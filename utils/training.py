import datetime
import math
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.training_components import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.segmentation_dataset import UnetDataset, unet_dataset_collate
from utils.runtime import seed_everything, show_config, worker_init_fn
from utils.training_callbacks import EvalCallback, LossHistory
from utils.training_epoch import fit_one_epoch


def _format_config_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def write_experiment_config(save_dir_root, config_dict):
    os.makedirs(save_dir_root, exist_ok=True)
    config_path = os.path.join(save_dir_root, "experiment_config.txt")
    with open(config_path, "w", encoding="utf-8") as file:
        file.write("Experiment Configuration\n")
        file.write("=" * 80 + "\n")
        for key, value in config_dict.items():
            file.write(f"{key}: {_format_config_value(value)}\n")
    print(f"配置文件已保存: {config_path}")


def count_trainable_params(model):
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def print_trainable_params(stage, model):
    trainable, total = count_trainable_params(model)
    ratio = trainable / total * 100 if total > 0 else 0.0
    print(
        f"[TrainableParams] {stage}: trainable={trainable:,} / "
        f"total={total:,} ({ratio:.2f}%)"
    )


def _create_dataloader(
    dataset, batch_size, num_workers, seed, shuffle, drop_last,
    collate_fn=unet_dataset_collate, pin_memory=True,
):
    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=seed),
    )


def _compute_lr_fit(batch_size, Init_lr, Min_lr, nbs=16, auto_scale_lr=False):
    if auto_scale_lr:
        scale = batch_size / nbs
        Init_lr_fit = Init_lr * scale
        Min_lr_fit = Min_lr * scale
    else:
        Init_lr_fit = Init_lr
        Min_lr_fit = Min_lr

    if Init_lr_fit <= 0 or Min_lr_fit <= 0:
        raise ValueError("学习率必须大于 0")
    if Min_lr_fit > Init_lr_fit:
        raise ValueError(
            f"Min_lr ({Min_lr_fit}) 不能大于 Init_lr ({Init_lr_fit})"
        )
    return Init_lr_fit, Min_lr_fit


def _compute_train_epoch_step(
    num_train, batch_size, gradient_accumulation_steps
):
    if (
        not isinstance(gradient_accumulation_steps, int)
        or gradient_accumulation_steps < 1
    ):
        raise ValueError("gradient_accumulation_steps 必须是大于等于 1 的整数")
    effective_batch_size = batch_size * gradient_accumulation_steps
    effective_batch_count = num_train // effective_batch_size
    return effective_batch_count * gradient_accumulation_steps


def _create_optimizer(
    model, optimizer_type, Init_lr_fit, momentum, weight_decay
):
    supported = {"adam", "adamw"}
    if optimizer_type not in supported:
        raise ValueError(
            f"不支持的优化器类型: {optimizer_type}，可选: {supported}"
        )
    if optimizer_type == 'adam':
        return optim.Adam(
            model.parameters(),
            Init_lr_fit,
            betas=(momentum, 0.999),
            weight_decay=weight_decay,
        )
    if optimizer_type == 'adamw':
        return optim.AdamW(
            model.parameters(),
            Init_lr_fit,
            betas=(momentum, 0.999),
            weight_decay=weight_decay,
        )
    raise AssertionError("不可达的优化器分支")


def _check_existing_run(save_dir):
    if not os.path.exists(save_dir):
        return

    existing_weights = [
        name for name in os.listdir(save_dir) if name.endswith('.pth')
    ]
    training_records = [
        name
        for name in os.listdir(save_dir)
        if os.path.isdir(os.path.join(save_dir, name))
        and name.startswith('loss_')
    ]
    log_files = [
        name
        for name in os.listdir(save_dir)
        if name.endswith('.txt')
        and ('loss' in name or 'val_loss' in name or 'miou' in name)
    ]
    if not (existing_weights or training_records or log_files):
        return

    print("\n检测到已有训练记录:")
    if existing_weights:
        print(f"  - 权重文件: {existing_weights}")
    if training_records:
        print(f"  - 训练记录目录: {training_records}")
    if log_files:
        print(f"  - 日志文件: {log_files}")

    raise RuntimeError(
        f"{save_dir} 已存在训练记录。为避免覆盖，停止运行。"
    )


def run_training(model_cls, config):
    Cuda = config.get("Cuda", True)
    if Cuda and not torch.cuda.is_available():
        print("[WARN] Cuda=True 但 CUDA 不可用，自动切换到 CPU 模式")
        Cuda = False

    seed = config.get("seed", 11)
    backbones = ("resnet50",)
    num_classes = config.get("num_classes", 2)
    pretrained = config.get("pretrained", True)
    input_shape = config.get("input_shape", [512, 512])
    Init_Epoch = config.get("Init_Epoch", 0)
    Freeze_Epoch = config.get("Freeze_Epoch", 50)
    Freeze_batch_size = config.get("Freeze_batch_size", 8)
    UnFreeze_Epoch = config.get("UnFreeze_Epoch", 150)
    Stop_Epoch = config.get("Stop_Epoch", 150)
    Scheduler_Total_Epoch = config.get("Scheduler_Total_Epoch", 150)
    Unfreeze_batch_size = config.get("Unfreeze_batch_size", 8)
    Freeze_Train = config.get("Freeze_Train", True)
    Init_lr = config.get("Init_lr", 1e-4)
    Min_lr = config.get("Min_lr", Init_lr * 0.01)
    optimizer_type = config.get("optimizer_type", "adam")
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0)
    lr_decay_type = config.get("lr_decay_type", "cos")

    dataset_dir = config.get("dataset_dir", "dataset")
    image_dir_name = config.get("image_dir_name", "images")
    mask_dir_name = config.get("mask_dir_name", "masks")
    split_dir_name = config.get("split_dir_name", "splits")
    save_dir_root = config.get("save_dir_root", "outputs/training")
    save_period = config.get("save_period", 10)
    eval_flag = config.get("eval_flag", True)
    eval_period = config.get("eval_period", 5)
    gradient_accumulation_steps = config.get(
        "gradient_accumulation_steps", 1
    )
    configured_effective_batch_size = config.get(
        "effective_batch_size", None
    )
    if (
        not isinstance(gradient_accumulation_steps, int)
        or gradient_accumulation_steps < 1
    ):
        raise ValueError("gradient_accumulation_steps 必须是大于等于 1 的整数")
    expected_effective_batch_sizes = {
        Freeze_batch_size * gradient_accumulation_steps,
        Unfreeze_batch_size * gradient_accumulation_steps,
    }
    if (
        configured_effective_batch_size is not None
        and expected_effective_batch_sizes != {configured_effective_batch_size}
    ):
        raise ValueError(
            "effective_batch_size 与冻结/解冻阶段配置不一致: "
            "expected={}, configured={}".format(
                sorted(expected_effective_batch_sizes),
                configured_effective_batch_size,
            )
        )

    dice_loss = config.get("dice_loss", True)
    cls_weights = config.get(
        "cls_weights", np.ones([num_classes], np.float32)
    )
    if not isinstance(cls_weights, np.ndarray):
        cls_weights = np.array(cls_weights, dtype=np.float32)
    num_workers = config.get("num_workers", 2)
    pin_memory = config.get("pin_memory", True)
    cudnn_benchmark = config.get("cudnn_benchmark", True)
    if not isinstance(pin_memory, bool):
        raise ValueError("pin_memory 必须是 bool")
    if not isinstance(cudnn_benchmark, bool):
        raise ValueError("cudnn_benchmark 必须是 bool")
    augmentation_config = config.get("augmentation_config", None)
    auto_scale_lr = config.get("auto_scale_lr", False)

    write_experiment_config(save_dir_root, config)
    seed_everything(seed)
    local_rank = 0
    for backbone in backbones:
        save_dir = os.path.join(save_dir_root, backbone)
        _check_existing_run(save_dir)

        print(f"\n{'=' * 20} 正在开始训练 Backbone: {backbone} {'=' * 20}")
        os.makedirs(save_dir, exist_ok=True)
        model = model_cls(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone=backbone,
        ).train()
        if not pretrained:
            weights_init(model)

        time_str = datetime.datetime.strftime(
            datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S'
        )
        log_dir = os.path.join(save_dir, "loss_" + time_str)
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)

        use_fp16 = config.get("fp16", True)
        if use_fp16 and not (Cuda and torch.cuda.is_available()):
            print(
                "[AMP] fp16 was requested but CUDA is unavailable, "
                "falling back to fp32"
            )
            use_fp16 = False
        scaler = torch.amp.GradScaler('cuda') if use_fp16 and Cuda else None
        print(f"[AMP] fp16={use_fp16}")
        if use_fp16:
            amp_config_path = os.path.join(save_dir, "amp_config.txt")
            with open(amp_config_path, "w", encoding="utf-8") as file:
                file.write(f"fp16: {use_fp16}\n")

        model_train = model.train()
        if Cuda:
            cudnn.benchmark = cudnn_benchmark
            if torch.cuda.device_count() > 1:
                model_train = torch.nn.DataParallel(model).cuda()
            else:
                model_train = model_train.cuda()

        split_root = os.path.join(
            dataset_dir,
            split_dir_name,
        )
        train_txt = os.path.join(split_root, "train.txt")
        val_txt = os.path.join(split_root, "val.txt")
        if not os.path.exists(train_txt):
            raise FileNotFoundError(f"训练集文件不存在: {train_txt}")
        if not os.path.exists(val_txt):
            raise FileNotFoundError(f"验证集文件不存在: {val_txt}")

        with open(train_txt, "r", encoding="utf-8") as file:
            train_lines = file.readlines()
        with open(val_txt, "r", encoding="utf-8") as file:
            val_lines = file.readlines()
        num_train = len(train_lines)
        num_val = len(val_lines)

        if local_rank == 0:
            show_config(
                num_classes=num_classes,
                backbone=backbone,
                model_path="",
                input_shape=input_shape,
                Init_Epoch=Init_Epoch,
                Freeze_Epoch=Freeze_Epoch,
                UnFreeze_Epoch=UnFreeze_Epoch,
                Freeze_batch_size=Freeze_batch_size,
                Unfreeze_batch_size=Unfreeze_batch_size,
                Freeze_Train=Freeze_Train,
                Init_lr=Init_lr,
                Min_lr=Min_lr,
                optimizer_type=optimizer_type,
                momentum=momentum,
                lr_decay_type=lr_decay_type,
                save_period=save_period,
                save_dir=save_dir,
                num_workers=num_workers,
                num_train=num_train,
                num_val=num_val,
            )
            print(
                "[Batch] micro_batch_size(freeze/unfreeze)=({}/{})，"
                "gradient_accumulation_steps={}，effective_batch_size={}".format(
                    Freeze_batch_size,
                    Unfreeze_batch_size,
                    gradient_accumulation_steps,
                    configured_effective_batch_size
                    if configured_effective_batch_size is not None
                    else Freeze_batch_size * gradient_accumulation_steps,
                )
            )
            print(
                "[Runtime] num_workers={}，pin_memory={}，"
                "cudnn_benchmark={}".format(
                    num_workers, pin_memory, cudnn_benchmark
                )
            )

        UnFreeze_flag = False
        batch_size = (
            Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        )
        print_trainable_params("before freeze", model)
        if Freeze_Train:
            model.freeze_backbone()
            print_trainable_params("after freeze_backbone", model)
        else:
            print("[Freeze] Freeze_Train=False，跳过 backbone 冻结。")

        Init_lr_fit, Min_lr_fit = _compute_lr_fit(
            batch_size,
            Init_lr,
            Min_lr,
            auto_scale_lr=auto_scale_lr,
        )
        print(
            f"[LR] 配置值: Init_lr={Init_lr}, Min_lr={Min_lr}, "
            f"auto_scale_lr={auto_scale_lr}"
        )
        print(
            f"[LR] 实际值: Init_lr_fit={Init_lr_fit}, "
            f"Min_lr_fit={Min_lr_fit}"
        )
        config_path = os.path.join(save_dir_root, "experiment_config.txt")
        if os.path.exists(config_path):
            with open(config_path, "a", encoding="utf-8") as file:
                file.write("\n# 实际学习率（_compute_lr_fit 计算后）:\n")
                file.write(f"auto_scale_lr: {auto_scale_lr}\n")
                file.write(f"Init_lr_fit: {Init_lr_fit}\n")
                file.write(f"Min_lr_fit: {Min_lr_fit}\n")

        optimizer = _create_optimizer(
            model,
            optimizer_type,
            Init_lr_fit,
            momentum,
            weight_decay,
        )
        lr_scheduler_func = get_lr_scheduler(
            lr_decay_type,
            Init_lr_fit,
            Min_lr_fit,
            Scheduler_Total_Epoch,
        )

        train_dataset = UnetDataset(
            train_lines,
            input_shape,
            num_classes,
            True,
            dataset_dir,
            augmentation_config,
            image_dir_name=image_dir_name,
            mask_dir_name=mask_dir_name,
        )
        val_dataset = UnetDataset(
            val_lines,
            input_shape,
            num_classes,
            False,
            dataset_dir,
            image_dir_name=image_dir_name,
            mask_dir_name=mask_dir_name,
        )
        gen = _create_dataloader(
            train_dataset,
            batch_size,
            num_workers,
            seed,
            shuffle=True,
            drop_last=True,
            pin_memory=pin_memory,
        )
        gen_val = _create_dataloader(
            val_dataset,
            batch_size,
            num_workers,
            seed,
            shuffle=False,
            drop_last=False,
            pin_memory=pin_memory,
        )
        eval_callback = EvalCallback(
            model,
            input_shape,
            num_classes,
            val_lines,
            dataset_dir,
            log_dir,
            Cuda,
            eval_flag=eval_flag,
            period=eval_period,
            loss_history=loss_history,
            image_dir_name=image_dir_name,
            mask_dir_name=mask_dir_name,
        ) if local_rank == 0 else None

        try:
            for epoch in range(Init_Epoch, Stop_Epoch):
                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    batch_size = Unfreeze_batch_size
                    Init_lr_fit, Min_lr_fit = _compute_lr_fit(
                        batch_size,
                        Init_lr,
                        Min_lr,
                        auto_scale_lr=auto_scale_lr,
                    )
                    lr_scheduler_func = get_lr_scheduler(
                        lr_decay_type,
                        Init_lr_fit,
                        Min_lr_fit,
                        Scheduler_Total_Epoch,
                    )
                    model.unfreeze_backbone()
                    print_trainable_params("after unfreeze_backbone", model)
                    epoch_step = _compute_train_epoch_step(
                        num_train,
                        batch_size,
                        gradient_accumulation_steps,
                    )
                    epoch_step_val = math.ceil(num_val / batch_size)
                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError(
                            "数据集过小，无法继续进行训练，请扩充数据集。"
                        )
                    gen = _create_dataloader(
                        train_dataset,
                        batch_size,
                        num_workers,
                        seed,
                        shuffle=True,
                        drop_last=True,
                        pin_memory=pin_memory,
                    )
                    gen_val = _create_dataloader(
                        val_dataset,
                        batch_size,
                        num_workers,
                        seed,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=pin_memory,
                    )
                    UnFreeze_flag = True

                set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
                epoch_step = _compute_train_epoch_step(
                    num_train,
                    batch_size,
                    gradient_accumulation_steps,
                )
                epoch_step_val = math.ceil(num_val / batch_size)
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError(
                        "数据集过小，无法继续进行训练，请扩充数据集。"
                    )

                fit_one_epoch(
                    model_train,
                    model,
                    loss_history,
                    eval_callback,
                    optimizer,
                    epoch,
                    epoch_step,
                    epoch_step_val,
                    gen,
                    gen_val,
                    Stop_Epoch,
                    Cuda,
                    dice_loss,
                    cls_weights,
                    num_classes,
                    use_fp16,
                    scaler,
                    save_period,
                    save_dir,
                    local_rank,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
        finally:
            loss_history.writer.close()
            if eval_callback is not None and eval_callback.best_miou_epoch >= 0:
                print(
                    f"[TrainLoop] Backbone={backbone}, "
                    f"best_val_mIoU={eval_callback.best_miou:.4f}, "
                    f"epoch={eval_callback.best_miou_epoch}"
                )
            print(f"Backbone {backbone} 训练完成，正在释放显存...")
            del model, model_train, optimizer, gen, gen_val
            torch.cuda.empty_cache()

    print("全部结束。")
