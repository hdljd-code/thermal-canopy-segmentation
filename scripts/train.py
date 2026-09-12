import argparse
import sys
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[0] != PROJECT_ROOT:
    sys.path.insert(0, PROJECT_ROOT)

from nets.model_api import get_model_class
from nets.unet import Unet
from utils.training import run_training


EXPERIMENTS = {
    "baseline_adam": {
        "model_name": "baseline",
        "optimizer_type": "adam",
        "weight_decay": 0.0,
    },
    "baseline_adamw": {
        "model_name": "baseline",
        "optimizer_type": "adamw",
        "weight_decay": 1e-4,
    },
    "asf": {
        "model_name": "asf",
        "optimizer_type": "adamw",
        "weight_decay": 1e-4,
    },
    "msb": {
        "model_name": "msb",
        "optimizer_type": "adamw",
        "weight_decay": 1e-4,
    },
    "asf_msb": {
        "model_name": "asf_msb",
        "optimizer_type": "adamw",
        "weight_decay": 1e-4,
    },
}


def build_config(experiment_name, run_id):
    experiment = EXPERIMENTS[experiment_name]
    model_name = experiment["model_name"]
    network_file = (
        "nets/unet.py"
        if model_name == "baseline"
        else "nets/{}.py".format(model_name)
    )
    return {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "script_name": "scripts/train.py",
        "network_file": network_file,
        "Cuda": True,
        "seed": 101,
        "num_classes": 2,
        "pretrained": True,
        "input_shape": [512, 512],
        "Init_Epoch": 0,
        "Freeze_Epoch": 50,
        "UnFreeze_Epoch": 150,
        "Stop_Epoch": 150,
        "Scheduler_Total_Epoch": 150,
        "Freeze_batch_size": 8,
        "Unfreeze_batch_size": 8,
        "Freeze_Train": True,
        "Init_lr": 1e-4,
        "Min_lr": 1e-6,
        "auto_scale_lr": False,
        "optimizer_type": experiment["optimizer_type"],
        "momentum": 0.9,
        "weight_decay": experiment["weight_decay"],
        "lr_decay_type": "cos",
        "dataset_dir": "dataset",
        "image_dir_name": "images",
        "mask_dir_name": "masks",
        "split_dir_name": "splits",
        "augmentation_config": {"brightness": 0.3},
        "save_dir_root": "outputs/training/{}/{}".format(
            experiment_name, run_id
        ),
        "save_period": 50,
        "eval_flag": True,
        "eval_period": 5,
        "dice_loss": True,
        "cls_weights": [1.0, 1.0],
        "num_workers": 2,
    }


def main():
    parser = argparse.ArgumentParser(description="论文实验训练入口")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=tuple(EXPERIMENTS),
        help="选择论文中的实验配置",
    )
    parser.add_argument(
        "--run-id",
        default="run1",
        help="本次运行的通用标识，仅用于区分输出目录",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="不加载预训练骨干权重",
    )
    args = parser.parse_args()

    config = build_config(args.experiment, args.run_id)
    config["pretrained"] = not args.no_pretrained
    model_name = EXPERIMENTS[args.experiment]["model_name"]
    model_class = (
        Unet if model_name == "baseline" else get_model_class(model_name)
    )
    run_training(model_cls=model_class, config=config)


if __name__ == "__main__":
    main()
