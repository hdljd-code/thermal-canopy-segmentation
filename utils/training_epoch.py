import os

import torch
import torch.nn as nn
from tqdm import tqdm

from nets.training_components import CE_Loss, Dice_loss
from utils.runtime import get_lr
from utils.segmentation_metrics import f_score


def _freeze_frozen_batchnorm_running_stats(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()


def _supervised_loss(outputs, pngs, labels, weights, num_classes, dice_loss):
    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
    if dice_loss:
        loss = loss + Dice_loss(outputs, labels)
    return loss


def fit_one_epoch(
    model_train, model, loss_history, eval_callback, optimizer,
    epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda,
    dice_loss, cls_weights, num_classes, fp16, scaler,
    save_period, save_dir, local_rank=0, gradient_accumulation_steps=1,
):
    if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps 必须是大于等于 1 的整数")

    total_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(
            total=epoch_step,
            desc=f'Epoch {epoch + 1}/{Epoch}',
            postfix=dict,
            mininterval=0.3,
        )
    model_train.train()
    if getattr(model, "_freeze_frozen_batchnorm", False):
        _freeze_frozen_batchnorm_running_stats(model_train)
    optimizer.zero_grad()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        accumulation_window_size = min(
            gradient_accumulation_steps,
            epoch_step - (iteration // gradient_accumulation_steps) * gradient_accumulation_steps,
        )
        should_step = (
            (iteration + 1) % gradient_accumulation_steps == 0
            or iteration + 1 == epoch_step
        )
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        if not fp16:
            outputs = model_train(imgs)
            loss = _supervised_loss(
                outputs, pngs, labels, weights, num_classes, dice_loss
            )
            with torch.no_grad():
                _f_score = f_score(outputs, labels)
            loss_value = loss.item()
            (loss / accumulation_window_size).backward()
            if should_step:
                optimizer.step()
                optimizer.zero_grad()
        else:
            with torch.amp.autocast('cuda'):
                outputs = model_train(imgs)
                loss = _supervised_loss(
                    outputs, pngs, labels, weights, num_classes, dice_loss
                )
                with torch.no_grad():
                    _f_score = f_score(outputs, labels)
            loss_value = loss.item()
            scaler.scale(loss / accumulation_window_size).backward()
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        total_loss += loss_value
        total_f_score += _f_score.item()
        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score': total_f_score / (iteration + 1),
                'lr': get_lr(optimizer),
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(
            total=epoch_step_val,
            desc=f'Epoch {epoch + 1}/{Epoch}',
            postfix=dict,
            mininterval=0.3,
        )

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs = model_train(imgs)
            loss = _supervised_loss(
                outputs, pngs, labels, weights, num_classes, dice_loss
            )
            _f_score = f_score(outputs, labels)
            val_loss += loss.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'val_loss': val_loss / (iteration + 1),
                'f_score': val_f_score / (iteration + 1),
                'lr': get_lr(optimizer),
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        current_val_loss = val_loss / epoch_step_val
        loss_history.append_loss(
            epoch + 1, total_loss / epoch_step, current_val_loss
        )

        eval_result = None
        if eval_callback is not None:
            eval_result = eval_callback.on_epoch_end(epoch + 1, model_train)
            if eval_result and eval_result.get("best_miou_refreshed"):
                print(
                    f"[Checkpoint] best_miou_epoch_weights.pth updated at "
                    f"epoch={epoch + 1}, val_mIoU={eval_result['best_miou']:.4f}"
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, "best_miou_epoch_weights.pth"),
                )
                with open(
                    os.path.join(save_dir, "best_source.txt"),
                    "w",
                    encoding="utf-8",
                ) as best_source_f:
                    best_source_f.write("best_miou_epoch_weights.pth\n")
                    best_source_f.write(f"val_mIoU={eval_result['best_miou']:.6f}\n")
                    best_source_f.write(f"epoch={epoch + 1}\n")

        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print(
            'Total Loss: %.3f || Val Loss: %.3f '
            % (total_loss / epoch_step, current_val_loss)
        )

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_dir,
                    'ep%03d-loss%.3f-val_loss%.3f.pth'
                    % (epoch + 1, total_loss / epoch_step, current_val_loss),
                ),
            )

        if len(loss_history.val_loss) <= 1 or current_val_loss <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_epoch_weights.pth"),
            )
            print(
                f'[Checkpoint] best_epoch_weights.pth updated at '
                f'epoch={epoch + 1}, val_loss={current_val_loss:.6f}'
            )

        torch.save(
            model.state_dict(),
            os.path.join(save_dir, "last_epoch_weights.pth"),
        )
