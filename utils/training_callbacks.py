import os
import shutil

import cv2
import matplotlib
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from .runtime import cvtColor, preprocess_input, resize_image
from .segmentation_metrics import compute_mIoU


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag
        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except Exception:
            pass

    def append_loss(self, epoch, loss, val_loss=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        if self.val_loss_flag:
            plt.plot(
                iters, self.val_loss, 'coral', linewidth=2, label='val loss'
            )

        try:
            num = 5 if len(self.losses) < 25 else 15
            plt.plot(
                iters,
                scipy.signal.savgol_filter(self.losses, num, 3),
                'green',
                linestyle='--',
                linewidth=2,
                label='smooth train loss',
            )
            if self.val_loss_flag:
                plt.plot(
                    iters,
                    scipy.signal.savgol_filter(self.val_loss, num, 3),
                    '#8B4513',
                    linestyle='--',
                    linewidth=2,
                    label='smooth val loss',
                )
        except Exception:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(
        self, net, input_shape, num_classes, image_ids, dataset_dir,
        log_dir, cuda, miou_out_path=None,
        eval_flag=True, period=1, loss_history=None, *,
        image_dir_name="images", mask_dir_name="masks",
    ):
        super(EvalCallback, self).__init__()
        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.dataset_dir = dataset_dir
        self.image_dir_name = image_dir_name
        self.mask_dir_name = mask_dir_name
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path or os.path.join(log_dir, ".temp_miou_out")
        self.eval_flag = eval_flag
        self.period = period
        self.writer = loss_history.writer if loss_history is not None else None
        self.loss_history = loss_history
        self.mious = [0]
        self.epoches = [0]
        self.best_miou = -1.0
        self.best_miou_epoch = -1

        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write("0\n")

    def get_miou_png(self, image):
        image = cvtColor(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(
            image, (self.input_shape[1], self.input_shape[0])
        )
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, np.float32)),
                (2, 0, 1),
            ),
            0,
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            prediction = self.net(images)[0]
            prediction = F.softmax(
                prediction.permute(1, 2, 0), dim=-1
            ).cpu().numpy()
            top = int((self.input_shape[0] - nh) // 2)
            left = int((self.input_shape[1] - nw) // 2)
            prediction = prediction[top:top + nh, left:left + nw]
            prediction = cv2.resize(
                prediction,
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR,
            )
            prediction = prediction.argmax(axis=-1)

        return Image.fromarray(np.uint8(prediction))

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period != 0 or not self.eval_flag:
            return {
                "evaluated": False,
                "best_miou_refreshed": False,
                "miou": None,
                "best_miou": self.best_miou,
                "best_miou_epoch": self.best_miou_epoch,
            }

        self.net = model_eval
        gt_dir = os.path.join(self.dataset_dir, self.mask_dir_name)
        pred_dir = os.path.join(self.miou_out_path, 'detection-results')
        os.makedirs(pred_dir, exist_ok=True)

        print("Get miou.")
        for image_id in tqdm(self.image_ids):
            image_path = os.path.join(
                self.dataset_dir,
                self.image_dir_name,
                image_id + ".jpg",
            )
            image = Image.open(image_path)
            self.get_miou_png(image).save(
                os.path.join(pred_dir, image_id + ".png")
            )

        print("Calculate miou.")
        _, ious, _, _ = compute_mIoU(
            gt_dir, pred_dir, self.image_ids, self.num_classes, None
        )
        current_miou = np.nanmean(ious) * 100
        self.mious.append(current_miou)
        self.epoches.append(epoch)

        with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
            f.write(str(current_miou))
            f.write("\n")

        val_loss_value = None
        if self.loss_history is not None and self.loss_history.val_loss:
            val_loss_value = self.loss_history.val_loss[-1]

        epoch_miou_csv = os.path.join(self.log_dir, "epoch_val_miou.csv")
        if not os.path.exists(epoch_miou_csv):
            with open(epoch_miou_csv, 'w', encoding='utf-8-sig') as csv_f:
                csv_f.write("epoch,val_loss,val_mIoU\n")
        with open(epoch_miou_csv, 'a', encoding='utf-8-sig') as csv_f:
            val_loss_str = '' if val_loss_value is None else str(val_loss_value)
            csv_f.write(f'{epoch},{val_loss_str},{current_miou}\n')

        print(
            f"[Eval] epoch={epoch}, val_loss={val_loss_value}, "
            f"val_mIoU={current_miou:.4f}"
        )
        best_miou_refreshed = False
        if current_miou > self.best_miou:
            self.best_miou = current_miou
            self.best_miou_epoch = epoch
            best_miou_refreshed = True
            print(
                f"[Eval] 更新 best val_mIoU: {self.best_miou:.4f} "
                f"@ epoch {epoch}"
            )

        plt.figure()
        plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='train miou')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Miou')
        plt.title('A Miou Curve')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
        plt.cla()
        plt.close("all")

        print("Get miou done.")
        shutil.rmtree(self.miou_out_path)
        return {
            "evaluated": True,
            "best_miou_refreshed": best_miou_refreshed,
            "miou": current_miou,
            "best_miou": self.best_miou,
            "best_miou_epoch": self.best_miou_epoch,
        }
