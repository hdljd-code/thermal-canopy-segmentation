from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(
    inputs, target, beta=1, smooth=1e-5, threhold=0.5, ignore_index=255
):
    n, c, h, w = inputs.size()
    _, ht, wt, ct = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(
            inputs, size=(ht, wt), mode="bilinear", align_corners=True
        )

    temp_inputs = torch.softmax(
        inputs.transpose(1, 2)
        .transpose(2, 3)
        .contiguous()
        .view(n, -1, c),
        -1,
    )
    temp_target = target.view(n, -1, ct)
    valid_mask = 1.0 - temp_target[..., -1:]
    temp_target = temp_target[..., :-1] * valid_mask
    temp_inputs = torch.gt(temp_inputs * valid_mask, threhold).float()

    tp = torch.sum(temp_target * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target, axis=[0, 1]) - tp
    score = ((1 + beta ** 2) * tp + smooth) / (
        (1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth
    )
    return torch.mean(score)


def fast_hist(label, prediction, num_classes):
    valid = (label >= 0) & (label < num_classes)
    return np.bincount(
        num_classes * label[valid].astype(int) + prediction[valid],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum(
        hist.sum(1) + hist.sum(0) - np.diag(hist), 1
    )


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def compute_mIoU(
    gt_dir, pred_dir, png_name_list, num_classes, name_classes=None
):
    hist = np.zeros((num_classes, num_classes))
    gt_images = [join(gt_dir, name + ".png") for name in png_name_list]
    pred_images = [join(pred_dir, name + ".png") for name in png_name_list]

    for index, (gt_path, pred_path) in enumerate(zip(gt_images, pred_images)):
        prediction = np.array(Image.open(pred_path))
        label = np.array(Image.open(gt_path))
        if label.shape != prediction.shape:
            raise ValueError(
                "Shape mismatch gt={} pred={}, {}, {}".format(
                    label.shape, prediction.shape, gt_path, pred_path
                )
            )

        valid = label != 255
        hist += fast_hist(
            label[valid].flatten(),
            prediction[valid].flatten(),
            num_classes,
        )
        if name_classes is not None and index > 0 and index % 10 == 0:
            print(
                "{} / {}: mIoU-{:.2f}%; mPA-{:.2f}%; Accuracy-{:.2f}%".format(
                    index,
                    len(gt_images),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist),
                )
            )

    ious = per_class_iu(hist)
    recalls = per_class_PA_Recall(hist)
    precisions = per_class_Precision(hist)
    if name_classes is not None:
        for index, class_name in enumerate(name_classes):
            print(
                "===>{}: IoU-{}; Recall-{}; Precision-{}".format(
                    class_name,
                    round(ious[index] * 100, 2),
                    round(recalls[index] * 100, 2),
                    round(precisions[index] * 100, 2),
                )
            )
    print(
        "===> mIoU: {}; mPA: {}; Accuracy: {}".format(
            round(np.nanmean(ious) * 100, 2),
            round(np.nanmean(recalls) * 100, 2),
            round(per_Accuracy(hist) * 100, 2),
        )
    )
    return np.array(hist, int), ious, recalls, precisions
