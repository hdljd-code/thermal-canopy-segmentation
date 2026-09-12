import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.runtime import cvtColor, preprocess_input


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train,
                 dataset_dir, augmentation_config=None, *,
                 image_dir_name="images", mask_dir_name="masks"):
        super(UnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_dir        = dataset_dir
        self.image_dir_name     = image_dir_name
        self.mask_dir_name      = mask_dir_name
        augmentation_config     = augmentation_config or {}
        self.augmentation_config = {
            "jitter":    augmentation_config.get("jitter", 0.3),
            "brightness": augmentation_config.get("brightness", 0.3),
            "scale_min": augmentation_config.get("scale_min", 0.25),
            "scale_max": augmentation_config.get("scale_max", 2.0),
        }

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]


        jpg = Image.open(
            os.path.join(
                self.dataset_dir, self.image_dir_name, name + ".jpg"
            )
        )
        png = Image.open(
            os.path.join(
                self.dataset_dir, self.mask_dir_name, name + ".png"
            )
        )


        jpg, png    = self.get_random_data(
            jpg,
            png,
            self.input_shape,
            random=self.train,
            **self.augmentation_config,
        )

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png         = np.array(png)


        png_for_onehot = png.copy()
        png_for_onehot[png_for_onehot == 255] = self.num_classes
        seg_labels  = np.eye(self.num_classes + 1)[png_for_onehot.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, brightness=0.3,
                        scale_min=0.25, scale_max=2.0, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))


        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label


        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(scale_min, scale_max)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)


        flip = self.rand()<.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)


        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)


        scale = np.random.uniform(1 - brightness, 1 + brightness)
        hue, saturation, value = cv2.split(
            cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
        )
        lookup = np.clip(
            np.arange(256, dtype=np.float32) * scale, 0, 255
        ).astype(image_data.dtype)
        image_data = cv2.merge(
            (hue, saturation, cv2.LUT(value, lookup))
        )
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def unet_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
