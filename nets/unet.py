import torch
import torch.nn as nn

from nets.resnet50 import resnet50


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.relu(self.conv1(outputs))
        return self.relu(self.conv2(outputs))


class Unet(nn.Module):
    def __init__(
        self, num_classes=21, pretrained=False, backbone="resnet50"
    ):
        super(Unet, self).__init__()
        if backbone != "resnet50":
            raise ValueError("公开基线仅支持 resnet50")

        self.resnet = resnet50(pretrained=pretrained)
        in_filters = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]

        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(
                out_filters[0], out_filters[0], kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_filters[0], out_filters[0], kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        feat1, feat2, feat3, feat4, feat5 = self.resnet(inputs)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        return self.final(self.up_conv(up1))

    def freeze_backbone(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.resnet.parameters():
            param.requires_grad = True
