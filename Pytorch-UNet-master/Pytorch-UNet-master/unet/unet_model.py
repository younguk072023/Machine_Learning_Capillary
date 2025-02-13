""" Full assembly of the parts to form the complete network """

from .unet_parts import *  # 점(.) 제거하여 불러오기
from .cbam import CBAM  # CBAM 모듈 불러오기

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.cbam1 = CBAM(64)  # CBAM 추가

        self.down1 = Down(64, 128)
        self.cbam2 = CBAM(128)

        self.down2 = Down(128, 256)
        self.cbam3 = CBAM(256)

        self.down3 = Down(256, 512)
        self.cbam4 = CBAM(512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.cbam5 = CBAM(1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.cbam6 = CBAM(512 // factor)

        self.up2 = Up(512, 256 // factor, bilinear)
        self.cbam7 = CBAM(256 // factor)

        self.up3 = Up(256, 128 // factor, bilinear)
        self.cbam8 = CBAM(128 // factor)

        self.up4 = Up(128, 64, bilinear)
        self.cbam9 = CBAM(64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.cbam1(x1)  # CBAM 적용

        x2 = self.down1(x1)
        x2 = self.cbam2(x2)

        x3 = self.down2(x2)
        x3 = self.cbam3(x3)

        x4 = self.down3(x3)
        x4 = self.cbam4(x4)

        x5 = self.down4(x4)
        x5 = self.cbam5(x5)

        x = self.up1(x5, x4)
        x = self.cbam6(x)

        x = self.up2(x, x3)
        x = self.cbam7(x)

        x = self.up3(x, x2)
        x = self.cbam8(x)

        x = self.up4(x, x1)
        x = self.cbam9(x)

        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)
