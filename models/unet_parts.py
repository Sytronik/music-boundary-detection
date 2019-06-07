"""
sub-parts of the U-Net model
"""

from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .cbam import CBAM


def force_size_same(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    diffX = a.shape[-2] - b.shape[-2]
    diffY = a.shape[-1] - b.shape[-1]

    if diffY > 0:
        b = F.pad(b, [diffY // 2, int(np.ceil(diffY / 2)), 0, 0])
    elif diffY < 0:
        a = F.pad(a, [(-diffY) // 2, int(np.ceil((-diffY) / 2)), 0, 0])

    if diffX > 0:
        b = F.pad(b, [0, 0, diffX // 2, int(np.ceil(diffX / 2))])
    elif diffX < 0:
        a = F.pad(a, [0, 0, (-diffX) // 2, int(np.ceil((-diffX) / 2))])

    return a, b


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, act_fn,
                 kernel_size, padding,
                 groups=1, stride=(1, 1)):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=padding, groups=groups, stride=stride),
            nn.BatchNorm2d(out_ch),
        )
        if act_fn:
            self.cba.add_module('2', act_fn)

    def forward(self, x):
        return self.cba(x)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act_fn: nn.Module,
                 last_act_fn=True, use_cbam=False):
        super().__init__()
        self.skipcba = ConvBNAct(in_ch, out_ch, None, kernel_size=(1, 1), padding=(0, 0))

        self.cba1 = ConvBNAct(in_ch, out_ch, act_fn, (3, 3), padding=(1, 1))
        self.cba2 = ConvBNAct(out_ch, out_ch, None, (3, 3), padding=(1, 1))

        self.act_fn = act_fn if last_act_fn else None
        self.cbam = CBAM(out_ch, reduction_ratio=min(16, out_ch)) if use_cbam else None

    def forward(self, x):
        residual = self.skipcba(x)

        out = self.cba1(x)
        out = self.cba2(out)
        if self.cbam:
            out = self.cbam(out)
        out += residual
        if self.act_fn:
            out = self.act_fn(out)

        return out


class FusionNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act_fn: nn.Module, use_cbam=False):
        super().__init__()
        self.cba1 = ConvBNAct(in_ch, out_ch, act_fn, (3, 3), padding=(1, 1))
        self.resblock = ResidualBlock(out_ch, out_ch, act_fn, last_act_fn=False, use_cbam=use_cbam)
        self.cba2 = ConvBNAct(out_ch, out_ch, act_fn, (3, 3), padding=(1, 1))

    def forward(self, x):
        x = self.cba1(x)
        x = self.resblock(x)
        x = self.cba2(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        # self.block = FusionNetBlock(in_ch, out_ch, nn.ReLU(inplace=True), use_cbam=use_cbam)
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, nn.ReLU(inplace=True), kernel_size, padding),
            ConvBNAct(out_ch, out_ch, nn.ReLU(inplace=True), kernel_size, padding),
        )
        # self.conv = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.block(x)
        return x


class DownAndConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.pool = nn.MaxPool2d((2, 2))

        # self.block = FusionNetBlock(in_ch, out_ch, nn.ReLU(inplace=True), use_cbam=use_cbam)
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, nn.ReLU(inplace=True), kernel_size, padding, stride=stride),
            ConvBNAct(out_ch, out_ch, nn.ReLU(inplace=True), kernel_size, padding),
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x


class UpAndConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear=False, kernel_size=(3, 3), upsample=(2, 2)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, upsample, stride=upsample)

        # self.block = FusionNetBlock(out_ch, out_ch, nn.ReLU(inplace=True), use_cbam=use_cbam)
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, nn.ReLU(inplace=True), kernel_size, padding),
            ConvBNAct(out_ch, out_ch, nn.ReLU(inplace=True), kernel_size, padding),
        )

    def forward(self, x, x_skip):
        x = self.up(x)
        x, x_skip = force_size_same(x, x_skip)

        out = torch.cat((x, x_skip), dim=-3)
        # out = (x + x_skip) / 2
        out = self.block(out)
        return out


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1))
        self.conv1 = nn.Conv2d(in_ch, out_ch, (1, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        x = self.conv1(x)
        x = self.gap(x)
        return x
