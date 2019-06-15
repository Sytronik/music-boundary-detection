"""
sub-parts of the U-Net model
"""

from typing import Tuple

import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor


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


class InConv(nn.Module):
    """ Double Convolution Blocks

    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, nn.ReLU(inplace=True), kernel_size, padding),
            ConvBNAct(out_ch, out_ch, nn.ReLU(inplace=True), kernel_size, padding),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class DownAndConv(nn.Module):
    """ Max Pooling and Double Convolution Blocks

    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.pool = nn.MaxPool2d((2, 2))

        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, nn.LeakyReLU(0.2, True), kernel_size, padding, stride=stride),
            ConvBNAct(out_ch, out_ch, nn.LeakyReLU(0.2, True), kernel_size, padding),
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x


class UpAndConv(nn.Module):
    """ Transposed Convolution and Double Convolution Blocks

    """
    def __init__(self, in_ch: int, out_ch: int, bilinear=False, kernel_size=(3, 3), upsample=(2, 2)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, upsample, stride=upsample)

        self.block = nn.Sequential(
            ConvBNAct(out_ch, out_ch, nn.ReLU(inplace=True), kernel_size, padding),
            ConvBNAct(out_ch, out_ch, nn.ReLU(inplace=True), kernel_size, padding),
        )

    def forward(self, x, x_skip):
        x = self.up(x)
        x, x_skip = force_size_same(x, x_skip)

        out = (x + x_skip) / 2
        out = self.block(out)
        return out


class OutConv(nn.Module):
    """ For summarizing channels and mel frequency axis.

    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (1, 5), padding=(0, 2))
        # self.conv1 = nn.Conv2d(in_ch, out_ch, (1, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        x = self.conv1(x)
        x = self.gap(x)
        return x
