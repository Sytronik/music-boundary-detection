# full assembly of the sub-parts to form the complete net

import torch
from torch import nn, Tensor
from .unet_parts import InConv, DownAndConv, UpAndConv, OutConv


class UNet(nn.Module):
    def __init__(self, ch_in, ch_out, ch_base=32, depth=4, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.inc = InConv(ch_in, ch_base,
                          kernel_size=kernel_size,
                          )
        upsample = (2 * stride[0], 2 * stride[1])

        self.downs = nn.ModuleList(
            [DownAndConv(ch_base * (2**ii), ch_base * (2**(ii + 1)),
                         kernel_size=kernel_size,
                         stride=stride)
             for ii in range(depth)]
        )
        self.ups = nn.ModuleList(
            [UpAndConv(ch_base * (2**(ii + 1)), ch_base * (2**ii),
                       kernel_size=kernel_size,
                       upsample=upsample)
             for ii in reversed(range(depth))]
        )

        # self.outc = OutConv(ch_base, ch_out)
        self.outc = OutConv(ch_base, ch_out)

    def forward(self, xin):
        x_skip = [Tensor] * len(self.downs)
        x_skip[0] = self.inc(xin)

        for idx, down in enumerate(self.downs[:-1]):
            x_skip[idx + 1] = down(x_skip[idx])

        x = self.downs[-1](x_skip[-1])

        for item_skip, up in zip(reversed(x_skip), self.ups):
            x = up(x, item_skip)

        x = self.outc(x)
        return x
