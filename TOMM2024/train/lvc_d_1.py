# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


from compressai.models.utils import (
    conv,
)

from compressai.layers import (
    conv3x3,
    conv1x1)

def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        if stride != 1:
            self.downsample = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2):
        super().__init__()
        self.subpel_conv = subpel_conv1x1(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU()
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch, leaky_relu_slope=0.01):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.conv2 = conv3x3(out_ch, out_ch)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        out = out + identity
        return out




class LVC_exp_spy_res(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__(
                    ResidualBlockWithStride(in_planes, mid_planes, stride=2),
                    ResidualBlock(mid_planes, mid_planes),
                    ResidualBlockWithStride(mid_planes, mid_planes, stride=2),
                    ResidualBlock(mid_planes, mid_planes),
                    ResidualBlockWithStride(mid_planes, mid_planes, stride=2),
                    ResidualBlock(mid_planes, mid_planes),
                    conv3x3(mid_planes, out_planes, stride=2),
                )

        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 192
            ):
                super().__init__(
                    ResidualBlock(in_planes, mid_planes),
                    ResidualBlockUpsample(mid_planes, mid_planes, 2),
                    ResidualBlock(mid_planes, mid_planes),
                    ResidualBlockUpsample(mid_planes, mid_planes, 2),
                    ResidualBlock(mid_planes, mid_planes),
                    ResidualBlockUpsample(mid_planes, mid_planes, 2),
                    ResidualBlock(mid_planes, mid_planes),
                    subpel_conv1x1(mid_planes, out_planes, 2),
                )

        self.img_encoder = Encoder(in_planes=3, mid_planes=192, out_planes=192)
        self.img_decoder = Decoder(in_planes=192, mid_planes=192, out_planes=3)
    
    def forward(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        reconstructions = []

        x_hat = self.forward_keyframe(frames[0])
        reconstructions.append(x_hat)


        return {
            "x_hat": reconstructions,
        }

    def forward_keyframe(self, x):
        y = self.img_encoder(x)
        x_hat = self.img_decoder(y)
        return x_hat

    
