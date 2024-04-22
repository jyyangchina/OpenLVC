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


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)
    return outfeature


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='border',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[-2:]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[-2:]}) are not the same.')
    flow = flow.permute(0, 2, 3, 1)
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


class MEBasic(nn.Module):
    def __init__(self):
        super(MEBasic, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self):
        super(ME_Spynet, self).__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList(
            [MEBasic() for intLevel in range(self.L)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1list = [im1_pre]
        im2list = [im2_pre]
        for intLevel in range(self.L - 1):
            im1list.append(F.avg_pool2d(
                im1list[intLevel], kernel_size=2, stride=2))
            im2list.append(F.avg_pool2d(
                im2list[intLevel], kernel_size=2, stride=2))

        shape_fine = im2list[self.L - 1].size()
        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        device = im1.device
        flowfileds = torch.zeros(
            zeroshape, dtype=torch.float32, device=device)
        for intLevel in range(self.L):
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            flowfileds = flowfiledsUpsample + \
                self.moduleBasic[intLevel](torch.cat([im1list[self.L - 1 - intLevel],
                                                      flow_warp(im2list[self.L - 1 - intLevel],
                                                                flowfiledsUpsample),
                                                      flowfiledsUpsample], 1))

        return flowfileds


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

        self.motion_estimation = ME_Spynet()
        self.mot_encoder = Encoder(in_planes=2, mid_planes=128, out_planes=128)
        self.mot_decoder = Decoder(in_planes=128, mid_planes=128, out_planes=2)
    

    def forward(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        reconstructions = []

        x_ref = frames[0]

        for i in range(1, len(frames)):
            x = frames[i]
            x_ref = self.forward_inter(x, x_ref)
            reconstructions.append(x_ref)

        return {
            "x_hat": reconstructions,
        }

    def forward_keyframe(self, x):
        y = self.img_encoder(x)
        x_hat = self.img_decoder(y)
        return x_hat
    
    def forward_inter(self, x_cur, x_ref):
        x_motion = self.motion_estimation(x_cur, x_ref)
        y_motion = self.mot_encoder(x_motion)
        motion_info = self.mot_decoder(y_motion)
        x_pred = self.forward_prediction(x_ref, motion_info)

        return x_pred

    def forward_prediction(self, x_ref, motion_info):
        x_pred = flow_warp(x_ref, motion_info)
        return x_pred

    def load_state_dict(self, state_dict, strict=True):

        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

