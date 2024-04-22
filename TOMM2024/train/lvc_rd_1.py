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


from compressai.entropy_models import GaussianConditional, EntropyBottleneck

from compressai.models.google import CompressionModel, get_scale_table
from compressai.models.utils import (
    conv,
    quantize_ste,
    update_registered_buffers,
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

        class HyperEncoder(nn.Sequential):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__(
                    conv3x3(in_planes, mid_planes),
                    nn.LeakyReLU(),
                    conv3x3(mid_planes, mid_planes),
                    nn.LeakyReLU(),
                    conv3x3(mid_planes, mid_planes, stride=2),
                    nn.LeakyReLU(),
                    conv3x3(mid_planes, mid_planes),
                    nn.LeakyReLU(),
                    conv3x3(mid_planes, out_planes, stride=2),
                )

        class HyperDecoder(nn.Sequential):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__(
                    conv3x3(in_planes, mid_planes),
                    nn.LeakyReLU(),
                    subpel_conv1x1(mid_planes, mid_planes, 2),
                    nn.LeakyReLU(),
                    conv3x3(mid_planes, mid_planes * 3 // 2),
                    nn.LeakyReLU(),
                    subpel_conv1x1(mid_planes * 3 // 2, mid_planes * 3 // 2, 2),
                    nn.LeakyReLU(),
                    conv3x3(mid_planes * 3 // 2, out_planes * 2),
                )

        class EntropyModel_hyper_latent(CompressionModel):
            def __init__(self, planes: int = 192, mid_planes: int = 192):
                super().__init__(entropy_bottleneck_channels=mid_planes)
                self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
                self.hyper_decoder = HyperDecoder(planes, mid_planes, planes)
                self.temporal_context = conv(planes, planes, kernel_size=3, stride=1)
                self.latent_mask = nn.Sequential(
                    nn.Conv2d(planes * 3, planes * 3, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(planes * 3, planes * 2, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(planes * 2, planes * 1, 1),
                )                
                self.entropy_parameters = nn.Sequential(
                    nn.Conv2d(planes * 3, planes * 3, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(planes * 3, planes * 2, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(planes * 2, planes * 2, 1),
                )
                self.gaussian_conditional = GaussianConditional(None)

            def forward(self, y, y_ref):
                z = self.hyper_encoder(y)
                z_hat, z_likelihoods = self.entropy_bottleneck(z)

                hyperprior = self.hyper_decoder(z_hat)
                if y_ref is None:
                    y_ref = torch.zeros_like(y)
                latentprior = self.temporal_context(y_ref)
                latent_mask = self.latent_mask(torch.cat((hyperprior, latentprior), dim=1))
                refined_latentprior = torch.sigmoid(latent_mask) * latentprior
                gaussian_params = self.entropy_parameters(torch.cat((hyperprior, refined_latentprior), dim=1))
                scales, means = gaussian_params.chunk(2, 1)
                _, y_likelihoods = self.gaussian_conditional(y, scales, means)
                y_hat = quantize_ste(y - means) + means
                return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

            def compress(self, y, y_ref):
                z = self.hyper_encoder(y)

                z_string = self.entropy_bottleneck.compress(z)
                z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

                hyperprior = self.hyper_decoder(z_hat)
                if y_ref is None:
                    y_ref = torch.zeros_like(y)
                latentprior = self.temporal_context(y_ref)
                latent_mask = self.latent_mask(torch.cat((hyperprior, latentprior), dim=1))
                refined_latentprior = torch.sigmoid(latent_mask) * latentprior
                gaussian_params = self.entropy_parameters(torch.cat((hyperprior, refined_latentprior), dim=1))
                scales, means = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales)
                y_string = self.gaussian_conditional.compress(y, indexes, means)
                y_hat = self.gaussian_conditional.quantize(y, "dequantize", means)
                return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}

            def decompress(self, strings, shape, y_ref):
                assert isinstance(strings, list) and len(strings) == 2
                z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

                hyperprior = self.hyper_decoder(z_hat)
                if y_ref is None:
                    b, c, h, w = hyperprior.size()
                    y_ref = torch.zeros((b, c//2, h, w), device=hyperprior.device)
                latentprior = self.temporal_context(y_ref)                
                latent_mask = self.latent_mask(torch.cat((hyperprior, latentprior), dim=1))
                refined_latentprior = torch.sigmoid(latent_mask) * latentprior
                gaussian_params = self.entropy_parameters(torch.cat((hyperprior, refined_latentprior), dim=1))
                scales, means = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales)
                y_hat = self.gaussian_conditional.decompress(
                    strings[0], indexes, z_hat.dtype, means
                )

                return y_hat

        self.img_encoder = Encoder(in_planes=3, mid_planes=192, out_planes=192)
        self.img_decoder = Decoder(in_planes=192, mid_planes=192, out_planes=3)
        self.img_entropymodel = EntropyModel_hyper_latent(planes=192, mid_planes=192)

        self.img_var_factor = nn.Parameter(torch.ones([4, 1, 192, 1, 1]))
        self.img_var_bias = nn.Parameter(torch.ones(1))

        
    def forward(self, frames, factor):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        reconstructions = []
        frames_likelihoods = []

        latent_prior_intra = None
        x_hat, likelihoods = self.forward_keyframe(frames[0], latent_prior_intra, factor)
        reconstructions.append(x_hat)
        frames_likelihoods.append(likelihoods)
        x_ref = x_hat.detach()  # stop gradient flow (cf: google2020 paper)

        x_hat, likelihoods = self.forward_keyframe(frames[-1], self.img_encoder(x_ref), factor)
        reconstructions.append(x_hat)
        frames_likelihoods.append(likelihoods)

        return {
            "x_hat": reconstructions,
            "likelihoods": frames_likelihoods,
        }

    def forward_keyframe(self, x, latent_prior_intra, factor):
        y = self.img_encoder(x)
        y = self.img_var_factor[factor] * y
        y_hat, likelihoods = self.img_entropymodel(y, latent_prior_intra)
        y_hat = self.img_var_bias * y_hat / self.img_var_factor[factor]
        x_hat = self.img_decoder(y_hat)
        return x_hat, {"keyframe": likelihoods}
    
    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list

    def load_state_dict(self, state_dict, strict=True):

        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    