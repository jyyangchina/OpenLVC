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
from mmcv.ops import ModulatedDeformConv2d as DCN


from compressai.entropy_models import GaussianConditional, EntropyBottleneck

from compressai.models.google import CompressionModel, get_scale_table
from compressai.models.utils import (
    conv,
    deconv,
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

    
class Unet(nn.Module):
    def __init__(self, in_ch=3*2, nf=64, base_ks=3, out_ch=1*3*9):
        super(Unet, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        self.out_conv = nn.Conv2d(
            nf*2, out_ch, base_ks, padding=base_ks//2
            )

    def forward(self, x):
        out1 = self.in_conv(x)  
        out2 = self.tr_conv(out1)
        out = self.out_conv(torch.cat([out1, out2], dim=1))
        return out
    

class VQE(nn.Module):

    def __init__(self, deform_ks: int = 3):
        super(VQE, self).__init__()
        self.size_dk = deform_ks**2
        self.deform_gr = 1

        self.qe_me_dcn = Unet(in_ch=3*2+2, out_ch=self.deform_gr*3*self.size_dk)
        self.qe_dcn = DCN(
            3, 64, kernel_size=(deform_ks, deform_ks), padding=deform_ks//2, deform_groups=self.deform_gr
            ) 
        self.qe_net = nn.Sequential(
            conv(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            conv(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            conv(64, 3, kernel_size=3, stride=1),
        )

    def forward(self, x_rec, x_ref, motion_info):
        motion_info_dcn = self.qe_me_dcn(torch.cat([x_rec, x_ref, motion_info], dim=1))
        off1, off2, mask = torch.chunk(motion_info_dcn, 3, dim=1)
        offset = 10.0 * torch.tanh(torch.cat((off1, off2), dim=1)) # max_residue_magnitude
        offset = offset + motion_info.flip(1).repeat(1, offset.size(1)//2, 1, 1)
        mask = torch.sigmoid(mask)
        x_rec_res = self.qe_net(self.qe_dcn(x_ref, offset, mask))
        return x_rec + x_rec_res


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

        self.res_encoder = Encoder(in_planes = 3, mid_planes=128, out_planes=128)
        self.res_decoder = Decoder(out_planes = 3, mid_planes=128, in_planes=128)
        self.res_entropymodel = EntropyModel_hyper_latent(planes=128, mid_planes=128)

        self.motion_estimation = ME_Spynet()
        self.mot_encoder = Encoder(in_planes=2, mid_planes=128, out_planes=128)
        self.mot_decoder = Decoder(in_planes=128, mid_planes=128, out_planes=2)
        self.mot_entropymodel = EntropyModel_hyper_latent(planes=128, mid_planes=128)
    
        self.img_var_factor = nn.Parameter(torch.ones([4, 1, 192, 1, 1]))
        self.img_var_bias = nn.Parameter(torch.ones(1))
        self.mot_var_factor = nn.Parameter(torch.ones([8, 1, 128, 1, 1]))
        self.mot_var_bias = nn.Parameter(torch.ones(1))
        self.res_var_factor = nn.Parameter(torch.ones([8, 1, 128, 1, 1]))
        self.res_var_bias = nn.Parameter(torch.ones(1))

        self.qe_net_pred = VQE()
        self.qe_net_rec = VQE()
        
    def forward(self, frames, factor):
        if not isinstance(frames, List):
            raise RuntimeError(f"Invalid number of frames: {len(frames)}.")

        reconstructions = []
        frames_likelihoods = []
        
        with torch.no_grad():
            latent_prior_intra = None
            x_hat, likelihoods = self.forward_keyframe(frames[0], latent_prior_intra, factor)
            # reconstructions.append(x_hat)
            # frames_likelihoods.append(likelihoods)
            x_ref = x_hat.detach()  # stop gradient flow (cf: google2020 paper)

        latent_prior_inter = {"latent_prior_motion": None, "latent_prior_res": None}
        for i in range(1, len(frames)-1):
            x = frames[i]
            x_ref, latent_prior_inter, likelihoods = self.forward_inter(x, x_ref, latent_prior_inter, factor)
            reconstructions.append(x_ref)
            frames_likelihoods.append(likelihoods)
            
        with torch.no_grad():
            x_ref_ = x_ref
            latent_prior_inter_ = latent_prior_inter
            
        x_ref, latent_prior_inter, likelihoods = self.forward_inter(frames[-1], x_ref_, latent_prior_inter_, factor+4)
        reconstructions.append(x_ref)
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

    def encode_keyframe(self, x, latent_prior_intra, factor):
        y = self.img_encoder(x)
        y = self.img_var_factor[factor] * y
        y_hat, out_keyframe = self.img_entropymodel.compress(y, latent_prior_intra)
        y_hat = self.img_var_bias * y_hat / self.img_var_factor[factor]
        x_hat = self.img_decoder(y_hat)

        return x_hat, out_keyframe

    def decode_keyframe(self, strings, shape, latent_prior_intra, factor):
        y_hat = self.img_entropymodel.decompress(strings, shape, latent_prior_intra)
        y_hat = self.img_var_bias * y_hat / self.img_var_factor[factor]
        x_hat = self.img_decoder(y_hat)

        return x_hat

    def encode_inter(self, x_cur, x_ref, latent_prior_inter, factor):
        # encode the motion information
        x_motion = self.motion_estimation(x_cur, x_ref)
        y_motion = self.mot_encoder(x_motion)
        y_motion = self.mot_var_factor[factor] * y_motion
        y_motion_ref = latent_prior_inter["latent_prior_motion"]
        y_motion_hat, out_motion = self.mot_entropymodel.compress(y_motion, y_motion_ref)
        y_motion_hat = self.mot_var_bias * y_motion_hat / self.mot_var_factor[factor]
        motion_info = self.mot_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)
        x_pred_qe = self.qe_net_pred(x_pred, x_ref, motion_info)

        # residual
        x_res = x_cur - x_pred_qe
        y_res = self.res_encoder(x_res)
        y_res = self.res_var_factor[factor] * y_res
        y_res_ref = latent_prior_inter["latent_prior_res"]
        y_res_hat, out_res = self.res_entropymodel.compress(y_res, y_res_ref)
        y_res_hat = self.res_var_bias * y_res_hat / self.res_var_factor[factor]
        x_res_hat = self.res_decoder(y_res_hat)

        # final reconstruction: prediction + residual
        x_rec = x_pred_qe + x_res_hat
        x_rec_qe = self.qe_net_rec(x_rec, x_ref, motion_info)

        return x_rec_qe, {"latent_prior_motion": y_motion_hat, "latent_prior_res": y_res_hat}, {
            "strings": {
                "motion": out_motion["strings"],
                "residual": out_res["strings"],
            },
            "shape": {"motion": out_motion["shape"], "residual": out_res["shape"]},
        }

    def decode_inter(self, x_ref, strings, shapes, latent_prior_inter, factor):
        y_motion_ref = latent_prior_inter["latent_prior_motion"]
        key = "motion"
        y_motion_hat = self.mot_entropymodel.decompress(strings[key], shapes[key], y_motion_ref)
        y_motion_hat = self.mot_var_bias * y_motion_hat / self.mot_var_factor[factor]

        # decode the space-scale flow information
        motion_info = self.mot_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)
        x_pred_qe = self.qe_net_pred(x_pred, x_ref, motion_info)

        # residual
        y_res_ref = latent_prior_inter["latent_prior_res"]
        key = "residual"
        y_res_hat = self.res_entropymodel.decompress(strings[key], shapes[key], y_res_ref)
        y_res_hat = self.res_var_bias * y_res_hat / self.res_var_factor[factor]
        x_res_hat = self.res_decoder(y_res_hat)
        x_rec = x_pred_qe + x_res_hat
        x_rec_qe = self.qe_net_rec(x_rec, x_ref, motion_info)

        return x_rec_qe, {"latent_prior_motion": y_motion_hat, "latent_prior_res": y_res_hat}

    def forward_prediction(self, x_ref, motion_info):
        x_pred = flow_warp(x_ref, motion_info)
        return x_pred
    
    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""

        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())

        return aux_loss_list


    def load_state_dict(self, state_dict, strict=True, update_buffer=True):

        # Dynamically update the entropy bottleneck buffers related to the CDFs
        if update_buffer:
            update_registered_buffers(
                self.img_entropymodel.gaussian_conditional,
                "img_entropymodel.gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )
            update_registered_buffers(
                self.img_entropymodel.entropy_bottleneck,
                "img_entropymodel.entropy_bottleneck",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )

            update_registered_buffers(
                self.res_entropymodel.gaussian_conditional,
                "res_entropymodel.gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )
            update_registered_buffers(
                self.res_entropymodel.entropy_bottleneck,
                "res_entropymodel.entropy_bottleneck",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )

            update_registered_buffers(
                self.mot_entropymodel.gaussian_conditional,
                "mot_entropymodel.gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )
            update_registered_buffers(
                self.mot_entropymodel.entropy_bottleneck,
                "mot_entropymodel.entropy_bottleneck",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )

        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()

        updated = self.img_entropymodel.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        # updated |= super().update(force=force)

        updated = self.res_entropymodel.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        # updated |= super().update(force=force)

        updated = self.mot_entropymodel.gaussian_conditional.update_scale_table(
            scale_table, force=force
        )
        # updated |= super().update(force=force)

        for m in self.modules():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
            
        return updated
