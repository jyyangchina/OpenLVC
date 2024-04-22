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

import argparse
import struct
import sys
import time
import os

from pathlib import Path
from typing import Any, IO, Dict,  Tuple, Union
from collections import defaultdict
import json
import glob

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch import Tensor
from torch.utils.model_zoo import tqdm
from pytorch_msssim import ms_ssim

import compressai

from compressai.transforms.functional import (
    rgb2ycbcr,
    yuv_444_to_420,
)
from lvc import LVC_exp_spy_res

torch.backends.cudnn.deterministic = True

Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


def write_string(fd, out_strings):
    bytes_cnt = 0
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


def convert_rgb_yuv420(frame: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return yuv_444_to_420(rgb2ycbcr(frame), mode="avg_pool")


def pad(x, p=2**6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def convert_output(t: Tensor, bitdepth: int = 8) -> np.array:
    assert bitdepth in (8, 10)
    # [0,1] fp ->  [0, 2**bitstream-1] uint
    dtype = np.uint8 if bitdepth == 8 else np.uint16
    t = (t.clamp(0, 1) * (2**bitdepth - 1)).cpu().squeeze()
    arr = t.numpy().astype(dtype)
    return arr


def write_frame(fout: IO[bytes], frame: Frame, bitdepth: np.uint = 8):
    for plane in frame:
        convert_output(plane, bitdepth).tofile(fout)


def compute_metrics_for_frame(org: Tensor, rec: Tensor, max_val: int) -> Dict[str, Any]:
    rec = torch.clamp(rec, 0.0, 1.0)
    mse = torch.mean((org - rec) ** 2)
    p = -10 * torch.log10(mse)
    m = ms_ssim(org, rec, data_range=1.0)

    return {
        "mse": mse * (max_val * max_val),
        "rgb_psnr": p,
        "msssim": m,
    }


def read_frame_to_torch(path, max_val):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0) / max_val
    return input_image


def encode_video(input, net, output, gop_size, num_frames, device, factor):
    name, defination, framerate, bitdepth, format = input.split("/")[-1].split("_")
    width, height = defination.split("x")
    width, height = int(width), int(height)
    framerate = int(framerate[:-3]) 
    bitdepth = int(bitdepth[:-3])
    max_val = 2**bitdepth - 1
    pngs = os.listdir(input)
    total_frames = len(pngs)

    num_frames = total_frames if num_frames == -1 else num_frames
    png_name_padding = 3

    avg_frame_enc_time = []

    f = Path(output).open("wb")
    with torch.no_grad():
        # write original image size
        write_uints(f, (height, width))
        # write original bitdepth
        write_uchars(f, (bitdepth,))
        # write number of coded frames
        write_uints(f, (num_frames,))
        # write number of gop size
        write_uints(f, (gop_size,))
        # write quality factor
        write_uints(f, (factor,))

        x_ref = None
        latent_prior_intra = None
        with tqdm(total=num_frames) as pbar:
            for i in range(num_frames):
                frm_enc_start = time.time()

                x_ori = read_frame_to_torch(
                    os.path.join(input, f"f{str(i+1).zfill(png_name_padding)}.png"), max_val).to(device)
                h, w = x_ori.size(2), x_ori.size(3)
                p = 64  # maximum 6 strides of 2
                x_cur = pad(x_ori, p)

                if i % gop_size == 0:
                    x_out, out_info = net.encode_keyframe(x_cur, latent_prior_intra, factor)
                    cnt = write_body(f, out_info["shape"], out_info["strings"])
                    print("I frame bits:", cnt)
                    latent_prior_inter = {"latent_prior_motion": None, "latent_prior_res": None}
                else:
                    # x_out, latent_prior_inter, out_info = net.encode_inter(x_cur, x_ref, latent_prior_inter, factor)
                    x_out, latent_prior_inter, out_info = net.encode_inter(x_cur, x_ref, latent_prior_inter, factor if i % 4 == 0 else factor+4)
                    cnt = 0
                    for shape, out in zip(
                        out_info["shape"].items(), out_info["strings"].items()
                    ):
                        cnt += write_body(f, shape[1], out[1])
                    print("P frame bits:", cnt)

                x_ref = x_out.clamp(0, 1)
                avg_frame_enc_time.append((time.time() - frm_enc_start))

                if i % gop_size == gop_size - 1:
                    latent_prior_intra = net.img_encoder(x_ref)
                pbar.update(1)

    f.close()

    size = filesize(output)
    bpp = float(size) * 8 / (h * w * num_frames)
    kbps = float(size) * 8 / (num_frames / framerate) / 1000
    return {"bpp": bpp, "kbps": kbps, "avg_frm_enc_time": np.mean(avg_frame_enc_time)}


def _encode(input, num_frames, ckpt_dir, coder, device, output, dir, gop_size, factor):
    compressai.set_entropy_coder(coder)
    enc_start = time.time()

    net = LVC_exp_spy_res()
    state_dict = torch.load(ckpt_dir)
    net.load_state_dict(state_dict)
    net.to(device).eval()

    if dir:
        if not os.path.exists(output):
            os.mkdir(output)
        bpp_list = []
        kbps_list = []
        for seq in glob.glob(input+"*"):
            print(seq)
            video_start = time.time()
            pre_fix = output + seq.split('/')[-1]
            out = encode_video(seq, net, pre_fix + '.bits', gop_size, num_frames, device, factor)
            video_end = time.time()
            print(seq.split('/')[-1])
            print(f"{out['bpp']:.3f} bpp | {out['kbps']:.3f} kbps")
            print("encode time:", video_end - video_start)
            bpp_list.append(out['bpp'])
            kbps_list.append(out['kbps'])
        enc_time = time.time() - enc_start
        print("average bpp:", np.mean(bpp_list))
        print("average kbps:", np.mean(kbps_list))
        print("Total time:", enc_time)

    else:
        if not Path(input).is_file():
            raise FileNotFoundError(f"{input} does not exist")
        out = encode_video(input, net, output, gop_size, num_frames, device, factor)
        enc_time = time.time() - enc_start
        print(
            f"{out['bpp']:.3f} bpp |"
            f" Encoded in {enc_time:.2f}s"
        )
        print(f"{out['kbps']:.3f} kbps |"
            f" Encoded in {out['avg_frm_enc_time']:.2f}s per frame)")


def decode_video(f, net, output):
    # read number of coded frames
    original_size = read_uints(f, 2)
    original_bitdepth = read_uchars(f, 1)[0]
    num_frames = read_uints(f, 1)[0]
    gop_size = read_uints(f, 1)[0]
    factor = read_uints(f, 1)[0]

    avg_frame_dec_time = []

    with torch.no_grad():
        x_ref = None
        latent_prior_intra = None
        with tqdm(total=num_frames) as pbar:
            for i in range(num_frames):
                frm_dec_start = time.time()

                if i % gop_size == 0:
                    strings, shape = read_body(f)
                    x_out = net.decode_keyframe(strings, shape, latent_prior_intra, factor)
                    latent_prior_inter = {"latent_prior_motion": None, "latent_prior_res": None}
                else:
                    mstrings, mshape = read_body(f)
                    rstrings, rshape = read_body(f)
                    inter_strings = {"motion": mstrings, "residual": rstrings}
                    inter_shapes = {"motion": mshape, "residual": rshape}

                    # x_out, latent_prior_inter = net.decode_inter(x_ref, inter_strings, inter_shapes, latent_prior_inter, factor)
                    x_out, latent_prior_inter = net.decode_inter(x_ref, inter_strings, inter_shapes, latent_prior_inter, factor if i % 4 == 0 else factor+4)

                x_ref = x_out.clamp(0, 1)
                avg_frame_dec_time.append((time.time() - frm_dec_start))
                
                if i % gop_size == gop_size - 1:
                    latent_prior_intra = net.img_encoder(x_ref)

                x_hat = crop(x_ref, original_size)

                if output is not None:
                    if Path(output).suffix == ".yuv":
                        rec = convert_rgb_yuv420(x_hat)
                        wopt = "wb" if i == 0 else "ab"
                        with Path(output).open(wopt) as fout:
                            write_frame(fout, rec, original_bitdepth)
                pbar.update(1)

    return {"avg_frm_dec_time": np.mean(avg_frame_dec_time)}


def _decode(inputpath, ckpt_dir, coder, device, output):
    compressai.set_entropy_coder(coder)

    dec_start = time.time()
    with Path(inputpath).open("rb") as f:
        start = time.time()
        net = LVC_exp_spy_res()
        
        state_dict = torch.load(ckpt_dir)
        net.load_state_dict(state_dict)
        net.to(device).eval()        
        load_time = time.time() - start

        out = decode_video(f, net, output)

    dec_time = time.time() - dec_start
    print(f"Decoded in {dec_time:.2f}s (model loading: {load_time:.2f}s)")


def eval_video(input, net, output_bits, output_yuv, output_json, output_log, gop_size, num_frames, device, factor):

    name, defination, framerate, bitdepth, format = input.split("/")[-1].split("_")
    width, height = defination.split("x")
    width, height = int(width), int(height)
    framerate = int(framerate[:-3]) 
    bitdepth = int(bitdepth[:-3])
    max_val = 2**bitdepth - 1
    pngs = os.listdir(input)
    total_frames = len(pngs)
    png_name_padding = 3
    num_frames = total_frames if num_frames == -1 else num_frames

    log_fp = open(output_log, 'w')
    avg_frame_enc_time = []
    results = defaultdict(list)
    f = Path(output_bits).open("wb")

    with torch.no_grad():
        # write_uints(f, (org_seq.height, org_seq.width))
        # write original bitdepth
        # write_uchars(f, (bitdepth,))
        # write number of coded frames
        # write_uints(f, (num_frames,))
        # write number of gop size
        # write_uints(f, (gop_size,))
        x_ref = None
        latent_prior_intra = None
        with tqdm(total=num_frames) as pbar:
            for i in range(num_frames):
                frm_enc_start = time.time()
                x_ori = read_frame_to_torch(
                    os.path.join(input, f"f{str(i+1).zfill(png_name_padding)}.png"), max_val).to(device)
                p = 64  
                x_cur = pad(x_ori, p)

                if i % gop_size == 0:
                    x_out, out_info = net.encode_keyframe(x_cur, latent_prior_intra, factor)
                    # cnt = write_body(f, out_info["shape"], out_info["strings"])
                    cnt = write_string(f, out_info["strings"])
                    latent_prior_inter = {"latent_prior_motion": None, "latent_prior_res": None}
                else:
                    # x_out, latent_prior_inter, out_info = net.encode_inter(x_cur, x_ref, latent_prior_inter, factor)
                    x_out, latent_prior_inter, out_info = net.encode_inter(x_cur, x_ref, latent_prior_inter, factor if i % 4 == 0 else factor+4)
                    cnt = 0
                    for shape, out in zip(
                        out_info["shape"].items(), out_info["strings"].items()
                    ):
                        # cnt += write_body(f, shape[1], out[1])
                        cnt += write_string(f, out[1])

                x_ref = x_out.clamp(0, 1)
                avg_frame_enc_time.append((time.time() - frm_enc_start))

                if i % gop_size == gop_size - 1:
                    latent_prior_intra = net.img_encoder(x_ref)

                x_hat = crop(x_ref, (height, width))

                if output_yuv is not None:
                    if Path(output_yuv).suffix == ".yuv":
                        rec = convert_rgb_yuv420(x_hat)
                        wopt = "wb" if i == 0 else "ab"
                        with Path(output_yuv).open(wopt) as fout:
                            write_frame(fout, rec, bitdepth)

                metrics = compute_metrics_for_frame(
                    x_ori, x_hat, max_val
                )
                msg = "{:d} {:s} {:.4f} {:.4f} {:.4f}".format(i, "I" if i % gop_size == 0 else "P", cnt/height/width, metrics["rgb_psnr"].data, metrics["msssim"].data)
                print(msg)
                log_fp.write(msg + '\n')
                for k, v in metrics.items():
                    results[k].append(v)

                pbar.update(1)

    f.close()
    log_fp.flush()

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }

    size = filesize(output_bits)
    seq_results["bpp"] = float(size) * 8 / (height * width * num_frames)
    seq_results["kbps"] = float(size) * 8 / (num_frames / framerate) / 1000
    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()

    with Path(output_json).open("wb") as f:
        f.write(json.dumps(seq_results, indent=2).encode())

    return seq_results


def _eval(input, num_frames, ckpt_dir, coder, device, output, gop_size, factor):

    compressai.set_entropy_coder(coder)
    enc_start = time.time()

    net = LVC_exp_spy_res()
    state_dict = torch.load(ckpt_dir)
    net.load_state_dict(state_dict)
    net.to(device).eval()

    if not os.path.exists(output):
        os.mkdir(output)
    rgb_psnr = []
    msssim = []
    bpp = []
    kbps = []
    data = open(output+"ave.txt", "a+")
    for seq in glob.glob(input+"*"):
        print(seq)
        video_start = time.time()
        pre_fix = output + seq.split('/')[-1]
        video_result = eval_video(seq, net, pre_fix + '.bits', pre_fix + '.yuv', pre_fix + '.json', pre_fix + '.txt', gop_size, num_frames, device, factor)
        video_end = time.time()
        rgb_psnr.append(video_result['rgb_psnr'])
        msssim.append(video_result['msssim'])
        bpp.append(video_result['bpp'])
        kbps.append(video_result['kbps'])
        print(seq.split('/')[-1])
        print("rgb_psnr:", video_result['rgb_psnr'])
        print("msssim:", video_result['msssim'])
        print("bpp:", video_result['bpp'])
        print("kbps:", video_result['kbps'])
        print("eval time:", video_end - video_start)
    enc_time = time.time() - enc_start
    print("rgb_psnr:", np.mean(rgb_psnr), file=data)
    print("msssim:", np.mean(msssim), file=data)
    print("bpp:", np.mean(bpp), file=data)
    print("kbps:", np.mean(kbps), file=data)
    print("Total time:", enc_time, file=data)

    total_bits = []
    for bits in glob.glob(output + "*.bits"):
        total_bits.append(filesize(bits))
    print("total mbits: ", round(float(sum(total_bits))/1e6, 3), file=data)
    data.close()


def encode(argv):
    parser = argparse.ArgumentParser(description="Encode image/video to bit-stream with checkpoint")
    parser.add_argument(
        "input",
        type=str,
        help="Input path, the first frame will be encoded with a NN image codec if the input is a raw yuv sequence",
    )
    parser.add_argument(
        "--dir",
        action="store_true",
        help="Input path is a dir where all videos need encoding",
    )
    parser.add_argument(
        "-f",
        "--num_frames",
        default=96,
        type=int,
        help="Number of frames to be coded. -1 will encode all frames of input (default: %(default)s)",
    )
    parser.add_argument(
        "-gop",
        "--gop_size",
        type=int,
        default=32,
        help="Group of pictures size (default: %(default)s)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=1,
        help="Quality factor (default: %(default)s)",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        help="Checkpoint to be used",
    )
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    args = parser.parse_args(argv)
    if not args.output:
        args.output = Path(Path(args.input).resolve().name).with_suffix(".bin")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    _encode(
        args.input,
        args.num_frames,
        args.ckpt_dir,
        args.coder,
        device,
        args.output,
        args.dir,
        args.gop_size,
        args.factor
    )


def decode(argv):
    parser = argparse.ArgumentParser(description="Decode bit-stream to image/video")
    parser.add_argument("input", type=str)   
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        help="Checkpoint to be used",
    )
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    args = parser.parse_args(argv)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    start_time = time.time()
    for bits in glob.glob(args.input+"*.bits"):
        print(bits)
        _decode(bits, args.ckpt_dir, args.coder, device, args.output + bits.split('/')[-1][:-5] + '.yuv')
    end_time = time.time() - start_time
    print("total decoding time: ", end_time)


def eval(argv):
    parser = argparse.ArgumentParser(description="Encode image/video to bit-stream with checkpoint")
    parser.add_argument(
        "input",
        type=str,
        help="Input path, the first frame will be encoded with a NN image codec if the input is a raw yuv sequence",
    )
    parser.add_argument(
        "-f",
        "--num_frames",
        default=-1,
        type=int,
        help="Number of frames to be coded. -1 will encode all frames of input (default: %(default)s)",
    )
    parser.add_argument(
        "-gop",
        "--gop_size",
        type=int,
        default=32,
        help="Group of pictures size (default: %(default)s)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=1,
        help="Quality factor (default: %(default)s)",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        help="Checkpoint to be used",
    )
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    args = parser.parse_args(argv)
    if not args.output:
        args.output = Path(Path(args.input).resolve().name).with_suffix(".bin")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    _eval(
        args.input,
        args.num_frames,
        args.ckpt_dir,
        args.coder,
        device,
        args.output,
        args.gop_size,
        args.factor
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("command", choices=["encode", "decode", "eval"])
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv[1:2])
    argv = argv[2:]
    torch.set_num_threads(1)  # just to be sure
    if args.command == "encode":
        encode(argv)
    elif args.command == "decode":
        decode(argv)
    elif args.command == "eval":
        eval(argv)


if __name__ == "__main__":
    main(sys.argv)