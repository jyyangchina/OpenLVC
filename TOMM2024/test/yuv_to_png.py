import argparse
import glob
import os
import subprocess
from concurrent import futures

from typing import Sequence


def _convert_yuvs_to_pngs(yuv_path: str, png_out_root: str):
    yuv_name = os.path.basename(os.path.splitext(yuv_path)[0])
    out_path_base = os.path.join(
        png_out_root, yuv_name)
    os.makedirs(out_path_base, exist_ok=True)
    # Beauty_1920x1080_120fps_8bit_P420
    # ffmpeg -pix_fmt yuv420p -s WidthxHeight -i Name.yuv -vframes Frame path_to_PNG/f%03d.png
    wid_hei = yuv_name.split('_')[1] 
    subprocess.run([
        "ffmpeg", "-pix_fmt", "yuv420p", "-s", wid_hei, "-i", yuv_path, out_path_base + "/f%03d.png"
    ], check=True, capture_output=True)

    return os.path.dirname(out_path_base)


def convert_yuvs_to_pngs(yuv_paths: Sequence[str],
                           png_out_root: str,
                           num_processes: int = 8):
    print(f"Starting conversion with {num_processes} processes...")
    pool = futures.ProcessPoolExecutor(num_processes)
    futs = []
    for yuv_path in yuv_paths:
        futs.append(
            pool.submit(_convert_yuvs_to_pngs, 
                        yuv_path, png_out_root=png_out_root))
    for i, fut in enumerate(futures.as_completed(futs), 1):
        out_p = fut.result()
        print(f"{i}/{len(futs)}: {out_p}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("yuv_dir",
                   help="Path to yuv file.")
    p.add_argument("--out_dir", help="Root dir of where to store PNGs.",
                   required=True)
    p.add_argument("--num_processes", type=int, default=8,
                   help="How many processes (cores) to use for conversion.")
    flags = p.parse_args()

    yuv_dir = flags.yuv_dir
    print(f"Checking for yuv files at {yuv_dir}, ")
    
    yuv_paths = sorted(
        glob.glob(os.path.join(yuv_dir, "*.yuv")))

    print(f"Found {len(yuv_paths)} yuvs...")
    convert_yuvs_to_pngs(yuv_paths, flags.out_dir, flags.num_processes)


if __name__ == "__main__":
    main()