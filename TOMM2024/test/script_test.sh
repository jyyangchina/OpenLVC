
# test rate-distortion performance
CUDA_VISIBLE_DEVICES=0 python codec.py eval /home/yangjy/dataset/UVG_1080P_png/ --ckpt_dir updated_model_psnr.pth.tar -o /home/yangjy/dataset/out/UVG_1080P/opensource_test/ --cuda --gop_size 32 -f 96 --factor 1

# encoding and decoding
CUDA_VISIBLE_DEVICES=0 python codec.py encode /home/yangjy/dataset/UVG_1080P_png/ --dir --ckpt_dir updated_model_psnr.pth.tar -o /home/yangjy/dataset/out/UVG_1080P/opensource_test/ --cuda --gop_size 32 -f 96 --factor 1
CUDA_VISIBLE_DEVICES=0 python codec.py decode /home/yangjy/dataset/out/UVG_1080P/opensource_test/ --ckpt_dir updated_model_psnr.pth.tar -o /home/yangjy/dataset/out/UVG_1080P/opensource_test/ --cuda 