# img enc
CUDA_VISIBLE_DEVICES=0 python train_video_d_1.py -d /home/yangjy/dataset/vimeo_septuplet/ --cuda -e 125 --batch-size 16 --test-batch-size 64 
mv checkpoint_best_loss.pth.tar checkpoint_best_loss_d_1_125e.pth.tar
# I cI
CUDA_VISIBLE_DEVICES=0 python train_video_rd_1.py -d /home/yangjy/dataset/vimeo_septuplet/ --cuda -e 200 --batch-size 16 --test-batch-size 64 --max_frames 2 --checkpoint checkpoint_best_loss_d_1_125e.pth.tar 
mv checkpoint_best_loss.pth.tar checkpoint_best_loss_rd_1_200e.pth.tar
# I cI 1e-5
CUDA_VISIBLE_DEVICES=0 python train_video_rd_1.py -d /home/yangjy/dataset/vimeo_septuplet/ --cuda -e 250 --batch-size 16 --test-batch-size 64 --max_frames 2 --checkpoint checkpoint_best_loss_rd_1_200e.pth.tar -lr 1e-5
mv checkpoint_best_loss.pth.tar checkpoint_cI.pth.tar


# MEMC
CUDA_VISIBLE_DEVICES=0 python train_video_d_2.py -d /home/yangjy/dataset/vimeo_septuplet/ --cuda -e 50 --batch-size 16 --test-batch-size 64 
mv checkpoint_best_loss.pth.tar checkpoint_best_loss_d_2_50e.pth.tar
# mot enc
CUDA_VISIBLE_DEVICES=0 python train_video_d_3.py -d /home/yangjy/dataset/vimeo_septuplet/ --cuda -e 125 --batch-size 16 --test-batch-size 64 --checkpoint checkpoint_best_loss_d_2_50e.pth.tar 
mv checkpoint_best_loss.pth.tar checkpoint_best_loss_d_3_125e.pth.tar
# PF-QE
CUDA_VISIBLE_DEVICES=0 python train_video_d_4.py -d /home/yangjy/dataset/vimeo_septuplet/ --cuda -e 175 --batch-size 16 --test-batch-size 64 --checkpoint checkpoint_best_loss_d_3_125e.pth.tar 
mv checkpoint_best_loss.pth.tar checkpoint_best_loss_d_4_175e.pth.tar
# res enc
CUDA_VISIBLE_DEVICES=0 python train_video_d_5.py -d /home/yangjy/dataset/vimeo_septuplet/ --cuda -e 250 --batch-size 16 --test-batch-size 64 --checkpoint checkpoint_best_loss_d_4_175e.pth.tar 
mv checkpoint_best_loss.pth.tar checkpoint_best_loss_d_5_250e.pth.tar
# RF-QE
CUDA_VISIBLE_DEVICES=0 python train_video_d_6.py -d /home/yangjy/dataset/vimeo_septuplet/ --cuda -e 300 --batch-size 16 --test-batch-size 64 --checkpoint checkpoint_best_loss_d_5_250e.pth.tar 
mv checkpoint_best_loss.pth.tar checkpoint_best_loss_d_6_300e.pth.tar
# P cP
CUDA_VISIBLE_DEVICES=0 python train_video_rd_2.py -d /home/yangjy/dataset/vimeo_septuplet/ --cuda -e 325 --batch-size 16 --test-batch-size 64 --max_frames 3 --checkpoint checkpoint_best_loss_d_6_300e.pth.tar 
mv checkpoint_best_loss.pth.tar checkpoint_best_loss_rd_2_325e.pth.tar
# P cP cP cP
CUDA_VISIBLE_DEVICES=0 python train_video_rd_3.py -d /home/yangjy/dataset/vimeo_septuplet/ --cuda -e 375 --batch-size 16 --test-batch-size 64 --max_frames 5 --checkpoint checkpoint_best_loss_rd_2_325e.pth.tar 
mv checkpoint_best_loss.pth.tar checkpoint_best_loss_rd_3_375e.pth.tar
# P cP cP cP 1e-5
CUDA_VISIBLE_DEVICES=0 python train_video_rd_3.py -d /home/yangjy/dataset/vimeo_septuplet/ --cuda -e 400 --batch-size 16 --test-batch-size 64 --max_frames 5 --checkpoint checkpoint_best_loss_rd_3_375e.pth.tar -lr 1e-5
mv checkpoint_best_loss.pth.tar checkpoint_psnr.pth.tar
