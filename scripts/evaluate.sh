#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python test_3dmatch.py \
--root /xxx/FCGF_data/threedmatch_test \
--voxel_size 0.025 \
--checkpoint snapshot/xxxxxxxx/checkpoints/checkpoint.pth \
--num_points 5000 \
--num_workers 8 \
