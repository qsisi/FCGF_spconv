#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--root /xxx/FCGF_data/threedmatch \
--voxel_size 0.025 \
--max_epoch 30 \
--num_workers 8 \
