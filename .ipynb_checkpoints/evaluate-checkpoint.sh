#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python benchmark_3dmatch.py --source /data/ghn/FCGF_data/threedmatch_test --target ./features_tmp --num_rand_keypoints 5000 \
              --voxel_size 0.025 --model ./snapshot/HardestContrastiveLoss_11271549/checkpoints/model_best_recall.pth \
              --evaluate_fmr
