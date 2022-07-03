#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
	--loss_type HardestContrastiveLoss \
	--model ResUNet_spconv \
	--num_workers 6 \
	--lr 1e-4
