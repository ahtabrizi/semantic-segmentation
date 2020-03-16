#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py \
  --dataset comma10k \
  --cv 2 \
  --arch network.deepv3.DeepWV3Plus \
  --class_uniform_pct 0.0 \
  --class_uniform_tile 300 \
  --lr 2e-2 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --adam \
  --crop_size 360 \
  --scale_min 1.0 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --max_epoch 90 \
  --img_wt_loss \
  --wt_bound 1.0 \
  --bs_mult 2 \
  --exp comma10k \
  --ckpt ./logs/ \
  --tb_path ./logs/

  #--lr 0.001 \
  #--snapshot ./pretrained_models/cityscapes_best.pth \
  #--apex \
  #--syncbn \
