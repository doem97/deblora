#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -u ./scripts/dota_linprob_v0.2.py \
    --arch "linprob" \
    --epochs 30 \
    --lr 0.001 \
    --batch_size 512 \
    --feat_dim 1280 \
    --feature_idx 0 \
    --dl_workers 4 \
    --random_seed 42 \
    --feature_dir "./output/features/sd15+lora_mid_t1" \
    --train_csv "/workspace/data/DOTA_v2/image_folder/hf_format/meta/train.csv" \
    --val_csv "/workspace/data/DOTA_v2/image_folder/hf_format/meta/val.csv" \
    --test_csv "/workspace/data/DOTA_v2/image_folder/hf_format/meta/val.csv" \
    --output_dir "./output/dotav1/sd15+lora_mid_t1/linprob_bs512_lr0.001_ep30"
