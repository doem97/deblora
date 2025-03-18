#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export LR=0.001
export EPOCHS=30
export BATCH_SIZE=512
export TRAIN_CSV="/workspace/data/DOTA_v2/image_folder/hf_format/meta/train.csv"
export VAL_CSV="/workspace/data/DOTA_v2/image_folder/hf_format/meta/val.csv"

python -u ./scripts/dota_linprob_v0.2.py --arch "linprob" \
    --epochs "$EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" --feat_dim 1280 --feature_idx 0 \
    --dl_workers 4 --random_seed 42 \
    --feature_dir "./output/features/sd15_0shot_mid_t1" \
    --train_csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --test_csv "$VAL_CSV" \
    --output_dir "./output/sd15_0shot_mid_t1/linprob_bs512_lr${LR}_ep30"
