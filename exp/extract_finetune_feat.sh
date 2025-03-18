#!/bin/bash

export IMG_SIZE=512
export ENSEMBLE_SIZE=8
export SD_ID="runwayml/stable-diffusion-v1-5"
export INPUT_PATH="/workspace/data/DOTA_v2/image_folder/hf_format"
export USE_CAT_PROMPT=0
# export TIME_STEPS="100_200_300"
export TIME_STEPS="50_100_500"
export OUTPUT_PATH="./output/features/sd15_ft_mid_t${TIME_STEPS}"

for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python ./scripts/dota_extract_feat_mid.py \
        --img_size $IMG_SIZE \
        --sd_id $SD_ID \
        --time_steps ${TIME_STEPS//_/ } \
        --ensemble_size $ENSEMBLE_SIZE \
        --input_path $INPUT_PATH \
        --n_workers 16 --worker_idx $i \
        --prompt_template "an ORS image of category_name" \
        --output_path $OUTPUT_PATH \
        --use_cat_prompt $USE_CAT_PROMPT &
done
wait
