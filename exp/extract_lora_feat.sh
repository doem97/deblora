#!/bin/bash

export IMG_SIZE=512
export ENSEMBLE_SIZE=8
export SD_ID="runwayml/stable-diffusion-v1-5"
export INPUT_PATH="/workspace/data/DOTA_v2/image_folder/hf_format"
export USE_CAT_PROMPT=0
export TIME_STEPS="1"
export LORA_PATH="/workspace/dso/gensar/lora/output/dotav1/sd15_lora/512_fp16_lr1e-3_ep30_wu0_bs32_r8_ORS"
export OUTPUT_PATH="./output/features/sd15_lora_mid_t${TIME_STEPS}"

for i in {4..7}; do
    CUDA_VISIBLE_DEVICES=$i python ./scripts/dota_extract_feat_mid+lora.py \
        --img_size $IMG_SIZE \
        --sd_id $SD_ID \
        --time_steps ${TIME_STEPS//_/ } \
        --ensemble_size $ENSEMBLE_SIZE \
        --input_path $INPUT_PATH \
        --n_workers 4 --worker_idx $((i - 4)) \
        --prompt_template "an ORS image of category_name" \
        --output_path $OUTPUT_PATH \
        --use_cat_prompt $USE_CAT_PROMPT \
        --lora_path $LORA_PATH &
done
wait
