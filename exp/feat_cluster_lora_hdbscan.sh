#!/bin/bash

################################################################################
# Feature Clustering for LoRA: HDBSCAN, Single GPU Version, FP16
################################################################################
export PYTHONBREAKPOINT=ipdb.set_trace

INPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/feature/train"
OUTPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_hdbscan_gpu/train"
MIN_CLUSTER_SIZE=5
MIN_SAMPLES=5
BATCH_SIZE=3000
SAMPLE_PERCENTAGE=20

python ./scripts/feat_cluster_hdbscan_gpu_fp16_v0.1.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --min_cluster_size "$MIN_CLUSTER_SIZE" \
    --min_samples "$MIN_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --sample_percentage "$SAMPLE_PERCENTAGE" \
    --random_seed 42
# Not used options:
# --no_normalize
