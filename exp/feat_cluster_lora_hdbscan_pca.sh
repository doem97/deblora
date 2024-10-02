#!/bin/bash
export PYTHONBREAKPOINT=ipdb.set_trace

# # ################################################################################
# # # Feature Clustering for LoRA: HDBSCAN, Single GPU Version, FP16, PCA 1280, no sampling
# # ################################################################################

# INPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/feature/train"
# OUTPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_hdbscan_gpu/pca1280_min15_eom0.0/train_v0.3"
# MIN_CLUSTER_SIZE=15
# MIN_SAMPLES=15
# BATCH_SIZE=3000
# PCA_DIM=1280                   # from 1280*8*8 to 1280
# CLUSTER_SELECTION_METHOD="eom" # between "leaf" and "eom", by default "eom"
# CLUSTER_SELECTION_EPSILON=0.0  # from 0.0 to 1.0, by default 0.0
# SCALER_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_hdbscan_gpu/scaler.joblib"
# PCA_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_hdbscan_gpu/pca_1280.joblib"

# python ./scripts/feat_cluster_hdbscan_gpu_fp16_v0.3.py \
#     --input_path "$INPUT_PATH" \
#     --output_path "$OUTPUT_PATH" \
#     --min_cluster_size "$MIN_CLUSTER_SIZE" \
#     --min_samples "$MIN_SAMPLES" \
#     --batch_size "$BATCH_SIZE" \
#     --pca_dim "$PCA_DIM" \
#     --scaler_path "$SCALER_PATH" \
#     --pca_path "$PCA_PATH" \
#     --cluster_selection_method "$CLUSTER_SELECTION_METHOD" \
#     --cluster_selection_epsilon "$CLUSTER_SELECTION_EPSILON" \
#     --random_seed 42
# # Not used options:
# # --no_normalize

# # PCA 1280
# INPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/feature/train"
# BASE_OUTPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_hdbscan_gpu/pca1280_min5_eom"
# MIN_CLUSTER_SIZE=5
# MIN_SAMPLES=5
# BATCH_SIZE=3000
# PCA_DIM=1280
# CLUSTER_SELECTION_METHOD="eom"
# SCALER_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_hdbscan_gpu/scaler.joblib"
# PCA_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_hdbscan_gpu/pca_1280.joblib"

# for EPSILON in 0.0; do
#     OUTPUT_PATH="${BASE_OUTPUT_PATH}${EPSILON}/"

#     echo "Running experiment with epsilon = $EPSILON"

#     python ./scripts/feat_cluster_hdbscan_gpu_fp16_v0.3.py \
#         --input_path "$INPUT_PATH" \
#         --output_path "$OUTPUT_PATH" \
#         --min_cluster_size "$MIN_CLUSTER_SIZE" \
#         --min_samples "$MIN_SAMPLES" \
#         --batch_size "$BATCH_SIZE" \
#         --pca_dim "$PCA_DIM" \
#         --scaler_path "$SCALER_PATH" \
#         --pca_path "$PCA_PATH" \
#         --cluster_selection_method "$CLUSTER_SELECTION_METHOD" \
#         --cluster_selection_epsilon "$EPSILON" \
#         --random_seed 42
# done

# # Grid search for PCA dimensions
# INPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/feature/train"
# BASE_OUTPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_hdbscan_gpu"
# MIN_CLUSTER_SIZE=15
# MIN_SAMPLES=15
# BATCH_SIZE=3000
# CLUSTER_SELECTION_METHOD="eom"
# SCALER_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_hdbscan_gpu/scaler.joblib"
# EPSILON=0.0

# for PCA_DIM in 2560 5120 10240; do
#     OUTPUT_PATH="${BASE_OUTPUT_PATH}/pca_${PCA_DIM}_min15_eom0.0/"
#     PCA_PATH="${BASE_OUTPUT_PATH}/pca_${PCA_DIM}.joblib"

#     echo "Running experiment with PCA_DIM = $PCA_DIM"

#     python ./scripts/feat_cluster_hdbscan_gpu_fp16_v0.3.py \
#         --input_path "$INPUT_PATH" \
#         --output_path "$OUTPUT_PATH" \
#         --min_cluster_size "$MIN_CLUSTER_SIZE" \
#         --min_samples "$MIN_SAMPLES" \
#         --batch_size "$BATCH_SIZE" \
#         --pca_dim "$PCA_DIM" \
#         --scaler_path "$SCALER_PATH" \
#         --pca_path "$PCA_PATH" \
#         --cluster_selection_method "$CLUSTER_SELECTION_METHOD" \
#         --cluster_selection_epsilon "$EPSILON" \
#         --random_seed 42
# done

# # USE PCA 5120 only, turn to fp32 version.
# INPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/feature/train"
# BASE_OUTPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_hdbscan_gpu_fp32"
# MIN_CLUSTER_SIZE=100
# MIN_SAMPLES=$MIN_CLUSTER_SIZE
# MAX_CLUSTER_SIZE=10000 # sample_number / 15, should lead to >15 clusters
# BATCH_SIZE=3000
# CLUSTER_SELECTION_METHOD="eom"
# SCALER_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_hdbscan_gpu_fp32/scaler.joblib"
# EPSILON=0.0
# PCA_DIM=5120
# OUTPUT_PATH="${BASE_OUTPUT_PATH}/pca_${PCA_DIM}_min${MIN_CLUSTER_SIZE}_max${MAX_CLUSTER_SIZE}_eom${EPSILON}/"
# PCA_PATH="${BASE_OUTPUT_PATH}/pca_${PCA_DIM}.joblib"

# python ./scripts/feat_cluster_hdbscan_gpu_fp32_v0.4.py \
#     --input_path "$INPUT_PATH" \
#     --output_path "$OUTPUT_PATH" \
#     --min_cluster_size "$MIN_CLUSTER_SIZE" \
#     --min_samples "$MIN_SAMPLES" \
#     --max_cluster_size "$MAX_CLUSTER_SIZE" \
#     --batch_size "$BATCH_SIZE" \
#     --pca_dim "$PCA_DIM" \
#     --scaler_path "$SCALER_PATH" \
#     --pca_path "$PCA_PATH" \
#     --cluster_selection_method "$CLUSTER_SELECTION_METHOD" \
#     --cluster_selection_epsilon "$EPSILON" \
#     --random_seed 42

# Kmeans: Grid search for PCA dimensions: 10240, 20480, 40960, 81920 (full)
INPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/feature/train"
BASE_OUTPUT_PATH="/workspace/dso/gensar/dift/output/dotav1/sd15+lora_mid_t1/clusters_kmeans_gpu_fp32"
N_CLUSTERS=32
BATCH_SIZE=81920
SCALER_PATH="${BASE_OUTPUT_PATH}/scaler.joblib"

for PCA_DIM in 5120 10240 20480; do
    OUTPUT_PATH="${BASE_OUTPUT_PATH}/pca${PCA_DIM}_n${N_CLUSTERS}/"
    PCA_PATH="${BASE_OUTPUT_PATH}/pca_${PCA_DIM}.joblib"

    echo "Running experiment with PCA_DIM = $PCA_DIM"

    python ./scripts/feat_cluster_kmeans_gpu_fp32_v0.3.py \
        --input_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --n_clusters "$N_CLUSTERS" \
        --batch_size "$BATCH_SIZE" \
        --pca_dim "$PCA_DIM" \
        --scaler_path "$SCALER_PATH" \
        --pca_path "$PCA_PATH" \
        --random_seed 42
done

################################################################################
# Telegram notification
################################################################################
# Pose finish message to my TG bot
BOT_TOKEN="7254878605:AAGHLOnoaj8W3oGUl-BbWlywnuSXSOWKOb0"
CHAT_ID="6148817210"
MESSAGE="Your bash script has finished running."

curl -s -X POST https://api.telegram.org/bot$BOT_TOKEN/sendMessage \
    -d chat_id=$CHAT_ID \
    -d text="$MESSAGE" \
    >/dev/null 2>&1
