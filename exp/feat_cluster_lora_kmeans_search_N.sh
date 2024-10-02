#!/bin/bash
export PYTHONBREAKPOINT=ipdb.set_trace

# Kmeans: Clustering with PCA 1280
BASE_DIR="PLEASE_SET_BASE_DIR"
INPUT_PATH="${BASE_DIR}/feature/train"
BASE_OUTPUT_PATH="${BASE_DIR}/clusters_kmeans_gpu_fp32"
BATCH_SIZE=3000
SCALER_PATH="${BASE_OUTPUT_PATH}/scaler.joblib"

# Fixed PCA dimension
PCA_DIM=1280
PCA_PATH="${BASE_OUTPUT_PATH}/pca_${PCA_DIM}.joblib"

# Grid search for number of clusters
N_CLUSTERS=32

OUTPUT_PATH="${BASE_OUTPUT_PATH}/pca${PCA_DIM}_n${N_CLUSTERS}/"

echo "Running experiment with PCA_DIM = $PCA_DIM and N_CLUSTERS = $N_CLUSTERS"

python ./scripts/feat_cluster_kmeans_gpu_v0.4.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --n_clusters "$N_CLUSTERS" \
    --batch_size "$BATCH_SIZE" \
    --scaler_path "$SCALER_PATH" \
    --pca_dim "$PCA_DIM" \
    --pca_path "$PCA_PATH" \
    --random_seed 42

################################################################################
# Telegram notification
################################################################################
# Post finish message to TG bot
BOT_TOKEN="YOUR_BOT_TOKEN"
CHAT_ID="YOUR_CHAT_ID"
MESSAGE="Your bash script has finished running."

curl -s -X POST https://api.telegram.org/bot$BOT_TOKEN/sendMessage \
    -d chat_id=$CHAT_ID \
    -d text="$MESSAGE" \
    >/dev/null 2>&1
