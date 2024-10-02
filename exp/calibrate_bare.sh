#!/bin/bash
export PYTHONBREAKPOINT=ipdb.set_trace

# Set your base directory here
export BASE_DIR="PLEASE_SET_YOUR_BASE_DIR"

# ################################################################################
# # Feature Calibration
# ################################################################################
export TARGET_CLUSTER=5
export ALPHA=0.5
export TARGET_CLASS="ground-track-field"
export INPUT_PATH="${BASE_DIR}/feature/train"
export CLUSTER_BASE_PATH="${BASE_DIR}/clusters_kmeans_gpu_fp32"
export CLUSTER_ASSIGNMENTS="${CLUSTER_BASE_PATH}/pca1280_n32/cluster_assignments.json"
export CLUSTER_CENTERS="${CLUSTER_BASE_PATH}/pca1280_n32/cluster_centers.npy"
export SCALER_PATH="${CLUSTER_BASE_PATH}/scaler.joblib"
export CALI_FEATURE_PATH="${CLUSTER_BASE_PATH}/pca1280_n32/cali/${TARGET_CLASS}_c${TARGET_CLUSTER}_a${ALPHA}/train"

python ./scripts/feat_cluster_calib_v0.2.py \
    --input_path "$INPUT_PATH" \
    --output_path "$CALI_FEATURE_PATH" \
    --cluster_assignments "$CLUSTER_ASSIGNMENTS" \
    --cluster_centers "$CLUSTER_CENTERS" \
    --scaler_path "$SCALER_PATH" \
    --target_class "$TARGET_CLASS" \
    --target_cluster "$TARGET_CLUSTER" \
    --alpha "$ALPHA"
# --threshold "$THRESHOLD"

################################################################################
# Linear Probing
################################################################################
# Set your CSV paths here
export TRAIN_CSV="PLEASE_SET_YOUR_TRAIN_CSV_PATH"
export VAL_CSV="PLEASE_SET_YOUR_VAL_CSV_PATH"
export TEST_CSV="PLEASE_SET_YOUR_TEST_CSV_PATH"

export CUDA_VISIBLE_DEVICES=7
export ARCH="linprob"
export LR="1e-3"
export BATCH_SIZE=512
export FEAT_DIM=1280
export FEATURE_IDX=0
export FEATURE_DIR="${CLUSTER_BASE_PATH}/pca1280_n32/cali/${TARGET_CLASS}_c${TARGET_CLUSTER}_a${ALPHA}"
export VAL_TEST_DIR="${BASE_DIR}/feature"

for EPOCHS in 3 5 10 30; do
    export OUTPUT_PATH="${CLUSTER_BASE_PATH}/pca1280_n32/cali/${TARGET_CLASS}_c${TARGET_CLUSTER}_a${ALPHA}/linprob_bs${BATCH_SIZE}_ep${EPOCHS}"

    python -u ./scripts/dota_linprob_v0.2.py \
        --arch "$ARCH" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE" \
        --feat_dim "$FEAT_DIM" \
        --feature_idx "$FEATURE_IDX" \
        --feature_dir "$FEATURE_DIR" \
        --train_csv "$TRAIN_CSV" \
        --val_csv "$VAL_CSV" \
        --test_csv "$TEST_CSV" \
        --val_test_dir "$VAL_TEST_DIR" \
        --output_dir "$OUTPUT_PATH"
done

################################################################################
# EOF: Telegram notification
################################################################################
BOT_TOKEN="YOUR_BOT_TOKEN"
CHAT_ID="YOUR_CHAT_ID"
MESSAGE="Your bash script has finished running."

curl -s -X POST https://api.telegram.org/bot$BOT_TOKEN/sendMessage \
    -d chat_id=$CHAT_ID \
    -d text="$MESSAGE" \
    >/dev/null 2>&1
