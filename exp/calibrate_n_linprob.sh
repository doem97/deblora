#!/bin/bash
export PYTHONBREAKPOINT=""
# export PYTHONBREAKPOINT=ipdb.set_trace

################################################################################
# Global Variables
################################################################################
BASE_DIR="PLEASE_SET_YOUR_BASE_DIR"
export FEAT_DIM=1280
export TARGET_CLASS="ground-track-field"
export INPUT_PATH="${BASE_DIR}/feature/train"
export CLUSTER_BASE_PATH="${BASE_DIR}/clusters_kmeans_gpu_fp32"
export CLUSTER_ASSIGNMENTS="${CLUSTER_BASE_PATH}/pca${FEAT_DIM}_n32/cluster_assignments.json"
export CLUSTER_CENTERS="${CLUSTER_BASE_PATH}/pca${FEAT_DIM}_n32/cluster_centers.npy"
export SCALER_PATH="${CLUSTER_BASE_PATH}/scaler.joblib"
export VAL_CSV="PLEASE_SET_YOUR_VAL_CSV_PATH"
export TEST_CSV="PLEASE_SET_YOUR_TEST_CSV_PATH"
export ARCH="linprob"
export LR="1e-3"
export BATCH_SIZE=512
export FEATURE_IDX=0
export VAL_TEST_DIR="${BASE_DIR}/feature"
export CALI_DIR="${CLUSTER_BASE_PATH}/pca${FEAT_DIM}_n32/cali"
export LIST_FILE="${BASE_DIR}/calibrated_folders.txt"

################################################################################
# Helper Functions
################################################################################
send_telegram_notification() {
    local MESSAGE="$1"
    local BOT_TOKEN="YOUR_BOT_TOKEN"
    local CHAT_ID="YOUR_CHAT_ID"
    curl -s -X POST https://api.telegram.org/bot${BOT_TOKEN}/sendMessage \
        -d chat_id=${CHAT_ID} \
        -d text="${MESSAGE}" \
        >/dev/null 2>&1
}

################################################################################
# Feature Calibration and Combination
################################################################################
calibrate_and_combine_features() {
    echo "Starting feature calibration and combination..."
    >"${LIST_FILE}" # Clear the list file

    for folder in "${CALI_DIR}"/ground-track-field_*; do
        if [ -d "$folder" ]; then
            CALI_FEATURE_PATH="${folder}/train"
            if [ -f "${CALI_FEATURE_PATH}/calibrated_train.csv" ]; then
                if [ ! -f "${CALI_FEATURE_PATH}/ori_calibrated_train.csv" ]; then
                    echo "Combining original GTF features with calibrated features for ${folder}"
                    cat "${CALI_FEATURE_PATH}/calibrated_train.csv" "${BASE_DIR}/ori_gtf.csv" >"${CALI_FEATURE_PATH}/ori_calibrated_train.csv"
                    echo "Created combined file: ${CALI_FEATURE_PATH}/ori_calibrated_train.csv"
                fi
                echo "${folder}" >>"${LIST_FILE}"
            fi
        fi
    done

    echo "Calibration and combination completed. List saved to ${LIST_FILE}"
}

################################################################################
# Linear Probing
################################################################################
run_linear_probing() {
    local GPU_ID=$1
    shift
    local FOLDERS=("$@")

    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    for folder in "${FOLDERS[@]}"; do
        COMBINED_CSV="${folder}/train/ori_calibrated_train.csv"
        OUTPUT_PATH="${folder}/comb_linprob_bs${BATCH_SIZE}_ep10"

        if [ ! -f "${OUTPUT_PATH}/predictions.csv" ]; then
            echo "GPU ${GPU_ID} - Running linear probing for ${folder}"
            python -u ./scripts/dota_linprob_v0.2.py \
                --arch "${ARCH}" \
                --epochs 10 \
                --lr "${LR}" \
                --batch_size "${BATCH_SIZE}" \
                --feat_dim "${FEAT_DIM}" \
                --feature_idx "${FEATURE_IDX}" \
                --feature_dir "" \
                --train_csv "${COMBINED_CSV}" \
                --val_csv "${VAL_CSV}" \
                --test_csv "${TEST_CSV}" \
                --val_test_dir "${VAL_TEST_DIR}" \
                --output_dir "${OUTPUT_PATH}"
        else
            echo "GPU ${GPU_ID} - Skipping linear probing for ${folder} (already exists)"
        fi
    done
}

################################################################################
# Main Execution
################################################################################
# Step 1: Calibrate and combine features
calibrate_and_combine_features

# Step 2: Read the list of folders
CALIBRATED_FOLDERS=($(cat "${LIST_FILE}"))

# Step 3: Distribute folders across GPUs
NUM_GPUS=8
FOLDERS_PER_GPU=$((${#CALIBRATED_FOLDERS[@]} / NUM_GPUS + 1))

for ((i = 0; i < NUM_GPUS; i++)); do
    START=$((i * FOLDERS_PER_GPU))
    END=$((START + FOLDERS_PER_GPU))
    SUBSET=("${CALIBRATED_FOLDERS[@]:START:FOLDERS_PER_GPU}")
    run_linear_probing $i "${SUBSET[@]}" &
done

# Wait for all background processes to finish
wait

################################################################################
# EOF: Telegram notification
################################################################################
send_telegram_notification "Your bash script has finished running."
