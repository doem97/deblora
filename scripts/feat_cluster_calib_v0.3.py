"""
NAME: feat_cluster_calib_v0.2.py
ENV: rapids
VERSION: 0.3
AUTHOR: @doem1997

Description:
    This script performs feature calibration on feature vectors using single
    target cluster centers. It uses the saved scaler to de-normalize the
    cluster centers, and then interpolates between the original target
    features and the de-normalized cluster center.

CHANGELOG:
    v0.3: Use csv instead of copying files, save abs path
    v0.2: Add multiprocessing to speed up the feature calibration process
    v0.1: Initial version
"""

import argparse
import json
import os
import signal
import sys

import joblib
import numpy as np
import torch
from tqdm import tqdm

import multiprocessing
import shutil
from functools import partial
import csv

###############################################################################
# Setup: Logger, Exception Handling (Exit Signal), and Global Seeds
###############################################################################
sys.path.append("/workspace/dso/gensar/dift/src")
# available log levels: debug, info, success, warning, error, critical, section
from utils.logger import CustomLogger
from utils.error_handler import excepthook, signal_handler


def setup_logger(output_path, log_file_name="feature_calibration.log"):
    return CustomLogger("main_logger", output_path, log_file_name)


def setup_error_handling():
    sys.excepthook = excepthook
    signal.signal(signal.SIGINT, signal_handler)


def setup_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


###############################################################################
# Feature Calibration Functions
###############################################################################
def load_cluster_assignments(cluster_assignments_file):
    with open(cluster_assignments_file, "r") as f:
        return json.load(f)


def load_cluster_centers(cluster_centers_file):
    return np.load(cluster_centers_file)


def load_scaler(scaler_path):
    return joblib.load(scaler_path)


def denormalize_cluster_center(cluster_center, scaler):
    # Reshape cluster_center to 2D array for inverse_transform
    cluster_center_2d = cluster_center.reshape(1, -1)
    denormalized_center = scaler.inverse_transform(cluster_center_2d)
    return denormalized_center.flatten()


def calibrate_features(features, cluster_center, alpha=0.5, threshold=None):
    """
    Calibrate features by moving them towards the cluster center.

    Args:
    features (np.ndarray): Original features
    cluster_center (np.ndarray): Center of the selected cluster
    alpha (float): Interpolation factor (0 <= alpha <= 1)
    threshold (float): Threshold for distance between features and cluster center

    Returns:
    np.ndarray: Calibrated features
    """
    if threshold is not None:
        distance = np.linalg.norm(features - cluster_center)
        if distance < threshold:
            return features  # No calibration needed
    return (1 - alpha) * features + alpha * cluster_center


# Move these functions outside of process_features
def calibrate_file(
    file_path,
    input_path,
    output_path,
    scaler,
    cluster_center,
    alpha,
    threshold,
):
    relative_path = os.path.relpath(file_path, input_path)
    output_file_path = os.path.join(output_path, relative_path)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Load the original feature
    feature = torch.load(file_path, map_location="cpu", weights_only=True)
    original_shape = feature.shape

    # Reshape to 2D for scaler operations
    feature_np = feature.view(1, -1).numpy()

    # Normalize the feature
    normalized_feature = scaler.transform(feature_np)

    # Reshape cluster center to match feature shape
    cluster_center = cluster_center.reshape(normalized_feature.shape)

    # Calibrate the normalized feature
    calibrated_feature = calibrate_features(
        normalized_feature,
        cluster_center,
        alpha,
        threshold,
    )

    # De-normalize the calibrated feature
    calibrated_feature = scaler.inverse_transform(calibrated_feature)

    # Convert back to tensor and reshape to original shape
    calibrated_feature = torch.from_numpy(calibrated_feature).view(
        original_shape
    )

    # Save the calibrated feature
    torch.save(calibrated_feature, output_file_path)


def copy_file(file_path, input_path, output_path):
    relative_path = os.path.relpath(file_path, input_path)
    output_file_path = os.path.join(output_path, relative_path)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    shutil.copy2(file_path, output_file_path)


def process_features(
    input_path,
    output_path,
    cluster_assignments,
    cluster_centers,
    scaler,
    target_class,
    target_cluster,
    alpha,
    threshold,
):

    logger.info(
        f"Processing features for {target_class} using cluster {target_cluster}"
    )

    # Get the center of the target cluster
    logger.section("Initialize the cluster centers")
    cluster_center = cluster_centers[target_cluster]
    logger.info(
        f"Cluster center shape: {cluster_center.shape}, dtype: {cluster_center.dtype}"
    )

    logger.section("Create lists for calibration and original features")
    files_to_calibrate = []
    files_to_keep = []

    for assignment in cluster_assignments:
        file_path = os.path.join(input_path, assignment["file"])
        if assignment["category"] == target_class:
            files_to_calibrate.append(file_path)
        else:
            files_to_keep.append(file_path)

    logger.info(f"Number of files to calibrate: {len(files_to_calibrate)}")
    logger.info(f"Number of files to keep: {len(files_to_keep)}")

    # 使用 partial 创建带有预填充参数的函数
    calibrate_file_partial = partial(
        calibrate_file,
        input_path=input_path,
        output_path=output_path,
        scaler=scaler,
        cluster_center=cluster_center,
        alpha=alpha,
        threshold=threshold,
    )
    logger.info("Created partial function for file calibration")

    # 使用多进程来并行校准文件
    logger.section("Feature Calibration")
    with multiprocessing.Pool() as pool:
        list(
            tqdm(
                pool.imap(calibrate_file_partial, files_to_calibrate),
                total=len(files_to_calibrate),
                desc="Calibrating files",
            )
        )
    logger.success(
        f"Calibrated features saved to {os.path.abspath(output_path)}"
    )

    # Generate combined CSV file
    logger.section("Generating combined CSV file")
    generate_combined_csv(
        files_to_keep,
        files_to_calibrate,
        os.path.join(output_path, "calibrated_train.csv"),
        input_path,
        output_path,
    )

    logger.success(
        f"Finished processing features for {target_class} saved to:"
    )
    logger.success(
        f"{os.path.abspath(os.path.join(output_path, 'calibrated_train.csv'))}"
    )


def generate_combined_csv(
    original_files, calibrated_files, output_csv, input_path, output_path
):
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path", "category"])

        # Write original files
        for file_path in original_files:
            abs_path = os.path.abspath(file_path)
            category = os.path.basename(os.path.dirname(file_path))
            writer.writerow([abs_path, category])

        # Write calibrated files
        for file_path in calibrated_files:
            calibrated_rel_path = os.path.relpath(file_path, input_path)
            calibrated_abs_path = os.path.abspath(
                os.path.join(output_path, calibrated_rel_path)
            )
            category = os.path.basename(os.path.dirname(file_path))
            writer.writerow([calibrated_abs_path, category])


###############################################################################
# Main Function
###############################################################################
def main(args):
    global logger
    logger = setup_logger(args.output_path)
    setup_seeds(args.random_seed)

    logger.info("Starting feature calibration process")
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Cluster assignments file: {args.cluster_assignments}")
    logger.info(f"Cluster centers file: {args.cluster_centers}")
    logger.info(f"Scaler file: {args.scaler_path}")
    logger.info(f"Target class: {args.target_class}")
    logger.info(f"Target cluster: {args.target_cluster}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info(f"Threshold: {args.threshold}")

    # Load cluster assignments and centers
    cluster_assignments = load_cluster_assignments(args.cluster_assignments)
    cluster_centers = load_cluster_centers(args.cluster_centers)
    scaler = load_scaler(args.scaler_path)
    # Process and calibrate features
    process_features(
        args.input_path,
        args.output_path,
        cluster_assignments,
        cluster_centers,
        scaler,
        args.target_class,
        args.target_cluster,
        args.alpha,
        args.threshold,
    )

    logger.success("Feature calibration completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate features based on cluster centers"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input features folder",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the calibrated features",
    )
    parser.add_argument(
        "--cluster_assignments",
        type=str,
        required=True,
        help="Path to the cluster assignments JSON file",
    )
    parser.add_argument(
        "--cluster_centers",
        type=str,
        required=True,
        help="Path to the cluster centers numpy file",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        required=True,
        help="Path to the saved scaler file",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default="ground-track-field",
        help="Target class for calibration",
    )
    parser.add_argument(
        "--target_cluster",
        type=int,
        default=5,
        help="Target cluster for calibration",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Interpolation factor for calibration",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for distance between features and cluster center",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    main(args)
