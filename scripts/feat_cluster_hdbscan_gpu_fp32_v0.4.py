# Note: This HDBSCAN version is stable, but the clusters are too small and
#   can not form centers with both head ant tail classes. So we turn to kmeans
#   instead.
# Environment: rapids
# v0.4:
#  - use float32 instead of float16
# v0.3:
#  - use of cluster_selection_epsilon (note: will not be effective when
#    doing prediction later, as the approximate_predict function is not aware
#    of this argument)
#  - due to above, use no sampling and on-the-fly
#    hdbscan (mean, the hdbscan will run on entire dataset to
#    compute global cluster centers)


import argparse
import json
import os
import pdb
import sys
import time
from collections import defaultdict

import cupy as cp
import joblib
import numpy as np
import pandas as pd
import torch
from cuml.cluster import HDBSCAN
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append("/workspace/dso/gensar/dift/src")
from utils.logger import CustomLogger


# Setup logger and pdb tracebacks
def setup_logger(output_path, log_file_name="clustering.log"):
    return CustomLogger("main_logger", output_path, log_file_name)


def excepthook(type, value, traceback):
    print("An error occurred:", value)
    pdb.post_mortem(traceback)


sys.excepthook = excepthook

# Global variables
COMPUTE_GLOBAL_NORM = False
COMPUTE_GLOBAL_PCA = False


# Set all seeds
def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    cp.random.seed(seed)


# Load and preprocess batch
def load_and_preprocess_batch(file_paths, batch_size):
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i : i + batch_size]
        features = []
        for file_path in batch:
            feature = (
                torch.load(file_path, map_location="cpu", weights_only=True)
                .squeeze(0)
                .reshape(-1)
                .numpy()
            )
            features.append(feature)
        yield np.array(features), batch


# Compute cluster representatives
def compute_cluster_representatives(features, labels):
    unique_labels = cp.unique(labels)
    representatives = []

    for label in unique_labels:
        if label != -1:  # Exclude noise points
            cluster_points = features[labels == label]
            centroid = cp.mean(cluster_points, axis=0)

            # Find the point closest to the centroid
            distances = cp.sum((cluster_points - centroid) ** 2, axis=1)
            closest_point_idx = cp.argmin(distances)
            representative = cluster_points[closest_point_idx]

            representatives.append(representative)

    return cp.array(representatives)


# Function to get current GPU memory usage
def get_gpu_memory_usage():
    total_memory = cp.cuda.Device(0).mem_info[1]
    free_memory = cp.cuda.Device(0).mem_info[0]
    used_memory = total_memory - free_memory
    return f"{used_memory / 1024**3:.2f}/{total_memory / 1024**3:.2f} GB"


# Cluster features
def cluster_features(
    input_path,
    output_path,
    min_cluster_size=5,
    min_samples=5,
    max_cluster_size=0,  # 新增参数
    cluster_selection_method="eom",
    cluster_selection_epsilon=0.0,
    batch_size=1000,
    normalize=True,
    random_seed=42,
    pca_dim=None,
    scaler_path=None,
    pca_path=None,
):
    logger = setup_logger(output_path)

    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Min cluster size: {min_cluster_size}")
    logger.info(f"Min samples: {min_samples}")
    logger.info(f"Max cluster size: {max_cluster_size}")  # 新增日志
    logger.info(f"Cluster selection method: {cluster_selection_method}")
    logger.info(f"Cluster selection epsilon: {cluster_selection_epsilon}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Normalize: {normalize}")
    logger.info(f"Random seed: {random_seed}")
    if pca_dim:
        logger.info(f"PCA dimension: {pca_dim}")

    logger.section("Directory Search")
    logger.info(f"Input directory: {input_path}")

    file_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_path)
        for file in files
        if file.endswith(".pt")
    ]
    logger.success(f"Found {len(file_paths)} .pt files")

    logger.section("Global Feature Pre-processing (On CPU)")

    # Initialize processors
    logger.info("Initializing scaler and PCA")
    scaler = None
    pca = None

    # Handle normalization
    if normalize:
        if COMPUTE_GLOBAL_NORM:
            logger.info("Computing global normalization")
            scaler = StandardScaler()
            all_features = []
            for batch, _ in tqdm(
                load_and_preprocess_batch(file_paths, batch_size),
                total=len(file_paths) // batch_size,
                desc="Loading data for normalization",
            ):
                all_features.append(batch)
            all_features = np.vstack(all_features)
            scaler.fit(all_features)
            os.makedirs(output_path, exist_ok=True)
            joblib.dump(scaler, os.path.join(output_path, "scaler.joblib"))
            logger.success(
                f"Computed and saved scaler to {os.path.join(output_path, 'scaler.joblib')}"
            )
        else:
            scaler = joblib.load(
                scaler_path or os.path.join(output_path, "scaler.joblib")
            )
            logger.success(
                f"Loaded scaler from {scaler_path or os.path.join(output_path, 'scaler.joblib')}"
            )

    # Handle PCA
    if pca_dim:
        if COMPUTE_GLOBAL_PCA:
            logger.info("Computing global PCA")
            pca = sklearnPCA(n_components=pca_dim)
            all_features = []
            for batch, _ in tqdm(
                load_and_preprocess_batch(file_paths, batch_size),
                total=len(file_paths) // batch_size,
                desc="Loading data for PCA",
            ):
                if normalize:
                    batch = scaler.transform(batch)
                all_features.append(batch)
            all_features = np.vstack(all_features)

            pca_start_time = time.time()
            pca.fit(all_features)
            pca_end_time = time.time()
            pca_duration = pca_end_time - pca_start_time
            logger.info(
                f"PCA computation completed in {pca_duration:.2f} seconds"
            )

            pca_save_path = pca_path or os.path.join(output_path, "pca.joblib")
            os.makedirs(os.path.dirname(pca_save_path), exist_ok=True)
            joblib.dump(pca, pca_save_path)
            logger.success(f"Computed and saved PCA to {pca_save_path}")
        else:
            pca = joblib.load(
                pca_path or os.path.join(output_path, "pca.joblib")
            )
            logger.success(
                f"Loaded PCA from {pca_path or os.path.join(output_path, 'pca.joblib')}"
            )

    # Load all data to GPU:0 in batches
    logger.section("Loading All Data to GPU:0")
    gpu_features = []
    file_paths_processed = []

    with tqdm(total=len(file_paths), desc="Loading data") as pbar:
        for batch, batch_files in load_and_preprocess_batch(
            file_paths, batch_size
        ):
            if normalize:
                batch = scaler.transform(batch)
            if pca_dim:
                batch = pca.transform(batch)
            gpu_batch = cp.asarray(
                batch, dtype=cp.float32
            )  # Changed to float32
            gpu_features.append(gpu_batch)
            file_paths_processed.extend(batch_files)

            pbar.update(len(batch))
            pbar.set_postfix_str(f"GPU VRAM: {get_gpu_memory_usage()}")

    # Concatenate all batches on GPU
    gpu_features = cp.vstack(gpu_features)
    logger.success(
        f"Loaded {gpu_features.shape[0]} samples to GPU:0 with shape {gpu_features.shape}"
    )
    logger.info(f"Final GPU VRAM occupied: {get_gpu_memory_usage()}")

    # Initialize HDBSCAN
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        max_cluster_size=max_cluster_size,  # 新增参数
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        gen_min_span_tree=False,
        verbose=2,
    )

    # Perform clustering on all data (on GPU)
    logger.section("Clustering")
    start_time = time.time()
    try:
        labels = hdbscan.fit_predict(gpu_features)
    except Exception as e:
        logger.error(f"Error during HDBSCAN clustering: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error(
            f"GPU memory usage: {cp.cuda.Device().mem_info[0] / 1024**3:.2f} GB / {cp.cuda.Device().mem_info[1] / 1024**3:.2f} GB"
        )
        raise

    end_time = time.time()
    total_time = end_time - start_time
    num_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    logger.success(f"HDBSCAN clustering completed in {total_time:.2f} seconds")
    logger.success(f"Number of cluster centers found: {num_clusters}")
    logger.success(
        f"Noise cluster (-1) {'found' if (-1 in labels) else 'not found'}"
    )

    # Compute and save cluster representatives
    logger.section("Computing Cluster Representatives")
    cluster_representatives = compute_cluster_representatives(
        gpu_features, labels
    )

    # Reverse transform cluster representatives (only PCA, not normalization)
    if pca_dim:
        logger.info("Reversing PCA transformation (on CPU)")
        pca_inverse_start_time = time.time()
        cluster_representatives = cp.asnumpy(cluster_representatives)
        cluster_representatives = pca.inverse_transform(
            cluster_representatives
        )
        logger.success(
            f"PCA inverse transformation completed in {time.time() - pca_inverse_start_time:.2f} seconds"
        )

    rep_file = os.path.join(output_path, "cluster_representatives.npy")
    np.save(rep_file, cluster_representatives)
    logger.success(
        f"Saved high-dimensional normalized cluster representatives to {rep_file}"
    )

    # Prepare cluster assignments
    logger.section("Preparing Cluster Assignments")
    all_assignments = []
    cluster_counts = defaultdict(int)

    labels = cp.asnumpy(labels)
    for file, cluster in zip(file_paths_processed, labels):
        cluster_counts[int(cluster)] += 1
        category = os.path.basename(os.path.dirname(file))
        all_assignments.append(
            {
                "file": file,
                "cluster": int(cluster),
                "category": category,
            }
        )

    logger.success(
        f"HDBSCAN clustering completed with {len(np.unique(labels))} clusters"
    )

    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, "cluster_assignments.json")
    with open(output_file, "w") as f:
        json.dump(all_assignments, f, indent=2)
    logger.success(f"Saved cluster assignments to {output_file}")

    end_time = time.time()
    total_time = end_time - start_time

    logger.section("Clustering Summary")
    logger.success(
        f"Clustering (center finding and cluster assignment) completed in {total_time:.2f} seconds"
    )
    print("\nFinal cluster sizes:")
    for cluster, count in sorted(cluster_counts.items()):
        print(f"  Cluster {cluster:3d}: {count:6d} samples")

    logger.section("Cluster-Category Matrix")
    categories = sorted(
        set(assignment["category"] for assignment in all_assignments)
    )
    cluster_labels = sorted(
        set(assignment["cluster"] for assignment in all_assignments)
    )

    matrix = {
        category: {cluster: 0 for cluster in cluster_labels}
        for category in categories
    }
    for assignment in all_assignments:
        matrix[assignment["category"]][assignment["cluster"]] += 1

    df = pd.DataFrame(matrix).T
    csv_file = os.path.join(output_path, "cluster_category_matrix.csv")
    df.to_csv(csv_file)
    logger.success(f"Cluster-category matrix saved to {csv_file}")

    # Print formatted cluster-category matrix
    print("\nCluster-category matrix:")

    # Calculate column widths for formatting
    col_widths = [
        max(
            len(str(cluster)),
            max(len(str(matrix[cat][cluster])) for cat in categories),
        )
        for cluster in cluster_labels
    ]
    cat_width = max(len(cat) for cat in categories)

    # Print header
    header = f"{'Category':<{cat_width}} " + " ".join(
        f"{str(cluster):>{width}}"
        for cluster, width in zip(cluster_labels, col_widths)
    )
    print(header)
    print(
        "-" * len(header)
    )  # Add a line under the header for better readability

    # Print rows
    for category in categories:
        row = f"{category:<{cat_width}} " + " ".join(
            f"{matrix[category][cluster]:>{width}}"
            for cluster, width in zip(cluster_labels, col_widths)
        )
        print(row)

    metadata = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "normalize": normalize,
    }
    metadata_file = os.path.join(output_path, "clustering_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.success(f"Saved clustering metadata to {metadata_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster image features using GPU HDBSCAN"
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
        help="Path to save the cluster assignments",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=5,
        help="The minimum size of clusters",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="The number of samples in a neighborhood for a point to be considered a core point",
    )
    parser.add_argument(
        "--max_cluster_size",
        type=int,
        default=0,
        help="The maximum size of clusters (default: 0, as in rapids.hdbscan)",
    )
    parser.add_argument(
        "--cluster_selection_method",
        type=str,
        default="eom",
        choices=["eom", "leaf"],
        help="The method used to select clusters from the condensed tree",
    )
    parser.add_argument(
        "--cluster_selection_epsilon",
        type=float,
        default=0.0,
        help="Distance threshold for merging clusters",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable feature normalization",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="Number of PCA components (default: no PCA)",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default=None,
        help="Path to the scaler joblib file (default: None, will use output_path/scaler.joblib)",
    )
    parser.add_argument(
        "--pca_path",
        type=str,
        default=None,
        help="Path to the PCA joblib file (default: None, will use output_path/pca.joblib)",
    )

    args = parser.parse_args()

    cluster_features(
        args.input_path,
        args.output_path,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        max_cluster_size=args.max_cluster_size,  # 新增参数
        cluster_selection_method=args.cluster_selection_method,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        random_seed=args.random_seed,
        pca_dim=args.pca_dim,
        scaler_path=args.scaler_path,
        pca_path=args.pca_path,
    )
