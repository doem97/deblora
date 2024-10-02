import os
import json
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import argparse
import time
from collections import defaultdict
import pandas as pd
import joblib

# Function for inference
def preprocess_and_predict(features, output_path):
    scaler_path = os.path.join(output_path, "scaler.joblib")
    pca_path = os.path.join(output_path, "pca.joblib")
    kmeans_path = os.path.join(output_path, "kmeans.joblib")
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        features = scaler.transform(features)
    
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        features = pca.transform(features)
    
    kmeans = joblib.load(kmeans_path)
    return kmeans.predict(features)

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def log_section(title):
    print(f"\n{'=' * 80}\n{title:^80}\n{'=' * 80}\n")

def log_info(message):
    print(f"ℹ️ {message}")

def log_success(message):
    print(f"✅ {message}")

def load_and_preprocess_batch(file_paths, batch_size):
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        features = []
        for file_path in batch:
            feature = torch.load(file_path, map_location='cpu').squeeze(0).reshape(-1).numpy()
            features.append(feature)
        yield np.array(features), batch

def cluster_features(input_path, n_clusters, output_path, batch_size=1000, normalize=True, use_pca=False, pca_components=50, random_seed=42):
    start_time = time.time()
    
    log_section("Feature Clustering with CPU MiniBatchKMeans")
    log_info(f"Input directory: {input_path}")
    
    file_paths = [os.path.join(root, file) for root, _, files in os.walk(input_path) for file in files if file.endswith('.pt')]
    log_success(f"Found {len(file_paths)} .pt files")

    log_section("Global Feature Processing")

    # Initialize processors
    scaler = StandardScaler(with_mean=normalize, with_std=normalize)
    pca = IncrementalPCA(n_components=pca_components) if use_pca else None

    # First pass: Compute global statistics
    for batch, _ in tqdm(load_and_preprocess_batch(file_paths, batch_size), total=len(file_paths)//batch_size, desc="Computing global statistics"):
        if normalize:
            scaler.partial_fit(batch)
        if use_pca:
            pca.partial_fit(batch)

    log_success("Global statistics computed")

    # Save preprocessing parameters
    os.makedirs(output_path, exist_ok=True)
    if normalize:
        joblib.dump(scaler, os.path.join(output_path, "scaler.joblib"))
        log_success(f"Saved scaler to {os.path.join(output_path, 'scaler.joblib')}")
    if use_pca:
        joblib.dump(pca, os.path.join(output_path, "pca.joblib"))
        log_success(f"Saved PCA to {os.path.join(output_path, 'pca.joblib')}")

    # Initialize K-means
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_seed, batch_size=batch_size)

    # Second pass: Apply transformations and perform clustering
    log_section("Clustering")
    for batch, _ in tqdm(load_and_preprocess_batch(file_paths, batch_size), total=len(file_paths)//batch_size, desc="Clustering"):
        if normalize:
            batch = scaler.transform(batch)
        if use_pca:
            batch = pca.transform(batch)
        kmeans.partial_fit(batch)
    
    # Save K-means model
    joblib.dump(kmeans, os.path.join(output_path, "kmeans.joblib"))
    log_success(f"Saved K-means model to {os.path.join(output_path, 'kmeans.joblib')}")

    log_section("Cluster Assignment")
    assignments = []
    cluster_counts = defaultdict(int)

    for batch, batch_files in tqdm(load_and_preprocess_batch(file_paths, batch_size), total=len(file_paths)//batch_size, desc="Assigning clusters"):
        if normalize:
            batch = scaler.transform(batch)
        if use_pca:
            batch = pca.transform(batch)
        labels = kmeans.predict(batch)
        
        for file, cluster in zip(batch_files, labels):
            cluster_counts[int(cluster)] += 1
            category = os.path.basename(os.path.dirname(file))
            assignments.append({"file": file, "cluster": int(cluster), "category": category})

    log_success(f"K-means clustering completed with {n_clusters} clusters")

    os.makedirs(output_path, exist_ok=True)
    
    cluster_centers_file = os.path.join(output_path, "cluster_centers.npy")
    np.save(cluster_centers_file, kmeans.cluster_centers_)
    log_success(f"Saved cluster centers to {cluster_centers_file}")

    output_file = os.path.join(output_path, "cluster_assignments.json")
    with open(output_file, 'w') as f:
        json.dump(assignments, f, indent=2)
    log_success(f"Saved cluster assignments to {output_file}")

    end_time = time.time()
    total_time = end_time - start_time
    
    log_section("Clustering Summary")
    log_success(f"Clustering completed in {total_time:.2f} seconds")
    print("\nFinal cluster sizes:")
    for cluster, count in sorted(cluster_counts.items()):
        print(f"  Cluster {cluster:3d}: {count:6d} samples")
    
    log_section("Cluster-Category Matrix")
    categories = sorted(set(assignment['category'] for assignment in assignments))
    cluster_labels = sorted(set(assignment['cluster'] for assignment in assignments))
    
    matrix = {category: {cluster: 0 for cluster in cluster_labels} for category in categories}
    for assignment in assignments:
        matrix[assignment['category']][assignment['cluster']] += 1
    
    df = pd.DataFrame(matrix).T
    csv_file = os.path.join(output_path, "cluster_category_matrix.csv")
    df.to_csv(csv_file)
    log_success(f"Cluster-category matrix saved to {csv_file}")

    # Print formatted cluster-category matrix
    print("\nCluster-category matrix:")
    
    # Calculate column widths for formatting
    col_widths = [max(len(str(cluster)), max(len(str(matrix[cat][cluster])) for cat in categories)) for cluster in cluster_labels]
    cat_width = max(len(cat) for cat in categories)
    
    # Print header
    header = f"{'Category':<{cat_width}} " + " ".join(f"{str(cluster):>{width}}" for cluster, width in zip(cluster_labels, col_widths))
    print(header)
    print('-' * len(header))  # Add a line under the header for better readability
    
    # Print rows
    for category in categories:
        row = f"{category:<{cat_width}} " + " ".join(f"{matrix[category][cluster]:>{width}}" for cluster, width in zip(cluster_labels, col_widths))
        print(row)

    metadata = {
        "n_clusters": n_clusters,
        "normalize": normalize,
        "use_pca": use_pca,
        "pca_components": pca_components if use_pca else None,
    }
    metadata_file = os.path.join(output_path, "clustering_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    log_success(f"Saved clustering metadata to {metadata_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster image features using CPU MiniBatchKMeans")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input features folder")
    parser.add_argument("--n_clusters", type=int, default=100, help="The number of clusters for K-means")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the cluster assignments")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--no_normalize", action="store_true", help="Disable feature normalization")
    parser.add_argument("--use_pca", action="store_true", help="Use PCA for dimensionality reduction")
    parser.add_argument("--pca_components", type=int, default=50, help="Number of components for PCA")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    cluster_features(args.input_path, args.n_clusters, args.output_path, 
                     args.batch_size, normalize=not args.no_normalize,
                     use_pca=args.use_pca, pca_components=args.pca_components, 
                     random_seed=args.random_seed)