import os
import json
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import argparse
import time
from collections import defaultdict

def load_and_preprocess_batch(file_paths, batch_size):
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        features = []
        for file_path in batch:
            feature = torch.load(file_path).squeeze(0).reshape(-1)
            features.append(feature.numpy())
        yield np.array(features), batch

def cluster_features(input_path, n_clusters, output_path, batch_size=1000, min_samples_per_cluster=15, verbose=True):
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Define output file paths
    output_file = os.path.join(output_path, "cluster_assignments.json")
    output_clusters = os.path.join(output_path, "cluster_centers.pt")
    
    if verbose:
        print(f"Scanning input directory: {input_path}")
    
    file_paths = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('.pt'):
                file_paths.append(os.path.join(root, file))
    
    if verbose:
        print(f"Found {len(file_paths)} .pt files")
        print(f"Initializing MiniBatchKMeans (n_clusters={n_clusters})")

    # Initialize MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=batch_size, verbose=verbose)

    # First pass: Fit MiniBatchKMeans
    if verbose:
        print("First pass: Clustering with MiniBatchKMeans")
    for batch, _ in tqdm(load_and_preprocess_batch(file_paths, batch_size), 
                         total=len(file_paths)//batch_size, 
                         desc="KMeans fitting"):
        kmeans.partial_fit(batch)

    # Second pass: Assign clusters and save results
    if verbose:
        print("Second pass: Assigning clusters and saving results")
    assignments = []
    cluster_counts = defaultdict(int)
    for batch, batch_files in tqdm(load_and_preprocess_batch(file_paths, batch_size), 
                                   total=len(file_paths)//batch_size, 
                                   desc="Cluster assignment"):
        batch_clusters = kmeans.predict(batch)
        for file, cluster in zip(batch_files, batch_clusters):
            cluster_counts[int(cluster)] += 1
            assignments.append({"file": file, "cluster": int(cluster)})

    # Reassign samples from small clusters
    if verbose:
        print(f"Reassigning samples from clusters with less than {min_samples_per_cluster} samples")
    
    small_clusters = [c for c, count in cluster_counts.items() if count < min_samples_per_cluster]
    large_clusters = [c for c, count in cluster_counts.items() if count >= min_samples_per_cluster]
    
    if small_clusters:
        for i, assignment in enumerate(assignments):
            if assignment["cluster"] in small_clusters:
                feature = torch.load(assignment["file"]).squeeze(0).reshape(-1).numpy()
                distances = kmeans.transform(feature.reshape(1, -1))
                new_cluster = large_clusters[np.argmin([distances[0][c] for c in large_clusters])]
                assignments[i]["cluster"] = int(new_cluster)
                cluster_counts[assignment["cluster"]] -= 1
                cluster_counts[new_cluster] += 1

    # Save cluster assignments
    if verbose:
        print(f"Saving cluster assignments to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(assignments, f, indent=2)

    # Save cluster centers
    if verbose:
        print(f"Saving cluster centers to {output_clusters}")
    centers = kmeans.cluster_centers_
    torch.save(torch.tensor(centers), output_clusters)

    end_time = time.time()
    total_time = end_time - start_time
    
    if verbose:
        print(f"Clustering completed in {total_time:.2f} seconds")
        print(f"Cluster assignments saved to {output_file}")
        print(f"Cluster centers saved to {output_clusters}")
        print("Final cluster sizes:")
        for cluster, count in sorted(cluster_counts.items()):
            print(f"Cluster {cluster}: {count} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster image features using Mini-batch K-means")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input features folder")
    parser.add_argument("--n_clusters", type=int, default=32, help="Number of clusters")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the cluster assignments and centers")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--min_samples_per_cluster", type=int, default=15, help="Minimum number of samples per cluster")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    cluster_features(args.input_path, args.n_clusters, args.output_path, 
                     args.batch_size, args.min_samples_per_cluster, args.verbose)