import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from src.processing.dataset_filtering import convert_to_dict_filtered
from src.clustering.hdbscan_clustering import hdbscan_clustering_iterative


def compare_params():
    print("Loading dataset...")
    df = convert_to_dict_filtered()
    
    # Optional: limit dataset for faster testing
    LIMIT = 2000 
    if LIMIT != -1 and len(df) > LIMIT:
        print(f"Limiting dataset to {LIMIT} points...")
        df = df.head(LIMIT)

    # Define parameters to compare
    # Format: (label, min_cluster_size, cluster_selection_epsilon, max_std_dev)
    configs = [
        ("Tighter (5, 0.001, 50)", 5, 1/1000.0, 50.0),
        ("Very Tight (3, 0.0005, 30)", 3, 5/10000.0, 30.0),
        ("Small Clusters (5, 0.001, 25)", 5, 1/1000.0, 25.0),
        ("Medium Epsilon (10, 0.005, 100)", 10, 5/1000.0, 100.0),
        ("High Epsilon (10, 0.01, 100)", 10, 1/100.0, 100.0),
        ("Extreme Epsilon (10, 0.05, 100)", 10, 5/100.0, 100.0),
    ]

    n_configs = len(configs)
    cols = 2
    rows = (n_configs + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    axes = axes.flatten()

    for i, (label, min_size, eps, std_dev) in enumerate(configs):
        print(f"\nRunning config: {label}")
        start_time = time.time()
        
        clustered_points, used_k, labels = hdbscan_clustering_iterative(
            df,
            min_cluster_size=min_size,
            cluster_selection_epsilon=eps,
            max_std_dev=std_dev
        )
        
        duration = time.time() - start_time
        print(f"  Found {used_k} clusters in {duration:.2f}s")

        ax = axes[i]
        
        # Plot noise points in gray
        noise_mask = (labels == -1)
        ax.scatter(df.loc[noise_mask, 'longitude'], df.loc[noise_mask, 'latitude'], 
                   c='lightgray', s=1, alpha=0.3, label='Noise')

        # Plot clusters with distinct colors
        # Filter out noise from df to plot clusters
        for cluster_id in range(used_k):
            cluster_mask = (labels == cluster_id)
            ax.scatter(df.loc[cluster_mask, 'longitude'], df.loc[cluster_mask, 'latitude'], 
                       s=5, alpha=0.7)

        ax.set_title(f"{label}\nClusters: {used_k}, Time: {duration:.2f}s")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide unused subplots
    for j in range(n_configs, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    print("\nDisplaying comparison plot...")
    plt.show()

    pass


if __name__ == "__main__":
    compare_params()
