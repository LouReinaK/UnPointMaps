import numpy as np
import hdbscan
import pandas as pd
from typing import List, Tuple, Optional

def hdbscan_clustering(dataset: any, min_cluster_size: int = 10, min_samples: Optional[int] = None, cluster_selection_epsilon: float = 0.5) -> Tuple[List[List[List[float]]], int, np.ndarray]:
    """
    Performs HDBSCAN clustering on the dataset.
    
    Args:
        dataset: pandas DataFrame (with 'latitude', 'longitude') or numpy array of points.
        min_cluster_size: The minimum number of points in a group for that group to be considered a cluster.
        min_samples: The number of samples in a neighborhood for a point to be considered a core point.
        cluster_selection_epsilon: A density threshold to split clusters. Points closer than this distance are considered to be in the same cluster.
        
    Returns:
        clusters: A list of clusters, where each cluster is a list of points [lat, lon]. Noise points (label -1) are excluded.
        n_clusters: The number of clusters found (excluding noise).
        labels: Array of cluster labels for each point (-1 indicates noise).
    """
    
    # Data preparation
    if hasattr(dataset, 'iloc'): # pandas DataFrame
        # Extract coordinates
        points_array = dataset[['latitude', 'longitude']].values
    else: # Assume numpy array or list
        points_array = np.array(dataset)

    # Apply HDBSCAN
    # Using the parameters specified in the prompt
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon
    ).fit(points_array)
    
    labels = clusterer.labels_
    # probabilities = clusterer.probabilities_ # Not used for return but available
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Organize points by cluster
    clusters = [[] for _ in range(n_clusters)]
    
    # We will ignore noise (-1) for the main clusters list
    unique_labels = sorted(list(set(labels)))
    if -1 in unique_labels:
        unique_labels.remove(-1)
        
    # Create a mapping from label to index (0 to n_clusters-1)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    
    for point, label in zip(points_array, labels):
        if label != -1:
            clusters[label_to_index[label]].append(point.tolist())
            
    return clusters, n_clusters, labels


def hdbscan_clustering_iterative(
    dataset: any, 
    min_cluster_size: int = 10, 
    min_samples: Optional[int] = None, 
    cluster_selection_epsilon: float = 0.5,
    max_cluster_size: int = 5000
) -> Tuple[List[List[List[float]]], int, np.ndarray]:
    """
    Performs iterative HDBSCAN clustering, splitting large clusters until all are below max_cluster_size.
    
    Args:
        dataset: pandas DataFrame (with 'latitude', 'longitude') or numpy array of points.
        min_cluster_size: The minimum number of points in a group for that group to be considered a cluster.
        min_samples: The number of samples in a neighborhood for a point to be considered a core point.
        cluster_selection_epsilon: A density threshold to split clusters.
        max_cluster_size: Maximum allowed cluster size. Clusters larger than this will be split.
        
    Returns:
        clusters: A list of final clusters, where each cluster is a list of points [lat, lon].
        n_clusters: The total number of clusters found.
        labels: Array of cluster labels for each point in the original dataset (-1 indicates noise).
    """
    
    # Data preparation
    if hasattr(dataset, 'iloc'): # pandas DataFrame
        points_array = dataset[['latitude', 'longitude']].values
    else:
        points_array = np.array(dataset)
    
    # Initialize labels array with -1 (noise)
    final_labels = np.full(len(points_array), -1, dtype=int)
    
    # Track points to process - initially all points
    points_to_process = [(points_array, np.arange(len(points_array)))]
    
    final_clusters = []
    next_cluster_id = 0
    iteration = 0
    
    while points_to_process:
        iteration += 1
        print(f"\nIteration {iteration}: Processing {len(points_to_process)} cluster(s)...")
        
        new_points_to_process = []
        
        for current_points, original_indices in points_to_process:
            if len(current_points) == 0:
                continue
                
            print(f"  Clustering {len(current_points)} points...")
            
            # Apply HDBSCAN to current subset
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon
            ).fit(current_points)
            
            labels = clusterer.labels_
            unique_labels = sorted(list(set(labels)))
            
            # Remove noise label for processing
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            # Process each cluster
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_points = current_points[cluster_mask]
                cluster_original_indices = original_indices[cluster_mask]
                
                cluster_size = len(cluster_points)
                
                if cluster_size > max_cluster_size:
                    # This cluster is too large - add it back for re-processing
                    print(f"    Found large cluster with {cluster_size} points (> {max_cluster_size}), will split...")
                    new_points_to_process.append((cluster_points, cluster_original_indices))
                else:
                    # This cluster is acceptable - save it
                    print(f"    Accepted cluster with {cluster_size} points (ID: {next_cluster_id})")
                    final_clusters.append(cluster_points.tolist())
                    final_labels[cluster_original_indices] = next_cluster_id
                    next_cluster_id += 1
            
            # Noise points remain as noise in final_labels
            noise_mask = labels == -1
            if np.any(noise_mask):
                noise_count = np.sum(noise_mask)
                print(f"    Found {noise_count} noise points")
        
        # Update points to process for next iteration
        points_to_process = new_points_to_process
        
        if points_to_process:
            print(f"  Need to split {len(points_to_process)} large cluster(s) in next iteration...")
    
    n_clusters = len(final_clusters)
    print(f"\nIterative clustering complete! Total clusters: {n_clusters}")
    
    return final_clusters, n_clusters, final_labels
