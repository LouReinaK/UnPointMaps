import numpy as np
import hdbscan
import pandas as pd
from typing import List, Tuple, Optional

def hdbscan_iterative_generator(
    dataset: any, 
    min_cluster_size: int = 10, 
    min_samples: Optional[int] = None, 
    cluster_selection_epsilon: float = 0.5,
    max_cluster_size: int = 5000
):
    """
    Generator for iterative HDBSCAN clustering.
    Yields intermediate states of clustering.
    
    Yields:
        (status, data)
        status: "intermediate" or "final"
        
        If "intermediate":
            data: list of all current cluster point arrays (completed + pending)
            
        If "final":
            data: (final_clusters_data, n_clusters, final_labels)
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

        # Yield intermediate state
        current_view_clusters = final_clusters + [p[0].tolist() for p in points_to_process]
        yield "intermediate", current_view_clusters
        
        new_points_to_process = []
        
        for current_points, original_indices in points_to_process:
            if len(current_points) == 0:
                continue
                
            print(f"  Clustering {len(current_points)} points...")
            
            # Apply HDBSCAN to current subset
            # Optimization: Explicitly use boruvka_kdtree for speed with Euclidean metric
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                metric='euclidean',
                algorithm='boruvka_kdtree',
                core_dist_n_jobs=-1 # Use all cores for this step
            ).fit(current_points)
            
            labels = clusterer.labels_
            # Optimization: np.unique is faster than sorted(list(set(...)))
            unique_labels = np.unique(labels)
            
            # Remove noise label for processing
            if -1 in unique_labels:
                # np.unique returns a numpy array, but we can iterate over it directly.
                # To remove -1 efficiently, we can just mask it or list comp.
                unique_labels = unique_labels[unique_labels != -1]
            
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
    
    yield "final", (final_clusters, n_clusters, final_labels)


def hdbscan_clustering_iterative(
    dataset: any, 
    min_cluster_size: int = 10, 
    min_samples: Optional[int] = None, 
    cluster_selection_epsilon: float = 0.5,
    max_cluster_size: int = 5000
) -> Tuple[List[List[List[float]]], int, np.ndarray]:
    """
    Performs iterative HDBSCAN clustering, splitting large clusters until all are below max_cluster_size.
    Wrapper around generator.
    """
    gen = hdbscan_iterative_generator(dataset, min_cluster_size, min_samples, cluster_selection_epsilon, max_cluster_size)
    result = None
    for status, data in gen:
        if status == "final":
            result = data
    return result
