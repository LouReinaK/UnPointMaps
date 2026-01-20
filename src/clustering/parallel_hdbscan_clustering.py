import numpy as np
import hdbscan
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import concurrent.futures
import multiprocessing

def _process_cluster_subset(args: Tuple[np.ndarray, np.ndarray, Dict[str, Any], int]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Helper function to process a single subset of points with HDBSCAN.
    Executed in parallel workers.
    
    Args:
        args: Tuple containing:
            - current_points: numpy array of points to cluster
            - original_indices: numpy array of original indices corresponding to current_points
            - hdbscan_params: Dictionary of parameters for HDBSCAN
            - max_cluster_size: Maximum allowed size for a cluster
            
    Returns:
        final_clusters: List of (points, indices) for accepted clusters
        retry_clusters: List of (points, indices) for clusters that need splitting
    """
    current_points, original_indices, hdbscan_params, max_cluster_size = args
    
    if len(current_points) < hdbscan_params.get('min_cluster_size', 5):
        # Too small to be a cluster (though the parent was a cluster, treating fragments as noise if HDBSCAN fails is one way, 
        # or just keeping it if it can't be split. Current logic implies we try to split. 
        # If input is small, HDBSCAN might label all as noise or one cluster.
        # Let's proceed with HDBSCAN attempt.)
        pass

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_params.get('min_cluster_size', 10),
            min_samples=hdbscan_params.get('min_samples', None),
            cluster_selection_epsilon=hdbscan_params.get('cluster_selection_epsilon', 0.5),
            metric='euclidean',
            algorithm='boruvka_kdtree',
            core_dist_n_jobs=1  # Disable inner parallelism to avoid oversubscription
        ).fit(current_points)
        
        labels = clusterer.labels_
        # Optimization: np.unique is faster
        unique_labels = np.unique(labels)
        
        if -1 in unique_labels:
            unique_labels = unique_labels[unique_labels != -1]
            
        final_clusters = []
        retry_clusters = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = current_points[cluster_mask]
            cluster_original_indices = original_indices[cluster_mask]
            
            cluster_size = len(cluster_points)
            
            if cluster_size > max_cluster_size:
                retry_clusters.append((cluster_points, cluster_original_indices))
            else:
                final_clusters.append((cluster_points, cluster_original_indices))
                
        return final_clusters, retry_clusters
        
    except Exception as e:
        print(f"Error in parallel worker: {e}")
        return [], []


def parallel_hdbscan_iterative_generator(
    dataset: Any, 
    min_cluster_size: int = 10, 
    min_samples: Optional[int] = None, 
    cluster_selection_epsilon: float = 0.5,
    max_cluster_size: int = 5000,
    max_workers: Optional[int] = None
):
    """
    Generator for parallelized iterative HDBSCAN clustering.
    Yields intermediate states of clustering.
    
    Yields:
        (status, data)
        status: "intermediate" or "final"
        
        If "intermediate":
            data: list of all current cluster point arrays (completed + pending)
            
        If "final":
            data: (final_clusters_data, n_clusters, final_labels)
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    # Data preparation
    if hasattr(dataset, 'iloc'): 
        points_array = dataset[['latitude', 'longitude']].values
    else:
        points_array = np.array(dataset)
        
    final_labels = np.full(len(points_array), -1, dtype=int)
    
    # We start with one big chunk
    points_to_process = [(points_array, np.arange(len(points_array)))]
    
    final_clusters_data = [] # List of points lists
    next_cluster_id = 0
    iteration = 0
    
    hdbscan_params = {
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'cluster_selection_epsilon': cluster_selection_epsilon
    }

    # Initialize ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        # Iterate until no more clusters to split
        while points_to_process:
            iteration += 1
            print(f"Iteration {iteration}: Processing {len(points_to_process)} cluster(s) with {max_workers} workers...")
            
            # Yield intermediate state (current view of map)
            # The current view consists of already finalized clusters AND the ones currently being processed (points_to_process)
            current_view_clusters = final_clusters_data + [p[0].tolist() for p in points_to_process]
            yield "intermediate", (current_view_clusters, iteration)
            
            new_points_to_process = []
            
            # Prepare arguments for parallel execution
            tasks = [
                (pts, idxs, hdbscan_params, max_cluster_size) 
                for pts, idxs in points_to_process 
                if len(pts) > 0
            ]
            
            if not tasks:
                break

            results = []
            if len(tasks) == 1:
                results.append(_process_cluster_subset(tasks[0]))
            else:
                results = list(executor.map(_process_cluster_subset, tasks))
            
            # Aggregate results
            for finals, retries in results:
                # Add retries to next iteration queue
                new_points_to_process.extend(retries)
                
                # Save final clusters
                for pts, idxs in finals:
                    final_clusters_data.append(pts.tolist())
                    # For intermediate labels, we can't easily update global labels array iteratively 
                    # without more complex state, but we do it at the end.
                    # However, we DO need to track them for the final result.
                    final_labels[idxs] = next_cluster_id
                    next_cluster_id += 1
            
            points_to_process = new_points_to_process
        
    n_clusters = next_cluster_id
    yield "final", (final_clusters_data, n_clusters, final_labels)


def parallel_hdbscan_clustering_iterative(
    dataset: Any, 
    min_cluster_size: int = 10, 
    min_samples: Optional[int] = None, 
    cluster_selection_epsilon: float = 0.5,
    max_cluster_size: int = 5000,
    max_workers: Optional[int] = None
) -> Tuple[List[List[List[float]]], int, np.ndarray]:
    """
    Performs parallelized iterative HDBSCAN clustering.
    Wrapper around the generator to return final result only.
    """
    gen = parallel_hdbscan_iterative_generator(
        dataset, min_cluster_size, min_samples, cluster_selection_epsilon, max_cluster_size, max_workers
    )
    
    result = None
    for status, data in gen:
        if status == "final":
            result = data
            
    return result

