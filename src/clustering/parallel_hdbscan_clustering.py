from typing import List, Tuple, Optional, Dict, Any
import concurrent.futures
import multiprocessing
import numpy as np
try:
    import hdbscan
except ImportError:
    hdbscan = None


def _process_cluster_subset(args: Tuple[np.ndarray,
                                        np.ndarray,
                                        Dict[str,
                                             Any]]) -> List[Tuple[np.ndarray,
                                                                 np.ndarray]]:
    """
    Helper function to process a single subset of points with HDBSCAN.
    Executed in parallel workers.

    Args:
        args: Tuple containing:
            - current_points: numpy array of points to cluster
            - original_indices: numpy array of original indices corresponding to current_points
            - hdbscan_params: Dictionary of parameters for HDBSCAN

    Returns:
        clusters: List of (points, indices) for all discovered clusters
    """
    if hdbscan is None:
        raise ImportError("hdbscan library is not installed.")

    current_points, original_indices, hdbscan_params = args

    if len(current_points) < hdbscan_params.get('min_cluster_size', 5):
        return []

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_params.get('min_cluster_size', 10),
            min_samples=hdbscan_params.get('min_samples', None),
            cluster_selection_epsilon=hdbscan_params.get(
                'cluster_selection_epsilon', 0.5),
            metric='haversine',
            # algorithm='boruvka_kdtree',
            core_dist_n_jobs=1  # Disable inner parallelism to avoid oversubscription
        ).fit(current_points)

        labels = clusterer.labels_
        unique_labels = np.unique(labels)

        if -1 in unique_labels:
            unique_labels = unique_labels[unique_labels != -1]

        clusters = []

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = current_points[cluster_mask]
            cluster_original_indices = original_indices[cluster_mask]

            clusters.append((cluster_points, cluster_original_indices))

        return clusters

    except Exception as e:
        print(f"Error in parallel worker: {e}")
        return []


def parallel_hdbscan_iterative_generator(
    dataset: Any,
    min_cluster_size: int = 3,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0005,
    max_std_dev: float = 30.0,
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

    final_clusters_data: List[List[List[float]]] = []  # List of points lists
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
            print(
                f"Iteration {iteration}: Processing {len(points_to_process)} cluster(s) with {max_workers} workers...")

            # Yield intermediate state (current view of map)
            # The current view consists of already finalized clusters AND the
            # ones currently being processed (points_to_process)
            current_view_clusters = final_clusters_data + \
                [p[0].tolist() for p in points_to_process]
            yield "intermediate", (current_view_clusters, iteration)

            new_points_to_process = []

            # Prepare arguments for parallel execution
            tasks = [
                (pts, idxs, hdbscan_params)
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

            # Aggregate all found clusters first
            temp_clusters = []
            for clusters_in_subset in results:
                temp_clusters.extend(clusters_in_subset)

            # Compute statistics on all found clusters in this iteration
            if temp_clusters:
                sizes = [len(c[0]) for c in temp_clusters]
                if len(sizes) > 1:
                    std_dev = np.std(sizes)
                    mean_size = np.mean(sizes)
                    print(f"    Cluster sizes std dev: {std_dev:.2f}, mean: {mean_size:.2f}")

                    for cluster_points, cluster_original_indices in temp_clusters:
                        cluster_size = len(cluster_points)
                        if std_dev > max_std_dev and cluster_size > mean_size:
                            print(f"    Re-processing large cluster with {cluster_size} points")
                            new_points_to_process.append((cluster_points, cluster_original_indices))
                        else:
                            print(f"    Accepted cluster with {cluster_size} points (ID: {next_cluster_id})")
                            final_clusters_data.append(cluster_points.tolist())
                            final_labels[cluster_original_indices] = next_cluster_id
                            next_cluster_id += 1
                else:
                    # Only one cluster, accept it
                    for cluster_points, cluster_original_indices in temp_clusters:
                        cluster_size = len(cluster_points)
                        print(f"    Accepted cluster with {cluster_size} points (ID: {next_cluster_id})")
                        final_clusters_data.append(cluster_points.tolist())
                        final_labels[cluster_original_indices] = next_cluster_id
                        next_cluster_id += 1

            points_to_process = new_points_to_process

    n_clusters = next_cluster_id
    yield "final", (final_clusters_data, n_clusters, final_labels)


def parallel_hdbscan_clustering_iterative(
    dataset: Any,
    min_cluster_size: int = 3,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0005,
    max_std_dev: float = 30.0,
    max_workers: Optional[int] = None
) -> Tuple[List[List[List[float]]], int, np.ndarray]:
    """
    Performs parallelized iterative HDBSCAN clustering.
    Wrapper around the generator to return final result only.
    """
    gen = parallel_hdbscan_iterative_generator(
        dataset,
        min_cluster_size,
        min_samples,
        cluster_selection_epsilon,
        max_std_dev,
        max_workers)

    result = None
    for status, data in gen:
        if status == "final":
            result = data

    if result is None:
        return [], 0, np.array([])
    return result  # type: ignore
