from typing import List, Tuple, Optional, Any
import multiprocessing
import numpy as np
try:
    import hdbscan
except ImportError:
    hdbscan = None

# Worker function must be top-level for multiprocessing pickling


def _hdbscan_worker(
    points_array: np.ndarray,
    original_indices_range: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int],
    cluster_selection_epsilon: float,
    max_std_dev: float,
    output_queue: multiprocessing.Queue
):
    if hdbscan is None:
        output_queue.put(("error", "hdbscan library is not installed. HDBSCAN clustering is unavailable."))
        return

    try:
        # Initialize labels array with -1 (noise)
        final_labels = np.full(len(points_array), -1, dtype=int)

        # Track points to process - initially all points
        points_to_process = [(points_array, original_indices_range)]

        final_clusters: List[Tuple[List[List[float]], List[int]]] = []
        next_cluster_id = 0
        iteration = 0

        while points_to_process:
            iteration += 1
            print(
                f"\nIteration {iteration}: Processing {len(points_to_process)} cluster(s)...")

            # Yield intermediate state via queue
            # Each item in lists is now (points_list, indices_list)
            current_view_clusters = final_clusters + \
                [(p[0].tolist(), p[1].tolist()) for p in points_to_process]
            output_queue.put(
                ("intermediate", (current_view_clusters, iteration)))

            new_points_to_process = []

            temp_clusters = []

            for current_points, original_indices in points_to_process:
                if len(current_points) == 0:
                    continue

                print(f"  Clustering {len(current_points)} points...")

                # Apply HDBSCAN to current subset
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    metric='haversine',
                    core_dist_n_jobs=-1  # Use all cores for this step
                ).fit(current_points)

                labels = clusterer.labels_
                # Optimization: np.unique is faster than sorted(list(set(...)))
                unique_labels = np.unique(labels)

                # Remove noise label for processing
                if -1 in unique_labels:
                    unique_labels = unique_labels[unique_labels != -1]

                # Collect all clusters
                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_points = current_points[cluster_mask]
                    cluster_original_indices = original_indices[cluster_mask]

                    temp_clusters.append((cluster_points, cluster_original_indices))

                # Noise points remain as noise in final_labels
                noise_mask = labels == -1
                if np.any(noise_mask):
                    noise_count = np.sum(noise_mask)
                    print(f"    Found {noise_count} noise points")

            # Compute statistics on all found clusters
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
                            final_clusters.append((cluster_points.tolist(), cluster_original_indices.tolist()))
                            final_labels[cluster_original_indices] = next_cluster_id
                            next_cluster_id += 1
                else:
                    # Only one cluster, accept it
                    for cluster_points, cluster_original_indices in temp_clusters:
                        cluster_size = len(cluster_points)
                        print(f"    Accepted cluster with {cluster_size} points (ID: {next_cluster_id})")
                        final_clusters.append((cluster_points.tolist(), cluster_original_indices.tolist()))
                        final_labels[cluster_original_indices] = next_cluster_id
                        next_cluster_id += 1

            # Update points to process for next iteration
            points_to_process = new_points_to_process

            if points_to_process:
                print(
                    f"  Need to split {len(points_to_process)} large cluster(s) in next iteration...")

        n_clusters = len(final_clusters)
        print(f"\nIterative clustering complete! Total clusters: {n_clusters}")

        output_queue.put(("final", (final_clusters, n_clusters, final_labels)))

    except Exception as e:
        print(f"Worker process error: {e}")
        output_queue.put(("error", str(e)))


def hdbscan_iterative_generator(
    dataset: Any,
    min_cluster_size: int = 3,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0005,
    max_std_dev: float = 30.0
):
    """
    Generator for iterative HDBSCAN clustering.
    Spawns a separate process to run clustering and yields results from a queue.

    Yields:
        (status, data)
        status: "intermediate" or "final"

        If "intermediate":
            data: list of all current cluster point arrays (completed + pending)

        If "final":
            data: (final_clusters_data, n_clusters, final_labels)
    """

    # Data preparation
    if hasattr(dataset, 'iloc'):  # pandas DataFrame
        points_array = dataset[['latitude', 'longitude']].values
    else:
        points_array = np.array(dataset)

    # Create Queue and Process
    output_queue: multiprocessing.Queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=_hdbscan_worker,
        args=(
            points_array,
            np.arange(len(points_array)),
            min_cluster_size,
            min_samples,
            cluster_selection_epsilon,
            max_std_dev,
            output_queue
        )
    )

    process.start()

    try:
        while True:
            # Block until an item is available
            try:
                item = output_queue.get()
            except Exception:
                break

            status = item[0]
            if status == "error":
                raise RuntimeError(f"Clustering worker failed: {item[1]}")

            yield item

            if status == "final":
                break

    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        else:
            process.join()


def hdbscan_clustering_iterative(
    dataset: Any,
    min_cluster_size: int = 3,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0005,
    max_std_dev: float = 30.0
) -> Tuple[List[List[List[float]]], int, np.ndarray]:
    """
    Performs iterative HDBSCAN clustering, splitting large clusters until the standard deviation
    of cluster sizes falls below max_std_dev.
    Wrapper around generator.
    """
    gen = hdbscan_iterative_generator(
        dataset,
        min_cluster_size,
        min_samples,
        cluster_selection_epsilon,
        max_std_dev)
    result = None
    for status, data in gen:
        if status == "final":
            result = data
    if result is None:
        return [], 0, np.array([])
    return result
