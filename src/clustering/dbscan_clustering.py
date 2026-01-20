from typing import List, Tuple, Any
import numpy as np
from sklearn.cluster import DBSCAN


def dbscan_clustering(dataset: Any,
                      eps: float = 0.3,
                      min_samples: int = 5) -> Tuple[List[List[List[float]]],
                                                     int,
                                                     np.ndarray]:
    """
    Performs DBSCAN clustering on the dataset.

    Args:
        dataset: pandas DataFrame (with 'latitude', 'longitude') or numpy array of points.
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
        clusters: A list of clusters, where each cluster is a list of points [lat, lon]. Noise points (label -1) are excluded.
        n_clusters: The number of clusters found (excluding noise).
        labels: Array of cluster labels (-1 is noise).
    """

    # Data preparation
    if hasattr(dataset, 'iloc'):  # pandas DataFrame
        # Extract coordinates
        points_array = dataset[['latitude', 'longitude']].values
    else:  # Assume numpy array or list
        points_array = np.array(dataset)

    # Handle empty dataset
    if points_array.size == 0:
        return [], 0, np.array([])

    # Apply DBSCAN
    # Using the parameters specified in the prompt
    # Optimization: algorithm='kd_tree' is usually fastest for low-dimensional
    # data (2D)
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="kd_tree",
        leaf_size=40,       # Slightly larger leaf size can speed up tree queries
        n_jobs=-1           # use all cores
    ).fit(points_array)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    n_clusters = len(unique_labels)
    sorted_labels = sorted(list(unique_labels))

    # Organize points by cluster
    # Optimization: Vectorized boolean indexing instead of iterating over all
    # points
    clusters = []

    # Pre-calculate masks for speed if needed, but simple boolean indexing is usually fast enough
    # and vastly faster than a python for-loop over N points
    for label in sorted_labels:
        # Extract all points belonging to this cluster at once
        cluster_points = points_array[labels == label]
        clusters.append(cluster_points.tolist())

    return clusters, n_clusters, labels


if __name__ == "__main__":
    # Example usage for testing
    np.random.seed(42)
    # Generate random points
    sample_points = np.array(
        [[np.random.uniform(-90, 90), np.random.uniform(-180, 180)] for _ in range(100)])

    clusters, n_clusters, labels = dbscan_clustering(
        sample_points, eps=10, min_samples=2)
    print(f"DBSCAN complete. Found {n_clusters} clusters.")
    for i, c in enumerate(clusters):
        print(f"Cluster {i}: {len(c)} points")