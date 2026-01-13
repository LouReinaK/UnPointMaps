import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Tuple

def dbscan_clustering(dataset: any, eps: float = 0.3, min_samples: int = 5) -> Tuple[List[List[List[float]]], int]:
    """
    Performs DBSCAN clustering on the dataset.
    
    Args:
        dataset: pandas DataFrame (with 'latitude', 'longitude') or numpy array of points.
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        
    Returns:
        clusters: A list of clusters, where each cluster is a list of points [lat, lon]. Noise points (label -1) are excluded.
        n_clusters: The number of clusters found (excluding noise).
    """
    
    # Data preparation
    if hasattr(dataset, 'iloc'): # pandas DataFrame
        # Extract coordinates
        points_array = dataset[['latitude', 'longitude']].values
    else: # Assume numpy array or list
        points_array = np.array(dataset)

    # Apply DBSCAN
    # Using the parameters specified in the prompt
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="auto",   # 'auto', 'ball_tree', 'kd_tree', 'brute'
        n_jobs=-1           # use all cores
    ).fit(points_array)
    
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Organize points by cluster
    # We will create a list of length n_clusters. 
    # Since labels can be -1, 0, 1, ..., we need to map them to 0, 1, ... or just ignore -1.
    # We will ignore noise (-1) for the main clusters list as typically noise isn't "a cluster" to draw a hull around.
    
    clusters = [[] for _ in range(n_clusters)]
    
    # We need to map label indices to 0..n_clusters-1
    # Unique labels (sorted) might be [-1, 0, 1, 2] or [0, 1, 2]
    # If -1 exists, 0 -> index 0
    
    for i, label in enumerate(labels):
        if label != -1:
            clusters[label].append(points_array[i].tolist())
            
    return clusters, n_clusters

if __name__ == "__main__":
    # Example usage for testing
    np.random.seed(42)
    # Generate random points
    sample_points = np.array([[np.random.uniform(-90, 90), np.random.uniform(-180, 180)] for _ in range(100)])
    
    clusters, n_clusters = dbscan_clustering(sample_points, eps=10, min_samples=2)
    print(f"DBSCAN complete. Found {n_clusters} clusters.")
    for i, c in enumerate(clusters):
        print(f"Cluster {i}: {len(c)} points")
