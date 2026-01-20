import unittest
import numpy as np
import time
from src.clustering.hdbscan_clustering import hdbscan_clustering_iterative
from src.clustering.parallel_hdbscan_clustering import parallel_hdbscan_clustering_iterative
from sklearn.datasets import make_blobs

class TestParallelHDBSCAN(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset
        # 3 large blobs to ensure splitting might happen if max_cluster_size is small
        self.n_samples = 2000
        self.centers = [[0, 0], [5, 5], [10, 0]]
        self.X, _ = make_blobs(n_samples=self.n_samples, centers=self.centers, cluster_std=0.5, random_state=42)
        
        # DataFrame simulation (list of dicts or just array)
        # The functions accept numpy arrays directly
        self.dataset = self.X
        
        self.params = {
            'min_cluster_size': 10,
            'cluster_selection_epsilon': 0.2,
            'max_cluster_size': 300 # Should force splitting of the 666-point blobs
        }

    def test_sequential_vs_parallel_logic(self):
        print("\n--- Testing Sequential Execution ---")
        start_time = time.time()
        clusters_seq, n_clusters_seq, labels_seq = hdbscan_clustering_iterative(
            self.dataset,
            **self.params
        )
        seq_time = time.time() - start_time
        print(f"Sequential took {seq_time:.4f}s. Found {n_clusters_seq} clusters.")
        
        print("\n--- Testing Parallel Execution ---")
        start_time = time.time()
        clusters_par, n_clusters_par, labels_par = parallel_hdbscan_clustering_iterative(
            self.dataset,
            **self.params
        )
        par_time = time.time() - start_time
        print(f"Parallel took {par_time:.4f}s. Found {n_clusters_par} clusters.")
        
        # Verify basic properties
        self.assertEqual(len(clusters_seq), n_clusters_seq)
        self.assertEqual(len(clusters_par), n_clusters_par)
        
        # Number of clusters should be roughly similar (exact match depends on floating point drift, but HDBSCAN is usually deterministic)
        # With identical inputs and logic, they should be identical.
        self.assertEqual(n_clusters_seq, n_clusters_par)
        
        # Check if noise count is similar
        noise_seq = np.sum(labels_seq == -1)
        noise_par = np.sum(labels_par == -1)
        print(f"Noise points: Seq={noise_seq}, Par={noise_par}")
        self.assertEqual(noise_seq, noise_par)

if __name__ == '__main__':
    unittest.main()
