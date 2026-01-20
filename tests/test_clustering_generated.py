import pytest
import numpy as np
import pandas as pd
from src.clustering.clustering import kmeans_clustering, find_optimal_k_elbow


class TestClustering:
    @pytest.fixture
    def sample_data(self):
        # Create 3 distinct clusters
        c1 = np.random.normal(loc=[0, 0], scale=0.1, size=(10, 2))
        c2 = np.random.normal(loc=[5, 5], scale=0.1, size=(10, 2))
        c3 = np.random.normal(loc=[10, 0], scale=0.1, size=(10, 2))
        return np.vstack([c1, c2, c3])

    def test_kmeans_clustering_fixed_k(self, sample_data):
        clusters, k, labels = kmeans_clustering(sample_data, k=3)
        assert k == 3
        assert len(clusters) == 3
        assert len(labels) == 30

    def test_kmeans_clustering_elbow(self, sample_data):
        # Elbow should find 3
        clusters, k, labels = kmeans_clustering(
            sample_data, k=None, method='elbow')
        assert k == 3
        assert len(clusters) == 3

    def test_kmeans_clustering_silhouette(self, sample_data):
        # Silhouette should find 3
        clusters, k, labels = kmeans_clustering(
            sample_data, k=None, method='silhouette')
        assert k == 3
        assert len(clusters) == 3

    def test_kmeans_dataframe(self, sample_data):
        df = pd.DataFrame(sample_data, columns=['latitude', 'longitude'])
        clusters, k, labels = kmeans_clustering(df, k=3)
        assert k == 3
        assert len(labels) == 30

    def test_find_optimal_k_elbow_small_data(self):
        # Not enough data for elbow
        data = np.array([[0, 0], [1, 1]])
        k = find_optimal_k_elbow(data, k_range=range(2, 3))
        assert k == 2
