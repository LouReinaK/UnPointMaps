
import pytest
import numpy as np
from src.clustering.dbscan_clustering import dbscan_clustering


class TestDBSCAN:
    @pytest.fixture
    def sample_data(self):
        # Create 2 blobs slightly separated
        c1 = np.random.normal(loc=[0, 0], scale=0.1, size=(50, 2))
        c2 = np.random.normal(loc=[1, 1], scale=0.1, size=(50, 2))
        # Add some noise
        noise = np.array([[0.5, 0.5], [10, 10]])
        return np.vstack([c1, c2, noise])

    def test_dbscan_basic(self, sample_data):
        # eps=0.3 should separate the two blobs and maybe classify noise
        clusters, k, labels = dbscan_clustering(
            sample_data, eps=0.3, min_samples=5)

        # We expect at least 2 clusters
        assert k >= 2
        assert len(labels) == 102

        # Verify noise is handled (label -1)
        # Note: implementation might not include noise in 'clusters' dict
        # check if noise point [10, 10] (last point) is labelled -1
        assert labels[-1] == -1

    def test_dbscan_empty(self):
        data = np.array([])
        clusters, k, labels = dbscan_clustering(data)
        assert k == 0
        assert len(clusters) == 0
