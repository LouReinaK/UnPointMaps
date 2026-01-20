
from server import update_app_state_with_clustering_results, app_state
import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to mock app_state components inside server.py but we can't easily do it after import if they are global
# but we can modify app_state object attributes.


class TestServerFeatures(unittest.TestCase):
    def setUp(self):
        app_state.current_clusters = {}
        app_state.labelled_cluster_ids = set()
        app_state.labelling_queue = MagicMock()
        app_state.embedding_service = MagicMock()
        app_state.embedding_service.get_embedding_model.return_value = True
        app_state.embedding_service.select_representative_indices.return_value = [
            0]

    def test_metadata_extraction_with_description(self):
        """Test metadata extraction when description column IS present"""
        data = {
            'latitude': [45.76],
            'longitude': [4.85],
            'title': ['Title'],
            'tags': ['Tag'],
            'description': ['Desc'],
            'cluster_label': [0]
        }
        df = pd.DataFrame(data)
        labels = np.array([0])

        update_app_state_with_clustering_results(df, labels)

        # Verify call arguments to EmbeddingService
        calls = app_state.embedding_service.select_representative_indices.call_args_list
        self.assertTrue(len(calls) > 0)
        # Check that 'Desc' was included in the text passed
        text_list = calls[0][0][0]
        self.assertTrue("Desc" in text_list[0])

    def test_metadata_extraction_without_description(self):
        """Test metadata extraction when description column IS missing (Regression Test)"""
        data = {
            'latitude': [45.76],
            'longitude': [4.85],
            'title': ['Title'],
            'tags': ['Tag'],
            # Missing description
            'cluster_label': [0]
        }
        df = pd.DataFrame(data)
        labels = np.array([0])

        update_app_state_with_clustering_results(df, labels)

        calls = app_state.embedding_service.select_representative_indices.call_args_list
        self.assertTrue(len(calls) > 0)
        text_list = calls[0][0][0]
        self.assertTrue("Title Tag" in text_list[0] or "Title" in text_list[0])

    def test_update_clusters_state(self):
        """Test that global state is updated correctly"""
        data = {
            'latitude': [45.76, 45.77],
            'longitude': [4.85, 4.86],
            'title': ['T1', 'T2'],
            'tags': ['t1', 't2'],
            'cluster_label': [-1, -1]  # Reset first
        }
        df = pd.DataFrame(data)
        labels = np.array([1, 1])  # Both in cluster 1

        update_app_state_with_clustering_results(df, labels)

        self.assertTrue(1 in app_state.current_clusters)
        self.assertEqual(len(app_state.current_clusters), 1)

        cluster_data = app_state.current_clusters[1]
        self.assertEqual(cluster_data['size'], 2)

    def test_hull_fallback(self):
        """Test hull computation failure handled gracefully"""
        # 2 points is not enough for convex hull usually (requires 3 for area, but Valid for generic hull if handle collinear)
        # compute_cluster_hulls handles it or raises exception. We check
        # fallback.
        data = {
            'latitude': [45.76, 45.76],  # Duplicate point
            'longitude': [4.85, 4.85],
            'title': ['T1', 'T2'],
            'cluster_label': [2, 2]
        }
        df = pd.DataFrame(data)
        # Add required columns to avoid warnings from our previous fix if
        # checked
        if 'tags' not in df.columns:
            df['tags'] = ''
        if 'description' not in df.columns:
            df['description'] = ''

        labels = np.array([2, 2])

        # We Mock compute_cluster_hulls to raise Exception
        with patch('server.compute_cluster_hulls', side_effect=Exception("Hull Error")):
            update_app_state_with_clustering_results(df, labels)

        # Check results
        self.assertTrue(2 in app_state.current_clusters)
        # Hull points should be equal to points (fallback behavior)
        cluster_data = app_state.current_clusters[2]
        # In current implementation, 'points' key holds the hull points (or
        # original points if fallback)
        self.assertEqual(len(cluster_data['points']), 2)


if __name__ == '__main__':
    unittest.main()
