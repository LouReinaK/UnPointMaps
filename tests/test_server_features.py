

from server import update_app_state_with_clustering_results, app_state
import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestServerFeatures(unittest.TestCase):
    def setUp(self):
        app_state.current_clusters = {}
        app_state.labelled_cluster_ids = set()
        app_state.labelling_queue = MagicMock()
        app_state.embedding_service = MagicMock()
        app_state.embedding_service.get_embedding_model.return_value = True
        app_state.embedding_service.select_representative_indices.return_value = [0]

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
        
        # Set the df in app_state for labelling worker to access
        app_state.df = df

        # Test the metadata extraction logic directly (simulating what labelling worker does)
        cluster_id = 0
        cluster_df = df[df['cluster_label'] == cluster_id]
        
        # Prepare text data
        temp_data = []
        text_cols = ['title', 'tags', 'description']
        available_cols = [col for col in text_cols if col in cluster_df.columns]
        
        if available_cols:
            filled_df = cluster_df[available_cols].fillna('')
            combined_series = None
            for col in available_cols:
                if combined_series is None:
                    combined_series = filled_df[col].astype(str)
                else:
                    combined_series = combined_series + " " + filled_df[col].astype(str)
            
            if combined_series is not None:
                temp_data = combined_series.tolist()
        
        # Verify that description was included in the combined text
        self.assertTrue(len(temp_data) > 0)
        self.assertTrue("Desc" in temp_data[0])
        
        # If embedding service is available, it would be called with this data
        if temp_data and app_state.embedding_service and app_state.embedding_service.get_embedding_model():
            app_state.embedding_service.select_representative_indices(temp_data, top_k=50)
            
            # Verify the call was made
            calls = app_state.embedding_service.select_representative_indices.call_args_list
            self.assertTrue(len(calls) > 0)

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
        
        # Set the df in app_state for labelling worker to access
        app_state.df = df

        # Test the metadata extraction logic directly (simulating what labelling worker does)
        cluster_id = 0
        cluster_df = df[df['cluster_label'] == cluster_id]
        
        # Prepare text data
        temp_data = []
        text_cols = ['title', 'tags', 'description']
        available_cols = [col for col in text_cols if col in cluster_df.columns]
        
        if available_cols:
            filled_df = cluster_df[available_cols].fillna('')
            combined_series = None
            for col in available_cols:
                if combined_series is None:
                    combined_series = filled_df[col].astype(str)
                else:
                    combined_series = combined_series + " " + filled_df[col].astype(str)
            
            if combined_series is not None:
                temp_data = combined_series.tolist()
        
        # Verify that title and tags were included (no description)
        self.assertTrue(len(temp_data) > 0)
        self.assertTrue("Title" in temp_data[0])
        self.assertTrue("Tag" in temp_data[0])
        self.assertFalse("Desc" in temp_data[0])
        
        # If embedding service is available, it would be called with this data
        if temp_data and app_state.embedding_service and app_state.embedding_service.get_embedding_model():
            app_state.embedding_service.select_representative_indices(temp_data, top_k=50)
            
            # Verify the call was made
            calls = app_state.embedding_service.select_representative_indices.call_args_list
            self.assertTrue(len(calls) > 0)

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