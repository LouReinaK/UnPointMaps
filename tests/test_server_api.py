import pytest
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient
from server import app


class TestServerAPI:

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'latitude': [45.0],
            'longitude': [4.0],
            'date': [pd.Timestamp('2023-01-01')],
            'cluster_label': [-1],
            'title': ['Test'],
            'description': ['Desc'],
            'tags': ['tag']
        })

    def test_get_index(self):
        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200
            # Ensure it serves html
            assert "text/html" in response.headers["content-type"]

    @patch('src.processing.dataset_filtering.convert_to_dict_filtered')
    @patch('server.convert_to_dict_filtered')
    @patch('server.TimeFilter')
    @patch('server.DatabaseManager')
    @patch('server.EmbeddingService')
    def test_get_stats(
            self,
            mock_emb,
            mock_db,
            mock_tf,
            mock_load_server,
            mock_load_src,
            sample_df):
        mock_load_server.return_value = sample_df
        mock_load_src.return_value = sample_df
        # Mock detect_events
        mock_tf.return_value.detect_events.return_value = [{
            'label': 'Event',
            'start_date_str': '2023-01-01',
            'end_date_str': '2023-01-02',
            'total_entries': 10
        }]

        # We need to run lifespan to init app_state
        with TestClient(app) as client:
            response = client.get("/api/stats")
            assert response.status_code == 200
            data = response.json()
            assert data['total_points'] == 1
            assert data['events'][0]['label'] == 'Event'

    @patch('src.processing.dataset_filtering.convert_to_dict_filtered')
    @patch('server.convert_to_dict_filtered')
    @patch('server.TimeFilter')
    @patch('server.DatabaseManager')
    @patch('server.EmbeddingService')
    @patch('server.threading.Thread')
    def test_run_clustering(
            self,
            mock_thread,
            mock_emb,
            mock_db,
            mock_tf,
            mock_load_server,
            mock_load_src,
            sample_df):
        mock_load_server.return_value = sample_df
        mock_load_src.return_value = sample_df

        with TestClient(app) as client:
            response = client.post(
                "/api/cluster",
                json={
                    "algorithm": "hdbscan"})
            assert response.status_code == 200
            assert response.json()["status"] == "started"

            # Ensure background task would have been started
            assert mock_thread.called
