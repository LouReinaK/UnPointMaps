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

    def test_tram_line_endpoint(self):
        # Mock some clusters in app_state
        from server import app_state
        app_state.current_clusters = {
            1: {
                'points': [[45.0, 4.0], [45.1, 4.1]],
                'size': 10
            },
            2: {
                'points': [[45.2, 4.2], [45.3, 4.3]],
                'size': 20
            }
        }
        
        with TestClient(app) as client:
            response = client.post(
                "/api/tram_line",
                json={"max_length": 0.01}
            )
            assert response.status_code == 200
            data = response.json()
            assert "path" in data
            assert isinstance(data["path"], list)
            assert len(data["path"]) > 0
            # Each point should be [lat, lon]
            for point in data["path"]:
                assert isinstance(point, list)
                assert len(point) == 2
                assert isinstance(point[0], float)
                assert isinstance(point[1], float)

    def test_tram_line_endpoint_no_clusters(self):
        from server import app_state
        app_state.current_clusters = {}
        
        with TestClient(app) as client:
            response = client.post(
                "/api/tram_line",
                json={"max_length": 0.01}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["path"] == []
            assert "error" in data

    def test_tram_line_endpoint_invalid_length(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/tram_line",
                json={"max_length": -1}
            )
            assert response.status_code == 200
            data = response.json()
            assert "error" in data
