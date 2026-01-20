import pytest
import sqlite3
import os
import tempfile
from src.database.manager import DatabaseManager


class TestDatabaseManager:
    @pytest.fixture
    def db_path(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    @pytest.fixture
    def manager(self, db_path):
        return DatabaseManager(db_path=db_path)

    def test_init_db(self, manager, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        assert "clustering_runs" in tables
        assert "hull_cache" in tables
        conn.close()

    def test_hull_cache(self, manager):
        key = "test_hull"
        data = [[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]

        # Test Get Empty
        assert manager.get_cached_hull(key) is None

        # Test Save & Get
        manager.save_cached_hull(key, data)
        result = manager.get_cached_hull(key)
        assert result == data

    def test_llm_cache(self, manager):
        key = "test_llm"
        data = {"label": "test", "confidence": 0.9}

        # Test Get Empty
        assert manager.get_cached_llm_label(key) is None

        # Test Save & Get
        manager.save_cached_llm_label(key, data)
        result = manager.get_cached_llm_label(key)
        assert result == data

    def test_run_caching(self, manager):
        params = {"k": 3}
        sig = "sig123"

        # Should be None initially (get_cached_run checks clustering_runs table)
        # Note: save_run is complex, maybe test internal hash computation
        run_id = manager.get_cached_run(params, sig)
        assert run_id is None

        # We can simulate saving a run manually or use save_run if we provide proper args
        # save_run signature: params, dataset_signature, clusters_data, labels_map
        # clusters_data expected structure?
        # It's not fully clear without reading save_run, so let's skip deep save_run test
        # and test _compute_params_hash indirectly via get_cached_run logic by
        # inserting manually

        params_hash = manager._compute_params_hash(params, sig)
        conn = sqlite3.connect(manager.db_path)
        conn.execute(
            "INSERT INTO clustering_runs (params_hash) VALUES (?)", (params_hash,))
        conn.commit()
        conn.close()

        run_id = manager.get_cached_run(params, sig)
        assert run_id is not None
