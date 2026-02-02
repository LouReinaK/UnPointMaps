import sqlite3
import json
import hashlib
import os
import io
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


class DatabaseManager:
    def __init__(self, db_path: str = "unpointmaps_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table to store clustering runs (execution of algorithm)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clustering_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                params_hash TEXT,
                params_json TEXT,
                dataset_signature TEXT
            )
        ''')

        # Table to store specific clusters from a run
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                cluster_index INTEGER,
                label TEXT,
                FOREIGN KEY (run_id) REFERENCES clustering_runs (id)
            )
        ''')

        # Table to store points for each cluster (to avoid re-reading/matching large DF if desired)
        # Using a simplistic approach: storing lat/lon directly.
        # For large datasets, referencing original indices might be better, but
        # we cache computed data as requested.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cluster_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER,
                latitude REAL,
                longitude REAL,
                original_index INTEGER,
                FOREIGN KEY (cluster_id) REFERENCES clusters (id)
            )
        ''')

        # Table to store event detection runs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                dataset_size INTEGER,
                dataset_signature TEXT
            )
        ''')

        # Table to store detected events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detected_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                event_index INTEGER,
                start_day INTEGER,
                end_day INTEGER,
                start_date_str TEXT,
                end_date_str TEXT,
                total_entries INTEGER,
                label TEXT,
                days_json TEXT,
                FOREIGN KEY (run_id) REFERENCES event_runs (id)
            )
        ''')

        # Table for generic hull cache (key-value store)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hull_cache (
                cache_key TEXT PRIMARY KEY,
                hull_data TEXT,
                created_at TEXT
            )
        ''')

        # Table for LLM label cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                label_data TEXT,
                created_at TEXT
            )
        ''')

        # Table for embeddings cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                text_hash TEXT PRIMARY KEY,
                embedding_blob BLOB,
                created_at TEXT
            )
        ''')

        # Table for generic clustering results (labels array)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clustering_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                params_hash TEXT,
                dataset_hash TEXT,
                labels_bytes BLOB,
                created_at TEXT
            )
        ''')

        # Table for direct cluster point ID to label cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cluster_point_label_cache (
                cluster_point_id TEXT PRIMARY KEY,
                label_data TEXT,
                created_at TEXT
            )
        ''')

        # Table for text cleaning cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS text_cleaning_cache (
                raw_text TEXT PRIMARY KEY,
                cleaned_text TEXT,
                created_at TEXT
            )
        ''')

        # Table for processed dataset cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                source_mtime REAL,
                data_json TEXT,
                created_at TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def get_cached_hull(self, key: str) -> Optional[List[List[List[float]]]]:
        """Retrieve cached hull geometry."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT hull_data FROM hull_cache WHERE cache_key = ?", (key,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return json.loads(row[0])
        except Exception as e:
            pass  # Fail silently on cache read error
        return None

    def save_cached_hull(self, key: str, hull_data: List[List[List[float]]]):
        """Save hull geometry to cache."""
        try:
            serialized = json.dumps(hull_data)
            timestamp = datetime.now().isoformat()

            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO hull_cache (cache_key, hull_data, created_at) VALUES (?, ?, ?)",
                (key, serialized, timestamp)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            pass  # Fail silently on cache write error

    def get_cached_llm_label(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached LLM label result."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT label_data FROM llm_cache WHERE cache_key = ?", (key,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return json.loads(row[0])
            return None
        except Exception as e:
            return None

    def save_cached_llm_label(self, key: str, data: Dict[str, Any]):
        """Save LLM label result to cache."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (cache_key, label_data, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(data), datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            pass

    def get_cleaned_text_batch(self, texts: List[str]) -> Dict[str, str]:
        """Retrieve cleaned text for a batch of strings."""
        if not texts:
            return {}
        try:
            conn = sqlite3.connect(self.db_path)
            # SQLite has a limit on parameters, so we might need to chunk if texts is huge
            # but for 5000 it's usually fine if we don't exceed the limit (default 999)
            # Let's chunk it for safety
            results = {}
            chunk_size = 900
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                placeholders = ",".join(["?"] * len(chunk))
                cursor = conn.execute(
                    f"SELECT raw_text, cleaned_text FROM text_cleaning_cache WHERE raw_text IN ({placeholders})",
                    chunk
                )
                for row in cursor.fetchall():
                    results[row[0]] = row[1]
            conn.close()
            return results
        except Exception as e:
            return {}

    def save_cleaned_text_batch(self, mapping: Dict[str, str]):
        """Save a batch of cleaned text mappings."""
        if not mapping:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            timestamp = datetime.now().isoformat()
            data = [(raw, cleaned, timestamp) for raw, cleaned in mapping.items()]
            conn.executemany(
                "INSERT OR REPLACE INTO text_cleaning_cache (raw_text, cleaned_text, created_at) VALUES (?, ?, ?)",
                data
            )
            conn.commit()
            conn.close()
        except Exception as e:
            pass

    def get_cached_cluster_point_label(self, cluster_point_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached label for a specific cluster point ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT label_data FROM cluster_point_label_cache WHERE cluster_point_id = ?", (cluster_point_id,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return json.loads(row[0])
            return None
        except Exception as e:
            return None
    
    def save_cached_cluster_point_label(self, cluster_point_id: str, data: Dict[str, Any]):
        """Save label for a specific cluster point ID to cache."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO cluster_point_label_cache (cluster_point_id, label_data, created_at) VALUES (?, ?, ?)",
                (cluster_point_id, json.dumps(data), datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            pass

    def get_cached_embeddings(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve cached embeddings for a list of texts."""
        if not texts:
            return {}

        # Compute hashes
        text_hashes = {
            t: hashlib.sha256(
                t.encode('utf-8')).hexdigest() for t in texts}
        # We need to map hash back to potentially multiple texts if collisions (unlikely but possible logic)
        # But here we just use the first match or assuming unique enough.
        # Actually, let's just lookup by hash.
        hashes_to_texts: Dict[str, List[str]] = {}
        for t, h in text_hashes.items():
            if h not in hashes_to_texts:
                hashes_to_texts[h] = []
            hashes_to_texts[h].append(t)

        unique_hashes = list(text_hashes.values())

        cached_embeddings = {}

        try:
            conn = sqlite3.connect(self.db_path)

            # Chunking because SQLite has variable limit
            chunk_size = 900
            for i in range(0, len(unique_hashes), chunk_size):
                chunk = unique_hashes[i:i + chunk_size]
                placeholders = ',' .join(['?'] * len(chunk))
                query = f"SELECT text_hash, embedding_blob FROM embeddings_cache WHERE text_hash IN ({placeholders})"

                cursor = conn.execute(query, chunk)
                for row in cursor:
                    text_hash, blob = row
                    # Assuming float32 as it is standard for
                    # sentence-transformers
                    embedding = np.frombuffer(blob, dtype=np.float32)

                    # Populate for all texts matching this hash
                    if text_hash in hashes_to_texts:
                        for t in hashes_to_texts[text_hash]:
                            cached_embeddings[t] = embedding

            conn.close()
        except Exception as e:
            # Fail silently or log
            pass

        return cached_embeddings

    def save_cached_embeddings(
            self, text_embedding_map: Dict[str, np.ndarray]):
        """Save embeddings to cache."""
        if not text_embedding_map:
            return

        try:
            # Increased timeout for concurrency
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            created_at = datetime.now().isoformat()

            data_to_insert = []
            for text, embedding in text_embedding_map.items():
                text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                # Ensure float32
                blob = embedding.astype(np.float32).tobytes()
                data_to_insert.append((text_hash, blob, created_at))

            # Use executemany for bulk insert
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings_cache (text_hash, embedding_blob, created_at) VALUES (?, ?, ?)",
                data_to_insert)
            conn.commit()
            conn.close()
        except Exception as e:
            # print(f"DB Write Error: {e}")
            pass

    def _compute_params_hash(
            self, params: Dict[str, Any], dataset_signature: str) -> str:
        """Compute a consistent hash for parameters and dataset."""
        # Sort keys to ensure consistent JSON
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{params_str}|{dataset_signature}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_cached_run(
            self, params: Dict[str, Any], dataset_signature: str) -> int | None:
        """Check if a run with these parameters already exists."""
        params_hash = self._compute_params_hash(params, dataset_signature)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id FROM clustering_runs WHERE params_hash = ? ORDER BY id DESC LIMIT 1",
            (params_hash,)
        )
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def save_run(self, params: Dict[str, Any], dataset_signature: str,
                 clusters_data: List[Any], labels_map: Dict[int, str]) -> int:
        """
        Save a new clustering run and its results.

        Args:
            params: Dictionary of clustering parameters.
            dataset_signature: Identifier for the dataset.
            clusters_data: List of clusters, where each cluster is a list of points (or list of [lat, lon]).
                           (This matches the output of hdbscan_clustering_iterative structure)
            labels_map: Dictionary mapping cluster_index to LLM label string.
        """
        params_hash = self._compute_params_hash(params, dataset_signature)
        timestamp = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 1. Insert Run
            cursor.execute(
                "INSERT INTO clustering_runs (timestamp, params_hash, params_json, dataset_signature) VALUES (?, ?, ?, ?)",
                (timestamp, params_hash, json.dumps(params), dataset_signature)
            )
            run_id = cursor.lastrowid
            if run_id is None:
                raise RuntimeError("Failed to get run_id after insertion")

            # 2. Insert Clusters and Points
            for cluster_idx, points in enumerate(clusters_data):
                # points expected to be list of [lat, lon]
                label_text = labels_map.get(
                    cluster_idx, f"Cluster {cluster_idx}")

                cursor.execute(
                    "INSERT INTO clusters (run_id, cluster_index, label) VALUES (?, ?, ?)",
                    (run_id, cluster_idx, label_text)
                )
                cluster_db_id = cursor.lastrowid

                # Batch insert points
                # Assuming points is a list of [lat, lon] or similar
                # We won't store original_index if not provided easily,
                # defaulting to NULL
                points_batch = []
                for point in points:
                    # Point is likely [lat, lon]
                    lat, lon = point[0], point[1]
                    points_batch.append((cluster_db_id, lat, lon))

                if points_batch:
                    cursor.executemany(
                        "INSERT INTO cluster_points (cluster_id, latitude, longitude) VALUES (?, ?, ?)",
                        points_batch)

            conn.commit()
            return run_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def load_run_data(
            self, run_id: int) -> Tuple[List[List[List[float]]], Dict[int, str]]:
        """
        Load clusters and labels for a given run ID.
        Returns (clustered_points, llm_labels)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get clusters
            cursor.execute(
                "SELECT id, cluster_index, label FROM clusters WHERE run_id = ? ORDER BY cluster_index",
                (run_id,
                 ))
            cluster_rows = cursor.fetchall()

            clustered_points = []
            llm_labels = {}

            for c_id, c_idx, label in cluster_rows:
                llm_labels[c_idx] = label

                # Get points for this cluster
                cursor.execute(
                    "SELECT latitude, longitude FROM cluster_points WHERE cluster_id = ?", (c_id,))
                points = [[row[0], row[1]] for row in cursor.fetchall()]
                clustered_points.append(points)

            return clustered_points, llm_labels

        finally:
            conn.close()

    def get_cached_events(self, dataset_size: int) -> Optional[List[Dict]]:
        """Retrieve cached events for a dataset of a given size."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Find the most recent run with matching dataset size
            cursor.execute(
                "SELECT id FROM event_runs WHERE dataset_size = ? ORDER BY id DESC LIMIT 1",
                (dataset_size,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            run_id = row[0]

            cursor.execute('''
                SELECT event_index, start_day, end_day, start_date_str, end_date_str,
                       total_entries, label, days_json
                FROM detected_events
                WHERE run_id = ?
                ORDER BY total_entries DESC
            ''', (run_id,))

            events = []
            for r in cursor.fetchall():
                events.append({
                    'id': r[0],
                    'start_day': r[1],
                    'end_day': r[2],
                    'start_date_str': r[3],
                    'end_date_str': r[4],
                    'total_entries': r[5],
                    'label': r[6],
                    'days': json.loads(r[7])
                })

            return events
        finally:
            conn.close()

    def save_events(self, dataset_size: int, events: List[Dict]):
        """Save detected events to cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            timestamp = datetime.now().isoformat()

            # Create run
            cursor.execute(
                "INSERT INTO event_runs (timestamp, dataset_size) VALUES (?, ?)",
                (timestamp, dataset_size)
            )
            run_id = cursor.lastrowid

            # Insert events
            batch = []
            for e in events:
                batch.append((
                    run_id,
                    e['id'],
                    e['start_day'],
                    e['end_day'],
                    e.get('start_date_str', ''),
                    e.get('end_date_str', ''),
                    e['total_entries'],
                    e['label'],
                    json.dumps(e['days'])
                ))

            cursor.executemany('''
                INSERT INTO detected_events
                (run_id, event_index, start_day, end_day, start_date_str, end_date_str,
                total_entries, label, days_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch)

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get_cached_labels(
            self, params: Dict[str, Any], dataset_hash: str) -> Optional[np.ndarray]:
        """Retrieve cached clustering labels."""
        params_hash = self._compute_params_hash(params, dataset_hash)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT labels_bytes FROM clustering_results WHERE params_hash = ? AND dataset_hash = ? ORDER BY id DESC LIMIT 1",
                (params_hash, dataset_hash)
            )
            row = cursor.fetchone()
            if row:
                return np.frombuffer(row[0], dtype=np.int32)
        except Exception:
            pass
        finally:
            conn.close()
        return None

    def save_cached_labels(self,
                           params: Dict[str,
                                        Any],
                           dataset_hash: str,
                           labels: np.ndarray):
        """Save clustering labels to cache."""
        params_hash = self._compute_params_hash(params, dataset_hash)
        timestamp = datetime.now().isoformat()
        labels_bytes = labels.astype(np.int32).tobytes()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO clustering_results (params_hash, dataset_hash, labels_bytes, created_at) VALUES (?, ?, ?, ?)",
                (params_hash, dataset_hash, labels_bytes, timestamp)
            )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

    def get_processed_data_cache(self, source_file: str) -> Optional[pd.DataFrame]:
        """Retrieve processed dataset from cache if source file hasn't changed."""
        try:
            if not os.path.exists(source_file):
                return None
            
            mtime = os.path.getmtime(source_file)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT data_json FROM processed_data_cache WHERE source_file = ? AND source_mtime = ? ORDER BY id DESC LIMIT 1",
                (source_file, mtime)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                df = pd.read_json(io.StringIO(row[0]), orient='records')
                # Ensure date is converted back to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                return df
        except Exception:
            pass
        return None

    def save_processed_data_cache(self, source_file: str, df: pd.DataFrame):
        """Save processed dataset to cache with source file metadata."""
        try:
            if not os.path.exists(source_file):
                return
            
            mtime = os.path.getmtime(source_file)
            data_json = df.to_json(orient='records', date_format='iso')
            timestamp = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO processed_data_cache (source_file, source_mtime, data_json, created_at) VALUES (?, ?, ?, ?)",
                (source_file, mtime, data_json, timestamp)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
