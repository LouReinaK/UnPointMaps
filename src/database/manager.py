import sqlite3
import json
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import os

class DatabaseManager:
    def __init__(self, db_path: str = "cache.db"):
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
                confidence REAL,
                FOREIGN KEY (run_id) REFERENCES clustering_runs (id)
            )
        ''')

        # Table to store points for each cluster (to avoid re-reading/matching large DF if desired)
        # Using a simplistic approach: storing lat/lon directly. 
        # For large datasets, referencing original indices might be better, but we cache computed data as requested.
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

        conn.commit()
        conn.close()

    def _compute_params_hash(self, params: Dict[str, Any], dataset_signature: str) -> str:
        """Compute a consistent hash for parameters and dataset."""
        # Sort keys to ensure consistent JSON
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{params_str}|{dataset_signature}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_cached_run(self, params: Dict[str, Any], dataset_signature: str) -> Optional[int]:
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
            
            # 2. Insert Clusters and Points
            for cluster_idx, points in enumerate(clusters_data):
                # points expected to be list of [lat, lon]
                label_text = labels_map.get(cluster_idx, f"Cluster {cluster_idx}")
                
                cursor.execute(
                    "INSERT INTO clusters (run_id, cluster_index, label) VALUES (?, ?, ?)",
                    (run_id, cluster_idx, label_text)
                )
                cluster_db_id = cursor.lastrowid
                
                # Batch insert points
                # Assuming points is a list of [lat, lon] or similar
                # We won't store original_index if not provided easily, defaulting to NULL
                points_batch = []
                for point in points:
                    # Point is likely [lat, lon]
                    lat, lon = point[0], point[1]
                    points_batch.append((cluster_db_id, lat, lon))
                
                if points_batch:
                    cursor.executemany(
                        "INSERT INTO cluster_points (cluster_id, latitude, longitude) VALUES (?, ?, ?)",
                        points_batch
                    )

            conn.commit()
            return run_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def load_run_data(self, run_id: int) -> Tuple[List[List[List[float]]], Dict[int, str]]:
        """
        Load clusters and labels for a given run ID.
        Returns (clustered_points, llm_labels)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get clusters
            cursor.execute("SELECT id, cluster_index, label FROM clusters WHERE run_id = ? ORDER BY cluster_index", (run_id,))
            cluster_rows = cursor.fetchall()
            
            clustered_points = []
            llm_labels = {}
            
            for c_id, c_idx, label in cluster_rows:
                llm_labels[c_idx] = label
                
                # Get points for this cluster
                cursor.execute("SELECT latitude, longitude FROM cluster_points WHERE cluster_id = ?", (c_id,))
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
