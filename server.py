from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import threading
import asyncio
import json
import logging
import queue
from typing import List, Dict, Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Project imports
from src.processing.dataset_filtering import convert_to_dict_filtered
from src.processing.time_filtering import TimeFilter
from src.clustering.hdbscan_clustering import hdbscan_clustering_iterative
from src.processing.llm_labelling import ConfigManager, LLMLabelingService, ClusterMetadata
from src.utils.hull_logic import compute_cluster_hulls

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnPointMaps_Server")

# Global State
class AppState:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.time_filter = TimeFilter()
        self.events = []
        self.current_clusters = {} # cluster_id -> {points, metadata, status, label}
        self.active_websockets: List[WebSocket] = []
        self.labelling_queue = queue.PriorityQueue()
        self.labelled_cluster_ids: Set[str] = set()
        self.stop_labelling_flag = threading.Event()
        self.labelling_thread = None
        self.db_manager = None # Not using DB for this live version for now, or could re-enable

app_state = AppState()

# Labelling Worker
def labelling_worker():
    logger.info("Labelling worker started.")
    
    # Initialize Service
    try:
        config_manager = ConfigManager()
        labeling_service = LLMLabelingService(config_manager)
        logger.info(f"LLM Service initialized using {config_manager.get_model()}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Service: {e}")
        return

    while not app_state.stop_labelling_flag.is_set():
        try:
            # item: (priority, cluster_id, retry_count)
            # Priority: 0 (User Click), 10 (Background)
            priority, cluster_id, retry_count = app_state.labelling_queue.get(timeout=1)
            
            if cluster_id in app_state.labelled_cluster_ids:
                app_state.labelling_queue.task_done()
                continue

            cluster_data = app_state.current_clusters.get(cluster_id)
            if not cluster_data:
                app_state.labelling_queue.task_done()
                continue
                
            logger.info(f"Processing label for {cluster_id} (Priority {priority})")
            
            # Prepare metadata for LLM
            # We need to extract text metadata from the dataframe for this cluster
            # The cluster_data should store indices or we filter df again (slower)
            # Better: cluster_data stores text_metadata ready to go.
            
            metadata_dict = {
                "cluster_id": str(cluster_id),
                "image_ids": [str(x) for x in cluster_data['sample_ids']],
                "text_metadata": cluster_data['text_metadata'],
                "cluster_size": cluster_data['size']
            }
            
            try:
                # Call LLM
                result = labeling_service.generate_cluster_label(metadata_dict)
                label = result.label
                
                # Update State
                app_state.current_clusters[cluster_id]['label'] = label
                app_state.labelled_cluster_ids.add(cluster_id)
                
                # Broadcast
                asyncio.run_coroutine_threadsafe(
                    broadcast_label(cluster_id, label),
                    loop
                )
                
            except Exception as e:
                logger.error(f"Error labelling {cluster_id}: {e}")
                # Retry logic if needed?
            
            app_state.labelling_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker Exception: {e}")

async def broadcast_label(cluster_id: str, label: str):
    message = json.dumps({
        "type": "label_update",
        "cluster_id": cluster_id,
        "label": label
    })
    to_remove = []
    for ws in app_state.active_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            to_remove.append(ws)
    for ws in to_remove:
        if ws in app_state.active_websockets:
            app_state.active_websockets.remove(ws)

# Start Labelling Thread
loop = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global loop
    loop = asyncio.get_running_loop()
    
    # Load Data
    logger.info("Loading dataset...")
    app_state.df = convert_to_dict_filtered()
    logger.info(f"Dataset loaded: {len(app_state.df)} records.")
    
    # Detect Events
    logger.info("Detecting events...")
    app_state.events = app_state.time_filter.detect_events(app_state.df)
    
    # Start Worker
    app_state.labelling_thread = threading.Thread(target=labelling_worker, daemon=True)
    app_state.labelling_thread.start()
    
    yield
    
    # Cleanup
    app_state.stop_labelling_flag.set()
    if app_state.labelling_thread:
        app_state.labelling_thread.join()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return HTMLResponse(open("static/index.html", encoding="utf-8").read())

@app.get("/api/stats")
async def get_stats():
    if app_state.df is None:
        return {"error": "Data not loaded"}
    
    df = app_state.df
    min_date = df['date'].min().isoformat()
    max_date = df['date'].max().isoformat()
    
    return {
        "total_points": len(df),
        "min_date": min_date,
        "max_date": max_date,
        "events": [
            {
                "id": i+1, 
                "label": e['label'], 
                "start": e['start_date_str'], 
                "end": e['end_date_str'],
                "count": int(e['total_entries'])
            } for i, e in enumerate(app_state.events[:10])
        ]
    }

@app.post("/api/cluster")
async def run_clustering(params: dict):
    """
    Params:
    {
        "min_year": 2010,
        "max_year": 2015,
        "start_hour": 8,
        "end_hour": 18,
        "exclude_events": [1, 2] # indices of events to exclude
    }
    """
    logger.info(f"Clustering request: {params}")
    
    # Filter Data
    df = app_state.df.copy()
    
    # Year Filter
    if "min_year" in params and "max_year" in params:
        df = df[
            (df['date'].dt.year >= params['min_year']) & 
            (df['date'].dt.year <= params['max_year'])
        ]
        
    # Time of Day Filter
    if "start_hour" in params and "end_hour" in params:
        # Assuming simple range for now
        s, e = params['start_hour'], params['end_hour']
        if s <= e:
            df = df[(df['hour'] >= s) & (df['hour'] <= e)]
        else:
             # Crosses midnight
            df = df[(df['hour'] >= s) | (df['hour'] <= e)]
            
    # Exclude Events
    if "exclude_events" in params and params['exclude_events']:
        events_to_exclude = []
        for idx in params['exclude_events']:
            if 0 <= idx-1 < len(app_state.events):
                events_to_exclude.append(app_state.events[idx-1])
        if events_to_exclude:
            df = app_state.time_filter.exclude_events(df, events_to_exclude)
            
    if len(df) == 0:
        return {"clusters": [], "count": 0}
        
    # Run HDBSCAN
    # Use standard params from run.py
    min_cluster_size = 10
    cluster_selection_epsilon = 1/1000.0 
    max_cluster_size = 1000
    
    clustered_points_groups, used_k, labels = hdbscan_clustering_iterative(
        df, 
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size
    )
    
    # Compute Hulls & Metadata
    # labels matches df rows.
    df['cluster_label'] = labels
    
    # Clear previous queue and state
    while not app_state.labelling_queue.empty():
        try:
            app_state.labelling_queue.get_nowait()
            app_state.labelling_queue.task_done()
        except:
            pass
    
    app_state.labelled_cluster_ids.clear()
    app_state.current_clusters.clear()
    
    result_clusters = []
    
    # Process each cluster
    # clustered_points_groups is list of list of [lat, lon].
    # But checking hdbscan_clustering_iterative source, it returns:
    # return final_clusters, next_cluster_id, final_labels
    # final_clusters is list of points.
    
    # However we need strict ID mapping. 
    # final_labels has IDs -1 (noise) to N.
    
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])
    
    for cluster_id in unique_labels:
        cluster_df = df[df['cluster_label'] == cluster_id]
        points = cluster_df[['latitude', 'longitude']].values.tolist()
        
        # Compute Hull
        # simple convex hull for visualization
        try:
             # Use alpha=None for auto-computed alpha shape logic
             hull_points_list = compute_cluster_hulls([points], alpha=None)[0]
             # compute_cluster_hulls returns [ [poly1, poly2...] ] for each cluster
             # If it returns a list of polygons (multipolygon), pass that valid structure to Leaflet
             hull_points = hull_points_list
        except Exception as e:
             logger.error(f"Hull error for {cluster_id}: {e}")
             hull_points = points # Fallback to points

        # Prepare Metadata
        # Collect top text metadata for LLM
        texts = []
        for col in ['title', 'tags', 'description']:
            if col in cluster_df.columns:
                texts.extend(cluster_df[col].dropna().astype(str).tolist())
        
        # Take a sample of valid texts
        valid_texts = [t for t in texts if len(t) > 3]
        text_sample = valid_texts[:50] # increased sample size for better context
        
        # Store in app state
        c_id_str = str(cluster_id)
        app_state.current_clusters[c_id_str] = {
            "size": len(cluster_df),
            "points": hull_points, 
            "text_metadata": text_sample,
            "sample_ids": cluster_df.index[:5].tolist(), # fake IDs using index
            "label": None
        }
        
        # Add to Labelling Queue
        app_state.labelling_queue.put((10, c_id_str, 0)) # Priority 10 (Background)
        
        result_clusters.append({
            "id": c_id_str,
            "points": hull_points,
            "size": len(cluster_df)
        })
        
    return {"clusters": result_clusters, "count": len(result_clusters)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app_state.active_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Expecting {"action": "prioritize", "cluster_id": "1"}
            msg = json.loads(data)
            if msg.get("action") == "prioritize":
                c_id = str(msg.get("cluster_id"))
                if c_id in app_state.current_clusters and c_id not in app_state.labelled_cluster_ids:
                    logger.info(f"Boosting priority for cluster {c_id}")
                    app_state.labelling_queue.put((0, c_id, 0)) # Priority 0 (High)
                    
    except WebSocketDisconnect:
        app_state.active_websockets.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
