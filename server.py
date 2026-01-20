from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import threading
import asyncio
import json
import logging
import queue
import time
from typing import List, Dict, Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import hashlib
import requests
import re

# Project imports
from src.processing.dataset_filtering import convert_to_dict_filtered
from src.processing.time_filtering import TimeFilter
from src.clustering.hdbscan_clustering import hdbscan_clustering_iterative, hdbscan_iterative_generator
from src.clustering.clustering import kmeans_clustering
from src.clustering.dbscan_clustering import dbscan_clustering
from src.clustering.parallel_hdbscan_clustering import parallel_hdbscan_clustering_iterative, parallel_hdbscan_iterative_generator
from src.processing.llm_labelling import ConfigManager, LLMLabelingService, ClusterMetadata
from src.utils.hull_logic import compute_cluster_hulls
from src.database.manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnPointMaps_Server")

# Global State
class AppState:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.time_filter: Optional[TimeFilter] = None # Lazy init
        self.events = []
        self.current_clusters: Dict[int, Dict] = {} # cluster_id -> {points, metadata, status, label}
        self.active_websockets: List[WebSocket] = []
        self.labelling_queue = queue.PriorityQueue()
        self.labelled_cluster_ids: Set[int] = set()
        self.stop_labelling_flag = threading.Event()
        self.labelling_thread = None
        self.db_manager = None 

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
            
            # Broadcast start
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(f"Labelling cluster {cluster_id}..."),
                loop
            )
            
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
            
            # Compute Hash for Cache
            # We explicitly exclude cluster_id from hash to allow cache hits across different runs 
            # where the same cluster content might get a different ID
            hash_input = {
                "image_ids": metadata_dict["image_ids"],
                "text_metadata": metadata_dict["text_metadata"]
            }
            cache_key = hashlib.sha256(json.dumps(hash_input, sort_keys=True).encode()).hexdigest()
            
            label = None
            cached_data = None
            
            if app_state.db_manager:
                cached_data = app_state.db_manager.get_cached_llm_label(cache_key)
            
            if cached_data and "label" in cached_data:
                label = cached_data["label"]
                logger.info(f"Using cached label for cluster {cluster_id}")
            else:
                try:
                    # Call LLM
                    result = labeling_service.generate_cluster_label(metadata_dict)
                    label = result.label
                    
                    # Cache the result
                    if app_state.db_manager:
                        app_state.db_manager.save_cached_llm_label(cache_key, {"label": label})
                        
                except Exception as e:
                    logger.error(f"Error labelling {cluster_id}: {e}")
                    # Retry logic if needed?
            
            if label:
                # Update State
                app_state.current_clusters[cluster_id]['label'] = label
                app_state.labelled_cluster_ids.add(cluster_id)
                
                # Broadcast
                asyncio.run_coroutine_threadsafe(
                    broadcast_label(cluster_id, label),
                    loop
                )
            
            app_state.labelling_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker Exception: {e}")

async def broadcast_label(cluster_id: str, label: str):
    message = json.dumps({
        "type": "label_update",
        "cluster_id": str(cluster_id), # Ensure string for consistency
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

async def broadcast_progress(message_text: str, iteration: Optional[int] = None):
    msg_data = {
        "type": "progress",
        "message": message_text
    }
    if iteration is not None:
        msg_data["iteration"] = iteration

    message = json.dumps(msg_data)
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
    
    # Init TimeFilter lazy
    app_state.time_filter = TimeFilter()
    
    # Init DB Manager
    app_state.db_manager = DatabaseManager()

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

async def broadcast_clusters(clusters_data: List[Dict]):
    message = json.dumps({
        "type": "cluster_update",
        "clusters": clusters_data
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

def update_app_state_with_clustering_results(df: pd.DataFrame, labels: np.ndarray):
    """
    Helper to process clustering results immediately.
    Updates AppState and returns the list of clusters for the frontend.
    """
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
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])
    
    for cluster_id_np in unique_labels:
        cluster_id = int(cluster_id_np)
        cluster_df = df[df['cluster_label'] == cluster_id]
        points = cluster_df[['latitude', 'longitude']].values.tolist()
        
        # Compute Hull
        try:
             hull_points_list = compute_cluster_hulls([points], alpha=None)[0]
             hull_points = hull_points_list
        except Exception as e:
             logger.error(f"Hull error for {cluster_id}: {e}")
             hull_points = points

        # Prepare Metadata
        texts = []
        for col in ['title', 'tags', 'description']:
            if col in cluster_df.columns:
                texts.extend(cluster_df[col].dropna().astype(str).tolist())
        
        valid_texts = [t for t in texts if len(t) > 3]
        text_sample = valid_texts[:50]
        
        # Generate placeholder image URLs for the cluster
        # In a real scenario, these would come from the dataset's image URL column
        sample_ids = cluster_df.index[:20].tolist()
        image_urls = []
        for i, sample_id in enumerate(sample_ids):
            text = f"Cluster+{cluster_id}+Image+{i+1}"
            image_url = f"https://via.placeholder.com/400x300?text={text}"
            image_urls.append(image_url)
        
        app_state.current_clusters[cluster_id] = {
            "size": len(cluster_df),
            "points": hull_points, 
            "text_metadata": text_sample,
            "sample_ids": sample_ids,
            "image_urls": image_urls,
            "label": None
        }
        
        app_state.labelling_queue.put((10, cluster_id, 0)) 
        
        result_clusters.append({
            "id": cluster_id,
            "points": hull_points,
            "size": len(cluster_df)
        })

    return result_clusters

def background_clustering_task(df: pd.DataFrame, params: dict):
    logger.info("Starting background clustering task...")
    
    # Check cache
    try:
        dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
        if app_state.db_manager:
            cached_labels = app_state.db_manager.get_cached_labels(params, dataset_hash)
            if cached_labels is not None:
                logger.info("Found cached clustering results. Skipping computation.")
                asyncio.run_coroutine_threadsafe(
                    broadcast_progress(f"Clustering complete (Cached)."),
                    loop
                )
                result_clusters = update_app_state_with_clustering_results(df, cached_labels)
                asyncio.run_coroutine_threadsafe(
                    broadcast_clusters(result_clusters),
                    loop
                )
                return
    except Exception as e:
        logger.error(f"Error checking cache: {e}")
        dataset_hash = "unknown" # Fallback if hashing fails, though unlikely

    min_cluster_size = 10
    cluster_selection_epsilon = 1/1000.0
    max_cluster_size = 1000
    
    algo = params.get("algorithm", "hdbscan")
    
    if algo == "parallel_hdbscan":
        gen = parallel_hdbscan_iterative_generator(
            df,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            max_cluster_size=max_cluster_size
        )
    else: # Normal iterative hdbscan
        gen = hdbscan_iterative_generator(
            df,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            max_cluster_size=max_cluster_size
        )
    
    # Run generator in a separate thread and use a queue to decouple processing
    result_queue = queue.Queue()
    iteration_times = []
    
    def generator_thread():
        try:
            for item in gen:
                result_queue.put(item)
                if item[0] == 'final':
                    break
        except Exception as e:
            logger.error(f"Generator thread error: {e}")
            
    t = threading.Thread(target=generator_thread, daemon=True)
    t.start()
    
    while True:
        try:
            # Wait for data
            item = result_queue.get(timeout=0.5)
        except queue.Empty:
            if not t.is_alive():
                break
            continue
            
        # Drain the queue to get the latest update (drop everything else)
        while not result_queue.empty():
            try:
                next_item = result_queue.get_nowait()
                item = next_item
                # If we encounter final, we must stop draining and process it
                if item[0] == 'final':
                    break
            except queue.Empty:
                break
                
        status, data = item
        
        if status == "intermediate":
            # Just broadcast visualization
            if isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[1], int):
                clusters_list, iteration = data
            else:
                clusters_list = data
                iteration = None
            
            msg = f"Refining clusters... ({len(clusters_list)} clusters found)"
            if iteration:
                msg += f" [Iter {iteration}]"

            asyncio.run_coroutine_threadsafe(
                broadcast_progress(msg, iteration=iteration),
                loop
            )
            clusters_for_ws = []
            
            # Compute hulls for intermediate clusters (potentially parallelize this if slow)
            # Or just send the first N hulls to save time/bandwidth
            
            # Note: We need temporary IDs.
            # Using points directly might be too heavy if we send ALL points. 
            # compute_cluster_hulls reduces them to polygons.
            
            # Optimization: compute hulls in batches or parallel? 
            # For now, sequential.
            try:
                t0 = time.time()
                # compute_cluster_hulls takes List[List[List[float]]] -> List[List[List[float]]] (hulls)
                # It handles exceptions internally? No, returns list corresponding to inputs.
                hulls_list = compute_cluster_hulls(clusters_list, alpha=None)
                dt = time.time() - t0
                
                logger.info(f"Iteration {iteration if iteration else '?'} hull computation for {len(clusters_list)} clusters took {dt:.4f}s")
                iteration_times.append({
                    "iteration": iteration if iteration is not None else "?",
                    "count": len(clusters_list),
                    "time": dt
                })
                
                for i, hull in enumerate(hulls_list):
                   clusters_for_ws.append({
                       "id": f"temp_{i}",
                       "points": hull,
                       "size": len(clusters_list[i]), 
                       "temp": True
                   })
                   
                asyncio.run_coroutine_threadsafe(
                    broadcast_clusters(clusters_for_ws),
                    loop
                )
            except Exception as e:
                logger.error(f"Error computing intermediate hulls: {e}")
                
        elif status == "final":
             final_clusters_data, n_clusters, final_labels = data
             logger.info(f"Background clustering complete: {n_clusters} clusters.")
             
             # Save to cache
             if app_state.db_manager:
                 try:
                     app_state.db_manager.save_cached_labels(params, dataset_hash, final_labels)
                 except Exception as e:
                     logger.error(f"Error saving to cache: {e}")

             # Log summary report and iteration details
             logger.info("=== Hull Computation Performance Report ===")
             for rec in iteration_times:
                 logger.info(f"  Iter {rec['iteration']}: {rec['count']} clusters -> {rec['time']:.4f}s")
             if iteration_times:
                 avg_time = sum(r['time'] for r in iteration_times) / len(iteration_times)
                 logger.info(f"  Average Time: {avg_time:.4f}s per iteration")
             logger.info("=========================================")

             asyncio.run_coroutine_threadsafe(
                broadcast_progress(f"Clustering complete. Found {n_clusters} clusters."),
                loop
             )
             
             # Update state fully
             result_clusters = update_app_state_with_clustering_results(df, final_labels)
             
             # Broadcast Final Result
             asyncio.run_coroutine_threadsafe(
                broadcast_clusters(result_clusters),
                loop
             )
             break

    t.join()

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
        s, e = params['start_hour'], params['end_hour']
        if s <= e:
            df = df[(df['hour'] >= s) & (df['hour'] <= e)]
        else:
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
        
    # Get Algorithm
    algo = params.get("algorithm", "hdbscan") # default
    
    # If using parallel hdbscan or normal hdbscan, running in background to support iterative updates
    if algo == "parallel_hdbscan" or algo == "hdbscan":
        threading.Thread(target=background_clustering_task, args=(df, params)).start()
        return {"clusters": [], "count": 0, "status": "started"}

    # Standard Blocking Execution (for other algorithms like kmeans, dbscan)
    min_cluster_size = 10
    cluster_selection_epsilon = 1/1000.0 
    max_cluster_size = 1000
    
    try:
        if algo == "kmeans":
            clustered_points_groups, used_k, labels = kmeans_clustering(df)
            
        elif algo == "dbscan":
            clustered_points_groups, used_k, labels = dbscan_clustering(df, eps=0.3, min_samples=5)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algo}")
            
        # Update State & Return
        result_clusters = update_app_state_with_clustering_results(df, labels)
        return {"clusters": result_clusters, "count": len(result_clusters)}

    except Exception as e:
        logger.error(f"Clustering error ({algo}): {e}")
        return {"clusters": [], "count": 0, "error": str(e)}

def get_flickr_image_url(user_id, photo_id):
    try:
        photo_url = f"https://www.flickr.com/photos/{user_id}/{photo_id}/"
        oembed_url = f"https://www.flickr.com/services/oembed/?url={requests.utils.quote(photo_url)}&format=json"
        # Use a short timeout to avoid hanging
        resp = requests.get(oembed_url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            html = data.get("html", "")
            # Extract src using regex
            match = re.search(r'src="([^"]+)"', html)
            if match:
                return match.group(1)
    except Exception as e:
        logger.warning(f"Failed to fetch Flickr URL for {user_id}/{photo_id}: {e}")
    return None

@app.get("/api/cluster_images")
async def get_cluster_images(cluster_id: str):
    """
    Return image URLs for a specific cluster.
    Resolves real Flickr URLs using oEmbed.
    """
    # Handle temporary clusters or non-integer IDs
    try:
        c_id = int(cluster_id)
    except ValueError:
        return {"images": [], "error": "Cannot fetch images for temporary or invalid cluster ID"}

    cluster_data = app_state.current_clusters.get(c_id)
    if cluster_data is None:
        return {"images": [], "error": "Cluster not found"}
    
    sample_ids = cluster_data.get("sample_ids", [])
    
    # We want max 20 images
    target_ids = sample_ids[:20]
    
    async def fetch_url(idx):
        try:
            # Safely get row from app_state.df
            if idx not in app_state.df.index:
                return None
            
            row = app_state.df.loc[idx]
            user_id = row['user']
            photo_id = row['id']
            
            # Helper runs synchronously, so we wrap it
            src_url = await asyncio.to_thread(get_flickr_image_url, user_id, photo_id)
            if src_url:
                return {
                    "url": src_url,
                    "page_url": f"https://www.flickr.com/photos/{user_id}/{photo_id}/"
                }
            return None
        except Exception as e:
            logger.error(f"Error resolving image {idx}: {e}")
            return None

    # Run fetches in parallel
    tasks = [fetch_url(idx) for idx in target_ids]
    results = await asyncio.gather(*tasks)
    
    image_data = [res for res in results if res]
    
    # Fallback if no real images found
    if not image_data:
         image_data.append({
             "url": f"https://placehold.co/400x300?text=Cluster+{cluster_id}",
             "page_url": None
         })
    
    return {"images": image_data, "cluster_id": cluster_id}


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
                try:
                    cluster_id = int(msg.get("cluster_id"))
                    if cluster_id in app_state.current_clusters and cluster_id not in app_state.labelled_cluster_ids:
                        logger.info(f"Boosting priority for cluster {cluster_id}")
                        app_state.labelling_queue.put((0, cluster_id, 0)) # Priority 0 (High)
                except (ValueError, TypeError):
                    # Ignore invalid or temporary cluster IDs
                    pass
                    
    except WebSocketDisconnect:
        app_state.active_websockets.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
