from typing import List, Dict, Optional, Set, Any
from contextlib import asynccontextmanager
import threading
import asyncio
import json
import logging
import queue
import time
import urllib.parse
import hashlib
import re
import requests
import pandas as pd
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Project imports
from src.processing.embedding_service import EmbeddingService
from src.processing.dataset_filtering import convert_to_dict_filtered
from src.processing.time_filtering import TimeFilter
from src.clustering.hdbscan_clustering import hdbscan_iterative_generator
from src.clustering.clustering import kmeans_clustering
from src.clustering.dbscan_clustering import dbscan_clustering
from src.clustering.parallel_hdbscan_clustering import parallel_hdbscan_iterative_generator
from src.processing.llm_labelling import ConfigManager, LLMLabelingService
from src.processing.remove_nonsignificative_words import clean_text_list
from src.processing.TFIDF import get_top_keywords
from src.utils.hull_logic import compute_cluster_hulls
from src.database.manager import DatabaseManager
from src.processing.tram_line import compute_tram_line


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnPointMaps_Server")

# Global State
class AppState:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.time_filter: Optional[TimeFilter] = None # Lazy init
        self.events = []
        self.current_clusters: Dict[Any, Dict] = {} # cluster_id -> {points, metadata, status, label}
        self.active_websockets: List[WebSocket] = []
        self.labelling_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.labelled_cluster_ids: Set[int] = set()
        self.stop_labelling_flag = threading.Event()
        self.labelling_thread: Optional[threading.Thread] = None
        self.db_manager: Optional[DatabaseManager] = None 
        self.embedding_service: Optional[EmbeddingService] = None
        self.labelling_method: str = "llm" # "llm" or "statistical"
        # Hull computation
        self.hull_queue: queue.Queue = queue.Queue()
        self.hull_thread: Optional[threading.Thread] = None
        self.stop_hull_flag = threading.Event()
        # Track which clusters have been broadcasted to frontend
        self.broadcasted_clusters: Set[Any] = set()

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
            
            # Compute metadata if not already done
            if not cluster_data.get('text_metadata'):
                cluster_indices = cluster_data['cluster_indices']
                cluster_df = app_state.df.loc[cluster_indices]
                
                # Prepare Metadata from Representative Sample (Closest to Centroid)
                sample_ids = []
                representative_df = pd.DataFrame()

                # Try Semantic Selection
                if app_state.embedding_service and app_state.embedding_service.get_embedding_model():
                    try:
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
                        if temp_data:
                            top_local_indices = app_state.embedding_service.select_representative_indices(temp_data, top_k=50)
                            
                            representative_df = cluster_df.iloc[top_local_indices]
                            sample_ids = representative_df.index.tolist()
                    except Exception as e:
                        logger.error(f"Semantic selection failed for cluster {cluster_id}: {e}")
                        sample_ids = []
                
                # Fallback: Spatial Centroid
                if not sample_ids:
                    try:
                        lats = np.array(cluster_df['latitude'].values.astype(float)).astype(np.float64)
                        lons = np.array(cluster_df['longitude'].values.astype(float)).astype(np.float64)
                        centroid_lat: float = np.mean(lats)
                        centroid_lon: float = np.mean(lons)
                        dists = (lats - centroid_lat)**2 + (lons - centroid_lon)**2
                        sorted_indices = np.argsort(dists)
                        n_sample = min(50, len(cluster_df))
                        top_indices = sorted_indices[:n_sample]
                        representative_df = cluster_df.iloc[top_indices]
                        sample_ids = representative_df.index.tolist()
                    except Exception as e:
                        logger.error(f"Spatial fallback failed for cluster {cluster_id}: {e}")
                        representative_df = cluster_df.head(50)
                        sample_ids = representative_df.index.tolist()

                # Extract Text
                texts = []
                for col in ['title', 'tags', 'description']:
                    if col in representative_df.columns:
                        texts.extend(representative_df[col].dropna().astype(str).tolist())
                
                cleaned_texts = clean_text_list(texts)
                valid_texts = [t for t in cleaned_texts if len(t) > 3]
                text_sample = valid_texts[:50]

                # Extract Keywords
                keywords = get_top_keywords(valid_texts, top_n=5, stop_words=None)
                keyword_list = [k[0] for k in keywords]
                
                provisional_label = ", ".join(keyword_list[:3]) if keyword_list else "Processing..."

                # Image URLs
                image_urls = []
                for i, sample_id in enumerate(sample_ids):
                    text = f"Cluster+{cluster_id}+Image+{i+1}"
                    image_url = f"https://via.placeholder.com/400x300?text={text}"
                    image_urls.append(image_url)
                
                # Update cluster_data
                cluster_data['text_metadata'] = text_sample
                cluster_data['keywords'] = keyword_list
                cluster_data['sample_ids'] = sample_ids
                cluster_data['image_urls'] = image_urls
                cluster_data['label'] = provisional_label
                
                # Broadcast provisional label
                if loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        broadcast_label(cluster_id, provisional_label),
                        loop
                    )
            
            # Broadcast start of labelling
            if loop is not None:
                asyncio.run_coroutine_threadsafe(
                    broadcast_progress(f"Labelling cluster {cluster_id}..."),
                    loop
                )
            
            # Prepare metadata for LLM
            metadata_dict = {
                "cluster_id": str(cluster_id),
                "image_ids": [str(x) for x in cluster_data['sample_ids']],
                "text_metadata": cluster_data['text_metadata'],
                "cluster_size": cluster_data['size']
            }
            
            # Determine Method
            if app_state.labelling_method == "statistical":
                # Statistical approach: Use top keywords
                if loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        broadcast_progress(f"Generating statistical label for cluster {cluster_id}..."),
                        loop
                    )
                keywords = cluster_data.get('keywords', [])
                if keywords:
                    label = ", ".join(keywords[:3]) # Top 3 keywords
                else:
                    label = "Unlabeled Cluster"
                logger.info(f"Statistical label for cluster {cluster_id}: {label}")
                
            else: 
                # LLM approach (Default)
                
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
                    if loop is not None:
                        asyncio.run_coroutine_threadsafe(
                            broadcast_progress(f"Using cached label for cluster {cluster_id}"),
                            loop
                        )
                else:
                    try:
                        if loop is not None:
                            asyncio.run_coroutine_threadsafe(
                                broadcast_progress(f"Preparing LLM prompt for cluster {cluster_id}..."),
                                loop
                            )
                        # Call LLM
                        result = labeling_service.generate_cluster_label(metadata_dict)
                        label = result.label
                        
                        if loop is not None:
                            asyncio.run_coroutine_threadsafe(
                                broadcast_progress(f"LLM generated label for cluster {cluster_id}"),
                                loop
                            )
                        
                        # Cache the result
                        if app_state.db_manager:
                            app_state.db_manager.save_cached_llm_label(cache_key, {"label": label})
                            
                    except Exception as e:
                        logger.error(f"Error labelling {cluster_id}: {e}")
                        if loop is not None:
                            asyncio.run_coroutine_threadsafe(
                                broadcast_progress(f"Failed to label cluster {cluster_id}: {str(e)}"),
                                loop
                            )
            
            if label:
                # Update State
                app_state.current_clusters[cluster_id]['label'] = label
                app_state.labelled_cluster_ids.add(cluster_id)
                
                # Broadcast
                if loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        broadcast_label(cluster_id, label),
                        loop
                    )
            
            app_state.labelling_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker Exception: {e}")

# Hull Computation Worker
def hull_worker():
    logger.info("Hull computation worker started.")
    
    while not app_state.stop_hull_flag.is_set():
        try:
            # Get hull computation request: (cluster_id, points_list)
            cluster_id, points_list = app_state.hull_queue.get(timeout=1)
            
            logger.info(f"Computing hull for cluster {cluster_id}")
            
            # Compute hull
            hull = []
            try:
                # Use concave hull with fixed alpha for more consistent results
                hulls_list = compute_cluster_hulls([points_list], alpha=0.01, auto_alpha_quantile=0.5)
                hull = hulls_list[0] if hulls_list else []
                
                logger.info(f"Hull computed for cluster {cluster_id}: {len(points_list)} input points -> {len(hull)} hull points")
                
            except Exception as e:
                logger.error(f"Hull computation failed for {cluster_id}: {e}")
            
            # Update cluster data
            if cluster_id in app_state.current_clusters:
                app_state.current_clusters[cluster_id]["points"] = hull
                
                # Check if this cluster has been broadcasted to frontend
                if cluster_id not in app_state.broadcasted_clusters:
                    # First time - broadcast as new cluster
                    if loop is not None:
                        cluster_data = app_state.current_clusters[cluster_id]
                        cluster_message = {
                            "type": "cluster_update",
                            "clusters": [{
                                "id": str(cluster_id),
                                "points": hull,
                                "size": cluster_data["size"],
                                
                                "label": cluster_data.get("label", "Processing...")
                            }]
                        }
                        asyncio.run_coroutine_threadsafe(
                            broadcast_clusters([{
                                "id": str(cluster_id),
                                "points": hull,
                                "size": cluster_data["size"],
                                
                                "label": cluster_data.get("label", "Processing...")
                            }]),
                            loop
                        )
                    app_state.broadcasted_clusters.add(cluster_id)
                else:
                    # Already broadcasted - send hull update
                    if loop is not None:
                        cluster_data = app_state.current_clusters[cluster_id]
                        update_message = {
                            "type": "hull_update",
                            "cluster_id": str(cluster_id),
                            "points": hull,
                            "size": cluster_data["size"]
                        }
                        asyncio.run_coroutine_threadsafe(
                            broadcast_hull_update(update_message),
                            loop
                        )
            
            app_state.hull_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Hull worker exception: {e}")

async def broadcast_hull_update(update_data: Dict):
    message = json.dumps(update_data)
    to_remove = []
    for ws in app_state.active_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            to_remove.append(ws)
    for ws in to_remove:
        if ws in app_state.active_websockets:
            app_state.active_websockets.remove(ws)

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
    msg_data: Dict[str, Any] = {
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
loop: Optional[asyncio.AbstractEventLoop] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global loop
    loop = asyncio.get_running_loop()
    
    # Init TimeFilter lazy
    app_state.time_filter = TimeFilter()
    
    # Init DB Manager
    app_state.db_manager = DatabaseManager()

    # Init Embedding Service
    app_state.embedding_service = EmbeddingService.get_instance()
    app_state.embedding_service.load_model()

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
    
    # Start Hull Worker
    app_state.hull_thread = threading.Thread(target=hull_worker, daemon=True)
    app_state.hull_thread.start()
    
    yield
    
    # Cleanup
    app_state.stop_labelling_flag.set()
    app_state.stop_hull_flag.set()
    if app_state.labelling_thread:
        app_state.labelling_thread.join()
    if app_state.hull_thread:
        app_state.hull_thread.join()

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

async def broadcast_cluster_remove(cluster_ids: List[Any]):
    message = json.dumps({
        "type": "cluster_remove",
        "cluster_ids": [str(cid) for cid in cluster_ids]
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

def update_app_state_with_clustering_results(df: pd.DataFrame, labels: np.ndarray, enqueue: bool = True):
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
        except Exception:
            pass
    
    app_state.labelled_cluster_ids.clear()
    app_state.current_clusters.clear()
    app_state.broadcasted_clusters.clear()
    
    result_clusters = []
    unique_labels = sorted([label for label in np.unique(labels) if label != -1])
    
    for cluster_id_np in unique_labels:
        cluster_id = int(cluster_id_np)
        cluster_df = df[df['cluster_label'] == cluster_id]
        points = cluster_df[['latitude', 'longitude']].values.tolist()
        
        # Store basic info with placeholder hull (raw points); proper hull computation will be enqueued
        cluster_indices = cluster_df.index.tolist()
        
        app_state.current_clusters[cluster_id] = {
            "size": len(cluster_df),
            "points": points,  # Placeholder: raw points - will be updated when proper hull computation completes
            "text_metadata": [],  # placeholder
            "keywords": [],  # placeholder
            "sample_ids": [],  # placeholder
            "image_urls": [],  # placeholder
            "cluster_indices": cluster_indices,  # for metadata computation
            "label": "Processing..."
        }
        
        # Enqueue hull computation
        app_state.hull_queue.put((cluster_id, points))
        
        if enqueue:
            app_state.labelling_queue.put((10, cluster_id, 0)) 
        
        result_clusters.append({
            "id": cluster_id,
            "points": [], # Don't send hull points yet
            "size": len(cluster_df),
            "label": "Processing..."
        })

    return result_clusters

def background_clustering_task(df: pd.DataFrame, params: dict, total_start: float, filter_time: float):
    logger.info("Starting background clustering task...")
    
    if loop is not None:
        asyncio.run_coroutine_threadsafe(
            broadcast_progress("Initializing clustering algorithm..."),
            loop
        )
    
    # Check cache
    cache_check_start = time.time()
    try:
        dataset_hash = hashlib.sha256(np.array(pd.util.hash_pandas_object(df, index=True)).tobytes()).hexdigest()
        if app_state.db_manager:
            cached_labels = app_state.db_manager.get_cached_labels(params, dataset_hash)
            if cached_labels is not None:
                cache_check_time = time.time() - cache_check_start
                logger.info(f"Cache check took {cache_check_time:.4f}s - Found cached clustering results. Skipping computation.")
                n_clusters = len(set(cached_labels)) - (1 if -1 in cached_labels else 0)  # Exclude noise
                if loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        broadcast_progress(f"Clustering complete (Cached). Found {n_clusters} clusters."),
                        loop
                    )
                result_clusters = update_app_state_with_clustering_results(df, cached_labels, enqueue=False)
                if loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        broadcast_clusters(result_clusters),
                        loop
                    )
                
                # Now enqueue for labelling
                for cluster in result_clusters:
                    app_state.labelling_queue.put((10, cluster['id'], 0))

                total_time = time.time() - total_start
                logger.info(f"Total clustering time (cached): {total_time:.4f}s")
                return

    except Exception as e:
        logger.error(f"Error checking cache: {e}")
        dataset_hash = "unknown" # Fallback if hashing fails, though unlikely
    cache_check_time = time.time() - cache_check_start
    logger.info(f"Cache check took {cache_check_time:.4f}s - No cache found, proceeding with computation.")

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
    result_queue: queue.Queue = queue.Queue()
    iteration_times: List[Dict[str, Any]] = []
    compute_start = time.time()
    
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
            # Store intermediate clusters but don't broadcast until hulls are computed
            if isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[1], int):
                clusters_list, iteration = data
            else:
                clusters_list = data
                iteration = None
            
            msg = f"Refining clusters... ({len(clusters_list)} clusters found)"
            if iteration:
                msg += f" [Iter {iteration}]"

            if loop is not None:
                asyncio.run_coroutine_threadsafe(
                    broadcast_progress(msg, iteration=iteration),
                    loop
                )
            
            # Track previous cluster IDs before updating
            previous_cluster_ids = set(app_state.current_clusters.keys())
            
            # Store clusters and enqueue hull computation (don't broadcast yet)
            # Only compute hulls for new clusters to avoid duplicate work
            current_cluster_ids = set()
            for i, cluster_data in enumerate(clusters_list):
                cid = i
                current_cluster_ids.add(cid)
                
                # Extract points and indices
                if isinstance(cluster_data, (tuple, list)) and len(cluster_data) >= 1:
                    points = cluster_data[0]
                else:
                    points = cluster_data
                
                if isinstance(cluster_data, (tuple, list)) and len(cluster_data) >= 2:
                    indices = cluster_data[1]
                    size = len(indices)
                    sample_ids = indices[:20]
                else:
                    indices = list(range(len(points)))
                    size = len(points)
                    sample_ids = indices[:20]
                
                # Check if cluster already exists in app_state to avoid duplicate computation
                if cid not in app_state.current_clusters:
                    # Store in app_state with raw points (no placeholder hull)
                    app_state.current_clusters[cid] = {
                         "size": size,
                         "points": points,  # Raw points - will be replaced with hull when computed
                         "text_metadata": [],
                         "keywords": [],
                         "sample_ids": sample_ids,
                         "image_urls": [], 
                         "label": "Clustering..."
                    }
                    
                    # Enqueue hull computation only for new clusters
                    app_state.hull_queue.put((cid, points))
            
            # Find clusters that were removed (split)
            removed_cluster_ids = previous_cluster_ids - current_cluster_ids
            
            # Send cluster remove messages for removed clusters
            if removed_cluster_ids and loop is not None:
                asyncio.run_coroutine_threadsafe(
                    broadcast_cluster_remove(list(removed_cluster_ids)),
                    loop
                )
            
            # Remove old clusters from app_state
            for cid in removed_cluster_ids:
                del app_state.current_clusters[cid]
                if cid in app_state.broadcasted_clusters:
                    app_state.broadcasted_clusters.remove(cid)
                
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

            if loop is not None:
                asyncio.run_coroutine_threadsafe(
                    broadcast_progress(f"Clustering complete. Found {n_clusters} clusters."),
                    loop
                )
            
            # Update state fully (hulls will be computed asynchronously)
            result_clusters = update_app_state_with_clustering_results(df, final_labels, enqueue=False)
            
            # Don't broadcast yet - wait for hulls to be computed
            
            # Now enqueue for labelling
            for cluster in result_clusters:
                app_state.labelling_queue.put((10, cluster['id'], 0))
                
            break

    t.join()
    compute_time = time.time() - compute_start
    logger.info(f"Clustering computation took {compute_time:.4f}s")
    total_time = time.time() - total_start
    logger.info(f"Total clustering time: {total_time:.4f}s")
    
    # Sort tasks by time (descending)
    task_times = [
        ("Data filtering", filter_time),
        ("Cache check", cache_check_time),
        ("Clustering computation", compute_time),
        ("State update", 0),  # Not measured in background
    ]
    task_times.sort(key=lambda x: x[1], reverse=True)
    logger.info("Tasks sorted by time (descending):")
    for task, t in task_times:
        if t > 0:
            logger.info(f"  {task}: {t:.4f}s")
    
    # Sort tasks by time (descending)
    task_times = [
        ("Data filtering", filter_time),
        ("Cache check", cache_check_time),
        ("Clustering computation", compute_time if 'compute_time' in locals() else 0),
        ("State update", 0),  # Not measured in background
    ]
    task_times.sort(key=lambda x: x[1], reverse=True)
    logger.info("Tasks sorted by time (descending):")
    for task, t in task_times:
        if t > 0:
            logger.info(f"  {task}: {t:.4f}s")

@app.post("/api/cluster")
async def run_clustering(params: dict):
    """
    Params:
    {
        "min_year": 2010,
        "max_year": 2015,
        "start_date": "2010-01-01",
        "end_date": "2015-12-31",
        "start_hour": 8,
        "end_hour": 18,
        "exclude_events": [1, 2] # indices of events to exclude
    }
    """
    logger.info(f"Clustering request: {params}")
    total_start = time.time()

    # Set Labelling Method
    if "labelling_method" in params:
        app_state.labelling_method = params["labelling_method"]
        logger.info(f"Labelling method set to: {app_state.labelling_method}")
    
    # Filter Data
    if app_state.df is None:
        return {"clusters": [], "count": 0, "error": "Data not loaded"}
    df = app_state.df.copy()
    
    filter_start = time.time()
    # Year Filter
    if "min_year" in params and "max_year" in params:
        df = df[
            (df['date'].dt.year >= params['min_year']) & 
            (df['date'].dt.year <= params['max_year'])
        ]
        
    # Date Filter
    if "start_date" in params and "end_date" in params:
        start_date = pd.to_datetime(params['start_date'])
        end_date = pd.to_datetime(params['end_date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
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
        if events_to_exclude and app_state.time_filter is not None:
            df = app_state.time_filter.exclude_events(df, events_to_exclude)

    # Include Events (Mutual Exclusion usually, or sequential filter?)
    if "include_events" in params and params['include_events']:
        events_to_include = []
        for idx in params['include_events']:
            if 0 <= idx-1 < len(app_state.events):
                events_to_include.append(app_state.events[idx-1])
        if events_to_include and app_state.time_filter is not None:
            df = app_state.time_filter.filter_by_events(df, events_to_include, exclude=False)

    filter_time = time.time() - filter_start
    logger.info(f"Data filtering took {filter_time:.4f}s, resulting in {len(df)} points")

    if len(df) == 0:
        total_time = time.time() - total_start
        logger.info(f"Total clustering time: {total_time:.4f}s (no data)")
        return {"clusters": [], "count": 0}
        
    # Get Algorithm
    algo = params.get("algorithm", "hdbscan") # default
    
    # If using parallel hdbscan or normal hdbscan, running in background to support iterative updates
    if algo == "parallel_hdbscan" or algo == "hdbscan":
        threading.Thread(target=background_clustering_task, args=(df, params, total_start, filter_time)).start()
        return {"clusters": [], "count": 0, "status": "started"}

    # Standard Blocking Execution (for other algorithms like kmeans, dbscan)
    min_cluster_size = 10
    cluster_selection_epsilon = 1/1000.0 
    max_cluster_size = 1000
    
    try:
        compute_start = time.time()
        if algo == "kmeans":
            clustered_points_groups, used_k, labels = kmeans_clustering(df)
            
        elif algo == "dbscan":
            clustered_points_groups, used_k, labels = dbscan_clustering(df, eps=0.3, min_samples=5)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algo}")
            
        compute_time = time.time() - compute_start
        logger.info(f"Clustering computation ({algo}) took {compute_time:.4f}s")
        
        # Update State & Return
        update_start = time.time()
        result_clusters = update_app_state_with_clustering_results(df, labels)
        update_time = time.time() - update_start
        logger.info(f"State update took {update_time:.4f}s")
        
        total_time = time.time() - total_start
        logger.info(f"Total clustering time: {total_time:.4f}s")
        return {"clusters": result_clusters, "count": len(result_clusters)}

    except Exception as e:
        logger.error(f"Clustering error ({algo}): {e}")
        total_time = time.time() - total_start
        logger.info(f"Total clustering time (with error): {total_time:.4f}s")
        return {"clusters": [], "count": 0, "error": str(e)}

def get_flickr_image_url(user_id, photo_id):
    try:
        photo_url = f"https://www.flickr.com/photos/{user_id}/{photo_id}/"
        oembed_url = f"https://www.flickr.com/services/oembed/?url={urllib.parse.quote(photo_url)}&format=json"
        # Use a short timeout to avoid hanging
        headers = {
            'User-Agent': 'UnPointMaps/1.0 (https://github.com/LouReinaK/UnPointMaps)'
        }
        resp = requests.get(oembed_url, headers=headers, timeout=3)
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
    # Convert cluster_id to appropriate type
    try:
        c_id = int(cluster_id)
    except ValueError:
        c_id = cluster_id

    cluster_data = app_state.current_clusters.get(c_id)
    if cluster_data is None:
        return {"images": [], "error": f"Cluster {c_id} not found"}
    
    sample_ids = cluster_data.get("sample_ids", [])
    
    # We want max 20 images
    target_ids = sample_ids[:20]
    
    async def fetch_url(idx):
        try:
            # Safely get row from app_state.df
            if app_state.df is None or idx not in app_state.df.index:
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


@app.post("/api/tram_line")
async def compute_tram_line_endpoint(params: dict):
    """
    Compute optimal tram line path.
    
    Params:
    {
        "max_length": 0.01  # in degrees (approx 1km)
    }
    """
    max_length = params.get("max_length", 0.01)
    if max_length <= 0:
        return {"error": "max_length must be positive"}
    
    # Get current clusters
    clusters = []
    for cluster_id, cluster_data in app_state.current_clusters.items():
        clusters.append({
            "id": cluster_id,
            "points": cluster_data["points"],
            "size": cluster_data["size"]
        })
    
    if not clusters:
        return {"path": [], "error": "No clusters available"}
    
    path = compute_tram_line(clusters, max_length)
    
    return {"path": path}


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
