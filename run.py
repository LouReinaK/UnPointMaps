from src.clustering.hdbscan_clustering import hdbscan_clustering, hdbscan_clustering_iterative
from src.clustering.clustering import plot_clusters, plot_k_distance
from src.processing.dataset_filtering import convert_to_dict_filtered
from src.visualization.map_visualisation import Visualisation
from src.utils.hull_logic import compute_cluster_hulls
from src.processing.llm_labelling import ConfigManager, LLMLabelingService
from src.database.manager import DatabaseManager
import webbrowser
import sys
import pandas as pd
import numpy as np

# Cache Setting
USE_CACHE = True
DB_PATH = "unpointmaps_cache.db"


TEST_CLUSTERS = False # Set to True to only test clustering and exit
TEST_HULLS = False    # Set to True to preview hulls on matplotlib and exit
ADD_POINTS = False  # Set to True to add individual points to the map
AUTO_OPEN_BROWSER = True  # Set to True to automatically open the generated map in a web browser

print("--- Starting UnPointMaps ---")
print("Loading and filtering dataset...")
df = convert_to_dict_filtered()
limit = -1
if limit != -1:
    print(f"Limiting dataset to first {limit} points for performance...")
    df = df.head(limit)

# HDBSCAN settings
min_cluster_size = 10
cluster_selection_epsilon = 1/1000.0  # Approx 1 meter in degrees
max_cluster_size = 1000  # Maximum allowed cluster size

print(f"Starting iterative clustering with HDBSCAN...")
print(f"  min_cluster_size={min_cluster_size}")
print(f"  cluster_selection_epsilon={cluster_selection_epsilon}")
print(f"  max_cluster_size={max_cluster_size}")
print()

# Prepare Cache and Parameters
db_manager = DatabaseManager(DB_PATH)
clustering_params = {
    "min_cluster_size": min_cluster_size,
    "cluster_selection_epsilon": cluster_selection_epsilon,
    "max_cluster_size": max_cluster_size,
    "limit": limit
}
# Simple signature for dataset
dataset_signature = "flickr_data2.csv" 

cached_run_id = db_manager.get_cached_run(clustering_params, dataset_signature) if USE_CACHE else None
clustered_points = []
used_k = 0
llm_labels = {}
labels = None

if cached_run_id:
    print(f"Found cached run (ID: {cached_run_id}). Loading data from database...")
    clustered_points, llm_labels = db_manager.load_run_data(cached_run_id)
    used_k = len(clustered_points)
    print(f"Loaded {used_k} clusters from cache.")
    # Mock labels to avoid errors
    labels = np.full(len(df), -1) 
else:
    clustered_points, used_k, labels = hdbscan_clustering_iterative(
        df, 
        min_cluster_size=min_cluster_size, 
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size
    )
    print(f"Clustering complete. Found {used_k} clusters.")

# LLM Labeling Integration
# LLM Labeling Integration
if not cached_run_id: # Only run if not cached
    llm_labels = {}
    try:
        print("\n--- Starting LLM Labeling ---")
        # Initialize ConfigManager and Service
        # Note: Ensure .env file exists with OPENROUTER_API_KEY
        config_manager = ConfigManager()
        labeling_service = LLMLabelingService(config_manager)
        
        print(f"Labelling {used_k} clusters using {config_manager.get_model()}...")
        
        for i in range(used_k):
            # Extract cluster points from original dataframe
            # labels numpy array matches df rows
            if labels is None: # Should not happen if not cached
                continue
                
            cluster_mask = (labels == i)
            cluster_df = df[cluster_mask]
            
            if cluster_df.empty:
                continue
                
            # Prepare text metadata from title and tags
            # Limit to top 50 points to fit in context window and save tokens
            text_metadata = []
            for _, row in cluster_df.head(50).iterrows():
                parts = []
                if pd.notna(row.get('title')) and str(row['title']).strip():
                    parts.append(str(row['title']))
                if pd.notna(row.get('tags')) and str(row['tags']).strip():
                    # Tags are often space separated in Flickr
                    parts.append(str(row['tags']))
                
                if parts:
                    text_metadata.append(" ".join(parts))
            
            # Fallback if no text data
            if not text_metadata:
                 text_metadata = ["No description available"]

            # Construct metadata object
            cluster_meta = {
                "cluster_id": f"cluster_{i}",
                "image_ids": [str(x) for x in cluster_df.index[:10]],
                "text_metadata": text_metadata, # Service handles joining
                "cluster_size": len(cluster_df)
            }
            
            # Generate Label
            try:
                print(f"  Processing Cluster {i} ({len(cluster_df)} points)... ", end="", flush=True)
                result = labeling_service.generate_cluster_label(cluster_meta)
                llm_labels[i] = result.label
                print(f"Done -> '{result.label}'")
            except Exception as e:
                print(f"Failed -> {e}")
                llm_labels[i] = f"Cluster {i}"
        
        # Save to Cache
        if USE_CACHE:
            print("Saving results to cache...")
            db_manager.save_run(clustering_params, dataset_signature, clustered_points, llm_labels)

    except Exception as e:
        print(f"LLM Labeling skipped or failed: {e}")
        # Prepare default labels if LLM fails
        for i in range(used_k):
            llm_labels[i] = f"Cluster {i}"

if TEST_CLUSTERS:
    # Use matplotlib for faster visualisation
    plot_clusters(clustered_points)

    # Plot histogram of cluster sizes
    if not clustered_points:
         cluster_sizes = []
    else:
         cluster_sizes = [len(cluster) for cluster in clustered_points]
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(cluster_sizes, bins=30, edgecolor='black')
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster Size')
    plt.ylabel('Frequency')
    plt.show()

    sys.exit(0)


if TEST_HULLS:
    if cached_run_id:
        print("Warning: Skipping TEST_HULLS because data was loaded from cache (labels array incomplete).")
    else:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        import numpy as np

        print("Previewing hulls comparison on Matplotlib...")
        alpha_values = [None, 1/1000, 5/1000, 1/100, 5/100]
        
        # Calculate grid size
        n_plots = len(alpha_values)
        cols = 2
        rows = (n_plots + 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 7 * rows))
        axes = axes.flatten()

        # Pre-calculate noise points
        noise_mask = (labels == -1)
        noise_points = df[noise_mask]
        
        # Simple list of colors for the clusters
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'olive']

        for idx, alpha in enumerate(alpha_values):
            ax = axes[idx]
            print(f"  Computing hulls for alpha={alpha}...")
            hulls = compute_cluster_hulls(clustered_points, alpha=alpha, auto_alpha_quantile=0.9)

            # Plot noise points in background
            ax.scatter(noise_points['longitude'], noise_points['latitude'], 
                      c='lightgray', s=5, alpha=0.3, label='Noise')

            for i, hull in enumerate(hulls):
                # Plot cluster points
                cluster = clustered_points[i]
                cluster_array = np.array(cluster)
                clat, clon = cluster_array[:, 0], cluster_array[:, 1]
                
                color = colors[i % len(colors)]
                ax.scatter(clon, clat, c=color, s=15, alpha=0.9, edgecolors='none')

                # Plot hull
                for poly_points in hull:
                    hull_array = np.array(poly_points)
                    if len(hull_array) >= 3:
                        lon_lat_hull = hull_array[:, [1, 0]]
                        poly = Polygon(lon_lat_hull.tolist(), closed=True, fill=True, alpha=0.3, 
                                      facecolor=color, edgecolor='black', linewidth=1)
                        ax.add_patch(poly)
                    elif len(hull_array) > 0:
                        ax.plot(hull_array[:, 1], hull_array[:, 0], color=color, linestyle='-', alpha=0.5)

            ax.set_title(f"Alpha = {alpha}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, alpha=0.2)
        
        # Hide unused subplots
        for j in range(n_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Concave Hull Comparison ({used_k} clusters)", fontsize=18, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        sys.exit(0)


vis = Visualisation()

if ADD_POINTS:
    if cached_run_id:
        print("Warning: Noise points cannot be accurately displayed when loading from cache (re-run logic required).")
    else:
        # Only add noise points (label -1) to the map
        noise_mask = labels == -1
        noise_points = df[noise_mask]
        print(f"Adding {len(noise_points)} noise points to map...")
        for _, row in noise_points.iterrows():
            vis.draw_point(row)

print("Computing cluster hulls...")
# Alpha parameter controls hull tightness:
# - Smaller values (e.g., 0.1-0.5) = tighter, more concave hulls
# - Larger values (e.g., 1.0-5.0) = looser hulls, closer to convex hull
# - None = auto-compute based on 'average' density of EACH cluster (Scaling)
alpha_value = 1/1000  # Set to None to enable auto-scaling
alpha_quantile = 0.9 # Only used if alpha_value is None. 0.95 = Keep 95% of triangles (pretty loose).

hulls = compute_cluster_hulls(clustered_points, alpha=alpha_value, auto_alpha_quantile=alpha_quantile)

print(f"Adding {len(hulls)} hulls to map...")
for i, hull in enumerate(hulls):
    # Retrieve label for this cluster
    label_text = llm_labels.get(i, f"Cluster {i}")
    
    for poly_points in hull:
        vis.draw_cluster(poly_points, popup=label_text, tooltip=label_text)

print("Generating output map 'output_map.html'...")
# Compute optimized extremes from dataframe for bounds if possible
try:
    min_lat = df["latitude"].min()
    max_lat = df["latitude"].max()
    min_lon = df["longitude"].min()
    max_lon = df["longitude"].max()
    bounds = [[min_lat, min_lon], [max_lat, max_lon]]
except Exception:
    bounds = None

vis.create_map("output_map.html", bounds=bounds)

if AUTO_OPEN_BROWSER:
    print("Opening map in web browser...")
    webbrowser.open("output_map.html")

print("--- Done ---")

