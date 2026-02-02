import webbrowser
import sys
import multiprocessing
import pandas as pd

from src.clustering.hdbscan_clustering import hdbscan_clustering_iterative
from src.clustering.parallel_hdbscan_clustering import parallel_hdbscan_clustering_iterative
from src.clustering.clustering import plot_clusters
from src.processing.dataset_filtering import convert_to_dict_filtered
from src.visualization.map_visualisation import Visualisation
from src.utils.hull_logic import compute_cluster_hulls
from src.processing.llm_labelling import ConfigManager, LLMLabelingService
from src.database.manager import DatabaseManager
from src.processing.remove_nonsignificative_words import clean_text_list, set_cache_enabled
from src.cli_menu import CLIMenu
from src.utils.dependency_utils import HAS_HDBSCAN, HAS_OPENAI, HAS_FOLIUM, HAS_MATPLOTLIB

# Configuration
DB_PATH = "unpointmaps_cache.db"
USE_PARALLEL_CLUSTERING = True  # Enable parallel processing by default
DATASET_FILE = "flickr_data2.csv"

# Cache Settings
USE_DATA_CACHE = True
USE_CLUSTERING_CACHE = True
USE_TEXT_CLEANING_CACHE = True
USE_LLM_CACHE = True

# Clustering Parameters
MIN_CLUSTER_SIZE = 3
CLUSTER_SELECTION_EPSILON = 5 / 10000.0  # Approx 0.5 meters in degrees
MAX_STD_DEV = 30.0  # Maximum allowed standard deviation of cluster sizes
LIMIT = -1  # Limiting dataset size (-1 for no limit)

# Hull Generation Parameters
# Alpha parameter controls hull tightness:
# - Smaller values (e.g., 0.1-0.5) = tighter, more concave hulls
# - Larger values (e.g., 1.0-5.0) = looser hulls, closer to convex hull
# - None = auto-compute based on 'average' density of EACH cluster (Scaling)
ALPHA_VALUE = 1 / 1000  # Set to None to enable auto-scaling
# Only used if ALPHA_VALUE is None. 0.95 = Keep 95% of triangles (pretty loose).
ALPHA_QUANTILE = 0.9

# LLM Labeling Parameters
MAX_LABELING_WORKERS = 5

TEST_CLUSTERS = False  # Set to True to only test clustering and exit
TEST_HULLS = False    # Set to True to preview hulls on matplotlib and exit
ADD_POINTS = False    # Set to True to add individual points to the map
# Set to True to automatically open the generated map in a web browser
AUTO_OPEN_BROWSER = True


def main():
    print("--- Starting UnPointMaps ---")
    
    # Configure shared component caches
    set_cache_enabled(USE_TEXT_CLEANING_CACHE)

    print("Loading and filtering dataset...")
    
    # Init DB Manager for data cache
    db_manager = DatabaseManager(DB_PATH)
    
    # Load Data (with persistent Cache)
    cached_df = db_manager.get_processed_data_cache(DATASET_FILE) if USE_DATA_CACHE else None
    if cached_df is not None:
        print(f"Loaded processed dataset from cache ({len(cached_df)} records)")
        df = cached_df
    else:
        print("Cache miss or file changed, processing dataset from scratch (this may take a while)...")
        df = convert_to_dict_filtered()
        if USE_DATA_CACHE:
            db_manager.save_processed_data_cache(DATASET_FILE, df)
        print(f"Dataset processed: {len(df)} records.")

    if LIMIT != -1:
        print(f"Limiting dataset to first {LIMIT} points for performance...")
        df = df.head(LIMIT)

    # === CLI INTERFACE ===
    # Invoke menu to filter dataset
    cli = CLIMenu()
    df = cli.run(df)

    if len(df) == 0:
        print("No data left after filtering. Exiting.")
        sys.exit(0)
    # =====================

    print("Starting iterative clustering with HDBSCAN...")
    print(f"  min_cluster_size={MIN_CLUSTER_SIZE}")
    print(f"  cluster_selection_epsilon={CLUSTER_SELECTION_EPSILON}")
    print(f"  max_std_dev={MAX_STD_DEV}")
    if USE_PARALLEL_CLUSTERING:
        print(
            f"  Parallel execution: Enabled (Workers: {multiprocessing.cpu_count()})")
    print()

    # Prepare Parameters
    clustering_params = {
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "cluster_selection_epsilon": CLUSTER_SELECTION_EPSILON,
        "max_std_dev": MAX_STD_DEV,
        "limit": LIMIT
    }
    # Simple signature for dataset
    dataset_signature = DATASET_FILE

    cached_run_id = db_manager.get_cached_run(
        clustering_params, dataset_signature) if USE_CLUSTERING_CACHE else None
    clustered_points = []
    used_k = 0
    llm_labels = {}
    labels = None

    if cached_run_id:
        print(
            f"Found cached run (ID: {cached_run_id}). Loading data from database...")
        clustered_points, llm_labels = db_manager.load_run_data(cached_run_id)
        used_k = len(clustered_points)
        print(f"Loaded {used_k} clusters from cache.")
        # Mock labels to avoid errors
        import numpy as np
        labels = np.full(len(df), -1)
    else:
        if not HAS_HDBSCAN:
            print("\n[!] Error: 'hdbscan' library is required for HDBSCAN clustering.")
            print("    Please install it with: pip install hdbscan")
            sys.exit(1)

        if USE_PARALLEL_CLUSTERING:
            clustered_points, used_k, labels = parallel_hdbscan_clustering_iterative(
                df,
                min_cluster_size=MIN_CLUSTER_SIZE,
                cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
                max_std_dev=MAX_STD_DEV
            )
        else:
            clustered_points, used_k, labels = hdbscan_clustering_iterative(
                df,
                min_cluster_size=MIN_CLUSTER_SIZE,
                cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
                max_std_dev=MAX_STD_DEV
            )
        print(f"Clustering complete. Found {used_k} clusters.")

    # LLM Labeling Integration
    if not cached_run_id:  # Only run if not cached
        llm_labels = {}
        if HAS_OPENAI:
            try:
                print("\n--- Starting LLM Labeling ---")
                # Initialize ConfigManager and Service
                # Note: Ensure .env file exists with OPENAI_API_KEY
                config_manager = ConfigManager()
                labeling_service = LLMLabelingService(config_manager)

                print(
                    f"Labelling {used_k} clusters using {config_manager.get_model()}...")

                clusters_to_process = []
                print("Preparing cluster metadata...")

                for i in range(used_k):
                    # Extract cluster points from original dataframe
                    # labels numpy array matches df rows
                    if labels is None:  # Should not happen if not cached
                        continue

                    cluster_mask = (labels == i)
                    cluster_df = df[cluster_mask]

                    if cluster_df.empty:
                        continue

                    # Prepare text metadata from title and tags
                    # Limit to top 50 points to fit in context window and save
                    # tokens
                    text_metadata = []
                    for _, row in cluster_df.head(50).iterrows():
                        parts = []
                        if pd.notna(
                                row.get('title')) and str(
                                row['title']).strip():
                            parts.append(str(row['title']))
                        if pd.notna(row.get('tags')) and str(row['tags']).strip():
                            # Tags are often space separated in Flickr
                            parts.append(str(row['tags']))

                        if parts:
                            text_metadata.append(" ".join(parts))

                    # Clean text before sending to LLM
                    text_metadata = clean_text_list(text_metadata)

                    # Fallback if no text data
                    if not text_metadata:
                        text_metadata = ["No description available"]

                    # Construct metadata object
                    cluster_meta = {
                        "cluster_id": f"cluster_{i}",
                        "image_ids": [str(x) for x in cluster_df.index[:10]],
                        "text_metadata": text_metadata,  # Service handles joining
                        "cluster_size": len(cluster_df)
                    }
                    clusters_to_process.append(cluster_meta)

                # Process in batches
                # Use a reasonable number of workers (e.g., 5 or 10) to allow concurrent processing
                # while the rate limiter imposes the speed limit.
                results = labeling_service.process_batch(
                    clusters_to_process, MAX_LABELING_WORKERS)
                llm_labels.update(results)

                # Fill in missing labels
                for i in range(used_k):
                    if i not in llm_labels:
                        llm_labels[i] = f"Cluster {i}"

                # Save to Cache
                if USE_CLUSTERING_CACHE:
                    print("Saving results to cache...")
                    db_manager.save_run(
                        clustering_params,
                        dataset_signature,
                        clustered_points,
                        llm_labels)

            except Exception as e:
                print(f"LLM Labeling skipped or failed: {e}")
                # Prepare default labels if LLM fails
                for i in range(used_k):
                    llm_labels[i] = f"Cluster {i}"
        else:
            print("\nWarning: 'openai' library not found. Skipping LLM Labeling. Default labels will be used.")
            # Prepare default labels if LLM fails
            for i in range(used_k):
                llm_labels[i] = f"Cluster {i}"

    if TEST_CLUSTERS:
        if not HAS_MATPLOTLIB:
            print("\nError: 'matplotlib' is required for TEST_CLUSTERS visualisation.")
            sys.exit(1)

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
        if not HAS_MATPLOTLIB:
            print("\nError: 'matplotlib' is required for TEST_HULLS visualisation.")
            sys.exit(1)

        if cached_run_id:
            print(
                "Warning: Skipping TEST_HULLS because data was loaded from cache (labels array incomplete).")
        else:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon
            import numpy as np

            print("Previewing hulls comparison on Matplotlib...")
            alpha_values = [None, 1 / 1000, 5 / 1000, 1 / 100, 5 / 100]

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
            colors = [
                'blue',
                'green',
                'red',
                'cyan',
                'magenta',
                'orange',
                'purple',
                'brown',
                'pink',
                'olive']

            for idx, alpha in enumerate(alpha_values):
                ax = axes[idx]
                print(f"  Computing hulls for alpha={alpha}...")
                hulls = compute_cluster_hulls(
                    clustered_points, alpha=alpha, auto_alpha_quantile=0.9)

                # Plot noise points in background
                ax.scatter(noise_points['longitude'], noise_points['latitude'],
                           c='lightgray', s=5, alpha=0.3, label='Noise')

                for i, hull in enumerate(hulls):
                    # Plot cluster points
                    cluster = clustered_points[i]
                    cluster_array = np.array(cluster)
                    clat, clon = cluster_array[:, 0], cluster_array[:, 1]

                    color = colors[i % len(colors)]
                    ax.scatter(
                        clon,
                        clat,
                        c=color,
                        s=15,
                        alpha=0.9,
                        edgecolors='none')

                    # Plot hull
                    for poly_points in hull:
                        hull_array = np.array(poly_points)
                        if len(hull_array) >= 3:
                            lon_lat_hull = hull_array[:, [1, 0]]
                            poly = Polygon(
                                lon_lat_hull.tolist(),
                                closed=True,
                                fill=True,
                                alpha=0.3,
                                facecolor=color,
                                edgecolor='black',
                                linewidth=1)
                            ax.add_patch(poly)
                        elif len(hull_array) > 0:
                            ax.plot(hull_array[:, 1], hull_array[:, 0],
                                    color=color, linestyle='-', alpha=0.5)

                ax.set_title(
                    f"Alpha = {alpha}",
                    fontsize=14,
                    fontweight='bold')
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.grid(True, alpha=0.2)

            # Hide unused subplots
            for j in range(n_plots, len(axes)):
                axes[j].axis('off')

            plt.suptitle(
                f"Concave Hull Comparison ({used_k} clusters)",
                fontsize=18,
                y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

            sys.exit(0)

    if HAS_FOLIUM:
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
        hulls = compute_cluster_hulls(
            clustered_points,
            alpha=ALPHA_VALUE,
            auto_alpha_quantile=ALPHA_QUANTILE)

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
    else:
        print("\n[!] Warning: 'folium' library not found. Skipping interactive map generation.")
        print("    Standard clustering and hulls will still be computed and cached.")

    print("--- Done ---")

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


if __name__ == '__main__':
    # On Windows, the multiprocessing module must typically be able
    # to import the main module without side effects.
    multiprocessing.freeze_support()
    main()
