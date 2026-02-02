import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import clustering and hull logic
try:
    from src.clustering.hdbscan_clustering import hdbscan_clustering_iterative
    from src.utils.hull_logic import get_alpha_shape
except ImportError:
    print("Could not import project modules. Ensure you are running this from the project root.")
    sys.exit(1)

def main():
    print("Loading data...")
    dataset_file = "flickr_data2.csv"
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} not found.")
        return

    df = pd.read_csv(dataset_file)
    # Using a subset for speed if large, but clustering needs some context
    # Let's use the first 5000 points or so
    subset_df = df.head(5000)

    print(f"Clustering {len(subset_df)} points with 'Very Tight' parameters...")
    # Use the parameters we just set as defaults
    clusters_data, n_clusters, labels = hdbscan_clustering_iterative(
        subset_df,
        min_cluster_size=3,
        cluster_selection_epsilon=0.0005,
        max_std_dev=30.0
    )

    if n_clusters == 0:
        print("No clusters found. Try different clustering parameters.")
        return

    print(f"Found {n_clusters} clusters.")

    # Find the largest cluster (usually the most interesting for hulls)
    cluster_sizes = [len(c[0]) for c in clusters_data]
    largest_idx = np.argmax(cluster_sizes)
    target_cluster_points = clusters_data[largest_idx][0] # List of [lat, lon]
    points_array = np.array(target_cluster_points)

    print(f"Targeting cluster #{largest_idx} with {len(points_array)} points.")

    # Define configurations to test: (Title, alpha, quantile)
    configs = [
        ("Auto (q=0.5)", None, 0.5),
        ("Auto (q=0.8)", None, 0.8),
        ("Auto (q=0.9) - Default", None, 0.9),
        ("Auto (q=0.95)", None, 0.95),
        ("Auto (q=0.99)", None, 0.99),
        ("Fixed Alpha (0.005)", 0.005, 0.95),
    ]

    # Setup subplots
    n = len(configs)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = axes.flatten()

    print(f"Generating {n} plots...")

    for idx, (title, alpha, quantile) in enumerate(configs):
        ax = axes[idx]
        
        try:
            # We need to swap [lat, lon] to [lon, lat] for plotting if we want standard x,y
            # Actually matplotlib doesn't care if we are consistent. 
            # Let's use lon as X (index 1) and lat as Y (index 0)
            x_pts = points_array[:, 1]
            y_pts = points_array[:, 0]

            hull_polygons = get_alpha_shape(
                target_cluster_points, 
                alpha=alpha, 
                auto_alpha_quantile=quantile
            )

            # Plot points
            ax.scatter(x_pts, y_pts, c='blue', alpha=0.5, s=5, label='Points')

            # Plot hull(s)
            found_hull = False
            for poly_coords in hull_polygons:
                # poly_coords is list of [lat, lon]
                poly = np.array(poly_coords)
                if len(poly) > 0:
                    found_hull = True
                    # Swap to [lon, lat] for plot
                    px = poly[:, 1]
                    py = poly[:, 0]
                    
                    # Close the loop
                    if not np.array_equal(poly[0], poly[-1]):
                        px = np.append(px, px[0])
                        py = np.append(py, py[0])

                    ax.plot(px, py, 'r-', linewidth=1.5)
                    ax.fill(px, py, 'r', alpha=0.1)

            ax.set_title(f"{title}\n{len(hull_polygons)} polygon(s)")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Equal aspect to avoid distortion of geographic coordinates
            ax.set_aspect('equal')

        except Exception as e:
            ax.set_title(f"{title} - Error")
            print(f"Error computing hull for {title}: {e}")

    # Hide empty subplots
    for i in range(len(configs), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"Hull Parameter Comparison (Real Data Cluster #{largest_idx})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = 'hull_real_data_comparison.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved visualization to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
