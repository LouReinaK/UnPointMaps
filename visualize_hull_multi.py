import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the hull logic
try:
    from src.utils.hull_logic import get_alpha_shape
except ImportError:
    print("Could not import hull_logic. Ensure you are running this from the project root.")
    sys.exit(1)

def generate_cluster(center, n_points, spread):
    return np.random.normal(center, spread, (n_points, 2))

def main():
    print("Generating random points...")
    # 1. Generate points
    np.random.seed(42) # For reproducibility
    # Cluster 1 (C shape part 1)
    c1 = generate_cluster([0, 0], 30, 0.3)
    # Cluster 2 (C shape part 2)
    c2 = generate_cluster([0, 2], 30, 0.3)
    # Cluster 3 (C shape part 3)
    c3 = generate_cluster([-1, 1], 30, 0.3)
    
    points = np.vstack([c1, c2, c3])
    points_list = points.tolist()

    # Define alphas to test
    # None represents "Auto"
    alphas = [0.2, 0.5, 0.8, 1.2, 2.0, None]
    
    # Setup subplots
    n = len(alphas)
    cols = 3
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()
    
    print(f"Generating {n} plots on a {rows}x{cols} grid...")

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        
        # Title
        display_alpha = f"alpha={alpha}" if alpha is not None else "alpha=Auto"
        # print(f"Computing alpha shape with {display_alpha}...")
        
        try:
            hull_polygons = get_alpha_shape(points_list, alpha=alpha, auto_alpha_quantile=0.9)
            
            # Plot points
            ax.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.5, s=10)
            
            # Plot hull(s)
            for i, poly_coords in enumerate(hull_polygons):
                poly = np.array(poly_coords)
                if len(poly) > 0:
                    # Close the loop if not closed
                    if not np.array_equal(poly[0], poly[-1]):
                        poly = np.vstack([poly, poly[0]])
                        
                    ax.plot(poly[:, 0], poly[:, 1], 'r-', linewidth=2)
                    ax.fill(poly[:, 0], poly[:, 1], 'r', alpha=0.2)
            
            ax.set_title(display_alpha)
            ax.grid(True)
            
        except Exception as e:
            ax.set_title(f"{display_alpha} - Error")
            print(f"Error computing hull for {display_alpha}: {e}")

    # Hide empty subplots
    for i in range(len(alphas), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Alpha Value Comparison", fontsize=16)
    plt.tight_layout()
    output_file = 'hull_example_multi.png'
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    main()
