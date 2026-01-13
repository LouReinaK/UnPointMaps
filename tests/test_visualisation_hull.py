
from src.visualization.map_visualisation import Visualisation
from src.utils.hull_logic import get_alpha_shape, compute_cluster_hulls
import pandas as pd

def test_hull():
    vis = Visualisation()
    
    # Test 1: Simple square with center point
    points = [
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [0.5, 0.5] # In the middle
    ]
    # Test with different alpha values
    print("Testing alpha shape with alpha=0.5 (tighter fit):")
    hull_tight = get_alpha_shape(points, alpha=0.5)
    print("Alpha shape (alpha=0.5) components:", len(hull_tight))
    print("First component:", hull_tight[0] if hull_tight else "None")
    
    print("\nTesting alpha shape with alpha=2.0 (looser fit):")
    hull_loose = get_alpha_shape(points, alpha=2.0)
    print("Alpha shape (alpha=2.0) components:", len(hull_loose))

    
    # Test 2: Random points handled via compute_cluster_hulls
    pts = [(0, 0), (2, 2), (0, 2), (2, 0), (1, 1)]
    
    try:
        hulls = compute_cluster_hulls([pts], alpha=0.5)
        for hull in hulls:
            vis.draw_cluster(hull)
        print("\ncompute_cluster_hulls and draw_cluster ran successfully")
    except Exception as e:
        print("compute_cluster_hulls or draw_cluster failed:", e)
        import traceback
        traceback.print_exc()
    
    # Test 3: More complex shape to show concave hull capability
    complex_points = [
        [0, 0], [1, 0], [2, 0],
        [0, 1], [2, 1],
        [0, 2], [1, 2], [2, 2]
    ]
    print("\nTesting complex shape (C-shaped cluster):")
    hull_complex = get_alpha_shape(complex_points, alpha=0.8)
    print("Alpha shape for C-shape:", hull_complex)

if __name__ == "__main__":
    test_hull()
