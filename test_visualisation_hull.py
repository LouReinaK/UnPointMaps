
from map_visualisation import Visualisation

def test_hull():
    vis = Visualisation()
    
    # Test 1: Simple square
    points = [
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [0.5, 0.5] # In the middle
    ]
    # Expected hull: (0,0)->(1,0)->(1,1)->(0,1) or similar order
    hull = vis.get_convex_hull(points)
    print("Hull of square + center:", hull)
    
    # Test 2: Random points handled via draw_cluster_hull
    pts = [(0, 0), (2, 2), (0, 2), (2, 0), (1, 1)]
    
    try:
        vis.draw_cluster_hull(pts)
        print("draw_cluster_hull ran successfully")
    except Exception as e:
        print("draw_cluster_hull failed:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hull()
