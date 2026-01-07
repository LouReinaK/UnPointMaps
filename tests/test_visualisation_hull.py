import unittest
import sys
import os

# Add parent directory to path to import modules from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from map_visualisation import Visualisation
import folium
import pandas as pd

class TestConvexHull(unittest.TestCase):
    def setUp(self):
        self.vis = Visualisation()

    def test_convex_hull_simple_square(self):
        # Square with a point in the middle
        points = [
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [0.5, 0.5]
        ]
        hull = self.vis.get_convex_hull(points)
        
        expected = {(0, 0), (1, 0), (1, 1), (0, 1)}
        result_set = set(tuple(p) for p in hull)
        
        self.assertEqual(result_set, expected)
        self.assertEqual(len(hull), 4)

    def test_convex_hull_triangle(self):
        points = [[0, 0], [1, 0], [0, 1]]
        hull = self.vis.get_convex_hull(points)
        expected = {(0, 0), (1, 0), (0, 1)}
        self.assertEqual(set(tuple(p) for p in hull), expected)
        self.assertEqual(len(hull), 3)

    def test_convex_hull_collinear(self):
        # Points on a line
        points = [[0, 0], [1, 1], [2, 2]]
        hull = self.vis.get_convex_hull(points)
        
        expected = {(0, 0), (2, 2)}
        self.assertEqual(set(tuple(p) for p in hull), expected)

    def test_empty_and_single_point(self):
        # Empty
        self.assertEqual(self.vis.get_convex_hull([]), [])
        
        # Single point
        self.assertEqual(self.vis.get_convex_hull([[1, 1]]), [[1, 1]])

        # Two points
        points = [[0, 0], [1, 1]]
        result = self.vis.get_convex_hull(points)
        self.assertEqual(len(result), 2)
        self.assertIn([0, 0], result)
        self.assertIn([1, 1], result)

    def test_duplicate_points(self):
        points = [[1, 1], [1, 1], [1, 1]]
        # Should be treated as single point
        self.assertEqual(self.vis.get_convex_hull(points), [[1, 1]])

    def test_draw_cluster_hull_with_points(self):
        # Test method that takes dicts (simulating pandas rows)
        pts = [
            {'latitude': 0, 'longitude': 0, 'title': 'P1'},
            {'latitude': 1, 'longitude': 1, 'title': 'P2'},
            {'latitude': 0, 'longitude': 1, 'title': 'P3'},
            {'latitude': 1, 'longitude': 0, 'title': 'P4'},
            {'latitude': 0.5, 'longitude': 0.5, 'title': 'P5'}
        ]
        
        initial_count = len(self.vis.clusters)
        self.vis.draw_cluster_hull(pts)
        self.assertEqual(len(self.vis.clusters), initial_count + 1)
        self.assertIsInstance(self.vis.clusters[-1], folium.Polygon)

if __name__ == '__main__':
    unittest.main()
