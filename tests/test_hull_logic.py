from unittest.mock import patch
import numpy as np
from src.utils.hull_logic import compute_cluster_hulls, get_alpha_shape


class TestHullLogic:
    @patch('src.utils.hull_logic._DB_MANAGER')
    def test_compute_cluster_hulls_triangle(self, mock_db):
        # 3 points forming a triangle
        points = [[0, 0], [1, 0], [0, 1]]
        # Mock cache miss
        mock_db.get_cached_hull.return_value = None

        hulls = compute_cluster_hulls([points], alpha=0)
        # With alpha=0, it should be convex hull (triangle)

        assert len(hulls) == 1
        # hulls[0] is a list of polygons (MultiPolygon support), so access the
        # first polygon
        assert len(hulls[0][0]) >= 3
        # Should contain original points
        assert [0, 0] in hulls[0][0]

    @patch('src.utils.hull_logic._DB_MANAGER')
    def test_compute_cluster_hulls_cached(self, mock_db):
        points = [[0, 0], [1, 0], [0, 1]]
        cached_result = [[[0, 0], [1, 0], [0, 1], [0, 0]]]  # Closed loop

        mock_db.get_cached_hull.return_value = cached_result

        hulls = compute_cluster_hulls([points])
        assert hulls[0] == cached_result

    @patch('src.utils.hull_logic._DB_MANAGER')
    def test_get_alpha_shape_basic(self, mock_db):
        # Mock cache miss
        mock_db.get_cached_hull.return_value = None

        # Grid of points
        x, y = np.meshgrid(range(5), range(5))
        points = np.column_stack((x.flatten(), y.flatten()))

        # Test alpha shape computation (simple execution check)
        hull = get_alpha_shape(points, alpha=0.5)
        assert hull is not None
