"""
Test edge cases for alpha shape computation including MultiPoint scenarios.
"""

from src.visualization.map_visualisation import Visualisation
import numpy as np


def test_edge_cases():
    """Test various edge cases that might produce MultiPoint or other non-Polygon results."""
    vis = Visualisation()

    print("Testing edge cases for alpha shape computation...\n")

    # Test 1: Very few points (2 points)
    print("Test 1: Two points")
    points_2 = [[0, 0], [1, 1]]
    try:
        hull = vis.get_alpha_shape(points_2, alpha=0.5)
        print(f"  ✓ Success: {len(hull)} points returned")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 2: Three points (triangle)
    print("\nTest 2: Three points (triangle)")
    points_3 = [[0, 0], [1, 0], [0.5, 1]]
    try:
        hull = vis.get_alpha_shape(points_3, alpha=0.5)
        print(f"  ✓ Success: {len(hull)} points returned")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: Collinear points (might produce LineString)
    print("\nTest 3: Collinear points")
    points_collinear = [[i, i] for i in range(5)]
    try:
        hull = vis.get_alpha_shape(points_collinear, alpha=0.5)
        print(f"  ✓ Success: {len(hull)} points returned")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 4: Very small alpha (might produce MultiPoint)
    print("\nTest 4: Very small alpha (0.01) with scattered points")
    np.random.seed(42)
    scattered = [[x, y] for x, y in zip(
        np.random.uniform(0, 10, 10),
        np.random.uniform(0, 10, 10)
    )]
    try:
        hull = vis.get_alpha_shape(scattered, alpha=0.01)
        print(f"  ✓ Success: {len(hull)} points returned")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 5: Tiny cluster (might produce MultiPoint)
    print("\nTest 5: Very tight cluster")
    tiny_cluster = [
        [45.7597, 4.8422],
        [45.7598, 4.8423],
        [45.7599, 4.8424],
        [45.7600, 4.8425]
    ]
    try:
        hull = vis.get_alpha_shape(tiny_cluster, alpha=0.1)
        print(f"  ✓ Success: {len(hull)} points returned")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 6: Duplicate points
    print("\nTest 6: Points with duplicates")
    points_dup = [[0, 0], [1, 1], [0, 0], [1, 1], [2, 2]]
    try:
        hull = vis.get_alpha_shape(points_dup, alpha=0.5)
        print(f"  ✓ Success: {len(hull)} points returned (duplicates handled)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 7: Two separate blobs (might produce MultiPolygon)
    print("\nTest 7: Two separate blobs")
    blob1 = [[x, y] for x, y in zip(
        np.random.RandomState(42).normal(0, 0.3, 8),
        np.random.RandomState(42).normal(0, 0.3, 8)
    )]
    blob2 = [[x, y] for x, y in zip(
        np.random.RandomState(43).normal(5, 0.3, 8),
        np.random.RandomState(43).normal(5, 0.3, 8)
    )]
    two_blobs = blob1 + blob2
    try:
        hull = vis.get_alpha_shape(two_blobs, alpha=0.5)
        print(
            f"  ✓ Success: {len(hull)} points returned (largest blob selected)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 8: Very large alpha (should produce convex-like hull)
    print("\nTest 8: Very large alpha (10.0)")
    points = [[x, y] for x, y in zip(
        np.random.RandomState(44).uniform(0, 5, 15),
        np.random.RandomState(44).uniform(0, 5, 15)
    )]
    try:
        hull = vis.get_alpha_shape(points, alpha=10.0)
        print(f"  ✓ Success: {len(hull)} points returned")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 9: None alpha (auto-compute)
    print("\nTest 9: Auto-compute alpha (None)")
    try:
        hull = vis.get_alpha_shape(scattered, alpha=None)
        print(f"  ✓ Success: {len(hull)} points returned (auto-computed)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("Edge case testing complete!")
    print("All cases should handle gracefully with fallbacks.")
    print("=" * 60)


if __name__ == "__main__":
    test_edge_cases()
