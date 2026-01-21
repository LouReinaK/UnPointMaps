import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist
from scipy import interpolate

def compute_weighted_centroids(clusters: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute centroids and weights for clusters.
    
    Args:
        clusters: List of cluster dicts with 'points' (list of [lat, lon] or list of polygons)
    
    Returns:
        centroids: (n_clusters, 2) array of [lat, lon]
        weights: (n_clusters,) array of weights
    """
    centroids = []
    weights = []
    
    for cluster in clusters:
        points_list = cluster['points']
        if not points_list or not isinstance(points_list, list):
            continue
        
        # Flatten if it's a list of polygons (list of lists of points)
        all_points = []
        for item in points_list:
            if isinstance(item, list) and len(item) > 0:
                if isinstance(item[0], (list, tuple)) and len(item[0]) >= 2:
                    # It's a polygon: list of [lat, lon]
                    all_points.extend(item)
                elif isinstance(item[0], (int, float)):
                    # It's a point: [lat, lon]
                    all_points.append(item)
        
        # Filter valid points (should be [lat, lon] pairs)
        valid_points = []
        for point in all_points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    lat, lon = float(point[0]), float(point[1])
                    valid_points.append([lat, lon])
                except (ValueError, TypeError):
                    continue
        
        if len(valid_points) == 0:
            continue
            
        points = np.array(valid_points)
        if len(points) == 0:
            continue
            
        centroid = np.mean(points, axis=0)
        centroids.append(centroid)
        weights.append(cluster['size'])
    
    return np.array(centroids), np.array(weights)

def compute_tram_line(clusters: List[Dict], max_length: float) -> List[List[float]]:
    """
    Compute a tram line path of maximum length that minimizes weighted distance to clusters.
    
    Uses a heuristic: find the principal direction of weighted centroids, 
    then create a smooth spline path along that direction of length max_length,
    centered on the weighted center of mass, with curvature to better fit clusters.
    
    Args:
        clusters: List of cluster dicts
        max_length: Maximum length in degrees (approximate, since lat/lon)
    
    Returns:
        path: List of [lat, lon] points defining the tram line spline
    """
    if not clusters:
        return []
    
    centroids, weights = compute_weighted_centroids(clusters)
    
    if len(centroids) == 0:
        return []
    
    # Compute weighted center of mass
    total_weight = np.sum(weights)
    center = np.average(centroids, axis=0, weights=weights)
    
    # If only one cluster, create a short curved line around it
    if len(centroids) == 1:
        # Create a small curved segment
        direction = np.array([0.001, 0.001])  # arbitrary small direction
        perp_direction = np.array([-direction[1], direction[0]])  # perpendicular
        half_length = min(max_length / 2, 0.005)
        
        # Create control points for a slight curve
        control_points = np.array([
            center - direction * half_length,
            center - direction * half_length * 0.5 + perp_direction * half_length * 0.2,
            center + direction * half_length * 0.5 + perp_direction * half_length * 0.2,
            center + direction * half_length
        ])
        
        # Create spline
        t = np.linspace(0, 1, 20)
        cs = interpolate.CubicSpline(np.linspace(0, 1, len(control_points)), control_points, axis=0)
        path = cs(t).tolist()
        return path
    
    # Compute covariance matrix weighted
    centered = centroids - center
    cov = np.cov(centered.T, aweights=weights)
    
    # Find principal direction (eigenvector of largest eigenvalue)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    principal_direction = eigenvecs[:, -1]  # eigenvector for largest eigenvalue
    
    # Normalize
    principal_direction = principal_direction / np.linalg.norm(principal_direction)
    perp_direction = np.array([-principal_direction[1], principal_direction[0]])  # perpendicular
    
    # Create control points for spline with curvature
    half_length = max_length / 2
    n_control = 5  # number of control points
    
    control_points = []
    for i in range(n_control):
        t = (i / (n_control - 1) - 0.5) * 2  # from -1 to 1
        base_point = center + principal_direction * t * half_length
        
        # Add curvature based on cluster distribution
        # Calculate how "off-center" the clusters are in the perpendicular direction
        perp_offset = np.dot(centered, perp_direction)  # projection onto perpendicular
        weighted_offset = np.average(perp_offset, weights=weights)
        
        # Add some curvature towards the clusters
        curvature_factor = 0.1 * weighted_offset * (1 - abs(t))  # more curvature in the middle
        curved_point = base_point + perp_direction * curvature_factor
        
        control_points.append(curved_point)
    
    control_points = np.array(control_points)
    
    # Create smooth spline through control points
    t_param = np.linspace(0, 1, len(control_points))
    t_eval = np.linspace(0, 1, 50)  # More points for smoother curve
    
    try:
        cs = interpolate.CubicSpline(t_param, control_points, axis=0)
        path = cs(t_eval).tolist()
    except Exception:
        # Fallback to linear interpolation if spline fails
        path = []
        for t in t_eval:
            idx = int(t * (len(control_points) - 1))
            frac = t * (len(control_points) - 1) - idx
            if idx < len(control_points) - 1:
                point = control_points[idx] + frac * (control_points[idx + 1] - control_points[idx])
            else:
                point = control_points[-1]
            path.append(point.tolist())
    
    return path

def evaluate_path_cost(path: List[List[float]], centroids: np.ndarray, weights: np.ndarray) -> float:
    """
    Evaluate the cost of a path: sum of weighted distances from centroids to path.
    
    Args:
        path: List of [lat, lon] points
        centroids: (n_clusters, 2) array
        weights: (n_clusters,) array
    
    Returns:
        cost: Total weighted distance
    """
    if not path or len(centroids) == 0:
        return float('inf')
    
    path_array = np.array(path)
    
    # For each centroid, find min distance to any point on path
    distances = cdist(centroids, path_array).min(axis=1)
    
    cost = np.sum(weights * distances)
    return cost