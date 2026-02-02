import numpy as np
from typing import List, Tuple, Dict, Optional
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

def compute_polynomial_tram_line(clusters: List[Dict], degree: int = 5, n_points: int = 100, 
                                  max_length: Optional[float] = None) -> List[List[float]]:
    """
    Compute a tram line path using polynomial regression that minimizes distance to cluster centroids.
    
    The function extracts all points from clusters, finds the principal direction,
    and fits a polynomial curve through the data to create a smooth tram line path.
    
    Args:
        clusters: List of cluster dicts with 'points' (list of [lat, lon] or list of polygons)
        degree: Degree of the polynomial (1=linear, 2=quadratic, 3=cubic, etc.)
        n_points: Number of points to generate along the polynomial curve
        max_length: Optional maximum length constraint (in degrees)
    
    Returns:
        path: List of [lat, lon] points defining the polynomial tram line
    """
    if not clusters:
        return []
    
    # Extract all points from all clusters
    all_points = []
    all_weights = []
    
    for cluster in clusters:
        points_list = cluster['points']
        if not points_list or not isinstance(points_list, list):
            continue
        
        # Flatten if it's a list of polygons
        cluster_points = []
        for item in points_list:
            if isinstance(item, list) and len(item) > 0:
                if isinstance(item[0], (list, tuple)) and len(item[0]) >= 2:
                    # It's a polygon: list of [lat, lon]
                    cluster_points.extend(item)
                elif isinstance(item[0], (int, float)):
                    # It's a point: [lat, lon]
                    cluster_points.append(item)
        
        # Filter valid points
        for point in cluster_points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    lat, lon = float(point[0]), float(point[1])
                    all_points.append([lat, lon])
                    # Weight each point by cluster size
                    all_weights.append(cluster.get('size', 1))
                except (ValueError, TypeError):
                    continue
    
    if len(all_points) == 0:
        return []
    
    points = np.array(all_points)
    weights = np.array(all_weights)
    print(f"[DEBUG] Points shape: {points.shape}, weights shape: {weights.shape}")
    
    # If only one point, return a small segment
    if len(points) == 1:
        center = points[0]
        offset = 0.001
        return [
            [center[0] - offset, center[1] - offset],
            center.tolist(),
            [center[0] + offset, center[1] + offset]
        ]
    
    # Compute weighted center
    weighted_center = np.average(points, axis=0, weights=weights)
    print(f"[DEBUG] Weighted center: {weighted_center}")
    
    # Center the points
    centered_points = points - weighted_center
    
    # Find principal direction using weighted PCA
    cov_matrix = np.cov(centered_points.T, aweights=weights)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    principal_direction = eigenvectors[:, -1]  # eigenvector of largest eigenvalue
    
    # Project all points onto the principal direction to get parameter t
    # t represents the position along the principal axis
    t_values = np.dot(centered_points, principal_direction)
    
    # Sort points by t for proper polynomial fitting
    sort_indices = np.argsort(t_values)
    t_sorted = t_values[sort_indices]
    points_sorted = points[sort_indices]
    weights_sorted = weights[sort_indices]
    
    # Fit polynomial: lat = poly(t) and lon = poly(t)
    # Using weighted polynomial regression
    lat_coeffs = np.polyfit(t_sorted, points_sorted[:, 0], degree, w=np.sqrt(weights_sorted))
    lon_coeffs = np.polyfit(t_sorted, points_sorted[:, 1], degree, w=np.sqrt(weights_sorted))
    
    # Create polynomial functions
    lat_poly = np.poly1d(lat_coeffs)
    lon_poly = np.poly1d(lon_coeffs)
    
    # Generate points along the polynomial curve
    t_min, t_max = t_sorted[0], t_sorted[-1]
    
    # Apply max_length constraint if specified
    if max_length is not None:
        # Estimate current length and scale if necessary
        t_range = t_max - t_min
        estimated_length = t_range * np.linalg.norm(principal_direction)
        if estimated_length > max_length:
            scale_factor = max_length / estimated_length
            t_center = (t_min + t_max) / 2
            t_min = t_center - (t_center - t_min) * scale_factor
            t_max = t_center + (t_max - t_center) * scale_factor
    
    t_eval = np.linspace(t_min, t_max, n_points)
    
    # Evaluate polynomials
    lat_values = lat_poly(t_eval)
    lon_values = lon_poly(t_eval)
    
    # Combine into path
    path = [[float(lat), float(lon)] for lat, lon in zip(lat_values, lon_values)]
    
    return path