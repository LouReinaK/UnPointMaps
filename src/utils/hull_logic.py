import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point, LineString, GeometryCollection
from shapely.ops import unary_union
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def get_alpha_shape(points, alpha=None, auto_alpha_quantile=0.95):
    """
    Computes the alpha shape (concave hull) of a set of 2D points.
    Input: list of [lat, lon] or (lat, lon)
           alpha: Threshold for circumradius of Delaunay triangles.
                  If None, auto-computed based on auto_alpha_quantile.
           auto_alpha_quantile: Percentile (0 to 1) to use if alpha is None. 
                  Higher = looser (more convex), Lower = tighter (more concave).
    Output: list of [lat, lon] forming the alpha shape boundary
    """
    # Remove duplicates
    unique_points = list(set(tuple(p) for p in points))
    
    if len(unique_points) <= 2:
        return [[list(p) for p in unique_points]]
    
    if len(unique_points) == 3:
        # For 3 points, just return them as a triangle
        return [[list(p) for p in unique_points]]

    # Convert to numpy array
    points_array = np.array(unique_points)
    
    try:
        tri = Delaunay(points_array)
        triangles = points_array[tri.simplices]
        
        # Calculate side lengths
        a = np.sqrt(np.sum((triangles[:, 0] - triangles[:, 1])**2, axis=1))
        b = np.sqrt(np.sum((triangles[:, 1] - triangles[:, 2])**2, axis=1))
        c = np.sqrt(np.sum((triangles[:, 2] - triangles[:, 0])**2, axis=1))
        
        s = (a + b + c) / 2
        # Area using Heron's formula
        val = s * (s - a) * (s - b) * (s - c)
        val = np.maximum(val, 0)
        area = np.sqrt(val)
        
        # Filter singularity
        mask = area > 1e-10
        
        # Circumradius R = abc / (4 * area)
        circum_r = np.full(area.shape, np.inf)
        circum_r[mask] = (a[mask] * b[mask] * c[mask]) / (4.0 * area[mask])
        
        # Auto-compute alpha if None
        if alpha is None:
            # Use percentile of circumradii (keeps most triangles but removes very large ones)
            # Filter out infinite radii first (collinear points)
            valid_r = circum_r[np.isfinite(circum_r)]
            if len(valid_r) > 0:
                # Convert 0-1 range to 0-100 for numpy
                alpha = np.percentile(valid_r, auto_alpha_quantile * 100)
            else:
                alpha = 1.0 # arbitrary default
        
        keep_indices = circum_r < alpha

        # ---------------------------------------------------------
        # FORCE INCLUSION: Ensure every point is covered by at least one triangle
        # ---------------------------------------------------------
        # Check which points are currently covered by the kept triangles
        point_covered = np.zeros(len(points_array), dtype=bool)
        kept_simplices_indices = tri.simplices[keep_indices]
        
        if len(kept_simplices_indices) > 0:
            covered_indices_flat = np.unique(kept_simplices_indices)
            point_covered[covered_indices_flat] = True
            
        missing_indices = np.flatnonzero(~point_covered)
        
        if len(missing_indices) > 0:
            # Build a map of point_index -> list of simplex_indices
            vertex_to_simplices = [[] for _ in range(len(points_array))]
            for idx, simplex in enumerate(tri.simplices):
                for v in simplex:
                    vertex_to_simplices[v].append(idx)
            
            for p_idx in missing_indices:
                incident_simplices = vertex_to_simplices[p_idx]
                if not incident_simplices:
                    continue
                
                # Find the incident simplex with minimal circumradius
                # We filter only triangles that have finite radius (valid polygons)
                valid_incidents = [i for i in incident_simplices if np.isfinite(circum_r[i])]
                
                if valid_incidents:
                    best_simplex_idx = min(valid_incidents, key=lambda i: circum_r[i])
                    # Force keep this triangle
                    keep_indices[best_simplex_idx] = True

        # ---------------------------------------------------------
        # CONNECTIVITY: Ensure single component using Minimum Spanning Tree
        # ---------------------------------------------------------
        if np.any(keep_indices):
            num_tri = len(tri.simplices)
            rows, cols, data = [], [], []

            # Build dual graph of triangles
            for i in range(num_tri):
                # Optimization: only consider finite triangles as valid nodes in our graph
                # If a triangle is infinite (collinear), we effectively remove it
                if not np.isfinite(circum_r[i]):
                     continue
                     
                for neighbor in tri.neighbors[i]:
                    if neighbor == -1 or neighbor < i:
                        continue
                    if not np.isfinite(circum_r[neighbor]):
                         continue
                    
                    # Weight logic: "Cost" to traverse. 
                    # If triangle is already kept, cost is 0. Else cost is its radius.
                    # Edge weight = max cost of the two triangles.
                    # This ensures we prefer paths through kept regions (0 cost) 
                    # and minimize the "widest" gap we must bridge.
                    cost_i = 0 if keep_indices[i] else circum_r[i]
                    cost_n = 0 if keep_indices[neighbor] else circum_r[neighbor]
                    w = max(cost_i, cost_n)
                    
                    rows.append(i)
                    cols.append(neighbor)
                    data.append(w)
            
            if rows:
                # Compute MST of the triangle graph
                graph_matrix = csr_matrix((data, (rows, cols)), shape=(num_tri, num_tri))
                mst = minimum_spanning_tree(graph_matrix)
                
                # Convert MST to adjacency list
                mst_coo = mst.tocoo()
                mst_adj = [set() for _ in range(num_tri)]
                degrees = np.zeros(num_tri, dtype=int)
                
                for r, c in zip(mst_coo.row, mst_coo.col):
                    mst_adj[r].add(c)
                    mst_adj[c].add(r)
                    degrees[r] += 1
                    degrees[c] += 1
                
                # Pruning: Remove leaves that are NOT in original 'keep_indices'
                # This trims the MST back to the minimal structure connecting the original components
                leaves = [i for i in range(num_tri) if degrees[i] == 1 and not keep_indices[i]]
                queue = leaves[:]
                
                eliminated = np.zeros(num_tri, dtype=bool)
                idx = 0
                while idx < len(queue):
                    leaf = queue[idx]
                    idx += 1
                    eliminated[leaf] = True
                    
                    for neighbor in mst_adj[leaf]:
                        if not eliminated[neighbor]:
                            degrees[neighbor] -= 1
                            if degrees[neighbor] == 1 and not keep_indices[neighbor]:
                                queue.append(neighbor)
                
                # Update keep_indices: Anything connected in the pruned MST (that isn't eliminated)
                # Note: We must also safeguard against triangles that were completely disconnected in the graph (degree 0)
                # Degree 0 nodes in MST are either isolated components (which keep_indices accounts for)
                # or nodes that were eliminated.
                
                # Actually, simpler:
                # The final set is (Original Kept) UNION (Intermediate Nodes in Pruned MST).
                # Intermediate nodes are those that were NOT eliminated.
                # However, nodes that were not in the graph at all (infinite triangles) are not eliminated but 
                # also not reachable.
                # So we just say: if 'eliminated' is false AND it was part of the finite graph?
                # Actually, 'eliminated' defaults to False.
                # If a node was NEVER in the MST (isolated, keep_indices=False), it has degree 0.
                # It won't be in 'leaves' initially.
                # Wait. Isolated nodes with keep_indices=False are simply not interesting.
                # Isolated nodes with keep_indices=True must be kept.
                # Nodes in MST that are NOT eliminated are the bridges + original nodes.
                
                # Correct logic:
                # 1. Any node originally in keep_indices MUST be kept (force kept).
                # 2. Any node not eliminated by the pruning process should be kept IF can be reached?
                # Actually, standard pruning logic says "Remove only if leaf and bad".
                # An isolated "bad" node is a leaf (degree 0). It should be removed.
                # So we need to handle degree 0 "bad" nodes too.
                
                bad_nodes = [i for i in range(num_tri) if not keep_indices[i]]
                for i in bad_nodes:
                    if degrees[i] == 0:
                        eliminated[i] = True
                        
                keep_indices = keep_indices | (~eliminated)
                # Ensure we don't accidentally include infinite triangles if our logic was loose
                keep_indices = keep_indices & np.isfinite(circum_r)

        # If filtering removed everything, or resulted in empty set
        if not np.any(keep_indices):
            # Fallback to Convex Hull
            hull = ConvexHull(points_array)
            hull_points = points_array[hull.vertices]
            return [[[float(p[0]), float(p[1])] for p in hull_points]]
            
        kept_triangles = triangles[keep_indices]
        polys = [Polygon(t) for t in kept_triangles]
        
        # Union the triangles
        concave_hull = unary_union(polys)
        
        # Extract coordinates
        coords = []
        if isinstance(concave_hull, Polygon):
            coords = list(concave_hull.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            return [[[float(p[0]), float(p[1])] for p in coords]]
        elif isinstance(concave_hull, MultiPolygon):
             polys_coords = []
             for poly in concave_hull.geoms:
                 coords = list(poly.exterior.coords)
                 if len(coords) > 1 and coords[0] == coords[-1]:
                    coords = coords[:-1]
                 polys_coords.append([[float(p[0]), float(p[1])] for p in coords])
             return polys_coords
        elif isinstance(concave_hull, GeometryCollection):
             polys_coords = []
             for g in concave_hull.geoms:
                 if isinstance(g, Polygon):
                     coords = list(g.exterior.coords)
                     if len(coords) > 1 and coords[0] == coords[-1]:
                        coords = coords[:-1]
                     polys_coords.append([[float(p[0]), float(p[1])] for p in coords])
             if polys_coords:
                 return polys_coords

        # Ensure we return valid coordinates
        # Fallback
        hull = ConvexHull(points_array)
        hull_points = points_array[hull.vertices]
        return [[[float(p[0]), float(p[1])] for p in hull_points]]
        
    except Exception as e:
        print(f"Concave hull computation failed: {e}. Using convex hull fallback.")
        try:
             hull = ConvexHull(points_array)
             hull_points = points_array[hull.vertices]
             return [[[float(p[0]), float(p[1])] for p in hull_points]]
        except:
             return [[[float(p[0]), float(p[1])] for p in unique_points]]

def compute_cluster_hulls(clusters, alpha: float | None = 1.0, auto_alpha_quantile=0.95):
    """
    Computes alpha shapes for a list of clusters.
    Returns a list of hulls (each hull is a LIST OF POLYGONS, where each polygon is a list of [lat, lon]).
    Note: The return type changed to support MultiPolygons.
    """
    hulls = []
    for cluster in clusters:
        coords = []
        for p in cluster:
            if isinstance(p, (pd.Series, dict)):
                lat = p.get('latitude') if isinstance(p, dict) else p['latitude']
                lon = p.get('longitude') if isinstance(p, dict) else p['longitude']
                coords.append([lat, lon])
            else:
                coords.append(p)
        
        if not coords:
            continue
            
        hull_polys = get_alpha_shape(coords, alpha=alpha, auto_alpha_quantile=auto_alpha_quantile)
        # Flatten the list of polygons into the hulls list
        # NO. We want to keep each cluster associated with its geometry.
        # But Visualisation.draw_cluster handles one geometry object at a time.
        # If we pass a list of polygons (list of list of points), Folium treats it as MultiPolygon.
        # So we just append the result of get_alpha_shape directly.
        hulls.append(hull_polys)
    return hulls
