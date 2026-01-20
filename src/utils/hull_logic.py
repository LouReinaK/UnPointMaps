import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point, LineString, GeometryCollection
from shapely.ops import unary_union
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import multiprocessing
import concurrent.futures

import json
import hashlib
from src.database.manager import DatabaseManager

# --- Optimization Step 4: Caching (via DatabaseManager) ---
# We use a global instance for the module. 
# Note: SQLite connections shouldn't be shared across processes, 
# but DatabaseManager creates a fresh connection per call, so it is safe.
_DB_MANAGER = DatabaseManager()

def _hash_points(points):
    """Creates a deterministic hash for the points to use in persistent caching."""
    if isinstance(points, np.ndarray):
        if not points.flags['C_CONTIGUOUS']:
            points = np.ascontiguousarray(points)
        data = points.tobytes()
    else:
        # list of lists
        data = str(points).encode('utf-8')
    return hashlib.md5(data).hexdigest()

def _get_cache_key(points, alpha, auto_alpha_quantile):
    # Combine point hash with parameters
    h = _hash_points(points)
    # Ensure params are strings
    a = str(alpha) if alpha is not None else "None"
    q = str(auto_alpha_quantile)
    return f"{h}_a{a}_q{q}"

def _cache_result(key, value):
    _DB_MANAGER.save_cached_hull(key, value)

def _get_from_cache(key):
    return _DB_MANAGER.get_cached_hull(key)


def get_alpha_shape(points, alpha=None, auto_alpha_quantile=0.95):
    """
    Computes the alpha shape (concave hull) of a set of 2D points.
    Optimized version with vectorization and caching.
    
    Input: list of [lat, lon] or (lat, lon)
           alpha: Threshold for circumradius of Delaunay triangles.
           auto_alpha_quantile: Percentile to use if alpha is None.
    Output: list of list of [lat, lon]
    """
    # Quick check for empty
    if not points:
        return []

    # 1. Caching Check
    cache_key = _get_cache_key(points, alpha, auto_alpha_quantile)
    cached_res = _get_from_cache(cache_key)
    if cached_res is not None:
        return cached_res

    # Remove duplicates
    unique_points = list(set(tuple(p) for p in points))
    
    if len(unique_points) <= 2:
        res = [[list(p) for p in unique_points]]
        _cache_result(cache_key, res)
        return res
    
    if len(unique_points) == 3:
        res = [[list(p) for p in unique_points]]
        _cache_result(cache_key, res)
        return res

    points_array = np.array(unique_points)
    
    try:
        # --- Optimization Step 3: Delaunay Triangulation ---
        tri = Delaunay(points_array)
        triangles = points_array[tri.simplices]
        
        # --- Optimization Step 9: Use Vectorized Operations ---
        # Calculate side lengths a, b, c
        # Edges: 0-1, 1-2, 2-0
        # Use np.hypot for speed on 2D vectors
        v01 = triangles[:, 0] - triangles[:, 1]
        v12 = triangles[:, 1] - triangles[:, 2]
        v20 = triangles[:, 2] - triangles[:, 0]
        
        a = np.hypot(v01[:, 0], v01[:, 1])
        b = np.hypot(v12[:, 0], v12[:, 1])
        c = np.hypot(v20[:, 0], v20[:, 1])
        
        s = (a + b + c) / 2.0
        # Area using Heron's formula (clamped to 0)
        val = np.maximum(s * (s - a) * (s - b) * (s - c), 0)
        area = np.sqrt(val)
        
        # Filter singularity (very small area)
        valid_tri_mask = area > 1e-10
        
        # Circumradius R = abc / (4 * area)
        circum_r = np.full(len(triangles), np.inf)
        circum_r[valid_tri_mask] = (a[valid_tri_mask] * b[valid_tri_mask] * c[valid_tri_mask]) / (4.0 * area[valid_tri_mask])
        
        # --- Optimization Step 5: Optimize Alpha Calculation ---
        if alpha is None:
            # Filter finite radii
            valid_r = circum_r[np.isfinite(circum_r)]
            if len(valid_r) > 0:
                # np.percentile is essentially efficient
                alpha = np.percentile(valid_r, auto_alpha_quantile * 100)
            else:
                alpha = 1.0
        
        keep_indices = circum_r < alpha

        # --- Optimization: Vectorized Force Inclusion ---
        # Ensure every point is covered by at least one kept triangle.
        
        # 1. Identify uncovered points
        kept_simplices = tri.simplices[keep_indices]
        if len(kept_simplices) > 0:
            covered_vertices = np.unique(kept_simplices)
        else:
            covered_vertices = np.array([], dtype=int)
            
        # If not all points covered
        if len(covered_vertices) < len(points_array):
            all_indices = np.arange(len(points_array))
            missing_indices = np.setdiff1d(all_indices, covered_vertices, assume_unique=True)
            
            if len(missing_indices) > 0:
                # We need: for each missing vertex v, find simplex s incident to v with min(circum_r[s])
                # Data structures:
                # simplex_ids: [0,0,0, 1,1,1, ...]
                # vertex_ids:  [v1,v2,v3, v4,v5,v6, ...]
                # radii:       [r0,r0,r0, r1,r1,r1, ...]
                
                num_simplices = len(tri.simplices)
                simplex_ids = np.repeat(np.arange(num_simplices), 3)
                vertex_ids = tri.simplices.flatten()
                radii_expanded = np.repeat(circum_r, 3)
                
                # Filter valid finite triangles
                valid_mask = np.isfinite(radii_expanded)
                vertex_ids = vertex_ids[valid_mask]
                simplex_ids = simplex_ids[valid_mask]
                radii_vals = radii_expanded[valid_mask]
                
                # Sort by vertex then radius
                sort_idx = np.lexsort((radii_vals, vertex_ids))
                
                sorted_vertices = vertex_ids[sort_idx]
                sorted_simplices = simplex_ids[sort_idx]
                
                # Group by vertex, take first (min radius)
                unique_v, unique_indices = np.unique(sorted_vertices, return_index=True)
                best_simplices = sorted_simplices[unique_indices]
                
                # Filter for only the missing vertices
                # unique_v is sorted, so is missing_indices.
                # Use searchsorted or isin
                mask_needed = np.isin(unique_v, missing_indices)
                simplices_to_add = best_simplices[mask_needed]
                
                keep_indices[simplices_to_add] = True

        # --- Optimization Step 7: Simplify Connectivity (Vectorized Graph) ---
        if np.any(keep_indices):
            num_tri = len(tri.simplices)
            
            # Construct Edge List from neighbors
            # tri.neighbors: (N, 3). -1 means no neighbor.
            src = np.repeat(np.arange(num_tri), 3)
            dst = tri.neighbors.flatten()
            
            # Filter valid and directed (dst > src) to avoid duplicates
            valid_edges = (dst != -1) & (dst > src)
            s = src[valid_edges]
            d = dst[valid_edges]
            
            # Note: We do NOT filter infinite circumradius triangles here.
            # Allowing them is crucial to maintain connectivity (acting as bridges)
            # even if they are degenerate.
            
            if len(s) > 0:
                # Weights: max(cost[u], cost[v])
                # Cost is 0 if kept, else radius
                costs = np.where(keep_indices, 0.0, circum_r)
                w = np.maximum(costs[s], costs[d])
                
                # Build MST using scipy
                graph_matrix = csr_matrix((w, (s, d)), shape=(num_tri, num_tri))
                mst = minimum_spanning_tree(graph_matrix)
                
                # Pruning Logic
                mst_coo = mst.tocoo()
                
                # Calculate degrees
                # np.bincount is fast for 0..N integers
                degrees = np.bincount(mst_coo.row, minlength=num_tri) + \
                          np.bincount(mst_coo.col, minlength=num_tri)
                
                # Adjacency list for traversal
                adj = [set() for _ in range(num_tri)]
                for u, v in zip(mst_coo.row, mst_coo.col):
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Iterative Pruning
                # Leaves that are not originally kept
                # 0-degree nodes (isolated) are dealt with later (keep if originally kept)
                leaves = [i for i in range(num_tri) if degrees[i] == 1 and not keep_indices[i]]
                
                eliminated = np.zeros(num_tri, dtype=bool)
                queue = leaves
                q_idx = 0
                
                while q_idx < len(queue):
                    u = queue[q_idx]
                    q_idx += 1
                    eliminated[u] = True
                    
                    # Neighbor
                    # In a tree, if degree was 1, there is exactly 1 neighbor in adj[u]
                    # But we must check if neighbor is not already eliminated (unlikely in tree traversal from leaves but safe)
                    for v in adj[u]:
                        if not eliminated[v]:
                            degrees[v] -= 1
                            if degrees[v] == 1 and not keep_indices[v]:
                                queue.append(v)
            
                # Update keep_indices
                # A node is kept if: 
                # 1. It was originally kept
                # 2. It is in the MST and NOT eliminated (bridge)
                
                # Identify nodes in MST (degree > 0 in original MST)
                initial_degrees = np.bincount(mst_coo.row, minlength=num_tri) + \
                                  np.bincount(mst_coo.col, minlength=num_tri)
                in_mst = initial_degrees > 0
                
                # Add bridges
                keep_indices = keep_indices | (in_mst & ~eliminated)
            
        # Final validity check: removed to allow infinite radius bridge triangles
        # keep_indices = keep_indices & np.isfinite(circum_r)
        
        # 8. Fallback to Convex Hull if empty
        if not np.any(keep_indices):
            return _fallback_convex(points_array, cache_key)
            
        kept_triangles = triangles[keep_indices]
        
        # Merge triangles
        # polys = [Polygon(t) for t in kept_triangles]
        # concave_hull = unary_union(polys)
        
        # Optimized: Extract boundary edges directly
        # kept_simplices = tri.simplices[keep_indices]
        # But we need indices relative to points_array, which tri.simplices already provides.
        res = _extract_boundary_edges_fast(tri.simplices[keep_indices], points_array)
        
        if not res:
             # Fallback if fast extraction failed or produced nothing (e.g. no cycles found)
             polys = [Polygon(t) for t in kept_triangles]
             concave_hull = unary_union(polys)
             res = _extract_multipolygon_coords(concave_hull)
             
        if not res:
            return _fallback_convex(points_array, cache_key)
            
        _cache_result(cache_key, res)
        return res

    except Exception as e:
        print(f"Concave hull failed: {e}")
        return _fallback_convex(points_array, cache_key)

def _fallback_convex(points_array, cache_key):
    try:
        hull = ConvexHull(points_array)
        res = [[[float(p[0]), float(p[1])] for p in points_array[hull.vertices]]]
        _cache_result(cache_key, res)
        return res
    except:
        return [[[float(p[0]), float(p[1])] for p in points_array]]

def _extract_boundary_edges_fast(triangles_indices, points_array):
    """
    Extracts the boundary polygon(s) from a set of triangle indices.
    This avoids shapely.ops.unary_union which is slow for many polygons.
    """
    # Create array of sorted edges: (N, 3, 2)
    # Each triangle has 3 edges. We sort vertex indices per edge to handle (u,v) == (v,u)
    edges = np.sort(np.vstack([
        triangles_indices[:, [0, 1]],
        triangles_indices[:, [1, 2]],
        triangles_indices[:, [2, 0]]
    ]), axis=1)
    
    # We turn rows into structured array or view to use unique
    # A fast way to find unique rows for integer arrays:
    # Convert to void type or allow numpy to sort rows directly
    # edges is (M, 2).
    
    # Pack edges into 64-bit integers if indices < 2^32 for speed (assuming indices fit)
    # Or just use row sorting
    
    # Lexsort: sort by col 0, then col 1
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    
    # Identify unique edges and their counts
    # diff between adjacent rows
    diff = np.diff(sorted_edges, axis=0)
    # check where diff is non-zero (i.e., new edge)
    # diff is (M-1, 2). If any col is non-zero, it's a change
    is_new = np.any(diff != 0, axis=1)
    is_new = np.concatenate(([True], is_new, [True])) # Sentinels
    
    # Indices where changes occur
    change_indices = np.flatnonzero(is_new)
    
    # Counts are differences between change indices
    counts = np.diff(change_indices)
    
    # Unique edges are at change_indices[:-1]
    unique_edges = sorted_edges[change_indices[:-1]]
    
    # Boundary edges appear exactly once (or odd times if geometry is weird, but for valid triangulation, 1 means boundary, 2 means internal)
    # Actually, in 2D Delaunay, edges are shared by at most 2 triangles. 
    # 1 = boundary
    # 2 = internal
    boundary_mask = (counts == 1)
    boundary_edges = unique_edges[boundary_mask]
    
    if len(boundary_edges) == 0:
        return []
        
    return _stitch_edges(boundary_edges, points_array)

def _stitch_edges(edges, points_array):
    """
    Stitches a list of (u, v) edges into loops (polygons).
    """
    # Build adjacency list
    adj = {}
    for u, v in edges:
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
        
    # Traverse to find loops
    polygons = []
    visited_edges = set()
    
    # We need to handle the fact that we might have disjoint loops (holes or islands)
    # For a simple polygon, we can just walk.
    # Note: Orientation matters for holes vs islands, but we just return coords here.
    
    for start_node in adj:
        if not adj[start_node]:
            continue
            
        # Try to find a loop starting from here
        # Eagerly pick a neighbor that hasn't been traversed
        
        # We need to track directed edges to avoid going back and forth?
        # A simple valid boundary traversal: preserve direction from triangulation?
        # We lost directionality in the edge sorting (u<v).
        # But we can reconstruct:
        # Just walking the graph of boundary edges should produce cycles.
        # Since every node in a valid 2-manifold boundary has degree 2 (or even), 
        # we can just walk.
        # If the union of triangles is a single component (enforced by MST), we usually get one outer loop + holes.
        
        # We use a set of visited nodes is NOT enough because a node can be part of multiple loops (touching at a vertex).
        # We need visited EDGES.
        pass
        
    # Simplified Graph Traversal
    # Convert edges to set for O(1) lookup
    edge_set = set(tuple(sorted((u, v))) for u, v in edges)
    
    loops = []
    
    while edge_set:
        # Start a new loop
        u, v = next(iter(edge_set)) # Arbitrary start edge
        edge_set.remove((u, v) if (u, v) in edge_set else (v, u))
        
        loop_coords = [points_array[u], points_array[v]]
        curr = v
        start = u
        
        while curr != start:
            # Find neighbor of curr in remaining edges
            found = False
            # Check neighbors of curr
            # Optimization: looking up in adj is faster than iterating edge_set
            # But we need to keep adj in sync or just check simple possibilities?
            # Creating a full graph is safer.
            pass
            break # Fallback to slower robust construction below
            
        loops.append(loop_coords)
       
    # --- Robust Graph Construction for Stitching ---
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Extract cycles
    # For a set of polygons, the boundary is a set of cycles.
    # nx.cycle_basis might give internal cycles which we don't want.
    # We want simple connected components of the boundary graph.
    # Since every node has degree 2 (ideally), components are just cycles.
    
    loops_coords = []
    try:
        # Get connected components of the edge graph
        components = list(nx.connected_components(G))
        for comp in components:
            if len(comp) < 3: continue
            
            # Subgraph for this component
            subg = G.subgraph(comp)
            
            # An Eulerian circuit exists if all degrees are even.
            # In a boundary graph, degrees should be 2.
            # If so, find cycle.
            try:
                cycle = list(nx.find_cycle(subg))
                # Cycle is list of edges (u,v). Extract vertices.
                path = [cycle[0][0]]
                for _, v in cycle:
                    path.append(v)
                    
                loops_coords.append([ [float(points_array[i][0]), float(points_array[i][1])] for i in path ])
            except:
                pass
    except:
        pass
        
    return loops_coords


def _extract_multipolygon_coords(geom):
    coords_list = []
    if isinstance(geom, Polygon):
        geoms = [geom]
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        geoms = geom.geoms
    else:
        return []
        
    for poly in geoms:
        if isinstance(poly, Polygon):
            c = list(poly.exterior.coords)
            if len(c) > 1 and c[0] == c[-1]: c = c[:-1]
            coords_list.append([[float(p[0]), float(p[1])] for p in c])
            
    return coords_list

# --- Optimization Step 6: Parallel Execution ---
def _compute_hull_wrapper(args):
    """Wrapper for parallel execution"""
    points, alpha, quantile = args
    return get_alpha_shape(points, alpha=alpha, auto_alpha_quantile=quantile)

def compute_cluster_hulls(clusters, alpha: float | None = 1.0, auto_alpha_quantile=0.95, max_workers=None):
    """
    Computes alpha shapes for a list of clusters.
    """
    clean_clusters = []
    for cluster in clusters:
        coords = []
        for p in cluster:
            if isinstance(p, (pd.Series, dict)):
                lat = p.get('latitude') if isinstance(p, dict) else p['latitude']
                lon = p.get('longitude') if isinstance(p, dict) else p['longitude']
                coords.append([lat, lon])
            else:
                coords.append(p)
        clean_clusters.append(coords)
    
    if not clean_clusters:
        return []

    # Only parallelize if we have enough work
    if len(clean_clusters) < 5:
        return [get_alpha_shape(c, alpha=alpha, auto_alpha_quantile=auto_alpha_quantile) if c else [] for c in clean_clusters]

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)

    tasks = [(c, alpha, auto_alpha_quantile) for c in clean_clusters]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_compute_hull_wrapper, tasks))

    return results
