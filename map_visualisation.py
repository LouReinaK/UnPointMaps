import folium
import pandas as pd

# Trinome 3


class Visualisation:
    def __init__(self):
        self.clusters = []
        self.markers = []

    def draw_cluster(self, points, **kwargs):
        kwargs = kwargs | {
            "color": "blue",
            "fill": True,
            "fill_color": "blue",
            "popup": "Shape around Paris",
        }

        self.clusters.append(folium.Polygon(locations=points, **kwargs))

    @staticmethod
    def get_convex_hull(points):
        """
        Computes the convex hull of a set of 2D points using the Monotone Chain algorithm.
        Input: list of [lat, lon] or (lat, lon)
        Output: list of [lat, lon] forming the convex hull
        """
        # Remove duplicates and sort
        unique_points = sorted(list(set(tuple(p) for p in points)))
        
        if len(unique_points) <= 2:
            return [list(p) for p in unique_points]

        # Cross product of two vectors OA and OB
        # Returns > 0 for counter-clockwise turn, < 0 for clockwise, 0 for collinear
        # Coordinates are (lat, lon) -> lat=y, lon=x for the algo logic (or vice versa, consistency matters)
        # Here: p[0]=lat, p[1]=lon. 
        # Vector OA = (a[0]-o[0], a[1]-o[1])
        # Cross product (2D) = x1*y2 - x2*y1. 
        # If we treat lat as x and lon as y:
        # (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build lower hull
        lower = []
        for p in unique_points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(unique_points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenate lower and upper hull
        return [list(p) for p in (lower[:-1] + upper[:-1])]

    def draw_cluster_hull(self, points, **kwargs):
        """
        Computes and draws the convex hull for a given set of points.
        points: List of [lat, lon] or Point objects
        """
        coords = []
        for p in points:
            if isinstance(p, Point):
                if hasattr(p, 'latitude') and hasattr(p, 'longitude'):
                    coords.append([p.latitude, p.longitude])
                elif hasattr(p, 'lat') and hasattr(p, 'long'):
                    coords.append([p.lat, p.long])
            else:
                coords.append(p)

        if not coords:
            return

        hull_points = self.get_convex_hull(coords)
        self.draw_cluster(hull_points, **kwargs)

    def draw_point(self, point, **kwargs):
        kwargs = kwargs | {"popup": point.name, "tooltip": f"Tooltip: {point.name}"}

        self.markers.append(
            folium.Marker(location=[point.latitude, point.longitude], **kwargs)
        )

    def create_map(self, output_file="map.html"):
        center_point = (
            [
                sum(p.location[0] for p in self.markers) / len(self.markers),
                sum(p.location[1] for p in self.markers) / len(self.markers),
            ]
            if self.markers
            else [46.8131, 1.6910]
        )

        zoom_start = 12

        m = folium.Map(location=center_point, zoom_start=zoom_start)
        if self.clusters:
            for cluster in self.clusters:
                cluster.add_to(m)

        if self.markers:
            for marker in self.markers:
                marker.add_to(m)

        folium.LayerControl().add_to(m)

        m.save(output_file)


if __name__ == "__main__":
    # Sample data points
    points = [
        Point(name="Point A", latitude=48.8566, longitude=2.3522),  # Paris
        # Point(name="Point B", latitude=34.0522, longitude=-118.2437),  # Los Angeles
    ]

    # Sample shape data
    shape_data = [
        [
            [48.85, 2.35],
            [48.86, 2.36],
            [48.87, 2.35],
            [48.86, 2.34],
        ]
    ]

    vis = Visualisation()
    for point in points:
        vis.draw_point(point)

    for shape in shape_data:
        vis.draw_cluster(shape)

    vis.create_map("output_map.html")
