try:
    import folium
except ImportError:
    folium = None
import pandas as pd

# Trinome 3


class Visualisation:
    def __init__(self):
        self.clusters = []
        self.markers = []
        self.points = []

    def draw_cluster(self, points, **kwargs):
        """
        Adds a polygon (cluster hull) to the map.
        points: list of [lat, lon]
        """
        if folium is None:
            return

        kwargs = {
            "color": "blue",
            "fill": True,
            "fill_color": "blue",
            "weight": 2,
            "opacity": 0.7,
            "fill_opacity": 0.3,
        } | kwargs

        self.clusters.append(folium.Polygon(locations=points, **kwargs))

    def draw_point(self, point, **kwargs):
        if folium is None:
            return

        if isinstance(point, (pd.Series, dict)):
            latitude = point['latitude']
            longitude = point['longitude']
            name = point.get('title', point.get('name', 'Unknown'))
        else:
            try:
                # Try to unpack as list/tuple
                latitude, longitude = point
                name = "Unknown"
            except (ValueError, TypeError):
                # Fallback if it's an object with attributes (like the old
                # Point, or similar)
                if hasattr(point, 'latitude') and hasattr(point, 'longitude'):
                    latitude = point.latitude
                    longitude = point.longitude
                    name = getattr(point, 'title', getattr(
                        point, 'name', 'Unknown'))
                else:
                    raise TypeError(f"Cannot interpret point: {point}")

        kwargs = kwargs | {"popup": name, "tooltip": f"Tooltip: {name}"}

        self.markers.append(
            folium.Marker(location=[latitude, longitude], **kwargs)
        )

    def create_map(self, output_file="map.html", bounds=None):
        if folium is None:
            print("Error: 'folium' library is missing. Cannot generate map. Please install it with 'pip install folium'.")
            return

        # Default center/zoom
        zoom_start = 12

        # If bounds provided explicitly, use them (optimized path)
        if bounds is not None:
            try:
                min_lat, min_lon = bounds[0]
                max_lat, max_lon = bounds[1]
                center_point = [(min_lat + max_lat) / 2,
                                (min_lon + max_lon) / 2]
            except Exception:
                # Malformed bounds: fall back to defaults
                bounds = None
                center_point = [45.767328, 4.833362]
        else:
            bounds = None
            center_point = [45.767328, 4.833362]

        m = folium.Map(location=center_point, zoom_start=zoom_start)
        print('Map initialized.')

        if self.clusters:
            print(f"Adding {len(self.clusters)} clusters to map...")
            for cluster in self.clusters:
                cluster.add_to(m)

        if self.markers:
            print(f"Adding {len(self.markers)} markers to map...")
            for marker in self.markers:
                marker.add_to(m)

        # If we tracked points, fit the map to their bounds
        if bounds:
            try:
                folium.FitBounds(bounds).add_to(m)
            except Exception:
                # Fall back silently if FitBounds is unavailable or fails
                pass

        folium.LayerControl().add_to(m)

        print(f"Saving map to {output_file}...")
        m.save(output_file)


if __name__ == "__main__":
    # Sample data points
    points = [
        [48.8566, 2.3522],  # Paris
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
