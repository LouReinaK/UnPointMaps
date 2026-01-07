import folium
import pandas as pd

from models import Point

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
