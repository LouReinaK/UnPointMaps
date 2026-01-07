import folium
import pandas as pd

from models import Point

# Trinome 3

# Sample data (replace with your CSV)
geo_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"


class Visualisation:
    def __init__(self, points):
        self.data = points
        self.clusters = []
        self.markers = []

    def draw_cluster(self, shape):
        self.clusters.append(shape)

        # folium.Choropleth(
        #     geo_data=geo_url,
        #     data=self.data,
        #     columns=["Country", "Value"],
        #     key_on="feature.properties.name",
        #     fill_color="RdYlGn_r",
        #     fill_opacity=0.7,
        #     line_opacity=0.2,
        #     legend_name="Sample Values"
        # ).add_to(m)

    def draw_point(self, point):
        self.markers.append(
            folium.Marker(
                location=[point.latitude, point.longitude],
                popup=point.name,
                tooltip=f"Tooltip: {point.name}",
            )
        )

    def create_map(self, output_file="map.html"):
        m = folium.Map(location=[30, 10], zoom_start=3)
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
        Point(name="Point B", latitude=34.0522, longitude=-118.2437),  # Los Angeles
    ]

    # Sample shape data
    shape_data = [
        folium.Polygon(
            locations=[
                [48.85, 2.35],
                [48.86, 2.36],
                [48.87, 2.35],
                [48.86, 2.34],
            ],
            color="blue",
            fill=True,
            fill_color="blue",
            popup="Shape around Paris",
        )
    ]

    vis = Visualisation(points)
    for point in points:
        vis.draw_point(point)

    for shape in shape_data:
        vis.draw_cluster(shape)

    vis.create_map("output_map.html")
