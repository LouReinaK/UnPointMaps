from clustering import kmeans_clustering
from dataset_filtering import convert_to_dict_filtered
from map_visualisation import Visualisation


df = convert_to_dict_filtered()
df = df.head(20)

k = 5

clustered_points, used_k = kmeans_clustering(df, k=k, method="elbow")


class Point:
    def __init__(self, name, latitude, longitude):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude

vis = Visualisation()
for _, row in df.iterrows():
    point = Point(row['title'], row['latitude'], row['longitude'])
    vis.draw_point(point)

for cluster in clustered_points:
    vis.draw_cluster(cluster)

vis.create_map("output_map.html")