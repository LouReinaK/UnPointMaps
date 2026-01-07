from clustering import kmeans_clustering
from dataset_filtering import convert_to_dict_filtered
from map_visualisation import Visualisation


print("--- Starting UnPointMaps ---")
print("Loading and filtering dataset...")
df = convert_to_dict_filtered()
print("Limiting dataset to first 100 points for performance...")
df = df.head(100)

k = 5

print(f"Starting clustering with k={k} and method='elbow'...")
clustered_points, used_k = kmeans_clustering(df, k=k, method="elbow")
print(f"Clustering complete. Used k={used_k}.")


vis = Visualisation()
# print("Adding points to map...")
# for _, row in df.iterrows():
#     vis.draw_point(row)

print("Adding clusters to map...")
for cluster in clustered_points:
    vis.draw_cluster(cluster)

print("Generating output map 'output_map.html'...")
vis.create_map("output_map.html")
print("--- Done ---")