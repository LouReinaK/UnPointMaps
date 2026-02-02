# service de clustering de points dans un espace 2D
# dans un premier temps on implémentera un k-means avec la distance
# euclidienne et on cherchera le meilleur k
from typing import List, Tuple, Any
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from scipy.spatial import ConvexHull
try:
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score
except ImportError:
    KMeans = None
    NearestNeighbors = None
    silhouette_score = None
from ..processing.dataset_filtering import convert_to_dict_filtered


def load_dataset(points) -> np.ndarray:
    """
        Convertit une liste de points en tableau numpy (lat, long)
    """
    points = convert_to_dict_filtered()
    return np.array([[p.lat, p.long] for p in points])


def find_optimal_k_elbow(
    dataset: np.ndarray,
    k_range: range = range(
        2,
        11)) -> int:
    """
        Trouve le k optimal en utilisant la méthode du coude (Elbow method)
    """
    if KMeans is None:
        raise ImportError("scikit-learn is required for KMeans clustering.")
    inertias = []
    k_values = list(k_range)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(dataset)
        inertias.append(kmeans.inertia_)

    # Calcul du coude en utilisant la méthode de la dérivée seconde
    if len(inertias) >= 3:
        differences = np.diff(inertias)
        second_differences = np.diff(differences)
        elbow_index = np.argmax(np.abs(second_differences)) + 1
        return k_values[elbow_index]

    return k_values[0]


def find_optimal_k_silhouette(
    dataset: np.ndarray,
    k_range: range = range(
        2,
        11)) -> int:
    """
        Trouve le k optimal en utilisant le score de silhouette
    """
    if KMeans is None or silhouette_score is None:
        raise ImportError("scikit-learn is required for KMeans clustering and silhouette score.")

    silhouette_scores = []
    k_values = list(k_range)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(dataset)
        score = silhouette_score(dataset, labels)
        silhouette_scores.append(score)

    # Le k optimal est celui qui maximise le score de silhouette
    optimal_index = np.argmax(silhouette_scores)
    return k_values[optimal_index]


def kmeans_clustering(dataset: Any, k: int | None = None,
                      method: str = "elbow") -> Tuple[List, int, np.ndarray]:
    """
        Fonction principale de clustering avec k-means
        peut trouver le k optimal avec la méthode du coude ou du score de silhouette si k n'est pas fourni
    """
    if KMeans is None:
        raise ImportError("scikit-learn is required for KMeans clustering.")

    # Préparation des données pour scikit-learn
    if hasattr(dataset, 'iloc'):  # pandas DataFrame
        # Extraction des coordonnées
        points_array = dataset[['latitude', 'longitude']].values
    else:  # Supposons numpy array ou liste
        points_array = np.array(dataset)

    # trouver k optimal si nécessaire
    if k is None:
        if method == 'elbow':
            k = find_optimal_k_elbow(
                points_array, k_range=range(2, min(11, len(points_array))))
        elif method == 'silhouette':
            k = find_optimal_k_silhouette(
                points_array, k_range=range(2, min(11, len(points_array))))
        else:
            # Default k if none of the above
            k = 5

    # Application du clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(points_array)

    # Organisation des points par cluster pour la visualisation
    clusters: List[List[List[float]]] = [[] for _ in range(k)]
    for i, label in enumerate(labels):
        clusters[label].append(points_array[i].tolist())

    return clusters, k, labels


def plot_k_distance(data, k):
    """
    Plots the k-distance graph to help find the optimal eps for DBSCAN.
    Sorts distances to the k-th nearest neighbor.

    Args:
        data: numpy array or array-like of points (lat, lon)
        k: min_samples used for DBSCAN (typically)
    """
    if data is None or len(data) == 0:
        print("No data to plot k-distance graph.")
        return

    # Convert to numpy array if needed
    if hasattr(data, 'iloc'):
        try:
            points = data[['latitude', 'longitude']].values
        except KeyError:
            print("Error: DataFrame must have 'latitude' and 'longitude' columns.")
            return
    else:
        points = np.array(data)

    if len(points) < k:
        print(
            f"Not enough points ({len(points)}) to compute {k}-nearest neighbors.")
        return

    # We need k neighbors. NearestNeighbors includes the point itself as the first neighbor (dist=0)
    # So we ask for k neighbors.
    if NearestNeighbors is None:
        raise ImportError("scikit-learn is required for parameter checking.")
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(points)
    distances, _ = neighbors_fit.kneighbors(points)

    # Sort distance values by the k-th neighbor distance (last column)
    # The distances are returned sorted by nearest.
    # column 0 is the point itself (0 distance).
    # column k-1 is the k-th neighbor (since we asked for k neighbors)
    sorted_distances = np.sort(distances[:, k - 1], axis=0)

    # Calculate first and second derivatives
    first_derivative = np.gradient(sorted_distances)
    second_derivative = np.gradient(first_derivative)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Original K-Distance Graph
    axes[0].plot(sorted_distances)
    axes[0].set_title(f"K-Distance Graph (k={k})")
    axes[0].set_ylabel(f"Distance to {k}-th nearest neighbor (Eps)")
    axes[0].grid(True)

    # First Derivative
    axes[1].plot(first_derivative, color='orange')
    axes[1].set_title("First Derivative")
    axes[1].set_ylabel("Rate of Change")
    axes[1].grid(True)

    # Second Derivative
    axes[2].plot(second_derivative, color='green')
    axes[2].set_title("Second Derivative")
    axes[2].set_ylabel("Curvature")
    axes[2].set_xlabel("Points sorted by distance to k-th nearest neighbor")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_clusters(data, titles=None):
    """
    Affiche un ou plusieurs graphiques matplotlib avec les clusters.
    Peut prendre en entrée :
    - Une liste de clusters (cas unique)
    - Une liste de listes de clusters (cas multiple) pour comparaison
    """
    # Détection si on a affaire à un seul résultat ou plusieurs
    is_multiple = False

    if data and isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        if isinstance(first_item, list) and len(first_item) > 0:
            first_sub = first_item[0]
            if isinstance(first_sub, (list, np.ndarray)
                          ) and len(first_sub) > 0:
                # Si le contenu est numérique, c'est un point => data est une liste de clusters (Single)
                # Sinon (si c'est un itérable), c'est un cluster => data est
                # une liste de résultats (Multiple)
                try:
                    if isinstance(first_sub[0], (int, float, np.number)):
                        is_multiple = False
                    else:
                        is_multiple = True
                except BaseException:
                    pass

    results = data if is_multiple else [data]

    if not results or not results[0]:
        print("Aucune donnée à afficher.")
        return

    num_plots = len(results)

    if titles and len(titles) != num_plots:
        print("Warning: Number of titles doesn't match number of plots.")
        titles = None

    # Determine layout
    cols = min(num_plots, 3)
    rows = (num_plots - 1) // cols + 1

    # squeeze=False assure que axes est toujours un tableau 2D, flatten() le
    # rend 1D
    fig, axes = plt.subplots(rows, cols, figsize=(
        6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    idx = 0
    for idx, clusters in enumerate(results):
        ax = axes[idx]

        for i, cluster in enumerate(clusters):
            points = np.array(cluster)
            if len(points) > 0:
                # Lat=0 (Y), Lon=1 (X)
                ax.scatter(points[:, 1], points[:, 0], label=f'Cluster {i}')

                # Draw convex hull if enough points
                if len(points) >= 3:
                    # Remove duplicates for hull calculation
                    unique_points = np.unique(points, axis=0)
                    if len(unique_points) >= 3:
                        try:
                            # QJ: Joggle inputs. This is robust against
                            # coplanar/collinear points
                            hull = ConvexHull(
                                unique_points, qhull_options='QJ')
                            hull_indices = np.append(
                                hull.vertices, hull.vertices[0])
                            ax.plot(
                                unique_points[hull_indices, 1], unique_points[hull_indices, 0], '--', alpha=0.5)
                        except Exception as e:
                            # Silently fail or log sparingly to avoid console spam
                            # print(f"Could not compute hull for cluster {i} in plot {idx}: {e}")
                            pass

        title = titles[idx] if titles else (
            f'Visualisation des Clusters ({idx+1})' if num_plots > 1 else 'Visualisation des Clusters')
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        ax.grid(True)

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Exemple d'utilisation
    # fixe la seed pour la reproductibilité
    np.random.seed(42)
    # Génère des points aléatoires
    sample_points = np.array(
        [[np.random.uniform(-90, 90), np.random.uniform(-180, 180)] for _ in range(100)])

    clustered_points, used_k, labels = kmeans_clustering(sample_points)
    print(f"Clustering effectué avec k={used_k}")

    plot_clusters(clustered_points)
