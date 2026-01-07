# service de clustering de points dans un espace 2D
# les points sont représentés par des objets de la classe Point et on utilisera les propriétés Point.lat et Point.long
# dans un premier temps on implémentera un k-means avec la distance euclidienne et on cherchera le meilleur k
import numpy as np
from dataset_filtering import convert_to_dict_filtered
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import List, Tuple


def load_dataset(points) -> np.ndarray:
    """
        Convertit une liste de points en tableau numpy (lat, long)
    """
    points = convert_to_dict_filtered()
    return np.array([[p.lat, p.long] for p in points])


def find_optimal_k_elbow(dataset: np.ndarray, k_range: range = range(2, 11)) -> int:
    """
        Trouve le k optimal en utilisant la méthode du coude (Elbow method)
    """
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


def find_optimal_k_silhouette(dataset: np.ndarray, k_range: range = range(2, 11)) -> int:
    """
        Trouve le k optimal en utilisant le score de silhouette
    """
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


def kmeans_clustering(dataset: any, k: int = None, 
                   method: str = "elbow") -> Tuple[List, int]:
    """
        Fonction principale de clustering avec k-means
        peut trouver le k optimal avec la méthode du coude ou du score de silhouette si k n'est pas fourni
    """

    # Préparation des données pour scikit-learn
    if hasattr(dataset, 'iloc'): # pandas DataFrame
        # Extraction des coordonnées
        points_array = dataset[['latitude', 'longitude']].values
    else: # Supposons numpy array ou liste
        points_array = np.array(dataset)

    # trouver k optimal si nécessaire
    if k is None and method=='elbow':
        k = find_optimal_k_elbow(points_array, k_range=range(2, min(11, len(points_array))))
    elif k is None and method=='silhouette':
        k = find_optimal_k_silhouette(points_array, k_range=range(2, min(11, len(points_array))))
    
    # Application du clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(points_array)
    
    # Organisation des points par cluster pour la visualisation
    clusters = [[] for _ in range(k)]
    for i, label in enumerate(labels):
        clusters[label].append(points_array[i].tolist())
        
    return clusters, k
    
    return points, k

if __name__ == "__main__":
    # Exemple d'utilisation
    # fixe la seed pour la reproductibilité
    np.random.seed(42)
    # Génère des points aléatoires
    sample_points = np.array([Point(lat=np.random.uniform(-90, 90), long=np.random.uniform(-180, 180)) for _ in range(100)])
    
    clustered_points, used_k = kmeans_clustering(sample_points)
    print(f"Clustering effectué avec k={used_k}")
    for p in clustered_points[:20]:  # Affiche les 20 premiers points clusterisés
        print(f"Point({p.lat}, {p.long}) -> Cluster {p.cluster}")
