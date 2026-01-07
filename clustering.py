# service de clustering de points dans un espace 2D
# les points sont représentés par des objets de la classe Point et on utilisera les propriétés Point.lat et Point.long
# dans un premier temps on implémentera un k-means avec la distance euclidienne et on cherchera le meilleur k
import numpy as np
from models import Point
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import List, Tuple


def points_to_array(points: List[Point]) -> np.ndarray:
    """
        Convertit une liste de points en tableau numpy (lat, long)
    """
    return np.array([[p.lat, p.long] for p in points])


def find_optimal_k_elbow(points: List[Point], k_range: range = range(2, 11)) -> int:
    """
        Trouve le k optimal en utilisant la méthode du coude (Elbow method)
    """
    X = points_to_array(points)
    inertias = []
    k_values = list(k_range)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Calcul du coude en utilisant la méthode de la dérivée seconde
    if len(inertias) >= 3:
        differences = np.diff(inertias)
        second_differences = np.diff(differences)
        elbow_index = np.argmax(np.abs(second_differences)) + 1
        return k_values[elbow_index]
    
    return k_values[0]


def find_optimal_k_silhouette(points: List[Point], k_range: range = range(2, 11)) -> int:
    """
        Trouve le k optimal en utilisant le score de silhouette
    """
    X = points_to_array(points)
    silhouette_scores = []
    k_values = list(k_range)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    # Le k optimal est celui qui maximise le score de silhouette
    optimal_index = np.argmax(silhouette_scores)
    return k_values[optimal_index]


def kmeans_clustering(points: List[Point], k: int = None, 
                   method: str = "elbow") -> Tuple[List[Point], int]:
    """
        Fonction principale de clustering avec k-means
        peut trouver le k optimal avec la méthode du coude ou du score de silhouette si k n'est pas fourni
    """
    if k is None and method=='elbow':
        k = find_optimal_k_elbow(points, k_range=range(2, min(11, len(points))))
    elif k is None and method=='silhouette':
        k = find_optimal_k_silhouette(points, k_range=range(2, min(11, len(points))))
    
    # Application du clustering
    X = points_to_array(points)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Ajout de l'attribut cluster à chaque point
    for i, point in enumerate(points):
        point.cluster = int(labels[i])
    
    return points, k

if __name__ == "__main__":
    # Exemple d'utilisation
    # fixe la seed pour la reproductibilité
    np.random.seed(42)
    # Génère des points aléatoires
    sample_points = [Point(lat=np.random.uniform(-90, 90), long=np.random.uniform(-180, 180)) for _ in range(100)]
    
    clustered_points, used_k = cluster_points(sample_points, k=3)
    print(f"Clustering effectué avec k={used_k}")
    for p in clustered_points[:20]:  # Affiche les 20 premiers points clusterisés
        print(f"Point({p.lat}, {p.long}) -> Cluster {p.cluster}")
