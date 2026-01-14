import pytest
import time
import sys
import os
import numpy as np

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Point
from ..src.clustering.clustering import kmeans_clustering

def generate_random_points(n_points):
    """Génère n points aléatoires"""
    return [
        Point(
            lat=np.random.uniform(-90, 90),
            long=np.random.uniform(-180, 180)
        ) 
        for _ in range(n_points)
    ]

def test_kmeans_performance_large_dataset():
    """
    Test de performance pour le clustering sur un grand jeu de données
    Vérifie que le temps d'exécution reste raisonnable
    """
    n_points = 10000
    k = 10
    points = generate_random_points(n_points)
    
    start_time = time.time()
    labels = kmeans_clustering(points, k)
    execution_time = time.time() - start_time
    
    print(f"\nTemps d'exécution pour {n_points} points et k={k}: {execution_time:.4f} secondes")
    
    # Vérifications de cohérence
    assert len(labels) == n_points
    # Le clustering devrait être raisonnablement rapide (moins de 2s sur une machine moderne pour 10k points)
    # On met une limite large pour éviter les échecs en CI/machines lentes
    assert execution_time < 5.0, f"Le clustering est trop lent: {execution_time}s"

def test_find_optimal_k_performance():
    """
    Test de performance pour la recherche du k optimal (méthode du coude et silhouette)
    """
    n_points = 1000
    points = generate_random_points(n_points)
    
    start_time = time.time()
    # On teste sur un range réduit pour le test de perf
    results = find_optimal_k(points, k_range=range(2, 6), method='both', plot=False)
    execution_time = time.time() - start_time
    
    print(f"\nTemps pour trouver k optimal (1000 points, k=2..5): {execution_time:.4f} secondes")
    
    assert 'elbow' in results
    assert 'silhouette' in results
    # Silhouette est coûteux en O(N^2), donc on vérifie juste que ça termine
    assert execution_time < 10.0

@pytest.mark.benchmark
def test_kmeans_scalability():
    """
    Test de scalabilité simple : le temps ne devrait pas exploser linéairement
    """
    results = []
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        points = generate_random_points(size)
        start_time = time.time()
        kmeans_clustering(points, k=5)
        duration = time.time() - start_time
        results.append(duration)
        print(f"Size {size}: {duration:.4f}s")
    
    # Simple vérification que le temps pour 10000 n'est pas 100x plus long que pour 1000
    # K-means est souvent O(N), donc on s'attend à environ 10x
    ratio = results[2] / results[0]
    print(f"Ratio 10k/1k: {ratio:.2f}")
    
    # Marge large car le "setup time" peut fausser les petits datasets
    assert ratio < 20.0
