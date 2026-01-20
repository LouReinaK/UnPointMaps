from collections import Counter
import pandas as pd
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from .dataset_filtering import convert_to_dict_filtered, get_info_geo

def cluster_to_texte(df: pd.DataFrame, cluster: list) -> str:
    """
        Concatène les tags des points du cluster en une chaine de caractères.
    """
    tags_list = []
    for p in cluster:
        # p est [latitude, longitude]
        info = get_info_geo(df, p[0], p[1])
        if info is not None and 'tags' in info and info['tags']:
            # Convertir en string et vérifier que ce n'est pas NaN
            tag = str(info['tags'])
            if tag and tag.lower() != 'nan':
                tags_list.append(tag)
    
    return ','.join(tags_list)

def calculer_pourcentage_mots(texte: str) -> Dict[str, float]:
    """
        Calcule le pourcentage d'apparition de chaque mot dans un texte donné.
    """
    if not texte or not texte.strip():
        return {}

    mots = texte.lower().split(',')
    compteur = Counter(mots)
    total_mots = len(mots)
    
    pourcentages = {
        mot: (count / total_mots) * 100 
        for mot, count in compteur.items()
    }
    
    return pourcentages

def tdidf_mots_clusters(df: pd.DataFrame, clusters: list) -> list:
    """
        Calcule le TF-IDF des mots de chaque cluster par rapport à tous les autres clusters.
    """
    # Convertir tous les clusters en textes
    textes_clusters = []
    for cluster in clusters:
        texte = cluster_to_texte(df, cluster)
        textes_clusters.append(texte if texte else "")
    
    # Vérifier que le cluster cible n'est pas vide
    if not any(textes_clusters) or all(not texte.strip() for texte in textes_clusters):
        return []
    
    # Calculer TF-IDF sur tous les clusters
    vectorizer = TfidfVectorizer(token_pattern=r'[^,]+', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(textes_clusters)
    
    # Récupérer les noms des features (mots)
    feature_names = vectorizer.get_feature_names_out()
    
    # Extraire les scores TF-IDF pour chaque cluster
    tfidf_scores_clusters = []
    for cluster_index in range(len(textes_clusters)):
        tfidf_scores = {
            feature_names[i].strip(): tfidf_matrix[cluster_index, i]
            for i in range(len(feature_names))
            if tfidf_matrix[cluster_index, i] > 0  # Ne garder que les mots présents dans ce cluster
        }
        tfidf_scores_clusters.append(tfidf_scores)
    
    return tfidf_scores_clusters

