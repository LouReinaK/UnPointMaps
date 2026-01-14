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

def tdidf_mots_cluster(df: pd.DataFrame, cluster: list) -> Dict[str, float]:
    """
        Calcule le tfidf des mots dans un cluster donné.
    """
    texte = cluster_to_texte(df, cluster)
    
    if not texte or not texte.strip():
        return {}
    
    vectorizer = TfidfVectorizer(token_pattern=r'[^,]+', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform([texte])
    
    # Récupérer les noms des features (mots)
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_scores = {
        feature_names[i].strip(): tfidf_matrix[0, i]
        for i in range(len(feature_names))
    }
    
    return tfidf_scores



if __name__ == "__main__":
    # mots les plus utilisés dans la base de flickr data
    df = convert_to_dict_filtered()
    textes = df['tags'].dropna().tolist()

    # Combiner tous les textes en un seul
    texte_combine = " ".join(textes)
    resultats = calculer_pourcentage_mots(texte_combine)

    # Afficher les 10 mots les plus fréquents
    mots_tries = sorted(resultats.items(), key=lambda x: x[1], reverse=True)
    print("\nLes 10 mots les plus fréquents:")
    for mot, pourcentage in mots_tries[:10]:
        print(f"Mot: '{mot}' - Pourcentage: {pourcentage:.8f}%")
