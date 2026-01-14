from collections import Counter
from typing import Dict


def calculer_pourcentage_mots(texte: str) -> Dict[str, float]:
    """
        Calcule le pourcentage d'apparition de chaque mot dans un texte donné.
    """
    if not texte or not texte.strip():
        return {}

    mots = texte.lower().split()
    compteur = Counter(mots)
    total_mots = len(mots)
    
    pourcentages = {
        mot: (count / total_mots) * 100 
        for mot, count in compteur.items()
    }
    
    return pourcentages

if __name__ == "__main__":
    texte_exemple = "Ceci est un exemple de texte. Ceci est un test. Lyon est une ville."
    resultats = calculer_pourcentage_mots(texte_exemple)
    for mot, pourcentage in resultats.items():
        print(f"Mot: '{mot}' - Pourcentage: {pourcentage:.8f}%")
