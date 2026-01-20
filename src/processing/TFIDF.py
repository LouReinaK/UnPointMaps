from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer


def get_top_keywords(texts: List[str], top_n: int = 10,
                     stop_words: str = 'english') -> List[tuple[str, float]]:
    """
    Extracts top keywords from a list of texts using TF-IDF logic.
    Since we only have one cluster's text here, it behaves like Term Frequency
    but with stopword removal and tokenization power of TfidfVectorizer.
    """
    if not texts:
        return []

    try:
        # Combine all texts into one document to find most frequent words in
        # this cluster
        combined_text = " ".join(texts)
        if not combined_text.strip():
            return []

        # We interpret 'texts' as a collection ensuring we can filter stopwords
        # If we pass a list of 1 element, idf is constant.
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=top_n)
        matrix = vectorizer.fit_transform([combined_text])

        feature_names = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]

        # Sort by score
        indices = scores.argsort()[::-1]
        top_keywords = [(feature_names[i], float(scores[i]))
                        for i in indices[:top_n]]

        return top_keywords
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []


def tdidf_mots_clusters(cluster_texts: List[str]) -> List[Dict[str, float]]:
    """
    Calcule le TF-IDF des mots de chaque cluster par rapport à tous les autres clusters.
    Input: List of strings (one string per cluster, containing all its text)
    """
    if not cluster_texts or all(not t.strip() for t in cluster_texts):
        return []

    try:
        # Calculer TF-IDF sur tous les clusters
        # Use 'english' as a safe default or make it configurable
        vectorizer = TfidfVectorizer(
            token_pattern=r'(?u)\b\w\w+\b',
            lowercase=True,
            stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)

        # Récupérer les noms des features (mots)
        feature_names = vectorizer.get_feature_names_out()

        # Extraire les scores TF-IDF pour chaque cluster
        tfidf_scores_clusters = []
        for cluster_index in range(len(cluster_texts)):
            tfidf_scores = {
                feature_names[i].strip(): float(tfidf_matrix[cluster_index, i])
                for i in range(len(feature_names))
                # Ne garder que les mots importants (> 0.05 par exemple)
                if tfidf_matrix[cluster_index, i] > 0.05
            }
            # Sort dict by value descending
            sorted_scores = dict(
                sorted(
                    tfidf_scores.items(),
                    key=lambda item: item[1],
                    reverse=True)[
                    :10])
            tfidf_scores_clusters.append(sorted_scores)

        return tfidf_scores_clusters
    except Exception as e:
        print(f"TFIDF Error: {e}")
        return []
