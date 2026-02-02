# remove non-significant words from text data
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from .TFIDF import get_top_keywords
from langdetect import detect, DetectorFactory
import re
from collections import Counter

# Pour avoir des résultats reproductibles
DetectorFactory.seed = 0

# Dictionnaire des stopwords pour différentes langues
STOPWORDS_DICT = {
    'french': set(stopwords.words('french')),
    'english': set(stopwords.words('english')),
    'spanish': set(stopwords.words('spanish')),
    'german': set(stopwords.words('german')),
    'italian': set(stopwords.words('italian')),
    'portuguese': set(stopwords.words('portuguese')),
    'russian': set(stopwords.words('russian')),
    'dutch': set(stopwords.words('dutch')),
}

# Mapping langdetect codes to NLTK language names
LANGUAGE_MAPPING = {
    'fr': 'french',
    'en': 'english',
    'es': 'spanish',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'ru': 'russian',
    'nl': 'dutch',
}


def get_stopwords_for_language(lang_code):
    """Retourne les stopwords pour une langue donnée"""
    lang_name = LANGUAGE_MAPPING.get(lang_code, 'english')
    return STOPWORDS_DICT.get(lang_name, STOPWORDS_DICT['english'])


def detect_language(text):
    """Détecte la langue d'un texte"""
    try:
        if pd.isna(text) or text == '':
            return 'en'
        lang = detect(str(text))
        return lang
    except Exception:
        return 'en'  # Par défaut, anglais


def remove_nonsignificant_words_multilang(text):
    """Supprime les stopwords basés sur la langue détectée"""
    try:
        if pd.isna(text) or text == '':
            return ''

        text_str = str(text)
        lang = detect_language(text_str)
        stop_words = get_stopwords_for_language(lang)

        # Tokenize the text
        # Tokenization par défaut
        # Get language name for word_tokenize
        lang_name = LANGUAGE_MAPPING.get(lang, 'french')
        words = word_tokenize(text_str, language=lang_name)
        # Remove stop words
        filtered_words = [
            word for word in words if word.lower() not in stop_words]
        # Reconstruct the text
        return ' '.join(filtered_words)
    except Exception:
        return text


def process_text_columns(df, text_columns):
    """Traite les colonnes de texte en supprimant les stopwords multilingues"""
    for col in text_columns:
        df[col] = df[col].astype(str).apply(
            remove_nonsignificant_words_multilang)
    return df


def langues_detectees(df, n=5000):
    """Affiche les langues détectées dans le dataset des 5000 premières lignes et les affiche en pourcentage"""
    all_texts = df['title'].tolist() + df['tags'].tolist()
    detected_languages = [detect_language(text) for text in all_texts[:n]]
    language_counts = Counter(detected_languages)
    total = sum(language_counts.values())
    print("Langues détectées dans le dataset:")
    for lang, count in language_counts.most_common():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{lang}: ({percentage:.2f}%)")


def clean_text_list(texts):
    """
    Takes a list of strings, cleans them using remove_nonsignificant_words_multilang,
    and returns the cleaned list.
    """
    cleaned = []
    for t in texts:
        c = remove_nonsignificant_words_multilang(t)
        if c and c.strip():
            cleaned.append(c)
    return cleaned


#Supprimer les mots fréquents dans le dataset qui ne sont pas des stopwords ou des mots significatifs
def clean_df_words(df, threshold=0.5):
    texts = (
        df['title'].dropna().astype(str).tolist()
        + df['tags'].dropna().astype(str).tolist()
    )
    texts=remove_nonsignificant_words_multilang(texts)
    frequent_words = get_top_keywords(texts, top_n=3)
    # Supprimer les mots fréquents du dataset
    df = df.map(lambda x: ' '.join(
        word for word in str(x).split() if word.lower() not in frequent_words) if isinstance(x, str) else x)
    return df


def get_most_frequent_non_stopwords(df, top_n=7, min_freq=5):
    """Retourne les mots les plus fréquents dans title+tags qui ne sont pas des stopwords.
    Méthode optimisée : utilise l'union des stopwords disponibles et un tokeniseur regex léger
    pour éviter la détection de langue coûteuse sur chaque texte.
    """
    texts = []
    if 'title' in df.columns:
        texts += df['title'].dropna().astype(str).tolist()
    if 'tags' in df.columns:
        texts += df['tags'].dropna().astype(str).tolist()

    if not texts:
        return []

    # Union de tous les stopwords connus (fr/en/...) pour un nettoyage global rapide
    stop_union = set()
    for s in STOPWORDS_DICT.values():
        stop_union.update(s)
    # NOTE: n'ajouter aucun token projet-spécifique ici (ex: 'lyon') afin que 'lyon' puisse
    # être détecté parmi les mots fréquents et éventuellement supprimé.

    words = []
    token_pattern = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        # lower once
        t_lower = t.lower()
        for m in token_pattern.findall(t_lower):
            # skip pure numbers and very short tokens
            if m.isdigit() or len(m) < 2:
                continue
            if m in stop_union:
                continue
            words.append(m)

    if not words:
        return []

    counts = Counter(words)
    frequent = [w for w, c in counts.most_common(top_n * 2) if c >= min_freq]
    return frequent[:top_n]


def remove_frequent_words_from_df(df, words_to_remove):
    """Supprime mots donnés (liste) des colonnes title et tags du dataframe et renvoie une copie."""
    df = df.copy()

    def _remove_from_text(text):
        if not isinstance(text, str):
            return text
        # remplace mot complet, insensible à la casse
        pattern = r"\b(" + "|".join(re.escape(w) for w in words_to_remove) + r")\b"
        return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    if 'title' in df.columns:
        df['title'] = df['title'].astype(str).apply(_remove_from_text)
    if 'tags' in df.columns:
        df['tags'] = df['tags'].astype(str).apply(_remove_from_text)

    return df


def remove_common_words_pipeline(df, top_n=10, min_freq=10):
    """Pipeline : détecte mots fréquents non-stopwords et les supprime.
    Retourne (df_cleaned, words_removed)
    """
    frequent = get_most_frequent_non_stopwords(df, top_n=top_n, min_freq=min_freq)
    if not frequent:
        return df.copy(), []
    df_clean = remove_frequent_words_from_df(df, frequent)
    return df_clean, frequent



