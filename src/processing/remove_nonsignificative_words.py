# remove non-significant words from text data
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from .dataset_filtering import convert_to_dict_filtered
from .TFIDF import get_top_keywords
from langdetect import detect, DetectorFactory
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
        c = remove_frequent_words(c)
        if c and c.strip():
            cleaned.append(c)
    return cleaned


#Supprimer les mots fréquents dans le dataset qui ne sont pas des stopwords ou des mots significatifs
def remove_frequent_words(texts, threshold=0.5):
    frequent_words = get_top_keywords(texts, top_n=10)
    if pd.isna(texts) or texts == '':
        return texts
    words = texts.split()
    filtered_words = [word for word in words if word.lower() not in frequent_words]
    return ' '.join(filtered_words)


def remove_word_lyon(df):
    """Supprime le mot 'lyon' d'un dataset donné"""
    df = df.applymap(lambda x: x.replace('lyon', '') if isinstance(x, str) else x)
    return df
    """Supprime le mot 'lyon' d'un dataset donné"""


if __name__ == '__main__':
    df = convert_to_dict_filtered()
    remove_word_lyon(df)
    langues_detectees(df)
    df = remove_frequent_words(df)
