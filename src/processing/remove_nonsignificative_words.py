# remove non-significant words from text data
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from .dataset_filtering import convert_to_dict_filtered
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


def langues_detectees(df):
    """Affiche les langues détectées dans le dataset"""
    all_texts = df['title'].tolist() + df['description'].tolist()
    detected_languages = [detect_language(text) for text in all_texts]
    language_counts = Counter(detected_languages)
    print("Langues détectées dans le dataset:")
    for lang, count in language_counts.most_common():
        print(f"{lang}: {count} occurrences")


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


if __name__ == '__main__':
    df = convert_to_dict_filtered()
    langues_detectees(df)
