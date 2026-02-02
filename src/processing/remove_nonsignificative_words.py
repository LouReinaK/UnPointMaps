# remove non-significant words from text data
import pandas as pd
import os
import logging
from multiprocessing import Pool
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    stopwords = None
    word_tokenize = None
from .TFIDF import get_top_keywords
try:
    from langdetect import detect, DetectorFactory
except ImportError:
    detect = None
    DetectorFactory = None
from collections import Counter
from functools import lru_cache

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Lazy database manager instance
_DB_MANAGER = None
_CACHE_ENABLED = True

def set_cache_enabled(enabled):
    """Globally enable or disable database caching for word removal."""
    global _CACHE_ENABLED
    _CACHE_ENABLED = enabled

def get_db_manager():
    """Returns the database manager instance, initializing it if necessary."""
    global _DB_MANAGER
    if not _CACHE_ENABLED:
        return None
    if _DB_MANAGER is None:
        try:
            from ..database.manager import DatabaseManager
            # Use the default path "unpointmaps_cache.db" which is the main project DB
            _DB_MANAGER = DatabaseManager()
        except Exception:
            return None
    return _DB_MANAGER

# Pour avoir des résultats reproductibles
if DetectorFactory:
    DetectorFactory.seed = 0

# Dictionnaire des stopwords pour différentes langues
try:
    STOPWORDS_DICT = {
        'french': set(stopwords.words('french')),
        'english': set(stopwords.words('english')),
        'spanish': set(stopwords.words('spanish')),
        'german': set(stopwords.words('german')),
        'italian': set(stopwords.words('italian')),
        'portuguese': set(stopwords.words('portuguese')),
        'russian': set(stopwords.words('russian')),
        'dutch': set(stopwords.words('dutch')),
    } if stopwords else {}
except Exception:
    STOPWORDS_DICT = {}

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


@lru_cache(maxsize=10000)
def detect_language(text):
    """Détecte la langue d'un texte"""
    if detect is None:
        return 'en'
    try:
        if pd.isna(text) or text == '':
            return 'en'
        lang = detect(str(text))
        return lang
    except Exception:
        return 'en'  # Par défaut, anglais


@lru_cache(maxsize=10000)
def _remove_nonsignificant_words_multilang_internal(text):
    """Internal function for cleaning a single string, with LRU cache."""
    try:
        if pd.isna(text) or text == '':
            return ''

        if word_tokenize is None:
            return text

        text_str = str(text)
        lang = detect_language(text_str)
        stop_words = get_stopwords_for_language(lang)

        # Tokenize the text
        lang_name = LANGUAGE_MAPPING.get(lang, 'french')
        words = word_tokenize(text_str, language=lang_name)
        # Remove stop words
        filtered_words = [
            word for word in words if word.lower() not in stop_words]
        # Reconstruct the text
        return ' '.join(filtered_words)
    except Exception:
        return text


def remove_nonsignificant_words_multilang(text):
    """Supprime les stopwords basés sur la langue détectée (avec cache LRU)"""
    return _remove_nonsignificant_words_multilang_internal(text)


def clean_texts_batched(texts):
    """
    Cleans a list of texts using batch DB lookups and LRU cache for maximum speed.
    """
    if not texts:
        return []

    # Map inputs to unique non-empty strings
    original_texts = [str(t) if not pd.isna(t) else "" for t in texts]
    unique_texts = list(set(t for t in original_texts if t.strip()))
    
    if not unique_texts:
        return original_texts

    logger.info(f"Cleaning batch of {len(original_texts)} texts ({len(unique_texts)} unique non-empty)")

    db = get_db_manager()
    cached_results = {}
    
    # 1. Try DB cache first
    if db:
        try:
            cached_results = db.get_cleaned_text_batch(unique_texts)
            if cached_results:
                logger.info(f"Cache hit: {len(cached_results)}/{len(unique_texts)} texts found in database")
        except Exception as e:
            logger.debug(f"DB cache read error: {e}")
            cached_results = {}
    
    # 2. Process what's missing
    to_process = [t for t in unique_texts if t not in cached_results]
    new_results = {}
    
    if to_process:
        # Use multiprocessing for large batches to speed up CPU-bound cleaning/detection
        if len(to_process) > 50:
            logger.info(f"Processing {len(to_process)} new texts using multiprocessing...")
            # Limit workers to avoid excessive overhead or memory usage
            num_workers = os.cpu_count() or 1
            try:
                with Pool(processes=num_workers) as pool:
                    clean_results_list = pool.map(_remove_nonsignificant_words_multilang_internal, to_process)
                for t, cleaned in zip(to_process, clean_results_list):
                    new_results[t] = cleaned
                    cached_results[t] = cleaned
                logger.info("Multiprocessing batch complete.")
            except Exception as e:
                logger.warning(f"Multiprocessing failed, falling back to sequential: {e}")
                # Fallback to sequential if multiprocessing fails
                for t in to_process:
                    cleaned = _remove_nonsignificant_words_multilang_internal(t)
                    new_results[t] = cleaned
                    cached_results[t] = cleaned
        else:
            if len(to_process) > 0:
                logger.debug(f"Processing {len(to_process)} texts sequentially")
            for t in to_process:
                cleaned = _remove_nonsignificant_words_multilang_internal(t)
                new_results[t] = cleaned
                cached_results[t] = cleaned
        
    # 3. Save new results to DB
    if db and new_results:
        try:
            db.save_cleaned_text_batch(new_results)
            logger.debug(f"Saved {len(new_results)} new cleaned texts to database")
        except Exception as e:
            logger.debug(f"DB cache write error: {e}")
            
    # Reconstruct the list in original order
    return [cached_results.get(t, t) if t else "" for t in original_texts]


def process_text_columns(df, text_columns):
    """Traite les colonnes de texte en supprimant les stopwords multilingues (version optimisée)"""
    for col in text_columns:
        if col in df.columns:
            texts = df[col].astype(str).tolist()
            df[col] = clean_texts_batched(texts)
    return df


def langues_detectees(df, n=5000):
    """Affiche les langues détectées dans le dataset des 5000 premières lignes et les affiche en pourcentage"""
    all_texts = [str(t) for t in (df['title'].tolist() + df['tags'].tolist())[:n] if t and not pd.isna(t)]
    
    if len(all_texts) > 100:
        logger.info(f"Detecting languages for {len(all_texts)} items using multiprocessing...")
        num_workers = min(os.cpu_count() or 1, 8)
        try:
            with Pool(processes=num_workers) as pool:
                detected_languages = pool.map(detect_language, all_texts)
        except Exception as e:
            logger.warning(f"Multiprocessing for language detection failed: {e}")
            detected_languages = [detect_language(text) for text in all_texts]
    else:
        detected_languages = [detect_language(text) for text in all_texts]
        
    language_counts = Counter(detected_languages)
    total = sum(language_counts.values())
    print("Langues détectées dans le dataset:")
    for lang, count in language_counts.most_common():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{lang}: ({percentage:.2f}%)")


def clean_text_list(texts):
    """
    Takes a list of strings, cleans them using batch processing,
    and returns the cleaned list.
    """
    if not texts:
        return []
    cleaned_full = clean_texts_batched(texts)
    # Filter out empty results as original behavior did
    return [c for c in cleaned_full if c and c.strip()]


#Supprimer les mots fréquents dans le dataset qui ne sont pas des stopwords ou des mots significatifs
def clean_df_words(df, threshold=0.5):
    # Collect all texts to clean in one batch
    texts_to_clean = []
    if 'title' in df.columns:
        texts_to_clean.extend(df['title'].dropna().astype(str).tolist())
    if 'tags' in df.columns:
        texts_to_clean.extend(df['tags'].dropna().astype(str).tolist())
    
    # Process everything in one batch
    cleaned_texts = clean_texts_batched(texts_to_clean)
    
    frequent_words = get_top_keywords(cleaned_texts, top_n=3)
    # Supprimer les mots fréquents du dataset
    df = df.map(lambda x: ' '.join(
        word for word in str(x).split() if word.lower() not in frequent_words) if isinstance(x, str) else x)
    return df


def remove_word_lyon(df):
    """Supprime le mot 'lyon' d'un dataset donné (insensible à la casse)"""
    df = df.map(lambda x: x.replace('lyon', '').replace('Lyon', '') if isinstance(x, str) else x)
    return df



