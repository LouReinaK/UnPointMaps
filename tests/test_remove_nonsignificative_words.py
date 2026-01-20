
import unittest
from unittest.mock import MagicMock, patch
import sys
import pandas as pd

# We need to mock nltk before importing the module because it uses nltk at
# top level
mock_nltk = MagicMock()
mock_stopwords = MagicMock()
mock_stopwords.words.return_value = ['le', 'la', 'the', 'a']
mock_nltk.corpus.stopwords = mock_stopwords
mock_nltk.tokenize.word_tokenize.side_effect = lambda t, language=None: t.split()

sys.modules['nltk'] = mock_nltk
sys.modules['nltk.corpus'] = mock_nltk.corpus
sys.modules['nltk.tokenize'] = mock_nltk.tokenize

# Now we can import the module
from src.processing.remove_nonsignificative_words import (
    detect_language,
    remove_nonsignificant_words_multilang,
    clean_text_list,
    process_text_columns
)


class TestRemoveNonsignificativeWords(unittest.TestCase):
    def test_detect_language(self):
        # We need to patch langdetect inside the function or usage
        # But wait, detect is imported in the module.
        # So we need to patch likely where it is used or patch the imported
        # symbol
        with patch('src.processing.remove_nonsignificative_words.detect') as mock_detect:
            mock_detect.return_value = 'fr'
            self.assertEqual(detect_language("Bonjour"), 'fr')

            mock_detect.side_effect = Exception("error")
            self.assertEqual(detect_language("err"), 'en')  # Fallback

    def test_remove_nonsignificant_words_multilang(self):
        with patch('src.processing.remove_nonsignificative_words.detect') as mock_detect:
            mock_detect.return_value = 'fr'
            # Our mock stopwords returns ['le', 'la', 'the', 'a']
            # Our mock tokenize uses split()

            text = "le chat mange la souris"
            cleaned = remove_nonsignificant_words_multilang(text)
            
            # Debug
            print("=== Debug info ===")
            print(f"Text: {text}")
            print(f"Cleaned: {cleaned}")
            
            self.assertEqual(cleaned, "chat mange souris")

    def test_clean_text_list(self):
        with patch('src.processing.remove_nonsignificative_words.detect') as mock_detect:
            mock_detect.return_value = 'en'
            texts = ["the cat", "a dog"]
            cleaned = clean_text_list(texts)
            self.assertEqual(cleaned, ["cat", "dog"])

    def test_process_text_columns(self):
        with patch('src.processing.remove_nonsignificative_words.detect') as mock_detect:
            mock_detect.return_value = 'en'
            df = pd.DataFrame({'text': ['the cat', 'a dog']})
            df_processed = process_text_columns(df, ['text'])
            self.assertEqual(df_processed['text'].tolist(), ["cat", "dog"])


if __name__ == '__main__':
    unittest.main()
