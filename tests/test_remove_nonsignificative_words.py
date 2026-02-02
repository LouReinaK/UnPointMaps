import unittest
from unittest.mock import MagicMock, patch
import sys
import pandas as pd

class TestRemoveNonsignificativeWords(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # We need to mock nltk before importing the module because it uses nltk at
        # top level
        cls.mock_nltk = MagicMock()
        cls.mock_stopwords = MagicMock()
        cls.mock_stopwords.words.return_value = ['le', 'la', 'the', 'a']
        cls.mock_nltk.corpus.stopwords = cls.mock_stopwords
        cls.mock_nltk.tokenize.word_tokenize.side_effect = lambda t, language=None: t.split()

        sys.modules['nltk'] = cls.mock_nltk
        sys.modules['nltk.corpus'] = cls.mock_nltk.corpus
        sys.modules['nltk.tokenize'] = cls.mock_nltk.tokenize
        
        # Now we can import the module
        global detect_language, remove_nonsignificant_words_multilang, clean_text_list, process_text_columns
        from src.processing.remove_nonsignificative_words import (
            detect_language,
            remove_nonsignificant_words_multilang,
            clean_text_list,
            process_text_columns
        )
        
    @classmethod
    def tearDownClass(cls):
        if 'nltk' in sys.modules:
            del sys.modules['nltk']
        if 'nltk.corpus' in sys.modules:
            del sys.modules['nltk.corpus']
        if 'nltk.tokenize' in sys.modules:
            del sys.modules['nltk.tokenize']
            
    def setUp(self):
        from src.processing.remove_nonsignificative_words import set_cache_enabled
        set_cache_enabled(False)

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


if __name__ == '__main__':
    unittest.main()
