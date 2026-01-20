from src.processing.TFIDF import get_top_keywords


class TestTFIDF:
    def test_get_top_keywords(self):
        texts = [
            "apple banana apple",
            "banana orange",
            "apple orange pear"
        ]

        # Test basic functionality
        keywords = get_top_keywords(texts, top_n=5)
        # We expect a list of tuples (word, score)
        assert len(keywords) > 0
        assert isinstance(keywords[0], tuple)
        # Check that common words are there
        words = [k[0] for k in keywords]
        assert "apple" in words
        assert "banana" in words

    def test_get_top_keywords_empty(self):
        assert get_top_keywords([]) == []
        assert get_top_keywords([""]) == []

    def test_get_top_keywords_single_doc(self):
        texts = ["hello world"]
        keywords = get_top_keywords(texts)
        assert len(keywords) > 0
        words = [k[0] for k in keywords]
        assert "hello" in words
        assert "world" in words
