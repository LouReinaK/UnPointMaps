
import sys
from unittest.mock import MagicMock, patch, Mock
import numpy as np
import pytest
from src.processing.embedding_service import EmbeddingService

# Mock nltk to avoid import errors
sys.modules['nltk'] = Mock()
sys.modules['nltk'].__spec__ = Mock()

# Mock importlib.util.find_spec to avoid nltk spec error
import importlib.util
original_find_spec = importlib.util.find_spec
def mock_find_spec(name, package=None):
    if name == 'nltk':
        return None
    else:
        return original_find_spec(name, package)
importlib.util.find_spec = mock_find_spec


class MockTensor:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def test_select_representative_indices_correct_ranking():
    service = EmbeddingService.get_instance()

    # Mock the model
    service.model = MagicMock()
    # Mock encode to return numpy array of embeddings
    # 3 items, dimension 2
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    service.model.encode.return_value = embeddings

    # Mock util.cos_sim - note: util is imported from sentence_transformers inside the method
    with patch('sentence_transformers.util.cos_sim') as mock_cos_sim:
        # Create a mock tensor with numpy method
        mock_tensor = MagicMock()
        mock_tensor.__getitem__.return_value = MockTensor([0.1, 0.2, 0.3])
        mock_cos_sim.return_value = mock_tensor

        texts = ["a", "b", "c"]
        indices = service.select_representative_indices(texts, top_k=2)

        # If the bug exists and was swallowed, it would return [0, 1] (range) due to exception handling
        # But we want to fail if it catches exception.
        # Ideally we want to assert that NO exception was logged/caught?
        # The existing code catches Exception so we can't easily see it fail unless we spy on logger or check return value correctness?
        # But the fallback returns range(N). If logic works, it returns sorted indices.
        # With scores [0.1, 0.2, 0.3], sorted ascending is [0, 1, 2] (indices of 0.1, 0.2, 0.3)
        # argsort gives [0, 1, 2]. [::-1] gives [2, 1, 0]. top 2 gives [2, 1].
        # If exception occurs, it returns [0, 1].

        assert indices == [2, 1]


def test_embedding_service_singleton():
    s1 = EmbeddingService.get_instance()
    s2 = EmbeddingService.get_instance()
    assert s1 is s2


def test_encode_no_model():
    service = EmbeddingService.get_instance()
    service.model = None
    assert service.encode(["test"]) is None


def test_select_representative_indices_empty():
    service = EmbeddingService.get_instance()
    assert service.select_representative_indices([], top_k=5) == []


def test_load_model_success():
    service = EmbeddingService.get_instance()
    # Reset model
    service.model = None

    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        service.load_model('test-model')
        mock_st.assert_called_with('test-model')
        assert service.model is not None


def test_load_model_failure():
    service = EmbeddingService.get_instance()
    service.model = None

    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_st.side_effect = Exception("Download failed")
        service.load_model('test-model')
        assert service.model is None


def test_encode_success():
    service = EmbeddingService.get_instance()
    service.model = MagicMock()
    service.model.encode.return_value = np.array([[1, 2]])

    res = service.encode(["hello"])
    assert isinstance(res, np.ndarray)
    assert res.shape == (1, 2)
