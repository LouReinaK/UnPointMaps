import logging
import numpy as np
from typing import List
from src.database.manager import DatabaseManager

logger = logging.getLogger("EmbeddingService")


class EmbeddingService:
    _instance = None
    model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_name: str = 'all-MiniLM-L6-v2'):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info("SentenceTransformer model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                self.model = None

    def get_embedding_model(self):
        return self.model

    def encode(self, texts: List[str]):
        if not self.model:
            return None

        if not texts:
            return np.array([])

        try:
            db_manager = DatabaseManager()
            cached_embeddings = db_manager.get_cached_embeddings(texts)

            texts_to_compute = [t for t in texts if t not in cached_embeddings]

            if texts_to_compute:
                # logger.info(f"Computing embeddings for {len(texts_to_compute)} new texts")
                new_embeddings = self.model.encode(texts_to_compute)

                # Save to cache
                new_cache_data = {}
                for t, emb in zip(texts_to_compute, new_embeddings):
                    new_cache_data[t] = emb
                    cached_embeddings[t] = emb

                db_manager.save_cached_embeddings(new_cache_data)

            # Reconstruct result array in correct order
            # Get dimension from one embedding
            if not cached_embeddings:
                return np.array([])

            first_emb = next(iter(cached_embeddings.values()))
            dim = first_emb.shape[0]

            result = np.zeros((len(texts), dim), dtype=np.float32)

            for i, text in enumerate(texts):
                if text in cached_embeddings:
                    result[i] = cached_embeddings[text]
                else:
                    # Should not happen
                    pass

            return result
        except Exception as e:
            logger.error(
                f"Error in cached encoding: {e}. Fallback to direct encoding.")
            return self.model.encode(texts)

    def select_representative_indices(
            self,
            texts: List[str],
            top_k: int = 50) -> List[int]:
        """
        Returns the indices of the texts closest to the semantic centroid.
        """
        if not self.model or not texts:
            # Fallback: just return first N indices if model failure
            return list(range(min(len(texts), top_k)))

        try:
            from sentence_transformers import util
            # Use self.encode to benefit from caching
            embeddings = self.encode(texts)

            centroid = np.mean(embeddings, axis=0)

            # Compute cosine similarities
            # util.cos_sim returns a tensor/array of shape (1, N)
            cos_scores = util.cos_sim(centroid, embeddings)[0]

            # Ensure cos_scores is always a numpy array
            cos_scores_np: np.ndarray
            if hasattr(cos_scores, 'cpu'):
                cos_scores_np = cos_scores.cpu().numpy()
            else:
                cos_scores_np = np.array(cos_scores)

            # Sort descending
            top_indices = np.argsort(cos_scores_np)[::-1][:top_k]
            return top_indices.tolist()
        except Exception as e:
            logger.error(f"Error selecting representative indices: {e}")
            return list(range(min(len(texts), top_k)))
