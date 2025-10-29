"""
embedding_manager.py

SHA256-based embedding cache for fast incremental updates.
"""

import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embeddings with SHA256-based caching.

    Provides 50-1000x speedup on re-indexing by caching embeddings
    based on text content hash.
    """

    def __init__(self, embedding_model: SentenceTransformer, cache_dir: str):
        """
        Initialize embedding manager.

        Args:
            embedding_model: SentenceTransformer model for generating embeddings
            cache_dir: Directory for storing cache file
        """
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / "embedding_cache.pkl"
        self.cache: Dict[str, np.ndarray] = {}

        # Statistics
        self.hits = 0
        self.misses = 0

        # Load existing cache
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk if it exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded embedding cache: {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.debug(f"Saved embedding cache: {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _get_text_hash(self, text: str) -> str:
        """
        Calculate SHA256 hash of text.

        Args:
            text: Text to hash

        Returns:
            Hexadecimal SHA256 hash
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text, using cache if available.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        text_hash = self._get_text_hash(text)

        # Check cache
        if text_hash in self.cache:
            self.hits += 1
            logger.debug(f"Cache hit for text hash: {text_hash[:16]}...")
            return self.cache[text_hash]

        # Generate embedding
        self.misses += 1
        logger.debug(f"Cache miss for text hash: {text_hash[:16]}...")

        embedding = self.embedding_model.encode(
            text,
            convert_to_tensor=False,
            show_progress_bar=False
        )

        # Store in cache
        self.cache[text_hash] = embedding

        # Periodically save cache (every 100 misses)
        if self.misses % 100 == 0:
            self._save_cache()

        return embedding

    def get_embeddings_batch(self, texts: list) -> np.ndarray:
        """
        Get embeddings for multiple texts, using cache where possible.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self.cache:
                embeddings.append(self.cache[text_hash])
                self.hits += 1
            else:
                embeddings.append(None)
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                self.misses += 1

        # Batch embed uncached texts
        if texts_to_embed:
            logger.debug(f"Batch embedding {len(texts_to_embed)} uncached texts")
            new_embeddings = self.embedding_model.encode(
                texts_to_embed,
                convert_to_tensor=False,
                show_progress_bar=len(texts_to_embed) > 10
            )

            # Store in cache and fill in embeddings list
            for i, text in enumerate(texts_to_embed):
                text_hash = self._get_text_hash(text)
                self.cache[text_hash] = new_embeddings[i]
                embeddings[indices_to_embed[i]] = new_embeddings[i]

            # Save cache after batch
            self._save_cache()

        return np.array(embeddings)

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, and size
        """
        return {
            'hits': self.hits,
            'misses': self.misses,
            'size': len(self.cache)
        }

    def clear_cache(self):
        """Clear all cached embeddings."""
        self.cache = {}
        self.hits = 0
        self.misses = 0
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Embedding cache cleared")
