"""
hybrid_retriever.py

Hybrid retrieval combining dense (FAISS) and sparse (BM25) search with RRF fusion.
Lean V2 implementation following leaner_rag_V2.md architecture.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import pickle

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

from ..models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval system combining FAISS (dense) + BM25 (sparse) with RRF fusion.

    Features:
    - Dense retrieval using FAISS HNSW index
    - Sparse retrieval using BM25
    - RRF (Reciprocal Rank Fusion) for result combination
    - Diversity deduplication
    - Persistent storage of indices
    """

    def __init__(self,
                 embedding_model,
                 reranker_model,
                 index_dir: str):
        """
        Initialize hybrid retriever.

        Args:
            embedding_model: SentenceTransformer model for dense embeddings
            reranker_model: CrossEncoder model for reranking
            index_dir: Directory to save/load indices
        """
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Retrieval indices
        self.dense_index = None
        self.sparse_index = None
        self.chunks = []
        self.chunk_lookup = {}  # chunk_id -> Chunk

        # BM25 preprocessing
        self.stemmer = PorterStemmer()
        self.tokenized_corpus = []

        # Try to load existing indices
        self._load_indices()

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Build dense (FAISS) and sparse (BM25) indices from chunks.

        Args:
            chunks: List of Chunk objects to index
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return

        logger.info(f"Building indices for {len(chunks)} chunks")

        self.chunks = chunks
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}

        # Build dense index (FAISS HNSW)
        self._build_dense_index(chunks)

        # Build sparse index (BM25)
        self._build_sparse_index(chunks)

        # Save indices to disk
        self._save_indices()

        logger.info(f"Indices built successfully: {len(chunks)} chunks indexed")

    def _build_dense_index(self, chunks: List[Chunk]) -> None:
        """Build FAISS HNSW index for dense retrieval."""
        logger.info("Building FAISS HNSW index...")

        # Collect embeddings
        embeddings = []
        for chunk in chunks:
            if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                embeddings.append(chunk.embedding)
            else:
                # Generate embedding if not present
                emb = self.embedding_model.encode(
                    chunk.text,
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
                chunk.embedding = emb
                embeddings.append(emb)

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity (inner product after normalization)
        faiss.normalize_L2(embeddings_array)

        # Build HNSW index
        dimension = embeddings_array.shape[1]
        self.dense_index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 connections
        self.dense_index.hnsw.efConstruction = 40
        self.dense_index.hnsw.efSearch = 64

        self.dense_index.add(embeddings_array)

        logger.info(f"FAISS index built: {self.dense_index.ntotal} vectors, {dimension}-dim")

    def _build_sparse_index(self, chunks: List[Chunk]) -> None:
        """Build BM25 index for sparse retrieval."""
        logger.info("Building BM25 index...")

        # Tokenize and preprocess corpus
        self.tokenized_corpus = []
        for chunk in chunks:
            tokens = self._preprocess_text(chunk.text)
            self.tokenized_corpus.append(tokens)

        # Build BM25 index
        self.sparse_index = BM25Okapi(
            self.tokenized_corpus,
            k1=1.2,  # Term frequency saturation
            b=0.75   # Length normalization
        )

        logger.info(f"BM25 index built: {len(self.tokenized_corpus)} documents")

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25.

        No stemming by default (preserves scientific terms like "protein").
        No stopword removal (better for scientific queries).
        """
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())

        # Filter out non-alphabetic tokens
        tokens = [t for t in tokens if t.isalpha() and len(t) > 1]

        return tokens

    def retrieve(self,
                query: str,
                top_k: int = 10,
                method: Optional[str] = None,
                diversity_threshold: float = 0.85) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return
            method: "dense", "sparse", or None for hybrid
            diversity_threshold: Cosine similarity threshold for deduplication

        Returns:
            List of RetrievalResult objects sorted by score
        """
        if not self.is_built():
            logger.error("Indices not built. Call build_index() first.")
            return []

        # Retrieve from both indices
        if method == "sparse":
            results = self._sparse_retrieve(query, top_k)
        elif method == "dense":
            results = self._dense_retrieve(query, top_k)
        else:
            # Hybrid: both methods + RRF fusion
            dense_results = self._dense_retrieve(query, top_k * 2)
            sparse_results = self._sparse_retrieve(query, top_k * 2)
            results = self._rrf_fusion(dense_results, sparse_results, top_k * 2)

        # Apply diversity deduplication
        if diversity_threshold < 1.0:
            results = self._apply_diversity_filtering(results, diversity_threshold)

        # Trim to top_k
        results = results[:top_k]

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def _dense_retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Dense retrieval using FAISS."""
        # Encode query
        query_emb = self.embedding_model.encode(
            query,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        query_emb = np.array([query_emb], dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_emb)

        # Search
        scores, indices = self.dense_index.search(query_emb, min(top_k, len(self.chunks)))

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    chunk=chunk,
                    score=float(score),
                    retrieval_method="dense",
                    rank=len(results) + 1
                ))

        return results

    def _sparse_retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Sparse retrieval using BM25."""
        # Preprocess query
        query_tokens = self._preprocess_text(query)

        # Get BM25 scores
        scores = self.sparse_index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:min(top_k, len(self.chunks))]

        # Build results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                chunk=chunk,
                score=float(scores[idx]),
                retrieval_method="sparse",
                rank=len(results) + 1
            ))

        return results

    def _rrf_fusion(self,
                   dense_results: List[RetrievalResult],
                   sparse_results: List[RetrievalResult],
                   top_k: int) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) for combining dense and sparse results.

        Formula: score(chunk) = Σ [1 / (k + rank)]
        where k=60 is the smoothing parameter.
        """
        k = 60  # RRF smoothing parameter

        # Collect all unique chunks
        chunk_scores = {}

        # Add dense scores
        for rank, result in enumerate(dense_results, 1):
            chunk_id = result.chunk_id
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {'chunk': result.chunk, 'score': 0.0}
            chunk_scores[chunk_id]['score'] += 1.0 / (k + rank)

        # Add sparse scores
        for rank, result in enumerate(sparse_results, 1):
            chunk_id = result.chunk_id
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {'chunk': result.chunk, 'score': 0.0}
            chunk_scores[chunk_id]['score'] += 1.0 / (k + rank)

        # Sort by RRF score
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        # Build results
        results = []
        for rank, (chunk_id, data) in enumerate(sorted_chunks[:top_k], 1):
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                chunk=data['chunk'],
                score=data['score'],
                retrieval_method="hybrid",
                rank=rank
            ))

        return results

    def _apply_diversity_filtering(self,
                                   results: List[RetrievalResult],
                                   threshold: float) -> List[RetrievalResult]:
        """
        Remove near-duplicate results based on text similarity.

        Args:
            results: List of retrieval results
            threshold: Cosine similarity threshold (0.85 = keep if < 85% similar)

        Returns:
            Filtered list of results
        """
        if len(results) <= 1:
            return results

        filtered = [results[0]]  # Always keep top result

        for result in results[1:]:
            # Check similarity with already selected results
            is_diverse = True
            for selected in filtered:
                similarity = self._embedding_similarity_between_chunks(result.chunk, selected.chunk)
                if similarity >= threshold:
                    is_diverse = False
                    break

            if is_diverse:
                filtered.append(result)

        return filtered

    def _embedding_similarity_between_chunks(self, chunk1: Chunk, chunk2: Chunk) -> float:
        """Calculate cosine similarity between two chunks using their stored embeddings.

        Falls back to on-the-fly encoding once if an embedding is missing, then caches it on the chunk.
        """
        # Ensure embeddings exist (fallback once if needed)
        if not hasattr(chunk1, 'embedding') or chunk1.embedding is None:
            chunk1.embedding = self.embedding_model.encode(
                chunk1.text, convert_to_tensor=False, show_progress_bar=False
            )
        if not hasattr(chunk2, 'embedding') or chunk2.embedding is None:
            chunk2.embedding = self.embedding_model.encode(
                chunk2.text, convert_to_tensor=False, show_progress_bar=False
            )

        emb1 = np.asarray(chunk1.embedding, dtype=np.float32)
        emb2 = np.asarray(chunk2.embedding, dtype=np.float32)

        # Cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def is_built(self) -> bool:
        """Check if indices are built."""
        return (self.dense_index is not None and
                self.sparse_index is not None and
                len(self.chunks) > 0)

    def clear_index(self) -> None:
        """Clear all indices and cached data."""
        self.dense_index = None
        self.sparse_index = None
        self.chunks = []
        self.chunk_lookup = {}
        self.tokenized_corpus = []

        # Delete saved files
        for file in ['chunks.json', 'faiss.index', 'bm25.pkl']:
            file_path = self.index_dir / file
            if file_path.exists():
                file_path.unlink()

        logger.info("Index cleared")

    def _save_indices(self) -> None:
        """Save indices to disk."""
        logger.info(f"Saving indices to {self.index_dir}")

        # Save chunks
        chunks_data = []
        for chunk in self.chunks:
            chunk_dict = {
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'text': chunk.text,
                'source_path': chunk.source_path,
                'token_count': chunk.token_count,
                'section': chunk.section,
                'page_number': chunk.page_number,
                'embedding': chunk.embedding.tolist() if hasattr(chunk, 'embedding') else None
            }
            chunks_data.append(chunk_dict)

        with open(self.index_dir / 'chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f)

        # Save FAISS index
        faiss.write_index(self.dense_index, str(self.index_dir / 'faiss.index'))

        # Save BM25 index
        bm25_data = {
            'tokenized_corpus': self.tokenized_corpus,
            'doc_len': self.sparse_index.doc_len,
            'avgdl': self.sparse_index.avgdl,
            'idf': self.sparse_index.idf
        }
        with open(self.index_dir / 'bm25.pkl', 'wb') as f:
            pickle.dump(bm25_data, f)

        logger.info("Indices saved successfully")

    def _load_indices(self) -> None:
        """Load indices from disk if they exist."""
        chunks_file = self.index_dir / 'chunks.json'
        faiss_file = self.index_dir / 'faiss.index'
        bm25_file = self.index_dir / 'bm25.pkl'

        if not all([chunks_file.exists(), faiss_file.exists(), bm25_file.exists()]):
            logger.debug("No existing indices found")
            return

        try:
            logger.info(f"Loading indices from {self.index_dir}")

            # Load chunks
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            self.chunks = []
            for chunk_dict in chunks_data:
                chunk = Chunk(
                    chunk_id=chunk_dict['chunk_id'],
                    doc_id=chunk_dict['doc_id'],
                    text=chunk_dict['text'],
                    source_path=chunk_dict['source_path'],
                    token_count=chunk_dict['token_count'],
                    section=chunk_dict.get('section'),
                    page_number=chunk_dict.get('page_number')
                )
                if chunk_dict.get('embedding'):
                    chunk.embedding = np.array(chunk_dict['embedding'], dtype=np.float32)
                self.chunks.append(chunk)

            self.chunk_lookup = {chunk.chunk_id: chunk for chunk in self.chunks}

            # Load FAISS index
            self.dense_index = faiss.read_index(str(faiss_file))

            # Load BM25 index
            with open(bm25_file, 'rb') as f:
                bm25_data = pickle.load(f)

            self.tokenized_corpus = bm25_data['tokenized_corpus']
            self.sparse_index = BM25Okapi(self.tokenized_corpus)

            logger.info(f"Indices loaded successfully: {len(self.chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            self.dense_index = None
            self.sparse_index = None
            self.chunks = []
            self.chunk_lookup = {}
