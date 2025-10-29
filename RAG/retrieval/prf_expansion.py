"""
prf_expansion.py

Pseudo-Relevance Feedback (PRF) style query expansion using embedding similarity.
Expands queries using terms from top BM25 results, avoiding dependency on external
vocabularies like WordNet.
"""

import logging
from typing import List, Set
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class PRFQueryExpander:
    """
    PRF-style query expansion using embedding similarity on top BM25 hits.

    This approach is superior to WordNet for scientific domains because:
    - Domain-aware: learns from actual corpus vocabulary (genes, assays, methods)
    - Deterministic: no external dependencies, works offline
    - Grounded: expansion terms come from relevant documents
    - Fast: only 3 documents analyzed, 2 terms added
    """

    def __init__(self, embedding_model: SentenceTransformer,
                 max_expansions: int = 2,
                 top_docs: int = 3,
                 min_term_length: int = 4,
                 stopwords: Set[str] = None):
        """
        Initialize PRF query expander.

        Args:
            embedding_model: Sentence transformer for term embeddings
            max_expansions: Maximum expansion terms to add (default: 2)
            top_docs: Number of top BM25 documents to analyze (default: 3)
            min_term_length: Minimum length for candidate terms (default: 4)
            stopwords: Set of stopwords to filter out
        """
        self.embedding_model = embedding_model
        self.max_expansions = max_expansions
        self.top_docs = top_docs
        self.min_term_length = min_term_length

        # Basic stopwords if none provided
        self.stopwords = stopwords or {
            'the', 'is', 'at', 'which', 'on', 'are', 'was', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'what', 'when', 'where', 'who', 'why', 'how'
        }

    def expand_query(self, query: str, top_bm25_chunks: List) -> str:
        """
        Expand query using terms from top BM25 results.

        Process:
        1. Extract candidate terms from top-3 BM25 hits
        2. Embed query and candidate terms
        3. Rank terms by embedding similarity to query
        4. Add top-N most similar terms to query

        Args:
            query: Original query string
            top_bm25_chunks: List of Chunk objects from BM25 retrieval (top results)

        Returns:
            Expanded query string with original query + expansion terms
        """
        if not top_bm25_chunks or self.max_expansions == 0:
            return query

        # Extract query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)

        # Extract candidate terms from top documents
        candidate_terms = self._extract_candidate_terms(
            query,
            top_bm25_chunks[:self.top_docs]
        )

        if not candidate_terms:
            logger.debug("No valid candidate terms found for expansion")
            return query

        # Rank terms by similarity to query
        expansion_terms = self._rank_terms_by_similarity(
            query_embedding,
            candidate_terms
        )

        if not expansion_terms:
            return query

        # Build expanded query
        expanded = f"{query} {' '.join(expansion_terms)}"
        logger.debug(f"Query expanded: '{query}' -> '{expanded}'")

        return expanded

    def _extract_candidate_terms(self, query: str, chunks: List) -> Set[str]:
        """
        Extract candidate terms from top chunks.

        Args:
            query: Original query (to filter out query terms)
            chunks: List of top BM25 chunks

        Returns:
            Set of candidate expansion terms
        """
        candidates = set()
        query_terms = set(query.lower().split())

        for chunk in chunks:
            # Simple tokenization: split on whitespace and basic punctuation
            text = chunk.text.lower()
            terms = text.replace(',', ' ').replace('.', ' ').replace(':', ' ').split()

            for term in terms:
                # Filter criteria
                if (len(term) >= self.min_term_length and
                        term.isalpha() and
                        term not in self.stopwords and
                        term not in query_terms):
                    candidates.add(term)

        return candidates

    def _rank_terms_by_similarity(self, query_embedding: np.ndarray,
                                  candidate_terms: Set[str]) -> List[str]:
        """
        Rank candidate terms by embedding similarity to query.

        Args:
            query_embedding: Query embedding vector
            candidate_terms: Set of candidate expansion terms

        Returns:
            List of top-N expansion terms sorted by similarity
        """
        if not candidate_terms:
            return []

        # Embed all candidate terms
        term_list = list(candidate_terms)
        term_embeddings = self.embedding_model.encode(
            term_list,
            convert_to_tensor=False,
            show_progress_bar=False
        )

        # Calculate cosine similarities
        similarities = []
        for i, term in enumerate(term_list):
            similarity = self._cosine_similarity(query_embedding, term_embeddings[i])
            similarities.append((term, similarity))

        # Sort by similarity (descending) and take top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        expansion_terms = [term for term, _ in similarities[:self.max_expansions]]

        return expansion_terms

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
