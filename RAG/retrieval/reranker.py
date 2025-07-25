#!/usr/bin/env python3
"""
reranker.py

Cross-encoder re-ranking for improving retrieval precision.
"""

import logging
from typing import List, Tuple, Optional
import time

from sentence_transformers import CrossEncoder
import torch

from .retrieval_config import RetrievalConfig

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Cross-encoder based re-ranking for retrieved passages.
    Uses pre-trained models to score query-passage pairs.
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        
        # Initialize cross-encoder model
        self.model = CrossEncoder(
            config.models.reranker_model,
            device=self._get_device()
        )
        
        # Performance optimization
        if hasattr(self.model, 'eval'):
            self.model.eval()
    
    def _get_device(self) -> str:
        """Determine device for model"""
        if self.config.models.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.models.device
    
    def rerank(self, query: str, results: List) -> List:
        """
        Re-rank retrieval results using cross-encoder
        
        Args:
            query: Original search query
            results: List of RetrievalResult objects
            
        Returns:
            Re-ranked list of RetrievalResult objects
        """
        if not results:
            return results
        
        start_time = time.time()
        
        # Prepare query-passage pairs
        pairs = []
        for result in results:
            pairs.append([query, result.chunk.text])
        
        try:
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Update result scores and sort
            for result, score in zip(results, scores):
                result.score = float(score)
                result.metadata = result.metadata or {}
                result.metadata['reranker_score'] = float(score)
                result.metadata['original_score'] = result.metadata.get('original_score', result.score)
            
            # Sort by new scores
            reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
                result.retrieval_method = "reranked"
            
            rerank_time = time.time() - start_time
            logger.debug(f"Re-ranked {len(results)} results in {rerank_time:.3f}s")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return results
    
    def score_pair(self, query: str, passage: str) -> float:
        """
        Score a single query-passage pair
        
        Args:
            query: Search query
            passage: Text passage
            
        Returns:
            Relevance score
        """
        try:
            score = self.model.predict([[query, passage]])
            return float(score[0])
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return 0.0
    
    def batch_score(self, query: str, passages: List[str]) -> List[float]:
        """
        Score multiple query-passage pairs in batch
        
        Args:
            query: Search query
            passages: List of text passages
            
        Returns:
            List of relevance scores
        """
        if not passages:
            return []
        
        try:
            pairs = [[query, passage] for passage in passages]
            scores = self.model.predict(pairs)
            return [float(score) for score in scores]
        except Exception as e:
            logger.error(f"Batch scoring failed: {e}")
            return [0.0] * len(passages)
    
    def get_model_info(self) -> dict:
        """Get information about the reranker model"""
        return {
            "model_name": self.config.models.reranker_model,
            "device": self.model.device if hasattr(self.model, 'device') else 'unknown',
            "max_length": getattr(self.model, 'max_length', 'unknown')
        }

class BiEncoderReranker:
    """
    Alternative bi-encoder based re-ranking using cosine similarity.
    Less accurate than cross-encoder but faster for large result sets.
    """
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def rerank(self, query: str, results: List) -> List:
        """Re-rank using bi-encoder similarity"""
        if not results:
            return results
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Encode passages
            passages = [result.chunk.text for result in results]
            passage_embeddings = self.embedding_model.encode(passages)
            
            # Calculate cosine similarities
            similarities = self.embedding_model.similarity(query_embedding, passage_embeddings)[0]
            
            # Update scores
            for result, similarity in zip(results, similarities):
                result.score = float(similarity)
                result.metadata = result.metadata or {}
                result.metadata['biencoder_score'] = float(similarity)
            
            # Sort and update ranks
            reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
                result.retrieval_method = "biencoder_reranked"
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Bi-encoder re-ranking failed: {e}")
            return results

class HybridReranker:
    """
    Hybrid re-ranker that combines cross-encoder and bi-encoder scores.
    """
    
    def __init__(self, config: RetrievalConfig, embedding_model):
        self.cross_encoder = CrossEncoderReranker(config)
        self.bi_encoder = BiEncoderReranker(embedding_model)
        self.cross_weight = 0.7
        self.bi_weight = 0.3
    
    def rerank(self, query: str, results: List) -> List:
        """Re-rank using hybrid approach"""
        if not results:
            return results
        
        # Get cross-encoder scores
        cross_results = self.cross_encoder.rerank(query, results.copy())
        
        # Get bi-encoder scores
        bi_results = self.bi_encoder.rerank(query, results.copy())
        
        # Combine scores
        result_dict = {r.chunk_id: r for r in results}
        
        for cross_result, bi_result in zip(cross_results, bi_results):
            original_result = result_dict[cross_result.chunk_id]
            
            # Weighted combination
            combined_score = (
                self.cross_weight * cross_result.score + 
                self.bi_weight * bi_result.score
            )
            
            original_result.score = combined_score
            original_result.metadata = original_result.metadata or {}
            original_result.metadata['cross_encoder_score'] = cross_result.score
            original_result.metadata['bi_encoder_score'] = bi_result.score
            original_result.metadata['hybrid_score'] = combined_score
        
        # Sort by combined scores
        final_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(final_results):
            result.rank = i + 1
            result.retrieval_method = "hybrid_reranked"
        
        return final_results 