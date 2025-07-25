#!/usr/bin/env python3
"""
adaptive_retrieval.py

Adaptive Top-K retrieval system with coverage heuristics for dynamic chunk selection.
Implements intelligent retrieval that adjusts the number of chunks based on coverage analysis.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import math

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from ..ingestion.chunk_objects import ChunkMetadata
from .hybrid_retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class CoverageMetrics:
    """Metrics for assessing retrieval coverage"""
    semantic_diversity: float
    content_coverage: float
    topic_breadth: float
    information_density: float
    redundancy_score: float
    overall_coverage: float

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive retrieval"""
    # Coverage thresholds
    min_coverage_score: float = 0.7
    max_coverage_score: float = 0.95
    diversity_threshold: float = 0.6
    
    # Retrieval parameters
    initial_top_k: int = 5
    max_top_k: int = 50
    increment_step: int = 3
    
    # Content analysis
    min_sentence_length: int = 10
    max_redundancy_ratio: float = 0.3
    
    # Query complexity factors
    enable_query_complexity_analysis: bool = True
    complexity_multiplier: float = 1.2

class SemanticCoverageAnalyzer:
    """
    Analyzes semantic coverage of retrieved chunks to determine adequacy
    """
    
    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        self.embedding_model = embedding_model
        
        # Initialize NLTK components
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.warning("NLTK data not found. Some features may be limited.")
    
    def calculate_coverage_metrics(self, 
                                 query: str,
                                 chunks: List[ChunkMetadata],
                                 chunk_embeddings: Optional[np.ndarray] = None) -> CoverageMetrics:
        """
        Calculate comprehensive coverage metrics for retrieved chunks
        
        Args:
            query: Original search query
            chunks: List of retrieved chunks
            chunk_embeddings: Optional precomputed embeddings
            
        Returns:
            CoverageMetrics object with detailed analysis
        """
        if not chunks:
            return CoverageMetrics(0, 0, 0, 0, 1.0, 0)
        
        # Extract texts for analysis
        texts = [chunk.text for chunk in chunks]
        
        # Calculate individual metrics
        semantic_diversity = self._calculate_semantic_diversity(texts, chunk_embeddings)
        content_coverage = self._calculate_content_coverage(query, texts)
        topic_breadth = self._calculate_topic_breadth(texts)
        information_density = self._calculate_information_density(texts)
        redundancy_score = self._calculate_redundancy_score(texts)
        
        # Calculate overall coverage score
        overall_coverage = self._calculate_overall_coverage(
            semantic_diversity, content_coverage, topic_breadth, 
            information_density, redundancy_score
        )
        
        return CoverageMetrics(
            semantic_diversity=semantic_diversity,
            content_coverage=content_coverage,
            topic_breadth=topic_breadth,
            information_density=information_density,
            redundancy_score=redundancy_score,
            overall_coverage=overall_coverage
        )
    
    def _calculate_semantic_diversity(self, 
                                    texts: List[str], 
                                    embeddings: Optional[np.ndarray] = None) -> float:
        """Calculate semantic diversity among chunks"""
        if len(texts) < 2:
            return 1.0 if len(texts) == 1 else 0.0
        
        try:
            # Use provided embeddings or generate them
            if embeddings is None and self.embedding_model:
                embeddings = self.embedding_model.encode(texts)
            elif embeddings is None:
                # Fallback to simple lexical diversity
                return self._lexical_diversity(texts)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings)
            
            # Remove diagonal (self-similarities)
            mask = np.ones(similarities.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            similarities_off_diag = similarities[mask]
            
            # Calculate diversity as 1 - average similarity
            avg_similarity = np.mean(similarities_off_diag)
            diversity = 1.0 - avg_similarity
            
            return max(0.0, min(1.0, diversity))
            
        except Exception as e:
            logger.warning(f"Error calculating semantic diversity: {e}")
            return self._lexical_diversity(texts)
    
    def _lexical_diversity(self, texts: List[str]) -> float:
        """Fallback lexical diversity calculation"""
        try:
            all_words = []
            for text in texts:
                words = word_tokenize(text.lower())
                words = [w for w in words if w.isalpha()]
                all_words.extend(words)
            
            if not all_words:
                return 0.0
            
            unique_words = len(set(all_words))
            total_words = len(all_words)
            
            return unique_words / total_words if total_words > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating lexical diversity: {e}")
            return 0.5
    
    def _calculate_content_coverage(self, query: str, texts: List[str]) -> float:
        """Calculate how well the chunks cover the query content"""
        try:
            # Extract key terms from query
            query_words = set(word_tokenize(query.lower()))
            query_words = {w for w in query_words if w.isalpha() and len(w) > 2}
            
            if not query_words:
                return 0.5
            
            # Extract words from all chunks
            chunk_words = set()
            for text in texts:
                words = word_tokenize(text.lower())
                words = {w for w in words if w.isalpha() and len(w) > 2}
                chunk_words.update(words)
            
            # Calculate coverage as intersection / query terms
            coverage = len(query_words.intersection(chunk_words)) / len(query_words)
            return min(1.0, coverage)
            
        except Exception as e:
            logger.warning(f"Error calculating content coverage: {e}")
            return 0.5
    
    def _calculate_topic_breadth(self, texts: List[str]) -> float:
        """Calculate topic breadth using simple clustering"""
        try:
            if len(texts) < 2:
                return 1.0 if len(texts) == 1 else 0.0
            
            # Simple topic indicator: sentence count variation
            sentence_counts = []
            for text in texts:
                sentences = sent_tokenize(text)
                sentence_counts.append(len(sentences))
            
            if not sentence_counts:
                return 0.0
            
            # Calculate variation in content length as proxy for topic breadth
            mean_sentences = np.mean(sentence_counts)
            std_sentences = np.std(sentence_counts)
            
            # Normalize variation score
            if mean_sentences > 0:
                variation_score = min(1.0, std_sentences / mean_sentences)
            else:
                variation_score = 0.0
            
            # Topic breadth increases with variation but caps at reasonable level
            topic_breadth = min(0.8, variation_score * 2)
            return topic_breadth
            
        except Exception as e:
            logger.warning(f"Error calculating topic breadth: {e}")
            return 0.5
    
    def _calculate_information_density(self, texts: List[str]) -> float:
        """Calculate information density of the chunk collection"""
        try:
            if not texts:
                return 0.0
            
            total_chars = sum(len(text) for text in texts)
            total_words = 0
            total_sentences = 0
            
            for text in texts:
                words = word_tokenize(text)
                sentences = sent_tokenize(text)
                total_words += len(words)
                total_sentences += len(sentences)
            
            if total_chars == 0:
                return 0.0
            
            # Calculate density metrics
            words_per_char = total_words / total_chars if total_chars > 0 else 0
            sentences_per_word = total_sentences / total_words if total_words > 0 else 0
            
            # Normalize and combine metrics
            density_score = (words_per_char * 100 + sentences_per_word * 10) / 2
            return min(1.0, density_score)
            
        except Exception as e:
            logger.warning(f"Error calculating information density: {e}")
            return 0.5
    
    def _calculate_redundancy_score(self, texts: List[str]) -> float:
        """Calculate redundancy score (higher = more redundant)"""
        try:
            if len(texts) < 2:
                return 0.0
            
            # Simple redundancy check using sentence overlap
            all_sentences = []
            for text in texts:
                sentences = sent_tokenize(text)
                sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]
                all_sentences.extend(sentences)
            
            if not all_sentences:
                return 0.0
            
            unique_sentences = len(set(all_sentences))
            total_sentences = len(all_sentences)
            
            redundancy = 1.0 - (unique_sentences / total_sentences)
            return redundancy
            
        except Exception as e:
            logger.warning(f"Error calculating redundancy score: {e}")
            return 0.0
    
    def _calculate_overall_coverage(self, 
                                  semantic_diversity: float,
                                  content_coverage: float,
                                  topic_breadth: float,
                                  information_density: float,
                                  redundancy_score: float) -> float:
        """Calculate overall coverage score"""
        # Weighted combination of metrics
        coverage_score = (
            semantic_diversity * 0.25 +
            content_coverage * 0.30 +
            topic_breadth * 0.20 +
            information_density * 0.15 +
            (1.0 - redundancy_score) * 0.10  # Less redundancy is better
        )
        
        return max(0.0, min(1.0, coverage_score))

class QueryComplexityAnalyzer:
    """
    Analyzes query complexity to adjust retrieval parameters
    """
    
    def __init__(self):
        self.complexity_indicators = {
            'multi_part': ['and', 'or', 'also', 'additionally', 'furthermore'],
            'comparison': ['compare', 'contrast', 'difference', 'versus', 'vs'],
            'analytical': ['analyze', 'explain', 'describe', 'discuss', 'evaluate'],
            'factual': ['what', 'when', 'where', 'who', 'which'],
            'complex_logical': ['because', 'therefore', 'thus', 'consequently', 'however']
        }
    
    def analyze_query_complexity(self, query: str) -> Dict[str, float]:
        """
        Analyze query complexity across multiple dimensions
        
        Args:
            query: Search query to analyze
            
        Returns:
            Dictionary with complexity scores
        """
        query_lower = query.lower()
        complexity_scores = {}
        
        # Check for different types of complexity
        for complexity_type, indicators in self.complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            complexity_scores[complexity_type] = min(1.0, score / 3.0)  # Normalize
        
        # Calculate overall complexity
        overall_complexity = np.mean(list(complexity_scores.values()))
        
        # Additional factors
        word_count = len(query.split())
        length_factor = min(1.0, word_count / 20.0)  # Longer queries are more complex
        
        question_marks = query.count('?')
        question_factor = min(1.0, question_marks / 2.0)
        
        # Final complexity score
        final_complexity = (
            overall_complexity * 0.6 +
            length_factor * 0.3 +
            question_factor * 0.1
        )
        
        complexity_scores['overall'] = final_complexity
        complexity_scores['length_factor'] = length_factor
        complexity_scores['question_factor'] = question_factor
        
        return complexity_scores

class AdaptiveTopKRetriever:
    """
    Adaptive retrieval system that dynamically adjusts top-k based on coverage analysis
    """
    
    def __init__(self,
                 base_retriever: HybridRetriever,
                 config: Optional[AdaptiveConfig] = None,
                 embedding_model: Optional[SentenceTransformer] = None):
        
        self.base_retriever = base_retriever
        self.config = config or AdaptiveConfig()
        self.embedding_model = embedding_model
        
        # Initialize analyzers
        self.coverage_analyzer = SemanticCoverageAnalyzer(embedding_model)
        self.complexity_analyzer = QueryComplexityAnalyzer()
        
        # Performance tracking
        self.adaptation_stats = {
            'total_queries': 0,
            'average_chunks_retrieved': 0.0,
            'coverage_improvements': 0,
            'early_stops': 0
        }
    
    def retrieve_adaptive(self, 
                         query: str,
                         max_iterations: int = 10) -> Tuple[List[RetrievalResult], CoverageMetrics]:
        """
        Perform adaptive retrieval with coverage-based stopping
        
        Args:
            query: Search query
            max_iterations: Maximum number of retrieval iterations
            
        Returns:
            Tuple of (final results, coverage metrics)
        """
        self.adaptation_stats['total_queries'] += 1
        
        # Analyze query complexity to determine initial parameters
        complexity_analysis = self.complexity_analyzer.analyze_query_complexity(query)
        
        # Adjust initial top-k based on complexity
        initial_k = self._adjust_initial_k(complexity_analysis)
        current_k = initial_k
        
        best_results = []
        best_coverage = None
        iteration = 0
        
        logger.info(f"Starting adaptive retrieval for query: '{query[:50]}...' "
                   f"(complexity: {complexity_analysis['overall']:.2f})")
        
        while iteration < max_iterations and current_k <= self.config.max_top_k:
            iteration += 1
            
            # Retrieve with current k
            results = self.base_retriever.retrieve(query, top_k=current_k)
            
            if not results:
                break
            
            # Extract chunks and calculate coverage
            chunks = [result.chunk for result in results]
            
            # Get embeddings if available
            embeddings = None
            if hasattr(results[0], 'embedding') and results[0].embedding is not None:
                embeddings = np.array([result.embedding for result in results])
            
            # Calculate coverage metrics
            coverage_metrics = self.coverage_analyzer.calculate_coverage_metrics(
                query, chunks, embeddings
            )
            
            logger.debug(f"Iteration {iteration}: k={current_k}, "
                        f"coverage={coverage_metrics.overall_coverage:.3f}")
            
            # Check if coverage is sufficient
            if coverage_metrics.overall_coverage >= self.config.min_coverage_score:
                best_results = results
                best_coverage = coverage_metrics
                self.adaptation_stats['early_stops'] += 1
                logger.info(f"Early stop at k={current_k} with coverage={coverage_metrics.overall_coverage:.3f}")
                break
            
            # Update best results if this is better
            if (best_coverage is None or 
                coverage_metrics.overall_coverage > best_coverage.overall_coverage):
                best_results = results
                best_coverage = coverage_metrics
                self.adaptation_stats['coverage_improvements'] += 1
            
            # Check for diminishing returns
            if iteration > 1 and self._check_diminishing_returns(coverage_metrics, best_coverage):
                logger.info(f"Diminishing returns detected at k={current_k}")
                break
            
            # Increment k for next iteration
            current_k += self.config.increment_step
        
        # Update statistics
        final_k = len(best_results)
        self.adaptation_stats['average_chunks_retrieved'] = (
            (self.adaptation_stats['average_chunks_retrieved'] * 
             (self.adaptation_stats['total_queries'] - 1) + final_k) /
            self.adaptation_stats['total_queries']
        )
        
        logger.info(f"Adaptive retrieval completed: {final_k} chunks retrieved, "
                   f"coverage={best_coverage.overall_coverage:.3f}")
        
        return best_results, best_coverage
    
    def _adjust_initial_k(self, complexity_analysis: Dict[str, float]) -> int:
        """Adjust initial k based on query complexity"""
        base_k = self.config.initial_top_k
        complexity_factor = complexity_analysis['overall']
        
        if self.config.enable_query_complexity_analysis:
            # More complex queries need more chunks initially
            adjusted_k = int(base_k * (1 + complexity_factor * self.config.complexity_multiplier))
            return min(adjusted_k, self.config.max_top_k)
        
        return base_k
    
    def _check_diminishing_returns(self, 
                                 current_coverage: CoverageMetrics,
                                 previous_best: CoverageMetrics) -> bool:
        """Check if coverage improvement is showing diminishing returns"""
        if previous_best is None:
            return False
        
        improvement = current_coverage.overall_coverage - previous_best.overall_coverage
        return improvement < 0.05  # Less than 5% improvement
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about adaptive retrieval performance"""
        return {
            'adaptation_stats': self.adaptation_stats.copy(),
            'config': {
                'min_coverage_score': self.config.min_coverage_score,
                'max_top_k': self.config.max_top_k,
                'initial_top_k': self.config.initial_top_k
            }
        }
    
    def optimize_thresholds(self, 
                          query_coverage_pairs: List[Tuple[str, float]]) -> AdaptiveConfig:
        """
        Optimize configuration thresholds based on historical performance
        
        Args:
            query_coverage_pairs: List of (query, target_coverage) pairs
            
        Returns:
            Optimized configuration
        """
        if not query_coverage_pairs:
            return self.config
        
        # Analyze target coverage distribution
        target_coverages = [coverage for _, coverage in query_coverage_pairs]
        
        # Set min_coverage_score to 75th percentile of targets
        min_coverage = np.percentile(target_coverages, 75)
        
        # Create optimized config
        optimized_config = AdaptiveConfig(
            min_coverage_score=min_coverage,
            max_coverage_score=max(target_coverages) * 1.1,
            diversity_threshold=self.config.diversity_threshold,
            initial_top_k=self.config.initial_top_k,
            max_top_k=self.config.max_top_k,
            increment_step=self.config.increment_step
        )
        
        logger.info(f"Optimized coverage threshold: {min_coverage:.3f}")
        
        return optimized_config 