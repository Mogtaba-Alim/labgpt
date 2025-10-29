"""
Deduplication and Similarity Filtering for Q&A Pairs

This module provides sophisticated deduplication using embeddings to identify
and remove near-duplicate Q&A pairs while preserving diversity.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict

from ..tasks.qa_generator import GroundedQA


@dataclass
class SimilarityCluster:
    """Represents a cluster of similar Q&A pairs."""
    representative_qa: GroundedQA
    similar_qas: List[GroundedQA] = field(default_factory=list)
    average_similarity: float = 0.0
    cluster_size: int = 0


@dataclass
class DeduplicationStats:
    """Statistics from deduplication process."""
    original_count: int = 0
    duplicates_removed: int = 0
    final_count: int = 0
    deduplication_rate: float = 0.0
    clusters_formed: int = 0
    average_cluster_size: float = 0.0
    similarity_distribution: Dict[str, int] = field(default_factory=dict)


class QADeduplicator:
    """Deduplicates Q&A pairs using embedding-based similarity."""
    
    def __init__(self, 
                 similarity_threshold: float = 0.92,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 preserve_diversity: bool = True):
        """
        Initialize the deduplicator.
        
        Args:
            similarity_threshold: Cosine similarity threshold for considering duplicates
            embedding_model: Sentence transformer model for embeddings
            preserve_diversity: Whether to preserve diversity across complexity levels
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self.preserve_diversity = preserve_diversity
        self.logger = logging.getLogger(__name__)
        self.stats = DeduplicationStats()
        
        # Initialize sentence transformer
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(embedding_model)
        except ImportError:
            self.logger.error("sentence-transformers not installed. Using fallback method.")
            self.embedding_model = None
    
    def deduplicate(self, qa_pairs: List[GroundedQA]) -> List[GroundedQA]:
        """
        Remove near-duplicate Q&A pairs while preserving diversity.
        
        Args:
            qa_pairs: List of Q&A pairs to deduplicate
            
        Returns:
            Deduplicated list of Q&A pairs
        """
        if not qa_pairs:
            return qa_pairs
        
        self.stats.original_count = len(qa_pairs)
        
        # Group by complexity level if preserving diversity
        if self.preserve_diversity:
            grouped_pairs = self._group_by_attributes(qa_pairs)
            deduplicated_pairs = []
            
            for group_key, group_pairs in grouped_pairs.items():
                group_deduplicated = self._deduplicate_group(group_pairs)
                deduplicated_pairs.extend(group_deduplicated)
        else:
            deduplicated_pairs = self._deduplicate_group(qa_pairs)
        
        # Update statistics
        self.stats.final_count = len(deduplicated_pairs)
        self.stats.duplicates_removed = self.stats.original_count - self.stats.final_count
        self.stats.deduplication_rate = (
            self.stats.duplicates_removed / self.stats.original_count 
            if self.stats.original_count > 0 else 0.0
        )
        
        self.logger.info(f"Deduplication complete: {self.stats.original_count} -> {self.stats.final_count} "
                        f"({self.stats.duplicates_removed} duplicates removed)")
        
        return deduplicated_pairs
    
    def _group_by_attributes(self, qa_pairs: List[GroundedQA]) -> Dict[str, List[GroundedQA]]:
        """Group Q&A pairs by attributes to preserve diversity."""
        groups = defaultdict(list)
        
        for qa in qa_pairs:
            # Create group key from attributes
            group_key = f"{qa.complexity_level}_{qa.task_focus_area}_{getattr(qa, 'is_negative_example', False)}"
            groups[group_key].append(qa)
        
        return dict(groups)
    
    def _deduplicate_group(self, qa_pairs: List[GroundedQA]) -> List[GroundedQA]:
        """Deduplicate a group of Q&A pairs."""
        if len(qa_pairs) <= 1:
            return qa_pairs
        
        # Generate embeddings
        embeddings = self._generate_embeddings(qa_pairs)
        if embeddings is None:
            # Fallback to lexical similarity
            return self._deduplicate_lexical(qa_pairs)
        
        # Find similar pairs using embeddings
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        clusters = self._form_clusters(qa_pairs, similarity_matrix)
        
        # Select representative from each cluster
        deduplicated = self._select_representatives(clusters)
        
        return deduplicated
    
    def _generate_embeddings(self, qa_pairs: List[GroundedQA]) -> Optional[np.ndarray]:
        """Generate embeddings for Q&A pairs."""
        if self.embedding_model is None:
            return None
        
        try:
            # Create combined text for each Q&A pair
            texts = []
            for qa in qa_pairs:
                # Combine question and answer for similarity comparison
                combined_text = f"Q: {qa.question} A: {qa.answer}"
                texts.append(combined_text)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return None
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix."""
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix
    
    def _form_clusters(self, qa_pairs: List[GroundedQA], 
                      similarity_matrix: np.ndarray) -> List[SimilarityCluster]:
        """Form clusters of similar Q&A pairs."""
        n = len(qa_pairs)
        visited = set()
        clusters = []
        
        for i in range(n):
            if i in visited:
                continue
            
            # Start new cluster
            cluster_indices = [i]
            visited.add(i)
            
            # Find similar pairs
            for j in range(i + 1, n):
                if j in visited:
                    continue
                
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    cluster_indices.append(j)
                    visited.add(j)
            
            # Create cluster
            representative_qa = qa_pairs[cluster_indices[0]]
            similar_qas = [qa_pairs[idx] for idx in cluster_indices[1:]]
            
            # Calculate average similarity within cluster
            avg_similarity = 0.0
            if len(cluster_indices) > 1:
                similarities = []
                for idx1 in cluster_indices:
                    for idx2 in cluster_indices:
                        if idx1 != idx2:
                            similarities.append(similarity_matrix[idx1, idx2])
                avg_similarity = np.mean(similarities) if similarities else 0.0
            
            cluster = SimilarityCluster(
                representative_qa=representative_qa,
                similar_qas=similar_qas,
                average_similarity=avg_similarity,
                cluster_size=len(cluster_indices)
            )
            
            clusters.append(cluster)
        
        self.stats.clusters_formed = len(clusters)
        if clusters:
            self.stats.average_cluster_size = np.mean([c.cluster_size for c in clusters])
        
        return clusters
    
    def _select_representatives(self, clusters: List[SimilarityCluster]) -> List[GroundedQA]:
        """Select representative Q&A pairs from clusters."""
        representatives = []
        
        for cluster in clusters:
            # Select the best representative from the cluster
            if not cluster.similar_qas:
                # Only one item in cluster
                representatives.append(cluster.representative_qa)
            else:
                # Select best QA from cluster based on quality scores
                all_qas = [cluster.representative_qa] + cluster.similar_qas
                best_qa = self._select_best_qa(all_qas)
                representatives.append(best_qa)
        
        return representatives
    
    def _select_best_qa(self, qa_candidates: List[GroundedQA]) -> GroundedQA:
        """Select the best Q&A from candidates."""
        # Scoring criteria (in order of importance):
        # 1. Confidence score (if available)
        # 2. Not a negative example (prefer positive)
        # 3. Has citations
        # 4. Longer, more detailed answer
        
        def qa_score(qa: GroundedQA) -> float:
            score = 0.0
            
            # Confidence score weight (most important)
            score += qa.confidence_score * 5.0

            # Prefer positive examples
            if not getattr(qa, 'is_negative_example', False):
                score += 2.0
            
            # Has citations bonus
            if qa.citations:
                score += 1.0
            
            # Answer length bonus (normalized)
            if qa.answer and qa.answer != "NOT_IN_CONTEXT":
                score += min(len(qa.answer) / 500, 1.0)  # Max 1.0 bonus
            
            # Focus area bonus (prefer specific areas over general)
            if qa.task_focus_area and qa.task_focus_area != "general":
                score += 0.5
            
            return score
        
        # Find QA with highest score
        best_qa = max(qa_candidates, key=qa_score)
        return best_qa
    
    def _deduplicate_lexical(self, qa_pairs: List[GroundedQA]) -> List[GroundedQA]:
        """Fallback lexical deduplication when embeddings are not available."""
        seen_questions = set()
        deduplicated = []
        
        for qa in qa_pairs:
            # Normalize question for comparison
            normalized_question = self._normalize_text(qa.question)
            
            if normalized_question not in seen_questions:
                seen_questions.add(normalized_question)
                deduplicated.append(qa)
        
        return deduplicated
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for lexical comparison."""
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def find_similar_pairs(self, qa_pairs: List[GroundedQA], 
                          min_similarity: float = 0.8) -> List[Tuple[GroundedQA, GroundedQA, float]]:
        """Find pairs of similar Q&A pairs above a threshold."""
        similar_pairs = []
        
        embeddings = self._generate_embeddings(qa_pairs)
        if embeddings is None:
            return similar_pairs
        
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        n = len(qa_pairs)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i, j]
                if similarity >= min_similarity:
                    similar_pairs.append((qa_pairs[i], qa_pairs[j], similarity))
        
        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    def analyze_diversity(self, qa_pairs: List[GroundedQA]) -> Dict[str, Any]:
        """Analyze diversity in Q&A pairs."""
        analysis = {
            'total_pairs': len(qa_pairs),
            'complexity_distribution': defaultdict(int),
            'focus_area_distribution': defaultdict(int),
            'symbol_type_distribution': defaultdict(int),
            'negative_example_count': 0,
            'unique_questions': 0,
            'unique_symbols': set(),
            'average_question_length': 0.0,
            'average_answer_length': 0.0
        }
        
        question_texts = set()
        question_lengths = []
        answer_lengths = []
        
        for qa in qa_pairs:
            # Distribution counts
            analysis['complexity_distribution'][qa.complexity_level] += 1
            analysis['focus_area_distribution'][qa.task_focus_area] += 1
            analysis['symbol_type_distribution'][qa.context_symbol_type] += 1

            if getattr(qa, 'is_negative_example', False):
                analysis['negative_example_count'] += 1
            
            # Unique tracking
            question_texts.add(self._normalize_text(qa.question))
            analysis['unique_symbols'].add(qa.context_symbol_name)
            
            # Length tracking
            question_lengths.append(len(qa.question))
            if qa.answer:
                answer_lengths.append(len(qa.answer))
        
        analysis['unique_questions'] = len(question_texts)
        analysis['unique_symbols'] = len(analysis['unique_symbols'])
        
        if question_lengths:
            analysis['average_question_length'] = np.mean(question_lengths)
        if answer_lengths:
            analysis['average_answer_length'] = np.mean(answer_lengths)
        
        # Convert defaultdicts to regular dicts
        analysis['complexity_distribution'] = dict(analysis['complexity_distribution'])
        analysis['focus_area_distribution'] = dict(analysis['focus_area_distribution'])
        analysis['symbol_type_distribution'] = dict(analysis['symbol_type_distribution'])
        
        return analysis
    
    def get_stats(self) -> DeduplicationStats:
        """Get deduplication statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset deduplication statistics."""
        self.stats = DeduplicationStats()


def create_deduplicator(similarity_threshold: float = 0.92, 
                       preserve_diversity: bool = True) -> QADeduplicator:
    """Create a pre-configured deduplicator."""
    return QADeduplicator(
        similarity_threshold=similarity_threshold,
        preserve_diversity=preserve_diversity
    ) 