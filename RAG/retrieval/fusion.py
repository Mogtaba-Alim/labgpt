#!/usr/bin/env python3
"""
fusion.py

Result fusion methods for combining dense and sparse retrieval results.
"""

import logging
from typing import List, Dict, Set
from collections import defaultdict, Counter
import numpy as np

from .retrieval_config import RetrievalConfig

logger = logging.getLogger(__name__)

class ResultFusion:
    """
    Fusion methods for combining retrieval results from different sources.
    Supports Reciprocal Rank Fusion (RRF), linear combination, and rank-based fusion.
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        
        # Fusion method mapping
        self.fusion_methods = {
            "rrf": self._reciprocal_rank_fusion,
            "linear": self._linear_fusion,
            "rank_sum": self._rank_sum_fusion
        }
    
    def fuse_results(self, all_results: List) -> List:
        """
        Fuse results from multiple retrieval methods
        
        Args:
            all_results: List of RetrievalResult objects from different methods
            
        Returns:
            Fused and sorted list of RetrievalResult objects
        """
        if not all_results:
            return []
        
        # Group results by retrieval method
        method_results = self._group_by_method(all_results)
        
        # Apply fusion method
        fusion_method = self.fusion_methods.get(
            self.config.fusion.fusion_method,
            self._reciprocal_rank_fusion
        )
        
        fused_results = fusion_method(method_results)
        
        logger.debug(f"Fused {len(all_results)} results into {len(fused_results)} unique results")
        
        return fused_results
    
    def _group_by_method(self, results: List) -> Dict[str, List]:
        """Group results by retrieval method"""
        method_groups = defaultdict(list)
        
        for result in results:
            method = result.retrieval_method
            method_groups[method].append(result)
        
        # Sort each group by score/rank
        for method, method_results in method_groups.items():
            method_results.sort(key=lambda x: x.score, reverse=True)
            # Update ranks within each method
            for i, result in enumerate(method_results):
                result.rank = i + 1
        
        return dict(method_groups)
    
    def _reciprocal_rank_fusion(self, method_results: Dict[str, List]) -> List:
        """
        Reciprocal Rank Fusion (RRF) - effective method for combining ranked lists
        
        RRF score = sum(1 / (k + rank)) for each method where item appears
        """
        k = self.config.fusion.rrf_k
        chunk_scores = defaultdict(float)
        chunk_objects = {}
        
        # Calculate RRF scores
        for method, results in method_results.items():
            for result in results:
                chunk_id = result.chunk_id
                rank = result.rank
                
                # RRF formula
                rrf_score = 1.0 / (k + rank)
                chunk_scores[chunk_id] += rrf_score
                
                # Store the result object (use the one with highest original score)
                if (chunk_id not in chunk_objects or 
                    result.score > chunk_objects[chunk_id].score):
                    chunk_objects[chunk_id] = result
        
        # Create fused results
        fused_results = []
        for chunk_id, fused_score in chunk_scores.items():
            result = chunk_objects[chunk_id]
            result.score = fused_score
            result.retrieval_method = "fused_rrf"
            result.metadata = result.metadata or {}
            result.metadata['fusion_score'] = fused_score
            result.metadata['rrf_k'] = k
            fused_results.append(result)
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update final ranks
        for i, result in enumerate(fused_results):
            result.rank = i + 1
        
        return fused_results
    
    def _linear_fusion(self, method_results: Dict[str, List]) -> List:
        """
        Linear combination of normalized scores from different methods
        """
        chunk_scores = defaultdict(float)
        chunk_objects = {}
        
        # Get method weights
        method_weights = {
            "dense": self.config.fusion.dense_weight,
            "sparse": self.config.fusion.sparse_weight
        }
        
        # Normalize scores within each method and combine
        for method, results in method_results.items():
            if not results:
                continue
            
            # Normalize scores to [0, 1] range
            scores = [r.score for r in results]
            if len(set(scores)) > 1:  # Check if scores vary
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score
                
                for result in results:
                    if score_range > 0:
                        normalized_score = (result.score - min_score) / score_range
                    else:
                        normalized_score = 1.0
                    
                    chunk_id = result.chunk_id
                    weight = method_weights.get(method, 0.5)
                    
                    chunk_scores[chunk_id] += weight * normalized_score
                    
                    # Store result object
                    if (chunk_id not in chunk_objects or 
                        result.score > chunk_objects[chunk_id].score):
                        chunk_objects[chunk_id] = result
            else:
                # All scores are the same, use uniform weighting
                for result in results:
                    chunk_id = result.chunk_id
                    weight = method_weights.get(method, 0.5)
                    chunk_scores[chunk_id] += weight
                    
                    if chunk_id not in chunk_objects:
                        chunk_objects[chunk_id] = result
        
        # Create fused results
        fused_results = []
        for chunk_id, fused_score in chunk_scores.items():
            result = chunk_objects[chunk_id]
            result.score = fused_score
            result.retrieval_method = "fused_linear"
            result.metadata = result.metadata or {}
            result.metadata['fusion_score'] = fused_score
            fused_results.append(result)
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(fused_results):
            result.rank = i + 1
        
        return fused_results
    
    def _rank_sum_fusion(self, method_results: Dict[str, List]) -> List:
        """
        Simple rank-based fusion: sum of inverse ranks
        """
        chunk_scores = defaultdict(float)
        chunk_objects = {}
        
        for method, results in method_results.items():
            max_rank = len(results)
            
            for result in results:
                chunk_id = result.chunk_id
                # Inverse rank scoring (higher rank = lower score)
                rank_score = (max_rank - result.rank + 1) / max_rank
                
                chunk_scores[chunk_id] += rank_score
                
                # Store result object
                if (chunk_id not in chunk_objects or 
                    result.score > chunk_objects[chunk_id].score):
                    chunk_objects[chunk_id] = result
        
        # Create fused results
        fused_results = []
        for chunk_id, fused_score in chunk_scores.items():
            result = chunk_objects[chunk_id]
            result.score = fused_score
            result.retrieval_method = "fused_rank_sum"
            result.metadata = result.metadata or {}
            result.metadata['fusion_score'] = fused_score
            fused_results.append(result)
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(fused_results):
            result.rank = i + 1
        
        return fused_results
    
    def analyze_method_overlap(self, method_results: Dict[str, List]) -> Dict:
        """
        Analyze overlap between different retrieval methods
        
        Returns:
            Dictionary with overlap statistics
        """
        method_chunks = {}
        for method, results in method_results.items():
            method_chunks[method] = set(r.chunk_id for r in results)
        
        methods = list(method_chunks.keys())
        overlaps = {}
        
        # Calculate pairwise overlaps
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                chunks1, chunks2 = method_chunks[method1], method_chunks[method2]
                
                intersection = len(chunks1.intersection(chunks2))
                union = len(chunks1.union(chunks2))
                
                overlaps[f"{method1}_vs_{method2}"] = {
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union > 0 else 0,
                    "overlap_ratio": intersection / min(len(chunks1), len(chunks2)) if min(len(chunks1), len(chunks2)) > 0 else 0
                }
        
        # Overall statistics
        all_chunks = set()
        for chunks in method_chunks.values():
            all_chunks.update(chunks)
        
        return {
            "method_sizes": {method: len(chunks) for method, chunks in method_chunks.items()},
            "total_unique_chunks": len(all_chunks),
            "overlaps": overlaps,
            "coverage": {
                method: len(chunks) / len(all_chunks) if all_chunks else 0 
                for method, chunks in method_chunks.items()
            }
        }
    
    def get_fusion_stats(self, method_results: Dict[str, List], fused_results: List) -> Dict:
        """Get statistics about the fusion process"""
        stats = {
            "input_methods": list(method_results.keys()),
            "input_counts": {method: len(results) for method, results in method_results.items()},
            "total_input_results": sum(len(results) for results in method_results.values()),
            "fused_result_count": len(fused_results),
            "fusion_method": self.config.fusion.fusion_method,
            "fusion_config": {
                "dense_weight": self.config.fusion.dense_weight,
                "sparse_weight": self.config.fusion.sparse_weight,
                "rrf_k": self.config.fusion.rrf_k
            }
        }
        
        # Add overlap analysis
        if len(method_results) > 1:
            stats["overlap_analysis"] = self.analyze_method_overlap(method_results)
        
        return stats 