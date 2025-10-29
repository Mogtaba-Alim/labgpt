"""
rrf_fusion.py

Reciprocal Rank Fusion (RRF) for combining dense and sparse retrieval results.
RRF is position-based, robust, and requires no score normalization.
"""

import logging
from typing import List, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


class RRFFusion:
    """
    Reciprocal Rank Fusion for combining retrieval results.

    RRF is the single fusion method used in the system because:
    - Position-based: Works with any scoring scheme
    - No normalization needed: Robust across different score ranges
    - Proven effective: Strong performance across benchmarks
    - Simple: Easy to understand and tune

    Formula: score(chunk) = Σ 1/(k + rank) for each method where chunk appears
    Default k=60 provides good smoothing for typical ranked lists.
    """

    def __init__(self, rrf_k: int = 60):
        """
        Initialize RRF fusion.

        Args:
            rrf_k: Smoothing parameter (default: 60)
                  - Lower k (40): More aggressive, favors top ranks
                  - Standard k (60): Balanced
                  - Higher k (100): More conservative, considers lower ranks more
        """
        self.rrf_k = rrf_k

    def fuse_results(self, all_results: List) -> List:
        """
        Fuse results from multiple retrieval methods using RRF.

        Args:
            all_results: List of RetrievalResult objects from different methods

        Returns:
            Fused and sorted list of RetrievalResult objects
        """
        if not all_results:
            return []

        # Group results by retrieval method
        method_results = self._group_by_method(all_results)

        # Apply RRF fusion
        fused_results = self._reciprocal_rank_fusion(method_results)

        logger.debug(
            f"RRF fusion: {len(all_results)} input results -> "
            f"{len(fused_results)} unique fused results"
        )

        return fused_results

    def _group_by_method(self, results: List) -> Dict[str, List]:
        """
        Group results by retrieval method and assign ranks within each method.

        Args:
            results: List of RetrievalResult objects

        Returns:
            Dictionary mapping method name to sorted list of results
        """
        method_groups = defaultdict(list)

        for result in results:
            method = result.retrieval_method
            method_groups[method].append(result)

        # Sort each group by score (descending) and assign ranks
        for method, method_results in method_groups.items():
            method_results.sort(key=lambda x: x.score, reverse=True)
            for i, result in enumerate(method_results):
                result.rank = i + 1

        return dict(method_groups)

    def _reciprocal_rank_fusion(self, method_results: Dict[str, List]) -> List:
        """
        Apply Reciprocal Rank Fusion algorithm.

        RRF score for each chunk is the sum of 1/(k + rank) across all methods
        where the chunk appears. Chunks appearing in multiple methods get higher
        scores.

        Args:
            method_results: Dictionary mapping method name to sorted result lists

        Returns:
            List of fused RetrievalResult objects sorted by RRF score
        """
        chunk_scores = defaultdict(float)
        chunk_objects = {}

        # Calculate RRF scores
        for method, results in method_results.items():
            for result in results:
                chunk_id = result.chunk_id
                rank = result.rank

                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (self.rrf_k + rank)
                chunk_scores[chunk_id] += rrf_score

                # Store the result object (use the one with highest original score)
                if (chunk_id not in chunk_objects or
                        result.score > chunk_objects[chunk_id].score):
                    chunk_objects[chunk_id] = result

        # Create fused results with RRF scores
        fused_results = []
        for chunk_id, rrf_score in chunk_scores.items():
            result = chunk_objects[chunk_id]
            result.score = rrf_score
            result.retrieval_method = "hybrid"
            result.metadata = result.metadata or {}
            result.metadata['rrf_score'] = rrf_score
            result.metadata['rrf_k'] = self.rrf_k
            fused_results.append(result)

        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x.score, reverse=True)

        # Update final ranks
        for i, result in enumerate(fused_results):
            result.rank = i + 1

        return fused_results

    def analyze_method_overlap(self, method_results: Dict[str, List]) -> Dict:
        """
        Analyze overlap between different retrieval methods.

        Useful for understanding how well dense and sparse methods complement
        each other. High overlap means methods agree; low overlap means they
        find different relevant chunks.

        Args:
            method_results: Dictionary mapping method name to result lists

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
                    "overlap_ratio": (intersection / min(len(chunks1), len(chunks2))
                                    if min(len(chunks1), len(chunks2)) > 0 else 0)
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

    def get_fusion_stats(self, method_results: Dict[str, List],
                        fused_results: List) -> Dict:
        """
        Get statistics about the fusion process.

        Args:
            method_results: Input results grouped by method
            fused_results: Output fused results

        Returns:
            Dictionary with fusion statistics
        """
        stats = {
            "input_methods": list(method_results.keys()),
            "input_counts": {method: len(results)
                           for method, results in method_results.items()},
            "total_input_results": sum(len(results)
                                      for results in method_results.values()),
            "fused_result_count": len(fused_results),
            "fusion_method": "rrf",
            "rrf_k": self.rrf_k
        }

        # Add overlap analysis
        if len(method_results) > 1:
            stats["overlap_analysis"] = self.analyze_method_overlap(method_results)

        return stats
