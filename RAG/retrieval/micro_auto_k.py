"""
micro_auto_k.py

Lightweight adaptive top-k selection for multi-aspect queries.
Simple heuristic-based approach that covers 80% of adaptive value with 15% of the code.
"""

import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class MicroAutoK:
    """
    Micro auto-k: widen retrieval for multi-aspect queries.

    This lightweight heuristic adjusts top-k based on:
    1. Query patterns (comparison words indicate multi-aspect)
    2. Score distribution (flat tail indicates low discrimination)

    Replaces complex 540-line adaptive_retrieval.py with simple 80-line heuristic
    while maintaining most of the value for common use cases.
    """

    # Comparison words that indicate multi-aspect queries
    COMPARISON_WORDS = {
        "compare", "comparison", "difference", "differences",
        "versus", "vs", "vs.", "both", "either", "contrast",
        "similarities", "distinction", "between"
    }

    def __init__(self, increment: int = 10,
                 max_multiplier: float = 2.0,
                 variance_threshold: float = 0.01):
        """
        Initialize micro auto-k.

        Args:
            increment: Number of chunks to add when condition met (default: 10)
            max_multiplier: Maximum multiplier for initial_k (default: 2.0)
            variance_threshold: Threshold for flat score tail detection (default: 0.01)
        """
        self.increment = increment
        self.max_multiplier = max_multiplier
        self.variance_threshold = variance_threshold

    def adjust_top_k(self, query: str, initial_k: int,
                    fused_scores: List[float]) -> int:
        """
        Adjust top-k based on query characteristics and score distribution.

        Args:
            query: User query string
            initial_k: Initial top-k value
            fused_scores: List of fusion scores (sorted descending)

        Returns:
            Adjusted top-k value
        """
        adjusted_k = initial_k
        adjustments = []

        # Rule 1: Check for comparison/multi-aspect query patterns
        if self._is_comparison_query(query):
            adjusted_k += self.increment
            adjustments.append("comparison_query")
            logger.debug(f"Micro auto-k: Detected comparison query, +{self.increment}")

        # Rule 2: Check for flat score tail (low discrimination)
        if self._has_flat_score_tail(fused_scores, initial_k):
            adjusted_k += self.increment
            adjustments.append("flat_score_tail")
            logger.debug(f"Micro auto-k: Detected flat score tail, +{self.increment}")

        # Apply ceiling (max 2x initial_k)
        max_k = int(initial_k * self.max_multiplier)
        if adjusted_k > max_k:
            adjusted_k = max_k
            adjustments.append(f"capped_at_{max_k}")

        # Log adjustment
        if adjusted_k != initial_k:
            logger.info(
                f"Micro auto-k: {initial_k} → {adjusted_k} "
                f"(reasons: {', '.join(adjustments)})"
            )

        return adjusted_k

    def _is_comparison_query(self, query: str) -> bool:
        """
        Check if query contains comparison/contrast words.

        Args:
            query: User query string

        Returns:
            True if query appears to be comparing/contrasting topics
        """
        query_lower = query.lower()
        return any(word in query_lower for word in self.COMPARISON_WORDS)

    def _has_flat_score_tail(self, scores: List[float], initial_k: int) -> bool:
        """
        Check if score tail is flat (low variance = low discrimination).

        A flat score tail indicates that the retrieval system cannot confidently
        discriminate between items, suggesting more results should be considered.

        Args:
            scores: List of scores (sorted descending)
            initial_k: Initial top-k value

        Returns:
            True if score tail has low variance
        """
        if len(scores) < initial_k or initial_k < 5:
            return False

        # Analyze last 5 scores before initial_k cutoff
        tail_start = max(0, initial_k - 5)
        tail_scores = scores[tail_start:initial_k]

        if len(tail_scores) < 2:
            return False

        # Calculate variance
        variance = np.var(tail_scores)

        # Low variance indicates flat tail
        return variance < self.variance_threshold

    def get_adjustment_reason(self, query: str, scores: List[float],
                             initial_k: int) -> str:
        """
        Get human-readable reason for adjustment decision.

        Args:
            query: User query
            scores: Fusion scores
            initial_k: Initial top-k

        Returns:
            String explaining adjustment reason
        """
        reasons = []

        if self._is_comparison_query(query):
            reasons.append("comparison/contrast query detected")

        if self._has_flat_score_tail(scores, initial_k):
            tail_start = max(0, initial_k - 5)
            tail_scores = scores[tail_start:initial_k]
            variance = np.var(tail_scores)
            reasons.append(f"flat score tail (variance: {variance:.4f})")

        if not reasons:
            return "no adjustment needed"

        return "; ".join(reasons)
