"""
Minimal quality filtering for RAG chunks.

Replaces the complex quality_filter.py and advanced_scoring.py with a simple
length + garbled text check. Semantic quality is delegated to the reranker.
"""

from typing import List
from dataclasses import dataclass


@dataclass
class Chunk:
    """Minimal chunk representation (will be defined in models.py)"""
    text: str
    token_count: int


class MinimalFilter:
    """
    Minimal filtering: remove garbage, delegate quality decisions to reranker.

    Philosophy: Simple filters catch obvious problems (too short, garbled text),
    while the reranker handles semantic quality assessment.
    """

    def __init__(self,
                 min_tokens: int = 30,
                 max_tokens: int = 600,
                 max_non_ascii_ratio: float = 0.3,
                 min_alpha_chars: int = 20):
        """
        Initialize minimal filter.

        Args:
            min_tokens: Minimum tokens per chunk (default: 30)
            max_tokens: Maximum tokens per chunk (default: 600)
            max_non_ascii_ratio: Maximum ratio of non-ASCII chars (default: 0.3)
            min_alpha_chars: Minimum alphabetic characters (default: 20)
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.max_non_ascii_ratio = max_non_ascii_ratio
        self.min_alpha_chars = min_alpha_chars

    def filter_chunk(self, chunk: Chunk) -> bool:
        """
        Check if chunk passes quality filters.

        Returns:
            True if chunk should be kept, False if it should be filtered out
        """
        # Length check
        if chunk.token_count < self.min_tokens:
            return False
        if chunk.token_count > self.max_tokens:
            return False

        text = chunk.text

        # Garbled text check (excessive non-ASCII)
        if len(text) > 0:
            ascii_chars = sum(1 for c in text if ord(c) < 128)
            if ascii_chars / len(text) < (1 - self.max_non_ascii_ratio):
                return False

        # Has alphabetic content (not all numbers/symbols)
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars < self.min_alpha_chars:
            return False

        return True

    def filter_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Filter a list of chunks.

        Args:
            chunks: List of chunks to filter

        Returns:
            List of chunks that passed the filters
        """
        return [c for c in chunks if self.filter_chunk(c)]

    def get_stats(self, chunks_before: List[Chunk], chunks_after: List[Chunk]) -> dict:
        """
        Get filtering statistics.

        Args:
            chunks_before: Chunks before filtering
            chunks_after: Chunks after filtering

        Returns:
            Dictionary with filtering statistics
        """
        total_before = len(chunks_before)
        total_after = len(chunks_after)
        filtered = total_before - total_after

        return {
            'total_before': total_before,
            'total_after': total_after,
            'filtered_count': filtered,
            'filtered_percentage': (filtered / total_before * 100) if total_before > 0 else 0,
            'pass_rate': (total_after / total_before * 100) if total_before > 0 else 0
        }
