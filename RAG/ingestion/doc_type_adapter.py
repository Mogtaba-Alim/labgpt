"""
doc_type_adapter.py

Document-type specific chunking strategies for optimal information density.
Different document types require different chunking approaches for best retrieval.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkingStrategy:
    """Configuration for document-type specific chunking."""
    target_tokens: Optional[int]
    min_tokens: int
    max_tokens: int
    split_on: List[str]
    preserve_boundaries: bool = True


class DocTypeAdapter:
    """
    Adapts chunking strategy based on document type.

    Different document types have different optimal chunk sizes and splitting logic:
    - Code/Notebooks: Smaller chunks aligned with functions/classes for precise retrieval
    - Papers/Protocols: Larger chunks preserving section context
    - Slides: Per-slide splitting maintaining presentation structure

    This ensures each document type is chunked optimally for its content structure.
    """

    # Document type detection patterns
    TYPE_PATTERNS = {
        'code': ['.py', '.r', '.c', '.cpp', '.java', '.js', '.ts', '.go'],
        'notebook': ['.ipynb'],
        'paper': ['.pdf', '.tex'],
        'slides': ['.pptx', '.ppt', '.key'],
        'protocol': ['.txt', '.md', '.rst'],
        'web': ['.html', '.htm']
    }

    # Chunking strategies per document type
    STRATEGIES = {
        'code': ChunkingStrategy(
            target_tokens=275,
            min_tokens=100,
            max_tokens=400,
            split_on=['function', 'class', 'method'],
            preserve_boundaries=True
        ),
        'notebook': ChunkingStrategy(
            target_tokens=300,
            min_tokens=50,
            max_tokens=500,
            split_on=['cell', 'markdown_section'],
            preserve_boundaries=True
        ),
        'paper': ChunkingStrategy(
            target_tokens=425,
            min_tokens=200,
            max_tokens=600,
            split_on=['section', 'subsection', 'paragraph'],
            preserve_boundaries=True
        ),
        'slides': ChunkingStrategy(
            target_tokens=None,  # Per-slide splitting, variable size
            min_tokens=20,
            max_tokens=800,
            split_on=['slide', 'bullet_group'],
            preserve_boundaries=True
        ),
        'protocol': ChunkingStrategy(
            target_tokens=400,
            min_tokens=150,
            max_tokens=550,
            split_on=['step', 'section', 'numbered_list'],
            preserve_boundaries=True
        ),
        'web': ChunkingStrategy(
            target_tokens=350,
            min_tokens=100,
            max_tokens=500,
            split_on=['article', 'section', 'paragraph'],
            preserve_boundaries=False  # HTML may have irregular structure
        ),
        'default': ChunkingStrategy(
            target_tokens=400,
            min_tokens=150,
            max_tokens=600,
            split_on=['paragraph', 'sentence'],
            preserve_boundaries=True
        )
    }

    def __init__(self):
        """Initialize document type adapter."""
        pass

    def detect_type(self, file_path: str) -> str:
        """
        Detect document type from file extension.

        Args:
            file_path: Path to document file

        Returns:
            Document type string (code/notebook/paper/slides/protocol/web/default)
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        for doc_type, patterns in self.TYPE_PATTERNS.items():
            if extension in patterns:
                logger.debug(f"Detected type '{doc_type}' for file: {file_path}")
                return doc_type

        logger.debug(f"Using default type for file: {file_path}")
        return 'default'

    def get_strategy(self, doc_type: str) -> ChunkingStrategy:
        """
        Get chunking strategy for document type.

        Args:
            doc_type: Document type string

        Returns:
            ChunkingStrategy configuration for the type
        """
        strategy = self.STRATEGIES.get(doc_type, self.STRATEGIES['default'])
        logger.debug(
            f"Strategy for '{doc_type}': target={strategy.target_tokens} tokens, "
            f"split_on={strategy.split_on}"
        )
        return strategy

    def get_strategy_for_file(self, file_path: str) -> ChunkingStrategy:
        """
        Detect type and get strategy for a file in one call.

        Args:
            file_path: Path to document file

        Returns:
            ChunkingStrategy configuration
        """
        doc_type = self.detect_type(file_path)
        return self.get_strategy(doc_type)

    def should_split_here(self, text: str, position: int,
                         doc_type: str, boundary_type: str) -> bool:
        """
        Determine if splitting should occur at a given position.

        Args:
            text: Full document text
            position: Character position being considered for split
            doc_type: Document type
            boundary_type: Type of boundary detected (e.g., 'function', 'section')

        Returns:
            True if split should occur at this boundary
        """
        strategy = self.get_strategy(doc_type)

        # Check if boundary type is in allowed split points
        if boundary_type not in strategy.split_on:
            return False

        # If preserve_boundaries is False, allow splitting anywhere
        if not strategy.preserve_boundaries:
            return True

        # For boundary-preserving strategies, respect the boundary type
        return True

    def get_splitting_params(self, file_path: str) -> Dict[str, Any]:
        """
        Get text splitter parameters for a document.

        Provides parameters that can be passed to TextSplitter or similar chunking utilities.

        Args:
            file_path: Path to document file

        Returns:
            Dictionary with chunk_size, chunk_overlap, and other splitting params
        """
        strategy = self.get_strategy_for_file(file_path)
        doc_type = self.detect_type(file_path)

        # Calculate overlap as 10% of target (or default for variable-size)
        if strategy.target_tokens:
            overlap = int(strategy.target_tokens * 0.1)
        else:
            overlap = 50  # Default overlap for variable-size chunks

        params = {
            'chunk_size': strategy.target_tokens or strategy.max_tokens,
            'chunk_overlap': overlap,
            'min_chunk_size': strategy.min_tokens,
            'max_chunk_size': strategy.max_tokens,
            'split_on': strategy.split_on,
            'preserve_boundaries': strategy.preserve_boundaries,
            'doc_type': doc_type
        }

        logger.info(
            f"Splitting params for {Path(file_path).name} ({doc_type}): "
            f"chunk_size={params['chunk_size']}, overlap={params['chunk_overlap']}"
        )

        return params

    def estimate_chunk_count(self, file_path: str, total_tokens: int) -> int:
        """
        Estimate number of chunks for a document.

        Args:
            file_path: Path to document file
            total_tokens: Total token count in document

        Returns:
            Estimated number of chunks
        """
        strategy = self.get_strategy_for_file(file_path)

        if strategy.target_tokens:
            # Account for overlap in estimation
            effective_chunk_size = strategy.target_tokens * 0.9
            estimated = int(total_tokens / effective_chunk_size)
        else:
            # For variable-size (like slides), estimate based on max
            estimated = int(total_tokens / (strategy.max_tokens * 0.5))

        return max(1, estimated)

    def get_type_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about all configured document types.

        Returns:
            Dictionary mapping doc type to strategy statistics
        """
        stats = {}
        for doc_type, strategy in self.STRATEGIES.items():
            stats[doc_type] = {
                'target_tokens': strategy.target_tokens,
                'min_tokens': strategy.min_tokens,
                'max_tokens': strategy.max_tokens,
                'split_points': strategy.split_on,
                'preserve_boundaries': strategy.preserve_boundaries,
                'extensions': self.TYPE_PATTERNS.get(doc_type, [])
            }
        return stats
