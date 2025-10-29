"""
Core data models for the RAG pipeline.

This module defines the minimal data structures needed for document processing
and retrieval, focusing on essential fields required for functionality.
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import uuid


@dataclass
class Chunk:
    """
    Minimal chunk representation with essential fields only.

    Fields are limited to what's actually needed for retrieval and display,
    avoiding the complexity of 20+ metadata fields.
    """
    chunk_id: str                    # Unique identifier (UUID)
    doc_id: str                      # Parent document identifier
    text: str                        # Chunk content
    source_path: str                 # File path for citation
    token_count: int                 # Token count for context budgeting
    section: Optional[str] = None    # Section title for citation display
    page_number: Optional[int] = None  # Page number for citation display

    @classmethod
    def create(cls, doc_id: str, text: str, source_path: str,
               token_count: int, section: Optional[str] = None,
               page_number: Optional[int] = None) -> 'Chunk':
        """
        Factory method to create a new chunk with auto-generated ID.

        Args:
            doc_id: Parent document identifier
            text: Chunk content
            source_path: Source file path
            token_count: Number of tokens
            section: Optional section title
            page_number: Optional page number

        Returns:
            New Chunk instance with generated UUID
        """
        return cls(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=text,
            source_path=source_path,
            token_count=token_count,
            section=section,
            page_number=page_number
        )

    def get_citation(self) -> str:
        """
        Generate a citation string for this chunk.

        Returns:
            Citation string with source, section, and/or page
        """
        parts = [self.source_path]

        if self.section:
            parts.append(f"Section: {self.section}")

        if self.page_number is not None:
            parts.append(f"Page {self.page_number}")

        return " | ".join(parts)


@dataclass
class RetrievalResult:
    """
    Result from retrieval system containing chunk and scoring information.
    """
    chunk_id: str                    # Chunk identifier
    chunk: Chunk                     # Full chunk object
    score: float                     # Retrieval/reranking score
    retrieval_method: str            # "dense", "sparse", or "hybrid"
    rank: int                        # Position in results (1-indexed)
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata

    def __str__(self) -> str:
        """String representation for display."""
        return (f"[Rank {self.rank}] Score: {self.score:.3f} | "
                f"{self.chunk.get_citation()}")


@dataclass
class SearchConfig:
    """
    Configuration for a single search request.
    """
    query: str                       # Search query
    top_k: int = 10                  # Number of results to return
    expand_query: bool = False       # Enable query expansion
    verify_citations: bool = False   # Enable citation verification
    extract_cited_spans: bool = False  # Extract supporting text spans

    # Advanced options (research mode)
    use_auto_k: bool = False         # Enable micro auto-k heuristic
    rerank: bool = True              # Enable cross-encoder reranking
    diversity_threshold: float = 0.85  # Similarity threshold for deduplication


@dataclass
class CitedSpan:
    """
    A specific text span that supports a claim, with confidence score.
    """
    start_offset: int                # Character start position
    end_offset: int                  # Character end position
    text: str                        # Span text content
    confidence: float                # Confidence score (0-1)

    def __str__(self) -> str:
        """String representation showing span and confidence."""
        return f'"{self.text}" (confidence: {self.confidence:.2f})'


@dataclass
class DocumentMetadata:
    """
    Essential metadata for a processed document.
    """
    doc_id: str                      # Unique document identifier
    source_path: str                 # Original file path
    doc_type: str                    # Document type: code, paper, slides, protocol
    title: Optional[str] = None      # Document title
    page_count: Optional[int] = None  # Number of pages (if applicable)
    chunk_count: int = 0             # Number of chunks created

    @classmethod
    def create(cls, source_path: str, doc_type: str = "paper",
               title: Optional[str] = None, page_count: Optional[int] = None) -> 'DocumentMetadata':
        """
        Factory method to create document metadata with auto-generated ID.

        Args:
            source_path: Path to source file
            doc_type: Type of document (code/paper/slides/protocol)
            title: Optional document title
            page_count: Optional page count

        Returns:
            New DocumentMetadata instance
        """
        return cls(
            doc_id=str(uuid.uuid4()),
            source_path=source_path,
            doc_type=doc_type,
            title=title,
            page_count=page_count
        )


@dataclass
class SnapshotInfo:
    """
    Snapshot information for reproducibility.
    """
    timestamp: str                   # ISO format timestamp
    model: str                       # Embedding model used
    doc_count: int                   # Number of documents
    chunk_count: int                 # Number of chunks
    doc_hashes: Dict[str, str]       # Document path -> SHA256 hash mapping
    snapshot_id: str                 # Unique snapshot identifier

    @classmethod
    def create(cls, model: str, doc_count: int, chunk_count: int,
               doc_hashes: Dict[str, str]) -> 'SnapshotInfo':
        """
        Create a new snapshot with auto-generated ID and timestamp.

        Args:
            model: Embedding model name
            doc_count: Number of documents in index
            chunk_count: Number of chunks in index
            doc_hashes: Mapping of document paths to their SHA256 hashes

        Returns:
            New SnapshotInfo instance
        """
        from datetime import datetime

        return cls(
            timestamp=datetime.now().isoformat(),
            model=model,
            doc_count=doc_count,
            chunk_count=chunk_count,
            doc_hashes=doc_hashes,
            snapshot_id=str(uuid.uuid4())
        )
