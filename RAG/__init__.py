"""
LabGPT RAG Package - Lean V2

A production-ready RAG (Retrieval-Augmented Generation) system with hybrid retrieval,
document-type adapters, and intelligent query processing.

Key Features:
- Hybrid retrieval (FAISS dense + BM25 sparse)
- Document-type specific chunking strategies
- Cross-encoder reranking
- PRF-style query expansion
- Cited-span extraction
- SHA256-based embedding cache
- Reproducibility snapshots
"""

# Core API
from .pipeline import RAGPipeline
from .models import (
    Chunk,
    RetrievalResult,
    SearchConfig,
    CitedSpan,
    DocumentMetadata,
    SnapshotInfo
)
from .snapshot import create_snapshot, load_snapshot, compare_snapshots

# Package metadata
__version__ = "2.0.0"
__author__ = "LabGPT Team"
__description__ = "LabGPT RAG Pipeline - Lean and Powerful"

__all__ = [
    # Main Pipeline
    "RAGPipeline",

    # Data Models
    "Chunk",
    "RetrievalResult",
    "SearchConfig",
    "CitedSpan",
    "DocumentMetadata",
    "SnapshotInfo",

    # Snapshot Utilities
    "create_snapshot",
    "load_snapshot",
    "compare_snapshots"
]


def get_version():
    """Get package version."""
    return __version__


def get_info():
    """Get package information."""
    return {
        "version": __version__,
        "description": __description__,
        "features": [
            "Hybrid Retrieval (FAISS + BM25)",
            "RRF Fusion",
            "Cross-Encoder Reranking",
            "PRF Query Expansion",
            "Micro Auto-K",
            "Document-Type Adapters",
            "SHA256 Embedding Cache",
            "Cited Span Extraction",
            "Reproducibility Snapshots"
        ]
    }
