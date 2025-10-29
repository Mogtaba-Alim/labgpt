"""
Ingestion Package for LabGPT RAG System

This package provides document ingestion capabilities including:
- Multi-format document loading
- Semantic and structural text splitting
- Document-type specific chunking strategies
- Lightweight quality filtering
- SHA256-based embedding caching
"""

from .document_loader import DocumentLoader
from .text_splitter import SemanticStructuralSplitter
from .doc_type_adapter import DocTypeAdapter
from .minimal_filter import MinimalFilter
from .embedding_manager import EmbeddingManager

__all__ = [
    'DocumentLoader',
    'SemanticStructuralSplitter',
    'DocTypeAdapter',
    'MinimalFilter',
    'EmbeddingManager'
]
