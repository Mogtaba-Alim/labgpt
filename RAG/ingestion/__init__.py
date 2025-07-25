#!/usr/bin/env python3
"""
Ingestion Package for LabGPT RAG System

This package provides enhanced document ingestion capabilities including:
- Structured chunk objects with metadata
- Semantic and structural text splitting
- Document structure extraction
- Quality-based filtering
"""

from .chunk_objects import ChunkMetadata, DocumentMetadata, ChunkStore
from .document_loader import DocumentLoader
from .text_splitter import SemanticStructuralSplitter
from .metadata_extractor import MetadataExtractor
from .quality_filter import QualityFilter

__all__ = [
    'ChunkMetadata',
    'DocumentMetadata', 
    'ChunkStore',
    'DocumentLoader',
    'SemanticStructuralSplitter',
    'MetadataExtractor',
    'QualityFilter'
] 