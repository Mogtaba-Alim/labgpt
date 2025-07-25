#!/usr/bin/env python3
"""
RAG Package - Complete Enhanced RAG System

This package provides a comprehensive RAG (Retrieval-Augmented Generation) system
with state-of-the-art capabilities implemented in a unified architecture:

Core System:
- Hybrid retrieval (dense + sparse)
- Semantic text splitting
- Rich metadata management
- Quality filtering and assessment

Advanced Features:
- Advanced chunk scoring and pruning
- SHA256-based embedding caching
- Index versioning and rollback
- Comprehensive telemetry
- Incremental updates

Intelligent Features:
- Adaptive top-k retrieval with coverage heuristics
- Answer guardrails with citation verification
- Per-document embedding management
- Git-based change detection
- FAISS reconstruction with tombstone management
"""

# Core Components
from .ingestion import (
    ChunkMetadata,
    DocumentMetadata, 
    ChunkStore,
    DocumentLoader,
    SemanticStructuralSplitter,
    MetadataExtractor,
    QualityFilter
)

from .retrieval import (
    HybridRetriever,
    QueryProcessor,
    QueryExpander,
    CrossEncoderReranker,
    RetrievalConfig,
    ResultFusion
)

# Advanced P2 Components
from .ingestion.advanced_scoring import AdvancedChunkScorer, IntelligentPruner, ScoringConfig
from .ingestion.embedding_cache import EmbeddingCache, CachedEmbeddingGenerator
from .ingestion.index_versioning import IndexVersionManager
from .ingestion.incremental_updates import IncrementalIndexUpdater, DocumentTracker, UpdateCoordinator
from .retrieval.telemetry import RetrievalTelemetry

# Intelligent P3/P4 Components
from .retrieval.adaptive_retrieval import AdaptiveTopKRetriever, AdaptiveConfig, CoverageMetrics
from .generation.answer_guardrails import AnswerGuardrails, GuardrailsConfig
from .ingestion.per_document_management import PerDocumentEmbeddingManager, FAISSIndexReconstructor

# Unified Main Pipeline
from .enhanced_ingestion import EnhancedIngestionPipeline

# Package metadata
__version__ = "1.0.0"
__author__ = "LabGPT Development Team"
__description__ = "Complete Enhanced RAG System with Advanced Intelligence"

# Main exports for easy access
__all__ = [
    # Main Pipeline
    "EnhancedIngestionPipeline",
    
    # Core Components
    "ChunkMetadata",
    "DocumentMetadata", 
    "ChunkStore",
    "DocumentLoader",
    "SemanticStructuralSplitter",
    "MetadataExtractor",
    "QualityFilter",
    "HybridRetriever",
    "QueryProcessor",
    "QueryExpander", 
    "CrossEncoderReranker",
    "RetrievalConfig",
    "ResultFusion",
    
    # Advanced Components
    "AdvancedChunkScorer",
    "IntelligentPruner",
    "ScoringConfig",
    "EmbeddingCache",
    "CachedEmbeddingGenerator",
    "IndexVersionManager",
    "IncrementalIndexUpdater",
    "DocumentTracker",
    "UpdateCoordinator",
    "RetrievalTelemetry",
    
    # Intelligent Components
    "AdaptiveTopKRetriever",
    "AdaptiveConfig",
    "CoverageMetrics",
    "AnswerGuardrails",
    "GuardrailsConfig",
    "PerDocumentEmbeddingManager",
    "FAISSIndexReconstructor"
]

def get_version():
    """Get package version"""
    return __version__

def get_system_info():
    """Get system information"""
    return {
        "version": __version__,
        "description": __description__,
        "features": [
            "Hybrid Retrieval (Dense + Sparse)",
            "Semantic Text Splitting",
            "Advanced Quality Assessment",
            "Embedding Caching with SHA256",
            "Index Versioning and Rollback",
            "Comprehensive Telemetry",
            "Incremental Updates",
            "Adaptive Top-K Retrieval",
            "Answer Guardrails with Citation Verification",
            "Per-Document Embedding Management",
            "Git-based Change Detection"
        ],
        "components": len(__all__)
    }

# Configuration for easy pipeline setup
DEFAULT_PIPELINE_CONFIG = {
    "enable_advanced_scoring": True,
    "enable_caching": True,
    "enable_versioning": True,
    "enable_telemetry": True,
    "enable_incremental": True,
    "enable_adaptive_retrieval": True,
    "enable_answer_guardrails": True,
    "enable_per_doc_management": True,
    "enable_git_tracking": True
}

def create_pipeline(storage_dir: str = "enhanced_storage", **kwargs):
    """
    Create a fully configured EnhancedIngestionPipeline with all features enabled
    
    Args:
        storage_dir: Storage directory for the pipeline
        **kwargs: Additional configuration options
        
    Returns:
        Configured EnhancedIngestionPipeline instance
    """
    config = DEFAULT_PIPELINE_CONFIG.copy()
    config.update(kwargs)
    
    return EnhancedIngestionPipeline(
        storage_dir=storage_dir,
        **config
    )

# Quick access functions
def create_basic_pipeline(storage_dir: str = "basic_storage"):
    """Create a basic pipeline with only core features"""
    return EnhancedIngestionPipeline(
        storage_dir=storage_dir,
        enable_advanced_scoring=False,
        enable_caching=False,
        enable_versioning=False,
        enable_telemetry=False,
        enable_incremental=False,
        enable_adaptive_retrieval=False,
        enable_answer_guardrails=False,
        enable_per_doc_management=False
    )

def create_production_pipeline(storage_dir: str = "production_storage"):
    """Create a production-ready pipeline with all enterprise features"""
    return EnhancedIngestionPipeline(
        storage_dir=storage_dir,
        enable_advanced_scoring=True,
        enable_caching=True,
        enable_versioning=True,
        enable_telemetry=True,
        enable_incremental=True,
        enable_adaptive_retrieval=True,
        enable_answer_guardrails=True,
        enable_per_doc_management=True,
        enable_git_tracking=True
    )

# Module initialization message
import logging
logger = logging.getLogger(__name__)
logger.info(f"RAG Package v{__version__} initialized - Enhanced RAG System with Advanced Intelligence") 