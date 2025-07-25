#!/usr/bin/env python3
"""
enhanced_ingestion.py

Unified enhanced ingestion pipeline integrating all advanced RAG capabilities:

Core Features (P1):
- Structured document loading with metadata extraction
- Semantic and structural text splitting
- Quality filtering and assessment
- Hybrid indexing for both dense and sparse retrieval
- Configurable processing pipeline

Advanced Features (P2):
- Advanced chunk scoring and pruning
- SHA256-based embedding caching
- Index versioning and rollback
- Comprehensive retrieval telemetry
- Incremental index updates

Intelligent Features (P3/P4):
- Adaptive top-k retrieval with coverage heuristics
- Answer guardrails with citation verification
- Per-document embedding management
- Git-based change tracking
"""

import os
import logging
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import asdict
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer

# Core ingestion components
from .ingestion import (
    ChunkMetadata, DocumentMetadata, ChunkStore,
    DocumentLoader, SemanticStructuralSplitter, 
    MetadataExtractor, QualityFilter
)
from .ingestion.text_splitter import SplittingConfig

# Advanced P2 components
from .ingestion.advanced_scoring import AdvancedChunkScorer, IntelligentPruner, ScoringConfig
from .ingestion.embedding_cache import EmbeddingCache, CachedEmbeddingGenerator
from .ingestion.index_versioning import IndexVersionManager
from .ingestion.incremental_updates import IncrementalIndexUpdater, DocumentTracker, UpdateCoordinator

# P3/P4 intelligent components
from .ingestion.per_document_management import PerDocumentEmbeddingManager, FAISSIndexReconstructor
from .retrieval.adaptive_retrieval import AdaptiveTopKRetriever, AdaptiveConfig, CoverageMetrics
from .generation.answer_guardrails import AnswerGuardrails, GuardrailsConfig

# Retrieval system
from .retrieval import RetrievalConfig, HybridRetriever
from .retrieval.telemetry import RetrievalTelemetry

logger = logging.getLogger(__name__)

class EnhancedIngestionPipeline:
    """
    Unified enhanced ingestion pipeline with all advanced RAG capabilities.
    
    This single pipeline integrates all features from the complete system overhaul:
    - Core document processing and semantic chunking
    - Advanced quality assessment and intelligent pruning
    - Embedding caching and incremental processing
    - Index versioning and rollback capabilities
    - Comprehensive monitoring and telemetry
    - Adaptive retrieval with coverage analysis
    - Answer verification with citation guardrails
    - Per-document embedding management
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 storage_dir: str = "enhanced_storage",
                 # Core features
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 # Advanced features (P2)
                 enable_advanced_scoring: bool = True,
                 enable_caching: bool = True,
                 enable_versioning: bool = True,
                 enable_telemetry: bool = True,
                 enable_incremental: bool = True,
                 # Intelligent features (P3/P4)
                 enable_adaptive_retrieval: bool = True,
                 enable_answer_guardrails: bool = True,
                 enable_per_doc_management: bool = True,
                 enable_git_tracking: bool = True):
        
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = RetrievalConfig.from_yaml(config_path)
        else:
            self.config = RetrievalConfig()
            logger.info("Using default configuration")
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Store model name for later use
        self.embedding_model_name = embedding_model_name
        
        # Feature enablement flags
        self.enable_advanced_scoring = enable_advanced_scoring
        self.enable_caching = enable_caching
        self.enable_versioning = enable_versioning
        self.enable_telemetry = enable_telemetry
        self.enable_incremental = enable_incremental
        self.enable_adaptive_retrieval = enable_adaptive_retrieval
        self.enable_answer_guardrails = enable_answer_guardrails
        self.enable_per_doc_management = enable_per_doc_management
        self.enable_git_tracking = enable_git_tracking
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        
        # Handle device selection
        device = self.config.models.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        
        # Initialize core components
        self._init_core_components()
        
        # Initialize advanced components (P2)
        self._init_advanced_components()
        
        # Initialize intelligent components (P3/P4)
        self._init_intelligent_components()
        
        # Finalize initialization with cross-dependencies
        self._finalize_initialization()
        
        # Processing statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'chunks_pruned': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_time': 0.0,
            'adaptive_retrievals': 0,
            'verified_answers': 0,
            'per_doc_updates': 0
        }
        
        logger.info("Enhanced ingestion pipeline initialized with all advanced features")
    
    def _init_core_components(self):
        """Initialize core P1 components"""
        # Document loading and processing
        self.document_loader = DocumentLoader()
        
        # Text splitter with configuration
        self.text_splitter = SemanticStructuralSplitter(self.config.splitting)
        
        self.metadata_extractor = MetadataExtractor()
        
        # Quality filter with configuration
        self.quality_filter = QualityFilter(
            min_quality_score=self.config.quality.min_quality_score,
            language=self.config.quality.language
        )
        
        # Storage system
        self.chunk_store = ChunkStore(self.storage_dir / "chunks")
        
        logger.info("Core components initialized")
    
    def _init_advanced_components(self):
        """Initialize advanced P2 components"""
        if self.enable_advanced_scoring:
            self.chunk_scorer = AdvancedChunkScorer(ScoringConfig())
            self.intelligent_pruner = IntelligentPruner()
            logger.info("Advanced scoring and pruning enabled")
        
        if self.enable_caching:
            cache_dir = self.storage_dir / "embedding_cache"
            self.embedding_cache = EmbeddingCache(cache_dir=cache_dir)
            self.cached_embedding_generator = CachedEmbeddingGenerator(
                self.embedding_model_name, self.embedding_cache
            )
            logger.info("Embedding caching enabled")
        
        if self.enable_versioning:
            version_dir = self.storage_dir / "versions"
            self.version_manager = IndexVersionManager(version_dir)
            logger.info("Index versioning enabled")
        
        if self.enable_telemetry:
            telemetry_dir = self.storage_dir / "telemetry"
            self.telemetry = RetrievalTelemetry(telemetry_dir)
            logger.info("Retrieval telemetry enabled")
        
        if self.enable_incremental:
            self.document_tracker = DocumentTracker(self.storage_dir / "document_tracking")
            # Incremental updater will be initialized later in _finalize_initialization
            logger.info("Incremental updates prepared")
    
    def _init_intelligent_components(self):
        """Initialize intelligent P3/P4 components"""
        if self.enable_per_doc_management:
            doc_storage_dir = self.storage_dir / "per_document"
            self.per_doc_manager = PerDocumentEmbeddingManager(
                storage_dir=str(doc_storage_dir),
                enable_git_tracking=self.enable_git_tracking
            )
            self.faiss_reconstructor = FAISSIndexReconstructor()
            logger.info("Per-document management enabled")
        
        if self.enable_adaptive_retrieval:
            self.adaptive_config = AdaptiveConfig()
            logger.info("Adaptive retrieval prepared")
        
        if self.enable_answer_guardrails:
            self.guardrails_config = GuardrailsConfig()
            self.answer_guardrails = AnswerGuardrails(
                self.guardrails_config, 
                self.embedding_model
            )
            logger.info("Answer guardrails enabled")
    
    def _config_to_dict(self, config_obj: Any) -> Dict[str, Any]:
        """Convert configuration object to JSON-serializable dictionary"""
        if hasattr(config_obj, '__dict__'):
            result = {}
            for key, value in config_obj.__dict__.items():
                if hasattr(value, '__dict__'):  # Nested dataclass
                    result[key] = self._config_to_dict(value)
                else:
                    result[key] = value
            return result
        else:
            return str(config_obj)  # Fallback for non-serializable objects
    
    def _finalize_initialization(self):
        """Finalize initialization of components with cross-dependencies"""
        if self.enable_incremental:
            # Initialize incremental updater with proper dependencies
            embedding_gen = self.cached_embedding_generator if self.enable_caching else None
            self.incremental_updater = IncrementalIndexUpdater(
                chunk_store=self.chunk_store,
                embedding_generator=embedding_gen,
                config={},  # Use empty config for now
                update_dir=str(self.storage_dir / "incremental_updates")
            )
            logger.info("Incremental updates fully initialized")
    
    def process_documents(self, 
                         document_paths: List[str],
                         batch_size: int = 32) -> Dict[str, Any]:
        """
        Process documents through the complete enhanced pipeline
        
        Args:
            document_paths: List of document file paths
            batch_size: Batch size for embedding generation
            
        Returns:
            Processing results with comprehensive statistics
        """
        start_time = time.time()
        all_chunks = []
        all_embeddings = []
        processed_docs = []
        
        logger.info(f"Processing {len(document_paths)} documents")
        
        for doc_path in document_paths:
            try:
                # Load document
                doc_result = self.document_loader.load_document(doc_path)
                if not doc_result:
                    logger.warning(f"Failed to load document: {doc_path}")
                    continue
                
                content, doc_metadata = doc_result
                
                # Split text into chunks
                chunks = self.text_splitter.split_document(
                    content, 
                    doc_metadata.__dict__,
                    getattr(doc_metadata, 'structure', None)
                )
                if not chunks:
                    logger.warning(f"No chunks extracted from: {doc_path}")
                    continue
                
                # Extract metadata for chunks
                chunks = [self.metadata_extractor.extract_metadata(chunk) for chunk in chunks]
                
                # Apply quality filtering
                chunks = self.quality_filter.filter_chunks(chunks)
                
                # Apply advanced scoring and pruning if enabled
                if self.enable_advanced_scoring:
                    # Score chunks
                    for chunk in chunks:
                        scored_chunk = self.chunk_scorer.score_chunk_advanced(chunk)
                        chunk.quality_score = getattr(scored_chunk, 'quality_score', 0.5)
                        chunk.domain_scores = getattr(scored_chunk, 'domain_scores', {})
                    
                    # Intelligent pruning
                    chunks = self.intelligent_pruner.prune_chunks(chunks)
                    self.stats['chunks_pruned'] += len(chunks)
                
                # Generate embeddings (with caching if enabled)
                if self.enable_caching:
                    embeddings = self.cached_embedding_generator.encode_batch(
                        [chunk.text for chunk in chunks], batch_size=batch_size
                    )
                    # Get cache statistics if available
                    if hasattr(self.cached_embedding_generator, '_generation_stats'):
                        stats = self.cached_embedding_generator._generation_stats
                        self.stats['cache_hits'] += stats.get('cache_hits', 0)
                        self.stats['cache_misses'] += stats.get('cache_misses', 0)
                else:
                    texts = [chunk.text for chunk in chunks]
                    embeddings = self.embedding_model.encode(texts, batch_size=batch_size)
                
                # Store in per-document management if enabled
                if self.enable_per_doc_management:
                    self.per_doc_manager.store_document_embeddings(
                        doc_metadata, chunks, embeddings
                    )
                    self.stats['per_doc_updates'] += 1
                
                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)
                processed_docs.append(doc_metadata)
                
                self.stats['documents_processed'] += 1
                self.stats['chunks_created'] += len(chunks)
                
                logger.info(f"Processed {doc_path}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {doc_path}: {e}")
                continue
        
        # Store chunks and embeddings
        if all_chunks:
            self.chunk_store.store_chunks(all_chunks)
            embeddings_array = np.array(all_embeddings)
            embeddings_path = self.storage_dir / "embeddings.npy"
            np.save(embeddings_path, embeddings_array)
            
            # Create version snapshot if enabled
            if self.enable_versioning:
                version_id = self.version_manager.create_version(
                    index_files={
                        "chunks": str(self.chunk_store.chunks_file),
                        "embeddings": str(embeddings_path)
                    },
                    config=self._config_to_dict(self.config),
                    model_version=self.embedding_model_name,
                    chunk_count=len(all_chunks),
                    embedding_dim=embeddings_array.shape[1] if len(embeddings_array.shape) > 1 else 0,
                    index_type="hybrid",
                    doc_snapshot_hash=hashlib.md5(str(all_chunks).encode()).hexdigest()[:16],
                    description=f"Processed {len(processed_docs)} documents"
                )
                logger.info(f"Created version snapshot: {version_id}")
        
        processing_time = time.time() - start_time
        self.stats['processing_time'] = processing_time
        
        results = {
            'success': True,
            'documents_processed': len(processed_docs),
            'total_chunks': len(all_chunks),
            'processing_time': processing_time,
            'stats': self.stats.copy()
        }
        
        logger.info(f"Document processing completed in {processing_time:.2f}s")
        return results
    
    def build_retrieval_system(self) -> HybridRetriever:
        """
        Build the complete hybrid retrieval system with all enhancements
        
        Returns:
            Configured HybridRetriever with all advanced features
        """
        # Load chunks and embeddings
        chunks = self.chunk_store.load_chunks()
        embeddings_path = self.storage_dir / "embeddings.npy"
        
        if not embeddings_path.exists():
            raise FileNotFoundError("Embeddings not found. Run process_documents first.")
        
        embeddings = np.load(embeddings_path)
        
        # Create hybrid retriever (indices are built automatically)
        retriever = HybridRetriever(
            chunk_store=self.chunk_store,
            config=self.config,
            embedding_model=self.embedding_model
        )
        
        logger.info("Hybrid retrieval system built successfully")
        return retriever
    
    def create_adaptive_retriever(self, base_retriever: HybridRetriever) -> Optional[AdaptiveTopKRetriever]:
        """
        Create adaptive retriever if enabled
        
        Args:
            base_retriever: Base hybrid retriever
            
        Returns:
            AdaptiveTopKRetriever if enabled, None otherwise
        """
        if not self.enable_adaptive_retrieval:
            return None
        
        adaptive_retriever = AdaptiveTopKRetriever(
            base_retriever=base_retriever,
            config=self.adaptive_config,
            embedding_model=self.embedding_model
        )
        
        logger.info("Adaptive retriever created")
        return adaptive_retriever
    
    def verify_answer(self, 
                     generated_text: str,
                     source_chunks: List[ChunkMetadata]) -> Optional[Dict[str, Any]]:
        """
        Verify generated answer using guardrails if enabled
        
        Args:
            generated_text: Generated answer text
            source_chunks: Source chunks used for generation
            
        Returns:
            Verification results if enabled, None otherwise
        """
        if not self.enable_answer_guardrails:
            return None
        
        verification_result = self.answer_guardrails.verify_answer(
            generated_text, source_chunks
        )
        
        if verification_result['verification_status'] == 'passed':
            self.stats['verified_answers'] += 1
        
        return verification_result
    
    def incremental_update(self, 
                          changed_documents: List[str]) -> Dict[str, Any]:
        """
        Perform incremental update for changed documents
        
        Args:
            changed_documents: List of changed document paths
            
        Returns:
            Update results
        """
        if not self.enable_incremental:
            logger.warning("Incremental updates not enabled")
            return {'success': False, 'reason': 'Feature not enabled'}
        
        start_time = time.time()
        
        # Process changed documents
        update_results = self.process_documents(changed_documents)
        
        # Update indices incrementally
        if self.enable_per_doc_management:
            # Use per-document management for granular updates
            for doc_path in changed_documents:
                self.per_doc_manager.update_document(doc_path)
                self.stats['per_doc_updates'] += 1
        
        update_time = time.time() - start_time
        
        results = {
            'success': True,
            'documents_updated': len(changed_documents),
            'update_time': update_time,
            'update_results': update_results
        }
        
        logger.info(f"Incremental update completed in {update_time:.2f}s")
        return results
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        system_stats = {
            'pipeline_stats': self.stats.copy(),
            'feature_status': {
                'advanced_scoring': self.enable_advanced_scoring,
                'caching': self.enable_caching,
                'versioning': self.enable_versioning,
                'telemetry': self.enable_telemetry,
                'incremental': self.enable_incremental,
                'adaptive_retrieval': self.enable_adaptive_retrieval,
                'answer_guardrails': self.enable_answer_guardrails,
                'per_doc_management': self.enable_per_doc_management
            }
        }
        
        # Add component-specific stats if available
        if hasattr(self, 'embedding_cache'):
            system_stats['cache_stats'] = self.embedding_cache.get_statistics()
        
        if hasattr(self, 'telemetry'):
            try:
                # Get telemetry metrics
                real_time_metrics = self.telemetry.get_real_time_metrics()
                system_stats['telemetry_stats'] = {
                    'total_queries': real_time_metrics.total_queries,
                    'avg_response_time': real_time_metrics.avg_response_time,
                    'p95_response_time': real_time_metrics.p95_response_time,
                    'cache_hit_rate': real_time_metrics.cache_hit_rate
                }
            except Exception as e:
                logger.warning(f"Could not retrieve telemetry stats: {e}")
                system_stats['telemetry_stats'] = {'error': 'No telemetry data available'}
        
        if hasattr(self, 'answer_guardrails'):
            system_stats['guardrails_stats'] = self.answer_guardrails.verification_stats
        
        return system_stats
    
    def export_configuration(self, export_path: str):
        """Export current configuration to file"""
        config_dict = {
            'retrieval_config': asdict(self.config),
            'feature_flags': {
                'advanced_scoring': self.enable_advanced_scoring,
                'caching': self.enable_caching,
                'versioning': self.enable_versioning,
                'telemetry': self.enable_telemetry,
                'incremental': self.enable_incremental,
                'adaptive_retrieval': self.enable_adaptive_retrieval,
                'answer_guardrails': self.enable_answer_guardrails,
                'per_doc_management': self.enable_per_doc_management
            },
            'statistics': self.get_system_statistics()
        }
        
        with open(export_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuration exported to {export_path}")


def main():
    """
    Command-line interface for the enhanced ingestion pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RAG Ingestion Pipeline")
    parser.add_argument("--docs", nargs="+", required=True, 
                       help="Document paths to process")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--storage", default="enhanced_storage", 
                       help="Storage directory")
    parser.add_argument("--disable-caching", action="store_true",
                       help="Disable embedding caching")
    parser.add_argument("--disable-adaptive", action="store_true",
                       help="Disable adaptive retrieval")
    parser.add_argument("--disable-guardrails", action="store_true",
                       help="Disable answer guardrails")
    parser.add_argument("--test-query", help="Test query for retrieval")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize pipeline
    pipeline = EnhancedIngestionPipeline(
        config_path=args.config,
        storage_dir=args.storage,
        enable_caching=not args.disable_caching,
        enable_adaptive_retrieval=not args.disable_adaptive,
        enable_answer_guardrails=not args.disable_guardrails
    )
    
    # Process documents
    results = pipeline.process_documents(args.docs)
    print(f"Processing Results: {results}")
    
    # Build retrieval system
    retriever = pipeline.build_retrieval_system()
    
    # Test query if provided
    if args.test_query:
        print(f"\nTesting query: {args.test_query}")
        
        # Standard retrieval
        results = retriever.retrieve(args.test_query, top_k=5)
        print(f"Standard retrieval: {len(results)} results")
        
        # Adaptive retrieval if enabled
        if not args.disable_adaptive:
            adaptive_retriever = pipeline.create_adaptive_retriever(retriever)
            if adaptive_retriever:
                adaptive_results, coverage = adaptive_retriever.retrieve_adaptive(args.test_query)
                print(f"Adaptive retrieval: {len(adaptive_results)} results, coverage: {coverage.overall_coverage:.3f}")
    
    # Export final statistics
    stats_path = Path(args.storage) / "pipeline_statistics.json"
    pipeline.export_configuration(str(stats_path))
    print(f"Statistics exported to: {stats_path}")


if __name__ == "__main__":
    main() 