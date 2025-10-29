"""
pipeline.py

Unified RAGPipeline class - main entry point for the RAG system.
Simple API with sensible defaults and optional preset-based configuration.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

from .models import Chunk, RetrievalResult, SearchConfig, DocumentMetadata
from .ingestion.document_loader import DocumentLoader
from .ingestion.text_splitter import SemanticStructuralSplitter
from .ingestion.doc_type_adapter import DocTypeAdapter
from .ingestion.minimal_filter import MinimalFilter
from .ingestion.embedding_manager import EmbeddingManager
from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.prf_expansion import PRFQueryExpander
from .retrieval.micro_auto_k import MicroAutoK
from .generation.cited_span_extractor import CitedSpanExtractor
from .snapshot import create_snapshot, load_snapshot, get_latest_snapshot

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Unified RAG pipeline with simple API and sensible defaults.

    Usage:
        # Quick start (default preset)
        rag = RAGPipeline(index_dir="my_rag")
        rag.add_documents(["paper1.pdf", "paper2.pdf"])
        results = rag.search("How does CRISPR work?")

        # Research preset (adds query expansion + telemetry + auto-k)
        rag = RAGPipeline(index_dir="my_rag", preset="research")
        results = rag.search("compare CRISPR-Cas9 vs Cas13",
                           expand_query=True, cited_spans=True)
    """

    PRESETS = {
        "default": {
            "enable_query_expansion": False,
            "enable_auto_k": False,
            "enable_telemetry": False,
            "enable_doc_type_adapters": True,
            "top_k": 10,
            "rerank": True,
            "diversity_threshold": 0.85
        },
        "research": {
            "enable_query_expansion": True,
            "enable_auto_k": True,
            "enable_telemetry": True,
            "enable_doc_type_adapters": True,
            "top_k": 15,
            "rerank": True,
            "diversity_threshold": 0.85
        }
    }

    def __init__(self,
                 index_dir: str,
                 preset: str = "default",
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = "auto"):
        """
        Initialize RAG pipeline.

        Args:
            index_dir: Directory for storing index files
            preset: Configuration preset ("default" or "research")
            embedding_model: SentenceTransformer model name
            reranker_model: Cross-encoder model name
            device: Device for models ("cpu", "cuda", "mps", or "auto")
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Load preset configuration
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from: {list(self.PRESETS.keys())}")

        self.config = self.PRESETS[preset].copy()
        self.preset = preset

        logger.info(f"Initializing RAGPipeline with preset: {preset}")

        # Auto-detect device if needed
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            logger.info(f"Auto-detected device: {device}")

        # Store model names for later reference
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model

        # Initialize models
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)

        logger.info(f"Loading reranker model: {reranker_model}")
        self.reranker_model = CrossEncoder(reranker_model, device=device)

        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = SemanticStructuralSplitter()
        self.doc_type_adapter = DocTypeAdapter() if self.config["enable_doc_type_adapters"] else None
        self.minimal_filter = MinimalFilter()
        self.embedding_manager = EmbeddingManager(
            embedding_model=self.embedding_model,
            cache_dir=str(self.index_dir / "cache")
        )
        self.hybrid_retriever = HybridRetriever(
            embedding_model=self.embedding_model,
            reranker_model=self.reranker_model,
            index_dir=str(self.index_dir)
        )

        # Optional components (research preset)
        self.query_expander = None
        self.auto_k = None
        self.span_extractor = None

        if self.config["enable_query_expansion"]:
            self.query_expander = PRFQueryExpander(
                embedding_model=self.embedding_model,
                max_expansions=2,
                top_docs=3
            )

        if self.config["enable_auto_k"]:
            self.auto_k = MicroAutoK(
                increment=10,
                max_multiplier=2.0,
                variance_threshold=0.01
            )

        # Always initialize span extractor (used on-demand)
        self.span_extractor = CitedSpanExtractor(
            embedding_model=self.embedding_model,
            similarity_threshold=0.6,
            max_spans=3
        )

        # Telemetry (if enabled)
        self.telemetry = defaultdict(list) if self.config["enable_telemetry"] else None

        # Document tracking
        self.documents: List[DocumentMetadata] = []

        # Load existing document metadata if available
        self._load_document_metadata()

        logger.info("RAGPipeline initialized successfully")

    def add_documents(self, paths: List[str],
                     recursive: bool = True,
                     file_extensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add documents to the RAG index.

        Handles document loading, chunking, filtering, embedding, and indexing.

        Args:
            paths: List of file paths or directories
            recursive: Process directories recursively (default: True)
            file_extensions: File extensions to include (default: ['.pdf', '.txt', '.md'])

        Returns:
            Dictionary with ingestion statistics
        """
        start_time = time.time()

        if isinstance(paths, str):
            paths = [paths]

        logger.info(f"Adding documents from {len(paths)} path(s)")

        # Default file extensions
        if file_extensions is None:
            file_extensions = ['.pdf', '.txt', '.md', '.tex', '.rst',
                             '.py', '.r', '.c', '.cpp', '.ipynb']

        # Collect all file paths
        all_files = []
        for path in paths:
            path_obj = Path(path)
            if path_obj.is_file():
                all_files.append(str(path_obj))
            elif path_obj.is_dir():
                if recursive:
                    for ext in file_extensions:
                        all_files.extend([str(p) for p in path_obj.rglob(f"*{ext}")])
                else:
                    for ext in file_extensions:
                        all_files.extend([str(p) for p in path_obj.glob(f"*{ext}")])
            else:
                logger.warning(f"Path not found: {path}")

        logger.info(f"Found {len(all_files)} files to process")

        # Process each file
        all_chunks = []
        doc_count = 0

        for file_path in all_files:
            try:
                # Load document
                content, metadata = self.document_loader.load_document(file_path)

                # Extract document structure
                document_structure = self.document_loader.extract_document_structure(
                    content, metadata.doc_type
                )

                # Get doc-type specific chunking strategy
                if self.doc_type_adapter:
                    doc_type = self.doc_type_adapter.detect_type(file_path)
                    chunk_params = self.doc_type_adapter.get_splitting_params(file_path)
                    self.text_splitter.config.target_chunk_size = chunk_params['chunk_size']
                    self.text_splitter.config.chunk_overlap = chunk_params['chunk_overlap']

                # Prepare doc_metadata dictionary for text splitter
                doc_metadata_dict = {
                    'doc_id': metadata.doc_id,
                    'doc_type': metadata.doc_type,
                    'source_path': metadata.source_path
                }

                # Split into chunks
                chunks = self.text_splitter.split_document(
                    content, doc_metadata_dict, document_structure
                )

                # Filter chunks
                filtered_chunks = [c for c in chunks if self.minimal_filter.filter_chunk(c)]

                # Generate embeddings
                for chunk in filtered_chunks:
                    embedding = self.embedding_manager.get_embedding(chunk.text)
                    chunk.embedding = embedding

                all_chunks.extend(filtered_chunks)
                doc_count += 1

                # Track document (metadata is already a DocumentMetadata object)
                metadata.chunk_count = len(filtered_chunks)
                self.documents.append(metadata)

                logger.info(
                    f"Processed {file_path}: {len(chunks)} chunks "
                    f"({len(filtered_chunks)} after filtering)"
                )

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue

        # Build index
        logger.info(f"Building index with {len(all_chunks)} chunks")
        self.hybrid_retriever.build_index(all_chunks)

        elapsed_time = time.time() - start_time

        stats = {
            "files_processed": doc_count,
            "total_chunks": len(all_chunks),
            "time_seconds": elapsed_time,
            "chunks_per_second": len(all_chunks) / elapsed_time if elapsed_time > 0 else 0
        }

        logger.info(
            f"Ingestion complete: {doc_count} files, {len(all_chunks)} chunks "
            f"in {elapsed_time:.2f}s"
        )

        # Save document metadata to disk
        self._save_document_metadata()

        return stats

    def search(self,
              query: str,
              top_k: Optional[int] = None,
              expand_query: bool = False,
              verify_citations: bool = False,
              cited_spans: bool = False,
              rerank: Optional[bool] = None,
              diversity_threshold: Optional[float] = None) -> List[RetrievalResult]:
        """
        Search the RAG index.

        Args:
            query: Search query
            top_k: Number of results to return (default: from preset)
            expand_query: Enable PRF query expansion (default: from preset)
            verify_citations: Enable citation verification (default: False)
            cited_spans: Extract cited spans from results (default: False)
            rerank: Enable cross-encoder reranking (default: from preset)
            diversity_threshold: Similarity threshold for deduplication (default: from preset)

        Returns:
            List of RetrievalResult objects sorted by score
        """
        start_time = time.time()

        # Use preset defaults if not specified
        if top_k is None:
            top_k = self.config["top_k"]
        if rerank is None:
            rerank = self.config["rerank"]
        if diversity_threshold is None:
            diversity_threshold = self.config["diversity_threshold"]

        # Expand query if requested
        expanded_query = query
        if expand_query and self.query_expander:
            # Get initial BM25 results for PRF
            initial_results = self.hybrid_retriever.retrieve(query, top_k=5, method="sparse")
            expanded_query = self.query_expander.expand_query(query, initial_results)
            logger.info(f"Expanded query: '{query}' -> '{expanded_query}'")

        # Retrieve results
        results = self.hybrid_retriever.retrieve(
            expanded_query,
            top_k=top_k * 3 if rerank else top_k,  # Over-retrieve for reranking
            diversity_threshold=diversity_threshold
        )

        # Apply micro auto-k adjustment
        if self.auto_k:
            # Extract scores from results
            fused_scores = [r.score for r in results]
            adjusted_k = self.auto_k.adjust_top_k(query, top_k, fused_scores)

            if adjusted_k != top_k:
                logger.info(f"Micro auto-k adjusted top_k: {top_k} -> {adjusted_k}")
                top_k = adjusted_k

        # Rerank if enabled
        if rerank:
            # Create query-document pairs for reranking
            pairs = [(query, r.chunk.text) for r in results]
            rerank_scores = self.reranker_model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False
            )

            # Update scores and re-sort
            for i, result in enumerate(results):
                result.score = float(rerank_scores[i])
                result.metadata = result.metadata or {}
                result.metadata['reranked'] = True

            results.sort(key=lambda x: x.score, reverse=True)

            # Update ranks
            for i, result in enumerate(results[:top_k]):
                result.rank = i + 1

        # Trim to top_k
        results = results[:top_k]

        # Extract cited spans if requested
        if cited_spans and self.span_extractor:
            for result in results:
                spans = self.span_extractor.extract_cited_spans(query, result.chunk)
                result.metadata = result.metadata or {}
                result.metadata['cited_spans'] = spans

        elapsed_time = time.time() - start_time

        # Track telemetry
        if self.telemetry is not None:
            self.telemetry['search_times'].append(elapsed_time)
            self.telemetry['query_lengths'].append(len(query.split()))
            if expand_query and self.query_expander and expanded_query != query:
                expansion_terms = len(expanded_query.split()) - len(query.split())
                self.telemetry['expansion_terms'].append(expansion_terms)
            if self.auto_k and adjusted_k != self.config["top_k"]:
                self.telemetry['auto_k_activations'].append(1)

        logger.info(f"Search completed in {elapsed_time:.3f}s: {len(results)} results")

        return results

    def snapshot(self) -> str:
        """
        Create a reproducibility snapshot of the current index.

        Returns:
            Path to created snapshot file
        """
        doc_paths = [doc.source_path for doc in self.documents]
        chunk_count = sum(doc.chunk_count for doc in self.documents)

        snapshot_path = create_snapshot(
            index_dir=str(self.index_dir),
            model_name=self.embedding_model_name,
            doc_paths=doc_paths,
            chunk_count=chunk_count
        )

        logger.info(f"Snapshot created: {snapshot_path}")
        return snapshot_path

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics and telemetry.

        Returns:
            Dictionary with stats (varies by preset)
        """
        stats = {
            "preset": self.preset,
            "index_dir": str(self.index_dir),
            "document_count": len(self.documents),
            "total_chunks": sum(doc.chunk_count for doc in self.documents)
        }

        # Add embedding cache stats
        if hasattr(self.embedding_manager, 'get_cache_stats'):
            stats['cache_stats'] = self.embedding_manager.get_cache_stats()

        # Add telemetry if available
        if self.telemetry:
            search_times = self.telemetry.get('search_times', [])
            if search_times:
                stats['avg_search_time'] = np.mean(search_times)
                stats['total_searches'] = len(search_times)

            expansion_terms = self.telemetry.get('expansion_terms', [])
            if expansion_terms:
                stats['avg_expansion_terms'] = np.mean(expansion_terms)

            auto_k_activations = self.telemetry.get('auto_k_activations', [])
            if auto_k_activations:
                stats['auto_k_activations'] = len(auto_k_activations)

            # Cache hit rate
            if 'cache_stats' in stats:
                cache_stats = stats['cache_stats']
                total_requests = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
                if total_requests > 0:
                    stats['cache_hit_rate'] = cache_stats.get('hits', 0) / total_requests

        return stats

    def status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.

        Returns:
            Dictionary with status information
        """
        latest_snapshot = get_latest_snapshot(str(self.index_dir))

        status = {
            "initialized": True,
            "index_dir": str(self.index_dir),
            "preset": self.preset,
            "documents": len(self.documents),
            "chunks": sum(doc.chunk_count for doc in self.documents),
            "embedding_model": self.embedding_model_name,
            "has_index": self.hybrid_retriever.is_built(),
            "latest_snapshot": latest_snapshot
        }

        return status

    def clear(self):
        """Clear the index and all cached data."""
        logger.warning("Clearing index and cache")
        self.hybrid_retriever.clear_index()
        self.embedding_manager.clear_cache()
        self.documents.clear()
        if self.telemetry:
            self.telemetry.clear()

        # Delete document metadata file
        metadata_file = self.index_dir / "documents_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()

        logger.info("Index and cache cleared")

    def _save_document_metadata(self) -> None:
        """Save document metadata to disk."""
        import json

        metadata_file = self.index_dir / "documents_metadata.json"

        # Convert DocumentMetadata objects to dictionaries
        docs_data = []
        for doc in self.documents:
            doc_dict = {
                "doc_id": doc.doc_id,
                "source_path": doc.source_path,
                "doc_type": doc.doc_type,
                "title": doc.title,
                "page_count": doc.page_count,
                "chunk_count": doc.chunk_count
            }
            docs_data.append(doc_dict)

        # Save to JSON
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2)

        logger.debug(f"Saved metadata for {len(docs_data)} documents to {metadata_file}")

    def _load_document_metadata(self) -> None:
        """Load document metadata from disk if it exists."""
        import json

        metadata_file = self.index_dir / "documents_metadata.json"

        if not metadata_file.exists():
            logger.debug("No existing document metadata found")
            return

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)

            # Convert dictionaries back to DocumentMetadata objects
            self.documents = []
            for doc_dict in docs_data:
                doc = DocumentMetadata(
                    doc_id=doc_dict["doc_id"],
                    source_path=doc_dict["source_path"],
                    doc_type=doc_dict["doc_type"],
                    title=doc_dict.get("title"),
                    page_count=doc_dict.get("page_count"),
                    chunk_count=doc_dict.get("chunk_count", 0)
                )
                self.documents.append(doc)

            logger.info(f"Loaded metadata for {len(self.documents)} documents from disk")

        except Exception as e:
            logger.warning(f"Failed to load document metadata: {e}")
            self.documents = []
