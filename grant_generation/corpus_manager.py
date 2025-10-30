"""
corpus_manager.py

Manages per-project RAG indices with base corpus integration.
Handles document indexing using RAG/pipeline.py with embedding cache benefits.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional
from RAG.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class GrantCorpusManager:
    """
    Manages RAG indices for grant generation projects.

    Features:
    - Copy base lab corpus index on project creation
    - Add user documents to project-specific index
    - Benefit from embedding cache (95%+ cache hit rate on re-indexing)
    - Use RAG/pipeline.py for all indexing operations
    """

    def __init__(self, base_index_dir: Optional[str], projects_dir: str):
        """
        Initialize corpus manager.

        Args:
            base_index_dir: Path to shared RAG index from main pipeline.
                           Updated every time new files are added to base corpus.
                           Can be None if no base corpus exists.
            projects_dir: Directory for storing per-project indices.
        """
        self.base_index_dir = Path(base_index_dir) if base_index_dir else None
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

        if self.base_index_dir:
            logger.info(f"Base corpus index: {self.base_index_dir}")
        else:
            logger.info("No base corpus configured (will start projects from scratch)")

    def create_project(self, project_id: str, copy_base: bool = True) -> str:
        """
        Create new project RAG index directory.

        If copy_base=True and base index exists, copies:
        - FAISS index files (faiss_index.bin)
        - Chunk files (chunks.npy, embeddings.npy)
        - Document metadata (documents_metadata.json)
        - BM25 index (bm25_index.pkl)
        - Embedding cache directory (cache/)

        Copying the cache means new documents only generate embeddings for new chunks.
        Re-indexing achieves 95%+ cache hit rate.

        Args:
            project_id: Unique project identifier
            copy_base: Whether to copy base corpus index

        Returns:
            Path to project index directory
        """
        project_index_dir = self.projects_dir / project_id / "rag_index"
        project_index_dir.mkdir(parents=True, exist_ok=True)

        if copy_base and self.base_index_dir and self.base_index_dir.exists():
            logger.info(f"Copying base corpus to project {project_id}")

            # Files to copy
            index_files = [
                'faiss_index.bin',
                'chunks.npy',
                'embeddings.npy',
                'documents_metadata.json',
                'bm25_index.pkl'
            ]

            for file in index_files:
                src = self.base_index_dir / file
                if src.exists():
                    dst = project_index_dir / file
                    try:
                        shutil.copy2(src, dst)
                        logger.debug(f"Copied {file} to project index")
                    except Exception as e:
                        logger.warning(f"Could not copy {file}: {e}")

            # Copy embedding cache directory (critical for cache hit rate)
            cache_src = self.base_index_dir / "cache"
            if cache_src.exists():
                cache_dst = project_index_dir / "cache"
                try:
                    shutil.copytree(cache_src, cache_dst, dirs_exist_ok=True)
                    logger.info(f"Copied embedding cache ({len(list(cache_src.glob('*')))} files)")
                except Exception as e:
                    logger.warning(f"Could not copy cache: {e}")

            logger.info(f"Base corpus copied to {project_index_dir}")
        else:
            logger.info(f"Created empty index directory for project {project_id}")

        return str(project_index_dir)

    def add_documents(self, project_id: str, file_paths: List[str]) -> Dict:
        """
        Add user documents to project RAG index.

        Uses RAG/pipeline.py with embedding cache. Benefits:
        - 95%+ cache hit rate if re-indexing with base corpus
        - Only new chunks require embedding generation
        - Fast incremental updates

        Args:
            project_id: Project identifier
            file_paths: List of file paths to index

        Returns:
            Statistics dict with keys:
            - files_processed: Number of files successfully indexed
            - total_chunks: Total chunks created
            - time_seconds: Time taken for indexing
            - chunks_per_second: Throughput metric
            - cache_hits: Number of embeddings retrieved from cache
            - cache_misses: Number of new embeddings generated

        Raises:
            FileNotFoundError: If project index directory doesn't exist
            Exception: If indexing fails
        """
        project_index_dir = self.projects_dir / project_id / "rag_index"

        if not project_index_dir.exists():
            raise FileNotFoundError(
                f"Project index directory not found: {project_index_dir}. "
                f"Call create_project() first."
            )

        logger.info(f"Adding {len(file_paths)} documents to project {project_id}")

        try:
            # Initialize RAGPipeline for this project
            # Using preset="research" for all advanced features:
            # - Query expansion (PRF)
            # - Adaptive top-k (MicroAutoK)
            # - Cross-encoder reranking
            # - Telemetry tracking
            rag = RAGPipeline(
                index_dir=str(project_index_dir),
                preset="research",
                device="auto"
            )

            # Add documents (automatically uses embedding cache)
            stats = rag.add_documents(
                paths=file_paths,
                recursive=False  # Files already collected, don't recurse
            )

            # Get cache statistics
            cache_stats = rag.embedding_manager.get_cache_stats()
            stats['cache_hits'] = cache_stats.get('hits', 0)
            stats['cache_misses'] = cache_stats.get('misses', 0)

            # Calculate cache hit rate
            total_embeddings = stats['cache_hits'] + stats['cache_misses']
            if total_embeddings > 0:
                cache_hit_rate = stats['cache_hits'] / total_embeddings
                logger.info(
                    f"Indexing complete: {stats['files_processed']} files, "
                    f"{stats['total_chunks']} chunks, "
                    f"cache hit rate: {cache_hit_rate:.1%}"
                )
            else:
                logger.info(
                    f"Indexing complete: {stats['files_processed']} files, "
                    f"{stats['total_chunks']} chunks"
                )

            return stats

        except Exception as e:
            logger.error(f"Error adding documents to project {project_id}: {e}")
            raise

    def get_pipeline(self, project_id: str) -> RAGPipeline:
        """
        Get RAGPipeline instance for a project.

        Always uses preset="research" for:
        - Query expansion (PRF-style using embedding similarity)
        - Adaptive top-k (MicroAutoK heuristic for multi-aspect queries)
        - Cross-encoder reranking (improved precision)
        - Telemetry tracking (performance metrics)

        Args:
            project_id: Project identifier

        Returns:
            Configured RAGPipeline instance

        Raises:
            FileNotFoundError: If project index doesn't exist
        """
        project_index_dir = self.projects_dir / project_id / "rag_index"

        if not project_index_dir.exists():
            raise FileNotFoundError(
                f"Project index directory not found: {project_index_dir}"
            )

        return RAGPipeline(
            index_dir=str(project_index_dir),
            preset="research",
            device="auto"
        )

    def get_project_status(self, project_id: str) -> Dict:
        """
        Get status information for a project's RAG index.

        Args:
            project_id: Project identifier

        Returns:
            Status dict with keys:
            - project_id: Project identifier
            - index_dir: Path to index directory
            - exists: Whether index directory exists
            - has_index: Whether FAISS index exists
            - document_count: Number of indexed documents
            - chunk_count: Total number of chunks
        """
        project_index_dir = self.projects_dir / project_id / "rag_index"

        status = {
            'project_id': project_id,
            'index_dir': str(project_index_dir),
            'exists': project_index_dir.exists()
        }

        if status['exists']:
            # Check if FAISS index exists
            status['has_index'] = (project_index_dir / "faiss_index.bin").exists()

            # Try to get RAGPipeline status
            try:
                rag = self.get_pipeline(project_id)
                rag_status = rag.status()
                status['document_count'] = rag_status.get('documents', 0)
                status['chunk_count'] = rag_status.get('chunks', 0)
            except Exception as e:
                logger.warning(f"Could not get RAG status: {e}")
                status['document_count'] = 0
                status['chunk_count'] = 0
        else:
            status['has_index'] = False
            status['document_count'] = 0
            status['chunk_count'] = 0

        return status

    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project's RAG index.

        Args:
            project_id: Project identifier

        Returns:
            True if deleted successfully, False otherwise
        """
        project_dir = self.projects_dir / project_id

        if project_dir.exists():
            try:
                shutil.rmtree(project_dir)
                logger.info(f"Deleted project {project_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting project {project_id}: {e}")
                return False
        else:
            logger.warning(f"Project {project_id} does not exist")
            return False
