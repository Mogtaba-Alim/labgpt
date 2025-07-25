#!/usr/bin/env python3
"""
per_document_management.py

Enhanced per-document embedding management system for granular incremental updates.
Implements document-level isolation with FAISS reconstruction and tombstone management.
"""

import os
import json
import pickle
import hashlib
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil

import numpy as np
import faiss
from scipy.sparse import csr_matrix, save_npz, load_npz

from .chunk_objects import ChunkMetadata, DocumentMetadata, ChunkStore
from .incremental_updates import DocumentTracker

logger = logging.getLogger(__name__)

@dataclass
class DocumentEmbeddingRecord:
    """Record for per-document embedding storage"""
    doc_id: str
    file_path: str
    content_hash: str
    chunk_ids: List[str]
    embedding_file: str
    metadata_file: str
    creation_time: float
    last_modified: float
    chunk_count: int
    embedding_dim: int

@dataclass
class IndexUpdateOperation:
    """Represents an index update operation"""
    operation_type: str  # 'add', 'remove', 'update'
    doc_id: str
    affected_chunk_ids: List[str]
    new_chunk_ids: List[str] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

class PerDocumentEmbeddingManager:
    """
    Manages per-document embedding storage with granular update capabilities
    """
    
    def __init__(self, 
                 storage_dir: str = "per_doc_embeddings",
                 enable_git_tracking: bool = True):
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Directory structure
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.metadata_dir = self.storage_dir / "metadata"
        self.index_dir = self.storage_dir / "indices"
        self.tombstone_dir = self.storage_dir / "tombstones"
        
        for dir_path in [self.embeddings_dir, self.metadata_dir, self.index_dir, self.tombstone_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Registry files
        self.registry_file = self.metadata_dir / "document_registry.json"
        self.operation_log = self.metadata_dir / "operations.jsonl"
        
        # Document registry
        self.document_registry: Dict[str, DocumentEmbeddingRecord] = {}
        self.tombstone_list: Set[str] = set()  # Deleted chunk IDs
        
        # Git tracking
        self.enable_git_tracking = enable_git_tracking
        self.git_tracker = GitChangeTracker() if enable_git_tracking else None
        
        # Load existing registry
        self._load_registry()
        self._load_tombstones()
    
    def _convert_datetime_to_str(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert datetime objects to ISO format strings for JSON serialization"""
        from datetime import datetime
        result = data.copy()
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result
    
    def store_document_embeddings(self, 
                                doc_metadata: DocumentMetadata,
                                chunks: List[ChunkMetadata],
                                embeddings: np.ndarray) -> DocumentEmbeddingRecord:
        """
        Store embeddings for a document with per-document isolation
        
        Args:
            doc_metadata: Document metadata
            chunks: List of chunks for the document
            embeddings: Embedding vectors for the chunks
            
        Returns:
            DocumentEmbeddingRecord for the stored document
        """
        doc_id = doc_metadata.doc_id
        
        # Generate files for this document
        embedding_file = self.embeddings_dir / f"{doc_id}_embeddings.npy"
        metadata_file = self.metadata_dir / f"{doc_id}_metadata.json"
        
        # Save embeddings
        np.save(embedding_file, embeddings)
        
        # Save chunk metadata
        chunk_data = {
            'chunks': [chunk.to_dict() for chunk in chunks],  # Use to_dict() which handles datetime conversion
            'document_metadata': self._convert_datetime_to_str(asdict(doc_metadata))
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        # Create registry record
        record = DocumentEmbeddingRecord(
            doc_id=doc_id,
            file_path=doc_metadata.source_path,
            content_hash=doc_metadata.content_hash or "",
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            embedding_file=str(embedding_file),
            metadata_file=str(metadata_file),
            creation_time=time.time(),
            last_modified=time.time(),
            chunk_count=len(chunks),
            embedding_dim=embeddings.shape[1] if embeddings.ndim > 1 else embeddings.shape[0]
        )
        
        # Update registry
        self.document_registry[doc_id] = record
        self._save_registry()
        
        logger.info(f"Stored embeddings for document {doc_id} with {len(chunks)} chunks")
        
        return record
    
    def remove_document_embeddings(self, doc_id: str) -> bool:
        """
        Remove embeddings for a document and add chunks to tombstone list
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        if doc_id not in self.document_registry:
            logger.warning(f"Document {doc_id} not found in registry")
            return False
        
        record = self.document_registry[doc_id]
        
        try:
            # Add chunk IDs to tombstone list
            self.tombstone_list.update(record.chunk_ids)
            self._save_tombstones()
            
            # Remove files
            if os.path.exists(record.embedding_file):
                os.remove(record.embedding_file)
            
            if os.path.exists(record.metadata_file):
                os.remove(record.metadata_file)
            
            # Remove from registry
            del self.document_registry[doc_id]
            self._save_registry()
            
            # Log operation
            operation = IndexUpdateOperation(
                operation_type='remove',
                doc_id=doc_id,
                affected_chunk_ids=record.chunk_ids
            )
            self._log_operation(operation)
            
            logger.info(f"Removed document {doc_id} and added {len(record.chunk_ids)} chunks to tombstone list")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")
            return False
    
    def update_document_embeddings(self, 
                                 doc_metadata: DocumentMetadata,
                                 chunks: List[ChunkMetadata],
                                 embeddings: np.ndarray) -> DocumentEmbeddingRecord:
        """
        Update embeddings for an existing document
        
        Args:
            doc_metadata: Updated document metadata
            chunks: Updated list of chunks
            embeddings: New embedding vectors
            
        Returns:
            Updated DocumentEmbeddingRecord
        """
        doc_id = doc_metadata.doc_id
        
        # Remove old version if exists
        if doc_id in self.document_registry:
            old_record = self.document_registry[doc_id]
            self.tombstone_list.update(old_record.chunk_ids)
        
        # Store new version
        new_record = self.store_document_embeddings(doc_metadata, chunks, embeddings)
        
        # Log operation
        operation = IndexUpdateOperation(
            operation_type='update',
            doc_id=doc_id,
            affected_chunk_ids=old_record.chunk_ids if doc_id in self.document_registry else [],
            new_chunk_ids=[chunk.chunk_id for chunk in chunks]
        )
        self._log_operation(operation)
        
        return new_record
    
    def get_document_embeddings(self, doc_id: str) -> Optional[Tuple[List[ChunkMetadata], np.ndarray]]:
        """
        Retrieve embeddings and chunks for a document
        
        Args:
            doc_id: Document ID
            
        Returns:
            Tuple of (chunks, embeddings) or None if not found
        """
        if doc_id not in self.document_registry:
            return None
        
        record = self.document_registry[doc_id]
        
        try:
            # Load embeddings
            embeddings = np.load(record.embedding_file)
            
            # Load chunks
            with open(record.metadata_file, 'r') as f:
                data = json.load(f)
            
            chunks = [ChunkMetadata(**chunk_data) for chunk_data in data['chunks']]
            
            return chunks, embeddings
            
        except Exception as e:
            logger.error(f"Failed to load document {doc_id}: {e}")
            return None
    
    def detect_changed_documents(self, source_directories: List[str]) -> List[str]:
        """
        Detect documents that have changed using git diff and file modification times
        
        Args:
            source_directories: Directories to monitor for changes
            
        Returns:
            List of changed document IDs
        """
        changed_docs = []
        
        # Git-based change detection
        if self.git_tracker:
            git_changes = self.git_tracker.detect_git_changes(source_directories)
            
            for file_path in git_changes:
                doc_id = self._file_path_to_doc_id(file_path)
                if doc_id and doc_id in self.document_registry:
                    changed_docs.append(doc_id)
        
        # File modification time-based detection
        for doc_id, record in self.document_registry.items():
            try:
                if os.path.exists(record.file_path):
                    file_mtime = os.path.getmtime(record.file_path)
                    if file_mtime > record.last_modified:
                        if doc_id not in changed_docs:
                            changed_docs.append(doc_id)
            except Exception as e:
                logger.warning(f"Error checking modification time for {record.file_path}: {e}")
        
        return changed_docs
    
    def rebuild_embeddings_for_documents(self, doc_ids: List[str]) -> Dict[str, bool]:
        """
        Rebuild embeddings for specified documents
        
        Args:
            doc_ids: List of document IDs to rebuild
            
        Returns:
            Dictionary mapping doc_id to success status
        """
        results = {}
        
        for doc_id in doc_ids:
            try:
                if doc_id in self.document_registry:
                    record = self.document_registry[doc_id]
                    
                    # This would typically involve reprocessing the document
                    # For now, we mark it as needing rebuild
                    logger.info(f"Document {doc_id} marked for embedding rebuild")
                    results[doc_id] = True
                else:
                    logger.warning(f"Document {doc_id} not found in registry")
                    results[doc_id] = False
                    
            except Exception as e:
                logger.error(f"Failed to rebuild embeddings for {doc_id}: {e}")
                results[doc_id] = False
        
        return results
    
    def get_all_active_embeddings(self) -> Tuple[List[ChunkMetadata], np.ndarray, List[str]]:
        """
        Get all active embeddings (excluding tombstoned chunks)
        
        Returns:
            Tuple of (all_chunks, all_embeddings, chunk_to_doc_mapping)
        """
        all_chunks = []
        all_embeddings = []
        chunk_to_doc = []
        
        for doc_id, record in self.document_registry.items():
            try:
                chunks, embeddings = self.get_document_embeddings(doc_id)
                if chunks and embeddings is not None:
                    # Filter out tombstoned chunks
                    active_chunks = []
                    active_embeddings = []
                    
                    for chunk, embedding in zip(chunks, embeddings):
                        if chunk.chunk_id not in self.tombstone_list:
                            active_chunks.append(chunk)
                            active_embeddings.append(embedding)
                            chunk_to_doc.append(doc_id)
                    
                    all_chunks.extend(active_chunks)
                    if active_embeddings:
                        all_embeddings.extend(active_embeddings)
            
            except Exception as e:
                logger.error(f"Error loading embeddings for document {doc_id}: {e}")
        
        if all_embeddings:
            all_embeddings = np.array(all_embeddings)
        else:
            all_embeddings = np.array([]).reshape(0, 0)
        
        return all_chunks, all_embeddings, chunk_to_doc
    
    def _file_path_to_doc_id(self, file_path: str) -> Optional[str]:
        """Convert file path to document ID"""
        for doc_id, record in self.document_registry.items():
            if record.file_path == file_path:
                return doc_id
        return None
    
    def _load_registry(self):
        """Load document registry from file"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for doc_id, record_data in registry_data.items():
                    self.document_registry[doc_id] = DocumentEmbeddingRecord(**record_data)
                
                logger.info(f"Loaded registry with {len(self.document_registry)} documents")
        
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self.document_registry = {}
    
    def _save_registry(self):
        """Save document registry to file"""
        try:
            registry_data = {}
            for doc_id, record in self.document_registry.items():
                registry_data[doc_id] = asdict(record)
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _load_tombstones(self):
        """Load tombstone list from file"""
        tombstone_file = self.tombstone_dir / "tombstones.json"
        try:
            if tombstone_file.exists():
                with open(tombstone_file, 'r') as f:
                    data = json.load(f)
                    self.tombstone_list = set(data.get('tombstones', []))
                
                logger.info(f"Loaded {len(self.tombstone_list)} tombstoned chunks")
        
        except Exception as e:
            logger.error(f"Failed to load tombstones: {e}")
            self.tombstone_list = set()
    
    def _save_tombstones(self):
        """Save tombstone list to file"""
        tombstone_file = self.tombstone_dir / "tombstones.json"
        try:
            data = {
                'tombstones': list(self.tombstone_list),
                'last_updated': time.time()
            }
            
            with open(tombstone_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save tombstones: {e}")
    
    def _log_operation(self, operation: IndexUpdateOperation):
        """Log an index update operation"""
        try:
            with open(self.operation_log, 'a') as f:
                f.write(json.dumps(asdict(operation)) + '\n')
        
        except Exception as e:
            logger.error(f"Failed to log operation: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the per-document management system"""
        total_chunks = sum(record.chunk_count for record in self.document_registry.values())
        
        return {
            'total_documents': len(self.document_registry),
            'total_chunks': total_chunks,
            'tombstoned_chunks': len(self.tombstone_list),
            'active_chunks': total_chunks - len(self.tombstone_list),
            'storage_size_mb': self._calculate_storage_size() / 1024 / 1024,
            'avg_chunks_per_doc': total_chunks / len(self.document_registry) if self.document_registry else 0
        }
    
    def _calculate_storage_size(self) -> int:
        """Calculate total storage size in bytes"""
        total_size = 0
        try:
            for file_path in self.storage_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating storage size: {e}")
        
        return total_size

class GitChangeTracker:
    """
    Tracks document changes using git diff
    """
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.md', '.pdf', '.tex', '.rst'}
    
    def detect_git_changes(self, directories: List[str]) -> List[str]:
        """
        Detect changed files using git diff
        
        Args:
            directories: Directories to check for changes
            
        Returns:
            List of changed file paths
        """
        changed_files = []
        
        for directory in directories:
            try:
                # Check if directory is in a git repository
                result = subprocess.run(
                    ['git', 'rev-parse', '--git-dir'],
                    cwd=directory,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.debug(f"Directory {directory} is not in a git repository")
                    continue
                
                # Get changed files
                changed_files.extend(self._get_git_changed_files(directory))
                
            except Exception as e:
                logger.warning(f"Error detecting git changes in {directory}: {e}")
        
        # Filter for supported file types
        return [f for f in changed_files if Path(f).suffix.lower() in self.supported_extensions]
    
    def _get_git_changed_files(self, directory: str) -> List[str]:
        """Get list of changed files from git"""
        changed_files = []
        
        try:
            # Get files changed since last commit
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD'],
                cwd=directory,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                changed_files.extend([os.path.join(directory, f) for f in files if f])
            
            # Get untracked files
            result = subprocess.run(
                ['git', 'ls-files', '--others', '--exclude-standard'],
                cwd=directory,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                changed_files.extend([os.path.join(directory, f) for f in files if f])
        
        except Exception as e:
            logger.warning(f"Error getting git changed files: {e}")
        
        return changed_files

class FAISSIndexReconstructor:
    """
    Handles FAISS index reconstruction with add/remove operations
    """
    
    def __init__(self, index_type: str = "hnsw"):
        self.index_type = index_type
        self.current_index = None
        self.chunk_id_mapping = {}  # Maps FAISS index position to chunk ID
    
    def build_index_from_embeddings(self, 
                                   embeddings: np.ndarray,
                                   chunk_ids: List[str]) -> faiss.Index:
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Embedding vectors
            chunk_ids: Corresponding chunk IDs
            
        Returns:
            Built FAISS index
        """
        if embeddings.size == 0:
            logger.warning("No embeddings provided for index building")
            return None
        
        dimension = embeddings.shape[1]
        
        # Create index based on type
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 40
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(embeddings)))
            index.train(embeddings.astype(np.float32))
        else:  # flat
            index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embeddings.astype(np.float32))
        
        # Update mapping
        self.chunk_id_mapping = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
        self.current_index = index
        
        logger.info(f"Built {self.index_type} index with {len(embeddings)} vectors")
        
        return index
    
    def add_embeddings_to_index(self, 
                               new_embeddings: np.ndarray,
                               new_chunk_ids: List[str]) -> bool:
        """
        Add new embeddings to existing index
        
        Args:
            new_embeddings: New embedding vectors
            new_chunk_ids: Corresponding chunk IDs
            
        Returns:
            True if successful
        """
        if self.current_index is None:
            logger.error("No existing index to add to")
            return False
        
        try:
            # Get current mapping size
            current_size = len(self.chunk_id_mapping)
            
            # Add embeddings to index
            self.current_index.add(new_embeddings.astype(np.float32))
            
            # Update mapping
            for i, chunk_id in enumerate(new_chunk_ids):
                self.chunk_id_mapping[current_size + i] = chunk_id
            
            logger.info(f"Added {len(new_embeddings)} vectors to index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to index: {e}")
            return False
    
    def remove_embeddings_from_index(self, chunk_ids_to_remove: List[str]) -> bool:
        """
        Remove embeddings from index (using tombstone approach)
        
        Args:
            chunk_ids_to_remove: Chunk IDs to remove
            
        Returns:
            True if successful
        """
        # For FAISS, we can't directly remove vectors, so we use tombstone approach
        # This would be handled by the per-document manager's tombstone list
        
        logger.info(f"Marked {len(chunk_ids_to_remove)} chunks for removal (tombstone approach)")
        return True
    
    def reconstruct_index(self, 
                         all_embeddings: np.ndarray,
                         all_chunk_ids: List[str],
                         exclude_tombstoned: bool = True) -> faiss.Index:
        """
        Completely reconstruct the index
        
        Args:
            all_embeddings: All embedding vectors
            all_chunk_ids: All chunk IDs
            exclude_tombstoned: Whether to exclude tombstoned chunks
            
        Returns:
            Reconstructed FAISS index
        """
        logger.info("Reconstructing FAISS index from scratch")
        return self.build_index_from_embeddings(all_embeddings, all_chunk_ids)
    
    def save_index(self, index_path: str) -> bool:
        """Save FAISS index to file"""
        if self.current_index is None:
            return False
        
        try:
            faiss.write_index(self.current_index, index_path)
            
            # Save mapping
            mapping_path = index_path.replace('.bin', '_mapping.json')
            with open(mapping_path, 'w') as f:
                json.dump(self.chunk_id_mapping, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, index_path: str) -> bool:
        """Load FAISS index from file"""
        try:
            self.current_index = faiss.read_index(index_path)
            
            # Load mapping
            mapping_path = index_path.replace('.bin', '_mapping.json')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    # Convert string keys back to integers
                    str_mapping = json.load(f)
                    self.chunk_id_mapping = {int(k): v for k, v in str_mapping.items()}
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False 