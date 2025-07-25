#!/usr/bin/env python3
"""
embedding_cache.py

SHA256-based embedding cache system for incremental processing.
Enables massive performance improvements for reprocessing and updates.
"""

import os
import pickle
import gzip
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Single cache entry for an embedding"""
    text_hash: str
    embedding: np.ndarray
    model_name: str
    embedding_dim: int
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0
    
    def __post_init__(self):
        if self.last_accessed == 0.0:
            self.last_accessed = time.time()

@dataclass
class CacheMetadata:
    """Metadata for the entire cache"""
    version: str = "1.0"
    total_entries: int = 0
    total_size_bytes: int = 0
    created_at: float = 0.0
    last_updated: float = 0.0
    model_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_updated == 0.0:
            self.last_updated = time.time()
        if self.model_info is None:
            self.model_info = {}

class EmbeddingCache:
    """
    SHA256-based embedding cache with compression and LRU eviction.
    
    Features:
    - Content-based caching using SHA256 hashes
    - Compressed storage to reduce disk usage
    - LRU eviction policy for size management
    - Batch operations for efficiency
    - Cache validation and integrity checks
    - Performance monitoring and statistics
    """
    
    def __init__(self, 
                 cache_dir: str = "embedding_cache",
                 max_size_gb: float = 10.0,
                 compression_level: int = 6,
                 enable_compression: bool = True):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.compression_level = compression_level
        self.enable_compression = enable_compression
        
        # Cache files
        self.metadata_file = self.cache_dir / "metadata.json"
        self.index_file = self.cache_dir / "index.json"
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # In-memory structures
        self._index: Dict[str, CacheEntry] = {}
        self._metadata: CacheMetadata = CacheMetadata()
        self._dirty = False
        
        # Performance tracking
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "total_requests": 0
        }
        
        # Load existing cache
        self._load_cache()
    
    def _generate_text_hash(self, text: str) -> str:
        """Generate SHA256 hash for text content"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _get_embedding_path(self, text_hash: str) -> Path:
        """Get file path for storing embedding"""
        # Use first 2 chars of hash for directory structure
        subdir = self.embeddings_dir / text_hash[:2]
        subdir.mkdir(exist_ok=True)
        
        extension = ".gz" if self.enable_compression else ".pkl"
        return subdir / f"{text_hash}{extension}"
    
    def _save_embedding(self, text_hash: str, embedding: np.ndarray) -> None:
        """Save embedding to disk"""
        embedding_path = self._get_embedding_path(text_hash)
        
        try:
            if self.enable_compression:
                with gzip.open(embedding_path, 'wb', compresslevel=self.compression_level) as f:
                    pickle.dump(embedding, f)
            else:
                with open(embedding_path, 'wb') as f:
                    pickle.dump(embedding, f)
        except Exception as e:
            logger.error(f"Failed to save embedding {text_hash}: {e}")
            raise
    
    def _load_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Load embedding from disk"""
        embedding_path = self._get_embedding_path(text_hash)
        
        if not embedding_path.exists():
            return None
        
        try:
            if self.enable_compression:
                with gzip.open(embedding_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(embedding_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load embedding {text_hash}: {e}")
            return None
    
    def _delete_embedding(self, text_hash: str) -> None:
        """Delete embedding file from disk"""
        embedding_path = self._get_embedding_path(text_hash)
        try:
            if embedding_path.exists():
                embedding_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete embedding {text_hash}: {e}")
    
    def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache
        
        Args:
            text: Text content
            model_name: Name of the embedding model
            
        Returns:
            Cached embedding if found, None otherwise
        """
        text_hash = self._generate_text_hash(text)
        self._stats["total_requests"] += 1
        
        if text_hash in self._index:
            entry = self._index[text_hash]
            
            # Check model compatibility
            if entry.model_name == model_name:
                # Update access statistics
                entry.access_count += 1
                entry.last_accessed = time.time()
                self._dirty = True
                
                # Load embedding from disk
                embedding = self._load_embedding(text_hash)
                if embedding is not None:
                    self._stats["hits"] += 1
                    logger.debug(f"Cache hit for text hash {text_hash[:8]}...")
                    return embedding
                else:
                    # Entry exists but file is missing - clean up
                    del self._index[text_hash]
                    self._dirty = True
        
        self._stats["misses"] += 1
        logger.debug(f"Cache miss for text hash {text_hash[:8]}...")
        return None
    
    def store_embedding(self, 
                       text: str, 
                       embedding: np.ndarray, 
                       model_name: str) -> None:
        """
        Store embedding in cache
        
        Args:
            text: Text content
            embedding: Embedding vector
            model_name: Name of the embedding model
        """
        text_hash = self._generate_text_hash(text)
        
        # Create cache entry
        entry = CacheEntry(
            text_hash=text_hash,
            embedding=embedding,
            model_name=model_name,
            embedding_dim=embedding.shape[0] if embedding.ndim == 1 else embedding.shape[-1],
            created_at=time.time()
        )
        
        # Save to disk
        self._save_embedding(text_hash, embedding)
        
        # Update index
        self._index[text_hash] = entry
        self._stats["stores"] += 1
        self._dirty = True
        
        # Check if eviction is needed
        self._maybe_evict()
        
        logger.debug(f"Stored embedding for text hash {text_hash[:8]}...")
    
    def batch_get_embeddings(self, 
                           texts: List[str], 
                           model_name: str) -> Tuple[List[Optional[np.ndarray]], List[str]]:
        """
        Batch get embeddings from cache
        
        Args:
            texts: List of text contents
            model_name: Name of the embedding model
            
        Returns:
            Tuple of (embeddings list with None for misses, list of missing texts)
        """
        embeddings = []
        missing_texts = []
        
        for text in texts:
            embedding = self.get_embedding(text, model_name)
            embeddings.append(embedding)
            if embedding is None:
                missing_texts.append(text)
        
        return embeddings, missing_texts
    
    def batch_store_embeddings(self, 
                             texts: List[str], 
                             embeddings: List[np.ndarray], 
                             model_name: str) -> None:
        """
        Batch store embeddings in cache
        
        Args:
            texts: List of text contents
            embeddings: List of embedding vectors
            model_name: Name of the embedding model
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
        
        for text, embedding in zip(texts, embeddings):
            self.store_embedding(text, embedding, model_name)
        
        # Save metadata after batch operation
        self._save_metadata()
    
    def _maybe_evict(self) -> None:
        """Evict entries if cache size exceeds limit"""
        current_size = self._estimate_cache_size()
        
        if current_size <= self.max_size_bytes:
            return
        
        logger.info(f"Cache size {current_size / 1024 / 1024:.1f}MB exceeds limit, starting eviction...")
        
        # Sort entries by last accessed time (LRU)
        entries_by_lru = sorted(
            self._index.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Evict oldest entries until under limit
        target_size = self.max_size_bytes * 0.8  # Evict to 80% of limit
        
        for text_hash, entry in entries_by_lru:
            if current_size <= target_size:
                break
            
            # Delete embedding file
            self._delete_embedding(text_hash)
            
            # Remove from index
            del self._index[text_hash]
            
            # Update statistics
            self._stats["evictions"] += 1
            
            # Estimate size reduction (rough approximation)
            embedding_size = entry.embedding_dim * 4  # float32
            current_size -= embedding_size
        
        self._dirty = True
        logger.info(f"Evicted entries, new cache size: {current_size / 1024 / 1024:.1f}MB")
    
    def _estimate_cache_size(self) -> int:
        """Estimate current cache size in bytes"""
        try:
            total_size = 0
            for file_path in self.embeddings_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception as e:
            logger.warning(f"Failed to estimate cache size: {e}")
            return 0
    
    def _load_cache(self) -> None:
        """Load cache metadata and index from disk"""
        try:
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    self._metadata = CacheMetadata(**metadata_dict)
            
            # Load index
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    index_data = json.load(f)
                    
                    for text_hash, entry_dict in index_data.items():
                        # Convert back to CacheEntry (without embedding data)
                        entry = CacheEntry(
                            text_hash=entry_dict["text_hash"],
                            embedding=np.array([]),  # Placeholder, loaded on demand
                            model_name=entry_dict["model_name"],
                            embedding_dim=entry_dict["embedding_dim"],
                            created_at=entry_dict["created_at"],
                            access_count=entry_dict.get("access_count", 0),
                            last_accessed=entry_dict.get("last_accessed", time.time())
                        )
                        self._index[text_hash] = entry
            
            logger.info(f"Loaded cache with {len(self._index)} entries")
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self._index = {}
            self._metadata = CacheMetadata()
    
    def _save_metadata(self) -> None:
        """Save cache metadata and index to disk"""
        if not self._dirty:
            return
        
        try:
            # Update metadata
            self._metadata.total_entries = len(self._index)
            self._metadata.total_size_bytes = self._estimate_cache_size()
            self._metadata.last_updated = time.time()
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(asdict(self._metadata), f, indent=2)
            
            # Save index (without embedding data)
            index_data = {}
            for text_hash, entry in self._index.items():
                index_data[text_hash] = {
                    "text_hash": entry.text_hash,
                    "model_name": entry.model_name,
                    "embedding_dim": entry.embedding_dim,
                    "created_at": entry.created_at,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed
                }
            
            with open(self.index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            self._dirty = False
            logger.debug("Cache metadata saved")
            
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def clear_cache(self) -> None:
        """Clear entire cache"""
        logger.info("Clearing embedding cache...")
        
        # Delete all embedding files
        for text_hash in list(self._index.keys()):
            self._delete_embedding(text_hash)
        
        # Clear in-memory structures
        self._index.clear()
        self._metadata = CacheMetadata()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "total_requests": 0
        }
        
        # Save empty state
        self._dirty = True
        self._save_metadata()
        
        logger.info("Cache cleared")
    
    def validate_cache(self) -> Dict[str, Any]:
        """Validate cache integrity and return report"""
        logger.info("Validating cache integrity...")
        
        report = {
            "total_entries": len(self._index),
            "valid_entries": 0,
            "invalid_entries": 0,
            "missing_files": 0,
            "corrupted_files": 0,
            "orphaned_files": 0,
            "total_size_bytes": 0
        }
        
        # Check index entries
        valid_hashes = set()
        for text_hash, entry in self._index.items():
            embedding_path = self._get_embedding_path(text_hash)
            
            if not embedding_path.exists():
                report["missing_files"] += 1
                continue
            
            try:
                embedding = self._load_embedding(text_hash)
                if embedding is None:
                    report["corrupted_files"] += 1
                    continue
                
                # Validate dimensions
                if embedding.shape[0] != entry.embedding_dim:
                    report["invalid_entries"] += 1
                    continue
                
                report["valid_entries"] += 1
                report["total_size_bytes"] += embedding_path.stat().st_size
                valid_hashes.add(text_hash)
                
            except Exception as e:
                logger.warning(f"Validation error for {text_hash}: {e}")
                report["corrupted_files"] += 1
        
        # Check for orphaned files
        all_files = list(self.embeddings_dir.rglob("*.pkl")) + list(self.embeddings_dir.rglob("*.gz"))
        for file_path in all_files:
            file_hash = file_path.stem.replace('.pkl', '').replace('.gz', '')
            if file_hash not in valid_hashes:
                report["orphaned_files"] += 1
        
        report["integrity_score"] = report["valid_entries"] / max(report["total_entries"], 1)
        
        logger.info(f"Cache validation complete: {report['valid_entries']}/{report['total_entries']} valid entries")
        
        return report
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        cache_size = self._estimate_cache_size()
        
        stats = {
            "cache_info": {
                "total_entries": len(self._index),
                "cache_size_mb": cache_size / 1024 / 1024,
                "cache_size_gb": cache_size / 1024 / 1024 / 1024,
                "max_size_gb": self.max_size_bytes / 1024 / 1024 / 1024,
                "utilization": cache_size / self.max_size_bytes,
                "compression_enabled": self.enable_compression
            },
            "performance": {
                "hit_rate": self._stats["hits"] / max(self._stats["total_requests"], 1),
                "miss_rate": self._stats["misses"] / max(self._stats["total_requests"], 1),
                "total_requests": self._stats["total_requests"],
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "stores": self._stats["stores"],
                "evictions": self._stats["evictions"]
            },
            "metadata": asdict(self._metadata)
        }
        
        return stats
    
    def cleanup_orphaned_files(self) -> int:
        """Remove orphaned embedding files and return count"""
        logger.info("Cleaning up orphaned files...")
        
        valid_hashes = set(self._index.keys())
        orphaned_count = 0
        
        all_files = list(self.embeddings_dir.rglob("*.pkl")) + list(self.embeddings_dir.rglob("*.gz"))
        for file_path in all_files:
            file_hash = file_path.stem.replace('.pkl', '').replace('.gz', '')
            
            if file_hash not in valid_hashes:
                try:
                    file_path.unlink()
                    orphaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete orphaned file {file_path}: {e}")
        
        logger.info(f"Cleaned up {orphaned_count} orphaned files")
        return orphaned_count
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save metadata"""
        self._save_metadata()

class CachedEmbeddingGenerator:
    """
    Embedding generator with integrated caching support
    """
    
    def __init__(self, 
                 model_name: str,
                 cache: Optional[EmbeddingCache] = None,
                 device: str = "auto"):
        
        self.model_name = model_name
        self.cache = cache or EmbeddingCache()
        
        # Handle device selection
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model
        self.model = SentenceTransformer(model_name, device=device)
        
        # Performance tracking
        self._generation_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "embeddings_generated": 0,
            "total_texts_processed": 0
        }
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode single text with caching
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        self._generation_stats["total_texts_processed"] += 1
        
        # Try cache first
        cached_embedding = self.cache.get_embedding(text, self.model_name)
        if cached_embedding is not None:
            self._generation_stats["cache_hits"] += 1
            return cached_embedding
        
        # Generate new embedding
        self._generation_stats["cache_misses"] += 1
        self._generation_stats["embeddings_generated"] += 1
        
        embedding = self.model.encode(text)
        
        # Store in cache
        self.cache.store_embedding(text, embedding, self.model_name)
        
        return embedding
    
    def encode_batch(self, 
                    texts: List[str], 
                    batch_size: int = 32,
                    show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode batch of texts with caching
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress
            
        Returns:
            Array of embeddings
        """
        # Check cache for all texts
        cached_embeddings, missing_texts = self.cache.batch_get_embeddings(texts, self.model_name)
        
        # Generate embeddings for missing texts
        if missing_texts:
            logger.info(f"Generating {len(missing_texts)} new embeddings "
                       f"({len(texts) - len(missing_texts)} cached)")
            
            new_embeddings = self.model.encode(
                missing_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar
            )
            
            # Store new embeddings in cache
            self.cache.batch_store_embeddings(missing_texts, new_embeddings, self.model_name)
            
            # Update statistics
            self._generation_stats["embeddings_generated"] += len(missing_texts)
        else:
            new_embeddings = np.array([])
        
        # Combine cached and new embeddings
        result_embeddings = []
        new_idx = 0
        
        for text, cached in zip(texts, cached_embeddings):
            if cached is not None:
                result_embeddings.append(cached)
                self._generation_stats["cache_hits"] += 1
            else:
                result_embeddings.append(new_embeddings[new_idx])
                new_idx += 1
                self._generation_stats["cache_misses"] += 1
        
        self._generation_stats["total_texts_processed"] += len(texts)
        
        return np.array(result_embeddings)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        total_processed = self._generation_stats["total_texts_processed"]
        
        return {
            "cache_hit_rate": self._generation_stats["cache_hits"] / max(total_processed, 1),
            "embeddings_generated": self._generation_stats["embeddings_generated"],
            "total_processed": total_processed,
            "cache_effectiveness": 1.0 - (self._generation_stats["embeddings_generated"] / max(total_processed, 1))
        } 