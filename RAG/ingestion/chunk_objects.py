#!/usr/bin/env python3
"""
chunk_objects.py

Structured chunk objects with rich metadata for the enhanced RAG system.
Replaces simple numpy arrays with proper data structures supporting metadata,
hierarchical relationships, and quality scoring.
"""

import json
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for source documents"""
    doc_id: str
    source_path: str
    doc_type: str  # 'pdf', 'txt', 'md', etc.
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    creation_date: Optional[datetime] = None
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    language: str = "en"
    content_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.content_hash is None:
            # Generate hash based on source path and file size
            self.content_hash = hashlib.md5(
                f"{self.source_path}_{self.file_size}".encode()
            ).hexdigest()

@dataclass 
class ChunkMetadata:
    """Enhanced chunk object with comprehensive metadata"""
    
    # Core identifiers
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    
    # Content
    text: str = ""
    token_count: int = 0
    char_count: int = 0
    
    # Document structure
    doc_type: str = ""  # pdf, txt, md, etc.
    source_path: str = ""
    section: Optional[str] = None
    hierarchy_path: List[str] = field(default_factory=list)  # ["Chapter 1", "Section 1.1", "Subsection 1.1.1"]
    page_number: Optional[int] = None
    
    # Quality metrics
    quality_score: float = 0.0
    density_score: float = 0.0  # unique tokens / total length
    stopword_ratio: float = 0.0
    symbol_coverage: float = 0.0  # for code/technical content
    
    # Hierarchical relationships
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    
    # Position information
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0  # position within document
    
    # Processing metadata
    created_at: datetime = field(default_factory=datetime.now)
    model_version: str = "v1.0"
    content_hash: str = field(default="")
    
    def __post_init__(self):
        self.char_count = len(self.text)
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.text.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert datetime to string
        result['created_at'] = self.created_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def get_hierarchy_string(self) -> str:
        """Get hierarchy as readable string"""
        return " → ".join(self.hierarchy_path) if self.hierarchy_path else ""
    
    def get_context_window(self, window_size: int = 200) -> tuple[int, int]:
        """Get character positions for context expansion"""
        start = max(0, self.start_char - window_size)
        end = self.end_char + window_size
        return start, end

class ChunkStore:
    """Storage and retrieval system for structured chunks"""
    
    def __init__(self, storage_dir: str = "chunk_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.chunks_file = self.storage_dir / "chunks_metadata.jsonl"
        self.embeddings_file = self.storage_dir / "embeddings.npy"
        self.documents_file = self.storage_dir / "documents_metadata.json"
        self.index_file = self.storage_dir / "chunk_index.json"
        
        self._chunks: Dict[str, ChunkMetadata] = {}
        self._documents: Dict[str, DocumentMetadata] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._chunk_to_embedding_idx: Dict[str, int] = {}
        
    def add_document(self, doc_metadata: DocumentMetadata) -> None:
        """Add document metadata"""
        self._documents[doc_metadata.doc_id] = doc_metadata
        logger.info(f"Added document metadata for {doc_metadata.doc_id}")
        
    def add_chunk(self, chunk: ChunkMetadata, embedding: Optional[np.ndarray] = None) -> None:
        """Add a chunk with optional embedding"""
        self._chunks[chunk.chunk_id] = chunk
        
        if embedding is not None:
            self._add_embedding(chunk.chunk_id, embedding)
            
        logger.debug(f"Added chunk {chunk.chunk_id} from doc {chunk.doc_id}")
    
    def _add_embedding(self, chunk_id: str, embedding: np.ndarray) -> None:
        """Add embedding for a chunk"""
        if self._embeddings is None:
            self._embeddings = embedding.reshape(1, -1)
            self._chunk_to_embedding_idx[chunk_id] = 0
        else:
            self._chunk_to_embedding_idx[chunk_id] = len(self._embeddings)
            self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)])
    
    def get_chunk(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Retrieve chunk by ID"""
        return self._chunks.get(chunk_id)
    
    def get_chunks_by_doc(self, doc_id: str) -> List[ChunkMetadata]:
        """Get all chunks for a document"""
        return [chunk for chunk in self._chunks.values() if chunk.doc_id == doc_id]
    
    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get embedding for a chunk"""
        if chunk_id not in self._chunk_to_embedding_idx or self._embeddings is None:
            return None
        idx = self._chunk_to_embedding_idx[chunk_id]
        return self._embeddings[idx]
    
    def get_all_embeddings(self) -> np.ndarray:
        """Get all embeddings as array"""
        return self._embeddings if self._embeddings is not None else np.array([])
    
    def get_chunk_ids_ordered(self) -> List[str]:
        """Get chunk IDs in embedding order"""
        if not self._chunk_to_embedding_idx:
            return []
        return sorted(self._chunk_to_embedding_idx.keys(), 
                     key=lambda x: self._chunk_to_embedding_idx[x])
    
    def filter_chunks(self, 
                     doc_type: Optional[str] = None,
                     min_quality_score: Optional[float] = None,
                     section_contains: Optional[str] = None) -> List[ChunkMetadata]:
        """Filter chunks by various criteria"""
        filtered = []
        for chunk in self._chunks.values():
            if doc_type and chunk.doc_type != doc_type:
                continue
            if min_quality_score and chunk.quality_score < min_quality_score:
                continue  
            if section_contains and (not chunk.section or section_contains not in chunk.section):
                continue
            filtered.append(chunk)
        return filtered
    
    def save(self) -> None:
        """Save all data to disk"""
        # Save chunks metadata as JSONL
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            for chunk in self._chunks.values():
                f.write(json.dumps(chunk.to_dict()) + '\n')
        
        # Save embeddings
        if self._embeddings is not None:
            np.save(self.embeddings_file, self._embeddings.astype('float32'))
        
        # Save documents metadata
        docs_data = {doc_id: asdict(doc) for doc_id, doc in self._documents.items()}
        # Convert datetime objects
        for doc_data in docs_data.values():
            if doc_data.get('creation_date'):
                doc_data['creation_date'] = doc_data['creation_date'].isoformat()
                
        with open(self.documents_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2)
        
        # Save chunk to embedding index mapping
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self._chunk_to_embedding_idx, f, indent=2)
            
        logger.info(f"Saved {len(self._chunks)} chunks and {len(self._documents)} documents")
    
    def load(self) -> None:
        """Load data from disk"""
        # Load chunks
        if self.chunks_file.exists():
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk_data = json.loads(line.strip())
                    chunk = ChunkMetadata.from_dict(chunk_data)
                    self._chunks[chunk.chunk_id] = chunk
        
        # Load embeddings
        if self.embeddings_file.exists():
            self._embeddings = np.load(self.embeddings_file)
        
        # Load documents
        if self.documents_file.exists():
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
                for doc_id, doc_data in docs_data.items():
                    if 'creation_date' in doc_data and doc_data['creation_date']:
                        doc_data['creation_date'] = datetime.fromisoformat(doc_data['creation_date'])
                    self._documents[doc_id] = DocumentMetadata(**doc_data)
        
        # Load index mapping
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self._chunk_to_embedding_idx = json.load(f)
                
        logger.info(f"Loaded {len(self._chunks)} chunks and {len(self._documents)} documents")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the chunk store"""
        if not self._chunks:
            return {"status": "empty"}
        
        chunks = list(self._chunks.values())
        
        # Quality score statistics
        quality_scores = [c.quality_score for c in chunks]
        
        # Token count statistics  
        token_counts = [c.token_count for c in chunks]
        
        # Document type distribution
        doc_types = {}
        for chunk in chunks:
            doc_types[chunk.doc_type] = doc_types.get(chunk.doc_type, 0) + 1
        
        return {
            "status": "loaded",
            "total_chunks": len(chunks),
            "total_documents": len(self._documents),
            "embedding_dimension": self._embeddings.shape[1] if self._embeddings is not None else None,
            "quality_scores": {
                "mean": np.mean(quality_scores),
                "std": np.std(quality_scores),
                "min": np.min(quality_scores),
                "max": np.max(quality_scores)
            },
            "token_counts": {
                "mean": np.mean(token_counts),
                "std": np.std(token_counts), 
                "min": np.min(token_counts),
                "max": np.max(token_counts),
                "total": np.sum(token_counts)
            },
            "doc_type_distribution": doc_types,
            "has_embeddings": self._embeddings is not None
        }
    
    def load_chunks(self) -> List[ChunkMetadata]:
        """Load and return all chunks"""
        self.load()  # Ensure data is loaded
        return list(self._chunks.values())
    
    def store_chunks(self, chunks: List[ChunkMetadata]) -> None:
        """Store chunks and save to disk"""
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self.save()  # Save to disk 