#!/usr/bin/env python3
"""
hybrid_retriever.py

Main hybrid retriever combining dense (FAISS) and sparse (BM25) retrieval
with configurable fusion and re-ranking.
"""

import logging
import time
from typing import List, Dict, Tuple, Optional, Any
import concurrent.futures
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from ..ingestion.chunk_objects import ChunkMetadata, ChunkStore
from .retrieval_config import RetrievalConfig
from .query_processor import QueryProcessor
from .reranker import CrossEncoderReranker
from .fusion import ResultFusion

logger = logging.getLogger(__name__)

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download(['punkt', 'stopwords'], quiet=True)

@dataclass
class RetrievalResult:
    """Result from retrieval with metadata"""
    chunk_id: str
    chunk: ChunkMetadata
    score: float
    retrieval_method: str  # "dense", "sparse", "hybrid"
    rank: int
    metadata: Dict[str, Any] = None

class HybridRetriever:
    """
    Hybrid retrieval system combining dense and sparse search with re-ranking
    """
    
    def __init__(self, 
                 chunk_store: ChunkStore,
                 config: RetrievalConfig,
                 embedding_model: Optional[SentenceTransformer] = None):
        
        self.chunk_store = chunk_store
        self.config = config
        
        # Initialize components
        self.embedding_model = embedding_model or SentenceTransformer(
            config.models.embedding_model,
            device=self._get_device()
        )
        
        self.query_processor = QueryProcessor(config)
        self.reranker = CrossEncoderReranker(config) if config.reranking.enable_reranking else None
        self.result_fusion = ResultFusion(config)
        
        # Initialize retrieval indices
        self.dense_index = None
        self.sparse_index = None
        self.chunk_lookup = {}  # chunk_id -> ChunkMetadata
        
        # Text preprocessing for BM25
        self.stemmer = PorterStemmer() if config.sparse.use_stemming else None
        try:
            self.stopwords = set(stopwords.words('english')) if config.sparse.remove_stopwords else set()
        except:
            self.stopwords = set()
        
        self._build_indices()
    
    def _get_device(self) -> str:
        """Determine device for models"""
        if self.config.models.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.models.device
    
    def _build_indices(self) -> None:
        """Build both dense and sparse indices"""
        logger.info("Building retrieval indices...")
        
        # Load chunk store data
        if not hasattr(self.chunk_store, '_chunks') or not self.chunk_store._chunks:
            self.chunk_store.load()
        
        chunks = list(self.chunk_store._chunks.values())
        if not chunks:
            logger.warning("No chunks found in chunk store")
            return
        
        # Build chunk lookup
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Build dense index
        self._build_dense_index(chunks)
        
        # Build sparse index
        self._build_sparse_index(chunks)
        
        logger.info(f"Built indices for {len(chunks)} chunks")
    
    def _build_dense_index(self, chunks: List[ChunkMetadata]) -> None:
        """Build FAISS dense index"""
        logger.info("Building dense (FAISS) index...")
        
        # Get embeddings
        embeddings = self.chunk_store.get_all_embeddings()
        
        if embeddings is None or len(embeddings) == 0:
            logger.error("No embeddings found in chunk store")
            return
        
        # Ensure float32 format
        embeddings = embeddings.astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        
        if self.config.dense.index_type == "flat":
            self.dense_index = faiss.IndexFlatIP(dimension)
        elif self.config.dense.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            nlist = min(int(np.sqrt(len(embeddings))), 1000)
            self.dense_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.dense_index.train(embeddings)
        else:  # hnsw (default)
            M = 32
            self.dense_index = faiss.IndexHNSWFlat(dimension, M)
            self.dense_index.hnsw.efConstruction = 40
        
        # Add embeddings
        self.dense_index.add(embeddings)
        
        logger.info(f"Dense index built with {self.dense_index.ntotal} vectors")
    
    def _build_sparse_index(self, chunks: List[ChunkMetadata]) -> None:
        """Build BM25 sparse index"""
        logger.info("Building sparse (BM25) index...")
        
        # Preprocess documents for BM25
        processed_docs = []
        for chunk in chunks:
            tokens = self._preprocess_text_for_bm25(chunk.text)
            processed_docs.append(tokens)
        
        # Create BM25 index
        if processed_docs:
            self.sparse_index = BM25Okapi(
                processed_docs,
                k1=self.config.sparse.k1,
                b=self.config.sparse.b
            )
            logger.info(f"BM25 index built with {len(processed_docs)} documents")
        else:
            logger.warning("No documents to index for BM25")
    
    def _preprocess_text_for_bm25(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing"""
        # Tokenize
        try:
            tokens = word_tokenize(text.lower())
        except:
            tokens = text.lower().split()
        
        # Remove non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]
        
        # Remove stopwords
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Apply stemming
        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None,
                filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """
        Main retrieval method combining dense and sparse search
        
        Args:
            query: Search query
            top_k: Number of results to return (uses config default if None)
            filters: Optional filters for chunks
            
        Returns:
            List of RetrievalResult objects
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = self.config.reranking.final_top_k
        
        # Process query (expansion, rewriting)
        processed_queries = self.query_processor.process_query(query)
        
        # Perform parallel retrieval if configured
        if self.config.parallel_retrieval and len(processed_queries) > 1:
            all_results = self._parallel_retrieve(processed_queries, filters)
        else:
            all_results = []
            for proc_query in processed_queries:
                results = self._single_query_retrieve(proc_query, filters)
                all_results.extend(results)
        
        # Fuse results from different queries and methods
        fused_results = self.result_fusion.fuse_results(all_results)
        
        # Apply re-ranking if enabled
        if self.reranker and len(fused_results) > 1:
            rerank_candidates = fused_results[:self.config.reranking.rerank_top_k]
            reranked_results = self.reranker.rerank(query, rerank_candidates)
            fused_results = reranked_results + fused_results[len(rerank_candidates):]
        
        # Apply final filtering and top-k
        final_results = self._apply_final_filtering(fused_results, filters)
        final_results = final_results[:top_k]
        
        # Log retrieval statistics
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(final_results)} results in {retrieval_time:.3f}s")
        
        return final_results
    
    def _single_query_retrieve(self, 
                              query: str,
                              filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """Retrieve results for a single query using both dense and sparse methods"""
        results = []
        
        # Dense retrieval
        if self.dense_index is not None:
            dense_results = self._dense_retrieve(query)
            results.extend(dense_results)
        
        # Sparse retrieval
        if self.sparse_index is not None:
            sparse_results = self._sparse_retrieve(query)
            results.extend(sparse_results)
        
        return results
    
    def _dense_retrieve(self, query: str) -> List[RetrievalResult]:
        """Perform dense retrieval using FAISS"""
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype('float32')
        
        # Search index
        scores, indices = self.dense_index.search(
            query_embedding, 
            self.config.dense.top_k
        )
        
        # Convert to RetrievalResult objects
        results = []
        chunk_ids = self.chunk_store.get_chunk_ids_ordered()
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(chunk_ids) and score >= self.config.dense.score_threshold:
                chunk_id = chunk_ids[idx]
                chunk = self.chunk_lookup.get(chunk_id)
                
                if chunk:
                    result = RetrievalResult(
                        chunk_id=chunk_id,
                        chunk=chunk,
                        score=float(score),
                        retrieval_method="dense",
                        rank=i + 1,
                        metadata={"original_query": query}
                    )
                    results.append(result)
        
        return results
    
    def _sparse_retrieve(self, query: str) -> List[RetrievalResult]:
        """Perform sparse retrieval using BM25"""
        if self.sparse_index is None:
            return []
        
        # Preprocess query
        query_tokens = self._preprocess_text_for_bm25(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.sparse_index.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:self.config.sparse.top_k]
        
        # Convert to RetrievalResult objects
        results = []
        chunk_ids = list(self.chunk_lookup.keys())
        
        for i, idx in enumerate(top_indices):
            if idx < len(chunk_ids):
                score = scores[idx]
                chunk_id = chunk_ids[idx]
                chunk = self.chunk_lookup[chunk_id]
                
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=float(score),
                    retrieval_method="sparse",
                    rank=i + 1,
                    metadata={"original_query": query}
                )
                results.append(result)
        
        return results
    
    def _parallel_retrieve(self, 
                          queries: List[str],
                          filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """Perform parallel retrieval for multiple queries"""
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_query = {
                executor.submit(self._single_query_retrieve, query, filters): query 
                for query in queries
            }
            
            for future in concurrent.futures.as_completed(future_to_query):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    query = future_to_query[future]
                    logger.error(f"Error retrieving for query '{query}': {e}")
        
        return all_results
    
    def _apply_final_filtering(self, 
                              results: List[RetrievalResult],
                              filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """Apply final filtering to results"""
        if not results:
            return results
        
        filtered_results = []
        
        for result in results:
            chunk = result.chunk
            
            # Quality filter
            if chunk.quality_score < self.config.filtering.min_chunk_quality:
                continue
            
            # Document type filter
            if (self.config.filtering.allowed_doc_types and 
                chunk.doc_type not in self.config.filtering.allowed_doc_types):
                continue
            
            # Section filter
            if (self.config.filtering.excluded_sections and 
                chunk.section and 
                any(excluded in chunk.section.lower() 
                    for excluded in self.config.filtering.excluded_sections)):
                continue
            
            # Custom filters
            if filters:
                if 'doc_id' in filters and chunk.doc_id != filters['doc_id']:
                    continue
                if 'doc_type' in filters and chunk.doc_type != filters['doc_type']:
                    continue
                if 'min_score' in filters and result.score < filters['min_score']:
                    continue
            
            filtered_results.append(result)
        
        # Apply diversity filtering if configured
        if self.config.filtering.diversity_threshold < 1.0:
            filtered_results = self._apply_diversity_filtering(filtered_results)
        
        return filtered_results
    
    def _apply_diversity_filtering(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply diversity filtering to avoid too similar results"""
        if len(results) <= 1:
            return results
        
        diverse_results = [results[0]]  # Always include top result
        threshold = self.config.filtering.diversity_threshold
        
        for result in results[1:]:
            # Check similarity with already selected results
            is_diverse = True
            for selected in diverse_results:
                similarity = self._calculate_text_similarity(
                    result.chunk.text, 
                    selected.chunk.text
                )
                if similarity > threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system"""
        stats = {
            "dense_index": {
                "type": self.config.dense.index_type,
                "total_vectors": self.dense_index.ntotal if self.dense_index else 0,
                "dimension": self.dense_index.d if self.dense_index else 0
            },
            "sparse_index": {
                "type": "BM25",
                "total_documents": len(self.sparse_index.corpus) if self.sparse_index else 0,
                "average_doc_length": self.sparse_index.avgdl if self.sparse_index else 0
            },
            "chunk_store": self.chunk_store.get_statistics(),
            "config": self.config.to_dict()
        }
        
        return stats
    
    def update_index(self, new_chunks: List[ChunkMetadata], new_embeddings: np.ndarray) -> None:
        """Update indices with new chunks (for incremental updates)"""
        logger.info(f"Updating indices with {len(new_chunks)} new chunks")
        
        # Update chunk lookup
        for chunk in new_chunks:
            self.chunk_lookup[chunk.chunk_id] = chunk
        
        # Update dense index
        if self.dense_index is not None and new_embeddings is not None:
            self.dense_index.add(new_embeddings.astype('float32'))
        
        # Update sparse index (requires rebuild for BM25)
        if self.sparse_index is not None:
            all_chunks = list(self.chunk_lookup.values())
            self._build_sparse_index(all_chunks)
        
        logger.info("Index update completed") 