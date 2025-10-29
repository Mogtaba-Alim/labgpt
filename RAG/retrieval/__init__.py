"""
Retrieval Package for LabGPT RAG System

This package provides retrieval capabilities including:
- Hybrid dense + sparse retrieval (FAISS + BM25)
- RRF fusion
- PRF-style query expansion
- Micro auto-k adaptive top-k selection
- Cross-encoder reranking
"""

from .hybrid_retriever import HybridRetriever
from .rrf_fusion import RRFFusion
from .prf_expansion import PRFQueryExpander
from .micro_auto_k import MicroAutoK
from .reranker import CrossEncoderReranker
from .retrieval_config import RetrievalConfig

__all__ = [
    'HybridRetriever',
    'RRFFusion',
    'PRFQueryExpander',
    'MicroAutoK',
    'CrossEncoderReranker',
    'RetrievalConfig'
]
