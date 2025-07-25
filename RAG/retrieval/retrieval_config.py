#!/usr/bin/env python3
"""
retrieval_config.py

YAML-based configuration system for flexible retrieval parameters
"""

import yaml
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for models used in retrieval"""
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    query_expansion_model: str = "gpt-3.5-turbo"
    device: str = "auto"  # "cpu", "cuda", "auto"
    
@dataclass
class DenseRetrievalConfig:
    """Configuration for dense retrieval"""
    top_k: int = 20
    score_threshold: float = 0.0
    use_gpu: bool = True
    index_type: str = "hnsw"  # "hnsw", "ivf", "flat"
    
@dataclass
class SparseRetrievalConfig:
    """Configuration for sparse (BM25) retrieval"""
    top_k: int = 20
    k1: float = 1.2
    b: float = 0.75
    use_stemming: bool = True
    remove_stopwords: bool = True
    
@dataclass
class QueryProcessingConfig:
    """Configuration for query processing and expansion"""
    enable_expansion: bool = True
    max_expansions: int = 3
    expansion_method: str = "llm"  # "llm", "wordnet", "embedding"
    include_original: bool = True
    query_rewriting: bool = True
    
@dataclass
class RerankingConfig:
    """Configuration for re-ranking"""
    enable_reranking: bool = True
    rerank_top_k: int = 50
    final_top_k: int = 10
    score_threshold: float = 0.0
    
@dataclass
class FusionConfig:
    """Configuration for result fusion"""
    fusion_method: str = "rrf"  # "rrf", "linear", "rank_sum"
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60
    
@dataclass
class FilteringConfig:
    """Configuration for result filtering"""
    min_chunk_quality: float = 0.3
    max_chunk_age_days: Optional[int] = None
    allowed_doc_types: Optional[List[str]] = None
    excluded_sections: Optional[List[str]] = None
    diversity_threshold: float = 0.8

@dataclass 
class SplittingConfig:
    """Configuration for text splitting"""
    target_chunk_size: int = 400  # Target tokens per chunk
    max_chunk_size: int = 512     # Maximum tokens per chunk
    min_chunk_size: int = 50      # Minimum tokens per chunk
    chunk_overlap: int = 50       # Overlap in tokens
    respect_sentence_boundaries: bool = True
    respect_section_boundaries: bool = True
    preserve_structure: bool = True

@dataclass
class QualityConfig:
    """Configuration for quality filtering"""
    min_quality_score: float = 0.3
    language: str = 'english'
    filter_very_short: bool = True
    filter_very_long: bool = True
    remove_duplicates: bool = True
    
@dataclass
class RetrievalConfig:
    """Main configuration class for retrieval system"""
    
    # Model configurations
    models: ModelConfig = None
    
    # Retrieval configurations
    dense: DenseRetrievalConfig = None
    sparse: SparseRetrievalConfig = None
    query_processing: QueryProcessingConfig = None
    reranking: RerankingConfig = None
    fusion: FusionConfig = None
    filtering: FilteringConfig = None
    
    # Ingestion configurations
    splitting: SplittingConfig = None
    quality: QualityConfig = None
    
    # System configurations
    cache_embeddings: bool = True
    parallel_retrieval: bool = True
    logging_level: str = "INFO"
    
    def __post_init__(self):
        # Initialize sub-configs if not provided
        if self.models is None:
            self.models = ModelConfig()
        if self.dense is None:
            self.dense = DenseRetrievalConfig()
        if self.sparse is None:
            self.sparse = SparseRetrievalConfig()
        if self.query_processing is None:
            self.query_processing = QueryProcessingConfig()
        if self.reranking is None:
            self.reranking = RerankingConfig()
        if self.fusion is None:
            self.fusion = FusionConfig()
        if self.filtering is None:
            self.filtering = FilteringConfig()
        if self.splitting is None:
            self.splitting = SplittingConfig()
        if self.quality is None:
            self.quality = QualityConfig()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'RetrievalConfig':
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Create nested objects
            config = cls()
            
            if 'models' in config_dict:
                config.models = ModelConfig(**config_dict['models'])
            if 'dense' in config_dict:
                config.dense = DenseRetrievalConfig(**config_dict['dense'])
            if 'sparse' in config_dict:
                config.sparse = SparseRetrievalConfig(**config_dict['sparse'])
            if 'query_processing' in config_dict:
                config.query_processing = QueryProcessingConfig(**config_dict['query_processing'])
            if 'reranking' in config_dict:
                config.reranking = RerankingConfig(**config_dict['reranking'])
            if 'fusion' in config_dict:
                config.fusion = FusionConfig(**config_dict['fusion'])
            if 'filtering' in config_dict:
                config.filtering = FilteringConfig(**config_dict['filtering'])
            if 'splitting' in config_dict:
                config.splitting = SplittingConfig(**config_dict['splitting'])
            if 'quality' in config_dict:
                config.quality = QualityConfig(**config_dict['quality'])
            
            # Set top-level attributes
            for key, value in config_dict.items():
                if hasattr(config, key) and not isinstance(getattr(config, key), (ModelConfig, DenseRetrievalConfig, SparseRetrievalConfig, QueryProcessingConfig, RerankingConfig, FusionConfig, FilteringConfig, SplittingConfig, QualityConfig)):
                    setattr(config, key, value)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def save_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = asdict(self)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        valid = True
        
        # Validate dense retrieval config
        if self.dense.top_k <= 0:
            logger.error("Dense retrieval top_k must be positive")
            valid = False
        
        # Validate sparse retrieval config
        if self.sparse.top_k <= 0:
            logger.error("Sparse retrieval top_k must be positive")
            valid = False
        if not 0 < self.sparse.b <= 1:
            logger.error("BM25 parameter b must be between 0 and 1")
            valid = False
        if self.sparse.k1 <= 0:
            logger.error("BM25 parameter k1 must be positive")
            valid = False
        
        # Validate reranking config
        if self.reranking.rerank_top_k < self.reranking.final_top_k:
            logger.warning("rerank_top_k should be >= final_top_k for optimal results")
        
        # Validate fusion config
        if self.fusion.fusion_method not in ["rrf", "linear", "rank_sum"]:
            logger.error(f"Invalid fusion method: {self.fusion.fusion_method}")
            valid = False
        
        # Validate weights sum to 1 for linear fusion
        if self.fusion.fusion_method == "linear":
            weight_sum = self.fusion.dense_weight + self.fusion.sparse_weight
            if abs(weight_sum - 1.0) > 0.01:
                logger.warning(f"Fusion weights sum to {weight_sum}, should be 1.0")
        
        return valid
    
    def get_effective_top_k(self) -> int:
        """Get the effective top_k for initial retrieval"""
        # Return the maximum of dense and sparse top_k to ensure we have enough candidates
        return max(self.dense.top_k, self.sparse.top_k)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from a dictionary of changes"""
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (ModelConfig, DenseRetrievalConfig, SparseRetrievalConfig, QueryProcessingConfig, RerankingConfig, FusionConfig, FilteringConfig)):
                    # Update nested config
                    config_obj = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(config_obj, sub_key):
                            setattr(config_obj, sub_key, sub_value)
                else:
                    setattr(self, key, value)

# Default configuration templates
DEFAULT_RESEARCH_CONFIG = {
    "models": {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "device": "auto"
    },
    "dense": {
        "top_k": 25,
        "score_threshold": 0.0
    },
    "sparse": {
        "top_k": 25,
        "k1": 1.2,
        "b": 0.75
    },
    "query_processing": {
        "enable_expansion": True,
        "max_expansions": 2,
        "query_rewriting": True
    },
    "reranking": {
        "enable_reranking": True,
        "rerank_top_k": 50,
        "final_top_k": 10
    },
    "fusion": {
        "fusion_method": "rrf",
        "dense_weight": 0.6,
        "sparse_weight": 0.4
    },
    "filtering": {
        "min_chunk_quality": 0.4,
        "diversity_threshold": 0.8
    }
}

DEFAULT_TECHNICAL_CONFIG = {
    "models": {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    },
    "dense": {
        "top_k": 30,
        "score_threshold": 0.1
    },
    "sparse": {
        "top_k": 20,
        "k1": 1.5,
        "b": 0.8
    },
    "query_processing": {
        "enable_expansion": True,
        "max_expansions": 3,
        "expansion_method": "embedding"
    },
    "reranking": {
        "enable_reranking": True,
        "rerank_top_k": 60,
        "final_top_k": 15
    },
    "fusion": {
        "fusion_method": "linear",
        "dense_weight": 0.7,
        "sparse_weight": 0.3
    }
}

def create_config_template(config_type: str = "research") -> RetrievalConfig:
    """Create a configuration template for specific use cases"""
    if config_type == "research":
        config = RetrievalConfig()
        config.update_from_dict(DEFAULT_RESEARCH_CONFIG)
        return config
    elif config_type == "technical":
        config = RetrievalConfig()
        config.update_from_dict(DEFAULT_TECHNICAL_CONFIG)
        return config
    else:
        return RetrievalConfig()  # Default configuration 