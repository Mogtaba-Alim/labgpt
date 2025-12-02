"""
LabGPT vLLM Inference Components

This module provides vLLM-based inference capabilities for LabGPT,
allowing testing with different base models via API calls to a vLLM server.

Main Components:
- vllm_inference: Main inference script
- vllm_model_config: Model configuration and management utilities
- example_vllm_usage: Usage examples and demonstrations

Usage:
    # From project root
    python vllm_inference.py "What is CRISPR?"
    
    # Or import as module
    from vllm_inference import VLLMClient, get_rag_answer_vllm
"""

from vllm_inference import VLLMClient, VLLMConfig, get_rag_answer_vllm
from vllm_model_config import MODEL_CONFIGS, ModelConfig

__version__ = "1.0.0"
__author__ = "LabGPT Team"
__description__ = "vLLM integration for LabGPT inference"

__all__ = [
    "VLLMClient",
    "VLLMConfig", 
    "get_rag_answer_vllm",
    "MODEL_CONFIGS",
    "ModelConfig"
]
