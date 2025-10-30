"""
LLM Client Abstraction Layer

This module provides a unified interface for working with different LLM backends:
- External APIs (Anthropic Claude, OpenAI GPT)
- Local models (Llama 3.1 8B via Hugging Face Transformers)
"""

from .local_llm_client import LocalLlamaClient
from .llm_client_wrapper import LLMClientWrapper

__all__ = [
    'LocalLlamaClient',
    'LLMClientWrapper',
]
