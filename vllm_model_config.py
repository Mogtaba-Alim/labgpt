#!/usr/bin/env python3
"""
vllm_model_config.py

Model configurations for vLLM inference.
"""

from typing import List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    model_id: str
    description: str
    recommended_params: dict
    tags: List[str]

# Predefined model configurations
MODEL_CONFIGS = {
    "gemma3-4b": ModelConfig(
        name="Gemma 3 4B",
        model_id="google/gemma-3-4b-it",
        description="Google's Gemma 3 4B instruction-tuned model",
        recommended_params={
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 800
        },
        tags=["instruction", "general", "google", "medium"]
    ),
    
    "qwen3-8b": ModelConfig(
        name="Qwen 3 8B",
        model_id="Qwen/Qwen3-8B",
        description="Alibaba's Qwen 3 8B instruction-tuned model",
        recommended_params={
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 800
        },
        tags=["instruction", "general", "multilingual", "medium"]
    ),
    
    "deepseek-r1-32b": ModelConfig(
        name="DeepSeek R1 Distill Qwen 32B",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        description="DeepSeek's R1 distilled model based on Qwen 32B",
        recommended_params={
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 1000
        },
        tags=["instruction", "reasoning", "large", "distilled"]
    ),
    
    "gpt-oss-120b": ModelConfig(
        name="GPT OSS 120B",
        model_id="openai/GPT-OSS-120B",
        description="OpenAI's open-source GPT 120B model",
        recommended_params={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 1200
        },
        tags=["instruction", "general", "reasoning", "large", "microsoft"]
    )
}

# Simple listing function
if __name__ == "__main__":
    print("Available models:")
    for key, config in MODEL_CONFIGS.items():
        print(f"  {key}: {config.name} ({config.model_id})")
