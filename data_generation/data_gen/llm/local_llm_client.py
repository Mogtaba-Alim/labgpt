"""
Local Llama 3.1 8B Client

This module provides a local inference client for Llama 3.1 8B Instruct using Hugging Face Transformers.
Supports 8-bit quantization for memory efficiency and auto-detection of GPU/CPU.
"""

import logging
import os
from typing import Optional, Dict, List, Any
import torch
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class LocalLlamaClient:
    """
    Local Llama 3.1 8B Instruct client with 8-bit quantization support.

    Features:
    - Auto-download from Hugging Face Hub
    - 8-bit quantization for memory efficiency (~8GB VRAM)
    - Auto-detect GPU (CUDA) vs CPU
    - Proper Llama 3.1 Instruct chat template
    - Batching support for improved throughput
    """

    def __init__(
        self,
        model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        load_in_8bit: bool = True,
        device: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the local Llama client.

        Args:
            model_path: Hugging Face model ID or local path
            load_in_8bit: Whether to use 8-bit quantization (reduces memory usage)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            hf_token: Hugging Face token for gated models (or set HUGGINGFACE_TOKEN env var)
        """
        self.model_path = model_path
        self.load_in_8bit = load_in_8bit
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")

        # Auto-detect device if not specified
        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = device

        # Check if quantization is supported on the device
        if self.load_in_8bit and self.device == "cpu":
            logger.warning(
                "8-bit quantization requires CUDA. Disabling quantization for CPU inference."
            )
            self.load_in_8bit = False

        logger.info(f"Initializing Local Llama 3.1 8B client on {self.device}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"8-bit quantization: {self.load_in_8bit}")

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _auto_detect_device(self) -> str:
        """Auto-detect available device (CUDA GPU or CPU)."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")

            if gpu_memory < 10 and self.load_in_8bit:
                logger.warning(
                    f"GPU has {gpu_memory:.1f}GB VRAM. 8-bit quantization recommended. "
                    "May encounter OOM errors with full precision."
                )
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU (inference will be slower ~10-30x)")
            logger.info("Recommend using GPU for better performance")

        return device

    def _load_model(self):
        """Load the model and tokenizer with appropriate configuration."""
        try:
            # Import here to avoid requiring transformers if not using privacy mode
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                token=self.hf_token,
                trust_remote_code=True
            )

            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Loading model (this may take a few minutes on first run)...")

            if self.load_in_8bit:
                # Load with 8-bit quantization
                logger.info("Using 8-bit quantization (reduces memory to ~8GB)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    load_in_8bit=True,
                    device_map="auto",
                    token=self.hf_token,
                    trust_remote_code=True
                )
            else:
                # Load in full precision
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    token=self.hf_token,
                    trust_remote_code=True
                )

                if self.device == "cpu":
                    self.model = self.model.to("cpu")

            self.model.eval()  # Set to evaluation mode
            logger.info("Model loaded successfully!")

        except Exception as e:
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                logger.error(
                    "Authentication error: Llama 3.1 is a gated model. "
                    "Please set HUGGINGFACE_TOKEN in .env file or pass hf_token parameter. "
                    "Get your token from: https://huggingface.co/settings/tokens"
                )
            elif "not found" in str(e).lower():
                logger.error(
                    f"Model not found: {self.model_path}. "
                    "Check model ID or path. For Llama 3.1, use: meta-llama/Meta-Llama-3.1-8B-Instruct"
                )
            else:
                logger.error(f"Error loading model: {e}")

            raise

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text from a list of chat messages.

        Args:
            messages: List of dicts with 'role' and 'content' keys
                     Example: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1, higher = more random)
            top_p: Nucleus sampling parameter

        Returns:
            Generated text content
        """
        try:
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096  # Llama 3.1 context length
            )

            # Move to appropriate device
            if self.device == "cuda" and not self.load_in_8bit:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated tokens (skip input)
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return generated_text.strip()

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "GPU out of memory! Try: (1) Close other GPU applications, "
                "(2) Use --device cpu, (3) Restart with smaller batch size"
            )
            raise
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    def batch_generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """
        Generate text for multiple message sequences in a batch.

        Args:
            messages_list: List of message sequences
            max_tokens: Maximum tokens to generate per sequence
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            List of generated text strings
        """
        # For now, process sequentially
        # TODO: Implement true batching for better performance
        results = []
        for messages in messages_list:
            result = self.generate(messages, max_tokens, temperature, top_p)
            results.append(result)
        return results

    def __del__(self):
        """Clean up resources on deletion."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
