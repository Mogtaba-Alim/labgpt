"""
Inference adapter service for programmatic model usage in chat and grant generation.
Wraps the inference.py module for easy integration with the web application.
"""

import sys
import os
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add parent directory to Python path to import inference
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import from inference.py
from inference import (
    build_rag,
    retrieve_relevant_chunks_rag,
    build_messages,
    LABGPT_SYSTEM
)


class InferenceAdapter:
    """
    Adapter for programmatic model inference in web application.
    Provides simplified interface with project-specific model loading.

    Features model caching: Multiple InferenceAdapter instances for the same
    model_path will share the same loaded model to reduce memory usage.
    """

    # Class-level model cache: {model_path: (tokenizer, model)}
    _model_cache: Dict[str, tuple] = {}

    def __init__(self, model_path: Optional[str] = None, rag_index_dir: Optional[str] = None):
        """
        Initialize inference adapter.

        Args:
            model_path: Path to trained model directory (contains adapter_config.json)
                       If None, uses default HuggingFace adapter
            rag_index_dir: Path to RAG index directory (None = RAG disabled)
        """
        self.model_path = model_path
        self.rag_index_dir = rag_index_dir
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

        # Load model if path provided (uses cache if available)
        if model_path:
            self._load_project_model()

    def _detect_bf16_support(self) -> bool:
        """Detect if BF16 is supported on the current GPU."""
        if not torch.cuda.is_available():
            return False
        try:
            major, minor = torch.cuda.get_device_capability()
            return major >= 8
        except Exception:
            return False

    def _load_project_model(self):
        """
        Load project-specific fine-tuned model from local path.

        Uses class-level cache to avoid loading the same model multiple times.
        If model is already in cache, reuses the cached instance.
        """
        # Check cache first
        if self.model_path in InferenceAdapter._model_cache:
            logging.info(f"Reusing cached model from {self.model_path}")
            self.tokenizer, self.model = InferenceAdapter._model_cache[self.model_path]
            self._model_loaded = True
            return

        try:
            # Get HuggingFace token
            hf_key = os.environ.get('HF_TOKEN', None)

            # Base model name
            base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

            # Auto-detect best dtype
            has_cuda = torch.cuda.is_available()
            bf16_supported = self._detect_bf16_support()
            compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
            dtype_name = "BF16" if bf16_supported else "FP16"

            logging.info(f"Loading project model from {self.model_path} with {dtype_name} + 4-bit")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, token=hf_key)

            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                quantization_config=bnb_config,
                token=hf_key,
                trust_remote_code=False,
            )

            # Load LoRA adapters from project directory
            self.model = PeftModel.from_pretrained(base_model, self.model_path, token=hf_key)
            self.model.config.use_cache = True

            # Cache the loaded model
            InferenceAdapter._model_cache[self.model_path] = (self.tokenizer, self.model)

            self._model_loaded = True
            logging.info(f"Project model loaded successfully from {self.model_path} and cached")

        except Exception as e:
            logging.error(f"Error loading project model: {e}")
            raise

    def _generate_text(self, messages: List[Dict[str, str]], max_new_tokens: int = 600,
                      temperature: float = 0.4, top_p: float = 0.9) -> str:
        """
        Generate text using loaded model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        if not self._model_loaded:
            # Fallback to inference.py default model
            from inference import generate_messages
            return generate_messages(messages, max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_p=top_p)

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def generate(
        self,
        query: str,
        use_rag: bool = False,
        top_k: int = 3,
        max_new_tokens: int = 600,
        temperature: float = 0.4,
        top_p: float = 0.9
    ) -> Dict[str, any]:
        """
        Generate response for a query.

        Args:
            query: User query/question
            use_rag: Whether to use RAG for context retrieval
            top_k: Number of RAG chunks to retrieve (if use_rag=True)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Dict with:
                - response: Generated text
                - citations: List of citation dicts (if use_rag=True)
                - context_used: Whether RAG context was used
        """
        if use_rag and self.rag_index_dir:
            # Retrieve relevant chunks
            chunks = retrieve_relevant_chunks_rag(
                query=query,
                index_dir=self.rag_index_dir,
                top_k=top_k,
                expand=False,
                cited_spans=True,
                preset="research"
            )

            # Build messages with context
            context_text = "\n\n".join([
                f"[Source: {chunk['source']}, Page {chunk['page']}, Section: {chunk['section']}]\n{chunk['text']}"
                for chunk in chunks
            ])

            messages = [
                {"role": "system", "content": LABGPT_SYSTEM},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
            ]

            # Generate response
            response = self._generate_text(messages, max_new_tokens, temperature, top_p)

            # Extract citations
            citations = [
                {
                    'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                    'source': chunk['source'],
                    'page': chunk['page'],
                    'section': chunk['section']
                }
                for chunk in chunks
            ]

            return {
                'response': response,
                'citations': citations,
                'context_used': True
            }
        else:
            # Direct generation without RAG
            messages = [
                {"role": "system", "content": LABGPT_SYSTEM},
                {"role": "user", "content": query}
            ]

            response = self._generate_text(messages, max_new_tokens, temperature, top_p)

            return {
                'response': response,
                'citations': [],
                'context_used': False
            }

    def generate_with_context(
        self,
        query: str,
        context_items: List[Dict],
        max_new_tokens: int = 600
    ) -> str:
        """
        Generate response with provided context (for grant generation).

        Args:
            query: User query/question
            context_items: List of context dicts with 'text' and 'citation' keys
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        messages = build_messages(query, context_items)
        response = self._generate_text(messages, max_new_tokens=max_new_tokens)
        return response

    def chat(
        self,
        messages: List[Dict[str, str]],
        use_rag: bool = False,
        top_k: int = 3,
        max_new_tokens: int = 600,
        temperature: float = 0.4,
        top_p: float = 0.9
    ) -> Dict[str, any]:
        """
        Multi-turn chat interface.

        Args:
            messages: List of message dicts with 'role' and 'content'
            use_rag: Whether to use RAG for the last user message
            top_k: Number of RAG chunks to retrieve
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Dict with response and optional citations
        """
        # Extract last user message for RAG retrieval
        last_user_msg = None
        for msg in reversed(messages):
            if msg['role'] == 'user':
                last_user_msg = msg['content']
                break

        if not last_user_msg:
            return {'response': 'No user message found', 'citations': [], 'context_used': False}

        # If RAG enabled, retrieve context and augment last message
        if use_rag and self.rag_index_dir:
            # Retrieve relevant chunks
            chunks = retrieve_relevant_chunks_rag(
                query=last_user_msg,
                index_dir=self.rag_index_dir,
                top_k=top_k,
                expand=False,
                cited_spans=True,
                preset="research"
            )

            # Build context text
            context_text = "\n\n".join([
                f"[Source: {chunk['source']}, Page {chunk['page']}, Section: {chunk['section']}]\n{chunk['text']}"
                for chunk in chunks
            ])

            # Augment last user message with context
            augmented_messages = messages[:-1] + [
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {last_user_msg}"}
            ]

            # Generate response
            response = self._generate_text(augmented_messages, max_new_tokens, temperature, top_p)

            # Extract citations
            citations = [
                {
                    'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                    'source': chunk['source'],
                    'page': chunk['page'],
                    'section': chunk['section']
                }
                for chunk in chunks
            ]

            return {
                'response': response,
                'citations': citations,
                'context_used': True
            }
        else:
            # Direct generation with conversation history
            response = self._generate_text(messages, max_new_tokens, temperature, top_p)
            return {
                'response': response,
                'citations': [],
                'context_used': False
            }
