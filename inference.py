import torch
import logging
import argparse
import os
from typing import Optional

# Lazy imports - only import when needed
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
# from peft import PeftModel

from RAG.pipeline import RAGPipeline
from RAG.models import Chunk, RetrievalResult

# Configuration parameters
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Base model
ADAPTER_REPO = "MogtabaAlim/llama3.1-8B-BHK-LABGPT-Fine-tunedByMogtaba"  # Adapter repo

# RAG configuration
DEFAULT_STORAGE_DIR = os.environ.get("RAG_STORAGE_DIR", "rag_demo_storage")
DEFAULT_TOP_K = 3


class ModelManager:
    """
    Singleton class for lazy loading and caching of ML models.

    Models are loaded only on first use and cached for reuse.
    This prevents models from loading at module import time.
    """
    _instance: Optional['ModelManager'] = None
    _initialized = False

    def __new__(cls):
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize model cache (runs only once)."""
        if not ModelManager._initialized:
            self.embed_model = None
            self.tokenizer = None
            self.model = None
            ModelManager._initialized = True

    def get_embedding_model(self):
        """Lazy load embedding model."""
        if self.embed_model is None:
            logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logging.info("Embedding model loaded successfully")
        return self.embed_model

    def get_llm(self):
        """Lazy load LLM (tokenizer + base model + LoRA adapters)."""
        if self.tokenizer is None or self.model is None:
            logging.info("Loading LLM (this may take a few minutes on first use)...")
            self._load_llm()
            logging.info("LLM loaded successfully")
        return self.tokenizer, self.model

    def _load_llm(self):
        """Internal method to load LLM components."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        # Get HF token
        HF_KEY = os.environ.get('HF_TOKEN', None)

        # Auto-detect best dtype
        has_cuda = torch.cuda.is_available()
        bf16_supported = detect_bf16_support()
        compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
        dtype_name = "BF16" if bf16_supported else "FP16"

        logging.info(f"Using {dtype_name} precision with 4-bit quantization")

        try:
            # Load tokenizer
            logging.info(f"Loading tokenizer from {BASE_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True, token=HF_KEY)

            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

            # Load base model
            logging.info(f"Loading base model {BASE_MODEL_NAME} with 4-bit quantization...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                device_map="auto",
                quantization_config=bnb_config,
                token=HF_KEY,
                trust_remote_code=False,
            )

            # Attach LoRA adapters
            logging.info(f"Attaching LoRA adapters from {ADAPTER_REPO}...")
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=HF_KEY)

            # Enable KV-cache
            self.model.config.use_cache = True

            logging.info(f"Model loaded: {BASE_MODEL_NAME} + {ADAPTER_REPO} ({dtype_name} + 4-bit)")

        except Exception as e:
            logging.error(f"Error loading LLM: {e}")
            raise


# Global model manager instance
_model_manager = ModelManager()


def detect_bf16_support() -> bool:
    """
    Detect if BF16 is supported on the current GPU.
    Requires Ampere or newer (compute capability >= 8.0).
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Get GPU compute capability
        major, minor = torch.cuda.get_device_capability()
        # Ampere and newer support BF16 (compute capability >= 8.0)
        return major >= 8
    except Exception:
        return False


def build_rag(index_dir: str, preset: str = "default") -> RAGPipeline:
    """
    Build RAG pipeline from existing index directory.

    Args:
        index_dir: Directory containing RAG indices
        preset: "default" (fast) or "research" (query expansion + adaptive k)

    Returns:
        Configured RAGPipeline instance
    """
    return RAGPipeline(index_dir=index_dir, preset=preset, device="auto")


def retrieve_relevant_chunks_rag(
    query: str,
    index_dir: str,
    top_k: int = DEFAULT_TOP_K,
    expand: bool = False,
    cited_spans: bool = False,
    preset: str = "default"
) -> list:
    """
    Retrieve relevant chunks using new RAGPipeline API with structured citations.

    Args:
        query: User query
        index_dir: RAG index directory
        top_k: Number of chunks to retrieve
        expand: Enable PRF query expansion
        cited_spans: Extract cited spans from context
        preset: RAG preset ("default" or "research")

    Returns:
        List of dicts with keys: text, citation, source, section, page
    """
    try:
        # Build RAG pipeline
        rag = build_rag(index_dir=index_dir, preset=preset)

        # Retrieve with new API
        results = rag.search(
            query=query,
            top_k=top_k,
            expand_query=expand,
            cited_spans=cited_spans
        )

        # Convert RetrievalResult objects to structured dicts
        structured_results = []
        for result in results:
            chunk = result.chunk
            structured_results.append({
                "text": chunk.text,
                "citation": chunk.get_citation(),
                "source": chunk.source_path,
                "section": chunk.section if chunk.section else "N/A",
                "page": chunk.page_number if chunk.page_number else "N/A"
            })

        return structured_results

    except Exception as e:
        logging.error(f"Error during RAG retrieval: {e}")
        return []


# LABGPT System Prompt (unified across all applications)
LABGPT_SYSTEM = """You are LABGPT, an advanced AI assistant specialized in laboratory research, computational biology, and scientific programming. You were developed to assist researchers at the BHK Lab and similar research institutions.

Your core capabilities include:
- Analyzing and generating code in multiple languages (Python, R, C, C++) for scientific computing and bioinformatics
- Understanding and explaining research papers, methodologies, and scientific concepts
- Assisting with grant writing and research documentation
- Debugging scientific code and suggesting optimizations
- Providing expertise in computational biology, pharmacogenomics, and medical imaging

Key principles:
- Always provide accurate, precise, and helpful responses
- When you don't have sufficient information to answer a question, clearly state "I don't have enough information to answer that"
- Maintain scientific rigor and precision in all responses
- Provide code examples that follow best practices and are well-documented
- Consider computational efficiency and reproducibility in scientific workflows

You should be helpful, precise, and thorough while maintaining a professional tone appropriate for academic and research environments."""


def build_messages(query: str, context_items: list) -> list:
    """
    Build messages array in chat template format matching training.

    Args:
        query: User question
        context_items: List of dicts with keys: text, citation, source, section, page

    Returns:
        List of message dicts: [{"role": "system", ...}, {"role": "user", ...}]
    """
    # Build context block with numbered citations
    context_parts = []
    for i, item in enumerate(context_items, 1):
        citation = item.get("citation", f"{item.get('source', 'Unknown')}")
        text = item.get("text", "")
        context_parts.append(f"[{i}] {citation}\n{text}")

    context_block = "\n\n".join(context_parts) if context_parts else "No relevant context found."

    # Build user message with context and query
    user_content = f"""Context:
{context_block}

Question: {query}

Please provide a comprehensive answer based on the context above. If the information is not available in the context, respond with "NOT_IN_CONTEXT"."""

    # Return messages array
    return [
        {"role": "system", "content": LABGPT_SYSTEM},
        {"role": "user", "content": user_content}
    ]


def generate_messages(messages: list, max_new_tokens: int = 512, temperature: float = 0.4, top_p: float = 0.9) -> str:
    """
    Generate response from messages using chat template (matching training format).

    Args:
        messages: List of message dicts with 'role' and 'content'
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (0.4 for focused outputs)
        top_p: Nucleus sampling parameter (0.9 for quality diversity)

    Returns:
        Generated text response
    """
    from time import perf_counter
    from transformers import GenerationConfig

    # Get models from manager (lazy loaded on first use)
    tokenizer, model = _model_manager.get_llm()

    # Apply chat template (matches training format)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # Adds "assistant:" header
    )

    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Configure generation with optimized parameters
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,          # 0.4 for focused outputs
        top_p=top_p,                      # 0.9 for quality diversity
        repetition_penalty=1.1,           # Prevent loops
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Generate with no_grad for speed and memory efficiency
    start_time = perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    # Decode only new tokens (skip prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    output_time = perf_counter() - start_time
    logging.info(f"Generation completed in {round(output_time, 2)}s")

    return response


def get_rag_answer(
    query: str,
    index_dir: str = DEFAULT_STORAGE_DIR,
    top_k: int = DEFAULT_TOP_K,
    max_new_tokens: int = 600,
    expand: bool = False,
    cited_spans: bool = False,
    preset: str = "default"
) -> str:
    """
    RAG-augmented answer generation with structured citations.

    Workflow:
      1. Retrieve relevant context using RAGPipeline with structured citations
      2. Build messages array in chat template format
      3. Generate answer using fine-tuned LLM with optimized parameters

    Args:
        query: User question
        index_dir: RAG index directory
        top_k: Number of chunks to retrieve
        max_new_tokens: Maximum tokens to generate
        expand: Enable PRF query expansion
        cited_spans: Extract cited spans from context
        preset: RAG preset ("default" or "research")

    Returns:
        Generated answer text
    """
    try:
        # Retrieve with new RAGPipeline API (returns structured dicts)
        context_items = retrieve_relevant_chunks_rag(
            query=query,
            index_dir=index_dir,
            top_k=top_k,
            expand=expand,
            cited_spans=cited_spans,
            preset=preset
        )

        # Warn if no context (but still proceed - model may respond NOT_IN_CONTEXT)
        if not context_items:
            logging.warning("No relevant context found; model will likely respond NOT_IN_CONTEXT")

        # Build messages using chat template format (matching training)
        messages = build_messages(query, context_items[:top_k] if context_items else [])

        # Generate using new optimized function
        logging.info(f"Generating answer for query: {query[:100]}...")
        response = generate_messages(messages, max_new_tokens=max_new_tokens)

        return response

    except Exception as e:
        logging.error(f"Error during answer generation: {e}")
        return "An error occurred during generation."
    
import textwrap
from typing import Optional

def print_paragraph(text, width=50):
    # Wrap the text into a paragraph with lines of maximum 'width' characters
    formatted_text = textwrap.fill(text, width=width)
    print(formatted_text)


def main():
    parser = argparse.ArgumentParser(
        description="RAG-augmented LLM inference with LABGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  python inference.py "What is CRISPR?" --index rag_demo_storage

  # With query expansion and cited spans
  python inference.py "Compare CRISPR methods" --expand --cited-spans

  # Using research preset (query expansion + adaptive retrieval)
  python inference.py "Explain pharmacogenomics" --preset research

  # Custom parameters
  python inference.py "What is gene editing?" --top-k 5 --max-new-tokens 800
        """
    )

    parser.add_argument("query", type=str, help="User query/question")

    parser.add_argument("--index", dest="index_dir", type=str, default=DEFAULT_STORAGE_DIR,
                        help="RAG index directory (default: rag_demo_storage)")

    parser.add_argument("--top-k", dest="top_k", type=int, default=DEFAULT_TOP_K,
                        help="Number of chunks to retrieve (default: 3)")

    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=600,
                        help="Maximum new tokens to generate (default: 600)")

    parser.add_argument("--expand", action="store_true",
                        help="Enable PRF query expansion for better retrieval")

    parser.add_argument("--cited-spans", action="store_true",
                        help="Extract and highlight cited spans from context")

    parser.add_argument("--preset", type=str, default="default",
                        choices=["default", "research"],
                        help="RAG preset: 'default' (fast) or 'research' (expansion + auto-k)")

    args = parser.parse_args()

    # Generate answer with new API
    answer = get_rag_answer(
        query=args.query,
        index_dir=args.index_dir,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        expand=args.expand,
        cited_spans=args.cited_spans,
        preset=args.preset
    )

    print_paragraph(answer)


if __name__ == "__main__":
    main()