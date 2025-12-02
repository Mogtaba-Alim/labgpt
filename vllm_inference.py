#!/usr/bin/env python3
"""
vllm_inference.py

LabGPT inference script that uses vLLM server API calls instead of local model loading.
Maintains the same interface as inference.py but delegates generation to a vLLM server.

Usage:
    python vllm_inference.py "What is CRISPR?" --model meta-llama/Meta-Llama-3.1-8B-Instruct
    python vllm_inference.py "Compare CRISPR methods" --model microsoft/DialoGPT-medium --expand
"""

import os
import logging
import argparse
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import requests
from time import perf_counter

# Import existing RAG components
from sentence_transformers import SentenceTransformer
from RAG.pipeline import RAGPipeline
from RAG.models import Chunk, RetrievalResult

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_STORAGE_DIR = os.environ.get("RAG_STORAGE_DIR", "rag_demo_storage")
DEFAULT_TOP_K = 3

# vLLM server configuration
DEFAULT_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "https://prompter.uhndata.io/proxy/v1/chat/completions")
DEFAULT_MODEL = "Qwen/Qwen2.5-8B-Instruct"

# Load embedding model for RAG (same as original)
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# LabGPT system prompt (same as original)
LABGPT_SYSTEM = """You are LABGPT, an advanced AI assistant specialized in laboratory research, computational biology, and scientific programming. You were developed to assist researchers at the BHK Lab and similar research institutions.

Your core capabilities include:
- Analyzing and generating code in multiple languages (Python, R, C, C++) for scientific computing and bioinformatics
- Understanding and explaining research papers, methodologies, and scientific concepts
- Assisting with grant writing and research documentation
- Debugging scientific code and suggesting optimizations
- Providing expertise in computational biology, pharmacogenomics, and medical imaging

Key principles:
- Always provide accurate, grounded responses based on the provided context
- When information is not available in the context, clearly state "I don't have enough information to answer that" or "That information is not in the provided context"
- Maintain scientific rigor and precision in all responses
- Provide code examples that follow best practices and are well-documented
- Consider computational efficiency and reproducibility in scientific workflows

You should be helpful, precise, and thorough while maintaining a professional tone appropriate for academic and research environments."""


@dataclass
class VLLMConfig:
    """Configuration for vLLM server connection."""
    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout: int = 120
    max_retries: int = 3


class VLLMClient:
    """Client for making API calls to vLLM server using OpenAI-compatible API."""
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.session = requests.Session()
        
        # Set up headers
        headers = {
            "Content-Type": "application/json",
        }
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        self.session.headers.update(headers)
        
        logging.info(f"VLLMClient initialized for {config.base_url} with model {config.model}")
    
    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.4,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate chat completion using vLLM's OpenAI-compatible API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated text response
        """
        # Handle both full URL and base URL cases
        if "/v1/chat/completions" in self.config.base_url:
            url = self.config.base_url
        else:
            url = f"{self.config.base_url}/v1/chat/completions"
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            **kwargs
        }
        
        start_time = perf_counter()
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                generation_time = perf_counter() - start_time
                logging.info(f"Generation completed in {round(generation_time, 2)}s (attempt {attempt + 1})")
                
                return content.strip()
                
            except requests.exceptions.RequestException as e:
                logging.warning(f"API request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
            except (KeyError, IndexError) as e:
                logging.error(f"Invalid response format: {e}")
                raise
    


# RAG functions (same as original inference.py)
def build_rag(index_dir: str, preset: str = "default") -> RAGPipeline:
    """Build RAG pipeline from existing index directory."""
    return RAGPipeline(index_dir=index_dir, preset=preset, device="auto")


def retrieve_relevant_chunks_rag(
    query: str,
    index_dir: str,
    top_k: int = DEFAULT_TOP_K,
    expand: bool = False,
    cited_spans: bool = False,
    preset: str = "default"
) -> list:
    """Retrieve relevant chunks using RAGPipeline API."""
    try:
        rag = build_rag(index_dir=index_dir, preset=preset)
        
        results = rag.search(
            query=query,
            top_k=top_k,
            expand_query=expand,
            cited_spans=cited_spans
        )
        
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


def build_messages(query: str, context_items: list) -> list:
    """Build messages array in chat template format (same as original)."""
    context_parts = []
    for i, item in enumerate(context_items, 1):
        citation = item.get("citation", f"{item.get('source', 'Unknown')}")
        text = item.get("text", "")
        context_parts.append(f"[{i}] {citation}\n{text}")
    
    context_block = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    
    user_content = f"""Context:
{context_block}

Question: {query}

Please provide a comprehensive answer based on the context above. If the information is not available in the context, respond with "NOT_IN_CONTEXT"."""
    
    return [
        {"role": "system", "content": LABGPT_SYSTEM},
        {"role": "user", "content": user_content}
    ]


def get_rag_answer_vllm(
    query: str,
    vllm_client: VLLMClient,
    index_dir: str = DEFAULT_STORAGE_DIR,
    top_k: int = DEFAULT_TOP_K,
    max_new_tokens: int = 600,
    expand: bool = False,
    cited_spans: bool = False,
    preset: str = "default",
    temperature: float = 0.4,
    top_p: float = 0.9
) -> str:
    """
    RAG-augmented answer generation using vLLM server.
    
    Same workflow as original but uses vLLM API instead of local model.
    """
    try:
        # Retrieve context (same as original)
        context_items = retrieve_relevant_chunks_rag(
            query=query,
            index_dir=index_dir,
            top_k=top_k,
            expand=expand,
            cited_spans=cited_spans,
            preset=preset
        )
        
        if not context_items:
            logging.warning("No relevant context found; model will likely respond NOT_IN_CONTEXT")
        
        # Build messages (same as original)
        messages = build_messages(query, context_items[:top_k] if context_items else [])
        
        # Generate using vLLM API
        logging.info(f"Generating answer for query: {query[:100]}... using model {vllm_client.config.model}")
        response = vllm_client.generate_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        return response
        
    except Exception as e:
        logging.error(f"Error during answer generation: {e}")
        return "An error occurred during generation."


def test_multiple_models(
    query: str,
    models: List[str],
    vllm_base_url: str,
    api_key: Optional[str] = None,
    index_dir: str = DEFAULT_STORAGE_DIR,
    **kwargs
) -> Dict[str, str]:
    """
    Test the same query across multiple models.
    
    Args:
        query: Question to ask
        models: List of model names to test
        vllm_base_url: Base URL of vLLM server
        api_key: API key for authentication (optional)
        index_dir: RAG index directory
        **kwargs: Additional parameters for generation
        
    Returns:
        Dict mapping model names to their responses
    """
    results = {}
    
    for model in models:
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing model: {model}")
        logging.info(f"{'='*60}")
        
        try:
            config = VLLMConfig(base_url=vllm_base_url, model=model, api_key=api_key)
            client = VLLMClient(config)
            
            response = get_rag_answer_vllm(
                query=query,
                vllm_client=client,
                index_dir=index_dir,
                **kwargs
            )
            
            results[model] = response
            
        except Exception as e:
            logging.error(f"Error testing model {model}: {e}")
            results[model] = f"ERROR: {str(e)}"
    
    return results


def print_paragraph(text, width=80):
    """Format text into paragraphs (same as original)."""
    import textwrap
    formatted_text = textwrap.fill(text, width=width)
    print(formatted_text)


def main():
    parser = argparse.ArgumentParser(
        description="RAG-augmented LLM inference using vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query with default model
  python vllm_inference.py "What is CRISPR?"
  
  # Specify different model
  python vllm_inference.py "Compare CRISPR methods" --model microsoft/DialoGPT-medium
  
  # Test multiple models
  python vllm_inference.py "Explain pharmacogenomics" \\
    --test-models meta-llama/Meta-Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct
  
  # With RAG enhancements
  python vllm_inference.py "What is gene editing?" --expand --cited-spans --preset research
  
  # Custom API endpoint
  python vllm_inference.py "Question" --vllm-url https://prompter.uhndata.io/proxy/v1/chat/completions
        """
    )
    
    parser.add_argument("query", type=str, help="User query/question")
    
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name to use (default: {DEFAULT_MODEL})")
    
    parser.add_argument("--vllm-url", type=str, default=DEFAULT_VLLM_BASE_URL,
                        help=f"API endpoint URL (default: {DEFAULT_VLLM_BASE_URL})")
    
    parser.add_argument("--prompter-api-key", type=str,
                        help="API key for Prompter server (if required)")
    
    parser.add_argument("--test-models", nargs="+", metavar="MODEL",
                        help="Test multiple models and compare results")
    
    parser.add_argument("--index", dest="index_dir", type=str, default=DEFAULT_STORAGE_DIR,
                        help="RAG index directory (default: rag_demo_storage)")
    
    parser.add_argument("--top-k", dest="top_k", type=int, default=DEFAULT_TOP_K,
                        help="Number of chunks to retrieve (default: 3)")
    
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=600,
                        help="Maximum new tokens to generate (default: 600)")
    
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature (default: 0.4)")
    
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling parameter (default: 0.9)")
    
    parser.add_argument("--expand", action="store_true",
                        help="Enable PRF query expansion for better retrieval")
    
    parser.add_argument("--cited-spans", action="store_true",
                        help="Extract and highlight cited spans from context")
    
    parser.add_argument("--preset", type=str, default="default",
                        choices=["default", "research"],
                        help="RAG preset: 'default' (fast) or 'research' (expansion + auto-k)")
    
    parser.add_argument("--output-json", type=str,
                        help="Save results to JSON file (useful for multi-model testing)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.test_models:
        # Test multiple models
        print(f"\n🧪 Testing query across {len(args.test_models)} models...")
        print(f"Query: {args.query}")
        print(f"API Endpoint: {args.vllm_url}")
        
        results = test_multiple_models(
            query=args.query,
            models=args.test_models,
            vllm_base_url=args.vllm_url,
            api_key=args.prompter_api_key,
            index_dir=args.index_dir,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            expand=args.expand,
            cited_spans=args.cited_spans,
            preset=args.preset
        )
        
        # Display results
        for model, response in results.items():
            print(f"\n{'='*80}")
            print(f"MODEL: {model}")
            print(f"{'='*80}")
            print_paragraph(response)
        
        # Save to JSON if requested
        if args.output_json:
            output_data = {
                "query": args.query,
                "models": results,
                "parameters": {
                    "top_k": args.top_k,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "expand": args.expand,
                    "cited_spans": args.cited_spans,
                    "preset": args.preset
                }
            }
            
            with open(args.output_json, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Results saved to {args.output_json}")
    
    else:
        # Single model inference
        config = VLLMConfig(
            base_url=args.vllm_url,
            model=args.model,
            api_key=args.prompter_api_key
        )
        
        client = VLLMClient(config)
        print(f"🤖 Using model: {args.model}")
        print(f"❓ Query: {args.query}")
        
        # Generate answer
        answer = get_rag_answer_vllm(
            query=args.query,
            vllm_client=client,
            index_dir=args.index_dir,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            expand=args.expand,
            cited_spans=args.cited_spans,
            preset=args.preset
        )
        
        print(f"\n{'='*80}")
        print("RESPONSE:")
        print(f"{'='*80}")
        print_paragraph(answer)
    
    return 0


if __name__ == "__main__":
    exit(main())
