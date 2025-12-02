# LabGPT vLLM Inference

A streamlined RAG-augmented inference system that uses vLLM server API calls for efficient model testing and evaluation.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [vLLM Inference](#vllm-inference)
- [RAG System](#rag-system)
- [Batch Testing](#batch-testing)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

LabGPT vLLM Inference provides a clean, efficient way to test multiple language models against your documents using a vLLM server endpoint. This architecture offers several advantages:

```
┌─────────────────────────────────────────────────────────────┐
│  RAG Pipeline (Local)                                       │
│  ─────────────────────────────────────────────────────────  │
│  Documents → Vector Index → Context Retrieval               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  vLLM Inference (Remote API)                               │
│  ─────────────────────────────────────────────────────────  │
│  Context + Query → vLLM Server → Model Response            │
└─────────────────────────────────────────────────────────────┘
```

**Key Benefits:**
- **No Local Model Loading**: Leverage remote GPU resources via API calls
- **Multi-Model Testing**: Easy comparison across different models
- **Scalable**: Test hundreds of prompts efficiently
- **Cost-Effective**: Pay per API call instead of maintaining local GPUs
- **RAG Integration**: Full retrieval-augmented generation with your documents

---

## Installation

### Prerequisites
- Python 3.8+
- Access to a vLLM server endpoint
- API key for authentication (if required)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd labgpt
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK resources:**
   ```bash
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
   ```

---

## Quick Start

### 1. Basic Single Model Query

```bash
python vllm_inference.py "What is CRISPR gene editing?" \
  --model Qwen/Qwen3-8B \
  --prompter-api-key "your-api-key" \
  --index indices/rag_demo_storage
```

### 2. Multi-Model Comparison

```bash
python vllm_inference.py "Explain pharmacogenomics" \
  --test-models Qwen/Qwen3-8B google/gemma-3-4b-it \
  --prompter-api-key "your-api-key" \
  --index indices/rag_demo_storage \
  --output-json results/comparison.json
```

### 3. Enhanced RAG Features

```bash
python vllm_inference.py "How does machine learning apply to drug discovery?" \
  --model Qwen/Qwen3-8B \
  --prompter-api-key "your-api-key" \
  --index indices/rag_demo_storage \
  --expand --cited-spans --preset research
```

---

## vLLM Inference

### Available Models

Check configured models:
```bash
python vllm_model_config.py
```

Current models:
- **Qwen/Qwen3-8B**: Alibaba's 8B parameter model
- **google/gemma-3-4b-it**: Google's 4B instruction-tuned model  
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B**: DeepSeek's 32B reasoning model
- **openai/GPT-OSS-120B**: OpenAI's 120B open-source model

### Command Line Options

```bash
python vllm_inference.py [QUERY] [OPTIONS]

Required:
  QUERY                    Your question or prompt

Model Selection:
  --model MODEL           Single model to use (default: Qwen/Qwen3-8B)
  --test-models MODEL...  Test multiple models and compare

API Configuration:
  --vllm-url URL          API endpoint (default: https://prompter.uhndata.io/proxy/v1/chat/completions)
  --prompter-api-key KEY  API key for authentication

RAG Configuration:
  --index DIR             RAG index directory (default: rag_demo_storage)
  --top-k N               Number of context chunks (default: 3)
  --expand                Enable query expansion for better retrieval
  --cited-spans           Extract specific supporting text segments
  --preset PRESET         RAG preset: 'default' or 'research'

Generation Parameters:
  --temperature FLOAT     Sampling temperature (default: 0.4)
  --top-p FLOAT          Nucleus sampling (default: 0.9)
  --max-new-tokens INT   Maximum tokens to generate (default: 600)

Output:
  --output-json FILE     Save results to JSON file
```

---

## RAG System

The RAG (Retrieval-Augmented Generation) system provides context-aware responses by retrieving relevant information from your document collection.

### Features

- **Hybrid Retrieval**: Combines dense (FAISS) and sparse (BM25) search
- **Query Expansion**: Enhances queries with related terms from initial results
- **Cited Spans**: Extracts specific text segments that support answers
- **Cross-Encoder Reranking**: Improves relevance scoring
- **Document Types**: Supports PDFs, text files, code, and structured documents

### RAG Presets

- **default**: Fast retrieval with basic features
- **research**: Enhanced retrieval with expansion, auto-k, and telemetry

### Building Your Own Index

```bash
# Using the RAG CLI
python -m RAG.cli ingest --docs /path/to/documents --index my_index

# Check index status
python -m RAG.cli status --index my_index

# Interactive search
python -m RAG.cli interactive --index my_index
```

---

## Batch Testing

### Sequential Testing with JSON Configuration

Create a `prompts_config.json`:

```json
{
  "settings": {
    "api_key": "your-api-key",
    "index_dir": "indices/rag_demo_storage",
    "models": ["Qwen/Qwen3-8B", "google/gemma-3-4b-it"]
  },
  "prompts": [
    {
      "id": "crispr_001",
      "category": "molecular_biology",
      "prompt": "How does CRISPR-Cas9 gene editing work?",
      "priority": "high"
    },
    {
      "id": "ml_001", 
      "category": "machine_learning",
      "prompt": "Explain supervised vs unsupervised learning",
      "priority": "medium"
    }
  ]
}
```

Run batch tests:
```bash
# Test all prompts
python sequential_batch_test.py prompts_config.json

# Filter by category or priority
python sequential_batch_test.py prompts_config.json --category molecular_biology
python sequential_batch_test.py prompts_config.json --priority high
```

### Parallel Processing for Large Batches

For hundreds of prompts, use the parallel processing script:
```bash
# Create prompts file (one per line)
cat > prompts.txt << 'EOF'
What is CRISPR gene editing?
How does machine learning work?
Explain pharmacogenomics
EOF

# Run with GNU Parallel (install: brew install parallel)
./efficient_batch_test.sh
```

---

## Configuration

### Environment Variables

```bash
# Optional: Set default values
export VLLM_BASE_URL="https://prompter.uhndata.io/proxy/v1/chat/completions"
export RAG_STORAGE_DIR="indices/rag_demo_storage"
export PROMPTER_API_KEY="your-api-key"
```

### Model Configuration

Edit `vllm_model_config.py` to add or modify available models:

```python
MODEL_CONFIGS = {
    "my-model": ModelConfig(
        name="My Custom Model",
        model_id="organization/model-name",
        description="Description of the model",
        recommended_params={"temperature": 0.4, "top_p": 0.9},
        tags=["custom", "specialized"]
    )
}
```

---

## Examples

### Research Paper Analysis

```bash
python vllm_inference.py \
  "What are the latest developments in CRISPR base editing?" \
  --model Qwen/Qwen3-8B \
  --prompter-api-key "your-key" \
  --index indices/rag_demo_storage \
  --expand --cited-spans --preset research \
  --output-json results/crispr_analysis.json
```

### Multi-Model Comparison

```bash
python vllm_inference.py \
  "Compare different approaches to cancer immunotherapy" \
  --test-models Qwen/Qwen3-8B google/gemma-3-4b-it deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --prompter-api-key "your-key" \
  --index indices/rag_demo_storage \
  --output-json results/immunotherapy_comparison.json
```

### Lab Procedure Query

```bash
python vllm_inference.py \
  "What safety protocols should I follow when working with cell cultures?" \
  --model Qwen/Qwen3-8B \
  --prompter-api-key "your-key" \
  --index indices/rag_demo_storage \
  --top-k 5
```

---

## Troubleshooting

### Common Issues

**1. 401 Unauthorized Error**
- Check your API key is correct
- Ensure the API key has proper permissions
- Verify the vLLM server endpoint is accessible

**2. ModuleNotFoundError**
- Install missing dependencies: `pip install -r requirements.txt`
- Download NLTK resources: `python -c "import nltk; nltk.download('punkt_tab')"`

**3. RAG Index Not Found**
- Check the index directory path: `--index indices/rag_demo_storage`
- Build the index if missing: `python -m RAG.cli ingest --docs /path/to/docs --index indices/rag_demo_storage`

**4. Empty or Poor Responses**
- Increase `--top-k` for more context
- Use `--expand` for better retrieval
- Try `--preset research` for enhanced features
- Check if your documents are properly indexed

**5. API Timeout or Rate Limiting**
- Reduce concurrent requests in batch testing
- Add delays between API calls
- Check vLLM server capacity and limits

### Performance Tips

- **Use appropriate `--top-k`**: 3-5 for focused queries, 5-10 for complex topics
- **Enable `--expand`**: Improves retrieval quality for complex queries
- **Batch processing**: Use sequential testing for systematic evaluation
- **Monitor API costs**: Track usage when testing many prompts

---

## Directory Structure

After cleanup, your repository contains:

```
labgpt/
├── vllm_inference.py        # Main vLLM inference script
├── vllm_model_config.py     # Model configurations
├── example_vllm_usage.py    # Usage examples
│   ├── vllm_inference.py    # Main inference script
│   ├── vllm_model_config.py # Model configurations
│   ├── example_vllm_usage.py # Usage examples
│   └── README.md            # vLLM documentation
├── RAG/                     # Retrieval-Augmented Generation
│   ├── pipeline.py          # Main RAG pipeline
│   ├── ingestion/           # Document processing
│   ├── retrieval/           # Search and ranking
│   └── generation/          # Answer generation
├── indices/                 # Pre-built RAG indices
│   └── rag_demo_storage/    # Default index
├── results/                 # Test results and outputs
├── data/                    # Source documents (optional)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

For more detailed information about specific components, see:
- [RAG System Documentation](RAG/README.md)