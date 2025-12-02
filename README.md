# LabGPT vLLM Inference

A streamlined RAG-augmented inference system that uses vLLM server API calls for efficient model testing and evaluation.

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
- **Multi-Model Testing**: Easy comparison across different models
- **Scalable**: Test hundreds of prompts efficiently
- **RAG Integration**: Full retrieval-augmented generation with your documents

---

## Installation

### Prerequisites
- Python 3.8+
- Access to UHN network
- PROMPTER API key

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

Current models:
- **Qwen/Qwen3-8B**: Alibaba's 8B parameter model
- **google/gemma-3-4b-it**: Google's 4B instruction-tuned model  
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B**: DeepSeek's 32B reasoning model
- **openai/GPT-OSS-120B**: OpenAI's 120B open-source model


## Batch Testing

The sequential batch testing system allows you to efficiently test multiple prompts against multiple models with model-specific optimized parameters.

### Quick Start

1. **Configure your settings** in `prompts_config.json`:

```json
{
  "settings": {
    "api_key": "your-api-key-here",
    "index_dir": "indices/rag_demo_storage",
    "output_dir": "results",
    "models": {
      "Qwen3-8B": {
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens": 800,
        "expand": false,
        "cited_spans": false,
        "preset": "default"
      },
      "DeepSeek-R1-Distill-Qwen-32B": {
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 1000,
        "expand": true,
        "cited_spans": true,
        "preset": "research"
      }
    }
  },
  "prompts": [
    {
      "id": "crispr_001",
      "prompt": "How does CRISPR-Cas9 gene editing work at the molecular level?"
    }
  ]
}
```

2. **Test with dry run** (recommended first step):

```bash
python batch_proc.py prompts_config.json --dry-run
```

3. **Run the full batch test**:

```bash
python batch_proc.py prompts_config.json
```

4. **Override models** (optional):

```bash
python batch_proc.py prompts_config.json --models "Qwen3-8B" "DeepSeek-R1-Distill-Qwen-32B"
```

### Model-Specific Optimization

Each model in the configuration has optimized parameters:

- **Smaller Models** (Qwen3-8B, Gemma-3-4b-it): Higher temperature, basic RAG preset
- **Larger Models** (DeepSeek-R1-32B, GPT-OSS-120B): Lower temperature, research preset with advanced features

### Output Structure

Results are saved with timestamps:

```
results/
├── crispr_001_Qwen3-8B_20251202_143022.json
├── crispr_001_DeepSeek-R1-Distill-Qwen-32B_20251202_143045.json
└── batch_summary_20251202_143200.json
```

The batch testing system provides a streamlined way to evaluate multiple models systematically.

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


## Configuration

### Environment Variables

```bash
# Optional: Set default values
export VLLM_BASE_URL="https://prompter.uhndata.io/proxy/v1/chat/completions"
export RAG_STORAGE_DIR="indices/rag_demo_storage"
export PROMPTER_API_KEY="your-api-key"
```

### Single Query Configuration

For individual queries, use command-line arguments with `vllm_inference.py`:

```bash
python vllm_inference.py "Your query" \
  --model "Qwen/Qwen2.5-8B-Instruct" \
  --prompter-api-key "your-key" \
  --index indices/rag_demo_storage \
  --temperature 0.4 \
  --top-p 0.9 \
  --max-new-tokens 800
```

### Batch Testing Configuration

For batch testing, configure models and prompts in `prompts_config.json`:

- **Model-specific parameters**: Each model has optimized temperature, token limits, and RAG settings
- **Prompt management**: Simple ID and prompt structure for easy management
- **Output control**: Configurable result directory and file naming

All configuration options are detailed above in the batch testing examples.

---

## Examples

### Single Query Example

```bash
python vllm_inference.py \
  "What are the latest developments in CRISPR base editing?" \
  --model "Qwen/Qwen2.5-8B-Instruct" \
  --prompter-api-key "your-key" \
  --index indices/rag_demo_storage \
  --expand --cited-spans --preset research \
  --output-json results/crispr_analysis.json
```

### Multi-Model Comparison

```bash
python vllm_inference.py \
  "Compare different approaches to cancer immunotherapy" \
  --test-models "Qwen/Qwen2.5-8B-Instruct" "google/gemma-3-4b-it" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --prompter-api-key "your-key" \
  --index indices/rag_demo_storage \
  --output-json results/immunotherapy_comparison.json
```

### Batch Testing Example

```bash
# Test all configured prompts and models
python batch_proc.py prompts_config.json

# Test specific models only
python batch_proc.py prompts_config.json --models "Qwen/Qwen2.5-8B-Instruct" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# Dry run to preview what will be executed
python batch_proc.py prompts_config.json --dry-run
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
- Add delays between API calls if making multiple requests
- Check vLLM server capacity and limits

**6. Batch Testing Issues**
- **Config file not found**: Check the path to `prompts_config.json`
- **Model errors**: Ensure model names use full Hugging Face IDs (e.g., `"Qwen/Qwen2.5-8B-Instruct"`)
- **API key errors**: Verify your API key is set correctly in `prompts_config.json`
- **Permission errors**: Use `--dry-run` first to test configuration

### Performance Tips

- **Use appropriate `--top-k`**: 3-5 for focused queries, 5-10 for complex topics
- **Enable `--expand`**: Improves retrieval quality for complex queries
- **Multi-model testing**: Use `--test-models` to compare responses across different models
- **Monitor API costs**: Track usage when testing multiple models

---

## Directory Structure

After cleanup, your repository contains:

```
labgpt/
├── vllm_inference.py           # Main vLLM inference script
├── batch_proc.py               # Batch testing script
├── prompts_config.json         # Batch testing configuration
├── RAG/                        # Retrieval-Augmented Generation
│   ├── pipeline.py             # Main RAG pipeline
│   ├── ingestion/              # Document processing
│   ├── retrieval/              # Search and ranking
│   └── generation/             # Answer generation
├── indices/                    # Pre-built RAG indices (gitignored)
│   └── rag_demo_storage/       # Default index
├── results/                    # Test results and outputs (gitignored)
├── data/                       # Source documents (optional)
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

---

For more detailed information about specific components, see:
- [RAG System Documentation](RAG/README.md)