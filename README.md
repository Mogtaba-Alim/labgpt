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
└── batch_summary_20251202_143200.json
```

The batch testing system provides a streamlined way to evaluate multiple models systematically.