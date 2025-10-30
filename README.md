# LabGPT Unified CLI

A comprehensive command-line interface for orchestrating all LabGPT pipelines: **RAG document indexing**, **synthetic data generation**, and **LLM fine-tuning**.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
  - [run-all](#run-all-complete-pipeline)
  - [rag](#rag-document-indexing)
  - [data-gen](#data-gen-synthetic-data-generation)
  - [train](#train-model-fine-tuning)
- [Configuration Options](#configuration-options)
- [Examples](#examples)
- [Output Structure](#output-structure)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Overview

The LabGPT Unified CLI orchestrates three powerful pipelines in a single, streamlined workflow:

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: RAG (Document Indexing)                          │
│  ─────────────────────────────────────────────────────────  │
│  Papers + Lab Docs → Vector Index → Retrieval System       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Data Generation (Synthetic Training Data)        │
│  ─────────────────────────────────────────────────────────  │
│  Code Repos + Papers → LLM → Instruction Dataset (JSONL)   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Training (Model Fine-tuning)                     │
│  ─────────────────────────────────────────────────────────  │
│  Instruction Dataset → LoRA Fine-tuning → Custom Model      │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Sequential Orchestration**: Automatic pipeline chaining with validation between stages
- **Multi-Repo Support**: Process multiple code repositories (local paths or GitHub URLs)
- **Unified Configuration**: Single command with comprehensive options for all pipelines
- **Progress Tracking**: Detailed logging with timestamped output
- **Error Recovery**: Clear error messages and automatic cleanup on failure
- **Flexible Execution**: Run all pipelines or individual stages as needed

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (for cloning repositories)
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM (32GB+ recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/labgpt.git
cd labgpt
```

### Step 2: Install Dependencies

Install all dependencies from the unified requirements file:

```bash
pip install -r requirements.txt
```

**Note:** For better dependency management, we recommend using a virtual environment:

```bash
# Using conda
conda create -n labgpt python=3.10
conda activate labgpt
pip install -r requirements.txt

# Or using venv
python -m venv labgpt-env
source labgpt-env/bin/activate  # On Windows: labgpt-env\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# For Data Generation (API mode)
CLAUDE_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key

# For Training (optional, for uploading to Hugging Face Hub)
HF_TOKEN=your_huggingface_token
```

---

## Quick Start

### Run Complete Pipeline

Process code repositories, generate training data, and fine-tune a model in one command:

```bash
python labgpt_cli.py run-all \
  --code-repos /path/to/your/repo \
  --papers /path/to/research/papers \
  --lab-docs /path/to/lab/documents \
  --output ./my-labgpt-output
```

This will:
1. **Index documents** from papers and lab-docs directories
2. **Generate instruction data** from code repository and papers
3. **Fine-tune Llama 3.1 8B** on the generated data

All outputs will be saved to `./my-labgpt-output/`:
- `rag-index/` - Vector index for retrieval
- `data_generation/` - Generated JSONL training data
- `model/` - Fine-tuned model weights
- `logs/` - Execution logs

---

## Commands

### `run-all`: Complete Pipeline

Run all three pipelines sequentially: RAG → Data Generation → Training.

**Syntax:**
```bash
python labgpt_cli.py run-all [OPTIONS]
```

**Required Arguments:**
- `--code-repos`: One or more code repository paths or GitHub URLs
- At least one of:
  - `--papers`: Path to research papers directory
  - `--lab-docs`: Path to lab documents directory

**Common Options:**
- `--output DIR`: Base output directory (default: `./labgpt-output`)

**Examples:**

```bash
# Basic usage with local repository
python labgpt_cli.py run-all \
  --code-repos /path/to/repo \
  --papers /path/to/papers \
  --lab-docs /path/to/lab-docs

# With multiple repositories (local + GitHub)
python labgpt_cli.py run-all \
  --code-repos /local/repo1 https://github.com/user/repo2 \
  --papers /path/to/papers \
  --output ./custom-output

# With custom training parameters
python labgpt_cli.py run-all \
  --code-repos /path/to/repo \
  --papers /path/to/papers \
  --num-epochs 5 \
  --learning-rate 1e-5 \
  --max-symbols 50
```

---

### `rag`: Document Indexing

Index research papers and lab documents for retrieval.

**Syntax:**
```bash
python labgpt_cli.py rag [OPTIONS]
```

**Required Arguments:**
- `--index DIR`: Output directory for RAG index
- At least one of:
  - `--papers DIR`: Path to research papers
  - `--lab-docs DIR`: Path to lab documents

**Options:**
- `--rag-preset {default,research}`: Configuration preset (default: `research`)

**Examples:**

```bash
# Index papers only
python labgpt_cli.py rag \
  --papers /path/to/papers \
  --index ./my-rag-index

# Index papers and lab documents
python labgpt_cli.py rag \
  --papers /path/to/papers \
  --lab-docs /path/to/lab-docs \
  --index ./my-rag-index \
  --rag-preset research
```

**Output:**
- Vector index files in `--index` directory
- Document metadata
- Embedding cache for fast incremental updates

**Query the Index:**
After indexing, use the RAG CLI to search:

```bash
python -m RAG.cli ask \
  --index ./my-rag-index \
  --query "How does CRISPR-Cas9 work?"
```

---

### `data-gen`: Synthetic Data Generation

Generate instruction-following training data from code repositories and research papers.

**Syntax:**
```bash
python labgpt_cli.py data-gen [OPTIONS]
```

**Required Arguments:**
- `--code-repos`: One or more code repository paths or GitHub URLs
- `--output DIR`: Output directory for generated data

**Optional Arguments:**
- `--papers DIR`: Path to research papers
- `--max-symbols INT`: Maximum symbols per file (default: 30)
- `--languages LANG [LANG ...]`: Languages to process (default: `python r c cpp`)
- `--train-ratio FLOAT`: Train/validation split (default: 0.8)

**Feature Flags:**
- `--no-debug`: Disable debug task generation
- `--no-negatives`: Disable negative example generation (NOT_IN_CONTEXT responses)
- `--no-critic`: Disable quality filtering
- `--no-dedup`: Disable deduplication

**Privacy Mode:**
- `--privacy`: Use local Llama model instead of API
- `--local-model-path PATH`: Path to local model

**Examples:**

```bash
# Basic data generation
python labgpt_cli.py data-gen \
  --code-repos /path/to/repo \
  --output ./training-data

# Multiple repositories with papers
python labgpt_cli.py data-gen \
  --code-repos /repo1 https://github.com/user/repo2 \
  --papers /path/to/papers \
  --output ./training-data \
  --max-symbols 50

# Privacy mode (local model, no API calls)
python labgpt_cli.py data-gen \
  --code-repos /path/to/repo \
  --output ./training-data \
  --privacy \
  --local-model-path meta-llama/Llama-3.1-8B

# Fast generation (no quality filtering)
python labgpt_cli.py data-gen \
  --code-repos /path/to/repo \
  --output ./training-data \
  --no-critic \
  --no-dedup
```

**Output:**
- `combined_instruct_train.jsonl` - Training data
- `combined_instruct_val.jsonl` - Validation data
- `repo_1/`, `repo_2/`, ... - Per-repository outputs

**Output Format:**
Each line in the JSONL file contains:
```json
{
  "messages": [
    {"role": "system", "content": "You are LABGPT, a helpful AI assistant..."},
    {"role": "user", "content": "How does the calculate_distance function work?"},
    {"role": "assistant", "content": "The calculate_distance function computes..."}
  ],
  "metadata": {
    "source": "repo_name/file.py",
    "task_type": "code_qa",
    "language": "python"
  }
}
```

---

### `train`: Model Fine-tuning

Fine-tune Llama 3.1 on instruction data using LoRA.

**Syntax:**
```bash
python labgpt_cli.py train [OPTIONS]
```

**Required Arguments:**
- `--train-file FILE`: Path to training JSONL file
- `--output DIR`: Output directory for trained model

**Model Options:**
- `--model-name MODEL`: Base model to fine-tune (default: `meta-llama/Llama-3.1-8B`)
- `--val-file FILE`: Validation JSONL file (optional)

**Training Hyperparameters:**
- `--num-epochs FLOAT`: Number of epochs (default: 3.0)
- `--batch-size INT`: Batch size per device (default: 1)
- `--gradient-accumulation INT`: Gradient accumulation steps (default: 16)
- `--learning-rate FLOAT`: Learning rate (default: 2e-5)
- `--max-seq-length INT`: Maximum sequence length (default: 8192)

**LoRA Configuration:**
- `--lora-rank INT`: LoRA rank (default: 16)
- `--lora-alpha INT`: LoRA alpha (default: 32)

**Optimization:**
- `--no-4bit`: Disable 4-bit quantization
- `--no-flash-attn`: Disable flash attention

**Examples:**

```bash
# Basic training
python labgpt_cli.py train \
  --train-file ./data/combined_instruct_train.jsonl \
  --output ./my-model

# With validation and custom hyperparameters
python labgpt_cli.py train \
  --train-file ./data/combined_instruct_train.jsonl \
  --val-file ./data/combined_instruct_val.jsonl \
  --output ./my-model \
  --num-epochs 5 \
  --learning-rate 1e-5 \
  --lora-rank 32

# Full precision training (no quantization)
python labgpt_cli.py train \
  --train-file ./data/combined_instruct_train.jsonl \
  --output ./my-model \
  --no-4bit
```

**Output:**
- LoRA adapter weights
- Tokenizer configuration
- Training checkpoints
- Training logs and metrics

**Load the Trained Model:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./my-model")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./my-model")
```

---

## Configuration Options

### Data Sources

| Option | Type | Description |
|--------|------|-------------|
| `--code-repos` | List[str] | Code repository paths or GitHub URLs (space-separated) |
| `--papers` | str | Path to research papers directory (PDF, TXT, MD, etc.) |
| `--lab-docs` | str | Path to lab documents directory |

### Output Locations

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | str | `./labgpt-output` | Base output directory for all pipelines |
| `--rag-index` | str | `{output}/rag-index` | RAG index directory |
| `--datagen-output` | str | `{output}/data_generation` | Data generation output |
| `--model-output` | str | `{output}/model` | Trained model output |

### RAG Pipeline

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--rag-preset` | str | `research` | Preset configuration (`default` or `research`) |
| `--rag-no-cleanup` | flag | False | Keep temporary documents after ingestion |

### Data Generation Pipeline

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max-symbols` | int | 30 | Maximum symbols to extract per file |
| `--languages` | List[str] | `python r c cpp` | Languages to process |
| `--train-ratio` | float | 0.8 | Train/validation split ratio |
| `--no-debug` | flag | False | Disable debug task generation |
| `--no-negatives` | flag | False | Disable negative example generation |
| `--no-critic` | flag | False | Disable quality filtering |
| `--no-dedup` | flag | False | Disable deduplication |
| `--privacy` | flag | False | Use local model instead of API |
| `--local-model-path` | str | None | Path to local model for privacy mode |

### Training Pipeline

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model-name` | str | `meta-llama/Llama-3.1-8B` | Base model to fine-tune |
| `--val-file` | str | None | Validation JSONL file (optional) |
| `--num-epochs` | float | 3.0 | Number of training epochs |
| `--batch-size` | int | 1 | Batch size per device |
| `--gradient-accumulation` | int | 16 | Gradient accumulation steps |
| `--learning-rate` | float | 2e-5 | Learning rate |
| `--max-seq-length` | int | 8192 | Maximum sequence length |
| `--lora-rank` | int | 16 | LoRA rank |
| `--lora-alpha` | int | 32 | LoRA alpha parameter |
| `--no-4bit` | flag | False | Disable 4-bit quantization |
| `--no-flash-attn` | flag | False | Disable flash attention |

---

## Examples

### Example 1: Quick Test Run

Test the pipeline on a small repository:

```bash
python labgpt_cli.py run-all \
  --code-repos https://github.com/bhklab/readii-fmcib \
  --output ./test-output \
  --max-symbols 10 \
  --num-epochs 1 \
  --no-critic \
  --no-dedup
```

### Example 2: Production Run with Multiple Repos

Process multiple repositories with full quality control:

```bash
python labgpt_cli.py run-all \
  --code-repos \
    /local/repo1 \
    https://github.com/user/repo2 \
    https://github.com/user/repo3 \
  --papers /data/papers \
  --lab-docs /data/lab-docs \
  --output ./production-labgpt \
  --max-symbols 50 \
  --num-epochs 3 \
  --learning-rate 2e-5
```

### Example 3: Privacy Mode (No API Calls)

Generate data and train using only local models:

```bash
python labgpt_cli.py run-all \
  --code-repos /path/to/sensitive/repo \
  --papers /path/to/papers \
  --output ./private-output \
  --privacy \
  --local-model-path meta-llama/Llama-3.1-8B
```

### Example 4: RAG Only for Document Search

Just index documents without training:

```bash
python labgpt_cli.py rag \
  --papers /data/papers \
  --lab-docs /data/lab-docs \
  --index ./document-index

# Then query interactively
python -m RAG.cli interactive --index ./document-index
```

### Example 5: Generate Data, Then Train Later

Split data generation and training into separate steps:

```bash
# Step 1: Generate training data
python labgpt_cli.py data-gen \
  --code-repos /path/to/repo \
  --papers /path/to/papers \
  --output ./my-data \
  --max-symbols 50

# Step 2: Train model (run later or on different machine)
python labgpt_cli.py train \
  --train-file ./my-data/combined_instruct_train.jsonl \
  --val-file ./my-data/combined_instruct_val.jsonl \
  --output ./my-model \
  --num-epochs 5
```

---

## Output Structure

After running `run-all`, your output directory will contain:

```
labgpt-output/
├── rag-index/                          # RAG vector index
│   ├── cache/                          # Embedding cache
│   ├── documents_metadata.json         # Document metadata
│   └── [FAISS and BM25 index files]
│
├── data_generation/                    # Generated training data
│   ├── combined_instruct_train.jsonl   # Training dataset
│   ├── combined_instruct_val.jsonl     # Validation dataset
│   ├── repo_1/                         # Per-repo outputs
│   └── repo_2/
│
├── model/                              # Fine-tuned model
│   ├── adapter_config.json             # LoRA configuration
│   ├── adapter_model.bin               # LoRA weights
│   ├── tokenizer_config.json           # Tokenizer config
│   ├── special_tokens_map.json
│   └── [checkpoint directories]
│
├── logs/                               # Execution logs
│   └── labgpt_20231115_143022.log      # Timestamped log file
│
└── temp/                               # Temporary files (auto-cleaned)
```

---

## Advanced Usage

### Resuming Failed Runs

If a pipeline fails, you can resume from the failed stage by running individual commands:

```bash
# If data generation failed after RAG completed
python labgpt_cli.py data-gen \
  --code-repos /path/to/repo \
  --output ./labgpt-output/data_generation

# If training failed after data generation completed
python labgpt_cli.py train \
  --train-file ./labgpt-output/data_generation/combined_instruct_train.jsonl \
  --output ./labgpt-output/model
```

### Multi-GPU Training

For faster training with multiple GPUs, use `torchrun` or `accelerate`:

```bash
# Generate data first
python labgpt_cli.py data-gen --code-repos /path/to/repo --output ./data

# Train with multiple GPUs
accelerate launch --multi_gpu --num_processes 4 \
  training/train_final.py \
  --train_file ./data/combined_instruct_train.jsonl \
  --output_dir ./model
```

### Custom Model Configuration

To fine-tune a different model:

```bash
python labgpt_cli.py run-all \
  --code-repos /path/to/repo \
  --papers /path/to/papers \
  --model-name meta-llama/Llama-3.2-11B \
  --output ./llama-11b-output
```

### Combining Multiple Data Sources

Generate data from different repositories, then combine manually:

```bash
# Generate from repo 1
python labgpt_cli.py data-gen \
  --code-repos /path/to/scientific-repo \
  --papers /path/to/papers \
  --output ./data-scientific

# Generate from repo 2
python labgpt_cli.py data-gen \
  --code-repos /path/to/web-repo \
  --output ./data-web

# Combine manually
cat ./data-scientific/combined_instruct_train.jsonl \
    ./data-web/combined_instruct_train.jsonl \
    > ./combined-all-train.jsonl

# Train on combined data
python labgpt_cli.py train \
  --train-file ./combined-all-train.jsonl \
  --output ./combined-model
```

---

## Troubleshooting

### Issue: Out of Memory (OOM) During Training

**Solution 1:** Reduce batch size and increase gradient accumulation:
```bash
python labgpt_cli.py train \
  --train-file ./data/train.jsonl \
  --output ./model \
  --batch-size 1 \
  --gradient-accumulation 32
```

**Solution 2:** Enable 4-bit quantization (if not already enabled):
```bash
python labgpt_cli.py train \
  --train-file ./data/train.jsonl \
  --output ./model
  # 4-bit is enabled by default
```

**Solution 3:** Reduce sequence length:
```bash
python labgpt_cli.py train \
  --train-file ./data/train.jsonl \
  --output ./model \
  --max-seq-length 4096
```

### Issue: API Rate Limit Errors (Data Generation)

**Solution 1:** Use privacy mode with local model:
```bash
python labgpt_cli.py data-gen \
  --code-repos /path/to/repo \
  --output ./data \
  --privacy \
  --local-model-path meta-llama/Llama-3.1-8B
```

**Solution 2:** Reduce generation volume:
```bash
python labgpt_cli.py data-gen \
  --code-repos /path/to/repo \
  --output ./data \
  --max-symbols 10  # Process fewer symbols per file
```

### Issue: GitHub Cloning Fails

**Solution:** Use SSH instead of HTTPS, or clone manually:
```bash
# Clone manually first
git clone https://github.com/user/repo /tmp/repo

# Then use local path
python labgpt_cli.py run-all \
  --code-repos /tmp/repo \
  --papers /path/to/papers
```

### Issue: RAG Ingestion Fails

**Solution:** Check document formats and permissions:
```bash
# Verify document directory structure
ls -R /path/to/papers

# Check file permissions
chmod -R +r /path/to/papers

# Try with verbose logging
python -m RAG.cli ingest \
  --docs /path/to/papers \
  --index ./my-index \
  --preset research
```

### Issue: Training Checkpoints Taking Too Much Space

**Solution:** Reduce checkpoint frequency and limit saved checkpoints:

Edit `training/train_final.py` or pass custom arguments:
```python
# In train_final.py, modify TrainingArguments:
save_steps=500,          # Save less frequently
save_total_limit=2,      # Keep only 2 checkpoints
```

### Issue: No GPU Available

**Solution 1:** Use CPU training (very slow):
```bash
CUDA_VISIBLE_DEVICES="" python labgpt_cli.py train \
  --train-file ./data/train.jsonl \
  --output ./model \
  --no-flash-attn  # Flash attention requires GPU
```

**Solution 2:** Use cloud GPU (Google Colab, AWS, etc.)

### Issue: Version Conflicts Between Pipelines

**Solution:** Use conda for better dependency management:
```bash
conda create -n labgpt python=3.10
conda activate labgpt
pip install -r requirements.txt
```

---

## Performance Tips

1. **Use Flash Attention:** Enable for 2-3x speedup during training (requires Ampere+ GPU)
2. **Enable Gradient Checkpointing:** Already enabled by default, saves memory
3. **Use Packing:** Already enabled in training, improves throughput for short sequences
4. **Parallelize Data Generation:** Generate data from multiple repos separately, then combine
5. **Cache Embeddings:** RAG automatically caches embeddings for faster incremental updates
6. **Use SSD Storage:** Significantly faster for document processing and model loading

---

## Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (for data generation in API mode)
CLAUDE_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Hugging Face (optional, for model upload)
HF_TOKEN=hf_...

# Custom model paths (optional)
LOCAL_MODEL_PATH=/path/to/llama-3.1-8b

# CUDA settings (optional)
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## Related Documentation

- [RAG Pipeline Guide](RAG/README.md)
- [Data Generation Guide](data_generation/README.md)
- [Training Documentation](training/train_final.py)
- [LabGPT Overview](README.md)

---

## Support

For issues or questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review logs in `{output}/logs/`
- Open an issue on GitHub

---

## License

Part of the LabGPT suite for laboratory research assistance.
