# LabGPT Data Generation Pipeline

A production-ready synthetic data generation system for creating high-quality instruction fine-tuning datasets from code repositories and research papers.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Setup

Create a `.env` file with your API keys:

```bash
CLAUDE_API_KEY=your_claude_api_key
OPENAI_KEY=your_openai_api_key
```

### Basic Usage

```bash
# Generate from a single repository
python run_comprehensive_data_gen.py \
  --repo https://github.com/user/repo \
  --output output_dir

# Generate from multiple repositories (all processed in one run)
python run_comprehensive_data_gen.py \
  --repo /path/to/repo1 https://github.com/user/repo2 /path/to/repo3 \
  --output output_dir

# Generate from code + papers
python run_comprehensive_data_gen.py \
  --repo /path/to/repo \
  --papers /path/to/papers \
  --output output_dir

# Multiple repos + papers (papers processed once, quality control across all repos)
python run_comprehensive_data_gen.py \
  --repo /path/repo1 https://github.com/user/repo2 \
  --papers /path/to/papers \
  --output output_dir

# Resume interrupted pipeline (checkpoints auto-loaded by default)
python run_comprehensive_data_gen.py \
  --repo /path/repo1 /path/repo2 /path/repo3 \
  --output output_dir

# Start fresh by clearing checkpoints
python run_comprehensive_data_gen.py \
  --repo /path/to/repo \
  --output output_dir \
  --clear_checkpoints
```

## Key Features

- **Multi-Language Support**: Python, R, C, C++ with case-insensitive extensions (`.py`, `.r/.R`, `.c/.C`, `.cpp/.CPP`, etc.)
- **Documentation Processing**: Automatically processes Markdown (`.md/.MD`) and R documentation (`.Rd/.rd`) files found in repositories
- **Grounded QA Generation**: Context-bound question-answer pairs with citations
- **Bug Injection**: 12 realistic bug types for debugging task generation
- **Negative Examples**: NOT_IN_CONTEXT responses for abstention training
- **Quality Control**: 6-dimensional LLM-based quality scoring
- **Deduplication**: Embedding-based similarity filtering (0.92 threshold)
- **Checkpoint System**: Automatic fault-tolerant processing with per-repository saves
- **Cross-Platform**: Works consistently on Windows, Linux, and macOS
- **Instruct Format**: Direct output for Llama fine-tuning
- **Privacy Mode GPU Coordination**: Automatic GPU lock management prevents conflicts with training tasks

## Output Format

The pipeline generates instruct format JSONL files ready for fine-tuning:

```
output_dir/
├── checkpoints/                      # Intermediate saves (auto-resume)
│   ├── repo_01_RepoName.json        # Per-repository checkpoint
│   ├── repo_02_RepoName.json        # Per-repository checkpoint
│   └── papers_dataset.json          # Papers checkpoint
├── code_instruct_train.jsonl        # Training data from code
├── code_instruct_val.jsonl          # Validation data from code
├── papers_instruct_train.jsonl      # Training data from papers
├── papers_instruct_val.jsonl        # Validation data from papers
├── combined_instruct_train.jsonl    # Combined training data
└── combined_instruct_val.jsonl      # Combined validation data
```

Each entry contains:
```json
{
  "messages": [
    {"role": "system", "content": "You are LABGPT..."},
    {"role": "user", "content": "Question about code..."},
    {"role": "assistant", "content": "Answer with citations..."}
  ]
}
```

## Testing

Test the pipeline on a sample repository:

```bash
# Quick test with small repository
python run_comprehensive_data_gen.py \
  --repo https://github.com/bhklab/readii-fmcib \
  --output test_output \
  --max_symbols 10 \
  --no_critic \
  --no_dedup

# Full test with all features
python run_comprehensive_data_gen.py \
  --repo https://github.com/bhklab/readii-fmcib \
  --papers ../sample_papers \
  --output test_output \
  --max_symbols 30
```

## Configuration

Key arguments:

- `--repo`: Path or GitHub URL to code repository (processes code + documentation files)
- `--papers`: Directory containing research papers (PDF, TXT, MD, Rd files)
- `--output`: Output directory for generated datasets
- `--max_symbols`: Maximum symbols to extract per file (default: 30)
- `--min_tokens`: Minimum tokens per symbol (default: 30)
- `--languages`: Languages to process (default: python r c cpp)
- `--train_ratio`: Train/validation split ratio (default: 0.8)
- `--no_debug`: Disable debugging task generation
- `--no_negatives`: Disable negative example generation
- `--no_critic`: Disable quality filtering
- `--no_dedup`: Disable deduplication
- `--clear_checkpoints`: Delete existing checkpoints and start fresh (use when changing parameters)
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--privacy`: Use local Llama 3.1 8B model instead of API (complete data privacy)
- `--local_model_path`: Path to local model for privacy mode
- `--device`: Device for local model (cpu/cuda/auto, default: auto)

## Privacy Mode GPU Coordination

When using `--privacy` mode, the pipeline automatically coordinates GPU usage with other tasks:

**Auto-Device Selection:**
- Checks if GPU is available
- Attempts to acquire Redis GPU lock (non-blocking)
- If GPU available and lock acquired: uses CUDA
- If GPU busy or locked by training: falls back to CPU automatically

**Benefits:**
- No GPU conflicts with training tasks
- Automatic fallback ensures pipeline continues
- CPU inference slower (~10-30x) but doesn't interfere with GPU tasks

**Example:**
```bash
# Privacy mode with auto device detection
python run_comprehensive_data_gen.py \
  --repo /path/to/repo \
  --output output_dir \
  --privacy

# Force CPU to ensure no GPU conflicts
python run_comprehensive_data_gen.py \
  --repo /path/to/repo \
  --output output_dir \
  --privacy \
  --device cpu
```

## Checkpoint System

The pipeline automatically saves progress after each repository and paper processing:

- **Auto-Resume**: By default, the pipeline resumes from existing checkpoints if interrupted
- **Per-Repository Saves**: Each repository is saved immediately after processing
- **Fault Tolerance**: Prevents data loss during long-running multi-repository processing
- **Fresh Start**: Use `--clear_checkpoints` when changing parameters like `--max_symbols`

Checkpoints are stored in `output_dir/checkpoints/` and loaded automatically on restart.

## Architecture

```
data_generation/
├── run_comprehensive_data_gen.py    # Main pipeline script
├── data_gen/
│   ├── symbols/                     # Multi-language code parsing (Python, R, C, C++)
│   ├── tasks/                       # QA, debug, negative generation
│   ├── critique/                    # Quality control & deduplication
│   ├── assembly/                    # Instruct format conversion
│   ├── paper_ingestion/             # Document loading (PDF, MD, Rd files)
│   └── config/                      # Task taxonomy configuration
└── requirements.txt                 # Dependencies
```

## Requirements

- Python 3.8+
- Claude API key (for code tasks)
- OpenAI API key (for paper processing)
- See [requirements.txt](requirements.txt) for dependencies

## Documentation

For detailed documentation, see [data_generation_comprehensive_guide.md](data_generation_comprehensive_guide.md)

## License

Part of the LabGPT suite for laboratory research assistance.
