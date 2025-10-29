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
# Generate from a GitHub repository
python run_comprehensive_data_gen.py \
  --repo https://github.com/user/repo \
  --output output_dir \
  --max_symbols 30

# Generate from local code
python run_comprehensive_data_gen.py \
  --repo /path/to/local/repo \
  --output output_dir

# Generate from code + papers
python run_comprehensive_data_gen.py \
  --repo /path/to/repo \
  --papers /path/to/papers \
  --output output_dir
```

## Key Features

- **Multi-Language Support**: Python, R, C, C++
- **Grounded QA Generation**: Context-bound question-answer pairs with citations
- **Bug Injection**: 12 realistic bug types for debugging task generation
- **Negative Examples**: NOT_IN_CONTEXT responses for abstention training
- **Quality Control**: 6-dimensional LLM-based quality scoring
- **Deduplication**: Embedding-based similarity filtering (0.92 threshold)
- **Instruct Format**: Direct output for Llama fine-tuning

## Output Format

The pipeline generates instruct format JSONL files ready for fine-tuning:

```
output_dir/
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

- `--repo`: Path or GitHub URL to code repository
- `--papers`: Directory containing research papers (PDF)
- `--output`: Output directory for generated datasets
- `--max_symbols`: Maximum symbols to extract per file (default: 30)
- `--min_tokens`: Minimum tokens per symbol (default: 30)
- `--languages`: Languages to process (default: python r c cpp)
- `--train_ratio`: Train/validation split ratio (default: 0.8)
- `--no_debug`: Disable debugging task generation
- `--no_negatives`: Disable negative example generation
- `--no_critic`: Disable quality filtering
- `--no_dedup`: Disable deduplication
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Architecture

```
data_generation/
├── run_comprehensive_data_gen.py    # Main pipeline script
├── data_gen/
│   ├── symbols/                     # Multi-language code parsing
│   ├── tasks/                       # QA, debug, negative generation
│   ├── critique/                    # Quality control & deduplication
│   ├── assembly/                    # Instruct format conversion
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
