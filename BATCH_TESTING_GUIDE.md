# Sequential Batch Testing Guide

This guide explains how to use the sequential batch testing system for vLLM inference.

## Files Created

1. **`prompts_config.json`** - Configuration file with prompts and settings
2. **`sequential_batch_test.py`** - Main batch testing script
3. **`BATCH_TESTING_GUIDE.md`** - This guide

## Quick Start

### 1. Configure Your Settings

Edit `prompts_config.json` to set your API key and preferences:

```json
{
  "settings": {
    "api_key": "your-actual-api-key-here",
    "index_dir": "indices/rag_demo_storage",
    "models": ["qwen3-8b", "gemma3-4b"],
    "output_dir": "results"
  }
}
```

### 2. Basic Usage

```bash
# Test all prompts with all models
python sequential_batch_test.py prompts_config.json

# Dry run to see what would be executed
python sequential_batch_test.py prompts_config.json --dry-run
```

### 3. Filtering Options

```bash
# Filter by category
python sequential_batch_test.py prompts_config.json --category molecular_biology

# Filter by priority
python sequential_batch_test.py prompts_config.json --priority high

# Combine filters
python sequential_batch_test.py prompts_config.json --category molecular_biology --priority high

# Override models from config
python sequential_batch_test.py prompts_config.json --models qwen3-8b deepseek-r1-32b
```

## Configuration Format

### Settings Section

- **`api_key`**: Your Prompter API key
- **`index_dir`**: Path to your RAG index directory
- **`models`**: List of model names to test (uses keys from `vllm_model_config.py`)
- **`output_dir`**: Directory to save results
- **`default_params`**: Default parameters for all tests

### Prompts Section

Each prompt has:
- **`id`**: Unique identifier
- **`category`**: Category for filtering (e.g., "molecular_biology", "machine_learning")
- **`priority`**: Priority level ("high", "medium", "low")
- **`prompt`**: The actual question/prompt text
- **`description`**: Human-readable description

## Output Structure

Results are saved in the specified output directory:

```
results/
├── crispr_001_qwen3-8b_20251202_143022.json
├── crispr_001_gemma3-4b_20251202_143045.json
├── ml_001_qwen3-8b_20251202_143108.json
├── ml_001_gemma3-4b_20251202_143131.json
└── batch_summary_20251202_143200.json
```

The batch summary contains:
- Total and successful test counts
- Applied filters
- Individual test results with timestamps

## Example Workflow

1. **Setup**: Edit `prompts_config.json` with your API key
2. **Test**: Run with `--dry-run` to verify configuration
3. **Execute**: Run without `--dry-run` to perform actual tests
4. **Analyze**: Review individual JSON files and batch summary

## Adding New Prompts

Add new prompts to the `prompts` array in `prompts_config.json`:

```json
{
  "id": "new_prompt_001",
  "category": "your_category",
  "priority": "medium",
  "prompt": "Your question here?",
  "description": "Brief description"
}
```

## Tips

- Use `--dry-run` first to verify your configuration
- Filter by category/priority to test specific subsets
- Check the batch summary for overall success rates
- Individual JSON files contain full model responses
- The script runs sequentially to avoid overwhelming the API

## Troubleshooting

- **"Config file not found"**: Check the path to `prompts_config.json`
- **API errors**: Verify your API key in the config file
- **Index errors**: Ensure the RAG index path is correct
- **Model errors**: Check that model names match those in `vllm_model_config.py`
