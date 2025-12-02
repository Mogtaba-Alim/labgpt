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

### 3. Model Override

```bash
# Override models from config (uses parameters from config if available)
python sequential_batch_test.py prompts_config.json --models "Qwen3-8B" "DeepSeek-R1-Distill-Qwen-32B"
```

Note: Category and priority filtering have been removed in the simplified format.

## Configuration Format

### Settings Section

- **`api_key`**: Your Prompter API key
- **`index_dir`**: Path to your RAG index directory
- **`models`**: Object with model names as keys and their specific parameters as values
- **`output_dir`**: Directory to save results

Each model configuration includes:
- **`temperature`**: Sampling temperature (0.0-1.0)
- **`top_p`**: Nucleus sampling parameter (0.0-1.0)
- **`max_tokens`**: Maximum tokens to generate
- **`expand`**: Enable query expansion (true/false)
- **`cited_spans`**: Enable cited span extraction (true/false)
- **`preset`**: RAG preset to use ("default" or "research")

### Prompts Section

Each prompt has:
- **`id`**: Unique identifier
- **`prompt`**: The actual question/prompt text

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
  "prompt": "Your question here?"
}
```

## Tips

- Use `--dry-run` first to verify your configuration
- Check the batch summary for overall success rates
- Individual JSON files contain full model responses
- The script runs sequentially to avoid overwhelming the API

## Troubleshooting

- **"Config file not found"**: Check the path to `prompts_config.json`
- **API errors**: Verify your API key in the config file
- **Index errors**: Ensure the RAG index path is correct
- **Model errors**: Check that model names match those in your config file
