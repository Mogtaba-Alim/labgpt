# LabGPT Model Training App

A simple Flask web application for fine-tuning AI models using datasets generated from the LabGPT data generation pipeline.

## Features

- **User-friendly Interface**: Clean, modern UI similar to the data generation app
- **Real-time Training Logs**: Live monitoring of training progress with detailed logs
- **Model Configuration**: Automatically configured for Llama 3.1 8B with LoRA fine-tuning
- **Progress Tracking**: Visual progress indicators and stage-by-stage updates
- **Results Summary**: Comprehensive results page with model information and usage instructions

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- At least 16GB RAM
- Generated training and validation JSONL files from the data generation pipeline (`combined_instruct_train.jsonl` and `combined_instruct_val.jsonl`)
- HuggingFace account and access to Llama 3.1 model (requires acceptance of license terms)

## Installation

1. Navigate to the training directory:
```bash
cd training
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Start the application**:
```bash
python app.py
```

2. **Access the web interface**:
   Open your browser and go to `http://localhost:5002`

3. **Configure training**:
   - **Training Dataset**: Provide the path to your `combined_instruct_train.jsonl` file
   - **Validation Dataset**: Provide the path to your `combined_instruct_val.jsonl` file (optional)
   - **Model Name**: Choose a name for your fine-tuned model (e.g., "my-labgpt-model")

4. **Monitor training**:
   - View real-time logs and progress updates
   - Track training stages: Data Validation → Environment Setup → Model Fine-Tuning
   - Monitor training metrics and completion status

5. **Access results**:
   - View training completion summary
   - Get model file locations and usage instructions
   - Copy code examples for using your fine-tuned model

## Running Training via Command Line

You can also run the training script directly from the terminal without using the web interface:

### Basic Usage

```bash
python train_final.py \
  --train_file path/to/combined_instruct_train.jsonl \
  --val_file path/to/combined_instruct_val.jsonl \
  --output_dir ./my-labgpt-model
```

### Input Data Format

The training script expects JSONL (JSON Lines) format where each line is a JSON object with:
```json
{
  "messages": [
    {"role": "system", "content": "You are LABGPT..."},
    {"role": "user", "content": "User question here"},
    {"role": "assistant", "content": "Assistant response here"}
  ],
  "metadata": {
    "source_type": "qa_pair",
    "language": "python",
    "complexity": "moderate"
  }
}
```

The script uses the official Llama 3.1 chat template to format these messages correctly.

## Configuration Parameters

### Data Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_file` | `combined_instruct_train.jsonl` | Path to training data file (JSONL format with messages) |
| `--val_file` | `""` | Path to validation data file (optional, JSONL format) |
| `--max_seq_length` | `8192` | Maximum sequence length for the model (tokens) |
| `--add_special_tokens` | `True` | Whether to add special tokens to the data |

### Model Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name_or_path` | `meta-llama/Llama-3.1-8B` | Base model to fine-tune (HuggingFace model ID or local path) |
| `--use_lora` | `True` | Use LoRA for parameter-efficient fine-tuning |
| `--lora_rank` | `16` | Rank for LoRA adaptation (higher = more parameters) |
| `--lora_alpha` | `32` | Alpha parameter for LoRA scaling |
| `--lora_dropout` | `0.05` | Dropout probability for LoRA layers |
| `--use_4bit` | `True` | Use 4-bit quantization (QLoRA) to reduce memory |
| `--use_nested_quant` | `True` | Use nested quantization (double quantization) for 4-bit |
| `--bnb_4bit_compute_dtype` | `float16` | Compute dtype for 4-bit quantization (float16/bfloat16) |
| `--bnb_4bit_quant_type` | `nf4` | Quantization type (nf4 or fp4) |
| `--use_flash_attn` | `True` | Use Flash Attention 2 for faster training (requires compatible GPU) |

### Training Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output_dir` | `./llama-3.1-8b-finetuned` | Output directory for model checkpoints |
| `--per_device_train_batch_size` | `1` | Training batch size per GPU/device |
| `--per_device_eval_batch_size` | `1` | Evaluation batch size per GPU/device |
| `--gradient_accumulation_steps` | `16` | Number of steps to accumulate gradients (effective batch size = batch_size × this) |
| `--learning_rate` | `2e-5` | Initial learning rate (0.00002) |
| `--weight_decay` | `0.01` | Weight decay for regularization |
| `--num_train_epochs` | `3.0` | Number of training epochs |
| `--warmup_ratio` | `0.1` | Fraction of steps for learning rate warmup |
| `--lr_scheduler_type` | `cosine` | Learning rate scheduler (linear/cosine/polynomial) |
| `--gradient_checkpointing` | `True` | Use gradient checkpointing to save memory |
| `--fp16` | `True` | Use FP16 mixed precision (auto-disabled if bf16 is available) |
| `--bf16` | `False` | Use BF16 mixed precision (auto-enabled on Ampere+ GPUs) |
| `--optim` | `paged_adamw_8bit` | Optimizer (paged_adamw_8bit/adamw_torch/adafactor) |
| `--logging_steps` | `10` | Log training metrics every N steps |
| `--eval_steps` | `100` | Run evaluation every N steps (minimum 200) |
| `--save_steps` | `100` | Save checkpoint every N steps (minimum 200) |
| `--save_total_limit` | `3` | Maximum number of checkpoints to keep |
| `--report_to` | `tensorboard` | Where to report metrics (tensorboard/wandb/none) |
| `--seed` | `42` | Random seed for reproducibility |

## Example Commands

### 1. Basic Training (Default Settings)
```bash
python train_final.py \
  --train_file data/combined_instruct_train.jsonl \
  --val_file data/combined_instruct_val.jsonl \
  --output_dir ./my-labgpt-model
```

### 2. Training with Custom Model Name
```bash
python train_final.py \
  --train_file ../data_generation/test_multi_lang_output_v4/combined_instruct_train.jsonl \
  --val_file ../data_generation/test_multi_lang_output_v4/combined_instruct_val.jsonl \
  --output_dir ./labgpt-research-assistant \
  --model_name_or_path meta-llama/Llama-3.1-8B
```

### 3. Training with Higher Learning Rate and More Epochs
```bash
python train_final.py \
  --train_file data/combined_instruct_train.jsonl \
  --val_file data/combined_instruct_val.jsonl \
  --output_dir ./labgpt-high-lr \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --warmup_ratio 0.15
```

### 4. Training with Larger Batch Size (Multi-GPU)
```bash
python train_final.py \
  --train_file data/combined_instruct_train.jsonl \
  --val_file data/combined_instruct_val.jsonl \
  --output_dir ./labgpt-multi-gpu \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8
```

### 5. Training with Custom LoRA Configuration
```bash
python train_final.py \
  --train_file data/combined_instruct_train.jsonl \
  --val_file data/combined_instruct_val.jsonl \
  --output_dir ./labgpt-lora-r32 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.1
```

### 6. Training without Validation Set
```bash
python train_final.py \
  --train_file data/combined_instruct_train.jsonl \
  --output_dir ./labgpt-no-val
```

### 7. Resume Training from Checkpoint
```bash
python train_final.py \
  --train_file data/combined_instruct_train.jsonl \
  --val_file data/combined_instruct_val.jsonl \
  --output_dir ./labgpt-resume \
  --model_name_or_path ./labgpt-resume/checkpoint-1000
```

### 8. Training with Weights & Biases Logging
```bash
python train_final.py \
  --train_file data/combined_instruct_train.jsonl \
  --val_file data/combined_instruct_val.jsonl \
  --output_dir ./labgpt-wandb \
  --report_to wandb
```

### 9. Memory-Optimized Training (for smaller GPUs)
```bash
python train_final.py \
  --train_file data/combined_instruct_train.jsonl \
  --val_file data/combined_instruct_val.jsonl \
  --output_dir ./labgpt-memory-opt \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_seq_length 4096 \
  --gradient_checkpointing True
```

### 10. Full Training Run with All Custom Parameters
```bash
python train_final.py \
  --train_file ../data_generation/test_multi_lang_output_v4/combined_instruct_train.jsonl \
  --val_file ../data_generation/test_multi_lang_output_v4/combined_instruct_val.jsonl \
  --output_dir ./labgpt-custom-full \
  --model_name_or_path meta-llama/Llama-3.1-8B \
  --max_seq_length 8192 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --logging_steps 10 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 3 \
  --seed 42
```

## Monitoring Training Progress

### TensorBoard (Default)
```bash
# In a separate terminal
tensorboard --logdir ./your-model-name/runs
```
Then open http://localhost:6006 in your browser.

### View Training Logs
```bash
# Training logs are written to stdout
python train_final.py --train_file data.jsonl --output_dir ./model 2>&1 | tee training.log
```

## Training Configuration

The app uses the following default training parameters:

- **Base Model**: Llama 3.1 8B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Batch Size**: 1 per device with gradient accumulation (16 steps)
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Max Sequence Length**: 8192 tokens
- **Mixed Precision**: FP16
- **Optimization**: AdamW 8-bit

## Output Files

After successful training, you'll find:

- **Model Directory**: `./[your-model-name]/`
  - PyTorch model files
  - LoRA adapter weights
  - Tokenizer configuration
  - Training checkpoints
  - TensorBoard logs

## Using Your Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Load your fine-tuned LoRA adapters
model = PeftModel.from_pretrained(base_model, "./your-model-name")

# Use the model
input_text = "Your prompt here"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Troubleshooting

- **Out of Memory**: Reduce batch size or sequence length in the training script
- **CUDA Errors**: Ensure you have CUDA-compatible PyTorch installed
- **File Not Found**: Verify that your dataset paths are correct and files exist
- **Permission Errors**: Ensure the app has write permissions in the current directory

## API Endpoints

- `GET /` - Main training configuration page
- `POST /` - Start training with provided parameters  
- `GET /training` - Training progress monitoring page
- `GET /api/status` - JSON API for training status updates
- `GET /results` - Training results and model information page

## Notes

### Important Updates (v2.0)

- **JSONL Format Required**: The training script now requires JSONL (JSON Lines) format with `messages` field containing chat-formatted conversations
- **Official Chat Template**: Uses Llama 3.1's official chat template via `tokenizer.apply_chat_template()` for proper special token handling
- **Auto bf16 Detection**: Automatically enables bfloat16 on Ampere+ GPUs (RTX 30/40 series, A100, A6000) for better training stability
- **Packing Enabled**: Sample packing is enabled by default for better GPU utilization with shorter samples
- **Legacy Format Removed**: Old JSON format with separate prompt/completion fields is no longer supported

### General Notes

- Training time varies based on dataset size and hardware (typically several hours)
- Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`
- The web app runs on port 5002 by default (different from the data generation app on 5001)
- All training logs are preserved and displayed in real-time
- Model files are saved locally in the specified model directory
- Flash Attention 2 requires `flash-attn` package: `pip install flash-attn --no-build-isolation`
- For multi-GPU training, the script automatically uses all available GPUs with `device_map="auto"`

### Data Generation Integration

- Use the data generation pipeline's instruct format output: `combined_instruct_train.jsonl` and `combined_instruct_val.jsonl`
- Located in: `data_generation/test_multi_lang_output_v4/` by default
- The training script is optimized for data generated by the comprehensive data generation pipeline 