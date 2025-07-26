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
- Generated training and validation JSON files from the data generation pipeline

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
   - **Training Dataset**: Provide the path to your `combined_dataset_train.json` file
   - **Validation Dataset**: Provide the path to your `combined_dataset_val.json` file
   - **Model Name**: Choose a name for your fine-tuned model (e.g., "my-labgpt-model")

4. **Monitor training**:
   - View real-time logs and progress updates
   - Track training stages: Data Validation → Environment Setup → Model Fine-Tuning
   - Monitor training metrics and completion status

5. **Access results**:
   - View training completion summary
   - Get model file locations and usage instructions
   - Copy code examples for using your fine-tuned model

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

- Training time varies based on dataset size and hardware (typically several hours)
- The app runs on port 5002 by default (different from the data generation app on 5001)
- All training logs are preserved and displayed in real-time
- Model files are saved locally in the specified model directory 