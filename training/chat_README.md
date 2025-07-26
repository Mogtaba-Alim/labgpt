# LabGPT Chat Application

A simple, ChatGPT-like interface for chatting with Hugging Face models.

## Features

- **Model Selection**: Choose from popular models or enter custom Hugging Face model IDs
- **Real-time Chat**: ChatGPT-style interface with message history
- **Model Loading**: Automatic download and initialization of selected models
- **Responsive Design**: Clean, modern UI that works on desktop and mobile

## Quick Start

1. **Install Requirements:**
```bash
pip install flask torch transformers accelerate bitsandbytes
```

2. **Run the Application:**
```bash
python chat_app.py
```

3. **Open Browser:**
```
http://localhost:5002
```

## How to Use

1. **Select a Model**: Choose from the sidebar or enter a custom model ID
2. **Wait for Loading**: The model will be downloaded and initialized
3. **Start Chatting**: Type messages and get responses from the AI

## Popular Models Included

- **Microsoft DialoGPT Medium**: Optimized for conversations
- **GPT-2 Medium**: Classic text generation model  
- **DistilGPT-2**: Smaller, faster version
- **Custom Fine-tuned Model**: Lab-specific model

## Custom Models

You can use any Hugging Face model by entering its ID in the custom model field:
- `gpt2`
- `microsoft/DialoGPT-small`
- `facebook/blenderbot-400M-distill`
- Any other compatible model

## Features

- **Conversation History**: Maintains context across messages
- **Typing Indicators**: Shows when AI is generating response
- **Message Timestamps**: Track conversation timing
- **Clear Chat**: Reset conversation history
- **Model Status**: Shows current model loading status

## Technical Details

- Built with Flask and Bootstrap
- Uses Hugging Face Transformers library
- Supports GPU acceleration with quantization
- Real-time status updates via AJAX

## Troubleshooting

- **Model Loading Fails**: Try smaller models like `distilgpt2`
- **Out of Memory**: Use CPU-only mode or smaller models
- **Slow Response**: Larger models need more time to generate responses

Enjoy chatting with AI models! 