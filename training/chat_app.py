#!/usr/bin/env python3
"""
chat_app.py

Simple Chat Application for Hugging Face Models
Allows users to select models and chat with them in a ChatGPT-like interface
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, request, render_template, jsonify, session
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import threading

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'chat-app-secret-key')

# Configuration
logging.basicConfig(level=logging.INFO)

# Global variables for model management
current_model = None
current_tokenizer = None
current_pipeline = None
model_loading_status = {
    'status': 'idle',  # idle, loading, loaded, error
    'model_name': '',
    'progress': '',
    'error': ''
}

# Popular model presets
POPULAR_MODELS = [
    {
        'name': 'Microsoft DialoGPT Medium',
        'model_id': 'microsoft/DialoGPT-medium',
        'description': 'Conversational AI model optimized for chat'
    },
    {
        'name': 'GPT-2 Medium',
        'model_id': 'gpt2-medium',
        'description': 'Classic GPT-2 model for text generation'
    },
    {
        'name': 'DistilGPT-2',
        'model_id': 'distilgpt2',
        'description': 'Smaller, faster version of GPT-2'
    },
    {
        'name': 'Custom Fine-tuned Model',
        'model_id': 'MogtabaAlim/llama3.1-8B-BHK-LABGPT-Fine-tunedByMogtaba',
        'description': 'Lab-specific fine-tuned model'
    }
]

class ModelManager:
    """Manages model loading and chat generation"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = None
    
    def load_model(self, model_name: str):
        """Load a model in a separate thread"""
        global model_loading_status
        
        def _load():
            try:
                model_loading_status.update({
                    'status': 'loading',
                    'model_name': model_name,
                    'progress': 'Downloading tokenizer...',
                    'error': ''
                })
                
                # Load tokenizer
                logging.info(f"Loading tokenizer for {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model_loading_status['progress'] = 'Downloading model...'
                
                # Load model with appropriate settings
                logging.info(f"Loading model {model_name}")
                
                try:
                    # Try with quantization first
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        load_in_8bit=True
                    )
                    logging.info("Model loaded with 8-bit quantization")
                except Exception as e:
                    logging.warning(f"8-bit loading failed: {e}")
                    try:
                        # Fallback to 4-bit
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            load_in_4bit=True
                        )
                        logging.info("Model loaded with 4-bit quantization")
                    except Exception as e2:
                        logging.warning(f"4-bit loading failed: {e2}")
                        # Final fallback to CPU
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,
                            device_map="cpu"
                        )
                        logging.info("Model loaded on CPU")
                
                model_loading_status['progress'] = 'Setting up pipeline...'
                
                # Create pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    do_sample=True,
                    temperature=0.7,
                    max_length=512
                )
                
                # Store globally
                self.model = model
                self.tokenizer = tokenizer
                self.pipeline = pipe
                self.model_name = model_name
                
                model_loading_status.update({
                    'status': 'loaded',
                    'progress': 'Model ready for chat!',
                })
                
                logging.info(f"Model {model_name} loaded successfully")
                
            except Exception as e:
                logging.error(f"Error loading model {model_name}: {e}")
                model_loading_status.update({
                    'status': 'error',
                    'error': str(e)
                })
        
        # Start loading in background
        thread = threading.Thread(target=_load)
        thread.daemon = True
        thread.start()
    
    def generate_response(self, message: str, conversation_history: list = None) -> str:
        """Generate a response to user message"""
        if not self.pipeline:
            return "No model loaded. Please select and load a model first."
        
        try:
            # Build context from conversation history
            context = ""
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages for context
                    context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"
            
            # Add current message
            prompt = f"{context}User: {message}\nAssistant:"
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the response
            generated_text = response[0]['generated_text']
            assistant_response = generated_text.split("Assistant:")[-1].strip()
            
            # Clean up response
            if "User:" in assistant_response:
                assistant_response = assistant_response.split("User:")[0].strip()
            
            return assistant_response
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

# Initialize model manager
model_manager = ModelManager()

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat_interface.html', popular_models=POPULAR_MODELS)

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load a selected model"""
    data = request.get_json()
    model_name = data.get('model_name', '').strip()
    
    if not model_name:
        return jsonify({'success': False, 'error': 'Model name is required'})
    
    # Start loading model
    model_manager.load_model(model_name)
    
    return jsonify({'success': True, 'message': 'Model loading started'})

@app.route('/api/model-status')
def model_status():
    """Get current model loading status"""
    return jsonify(model_loading_status)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.get_json()
    message = data.get('message', '').strip()
    conversation_history = data.get('history', [])
    
    if not message:
        return jsonify({'success': False, 'error': 'Message is required'})
    
    if model_loading_status['status'] != 'loaded':
        return jsonify({
            'success': False, 
            'error': 'No model loaded. Please load a model first.'
        })
    
    # Generate response
    response = model_manager.generate_response(message, conversation_history)
    
    return jsonify({
        'success': True,
        'response': response,
        'model_name': model_manager.model_name
    })

@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    return jsonify({'success': True, 'message': 'Chat cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 