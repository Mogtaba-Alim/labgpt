import os
import json
import subprocess
import threading
import sys
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from datetime import datetime
import time

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key_here_for_training"  # Change to a secure random key in production

# Status tracking
training_status = {
    'is_training': False,
    'current_stage': '',
    'progress': 0,
    'message': '',
    'start_time': None,
    'logs': [],
    'model_name': '',
    'train_file': '',
    'val_file': ''
}

def log_message(message):
    """Add a message to the training logs"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    training_status['logs'].append(log_entry)
    training_status['message'] = message
    print(log_entry)

def run_training_pipeline(train_file, val_file, model_name):
    """Run the model training pipeline in a separate thread"""
    global training_status
    
    try:
        training_status['is_training'] = True
        training_status['current_stage'] = 'Initializing'
        training_status['progress'] = 0
        training_status['start_time'] = datetime.now()
        training_status['logs'] = []
        training_status['model_name'] = model_name
        training_status['train_file'] = train_file
        training_status['val_file'] = val_file
        
        log_message("Starting model training pipeline...")
        log_message(f"Training file: {train_file}")
        log_message(f"Validation file: {val_file}")
        log_message(f"Model name: {model_name}")
        
        # Stage 1: Validate files
        training_status['current_stage'] = 'Validating Input Files'
        training_status['progress'] = 10
        log_message("Validating training and validation files...")
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not os.path.exists(val_file):
            raise FileNotFoundError(f"Validation file not found: {val_file}")
        
        # Check file contents
        with open(train_file, 'r') as f:
            train_data = json.load(f)
            log_message(f"Training dataset contains {len(train_data)} examples")
        
        with open(val_file, 'r') as f:
            val_data = json.load(f)
            log_message(f"Validation dataset contains {len(val_data)} examples")
        
        training_status['progress'] = 20
        log_message("Input files validated successfully!")
        
        # Stage 2: Prepare training command
        training_status['current_stage'] = 'Preparing Training Environment'
        training_status['progress'] = 30
        log_message("Preparing training command and environment...")
        
        # Build the training command
        train_cmd = [
            sys.executable, "train_final.py",
            "--train_file", train_file,
            "--val_file", val_file,
            "--output_dir", f"./{model_name}",
            "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "1",
            "--gradient_accumulation_steps", "16",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "3",
            "--logging_steps", "10",
            "--eval_steps", "100",
            "--save_steps", "100",
            "--warmup_ratio", "0.1",
            "--lr_scheduler_type", "cosine",
            "--optim", "paged_adamw_8bit",
            "--bf16", "False",
            "--fp16", "True",
            "--gradient_checkpointing", "True",
            "--max_seq_length", "8192"
        ]
        
        log_message(f"Training command: {' '.join(train_cmd)}")
        training_status['progress'] = 40
        
        # Stage 3: Start training
        training_status['current_stage'] = 'Model Training in Progress'
        training_status['progress'] = 50
        log_message("Starting model training...")
        log_message("This may take several hours depending on your hardware...")
        
        # Execute training
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream the output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                log_message(output.strip())
                
                # Update progress based on training logs
                if "epoch" in output.lower():
                    training_status['progress'] = min(90, training_status['progress'] + 5)
                elif "saving model" in output.lower():
                    training_status['progress'] = 95
        
        # Check if training completed successfully
        return_code = process.poll()
        if return_code == 0:
            training_status['progress'] = 100
            training_status['current_stage'] = 'Completed'
            log_message("Model training completed successfully!")
            log_message(f"Model saved to: ./{model_name}")
        else:
            raise RuntimeError(f"Training failed with return code {return_code}")
        
    except Exception as e:
        training_status['current_stage'] = 'Error'
        training_status['progress'] = 0
        log_message(f"Error in training pipeline: {str(e)}")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")
    
    finally:
        training_status['is_training'] = False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        train_file = request.form.get('train_file', '').strip()
        val_file = request.form.get('val_file', '').strip()
        model_name = request.form.get('model_name', '').strip()
        
        # Validation
        if not train_file:
            flash('Please provide the path to the training JSON file.', 'error')
            return render_template('index.html')
        
        if not val_file:
            flash('Please provide the path to the validation JSON file.', 'error')
            return render_template('index.html')
        
        if not model_name:
            flash('Please provide a name for the output model.', 'error')
            return render_template('index.html')
        
        if not os.path.exists(train_file):
            flash('The specified training file does not exist.', 'error')
            return render_template('index.html')
        
        if not os.path.exists(val_file):
            flash('The specified validation file does not exist.', 'error')
            return render_template('index.html')
        
        # Store in session
        session['train_file'] = train_file
        session['val_file'] = val_file
        session['model_name'] = model_name
        
        # Start training in background
        training_thread = threading.Thread(
            target=run_training_pipeline, 
            args=(train_file, val_file, model_name)
        )
        training_thread.daemon = True
        training_thread.start()
        
        flash('Model training started successfully!', 'success')
        return redirect(url_for('training'))
    
    return render_template('index.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/api/status')
def api_status():
    """API endpoint to get current training status"""
    return jsonify(training_status)

@app.route('/results')
def results():
    # Check if training is complete
    if training_status['current_stage'] != 'Completed':
        flash('Training is not yet complete.', 'warning')
        return redirect(url_for('training'))
    
    # Check for output model
    model_dir = f"./{training_status['model_name']}"
    
    results_info = {
        'model_exists': os.path.exists(model_dir),
        'model_name': training_status['model_name'],
        'model_dir': model_dir,
        'train_file': training_status['train_file'],
        'val_file': training_status['val_file']
    }
    
    return render_template('results.html', results=results_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 