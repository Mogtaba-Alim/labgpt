import os
import json
import subprocess
import threading
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from datetime import datetime
import time

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key_here_for_data_generation"  # Change to a secure random key in production

# Status tracking
processing_status = {
    'is_processing': False,
    'current_stage': '',
    'progress': 0,
    'message': '',
    'start_time': None,
    'logs': []
}

def log_message(message):
    """Add a message to the processing logs"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    processing_status['logs'].append(log_entry)
    processing_status['message'] = message
    print(log_entry)

def run_processing_pipeline(repo_urls, papers_dir):
    """Run the data generation pipeline in a separate thread"""
    global processing_status
    
    try:
        processing_status['is_processing'] = True
        processing_status['current_stage'] = 'Initializing'
        processing_status['progress'] = 0
        processing_status['start_time'] = datetime.now()
        processing_status['logs'] = []
        
        log_message("Starting data generation pipeline...")
        
        # Stage 1: Generate code fine-tune data
        processing_status['current_stage'] = 'Generating Code Fine-tune Data'
        processing_status['progress'] = 10
        log_message(f"Processing {len(repo_urls)} user-provided GitHub repositories...")
        
        # Import and run generateCodeFineTune with user-provided repos
        import generateCodeFineTune
        
        # Use the function designed for web app with user-provided repos
        log_message("Running code fine-tuning data generation from user repositories...")
        generateCodeFineTune.create_dataset_from_user_repos(repo_urls)
        
        processing_status['progress'] = 50
        log_message("Code fine-tune data generation completed!")
        
        # Stage 2: Generate final data output
        processing_status['current_stage'] = 'Generating Final Data Output'
        processing_status['progress'] = 60
        log_message(f"Processing research papers from user-provided directory: {papers_dir}")
        
        # Import and run createFinalDataOutput with provided papers directory
        import createFinalDataOutput
        
        # Use the new function that handles everything with user-provided papers_dir
        log_message("Processing research papers and combining with code data...")
        train_size, val_size = createFinalDataOutput.process_papers_and_combine_with_code(papers_dir)
        
        processing_status['progress'] = 90
        log_message("Research papers processed and data combined!")
        
        combined_train_data = train_size  # Store sizes for logging
        combined_val_data = val_size
        
        processing_status['progress'] = 100
        processing_status['current_stage'] = 'Completed'
        log_message("Data generation pipeline completed successfully!")
        log_message(f"Final training dataset: combined_dataset_train.json ({combined_train_data} entries)")
        log_message(f"Final validation dataset: combined_dataset_val.json ({combined_val_data} entries)")
        
    except Exception as e:
        processing_status['current_stage'] = 'Error'
        processing_status['progress'] = 0
        log_message(f"Error in processing pipeline: {str(e)}")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")
    
    finally:
        processing_status['is_processing'] = False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        repo_urls = []
        papers_dir = request.form.get('papers_dir', '').strip()
        
        # Collect all repo URLs
        for key in request.form:
            if key.startswith('repo_url_') and request.form[key].strip():
                repo_urls.append(request.form[key].strip())
        
        # Validation
        if not repo_urls:
            flash('Please provide at least one GitHub repository URL.', 'error')
            return render_template('index.html')
        
        if not papers_dir:
            flash('Please provide the research papers directory path.', 'error')
            return render_template('index.html')
        
        if not os.path.exists(papers_dir):
            flash('The specified research papers directory does not exist.', 'error')
            return render_template('index.html')
        
        # Store in session
        session['repo_urls'] = repo_urls
        session['papers_dir'] = papers_dir
        
        # Start processing in background
        processing_thread = threading.Thread(
            target=run_processing_pipeline, 
            args=(repo_urls, papers_dir)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        flash('Data generation pipeline started successfully!', 'success')
        return redirect(url_for('processing'))
    
    return render_template('index.html')

@app.route('/processing')
def processing():
    return render_template('processing.html')

@app.route('/api/status')
def api_status():
    """API endpoint to get current processing status"""
    return jsonify(processing_status)

@app.route('/results')
def results():
    # Check if processing is complete
    if processing_status['current_stage'] != 'Completed':
        flash('Processing is not yet complete.', 'warning')
        return redirect(url_for('processing'))
    
    # Check for output files
    train_file = 'combined_dataset_train.json'
    val_file = 'combined_dataset_val.json'
    
    results_info = {
        'train_exists': os.path.exists(train_file),
        'val_exists': os.path.exists(val_file),
        'train_size': 0,
        'val_size': 0,
        'train_file': train_file,
        'val_file': val_file
    }
    
    if results_info['train_exists']:
        try:
            with open(train_file, 'r') as f:
                train_data = json.load(f)
                results_info['train_size'] = len(train_data)
        except:
            pass
    
    if results_info['val_exists']:
        try:
            with open(val_file, 'r') as f:
                val_data = json.load(f)
                results_info['val_size'] = len(val_data)
        except:
            pass
    
    return render_template('results.html', results=results_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 