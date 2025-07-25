#!/usr/bin/env python3
"""
rag_app.py

RAG Document Processing Web Application
Processes lab documents by running data ingestion and index building
"""

import os
import subprocess
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import json

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'rag-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for tracking processing status
processing_status = {
    'status': 'idle',  # idle, processing, completed, error
    'stage': '',       # data_ingestion, index_building
    'progress': 0,     # 0-100
    'message': '',
    'error': '',
    'logs': []
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class ProcessingManager:
    """Manages the document processing pipeline"""
    
    def __init__(self):
        self.current_directory = None
        self.process_thread = None
        self.output_files = ['chunks.npy', 'embeddings.npy', 'faiss_index.bin']
    
    def start_processing(self, directory_path: str, override_existing: bool = False):
        """Start the processing pipeline in a separate thread"""
        if processing_status['status'] == 'processing':
            return False, "Processing already in progress"
        
        if not os.path.exists(directory_path):
            return False, f"Directory does not exist: {directory_path}"
        
        if not os.path.isdir(directory_path):
            return False, f"Path is not a directory: {directory_path}"
        
        # Check if directory contains any files
        files = list(Path(directory_path).glob("*.*"))
        if not files:
            return False, f"Directory is empty: {directory_path}"
        
        # Check for existing output files
        existing_files = self.get_existing_files()
        if existing_files and not override_existing:
            file_list = ", ".join(existing_files)
            return False, f"Output files already exist: {file_list}. Use override option to replace them."
        
        self.current_directory = directory_path
        self.override_existing = override_existing
        self.process_thread = threading.Thread(target=self._run_processing)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        return True, "Processing started successfully"
    
    def get_existing_files(self) -> List[str]:
        """Get list of existing output files"""
        existing = []
        for file_path in self.output_files:
            if os.path.exists(file_path):
                existing.append(file_path)
        return existing
    
    def cleanup_existing_files(self):
        """Remove existing output files"""
        removed_files = []
        for file_path in self.output_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                    self._log_message(f"Removed existing file: {file_path}")
                except Exception as e:
                    self._log_message(f"Warning: Could not remove {file_path}: {e}")
        
        if removed_files:
            self._log_message(f"Cleaned up {len(removed_files)} existing files")
        return removed_files
    
    def _run_processing(self):
        """Run the complete processing pipeline"""
        global processing_status
        
        try:
            # Reset status
            processing_status.update({
                'status': 'processing',
                'stage': 'cleanup' if hasattr(self, 'override_existing') and self.override_existing else 'data_ingestion',
                'progress': 0,
                'message': 'Starting processing...',
                'error': '',
                'logs': []
            })
            
            # Stage 0: Cleanup existing files if override is enabled
            if hasattr(self, 'override_existing') and self.override_existing:
                self._log_message("Override option enabled - cleaning up existing files...")
                self.cleanup_existing_files()
                processing_status.update({
                    'stage': 'data_ingestion',
                    'progress': 5,
                    'message': 'Cleanup completed. Starting data ingestion...'
                })
            
            # Stage 1: Data Ingestion
            self._log_message("Starting data ingestion process...")
            result = self._run_data_ingestion()
            
            if not result:
                processing_status.update({
                    'status': 'error',
                    'error': 'Data ingestion failed'
                })
                return
            
            # Adjust progress based on whether cleanup was performed
            base_progress = 5 if hasattr(self, 'override_existing') and self.override_existing else 0
            processing_status.update({
                'stage': 'index_building',
                'progress': base_progress + 45,  # 50% total, but adjusted for cleanup
                'message': 'Data ingestion completed. Starting index building...'
            })
            
            # Stage 2: Index Building
            self._log_message("Starting index building process...")
            result = self._run_index_building()
            
            if not result:
                processing_status.update({
                    'status': 'error',
                    'error': 'Index building failed'
                })
                return
            
            # Success
            processing_status.update({
                'status': 'completed',
                'stage': 'completed',
                'progress': 100,
                'message': 'Processing completed successfully!'
            })
            self._log_message("RAG processing pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Processing error: {e}")
            processing_status.update({
                'status': 'error',
                'error': str(e)
            })
    
    def _run_data_ingestion(self) -> bool:
        """Run data_ingestion.py with the specified directory"""
        try:
            # Modify the data_ingestion.py to accept directory parameter
            env = os.environ.copy()
            env['DATA_FOLDER'] = self.current_directory
            
            process = subprocess.Popen(
                ['python', 'data_ingestion.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self._log_message(output.strip())
            
            stderr = process.stderr.read()
            if stderr:
                self._log_message(f"STDERR: {stderr}")
            
            return_code = process.poll()
            return return_code == 0
            
        except Exception as e:
            self._log_message(f"Error running data ingestion: {e}")
            return False
    
    def _run_index_building(self) -> bool:
        """Run index_builder.py"""
        try:
            process = subprocess.Popen(
                ['python', 'index_builder.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self._log_message(output.strip())
            
            stderr = process.stderr.read()
            if stderr:
                self._log_message(f"STDERR: {stderr}")
            
            return_code = process.poll()
            return return_code == 0
            
        except Exception as e:
            self._log_message(f"Error running index building: {e}")
            return False
    
    def _log_message(self, message: str):
        """Add message to processing logs"""
        global processing_status
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        processing_status['logs'].append(log_entry)
        logging.info(message)
        
        # Keep only last 100 log entries
        if len(processing_status['logs']) > 100:
            processing_status['logs'] = processing_status['logs'][-100:]

# Initialize processing manager
processor = ProcessingManager()

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page for RAG document processing"""
    if request.method == 'POST':
        directory_path = request.form.get('directory_path', '').strip()
        override_existing = request.form.get('override_existing') == 'on'
        
        if not directory_path:
            flash('Please provide a directory path.', 'error')
            return render_template('rag_index.html')
        
        # Expand user path if needed
        directory_path = os.path.expanduser(directory_path)
        directory_path = os.path.abspath(directory_path)
        
        # Check for existing files to show user what will be overridden
        existing_files = processor.get_existing_files()
        if existing_files and override_existing:
            file_list = ", ".join(existing_files)
            flash(f'Override enabled: Will replace existing files: {file_list}', 'warning')
        
        success, message = processor.start_processing(directory_path, override_existing)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('processing'))
        else:
            flash(message, 'error')
            return render_template('rag_index.html', 
                                 directory_path=directory_path,
                                 existing_files=existing_files)
    
    # Check for existing files on GET request
    existing_files = processor.get_existing_files()
    return render_template('rag_index.html', existing_files=existing_files)

@app.route('/processing')
def processing():
    """Processing status page"""
    return render_template('rag_processing.html')

@app.route('/api/status')
def get_status():
    """API endpoint to get current processing status"""
    return jsonify(processing_status)

@app.route('/api/existing-files')
def get_existing_files():
    """API endpoint to check for existing output files"""
    existing_files = processor.get_existing_files()
    return jsonify({
        'existing_files': existing_files,
        'has_existing': len(existing_files) > 0
    })

@app.route('/reset')
def reset():
    """Reset the processing status"""
    global processing_status
    if processing_status['status'] != 'processing':
        processing_status.update({
            'status': 'idle',
            'stage': '',
            'progress': 0,
            'message': '',
            'error': '',
            'logs': []
        })
        flash('Status reset successfully.', 'success')
    else:
        flash('Cannot reset while processing is in progress.', 'error')
    
    return redirect(url_for('index'))

@app.route('/results')
def results():
    """Show processing results"""
    if processing_status['status'] != 'completed':
        flash('Processing not completed yet.', 'error')
        return redirect(url_for('index'))
    
    # Check if output files exist
    output_files = {
        'chunks.npy': os.path.exists('chunks.npy'),
        'embeddings.npy': os.path.exists('embeddings.npy'),
        'faiss_index.bin': os.path.exists('faiss_index.bin')
    }
    
    # Get file sizes
    file_info = {}
    for filename, exists in output_files.items():
        if exists:
            file_path = filename
            size = os.path.getsize(file_path)
            file_info[filename] = {
                'exists': True,
                'size': f"{size / (1024*1024):.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
            }
        else:
            file_info[filename] = {'exists': False, 'size': 'N/A'}
    
    return render_template('rag_results.html', file_info=file_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 