"""
Log parser utility for extracting progress information from pipeline logs.
"""

import re
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class LogParser:
    """
    Parses log files from labgpt_cli orchestrator to extract progress information.
    """

    # Regex patterns for different pipeline stages
    RAG_PATTERNS = {
        'loading_documents': r'Loading documents from',
        'creating_chunks': r'Creating chunks',
        'generating_embeddings': r'Generating embeddings',
        'building_index': r'Building (FAISS|BM25) index',
        'completed': r'RAG pipeline completed',
        'error': r'ERROR|Error|Exception'
    }

    DATA_GEN_PATTERNS = {
        'extracting_symbols': r'Extracting symbols from',
        'generating_qa': r'Generating Q&A pairs',
        'generating_debug': r'Generating debug tasks',
        'quality_check': r'Running quality critic',
        'deduplication': r'Deduplicating examples',
        'formatting': r'Formatting (instruct|comprehensive) data',
        'completed': r'Data generation completed',
        'error': r'ERROR|Error|Exception'
    }

    TRAINING_PATTERNS = {
        'loading_model': r'Loading (base )?model',
        'loading_data': r'Loading (training|validation) data',
        'training_started': r'Training (started|epoch)',
        'step': r'Step (\d+)/(\d+)',
        'epoch': r'Epoch (\d+)/(\d+)',
        'loss': r'loss[:\s]+([\d.]+)',
        'completed': r'Training completed',
        'error': r'ERROR|Error|Exception'
    }

    @staticmethod
    def parse_log_file(log_path: str, job_type: str) -> Dict[str, any]:
        """
        Parse a log file and extract progress information.

        Args:
            log_path: Path to log file
            job_type: Type of job ('rag', 'data_generation', 'training')

        Returns:
            Dict with progress info: {
                'progress_percentage': int,
                'current_step': str,
                'stages_completed': list,
                'errors': list
            }
        """
        if not Path(log_path).exists():
            return {
                'progress_percentage': 0,
                'current_step': 'Waiting to start...',
                'stages_completed': [],
                'errors': []
            }

        # Select patterns based on job type
        if job_type == 'rag':
            patterns = LogParser.RAG_PATTERNS
            total_stages = 5
        elif job_type == 'data_generation':
            patterns = LogParser.DATA_GEN_PATTERNS
            total_stages = 6
        elif job_type == 'training':
            patterns = LogParser.TRAINING_PATTERNS
            total_stages = 5
        else:
            return {
                'progress_percentage': 0,
                'current_step': 'Unknown job type',
                'stages_completed': [],
                'errors': []
            }

        # Read log file
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
        except Exception as e:
            return {
                'progress_percentage': 0,
                'current_step': f'Error reading log: {str(e)}',
                'stages_completed': [],
                'errors': [str(e)]
            }

        # Extract stages and errors
        stages_completed = []
        errors = []
        current_step = 'Processing...'

        for stage_name, pattern in patterns.items():
            if re.search(pattern, log_content, re.IGNORECASE):
                if stage_name == 'completed':
                    stages_completed.append('completed')
                    current_step = 'Completed'
                elif stage_name == 'error':
                    # Extract actual error messages
                    error_matches = re.findall(r'(ERROR|Error|Exception)[:\s]+(.*?)(?:\n|$)', log_content)
                    errors.extend([msg[1].strip() for msg in error_matches[:5]])  # Limit to 5 errors
                else:
                    stages_completed.append(stage_name)
                    current_step = stage_name.replace('_', ' ').title()

        # Calculate progress percentage
        if 'completed' in stages_completed:
            progress_percentage = 100
        elif errors:
            progress_percentage = min((len(stages_completed) / total_stages) * 100, 95)
        else:
            progress_percentage = min((len(stages_completed) / total_stages) * 100, 95)

        # Extract specific metrics for training
        if job_type == 'training':
            step_match = re.search(r'Step (\d+)/(\d+)', log_content)
            if step_match:
                current, total = int(step_match.group(1)), int(step_match.group(2))
                progress_percentage = min((current / total) * 100, 95)
                current_step = f'Training: Step {current}/{total}'

        return {
            'progress_percentage': int(progress_percentage),
            'current_step': current_step,
            'stages_completed': stages_completed,
            'errors': errors
        }

    @staticmethod
    def tail_log_file(log_path: str, num_lines: int = 50) -> List[str]:
        """
        Get the last N lines of a log file.

        Args:
            log_path: Path to log file
            num_lines: Number of lines to retrieve

        Returns:
            List of log lines
        """
        if not Path(log_path).exists():
            return []

        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                return lines[-num_lines:] if lines else []
        except Exception:
            return []

    @staticmethod
    def get_training_metrics(log_path: str) -> Dict[str, any]:
        """
        Extract training metrics from log file.

        Args:
            log_path: Path to training log file

        Returns:
            Dict with training metrics (loss, steps, epochs, etc.)
        """
        if not Path(log_path).exists():
            return {}

        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
        except Exception:
            return {}

        metrics = {}

        # Extract loss values
        loss_matches = re.findall(r'loss[:\s]+([\d.]+)', log_content, re.IGNORECASE)
        if loss_matches:
            losses = [float(l) for l in loss_matches]
            metrics['current_loss'] = losses[-1]
            metrics['min_loss'] = min(losses)
            metrics['loss_history'] = losses[-20:]  # Last 20 loss values

        # Extract step information
        step_match = re.search(r'Step (\d+)/(\d+)', log_content)
        if step_match:
            metrics['current_step'] = int(step_match.group(1))
            metrics['total_steps'] = int(step_match.group(2))

        # Extract epoch information
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', log_content)
        if epoch_match:
            metrics['current_epoch'] = int(epoch_match.group(1))
            metrics['total_epochs'] = int(epoch_match.group(2))

        return metrics
