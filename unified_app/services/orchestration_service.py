"""
Orchestration service that wraps labgpt_cli.py for web application use.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to Python path to import labgpt_cli
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from labgpt_cli import PipelineOrchestrator
from unified_app.config import Config
from unified_app.services.file_manager import FileManager


class OrchestrationService:
    """
    Wraps the PipelineOrchestrator from labgpt_cli.py with project-specific paths
    and state management for the web application.
    """

    def __init__(self, project_dir: Path):
        """
        Initialize orchestration service for a specific project.

        Args:
            project_dir: Path to project directory
        """
        self.project_dir = project_dir
        self.orchestrator = PipelineOrchestrator(base_output_dir=str(project_dir))

        # Create log directory
        self.log_dir = project_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run_rag_pipeline(
        self,
        project_id: int,
        papers_paths: List[str],
        lab_docs_paths: List[str],
        rag_preset: str = "research"
    ) -> Dict[str, any]:
        """
        Run RAG indexing pipeline combining papers and lab documents.

        Args:
            project_id: Database ID of the project
            papers_paths: List of paths to research paper files/directories
            lab_docs_paths: List of paths to lab document files/directories
            rag_preset: RAG preset ('default' or 'research')

        Returns:
            Dict with result information including index path and log file
        """
        # Create temporary combined directory for papers
        papers_temp_dir = self.project_dir / 'research_papers'
        papers_temp_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary combined directory for lab docs
        lab_docs_temp_dir = self.project_dir / 'lab_documents'
        lab_docs_temp_dir.mkdir(parents=True, exist_ok=True)

        # Copy papers from user-provided paths to temp directory
        papers_copied = []
        if papers_paths:
            for papers_path in papers_paths:
                if papers_path and os.path.exists(papers_path):
                    try:
                        copied = FileManager.copy_directory_contents(
                            papers_path, papers_temp_dir, file_type='papers'
                        )
                        papers_copied.extend(copied)
                        print(f"Copied {len(copied)} papers from {papers_path} to temp directory.")
                    except Exception as e:
                        print(f"Error copying papers from {papers_path}: {e}")

            # Register copied papers in database
            if papers_copied:
                try:
                    FileManager.register_research_papers(project_id, papers_copied)
                    print(f"Registered {len(papers_copied)} research papers in database.")
                except Exception as e:
                    print(f"Error registering papers in database: {e}")

        # Copy lab documents from user-provided paths to temp directory
        lab_docs_copied = []
        if lab_docs_paths:
            for lab_docs_path in lab_docs_paths:
                if lab_docs_path and os.path.exists(lab_docs_path):
                    try:
                        copied = FileManager.copy_directory_contents(
                            lab_docs_path, lab_docs_temp_dir, file_type='lab_docs'
                        )
                        lab_docs_copied.extend(copied)
                        print(f"Copied {len(copied)} lab documents from {lab_docs_path} to temp directory.")
                    except Exception as e:
                        print(f"Error copying lab documents from {lab_docs_path}: {e}")

            # Register copied lab documents in database
            if lab_docs_copied:
                try:
                    FileManager.register_lab_documents(project_id, lab_docs_copied)
                    print(f"Registered {len(lab_docs_copied)} lab documents in database.")
                except Exception as e:
                    print(f"Error registering lab documents in database: {e}")

        # RAG index output directory
        index_dir = self.project_dir / 'rag-index'
        index_dir.mkdir(parents=True, exist_ok=True)

        # Run orchestrator
        result = self.orchestrator.run_rag_pipeline(
            papers_dir=str(papers_temp_dir) if papers_copied else None,
            lab_docs_dir=str(lab_docs_temp_dir) if lab_docs_copied else None,
            index_dir=str(index_dir),
            rag_preset=rag_preset
        )

        # Check if pipeline succeeded (labgpt_cli returns "status": "success")
        success = result.get('status') == 'success'

        return {
            'success': success,
            'index_dir': str(index_dir),
            'log_file': result.get('log_file'),
            'num_documents': result.get('documents_indexed', 0),
            'num_chunks': result.get('num_chunks', 0),
            'papers_copied': len(papers_copied),
            'lab_docs_copied': len(lab_docs_copied),
            'error': result.get('error')
        }

    def run_data_generation(
        self,
        code_repo_paths: List[str],
        papers_dir: Optional[str] = None,
        max_symbols: int = 30,
        languages: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Run data generation pipeline on code repositories and papers.

        Args:
            code_repo_paths: List of paths to code repositories
            papers_dir: Optional path to research papers directory
            max_symbols: Maximum symbols to extract per file
            languages: List of languages to process

        Returns:
            Dict with result information including output paths and log file
        """
        if languages is None:
            languages = Config.DEFAULT_LANGUAGES

        # Output directory for generated data
        output_dir = self.project_dir / 'data-generation'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Papers directory (use project papers if available)
        if papers_dir is None:
            papers_temp = self.project_dir / 'research_papers'
            papers_dir = str(papers_temp) if papers_temp.exists() else None

        # Run orchestrator
        result = self.orchestrator.run_data_generation(
            code_repos=code_repo_paths,
            papers_dir=papers_dir,
            output_dir=str(output_dir),
            max_symbols=max_symbols,
            languages=languages
        )

        # Check if pipeline succeeded (labgpt_cli returns "status": "success")
        success = result.get('status') == 'success'

        return {
            'success': success,
            'output_dir': str(output_dir),
            'log_file': result.get('log_file'),
            'train_file': result.get('train_file'),
            'val_file': result.get('val_file'),
            'num_examples': result.get('train_count', 0) + result.get('val_count', 0),
            'error': result.get('error')
        }

    def run_training(
        self,
        train_file: str,
        val_file: str,
        base_model: str,
        output_model_name: str,
        **training_kwargs
    ) -> Dict[str, any]:
        """
        Run model training pipeline.

        Args:
            train_file: Path to training data (JSONL)
            val_file: Path to validation data (JSONL)
            base_model: Base model name/path
            output_model_name: Name for output model
            **training_kwargs: Additional training parameters

        Returns:
            Dict with result information including model path and log file
        """
        # Model output directory
        model_output = self.project_dir / 'training' / output_model_name
        model_output.mkdir(parents=True, exist_ok=True)

        # Run orchestrator
        result = self.orchestrator.run_training(
            train_file=train_file,
            val_file=val_file,
            model_output=str(model_output),
            model_name=base_model,
            **training_kwargs
        )

        # Check if pipeline succeeded (labgpt_cli returns "status": "success")
        success = result.get('status') == 'success'

        return {
            'success': success,
            'model_path': str(model_output),
            'log_file': result.get('log_file'),
            'final_loss': result.get('final_loss'),
            'total_steps': result.get('total_steps'),
            'error': result.get('error')
        }

    def get_generated_data_files(self) -> Dict[str, Optional[str]]:
        """
        Get paths to generated training data files.

        Returns:
            Dict with 'train_file' and 'val_file' paths (or None if not found)
        """
        data_gen_dir = self.project_dir / 'data-generation'

        # Look for combined instruct files (preferred)
        train_file = data_gen_dir / 'combined_instruct_train.jsonl'
        val_file = data_gen_dir / 'combined_instruct_val.jsonl'

        if not train_file.exists():
            # Fallback to code-only files
            train_file = data_gen_dir / 'code_instruct_train.jsonl'
            val_file = data_gen_dir / 'code_instruct_val.jsonl'

        return {
            'train_file': str(train_file) if train_file.exists() else None,
            'val_file': str(val_file) if val_file.exists() else None
        }

    def get_rag_index_dir(self) -> Optional[str]:
        """
        Get path to RAG index directory.

        Returns:
            Path to RAG index or None if not found
        """
        index_dir = self.project_dir / 'rag-index'
        return str(index_dir) if index_dir.exists() else None

    def get_trained_model_path(self) -> Optional[str]:
        """
        Get path to trained model directory.

        Returns:
            Path to trained model or None if not found
        """
        training_dir = self.project_dir / 'training'
        if not training_dir.exists():
            return None

        # Find first subdirectory (model output)
        model_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
        return str(model_dirs[0]) if model_dirs else None
