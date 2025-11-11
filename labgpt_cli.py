#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LabGPT Unified CLI - Orchestrates RAG, Data Generation, and Training Pipelines

This tool provides a unified interface for running all LabGPT pipelines:
1. RAG: Document indexing for retrieval
2. Data Generation: Synthetic instruction data from code and papers
3. Training: Fine-tuning Llama models

Usage:
    labgpt run-all --code-repos /path/to/repo --papers /path/to/papers --lab-docs /path/to/lab-docs
    labgpt rag --papers /path/to/papers --lab-docs /path/to/lab-docs --index ./rag-index
    labgpt data-gen --code-repos /path/to/repo --papers /path/to/papers --output ./data
    labgpt train --train-file ./data/combined_instruct_train.jsonl --output ./model
"""

import os
import sys
import json
import shutil
import argparse
import logging
import subprocess
import tempfile
import signal
import atexit
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the execution of RAG, Data Generation, and Training pipelines."""

    def __init__(self, base_output_dir: str = "./labgpt-output"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Setup directories
        self.logs_dir = self.base_output_dir / "logs"
        self.temp_dir = self.base_output_dir / "temp"
        self.logs_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

        # Track child processes for cleanup
        self.child_processes = []

        # Register cleanup handler
        atexit.register(self.cleanup_processes)

        # Setup log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"labgpt_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)

        logger.info(f"LabGPT Orchestrator initialized. Logs: {log_file}")

    def cleanup_processes(self):
        """Kill all tracked child processes and their process groups."""
        if not self.child_processes:
            return

        logger.info(f"Cleaning up {len(self.child_processes)} child process(es)...")

        for proc in self.child_processes:
            try:
                if proc.poll() is None:  # Process still running
                    # With start_new_session=True, the process is its own session leader
                    # Session ID equals PID, so we can kill the entire session
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        logger.info(f"Sent SIGTERM to process session {proc.pid}")

                        # Wait briefly for graceful shutdown
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            # Force kill if still running
                            os.killpg(proc.pid, signal.SIGKILL)
                            logger.warning(f"Sent SIGKILL to process session {proc.pid}")
                    except ProcessLookupError:
                        # Process already exited
                        pass
            except (OSError, AttributeError) as e:
                logger.debug(f"Cleanup error for PID {proc.pid}: {e}")

        self.child_processes.clear()

    def run_subprocess_with_logging(self, cmd: List[str], log_prefix: str, cwd: str = None) -> tuple:
        """
        Run subprocess with process group handling and streamed logging.

        Args:
            cmd: Command and arguments to run
            log_prefix: Prefix for log file name
            cwd: Working directory (default: current directory)

        Returns:
            Tuple of (CompletedProcess with stdout/stderr as strings, Path to log file)

        Raises:
            subprocess.CalledProcessError: If subprocess fails
        """
        # Create log file for subprocess output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = self.logs_dir / f"{log_prefix}_{timestamp}.log"

        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Subprocess logs: {log_file_path}")

        # Initialize process to None (in case exception occurs before it's created)
        process = None

        try:
            with open(log_file_path, 'w') as log_file:
                # On Unix: start_new_session creates process group
                # On Windows: creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                popen_kwargs = {
                    'stdout': log_file,
                    'stderr': subprocess.STDOUT,  # Combine stderr with stdout
                    'text': True,
                    'cwd': cwd or os.getcwd(),
                }

                # Platform-specific process group creation
                if os.name == 'posix':  # Unix/Linux/Mac
                    # start_new_session=True creates a new process session (equivalent to setsid)
                    popen_kwargs['start_new_session'] = True
                else:  # Windows
                    popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

                # Start subprocess
                process = subprocess.Popen(cmd, **popen_kwargs)
                self.child_processes.append(process)

                # Wait for completion
                return_code = process.wait()

                # Remove from tracked processes (successfully completed)
                if process in self.child_processes:
                    self.child_processes.remove(process)

            # Read logs for return
            with open(log_file_path, 'r') as log_file:
                output = log_file.read()

            # Check return code
            if return_code != 0:
                logger.error(f"Subprocess failed with return code {return_code}")
                logger.error(f"Output:\n{output}")
                raise subprocess.CalledProcessError(return_code, cmd, output=output)

            logger.info(f"Subprocess completed successfully")
            completed_process = subprocess.CompletedProcess(cmd, return_code, stdout=output, stderr='')
            return completed_process, log_file_path

        except Exception as e:
            # Remove from tracked processes on error
            if process in self.child_processes:
                self.child_processes.remove(process)
            raise

    def run_rag_pipeline(
        self,
        papers_dir: Optional[str] = None,
        lab_docs_dir: Optional[str] = None,
        index_dir: str = None,
        rag_preset: str = "research",
        cleanup_temp: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the RAG pipeline to index documents.

        Args:
            papers_dir: Path to research papers directory
            lab_docs_dir: Path to lab documents directory
            index_dir: Output directory for RAG index
            rag_preset: RAG preset (default, research)
            cleanup_temp: Whether to cleanup temp directory after ingestion

        Returns:
            Dict with status and index_path
        """
        logger.info("=" * 80)
        logger.info("STAGE 1/3: RAG Pipeline - Document Indexing")
        logger.info("=" * 80)

        if not papers_dir and not lab_docs_dir:
            logger.warning("No documents provided for RAG. Skipping RAG pipeline.")
            return {"status": "skipped", "reason": "no_documents"}

        # Set default index directory
        if index_dir is None:
            index_dir = str(self.base_output_dir / "rag-index")

        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for combined documents
        temp_docs_dir = self.temp_dir / "rag_documents"
        temp_docs_dir.mkdir(exist_ok=True)

        try:
            # Copy papers to temp directory
            if papers_dir and os.path.exists(papers_dir):
                logger.info(f"Copying papers from {papers_dir} to temp directory...")
                papers_temp = temp_docs_dir / "papers"
                shutil.copytree(papers_dir, papers_temp, dirs_exist_ok=True)
                logger.info(f"Copied papers to {papers_temp}")

            # Copy lab documents to temp directory
            if lab_docs_dir and os.path.exists(lab_docs_dir):
                logger.info(f"Copying lab documents from {lab_docs_dir} to temp directory...")
                lab_docs_temp = temp_docs_dir / "lab_docs"
                shutil.copytree(lab_docs_dir, lab_docs_temp, dirs_exist_ok=True)
                logger.info(f"Copied lab documents to {lab_docs_temp}")

            # Run RAG ingestion
            logger.info(f"Starting RAG ingestion into index: {index_path}")
            logger.info(f"Documents source: {temp_docs_dir}")

            cmd = [
                sys.executable, "-m", "RAG.cli",
                "ingest",
                "--docs", str(temp_docs_dir),
                "--index", str(index_path),
                "--preset", rag_preset,
                "--device", "auto",  # Auto-select device: GPU if available, otherwise CPU
            ]

            # Run with process group handling and streamed logging
            result, log_file_path = self.run_subprocess_with_logging(cmd, "rag_ingestion")

            logger.info("RAG ingestion output:")
            logger.info(result.stdout)

            # Verify index was created
            if not (index_path / "documents_metadata.json").exists():
                raise RuntimeError("RAG index creation failed - metadata file not found")

            logger.info(f"✓ RAG pipeline completed successfully")
            logger.info(f"✓ Index created at: {index_path}")

            return {
                "status": "success",
                "index_path": str(index_path),
                "documents_indexed": self._count_indexed_documents(index_path),
                "log_file": str(log_file_path)
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"RAG pipeline failed with error:\n{e.stderr}")
            raise RuntimeError(f"RAG pipeline failed: {e.stderr}")

        finally:
            # Cleanup temp directory if requested
            if cleanup_temp and temp_docs_dir.exists():
                logger.info(f"Cleaning up temporary documents directory: {temp_docs_dir}")
                shutil.rmtree(temp_docs_dir)

    def run_data_generation(
        self,
        code_repos: List[str],
        papers_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_symbols: Optional[int] = None,
        languages: Optional[List[str]] = None,
        train_ratio: float = 0.8,
        no_debug: bool = False,
        no_negatives: bool = False,
        no_critic: bool = False,
        no_dedup: bool = False,
        clear_checkpoints: bool = False,
        privacy: bool = False,
        local_model_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the data generation pipeline.

        Args:
            code_repos: List of code repository paths or GitHub URLs
            papers_dir: Path to research papers directory
            output_dir: Output directory for generated data
            max_symbols: Maximum symbols to extract per file
            languages: Languages to process (default: python, r, c, cpp)
            train_ratio: Train/validation split ratio
            no_debug: Disable debug task generation
            no_negatives: Disable negative example generation
            no_critic: Disable quality filtering
            no_dedup: Disable deduplication
            clear_checkpoints: Delete existing checkpoints and start fresh
            privacy: Use local model instead of API
            local_model_path: Path to local model for privacy mode

        Returns:
            Dict with status and output paths
        """
        logger.info("=" * 80)
        logger.info("STAGE 2/3: Data Generation Pipeline - Synthetic Instruction Data")
        logger.info("=" * 80)

        if not code_repos:
            logger.warning("No code repositories provided. Data generation will only use papers if available.")
            if not papers_dir:
                raise ValueError("At least one of code_repos or papers_dir must be provided")

        # Set default output directory
        if output_dir is None:
            output_dir = str(self.base_output_dir / "data_generation")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Validate local repositories exist (URLs will be handled by data generation script)
        for repo in code_repos:
            if not (repo.startswith("http://") or repo.startswith("https://")):
                if not os.path.exists(repo):
                    raise ValueError(f"Local repository path does not exist: {repo}")

        try:
            logger.info(f"\nProcessing {len(code_repos)} repositories in single pipeline run")

            # Build command - pass ALL repos in a single call
            cmd = [
                sys.executable,
                "data_generation/run_comprehensive_data_gen.py",
                "--repo", *code_repos,  # Pass all repos at once
                "--output", str(output_path),
                "--train_ratio", str(train_ratio),
            ]

            # Only add max_symbols and languages if explicitly provided
            if max_symbols is not None:
                cmd.extend(["--max_symbols", str(max_symbols)])
            if languages is not None:
                cmd.extend(["--languages", *languages])

            # Add papers directory if provided
            if papers_dir and os.path.exists(papers_dir):
                cmd.extend(["--papers", papers_dir])

            # Add flags
            if no_debug:
                cmd.append("--no_debug")
            if no_negatives:
                cmd.append("--no_negatives")
            if no_critic:
                cmd.append("--no_critic")
            if no_dedup:
                cmd.append("--no_dedup")
            if clear_checkpoints:
                cmd.append("--clear_checkpoints")
            if privacy:
                cmd.append("--privacy")
                if local_model_path:
                    cmd.extend(["--local_model_path", local_model_path])

            # Run with process group handling and streamed logging
            result, log_file_path = self.run_subprocess_with_logging(cmd, "data_generation")

            logger.info("Data generation output:")
            logger.info(result.stdout)

            # Output files are already combined by the data generation pipeline
            combined_train = output_path / "combined_instruct_train.jsonl"
            combined_val = output_path / "combined_instruct_val.jsonl"

            # Count examples
            train_count = self._count_jsonl_lines(combined_train)
            val_count = self._count_jsonl_lines(combined_val)

            logger.info(f"✓ Data generation completed successfully")
            logger.info(f"✓ Training examples: {train_count}")
            logger.info(f"✓ Validation examples: {val_count}")
            logger.info(f"✓ Output directory: {output_path}")

            return {
                "status": "success",
                "output_dir": str(output_path),
                "train_file": str(combined_train),
                "val_file": str(combined_val),
                "train_count": train_count,
                "val_count": val_count,
                "log_file": str(log_file_path)
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Data generation failed with error:\n{e.stderr}")
            raise RuntimeError(f"Data generation failed: {e.stderr}")

    def run_training(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        model_output: Optional[str] = None,
        model_name: str = "meta-llama/Llama-3.1-8B",
        max_seq_length: int = 8192,
        num_epochs: float = 3.0,
        batch_size: int = 1,
        gradient_accumulation: int = 16,
        learning_rate: float = 2e-5,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        use_4bit: bool = True,
        use_flash_attn: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the training pipeline.

        Args:
            train_file: Path to training JSONL file
            val_file: Path to validation JSONL file (optional)
            model_output: Output directory for trained model
            model_name: Base model to fine-tune
            max_seq_length: Maximum sequence length
            num_epochs: Number of training epochs
            batch_size: Batch size per device
            gradient_accumulation: Gradient accumulation steps
            learning_rate: Learning rate
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            use_4bit: Use 4-bit quantization
            use_flash_attn: Use flash attention

        Returns:
            Dict with status and model path
        """
        logger.info("=" * 80)
        logger.info("STAGE 3/3: Training Pipeline - Model Fine-tuning")
        logger.info("=" * 80)

        if not os.path.exists(train_file):
            raise ValueError(f"Training file does not exist: {train_file}")

        # Set default model output directory
        if model_output is None:
            model_output = str(self.base_output_dir / "model")

        model_path = Path(model_output)
        model_path.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            "training/train_final.py",
            "--train_file", train_file,
            "--output_dir", str(model_path),
            "--model_name_or_path", model_name,
            "--max_seq_length", str(max_seq_length),
            "--num_train_epochs", str(num_epochs),
            "--per_device_train_batch_size", str(batch_size),
            "--gradient_accumulation_steps", str(gradient_accumulation),
            "--learning_rate", str(learning_rate),
            "--lora_rank", str(lora_rank),
            "--lora_alpha", str(lora_alpha),
        ]

        if val_file and os.path.exists(val_file):
            cmd.extend(["--val_file", val_file])

        if not use_4bit:
            cmd.append("--use_4bit=False")

        if not use_flash_attn:
            cmd.append("--use_flash_attn=False")

        logger.info("Note: Training may take several hours depending on dataset size and hardware.")

        try:
            # Run with process group handling and streamed logging
            result, log_file_path = self.run_subprocess_with_logging(cmd, "training")

            logger.info("Training output:")
            logger.info(result.stdout)

            logger.info(f"✓ Training completed successfully")
            logger.info(f"✓ Model saved to: {model_path}")

            return {
                "status": "success",
                "model_path": str(model_path),
                "log_file": str(log_file_path)
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with error:\n{e.stderr}")
            raise RuntimeError(f"Training failed: {e.stderr}")

    def run_all_pipelines(
        self,
        code_repos: List[str],
        papers_dir: Optional[str] = None,
        lab_docs_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run all three pipelines sequentially: RAG → Data Generation → Training.

        Args:
            code_repos: List of code repository paths or GitHub URLs
            papers_dir: Path to research papers directory
            lab_docs_dir: Path to lab documents directory
            **kwargs: Additional arguments for individual pipelines

        Returns:
            Dict with status and outputs from all pipelines
        """
        logger.info("=" * 80)
        logger.info("LabGPT Unified Pipeline - Running All Stages")
        logger.info("=" * 80)

        start_time = time.time()
        results = {}

        try:
            # Stage 1: RAG
            logger.info("\n" + "=" * 80)
            logger.info("Starting Stage 1/3: RAG Pipeline")
            logger.info("=" * 80 + "\n")

            rag_result = self.run_rag_pipeline(
                papers_dir=papers_dir,
                lab_docs_dir=lab_docs_dir,
                **kwargs
            )
            results["rag"] = rag_result
            logger.info(f"\n✓ Stage 1/3 completed in {time.time() - start_time:.1f}s")

            # Stage 2: Data Generation
            stage2_start = time.time()
            logger.info("\n" + "=" * 80)
            logger.info("Starting Stage 2/3: Data Generation Pipeline")
            logger.info("=" * 80 + "\n")

            datagen_result = self.run_data_generation(
                code_repos=code_repos,
                papers_dir=papers_dir,
                **kwargs
            )
            results["data_generation"] = datagen_result
            logger.info(f"\n✓ Stage 2/3 completed in {time.time() - stage2_start:.1f}s")

            # Stage 3: Training
            stage3_start = time.time()
            logger.info("\n" + "=" * 80)
            logger.info("Starting Stage 3/3: Training Pipeline")
            logger.info("=" * 80 + "\n")

            training_result = self.run_training(
                train_file=datagen_result["train_file"],
                val_file=datagen_result.get("val_file"),
                **kwargs
            )
            results["training"] = training_result
            logger.info(f"\n✓ Stage 3/3 completed in {time.time() - stage3_start:.1f}s")

            # Final summary
            total_time = time.time() - start_time
            logger.info("\n" + "=" * 80)
            logger.info("ALL PIPELINES COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"\nTotal execution time: {total_time / 60:.1f} minutes")
            logger.info(f"\nResults summary:")
            logger.info(f"  RAG Index: {results['rag'].get('index_path', 'N/A')}")
            logger.info(f"  Training Data: {datagen_result['train_count']} train, {datagen_result['val_count']} val")
            logger.info(f"  Trained Model: {results['training']['model_path']}")
            logger.info(f"\nAll outputs saved to: {self.base_output_dir}")

            return {
                "status": "success",
                "results": results,
                "total_time": total_time
            }

        except Exception as e:
            logger.error(f"\n{'=' * 80}")
            logger.error(f"PIPELINE FAILED")
            logger.error(f"{'=' * 80}")
            logger.error(f"Error: {str(e)}")
            logger.error(f"Check logs at: {self.logs_dir}")
            raise

    # Helper methods

    def _count_indexed_documents(self, index_path: Path) -> int:
        """Count the number of documents in a RAG index."""
        try:
            metadata_file = index_path / "documents_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return len(metadata.get("documents", []))
        except Exception as e:
            logger.warning(f"Could not count indexed documents: {e}")
        return 0

    def _combine_jsonl_files(
        self,
        source_dirs: List[Path],
        filename: str,
        output_file: Path
    ):
        """Combine JSONL files from multiple directories."""
        with open(output_file, 'w') as outf:
            for source_dir in source_dirs:
                source_file = source_dir / filename
                if source_file.exists():
                    with open(source_file, 'r') as inf:
                        for line in inf:
                            outf.write(line)

        logger.info(f"Combined {len(source_dirs)} files into {output_file}")

    def _count_jsonl_lines(self, file_path: Path) -> int:
        """Count the number of lines in a JSONL file."""
        if not file_path.exists():
            return 0
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)


def setup_argparse() -> argparse.ArgumentParser:
    """Setup argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="LabGPT Unified CLI - Orchestrates RAG, Data Generation, and Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all pipelines
  %(prog)s run-all --code-repos /path/to/repo --papers /path/to/papers --lab-docs /path/to/lab-docs

  # Run with multiple code repositories
  %(prog)s run-all --code-repos /path/repo1 https://github.com/user/repo2 --papers /path/papers

  # Run individual pipelines
  %(prog)s rag --papers /path/to/papers --lab-docs /path/to/lab-docs --index ./rag-index
  %(prog)s data-gen --code-repos /path/to/repo --papers /path/to/papers --output ./data
  %(prog)s train --train-file ./data/combined_instruct_train.jsonl --output ./model

For more information, see README_UNIFIED.md
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run-all command
    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Run all three pipelines sequentially (RAG → Data Gen → Training)"
    )
    add_common_args(run_all_parser)
    add_rag_args(run_all_parser)
    add_datagen_args(run_all_parser)
    add_training_args(run_all_parser)

    # rag command
    rag_parser = subparsers.add_parser("rag", help="Run RAG pipeline only")
    add_rag_specific_args(rag_parser)
    add_rag_args(rag_parser)

    # data-gen command
    datagen_parser = subparsers.add_parser("data-gen", help="Run data generation pipeline only")
    add_datagen_specific_args(datagen_parser)
    add_datagen_args(datagen_parser)

    # train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline only")
    add_training_specific_args(train_parser)
    add_training_args(train_parser)

    return parser


def add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments for all pipelines."""
    parser.add_argument(
        "--code-repos",
        nargs="+",
        help="Space-separated list of code repository paths or GitHub URLs"
    )
    parser.add_argument(
        "--papers",
        type=str,
        help="Path to research papers directory"
    )
    parser.add_argument(
        "--lab-docs",
        type=str,
        help="Path to lab documents directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./labgpt-output",
        help="Base output directory for all pipelines (default: ./labgpt-output)"
    )


def add_rag_specific_args(parser: argparse.ArgumentParser):
    """Add RAG-specific required arguments."""
    parser.add_argument(
        "--papers",
        type=str,
        help="Path to research papers directory"
    )
    parser.add_argument(
        "--lab-docs",
        type=str,
        help="Path to lab documents directory"
    )
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Output directory for RAG index"
    )


def add_rag_args(parser: argparse.ArgumentParser):
    """Add RAG pipeline arguments."""
    rag_group = parser.add_argument_group("RAG options")
    rag_group.add_argument(
        "--rag-index",
        type=str,
        help="Output directory for RAG index (default: {output}/rag-index)"
    )
    rag_group.add_argument(
        "--rag-preset",
        type=str,
        default="research",
        choices=["default", "research"],
        help="RAG preset configuration (default: research)"
    )
    rag_group.add_argument(
        "--rag-no-cleanup",
        action="store_true",
        help="Keep temporary documents directory after RAG ingestion"
    )


def add_datagen_specific_args(parser: argparse.ArgumentParser):
    """Add data generation specific required arguments."""
    parser.add_argument(
        "--code-repos",
        nargs="+",
        required=True,
        help="Space-separated list of code repository paths or GitHub URLs"
    )
    parser.add_argument(
        "--papers",
        type=str,
        help="Path to research papers directory (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for generated data"
    )


def add_datagen_args(parser: argparse.ArgumentParser):
    """Add data generation pipeline arguments (optional parameters)."""
    datagen_group = parser.add_argument_group("Data Generation options")
    datagen_group.add_argument(
        "--datagen-output",
        type=str,
        help="Output directory for data generation (default: {output}/data_generation)"
    )
    datagen_group.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum symbols to extract per file (default: 30 if not specified)"
    )
    datagen_group.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to process (default: python r c cpp if not specified)"
    )
    datagen_group.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/validation split ratio (default: 0.8)"
    )
    datagen_group.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug task generation"
    )
    datagen_group.add_argument(
        "--no-negatives",
        action="store_true",
        help="Disable negative example generation"
    )
    datagen_group.add_argument(
        "--no-critic",
        action="store_true",
        help="Disable quality filtering"
    )
    datagen_group.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication"
    )
    datagen_group.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Delete existing checkpoints and start fresh (use when changing parameters)"
    )
    datagen_group.add_argument(
        "--privacy",
        action="store_true",
        help="Use local model instead of API (privacy mode)"
    )
    datagen_group.add_argument(
        "--local-model-path",
        type=str,
        help="Path to local model for privacy mode"
    )


def add_training_specific_args(parser: argparse.ArgumentParser):
    """Add training specific required arguments."""
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for trained model"
    )


def add_training_args(parser: argparse.ArgumentParser):
    """Add training pipeline arguments."""
    training_group = parser.add_argument_group("Training options")
    training_group.add_argument(
        "--model-output",
        type=str,
        help="Output directory for trained model (default: {output}/model)"
    )
    training_group.add_argument(
        "--val-file",
        type=str,
        help="Path to validation JSONL file (optional)"
    )
    training_group.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Base model to fine-tune (default: meta-llama/Llama-3.1-8B)"
    )
    training_group.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length (default: 8192)"
    )
    training_group.add_argument(
        "--num-epochs",
        type=float,
        default=3.0,
        help="Number of training epochs (default: 3.0)"
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device (default: 1)"
    )
    training_group.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16)"
    )
    training_group.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    training_group.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    training_group.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    training_group.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    training_group.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable flash attention"
    )


def main():
    """Main entry point for the CLI."""
    parser = setup_argparse()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        # Convert args to dict
        args_dict = vars(args)
        command = args_dict.pop("command")

        # Initialize orchestrator
        if command == "run-all":
            base_output = args_dict.get("output", "./labgpt-output")
        elif command == "rag":
            base_output = str(Path(args_dict.get("index")).parent)
        elif command == "data-gen":
            base_output = str(Path(args_dict.get("output")).parent)
        elif command == "train":
            base_output = str(Path(args_dict.get("output")).parent)
        else:
            base_output = "./labgpt-output"

        orchestrator = PipelineOrchestrator(base_output_dir=base_output)

        # Process command-specific argument names
        if command == "run-all":
            # Map argument names to function parameters
            kwargs = {
                "code_repos": args_dict.get("code_repos", []),
                "papers_dir": args_dict.get("papers"),
                "lab_docs_dir": args_dict.get("lab_docs"),
                "index_dir": args_dict.get("rag_index"),
                "rag_preset": args_dict.get("rag_preset", "research"),
                "cleanup_temp": not args_dict.get("rag_no_cleanup", False),
                "output_dir": args_dict.get("datagen_output"),
                "max_symbols": args_dict.get("max_symbols", 30),
                "languages": args_dict.get("languages"),
                "train_ratio": args_dict.get("train_ratio", 0.8),
                "no_debug": args_dict.get("no_debug", False),
                "no_negatives": args_dict.get("no_negatives", False),
                "no_critic": args_dict.get("no_critic", False),
                "no_dedup": args_dict.get("no_dedup", False),
                "clear_checkpoints": args_dict.get("clear_checkpoints", False),
                "privacy": args_dict.get("privacy", False),
                "local_model_path": args_dict.get("local_model_path"),
                "model_output": args_dict.get("model_output"),
                "val_file": args_dict.get("val_file"),
                "model_name": args_dict.get("model_name", "meta-llama/Llama-3.1-8B"),
                "max_seq_length": args_dict.get("max_seq_length", 8192),
                "num_epochs": args_dict.get("num_epochs", 3.0),
                "batch_size": args_dict.get("batch_size", 1),
                "gradient_accumulation": args_dict.get("gradient_accumulation", 16),
                "learning_rate": args_dict.get("learning_rate", 2e-5),
                "lora_rank": args_dict.get("lora_rank", 16),
                "lora_alpha": args_dict.get("lora_alpha", 32),
                "use_4bit": not args_dict.get("no_4bit", False),
                "use_flash_attn": not args_dict.get("no_flash_attn", False),
            }

            result = orchestrator.run_all_pipelines(**kwargs)

        elif command == "rag":
            result = orchestrator.run_rag_pipeline(
                papers_dir=args_dict.get("papers"),
                lab_docs_dir=args_dict.get("lab_docs"),
                index_dir=args_dict.get("index"),
                rag_preset=args_dict.get("rag_preset", "research"),
            )

        elif command == "data-gen":
            result = orchestrator.run_data_generation(
                code_repos=args_dict.get("code_repos"),
                papers_dir=args_dict.get("papers"),
                output_dir=args_dict.get("output"),
                max_symbols=args_dict.get("max_symbols", 30),
                languages=args_dict.get("languages"),
                train_ratio=args_dict.get("train_ratio", 0.8),
                no_debug=args_dict.get("no_debug", False),
                no_negatives=args_dict.get("no_negatives", False),
                no_critic=args_dict.get("no_critic", False),
                no_dedup=args_dict.get("no_dedup", False),
                clear_checkpoints=args_dict.get("clear_checkpoints", False),
                privacy=args_dict.get("privacy", False),
                local_model_path=args_dict.get("local_model_path"),
            )

        elif command == "train":
            result = orchestrator.run_training(
                train_file=args_dict.get("train_file"),
                val_file=args_dict.get("val_file"),
                model_output=args_dict.get("output"),
                model_name=args_dict.get("model_name", "meta-llama/Llama-3.1-8B"),
                max_seq_length=args_dict.get("max_seq_length", 8192),
                num_epochs=args_dict.get("num_epochs", 3.0),
                batch_size=args_dict.get("batch_size", 1),
                gradient_accumulation=args_dict.get("gradient_accumulation", 16),
                learning_rate=args_dict.get("learning_rate", 2e-5),
                lora_rank=args_dict.get("lora_rank", 16),
                lora_alpha=args_dict.get("lora_alpha", 32),
                use_4bit=not args_dict.get("no_4bit", False),
                use_flash_attn=not args_dict.get("no_flash_attn", False),
            )

        # Print summary
        print("\n" + "=" * 80)
        print(f"✓ {command.upper()} COMPLETED SUCCESSFULLY")
        print("=" * 80)

        sys.exit(0)

    except KeyboardInterrupt:
        logger.error("\n\nInterrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
