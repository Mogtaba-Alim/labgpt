#!/usr/bin/env python3
"""
run_comprehensive_data_gen.py

Comprehensive multi-language data generation pipeline that addresses all limitations:

1. Multi-language support (Python, R, C, C++)
2. Proper instruct format output for fine-tuning
3. Integration with existing pipeline components
4. Enhanced quality control and filtering
5. Command line interface for easy usage

Features:
- Universal symbol extraction across multiple languages
- Grounded QA generation with citations and NOT_IN_CONTEXT handling
- Debug task generation with realistic bug injection
- Negative example generation for abstention training
- Multi-chunk paper QA with cross-section synthesis
- Quality critic filtering and embedding-based deduplication
- Proper instruct format output with system/user/assistant messages
- Integration with existing createFinalDataOutput.py workflow

Outputs:
- Standard JSON format for compatibility
- Instruct JSONL format for direct fine-tuning
- Combined datasets with proper train/val splits
"""

import os
import sys
import json
import argparse
import random
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
import anthropic
from openai import OpenAI
import PyPDF2
from urllib.parse import urlparse

# Project root adjustment
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import universal components
from data_gen.symbols.universal_symbol_extractor import (
    UniversalSymbolExtractor, 
    create_universal_extraction_config,
    UniversalCodeSymbol
)
from data_gen.symbols.multi_language_parser import Language
from data_gen.assembly.instruct_formatter import create_instruct_formatter
from data_gen.assembly import get_config_manager
from data_gen.tasks import (
    GroundedQAGenerator,
    GroundedQA,
    create_debug_generator,
    create_negative_generator,
    NegativeExample,
)
from data_gen.critique import QualityCritic, create_deduplicator
from data_gen.tasks.paper_qa_generator import create_paper_qa_generator
from data_gen.paper_ingestion import PaperLoader, PaperSplitter, SplittingConfig


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_generation.log')
        ]
    )


def convert_negative_to_grounded_qa(negative: NegativeExample, symbol: UniversalCodeSymbol) -> GroundedQA:
    """Convert a NegativeExample to GroundedQA format for compatibility."""
    return GroundedQA(
        question=negative.question,
        answer=negative.expected_response,
        context_symbol_text=negative.context_code,
        context_symbol_name=negative.context_symbol_name,
        context_symbol_type=negative.context_symbol_type,
        context_start_line=getattr(symbol, 'start_line', 0),
        context_end_line=getattr(symbol, 'end_line', 0),
        citations=[],
        task_focus_area="negative_example",
        complexity_level=negative.difficulty_level,
        is_negative_example=True,
        requires_external_knowledge=True
    )


def read_text(path: Path) -> str:
    """Read text file with error handling."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        return ""


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file using PyPDF2."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return ""


def clone_repository(repo_url: str, target_dir: Path) -> bool:
    """Clone a git repository to target directory."""
    try:
        # Parse URL to get repo name
        parsed = urlparse(repo_url)
        if not parsed.netloc:
            # Assume it's a local path
            return True
            
        logging.info(f"Cloning repository {repo_url} to {target_dir}")
        result = subprocess.run(
            ["git", "clone", repo_url, str(target_dir)],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"Successfully cloned repository")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error cloning repository {repo_url}: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error cloning repository: {e}")
        return False


def write_json(path: Path, data: Any):
    """Write JSON file with directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, data: List[Dict[str, Any]]):
    """Write JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_train_val(items: List[Any], train_ratio: float) -> tuple[List[Any], List[Any]]:
    """Split data into train and validation sets."""
    random.shuffle(items)
    k = int(len(items) * max(0.0, min(1.0, train_ratio)))
    return items[:k], items[k:]


def build_llm_clients(
    privacy_mode: bool = False,
    local_model_path: Optional[str] = None,
    device: Optional[str] = None
):
    """
    Initialize LLM clients for production use or privacy mode.

    Args:
        privacy_mode: If True, use local Llama 3.1 8B instead of API clients
        local_model_path: Path or HF model ID for local model (privacy mode only)
        device: Device to use for local model ('cuda', 'cpu', or None for auto-detect)

    Returns:
        Tuple of (claude_client, openai_client) or wrapped local clients in privacy mode
    """
    # Load environment variables
    load_dotenv()

    if privacy_mode:
        # Use local Llama 3.1 8B model
        logging.info("=" * 60)
        logging.info("PRIVACY MODE ENABLED")
        logging.info("Using local Llama 3.1 8B - No data sent to external APIs")
        logging.info("=" * 60)

        from data_gen.llm import create_privacy_mode_client

        # Get HF token from environment
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            logging.warning(
                "HUGGINGFACE_TOKEN not set. This is required for gated models like Llama 3.1. "
                "Get your token from: https://huggingface.co/settings/tokens"
            )

        # Use default model path if not specified
        if local_model_path is None:
            local_model_path = os.getenv('LOCAL_MODEL_PATH', 'meta-llama/Meta-Llama-3.1-8B-Instruct')

        # Create privacy mode clients
        anthropic_client, openai_client = create_privacy_mode_client(
            model_path=local_model_path,
            load_in_8bit=True,  # Use 8-bit quantization for memory efficiency
            device=device,
            hf_token=hf_token
        )

        logging.info("Privacy mode clients initialized successfully")
        logging.info(f"Model: {local_model_path}")
        logging.info("All code and papers will be processed locally")

        return anthropic_client, openai_client

    else:
        # Use production API clients
        # Get API keys from environment
        anthropic_key = os.getenv('CLAUDE_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        openai_key = os.getenv('OPENAI_KEY') or os.getenv('OPENAI_API_KEY')

        if not anthropic_key:
            raise ValueError("CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable is required")
        if not openai_key:
            raise ValueError("OPENAI_KEY or OPENAI_API_KEY environment variable is required")

        # Initialize clients
        anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        openai_client = OpenAI(api_key=openai_key)

        logging.info("Initialized production LLM clients (Anthropic Claude + OpenAI)")
        return anthropic_client, openai_client


def extract_bug_type(debug_task):
    """Extract bug type from debug task safely."""
    try:
        bug_injection = getattr(debug_task, 'bug_injection', None)
        if bug_injection:
            bug_type = getattr(bug_injection, 'bug_type', None)
            if bug_type:
                if hasattr(bug_type, 'value'):
                    return bug_type.value
                else:
                    return str(bug_type)
        return 'logic_error'
    except Exception:
        return 'logic_error'


def process_repo_documentation(
    repo_path: Path,
    config_manager,
    openai_client,
) -> List[Dict[str, Any]]:
    """
    Process documentation files (.md, .Rd) found in a code repository.
    Uses the same pipeline as papers (chunking + QA generation).

    Args:
        repo_path: Path to code repository
        config_manager: Configuration manager
        openai_client: OpenAI client for paper Q&A generation

    Returns:
        List of dataset entries from documentation files
    """
    # Initialize paper processing components
    paper_loader = PaperLoader()
    paper_splitter = PaperSplitter(
        config=SplittingConfig(
            target_chunk_size=600,
            max_chunk_size=800,
            min_chunk_size=200,
            chunk_overlap=50
        )
    )
    paper_gen = create_paper_qa_generator(config_manager, openai_client)

    # Discover documentation files in repo
    md_files = list(repo_path.glob("**/*.md")) + list(repo_path.glob("**/*.MD"))
    rd_files = list(repo_path.glob("**/*.Rd")) + list(repo_path.glob("**/*.rd"))
    doc_files = md_files + rd_files

    if not doc_files:
        logging.info("No documentation files (.md, .Rd) found in repository")
        return []

    logging.info(f"Found {len(md_files)} Markdown and {len(rd_files)} R documentation files in repository")

    dataset: List[Dict[str, Any]] = []

    for file_path in doc_files:
        try:
            logging.info(f"Processing documentation file: {file_path.name}")

            # Load document
            content, metadata = paper_loader.load_document(str(file_path))

            if not content.strip():
                logging.warning(f"No content extracted from {file_path.name}")
                continue

            # Split into semantic chunks
            doc_chunks = paper_splitter.split_document(content, metadata)
            logging.info(f"Split {file_path.name} into {len(doc_chunks)} chunks")

            if not doc_chunks:
                logging.warning(f"No chunks generated for {file_path.name}")
                continue

            # Convert to chunk texts for classification
            chunk_texts = [chunk.content for chunk in doc_chunks]

            # Classify chunks
            classified = paper_gen.classify_chunks(chunk_texts)

            # Generate QA pairs using multi-strategy approach
            qa_pairs = paper_gen.generate_integrative_qa_pairs(
                paper_chunks=classified,
                count=len(doc_chunks)
            )

            # Format dataset entries
            for qa in qa_pairs:
                # Combine source chunks as context
                context = "\n\n".join([chunk.content for chunk in qa.source_chunks])

                entry = {
                    "context_code": context,
                    "question": qa.question,
                    "answer": qa.answer,
                    "task": "documentation_qa",
                    "language": "documentation",
                    "complexity": qa.difficulty_level,
                    "metadata": {
                        "file": str(file_path.relative_to(repo_path)),
                        "doc_type": metadata.get("doc_type", "unknown"),
                        "source": "repo_documentation",
                        "integration_type": qa.integration_type,
                        "chunk_indices": qa.chunk_indices,
                        "requires_cross_reference": qa.requires_cross_reference
                    }
                }
                dataset.append(entry)

            logging.info(f"Generated {len(qa_pairs)} QA pairs from {file_path.name}")

        except Exception as e:
            logging.error(f"Error processing documentation file {file_path}: {e}")
            continue

    logging.info(f"Total QA pairs from repository documentation: {len(dataset)}")
    return dataset


def generate_from_code(
    repo_path: Path,
    config_manager,
    min_tokens: int,
    max_symbols: int,
    languages: List[str],
    include_debug: bool,
    include_negatives: bool,
    apply_critic: bool,
    apply_dedup: bool,
    privacy_mode: bool = False,
    local_model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Process a code repository and return a list of dataset entries."""

    extract_config = create_universal_extraction_config(
        include_private=False,
        max_symbols=max_symbols,
        min_tokens=min_tokens,
        languages=languages
    )
    extractor = UniversalSymbolExtractor(extract_config)

    # Initialize LLM clients (production APIs or local model)
    anthropic_client, openai_client = build_llm_clients(
        privacy_mode=privacy_mode,
        local_model_path=local_model_path,
        device=device
    )
    
    qa_gen = GroundedQAGenerator(config_manager, anthropic_client)
    debug_gen = create_debug_generator(config_manager, anthropic_client)
    neg_gen = create_negative_generator(config_manager)
    
    critic = QualityCritic(config_manager, anthropic_client) if apply_critic else None
    deduper = create_deduplicator() if apply_dedup else None
    
    # Extract symbols from repository
    symbols_by_file = extractor.extract_from_repo(str(repo_path))
    dataset: List[Dict[str, Any]] = []
    
    logging.info(f"Found {len(symbols_by_file)} files with symbols")
    
    for rel_path, symbols in symbols_by_file.items():
        file_entries = []
        
        logging.info(f"Processing {rel_path} with {len(symbols)} symbols")
        
        for symbol in symbols:
            complexity_tier = extractor.get_complexity_tier(symbol)

            # Generate grounded QA
            try:
                logging.debug(f"Generating QA for {symbol.name} (type: {symbol.symbol_type}, complexity: {complexity_tier})")
                qa_pairs = qa_gen.generate_qa_pairs(symbol, complexity_tier)
                logging.debug(f"Generated {len(qa_pairs)} QA pairs for {symbol.name}")
            except Exception as e:
                logging.error(f"Error generating QA for {symbol.name}: {e}", exc_info=True)
                qa_pairs = []

            # Optionally add negative examples
            if include_negatives:
                try:
                    logging.debug(f"Generating negative examples for {symbol.name}")
                    negative_examples = neg_gen.generate_negative_examples(symbol, complexity_tier, count=2)
                    logging.debug(f"Generated {len(negative_examples)} negative examples for {symbol.name}")
                    # Convert NegativeExample objects to GroundedQA for compatibility
                    for neg_ex in negative_examples:
                        qa_pairs.append(convert_negative_to_grounded_qa(neg_ex, symbol))
                except Exception as e:
                    logging.error(f"Error generating negative examples for {symbol.name}: {e}", exc_info=True)

            # Optionally add debug tasks
            debug_tasks = []
            if include_debug:
                try:
                    logging.debug(f"Generating debug tasks for {symbol.name}")
                    debug_tasks = debug_gen.generate_debug_tasks(symbol, complexity_tier)
                    logging.debug(f"Generated {len(debug_tasks)} debug tasks for {symbol.name}")
                except Exception as e:
                    logging.error(f"Error generating debug tasks for {symbol.name}: {e}", exc_info=True)
            
            # Optionally quality-filter QA
            if critic and qa_pairs:
                try:
                    logging.debug(f"Applying quality critic to {len(qa_pairs)} QA pairs for {symbol.name}")
                    high_quality = critic.filter_high_quality(qa_pairs)
                    logging.debug(f"Quality critic filtered to {len(high_quality)} high-quality pairs for {symbol.name}")
                except Exception as e:
                    logging.error(f"Error applying critic to {symbol.name}: {e}", exc_info=True)
                    high_quality = qa_pairs
            else:
                high_quality = qa_pairs

            # Optionally deduplicate QA
            if deduper and high_quality:
                try:
                    logging.debug(f"Deduplicating {len(high_quality)} QA pairs for {symbol.name}")
                    high_quality = deduper.deduplicate(high_quality)
                    logging.debug(f"Deduplication resulted in {len(high_quality)} unique pairs for {symbol.name}")
                except Exception as e:
                    logging.error(f"Error deduplicating {symbol.name}: {e}", exc_info=True)
            
            if not high_quality and not debug_tasks:
                continue
            
            entry = {
                "repo": str(repo_path),
                "file": rel_path,
                "language": symbol.language.value,
                "symbol_name": symbol.name,
                "symbol_type": symbol.symbol_type,
                "start_line": symbol.start_line,
                "end_line": symbol.end_line,
                "source_code": symbol.source_code,  # Add source code for instruct formatting
                "qa_pairs": [
                    {
                        "question": getattr(qa, 'question', ''),
                        "answer": getattr(qa, 'answer', getattr(qa, 'expected_response', 'NOT_IN_CONTEXT')),
                        "citations": getattr(qa, 'citations', []),
                        "context_start_line": getattr(qa, 'context_start_line', symbol.start_line),
                        "context_end_line": getattr(qa, 'context_end_line', symbol.end_line),
                        "focus": getattr(qa, 'task_focus_area', 'general'),
                        "complexity": getattr(qa, 'complexity_level', complexity_tier),
                        "negative": getattr(qa, 'is_negative_example', False),
                        "confidence": getattr(qa, "confidence_score", 0.0),
                    }
                    for qa in high_quality
                ],
                "debugging_tasks": [
                    {
                        "task_type": getattr(t, 'task_type', 'find_bug'),
                        "question": getattr(t, 'question', 'Analyze this code for bugs.'),
                        "expected_answer": getattr(t, 'expected_answer', 'No bugs found.'),
                        "bug_type": extract_bug_type(t),
                        "bug_location": getattr(getattr(t, 'bug_injection', None), 'bug_location', 0) if hasattr(t, 'bug_injection') else 0,
                        "severity": getattr(getattr(t, 'bug_injection', None), 'severity', 'medium') if hasattr(t, 'bug_injection') else 'medium',
                        "difficulty": getattr(t, 'difficulty_level', 'moderate'),
                    }
                    for t in debug_tasks
                ],
            }
            file_entries.append(entry)
        
        # Aggregate per file
        dataset.extend(file_entries)

    logging.info(f"Generated {len(dataset)} dataset entries from code")

    # Also process documentation files (.md, .Rd) found in repository
    logging.info("Processing documentation files in repository...")
    doc_dataset = process_repo_documentation(repo_path, config_manager, openai_client)

    if doc_dataset:
        dataset.extend(doc_dataset)
        logging.info(f"Added {len(doc_dataset)} dataset entries from documentation")

    logging.info(f"Total dataset size (code + documentation): {len(dataset)}")
    return dataset


def generate_from_papers(
    papers_dir: Path,
    config_manager,
    apply_critic: bool,
    privacy_mode: bool = False,
    local_model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate QA pairs from research papers using semantic chunking."""
    if not papers_dir.exists():
        logging.warning(f"Papers directory {papers_dir} does not exist")
        return []

    # Initialize LLM clients (production APIs or local model)
    _, openai_client = build_llm_clients(
        privacy_mode=privacy_mode,
        local_model_path=local_model_path,
        device=device
    )
    paper_gen = create_paper_qa_generator(config_manager, openai_client)

    # Initialize paper loading and chunking components
    paper_loader = PaperLoader()
    paper_splitter = PaperSplitter(
        config=SplittingConfig(
            target_chunk_size=600,
            max_chunk_size=800,
            min_chunk_size=200,
            chunk_overlap=50
        )
    )

    # Collect PDF, text, markdown, and R documentation files
    pdf_files = list(papers_dir.glob("**/*.pdf"))
    text_files = list(papers_dir.glob("**/*.txt"))
    md_files = list(papers_dir.glob("**/*.md")) + list(papers_dir.glob("**/*.MD"))
    rd_files = list(papers_dir.glob("**/*.Rd")) + list(papers_dir.glob("**/*.rd"))
    all_files = pdf_files + text_files + md_files + rd_files
    dataset: List[Dict[str, Any]] = []

    logging.info(f"Found {len(pdf_files)} PDF, {len(text_files)} TXT, {len(md_files)} MD, and {len(rd_files)} Rd files in papers directory")

    for file_path in all_files:
        try:
            logging.info(f"Processing paper: {file_path.name}")

            # Load document with enhanced loading
            content, metadata = paper_loader.load_document(str(file_path))

            if not content.strip():
                logging.warning(f"No content extracted from {file_path.name}")
                continue

            # Split into semantic chunks
            paper_chunks = paper_splitter.split_document(content, metadata)
            logging.info(f"Split {file_path.name} into {len(paper_chunks)} chunks")

            if not paper_chunks:
                logging.warning(f"No chunks generated for {file_path.name}")
                continue

            # Convert PaperChunk objects to simple chunk texts for classification
            chunk_texts = [chunk.content for chunk in paper_chunks]

            # Classify chunks
            classified = paper_gen.classify_chunks(chunk_texts)

            logging.info(f"Generating QA pairs from {len(paper_chunks)} chunks for {file_path.name}")
            logging.info(f"  - Random combinations: up to 10 (2 chunks) + 5 (4 chunks) + 3 (6 chunks)")
            logging.info(f"  - Consecutive windows: 2-chunk, 4-chunk, and 6-chunk windows")

            # Generate integrative QA pairs using multi-strategy approach
            # This will generate QAs using:
            # 1. Random chunk combinations (2, 4, 6 chunks)
            # 2. Consecutive windows (2, 4, 6 chunks)
            qa_pairs = paper_gen.generate_integrative_qa_pairs(classified, count=len(paper_chunks))

            logging.info(f"Generated {len(qa_pairs)} QA pairs for {file_path.name}")

            # Create dataset entry
            entries = [
                {
                    "repo": "research_papers",
                    "file": str(file_path.relative_to(papers_dir)),
                    "language": "research_paper",
                    "chunk_count": len(paper_chunks),
                    "qa_pairs": [
                        {
                            "question": qa.question,
                            "answer": qa.answer,
                            "chunk_indices": getattr(qa, 'chunk_indices', []),
                            "integration_type": getattr(qa, 'integration_type', 'synthesis'),
                            "requires_cross_reference": getattr(qa, 'requires_cross_reference', False),
                        }
                        for qa in qa_pairs
                    ],
                }
                for _ in [0]  # Single entry per file
            ]
            dataset.extend(entries)

        except Exception as e:
            logging.error(f"Error processing paper {file_path}: {e}", exc_info=True)

    logging.info(f"Generated {len(dataset)} dataset entries from {len(all_files)} papers")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive multi-language data generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from single repository
  python run_comprehensive_data_gen.py --repo /path/to/repo

  # Generate from multiple repositories (uses checkpoints - safe to interrupt!)
  python run_comprehensive_data_gen.py --repo /path/repo1 /path/repo2 https://github.com/user/repo3

  # Generate from code and papers
  python run_comprehensive_data_gen.py --repo /path/to/repo --papers /path/to/papers

  # Multiple repos with papers (checkpoints protect your progress)
  python run_comprehensive_data_gen.py --repo /path/repo1 https://github.com/user/repo2 --papers /path/to/papers

  # Resume interrupted pipeline (checkpoints auto-loaded)
  python run_comprehensive_data_gen.py --repo /path/repo1 /path/repo2 /path/repo3

  # Start fresh by clearing checkpoints
  python run_comprehensive_data_gen.py --repo /path/to/repo --clear_checkpoints

  # Quick test with limited output
  python run_comprehensive_data_gen.py --repo /path/to/repo --max_symbols 5 --no_critic --no_dedup
        """
    )
    
    # Input/Output
    parser.add_argument("--repo", nargs="+", help="Path(s) to local code repository(ies) or GitHub URL(s) to process (space-separated).")
    parser.add_argument("--papers", type=str, default="", help="Directory of pre-chunked .txt files for papers.")
    parser.add_argument("--output", type=str, default="comprehensive_outputs", help="Output directory for JSON files.")
    
    # Language support
    parser.add_argument("--languages", nargs="+", default=["python", "r", "c", "cpp"], 
                       help="Languages to support (python, r, c, cpp)")
    
    
    # Data generation
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--min_tokens", type=int, default=30, help="Min tokens per symbol.")
    parser.add_argument("--max_symbols", type=int, default=30, help="Max symbols per file.")
    
    # Features
    parser.add_argument("--no_debug", action="store_true", help="Disable debug task generation.")
    parser.add_argument("--no_negatives", action="store_true", help="Disable extra negative examples.")
    parser.add_argument("--no_critic", action="store_true", help="Disable critic filtering.")
    parser.add_argument("--no_dedup", action="store_true", help="Disable deduplication.")
    parser.add_argument("--clear_checkpoints", action="store_true", help="Delete existing checkpoints and start fresh.")

    # Privacy Mode - Local LLM
    parser.add_argument("--privacy", action="store_true",
                       help="Enable privacy mode - use local Llama 3.1 8B instead of API clients")
    parser.add_argument("--local_model_path", type=str, default=None,
                       help="Path or HF model ID for local model (default: meta-llama/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                       help="Device to use for local model (default: auto-detect)")

    # Debugging
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate inputs
    if not args.repo and not args.papers:
        parser.error("Must specify either --repo or --papers (or both)")

    # Setup output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoints directory for intermediate saves
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Clear checkpoints if requested
    if args.clear_checkpoints:
        checkpoint_files = list(checkpoints_dir.glob("*.json"))
        if checkpoint_files:
            logging.info(f"Clearing {len(checkpoint_files)} existing checkpoint(s)...")
            for ckpt in checkpoint_files:
                try:
                    ckpt.unlink()
                    logging.info(f"  Deleted: {ckpt.name}")
                except Exception as e:
                    logging.warning(f"  Failed to delete {ckpt.name}: {e}")
        else:
            logging.info("No checkpoints to clear")

    logging.info(f"Starting comprehensive data generation pipeline")
    logging.info(f"Languages: {args.languages}")
    logging.info(f"Output directory: {out_dir}")
    logging.info(f"Checkpoints directory: {checkpoints_dir}")

    # Load config
    config_manager = get_config_manager()

    # Code processing - handle multiple repositories
    code_dataset: List[Dict[str, Any]] = []
    if args.repo:
        repos_to_process = args.repo if isinstance(args.repo, list) else [args.repo]
        logging.info(f"Processing {len(repos_to_process)} code repository(ies)")

        # Check for existing checkpoints to resume from
        existing_checkpoints = list(checkpoints_dir.glob("repo_*.json"))
        if existing_checkpoints:
            logging.info(f"Found {len(existing_checkpoints)} existing checkpoint(s)")

        # Track temp directories for cleanup
        temp_dirs = []

        try:
            # Process each repository
            for i, repo in enumerate(repos_to_process, 1):
                logging.info(f"\n{'='*60}")
                logging.info(f"Repository {i}/{len(repos_to_process)}: {repo}")
                logging.info(f"{'='*60}")

                # Generate checkpoint filename from repo name
                repo_name = Path(repo).name.replace('.git', '')
                checkpoint_file = checkpoints_dir / f"repo_{i:02d}_{repo_name}.json"

                # Check if checkpoint exists - skip processing if it does
                if checkpoint_file.exists():
                    try:
                        logging.info(f"Loading existing checkpoint: {checkpoint_file.name}")
                        with open(checkpoint_file, 'r', encoding='utf-8') as f:
                            repo_data = json.load(f)
                        code_dataset.extend(repo_data)
                        logging.info(f"Loaded {len(repo_data)} entries from checkpoint {i}")
                        continue  # Skip to next repo
                    except Exception as e:
                        logging.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
                        logging.info("Reprocessing repository...")

                # Handle GitHub URLs by cloning to a temporary directory
                repo_path = Path(repo)
                temp_dir = None

                # Check if it's a URL or local path
                parsed = urlparse(repo)
                if parsed.netloc:  # It's a URL
                    temp_dir = Path(tempfile.mkdtemp(prefix="cloned_repo_"))
                    if clone_repository(repo, temp_dir):
                        repo_path = temp_dir
                        temp_dirs.append(temp_dir)
                    else:
                        logging.error(f"Failed to clone repository {repo}, skipping")
                        continue
                elif not repo_path.exists():
                    logging.error(f"Local repository path does not exist: {repo}, skipping")
                    continue

                if repo_path and repo_path.exists():
                    repo_data = generate_from_code(
                        repo_path,
                        config_manager,
                        args.min_tokens,
                        args.max_symbols,
                        args.languages,
                        include_debug=not args.no_debug,
                        include_negatives=not args.no_negatives,
                        apply_critic=not args.no_critic,
                        apply_dedup=not args.no_dedup,
                        privacy_mode=args.privacy,
                        local_model_path=args.local_model_path,
                        device=args.device,
                    )
                    code_dataset.extend(repo_data)
                    logging.info(f"Added {len(repo_data)} entries from repository {i}")

                    # Save checkpoint immediately after processing
                    try:
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            json.dump(repo_data, f, indent=2, ensure_ascii=False)
                        logging.info(f"✓ Checkpoint saved: {checkpoint_file.name} ({len(repo_data)} entries)")
                    except Exception as e:
                        logging.error(f"Failed to save checkpoint {checkpoint_file}: {e}")

            logging.info(f"\nTotal entries from all {len(repos_to_process)} repositories: {len(code_dataset)}")
        finally:
            # Clean up temporary directories
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                        logging.info(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        logging.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
    
    # Papers processing (optional)
    papers_dataset: List[Dict[str, Any]] = []
    if args.papers:
        papers_checkpoint_file = checkpoints_dir / "papers_dataset.json"

        # Check if papers checkpoint exists
        if papers_checkpoint_file.exists():
            try:
                logging.info(f"Loading existing papers checkpoint: {papers_checkpoint_file.name}")
                with open(papers_checkpoint_file, 'r', encoding='utf-8') as f:
                    papers_dataset = json.load(f)
                logging.info(f"Loaded {len(papers_dataset)} entries from papers checkpoint")
            except Exception as e:
                logging.warning(f"Failed to load papers checkpoint: {e}")
                logging.info("Reprocessing papers...")
                papers_dataset = []

        # Process papers if no checkpoint or checkpoint failed to load
        if not papers_dataset:
            logging.info(f"Processing papers directory: {args.papers}")
            papers_dataset = generate_from_papers(
                Path(args.papers),
                config_manager,
                apply_critic=not args.no_critic,
                privacy_mode=args.privacy,
                local_model_path=args.local_model_path,
                device=args.device,
            )

            # Save papers checkpoint
            try:
                with open(papers_checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(papers_dataset, f, indent=2, ensure_ascii=False)
                logging.info(f"✓ Papers checkpoint saved: {papers_checkpoint_file.name} ({len(papers_dataset)} entries)")
            except Exception as e:
                logging.error(f"Failed to save papers checkpoint: {e}")
    
    # Split datasets
    code_train, code_val = split_train_val(code_dataset, args.train_ratio) if code_dataset else ([], [])
    papers_train, papers_val = split_train_val(papers_dataset, args.train_ratio) if papers_dataset else ([], [])

    # Generate instruct format (only format we output)
    logging.info("Converting to instruct format")
    formatter = create_instruct_formatter(system_prompts=True)

    # Convert code datasets
    code_train_instruct = formatter.convert_dataset_to_instruct(code_train)
    code_val_instruct = formatter.convert_dataset_to_instruct(code_val)

    # Convert paper datasets
    papers_train_instruct = formatter.convert_paper_dataset_to_instruct(papers_train)
    papers_val_instruct = formatter.convert_paper_dataset_to_instruct(papers_val)

    # Save instruct format
    write_jsonl(out_dir / "code_instruct_train.jsonl", code_train_instruct)
    write_jsonl(out_dir / "code_instruct_val.jsonl", code_val_instruct)
    write_jsonl(out_dir / "papers_instruct_train.jsonl", papers_train_instruct)
    write_jsonl(out_dir / "papers_instruct_val.jsonl", papers_val_instruct)

    # Combined instruct format
    combined_train_instruct = code_train_instruct + papers_train_instruct
    combined_val_instruct = code_val_instruct + papers_val_instruct
    write_jsonl(out_dir / "combined_instruct_train.jsonl", combined_train_instruct)
    write_jsonl(out_dir / "combined_instruct_val.jsonl", combined_val_instruct)

    logging.info(f"Generated {len(combined_train_instruct)} training and {len(combined_val_instruct)} validation instruct examples")
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Total code entries: {len(code_dataset)}")
    print(f"Total paper entries: {len(papers_dataset)}")
    print(f"Training examples: {len(combined_train_instruct)}")
    print(f"Validation examples: {len(combined_val_instruct)}")

    print(f"\nOutput files saved to: {out_dir}")
    print("\nInstruct format files:")
    for filename in ["code_instruct_train.jsonl", "code_instruct_val.jsonl",
                     "papers_instruct_train.jsonl", "papers_instruct_val.jsonl",
                     "combined_instruct_train.jsonl", "combined_instruct_val.jsonl"]:
        filepath = out_dir / filename
        if filepath.exists():
            print(f"  - {filepath}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()