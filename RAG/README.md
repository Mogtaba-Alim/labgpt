# LabGPT RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system for processing documents and enabling intelligent hybrid search.

## Quick Start

### Installation

```bash
# Navigate to the RAG directory
cd RAG

# Install dependencies
pip install -r requirements.txt
```

**Important:** All commands below should be run from the **parent directory** (labgpt repository root), not from inside RAG/.

```bash
# Navigate back to parent directory
cd ..
```

### Basic Usage

```python
from RAG.pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(index_dir="my_rag")

# Add documents
rag.add_documents(["papers/"])

# Search
results = rag.search("How does CRISPR work?")

# Display results
for result in results:
    print(f"[{result.rank}] {result.chunk.source_path}")
    print(f"Score: {result.score:.3f}\n")
```

### CLI Usage

```bash
# Ingest documents (auto-detect device)
python -m RAG.cli ingest --docs papers/ --index my_rag

# Ingest with specific device (cpu/cuda/auto)
python -m RAG.cli ingest --docs papers/ --index my_rag --device cpu

# Search
python -m RAG.cli ask --index my_rag --query "How does CRISPR work?"

# Check status
python -m RAG.cli status --index my_rag
```

**Device Selection:**
- `--device auto` (default): Automatically uses GPU if available and not locked by other tasks
- `--device cuda`: Forces GPU usage (may conflict with training)
- `--device cpu`: Uses CPU (slower but no GPU conflicts)

## Key Features

- **Hybrid Retrieval**: FAISS (dense) + BM25 (sparse) with RRF fusion
- **Document-Type Adapters**: Optimal chunking for code, papers, slides, protocols
- **Cross-Encoder Reranking**: Improved precision with joint query-document scoring
- **Query Expansion**: PRF-style expansion using embedding similarity
- **Cited Spans**: Extract supporting text with character offsets
- **Embedding Cache**: SHA256-based caching for 50-1000x speedup on incremental updates
- **Reproducibility Snapshots**: SHA256 verification for exact index state tracking
- **Device-Aware Loading**: Auto-detects GPU availability with lock coordination to prevent conflicts
- **Model Caching**: Multiple pipeline instances share cached models to reduce memory usage

## Testing

Test all pipeline features using the automated test script:

```bash
# Make sure you're in labgpt directory (not RAG/)
cd /path/to/labgpt

# Run comprehensive feature test
bash RAG/test_rag_features.sh /path/to/your/corpus "test query"

# Example
bash RAG/test_rag_features.sh /path/to/documents/ "machine learning"
```

The test script runs 11 tests covering:
- Basic ingestion and search
- Query expansion (PRF)
- Cited span extraction
- Embedding cache performance
- Reproducibility snapshots
- Research preset with all enhancements

See [test_commands.txt](test_commands.txt) for individual CLI commands.

## Documentation

For complete documentation, see [rag_comprehensive_guide.md](rag_comprehensive_guide.md)

## Architecture

```
RAG/
├── pipeline.py          # Main API
├── cli.py              # Command-line interface
├── models.py           # Data models
├── ingestion/          # Document loading and chunking
├── retrieval/          # Hybrid search and fusion
└── generation/         # Span extraction and guardrails
```

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt) for dependencies

## License

Part of the LabGPT suite for scientific document processing.
