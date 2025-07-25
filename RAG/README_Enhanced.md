# Enhanced RAG System - Complete Implementation

This directory contains the unified enhanced RAG (Retrieval-Augmented Generation) system with state-of-the-art ingestion and retrieval capabilities. This represents a complete transformation from the original simple character-based chunking system to an intelligent, adaptive, production-ready RAG platform.

## System Overview

The Enhanced RAG System integrates all advanced capabilities into a single, unified pipeline:

### Core Features
- **Hybrid Retrieval**: Combines dense (FAISS) + sparse (BM25) retrieval with sophisticated fusion methods
- **Semantic Text Splitting**: Token-aware chunking that preserves document structure and semantic boundaries  
- **Rich Metadata Management**: 20+ metadata fields per chunk including hierarchical structure and quality scores
- **Quality Assessment**: Multi-dimensional quality filtering and intelligent content pruning
- **Configurable Pipeline**: YAML-based configuration for all processing parameters

### Advanced Features  
- **Embedding Caching**: SHA256-based content caching for incremental processing efficiency
- **Index Versioning**: Complete version control with rollback capabilities for production safety
- **Retrieval Telemetry**: Comprehensive monitoring and analytics for continuous optimization
- **Incremental Updates**: Real-time content updates without full system rebuilds

### Intelligent Features
- **Adaptive Retrieval**: Dynamic top-k adjustment based on coverage heuristics
- **Answer Guardrails**: Citation verification and factual accuracy checking
- **Per-Document Management**: Granular document-level embedding management
- **Git Integration**: Sophisticated change detection for version-controlled document collections

## Installation & Setup

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install core dependencies
pip install -r requirements_enhanced.txt

# Download required NLTK data
python -c "
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 
               'maxent_ne_chunker', 'words'])
"
```

### Verify Installation
```python
# Test basic functionality
from RAG import EnhancedIngestionPipeline, get_system_info

print("🧪 Testing installation...")
info = get_system_info()
print(f"✅ System: {info['description']}")
print(f"📦 Version: {info['version']}")

pipeline = EnhancedIngestionPipeline()
print(f"✅ Pipeline initialized on device: {pipeline.embedding_model.device}")
print("🎉 Installation successful!")
```

### Quick Test
Run this simple test to verify everything works:
```bash
# In the RAG directory, activate your conda environment
conda activate labgpt

# Run the comprehensive test (see below for test file)
python test_rag_system.py
```

## Quick Start

### Basic Document Processing
```python
from RAG import create_pipeline

# Create a basic pipeline with core features only
pipeline = create_pipeline(
    storage_dir="my_rag_storage",
    enable_advanced_scoring=False,
    enable_caching=False,
    enable_adaptive_retrieval=False
)

# Process documents
document_paths = ["path/to/document1.pdf", "path/to/document2.txt"]
results = pipeline.process_documents(document_paths)

print(f"✅ Processed {results['documents_processed']} documents")
print(f"📄 Created {results['total_chunks']} chunks")
print(f"⏱️ Processing time: {results['processing_time']:.2f}s")

# Build retrieval system
retriever = pipeline.build_retrieval_system()

# Test retrieval
query = "machine learning algorithms"
search_results = retriever.retrieve(query, top_k=5)

for i, result in enumerate(search_results, 1):
    print(f"\n{i}. Score: {result.score:.3f} | Method: {result.retrieval_method}")
    print(f"   Text: {result.chunk.text[:200]}...")
```

### Production-Ready Pipeline
```python
from RAG import create_production_pipeline

# Create fully-featured production pipeline
pipeline = create_production_pipeline(storage_dir="production_storage")

# Process documents with all advanced features
results = pipeline.process_documents(document_paths, batch_size=32)

# Build retrieval system with telemetry
retriever = pipeline.build_retrieval_system()

# Create adaptive retriever for intelligent top-k selection
adaptive_retriever = pipeline.create_adaptive_retriever(retriever)

# Test with adaptive retrieval
query = "deep learning neural networks optimization"
adaptive_results, coverage_metrics = adaptive_retriever.retrieve_adaptive(query)

print(f"📊 Retrieved {len(adaptive_results)} chunks")
print(f"📈 Coverage score: {coverage_metrics.overall_coverage:.3f}")
print(f"🎯 Semantic diversity: {coverage_metrics.semantic_diversity:.3f}")

# Verify answer quality (if you have generated text)
generated_answer = "Your LLM-generated answer here..."
verification = pipeline.verify_answer(generated_answer, [r.chunk for r in adaptive_results])
if verification:
    print(f"✅ Answer verification: {verification['verification_status']}")
```

## Advanced Usage

### Custom Configuration
```python
# Create pipeline with specific feature configuration
from RAG import EnhancedIngestionPipeline

pipeline = EnhancedIngestionPipeline(
    config_path="custom_config.yaml",
    storage_dir="custom_storage",
    
    # Core features
    embedding_model_name="all-mpnet-base-v2",
    
    # Advanced features (toggle as needed)
    enable_advanced_scoring=True,    # Intelligent quality assessment
    enable_caching=True,            # SHA256-based embedding cache
    enable_versioning=True,         # Index version control
    enable_telemetry=True,          # Performance monitoring
    enable_incremental=True,        # Real-time updates
    
    # Intelligent features
    enable_adaptive_retrieval=True,  # Dynamic top-k adjustment
    enable_answer_guardrails=True,  # Citation verification
    enable_per_doc_management=True, # Document-level management
    enable_git_tracking=True        # Git-based change detection
)
```

### Configuration File Example
Create a `custom_config.yaml`:
```yaml
# Model Configuration
models:
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  device: "auto"  # "cpu", "cuda", or "auto"

# Dense Retrieval Settings
dense:
  top_k: 20
  index_type: "hnsw"  # "hnsw", "ivf", "flat"
  nprobe: 10
  ef_search: 128

# Sparse Retrieval Settings  
sparse:
  top_k: 20
  k1: 1.2    # BM25 term frequency saturation
  b: 0.75    # BM25 length normalization

# Result Fusion
fusion:
  fusion_method: "rrf"  # "rrf", "linear", "rank_sum"
  dense_weight: 0.6
  sparse_weight: 0.4
  rrf_k: 60

# Text Splitting
splitting:
  target_chunk_size: 400  # tokens
  chunk_overlap: 50       # tokens
  respect_sentence_boundaries: true
  preserve_section_structure: true

# Quality Filtering
quality:
  min_quality_score: 0.4
  filter_very_short: true
  filter_very_long: true
  remove_duplicates: true

# Query Processing
query_processing:
  enable_expansion: true
  max_expansions: 3
  expansion_methods: ["wordnet", "embedding"]

# Re-ranking
reranking:
  enable: true
  rerank_top_k: 50
  final_top_k: 10
```

### Incremental Updates
```python
# Monitor and update documents in real-time
def monitor_document_changes():
    # Initial processing
    initial_docs = ["docs/paper1.pdf", "docs/paper2.pdf"]
    pipeline.process_documents(initial_docs)
    
    # Later, add new documents
    new_docs = ["docs/paper3.pdf", "docs/paper4.pdf"] 
    update_results = pipeline.incremental_update(new_docs)
    
    print(f"📝 Updated {update_results['documents_updated']} documents")
    print(f"⏱️ Update time: {update_results['update_time']:.2f}s")
    
    # The retrieval system automatically uses updated indices
    retriever = pipeline.build_retrieval_system()
```

### Per-Document Management
```python
# Fine-grained document control
if pipeline.enable_per_doc_management:
    doc_manager = pipeline.per_doc_manager
    
    # Get document information
    doc_info = doc_manager.get_document_info("document_id")
    print(f"📄 Document: {doc_info['metadata']['title']}")
    print(f"📊 Chunks: {len(doc_info['chunks'])}")
    print(f"🔄 Last updated: {doc_info['last_updated']}")
    
    # Update specific document
    doc_manager.update_document("path/to/updated_document.pdf")
```

## Testing & Evaluation

### 1. System Functionality Tests

#### Basic Pipeline Test
```python
def test_basic_functionality():
    """Test core pipeline functionality"""
    from RAG import create_basic_pipeline
    import os
    import shutil
    
    # Create test document
    test_doc = "test_document.txt"
    with open(test_doc, 'w') as f:
        f.write("""Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that focuses on 
algorithms that can learn and improve from experience without being 
explicitly programmed. The main types include supervised learning, 
unsupervised learning, and reinforcement learning.

Supervised Learning

In supervised learning, algorithms learn from labeled training data to 
make predictions on new, unseen data. Common algorithms include linear 
regression, decision trees, and neural networks.
""")
    
    try:
        print("🧪 Testing Basic Pipeline Functionality...")
        
        # Initialize pipeline
        pipeline = create_basic_pipeline(storage_dir="test_storage")
        print("✅ Pipeline initialized successfully")
        
        # Test document processing
        results = pipeline.process_documents([test_doc])
        assert results['documents_processed'] > 0, "No documents processed"
        assert results['total_chunks'] > 0, "No chunks created"
        print(f"✅ Processed {results['documents_processed']} documents, {results['total_chunks']} chunks")
        
        # Test retrieval system
        retriever = pipeline.build_retrieval_system()
        print("✅ Retrieval system built successfully")
        
        # Test basic retrieval
        search_results = retriever.retrieve("machine learning", top_k=3)
        assert len(search_results) > 0, "No search results returned"
        print(f"✅ Retrieved {len(search_results)} relevant chunks")
        
        # Test different query types
        test_queries = [
            "supervised learning algorithms",
            "what is machine learning?", 
            "neural networks decision trees"
        ]
        
        for query in test_queries:
            results = retriever.retrieve(query, top_k=2)
            print(f"   Query: '{query}' → {len(results)} results")
            if results:
                print(f"     Best score: {results[0].score:.3f}")
        
        print("🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if os.path.exists(test_doc):
            os.remove(test_doc)
        if os.path.exists("test_storage"):
            shutil.rmtree("test_storage", ignore_errors=True)
```

#### Advanced Features Test
```python
def test_advanced_features():
    """Test advanced pipeline features"""
    from RAG import create_production_pipeline
    import os
    import shutil
    
    # Create test document
    test_doc = "advanced_test_doc.txt"
    with open(test_doc, 'w') as f:
        f.write("""Deep Learning and Neural Networks

Deep learning is a subset of machine learning that uses artificial neural 
networks with multiple layers to model and understand complex patterns in data.
Neural networks consist of interconnected nodes that process information
through weighted connections and activation functions.

Applications include computer vision, natural language processing, and 
speech recognition. Popular architectures include convolutional neural 
networks for image processing and recurrent neural networks for sequences.
""")
    
    try:
        print("🧪 Testing Advanced Pipeline Features...")
        
        pipeline = create_production_pipeline(storage_dir="advanced_test_storage")
        print("✅ Production pipeline initialized")
        
        # Process documents for testing
        results = pipeline.process_documents([test_doc])
        print(f"✅ Processed {results['total_chunks']} chunks for testing")
        
        # Test caching (process twice)
        if pipeline.enable_caching:
            print("📊 Testing embedding caching...")
            initial_stats = pipeline.get_system_statistics()
            
            # Process again to test cache
            pipeline.process_documents([test_doc])
            final_stats = pipeline.get_system_statistics()
            
            if 'cache_stats' in final_stats:
                cache_stats = final_stats['cache_stats']
                print(f"   Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
                print("✅ Embedding caching tested")
            else:
                print("⚠️ Cache stats not available")
        
        # Test adaptive retrieval
        if pipeline.enable_adaptive_retrieval:
            print("📊 Testing adaptive retrieval...")
            retriever = pipeline.build_retrieval_system()
            adaptive_retriever = pipeline.create_adaptive_retriever(retriever)
            
            if adaptive_retriever:
                complex_query = "deep learning neural networks applications"
                results, coverage = adaptive_retriever.retrieve_adaptive(complex_query)
                
                print(f"   Retrieved {len(results)} chunks, coverage: {coverage.overall_coverage:.3f}")
                print("✅ Adaptive retrieval tested")
            else:
                print("⚠️ Adaptive retriever not available")
        
        # Test answer guardrails
        if pipeline.enable_answer_guardrails:
            print("📊 Testing answer guardrails...")
            retriever = pipeline.build_retrieval_system()
            
            test_answer = "Deep learning uses neural networks with multiple layers to process data."
            test_chunks = [result.chunk for result in retriever.retrieve("deep learning", top_k=3)]
            
            verification = pipeline.verify_answer(test_answer, test_chunks)
            if verification:
                print(f"   Verification status: {verification['verification_status']}")
                print(f"   Claims processed: {len(verification.get('verification_results', []))}")
                print("✅ Answer guardrails tested")
            else:
                print("⚠️ Answer guardrails not available")
        
        print("🎉 All advanced features tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if os.path.exists(test_doc):
            os.remove(test_doc)
        if os.path.exists("advanced_test_storage"):
            shutil.rmtree("advanced_test_storage", ignore_errors=True)
```

### 2. Performance Evaluation

#### Retrieval Quality Assessment
```python
def evaluate_retrieval_quality():
    """Evaluate retrieval quality with test queries"""
    
    # Define test queries with expected relevant content
    test_cases = [
        {
            "query": "machine learning algorithms",
            "expected_terms": ["algorithm", "learning", "model", "training"],
            "expected_concepts": ["supervised", "unsupervised", "neural network"]
        },
        {
            "query": "what is deep learning?",
            "expected_terms": ["deep", "learning", "neural", "network"],
            "expected_concepts": ["layers", "backpropagation", "activation"]
        },
        {
            "query": "natural language processing techniques",
            "expected_terms": ["language", "processing", "text", "nlp"],
            "expected_concepts": ["tokenization", "embedding", "transformer"]
        }
    ]
    
    pipeline = create_production_pipeline()
    retriever = pipeline.build_retrieval_system()
    
    results = {}
    
    for test_case in test_cases:
        query = test_case["query"]
        
        # Test different retrieval methods
        dense_results = retriever._dense_retrieve(query, top_k=10)
        sparse_results = retriever._sparse_retrieve(query, top_k=10)
        hybrid_results = retriever.retrieve(query, top_k=10)
        
        # Evaluate term coverage
        def calculate_term_coverage(results, expected_terms):
            found_terms = 0
            total_text = " ".join([r.chunk.text.lower() for r in results])
            
            for term in expected_terms:
                if term.lower() in total_text:
                    found_terms += 1
            
            return found_terms / len(expected_terms) if expected_terms else 0
        
        results[query] = {
            "dense_coverage": calculate_term_coverage(dense_results, test_case["expected_terms"]),
            "sparse_coverage": calculate_term_coverage(sparse_results, test_case["expected_terms"]),
            "hybrid_coverage": calculate_term_coverage(hybrid_results, test_case["expected_terms"]),
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "hybrid_count": len(hybrid_results)
        }
    
    # Print evaluation results
    print("\n📊 Retrieval Quality Evaluation")
    print("=" * 50)
    
    for query, metrics in results.items():
        print(f"\nQuery: {query}")
        print(f"  Dense coverage:  {metrics['dense_coverage']:.2%} ({metrics['dense_count']} results)")
        print(f"  Sparse coverage: {metrics['sparse_coverage']:.2%} ({metrics['sparse_count']} results)")
        print(f"  Hybrid coverage: {metrics['hybrid_coverage']:.2%} ({metrics['hybrid_count']} results)")
    
    return results
```

#### Performance Benchmarking
```python
def benchmark_performance():
    """Benchmark system performance"""
    import time
    import psutil
    import os
    
    # Create test datasets of different sizes
    test_datasets = {
        "small": ["doc1.txt", "doc2.txt"],         # 2 documents
        "medium": [f"doc{i}.txt" for i in range(10)],  # 10 documents  
        "large": [f"doc{i}.txt" for i in range(50)]    # 50 documents
    }
    
    # Test different pipeline configurations
    configs = {
        "basic": {
            "enable_caching": False,
            "enable_advanced_scoring": False,
            "enable_adaptive_retrieval": False
        },
        "cached": {
            "enable_caching": True,
            "enable_advanced_scoring": False,
            "enable_adaptive_retrieval": False
        },
        "full": {
            "enable_caching": True,
            "enable_advanced_scoring": True,
            "enable_adaptive_retrieval": True
        }
    }
    
    benchmark_results = {}
    
    for config_name, config in configs.items():
        print(f"\n🔧 Testing {config_name} configuration...")
        
        pipeline = EnhancedIngestionPipeline(
            storage_dir=f"benchmark_{config_name}",
            **config
        )
        
        config_results = {}
        
        for dataset_name, documents in test_datasets.items():
            print(f"  📄 Processing {dataset_name} dataset ({len(documents)} docs)...")
            
            # Measure memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time document processing
            start_time = time.time()
            results = pipeline.process_documents(documents)
            processing_time = time.time() - start_time
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            # Build retrieval system
            start_time = time.time()
            retriever = pipeline.build_retrieval_system()
            indexing_time = time.time() - start_time
            
            # Test retrieval speed
            test_queries = ["machine learning", "neural networks", "data analysis"]
            retrieval_times = []
            
            for query in test_queries:
                start_time = time.time()
                retriever.retrieve(query, top_k=10)
                retrieval_times.append(time.time() - start_time)
            
            avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
            
            config_results[dataset_name] = {
                "processing_time": processing_time,
                "indexing_time": indexing_time,
                "avg_retrieval_time": avg_retrieval_time,
                "memory_used_mb": memory_used,
                "chunks_created": results['total_chunks'],
                "chunks_per_second": results['total_chunks'] / processing_time if processing_time > 0 else 0
            }
        
        benchmark_results[config_name] = config_results
    
    # Print benchmark results
    print("\n📊 Performance Benchmark Results")
    print("=" * 60)
    
    for config_name, config_results in benchmark_results.items():
        print(f"\n🔧 {config_name.upper()} Configuration:")
        
        for dataset_name, metrics in config_results.items():
            print(f"  📄 {dataset_name} dataset:")
            print(f"    Processing: {metrics['processing_time']:.2f}s ({metrics['chunks_per_second']:.1f} chunks/s)")
            print(f"    Indexing:   {metrics['indexing_time']:.2f}s")
            print(f"    Retrieval:  {metrics['avg_retrieval_time']:.3f}s avg")
            print(f"    Memory:     {metrics['memory_used_mb']:.1f} MB")
            print(f"    Chunks:     {metrics['chunks_created']}")
    
    return benchmark_results
```

### 3. System Monitoring

#### Real-time Statistics
```python
def monitor_system_performance():
    """Monitor system performance in real-time"""
    
    pipeline = create_production_pipeline()
    
    # Get comprehensive statistics
    stats = pipeline.get_system_statistics()
    
    print("📊 System Performance Dashboard")
    print("=" * 40)
    
    # Pipeline statistics
    pipeline_stats = stats['pipeline_stats']
    print(f"\n📄 Documents processed: {pipeline_stats['documents_processed']}")
    print(f"📝 Chunks created: {pipeline_stats['chunks_created']}")
    print(f"🗑️ Chunks pruned: {pipeline_stats['chunks_pruned']}")
    print(f"⏱️ Total processing time: {pipeline_stats['processing_time']:.2f}s")
    
    # Feature status
    features = stats['feature_status']
    print(f"\n🔧 Feature Status:")
    for feature, enabled in features.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {feature.replace('_', ' ').title()}")
    
    # Cache statistics (if enabled)
    if 'cache_stats' in stats:
        cache_stats = stats['cache_stats']
        print(f"\n💾 Cache Performance:")
        print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"  Cache size: {cache_stats['cache_size']} entries")
        print(f"  Memory usage: {cache_stats['memory_usage_mb']:.1f} MB")
    
    # Telemetry data (if enabled)
    if 'telemetry_stats' in stats:
        telemetry = stats['telemetry_stats']
        print(f"\n📈 Retrieval Analytics:")
        print(f"  Queries processed: {telemetry['total_queries']}")
        print(f"  Avg response time: {telemetry['avg_response_time']:.3f}s")
        print(f"  P95 response time: {telemetry['p95_response_time']:.3f}s")
    
    # Guardrails statistics (if enabled)
    if 'guardrails_stats' in stats:
        guardrails = stats['guardrails_stats']
        print(f"\n🛡️ Answer Verification:")
        print(f"  Total verifications: {guardrails['total_verifications']}")
        print(f"  Success rate: {guardrails['successful_verifications']}/{guardrails['total_verifications']}")
        print(f"  Claims processed: {guardrails['claims_processed']}")
```

### 4. Evaluation Best Practices

#### Complete Test Suite
A comprehensive test file `test_rag_system.py` is provided that tests all system functionality:

```bash
# Run the complete test suite
conda activate labgpt
python test_rag_system.py
```

This test suite will:
- ✅ Test installation and imports
- ✅ Test basic pipeline functionality  
- ✅ Test advanced features (caching, adaptive retrieval, guardrails)
- ✅ Test configuration options
- ✅ Evaluate retrieval quality
- ✅ Benchmark performance
- ✅ Clean up all test files automatically

The test creates temporary documents, runs comprehensive tests, and cleans up everything afterward. Expected output:
```
🧪 Enhanced RAG System - Comprehensive Test Suite
📝 Creating test documents...
✅ Created 3 test documents

🧪 Testing Installation...
✅ System: Complete Enhanced RAG System with Advanced Intelligence
📦 Version: 1.0.0
🔧 Features: 11 available

[... detailed test results ...]

🎉 ALL TESTS PASSED! RAG System is fully functional!
```

## Command Line Interface

### Basic Usage
```bash
# Process documents with default settings
python -m RAG.enhanced_ingestion docs/*.pdf --storage my_storage

# Use custom configuration
python -m RAG.enhanced_ingestion docs/*.pdf --config custom_config.yaml

# Enable specific features
python -m RAG.enhanced_ingestion docs/*.pdf \
    --enable-caching \
    --enable-adaptive \
    --enable-guardrails

# Test retrieval with a query
python -m RAG.enhanced_ingestion docs/*.pdf \
    --test-query "machine learning algorithms" \
    --storage test_storage
```

### Advanced Options
```bash
# Disable specific features
python -m RAG.enhanced_ingestion docs/*.pdf \
    --disable-caching \
    --disable-adaptive \
    --disable-guardrails

# Export system statistics
python -m RAG.enhanced_ingestion docs/*.pdf \
    --export-stats results.json

# Incremental update mode
python -m RAG.enhanced_ingestion new_docs/*.pdf \
    --storage existing_storage \
    --incremental-update
```

## Production Deployment

### Recommended Configuration
```python
# Production-ready configuration
production_pipeline = EnhancedIngestionPipeline(
    config_path="production_config.yaml",
    storage_dir="/data/rag_storage",
    
    # Enable all production features
    enable_advanced_scoring=True,
    enable_caching=True,
    enable_versioning=True,
    enable_telemetry=True,
    enable_incremental=True,
    enable_adaptive_retrieval=True,
    enable_answer_guardrails=True,
    enable_per_doc_management=True,
    enable_git_tracking=True
)
```

### Monitoring Setup
```python
# Set up logging for production
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

# Enable detailed telemetry
pipeline.telemetry.enable_detailed_logging()
```

### Performance Optimization
```yaml
# production_config.yaml
models:
  device: "cuda"  # Use GPU if available
  
dense:
  index_type: "hnsw"  # Faster for large datasets
  ef_search: 256      # Higher accuracy
  
processing:
  batch_size: 64      # Optimize for your hardware
  num_threads: 8      # Parallel processing
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements_enhanced.txt`

2. **NLTK Data Missing**: 
   ```python
   import nltk
   nltk.download('all')  # Download all required data
   ```

3. **GPU Memory Issues**: Reduce batch size or use CPU
   ```python
   pipeline = EnhancedIngestionPipeline(
       embedding_model_name="all-MiniLM-L6-v2",  # Smaller model
       # ... other config
   )
   ```

4. **Poor Retrieval Quality**: 
   - Adjust quality filtering thresholds
   - Try different embedding models
   - Tune fusion weights

5. **Slow Performance**:
   - Enable caching
   - Use GPU acceleration
   - Increase batch sizes
   - Use parallel processing

### Debug Mode
```python
# Enable debug logging for detailed information
import logging
logging.getLogger('RAG').setLevel(logging.DEBUG)

# Test individual components
pipeline = create_basic_pipeline()
pipeline.export_configuration("debug_config.json")
```

### Support

For issues and questions:
1. Check the implementation log in `implementation.md`
2. Review the configuration in `config/default_config.yaml`
3. Enable debug logging to identify specific issues
4. Test individual components separately

---

This unified Enhanced RAG System provides a comprehensive solution for document processing and intelligent retrieval, with extensive configuration options and monitoring capabilities to meet production requirements. 