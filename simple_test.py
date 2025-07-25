#!/usr/bin/env python3
"""
Simple test to verify RAG system fixes
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_basic_imports():
    """Test basic imports work"""
    try:
        from RAG import create_basic_pipeline, EnhancedIngestionPipeline
        print("✅ Basic imports work")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pipeline_creation():
    """Test pipeline creation"""
    try:
        from RAG import create_basic_pipeline
        pipeline = create_basic_pipeline(storage_dir="test_simple")
        print("✅ Pipeline creation works")
        return True
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        return False

def test_chunk_store_methods():
    """Test that ChunkStore has required methods"""
    try:
        from RAG.ingestion.chunk_objects import ChunkStore
        store = ChunkStore(Path("test_chunks"))
        
        # Check if methods exist
        assert hasattr(store, 'store_chunks'), "Missing store_chunks method"
        assert hasattr(store, 'load_chunks'), "Missing load_chunks method"
        
        print("✅ ChunkStore methods exist")
        return True
    except Exception as e:
        print(f"❌ ChunkStore test failed: {e}")
        return False

def test_advanced_scoring_methods():
    """Test that AdvancedChunkScorer has required methods"""
    try:
        from RAG.ingestion.advanced_scoring import AdvancedChunkScorer
        
        # Create scorer with default config
        scorer = AdvancedChunkScorer()
        
        # Check if methods exist
        assert hasattr(scorer, 'score_chunk_advanced'), "Missing score_chunk_advanced method"
        
        print("✅ AdvancedChunkScorer methods exist")
        return True
    except Exception as e:
        print(f"❌ AdvancedChunkScorer test failed: {e}")
        return False

def test_text_splitting():
    """Test text splitting directly"""
    try:
        from RAG.ingestion.text_splitter import SemanticStructuralSplitter
        from RAG.retrieval.retrieval_config import RetrievalConfig, SplittingConfig
        
        # Use more lenient splitting config
        splitting_config = SplittingConfig(
            target_chunk_size=100,
            max_chunk_size=200,
            min_chunk_size=20,  # Much lower minimum
            chunk_overlap=10,
            respect_sentence_boundaries=True,
            respect_section_boundaries=False,  # Less strict
            preserve_structure=False  # Less strict
        )
        
        splitter = SemanticStructuralSplitter(splitting_config)
        
        test_text = """Machine learning is a powerful subset of artificial intelligence. It focuses on algorithms that can learn and improve from experience without being explicitly programmed. 

The main types include supervised learning, unsupervised learning, and reinforcement learning. Each type has different applications and strengths.

Supervised learning uses labeled data to train models. Common algorithms include linear regression, decision trees, and neural networks. These methods are widely used in classification and prediction tasks.

Unsupervised learning finds patterns in unlabeled data. Clustering and dimensionality reduction are popular techniques. These methods help discover hidden structures in complex datasets."""
        
        test_metadata = {
            'doc_id': 'test_doc',
            'source_path': 'test.txt',
            'doc_type': 'txt'
        }
        
        chunks = splitter.split_document(test_text, test_metadata, None)
        print(f"✅ Text splitting: created {len(chunks)} chunks")
        
        if chunks:
            print(f"   First chunk: {chunks[0].text[:100]}...")
            print(f"   Chunk lengths: {[len(c.text) for c in chunks]}")
        else:
            print(f"   Text length: {len(test_text)} characters")
            print(f"   Config: min={splitting_config.min_chunk_size}, target={splitting_config.target_chunk_size}")
            
        return len(chunks) > 0
        
    except Exception as e:
        print(f"❌ Text splitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_processing():
    """Test simple document processing"""
    try:
        from RAG import create_basic_pipeline
        
        # Create test file with substantial content
        test_content = """Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn and improve from experience without being explicitly programmed. This field has revolutionized many industries and continues to grow rapidly.

The main types of machine learning include supervised learning, unsupervised learning, and reinforcement learning. Each type has its own strengths and applications.

Supervised Learning
Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. Common algorithms include linear regression for continuous predictions, decision trees for classification and regression, neural networks for complex pattern recognition, and support vector machines for classification tasks.

The training process involves feeding the algorithm input-output pairs so it can learn to map inputs to correct outputs. This process requires careful validation to ensure the model generalizes well to new data.

Unsupervised Learning
Unsupervised learning finds hidden patterns in data without labeled examples. Key techniques include clustering algorithms like k-means, dimensionality reduction methods like PCA, and association rule learning for market basket analysis.

These methods are particularly useful for exploratory data analysis and discovering underlying structures in complex datasets.

Applications
Machine learning is used in numerous applications including recommendation systems for e-commerce, fraud detection in financial services, medical diagnosis and drug discovery, autonomous vehicles for transportation, and natural language processing for human-computer interaction.

The field continues to evolve with new architectures, algorithms, and applications being developed regularly."""
        test_file = "simple_test_doc.txt"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Create pipeline with all advanced features disabled and lenient config
        from RAG import EnhancedIngestionPipeline
        from RAG.retrieval.retrieval_config import SplittingConfig
        
        # Create a temporary config file with lenient settings
        import tempfile
        import yaml
        
        lenient_config = {
            'splitting': {
                'target_chunk_size': 100,
                'max_chunk_size': 200, 
                'min_chunk_size': 20,
                'chunk_overlap': 10,
                'respect_sentence_boundaries': True,
                'respect_section_boundaries': False,
                'preserve_structure': False
            }
        }
        
        config_file = "lenient_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(lenient_config, f)
        
        pipeline = EnhancedIngestionPipeline(
            config_path=config_file,
            storage_dir="simple_processing_test",
            enable_advanced_scoring=False,
            enable_caching=False,
            enable_versioning=False,
            enable_telemetry=False,
            enable_incremental=False,
            enable_adaptive_retrieval=False,
            enable_answer_guardrails=False,
            enable_per_doc_management=False
        )
        
        print("✅ Simple pipeline created")
        
        # Try to process document
        results = pipeline.process_documents([test_file])
        print(f"✅ Document processing completed: {results['documents_processed']} docs, {results['total_chunks']} chunks")
        
        # Test retrieval if chunks were created
        if results['total_chunks'] > 0:
            retriever = pipeline.build_retrieval_system()
            search_results = retriever.retrieve("machine learning", top_k=3)
            print(f"✅ Retrieval test: found {len(search_results)} results")
        else:
            print("⚠️ No chunks created - retrieval test skipped")
        
        # Cleanup
        os.remove(test_file)
        if os.path.exists(config_file):
            os.remove(config_file)
        if os.path.exists("simple_processing_test"):
            shutil.rmtree("simple_processing_test", ignore_errors=True)
        
        return results['total_chunks'] > 0  # Only pass if chunks were actually created
        
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        # Cleanup on error
        if os.path.exists("simple_test_doc.txt"):
            os.remove("simple_test_doc.txt")
        if os.path.exists("lenient_config.yaml"):
            os.remove("lenient_config.yaml")
        if os.path.exists("simple_processing_test"):
            shutil.rmtree("simple_processing_test", ignore_errors=True)
        return False

def main():
    """Run simple tests"""
    print("🧪 Running Simple RAG System Tests")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Pipeline Creation", test_pipeline_creation),
        ("ChunkStore Methods", test_chunk_store_methods),
        ("AdvancedScoring Methods", test_advanced_scoring_methods),
        ("Text Splitting", test_text_splitting),
        ("Document Processing", test_document_processing)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Core functionality is working.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    # Cleanup
    cleanup_patterns = ["test_simple", "test_chunks", "simple_processing_test"]
    for pattern in cleanup_patterns:
        if os.path.exists(pattern):
            if os.path.isdir(pattern):
                shutil.rmtree(pattern, ignore_errors=True)
            else:
                os.remove(pattern)

if __name__ == "__main__":
    main() 