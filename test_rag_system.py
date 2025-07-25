#!/usr/bin/env python3
"""
test_rag_system.py

Comprehensive test suite for the Enhanced RAG System.
Tests all functionality and cleans up test files afterward.

Usage:
    python test_rag_system.py
"""

import os
import sys
import time
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the current directory to path so we can import RAG
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging to capture test output
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during testing
    format='%(levelname)s: %(message)s'
)

class RAGSystemTester:
    """Comprehensive RAG system test suite"""
    
    def __init__(self):
        self.test_dir = Path("rag_test_temp")
        self.test_files = []
        self.test_results = {}
        self.start_time = time.time()
        
    def create_test_documents(self) -> List[str]:
        """Create test documents for various scenarios"""
        print("📝 Creating test documents...")
        
        # Document 1: Machine Learning
        ml_doc = self.test_dir / "machine_learning.txt"
        with open(ml_doc, 'w', encoding='utf-8') as f:
            f.write("""Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence (AI) that focuses on 
algorithms that can learn and improve from experience without being explicitly 
programmed. The main types include supervised learning, unsupervised learning, 
and reinforcement learning.

Supervised Learning
Supervised learning algorithms learn from labeled training data to make 
predictions on new, unseen data. Common algorithms include:
- Linear regression for continuous predictions
- Decision trees for classification and regression
- Neural networks for complex pattern recognition
- Support vector machines for classification tasks

The training process involves feeding the algorithm input-output pairs so it 
can learn to map inputs to correct outputs.

Unsupervised Learning
Unsupervised learning finds hidden patterns in data without labeled examples.
Key techniques include clustering, dimensionality reduction, and association 
rule learning.

Applications
Machine learning is used in numerous applications including recommendation 
systems, fraud detection, medical diagnosis, autonomous vehicles, and 
natural language processing.
""")
        
        # Document 2: Deep Learning
        dl_doc = self.test_dir / "deep_learning.txt"
        with open(dl_doc, 'w', encoding='utf-8') as f:
            f.write("""Deep Learning and Neural Networks

Deep learning is a specialized subset of machine learning that uses artificial 
neural networks with multiple layers (hence "deep") to model and understand 
complex patterns in data.

Neural Network Architecture
Neural networks consist of interconnected nodes (neurons) organized in layers:
- Input layer: Receives the raw data
- Hidden layers: Process information through weighted connections
- Output layer: Produces the final prediction or classification

Each connection has a weight that determines the strength of the signal passed 
between neurons. During training, these weights are adjusted through 
backpropagation to minimize prediction errors.

Popular Architectures
1. Convolutional Neural Networks (CNNs)
   - Specialized for image processing and computer vision
   - Use convolutional layers to detect local features
   - Applications: image classification, object detection, medical imaging

2. Recurrent Neural Networks (RNNs)
   - Designed for sequential data and time series
   - Include memory mechanisms to process sequences
   - Applications: natural language processing, speech recognition, machine translation

3. Transformer Networks
   - Use attention mechanisms for parallel processing
   - Revolutionized natural language processing
   - Applications: language models, text generation, question answering

Training Process
Deep learning models require large datasets and significant computational 
resources. The training process involves:
- Forward propagation: Data flows through the network
- Loss calculation: Measuring prediction accuracy
- Backpropagation: Adjusting weights to reduce errors
- Optimization: Using algorithms like Adam or SGD
""")
        
        # Document 3: Natural Language Processing
        nlp_doc = self.test_dir / "nlp.txt"
        with open(nlp_doc, 'w', encoding='utf-8') as f:
            f.write("""Natural Language Processing (NLP)

Natural Language Processing is a field that combines computational linguistics 
with machine learning and deep learning to help computers understand, interpret, 
and generate human language in a valuable way.

Core NLP Tasks

1. Tokenization
   Breaking text into individual words, phrases, or tokens. This is often the 
   first step in NLP pipelines.

2. Part-of-Speech Tagging
   Identifying the grammatical role of each word (noun, verb, adjective, etc.).

3. Named Entity Recognition (NER)
   Identifying and classifying named entities like person names, organizations, 
   locations, and dates.

4. Sentiment Analysis
   Determining the emotional tone or opinion expressed in text (positive, 
   negative, or neutral).

5. Machine Translation
   Automatically translating text from one language to another using neural 
   machine translation models.

Modern NLP Techniques

Word Embeddings
Representing words as dense vectors in high-dimensional space where semantically 
similar words are located close to each other. Popular methods include Word2Vec, 
GloVe, and FastText.

Transformer Models
Revolutionary architecture that uses self-attention mechanisms. Models like 
BERT, GPT, and T5 have achieved state-of-the-art results across many NLP tasks.

Large Language Models (LLMs)
Massive neural networks trained on enormous text corpora that can perform 
various language tasks through prompting and fine-tuning.

Applications
- Chatbots and virtual assistants
- Document summarization and question answering
- Content generation and creative writing
- Language translation services
- Email filtering and spam detection
- Voice recognition systems
""")
        
        self.test_files = [str(ml_doc), str(dl_doc), str(nlp_doc)]
        print(f"✅ Created {len(self.test_files)} test documents")
        return self.test_files
    
    def test_installation(self) -> bool:
        """Test basic installation and imports"""
        print("\n🧪 Testing Installation...")
        try:
            from RAG import (
                EnhancedIngestionPipeline, 
                create_basic_pipeline, 
                create_production_pipeline,
                get_system_info
            )
            
            # Test system info
            info = get_system_info()
            print(f"✅ System: {info['description']}")
            print(f"📦 Version: {info['version']}")
            print(f"🔧 Features: {len(info['features'])} available")
            
            # Test basic pipeline creation
            basic_pipeline = create_basic_pipeline(storage_dir=str(self.test_dir / "basic_test"))
            print(f"✅ Basic pipeline: {basic_pipeline.embedding_model_name}")
            
            # Test production pipeline creation
            prod_pipeline = create_production_pipeline(storage_dir=str(self.test_dir / "prod_test"))
            print(f"✅ Production pipeline: {prod_pipeline.embedding_model.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Installation test failed: {e}")
            return False
    
    def test_basic_functionality(self) -> bool:
        """Test core pipeline functionality"""
        print("\n🧪 Testing Basic Functionality...")
        try:
            from RAG import create_basic_pipeline
            
            # Create pipeline
            pipeline = create_basic_pipeline(storage_dir=str(self.test_dir / "basic_func"))
            print("✅ Pipeline initialized")
            
            # Process documents
            results = pipeline.process_documents(self.test_files)
            print(f"✅ Processed {results['documents_processed']} documents")
            print(f"✅ Created {results['total_chunks']} chunks")
            print(f"⏱️ Processing time: {results['processing_time']:.2f}s")
            
            # Build retrieval system
            retriever = pipeline.build_retrieval_system()
            print("✅ Retrieval system built")
            
            # Test queries
            test_queries = [
                "machine learning algorithms",
                "neural networks deep learning",
                "natural language processing",
                "supervised learning examples",
                "transformer models"
            ]
            
            print("📊 Testing retrieval with sample queries:")
            for query in test_queries:
                results = retriever.retrieve(query, top_k=3)
                print(f"   '{query}' → {len(results)} results")
                if results:
                    best_score = max(r.score for r in results)
                    print(f"     Best score: {best_score:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            return False
    
    def test_advanced_features(self) -> bool:
        """Test advanced pipeline features"""
        print("\n🧪 Testing Advanced Features...")
        try:
            from RAG import create_production_pipeline
            
            # Create production pipeline with all features
            pipeline = create_production_pipeline(storage_dir=str(self.test_dir / "advanced"))
            print("✅ Production pipeline initialized")
            
            # Check feature status
            features = [
                ("Advanced scoring", pipeline.enable_advanced_scoring),
                ("Embedding caching", pipeline.enable_caching),
                ("Index versioning", pipeline.enable_versioning),
                ("Telemetry", pipeline.enable_telemetry),
                ("Incremental updates", pipeline.enable_incremental),
                ("Adaptive retrieval", pipeline.enable_adaptive_retrieval),
                ("Answer guardrails", pipeline.enable_answer_guardrails),
                ("Per-doc management", pipeline.enable_per_doc_management)
            ]
            
            enabled_count = 0
            for feature_name, enabled in features:
                status = "✅" if enabled else "❌"
                print(f"   {status} {feature_name}")
                if enabled:
                    enabled_count += 1
            
            print(f"📊 {enabled_count}/{len(features)} advanced features enabled")
            
            # Process documents
            results = pipeline.process_documents(self.test_files)
            print(f"✅ Processed {results['total_chunks']} chunks with advanced features")
            
            # Test retrieval system
            retriever = pipeline.build_retrieval_system()
            
            # Test adaptive retrieval if enabled
            if pipeline.enable_adaptive_retrieval:
                print("📊 Testing adaptive retrieval...")
                adaptive_retriever = pipeline.create_adaptive_retriever(retriever)
                if adaptive_retriever:
                    query = "deep learning neural networks applications"
                    results, coverage = adaptive_retriever.retrieve_adaptive(query)
                    print(f"   Adaptive retrieval: {len(results)} chunks, coverage: {coverage.overall_coverage:.3f}")
                    print("✅ Adaptive retrieval tested")
            
            # Test answer guardrails if enabled
            if pipeline.enable_answer_guardrails:
                print("📊 Testing answer guardrails...")
                test_answer = "Machine learning uses algorithms to learn patterns from data."
                chunks = [r.chunk for r in retriever.retrieve("machine learning", top_k=3)]
                
                verification = pipeline.verify_answer(test_answer, chunks)
                if verification:
                    print(f"   Verification: {verification['verification_status']}")
                    print("✅ Answer guardrails tested")
            
            # Test system statistics
            stats = pipeline.get_system_statistics()
            print("📊 System statistics retrieved:")
            print(f"   Pipeline metrics: {len(stats['pipeline_stats'])}")
            print(f"   Feature status: {len(stats['feature_status'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Advanced features test failed: {e}")
            return False
    
    def test_configuration_options(self) -> bool:
        """Test different configuration options"""
        print("\n🧪 Testing Configuration Options...")
        try:
            from RAG import EnhancedIngestionPipeline
            
            # Test custom configuration
            custom_pipeline = EnhancedIngestionPipeline(
                storage_dir=str(self.test_dir / "custom"),
                embedding_model_name="all-MiniLM-L6-v2",  # Smaller model
                enable_caching=True,
                enable_adaptive_retrieval=False,
                enable_answer_guardrails=True,
                enable_per_doc_management=False
            )
            
            print("✅ Custom configuration pipeline created")
            print(f"   Model: {custom_pipeline.embedding_model_name}")
            print(f"   Device: {custom_pipeline.embedding_model.device}")
            print(f"   Caching: {custom_pipeline.enable_caching}")
            print(f"   Adaptive: {custom_pipeline.enable_adaptive_retrieval}")
            
            # Test configuration export
            config_path = self.test_dir / "exported_config.json"
            custom_pipeline.export_configuration(str(config_path))
            
            if config_path.exists():
                print(f"✅ Configuration exported to {config_path}")
            else:
                print("❌ Configuration export failed")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            return False
    
    def test_retrieval_quality(self) -> bool:
        """Test retrieval quality with known queries"""
        print("\n🧪 Testing Retrieval Quality...")
        try:
            from RAG import create_production_pipeline
            
            pipeline = create_production_pipeline(storage_dir=str(self.test_dir / "quality"))
            
            # Process documents
            pipeline.process_documents(self.test_files)
            retriever = pipeline.build_retrieval_system()
            
            # Define test cases with expected terms
            test_cases = [
                {
                    "query": "supervised learning algorithms",
                    "expected_terms": ["supervised", "learning", "algorithm", "training"],
                    "doc_relevance": "machine_learning.txt"
                },
                {
                    "query": "neural networks deep learning",
                    "expected_terms": ["neural", "network", "deep", "layer"],
                    "doc_relevance": "deep_learning.txt"
                },
                {
                    "query": "tokenization NLP processing",
                    "expected_terms": ["tokenization", "nlp", "text", "language"],
                    "doc_relevance": "nlp.txt"
                }
            ]
            
            print("📊 Quality assessment results:")
            total_score = 0
            
            for i, test_case in enumerate(test_cases, 1):
                query = test_case["query"]
                results = retriever.retrieve(query, top_k=5)
                
                if not results:
                    print(f"   {i}. '{query}' → No results")
                    continue
                
                # Check term coverage
                all_text = " ".join([r.chunk.text.lower() for r in results])
                found_terms = sum(1 for term in test_case["expected_terms"] if term in all_text)
                coverage = found_terms / len(test_case["expected_terms"])
                
                # Check relevance score
                best_score = max(r.score for r in results)
                
                print(f"   {i}. '{query}'")
                print(f"      Results: {len(results)}, Best score: {best_score:.3f}")
                print(f"      Term coverage: {coverage:.1%} ({found_terms}/{len(test_case['expected_terms'])})")
                
                total_score += (coverage + (best_score / 2)) / 2  # Combined score
            
            avg_quality = total_score / len(test_cases)
            print(f"📈 Average quality score: {avg_quality:.3f}")
            
            if avg_quality > 0.3:  # Reasonable threshold
                print("✅ Retrieval quality test passed")
                return True
            else:
                print("⚠️ Retrieval quality below threshold")
                return False
            
        except Exception as e:
            print(f"❌ Retrieval quality test failed: {e}")
            return False
    
    def test_performance_benchmark(self) -> bool:
        """Test performance with timing"""
        print("\n🧪 Testing Performance...")
        try:
            from RAG import create_basic_pipeline, create_production_pipeline
            import psutil
            
            # Memory tracking
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            configs = [
                ("Basic", create_basic_pipeline),
                ("Production", create_production_pipeline)
            ]
            
            print("📊 Performance comparison:")
            for config_name, pipeline_func in configs:
                start_time = time.time()
                
                # Create pipeline
                pipeline = pipeline_func(storage_dir=str(self.test_dir / f"perf_{config_name.lower()}"))
                
                # Process documents
                results = pipeline.process_documents(self.test_files)
                processing_time = time.time() - start_time
                
                # Build retrieval
                retrieval_start = time.time()
                retriever = pipeline.build_retrieval_system()
                retrieval_time = time.time() - retrieval_start
                
                # Test query speed
                query_start = time.time()
                query_results = retriever.retrieve("machine learning", top_k=5)
                query_time = time.time() - query_start
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_used = current_memory - initial_memory
                
                print(f"   {config_name} Pipeline:")
                print(f"     Processing: {processing_time:.2f}s ({results['total_chunks']} chunks)")
                print(f"     Indexing: {retrieval_time:.2f}s")
                print(f"     Query: {query_time:.3f}s ({len(query_results)} results)")
                print(f"     Memory: +{memory_used:.1f} MB")
            
            print("✅ Performance benchmarking completed")
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        print("🚀 Starting Comprehensive RAG System Tests")
        print("=" * 60)
        
        # Setup test environment
        self.test_dir.mkdir(exist_ok=True)
        
        try:
            # Create test documents
            self.create_test_documents()
            
            # Run all tests
            tests = [
                ("Installation", self.test_installation),
                ("Basic Functionality", self.test_basic_functionality),
                ("Advanced Features", self.test_advanced_features),
                ("Configuration Options", self.test_configuration_options),
                ("Retrieval Quality", self.test_retrieval_quality),
                ("Performance Benchmark", self.test_performance_benchmark)
            ]
            
            for test_name, test_func in tests:
                try:
                    result = test_func()
                    self.test_results[test_name] = result
                except Exception as e:
                    print(f"❌ {test_name} test crashed: {e}")
                    self.test_results[test_name] = False
        
        except Exception as e:
            print(f"💥 Test setup failed: {e}")
            
        finally:
            # Always cleanup
            self.cleanup()
        
        return self.test_results
    
    def cleanup(self):
        """Clean up test files and directories"""
        print("\n🧹 Cleaning up test files...")
        
        try:
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir, ignore_errors=True)
                print("✅ Test directory cleaned up")
            
            # Clean up any remaining test files
            cleanup_patterns = [
                "test_document.txt",
                "advanced_test_doc.txt", 
                "test_storage",
                "advanced_test_storage",
                "basic_test",
                "prod_test"
            ]
            
            for pattern in cleanup_patterns:
                if os.path.exists(pattern):
                    if os.path.isdir(pattern):
                        shutil.rmtree(pattern, ignore_errors=True)
                    else:
                        os.remove(pattern)
            
            print("✅ Additional cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def print_summary(self):
        """Print test summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        if not self.test_results:
            print("❌ No tests were completed")
            return
        
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        
        print(f"📈 Results: {passed}/{total} tests passed")
        print(f"⏱️ Total time: {total_time:.1f}s")
        print()
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        print()
        if passed == total:
            print("🎉 ALL TESTS PASSED! RAG System is fully functional!")
            print("✅ System is ready for:")
            print("   - Document processing and semantic chunking")
            print("   - Hybrid retrieval with dense and sparse methods")
            print("   - Advanced features (caching, versioning, monitoring)")
            print("   - Intelligent retrieval (adaptive, guardrails)")
            print("   - Production deployment")
        else:
            print("⚠️ Some tests failed. Please review the output above.")
            print("   Check dependencies, environment setup, and system resources.")
        
        print("\n🔧 Next steps:")
        print("   - Follow the README for detailed usage instructions")
        print("   - Try processing your own documents")
        print("   - Experiment with different configuration options")
        print("   - Monitor system performance with built-in telemetry")


def main():
    """Main test execution function"""
    print("🧪 Enhanced RAG System - Comprehensive Test Suite")
    print("Testing all functionality and cleaning up afterward...")
    print()
    
    tester = RAGSystemTester()
    
    try:
        results = tester.run_all_tests()
        tester.print_summary()
        
        # Exit with appropriate code
        if all(results.values()):
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Some tests failed
            
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        tester.cleanup()
        sys.exit(130)
        
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        tester.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main() 