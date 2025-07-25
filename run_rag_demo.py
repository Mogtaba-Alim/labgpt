#!/usr/bin/env python3
"""
run_rag_demo.py

Complete Enhanced RAG System Demonstration Script

This script demonstrates the full RAG pipeline:
1. Loads documents from a specified directory
2. Processes them through the enhanced ingestion pipeline
3. Builds a hybrid retrieval system
4. Provides interactive querying with detailed results

Usage:
    python run_rag_demo.py /path/to/documents
    python run_rag_demo.py /path/to/documents --advanced
    python run_rag_demo.py /path/to/documents --config custom_config.yaml
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_rag_system(documents_dir: str, 
                     storage_dir: str = "rag_demo_storage",
                     config_path: str = None,
                     use_advanced: bool = False) -> tuple:
    """
    Set up the RAG system with documents from the specified directory
    
    Args:
        documents_dir: Path to directory containing documents
        storage_dir: Directory for RAG system storage
        config_path: Optional path to custom configuration file
        use_advanced: Whether to use advanced features
        
    Returns:
        Tuple of (pipeline, retriever, processing_results)
    """
    try:
        # Import RAG components
        from RAG import create_basic_pipeline, create_production_pipeline
        
        print("🚀 Initializing Enhanced RAG System...")
        print(f"📂 Documents directory: {documents_dir}")
        print(f"💾 Storage directory: {storage_dir}")
        
        # Create pipeline based on configuration
        if config_path or use_advanced:
            print("🔧 Creating enhanced pipeline with custom configuration...")
            from RAG import EnhancedIngestionPipeline
            
            pipeline = EnhancedIngestionPipeline(
                config_path=config_path,
                storage_dir=storage_dir,
                enable_advanced_scoring=use_advanced,
                enable_caching=use_advanced,
                enable_versioning=use_advanced,
                enable_telemetry=use_advanced,
                enable_incremental=use_advanced,
                enable_adaptive_retrieval=use_advanced,
                enable_answer_guardrails=use_advanced,
                enable_per_doc_management=use_advanced
            )
            
            if use_advanced:
                # Show enabled features
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
                
                enabled_features = [name for name, enabled in features if enabled]
                print(f"✅ Advanced features enabled: {', '.join(enabled_features)}")
            
        else:
            print("🔧 Creating basic pipeline...")
            pipeline = create_basic_pipeline(storage_dir=storage_dir)
        
        print(f"🤖 Model: {pipeline.embedding_model_name}")
        print(f"🖥️ Device: {pipeline.embedding_model.device}")
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        raise

def find_documents(documents_dir: str) -> List[str]:
    """
    Find all supported documents in the directory
    
    Args:
        documents_dir: Directory to search for documents
        
    Returns:
        List of document file paths
    """
    supported_extensions = {'.pdf', '.txt', '.md', '.tex', '.rst'}
    documents = []
    
    documents_path = Path(documents_dir)
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    
    print(f"🔍 Searching for documents in {documents_dir}...")
    
    for file_path in documents_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            documents.append(str(file_path))
    
    if not documents:
        raise ValueError(f"No supported documents found in {documents_dir}")
    
    print(f"📄 Found {len(documents)} documents:")
    for doc in documents:
        print(f"   - {Path(doc).name} ({Path(doc).suffix})")
    
    return documents

def process_documents(pipeline, documents: List[str]) -> Dict[str, Any]:
    """
    Process documents through the RAG pipeline
    
    Args:
        pipeline: RAG pipeline instance
        documents: List of document paths
        
    Returns:
        Processing results dictionary
    """
    print(f"\n📊 Processing {len(documents)} documents...")
    start_time = time.time()
    
    try:
        results = pipeline.process_documents(documents)
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.2f}s")
        print(f"📄 Documents processed: {results['documents_processed']}")
        print(f"📝 Chunks created: {results['total_chunks']}")
        
        if results['total_chunks'] == 0:
            print("⚠️ Warning: No chunks were created. This might indicate:")
            print("   - Documents are too short for current chunk size settings")
            print("   - Content doesn't meet quality thresholds")
            print("   - Try using --advanced flag for better text processing")
        
        return results
        
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        raise

def build_retrieval_system(pipeline):
    """
    Build the retrieval system
    
    Args:
        pipeline: RAG pipeline instance
        
    Returns:
        Retriever instance
    """
    print("\n🔗 Building retrieval system...")
    
    try:
        retriever = pipeline.build_retrieval_system()
        print("✅ Retrieval system built successfully")
        
        # Test if advanced features are available
        if hasattr(pipeline, 'enable_adaptive_retrieval') and pipeline.enable_adaptive_retrieval:
            adaptive_retriever = pipeline.create_adaptive_retriever(retriever)
            if adaptive_retriever:
                print("🎯 Adaptive retrieval system enabled")
                return adaptive_retriever, True
        
        return retriever, False
        
    except Exception as e:
        print(f"❌ Failed to build retrieval system: {e}")
        raise

def format_chunk_result(chunk_result, index: int) -> str:
    """
    Format a chunk result for display
    
    Args:
        chunk_result: Chunk result object
        index: Result index
        
    Returns:
        Formatted string
    """
    chunk = chunk_result.chunk
    score = chunk_result.score
    
    # Get retrieval method if available
    method = getattr(chunk_result, 'retrieval_method', 'hybrid')
    
    # Format the result
    result_str = f"\n{'='*60}\n"
    result_str += f"📄 Result #{index}\n"
    result_str += f"{'='*60}\n"
    result_str += f"🎯 Score: {score:.4f} (Method: {method})\n"
    result_str += f"📊 Quality: {chunk.quality_score:.3f}\n"
    result_str += f"📁 Source: {Path(chunk.source_path).name}\n"
    
    if chunk.doc_type:
        result_str += f"📋 Type: {chunk.doc_type}\n"
    
    if chunk.section:
        result_str += f"📑 Section: {chunk.section}\n"
        
    if chunk.page_number:
        result_str += f"📖 Page: {chunk.page_number}\n"
    
    result_str += f"📏 Length: {chunk.char_count} chars, {chunk.token_count} tokens\n"
    
    if chunk.hierarchy_path:
        result_str += f"🗂️ Hierarchy: {' → '.join(chunk.hierarchy_path)}\n"
    
    result_str += f"\n📖 Content:\n"
    result_str += f"{'-'*60}\n"
    result_str += f"{chunk.text}\n"
    result_str += f"{'-'*60}"
    
    return result_str

def interactive_query_loop(retriever, pipeline, is_adaptive: bool = False):
    """
    Run interactive query loop
    
    Args:
        retriever: Retriever instance
        pipeline: Pipeline instance
        is_adaptive: Whether using adaptive retrieval
    """
    print(f"\n🎪 Interactive RAG Query Interface")
    print(f"{'='*60}")
    print(f"🔍 Enter queries to search through your documents")
    print(f"💡 Type 'help' for commands, 'quit' to exit")
    
    if is_adaptive:
        print(f"🎯 Using adaptive retrieval (dynamic top-k)")
    
    print(f"{'='*60}")
    
    while True:
        try:
            # Get query from user
            query = input("\n🔍 Query: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if query.lower() == 'help':
                print("\n📖 Available commands:")
                print("  • Enter any text to search")
                print("  • 'stats' - Show system statistics")
                print("  • 'help' - Show this help")
                print("  • 'quit' - Exit the program")
                continue
                
            if query.lower() == 'stats':
                print("\n📊 System Statistics:")
                try:
                    stats = pipeline.get_system_statistics()
                    for category, data in stats.items():
                        print(f"\n{category}:")
                        if isinstance(data, dict):
                            for key, value in data.items():
                                print(f"  {key}: {value}")
                        else:
                            print(f"  {data}")
                except Exception as e:
                    print(f"⚠️ Could not retrieve stats: {e}")
                continue
            
            # Perform search
            print(f"\n🔍 Searching for: '{query}'")
            start_time = time.time()
            
            if is_adaptive:
                # Use adaptive retrieval
                results, coverage_metrics = retriever.retrieve_adaptive(query)
                search_time = time.time() - start_time
                
                print(f"⏱️ Search completed in {search_time:.3f}s")
                print(f"📊 Found {len(results)} results")
                print(f"📈 Coverage score: {coverage_metrics.overall_coverage:.3f}")
                print(f"🎯 Semantic diversity: {coverage_metrics.semantic_diversity:.3f}")
                
            else:
                # Use standard retrieval
                results = retriever.retrieve(query, top_k=5)
                search_time = time.time() - start_time
                
                print(f"⏱️ Search completed in {search_time:.3f}s")
                print(f"📊 Found {len(results)} results")
            
            if not results:
                print("❌ No results found. Try:")
                print("   • Different keywords")
                print("   • Broader search terms")
                print("   • Check if documents were processed correctly")
                continue
            
            # Display results
            for i, result in enumerate(results, 1):
                print(format_chunk_result(result, i))
            
            # Ask if user wants to see answer guardrails (if available)
            if hasattr(pipeline, 'enable_answer_guardrails') and pipeline.enable_answer_guardrails:
                verify_answer = input(f"\n🛡️ Enter an answer to verify with guardrails (or press Enter to skip): ").strip()
                if verify_answer:
                    try:
                        chunks = [r.chunk for r in results]
                        verification = pipeline.verify_answer(verify_answer, chunks)
                        if verification:
                            print(f"\n🛡️ Answer Verification:")
                            print(f"   Status: {verification['verification_status']}")
                            print(f"   Claims processed: {len(verification.get('verification_results', []))}")
                    except Exception as e:
                        print(f"⚠️ Answer verification failed: {e}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during search: {e}")
            continue

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Enhanced RAG System Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_rag_demo.py ./documents
    python run_rag_demo.py ./papers --advanced
    python run_rag_demo.py ./docs --config custom.yaml --storage ./my_rag
        """
    )
    
    parser.add_argument(
        'documents_dir',
        help='Directory containing documents to process'
    )
    
    parser.add_argument(
        '--advanced',
        action='store_true',
        help='Use production pipeline with all advanced features'
    )
    
    parser.add_argument(
        '--config',
        help='Path to custom configuration YAML file'
    )
    
    parser.add_argument(
        '--storage',
        default='rag_demo_storage',
        help='Directory for RAG system storage (default: rag_demo_storage)'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild of the index (ignore existing storage)'
    )
    
    args = parser.parse_args()
    
    try:
        # Clean up storage if rebuild requested
        if args.rebuild and os.path.exists(args.storage):
            import shutil
            print(f"🗑️ Removing existing storage: {args.storage}")
            shutil.rmtree(args.storage)
        
        # Find documents
        documents = find_documents(args.documents_dir)
        
        # Setup RAG system
        pipeline = setup_rag_system(
            documents_dir=args.documents_dir,
            storage_dir=args.storage,
            config_path=args.config,
            use_advanced=args.advanced
        )
        
        # Process documents
        processing_results = process_documents(pipeline, documents)
        
        if processing_results['total_chunks'] == 0:
            print("\n❌ No chunks were created. Cannot proceed with retrieval.")
            print("💡 Try using the --advanced flag or check your documents.")
            return 1
        
        # Build retrieval system
        retriever, is_adaptive = build_retrieval_system(pipeline)
        
        # Run interactive query loop
        interactive_query_loop(retriever, pipeline, is_adaptive)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        logger.exception("Fatal error occurred")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 