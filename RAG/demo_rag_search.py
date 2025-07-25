#!/usr/bin/env python3
"""
demo_rag_search.py

Demonstration script for using the generated RAG files to perform semantic search
on processed lab documents.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
import os
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)

class RAGSearcher:
    """Simple RAG search system using generated files"""
    
    def __init__(self, 
                 chunks_file: str = "chunks.npy",
                 embeddings_file: str = "embeddings.npy", 
                 index_file: str = "faiss_index.bin",
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        
        self.chunks_file = chunks_file
        self.embeddings_file = embeddings_file
        self.index_file = index_file
        self.model_name = model_name
        
        self.chunks = None
        self.embeddings = None
        self.index = None
        self.model = None
        
    def load_rag_system(self):
        """Load all RAG components"""
        logging.info("Loading RAG system...")
        
        # Check if files exist
        for file_path in [self.chunks_file, self.embeddings_file, self.index_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load chunks
        logging.info(f"Loading chunks from {self.chunks_file}")
        self.chunks = np.load(self.chunks_file, allow_pickle=True)
        logging.info(f"Loaded {len(self.chunks)} text chunks")
        
        # Load embeddings
        logging.info(f"Loading embeddings from {self.embeddings_file}")
        self.embeddings = np.load(self.embeddings_file)
        logging.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
        
        # Load FAISS index
        logging.info(f"Loading FAISS index from {self.index_file}")
        self.index = faiss.read_index(self.index_file)
        logging.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Initialize embedding model
        logging.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        logging.info("RAG system loaded successfully!")
        
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform semantic search on the document collection
        
        Args:
            query: Search query string
            k: Number of top results to return
            
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        if not all([self.chunks is not None, self.index is not None, self.model is not None]):
            raise RuntimeError("RAG system not loaded. Call load_rag_system() first.")
        
        logging.info(f"Searching for: '{query}'")
        
        # Encode the query
        query_embedding = self.model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):  # Ensure valid index
                chunk = self.chunks[idx]
                # Convert distance to similarity score (higher is better)
                similarity = 1.0 / (1.0 + distance)
                results.append((chunk, similarity))
                logging.info(f"Result {i+1}: Similarity={similarity:.3f}")
        
        return results
    
    def get_stats(self) -> dict:
        """Get statistics about the loaded RAG system"""
        if self.chunks is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "num_chunks": len(self.chunks),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else None,
            "index_size": self.index.ntotal if self.index is not None else None,
            "model_name": self.model_name
        }

def main():
    """Demo the RAG search functionality"""
    print("=== LabGPT RAG Search Demo ===\n")
    
    # Initialize the searcher
    searcher = RAGSearcher()
    
    try:
        # Load the RAG system
        searcher.load_rag_system()
        
        # Show system stats
        stats = searcher.get_stats()
        print(f"System Status: {stats['status']}")
        print(f"Number of chunks: {stats['num_chunks']}")
        print(f"Embedding dimension: {stats['embedding_dim']}")
        print(f"Index size: {stats['index_size']}")
        print(f"Model: {stats['model_name']}\n")
        
        # Demo queries
        demo_queries = [
            "machine learning algorithms",
            "protein structure prediction",
            "experimental methodology",
            "statistical analysis",
            "research objectives"
        ]
        
        print("=== Demo Searches ===\n")
        
        for query in demo_queries:
            print(f"Query: '{query}'")
            print("-" * 50)
            
            try:
                results = searcher.search(query, k=3)
                
                if results:
                    for i, (chunk, score) in enumerate(results, 1):
                        print(f"Result {i} (Score: {score:.3f}):")
                        # Show first 200 characters of the chunk
                        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                        print(f"  {preview}")
                        print()
                else:
                    print("  No results found")
                    
            except Exception as e:
                print(f"  Error searching: {e}")
            
            print("=" * 60)
            print()
        
        # Interactive search
        print("=== Interactive Search ===")
        print("Enter your search queries (or 'quit' to exit):\n")
        
        while True:
            try:
                query = input("Search: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                results = searcher.search(query, k=5)
                
                if results:
                    print(f"\nFound {len(results)} results:")
                    for i, (chunk, score) in enumerate(results, 1):
                        print(f"\nResult {i} (Score: {score:.3f}):")
                        preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                        print(f"{preview}")
                else:
                    print("No results found.")
                    
                print("\n" + "=" * 60 + "\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Search error: {e}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you've run the RAG processing pipeline first!")
        print("Use the web interface at http://localhost:5001 to process your documents.")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        logging.error(f"Demo error: {e}")
    
    print("\nDemo completed. Thank you for using LabGPT RAG!")

if __name__ == "__main__":
    main() 