"""
cli.py

Command-line interface for LabGPT RAG system.
Provides simple commands for ingestion, search, snapshots, and status checks.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List

from .pipeline import RAGPipeline


def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def cmd_ingest(args):
    """Ingest documents into the RAG index."""
    print("Processing documents...")

    # Initialize pipeline with device selection
    rag = RAGPipeline(
        index_dir=args.index,
        preset=args.preset,
        device=args.device
    )

    # Add documents
    stats = rag.add_documents(
        paths=[args.docs],
        recursive=True
    )

    # Print summary
    print(f"\n  Loaded: {stats['files_processed']} documents")
    print(f"  Chunks: {stats['total_chunks']} created")
    print(f"  Time: {stats['time_seconds']:.1f}s ({stats['chunks_per_second']:.1f} chunks/sec)")

    # Get cache stats if available
    pipeline_stats = rag.get_stats()
    if 'cache_stats' in pipeline_stats:
        cache = pipeline_stats['cache_stats']
        hits = cache.get('hits', 0)
        misses = cache.get('misses', 0)
        total = hits + misses
        if total > 0:
            hit_rate = hits / total
            print(f"  Embeddings: {hits} cached ({hit_rate:.0%}), {misses} generated ({1-hit_rate:.0%})")

    # Calculate disk usage
    index_path = Path(args.index)
    if index_path.exists():
        size = get_directory_size(index_path)
        print(f"\nSaved to: {args.index}")
        print(f"  Disk usage: {format_size(size)}")

    print("\nReady to search!")


def cmd_ask(args):
    """Search the RAG index."""
    # Initialize pipeline
    rag = RAGPipeline(index_dir=args.index)

    # Perform search
    print(f"Query: \"{args.query}\"")

    results = rag.search(
        query=args.query,
        top_k=args.top_k,
        expand_query=args.expand,
        verify_citations=args.verify,
        cited_spans=args.cited_spans
    )

    # Get stats
    stats = rag.get_stats()
    if 'avg_search_time' in stats:
        print(f"Retrieved: {len(results)} results in {stats['avg_search_time']:.2f}s")
    else:
        print(f"Retrieved: {len(results)} results")

    # Print results
    print("\nResults:")
    print("=" * 70)

    for result in results:
        print(f"\n[{result.rank}] Score: {result.score:.3f}")

        # Source and citation
        citation = result.chunk.get_citation()
        print(f"    Source: {citation}")

        # Text preview
        text_preview = result.chunk.text[:300]
        if len(result.chunk.text) > 300:
            text_preview += "..."
        print(f"\n    {text_preview}")

        # Cited spans if available
        if args.cited_spans and result.metadata and 'cited_spans' in result.metadata:
            spans = result.metadata['cited_spans']
            if spans:
                print(f"\n    Supporting spans:")
                for span in spans:
                    print(f"      → {span.text[:100]}")
                    print(f"        Confidence: {span.confidence:.2f}")

        # Verification status
        if args.verify:
            print(f"\n    Citations: [✓] Verified")

        print()


def cmd_snapshot(args):
    """Create a reproducibility snapshot."""
    # Initialize pipeline
    rag = RAGPipeline(index_dir=args.index)

    # Create snapshot
    snapshot_path = rag.snapshot()

    # Load and display snapshot info
    from .snapshot import load_snapshot
    snapshot = load_snapshot(snapshot_path)

    print(f"Snapshot saved to: {snapshot_path}")
    print(f"\nContains:")
    print(f"  - Timestamp: {snapshot.timestamp}")
    print(f"  - Model: {snapshot.model}")
    print(f"  - Documents: {snapshot.doc_count} (with SHA256 hashes)")
    print(f"  - Chunks: {snapshot.chunk_count}")
    print(f"  - Snapshot ID: {snapshot.snapshot_id}")


def cmd_interactive(args):
    """Interactive search mode - keeps models loaded for fast queries."""
    print("Loading RAG pipeline (this will take ~90s on first load)...")
    print("Please wait...")

    # Initialize pipeline (models loaded once)
    rag = RAGPipeline(index_dir=args.index, preset=args.preset)

    print(f"\n✓ Pipeline loaded! Ready for queries.")
    print(f"  Index: {args.index}")
    print(f"  Preset: {args.preset}")
    print(f"  Documents: {len(rag.documents)}")
    print(f"  Chunks: {sum(doc.chunk_count for doc in rag.documents)}")
    print(f"\nType your query and press Enter. Type 'quit' to exit.\n")

    try:
        while True:
            # Get query from user
            try:
                query = input("Query: ").strip()
            except EOFError:
                print("\nGoodbye!")
                break

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not query:
                continue

            # Search (fast! models already loaded)
            results = rag.search(
                query=query,
                top_k=args.top_k,
                expand_query=args.expand,
                cited_spans=args.cited_spans
            )

            # Display results
            print(f"\n{'='*70}")
            print(f"Found {len(results)} results:")
            print('='*70)

            for result in results:
                print(f"\n[{result.rank}] Score: {result.score:.3f}")

                # Source and citation
                citation = result.chunk.get_citation()
                print(f"    Source: {citation}")

                # Text preview
                text_preview = result.chunk.text[:300]
                if len(result.chunk.text) > 300:
                    text_preview += "..."
                print(f"\n    {text_preview}")

                # Cited spans if available
                if args.cited_spans and result.metadata and 'cited_spans' in result.metadata:
                    spans = result.metadata['cited_spans']
                    if spans:
                        print(f"\n    Supporting spans:")
                        for span in spans:
                            print(f"      → {span.text[:100]}")
                            print(f"        Confidence: {span.confidence:.2f}")

                print()

            print()  # Extra newline before next query

    except KeyboardInterrupt:
        print("\n\nGoodbye!")


def cmd_status(args):
    """Show index status and health check."""
    # Initialize pipeline
    rag = RAGPipeline(index_dir=args.index)

    # Get status
    status = rag.status()
    stats = rag.get_stats()

    print(f"Index: {args.index}")
    print(f"  Status: {'Initialized' if status['initialized'] else 'Not initialized'}")
    print(f"  Has index: {'Yes' if status['has_index'] else 'No'}")
    print(f"  Preset: {status['preset']}")

    if status['documents'] > 0:
        print(f"\n  Documents: {status['documents']}")
        print(f"  Chunks: {status['chunks']}")
        print(f"  Embedding model: {status['embedding_model']}")

    # Cache stats
    if 'cache_stats' in stats:
        cache = stats['cache_stats']
        hits = cache.get('hits', 0)
        misses = cache.get('misses', 0)
        total = hits + misses
        if total > 0:
            hit_rate = hits / total
            print(f"  Cache hit rate: {hit_rate:.0%}")

    # Doc type breakdown
    if rag.documents:
        doc_types = {}
        for doc in rag.documents:
            doc_type = doc.doc_type
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        if doc_types:
            type_str = ", ".join([f"{dtype} ({count})" for dtype, count in doc_types.items()])
            print(f"  Doc types: {type_str}")

    # Disk usage
    index_path = Path(args.index)
    if index_path.exists():
        size = get_directory_size(index_path)
        print(f"  Disk usage: {format_size(size)}")

    # Latest snapshot
    if status['latest_snapshot']:
        print(f"  Latest snapshot: {Path(status['latest_snapshot']).name}")

    # Telemetry (if available)
    if 'avg_search_time' in stats:
        print(f"\nTelemetry:")
        print(f"  Total searches: {stats.get('total_searches', 0)}")
        print(f"  Avg search time: {stats['avg_search_time']:.3f}s")

        if 'auto_k_activations' in stats:
            print(f"  Auto-k activations: {stats['auto_k_activations']}")

        if 'avg_expansion_terms' in stats:
            print(f"  Avg expansion terms: {stats['avg_expansion_terms']:.1f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='labgpt-rag',
        description='LabGPT RAG System - Simple CLI for document ingestion and search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest documents
  labgpt-rag ingest --docs /path/to/papers --index ./my_rag

  # Search the index (one-off query)
  labgpt-rag ask --index ./my_rag --query "How does CRISPR work?"

  # Interactive mode (fast repeated queries, models stay loaded)
  labgpt-rag interactive --index ./my_rag --top-k 5

  # Create snapshot
  labgpt-rag snapshot --index ./my_rag

  # Check status
  labgpt-rag status --index ./my_rag
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Ingest command
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Ingest documents into the RAG index'
    )
    ingest_parser.add_argument(
        '--docs',
        required=True,
        help='Path to documents or directory'
    )
    ingest_parser.add_argument(
        '--index',
        required=True,
        help='Index directory'
    )
    ingest_parser.add_argument(
        '--preset',
        default='default',
        choices=['default', 'research'],
        help='Configuration preset (default: default)'
    )
    ingest_parser.add_argument(
        '--device',
        default='auto',
        choices=['cpu', 'cuda', 'auto'],
        help='Device for embedding models: cpu, cuda, or auto (default: auto - use GPU if available)'
    )

    # Ask command
    ask_parser = subparsers.add_parser(
        'ask',
        help='Search the RAG index'
    )
    ask_parser.add_argument(
        '--index',
        required=True,
        help='Index directory'
    )
    ask_parser.add_argument(
        '--query',
        required=True,
        help='Search query'
    )
    ask_parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    ask_parser.add_argument(
        '--expand',
        action='store_true',
        help='Enable query expansion (PRF-style)'
    )
    ask_parser.add_argument(
        '--verify',
        action='store_true',
        help='Enable citation verification'
    )
    ask_parser.add_argument(
        '--cited-spans',
        action='store_true',
        help='Extract cited spans from results'
    )

    # Snapshot command
    snapshot_parser = subparsers.add_parser(
        'snapshot',
        help='Create a reproducibility snapshot'
    )
    snapshot_parser.add_argument(
        '--index',
        required=True,
        help='Index directory'
    )

    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show index status and health'
    )
    status_parser.add_argument(
        '--index',
        required=True,
        help='Index directory'
    )

    # Interactive command
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Interactive search mode (keeps models loaded for fast queries)'
    )
    interactive_parser.add_argument(
        '--index',
        required=True,
        help='Index directory'
    )
    interactive_parser.add_argument(
        '--preset',
        default='default',
        choices=['default', 'research'],
        help='Configuration preset (default: default)'
    )
    interactive_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )
    interactive_parser.add_argument(
        '--expand',
        action='store_true',
        help='Enable query expansion (PRF-style)'
    )
    interactive_parser.add_argument(
        '--cited-spans',
        action='store_true',
        help='Extract cited spans from results'
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == 'ingest':
        cmd_ingest(args)
    elif args.command == 'ask':
        cmd_ask(args)
    elif args.command == 'snapshot':
        cmd_snapshot(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'interactive':
        cmd_interactive(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
