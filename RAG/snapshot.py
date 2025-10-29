"""
snapshot.py

Reproducibility snapshot functionality for RAG indices.
Simple 10-line alternative to full versioning system.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from .models import SnapshotInfo


def create_snapshot(index_dir: str, model_name: str,
                   doc_paths: List[str], chunk_count: int) -> str:
    """
    Create a reproducibility snapshot of the current index.

    This captures essential information needed to verify that two indices
    are identical, without the complexity of full version control.

    Args:
        index_dir: Path to index directory
        model_name: Name of embedding model used
        doc_paths: List of paths to source documents
        chunk_count: Number of chunks in index

    Returns:
        Path to created snapshot file
    """
    # Calculate SHA256 hashes for all documents
    doc_hashes = {}
    for doc_path in doc_paths:
        try:
            with open(doc_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            doc_hashes[doc_path] = file_hash
        except Exception as e:
            doc_hashes[doc_path] = f"ERROR: {str(e)}"

    # Create snapshot info
    snapshot = SnapshotInfo.create(
        model=model_name,
        doc_count=len(doc_paths),
        chunk_count=chunk_count,
        doc_hashes=doc_hashes
    )

    # Save snapshot to file
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    snapshot_filename = f"snapshot_{int(time.time())}.json"
    snapshot_path = index_path / snapshot_filename

    snapshot_dict = {
        "snapshot_id": snapshot.snapshot_id,
        "timestamp": snapshot.timestamp,
        "model": snapshot.model,
        "doc_count": snapshot.doc_count,
        "chunk_count": snapshot.chunk_count,
        "doc_hashes": snapshot.doc_hashes
    }

    with open(snapshot_path, 'w') as f:
        json.dump(snapshot_dict, f, indent=2)

    return str(snapshot_path)


def load_snapshot(snapshot_path: str) -> SnapshotInfo:
    """
    Load a snapshot from file.

    Args:
        snapshot_path: Path to snapshot JSON file

    Returns:
        SnapshotInfo object
    """
    with open(snapshot_path, 'r') as f:
        data = json.load(f)

    return SnapshotInfo(
        timestamp=data["timestamp"],
        model=data["model"],
        doc_count=data["doc_count"],
        chunk_count=data["chunk_count"],
        doc_hashes=data["doc_hashes"],
        snapshot_id=data["snapshot_id"]
    )


def compare_snapshots(snapshot1: SnapshotInfo, snapshot2: SnapshotInfo) -> Dict:
    """
    Compare two snapshots to check reproducibility.

    Args:
        snapshot1: First snapshot
        snapshot2: Second snapshot

    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "identical": True,
        "differences": []
    }

    # Compare models
    if snapshot1.model != snapshot2.model:
        comparison["identical"] = False
        comparison["differences"].append(
            f"Model mismatch: {snapshot1.model} vs {snapshot2.model}"
        )

    # Compare document counts
    if snapshot1.doc_count != snapshot2.doc_count:
        comparison["identical"] = False
        comparison["differences"].append(
            f"Document count mismatch: {snapshot1.doc_count} vs {snapshot2.doc_count}"
        )

    # Compare chunk counts
    if snapshot1.chunk_count != snapshot2.chunk_count:
        comparison["identical"] = False
        comparison["differences"].append(
            f"Chunk count mismatch: {snapshot1.chunk_count} vs {snapshot2.chunk_count}"
        )

    # Compare document hashes
    all_docs = set(snapshot1.doc_hashes.keys()) | set(snapshot2.doc_hashes.keys())

    for doc in all_docs:
        hash1 = snapshot1.doc_hashes.get(doc)
        hash2 = snapshot2.doc_hashes.get(doc)

        if hash1 is None:
            comparison["identical"] = False
            comparison["differences"].append(f"Document missing in snapshot1: {doc}")
        elif hash2 is None:
            comparison["identical"] = False
            comparison["differences"].append(f"Document missing in snapshot2: {doc}")
        elif hash1 != hash2:
            comparison["identical"] = False
            comparison["differences"].append(f"Document hash mismatch: {doc}")

    return comparison


def list_snapshots(index_dir: str) -> List[str]:
    """
    List all snapshot files in an index directory.

    Args:
        index_dir: Path to index directory

    Returns:
        List of snapshot file paths
    """
    index_path = Path(index_dir)
    if not index_path.exists():
        return []

    snapshot_files = list(index_path.glob("snapshot_*.json"))
    return [str(f) for f in sorted(snapshot_files, reverse=True)]


def get_latest_snapshot(index_dir: str) -> str:
    """
    Get the most recent snapshot file in an index directory.

    Args:
        index_dir: Path to index directory

    Returns:
        Path to latest snapshot file, or None if no snapshots exist
    """
    snapshots = list_snapshots(index_dir)
    return snapshots[0] if snapshots else None
