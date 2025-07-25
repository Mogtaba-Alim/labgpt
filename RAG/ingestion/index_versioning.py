#!/usr/bin/env python3
"""
index_versioning.py

Index versioning and rollback system with manifest tracking.
Enables safe experimentation, A/B testing, and production rollbacks.
"""

import os
import json
import shutil
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tarfile
import tempfile

import numpy as np
import faiss

logger = logging.getLogger(__name__)

@dataclass
class IndexManifest:
    """Manifest for a versioned index"""
    version_id: str
    model_version: str
    build_datetime: str
    doc_snapshot_hash: str
    chunk_count: int
    embedding_dim: int
    index_type: str
    config_hash: str
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    file_checksums: Dict[str, str]
    created_by: str = "system"
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class VersionComparison:
    """Comparison between two index versions"""
    version_a: str
    version_b: str
    chunk_count_diff: int
    config_changes: Dict[str, Any]
    performance_diff: Dict[str, float]
    compatibility: str  # "compatible", "breaking", "unknown"
    migration_required: bool
    notes: str = ""

class IndexVersionManager:
    """
    Manages versioned storage of RAG indices with rollback capabilities.
    
    Features:
    - Versioned index storage with manifests
    - Rollback to previous versions
    - Index comparison and diff tools
    - Automated backup and archival
    - Migration tools for version upgrades
    - Performance tracking across versions
    """
    
    def __init__(self, 
                 versions_dir: str = "index_versions",
                 max_versions: int = 10,
                 enable_compression: bool = True):
        
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(exist_ok=True)
        
        self.max_versions = max_versions
        self.enable_compression = enable_compression
        
        # Directory structure
        self.active_dir = self.versions_dir / "active"
        self.archive_dir = self.versions_dir / "archive"
        self.metadata_dir = self.versions_dir / "metadata"
        
        for dir_path in [self.active_dir, self.archive_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Version tracking
        self.versions_file = self.metadata_dir / "versions.json"
        self.current_version_file = self.metadata_dir / "current_version.txt"
        
        # Load existing versions
        self._versions: Dict[str, IndexManifest] = {}
        self._load_versions()
    
    def create_version(self,
                      index_files: Dict[str, str],
                      config: Dict[str, Any],
                      model_version: str,
                      chunk_count: int,
                      embedding_dim: int,
                      index_type: str,
                      doc_snapshot_hash: str,
                      performance_metrics: Optional[Dict[str, float]] = None,
                      description: str = "",
                      tags: Optional[List[str]] = None) -> str:
        """
        Create a new version of the index
        
        Args:
            index_files: Dictionary mapping file types to file paths
            config: Configuration used to build the index
            model_version: Version of the embedding model
            chunk_count: Number of chunks in the index
            embedding_dim: Dimension of embeddings
            index_type: Type of index (e.g., "hnsw", "ivf")
            doc_snapshot_hash: Hash of the document collection
            performance_metrics: Optional performance metrics
            description: Human-readable description
            tags: Optional tags for categorization
            
        Returns:
            Version ID of the created version
        """
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        
        logger.info(f"Creating index version {version_id}")
        
        # Create version directory
        version_dir = self.active_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Copy index files and calculate checksums
        file_checksums = {}
        for file_type, source_path in index_files.items():
            if not os.path.exists(source_path):
                logger.warning(f"Source file {source_path} does not exist, skipping")
                continue
            
            dest_path = version_dir / f"{file_type}"
            shutil.copy2(source_path, dest_path)
            
            # Calculate checksum
            file_checksums[file_type] = self._calculate_file_checksum(dest_path)
        
        # Calculate config hash
        config_hash = self._calculate_config_hash(config)
        
        # Create manifest
        manifest = IndexManifest(
            version_id=version_id,
            model_version=model_version,
            build_datetime=datetime.now().isoformat(),
            doc_snapshot_hash=doc_snapshot_hash,
            chunk_count=chunk_count,
            embedding_dim=embedding_dim,
            index_type=index_type,
            config_hash=config_hash,
            performance_metrics=performance_metrics or {},
            metadata={
                "config": config,
                "creation_timestamp": time.time()
            },
            file_checksums=file_checksums,
            description=description,
            tags=tags or []
        )
        
        # Save manifest
        manifest_path = version_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(asdict(manifest), f, indent=2)
        
        # Update versions registry
        self._versions[version_id] = manifest
        self._save_versions()
        
        # Set as current version
        self._set_current_version(version_id)
        
        # Clean up old versions if necessary
        self._cleanup_old_versions()
        
        logger.info(f"Created index version {version_id} with {len(file_checksums)} files")
        
        return version_id
    
    def rollback_to_version(self, version_id: str, target_dir: str) -> bool:
        """
        Rollback to a specific version
        
        Args:
            version_id: ID of the version to rollback to
            target_dir: Directory to restore the index files to
            
        Returns:
            True if rollback successful, False otherwise
        """
        if version_id not in self._versions:
            logger.error(f"Version {version_id} not found")
            return False
        
        logger.info(f"Rolling back to version {version_id}")
        
        # Get version directory
        version_dir = self._get_version_dir(version_id)
        if not version_dir.exists():
            logger.error(f"Version directory for {version_id} not found")
            return False
        
        # Ensure target directory exists
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy all files except manifest
            for file_path in version_dir.iterdir():
                if file_path.name != "manifest.json":
                    dest_path = Path(target_dir) / file_path.name
                    shutil.copy2(file_path, dest_path)
                    logger.debug(f"Restored {file_path.name}")
            
            # Update current version
            self._set_current_version(version_id)
            
            logger.info(f"Successfully rolled back to version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def list_versions(self) -> List[IndexManifest]:
        """List all available versions"""
        return list(self._versions.values())
    
    def get_version_info(self, version_id: str) -> Optional[IndexManifest]:
        """Get detailed information about a specific version"""
        return self._versions.get(version_id)
    
    def get_current_version(self) -> Optional[str]:
        """Get the current active version ID"""
        try:
            if self.current_version_file.exists():
                with open(self.current_version_file, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read current version: {e}")
        return None
    
    def compare_versions(self, version_a: str, version_b: str) -> Optional[VersionComparison]:
        """
        Compare two versions and return differences
        
        Args:
            version_a: First version ID
            version_b: Second version ID
            
        Returns:
            VersionComparison object or None if comparison fails
        """
        if version_a not in self._versions or version_b not in self._versions:
            logger.error("One or both versions not found")
            return None
        
        manifest_a = self._versions[version_a]
        manifest_b = self._versions[version_b]
        
        # Calculate differences
        chunk_count_diff = manifest_b.chunk_count - manifest_a.chunk_count
        
        # Compare configurations
        config_a = manifest_a.metadata.get("config", {})
        config_b = manifest_b.metadata.get("config", {})
        config_changes = self._diff_configs(config_a, config_b)
        
        # Compare performance metrics
        perf_diff = {}
        for metric in set(manifest_a.performance_metrics.keys()) | set(manifest_b.performance_metrics.keys()):
            val_a = manifest_a.performance_metrics.get(metric, 0.0)
            val_b = manifest_b.performance_metrics.get(metric, 0.0)
            perf_diff[metric] = val_b - val_a
        
        # Determine compatibility
        compatibility = self._assess_compatibility(manifest_a, manifest_b)
        migration_required = (
            manifest_a.model_version != manifest_b.model_version or
            manifest_a.embedding_dim != manifest_b.embedding_dim or
            compatibility == "breaking"
        )
        
        return VersionComparison(
            version_a=version_a,
            version_b=version_b,
            chunk_count_diff=chunk_count_diff,
            config_changes=config_changes,
            performance_diff=perf_diff,
            compatibility=compatibility,
            migration_required=migration_required
        )
    
    def delete_version(self, version_id: str, archive_first: bool = True) -> bool:
        """
        Delete a version
        
        Args:
            version_id: ID of the version to delete
            archive_first: Whether to archive before deletion
            
        Returns:
            True if deletion successful, False otherwise
        """
        if version_id not in self._versions:
            logger.error(f"Version {version_id} not found")
            return False
        
        # Check if it's the current version
        current = self.get_current_version()
        if current == version_id:
            logger.error("Cannot delete current active version")
            return False
        
        logger.info(f"Deleting version {version_id}")
        
        try:
            version_dir = self._get_version_dir(version_id)
            
            # Archive if requested
            if archive_first and version_dir.exists():
                self._archive_version(version_id)
            
            # Delete version directory
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            # Remove from registry
            del self._versions[version_id]
            self._save_versions()
            
            logger.info(f"Deleted version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False
    
    def archive_version(self, version_id: str) -> bool:
        """
        Archive a version to compressed storage
        
        Args:
            version_id: ID of the version to archive
            
        Returns:
            True if archival successful, False otherwise
        """
        return self._archive_version(version_id)
    
    def validate_version(self, version_id: str) -> Dict[str, Any]:
        """
        Validate version integrity
        
        Args:
            version_id: ID of the version to validate
            
        Returns:
            Validation report
        """
        if version_id not in self._versions:
            return {"status": "error", "message": "Version not found"}
        
        manifest = self._versions[version_id]
        version_dir = self._get_version_dir(version_id)
        
        report = {
            "version_id": version_id,
            "status": "valid",
            "issues": [],
            "file_validation": {},
            "manifest_valid": True
        }
        
        # Check if directory exists
        if not version_dir.exists():
            report["status"] = "error"
            report["issues"].append("Version directory missing")
            return report
        
        # Validate manifest file
        manifest_path = version_dir / "manifest.json"
        if not manifest_path.exists():
            report["manifest_valid"] = False
            report["issues"].append("Manifest file missing")
        
        # Validate file checksums
        for file_type, expected_checksum in manifest.file_checksums.items():
            file_path = version_dir / file_type
            
            if not file_path.exists():
                report["file_validation"][file_type] = {
                    "status": "missing",
                    "expected_checksum": expected_checksum
                }
                report["issues"].append(f"File {file_type} missing")
                continue
            
            actual_checksum = self._calculate_file_checksum(file_path)
            if actual_checksum != expected_checksum:
                report["file_validation"][file_type] = {
                    "status": "corrupted",
                    "expected_checksum": expected_checksum,
                    "actual_checksum": actual_checksum
                }
                report["issues"].append(f"File {file_type} corrupted")
            else:
                report["file_validation"][file_type] = {
                    "status": "valid",
                    "checksum": actual_checksum
                }
        
        if report["issues"]:
            report["status"] = "invalid"
        
        return report
    
    def export_version(self, version_id: str, export_path: str) -> bool:
        """
        Export a version to a portable format
        
        Args:
            version_id: ID of the version to export
            export_path: Path for the exported file
            
        Returns:
            True if export successful, False otherwise
        """
        if version_id not in self._versions:
            logger.error(f"Version {version_id} not found")
            return False
        
        version_dir = self._get_version_dir(version_id)
        if not version_dir.exists():
            logger.error(f"Version directory for {version_id} not found")
            return False
        
        try:
            logger.info(f"Exporting version {version_id} to {export_path}")
            
            with tarfile.open(export_path, "w:gz") as tar:
                tar.add(version_dir, arcname=version_id)
            
            logger.info(f"Successfully exported version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def import_version(self, import_path: str, version_id: Optional[str] = None) -> Optional[str]:
        """
        Import a version from exported file
        
        Args:
            import_path: Path to the exported version file
            version_id: Optional new version ID (generates one if None)
            
        Returns:
            Version ID of imported version, or None if import failed
        """
        if not os.path.exists(import_path):
            logger.error(f"Import file {import_path} not found")
            return None
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract archive
                with tarfile.open(import_path, "r:gz") as tar:
                    tar.extractall(temp_dir)
                
                # Find extracted directory
                extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
                if not extracted_dirs:
                    logger.error("No directory found in archive")
                    return None
                
                extracted_dir = Path(temp_dir) / extracted_dirs[0]
                
                # Load manifest
                manifest_path = extracted_dir / "manifest.json"
                if not manifest_path.exists():
                    logger.error("Manifest not found in archive")
                    return None
                
                with open(manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                
                # Generate new version ID if not provided
                if version_id is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    version_id = f"imported_{timestamp}_{manifest_data['version_id']}"
                
                # Create new version directory
                new_version_dir = self.active_dir / version_id
                new_version_dir.mkdir(exist_ok=True)
                
                # Copy files
                for item in extracted_dir.iterdir():
                    dest_path = new_version_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dest_path)
                    else:
                        shutil.copytree(item, dest_path)
                
                # Update manifest with new version ID
                manifest_data["version_id"] = version_id
                with open(new_version_dir / "manifest.json", 'w') as f:
                    json.dump(manifest_data, f, indent=2)
                
                # Add to registry
                manifest = IndexManifest(**manifest_data)
                self._versions[version_id] = manifest
                self._save_versions()
                
                logger.info(f"Successfully imported version {version_id}")
                return version_id
                
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return None
    
    def _get_version_dir(self, version_id: str) -> Path:
        """Get the directory path for a version"""
        return self.active_dir / version_id
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _load_versions(self) -> None:
        """Load versions registry from disk"""
        try:
            if self.versions_file.exists():
                with open(self.versions_file, 'r') as f:
                    versions_data = json.load(f)
                    
                    for version_id, manifest_data in versions_data.items():
                        self._versions[version_id] = IndexManifest(**manifest_data)
                
                logger.info(f"Loaded {len(self._versions)} versions")
            
        except Exception as e:
            logger.warning(f"Failed to load versions: {e}")
            self._versions = {}
    
    def _save_versions(self) -> None:
        """Save versions registry to disk"""
        try:
            versions_data = {}
            for version_id, manifest in self._versions.items():
                versions_data[version_id] = asdict(manifest)
            
            with open(self.versions_file, 'w') as f:
                json.dump(versions_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")
    
    def _set_current_version(self, version_id: str) -> None:
        """Set the current active version"""
        try:
            with open(self.current_version_file, 'w') as f:
                f.write(version_id)
        except Exception as e:
            logger.error(f"Failed to set current version: {e}")
    
    def _cleanup_old_versions(self) -> None:
        """Clean up old versions if exceeding max_versions"""
        if len(self._versions) <= self.max_versions:
            return
        
        # Sort versions by creation time
        sorted_versions = sorted(
            self._versions.items(),
            key=lambda x: x[1].metadata.get("creation_timestamp", 0)
        )
        
        # Archive and delete oldest versions
        versions_to_remove = len(self._versions) - self.max_versions
        for i in range(versions_to_remove):
            version_id, _ = sorted_versions[i]
            
            # Don't delete current version
            if version_id != self.get_current_version():
                logger.info(f"Archiving old version {version_id}")
                self._archive_version(version_id)
                
                # Delete from active storage
                version_dir = self._get_version_dir(version_id)
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                
                del self._versions[version_id]
        
        self._save_versions()
    
    def _archive_version(self, version_id: str) -> bool:
        """Archive a version to compressed storage"""
        if version_id not in self._versions:
            return False
        
        try:
            version_dir = self._get_version_dir(version_id)
            archive_path = self.archive_dir / f"{version_id}.tar.gz"
            
            if self.enable_compression:
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add(version_dir, arcname=version_id)
            else:
                with tarfile.open(archive_path, "w") as tar:
                    tar.add(version_dir, arcname=version_id)
            
            logger.info(f"Archived version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive version {version_id}: {e}")
            return False
    
    def _diff_configs(self, config_a: Dict, config_b: Dict) -> Dict[str, Any]:
        """Compare two configurations and return differences"""
        changes = {}
        
        all_keys = set(config_a.keys()) | set(config_b.keys())
        
        for key in all_keys:
            val_a = config_a.get(key)
            val_b = config_b.get(key)
            
            if val_a != val_b:
                changes[key] = {
                    "from": val_a,
                    "to": val_b
                }
        
        return changes
    
    def _assess_compatibility(self, manifest_a: IndexManifest, manifest_b: IndexManifest) -> str:
        """Assess compatibility between two versions"""
        # Breaking changes
        if (manifest_a.model_version != manifest_b.model_version or
            manifest_a.embedding_dim != manifest_b.embedding_dim):
            return "breaking"
        
        # Configuration changes that might affect compatibility
        config_a = manifest_a.metadata.get("config", {})
        config_b = manifest_b.metadata.get("config", {})
        
        # Check for significant config changes
        breaking_config_keys = ["embedding_model", "chunk_size", "index_type"]
        for key in breaking_config_keys:
            if config_a.get(key) != config_b.get(key):
                return "breaking"
        
        return "compatible"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get version manager statistics"""
        active_versions = [v for v in self._versions.values() 
                          if (self.active_dir / v.version_id).exists()]
        archived_versions = len(list(self.archive_dir.glob("*.tar.gz")))
        
        # Calculate storage usage
        active_size = sum(
            sum(f.stat().st_size for f in (self.active_dir / v.version_id).rglob("*") if f.is_file())
            for v in active_versions
        )
        
        archive_size = sum(f.stat().st_size for f in self.archive_dir.glob("*.tar.gz"))
        
        return {
            "total_versions": len(self._versions),
            "active_versions": len(active_versions),
            "archived_versions": archived_versions,
            "current_version": self.get_current_version(),
            "storage": {
                "active_size_mb": active_size / 1024 / 1024,
                "archive_size_mb": archive_size / 1024 / 1024,
                "total_size_mb": (active_size + archive_size) / 1024 / 1024
            },
            "oldest_version": min(self._versions.values(), 
                                key=lambda x: x.metadata.get("creation_timestamp", 0)).version_id if self._versions else None,
            "newest_version": max(self._versions.values(), 
                                key=lambda x: x.metadata.get("creation_timestamp", 0)).version_id if self._versions else None
        } 