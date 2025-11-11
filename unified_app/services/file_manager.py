"""
File management service for handling uploads and directory organization.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
from werkzeug.utils import secure_filename
from unified_app.config import Config
from unified_app.extensions import db
from unified_app.models import Project, CodeRepo, ResearchPaper, LabDocument


class FileManager:
    """
    Handles file uploads, directory creation, and project file organization.
    """

    ALLOWED_EXTENSIONS = {
        'papers': {'.pdf', '.txt', '.md'},
        'lab_docs': {'.pdf', '.txt', '.md', '.py', '.r', '.c', '.cpp', '.h', '.hpp'},
        'code': {'.py', '.r', '.c', '.cpp', '.h', '.hpp'}
    }

    @staticmethod
    def create_project_structure(project_name: str) -> Path:
        """
        Create directory structure for a new project.

        Args:
            project_name: Name of the project

        Returns:
            Path to project directory
        """
        # Create safe directory name
        safe_name = secure_filename(project_name)
        project_dir = Config.PROJECTS_BASE_DIR / safe_name

        # Handle name collisions by appending a number
        counter = 1
        original_dir = project_dir
        while project_dir.exists():
            project_dir = Path(f"{original_dir}_{counter}")
            counter += 1

        # Create subdirectories
        subdirs = [
            'code_repos',
            'research_papers',
            'lab_documents',
            'rag-index',
            'data-generation',
            'training',
            'logs'
        ]

        for subdir in subdirs:
            (project_dir / subdir).mkdir(parents=True, exist_ok=True)

        return project_dir

    @staticmethod
    def save_uploaded_files(files: List, destination_dir: Path, file_type: str = 'papers') -> List[Tuple[str, Path]]:
        """
        Save uploaded files to project directory.

        Args:
            files: List of FileStorage objects from Flask request
            destination_dir: Directory to save files
            file_type: Type of files ('papers', 'lab_docs', 'code')

        Returns:
            List of tuples: (original_filename, saved_path)
        """
        destination_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        allowed_exts = FileManager.ALLOWED_EXTENSIONS.get(file_type, set())

        for file in files:
            if file and file.filename:
                # Check extension
                ext = Path(file.filename).suffix.lower()
                if ext not in allowed_exts:
                    continue

                # Save file with secure filename
                filename = secure_filename(file.filename)
                filepath = destination_dir / filename

                # Handle duplicate filenames
                counter = 1
                original_filepath = filepath
                while filepath.exists():
                    name_parts = filename.rsplit('.', 1)
                    if len(name_parts) == 2:
                        new_filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    else:
                        new_filename = f"{filename}_{counter}"
                    filepath = destination_dir / new_filename
                    counter += 1

                file.save(str(filepath))
                saved_files.append((file.filename, filepath))

        return saved_files

    @staticmethod
    def copy_directory_contents(source_dir: str, destination_dir: Path, file_type: str = 'papers') -> List[Tuple[str, Path]]:
        """
        Copy files from a source directory to project directory.

        Args:
            source_dir: Source directory path
            destination_dir: Destination directory
            file_type: Type of files to filter

        Returns:
            List of tuples: (original_filename, copied_path)
        """
        source_path = Path(source_dir)
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError(f"Source directory does not exist: {source_dir}")

        destination_dir.mkdir(parents=True, exist_ok=True)
        copied_files = []

        allowed_exts = FileManager.ALLOWED_EXTENSIONS.get(file_type, set())

        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in allowed_exts:
                    # Preserve relative path structure
                    rel_path = file_path.relative_to(source_path)
                    dest_path = destination_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(file_path, dest_path)
                    copied_files.append((file_path.name, dest_path))

        return copied_files

    @staticmethod
    def register_code_repo(project_id: int, source_type: str, source_path: str, local_path: Path) -> CodeRepo:
        """
        Register a code repository in the database.

        Args:
            project_id: Project ID
            source_type: 'local' or 'github'
            source_path: Original path/URL
            local_path: Path where code is stored

        Returns:
            Created CodeRepo object
        """
        repo = CodeRepo(
            project_id=project_id,
            source_type=source_type,
            source_path=source_path,
            local_path=str(local_path)
        )
        db.session.add(repo)
        db.session.commit()
        return repo

    @staticmethod
    def register_research_papers(project_id: int, files: List[Tuple[str, Path]]) -> List[ResearchPaper]:
        """
        Register research papers in the database.

        Args:
            project_id: Project ID
            files: List of tuples (filename, filepath)

        Returns:
            List of created ResearchPaper objects
        """
        papers = []
        for filename, filepath in files:
            file_size = filepath.stat().st_size if filepath.exists() else 0
            file_type = filepath.suffix.lower()[1:]  # Remove leading dot

            paper = ResearchPaper(
                project_id=project_id,
                filename=filename,
                file_path=str(filepath),
                file_size=file_size,
                file_type=file_type
            )
            db.session.add(paper)
            papers.append(paper)

        db.session.commit()
        return papers

    @staticmethod
    def register_lab_documents(project_id: int, files: List[Tuple[str, Path]]) -> List[LabDocument]:
        """
        Register lab documents in the database.

        Args:
            project_id: Project ID
            files: List of tuples (filename, filepath)

        Returns:
            List of created LabDocument objects
        """
        documents = []
        for filename, filepath in files:
            file_size = filepath.stat().st_size if filepath.exists() else 0
            file_type = filepath.suffix.lower()[1:]  # Remove leading dot

            doc = LabDocument(
                project_id=project_id,
                filename=filename,
                file_path=str(filepath),
                file_size=file_size,
                file_type=file_type
            )
            db.session.add(doc)
            documents.append(doc)

        db.session.commit()
        return documents

    @staticmethod
    def get_file_size(path: Path) -> int:
        """Get file or directory size in bytes"""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return 0

    @staticmethod
    def cleanup_temp_files(project_dir: Path):
        """
        Clean up temporary files for a project.

        Args:
            project_dir: Project directory path
        """
        temp_dirs = ['temp', 'uploads']
        for temp_dir in temp_dirs:
            temp_path = project_dir / temp_dir
            if temp_path.exists():
                shutil.rmtree(temp_path)
