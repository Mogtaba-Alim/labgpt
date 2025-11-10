"""
Enhanced document loader for research papers.

Uses PyMuPDF (fitz) for robust PDF extraction with page markers.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, List
import hashlib
import re

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logging.warning("PyMuPDF not installed, falling back to PyPDF2")

# Fallback to PyPDF2 if PyMuPDF not available
if not HAS_PYMUPDF:
    import PyPDF2

logger = logging.getLogger(__name__)


class PaperLoader:
    """Enhanced document loader with metadata extraction and structure preservation."""

    def __init__(self, supported_extensions: List[str] = None):
        # Support PDFs, text files, Markdown, and R documentation files
        self.supported_extensions = supported_extensions or ['.pdf', '.txt', '.md', '.Rd', '.rd']

    def load_document(self, file_path: str) -> Tuple[str, Dict]:
        """
        Load document and extract both content and metadata.

        Returns:
            Tuple of (content_text, metadata_dict)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")

        # Generate document ID
        doc_id = self._generate_doc_id(file_path)

        # Basic metadata
        metadata = {
            'doc_id': doc_id,
            'source_path': str(file_path),
            'file_name': file_path.name,
            'doc_type': extension[1:],  # Remove leading dot
        }

        # Extract content based on file type
        if extension == '.pdf':
            content, pdf_metadata = self._load_pdf(file_path)
            metadata.update(pdf_metadata)
        else:
            content = self._load_text_file(file_path)
            # Try to extract title from content
            metadata['title'] = self._extract_title_from_text(content)

        logger.info(f"Loaded document: {file_path.name} ({len(content)} chars, {metadata.get('page_count', 'N/A')} pages)")
        return content, metadata

    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate a unique document ID based on file path and name."""
        path_hash = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()[:8]
        name = file_path.stem
        return f"{name}_{path_hash}"

    def _load_pdf(self, file_path: Path) -> Tuple[str, Dict]:
        """Load PDF content and extract metadata."""
        if HAS_PYMUPDF:
            return self._load_pdf_pymupdf(file_path)
        else:
            return self._load_pdf_pypdf2(file_path)

    def _load_pdf_pymupdf(self, file_path: Path) -> Tuple[str, Dict]:
        """Load PDF using PyMuPDF (fitz) with page markers."""
        try:
            doc = fitz.open(file_path)

            # Extract text content with page markers
            text_content = []
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text("text")
                if page_text.strip():
                    # Add page markers for later reference
                    text_content.append(f"[PAGE {page_num}]\n{page_text}")

            content = "\n".join(text_content)

            # Extract PDF metadata
            pdf_metadata = {
                'title': doc.metadata.get('title', '').strip() or file_path.stem,
                'authors': self._parse_authors(doc.metadata.get('author', '')),
                'page_count': len(doc),
                'subject': doc.metadata.get('subject', ''),
            }

            doc.close()
            return content, pdf_metadata

        except Exception as e:
            logger.error(f"Error reading PDF {file_path} with PyMuPDF: {e}")
            return "", {}

    def _load_pdf_pypdf2(self, file_path: Path) -> Tuple[str, Dict]:
        """Load PDF using PyPDF2 as fallback."""
        try:
            text_content = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"[PAGE {page_num}]\n{page_text}")

            content = "\n".join(text_content)

            # Basic metadata
            pdf_metadata = {
                'title': file_path.stem,
                'page_count': len(pdf_reader.pages),
            }

            return content, pdf_metadata

        except Exception as e:
            logger.error(f"Error reading PDF {file_path} with PyPDF2: {e}")
            return "", {}

    def _load_text_file(self, file_path: Path) -> str:
        """Load text-based files with encoding detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                return ""

        logger.warning(f"Could not decode file {file_path} with any encoding")
        return ""

    def _parse_authors(self, author_str: str) -> List[str]:
        """Parse author string into list of author names."""
        if not author_str:
            return []

        # Split by common delimiters
        authors = re.split(r'[;,]|\sand\s', author_str)
        # Clean and filter
        authors = [a.strip() for a in authors if a.strip()]
        return authors

    def _extract_title_from_text(self, content: str) -> str:
        """Extract title from text content (first non-empty line)."""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                return line
        return ""
