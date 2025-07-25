#!/usr/bin/env python3
"""
document_loader.py

Enhanced document loading with metadata extraction and content preprocessing.
Supports PDF, TXT, MD, and other document types with structure preservation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
import re

from .chunk_objects import DocumentMetadata

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

class DocumentLoader:
    """Enhanced document loader with metadata extraction and structure preservation"""
    
    def __init__(self, supported_extensions: Optional[List[str]] = None):
        self.supported_extensions = supported_extensions or ['.pdf', '.txt', '.md', '.tex', '.rst']
        
    def load_document(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """
        Load document and extract both content and metadata
        
        Returns:
            Tuple of (content_text, document_metadata)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
            
        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        # Extract basic file metadata
        file_stats = file_path.stat()
        doc_metadata = DocumentMetadata(
            doc_id=self._generate_doc_id(file_path),
            source_path=str(file_path),
            doc_type=extension[1:],  # Remove leading dot
            creation_date=datetime.fromtimestamp(file_stats.st_ctime),
            file_size=file_stats.st_size
        )
        
        # Extract content based on file type
        if extension == '.pdf':
            content, pdf_metadata = self._load_pdf(file_path)
            # Merge PDF-specific metadata
            doc_metadata.title = pdf_metadata.get('title')
            doc_metadata.authors = pdf_metadata.get('authors')
            doc_metadata.page_count = pdf_metadata.get('page_count')
        else:
            content = self._load_text_file(file_path)
            # Try to extract title from content for text files
            doc_metadata.title = self._extract_title_from_text(content)
        
        # Generate content hash
        doc_metadata.content_hash = hashlib.md5(content.encode()).hexdigest()
        
        logger.info(f"Loaded document: {file_path.name} ({len(content)} chars)")
        return content, doc_metadata
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate a unique document ID based on file path and name"""
        # Use file path hash to ensure uniqueness
        path_hash = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()[:8]
        name = file_path.stem
        return f"{name}_{path_hash}"
    
    def _load_pdf(self, file_path: Path) -> Tuple[str, Dict]:
        """Load PDF content and extract metadata"""
        try:
            doc = fitz.open(file_path)
            
            # Extract text content
            text_content = []
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text("text")
                if page_text.strip():
                    # Add page markers for later reference
                    text_content.append(f"[PAGE {page_num}]\n{page_text}")
            
            content = "\n".join(text_content)
            
            # Extract PDF metadata
            pdf_metadata = {
                'title': doc.metadata.get('title', '').strip(),
                'authors': self._parse_authors(doc.metadata.get('author', '')),
                'page_count': len(doc),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', '')
            }
            
            doc.close()
            return content, pdf_metadata
            
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return "", {}
    
    def _load_text_file(self, file_path: Path) -> str:
        """Load text-based files with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
                
        # Fallback: read with errors ignored
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                logger.warning(f"Loaded {file_path} with ignored encoding errors")
                return content
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return ""
    
    def _parse_authors(self, author_string: str) -> Optional[List[str]]:
        """Parse author string into list of individual authors"""
        if not author_string or not author_string.strip():
            return None
            
        # Common author separators
        separators = [';', ',', ' and ', '&', '\n']
        authors = [author_string]
        
        for sep in separators:
            new_authors = []
            for author in authors:
                new_authors.extend([a.strip() for a in author.split(sep)])
            authors = new_authors
        
        # Filter out empty authors and normalize
        authors = [author for author in authors if author and len(author) > 1]
        return authors if authors else None
    
    def _extract_title_from_text(self, content: str) -> Optional[str]:
        """Extract title from text content using heuristics"""
        lines = content.split('\n')[:10]  # Check first 10 lines
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Heuristics for title detection
            if (
                len(line) > 10 and len(line) < 200 and  # Reasonable length
                not line.startswith('#') and  # Not markdown header
                not line.lower().startswith('abstract') and
                not line.lower().startswith('introduction') and
                not re.match(r'^\d+\.', line) and  # Not numbered section
                re.search(r'[A-Z].*[a-z]', line)  # Contains mixed case
            ):
                # Clean up potential title
                title = re.sub(r'[^\w\s\-\:\.]', '', line)
                return title.strip()
        
        return None
    
    def extract_document_structure(self, content: str, doc_type: str) -> Dict:
        """
        Extract document structure including headings, sections, and TOC
        
        Returns:
            Dictionary with structure information including headings hierarchy
        """
        structure = {
            'headings': [],
            'sections': [],
            'toc': [],
            'has_structure': False
        }
        
        if doc_type == 'pdf':
            structure.update(self._extract_pdf_structure(content))
        elif doc_type == 'md':
            structure.update(self._extract_markdown_structure(content))
        else:
            structure.update(self._extract_text_structure(content))
        
        structure['has_structure'] = len(structure['headings']) > 0
        return structure
    
    def _extract_pdf_structure(self, content: str) -> Dict:
        """Extract structure from PDF content"""
        headings = []
        sections = []
        
        lines = content.split('\n')
        current_section = None
        section_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip page markers
            if re.match(r'^\[PAGE \d+\]$', line):
                continue
                
            # Detect headings using various heuristics
            if self._is_likely_heading(line, lines, i):
                # Save previous section
                if current_section and section_content:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(section_content).strip(),
                        'start_line': i - len(section_content),
                        'end_line': i
                    })
                
                # Start new section
                headings.append({
                    'text': line,
                    'level': self._estimate_heading_level(line),
                    'line_number': i
                })
                current_section = line
                section_content = []
            else:
                if line:  # Don't add empty lines
                    section_content.append(line)
        
        # Add final section
        if current_section and section_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(section_content).strip(),
                'start_line': len(lines) - len(section_content),
                'end_line': len(lines)
            })
        
        return {
            'headings': headings,
            'sections': sections,
            'toc': [h['text'] for h in headings]
        }
    
    def _extract_markdown_structure(self, content: str) -> Dict:
        """Extract structure from Markdown content"""
        headings = []
        sections = []
        
        lines = content.split('\n')
        current_section = None
        section_content = []
        
        for i, line in enumerate(lines):
            # Markdown heading detection
            if line.strip().startswith('#'):
                # Save previous section
                if current_section and section_content:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(section_content).strip(),
                        'start_line': i - len(section_content),
                        'end_line': i
                    })
                
                # Extract heading
                heading_match = re.match(r'^(#+)\s*(.*)', line.strip())
                if heading_match:
                    level = len(heading_match.group(1))
                    text = heading_match.group(2)
                    headings.append({
                        'text': text,
                        'level': level,
                        'line_number': i
                    })
                    current_section = text
                    section_content = []
            else:
                if line.strip():
                    section_content.append(line)
        
        # Add final section
        if current_section and section_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(section_content).strip(),
                'start_line': len(lines) - len(section_content),
                'end_line': len(lines)
            })
        
        return {
            'headings': headings,
            'sections': sections,
            'toc': [h['text'] for h in headings]
        }
    
    def _extract_text_structure(self, content: str) -> Dict:
        """Extract structure from plain text using heuristics"""
        headings = []
        sections = []
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if self._is_likely_heading(line, lines, i):
                headings.append({
                    'text': line,
                    'level': self._estimate_heading_level(line),
                    'line_number': i
                })
        
        return {
            'headings': headings,
            'sections': sections,
            'toc': [h['text'] for h in headings]
        }
    
    def _is_likely_heading(self, line: str, lines: List[str], line_idx: int) -> bool:
        """Determine if a line is likely a heading using heuristics"""
        if not line or len(line) < 3:
            return False
            
        # Skip very long lines
        if len(line) > 200:
            return False
            
        # Check for common heading patterns
        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
            r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # Title Case
            r'.*Introduction.*',
            r'.*Conclusion.*',
            r'.*Abstract.*',
            r'.*References.*',
            r'.*Methodology.*',
            r'.*Results.*',
            r'.*Discussion.*'
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        # Check if line stands alone (empty lines before/after)
        prev_empty = line_idx == 0 or not lines[line_idx - 1].strip()
        next_empty = line_idx == len(lines) - 1 or not lines[line_idx + 1].strip()
        
        if prev_empty and next_empty and len(line) < 100:
            return True
            
        return False
    
    def _estimate_heading_level(self, heading: str) -> int:
        """Estimate heading level (1-6) based on content"""
        # Check for numbered sections
        if re.match(r'^\d+\.', heading):
            return 1
        elif re.match(r'^\d+\.\d+', heading):
            return 2
        elif re.match(r'^\d+\.\d+\.\d+', heading):
            return 3
            
        # Check for common top-level headings
        top_level_keywords = ['abstract', 'introduction', 'conclusion', 'references']
        if any(keyword in heading.lower() for keyword in top_level_keywords):
            return 1
            
        # Check for all caps (often main headings)
        if heading.isupper():
            return 1
            
        # Default to level 2
        return 2
    
    def batch_load_documents(self, directory: str, recursive: bool = True) -> List[Tuple[str, DocumentMetadata]]:
        """
        Load all supported documents from a directory
        
        Returns:
            List of (content, metadata) tuples
        """
        directory = Path(directory)
        documents = []
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    content, metadata = self.load_document(file_path)
                    documents.append((content, metadata))
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents 