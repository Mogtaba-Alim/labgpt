#!/usr/bin/env python3
"""
text_splitter.py

Semantic and structural text splitter that respects document boundaries,
uses sentence tokenization, and implements token-budget based chunking.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import nltk
from nltk.tokenize import sent_tokenize
import tiktoken

from .chunk_objects import ChunkMetadata

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

@dataclass
class SplittingConfig:
    """Configuration for text splitting"""
    target_chunk_size: int = 400  # Target tokens per chunk
    max_chunk_size: int = 512     # Maximum tokens per chunk
    min_chunk_size: int = 50      # Minimum tokens per chunk
    chunk_overlap: int = 50       # Overlap in tokens
    respect_sentence_boundaries: bool = True
    respect_section_boundaries: bool = True
    preserve_structure: bool = True
    
class SemanticStructuralSplitter:
    """
    Advanced text splitter that:
    1. Respects document structure (sections, headings)
    2. Uses sentence boundaries for semantic coherence
    3. Implements token-budget based chunking
    4. Preserves hierarchical context
    """
    
    def __init__(self, config: Optional[SplittingConfig] = None):
        self.config = config or SplittingConfig()
        
        # Initialize tokenizer (using tiktoken for accurate token counting)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except Exception:
            logger.warning("Could not load tiktoken, falling back to approximate tokenization")
            self.tokenizer = None
    
    def split_document(self, 
                      content: str, 
                      doc_metadata: Dict,
                      document_structure: Optional[Dict] = None) -> List[ChunkMetadata]:
        """
        Split document into semantically coherent chunks
        
        Args:
            content: Document text content
            doc_metadata: Document metadata dictionary
            document_structure: Optional structure information from document loader
            
        Returns:
            List of ChunkMetadata objects
        """
        chunks = []
        
        if document_structure and document_structure.get('has_structure') and self.config.preserve_structure:
            # Use structure-aware splitting
            chunks = self._split_with_structure(content, doc_metadata, document_structure)
        else:
            # Fall back to semantic splitting without structure
            chunks = self._split_semantic(content, doc_metadata)
        
        # Post-process chunks to ensure quality
        chunks = self._post_process_chunks(chunks)
        
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks
    
    def _split_with_structure(self, 
                             content: str, 
                             doc_metadata: Dict, 
                             structure: Dict) -> List[ChunkMetadata]:
        """Split document using structural information"""
        chunks = []
        sections = structure.get('sections', [])
        
        if not sections:
            # No sections found, fall back to semantic splitting
            return self._split_semantic(content, doc_metadata)
        
        for section in sections:
            section_title = section['title']
            section_content = section['content']
            
            # Determine hierarchy path
            hierarchy_path = self._build_hierarchy_path(section_title, structure['headings'])
            
            # Split section content into chunks
            section_chunks = self._split_section_content(
                section_content, 
                doc_metadata, 
                section_title,
                hierarchy_path
            )
            
            chunks.extend(section_chunks)
        
        return chunks
    
    def _split_semantic(self, content: str, doc_metadata: Dict) -> List[ChunkMetadata]:
        """Split document using semantic approach without structure"""
        # First, split into sentences
        sentences = self._split_into_sentences(content)
        
        # Then pack sentences into token-budget chunks
        chunks = self._pack_sentences_into_chunks(sentences, doc_metadata)
        
        return chunks
    
    def _split_section_content(self, 
                              section_content: str,
                              doc_metadata: Dict,
                              section_title: str,
                              hierarchy_path: List[str]) -> List[ChunkMetadata]:
        """Split a single section into chunks"""
        # Split section into sentences
        sentences = self._split_into_sentences(section_content)
        
        # Pack sentences into chunks
        chunks = self._pack_sentences_into_chunks(
            sentences, 
            doc_metadata, 
            section_title=section_title,
            hierarchy_path=hierarchy_path
        )
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        if not text.strip():
            return []
        
        try:
            sentences = sent_tokenize(text)
            # Clean up sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}")
            # Fallback: split on periods, exclamations, questions
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _pack_sentences_into_chunks(self, 
                                   sentences: List[str],
                                   doc_metadata: Dict,
                                   section_title: Optional[str] = None,
                                   hierarchy_path: Optional[List[str]] = None) -> List[ChunkMetadata]:
        """Pack sentences into token-budget chunks"""
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # Check if adding this sentence would exceed max chunk size
            if (current_tokens + sentence_tokens > self.config.max_chunk_size and 
                current_chunk_sentences):
                
                # Create chunk from current sentences
                chunk = self._create_chunk_from_sentences(
                    current_chunk_sentences, 
                    doc_metadata,
                    section_title,
                    hierarchy_path,
                    len(chunks)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, 
                    self.config.chunk_overlap
                )
                current_chunk_sentences = overlap_sentences + [sentence]
                current_tokens = sum(self._count_tokens(s) for s in current_chunk_sentences)
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens
                
            # If we've reached target size, consider finishing chunk
            if (current_tokens >= self.config.target_chunk_size and 
                len(current_chunk_sentences) > 0):
                
                chunk = self._create_chunk_from_sentences(
                    current_chunk_sentences,
                    doc_metadata,
                    section_title, 
                    hierarchy_path,
                    len(chunks)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences,
                    self.config.chunk_overlap
                )
                current_chunk_sentences = overlap_sentences
                current_tokens = sum(self._count_tokens(s) for s in current_chunk_sentences)
        
        # Handle remaining sentences
        if current_chunk_sentences:
            chunk = self._create_chunk_from_sentences(
                current_chunk_sentences,
                doc_metadata,
                section_title,
                hierarchy_path,
                len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_sentences(self,
                                    sentences: List[str],
                                    doc_metadata: Dict,
                                    section_title: Optional[str],
                                    hierarchy_path: Optional[List[str]],
                                    chunk_index: int) -> ChunkMetadata:
        """Create a ChunkMetadata object from sentences"""
        text = ' '.join(sentences)
        token_count = self._count_tokens(text)
        
        chunk = ChunkMetadata(
            doc_id=doc_metadata.get('doc_id', ''),
            text=text,
            token_count=token_count,
            doc_type=doc_metadata.get('doc_type', ''),
            source_path=doc_metadata.get('source_path', ''),
            section=section_title,
            hierarchy_path=hierarchy_path or [],
            chunk_index=chunk_index
        )
        
        return chunk
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """Get sentences for overlap from the end of current chunk"""
        if not sentences or overlap_tokens <= 0:
            return []
        
        overlap_sentences = []
        tokens_collected = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_tokens = self._count_tokens(sentence)
            if tokens_collected + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                tokens_collected += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback: approximate token count (1 token ≈ 4 characters)
        return len(text) // 4
    
    def _build_hierarchy_path(self, section_title: str, headings: List[Dict]) -> List[str]:
        """Build hierarchy path for a section based on headings"""
        hierarchy = []
        
        # Find the heading for this section
        target_heading = None
        for heading in headings:
            if heading['text'] == section_title:
                target_heading = heading
                break
        
        if not target_heading:
            return [section_title]
        
        target_level = target_heading['level']
        target_line = target_heading['line_number']
        
        # Build hierarchy by finding parent headings
        for heading in reversed(headings):
            if (heading['line_number'] < target_line and 
                heading['level'] < target_level):
                hierarchy.insert(0, heading['text'])
                target_level = heading['level']
        
        # Add the current section
        hierarchy.append(section_title)
        
        return hierarchy
    
    def _post_process_chunks(self, chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """Post-process chunks to ensure quality and consistency"""
        processed_chunks = []
        
        for chunk in chunks:
            # Skip chunks that are too small
            if chunk.token_count < self.config.min_chunk_size:
                logger.debug(f"Skipping chunk with {chunk.token_count} tokens (too small)")
                continue
            
            # Clean up text
            chunk.text = self._clean_chunk_text(chunk.text)
            
            # Recalculate token count after cleaning
            chunk.token_count = self._count_tokens(chunk.text)
            
            # Skip if still too small after cleaning
            if chunk.token_count < self.config.min_chunk_size:
                continue
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean and normalize chunk text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers if present
        text = re.sub(r'\[PAGE \d+\]\s*', '', text)
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 3 or not line:  # Keep empty lines and lines > 3 chars
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def split_text_simple(self, text: str, doc_id: str = "unknown") -> List[ChunkMetadata]:
        """
        Simple splitting method for quick use cases
        
        Args:
            text: Raw text to split
            doc_id: Document identifier
            
        Returns:
            List of ChunkMetadata objects
        """
        doc_metadata = {
            'doc_id': doc_id,
            'doc_type': 'text',
            'source_path': 'unknown'
        }
        
        return self._split_semantic(text, doc_metadata)
    
    def get_splitting_stats(self, chunks: List[ChunkMetadata]) -> Dict:
        """Get statistics about the splitting results"""
        if not chunks:
            return {"status": "no_chunks"}
        
        token_counts = [chunk.token_count for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "token_stats": {
                "total": sum(token_counts),
                "mean": sum(token_counts) / len(token_counts),
                "min": min(token_counts),
                "max": max(token_counts),
                "target_size": self.config.target_chunk_size
            },
            "structure_preservation": {
                "chunks_with_sections": sum(1 for c in chunks if c.section),
                "chunks_with_hierarchy": sum(1 for c in chunks if c.hierarchy_path),
                "unique_sections": len(set(c.section for c in chunks if c.section))
            },
            "config": {
                "target_chunk_size": self.config.target_chunk_size,
                "max_chunk_size": self.config.max_chunk_size,
                "min_chunk_size": self.config.min_chunk_size,
                "chunk_overlap": self.config.chunk_overlap
            }
        } 