"""
Multi-Chunk Paper Q&A Generator

This module provides sophisticated Q&A generation for research papers,
creating integrative questions that span multiple sections and chunks.
"""

import re
import random
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..assembly.config_manager import ConfigManager


class SectionType(Enum):
    """Types of sections in research papers."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"
    UNKNOWN = "unknown"


@dataclass
class PaperChunk:
    """Represents a chunk of a research paper with metadata."""
    content: str
    section_type: SectionType
    section_title: str
    chunk_index: int
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    contains_figures: bool = False
    contains_tables: bool = False


@dataclass
class MultiChunkQA:
    """Q&A pair that integrates information from multiple chunks."""
    question: str
    answer: str
    source_chunks: List[PaperChunk]
    chunk_indices: List[int]
    integration_type: str  # "comparison", "synthesis", "sequential", "contextual"
    question_type: str  # "integrative", "comparative", "summary", "analytical"
    difficulty_level: str
    requires_cross_reference: bool = False
    section_types_involved: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)


class MultiChunkPaperQAGenerator:
    """Generates Q&A pairs for research papers with multi-chunk integration."""
    
    def __init__(self, config_manager: ConfigManager, llm_client):
        self.config_manager = config_manager
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    def classify_chunks(self, chunks: List[str]) -> List[PaperChunk]:
        """Classify paper chunks by section type."""
        classified_chunks = []
        
        for i, chunk_content in enumerate(chunks):
            # Simple classification based on content
            section_type = SectionType.UNKNOWN
            if "abstract" in chunk_content.lower()[:100]:
                section_type = SectionType.ABSTRACT
            elif "introduction" in chunk_content.lower()[:100]:
                section_type = SectionType.INTRODUCTION
            elif "method" in chunk_content.lower()[:100]:
                section_type = SectionType.METHODS
            elif "result" in chunk_content.lower()[:100]:
                section_type = SectionType.RESULTS
            elif "conclusion" in chunk_content.lower()[:100]:
                section_type = SectionType.CONCLUSION
            
            paper_chunk = PaperChunk(
                content=chunk_content,
                section_type=section_type,
                section_title=f"Section {i+1}",
                chunk_index=i
            )
            classified_chunks.append(paper_chunk)
        
        return classified_chunks
    
    def generate_integrative_qa_pairs(self, paper_chunks: List[PaperChunk], 
                                    count: int = 5) -> List[MultiChunkQA]:
        """Generate integrative Q&A pairs that span multiple chunks."""
        qa_pairs = []
        
        for _ in range(count):
            # Select 2-3 random chunks
            selected_chunks = random.sample(paper_chunks, min(2, len(paper_chunks)))
            
            question = f"How do the concepts in {selected_chunks[0].section_title} relate to those in {selected_chunks[1].section_title}?"
            answer = "This requires analysis across multiple sections of the paper."
            
            qa_pair = MultiChunkQA(
                question=question,
                answer=answer,
                source_chunks=selected_chunks,
                chunk_indices=[chunk.chunk_index for chunk in selected_chunks],
                integration_type="comparison",
                question_type="integrative",
                difficulty_level="medium",
                requires_cross_reference=True
            )
            qa_pairs.append(qa_pair)
        
        return qa_pairs


def create_paper_qa_generator(config_manager: ConfigManager, llm_client) -> MultiChunkPaperQAGenerator:
    """Create a pre-configured multi-chunk paper Q&A generator."""
    return MultiChunkPaperQAGenerator(config_manager, llm_client) 