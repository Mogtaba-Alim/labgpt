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
        """
        Generate integrative Q&A pairs that span multiple chunks.
        Uses two strategies:
        1. Random chunk combinations (2, 4, 6 chunks) with uniqueness tracking
        2. Consecutive window sampling (2, 4, 6 chunk windows)
        """
        qa_pairs = []

        # Track used chunk combinations to avoid duplicates
        used_combinations: Set[Tuple[int, ...]] = set()

        # Strategy 1: Random chunk combinations
        # - Max 10 QAs using 2 chunks
        # - Max 5 QAs using 4 chunks
        # - Max 3 QAs using 6 chunks
        random_qa_config = [
            (2, min(10, count // 3)),      # 2 chunks, up to 10 QAs
            (4, min(5, count // 4)),       # 4 chunks, up to 5 QAs
            (6, min(3, count // 5))        # 6 chunks, up to 3 QAs
        ]

        for chunk_count, max_qa in random_qa_config:
            generated_count = 0
            attempts = 0
            max_attempts = max_qa * 10  # Allow more attempts to find unique combinations

            while generated_count < max_qa and attempts < max_attempts:
                attempts += 1

                # Randomly select chunks
                selected_chunks = self._select_random_chunks(paper_chunks, chunk_count, used_combinations)

                if not selected_chunks:
                    continue

                # Generate question and answer
                question, answer = self._generate_integrative_question_answer(selected_chunks)

                if question and answer:
                    qa_pair = MultiChunkQA(
                        question=question,
                        answer=answer,
                        source_chunks=selected_chunks,
                        chunk_indices=[chunk.chunk_index for chunk in selected_chunks],
                        integration_type=self._determine_integration_type(selected_chunks),
                        question_type="integrative_random",
                        difficulty_level=self._determine_difficulty(chunk_count),
                        requires_cross_reference=True,
                        section_types_involved=[chunk.section_type.value for chunk in selected_chunks]
                    )
                    qa_pairs.append(qa_pair)
                    generated_count += 1

                    # Mark this combination as used
                    combo_tuple = tuple(sorted([chunk.chunk_index for chunk in selected_chunks]))
                    used_combinations.add(combo_tuple)

        # Strategy 2: Consecutive window sampling
        # - 1 QA per 2-chunk window
        # - 2 QAs per 4-chunk window
        # - 3 QAs per 6-chunk window
        window_qa_config = [
            (2, 1),  # 2-chunk windows, 1 QA each
            (4, 2),  # 4-chunk windows, 2 QAs each
            (6, 3)   # 6-chunk windows, 3 QAs each
        ]

        for window_size, qa_per_window in window_qa_config:
            windows = self._create_consecutive_windows(paper_chunks, window_size)

            for window in windows:
                if not window:
                    continue

                # Generate multiple QAs for this window
                for _ in range(qa_per_window):
                    question, answer = self._generate_integrative_question_answer(window)

                    if question and answer:
                        qa_pair = MultiChunkQA(
                            question=question,
                            answer=answer,
                            source_chunks=window,
                            chunk_indices=[chunk.chunk_index for chunk in window],
                            integration_type=self._determine_integration_type(window),
                            question_type="integrative_consecutive",
                            difficulty_level=self._determine_difficulty(window_size),
                            requires_cross_reference=True,
                            section_types_involved=[chunk.section_type.value for chunk in window]
                        )
                        qa_pairs.append(qa_pair)

        return qa_pairs
    
    def _generate_integrative_question_answer(self, chunks: List[PaperChunk]) -> tuple[str, str]:
        """Generate an integrative question and answer using GPT-4o."""

        system_prompt = """You are a research expert who creates high-quality questions and answers about academic papers.

Your task is to generate integrative questions that require synthesizing information across multiple sections of a research paper. The questions should:
1. Require understanding of concepts from multiple sections
2. Test comprehension of relationships between different parts of the paper
3. Be answerable using the provided content
4. Be educational and valuable for training a research assistant AI

CRITICAL: When generating the answer, write naturally and directly. Do NOT use phrases like:
- "Based on the provided sections"
- "According to the paper"
- "The sections describe"
- "From the content provided"

Instead, answer directly as if you're explaining the research. State the information clearly and naturally.

Example of good answer: "The study uses a randomized controlled trial design with 500 participants divided into treatment and control groups."
Example of bad answer: "Based on the provided sections, the study uses a randomized controlled trial design..."

Generate exactly one question and one comprehensive answer."""

        # Prepare context from chunks
        chunk_contexts = []
        for i, chunk in enumerate(chunks):
            chunk_contexts.append(f"**Section {i+1} ({chunk.section_type.value}):**\n{chunk.content[:1000]}...")

        user_prompt = f"""Research paper sections:

{chr(10).join(chunk_contexts)}

Generate:
1. One integrative question that requires understanding multiple sections
2. One comprehensive answer that synthesizes information from the sections (answer directly without meta-references)

Format your response as:
QUESTION: [your question]
ANSWER: [your detailed answer]"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.2
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse question and answer
            question, answer = self._parse_question_answer(response_text)
            return question, answer
            
        except Exception as e:
            self.logger.error(f"Error generating integrative QA: {e}")
            return "", ""
    
    def _parse_question_answer(self, response_text: str) -> tuple[str, str]:
        """Parse question and answer from GPT-4o response."""
        try:
            lines = response_text.split('\n')
            question = ""
            answer = ""
            current_section = None
            
            for line in lines:
                if line.strip().startswith('QUESTION:'):
                    current_section = 'question'
                    question = line.replace('QUESTION:', '').strip()
                elif line.strip().startswith('ANSWER:'):
                    current_section = 'answer'
                    answer = line.replace('ANSWER:', '').strip()
                elif current_section == 'question' and line.strip():
                    question += ' ' + line.strip()
                elif current_section == 'answer' and line.strip():
                    answer += ' ' + line.strip()
            
            return question.strip(), answer.strip()
            
        except Exception as e:
            self.logger.error(f"Error parsing QA response: {e}")
            return "", ""
    
    def _determine_integration_type(self, chunks: List[PaperChunk]) -> str:
        """Determine the type of integration based on chunk sections."""
        section_types = [chunk.section_type for chunk in chunks]

        if SectionType.INTRODUCTION in section_types and SectionType.CONCLUSION in section_types:
            return "contextual"
        elif SectionType.METHODS in section_types and SectionType.RESULTS in section_types:
            return "sequential"
        elif len(set(section_types)) == len(section_types):
            return "synthesis"
        else:
            return "comparison"

    def _select_random_chunks(self, paper_chunks: List[PaperChunk],
                             chunk_count: int,
                             used_combinations: Set[Tuple[int, ...]]) -> List[PaperChunk]:
        """
        Randomly select chunk_count chunks from paper_chunks.
        Ensures the combination hasn't been used before.
        """
        if len(paper_chunks) < chunk_count:
            return []

        # Try to find a unique combination
        max_attempts = 50
        for _ in range(max_attempts):
            selected = random.sample(paper_chunks, chunk_count)
            combo_tuple = tuple(sorted([chunk.chunk_index for chunk in selected]))

            if combo_tuple not in used_combinations:
                return selected

        # If we couldn't find a unique combination, return empty
        return []

    def _create_consecutive_windows(self, paper_chunks: List[PaperChunk],
                                   window_size: int) -> List[List[PaperChunk]]:
        """
        Create non-overlapping consecutive windows of chunks.
        Each window contains window_size consecutive chunks.
        """
        windows = []

        # Create non-overlapping windows by stepping through chunks
        for i in range(0, len(paper_chunks), window_size):
            window = paper_chunks[i:i + window_size]
            # Only include complete windows
            if len(window) == window_size:
                windows.append(window)

        return windows

    def _determine_difficulty(self, chunk_count: int) -> str:
        """Determine difficulty level based on number of chunks."""
        if chunk_count <= 2:
            return "medium"
        elif chunk_count <= 4:
            return "hard"
        else:
            return "very_hard"


def create_paper_qa_generator(config_manager: ConfigManager, llm_client) -> MultiChunkPaperQAGenerator:
    """Create a pre-configured multi-chunk paper Q&A generator."""
    return MultiChunkPaperQAGenerator(config_manager, llm_client) 