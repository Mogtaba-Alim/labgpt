#!/usr/bin/env python3
"""
quality_filter.py

Quality filtering and scoring for text chunks to remove low-quality content
and noise during ingestion.
"""

import re
import logging
from typing import List, Dict, Optional, Set
import string
from collections import Counter

import nltk
from nltk.corpus import stopwords

from .chunk_objects import ChunkMetadata

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

class QualityFilter:
    """
    Quality assessment and filtering for text chunks.
    
    Evaluates chunks based on:
    - Content density (unique tokens vs total length)
    - Stopword ratio
    - Symbol coverage (for technical content)
    - Language detection
    - Noise patterns
    """
    
    def __init__(self, 
                 min_quality_score: float = 0.3,
                 language: str = 'english'):
        self.min_quality_score = min_quality_score
        self.language = language
        
        # Load stopwords
        try:
            self.stopwords = set(stopwords.words(language))
        except:
            self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Common noise patterns
        self.noise_patterns = [
            r'^[^a-zA-Z]*$',  # No letters
            r'^\s*$',         # Only whitespace
            r'^(.)\1{10,}',   # Repeated characters
            r'^\d+$',         # Only numbers
            r'^[^\w\s]*$',    # Only symbols
        ]
        
        # Technical/code indicators
        self.code_indicators = {
            'programming': ['def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'else:', 'for ', 'while '],
            'math': ['$$', '\\begin{', '\\end{', '\\sum', '\\int', '\\frac'],
            'citation': ['[1]', '[2]', 'et al.', 'doi:', 'arxiv:', 'www.'],
            'url': ['http://', 'https://', 'www.', '.com', '.org', '.edu']
        }
    
    def assess_chunk_quality(self, chunk: ChunkMetadata) -> ChunkMetadata:
        """
        Assess and score a chunk's quality, updating the chunk metadata
        
        Args:
            chunk: ChunkMetadata object to assess
            
        Returns:
            Updated ChunkMetadata with quality scores
        """
        text = chunk.text
        
        # Calculate individual quality metrics
        density_score = self._calculate_density_score(text)
        stopword_ratio = self._calculate_stopword_ratio(text)
        symbol_coverage = self._calculate_symbol_coverage(text)
        noise_score = self._calculate_noise_score(text)
        length_score = self._calculate_length_score(text, chunk.token_count)
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_quality(
            density_score, stopword_ratio, symbol_coverage, 
            noise_score, length_score, text
        )
        
        # Update chunk metadata
        chunk.quality_score = quality_score
        chunk.density_score = density_score
        chunk.stopword_ratio = stopword_ratio
        chunk.symbol_coverage = symbol_coverage
        
        logger.debug(f"Quality assessment - Quality: {quality_score:.3f}, "
                    f"Density: {density_score:.3f}, Stopwords: {stopword_ratio:.3f}")
        
        return chunk
    
    def filter_chunks(self, chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """
        Filter chunks based on quality thresholds
        
        Args:
            chunks: List of ChunkMetadata objects
            
        Returns:
            Filtered list of high-quality chunks
        """
        filtered_chunks = []
        
        for chunk in chunks:
            # Assess quality if not already done
            if chunk.quality_score == 0.0:
                chunk = self.assess_chunk_quality(chunk)
            
            # Apply filters
            if self._should_keep_chunk(chunk):
                filtered_chunks.append(chunk)
            else:
                logger.debug(f"Filtered out chunk with quality {chunk.quality_score:.3f}")
        
        logger.info(f"Quality filtering: {len(filtered_chunks)}/{len(chunks)} chunks kept")
        return filtered_chunks
    
    def _calculate_density_score(self, text: str) -> float:
        """Calculate content density (unique tokens / total length)"""
        if not text or len(text) < 10:
            return 0.0
        
        # Tokenize and count unique tokens
        tokens = re.findall(r'\b\w+\b', text.lower())
        if not tokens:
            return 0.0
        
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        # Density = unique tokens / total tokens
        density = unique_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Normalize to 0-1 scale (typical range is 0.3-0.8)
        normalized_density = min(density / 0.8, 1.0)
        
        return normalized_density
    
    def _calculate_stopword_ratio(self, text: str) -> float:
        """Calculate ratio of stopwords to total words"""
        tokens = re.findall(r'\b\w+\b', text.lower())
        if not tokens:
            return 1.0  # All stopwords if no tokens
        
        stopword_count = sum(1 for token in tokens if token in self.stopwords)
        return stopword_count / len(tokens)
    
    def _calculate_symbol_coverage(self, text: str) -> float:
        """Calculate coverage of symbols/punctuation (useful for technical content)"""
        if not text:
            return 0.0
        
        symbol_count = sum(1 for char in text if char in string.punctuation)
        return symbol_count / len(text)
    
    def _calculate_noise_score(self, text: str) -> float:
        """Calculate noise score (0 = no noise, 1 = high noise)"""
        noise_score = 0.0
        
        # Check for noise patterns
        for pattern in self.noise_patterns:
            if re.search(pattern, text):
                noise_score += 0.3
        
        # Check for excessive repetition
        lines = text.split('\n')
        if len(lines) > 3:
            unique_lines = len(set(line.strip() for line in lines))
            repetition_ratio = 1 - (unique_lines / len(lines))
            noise_score += repetition_ratio * 0.4
        
        # Check for excessive short words
        tokens = text.split()
        if tokens:
            short_word_ratio = sum(1 for token in tokens if len(token) <= 2) / len(tokens)
            if short_word_ratio > 0.5:
                noise_score += 0.3
        
        return min(noise_score, 1.0)
    
    def _calculate_length_score(self, text: str, token_count: int) -> float:
        """Score based on text length (prefer moderate lengths)"""
        if token_count == 0:
            return 0.0
        
        # Ideal range: 100-500 tokens
        if 100 <= token_count <= 500:
            return 1.0
        elif 50 <= token_count < 100:
            return 0.8
        elif 500 < token_count <= 800:
            return 0.8
        elif 20 <= token_count < 50:
            return 0.6
        elif 800 < token_count <= 1000:
            return 0.6
        else:
            return 0.4
    
    def _calculate_overall_quality(self, 
                                 density_score: float,
                                 stopword_ratio: float, 
                                 symbol_coverage: float,
                                 noise_score: float,
                                 length_score: float,
                                 text: str) -> float:
        """Calculate overall quality score from individual metrics"""
        
        # Base quality from density and length
        base_quality = (density_score * 0.4) + (length_score * 0.3)
        
        # Adjust for stopword ratio (moderate ratio is good)
        stopword_adjustment = 1.0 - abs(stopword_ratio - 0.3)  # Ideal ~30% stopwords
        base_quality += stopword_adjustment * 0.2
        
        # Adjust for technical content (symbols can be good for code/math)
        content_type = self._detect_content_type(text)
        if content_type in ['programming', 'math']:
            # For technical content, some symbols are expected
            symbol_adjustment = min(symbol_coverage * 2, 0.1)
        else:
            # For regular text, too many symbols are bad
            symbol_adjustment = -min(symbol_coverage * 0.5, 0.1)
        
        base_quality += symbol_adjustment
        
        # Penalize for noise
        final_quality = base_quality * (1.0 - noise_score)
        
        # Bonus for well-structured content
        if self._has_good_structure(text):
            final_quality += 0.1
        
        return max(0.0, min(1.0, final_quality))
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content (regular, programming, math, etc.)"""
        text_lower = text.lower()
        
        # Count indicators for each type
        type_scores = {}
        for content_type, indicators in self.code_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            type_scores[content_type] = score
        
        # Return type with highest score, or 'regular' if no clear match
        if not type_scores or max(type_scores.values()) == 0:
            return 'regular'
        
        return max(type_scores, key=type_scores.get)
    
    def _has_good_structure(self, text: str) -> bool:
        """Check if text has good structural indicators"""
        # Look for paragraph breaks
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            return True
        
        # Look for sentence structure
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 2:
            avg_sentence_length = sum(len(s.strip().split()) for s in sentences) / len(sentences)
            if 5 < avg_sentence_length < 30:  # Reasonable sentence length
                return True
        
        # Look for list structures
        if re.search(r'^\s*[-*•]\s+', text, re.MULTILINE):
            return True
        
        # Look for numbered items
        if re.search(r'^\s*\d+[.)]\s+', text, re.MULTILINE):
            return True
        
        return False
    
    def _should_keep_chunk(self, chunk: ChunkMetadata) -> bool:
        """Determine if a chunk should be kept based on quality thresholds"""
        
        # Basic quality threshold
        if chunk.quality_score < self.min_quality_score:
            return False
        
        # Additional hard filters
        
        # Minimum text length
        if len(chunk.text.strip()) < 20:
            return False
        
        # Maximum noise ratio
        noise_score = self._calculate_noise_score(chunk.text)
        if noise_score > 0.8:
            return False
        
        # Check for completely garbled text
        if self._is_garbled_text(chunk.text):
            return False
        
        return True
    
    def _is_garbled_text(self, text: str) -> bool:
        """Check if text appears to be garbled or corrupted"""
        # Check for excessive non-ASCII characters
        non_ascii_ratio = sum(1 for char in text if ord(char) > 127) / len(text) if text else 0
        if non_ascii_ratio > 0.3:
            return True
        
        # Check for excessive consecutive non-letter characters
        non_letter_runs = re.findall(r'[^a-zA-Z\s]{5,}', text)
        if len(non_letter_runs) > 3:
            return True
        
        # Check for extremely high symbol density
        if len(text) > 50:
            symbol_ratio = sum(1 for char in text if not char.isalnum() and not char.isspace()) / len(text)
            if symbol_ratio > 0.6:
                return True
        
        return False
    
    def get_quality_statistics(self, chunks: List[ChunkMetadata]) -> Dict:
        """Get quality statistics for a list of chunks"""
        if not chunks:
            return {"status": "no_chunks"}
        
        quality_scores = [chunk.quality_score for chunk in chunks]
        density_scores = [chunk.density_score for chunk in chunks]
        stopword_ratios = [chunk.stopword_ratio for chunk in chunks]
        
        # Content type distribution
        content_types = Counter()
        for chunk in chunks:
            content_type = self._detect_content_type(chunk.text)
            content_types[content_type] += 1
        
        return {
            "total_chunks": len(chunks),
            "quality_scores": {
                "mean": sum(quality_scores) / len(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores),
                "above_threshold": sum(1 for q in quality_scores if q >= self.min_quality_score)
            },
            "density_scores": {
                "mean": sum(density_scores) / len(density_scores),
                "min": min(density_scores),
                "max": max(density_scores)
            },
            "stopword_ratios": {
                "mean": sum(stopword_ratios) / len(stopword_ratios),
                "min": min(stopword_ratios),
                "max": max(stopword_ratios)
            },
            "content_types": dict(content_types),
            "filter_threshold": self.min_quality_score
        } 