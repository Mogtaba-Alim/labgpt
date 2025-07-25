#!/usr/bin/env python3
"""
advanced_scoring.py

Advanced chunk scoring and pruning system with domain-specific metrics.
Builds on the basic QualityFilter to provide sophisticated content assessment.
"""

import re
import logging
import math
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from .chunk_objects import ChunkMetadata
from .quality_filter import QualityFilter

logger = logging.getLogger(__name__)

@dataclass
class ScoringConfig:
    """Configuration for advanced scoring system"""
    # Domain-specific weights
    research_weight: float = 1.2
    technical_weight: float = 1.1
    general_weight: float = 1.0
    
    # Scoring thresholds
    min_information_density: float = 0.3
    max_repetition_ratio: float = 0.7
    min_semantic_coherence: float = 0.4
    
    # Pruning thresholds  
    aggressive_pruning_threshold: float = 0.2
    conservative_pruning_threshold: float = 0.4
    
    # Content type specific settings
    enable_code_scoring: bool = True
    enable_math_scoring: bool = True
    enable_table_scoring: bool = True

class AdvancedChunkScorer:
    """
    Advanced scoring system for chunks with domain-specific metrics.
    
    Provides sophisticated assessment beyond basic quality filtering:
    - Information density analysis
    - Semantic coherence scoring  
    - Domain-specific value assessment
    - Content type optimization
    - Intelligent pruning strategies
    """
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        
        # Initialize base quality filter
        self.base_filter = QualityFilter()
        
        # Domain-specific scoring patterns
        self.domain_indicators = {
            'research': {
                'high_value': [
                    r'\b(?:hypothesis|methodology|experiment|analysis|results?|conclusion|findings?)\b',
                    r'\b(?:significant|correlation|statistical|p-value|confidence)\b',
                    r'\b(?:study|research|investigation|survey|trial|observation)\b',
                    r'\b(?:data|evidence|measurement|assessment|evaluation)\b'
                ],
                'medium_value': [
                    r'\b(?:background|literature|review|previous|prior)\b',
                    r'\b(?:method|approach|technique|procedure)\b',
                    r'\b(?:discussion|interpretation|implication)\b'
                ],
                'low_value': [
                    r'\b(?:acknowledgment|funding|conflict|interest)\b',
                    r'\b(?:appendix|supplementary|additional)\b'
                ]
            },
            'technical': {
                'high_value': [
                    r'\b(?:algorithm|implementation|architecture|design)\b',
                    r'\b(?:performance|optimization|efficiency|scalability)\b',
                    r'\b(?:function|method|class|module|interface)\b',
                    r'\b(?:data|structure|model|framework|system)\b'
                ],
                'medium_value': [
                    r'\b(?:configuration|setup|installation|deployment)\b',
                    r'\b(?:example|demonstration|tutorial|guide)\b',
                    r'\b(?:specification|requirement|standard)\b'
                ],
                'low_value': [
                    r'\b(?:changelog|version|update|release)\b',
                    r'\b(?:license|copyright|disclaimer)\b'
                ]
            },
            'biological': {
                'high_value': [
                    r'\b(?:protein|gene|DNA|RNA|sequence|structure)\b',
                    r'\b(?:cell|tissue|organ|organism|species)\b',
                    r'\b(?:pathway|interaction|binding|expression)\b',
                    r'\b(?:function|mechanism|process|regulation)\b'
                ],
                'medium_value': [
                    r'\b(?:experiment|assay|analysis|measurement)\b',
                    r'\b(?:condition|treatment|culture|medium)\b',
                    r'\b(?:sample|specimen|material|preparation)\b'
                ],
                'low_value': [
                    r'\b(?:buffer|solution|reagent|equipment)\b',
                    r'\b(?:protocol|procedure|step|instruction)\b'
                ]
            }
        }
        
        # Content type patterns
        self.content_patterns = {
            'code': [
                r'def\s+\w+\s*\(',
                r'class\s+\w+\s*[:\(]',
                r'import\s+\w+',
                r'from\s+\w+\s+import',
                r'if\s+.*:\s*$',
                r'for\s+\w+\s+in\s+',
                r'while\s+.*:\s*$',
                r'return\s+',
                r'print\s*\(',
                r'=\s*\[.*\]',
                r'=\s*\{.*\}'
            ],
            'math': [
                r'\$.*\$',
                r'\\begin\{.*\}',
                r'\\end\{.*\}',
                r'\\frac\{.*\}\{.*\}',
                r'\\sum_',
                r'\\int_',
                r'\\alpha|\\beta|\\gamma|\\delta|\\epsilon',
                r'\\mathbf\{.*\}',
                r'\\partial',
                r'\\nabla'
            ],
            'table': [
                r'\|.*\|.*\|',
                r'^\s*[-+|=\s]+$',
                r'^\s*\w+\s*\|\s*\w+',
                r'&\s*\\\\',
                r'\\hline',
                r'\\begin\{table\}',
                r'\\end\{table\}'
            ],
            'citation': [
                r'\[(\d+)\]',
                r'\([A-Za-z]+\s+et\s+al\.?,\s*\d{4}\)',
                r'\([A-Za-z]+,\s*\d{4}\)',
                r'doi:\s*\S+',
                r'arxiv:\s*\S+',
                r'pmid:\s*\d+'
            ]
        }
        
        # Noise patterns (indicating low-quality content)
        self.noise_patterns = [
            r'^[^a-zA-Z]*$',          # No letters
            r'^\s*$',                 # Only whitespace  
            r'^(.)\1{10,}',           # Repeated characters
            r'^[^\w\s]{10,}',         # Many non-word characters
            r'\b\w{1,2}\b.*\b\w{1,2}\b.*\b\w{1,2}\b',  # Many very short words
            r'(error|warning|exception|traceback)',     # Error messages
            r'^\d+[\.\)]\s*$',        # Lone numbering
            r'^(page|figure|table)\s+\d+\s*$',         # Lone references
        ]
    
    def score_chunk_advanced(self, chunk: ChunkMetadata) -> ChunkMetadata:
        """
        Apply advanced scoring to a chunk
        
        Args:
            chunk: ChunkMetadata object to score
            
        Returns:
            Enhanced ChunkMetadata with advanced scores
        """
        text = chunk.text
        
        # Start with base quality assessment
        chunk = self.base_filter.assess_chunk_quality(chunk)
        
        # Apply advanced scoring metrics
        information_density = self._calculate_information_density(text)
        semantic_coherence = self._calculate_semantic_coherence(text)
        domain_value = self._calculate_domain_value(text, chunk.doc_type)
        content_type_score = self._calculate_content_type_score(text)
        repetition_penalty = self._calculate_repetition_penalty(text)
        noise_penalty = self._calculate_noise_penalty(text)
        
        # Calculate composite advanced score
        advanced_score = self._calculate_composite_score(
            base_quality=chunk.quality_score,
            information_density=information_density,
            semantic_coherence=semantic_coherence,
            domain_value=domain_value,
            content_type_score=content_type_score,
            repetition_penalty=repetition_penalty,
            noise_penalty=noise_penalty
        )
        
        # Store advanced metrics in chunk metadata
        chunk.quality_score = advanced_score
        
        # Add detailed scoring breakdown (would need to extend ChunkMetadata)
        if not hasattr(chunk, 'advanced_scores'):
            chunk.advanced_scores = {}
        
        chunk.advanced_scores = {
            'information_density': information_density,
            'semantic_coherence': semantic_coherence,
            'domain_value': domain_value,
            'content_type_score': content_type_score,
            'repetition_penalty': repetition_penalty,
            'noise_penalty': noise_penalty,
            'composite_score': advanced_score
        }
        
        logger.debug(f"Advanced scoring - Chunk {chunk.chunk_id}: {advanced_score:.3f}")
        
        return chunk
    
    def _calculate_information_density(self, text: str) -> float:
        """Calculate information density using entropy-based metrics"""
        if not text or len(text) < 10:
            return 0.0
        
        # Tokenize and calculate entropy
        words = word_tokenize(text.lower())
        if len(words) < 5:
            return 0.0
        
        # Calculate word frequency entropy
        word_counts = Counter(words)
        total_words = len(words)
        entropy = 0.0
        
        for count in word_counts.values():
            probability = count / total_words
            entropy -= probability * math.log2(probability)
        
        # Normalize entropy by maximum possible entropy
        max_entropy = math.log2(len(word_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate unique content ratio
        unique_words = len(word_counts)
        unique_ratio = unique_words / total_words
        
        # Combine metrics
        information_density = (normalized_entropy * 0.6) + (unique_ratio * 0.4)
        
        return min(information_density, 1.0)
    
    def _calculate_semantic_coherence(self, text: str) -> float:
        """Calculate semantic coherence using linguistic patterns"""
        if not text or len(text) < 20:
            return 0.0
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.5  # Single sentence gets moderate score
        
        # Calculate average sentence length variance
        sentence_lengths = [len(s.split()) for s in sentences]
        if not sentence_lengths:
            return 0.0
        
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        length_score = 1.0 / (1.0 + length_variance / 100)  # Penalize high variance
        
        # Check for discourse markers
        discourse_markers = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'additionally', 'similarly', 'conversely', 'nevertheless', 'thus',
            'hence', 'accordingly', 'specifically', 'namely', 'particularly'
        ]
        
        marker_count = sum(1 for marker in discourse_markers if marker in text.lower())
        marker_score = min(marker_count / len(sentences), 0.5)  # Cap at 0.5
        
        # Check for pronoun coherence
        pronouns = ['it', 'they', 'this', 'that', 'these', 'those', 'which', 'who']
        pronoun_count = sum(1 for pronoun in pronouns if f' {pronoun} ' in text.lower())
        pronoun_score = min(pronoun_count / len(sentences), 0.3)  # Cap at 0.3
        
        # Combine coherence metrics
        coherence = (length_score * 0.5) + (marker_score * 0.3) + (pronoun_score * 0.2)
        
        return min(coherence, 1.0)
    
    def _calculate_domain_value(self, text: str, doc_type: str = "") -> float:
        """Calculate domain-specific value score"""
        text_lower = text.lower()
        
        # Determine likely domain based on content
        domain_scores = {}
        for domain, patterns in self.domain_indicators.items():
            score = 0.0
            
            # High value patterns
            for pattern in patterns['high_value']:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 3.0
            
            # Medium value patterns
            for pattern in patterns['medium_value']:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 2.0
            
            # Low value patterns (negative weight)
            for pattern in patterns['low_value']:
                matches = len(re.findall(pattern, text_lower))
                score -= matches * 1.0
            
            # Normalize by text length
            domain_scores[domain] = score / (len(text) / 1000) if len(text) > 0 else 0
        
        # Get maximum domain score
        max_score = max(domain_scores.values()) if domain_scores else 0
        
        # Apply domain-specific weighting
        if max_score > 0:
            best_domain = max(domain_scores, key=domain_scores.get)
            if best_domain == 'research':
                max_score *= self.config.research_weight
            elif best_domain == 'technical':
                max_score *= self.config.technical_weight
            else:
                max_score *= self.config.general_weight
        
        # Normalize to 0-1 range
        return min(max_score / 5.0, 1.0)  # Assuming max raw score of 5
    
    def _calculate_content_type_score(self, text: str) -> float:
        """Calculate score based on special content types"""
        total_score = 0.0
        
        # Score different content types
        for content_type, patterns in self.content_patterns.items():
            type_score = 0.0
            
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.MULTILINE))
                type_score += matches
            
            # Apply type-specific weighting
            if content_type == 'code' and self.config.enable_code_scoring:
                total_score += type_score * 0.8  # Code is valuable
            elif content_type == 'math' and self.config.enable_math_scoring:
                total_score += type_score * 1.0  # Math is very valuable
            elif content_type == 'table' and self.config.enable_table_scoring:
                total_score += type_score * 0.6  # Tables are moderately valuable
            elif content_type == 'citation':
                total_score += type_score * 0.4  # Citations add some value
        
        # Normalize by text length
        normalized_score = total_score / (len(text) / 500) if len(text) > 0 else 0
        
        return min(normalized_score, 1.0)
    
    def _calculate_repetition_penalty(self, text: str) -> float:
        """Calculate penalty for excessive repetition"""
        if not text or len(text) < 50:
            return 0.0
        
        # Check for repeated lines
        lines = text.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        
        if len(lines) > 0:
            line_repetition = 1.0 - (len(unique_lines) / len(lines))
        else:
            line_repetition = 0.0
        
        # Check for repeated phrases
        words = text.split()
        if len(words) >= 6:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            unique_trigrams = set(trigrams)
            trigram_repetition = 1.0 - (len(unique_trigrams) / len(trigrams))
        else:
            trigram_repetition = 0.0
        
        # Combine repetition measures
        repetition_ratio = (line_repetition * 0.7) + (trigram_repetition * 0.3)
        
        # Convert to penalty (higher repetition = higher penalty)
        penalty = repetition_ratio * 2.0  # Scale penalty
        
        return min(penalty, 1.0)
    
    def _calculate_noise_penalty(self, text: str) -> float:
        """Calculate penalty for noise patterns"""
        if not text:
            return 1.0  # Maximum penalty for empty text
        
        penalty = 0.0
        
        # Check each noise pattern
        for pattern in self.noise_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                penalty += 0.2
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if punct_ratio > 0.3:
            penalty += (punct_ratio - 0.3) * 2.0
        
        # Check for excessive numbers
        digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
        if digit_ratio > 0.4:
            penalty += (digit_ratio - 0.4) * 1.5
        
        return min(penalty, 1.0)
    
    def _calculate_composite_score(self, 
                                  base_quality: float,
                                  information_density: float,
                                  semantic_coherence: float,
                                  domain_value: float,
                                  content_type_score: float,
                                  repetition_penalty: float,
                                  noise_penalty: float) -> float:
        """Calculate final composite score"""
        
        # Base score from quality assessment
        score = base_quality * 0.3
        
        # Add information content
        score += information_density * 0.25
        
        # Add coherence
        score += semantic_coherence * 0.2
        
        # Add domain value
        score += domain_value * 0.15
        
        # Add content type bonus
        score += content_type_score * 0.1
        
        # Apply penalties
        score *= (1.0 - repetition_penalty * 0.5)  # Reduce impact of repetition
        score *= (1.0 - noise_penalty * 0.7)       # Strong penalty for noise
        
        return max(0.0, min(score, 1.0))

class IntelligentPruner:
    """
    Intelligent pruning system that uses advanced scores to make pruning decisions
    """
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
    
    def prune_chunks(self, chunks: List[ChunkMetadata], 
                    pruning_strategy: str = "adaptive") -> List[ChunkMetadata]:
        """
        Prune chunks using intelligent strategies
        
        Args:
            chunks: List of scored chunks
            pruning_strategy: "aggressive", "conservative", "adaptive"
            
        Returns:
            Pruned list of chunks
        """
        if not chunks:
            return chunks
        
        if pruning_strategy == "aggressive":
            threshold = self.config.aggressive_pruning_threshold
        elif pruning_strategy == "conservative":
            threshold = self.config.conservative_pruning_threshold
        else:  # adaptive
            threshold = self._calculate_adaptive_threshold(chunks)
        
        # Apply basic threshold pruning
        pruned = [chunk for chunk in chunks if chunk.quality_score >= threshold]
        
        # Apply additional intelligent filters
        pruned = self._apply_diversity_pruning(pruned)
        pruned = self._apply_redundancy_removal(pruned)
        
        logger.info(f"Pruned {len(chunks) - len(pruned)} chunks using {pruning_strategy} strategy")
        
        return pruned
    
    def _calculate_adaptive_threshold(self, chunks: List[ChunkMetadata]) -> float:
        """Calculate adaptive threshold based on score distribution"""
        scores = [chunk.quality_score for chunk in chunks]
        
        if not scores:
            return self.config.conservative_pruning_threshold
        
        # Use percentile-based threshold
        scores.sort()
        percentile_25 = scores[len(scores) // 4] if len(scores) > 4 else min(scores)
        median = scores[len(scores) // 2]
        
        # Adaptive threshold between 25th percentile and median
        adaptive_threshold = (percentile_25 + median) / 2
        
        # Ensure within reasonable bounds
        adaptive_threshold = max(adaptive_threshold, self.config.aggressive_pruning_threshold)
        adaptive_threshold = min(adaptive_threshold, self.config.conservative_pruning_threshold)
        
        return adaptive_threshold
    
    def _apply_diversity_pruning(self, chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """Remove chunks that are too similar to higher-scoring ones"""
        if len(chunks) <= 1:
            return chunks
        
        # Sort by score (descending)
        sorted_chunks = sorted(chunks, key=lambda x: x.quality_score, reverse=True)
        
        diverse_chunks = [sorted_chunks[0]]  # Always keep highest scoring
        
        for chunk in sorted_chunks[1:]:
            # Check similarity with already selected chunks
            is_diverse = True
            for selected in diverse_chunks:
                similarity = self._calculate_text_similarity(chunk.text, selected.text)
                if similarity > 0.85:  # High similarity threshold
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_chunks.append(chunk)
        
        return diverse_chunks
    
    def _apply_redundancy_removal(self, chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """Remove redundant chunks based on content overlap"""
        if len(chunks) <= 1:
            return chunks
        
        non_redundant = []
        
        for chunk in chunks:
            is_redundant = False
            
            for existing in non_redundant:
                # Check for high content overlap
                overlap = self._calculate_content_overlap(chunk.text, existing.text)
                if overlap > 0.7:  # 70% overlap threshold
                    # Keep the higher scoring chunk
                    if chunk.quality_score > existing.quality_score:
                        non_redundant.remove(existing)
                        non_redundant.append(chunk)
                    is_redundant = True
                    break
            
            if not is_redundant:
                non_redundant.append(chunk)
        
        return non_redundant
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_content_overlap(self, text1: str, text2: str) -> float:
        """Calculate content overlap ratio"""
        # Split into sentences
        sentences1 = set(re.split(r'[.!?]+', text1.lower()))
        sentences2 = set(re.split(r'[.!?]+', text2.lower()))
        
        # Remove empty sentences
        sentences1 = {s.strip() for s in sentences1 if s.strip()}
        sentences2 = {s.strip() for s in sentences2 if s.strip()}
        
        if not sentences1 or not sentences2:
            return 0.0
        
        intersection = len(sentences1.intersection(sentences2))
        smaller_set = min(len(sentences1), len(sentences2))
        
        return intersection / smaller_set if smaller_set > 0 else 0.0
    
    def get_pruning_statistics(self, original_chunks: List[ChunkMetadata], 
                              pruned_chunks: List[ChunkMetadata]) -> Dict:
        """Get statistics about the pruning process"""
        original_count = len(original_chunks)
        pruned_count = len(pruned_chunks)
        removed_count = original_count - pruned_count
        
        if original_count == 0:
            return {"status": "no_chunks"}
        
        # Calculate score distributions
        original_scores = [c.quality_score for c in original_chunks]
        pruned_scores = [c.quality_score for c in pruned_chunks]
        
        return {
            "original_count": original_count,
            "pruned_count": pruned_count,
            "removed_count": removed_count,
            "removal_ratio": removed_count / original_count,
            "score_improvement": {
                "original_mean": np.mean(original_scores),
                "pruned_mean": np.mean(pruned_scores) if pruned_scores else 0,
                "original_min": np.min(original_scores),
                "pruned_min": np.min(pruned_scores) if pruned_scores else 0,
                "improvement": (np.mean(pruned_scores) - np.mean(original_scores)) if pruned_scores else 0
            }
        } 