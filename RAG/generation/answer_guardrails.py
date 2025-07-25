#!/usr/bin/env python3
"""
answer_guardrails.py

Answer guardrails system with citation verification for factual accuracy and source attribution.
Implements post-generation verification to ensure each factual claim maps to retrieved sources.
"""

import re
import logging
import time
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

from ..ingestion.chunk_objects import ChunkMetadata

logger = logging.getLogger(__name__)

@dataclass
class FactualClaim:
    """Represents a factual claim in generated text"""
    sentence: str
    claim_text: str
    claim_type: str  # 'numeric', 'entity', 'relationship', 'definition'
    confidence: float
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    sentence_index: int = 0

@dataclass
class CitationMapping:
    """Represents mapping between claim and source chunk"""
    claim: FactualClaim
    chunk: ChunkMetadata
    similarity_score: float
    overlap_score: float
    confidence_score: float
    verification_method: str
    supporting_text: str = ""

@dataclass
class VerificationResult:
    """Result of citation verification process"""
    claim: FactualClaim
    is_verified: bool
    citations: List[CitationMapping]
    confidence: float
    failure_reason: Optional[str] = None

@dataclass
class GuardrailsConfig:
    """Configuration for answer guardrails"""
    # Verification thresholds
    min_similarity_threshold: float = 0.7
    min_overlap_threshold: float = 0.3
    min_confidence_threshold: float = 0.6
    
    # Claim detection
    min_claim_length: int = 10
    factual_claim_patterns: List[str] = field(default_factory=lambda: [
        r'\b\d+\.?\d*\s*(percent|%|million|billion|thousand|years?|days?|hours?)\b',
        r'\b(is|are|was|were|has|have|contains?|includes?)\b',
        r'\b(according to|research shows|studies indicate|data suggests)\b',
        r'\b(significantly|substantially|dramatically|markedly)\b'
    ])
    
    # Citation requirements
    require_citation_for_numbers: bool = True
    require_citation_for_entities: bool = True
    require_citation_for_claims: bool = True
    
    # Regeneration settings
    enable_regeneration: bool = True
    max_regeneration_attempts: int = 3
    regeneration_strictness_levels: List[str] = field(default_factory=lambda: [
        "moderate", "strict", "very_strict"
    ])

class FactualClaimExtractor:
    """
    Extracts factual claims from generated text for verification
    """
    
    def __init__(self, config: GuardrailsConfig):
        self.config = config
        
        # Initialize NLTK components
        self._init_nltk()
        
        # Compile factual claim patterns
        self.claim_patterns = [re.compile(pattern, re.IGNORECASE) 
                              for pattern in config.factual_claim_patterns]
        
        # Entity types that require citation
        self.citation_required_entities = {
            'PERSON', 'ORGANIZATION', 'GPE', 'DATE', 'TIME', 'PERCENT', 
            'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'
        }
    
    def _init_nltk(self):
        """Initialize required NLTK components"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('chunkers/maxent_ne_chunker')
            nltk.data.find('corpora/words')
        except LookupError:
            logger.warning("Some NLTK data missing. NER features may be limited.")
    
    def extract_factual_claims(self, text: str) -> List[FactualClaim]:
        """
        Extract factual claims from generated text
        
        Args:
            text: Generated text to analyze
            
        Returns:
            List of extracted factual claims
        """
        claims = []
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < self.config.min_claim_length:
                continue
            
            # Extract different types of claims
            sentence_claims = []
            
            # Numerical claims
            sentence_claims.extend(self._extract_numerical_claims(sentence, i))
            
            # Entity-based claims
            sentence_claims.extend(self._extract_entity_claims(sentence, i))
            
            # Pattern-based factual claims
            sentence_claims.extend(self._extract_pattern_claims(sentence, i))
            
            # Relationship claims
            sentence_claims.extend(self._extract_relationship_claims(sentence, i))
            
            claims.extend(sentence_claims)
        
        return claims
    
    def _extract_numerical_claims(self, sentence: str, sentence_index: int) -> List[FactualClaim]:
        """Extract claims containing numerical information"""
        claims = []
        
        # Pattern for numbers with units or context
        numerical_patterns = [
            r'\b\d+\.?\d*\s*(percent|%|million|billion|thousand|years?|days?|hours?|kg|km|cm|m)\b',
            r'\b\d+\.?\d*\s*(increased|decreased|rose|fell|grew|declined)\b',
            r'\b(approximately|roughly|about|over|under|more than|less than)\s+\d+\.?\d*\b'
        ]
        
        for pattern in numerical_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                claim_text = match.group()
                
                claim = FactualClaim(
                    sentence=sentence,
                    claim_text=claim_text,
                    claim_type='numeric',
                    confidence=0.8,  # High confidence for numerical claims
                    sentence_index=sentence_index
                )
                
                claims.append(claim)
        
        return claims
    
    def _extract_entity_claims(self, sentence: str, sentence_index: int) -> List[FactualClaim]:
        """Extract claims involving named entities"""
        claims = []
        
        try:
            # Tokenize and tag
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            # Named entity recognition
            tree = ne_chunk(pos_tags)
            
            entities = []
            for subtree in tree:
                if isinstance(subtree, Tree):
                    entity_text = ' '.join([token for token, pos in subtree.leaves()])
                    entity_type = subtree.label()
                    
                    if entity_type in self.citation_required_entities:
                        entities.append((entity_text, entity_type))
            
            # Create claims for sentences with entities
            if entities:
                claim = FactualClaim(
                    sentence=sentence,
                    claim_text=sentence,
                    claim_type='entity',
                    confidence=0.7,
                    entities=[entity[0] for entity in entities],
                    sentence_index=sentence_index
                )
                claims.append(claim)
        
        except Exception as e:
            logger.warning(f"Error extracting entity claims: {e}")
        
        return claims
    
    def _extract_pattern_claims(self, sentence: str, sentence_index: int) -> List[FactualClaim]:
        """Extract claims based on linguistic patterns"""
        claims = []
        
        for pattern in self.claim_patterns:
            matches = pattern.finditer(sentence)
            for match in matches:
                # Extract surrounding context for the claim
                start = max(0, match.start() - 50)
                end = min(len(sentence), match.end() + 50)
                claim_text = sentence[start:end].strip()
                
                claim = FactualClaim(
                    sentence=sentence,
                    claim_text=claim_text,
                    claim_type='factual_pattern',
                    confidence=0.6,
                    sentence_index=sentence_index
                )
                
                claims.append(claim)
        
        return claims
    
    def _extract_relationship_claims(self, sentence: str, sentence_index: int) -> List[FactualClaim]:
        """Extract claims expressing relationships or causality"""
        relationship_patterns = [
            r'\b(causes?|leads? to|results? in|due to|because of)\b',
            r'\b(associated with|correlated with|linked to|related to)\b',
            r'\b(compared to|in contrast to|versus|vs\.?)\b',
            r'\b(higher than|lower than|greater than|less than)\b'
        ]
        
        claims = []
        
        for pattern_str in relationship_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(sentence):
                claim = FactualClaim(
                    sentence=sentence,
                    claim_text=sentence,
                    claim_type='relationship',
                    confidence=0.6,
                    sentence_index=sentence_index
                )
                claims.append(claim)
                break  # Only one relationship claim per sentence
        
        return claims

class CitationVerifier:
    """
    Verifies factual claims against retrieved source chunks
    """
    
    def __init__(self, 
                 config: GuardrailsConfig,
                 embedding_model: Optional[SentenceTransformer] = None):
        self.config = config
        self.embedding_model = embedding_model
    
    def verify_claims(self, 
                     claims: List[FactualClaim],
                     source_chunks: List[ChunkMetadata]) -> List[VerificationResult]:
        """
        Verify factual claims against source chunks
        
        Args:
            claims: List of factual claims to verify
            source_chunks: List of source chunks for verification
            
        Returns:
            List of verification results
        """
        verification_results = []
        
        # Prepare chunk texts and embeddings if needed
        chunk_texts = [chunk.text for chunk in source_chunks]
        chunk_embeddings = None
        
        if self.embedding_model:
            chunk_embeddings = self.embedding_model.encode(chunk_texts)
        
        for claim in claims:
            result = self._verify_single_claim(claim, source_chunks, chunk_texts, chunk_embeddings)
            verification_results.append(result)
        
        return verification_results
    
    def _verify_single_claim(self, 
                           claim: FactualClaim,
                           source_chunks: List[ChunkMetadata],
                           chunk_texts: List[str],
                           chunk_embeddings: Optional[np.ndarray]) -> VerificationResult:
        """Verify a single factual claim"""
        
        citations = []
        
        # Try different verification methods
        for i, (chunk, chunk_text) in enumerate(zip(source_chunks, chunk_texts)):
            
            # Method 1: Lexical overlap
            overlap_score = self._calculate_lexical_overlap(claim.claim_text, chunk_text)
            
            # Method 2: Semantic similarity (if embeddings available)
            similarity_score = 0.0
            if chunk_embeddings is not None and self.embedding_model:
                claim_embedding = self.embedding_model.encode([claim.claim_text])
                similarity_score = cosine_similarity(
                    claim_embedding, 
                    chunk_embeddings[i:i+1]
                )[0][0]
            
            # Method 3: Entity overlap
            entity_overlap_score = self._calculate_entity_overlap(claim, chunk_text)
            
            # Combine scores
            confidence_score = self._calculate_combined_confidence(
                overlap_score, similarity_score, entity_overlap_score, claim.claim_type
            )
            
            # Check if citation meets threshold
            if (overlap_score >= self.config.min_overlap_threshold or
                similarity_score >= self.config.min_similarity_threshold or
                confidence_score >= self.config.min_confidence_threshold):
                
                # Extract supporting text
                supporting_text = self._extract_supporting_text(claim.claim_text, chunk_text)
                
                citation = CitationMapping(
                    claim=claim,
                    chunk=chunk,
                    similarity_score=similarity_score,
                    overlap_score=overlap_score,
                    confidence_score=confidence_score,
                    verification_method=self._determine_verification_method(
                        overlap_score, similarity_score, entity_overlap_score
                    ),
                    supporting_text=supporting_text
                )
                
                citations.append(citation)
        
        # Sort citations by confidence
        citations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Determine if claim is verified
        is_verified = len(citations) > 0 and citations[0].confidence_score >= self.config.min_confidence_threshold
        
        # Calculate overall confidence
        overall_confidence = citations[0].confidence_score if citations else 0.0
        
        failure_reason = None
        if not is_verified:
            if not citations:
                failure_reason = "No supporting citations found"
            else:
                failure_reason = f"Insufficient confidence: {overall_confidence:.2f} < {self.config.min_confidence_threshold}"
        
        return VerificationResult(
            claim=claim,
            is_verified=is_verified,
            citations=citations[:3],  # Keep top 3 citations
            confidence=overall_confidence,
            failure_reason=failure_reason
        )
    
    def _calculate_lexical_overlap(self, claim_text: str, chunk_text: str) -> float:
        """Calculate lexical overlap between claim and chunk"""
        try:
            claim_words = set(word_tokenize(claim_text.lower()))
            chunk_words = set(word_tokenize(chunk_text.lower()))
            
            # Remove stopwords and short words
            claim_words = {w for w in claim_words if len(w) > 2 and w.isalpha()}
            chunk_words = {w for w in chunk_words if len(w) > 2 and w.isalpha()}
            
            if not claim_words:
                return 0.0
            
            overlap = len(claim_words.intersection(chunk_words))
            return overlap / len(claim_words)
            
        except Exception as e:
            logger.warning(f"Error calculating lexical overlap: {e}")
            return 0.0
    
    def _calculate_entity_overlap(self, claim: FactualClaim, chunk_text: str) -> float:
        """Calculate entity overlap between claim and chunk"""
        if not claim.entities:
            return 0.0
        
        chunk_text_lower = chunk_text.lower()
        matching_entities = 0
        
        for entity in claim.entities:
            if entity.lower() in chunk_text_lower:
                matching_entities += 1
        
        return matching_entities / len(claim.entities)
    
    def _calculate_combined_confidence(self, 
                                     overlap_score: float,
                                     similarity_score: float,
                                     entity_overlap_score: float,
                                     claim_type: str) -> float:
        """Calculate combined confidence score"""
        
        # Different weights based on claim type
        if claim_type == 'numeric':
            weights = {'overlap': 0.4, 'similarity': 0.3, 'entity': 0.3}
        elif claim_type == 'entity':
            weights = {'overlap': 0.3, 'similarity': 0.3, 'entity': 0.4}
        else:
            weights = {'overlap': 0.35, 'similarity': 0.45, 'entity': 0.2}
        
        combined_score = (
            overlap_score * weights['overlap'] +
            similarity_score * weights['similarity'] +
            entity_overlap_score * weights['entity']
        )
        
        return combined_score
    
    def _determine_verification_method(self, 
                                     overlap_score: float,
                                     similarity_score: float,
                                     entity_overlap_score: float) -> str:
        """Determine which verification method was most successful"""
        scores = {
            'lexical': overlap_score,
            'semantic': similarity_score,
            'entity': entity_overlap_score
        }
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _extract_supporting_text(self, claim_text: str, chunk_text: str) -> str:
        """Extract most relevant supporting text from chunk"""
        # Simple approach: find sentences with highest word overlap
        chunk_sentences = sent_tokenize(chunk_text)
        claim_words = set(word_tokenize(claim_text.lower()))
        
        best_sentence = ""
        best_overlap = 0
        
        for sentence in chunk_sentences:
            sentence_words = set(word_tokenize(sentence.lower()))
            overlap = len(claim_words.intersection(sentence_words))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence
        
        return best_sentence if best_sentence else chunk_text[:200] + "..."

class AnswerGuardrails:
    """
    Main answer guardrails system coordinating verification and regeneration
    """
    
    def __init__(self, 
                 config: Optional[GuardrailsConfig] = None,
                 embedding_model: Optional[SentenceTransformer] = None):
        
        self.config = config or GuardrailsConfig()
        self.embedding_model = embedding_model
        
        # Initialize components
        self.claim_extractor = FactualClaimExtractor(self.config)
        self.citation_verifier = CitationVerifier(self.config, embedding_model)
        
        # Performance tracking
        self.verification_stats = {
            'total_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'regenerations_triggered': 0,
            'claims_processed': 0
        }
    
    def verify_answer(self, 
                     generated_text: str,
                     source_chunks: List[ChunkMetadata]) -> Dict[str, Any]:
        """
        Perform complete verification of generated answer
        
        Args:
            generated_text: Generated answer text
            source_chunks: Source chunks used for generation
            
        Returns:
            Verification report with results and recommendations
        """
        self.verification_stats['total_verifications'] += 1
        
        # Extract factual claims
        claims = self.claim_extractor.extract_factual_claims(generated_text)
        self.verification_stats['claims_processed'] += len(claims)
        
        if not claims:
            return {
                'verification_status': 'passed',
                'reason': 'No factual claims detected',
                'claims': [],
                'verification_results': [],
                'requires_regeneration': False
            }
        
        # Verify claims against sources
        verification_results = self.citation_verifier.verify_claims(claims, source_chunks)
        
        # Analyze verification results
        verified_claims = [r for r in verification_results if r.is_verified]
        failed_claims = [r for r in verification_results if not r.is_verified]
        
        # Calculate overall verification score
        verification_score = len(verified_claims) / len(claims) if claims else 1.0
        
        # Determine if regeneration is needed
        requires_regeneration = (
            self.config.enable_regeneration and
            verification_score < 0.8 and  # Less than 80% of claims verified
            len(failed_claims) > 0
        )
        
        # Update statistics
        if verification_score >= 0.8:
            self.verification_stats['successful_verifications'] += 1
        else:
            self.verification_stats['failed_verifications'] += 1
            if requires_regeneration:
                self.verification_stats['regenerations_triggered'] += 1
        
        return {
            'verification_status': 'passed' if verification_score >= 0.8 else 'failed',
            'verification_score': verification_score,
            'total_claims': len(claims),
            'verified_claims': len(verified_claims),
            'failed_claims': len(failed_claims),
            'claims': [self._claim_to_dict(claim) for claim in claims],
            'verification_results': [self._result_to_dict(result) for result in verification_results],
            'requires_regeneration': requires_regeneration,
            'failed_claim_details': [self._result_to_dict(result) for result in failed_claims],
            'regeneration_guidance': self._generate_regeneration_guidance(failed_claims) if requires_regeneration else None
        }
    
    def _claim_to_dict(self, claim: FactualClaim) -> Dict[str, Any]:
        """Convert FactualClaim to dictionary"""
        return {
            'sentence': claim.sentence,
            'claim_text': claim.claim_text,
            'claim_type': claim.claim_type,
            'confidence': claim.confidence,
            'entities': claim.entities,
            'sentence_index': claim.sentence_index
        }
    
    def _result_to_dict(self, result: VerificationResult) -> Dict[str, Any]:
        """Convert VerificationResult to dictionary"""
        return {
            'claim': self._claim_to_dict(result.claim),
            'is_verified': result.is_verified,
            'confidence': result.confidence,
            'failure_reason': result.failure_reason,
            'citations': [
                {
                    'chunk_id': citation.chunk.chunk_id,
                    'similarity_score': citation.similarity_score,
                    'overlap_score': citation.overlap_score,
                    'confidence_score': citation.confidence_score,
                    'verification_method': citation.verification_method,
                    'supporting_text': citation.supporting_text
                }
                for citation in result.citations
            ]
        }
    
    def _generate_regeneration_guidance(self, failed_claims: List[VerificationResult]) -> Dict[str, Any]:
        """Generate guidance for answer regeneration"""
        guidance = {
            'failed_claim_types': list(set(r.claim.claim_type for r in failed_claims)),
            'common_issues': [],
            'suggested_instructions': []
        }
        
        # Analyze common failure patterns
        numeric_failures = [r for r in failed_claims if r.claim.claim_type == 'numeric']
        entity_failures = [r for r in failed_claims if r.claim.claim_type == 'entity']
        
        if numeric_failures:
            guidance['common_issues'].append('Numerical claims lack source verification')
            guidance['suggested_instructions'].append(
                'Only include numerical information that is directly stated in the provided sources'
            )
        
        if entity_failures:
            guidance['common_issues'].append('Entity mentions not found in sources')
            guidance['suggested_instructions'].append(
                'Only mention entities, organizations, or names that appear in the source documents'
            )
        
        # Add general guidance
        guidance['suggested_instructions'].extend([
            'Base all factual statements on the provided source materials',
            'Include specific citations for all claims',
            'Avoid making inferences not directly supported by sources'
        ])
        
        return guidance
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """Get verification performance statistics"""
        total = self.verification_stats['total_verifications']
        
        stats = self.verification_stats.copy()
        if total > 0:
            stats['success_rate'] = self.verification_stats['successful_verifications'] / total
            stats['failure_rate'] = self.verification_stats['failed_verifications'] / total
            stats['regeneration_rate'] = self.verification_stats['regenerations_triggered'] / total
            stats['avg_claims_per_answer'] = self.verification_stats['claims_processed'] / total
        
        return stats 