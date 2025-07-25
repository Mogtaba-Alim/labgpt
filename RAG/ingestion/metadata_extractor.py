#!/usr/bin/env python3
"""
metadata_extractor.py

Metadata extraction and enhancement for document chunks.
Extracts additional context, keywords, and semantic information.
"""

import re
import logging
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter
import hashlib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

from .chunk_objects import ChunkMetadata

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK data...")
    nltk.download(['stopwords', 'punkt', 'averaged_perceptron_tagger', 'wordnet'], quiet=True)

class MetadataExtractor:
    """
    Extract and enhance metadata for document chunks including:
    - Keywords and key phrases
    - Named entities (simple pattern-based)
    - Topic indicators
    - Semantic tags
    - Context relationships
    """
    
    def __init__(self, language: str = 'english'):
        self.language = language
        
        # Initialize NLTK components
        try:
            self.stopwords = set(stopwords.words(language))
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but'])
            self.lemmatizer = None
        
        # Domain-specific term patterns
        self.domain_patterns = {
            'research': {
                'patterns': [
                    r'\b(?:hypothesis|methodology|experiment|analysis|results?|conclusion|findings?)\b',
                    r'\b(?:study|research|investigation|survey|trial)\b',
                    r'\b(?:significant|correlation|statistical|p-value|confidence)\b'
                ],
                'weight': 2.0
            },
            'technical': {
                'patterns': [
                    r'\b(?:algorithm|implementation|system|framework|architecture)\b',
                    r'\b(?:performance|optimization|efficiency|scalability)\b',
                    r'\b(?:function|method|class|module|interface)\b'
                ],
                'weight': 1.8
            },
            'biological': {
                'patterns': [
                    r'\b(?:protein|gene|DNA|RNA|sequence|structure)\b',
                    r'\b(?:cell|tissue|organ|organism|species)\b',
                    r'\b(?:mutation|expression|pathway|receptor)\b'
                ],
                'weight': 2.0
            },
            'mathematical': {
                'patterns': [
                    r'\b(?:theorem|proof|equation|formula|matrix)\b',
                    r'\b(?:function|variable|parameter|optimization)\b',
                    r'\b(?:probability|statistics|distribution|model)\b'
                ],
                'weight': 1.8
            }
        }
        
        # Citation patterns
        self.citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\(([A-Za-z]+\s+et\s+al\.?,\s*\d{4})\)',  # (Smith et al., 2023)
            r'\(([A-Za-z]+,\s*\d{4})\)',  # (Smith, 2023)
            r'doi:\s*([^\s]+)',  # DOI references
        ]
        
        # Common academic section indicators
        self.section_types = {
            'abstract': r'\babstract\b',
            'introduction': r'\bintroduction\b',
            'methods': r'\b(?:methods?|methodology)\b',
            'results': r'\bresults?\b',
            'discussion': r'\bdiscussion\b',
            'conclusion': r'\bconclusions?\b',
            'references': r'\breferences?\b',
            'appendix': r'\bappendix\b'
        }
    
    def extract_metadata(self, chunk: ChunkMetadata) -> ChunkMetadata:
        """
        Extract comprehensive metadata for a chunk
        
        Args:
            chunk: ChunkMetadata object to enhance
            
        Returns:
            Enhanced ChunkMetadata with additional metadata
        """
        text = chunk.text
        
        # Extract keywords and key phrases
        keywords = self._extract_keywords(text)
        key_phrases = self._extract_key_phrases(text)
        
        # Detect domain and topics
        domain_scores = self._calculate_domain_scores(text)
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else None
        
        # Extract citations and references
        citations = self._extract_citations(text)
        
        # Detect section type
        section_type = self._detect_section_type(text, chunk.section)
        
        # Calculate semantic features
        semantic_features = self._extract_semantic_features(text)
        
        # Store in chunk metadata (we'll extend ChunkMetadata to include these)
        # For now, we'll store in a custom metadata dict
        enhanced_metadata = {
            'keywords': keywords,
            'key_phrases': key_phrases,
            'domain_scores': domain_scores,
            'primary_domain': primary_domain,
            'citations': citations,
            'section_type': section_type,
            'semantic_features': semantic_features,
            'extraction_version': '1.0'
        }
        
        # Store as JSON string in a hypothetical metadata field
        # (This would require extending ChunkMetadata class)
        logger.debug(f"Extracted metadata for chunk {chunk.chunk_id}: "
                    f"domain={primary_domain}, keywords={len(keywords)}")
        
        return chunk
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text"""
        if not text:
            return []
        
        try:
            # Tokenize and get POS tags
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # Filter for important POS tags (nouns, adjectives)
            important_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}
            candidates = [
                token for token, pos in pos_tags 
                if pos in important_pos and 
                token not in self.stopwords and 
                len(token) > 2 and 
                token.isalpha()
            ]
            
            # Lemmatize if available
            if self.lemmatizer:
                candidates = [self.lemmatizer.lemmatize(word) for word in candidates]
            
            # Count frequency and return top keywords
            keyword_counts = Counter(candidates)
            keywords = [word for word, count in keyword_counts.most_common(max_keywords)]
            
            return keywords
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            # Fallback: simple frequency analysis
            return self._extract_keywords_simple(text, max_keywords)
    
    def _extract_keywords_simple(self, text: str, max_keywords: int = 10) -> List[str]:
        """Simple keyword extraction fallback"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in self.stopwords]
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 5) -> List[str]:
        """Extract key phrases (2-3 word combinations)"""
        if not text:
            return []
        
        # Simple n-gram extraction
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        words = [w for w in words if w not in self.stopwords and len(w) > 2]
        
        # Extract 2-grams and 3-grams
        phrases = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            phrases.append(bigram)
            
            if i < len(words) - 2:
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                phrases.append(trigram)
        
        # Count and return most frequent
        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(max_phrases)]
    
    def _calculate_domain_scores(self, text: str) -> Dict[str, float]:
        """Calculate domain relevance scores"""
        domain_scores = {}
        text_lower = text.lower()
        
        for domain, config in self.domain_patterns.items():
            score = 0.0
            patterns = config['patterns']
            weight = config.get('weight', 1.0)
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * weight
            
            # Normalize by text length
            if len(text) > 0:
                domain_scores[domain] = score / (len(text) / 1000)  # Per 1000 chars
            else:
                domain_scores[domain] = 0.0
        
        return domain_scores
    
    def _extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract citation references from text"""
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citation = {
                    'text': match.group(0),
                    'reference': match.group(1) if match.groups() else match.group(0),
                    'position': match.start()
                }
                citations.append(citation)
        
        return citations
    
    def _detect_section_type(self, text: str, section_title: Optional[str] = None) -> Optional[str]:
        """Detect the type of academic section"""
        
        # First check section title if available
        if section_title:
            title_lower = section_title.lower()
            for section_type, pattern in self.section_types.items():
                if re.search(pattern, title_lower):
                    return section_type
        
        # Then check text content
        text_lower = text.lower()
        section_scores = {}
        
        for section_type, pattern in self.section_types.items():
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                section_scores[section_type] = matches
        
        # Return highest scoring section type
        if section_scores:
            return max(section_scores, key=section_scores.get)
        
        return None
    
    def _extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Extract semantic features for the text"""
        features = {}
        
        # Basic text statistics
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
        
        # Complexity indicators
        features['long_word_ratio'] = sum(1 for word in text.split() if len(word) > 6) / len(text.split()) if text.split() else 0
        features['punctuation_density'] = sum(1 for char in text if not char.isalnum() and not char.isspace()) / len(text) if text else 0
        
        # Content type indicators
        features['number_density'] = len(re.findall(r'\d+', text)) / len(text) if text else 0
        features['capital_ratio'] = sum(1 for char in text if char.isupper()) / len(text) if text else 0
        
        # Question/interrogative content
        features['question_density'] = text.count('?') / len(text) if text else 0
        
        return features
    
    def extract_cross_references(self, chunks: List[ChunkMetadata]) -> Dict[str, List[str]]:
        """
        Find cross-references between chunks based on shared content
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary mapping chunk_id to list of related chunk_ids
        """
        cross_references = {}
        
        # Extract keywords for all chunks
        chunk_keywords = {}
        for chunk in chunks:
            keywords = self._extract_keywords(chunk.text, max_keywords=20)
            chunk_keywords[chunk.chunk_id] = set(keywords)
        
        # Find relationships based on keyword overlap
        for i, chunk1 in enumerate(chunks):
            related_chunks = []
            keywords1 = chunk_keywords[chunk1.chunk_id]
            
            for j, chunk2 in enumerate(chunks):
                if i != j:  # Don't compare with self
                    keywords2 = chunk_keywords[chunk2.chunk_id]
                    
                    # Calculate Jaccard similarity
                    if keywords1 and keywords2:
                        overlap = len(keywords1.intersection(keywords2))
                        union = len(keywords1.union(keywords2))
                        similarity = overlap / union if union > 0 else 0
                        
                        # If similarity is high enough, consider them related
                        if similarity > 0.3:  # Threshold can be adjusted
                            related_chunks.append(chunk2.chunk_id)
            
            cross_references[chunk1.chunk_id] = related_chunks
        
        return cross_references
    
    def extract_document_themes(self, chunks: List[ChunkMetadata]) -> Dict[str, float]:
        """
        Extract major themes across all chunks in a document
        
        Args:
            chunks: All chunks from a document
            
        Returns:
            Dictionary of themes with relevance scores
        """
        # Combine all text
        combined_text = ' '.join(chunk.text for chunk in chunks)
        
        # Extract keywords with higher frequency threshold
        all_keywords = self._extract_keywords(combined_text, max_keywords=50)
        
        # Group keywords into themes using simple clustering
        theme_groups = self._cluster_keywords_into_themes(all_keywords, combined_text)
        
        # Calculate theme scores
        theme_scores = {}
        for theme, keywords in theme_groups.items():
            score = 0.0
            for keyword in keywords:
                # Count occurrences across all chunks
                occurrences = sum(chunk.text.lower().count(keyword.lower()) for chunk in chunks)
                score += occurrences
            
            # Normalize by total text length
            total_length = sum(len(chunk.text) for chunk in chunks)
            theme_scores[theme] = score / (total_length / 1000) if total_length > 0 else 0
        
        return theme_scores
    
    def _cluster_keywords_into_themes(self, keywords: List[str], text: str) -> Dict[str, List[str]]:
        """Simple keyword clustering into themes"""
        themes = {}
        
        # Predefined theme categories
        research_terms = ['research', 'study', 'analysis', 'method', 'result', 'data', 'experiment']
        technical_terms = ['system', 'algorithm', 'model', 'implementation', 'performance', 'framework']
        biological_terms = ['protein', 'gene', 'cell', 'biological', 'molecular', 'sequence']
        mathematical_terms = ['equation', 'function', 'mathematical', 'calculation', 'formula', 'optimization']
        
        theme_categories = {
            'research': research_terms,
            'technical': technical_terms,
            'biological': biological_terms,
            'mathematical': mathematical_terms
        }
        
        # Classify keywords into themes
        for theme, seed_terms in theme_categories.items():
            theme_keywords = []
            for keyword in keywords:
                # Check if keyword is related to any seed terms
                if any(seed in keyword.lower() or keyword.lower() in seed for seed in seed_terms):
                    theme_keywords.append(keyword)
            
            if theme_keywords:
                themes[theme] = theme_keywords
        
        # Put remaining keywords in 'general' theme
        classified_keywords = set()
        for keywords_list in themes.values():
            classified_keywords.update(keywords_list)
        
        remaining = [k for k in keywords if k not in classified_keywords]
        if remaining:
            themes['general'] = remaining
        
        return themes 