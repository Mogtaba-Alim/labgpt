#!/usr/bin/env python3
"""
query_processor.py

Query processing, expansion, and rewriting for enhanced retrieval.
"""

import logging
import re
from typing import List, Dict, Optional, Set
from abc import ABC, abstractmethod

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
import numpy as np

from .retrieval_config import RetrievalConfig

logger = logging.getLogger(__name__)

# Ensure NLTK data
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download(['wordnet', 'punkt'], quiet=True)

class QueryExpander(ABC):
    """Abstract base class for query expansion methods"""
    
    @abstractmethod
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand a query into multiple related queries"""
        pass

class WordNetExpander(QueryExpander):
    """Query expansion using WordNet synonyms"""
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query using WordNet synonyms"""
        expanded_queries = [query]  # Always include original
        
        try:
            # Tokenize query
            tokens = word_tokenize(query.lower())
            
            # Get synonyms for each meaningful word
            for token in tokens:
                if len(token) > 3 and token.isalpha():  # Skip short words and non-alphabetic
                    synonyms = self._get_synonyms(token)
                    
                    # Create expanded queries by replacing the token
                    for synonym in synonyms[:2]:  # Limit synonyms per word
                        expanded_query = query.replace(token, synonym)
                        if expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)
                            
                        if len(expanded_queries) >= max_expansions + 1:  # +1 for original
                            break
                    
                    if len(expanded_queries) >= max_expansions + 1:
                        break
            
        except Exception as e:
            logger.warning(f"WordNet expansion failed: {e}")
        
        return expanded_queries
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet"""
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and len(synonym) > 2:
                    synonyms.add(synonym)
        
        return list(synonyms)

class EmbeddingExpander(QueryExpander):
    """Query expansion using embedding similarity"""
    
    def __init__(self, embedding_model: SentenceTransformer, vocabulary: Optional[Set[str]] = None):
        self.embedding_model = embedding_model
        self.vocabulary = vocabulary or set()
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query using embedding-based similarity"""
        expanded_queries = [query]
        
        if not self.vocabulary:
            logger.warning("No vocabulary available for embedding expansion")
            return expanded_queries
        
        try:
            # Encode original query
            query_embedding = self.embedding_model.encode([query])
            
            # Get candidate terms from vocabulary
            candidates = list(self.vocabulary)[:1000]  # Limit for performance
            
            if candidates:
                candidate_embeddings = self.embedding_model.encode(candidates)
                
                # Calculate similarities
                similarities = np.dot(query_embedding, candidate_embeddings.T)[0]
                
                # Get top similar terms
                top_indices = np.argsort(similarities)[::-1][:10]
                
                # Create expanded queries
                for idx in top_indices:
                    if similarities[idx] > 0.7:  # Similarity threshold
                        similar_term = candidates[idx]
                        expanded_query = f"{query} {similar_term}"
                        expanded_queries.append(expanded_query)
                        
                        if len(expanded_queries) >= max_expansions + 1:
                            break
        
        except Exception as e:
            logger.warning(f"Embedding expansion failed: {e}")
        
        return expanded_queries

class LLMExpander(QueryExpander):
    """Query expansion using LLM (placeholder for now)"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        # In a real implementation, you would initialize the LLM client here
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query using LLM (placeholder implementation)"""
        # This is a placeholder - in real implementation would call LLM API
        expanded_queries = [query]
        
        # Simple rule-based expansion as fallback
        query_lower = query.lower()
        
        # Add scientific/research variations
        if any(term in query_lower for term in ['algorithm', 'method', 'approach']):
            expanded_queries.append(query + " technique")
            expanded_queries.append(query + " methodology")
        
        if any(term in query_lower for term in ['analysis', 'study']):
            expanded_queries.append(query + " research")
            expanded_queries.append(query + " investigation")
        
        if 'protein' in query_lower:
            expanded_queries.append(query + " structure")
            expanded_queries.append(query + " function")
        
        return expanded_queries[:max_expansions + 1]

class QueryProcessor:
    """Main query processing class with expansion and rewriting"""
    
    def __init__(self, config: RetrievalConfig, embedding_model: Optional[SentenceTransformer] = None):
        self.config = config
        self.embedding_model = embedding_model
        
        # Initialize expanders based on config
        self.expanders = self._initialize_expanders()
        
        # Query rewriting patterns
        self.rewrite_patterns = self._initialize_rewrite_patterns()
    
    def _initialize_expanders(self) -> Dict[str, QueryExpander]:
        """Initialize query expanders based on configuration"""
        expanders = {}
        
        if self.config.query_processing.expansion_method == "wordnet":
            expanders["wordnet"] = WordNetExpander()
        elif self.config.query_processing.expansion_method == "embedding":
            if self.embedding_model:
                expanders["embedding"] = EmbeddingExpander(self.embedding_model)
        elif self.config.query_processing.expansion_method == "llm":
            expanders["llm"] = LLMExpander(self.config.models.query_expansion_model)
        
        # Always include WordNet as fallback
        if "wordnet" not in expanders:
            expanders["wordnet"] = WordNetExpander()
        
        return expanders
    
    def _initialize_rewrite_patterns(self) -> List[Dict[str, str]]:
        """Initialize query rewriting patterns"""
        return [
            # Question to statement
            {"pattern": r"what is (.+)\?", "replacement": r"\1"},
            {"pattern": r"how does (.+) work\?", "replacement": r"\1 mechanism function"},
            {"pattern": r"why (.+)\?", "replacement": r"\1 reason cause"},
            
            # Academic/research language
            {"pattern": r"explain (.+)", "replacement": r"\1 explanation description"},
            {"pattern": r"describe (.+)", "replacement": r"\1 description characteristics"},
            {"pattern": r"find (.+)", "replacement": r"\1 information"},
            
            # Domain-specific expansions
            {"pattern": r"machine learning", "replacement": "machine learning ML artificial intelligence"},
            {"pattern": r"deep learning", "replacement": "deep learning neural networks"},
            {"pattern": r"protein structure", "replacement": "protein structure conformation fold"},
        ]
    
    def process_query(self, query: str) -> List[str]:
        """
        Process query with expansion and rewriting
        
        Args:
            query: Original query
            
        Returns:
            List of processed queries (original + expansions/rewrites)
        """
        processed_queries = []
        
        # Clean and normalize query
        cleaned_query = self._clean_query(query)
        
        # Add original query if configured
        if self.config.query_processing.include_original:
            processed_queries.append(cleaned_query)
        
        # Apply query rewriting
        if self.config.query_processing.query_rewriting:
            rewritten_queries = self._rewrite_query(cleaned_query)
            processed_queries.extend(rewritten_queries)
        
        # Apply query expansion
        if self.config.query_processing.enable_expansion:
            expanded_queries = self._expand_query(cleaned_query)
            processed_queries.extend(expanded_queries)
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in processed_queries:
            if q not in seen:
                unique_queries.append(q)
                seen.add(q)
        
        # Limit to max expansions
        max_total = self.config.query_processing.max_expansions + 1  # +1 for original
        final_queries = unique_queries[:max_total]
        
        logger.debug(f"Processed query '{query}' into {len(final_queries)} variants")
        return final_queries
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query"""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters that might interfere with retrieval
        query = re.sub(r'[^\w\s\-\.\?]', '', query)
        
        return query
    
    def _rewrite_query(self, query: str) -> List[str]:
        """Apply query rewriting patterns"""
        rewritten = []
        
        for pattern_dict in self.rewrite_patterns:
            pattern = pattern_dict["pattern"]
            replacement = pattern_dict["replacement"]
            
            if re.search(pattern, query, re.IGNORECASE):
                rewritten_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                if rewritten_query != query:
                    rewritten.append(rewritten_query)
        
        return rewritten
    
    def _expand_query(self, query: str) -> List[str]:
        """Apply query expansion using configured expanders"""
        all_expansions = []
        
        for expander_name, expander in self.expanders.items():
            try:
                expansions = expander.expand_query(
                    query, 
                    max_expansions=self.config.query_processing.max_expansions
                )
                # Skip the original query (first element)
                all_expansions.extend(expansions[1:])
            except Exception as e:
                logger.warning(f"Expansion with {expander_name} failed: {e}")
        
        return all_expansions
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Simple keyword extraction
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def detect_query_intent(self, query: str) -> str:
        """Detect the intent/type of the query"""
        query_lower = query.lower()
        
        # Question types
        if query_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            return 'question'
        
        # Factual lookup
        if any(word in query_lower for word in ['define', 'definition', 'meaning']):
            return 'definition'
        
        # Comparison
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        
        # Procedure/method
        if any(word in query_lower for word in ['how to', 'steps', 'procedure', 'method']):
            return 'procedure'
        
        # Research/analysis
        if any(word in query_lower for word in ['analyze', 'research', 'study', 'investigate']):
            return 'research'
        
        return 'general'
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Get statistics about query processing"""
        return {
            "available_expanders": len(self.expanders),
            "rewrite_patterns": len(self.rewrite_patterns),
            "max_expansions": self.config.query_processing.max_expansions,
            "expansion_enabled": self.config.query_processing.enable_expansion,
            "rewriting_enabled": self.config.query_processing.query_rewriting
        } 