"""
cited_span_extractor.py

Extract specific text spans that support claims for transparent citation.
Enables highlighting exact source text that supports each answer claim.
"""

import logging
from typing import List
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from ..models import CitedSpan, Chunk

logger = logging.getLogger(__name__)


class CitedSpanExtractor:
    """
    Extract specific text spans from chunks that support claims.

    Uses embedding similarity to identify sentences that support a given claim,
    providing character offsets for highlighting in UI and improving transparency.

    This enables users to verify claims by seeing the exact source text, reducing
    hallucination concerns and improving trust in generated answers.
    """

    def __init__(self, embedding_model: SentenceTransformer,
                 similarity_threshold: float = 0.6,
                 max_spans: int = 3):
        """
        Initialize cited span extractor.

        Args:
            embedding_model: Sentence transformer for embeddings
            similarity_threshold: Minimum cosine similarity for citation (default: 0.6)
            max_spans: Maximum number of supporting spans to return (default: 3)
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_spans = max_spans

    def extract_cited_spans(self, claim: str, chunk: Chunk) -> List[CitedSpan]:
        """
        Extract specific text spans that support a claim.

        Process:
        1. Embed the claim
        2. Split chunk into sentences
        3. Embed each sentence
        4. Calculate cosine similarity to claim
        5. Return top-N sentences above threshold with offsets

        Args:
            claim: The claim/statement to find support for
            chunk: Chunk containing potential supporting text

        Returns:
            List of CitedSpan objects with offsets, text, and confidence scores
        """
        if not claim or not chunk.text:
            return []

        # Embed claim
        claim_embedding = self.embedding_model.encode(
            claim,
            convert_to_tensor=False,
            show_progress_bar=False
        )

        # Split chunk into sentences
        try:
            sentences = sent_tokenize(chunk.text)
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}. Using simple split.")
            sentences = [s.strip() for s in chunk.text.split('.') if s.strip()]

        if not sentences:
            return []

        # Score each sentence by similarity to claim
        span_candidates = []
        current_offset = 0

        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.strip()) < 20:
                current_offset += len(sentence) + 1
                continue

            # Embed sentence
            sentence_embedding = self.embedding_model.encode(
                sentence,
                convert_to_tensor=False,
                show_progress_bar=False
            )

            # Calculate cosine similarity
            similarity = self._cosine_similarity(claim_embedding, sentence_embedding)

            # Keep if above threshold
            if similarity > self.similarity_threshold:
                # Find actual start position in original text
                actual_start = chunk.text.find(sentence, current_offset)
                if actual_start == -1:
                    actual_start = current_offset

                actual_end = actual_start + len(sentence)

                span_candidates.append({
                    'start_offset': actual_start,
                    'end_offset': actual_end,
                    'text': sentence.strip(),
                    'confidence': float(similarity)
                })

                logger.debug(
                    f"Found supporting span (confidence: {similarity:.3f}): "
                    f"{sentence[:50]}..."
                )

            current_offset += len(sentence) + 1

        # Sort by confidence and take top-N
        span_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        top_spans = span_candidates[:self.max_spans]

        # Convert to CitedSpan objects
        cited_spans = [
            CitedSpan(
                start_offset=span['start_offset'],
                end_offset=span['end_offset'],
                text=span['text'],
                confidence=span['confidence']
            )
            for span in top_spans
        ]

        if cited_spans:
            logger.info(
                f"Extracted {len(cited_spans)} cited spans for claim "
                f"(avg confidence: {np.mean([s.confidence for s in cited_spans]):.3f})"
            )

        return cited_spans

    def extract_spans_for_claims(self, claims: List[str],
                                 chunk: Chunk) -> List[List[CitedSpan]]:
        """
        Extract supporting spans for multiple claims from the same chunk.

        Useful when an answer contains multiple claims that need citation.

        Args:
            claims: List of claims/statements
            chunk: Chunk to extract support from

        Returns:
            List of lists, one list of CitedSpans per claim
        """
        all_spans = []
        for claim in claims:
            spans = self.extract_cited_spans(claim, chunk)
            all_spans.append(spans)

        return all_spans

    def merge_overlapping_spans(self, spans: List[CitedSpan]) -> List[CitedSpan]:
        """
        Merge overlapping or adjacent cited spans.

        When multiple claims cite overlapping text, merge them to avoid
        redundant highlighting in UI.

        Args:
            spans: List of CitedSpan objects

        Returns:
            List of merged CitedSpan objects
        """
        if not spans:
            return []

        # Sort by start offset
        sorted_spans = sorted(spans, key=lambda s: s.start_offset)

        merged = []
        current = sorted_spans[0]

        for next_span in sorted_spans[1:]:
            # Check if overlapping or adjacent (within 10 chars)
            if next_span.start_offset <= current.end_offset + 10:
                # Merge spans
                new_end = max(current.end_offset, next_span.end_offset)
                new_confidence = max(current.confidence, next_span.confidence)

                # Update current span
                current = CitedSpan(
                    start_offset=current.start_offset,
                    end_offset=new_end,
                    text=current.text,  # Keep original text
                    confidence=new_confidence
                )
            else:
                # No overlap, add current and move to next
                merged.append(current)
                current = next_span

        # Add the last span
        merged.append(current)

        logger.debug(f"Merged {len(spans)} spans into {len(merged)} spans")
        return merged

    def get_highlighted_text(self, chunk_text: str,
                            spans: List[CitedSpan],
                            highlight_format: str = "**{}**") -> str:
        """
        Get chunk text with cited spans highlighted.

        Useful for displaying chunks with supporting text emphasized.

        Args:
            chunk_text: Original chunk text
            spans: List of CitedSpan objects to highlight
            highlight_format: Format string for highlighting (default: markdown bold)

        Returns:
            Text with cited spans highlighted
        """
        if not spans:
            return chunk_text

        # Sort spans by start offset (reverse to process from end)
        sorted_spans = sorted(spans, key=lambda s: s.start_offset, reverse=True)

        # Apply highlighting from end to start (to preserve offsets)
        highlighted_text = chunk_text
        for span in sorted_spans:
            before = highlighted_text[:span.start_offset]
            span_text = highlighted_text[span.start_offset:span.end_offset]
            after = highlighted_text[span.end_offset:]

            highlighted_span = highlight_format.format(span_text)
            highlighted_text = before + highlighted_span + after

        return highlighted_text

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def validate_span(self, span: CitedSpan, chunk: Chunk) -> bool:
        """
        Validate that a cited span is valid for the chunk.

        Checks that offsets are within bounds and text matches.

        Args:
            span: CitedSpan to validate
            chunk: Chunk the span references

        Returns:
            True if valid, False otherwise
        """
        # Check offsets are within bounds
        if span.start_offset < 0 or span.end_offset > len(chunk.text):
            return False

        # Check start < end
        if span.start_offset >= span.end_offset:
            return False

        # Check text matches (allowing for minor whitespace differences)
        expected_text = chunk.text[span.start_offset:span.end_offset].strip()
        actual_text = span.text.strip()

        return expected_text == actual_text or expected_text in actual_text
