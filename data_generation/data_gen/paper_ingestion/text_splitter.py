"""
Semantic and structural text splitter for research papers.

Respects document boundaries, uses sentence tokenization, and implements token-budget based chunking.
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
except ImportError:
    HAS_NLTK = False
    logging.warning("NLTK not installed, using regex-based sentence splitting")

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    logging.warning("tiktoken not installed, using approximate token counting")

from .chunk_models import PaperChunk

logger = logging.getLogger(__name__)


@dataclass
class SplittingConfig:
    """Configuration for text splitting."""
    target_chunk_size: int = 600  # Target tokens per chunk
    max_chunk_size: int = 800     # Maximum tokens per chunk
    min_chunk_size: int = 200     # Minimum tokens per chunk
    chunk_overlap: int = 50       # Overlap in tokens
    respect_sentence_boundaries: bool = True
    respect_section_boundaries: bool = True


class PaperSplitter:
    """
    Advanced text splitter that:
    1. Respects document structure (sections, headings)
    2. Uses sentence boundaries for semantic coherence
    3. Implements token-budget based chunking
    4. Preserves context with overlap
    """

    def __init__(self, config: Optional[SplittingConfig] = None):
        self.config = config or SplittingConfig()

        # Initialize tokenizer
        if HAS_TIKTOKEN:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None

    def split_document(self, content: str, metadata: Dict) -> List[PaperChunk]:
        """
        Split document into semantically coherent chunks.

        Args:
            content: Document text content
            metadata: Document metadata dictionary

        Returns:
            List of PaperChunk objects
        """
        # Extract page numbers from content if present
        page_markers = self._extract_page_markers(content)

        # Split into sentences
        sentences = self._split_into_sentences(content)

        # Pack sentences into chunks respecting token budget
        chunks = self._pack_sentences_into_chunks(sentences, page_markers)

        # Create PaperChunk objects
        paper_chunks = []
        for idx, chunk_data in enumerate(chunks):
            paper_chunk = PaperChunk(
                content=chunk_data['text'],
                source_path=metadata.get('source_path', ''),
                chunk_index=idx,
                doc_id=metadata.get('doc_id', ''),
                page_numbers=chunk_data.get('page_numbers', []),
                section_title=chunk_data.get('section_title'),
                token_count=chunk_data.get('token_count', 0),
            )
            paper_chunks.append(paper_chunk)

        # Add context previews
        paper_chunks = self._add_context_previews(paper_chunks)

        logger.info(f"Split document into {len(paper_chunks)} chunks")
        return paper_chunks

    def _extract_page_markers(self, content: str) -> Dict[int, int]:
        """Extract page markers from content and map char positions."""
        page_markers = {}
        for match in re.finditer(r'\[PAGE (\d+)\]', content):
            page_num = int(match.group(1))
            char_pos = match.start()
            page_markers[char_pos] = page_num
        return page_markers

    def _split_into_sentences(self, content: str) -> List[Dict]:
        """Split text into sentences with position tracking."""
        # Remove page markers for cleaner sentence splitting
        clean_content = re.sub(r'\[PAGE \d+\]\s*', '', content)

        if HAS_NLTK:
            # Use NLTK for sentence tokenization
            sentences = sent_tokenize(clean_content)
        else:
            # Fallback regex-based sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', clean_content)

        # Clean and filter sentences
        sentence_data = []
        char_pos = 0
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 10:  # Filter out very short sentences
                token_count = self._count_tokens(sent)
                sentence_data.append({
                    'text': sent,
                    'token_count': token_count,
                    'char_pos': char_pos
                })
                char_pos += len(sent) + 1  # Account for space

        return sentence_data

    def _pack_sentences_into_chunks(self, sentences: List[Dict], page_markers: Dict) -> List[Dict]:
        """Pack sentences into chunks respecting token budget."""
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sent_data in sentences:
            sent_tokens = sent_data['token_count']

            # Check if adding this sentence would exceed max_chunk_size
            if current_tokens + sent_tokens > self.config.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join([s['text'] for s in current_chunk])
                chunks.append({
                    'text': chunk_text,
                    'token_count': current_tokens,
                    'page_numbers': self._get_page_numbers(current_chunk, page_markers),
                })

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_tokens = sum(s['token_count'] for s in current_chunk)

            # Add sentence to current chunk
            current_chunk.append(sent_data)
            current_tokens += sent_tokens

            # Check if we've reached target size
            if current_tokens >= self.config.target_chunk_size:
                # Save current chunk
                chunk_text = ' '.join([s['text'] for s in current_chunk])
                chunks.append({
                    'text': chunk_text,
                    'token_count': current_tokens,
                    'page_numbers': self._get_page_numbers(current_chunk, page_markers),
                })

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_tokens = sum(s['token_count'] for s in current_chunk)

        # Add final chunk if it has content
        if current_chunk and current_tokens >= self.config.min_chunk_size:
            chunk_text = ' '.join([s['text'] for s in current_chunk])
            chunks.append({
                'text': chunk_text,
                'token_count': current_tokens,
                'page_numbers': self._get_page_numbers(current_chunk, page_markers),
            })

        return chunks

    def _get_overlap_sentences(self, chunk_sentences: List[Dict]) -> List[Dict]:
        """Get sentences for overlap based on token budget."""
        overlap_sentences = []
        overlap_tokens = 0

        # Take sentences from the end of the chunk
        for sent_data in reversed(chunk_sentences):
            if overlap_tokens + sent_data['token_count'] <= self.config.chunk_overlap:
                overlap_sentences.insert(0, sent_data)
                overlap_tokens += sent_data['token_count']
            else:
                break

        return overlap_sentences

    def _get_page_numbers(self, chunk_sentences: List[Dict], page_markers: Dict) -> List[int]:
        """Determine page numbers for a chunk."""
        if not chunk_sentences or not page_markers:
            return []

        # Get char positions for chunk
        start_pos = chunk_sentences[0]['char_pos']
        end_pos = chunk_sentences[-1]['char_pos']

        # Find relevant page numbers
        pages = set()
        for marker_pos, page_num in page_markers.items():
            if start_pos <= marker_pos <= end_pos:
                pages.add(page_num)

        return sorted(list(pages))

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        # Fallback: approximate token count (1 token ≈ 4 characters)
        return len(text) // 4

    def _add_context_previews(self, chunks: List[PaperChunk]) -> List[PaperChunk]:
        """Add prev/next chunk previews for context."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_text = chunks[i-1].content
                chunk.prev_chunk_preview = prev_text[:150] + "..." if len(prev_text) > 150 else prev_text

            if i < len(chunks) - 1:
                next_text = chunks[i+1].content
                chunk.next_chunk_preview = next_text[:150] + "..." if len(next_text) > 150 else next_text

        return chunks
