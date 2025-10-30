"""
Data models for paper chunks.

Simplified models adapted for data generation pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class PaperChunk:
    """Represents a chunk of text from a research paper."""
    content: str
    source_path: str
    chunk_index: int
    doc_id: str

    # Metadata
    page_numbers: Optional[list] = field(default_factory=list)
    section_title: Optional[str] = None
    token_count: int = 0

    # For maintaining context
    prev_chunk_preview: Optional[str] = None
    next_chunk_preview: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            'content': self.content,
            'source_path': self.source_path,
            'chunk_index': self.chunk_index,
            'doc_id': self.doc_id,
            'page_numbers': self.page_numbers,
            'section_title': self.section_title,
            'token_count': self.token_count,
        }
