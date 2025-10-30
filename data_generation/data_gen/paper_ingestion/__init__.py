"""
Paper ingestion components for enhanced document loading and chunking.
"""

from .document_loader import PaperLoader
from .text_splitter import PaperSplitter, SplittingConfig
from .chunk_models import PaperChunk

__all__ = [
    'PaperLoader',
    'PaperSplitter',
    'SplittingConfig',
    'PaperChunk',
]
