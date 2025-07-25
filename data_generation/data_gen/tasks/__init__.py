"""
Task generation package for synthetic data creation.

This package provides various task generators for creating training data
from code symbols, including Q&A, completion, debugging, and refactoring tasks.
"""

from .qa_generator import (
    GroundedQAGenerator,
    GroundedQA,
    QAGenerationStats
)

from .bug_injector import (
    BugInjector,
    BugInjection,
    BugType,
    create_bug_injector
)

from .debug_generator import (
    DebugGenerator,
    DebugTask,
    DebugGenerationStats,
    create_debug_generator
)

from .negative_generator import (
    EnhancedNegativeGenerator,
    NegativeExample,
    NegativeExampleType,
    create_negative_generator
)

from .paper_qa_generator import (
    MultiChunkPaperQAGenerator,
    PaperChunk,
    MultiChunkQA,
    SectionType,
    create_paper_qa_generator
)

from .selective_questioner import (
    SelectiveQuestioner,
    QuestionTemplate,
    QuestionCategory,
    create_selective_questioner
)

__all__ = [
    # P1 Components
    'GroundedQAGenerator',
    'GroundedQA', 
    'QAGenerationStats',
    
    # P2 Components - Bug Injection
    'BugInjector',
    'BugInjection',
    'BugType',
    'create_bug_injector',
    
    # P2 Components - Debug Generation
    'DebugGenerator',
    'DebugTask',
    'DebugGenerationStats',
    'create_debug_generator',
    
    # P2 Components - Enhanced Negatives
    'EnhancedNegativeGenerator',
    'NegativeExample',
    'NegativeExampleType',
    'create_negative_generator',
    
    # P2 Components - Multi-Chunk Paper QA
    'MultiChunkPaperQAGenerator',
    'PaperChunk',
    'MultiChunkQA',
    'SectionType',
    'create_paper_qa_generator',
    
    # P2 Components - Selective Questioning
    'SelectiveQuestioner',
    'QuestionTemplate',
    'QuestionCategory',
    'create_selective_questioner'
] 