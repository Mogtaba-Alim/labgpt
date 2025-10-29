"""
Symbol extraction package for code analysis.

This package provides multi-language parsing and extraction of code symbols
(functions, classes, methods, structs) with detailed complexity metrics.
"""

from .universal_symbol_extractor import (
    UniversalSymbolExtractor,
    UniversalExtractionConfig,
    UniversalExtractionStats,
    create_universal_extraction_config
)

from .multi_language_parser import (
    MultiLanguageParser,
    UniversalCodeSymbol,
    UniversalComplexityMetrics,
    Language
)

__all__ = [
    'UniversalSymbolExtractor',
    'UniversalExtractionConfig',
    'UniversalExtractionStats',
    'create_universal_extraction_config',
    'MultiLanguageParser',
    'UniversalCodeSymbol',
    'UniversalComplexityMetrics',
    'Language'
] 