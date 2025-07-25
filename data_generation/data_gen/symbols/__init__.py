"""
Symbol extraction package for code analysis.

This package provides AST-based parsing and extraction of code symbols
(functions, classes, methods) with detailed complexity metrics.
"""

from .ast_parser import (
    ASTParser,
    CodeSymbol,
    SymbolType,
    ComplexityMetrics,
    ComplexityAnalyzer
)

from .symbol_extractor import (
    SymbolExtractor,
    ExtractionConfig,
    ExtractionStats,
    create_extraction_config
)

__all__ = [
    'ASTParser',
    'CodeSymbol', 
    'SymbolType',
    'ComplexityMetrics',
    'ComplexityAnalyzer',
    'SymbolExtractor',
    'ExtractionConfig',
    'ExtractionStats',
    'create_extraction_config'
] 