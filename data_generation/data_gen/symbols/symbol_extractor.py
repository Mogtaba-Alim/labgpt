"""
Symbol Extractor for Code Analysis

This module provides high-level extraction of code symbols from files using the AST parser,
with intelligent filtering, token budget management, and symbol ranking.
"""

import os
import glob
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .ast_parser import ASTParser, CodeSymbol, SymbolType


@dataclass
class ExtractionConfig:
    """Configuration for symbol extraction."""
    token_budget: Tuple[int, int] = (200, 400)  # (min_tokens, max_tokens)
    max_symbols_per_file: int = 10
    complexity_filter: Optional[str] = None  # "simple", "moderate", "complex", "very_complex"
    include_private: bool = False
    include_methods: bool = True
    include_classes: bool = True
    include_functions: bool = True
    min_lines_of_code: int = 3
    max_lines_of_code: int = 100
    supported_extensions: Tuple[str, ...] = ('.py',)


@dataclass
class ExtractionStats:
    """Statistics from symbol extraction process."""
    files_processed: int = 0
    files_with_errors: int = 0
    total_symbols_found: int = 0
    symbols_after_filtering: int = 0
    symbols_by_type: Dict[str, int] = field(default_factory=dict)
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    token_distribution: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class SymbolExtractor:
    """High-level symbol extractor for code analysis."""
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """Initialize the symbol extractor with configuration."""
        self.config = config or ExtractionConfig()
        self.parser = ASTParser(token_budget=self.config.token_budget)
        self.stats = ExtractionStats()
        self.logger = logging.getLogger(__name__)
    
    def extract_from_file(self, file_path: str) -> List[CodeSymbol]:
        """
        Extract symbols from a single Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of extracted and filtered code symbols
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            
            # Parse and extract symbols
            symbols = self.parser.parse_file(file_path, source_code)
            
            # Filter symbols based on configuration
            filtered_symbols = self._filter_symbols(symbols)
            
            # Rank and limit symbols
            ranked_symbols = self._rank_and_limit_symbols(filtered_symbols)
            
            # Update statistics
            self._update_stats(symbols, filtered_symbols, ranked_symbols)
            
            self.logger.info(f"Extracted {len(ranked_symbols)} symbols from {file_path}")
            return ranked_symbols
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.logger.error(error_msg)
            self.stats.files_with_errors += 1
            self.stats.errors.append(error_msg)
            return []
        finally:
            self.stats.files_processed += 1
    
    def extract_from_directory(self, directory_path: str, recursive: bool = True) -> List[CodeSymbol]:
        """
        Extract symbols from all Python files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of all extracted symbols
        """
        all_symbols = []
        
        # Find all Python files
        pattern = "**/*" if recursive else "*"
        for ext in self.config.supported_extensions:
            file_pattern = os.path.join(directory_path, f"{pattern}{ext}")
            for file_path in glob.glob(file_pattern, recursive=recursive):
                if os.path.isfile(file_path):
                    symbols = self.extract_from_file(file_path)
                    all_symbols.extend(symbols)
        
        self.logger.info(f"Extracted {len(all_symbols)} total symbols from {directory_path}")
        return all_symbols
    
    def extract_from_repo(self, repo_path: str) -> Dict[str, List[CodeSymbol]]:
        """
        Extract symbols from a repository, organized by file.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Dictionary mapping file paths to extracted symbols
        """
        symbols_by_file = {}
        
        for ext in self.config.supported_extensions:
            file_pattern = os.path.join(repo_path, f"**/*{ext}")
            for file_path in glob.glob(file_pattern, recursive=True):
                if os.path.isfile(file_path):
                    symbols = self.extract_from_file(file_path)
                    if symbols:  # Only include files with extracted symbols
                        relative_path = os.path.relpath(file_path, repo_path)
                        symbols_by_file[relative_path] = symbols
        
        return symbols_by_file
    
    def _filter_symbols(self, symbols: List[CodeSymbol]) -> List[CodeSymbol]:
        """Filter symbols based on configuration criteria."""
        filtered = []
        
        for symbol in symbols:
            # Filter by symbol type
            if not self._should_include_symbol_type(symbol):
                continue
            
            # Filter by visibility
            if symbol.is_private and not self.config.include_private:
                continue
            
            # Filter by complexity
            if self.config.complexity_filter:
                complexity_tier = self.parser.get_complexity_tier(symbol)
                if complexity_tier != self.config.complexity_filter:
                    continue
            
            # Filter by lines of code
            if not (self.config.min_lines_of_code <= 
                   symbol.complexity.lines_of_code <= 
                   self.config.max_lines_of_code):
                continue
            
            # Filter by token budget (already done in parser, but double-check)
            if not (self.config.token_budget[0] <= 
                   symbol.token_count <= 
                   self.config.token_budget[1]):
                continue
            
            filtered.append(symbol)
        
        return filtered
    
    def _should_include_symbol_type(self, symbol: CodeSymbol) -> bool:
        """Check if symbol type should be included based on configuration."""
        if symbol.symbol_type == SymbolType.FUNCTION and not self.config.include_functions:
            return False
        if symbol.symbol_type == SymbolType.CLASS and not self.config.include_classes:
            return False
        if symbol.symbol_type in (SymbolType.METHOD, SymbolType.STATIC_METHOD, 
                                SymbolType.CLASS_METHOD, SymbolType.PROPERTY) and not self.config.include_methods:
            return False
        return True
    
    def _rank_and_limit_symbols(self, symbols: List[CodeSymbol]) -> List[CodeSymbol]:
        """Rank symbols by importance and limit to max_symbols_per_file."""
        if len(symbols) <= self.config.max_symbols_per_file:
            return symbols
        
        # Calculate importance score for each symbol
        scored_symbols = []
        for symbol in symbols:
            score = self._calculate_importance_score(symbol)
            scored_symbols.append((score, symbol))
        
        # Sort by score (descending) and take top symbols
        scored_symbols.sort(key=lambda x: x[0], reverse=True)
        return [symbol for _, symbol in scored_symbols[:self.config.max_symbols_per_file]]
    
    def _calculate_importance_score(self, symbol: CodeSymbol) -> float:
        """Calculate importance score for ranking symbols."""
        score = 0.0
        
        # Base score by symbol type
        type_scores = {
            SymbolType.CLASS: 10.0,
            SymbolType.FUNCTION: 8.0,
            SymbolType.METHOD: 6.0,
            SymbolType.ASYNC_FUNCTION: 7.0,
            SymbolType.PROPERTY: 4.0,
            SymbolType.STATIC_METHOD: 5.0,
            SymbolType.CLASS_METHOD: 5.0,
        }
        score += type_scores.get(symbol.symbol_type, 5.0)
        
        # Public API bonus
        if symbol.is_public_api:
            score += 5.0
        
        # Docstring bonus
        if symbol.docstring:
            score += 3.0
            # Longer docstrings indicate more important functions
            score += min(len(symbol.docstring) / 100, 2.0)
        
        # Complexity bonus (moderate complexity is preferred)
        complexity_tier = self.parser.get_complexity_tier(symbol)
        complexity_bonuses = {
            "simple": 2.0,
            "moderate": 4.0,  # Sweet spot
            "complex": 3.0,
            "very_complex": 1.0
        }
        score += complexity_bonuses.get(complexity_tier, 0.0)
        
        # Parameter count bonus (functions with parameters are more interesting)
        score += min(symbol.complexity.num_parameters * 0.5, 3.0)
        
        # Function call bonus (functions that call other functions are more complex)
        score += min(symbol.complexity.num_function_calls * 0.1, 2.0)
        
        # Lines of code bonus (moderate size preferred)
        loc = symbol.complexity.lines_of_code
        if 10 <= loc <= 30:
            score += 2.0
        elif 5 <= loc <= 50:
            score += 1.0
        
        # Penalize very private functions (double underscore)
        if symbol.name.startswith('__') and not symbol.name.endswith('__'):
            score -= 2.0
        
        return score
    
    def _update_stats(self, all_symbols: List[CodeSymbol], 
                     filtered_symbols: List[CodeSymbol], 
                     final_symbols: List[CodeSymbol]):
        """Update extraction statistics."""
        self.stats.total_symbols_found += len(all_symbols)
        self.stats.symbols_after_filtering += len(final_symbols)
        
        # Update type distribution
        for symbol in final_symbols:
            symbol_type_str = symbol.symbol_type.value
            self.stats.symbols_by_type[symbol_type_str] = (
                self.stats.symbols_by_type.get(symbol_type_str, 0) + 1
            )
            
            # Update complexity distribution
            complexity_tier = self.parser.get_complexity_tier(symbol)
            self.stats.complexity_distribution[complexity_tier] = (
                self.stats.complexity_distribution.get(complexity_tier, 0) + 1
            )
            
            # Update token distribution
            token_range = self._get_token_range(symbol.token_count)
            self.stats.token_distribution[token_range] = (
                self.stats.token_distribution.get(token_range, 0) + 1
            )
    
    def _get_token_range(self, token_count: int) -> str:
        """Get token range bucket for statistics."""
        if token_count < 100:
            return "< 100"
        elif token_count < 200:
            return "100-199"
        elif token_count < 300:
            return "200-299"
        elif token_count < 400:
            return "300-399"
        else:
            return "400+"
    
    def get_stats(self) -> ExtractionStats:
        """Get extraction statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset extraction statistics."""
        self.stats = ExtractionStats()
    
    def export_symbols(self, symbols: List[CodeSymbol], output_path: str, format: str = "json"):
        """
        Export symbols to a file.
        
        Args:
            symbols: List of symbols to export
            output_path: Output file path
            format: Export format ("json", "yaml", "csv")
        """
        import json
        
        if format == "json":
            self._export_json(symbols, output_path)
        elif format == "yaml":
            self._export_yaml(symbols, output_path)
        elif format == "csv":
            self._export_csv(symbols, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, symbols: List[CodeSymbol], output_path: str):
        """Export symbols to JSON format."""
        import json
        from dataclasses import asdict
        
        data = []
        for symbol in symbols:
            symbol_dict = asdict(symbol)
            # Convert sets to lists for JSON serialization
            symbol_dict['dependencies'] = list(symbol_dict['dependencies'])
            data.append(symbol_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _export_yaml(self, symbols: List[CodeSymbol], output_path: str):
        """Export symbols to YAML format."""
        try:
            import yaml
            from dataclasses import asdict
            
            data = []
            for symbol in symbols:
                symbol_dict = asdict(symbol)
                symbol_dict['dependencies'] = list(symbol_dict['dependencies'])
                data.append(symbol_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            raise ImportError("PyYAML is required for YAML export")
    
    def _export_csv(self, symbols: List[CodeSymbol], output_path: str):
        """Export symbols to CSV format (simplified)."""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'name', 'type', 'start_line', 'end_line', 'token_count',
                'complexity_tier', 'lines_of_code', 'num_parameters',
                'has_docstring', 'is_public_api', 'parent_class'
            ])
            
            # Write data
            for symbol in symbols:
                complexity_tier = self.parser.get_complexity_tier(symbol)
                writer.writerow([
                    symbol.name,
                    symbol.symbol_type.value,
                    symbol.start_line,
                    symbol.end_line,
                    symbol.token_count,
                    complexity_tier,
                    symbol.complexity.lines_of_code,
                    symbol.complexity.num_parameters,
                    bool(symbol.docstring),
                    symbol.is_public_api,
                    symbol.parent_class or ""
                ])


def create_extraction_config(
    complexity_level: str = "moderate",
    include_private: bool = False,
    max_symbols: int = 10,
    token_range: Tuple[int, int] = (200, 400)
) -> ExtractionConfig:
    """
    Create a pre-configured extraction config for common use cases.
    
    Args:
        complexity_level: Target complexity ("simple", "moderate", "complex", "all")
        include_private: Whether to include private symbols
        max_symbols: Maximum symbols per file
        token_range: (min_tokens, max_tokens)
        
    Returns:
        Configured ExtractionConfig
    """
    config = ExtractionConfig(
        token_budget=token_range,
        max_symbols_per_file=max_symbols,
        include_private=include_private,
        complexity_filter=complexity_level if complexity_level != "all" else None
    )
    
    return config 