"""
Universal Symbol Extractor for Multi-Language Code Analysis

This module provides high-level extraction of code symbols from files in multiple programming
languages (Python, R, C, C++) with intelligent filtering, token budget management, and symbol ranking.
"""

import os
import glob
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .multi_language_parser import MultiLanguageParser, UniversalCodeSymbol, Language


@dataclass
class UniversalExtractionConfig:
    """Configuration for universal symbol extraction."""
    min_tokens: int = 30
    max_symbols_per_file: int = 30
    include_private: bool = False
    min_lines_of_code: int = 3
    supported_languages: List[Language] = field(default_factory=lambda: [
        Language.PYTHON, Language.R, Language.C, Language.CPP
    ])


@dataclass
class UniversalExtractionStats:
    """Statistics from universal symbol extraction process."""
    files_processed: int = 0
    files_with_errors: int = 0
    total_symbols_found: int = 0
    symbols_after_filtering: int = 0
    symbols_by_type: Dict[str, int] = field(default_factory=dict)
    symbols_by_language: Dict[str, int] = field(default_factory=dict)
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    token_distribution: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class UniversalSymbolExtractor:
    """Universal symbol extractor for multi-language code analysis."""
    
    def __init__(self, config: Optional[UniversalExtractionConfig] = None):
        """Initialize the universal symbol extractor with configuration."""
        self.config = config or UniversalExtractionConfig()
        self.parser = MultiLanguageParser()
        self.stats = UniversalExtractionStats()
        self.logger = logging.getLogger(__name__)
    
    def extract_from_file(self, file_path: str) -> List[UniversalCodeSymbol]:
        """
        Extract symbols from a single file.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            List of extracted and filtered code symbols
        """
        try:
            # Check if file is supported
            if not self.parser.is_supported(file_path):
                return []
            
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
    
    def extract_from_directory(self, directory_path: str, recursive: bool = True) -> List[UniversalCodeSymbol]:
        """
        Extract symbols from all supported files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of all extracted symbols
        """
        all_symbols = []
        
        # Find all supported files
        supported_extensions = self.parser.get_supported_extensions()
        pattern = "**/*" if recursive else "*"
        
        for ext in supported_extensions:
            file_pattern = os.path.join(directory_path, f"{pattern}{ext}")
            for file_path in glob.glob(file_pattern, recursive=recursive):
                if os.path.isfile(file_path):
                    symbols = self.extract_from_file(file_path)
                    all_symbols.extend(symbols)
        
        self.logger.info(f"Extracted {len(all_symbols)} total symbols from {directory_path}")
        return all_symbols
    
    def extract_from_repo(self, repo_path: str) -> Dict[str, List[UniversalCodeSymbol]]:
        """
        Extract symbols from a repository, organized by file.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Dictionary mapping file paths to extracted symbols
        """
        symbols_by_file = {}
        supported_extensions = self.parser.get_supported_extensions()
        
        for ext in supported_extensions:
            file_pattern = os.path.join(repo_path, f"**/*{ext}")
            for file_path in glob.glob(file_pattern, recursive=True):
                if os.path.isfile(file_path):
                    symbols = self.extract_from_file(file_path)
                    if symbols:  # Only include files with extracted symbols
                        relative_path = os.path.relpath(file_path, repo_path)
                        symbols_by_file[relative_path] = symbols
        
        return symbols_by_file
    
    def _filter_symbols(self, symbols: List[UniversalCodeSymbol]) -> List[UniversalCodeSymbol]:
        """Filter symbols based on configuration criteria."""
        filtered = []

        for symbol in symbols:
            # Filter by language
            if symbol.language not in self.config.supported_languages:
                continue

            # Filter by visibility
            if symbol.is_private and not self.config.include_private:
                continue

            # Filter by minimum lines of code
            if symbol.complexity.lines_of_code < self.config.min_lines_of_code:
                continue

            # Filter by minimum token count
            if symbol.complexity.token_count < self.config.min_tokens:
                continue

            filtered.append(symbol)

        return filtered

    def _rank_and_limit_symbols(self, symbols: List[UniversalCodeSymbol]) -> List[UniversalCodeSymbol]:
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
    
    def _calculate_importance_score(self, symbol: UniversalCodeSymbol) -> float:
        """Calculate importance score for ranking symbols."""
        score = 0.0
        
        # Base score by symbol type
        type_scores = {
            'class': 10.0,
            'function': 8.0,
            'method': 6.0,
            'struct': 9.0,
        }
        score += type_scores.get(symbol.symbol_type, 5.0)
        
        # Public API bonus
        if symbol.is_public_api:
            score += 5.0
        
        # Documentation bonus
        if symbol.docstring:
            score += 3.0
            score += min(len(symbol.docstring) / 100, 2.0)
        
        # Comments bonus
        if symbol.comments:
            score += len(symbol.comments) * 0.5
        
        # Complexity bonus (moderate complexity is preferred)
        complexity_tier = self.get_complexity_tier(symbol)
        complexity_bonuses = {
            "simple": 1.0,
            "moderate": 4.0,
            "complex": 3.0,
            "very_complex": 2.0
        }
        score += complexity_bonuses.get(complexity_tier, 0.0)
        
        # Parameter count bonus (functions with parameters are more interesting)
        score += min(symbol.complexity.num_parameters * 0.5, 3.0)
        
        # Function call bonus
        score += min(symbol.complexity.num_function_calls * 0.1, 2.0)
        
        # Lines of code bonus (moderate size preferred)
        loc = symbol.complexity.lines_of_code
        if 10 <= loc <= 30:
            score += 2.0
        elif 5 <= loc <= 50:
            score += 1.0
        
        # Language-specific adjustments
        language_bonuses = {
            Language.PYTHON: 1.0,
            Language.R: 1.2,  # R functions might be rarer
            Language.C: 1.1,
            Language.CPP: 1.1,
        }
        score += language_bonuses.get(symbol.language, 1.0)
        
        # Penalize very private functions
        if symbol.name.startswith('__') or symbol.name.startswith('_.'):
            score -= 2.0
        
        return score
    
    def get_complexity_tier(self, symbol: UniversalCodeSymbol) -> str:
        """Determine complexity tier for symbol."""
        score = symbol.complexity.complexity_score
        
        if score < 10:
            return "simple"
        elif score < 25:
            return "moderate"
        elif score < 50:
            return "complex"
        else:
            return "very_complex"
    
    def _update_stats(self, all_symbols: List[UniversalCodeSymbol], 
                     filtered_symbols: List[UniversalCodeSymbol], 
                     final_symbols: List[UniversalCodeSymbol]):
        """Update extraction statistics."""
        self.stats.total_symbols_found += len(all_symbols)
        self.stats.symbols_after_filtering += len(final_symbols)
        
        # Update type and language distribution
        for symbol in final_symbols:
            # Type distribution
            self.stats.symbols_by_type[symbol.symbol_type] = (
                self.stats.symbols_by_type.get(symbol.symbol_type, 0) + 1
            )
            
            # Language distribution
            language_str = symbol.language.value
            self.stats.symbols_by_language[language_str] = (
                self.stats.symbols_by_language.get(language_str, 0) + 1
            )
            
            # Complexity distribution
            complexity_tier = self.get_complexity_tier(symbol)
            self.stats.complexity_distribution[complexity_tier] = (
                self.stats.complexity_distribution.get(complexity_tier, 0) + 1
            )
            
            # Token distribution
            token_range = self._get_token_range(symbol.complexity.token_count)
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
    
    def get_stats(self) -> UniversalExtractionStats:
        """Get extraction statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset extraction statistics."""
        self.stats = UniversalExtractionStats()
    
    def export_symbols(self, symbols: List[UniversalCodeSymbol], output_path: str, format: str = "json"):
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
    
    def _export_json(self, symbols: List[UniversalCodeSymbol], output_path: str):
        """Export symbols to JSON format."""
        import json
        from dataclasses import asdict
        
        data = []
        for symbol in symbols:
            symbol_dict = asdict(symbol)
            # Convert sets to lists and enums to strings for JSON serialization
            symbol_dict['dependencies'] = list(symbol_dict['dependencies'])
            symbol_dict['language'] = symbol_dict['language'].value
            data.append(symbol_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _export_yaml(self, symbols: List[UniversalCodeSymbol], output_path: str):
        """Export symbols to YAML format."""
        try:
            import yaml
            from dataclasses import asdict
            
            data = []
            for symbol in symbols:
                symbol_dict = asdict(symbol)
                symbol_dict['dependencies'] = list(symbol_dict['dependencies'])
                symbol_dict['language'] = symbol_dict['language'].value
                data.append(symbol_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            raise ImportError("PyYAML is required for YAML export")
    
    def _export_csv(self, symbols: List[UniversalCodeSymbol], output_path: str):
        """Export symbols to CSV format (simplified)."""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'name', 'type', 'language', 'start_line', 'end_line', 'token_count',
                'complexity_tier', 'lines_of_code', 'num_parameters',
                'has_docstring', 'has_comments', 'is_public_api', 'parent_scope'
            ])
            
            # Write data
            for symbol in symbols:
                complexity_tier = self.get_complexity_tier(symbol)
                writer.writerow([
                    symbol.name,
                    symbol.symbol_type,
                    symbol.language.value,
                    symbol.start_line,
                    symbol.end_line,
                    symbol.complexity.token_count,
                    complexity_tier,
                    symbol.complexity.lines_of_code,
                    symbol.complexity.num_parameters,
                    bool(symbol.docstring),
                    bool(symbol.comments),
                    symbol.is_public_api,
                    symbol.parent_scope or ""
                ])


def create_universal_extraction_config(
    include_private: bool = False,
    max_symbols: int = 30,
    min_tokens: int = 30,
    languages: Optional[List[str]] = None
) -> UniversalExtractionConfig:
    """
    Create a pre-configured universal extraction config for common use cases.

    Args:
        include_private: Whether to include private symbols
        max_symbols: Maximum symbols per file
        min_tokens: Minimum tokens per symbol
        languages: List of language names to support

    Returns:
        Configured UniversalExtractionConfig
    """
    supported_languages = [Language.PYTHON, Language.R, Language.C, Language.CPP]
    if languages:
        language_map = {
            'python': Language.PYTHON,
            'r': Language.R,
            'c': Language.C,
            'cpp': Language.CPP,
            'c++': Language.CPP,
        }
        supported_languages = [language_map[lang.lower()] for lang in languages
                             if lang.lower() in language_map]

    config = UniversalExtractionConfig(
        min_tokens=min_tokens,
        max_symbols_per_file=max_symbols,
        include_private=include_private,
        supported_languages=supported_languages
    )

    return config