"""
Multi-Language Parser for Code Symbol Extraction

This module provides universal parsing of code symbols from multiple programming languages
including Python, R, C, and C++ using language-specific parsing strategies.
"""

import re
import os
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging



class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    R = "r"
    C = "c"
    CPP = "cpp"
    UNKNOWN = "unknown"


@dataclass
class UniversalComplexityMetrics:
    """Universal complexity metrics that work across languages."""
    lines_of_code: int = 0
    num_parameters: int = 0
    num_return_statements: int = 0
    num_conditional_blocks: int = 0
    num_loop_blocks: int = 0
    num_try_catch_blocks: int = 0
    nesting_depth: int = 0
    num_function_calls: int = 0
    token_count: int = 0
    comment_density: float = 0.0
    complexity_score: float = 0.0


@dataclass
class UniversalCodeSymbol:
    """Universal code symbol that works across languages."""
    name: str
    symbol_type: str  # 'function', 'class', 'method', 'struct', 'namespace'
    language: Language
    start_line: int
    end_line: int
    source_code: str
    docstring: Optional[str] = None
    comments: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    parent_scope: Optional[str] = None
    complexity: UniversalComplexityMetrics = field(default_factory=UniversalComplexityMetrics)
    dependencies: Set[str] = field(default_factory=set)
    is_private: bool = False
    is_public_api: bool = True
    file_path: str = ""


class BaseLanguageParser:
    """Base class for language-specific parsers."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def parse_file(self, file_path: str, source_code: str) -> List[UniversalCodeSymbol]:
        """Parse a file and extract symbols."""
        raise NotImplementedError
    
    def extract_functions(self, source_code: str) -> List[UniversalCodeSymbol]:
        """Extract function symbols from source code."""
        raise NotImplementedError
    
    def extract_classes(self, source_code: str) -> List[UniversalCodeSymbol]:
        """Extract class/struct symbols from source code."""
        raise NotImplementedError
    
    def calculate_complexity(self, source_code: str) -> UniversalComplexityMetrics:
        """Calculate complexity metrics for a code block."""
        metrics = UniversalComplexityMetrics()
        
        lines = source_code.strip().split('\n')
        metrics.lines_of_code = len([line for line in lines if line.strip()])
        
        # Count common patterns across languages
        metrics.num_conditional_blocks = self._count_conditionals(source_code)
        metrics.num_loop_blocks = self._count_loops(source_code)
        metrics.num_return_statements = self._count_returns(source_code)
        metrics.nesting_depth = self._calculate_nesting_depth(source_code)
        metrics.token_count = len(source_code.split())
        metrics.comment_density = self._calculate_comment_density(source_code)
        
        # Simple complexity score
        metrics.complexity_score = (
            metrics.num_conditional_blocks * 2 +
            metrics.num_loop_blocks * 2 +
            metrics.nesting_depth * 3 +
            metrics.lines_of_code * 0.1
        )
        
        return metrics
    
    def _count_conditionals(self, source_code: str) -> int:
        """Count conditional statements."""
        patterns = [r'\bif\b', r'\belse\b', r'\belif\b', r'\belse if\b', r'\bswitch\b', r'\bcase\b']
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, source_code, re.IGNORECASE))
        return count
    
    def _count_loops(self, source_code: str) -> int:
        """Count loop statements."""
        patterns = [r'\bfor\b', r'\bwhile\b', r'\bdo\b', r'\brepeat\b']
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, source_code, re.IGNORECASE))
        return count
    
    def _count_returns(self, source_code: str) -> int:
        """Count return statements."""
        return len(re.findall(r'\breturn\b', source_code, re.IGNORECASE))
    
    def _calculate_nesting_depth(self, source_code: str) -> int:
        """Calculate maximum nesting depth."""
        depth = 0
        max_depth = 0
        
        for char in source_code:
            if char == '{':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == '}':
                depth = max(0, depth - 1)
        
        return max_depth
    
    def _calculate_comment_density(self, source_code: str) -> float:
        """Calculate comment density (comments / total lines)."""
        lines = source_code.split('\n')
        comment_lines = 0
        total_lines = len(lines)
        
        if total_lines == 0:
            return 0.0
        
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('#') or 
                stripped.startswith('//') or 
                stripped.startswith('/*') or
                stripped.startswith('*')):
                comment_lines += 1
        
        return comment_lines / total_lines


class PythonParser(BaseLanguageParser):
    """Parser for Python files."""

    def parse_file(self, file_path: str, source_code: str) -> List[UniversalCodeSymbol]:
        """Parse Python file using regex patterns."""
        symbols = []
        symbols.extend(self.extract_functions(source_code))
        symbols.extend(self.extract_classes(source_code))

        # Add file path to all symbols
        for symbol in symbols:
            symbol.file_path = file_path

        return symbols

    def extract_functions(self, source_code: str) -> List[UniversalCodeSymbol]:
        """Extract Python functions using regex patterns."""
        functions = []
        pattern = r'^(\s*)def\s+(\w+)\s*\((.*?)\).*?:(.*)$'
        
        lines = source_code.split('\n')
        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                indent, name, params, rest = match.groups()
                
                # Find function end
                start_line = i + 1
                end_line = self._find_function_end(lines, i, len(indent))
                
                function_code = '\n'.join(lines[i:end_line])
                
                # Extract parameters
                param_list = [p.strip().split(':')[0].split('=')[0].strip() 
                             for p in params.split(',') if p.strip()]
                
                complexity = self.calculate_complexity(function_code)
                complexity.num_parameters = len(param_list)
                
                symbol = UniversalCodeSymbol(
                    name=name,
                    symbol_type='function',
                    language=Language.PYTHON,
                    start_line=start_line,
                    end_line=end_line,
                    source_code=function_code,
                    parameters=param_list,
                    complexity=complexity,
                    is_private=name.startswith('_')
                )
                functions.append(symbol)
        
        return functions
    
    def _find_function_end(self, lines: List[str], start_idx: int, base_indent: int) -> int:
        """Find the end of a Python function."""
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue

            # If we find a line with same or less indentation, function ends
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= base_indent and line.strip():
                return i

        return len(lines)

    def extract_classes(self, source_code: str) -> List[UniversalCodeSymbol]:
        """Extract Python classes."""
        classes = []

        # Python class pattern: class ClassName(Base):
        pattern = r'^(\s*)class\s+(\w+)(?:\s*\((.*?)\))?\s*:'

        lines = source_code.split('\n')
        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                indent, name, bases = match.groups()

                # Find class end
                start_line = i + 1
                end_line = self._find_class_end(lines, i, len(indent))

                class_code = '\n'.join(lines[i:end_line])

                # Extract base classes
                base_list = []
                if bases:
                    base_list = [b.strip() for b in bases.split(',') if b.strip()]

                complexity = self.calculate_complexity(class_code)
                complexity.num_parameters = len(base_list)

                symbol = UniversalCodeSymbol(
                    name=name,
                    symbol_type='class',
                    language=Language.PYTHON,
                    start_line=start_line,
                    end_line=end_line,
                    source_code=class_code,
                    parameters=base_list,
                    complexity=complexity,
                    is_private=name.startswith('_'),
                    is_public_api=not name.startswith('_')
                )
                classes.append(symbol)

        return classes

    def _find_class_end(self, lines: List[str], start_idx: int, base_indent: int) -> int:
        """Find the end of a Python class."""
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue

            # If we find a line with same or less indentation, class ends
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= base_indent and line.strip():
                return i

        return len(lines)


class RParser(BaseLanguageParser):
    """Parser for R files."""
    
    def parse_file(self, file_path: str, source_code: str) -> List[UniversalCodeSymbol]:
        """Parse R file."""
        symbols = []
        symbols.extend(self.extract_functions(source_code))
        
        # Add file path to all symbols
        for symbol in symbols:
            symbol.file_path = file_path
        
        return symbols
    
    def extract_functions(self, source_code: str) -> List[UniversalCodeSymbol]:
        """Extract R functions."""
        functions = []
        
        # R function patterns: name <- function(params) { ... }
        pattern = r'(\w+)\s*<-\s*function\s*\((.*?)\)\s*\{'
        
        lines = source_code.split('\n')
        full_text = source_code
        
        for match in re.finditer(pattern, full_text, re.MULTILINE | re.DOTALL):
            name = match.group(1)
            params = match.group(2)
            start_pos = match.start()
            
            # Find start line
            start_line = full_text[:start_pos].count('\n') + 1
            
            # Find matching closing brace
            brace_count = 0
            pos = match.end() - 1  # Start from the opening brace
            end_pos = pos
            
            for i in range(pos, len(full_text)):
                if full_text[i] == '{':
                    brace_count += 1
                elif full_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            end_line = full_text[:end_pos].count('\n') + 1
            function_code = full_text[start_pos:end_pos]
            
            # Extract parameters
            param_list = [p.strip().split('=')[0].strip() 
                         for p in params.split(',') if p.strip()]
            
            # Extract comments (R uses # for comments)
            comments = re.findall(r'#.*', function_code)
            
            complexity = self.calculate_complexity(function_code)
            complexity.num_parameters = len(param_list)
            
            symbol = UniversalCodeSymbol(
                name=name,
                symbol_type='function',
                language=Language.R,
                start_line=start_line,
                end_line=end_line,
                source_code=function_code,
                comments=comments,
                parameters=param_list,
                complexity=complexity,
                is_private=name.startswith('.'),  # R private functions start with .
                is_public_api=not name.startswith('.')
            )
            functions.append(symbol)
        
        return functions


class CParser(BaseLanguageParser):
    """Parser for C files."""
    
    def parse_file(self, file_path: str, source_code: str) -> List[UniversalCodeSymbol]:
        """Parse C file."""
        symbols = []
        symbols.extend(self.extract_functions(source_code))
        symbols.extend(self.extract_structs(source_code))
        
        # Add file path to all symbols
        for symbol in symbols:
            symbol.file_path = file_path
        
        return symbols
    
    def extract_functions(self, source_code: str) -> List[UniversalCodeSymbol]:
        """Extract C functions."""
        functions = []
        
        # C function pattern: return_type function_name(params) { ... }
        pattern = r'(\w+(?:\s*\*)*)\s+(\w+)\s*\((.*?)\)\s*\{'
        
        for match in re.finditer(pattern, source_code, re.MULTILINE | re.DOTALL):
            return_type = match.group(1).strip()
            name = match.group(2)
            params = match.group(3)
            start_pos = match.start()
            
            # Skip if this looks like a macro or keyword
            if return_type in ['if', 'while', 'for', 'switch']:
                continue
            
            # Find start line
            start_line = source_code[:start_pos].count('\n') + 1
            
            # Find matching closing brace
            brace_count = 0
            pos = match.end() - 1  # Start from the opening brace
            end_pos = pos
            
            for i in range(pos, len(source_code)):
                if source_code[i] == '{':
                    brace_count += 1
                elif source_code[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            end_line = source_code[:end_pos].count('\n') + 1
            function_code = source_code[start_pos:end_pos]
            
            # Extract parameters
            param_list = []
            if params.strip():
                for param in params.split(','):
                    param = param.strip()
                    if param and param != 'void':
                        # Extract parameter name (last token)
                        tokens = param.split()
                        if tokens:
                            param_list.append(tokens[-1].strip('*'))
            
            # Extract comments
            comments = re.findall(r'//.*|/\*.*?\*/', function_code, re.DOTALL)
            
            complexity = self.calculate_complexity(function_code)
            complexity.num_parameters = len(param_list)
            
            symbol = UniversalCodeSymbol(
                name=name,
                symbol_type='function',
                language=Language.C,
                start_line=start_line,
                end_line=end_line,
                source_code=function_code,
                comments=comments,
                parameters=param_list,
                return_type=return_type,
                complexity=complexity,
                is_private=name.startswith('_'),  # C private functions often start with _
                is_public_api=not name.startswith('_')
            )
            functions.append(symbol)
        
        return functions
    
    def extract_structs(self, source_code: str) -> List[UniversalCodeSymbol]:
        """Extract C structs."""
        structs = []
        
        # C struct pattern: struct name { ... };
        pattern = r'struct\s+(\w+)\s*\{([^}]*)\}'
        
        for match in re.finditer(pattern, source_code, re.MULTILINE | re.DOTALL):
            name = match.group(1)
            body = match.group(2)
            start_pos = match.start()
            end_pos = match.end()
            
            start_line = source_code[:start_pos].count('\n') + 1
            end_line = source_code[:end_pos].count('\n') + 1
            
            struct_code = match.group(0)
            
            # Extract member variables as "parameters"
            members = []
            for line in body.split('\n'):
                line = line.strip()
                if line and not line.startswith('//') and not line.startswith('/*'):
                    # Simple member extraction
                    tokens = line.split()
                    if len(tokens) >= 2:
                        members.append(tokens[-1].rstrip(';'))
            
            complexity = self.calculate_complexity(struct_code)
            complexity.num_parameters = len(members)
            
            symbol = UniversalCodeSymbol(
                name=name,
                symbol_type='struct',
                language=Language.C,
                start_line=start_line,
                end_line=end_line,
                source_code=struct_code,
                parameters=members,
                complexity=complexity,
                is_public_api=True
            )
            structs.append(symbol)
        
        return structs


class CppParser(CParser):
    """Parser for C++ files."""
    
    def parse_file(self, file_path: str, source_code: str) -> List[UniversalCodeSymbol]:
        """Parse C++ file."""
        symbols = []
        symbols.extend(self.extract_functions(source_code))
        symbols.extend(self.extract_classes(source_code))
        symbols.extend(self.extract_structs(source_code))
        
        # Add file path to all symbols
        for symbol in symbols:
            symbol.file_path = file_path
            symbol.language = Language.CPP
        
        return symbols
    
    def extract_classes(self, source_code: str) -> List[UniversalCodeSymbol]:
        """Extract C++ classes."""
        classes = []
        
        # C++ class pattern: class name { ... };
        pattern = r'class\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
        
        for match in re.finditer(pattern, source_code, re.MULTILINE | re.DOTALL):
            name = match.group(1)
            body = match.group(2)
            start_pos = match.start()
            end_pos = match.end()
            
            start_line = source_code[:start_pos].count('\n') + 1
            end_line = source_code[:end_pos].count('\n') + 1
            
            class_code = match.group(0)
            
            # Extract methods and members
            methods = []
            members = []
            
            # Simple extraction of public methods
            method_pattern = r'(\w+(?:\s*\*)*)\s+(\w+)\s*\([^)]*\)\s*(?:\{|;)'
            for method_match in re.finditer(method_pattern, body):
                method_name = method_match.group(2)
                if method_name not in ['public', 'private', 'protected']:
                    methods.append(method_name)
            
            complexity = self.calculate_complexity(class_code)
            complexity.num_parameters = len(methods)  # Use methods as parameters for classes
            
            symbol = UniversalCodeSymbol(
                name=name,
                symbol_type='class',
                language=Language.CPP,
                start_line=start_line,
                end_line=end_line,
                source_code=class_code,
                parameters=methods,
                complexity=complexity,
                is_public_api=True
            )
            classes.append(symbol)
        
        return classes


class MultiLanguageParser:
    """Universal parser that handles multiple programming languages."""
    
    def __init__(self):
        self.parsers = {
            Language.PYTHON: PythonParser(),
            Language.R: RParser(),
            Language.C: CParser(),
            Language.CPP: CppParser(),
        }
        self.logger = logging.getLogger(__name__)
    
    def detect_language(self, file_path: str) -> Language:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.py':
            return Language.PYTHON
        elif ext == '.r':
            return Language.R
        elif ext == '.c':
            return Language.C
        elif ext in ['.cpp', '.cxx', '.cc', '.hpp', '.h++']:
            return Language.CPP
        elif ext == '.h':
            # Could be C or C++, default to C
            return Language.C
        else:
            return Language.UNKNOWN
    
    def parse_file(self, file_path: str, source_code: str) -> List[UniversalCodeSymbol]:
        """Parse a file using the appropriate language parser."""
        language = self.detect_language(file_path)
        
        if language == Language.UNKNOWN:
            self.logger.warning(f"Unknown language for file: {file_path}")
            return []
        
        if language not in self.parsers:
            self.logger.warning(f"No parser available for language: {language}")
            return []
        
        try:
            return self.parsers[language].parse_file(file_path, source_code)
        except Exception as e:
            self.logger.error(f"Error parsing {file_path} with {language} parser: {e}")
            return []
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return ['.py', '.r', '.c', '.cpp', '.cxx', '.cc', '.h', '.hpp', '.h++']
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file is supported by any parser."""
        return self.detect_language(file_path) != Language.UNKNOWN