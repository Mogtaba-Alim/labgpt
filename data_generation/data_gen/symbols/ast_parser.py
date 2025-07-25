"""
AST Parser for Code Symbol Extraction and Complexity Analysis

This module provides sophisticated parsing of Python code to extract functions, classes,
and other code symbols with detailed complexity metrics and contextual information.
"""

import ast
import re
import sys
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import tokenize
from io import StringIO


class SymbolType(Enum):
    """Types of code symbols that can be extracted."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    ASYNC_FUNCTION = "async_function"
    PROPERTY = "property"
    STATIC_METHOD = "static_method"
    CLASS_METHOD = "class_method"


@dataclass
class ComplexityMetrics:
    """Detailed complexity metrics for a code symbol."""
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    lines_of_code: int = 0
    num_parameters: int = 0
    num_return_statements: int = 0
    num_conditional_blocks: int = 0
    num_loop_blocks: int = 0
    num_try_except_blocks: int = 0
    nesting_depth: int = 0
    num_function_calls: int = 0
    num_imports: int = 0
    halstead_volume: float = 0.0
    maintainability_index: float = 0.0


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, method) with metadata."""
    name: str
    symbol_type: SymbolType
    start_line: int
    end_line: int
    source_code: str
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_annotation: Optional[str] = None
    parent_class: Optional[str] = None
    complexity: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    imports: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    token_count: int = 0
    is_private: bool = False
    is_public_api: bool = True


class ComplexityAnalyzer(ast.NodeVisitor):
    """Analyzes complexity metrics for code symbols."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset analyzer state for new symbol analysis."""
        self.cyclomatic_complexity = 1  # Base complexity
        self.cognitive_complexity = 0
        self.nesting_depth = 0
        self.max_nesting_depth = 0
        self.current_depth = 0
        self.return_count = 0
        self.conditional_count = 0
        self.loop_count = 0
        self.try_except_count = 0
        self.function_call_count = 0
        self.operators = []
        self.operands = []
        
    def visit_If(self, node):
        """Visit if statements and calculate complexity."""
        self.cyclomatic_complexity += 1
        self.conditional_count += 1
        self.current_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_depth)
        
        # Cognitive complexity increases with nesting
        self.cognitive_complexity += 1 + self.current_depth - 1
        
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_While(self, node):
        """Visit while loops."""
        self.cyclomatic_complexity += 1
        self.loop_count += 1
        self.current_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_depth)
        self.cognitive_complexity += 1 + self.current_depth - 1
        
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_For(self, node):
        """Visit for loops."""
        self.cyclomatic_complexity += 1
        self.loop_count += 1
        self.current_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_depth)
        self.cognitive_complexity += 1 + self.current_depth - 1
        
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_ExceptHandler(self, node):
        """Visit exception handlers."""
        self.cyclomatic_complexity += 1
        self.try_except_count += 1
        self.cognitive_complexity += 1
        self.generic_visit(node)
    
    def visit_Return(self, node):
        """Visit return statements."""
        self.return_count += 1
        if self.current_depth > 0:  # Return inside control structure
            self.cognitive_complexity += 1
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit function calls."""
        self.function_call_count += 1
        # Track operators for Halstead metrics
        if isinstance(node.func, ast.Name):
            self.operators.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.operators.append(node.func.attr)
        self.generic_visit(node)
    
    def visit_Name(self, node):
        """Visit variable names for Halstead metrics."""
        self.operands.append(node.id)
        self.generic_visit(node)
    
    def calculate_halstead_metrics(self) -> float:
        """Calculate Halstead volume metric."""
        if not self.operators and not self.operands:
            return 0.0
        
        n1 = len(set(self.operators))  # Unique operators
        n2 = len(set(self.operands))   # Unique operands
        N1 = len(self.operators)       # Total operators
        N2 = len(self.operands)        # Total operands
        
        if n1 == 0 or n2 == 0:
            return 0.0
        
        vocabulary = n1 + n2
        length = N1 + N2
        
        if vocabulary <= 0:
            return 0.0
        
        import math
        volume = length * math.log2(vocabulary)
        return volume
    
    def calculate_maintainability_index(self, loc: int, halstead_volume: float) -> float:
        """Calculate maintainability index."""
        if loc <= 0:
            return 100.0
        
        import math
        mi = (171 - 5.2 * math.log(halstead_volume + 1) - 
              0.23 * self.cyclomatic_complexity - 16.2 * math.log(loc))
        return max(0.0, min(100.0, mi))


class ASTParser:
    """Advanced AST parser for extracting code symbols with complexity analysis."""
    
    def __init__(self, token_budget: Tuple[int, int] = (200, 400)):
        """
        Initialize parser with token budget constraints.
        
        Args:
            token_budget: (min_tokens, max_tokens) for symbol extraction
        """
        self.min_tokens, self.max_tokens = token_budget
        self.complexity_analyzer = ComplexityAnalyzer()
    
    def parse_file(self, file_path: str, source_code: str) -> List[CodeSymbol]:
        """
        Parse a Python file and extract all code symbols.
        
        Args:
            file_path: Path to the source file
            source_code: Source code content
            
        Returns:
            List of extracted code symbols
        """
        try:
            tree = ast.parse(source_code)
            return self._extract_symbols(tree, source_code, file_path)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return []
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
    
    def _extract_symbols(self, tree: ast.AST, source_code: str, file_path: str) -> List[CodeSymbol]:
        """Extract all symbols from AST."""
        symbols = []
        source_lines = source_code.split('\n')
        
        # Extract top-level imports
        file_imports = self._extract_imports(tree)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = self._extract_function(node, source_lines, file_imports)
                if symbol and self._meets_token_budget(symbol):
                    symbols.append(symbol)
            
            elif isinstance(node, ast.ClassDef):
                class_symbol = self._extract_class(node, source_lines, file_imports)
                if class_symbol and self._meets_token_budget(class_symbol):
                    symbols.append(class_symbol)
                
                # Extract methods from class
                for method_node in node.body:
                    if isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_symbol = self._extract_method(
                            method_node, node.name, source_lines, file_imports
                        )
                        if method_symbol and self._meets_token_budget(method_symbol):
                            symbols.append(method_symbol)
        
        return symbols
    
    def _extract_function(self, node: ast.FunctionDef, source_lines: List[str], 
                         file_imports: List[str]) -> Optional[CodeSymbol]:
        """Extract function symbol with metadata."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Extract source code
        source_code = '\n'.join(source_lines[start_line-1:end_line])
        
        # Determine function type
        symbol_type = SymbolType.ASYNC_FUNCTION if isinstance(node, ast.AsyncFunctionDef) else SymbolType.FUNCTION
        
        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
                if decorator.id in ['property']:
                    symbol_type = SymbolType.PROPERTY
                elif decorator.id in ['staticmethod']:
                    symbol_type = SymbolType.STATIC_METHOD
                elif decorator.id in ['classmethod']:
                    symbol_type = SymbolType.CLASS_METHOD
        
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            parameters.append(arg.arg)
        
        # Extract return annotation
        return_annotation = None
        if node.returns:
            return_annotation = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Calculate complexity metrics
        complexity = self._calculate_complexity(node, source_code)
        
        # Count tokens
        token_count = self._count_tokens(source_code)
        
        # Determine visibility
        is_private = node.name.startswith('_')
        is_public_api = not is_private and docstring is not None
        
        # Extract dependencies
        dependencies = self._extract_dependencies(node)
        
        return CodeSymbol(
            name=node.name,
            symbol_type=symbol_type,
            start_line=start_line,
            end_line=end_line,
            source_code=source_code,
            docstring=docstring,
            decorators=decorators,
            parameters=parameters,
            return_annotation=return_annotation,
            complexity=complexity,
            imports=file_imports,
            dependencies=dependencies,
            token_count=token_count,
            is_private=is_private,
            is_public_api=is_public_api
        )
    
    def _extract_class(self, node: ast.ClassDef, source_lines: List[str], 
                      file_imports: List[str]) -> Optional[CodeSymbol]:
        """Extract class symbol with metadata."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Extract only class definition (without methods)
        class_def_end = start_line
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                break
            class_def_end = item.end_lineno or item.lineno
        
        source_code = '\n'.join(source_lines[start_line-1:class_def_end])
        
        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
        
        # Extract base classes
        parameters = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                parameters.append(base.id)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Calculate complexity (class-level)
        complexity = self._calculate_complexity(node, source_code)
        
        # Count tokens
        token_count = self._count_tokens(source_code)
        
        # Determine visibility
        is_private = node.name.startswith('_')
        is_public_api = not is_private and docstring is not None
        
        # Extract dependencies
        dependencies = self._extract_dependencies(node)
        
        return CodeSymbol(
            name=node.name,
            symbol_type=SymbolType.CLASS,
            start_line=start_line,
            end_line=class_def_end,
            source_code=source_code,
            docstring=docstring,
            decorators=decorators,
            parameters=parameters,  # Base classes
            complexity=complexity,
            imports=file_imports,
            dependencies=dependencies,
            token_count=token_count,
            is_private=is_private,
            is_public_api=is_public_api
        )
    
    def _extract_method(self, node: ast.FunctionDef, class_name: str, source_lines: List[str], 
                       file_imports: List[str]) -> Optional[CodeSymbol]:
        """Extract method symbol with class context."""
        symbol = self._extract_function(node, source_lines, file_imports)
        if symbol:
            symbol.symbol_type = SymbolType.METHOD
            symbol.parent_class = class_name
        return symbol
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from the file."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports
    
    def _extract_dependencies(self, node: ast.AST) -> Set[str]:
        """Extract function/variable dependencies from node."""
        dependencies = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                dependencies.add(child.id)
            elif isinstance(child, ast.Attribute):
                dependencies.add(child.attr)
        return dependencies
    
    def _calculate_complexity(self, node: ast.AST, source_code: str) -> ComplexityMetrics:
        """Calculate detailed complexity metrics for a code symbol."""
        self.complexity_analyzer.reset()
        self.complexity_analyzer.visit(node)
        
        lines_of_code = len([line for line in source_code.split('\n') if line.strip()])
        
        # Calculate parameters for functions
        num_parameters = 0
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            num_parameters = len(node.args.args)
        
        # Calculate Halstead metrics
        halstead_volume = self.complexity_analyzer.calculate_halstead_metrics()
        maintainability_index = self.complexity_analyzer.calculate_maintainability_index(
            lines_of_code, halstead_volume
        )
        
        return ComplexityMetrics(
            cyclomatic_complexity=self.complexity_analyzer.cyclomatic_complexity,
            cognitive_complexity=self.complexity_analyzer.cognitive_complexity,
            lines_of_code=lines_of_code,
            num_parameters=num_parameters,
            num_return_statements=self.complexity_analyzer.return_count,
            num_conditional_blocks=self.complexity_analyzer.conditional_count,
            num_loop_blocks=self.complexity_analyzer.loop_count,
            num_try_except_blocks=self.complexity_analyzer.try_except_count,
            nesting_depth=self.complexity_analyzer.max_nesting_depth,
            num_function_calls=self.complexity_analyzer.function_call_count,
            halstead_volume=halstead_volume,
            maintainability_index=maintainability_index
        )
    
    def _count_tokens(self, source_code: str) -> int:
        """Count tokens in source code using Python tokenizer."""
        try:
            tokens = list(tokenize.generate_tokens(StringIO(source_code).readline))
            # Filter out ENCODING, ENDMARKER, NL, NEWLINE, COMMENT tokens
            filtered_tokens = [
                token for token in tokens 
                if token.type not in (tokenize.ENCODING, tokenize.ENDMARKER, 
                                    tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT)
            ]
            return len(filtered_tokens)
        except tokenize.TokenError:
            # Fallback to simple whitespace splitting
            return len(source_code.split())
    
    def _meets_token_budget(self, symbol: CodeSymbol) -> bool:
        """Check if symbol meets token budget constraints."""
        return self.min_tokens <= symbol.token_count <= self.max_tokens
    
    def get_complexity_tier(self, symbol: CodeSymbol) -> str:
        """Determine complexity tier for symbol."""
        complexity = symbol.complexity
        
        # Calculate composite complexity score
        score = (complexity.cyclomatic_complexity * 2 + 
                complexity.cognitive_complexity * 1.5 + 
                complexity.lines_of_code * 0.1 +
                complexity.nesting_depth * 3)
        
        if score < 10:
            return "simple"
        elif score < 25:
            return "moderate"
        elif score < 50:
            return "complex"
        else:
            return "very_complex" 