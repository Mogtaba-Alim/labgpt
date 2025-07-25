"""
Bug Injector for Debugging Task Generation

This module provides sophisticated bug injection capabilities that apply realistic
code mutations to generate debugging training data with various types of common bugs.
"""

import ast
import re
import random
import copy
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..symbols import CodeSymbol


class BugType(Enum):
    """Types of bugs that can be injected."""
    OFF_BY_ONE = "off_by_one"
    NULL_POINTER = "null_pointer"
    TYPE_MISMATCH = "type_mismatch"
    LOGIC_ERROR = "logic_error"
    BOUNDARY_CONDITION = "boundary_condition"
    VARIABLE_NAME = "variable_name"
    OPERATOR_ERROR = "operator_error"
    INDENTATION_ERROR = "indentation_error"
    MISSING_RETURN = "missing_return"
    INFINITE_LOOP = "infinite_loop"
    EXCEPTION_HANDLING = "exception_handling"
    RESOURCE_LEAK = "resource_leak"


@dataclass
class BugInjection:
    """Represents a bug injection with metadata."""
    bug_type: BugType
    original_code: str
    buggy_code: str
    bug_location: int  # Line number where bug was injected
    bug_description: str
    fix_explanation: str
    severity: str  # "low", "medium", "high", "critical"
    detection_difficulty: str  # "easy", "medium", "hard"
    affected_lines: List[int] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    fix_suggestions: List[str] = field(default_factory=list)


@dataclass
class InjectionStats:
    """Statistics from bug injection process."""
    total_attempts: int = 0
    successful_injections: int = 0
    failed_injections: int = 0
    bugs_by_type: Dict[str, int] = field(default_factory=dict)
    bugs_by_severity: Dict[str, int] = field(default_factory=dict)
    average_detection_difficulty: float = 0.0


class BugInjector:
    """Injects realistic bugs into code for debugging task generation."""
    
    def __init__(self, 
                 enabled_bug_types: Optional[List[BugType]] = None,
                 preserve_syntax: bool = True,
                 difficulty_distribution: Optional[Dict[str, float]] = None):
        """
        Initialize the bug injector.
        
        Args:
            enabled_bug_types: List of bug types to enable (None = all)
            preserve_syntax: Whether to ensure syntactic correctness
            difficulty_distribution: Distribution of difficulty levels
        """
        self.enabled_bug_types = enabled_bug_types or list(BugType)
        self.preserve_syntax = preserve_syntax
        self.difficulty_distribution = difficulty_distribution or {
            "easy": 0.4,
            "medium": 0.4, 
            "hard": 0.2
        }
        self.logger = logging.getLogger(__name__)
        self.stats = InjectionStats()
        
        # Bug injection patterns
        self._init_bug_patterns()
    
    def _init_bug_patterns(self):
        """Initialize bug injection patterns."""
        self.bug_patterns = {
            BugType.OFF_BY_ONE: [
                # Range boundary errors
                (r'range\((\w+)\)', r'range(\1 + 1)'),  # range(n) -> range(n+1)
                (r'range\((\w+),\s*(\w+)\)', r'range(\1, \2 + 1)'),  # range(a,b) -> range(a,b+1)
                (r'for\s+(\w+)\s+in\s+range\(len\((\w+)\)\):', r'for \1 in range(len(\2) + 1):'),
                # Array indexing errors
                (r'(\w+)\[(\w+)\]', r'\1[\2 + 1]'),  # arr[i] -> arr[i+1]
                (r'(\w+)\[(\w+)\s*-\s*1\]', r'\1[\2]'),  # arr[i-1] -> arr[i]
            ],
            
            BugType.NULL_POINTER: [
                # None checks
                (r'if\s+(\w+)\s*:', r'if \1 is not None:'),
                (r'(\w+)\s*=\s*None', r'\1 = None  # Bug: should initialize'),
                (r'return\s+(\w+)', r'return \1 if \1 is not None else None'),
            ],
            
            BugType.TYPE_MISMATCH: [
                # String/int conversions
                (r'(\w+)\s*=\s*input\(\)', r'\1 = input()  # Bug: string instead of int'),
                (r'(\w+)\s*=\s*int\((\w+)\)', r'\1 = \2  # Bug: missing int() conversion'),
                (r'(\w+)\s*\+\s*(\w+)', r'str(\1) + str(\2)  # Bug: string concatenation instead of addition'),
            ],
            
            BugType.LOGIC_ERROR: [
                # Condition inversions
                (r'if\s+(\w+)\s*==\s*(\w+):', r'if \1 != \2:'),
                (r'if\s+(\w+)\s*<\s*(\w+):', r'if \1 > \2:'),
                (r'if\s+(\w+)\s*>\s*(\w+):', r'if \1 < \2:'),
                (r'and\s+', r'or '),
                (r'or\s+', r'and '),
            ],
            
            BugType.OPERATOR_ERROR: [
                # Arithmetic operators
                (r'(\w+)\s*\+\s*(\w+)', r'\1 - \2'),
                (r'(\w+)\s*\*\s*(\w+)', r'\1 + \2'),
                (r'(\w+)\s*//\s*(\w+)', r'\1 / \2'),  # Integer vs float division
                (r'(\w+)\s*%\s*(\w+)', r'\1 // \2'),
            ],
            
            BugType.VARIABLE_NAME: [
                # Common typos
                (r'\blenth\b', r'lenght'),  # length typo
                (r'\bcount\b', r'count_'),  # Variable name conflict
                (r'\bindex\b', r'indx'),    # index typo
                (r'\bvalue\b', r'val'),     # Abbreviation inconsistency
            ]
        }
    
    def inject_bug(self, symbol: CodeSymbol, 
                  bug_type: Optional[BugType] = None) -> Optional[BugInjection]:
        """
        Inject a bug into a code symbol.
        
        Args:
            symbol: Code symbol to inject bug into
            bug_type: Specific bug type (None = random from enabled)
            
        Returns:
            BugInjection object if successful, None if failed
        """
        self.stats.total_attempts += 1
        
        try:
            # Select bug type
            if bug_type is None:
                bug_type = random.choice(self.enabled_bug_types)
            
            # Select injection method based on bug type
            injection_method = getattr(self, f'_inject_{bug_type.value}', None)
            if injection_method is None:
                injection_method = self._inject_generic_pattern
            
            # Attempt injection
            injection = injection_method(symbol, bug_type)
            
            if injection:
                self.stats.successful_injections += 1
                self._update_stats(injection)
                self.logger.info(f"Successfully injected {bug_type.value} bug into {symbol.name}")
                return injection
            else:
                self.stats.failed_injections += 1
                self.logger.warning(f"Failed to inject {bug_type.value} bug into {symbol.name}")
                return None
                
        except Exception as e:
            self.stats.failed_injections += 1
            self.logger.error(f"Error injecting bug: {e}")
            return None
    
    def _inject_generic_pattern(self, symbol: CodeSymbol, bug_type: BugType) -> Optional[BugInjection]:
        """Inject bug using pattern matching."""
        if bug_type not in self.bug_patterns:
            return None
        
        patterns = self.bug_patterns[bug_type]
        original_code = symbol.source_code
        
        for pattern, replacement in patterns:
            if re.search(pattern, original_code):
                buggy_code = re.sub(pattern, replacement, original_code, count=1)
                
                # Find the line where change occurred
                bug_location = self._find_change_location(original_code, buggy_code)
                
                # Validate syntax if required
                if self.preserve_syntax and not self._is_valid_syntax(buggy_code):
                    continue
                
                return self._create_bug_injection(
                    symbol, bug_type, original_code, buggy_code, bug_location
                )
        
        return None
    
    def _inject_off_by_one(self, symbol: CodeSymbol, bug_type: BugType) -> Optional[BugInjection]:
        """Inject off-by-one errors."""
        original_code = symbol.source_code
        lines = original_code.split('\n')
        buggy_lines = lines.copy()
        
        # Look for range() calls
        for i, line in enumerate(lines):
            if 'range(' in line and 'len(' in line:
                # Change range(len(arr)) to range(len(arr) + 1)
                modified_line = re.sub(
                    r'range\(len\((\w+)\)\)',
                    r'range(len(\1) + 1)',
                    line
                )
                if modified_line != line:
                    buggy_lines[i] = modified_line
                    buggy_code = '\n'.join(buggy_lines)
                    
                    if not self.preserve_syntax or self._is_valid_syntax(buggy_code):
                        return BugInjection(
                            bug_type=bug_type,
                            original_code=original_code,
                            buggy_code=buggy_code,
                            bug_location=i + 1,
                            bug_description="Off-by-one error in range bounds",
                            fix_explanation="Remove the '+ 1' from range(len(arr) + 1) to fix the boundary condition",
                            severity="medium",
                            detection_difficulty="easy",
                            affected_lines=[i + 1],
                            symptoms=["IndexError: list index out of range"],
                            fix_suggestions=["Check loop bounds", "Verify array indexing"]
                        )
        
        return None
    
    def _inject_logic_error(self, symbol: CodeSymbol, bug_type: BugType) -> Optional[BugInjection]:
        """Inject logic errors like condition inversions."""
        original_code = symbol.source_code
        lines = original_code.split('\n')
        buggy_lines = lines.copy()
        
        # Look for comparison operators to invert
        for i, line in enumerate(lines):
            if 'if ' in line:
                # Invert comparison operators
                modifications = [
                    (r'==', '!='),
                    (r'!=', '=='),
                    (r'<=', '>'),
                    (r'>=', '<'),
                    (r'(?<![<>=])<(?!=)', '>'),  # < not preceded/followed by other operators
                    (r'(?<![<>=])>(?!=)', '<'),  # > not preceded/followed by other operators
                ]
                
                for pattern, replacement in modifications:
                    if re.search(pattern, line):
                        modified_line = re.sub(pattern, replacement, line, count=1)
                        buggy_lines[i] = modified_line
                        buggy_code = '\n'.join(buggy_lines)
                        
                        if not self.preserve_syntax or self._is_valid_syntax(buggy_code):
                            return BugInjection(
                                bug_type=bug_type,
                                original_code=original_code,
                                buggy_code=buggy_code,
                                bug_location=i + 1,
                                bug_description=f"Logic error: inverted comparison operator",
                                fix_explanation=f"Change '{replacement}' back to '{pattern}' to fix the condition",
                                severity="high",
                                detection_difficulty="medium",
                                affected_lines=[i + 1],
                                symptoms=["Incorrect program behavior", "Wrong results"],
                                fix_suggestions=["Review conditional logic", "Test edge cases"]
                            )
                        break
        
        return None
    
    def _inject_variable_name(self, symbol: CodeSymbol, bug_type: BugType) -> Optional[BugInjection]:
        """Inject variable name errors (typos, scope issues)."""
        original_code = symbol.source_code
        
        # Find variable names in the code
        variables = self._extract_variable_names(original_code)
        if not variables:
            return None
        
        # Select a variable to introduce a typo
        target_var = random.choice(list(variables))
        
        # Create typo variants
        typo_variants = [
            target_var + '_',  # Add underscore
            target_var[:-1] if len(target_var) > 3 else target_var + 'x',  # Remove last char or add x
            target_var.replace('e', 'a') if 'e' in target_var else target_var + 's',  # Common typos
        ]
        
        for typo in typo_variants:
            if typo != target_var and typo not in variables:
                # Replace one occurrence of the variable with typo
                buggy_code = re.sub(r'\b' + re.escape(target_var) + r'\b', typo, original_code, count=1)
                
                if buggy_code != original_code:
                    bug_location = self._find_change_location(original_code, buggy_code)
                    
                    return BugInjection(
                        bug_type=bug_type,
                        original_code=original_code,
                        buggy_code=buggy_code,
                        bug_location=bug_location,
                        bug_description=f"Variable name typo: '{typo}' should be '{target_var}'",
                        fix_explanation=f"Correct the variable name from '{typo}' to '{target_var}'",
                        severity="medium",
                        detection_difficulty="easy",
                        affected_lines=[bug_location],
                        symptoms=["NameError: name not defined", "Undefined variable"],
                        fix_suggestions=["Check variable spelling", "Use IDE auto-completion"]
                    )
        
        return None
    
    def _inject_type_mismatch(self, symbol: CodeSymbol, bug_type: BugType) -> Optional[BugInjection]:
        """Inject type mismatch errors."""
        original_code = symbol.source_code
        lines = original_code.split('\n')
        buggy_lines = lines.copy()
        
        # Look for int() conversions to remove
        for i, line in enumerate(lines):
            if 'int(' in line:
                # Remove int() conversion
                modified_line = re.sub(r'int\(([^)]+)\)', r'\1', line)
                if modified_line != line:
                    buggy_lines[i] = modified_line
                    buggy_code = '\n'.join(buggy_lines)
                    
                    return BugInjection(
                        bug_type=bug_type,
                        original_code=original_code,
                        buggy_code=buggy_code,
                        bug_location=i + 1,
                        bug_description="Missing type conversion from string to integer",
                        fix_explanation="Add int() conversion around the input/variable",
                        severity="medium",
                        detection_difficulty="medium",
                        affected_lines=[i + 1],
                        symptoms=["TypeError: unsupported operand types", "String concatenation instead of addition"],
                        fix_suggestions=["Add proper type conversion", "Validate input types"]
                    )
            
            # Look for arithmetic operations that might need string conversion
            if '+' in line and ('input(' in line or '"' in line or "'" in line):
                # Add improper string conversion
                modified_line = re.sub(r'(\w+)\s*\+\s*(\w+)', r'str(\1) + str(\2)', line)
                if modified_line != line:
                    buggy_lines[i] = modified_line
                    buggy_code = '\n'.join(buggy_lines)
                    
                    return BugInjection(
                        bug_type=bug_type,
                        original_code=original_code,
                        buggy_code=buggy_code,
                        bug_location=i + 1,
                        bug_description="String concatenation used instead of arithmetic addition",
                        fix_explanation="Remove str() conversions to perform numeric addition",
                        severity="medium",
                        detection_difficulty="medium",
                        affected_lines=[i + 1],
                        symptoms=["Unexpected string concatenation", "Wrong calculation results"],
                        fix_suggestions=["Ensure numeric types for arithmetic", "Check operation intent"]
                    )
        
        return None
    
    def _extract_variable_names(self, code: str) -> Set[str]:
        """Extract variable names from code."""
        try:
            tree = ast.parse(code)
            variables = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Load)):
                    # Filter out built-in names and common function names
                    if not node.id.startswith('__') and node.id not in ['len', 'range', 'print', 'input', 'str', 'int', 'float']:
                        variables.add(node.id)
            
            return variables
        except SyntaxError:
            # Fallback to regex extraction
            variables = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code))
            return variables - {'def', 'class', 'if', 'else', 'for', 'while', 'return', 'import', 'from'}
    
    def _find_change_location(self, original: str, modified: str) -> int:
        """Find the line number where code was changed."""
        original_lines = original.split('\n')
        modified_lines = modified.split('\n')
        
        for i, (orig_line, mod_line) in enumerate(zip(original_lines, modified_lines)):
            if orig_line != mod_line:
                return i + 1
        
        # If lines were added/removed
        return min(len(original_lines), len(modified_lines))
    
    def _is_valid_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _create_bug_injection(self, symbol: CodeSymbol, bug_type: BugType,
                            original_code: str, buggy_code: str, 
                            bug_location: int) -> BugInjection:
        """Create a BugInjection object with metadata."""
        
        # Determine severity based on bug type
        severity_map = {
            BugType.OFF_BY_ONE: "medium",
            BugType.NULL_POINTER: "high",
            BugType.TYPE_MISMATCH: "medium",
            BugType.LOGIC_ERROR: "high",
            BugType.BOUNDARY_CONDITION: "medium",
            BugType.VARIABLE_NAME: "low",
            BugType.OPERATOR_ERROR: "medium",
            BugType.MISSING_RETURN: "high",
            BugType.INFINITE_LOOP: "critical",
        }
        
        # Determine detection difficulty
        difficulty_map = {
            BugType.VARIABLE_NAME: "easy",
            BugType.TYPE_MISMATCH: "medium",
            BugType.OFF_BY_ONE: "medium",
            BugType.LOGIC_ERROR: "hard",
            BugType.NULL_POINTER: "medium",
            BugType.OPERATOR_ERROR: "easy",
        }
        
        return BugInjection(
            bug_type=bug_type,
            original_code=original_code,
            buggy_code=buggy_code,
            bug_location=bug_location,
            bug_description=f"{bug_type.value} error injected",
            fix_explanation=f"Fix the {bug_type.value} issue",
            severity=severity_map.get(bug_type, "medium"),
            detection_difficulty=difficulty_map.get(bug_type, "medium"),
            affected_lines=[bug_location],
            symptoms=[f"Issues related to {bug_type.value}"],
            fix_suggestions=[f"Review {bug_type.value} patterns"]
        )
    
    def _update_stats(self, injection: BugInjection):
        """Update injection statistics."""
        bug_type_str = injection.bug_type.value
        self.stats.bugs_by_type[bug_type_str] = (
            self.stats.bugs_by_type.get(bug_type_str, 0) + 1
        )
        
        self.stats.bugs_by_severity[injection.severity] = (
            self.stats.bugs_by_severity.get(injection.severity, 0) + 1
        )
    
    def inject_multiple_bugs(self, symbol: CodeSymbol, 
                           count: int = 1) -> List[BugInjection]:
        """Inject multiple different bugs into a symbol."""
        injections = []
        current_symbol = symbol
        
        for _ in range(count):
            # Try different bug types to avoid conflicts
            available_types = [bt for bt in self.enabled_bug_types 
                             if bt not in [inj.bug_type for inj in injections]]
            
            if not available_types:
                break
            
            injection = self.inject_bug(current_symbol, random.choice(available_types))
            if injection:
                injections.append(injection)
                # Create new symbol with buggy code for next iteration
                current_symbol = copy.deepcopy(symbol)
                current_symbol.source_code = injection.buggy_code
        
        return injections
    
    def get_stats(self) -> InjectionStats:
        """Get injection statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset injection statistics."""
        self.stats = InjectionStats()


def create_bug_injector(difficulty_level: str = "mixed") -> BugInjector:
    """Create a pre-configured bug injector."""
    if difficulty_level == "easy":
        enabled_types = [BugType.VARIABLE_NAME, BugType.OPERATOR_ERROR, BugType.TYPE_MISMATCH]
        difficulty_dist = {"easy": 0.8, "medium": 0.2, "hard": 0.0}
    elif difficulty_level == "medium":
        enabled_types = [BugType.OFF_BY_ONE, BugType.LOGIC_ERROR, BugType.TYPE_MISMATCH, BugType.NULL_POINTER]
        difficulty_dist = {"easy": 0.2, "medium": 0.6, "hard": 0.2}
    elif difficulty_level == "hard":
        enabled_types = [BugType.LOGIC_ERROR, BugType.BOUNDARY_CONDITION, BugType.NULL_POINTER]
        difficulty_dist = {"easy": 0.1, "medium": 0.3, "hard": 0.6}
    else:  # mixed
        enabled_types = list(BugType)
        difficulty_dist = {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    
    return BugInjector(
        enabled_bug_types=enabled_types,
        preserve_syntax=True,
        difficulty_distribution=difficulty_dist
    ) 