"""
Debug Task Generator

This module generates debugging tasks by injecting realistic bugs into code
and creating training examples for debugging and error detection.
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

from ..symbols import CodeSymbol, SymbolType
from ..assembly.config_manager import ConfigManager, TaskConfig
from .bug_injector import BugInjector, BugInjection, BugType, create_bug_injector


@dataclass
class DebugTask:
    """Represents a debugging task with context and solutions."""
    task_type: str  # "find_bug", "fix_bug", "explain_bug", "identify_symptom"
    question: str
    original_code: str
    buggy_code: str
    context_symbol_name: str
    context_symbol_type: str
    context_start_line: int
    context_end_line: int
    bug_injection: BugInjection
    expected_answer: str
    fix_explanation: str
    hints: List[str] = field(default_factory=list)
    difficulty_level: str = ""
    task_focus_area: str = "debugging"
    requires_execution: bool = False  # Whether task requires running code
    test_cases: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DebugGenerationStats:
    """Statistics from debug task generation."""
    total_tasks_generated: int = 0
    tasks_by_type: Dict[str, int] = field(default_factory=dict)
    tasks_by_bug_type: Dict[str, int] = field(default_factory=dict)
    tasks_by_difficulty: Dict[str, int] = field(default_factory=dict)
    successful_injections: int = 0
    failed_injections: int = 0


class DebugGenerator:
    """Generates debugging tasks using bug injection."""
    
    def __init__(self, config_manager: ConfigManager, llm_client):
        """
        Initialize the debug generator.
        
        Args:
            config_manager: Configuration manager for task specifications
            llm_client: LLM client for generating explanations and hints
        """
        self.config_manager = config_manager
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        self.stats = DebugGenerationStats()
        
        # Initialize bug injector
        self.bug_injector = create_bug_injector(difficulty_level="mixed")
        
        # Task type templates
        self.task_templates = self._init_task_templates()
    
    def _init_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize task templates for different debugging scenarios."""
        return {
            "find_bug": {
                "description": "Identify the location and type of bug in the code",
                "question_template": "Where is the bug in this code? What type of bug is it?",
                "answer_template": "The bug is on line {line} and is a {bug_type} error: {description}",
                "difficulty_weight": 0.3
            },
            
            "fix_bug": {
                "description": "Provide the corrected version of the buggy code",
                "question_template": "Fix the bug in this code. Provide the corrected version.",
                "answer_template": "The bug should be fixed by: {fix_explanation}\n\nCorrected code:\n{original_code}",
                "difficulty_weight": 0.4
            },
            
            "explain_bug": {
                "description": "Explain why the bug occurs and its consequences",
                "question_template": "Explain what's wrong with this code and why it causes problems.",
                "answer_template": "This code has a {bug_type} error: {description}. {fix_explanation}. This can cause: {symptoms}",
                "difficulty_weight": 0.2
            },
            
            "identify_symptom": {
                "description": "Identify what symptoms this bug would cause",
                "question_template": "What error or unexpected behavior would this code produce?",
                "answer_template": "This code would likely produce: {symptoms}. The root cause is: {description}",
                "difficulty_weight": 0.1
            }
        }
    
    def generate_debug_tasks(self, symbol: CodeSymbol, complexity_level: str) -> List[DebugTask]:
        """
        Generate debugging tasks for a code symbol.
        
        Args:
            symbol: Code symbol to generate debug tasks for
            complexity_level: Complexity level of the symbol
            
        Returns:
            List of debugging tasks
        """
        debug_tasks = []
        
        # Get task configuration
        task_config = self.config_manager.get_task_config('debugging_tasks', complexity_level)
        
        # Generate multiple bug injections
        num_bugs = min(task_config.count, 3)  # Limit to avoid too many bugs
        bug_injections = self.bug_injector.inject_multiple_bugs(symbol, num_bugs)
        
        if not bug_injections:
            self.logger.warning(f"Failed to inject bugs for symbol {symbol.name}")
            return debug_tasks
        
        # Create different types of debugging tasks for each bug injection
        for bug_injection in bug_injections:
            # Determine which task types to generate
            task_types = self._select_task_types(bug_injection, complexity_level)
            
            for task_type in task_types:
                debug_task = self._create_debug_task(
                    symbol, bug_injection, task_type, complexity_level
                )
                if debug_task:
                    debug_tasks.append(debug_task)
        
        # Update statistics
        self._update_stats(debug_tasks)
        
        return debug_tasks
    
    def _select_task_types(self, bug_injection: BugInjection, complexity_level: str) -> List[str]:
        """Select appropriate task types based on bug injection and complexity."""
        all_task_types = list(self.task_templates.keys())
        
        # For simple bugs, focus on identification and fixing
        if complexity_level == "simple":
            return random.sample(["find_bug", "fix_bug"], k=min(2, len(all_task_types)))
        
        # For complex bugs, include explanation tasks
        elif complexity_level in ["complex", "very_complex"]:
            return random.sample(all_task_types, k=min(3, len(all_task_types)))
        
        # For moderate complexity, balanced selection
        else:
            return random.sample(all_task_types, k=min(2, len(all_task_types)))
    
    def _create_debug_task(self, symbol: CodeSymbol, bug_injection: BugInjection,
                          task_type: str, complexity_level: str) -> Optional[DebugTask]:
        """Create a specific debugging task."""
        try:
            template = self.task_templates[task_type]
            
            # Generate question
            question = self._generate_question(template, symbol, bug_injection, task_type)
            
            # Generate expected answer
            expected_answer = self._generate_answer(template, symbol, bug_injection, task_type)
            
            # Generate hints
            hints = self._generate_hints(bug_injection, task_type)
            
            # Generate test cases if applicable
            test_cases = self._generate_test_cases(symbol, bug_injection, task_type)
            
            return DebugTask(
                task_type=task_type,
                question=question,
                original_code=bug_injection.original_code,
                buggy_code=bug_injection.buggy_code,
                context_symbol_name=symbol.name,
                context_symbol_type=symbol.symbol_type.value,
                context_start_line=symbol.start_line,
                context_end_line=symbol.end_line,
                bug_injection=bug_injection,
                expected_answer=expected_answer,
                fix_explanation=bug_injection.fix_explanation,
                hints=hints,
                difficulty_level=complexity_level,
                task_focus_area="debugging",
                requires_execution=(task_type == "identify_symptom"),
                test_cases=test_cases
            )
            
        except Exception as e:
            self.logger.error(f"Error creating debug task: {e}")
            return None
    
    def _generate_question(self, template: Dict[str, Any], symbol: CodeSymbol,
                          bug_injection: BugInjection, task_type: str) -> str:
        """Generate a question for the debugging task."""
        base_question = template["question_template"]
        
        # Customize question based on task type and symbol
        if task_type == "find_bug":
            question = f"Analyze this {symbol.symbol_type.value} and identify any bugs:\n\n```python\n{bug_injection.buggy_code}\n```\n\n{base_question}"
        
        elif task_type == "fix_bug":
            question = f"This {symbol.symbol_type.value} contains a bug:\n\n```python\n{bug_injection.buggy_code}\n```\n\n{base_question}"
        
        elif task_type == "explain_bug":
            question = f"Review this {symbol.symbol_type.value}:\n\n```python\n{bug_injection.buggy_code}\n```\n\n{base_question}"
        
        elif task_type == "identify_symptom":
            question = f"If you run this {symbol.symbol_type.value}:\n\n```python\n{bug_injection.buggy_code}\n```\n\n{base_question}"
        
        else:
            question = f"Debug this {symbol.symbol_type.value}:\n\n```python\n{bug_injection.buggy_code}\n```\n\n{base_question}"
        
        return question
    
    def _generate_answer(self, template: Dict[str, Any], symbol: CodeSymbol,
                        bug_injection: BugInjection, task_type: str) -> str:
        """Generate the expected answer for the debugging task."""
        answer_template = template["answer_template"]
        
        # Format the answer template with bug injection data
        answer = answer_template.format(
            line=bug_injection.bug_location,
            bug_type=bug_injection.bug_type.value,
            description=bug_injection.bug_description,
            fix_explanation=bug_injection.fix_explanation,
            symptoms=", ".join(bug_injection.symptoms),
            original_code=bug_injection.original_code
        )
        
        return answer
    
    def _generate_hints(self, bug_injection: BugInjection, task_type: str) -> List[str]:
        """Generate hints for the debugging task."""
        hints = []
        
        # General hints based on bug type
        bug_type_hints = {
            BugType.OFF_BY_ONE: [
                "Check loop boundaries and array indices",
                "Look for range() function calls",
                "Verify that array accesses are within bounds"
            ],
            BugType.LOGIC_ERROR: [
                "Review conditional statements carefully",
                "Check if comparisons are using correct operators",
                "Test with different input values"
            ],
            BugType.TYPE_MISMATCH: [
                "Check data types being used in operations",
                "Look for missing type conversions",
                "Verify input handling and processing"
            ],
            BugType.VARIABLE_NAME: [
                "Check variable name spelling",
                "Look for typos in variable references",
                "Verify variable scope and definition"
            ],
            BugType.NULL_POINTER: [
                "Check for None values",
                "Look for missing initialization",
                "Verify variable assignment before use"
            ]
        }
        
        # Add bug-specific hints
        if bug_injection.bug_type in bug_type_hints:
            hints.extend(bug_type_hints[bug_injection.bug_type][:2])  # Limit to 2 hints
        
        # Add task-specific hints
        if task_type == "find_bug":
            hints.append(f"Focus on line {bug_injection.bug_location} and surrounding code")
        elif task_type == "fix_bug":
            hints.extend(bug_injection.fix_suggestions[:1])  # One fix suggestion
        
        return hints
    
    def _generate_test_cases(self, symbol: CodeSymbol, bug_injection: BugInjection,
                           task_type: str) -> List[Dict[str, Any]]:
        """Generate test cases for the debugging task."""
        if symbol.symbol_type not in [SymbolType.FUNCTION, SymbolType.METHOD]:
            return []
        
        test_cases = []
        
        # Generate simple test cases based on function signature
        if symbol.parameters:
            # Create test case that might trigger the bug
            if bug_injection.bug_type == BugType.OFF_BY_ONE:
                test_cases.append({
                    "input": [10],  # Example input that might cause index error
                    "expected_error": "IndexError",
                    "description": "Test case that triggers the off-by-one error"
                })
            
            elif bug_injection.bug_type == BugType.TYPE_MISMATCH:
                test_cases.append({
                    "input": ["5"],  # String instead of int
                    "expected_error": "TypeError",
                    "description": "Test case with incorrect input type"
                })
        
        return test_cases
    
    def generate_comparative_debug_task(self, symbol: CodeSymbol, 
                                      complexity_level: str) -> Optional[DebugTask]:
        """Generate a task comparing correct vs buggy code."""
        # Inject a bug
        bug_injection = self.bug_injector.inject_bug(symbol)
        if not bug_injection:
            return None
        
        question = f"""Compare these two versions of the same {symbol.symbol_type.value}:

**Version A (Original):**
```python
{bug_injection.original_code}
```

**Version B (Modified):**
```python
{bug_injection.buggy_code}
```

Question: What is the difference between Version A and Version B? Which version is correct and why?"""
        
        answer = f"""Version A is correct. Version B contains a {bug_injection.bug_type.value} error on line {bug_injection.bug_location}.

**Problem:** {bug_injection.bug_description}

**Fix:** {bug_injection.fix_explanation}

**Why Version A is better:** The original code handles the logic correctly, while Version B introduces a bug that could cause: {', '.join(bug_injection.symptoms)}"""
        
        return DebugTask(
            task_type="compare_versions",
            question=question,
            original_code=bug_injection.original_code,
            buggy_code=bug_injection.buggy_code,
            context_symbol_name=symbol.name,
            context_symbol_type=symbol.symbol_type.value,
            context_start_line=symbol.start_line,
            context_end_line=symbol.end_line,
            bug_injection=bug_injection,
            expected_answer=answer,
            fix_explanation=bug_injection.fix_explanation,
            hints=["Compare the code line by line", "Look for subtle differences"],
            difficulty_level=complexity_level,
            task_focus_area="debugging"
        )
    
    def generate_progressive_debug_task(self, symbol: CodeSymbol,
                                      complexity_level: str) -> List[DebugTask]:
        """Generate a progressive debugging task (multiple related bugs)."""
        tasks = []
        
        # Start with the original code
        current_symbol = symbol
        
        # Inject multiple bugs progressively
        for i in range(2):  # Maximum 2 progressive bugs
            bug_injection = self.bug_injector.inject_bug(current_symbol)
            if not bug_injection:
                break
            
            question = f"""This is step {i+1} of debugging the {symbol.symbol_type.value} `{symbol.name}`:

```python
{bug_injection.buggy_code}
```

Find and fix the bug in this version."""
            
            answer = f"""Step {i+1} Bug: {bug_injection.bug_description}
Location: Line {bug_injection.bug_location}
Fix: {bug_injection.fix_explanation}

Corrected code:
```python
{bug_injection.original_code}
```"""
            
            task = DebugTask(
                task_type=f"progressive_debug_step_{i+1}",
                question=question,
                original_code=bug_injection.original_code,
                buggy_code=bug_injection.buggy_code,
                context_symbol_name=symbol.name,
                context_symbol_type=symbol.symbol_type.value,
                context_start_line=symbol.start_line,
                context_end_line=symbol.end_line,
                bug_injection=bug_injection,
                expected_answer=answer,
                fix_explanation=bug_injection.fix_explanation,
                difficulty_level=complexity_level,
                task_focus_area="debugging"
            )
            
            tasks.append(task)
            
            # Use the buggy code as the starting point for the next iteration
            import copy
            current_symbol = copy.deepcopy(symbol)
            current_symbol.source_code = bug_injection.buggy_code
        
        return tasks
    
    def _update_stats(self, debug_tasks: List[DebugTask]):
        """Update generation statistics."""
        self.stats.total_tasks_generated += len(debug_tasks)
        
        for task in debug_tasks:
            # Update task type distribution
            self.stats.tasks_by_type[task.task_type] = (
                self.stats.tasks_by_type.get(task.task_type, 0) + 1
            )
            
            # Update bug type distribution
            bug_type = task.bug_injection.bug_type.value
            self.stats.tasks_by_bug_type[bug_type] = (
                self.stats.tasks_by_bug_type.get(bug_type, 0) + 1
            )
            
            # Update difficulty distribution
            self.stats.tasks_by_difficulty[task.difficulty_level] = (
                self.stats.tasks_by_difficulty.get(task.difficulty_level, 0) + 1
            )
    
    def get_stats(self) -> DebugGenerationStats:
        """Get generation statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset generation statistics."""
        self.stats = DebugGenerationStats()


def create_debug_generator(config_manager: ConfigManager, llm_client, 
                          difficulty_level: str = "mixed") -> DebugGenerator:
    """Create a pre-configured debug generator."""
    generator = DebugGenerator(config_manager, llm_client)
    generator.bug_injector = create_bug_injector(difficulty_level)
    return generator 