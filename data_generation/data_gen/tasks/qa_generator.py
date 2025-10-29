"""
Grounded Q&A Generator

This module provides sophisticated Q&A generation with explicit grounding requirements,
context citations, and NOT_IN_CONTEXT handling for training data creation.
"""

import re
import random
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

from ..symbols import UniversalCodeSymbol
from ..assembly.config_manager import ConfigManager, TaskConfig


@dataclass
class GroundedQA:
    """Represents a grounded Q&A pair with context and citations."""
    question: str
    answer: str
    context_symbol_text: str
    context_symbol_name: str
    context_symbol_type: str
    context_start_line: int
    context_end_line: int
    citations: List[str] = field(default_factory=list)  # Line ranges or specific citations
    confidence_score: float = 0.0
    is_negative_example: bool = False
    task_focus_area: str = ""
    complexity_level: str = ""
    requires_external_knowledge: bool = False


@dataclass
class QAGenerationStats:
    """Statistics from Q&A generation process."""
    total_questions_generated: int = 0
    grounded_questions: int = 0
    negative_examples: int = 0
    questions_by_complexity: Dict[str, int] = field(default_factory=dict)
    questions_by_focus_area: Dict[str, int] = field(default_factory=dict)
    average_confidence: float = 0.0
    citation_coverage: float = 0.0


class GroundedQAGenerator:
    """Generates grounded Q&A pairs with explicit context citations."""
    
    def __init__(self, config_manager: ConfigManager, llm_client):
        """
        Initialize the grounded Q&A generator.
        
        Args:
            config_manager: Configuration manager for task specifications
            llm_client: LLM client for generating Q&A pairs
        """
        self.config_manager = config_manager
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        self.stats = QAGenerationStats()
    
    def generate_qa_pairs(self, symbol: UniversalCodeSymbol, complexity_level: str) -> List[GroundedQA]:
        """
        Generate grounded Q&A pairs for a code symbol.

        Args:
            symbol: Code symbol to generate Q&A pairs for
            complexity_level: Complexity level of the symbol

        Returns:
            List of grounded Q&A pairs
        """
        qa_pairs = []
        
        # Get task configuration
        task_config = self.config_manager.get_task_config('qa_pairs', complexity_level)
        
        # Generate positive examples
        positive_pairs = self._generate_positive_qa_pairs(symbol, complexity_level, task_config)
        qa_pairs.extend(positive_pairs)
        
        # Generate negative examples if enabled
        if self.config_manager.should_generate_negative_examples():
            negative_pairs = self._generate_negative_qa_pairs(symbol, complexity_level)
            qa_pairs.extend(negative_pairs)
        
        # Update statistics
        self._update_stats(qa_pairs, complexity_level)
        
        return qa_pairs
    
    def _generate_positive_qa_pairs(self, symbol: UniversalCodeSymbol, complexity_level: str,
                                  task_config: TaskConfig) -> List[GroundedQA]:
        """Generate positive (grounded) Q&A pairs."""
        qa_pairs = []
        
        # Use templates if available
        if task_config.templates:
            template_pairs = self._generate_from_templates(symbol, complexity_level, task_config)
            qa_pairs.extend(template_pairs)
        
        # Generate focus-area specific questions
        if task_config.focus_areas:
            focus_pairs = self._generate_focus_area_questions(symbol, complexity_level, task_config)
            qa_pairs.extend(focus_pairs)
        
        # Generate additional questions to meet count requirement
        remaining_count = max(0, task_config.count - len(qa_pairs))
        if remaining_count > 0:
            additional_pairs = self._generate_general_questions(symbol, complexity_level, remaining_count)
            qa_pairs.extend(additional_pairs)
        
        return qa_pairs[:task_config.count]  # Ensure we don't exceed the count
    
    def _generate_from_templates(self, symbol: UniversalCodeSymbol, complexity_level: str,
                               task_config: TaskConfig) -> List[GroundedQA]:
        """Generate Q&A pairs from predefined templates."""
        qa_pairs = []

        for template in task_config.templates:
            # Fill template with symbol information
            question = template.format(
                symbol_name=symbol.name,
                symbol_type=symbol.symbol_type,
                parent_class=symbol.parent_scope or "",
                num_parameters=symbol.complexity.num_parameters
            )
            
            # Generate answer using LLM
            answer = self._generate_grounded_answer(question, symbol)
            
            if answer and answer != "NOT_IN_CONTEXT":
                citations = self._extract_citations(answer, symbol)
                qa_pairs.append(GroundedQA(
                    question=question,
                    answer=answer,
                    context_symbol_text=symbol.source_code,
                    context_symbol_name=symbol.name,
                    context_symbol_type=symbol.symbol_type,
                    context_start_line=symbol.start_line,
                    context_end_line=symbol.end_line,
                    citations=citations,
                    task_focus_area="template",
                    complexity_level=complexity_level,
                    is_negative_example=False
                ))
        
        return qa_pairs
    
    def _generate_focus_area_questions(self, symbol: UniversalCodeSymbol, complexity_level: str,
                                     task_config: TaskConfig) -> List[GroundedQA]:
        """Generate questions based on focus areas."""
        qa_pairs = []
        
        for focus_area in task_config.focus_areas:
            question = self._create_focus_area_question(symbol, focus_area, complexity_level)
            if question:
                answer = self._generate_grounded_answer(question, symbol)
                
                if answer and answer != "NOT_IN_CONTEXT":
                    citations = self._extract_citations(answer, symbol)
                    qa_pairs.append(GroundedQA(
                        question=question,
                        answer=answer,
                        context_symbol_text=symbol.source_code,
                        context_symbol_name=symbol.name,
                        context_symbol_type=symbol.symbol_type,
                        context_start_line=symbol.start_line,
                        context_end_line=symbol.end_line,
                        citations=citations,
                        task_focus_area=focus_area,
                        complexity_level=complexity_level,
                        is_negative_example=False
                    ))
        
        return qa_pairs
    
    def _create_focus_area_question(self, symbol: UniversalCodeSymbol, focus_area: str,
                                  complexity_level: str) -> Optional[str]:
        """Create a question based on a specific focus area."""
        focus_area_templates = {
            "basic_functionality": [
                f"What does the {symbol.symbol_type} `{symbol.name}` do?",
                f"What is the purpose of `{symbol.name}`?",
                f"How does `{symbol.name}` work?"
            ],
            "parameter_description": [
                f"What parameters does `{symbol.name}` accept?",
                f"Describe the input parameters for `{symbol.name}`.",
                f"What are the arguments to `{symbol.name}` and what do they represent?"
            ],
            "edge_cases": [
                f"What edge cases does `{symbol.name}` handle?",
                f"How does `{symbol.name}` handle invalid inputs?",
                f"What happens when `{symbol.name}` receives unexpected parameters?"
            ],
            "algorithm_explanation": [
                f"What algorithm does `{symbol.name}` implement?",
                f"Explain the logic flow in `{symbol.name}`.",
                f"What computational approach does `{symbol.name}` use?"
            ],
            "performance_considerations": [
                f"What are the performance characteristics of `{symbol.name}`?",
                f"How efficient is `{symbol.name}`?",
                f"What is the time complexity of `{symbol.name}`?"
            ],
            "architecture_explanation": [
                f"How does `{symbol.name}` fit into the overall architecture?",
                f"What role does `{symbol.name}` play in the system?",
                f"How does `{symbol.name}` interact with other components?"
            ],
            "design_patterns": [
                f"What design patterns are used in `{symbol.name}`?",
                f"How does `{symbol.name}` implement the pattern?",
                f"What architectural patterns does `{symbol.name}` follow?"
            ]
        }
        
        if focus_area in focus_area_templates:
            return random.choice(focus_area_templates[focus_area])
        
        # Generic fallback
        return f"Explain the {focus_area} of `{symbol.name}`."
    
    def _generate_general_questions(self, symbol: UniversalCodeSymbol, complexity_level: str,
                                  count: int) -> List[GroundedQA]:
        """Generate general questions for the symbol."""
        qa_pairs = []

        general_templates = [
            f"What does `{symbol.name}` return?",
            f"What are the side effects of `{symbol.name}`?",
            f"What exceptions can `{symbol.name}` raise?",
            f"How should `{symbol.name}` be used?",
            f"What are the dependencies of `{symbol.name}`?",
            f"What is the complexity of `{symbol.name}`?",
            f"How is `{symbol.name}` tested?",
            f"What are the preconditions for `{symbol.name}`?",
            f"What are the postconditions of `{symbol.name}`?",
            f"How does `{symbol.name}` handle errors?"
        ]

        # Add symbol-specific questions
        if symbol.symbol_type == "class":
            general_templates.extend([
                f"What attributes does class `{symbol.name}` have?",
                f"What methods are available in class `{symbol.name}`?",
                f"How is class `{symbol.name}` initialized?",
                f"What is the inheritance hierarchy of `{symbol.name}`?"
            ])
        elif symbol.symbol_type in ("function", "method"):
            general_templates.extend([
                f"What is the signature of `{symbol.name}`?",
                f"What validation does `{symbol.name}` perform?",
                f"What is the return type of `{symbol.name}`?"
            ])
        
        # Randomly select questions
        selected_questions = random.sample(
            general_templates, 
            min(count, len(general_templates))
        )
        
        for question in selected_questions:
            answer = self._generate_grounded_answer(question, symbol)
            
            if answer and answer != "NOT_IN_CONTEXT":
                citations = self._extract_citations(answer, symbol)
                qa_pairs.append(GroundedQA(
                    question=question,
                    answer=answer,
                    context_symbol_text=symbol.source_code,
                    context_symbol_name=symbol.name,
                    context_symbol_type=symbol.symbol_type,
                    context_start_line=symbol.start_line,
                    context_end_line=symbol.end_line,
                    citations=citations,
                    task_focus_area="general",
                    complexity_level=complexity_level,
                    is_negative_example=False
                ))
        
        return qa_pairs
    
    def _generate_negative_qa_pairs(self, symbol: UniversalCodeSymbol, complexity_level: str) -> List[GroundedQA]:
        """Generate negative examples that should result in NOT_IN_CONTEXT."""
        qa_pairs = []
        
        negative_config = self.config_manager.get_negative_example_config()
        if not negative_config.enabled:
            return qa_pairs
        
        # Calculate number of negative examples
        total_positive = self.config_manager.get_task_config('qa_pairs', complexity_level).count
        num_negative = max(1, int(total_positive * negative_config.percentage_of_total))
        
        negative_questions = []
        
        # Generate impossible questions (ask about non-existent parameters)
        if "impossible_question" in negative_config.types:
            impossible_questions = self._generate_impossible_questions(symbol)
            negative_questions.extend(impossible_questions)
        
        # Generate out-of-scope questions
        if "out_of_scope" in negative_config.types:
            out_of_scope_questions = self._generate_out_of_scope_questions(symbol)
            negative_questions.extend(out_of_scope_questions)
        
        # Generate insufficient context questions
        if "insufficient_context" in negative_config.types:
            insufficient_context_questions = self._generate_insufficient_context_questions(symbol)
            negative_questions.extend(insufficient_context_questions)
        
        # Select subset of negative questions
        selected_negative = random.sample(
            negative_questions, 
            min(num_negative, len(negative_questions))
        )
        
        for question in selected_negative:
            qa_pairs.append(GroundedQA(
                question=question,
                answer=negative_config.expected_response,
                context_symbol_text=symbol.source_code,
                context_symbol_name=symbol.name,
                context_symbol_type=symbol.symbol_type,
                context_start_line=symbol.start_line,
                context_end_line=symbol.end_line,
                citations=[],
                task_focus_area="negative_example",
                complexity_level=complexity_level,
                is_negative_example=True,
                requires_external_knowledge=True
            ))
        
        return qa_pairs
    
    def _generate_impossible_questions(self, symbol: UniversalCodeSymbol) -> List[str]:
        """Generate questions about non-existent elements."""
        questions = []
        
        # Ask about non-existent parameters
        fake_params = ["config", "settings", "options", "metadata", "context", "handler"]
        existing_params = set(symbol.parameters)
        non_existent_params = [p for p in fake_params if p not in existing_params]
        
        for param in non_existent_params[:2]:
            questions.append(f"What does the `{param}` parameter do in `{symbol.name}`?")
            questions.append(f"How is the `{param}` parameter validated in `{symbol.name}`?")
        
        # Ask about non-existent methods (for classes)
        if symbol.symbol_type == "class":
            fake_methods = ["initialize", "configure", "setup", "teardown", "validate"]
            for method in fake_methods[:2]:
                questions.append(f"How does the `{method}` method work in class `{symbol.name}`?")
        
        # Ask about non-existent return values
        questions.append(f"What does `{symbol.name}` return when the `strict` flag is set?")
        questions.append(f"How does `{symbol.name}` handle the `timeout` parameter?")
        
        return questions
    
    def _generate_out_of_scope_questions(self, symbol: UniversalCodeSymbol) -> List[str]:
        """Generate questions outside the scope of the symbol."""
        questions = [
            f"What database does `{symbol.name}` connect to?",
            f"How does `{symbol.name}` handle authentication?",
            f"What network protocols does `{symbol.name}` support?",
            f"How is `{symbol.name}` deployed in production?",
            f"What monitoring tools work with `{symbol.name}`?",
            f"What are the security implications of using `{symbol.name}`?",
            f"How does `{symbol.name}` scale horizontally?",
            f"What are the licensing terms for `{symbol.name}`?",
            f"What operating systems support `{symbol.name}`?",
            f"How does `{symbol.name}` integrate with Kubernetes?"
        ]
        
        return random.sample(questions, min(3, len(questions)))
    
    def _generate_insufficient_context_questions(self, symbol: UniversalCodeSymbol) -> List[str]:
        """Generate questions requiring external knowledge."""
        questions = [
            f"What was the original motivation for creating `{symbol.name}`?",
            f"Who are the main contributors to `{symbol.name}`?",
            f"What version of the library first introduced `{symbol.name}`?",
            f"What other libraries provide similar functionality to `{symbol.name}`?",
            f"What are the performance benchmarks for `{symbol.name}`?",
            f"What real-world projects use `{symbol.name}`?",
            f"What are the future development plans for `{symbol.name}`?",
            f"How does `{symbol.name}` compare to industry standards?",
            f"What academic papers reference `{symbol.name}`?",
            f"What conferences have featured talks about `{symbol.name}`?"
        ]
        
        return random.sample(questions, min(2, len(questions)))
    
    def _generate_grounded_answer(self, question: str, symbol: UniversalCodeSymbol) -> str:
        """Generate a grounded answer using the LLM with strict instructions."""

        system_prompt = """You are a code analysis expert. Answer questions about the provided code symbol with strict grounding requirements.

CRITICAL INSTRUCTIONS:
1. ONLY use information directly visible in the provided code symbol
2. If the answer requires information not in the code, respond with exactly: "NOT_IN_CONTEXT"
3. Include specific line references or code snippets when possible
4. Be precise and factual - do not speculate or infer beyond what's directly shown
5. For implementation details, refer to specific lines or code sections
6. If asking about something that doesn't exist in the code, respond with "NOT_IN_CONTEXT"

The code symbol provided is the ONLY context available. Do not reference external knowledge."""

        user_prompt = f"""Code Symbol: {symbol.name} (Type: {symbol.symbol_type})
Location: Lines {symbol.start_line}-{symbol.end_line}

Code:
```python
{symbol.source_code}
```

Question: {question}

Answer (following the strict grounding requirements above):"""

        try:
            response = self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1  # Low temperature for consistency
            )
            
            answer = response.content[0].text.strip()
            
            # Post-process to ensure grounding
            if self._requires_external_knowledge(answer, symbol):
                return "NOT_IN_CONTEXT"
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return "NOT_IN_CONTEXT"
    
    def _requires_external_knowledge(self, answer: str, symbol: UniversalCodeSymbol) -> bool:
        """Check if the answer requires knowledge not in the symbol."""
        
        # If answer is already NOT_IN_CONTEXT, it's correctly grounded
        if "NOT_IN_CONTEXT" in answer:
            return False
        
        # Check for external references that shouldn't be in a grounded answer
        external_indicators = [
            "according to the documentation",
            "typically",
            "usually",
            "in most cases",
            "generally",
            "common practice",
            "best practice",
            "industry standard",
            "recommended approach",
            "external library",
            "framework",
            "documented in",
            "see documentation",
            "refer to manual"
        ]
        
        answer_lower = answer.lower()
        for indicator in external_indicators:
            if indicator in answer_lower:
                return True
        
        return False
    
    def _extract_citations(self, answer: str, symbol: UniversalCodeSymbol) -> List[str]:
        """Extract citations/line references from the answer."""
        citations = []
        
        # Look for line number references
        line_patterns = [
            r'line (\d+)',
            r'lines (\d+)-(\d+)',
            r'on line (\d+)',
            r'at line (\d+)'
        ]
        
        for pattern in line_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    citations.append(f"lines {match[0]}-{match[1]}")
                else:
                    citations.append(f"line {match}")
        
        # Look for code snippet references
        code_snippets = re.findall(r'`([^`]+)`', answer)
        for snippet in code_snippets:
            if snippet in symbol.source_code:
                citations.append(f"code: {snippet}")
        
        return citations
    
    def _update_stats(self, qa_pairs: List[GroundedQA], complexity_level: str):
        """Update generation statistics."""
        self.stats.total_questions_generated += len(qa_pairs)

        for qa in qa_pairs:
            if getattr(qa, 'is_negative_example', False):
                self.stats.negative_examples += 1
            else:
                self.stats.grounded_questions += 1
            
            # Update complexity distribution
            self.stats.questions_by_complexity[complexity_level] = (
                self.stats.questions_by_complexity.get(complexity_level, 0) + 1
            )
            
            # Update focus area distribution
            focus_area = qa.task_focus_area
            self.stats.questions_by_focus_area[focus_area] = (
                self.stats.questions_by_focus_area.get(focus_area, 0) + 1
            )
    
    def get_stats(self) -> QAGenerationStats:
        """Get generation statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset generation statistics."""
        self.stats = QAGenerationStats() 