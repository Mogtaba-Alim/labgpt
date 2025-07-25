"""
Selective Questioner for Section-Type Specific Questions

This module provides intelligent question selection and generation based on
section types, content characteristics, and complexity levels.
"""

import random
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..symbols import CodeSymbol, SymbolType
from ..assembly.config_manager import ConfigManager
from .paper_qa_generator import PaperChunk, SectionType


class QuestionCategory(Enum):
    """Categories of questions that can be generated."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    COMPARATIVE = "comparative"
    EVALUATIVE = "evaluative"
    SYNTHESIS = "synthesis"
    APPLICATION = "application"


@dataclass
class QuestionTemplate:
    """Template for generating specific types of questions."""
    template: str
    category: QuestionCategory
    difficulty: str  # "easy", "medium", "hard"
    required_elements: List[str] = field(default_factory=list)  # What content must be present
    focus_areas: List[str] = field(default_factory=list)
    answer_pattern: str = ""  # Expected answer structure
    weight: float = 1.0  # Relative importance/frequency


@dataclass
class SelectiveQuestionConfig:
    """Configuration for selective questioning by content type."""
    section_templates: Dict[str, List[QuestionTemplate]] = field(default_factory=dict)
    symbol_templates: Dict[str, List[QuestionTemplate]] = field(default_factory=dict)
    complexity_modifiers: Dict[str, Dict[str, float]] = field(default_factory=dict)
    question_distribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class SelectionStats:
    """Statistics from selective question generation."""
    questions_generated: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    by_section_type: Dict[str, int] = field(default_factory=dict)
    by_difficulty: Dict[str, int] = field(default_factory=dict)
    average_relevance_score: float = 0.0


class SelectiveQuestioner:
    """Generates targeted questions based on content type and characteristics."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the selective questioner.
        
        Args:
            config_manager: Configuration manager for question specifications
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.stats = SelectionStats()
        
        # Initialize question templates
        self.config = self._init_selective_config()
        
        # Content analysis patterns
        self.content_indicators = self._init_content_indicators()
    
    def _init_selective_config(self) -> SelectiveQuestionConfig:
        """Initialize selective questioning configuration."""
        config = SelectiveQuestionConfig()
        
        # Section-specific templates for research papers
        config.section_templates = {
            "abstract": [
                QuestionTemplate(
                    template="What is the main objective of this research?",
                    category=QuestionCategory.FACTUAL,
                    difficulty="easy",
                    required_elements=["objective", "goal", "aim"],
                    focus_areas=["research_goal"],
                    answer_pattern="The research aims to..."
                ),
                QuestionTemplate(
                    template="What methodology is briefly described in this abstract?",
                    category=QuestionCategory.PROCEDURAL,
                    difficulty="medium",
                    required_elements=["method", "approach", "technique"],
                    focus_areas=["methodology_overview"],
                    answer_pattern="The methodology involves..."
                ),
                QuestionTemplate(
                    template="What are the key contributions claimed in this abstract?",
                    category=QuestionCategory.ANALYTICAL,
                    difficulty="medium",
                    required_elements=["contribution", "novel", "advance"],
                    focus_areas=["contributions"],
                    answer_pattern="The key contributions are..."
                )
            ],
            
            "introduction": [
                QuestionTemplate(
                    template="What problem does this research address?",
                    category=QuestionCategory.FACTUAL,
                    difficulty="easy",
                    required_elements=["problem", "issue", "challenge"],
                    focus_areas=["problem_statement"],
                    answer_pattern="The research addresses..."
                ),
                QuestionTemplate(
                    template="What gaps in existing research are identified?",
                    category=QuestionCategory.ANALYTICAL,
                    difficulty="medium",
                    required_elements=["gap", "limitation", "lack"],
                    focus_areas=["research_gaps"],
                    answer_pattern="The identified gaps include..."
                ),
                QuestionTemplate(
                    template="How does this work motivate the proposed approach?",
                    category=QuestionCategory.CONCEPTUAL,
                    difficulty="hard",
                    required_elements=["motivation", "justify", "rationale"],
                    focus_areas=["motivation"],
                    answer_pattern="The motivation stems from..."
                )
            ],
            
            "methodology": [
                QuestionTemplate(
                    template="What is the proposed approach or method?",
                    category=QuestionCategory.PROCEDURAL,
                    difficulty="easy",
                    required_elements=["approach", "method", "algorithm"],
                    focus_areas=["core_method"],
                    answer_pattern="The proposed method is..."
                ),
                QuestionTemplate(
                    template="How does this method differ from existing approaches?",
                    category=QuestionCategory.COMPARATIVE,
                    difficulty="medium",
                    required_elements=["difference", "novel", "unlike"],
                    focus_areas=["methodological_novelty"],
                    answer_pattern="The method differs by..."
                ),
                QuestionTemplate(
                    template="What are the theoretical foundations of this approach?",
                    category=QuestionCategory.CONCEPTUAL,
                    difficulty="hard",
                    required_elements=["theory", "foundation", "principle"],
                    focus_areas=["theoretical_basis"],
                    answer_pattern="The theoretical foundation is..."
                )
            ],
            
            "results": [
                QuestionTemplate(
                    template="What are the main experimental findings?",
                    category=QuestionCategory.FACTUAL,
                    difficulty="easy",
                    required_elements=["result", "finding", "outcome"],
                    focus_areas=["key_results"],
                    answer_pattern="The main findings are..."
                ),
                QuestionTemplate(
                    template="How do the results compare to baseline methods?",
                    category=QuestionCategory.COMPARATIVE,
                    difficulty="medium",
                    required_elements=["compare", "baseline", "benchmark"],
                    focus_areas=["comparative_performance"],
                    answer_pattern="Compared to baselines..."
                ),
                QuestionTemplate(
                    template="What statistical significance do the results show?",
                    category=QuestionCategory.ANALYTICAL,
                    difficulty="medium",
                    required_elements=["significant", "statistical", "p-value"],
                    focus_areas=["statistical_analysis"],
                    answer_pattern="The statistical analysis shows..."
                )
            ],
            
            "conclusion": [
                QuestionTemplate(
                    template="What are the main conclusions of this research?",
                    category=QuestionCategory.SYNTHESIS,
                    difficulty="easy",
                    required_elements=["conclude", "summary", "main"],
                    focus_areas=["conclusions"],
                    answer_pattern="The main conclusions are..."
                ),
                QuestionTemplate(
                    template="What limitations are acknowledged?",
                    category=QuestionCategory.EVALUATIVE,
                    difficulty="medium",
                    required_elements=["limitation", "constraint", "restrict"],
                    focus_areas=["limitations"],
                    answer_pattern="The acknowledged limitations include..."
                ),
                QuestionTemplate(
                    template="What future work is suggested?",
                    category=QuestionCategory.APPLICATION,
                    difficulty="medium",
                    required_elements=["future", "next", "extend"],
                    focus_areas=["future_work"],
                    answer_pattern="Future work includes..."
                )
            ]
        }
        
        # Symbol-specific templates for code
        config.symbol_templates = {
            "function": [
                QuestionTemplate(
                    template="What does this function do?",
                    category=QuestionCategory.FACTUAL,
                    difficulty="easy",
                    required_elements=["function", "def"],
                    focus_areas=["functionality"],
                    answer_pattern="This function..."
                ),
                QuestionTemplate(
                    template="What are the parameters and their purposes?",
                    category=QuestionCategory.PROCEDURAL,
                    difficulty="easy",
                    required_elements=["parameter", "argument", "input"],
                    focus_areas=["parameters"],
                    answer_pattern="The parameters are..."
                ),
                QuestionTemplate(
                    template="What algorithm or logic does this function implement?",
                    category=QuestionCategory.ANALYTICAL,
                    difficulty="medium",
                    required_elements=["algorithm", "logic", "implementation"],
                    focus_areas=["algorithm"],
                    answer_pattern="The function implements..."
                ),
                QuestionTemplate(
                    template="What edge cases does this function handle?",
                    category=QuestionCategory.EVALUATIVE,
                    difficulty="hard",
                    required_elements=["edge", "exception", "error"],
                    focus_areas=["edge_cases"],
                    answer_pattern="The function handles..."
                )
            ],
            
            "class": [
                QuestionTemplate(
                    template="What is the purpose of this class?",
                    category=QuestionCategory.CONCEPTUAL,
                    difficulty="easy",
                    required_elements=["class", "purpose", "represents"],
                    focus_areas=["class_purpose"],
                    answer_pattern="This class..."
                ),
                QuestionTemplate(
                    template="What design patterns does this class implement?",
                    category=QuestionCategory.ANALYTICAL,
                    difficulty="hard",
                    required_elements=["pattern", "design", "architecture"],
                    focus_areas=["design_patterns"],
                    answer_pattern="The class implements..."
                ),
                QuestionTemplate(
                    template="How do the methods in this class work together?",
                    category=QuestionCategory.SYNTHESIS,
                    difficulty="medium",
                    required_elements=["method", "interaction", "together"],
                    focus_areas=["method_interaction"],
                    answer_pattern="The methods work together by..."
                )
            ]
        }
        
        # Complexity modifiers
        config.complexity_modifiers = {
            "simple": {
                "factual": 0.5,
                "procedural": 0.3,
                "analytical": 0.1,
                "conceptual": 0.1
            },
            "moderate": {
                "factual": 0.3,
                "procedural": 0.3,
                "analytical": 0.3,
                "conceptual": 0.1
            },
            "complex": {
                "factual": 0.2,
                "procedural": 0.2,
                "analytical": 0.4,
                "conceptual": 0.2
            },
            "very_complex": {
                "factual": 0.1,
                "procedural": 0.2,
                "analytical": 0.4,
                "conceptual": 0.3
            }
        }
        
        return config
    
    def _init_content_indicators(self) -> Dict[str, List[str]]:
        """Initialize content indicators for intelligent selection."""
        return {
            "has_code": ["def ", "class ", "import ", "return ", "if ", "for "],
            "has_math": ["equation", "formula", "\\(", "\\[", "$"],
            "has_data": ["dataset", "data", "table", "figure", "chart"],
            "has_algorithm": ["algorithm", "method", "approach", "procedure"],
            "has_experiment": ["experiment", "test", "evaluation", "benchmark"],
            "has_comparison": ["compare", "versus", "than", "better", "worse"],
            "has_statistics": ["significant", "p-value", "correlation", "regression"],
            "has_implementation": ["implement", "code", "software", "system"]
        }
    
    def select_questions_for_paper_chunk(self, chunk: PaperChunk, 
                                       count: int = 3) -> List[QuestionTemplate]:
        """
        Select appropriate questions for a paper chunk based on section type.
        
        Args:
            chunk: Paper chunk to generate questions for
            count: Number of questions to select
            
        Returns:
            List of selected question templates
        """
        section_type = chunk.section_type.value
        available_templates = self.config.section_templates.get(section_type, [])
        
        if not available_templates:
            # Fallback to general templates
            available_templates = self._get_general_templates()
        
        # Filter templates based on content analysis
        relevant_templates = self._filter_templates_by_content(available_templates, chunk.content)
        
        # Select templates with weighted random selection
        selected_templates = self._weighted_template_selection(relevant_templates, count)
        
        # Update statistics
        self._update_stats_for_selection(selected_templates, section_type)
        
        return selected_templates
    
    def select_questions_for_code_symbol(self, symbol: CodeSymbol, 
                                       complexity_level: str,
                                       count: int = 3) -> List[QuestionTemplate]:
        """
        Select appropriate questions for a code symbol.
        
        Args:
            symbol: Code symbol to generate questions for
            complexity_level: Complexity level of the symbol
            count: Number of questions to select
            
        Returns:
            List of selected question templates
        """
        symbol_type = symbol.symbol_type.value
        available_templates = self.config.symbol_templates.get(symbol_type, [])
        
        if not available_templates:
            # Use function templates as fallback
            available_templates = self.config.symbol_templates.get("function", [])
        
        # Apply complexity modifiers
        complexity_weights = self.config.complexity_modifiers.get(complexity_level, {})
        filtered_templates = self._apply_complexity_filter(available_templates, complexity_weights)
        
        # Filter by content analysis
        relevant_templates = self._filter_templates_by_content(filtered_templates, symbol.source_code)
        
        # Select with weighted selection
        selected_templates = self._weighted_template_selection(relevant_templates, count)
        
        # Update statistics
        self._update_stats_for_selection(selected_templates, symbol_type)
        
        return selected_templates
    
    def _filter_templates_by_content(self, templates: List[QuestionTemplate], 
                                   content: str) -> List[QuestionTemplate]:
        """Filter templates based on content analysis."""
        relevant_templates = []
        content_lower = content.lower()
        
        for template in templates:
            relevance_score = 0.0
            
            # Check required elements
            required_count = 0
            for element in template.required_elements:
                if element.lower() in content_lower:
                    required_count += 1
            
            if template.required_elements:
                relevance_score += (required_count / len(template.required_elements)) * 0.6
            else:
                relevance_score += 0.3  # No requirements means generally applicable
            
            # Check content indicators
            for indicator_type, indicators in self.content_indicators.items():
                if any(indicator in content_lower for indicator in indicators):
                    if indicator_type in template.focus_areas:
                        relevance_score += 0.2
            
            # Apply base weight
            relevance_score *= template.weight
            
            if relevance_score > 0.2:  # Minimum relevance threshold
                template_copy = QuestionTemplate(
                    template=template.template,
                    category=template.category,
                    difficulty=template.difficulty,
                    required_elements=template.required_elements,
                    focus_areas=template.focus_areas,
                    answer_pattern=template.answer_pattern,
                    weight=relevance_score  # Store calculated relevance
                )
                relevant_templates.append(template_copy)
        
        return relevant_templates
    
    def _apply_complexity_filter(self, templates: List[QuestionTemplate],
                               complexity_weights: Dict[str, float]) -> List[QuestionTemplate]:
        """Apply complexity-based filtering to templates."""
        if not complexity_weights:
            return templates
        
        filtered_templates = []
        for template in templates:
            category_weight = complexity_weights.get(template.category.value, 0.1)
            
            # Adjust template weight based on complexity
            adjusted_weight = template.weight * category_weight
            
            if adjusted_weight > 0.05:  # Minimum threshold
                template_copy = QuestionTemplate(
                    template=template.template,
                    category=template.category,
                    difficulty=template.difficulty,
                    required_elements=template.required_elements,
                    focus_areas=template.focus_areas,
                    answer_pattern=template.answer_pattern,
                    weight=adjusted_weight
                )
                filtered_templates.append(template_copy)
        
        return filtered_templates
    
    def _weighted_template_selection(self, templates: List[QuestionTemplate],
                                   count: int) -> List[QuestionTemplate]:
        """Select templates using weighted random selection."""
        if len(templates) <= count:
            return templates
        
        # Calculate selection probabilities
        total_weight = sum(template.weight for template in templates)
        if total_weight == 0:
            return random.sample(templates, count)
        
        probabilities = [template.weight / total_weight for template in templates]
        
        # Weighted selection without replacement
        selected = []
        remaining_templates = templates.copy()
        remaining_probs = probabilities.copy()
        
        for _ in range(count):
            if not remaining_templates:
                break
            
            # Normalize probabilities
            prob_sum = sum(remaining_probs)
            if prob_sum > 0:
                normalized_probs = [p / prob_sum for p in remaining_probs]
                
                # Select based on probabilities
                selected_idx = random.choices(range(len(remaining_templates)), 
                                            weights=normalized_probs)[0]
            else:
                selected_idx = random.randint(0, len(remaining_templates) - 1)
            
            selected.append(remaining_templates[selected_idx])
            
            # Remove selected template
            remaining_templates.pop(selected_idx)
            remaining_probs.pop(selected_idx)
        
        return selected
    
    def _get_general_templates(self) -> List[QuestionTemplate]:
        """Get general templates for unknown section types."""
        return [
            QuestionTemplate(
                template="What are the main points discussed in this section?",
                category=QuestionCategory.FACTUAL,
                difficulty="easy",
                required_elements=[],
                focus_areas=["general"],
                answer_pattern="The main points are..."
            ),
            QuestionTemplate(
                template="How does this section contribute to the overall work?",
                category=QuestionCategory.ANALYTICAL,
                difficulty="medium",
                required_elements=[],
                focus_areas=["general"],
                answer_pattern="This section contributes by..."
            ),
            QuestionTemplate(
                template="What concepts or terminology are introduced here?",
                category=QuestionCategory.CONCEPTUAL,
                difficulty="medium",
                required_elements=[],
                focus_areas=["general"],
                answer_pattern="The concepts introduced include..."
            )
        ]
    
    def generate_adaptive_question_set(self, content: str, content_type: str,
                                     complexity_level: str = "medium",
                                     focus_areas: Optional[List[str]] = None) -> List[QuestionTemplate]:
        """
        Generate an adaptive question set based on content characteristics.
        
        Args:
            content: Content to analyze
            content_type: Type of content ("code", "paper", "documentation")
            complexity_level: Complexity level for question selection
            focus_areas: Specific areas to focus on
            
        Returns:
            Adaptive question set
        """
        # Analyze content characteristics
        content_analysis = self._analyze_content_characteristics(content)
        
        # Select base templates
        if content_type == "code":
            base_templates = self.config.symbol_templates.get("function", [])
        elif content_type == "paper":
            base_templates = self._get_section_templates_by_content(content)
        else:
            base_templates = self._get_general_templates()
        
        # Filter by content analysis
        relevant_templates = self._filter_templates_by_content(base_templates, content)
        
        # Apply focus area filtering if specified
        if focus_areas:
            relevant_templates = [
                t for t in relevant_templates 
                if any(area in t.focus_areas for area in focus_areas)
            ]
        
        # Apply complexity filtering
        complexity_weights = self.config.complexity_modifiers.get(complexity_level, {})
        filtered_templates = self._apply_complexity_filter(relevant_templates, complexity_weights)
        
        return filtered_templates
    
    def _analyze_content_characteristics(self, content: str) -> Dict[str, float]:
        """Analyze content characteristics for adaptive selection."""
        analysis = {}
        content_lower = content.lower()
        
        for char_type, indicators in self.content_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            analysis[char_type] = min(score / len(indicators), 1.0)  # Normalize to 0-1
        
        return analysis
    
    def _get_section_templates_by_content(self, content: str) -> List[QuestionTemplate]:
        """Get templates based on detected section characteristics."""
        content_lower = content.lower()
        
        # Simple section detection
        if any(word in content_lower for word in ["abstract", "summary"]):
            return self.config.section_templates.get("abstract", [])
        elif any(word in content_lower for word in ["introduction", "background"]):
            return self.config.section_templates.get("introduction", [])
        elif any(word in content_lower for word in ["method", "approach", "algorithm"]):
            return self.config.section_templates.get("methodology", [])
        elif any(word in content_lower for word in ["result", "finding", "experiment"]):
            return self.config.section_templates.get("results", [])
        elif any(word in content_lower for word in ["conclusion", "summary", "future"]):
            return self.config.section_templates.get("conclusion", [])
        else:
            return self._get_general_templates()
    
    def _update_stats_for_selection(self, templates: List[QuestionTemplate], 
                                  content_type: str):
        """Update statistics for question selection."""
        self.stats.questions_generated += len(templates)
        
        for template in templates:
            # Update category distribution
            category = template.category.value
            self.stats.by_category[category] = self.stats.by_category.get(category, 0) + 1
            
            # Update section type distribution
            self.stats.by_section_type[content_type] = (
                self.stats.by_section_type.get(content_type, 0) + 1
            )
            
            # Update difficulty distribution
            self.stats.by_difficulty[template.difficulty] = (
                self.stats.by_difficulty.get(template.difficulty, 0) + 1
            )
        
        # Update average relevance score
        if templates:
            total_relevance = sum(template.weight for template in templates)
            avg_relevance = total_relevance / len(templates)
            
            # Running average
            n = self.stats.questions_generated
            self.stats.average_relevance_score = (
                (self.stats.average_relevance_score * (n - len(templates)) + 
                 avg_relevance * len(templates)) / n
            )
    
    def get_stats(self) -> SelectionStats:
        """Get selection statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset selection statistics."""
        self.stats = SelectionStats()


def create_selective_questioner(config_manager: ConfigManager) -> SelectiveQuestioner:
    """Create a pre-configured selective questioner."""
    return SelectiveQuestioner(config_manager) 