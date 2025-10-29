"""
Enhanced Negative Example Generator

This module provides sophisticated generation of negative examples and abstention
training data, creating impossible queries and out-of-context scenarios.
"""

import random
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..symbols import UniversalCodeSymbol
from ..assembly.config_manager import ConfigManager, NegativeExampleConfig


class NegativeExampleType(Enum):
    """Types of negative examples that can be generated."""
    IMPOSSIBLE_PARAMETER = "impossible_parameter"
    NONEXISTENT_METHOD = "nonexistent_method"
    OUT_OF_SCOPE_DOMAIN = "out_of_scope_domain"
    EXTERNAL_DEPENDENCY = "external_dependency"
    IMPLEMENTATION_DETAIL = "implementation_detail"
    HISTORICAL_QUESTION = "historical_question"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    DEPLOYMENT_SPECIFIC = "deployment_specific"
    CROSS_LANGUAGE = "cross_language"
    SPECULATIVE_FUTURE = "speculative_future"


@dataclass
class NegativeExample:
    """Represents a negative example with metadata."""
    question: str
    expected_response: str
    negative_type: NegativeExampleType
    context_symbol_name: str
    context_symbol_type: str
    context_code: str
    difficulty_level: str
    explanation: str  # Why this should result in NOT_IN_CONTEXT
    trap_indicators: List[str] = field(default_factory=list)  # What makes this tempting to answer
    abstention_cues: List[str] = field(default_factory=list)  # Cues that should trigger abstention


@dataclass
class NegativeGenerationStats:
    """Statistics from negative example generation."""
    total_generated: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_difficulty: Dict[str, int] = field(default_factory=dict)
    average_trap_indicators: float = 0.0


class EnhancedNegativeGenerator:
    """Generates sophisticated negative examples for abstention training."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the enhanced negative generator.
        
        Args:
            config_manager: Configuration manager for negative example settings
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.stats = NegativeGenerationStats()
        
        # Get negative example configuration
        self.negative_config = config_manager.get_negative_example_config()
        
        # Initialize question templates
        self.question_templates = self._init_question_templates()
        
        # Common technical domains for out-of-scope questions
        self.external_domains = [
            "database", "networking", "security", "deployment", "monitoring",
            "authentication", "caching", "logging", "testing", "documentation",
            "version control", "CI/CD", "containerization", "cloud platforms"
        ]
    
    def _init_question_templates(self) -> Dict[NegativeExampleType, List[Dict[str, Any]]]:
        """Initialize question templates for different negative example types."""
        return {
            NegativeExampleType.IMPOSSIBLE_PARAMETER: [
                {
                    "template": "What does the `{fake_param}` parameter do in `{symbol_name}`?",
                    "fake_params": ["config", "metadata", "context", "options", "settings", "handler", "callback", "timeout", "retry_count", "max_attempts"],
                    "explanation": "Parameter does not exist in the function signature"
                },
                {
                    "template": "How does `{symbol_name}` handle the `{fake_param}` parameter when it's set to {fake_value}?",
                    "fake_params": ["debug", "verbose", "strict", "async_mode", "cache_enabled"],
                    "fake_values": ["True", "False", "None", "'auto'", "0"],
                    "explanation": "Parameter and behavior do not exist in the code"
                }
            ],
            
            NegativeExampleType.NONEXISTENT_METHOD: [
                {
                    "template": "How does the `{fake_method}` method work in class `{symbol_name}`?",
                    "fake_methods": ["initialize", "configure", "setup", "teardown", "validate", "reset", "cleanup", "refresh", "update", "sync"],
                    "explanation": "Method does not exist in the class definition"
                },
                {
                    "template": "When should you call `{symbol_name}.{fake_method}()` instead of `{symbol_name}.{real_method}()`?",
                    "fake_methods": ["prepare", "finalize", "commit", "rollback", "flush"],
                    "explanation": "Comparison involves a non-existent method"
                }
            ],
            
            NegativeExampleType.OUT_OF_SCOPE_DOMAIN: [
                {
                    "template": "What {domain} does `{symbol_name}` connect to?",
                    "domains": ["database", "API endpoint", "message queue", "cache server", "authentication service"],
                    "explanation": "Question assumes external system integration not present in code"
                },
                {
                    "template": "How does `{symbol_name}` handle {domain_concept}?",
                    "domain_concepts": ["user authentication", "data encryption", "network timeouts", "database transactions", "logging levels"],
                    "explanation": "Question about domain-specific concepts not implemented in the code"
                }
            ],
            
            NegativeExampleType.EXTERNAL_DEPENDENCY: [
                {
                    "template": "What version of {library} does `{symbol_name}` require?",
                    "libraries": ["numpy", "pandas", "requests", "flask", "django", "tensorflow", "pytorch"],
                    "explanation": "Question about external library dependencies not determinable from code alone"
                },
                {
                    "template": "How does `{symbol_name}` integrate with {external_system}?",
                    "external_systems": ["Docker", "Kubernetes", "AWS", "Redis", "PostgreSQL", "MongoDB"],
                    "explanation": "Question about external system integration beyond code scope"
                }
            ],
            
            NegativeExampleType.IMPLEMENTATION_DETAIL: [
                {
                    "template": "What algorithm does `{symbol_name}` use internally for {operation}?",
                    "operations": ["optimization", "hashing", "sorting", "compression", "encryption"],
                    "explanation": "Question about internal implementation details not visible in the provided code"
                },
                {
                    "template": "Why did the developers choose {design_choice} for `{symbol_name}`?",
                    "design_choices": ["this specific algorithm", "this data structure", "this approach", "this pattern"],
                    "explanation": "Question about design rationale requiring historical/external knowledge"
                }
            ],
            
            NegativeExampleType.HISTORICAL_QUESTION: [
                {
                    "template": "When was `{symbol_name}` first introduced?",
                    "explanation": "Question requires historical information not in code"
                },
                {
                    "template": "Who originally wrote `{symbol_name}` and why?",
                    "explanation": "Question about authorship and motivation beyond code scope"
                },
                {
                    "template": "What version of the library first included `{symbol_name}`?",
                    "explanation": "Question about version history not determinable from code"
                }
            ],
            
            NegativeExampleType.PERFORMANCE_BENCHMARK: [
                {
                    "template": "What is the performance benchmark for `{symbol_name}` with {data_size} items?",
                    "data_sizes": ["1000", "10000", "1 million", "large datasets"],
                    "explanation": "Question requires empirical performance data not in code"
                },
                {
                    "template": "How does `{symbol_name}` compare to {alternative} in terms of speed?",
                    "alternatives": ["standard library implementations", "other algorithms", "competitors"],
                    "explanation": "Question requires comparative performance analysis beyond code scope"
                }
            ],
            
            NegativeExampleType.DEPLOYMENT_SPECIFIC: [
                {
                    "template": "How should `{symbol_name}` be configured in production?",
                    "explanation": "Question about deployment configuration not determinable from code"
                },
                {
                    "template": "What are the scalability limits of `{symbol_name}` in a distributed system?",
                    "explanation": "Question about deployment scalability beyond code scope"
                }
            ],
            
            NegativeExampleType.CROSS_LANGUAGE: [
                {
                    "template": "How would you implement `{symbol_name}` in {language}?",
                    "languages": ["Java", "C++", "JavaScript", "Go", "Rust"],
                    "explanation": "Question about implementation in different programming languages"
                },
                {
                    "template": "What is the {language} equivalent of `{symbol_name}`?",
                    "languages": ["Java", "C#", "JavaScript", "Go"],
                    "explanation": "Question requires knowledge of other programming languages"
                }
            ],
            
            NegativeExampleType.SPECULATIVE_FUTURE: [
                {
                    "template": "How will `{symbol_name}` be improved in future versions?",
                    "explanation": "Question about future development plans not determinable from current code"
                },
                {
                    "template": "What new features are planned for `{symbol_name}`?",
                    "explanation": "Question about future roadmap beyond current implementation"
                }
            ]
        }
    
    def generate_negative_examples(self, symbol: UniversalCodeSymbol,
                                 complexity_level: str,
                                 count: Optional[int] = None) -> List[NegativeExample]:
        """
        Generate negative examples for a code symbol.
        
        Args:
            symbol: Code symbol to generate negative examples for
            complexity_level: Complexity level of the symbol
            count: Number of examples to generate (None = use config)
            
        Returns:
            List of negative examples
        """
        if not self.negative_config.enabled:
            return []
        
        # Determine count
        if count is None:
            # Calculate based on configuration percentage
            base_count = 3  # Assume 3 positive examples
            count = max(1, int(base_count * self.negative_config.percentage_of_total))
        
        negative_examples = []
        
        # Select negative example types to generate
        selected_types = self._select_negative_types(symbol, complexity_level, count)
        
        for neg_type in selected_types:
            example = self._generate_negative_example(symbol, neg_type, complexity_level)
            if example:
                negative_examples.append(example)
        
        # Update statistics
        self._update_stats(negative_examples)
        
        return negative_examples
    
    def _select_negative_types(self, symbol: UniversalCodeSymbol, complexity_level: str,
                             count: int) -> List[NegativeExampleType]:
        """Select appropriate negative example types."""

        # Base types that work for all symbols
        base_types = [
            NegativeExampleType.OUT_OF_SCOPE_DOMAIN,
            NegativeExampleType.EXTERNAL_DEPENDENCY,
            NegativeExampleType.HISTORICAL_QUESTION
        ]

        # Add symbol-specific types
        if symbol.symbol_type == "function":
            base_types.extend([
                NegativeExampleType.IMPOSSIBLE_PARAMETER,
                NegativeExampleType.PERFORMANCE_BENCHMARK
            ])
        
        elif symbol.symbol_type == "class":
            base_types.extend([
                NegativeExampleType.NONEXISTENT_METHOD,
                NegativeExampleType.IMPLEMENTATION_DETAIL
            ])
        
        # Add complexity-based types
        if complexity_level in ["complex", "very_complex"]:
            base_types.extend([
                NegativeExampleType.DEPLOYMENT_SPECIFIC,
                NegativeExampleType.CROSS_LANGUAGE
            ])
        
        # Add some advanced types
        base_types.extend([
            NegativeExampleType.SPECULATIVE_FUTURE
        ])
        
        # Remove duplicates and randomly select
        unique_types = list(set(base_types))
        return random.sample(unique_types, min(count, len(unique_types)))
    
    def _generate_negative_example(self, symbol: UniversalCodeSymbol,
                                 neg_type: NegativeExampleType,
                                 complexity_level: str) -> Optional[NegativeExample]:
        """Generate a specific negative example."""
        try:
            templates = self.question_templates.get(neg_type, [])
            if not templates:
                return None
            
            template_data = random.choice(templates)
            
            # Generate question based on template
            question = self._format_question_template(template_data, symbol)
            
            # Create explanation
            explanation = template_data.get("explanation", "Question requires external knowledge")
            
            # Identify trap indicators (things that make it tempting to answer)
            trap_indicators = self._identify_trap_indicators(question, symbol, neg_type)
            
            # Identify abstention cues (things that should trigger NOT_IN_CONTEXT)
            abstention_cues = self._identify_abstention_cues(question, symbol, neg_type)
            
            return NegativeExample(
                question=question,
                expected_response=self.negative_config.expected_response,
                negative_type=neg_type,
                context_symbol_name=symbol.name,
                context_symbol_type=symbol.symbol_type,
                context_code=symbol.source_code,
                difficulty_level=complexity_level,
                explanation=explanation,
                trap_indicators=trap_indicators,
                abstention_cues=abstention_cues
            )
            
        except Exception as e:
            self.logger.error(f"Error generating negative example: {e}")
            return None
    
    def _format_question_template(self, template_data: Dict[str, Any],
                                symbol: UniversalCodeSymbol) -> str:
        """Format a question template with symbol data."""
        template = template_data["template"]
        
        # Build all format arguments at once to avoid sequential formatting issues
        format_args = {"symbol_name": symbol.name}
        
        # Handle fake parameters
        if "fake_params" in template_data:
            fake_param = random.choice(template_data["fake_params"])
            # Ensure fake parameter doesn't exist in real parameters
            while fake_param in symbol.parameters:
                fake_param = random.choice(template_data["fake_params"])
            format_args["fake_param"] = fake_param
        
        # Handle fake values
        if "fake_values" in template_data:
            format_args["fake_value"] = random.choice(template_data["fake_values"])
        
        # Handle fake methods
        if "fake_methods" in template_data:
            format_args["fake_method"] = random.choice(template_data["fake_methods"])
            
            # For comparison questions, add a real method if available
            if "{real_method}" in template:
                real_methods = self._extract_real_methods(symbol)
                if real_methods:
                    format_args["real_method"] = random.choice(real_methods)
        
        # Handle domains
        if "domains" in template_data:
            format_args["domain"] = random.choice(template_data["domains"])
        
        # Handle domain concepts
        if "domain_concepts" in template_data:
            format_args["domain_concept"] = random.choice(template_data["domain_concepts"])
        
        # Handle libraries
        if "libraries" in template_data:
            format_args["library"] = random.choice(template_data["libraries"])
        
        # Handle external systems
        if "external_systems" in template_data:
            format_args["external_system"] = random.choice(template_data["external_systems"])
        
        # Handle operations
        if "operations" in template_data:
            format_args["operation"] = random.choice(template_data["operations"])
        
        # Handle design choices
        if "design_choices" in template_data:
            format_args["design_choice"] = random.choice(template_data["design_choices"])
        
        # Handle data sizes
        if "data_sizes" in template_data:
            format_args["data_size"] = random.choice(template_data["data_sizes"])
        
        # Handle alternatives
        if "alternatives" in template_data:
            format_args["alternative"] = random.choice(template_data["alternatives"])
        
        # Handle languages
        if "languages" in template_data:
            format_args["language"] = random.choice(template_data["languages"])
        
        # Format with all arguments at once, using partial formatting to ignore missing keys
        try:
            question = template.format(**format_args)
        except KeyError as e:
            self.logger.warning(f"Missing format key {e} in template: {template}")
            # Return a fallback question
            question = f"What can you tell me about `{symbol.name}`?"
        
        return question
    
    def _extract_real_methods(self, symbol: UniversalCodeSymbol) -> List[str]:
        """Extract real method names from class code."""
        if symbol.symbol_type != "class":
            return []
        
        # Simple regex to find method definitions
        method_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        methods = re.findall(method_pattern, symbol.source_code)
        
        # Filter out special methods
        return [m for m in methods if not m.startswith('__')]
    
    def _identify_trap_indicators(self, question: str, symbol: UniversalCodeSymbol,
                                neg_type: NegativeExampleType) -> List[str]:
        """Identify what makes this question tempting to answer incorrectly."""
        traps = []
        
        # Check if question mentions real symbol name
        if symbol.name in question:
            traps.append("Question references real symbol name")
        
        # Check if question uses technical terminology
        technical_terms = ["algorithm", "implementation", "performance", "optimization", "configuration"]
        if any(term in question.lower() for term in technical_terms):
            traps.append("Uses technical terminology that might trigger confident response")
        
        # Check if question seems specific and detailed
        if len(question.split()) > 10:
            traps.append("Question appears detailed and specific")
        
        # Type-specific traps
        if neg_type == NegativeExampleType.IMPOSSIBLE_PARAMETER:
            traps.append("Parameter name sounds plausible for this type of function")
        
        elif neg_type == NegativeExampleType.PERFORMANCE_BENCHMARK:
            traps.append("Question asks for specific metrics which seem answerable")
        
        return traps
    
    def _identify_abstention_cues(self, question: str, symbol: UniversalCodeSymbol,
                                neg_type: NegativeExampleType) -> List[str]:
        """Identify cues that should trigger abstention."""
        cues = []
        
        # General abstention cues
        external_indicators = [
            "version", "when", "who", "why", "originally", "first", "history",
            "production", "deployment", "scale", "benchmark", "compare",
            "integrate", "connect", "require", "equivalent"
        ]
        
        if any(indicator in question.lower() for indicator in external_indicators):
            cues.append("Question contains external knowledge indicators")
        
        # Check for questions about non-existent elements
        if neg_type in [NegativeExampleType.IMPOSSIBLE_PARAMETER, NegativeExampleType.NONEXISTENT_METHOD]:
            cues.append("Question asks about elements not present in the code")
        
        # Check for domain-specific questions
        if neg_type == NegativeExampleType.OUT_OF_SCOPE_DOMAIN:
            cues.append("Question assumes domain knowledge not in code")
        
        # Check for comparative questions
        if any(word in question.lower() for word in ["compare", "versus", "better", "faster", "alternative"]):
            cues.append("Question requires external comparison")
        
        return cues
    
    def generate_adversarial_examples(self, symbol: UniversalCodeSymbol,
                                    complexity_level: str) -> List[NegativeExample]:
        """Generate adversarial examples designed to be particularly tricky."""
        adversarial_examples = []

        # Create questions that seem answerable but require external knowledge
        adversarial_templates = [
            f"Based on the code structure, what design pattern does `{symbol.name}` implement?",
            f"What are the time complexity characteristics of `{symbol.name}`?",
            f"How does `{symbol.name}` ensure thread safety?",
            f"What testing strategy would be most appropriate for `{symbol.name}`?",
            f"How does `{symbol.name}` handle edge cases in production?",
        ]

        for template in adversarial_templates[:2]:  # Limit to 2 adversarial examples
            example = NegativeExample(
                question=template,
                expected_response=self.negative_config.expected_response,
                negative_type=NegativeExampleType.IMPLEMENTATION_DETAIL,
                context_symbol_name=symbol.name,
                context_symbol_type=symbol.symbol_type,
                context_code=symbol.source_code,
                difficulty_level="hard",  # Adversarial examples are always hard
                explanation="Question appears answerable from code but requires external analysis",
                trap_indicators=[
                    "Question seems directly related to the code",
                    "Uses technical terminology suggesting code analysis",
                    "Appears to ask for implementation details"
                ],
                abstention_cues=[
                    "Requires analysis beyond what's explicitly shown",
                    "Asks for conclusions that need external knowledge",
                    "Seeks design pattern identification requiring experience"
                ]
            )
            adversarial_examples.append(example)
        
        return adversarial_examples
    
    def _update_stats(self, negative_examples: List[NegativeExample]):
        """Update generation statistics."""
        self.stats.total_generated += len(negative_examples)
        
        for example in negative_examples:
            # Update type distribution
            neg_type = example.negative_type.value
            self.stats.by_type[neg_type] = self.stats.by_type.get(neg_type, 0) + 1
            
            # Update difficulty distribution
            self.stats.by_difficulty[example.difficulty_level] = (
                self.stats.by_difficulty.get(example.difficulty_level, 0) + 1
            )
        
        # Update average trap indicators
        if negative_examples:
            total_traps = sum(len(ex.trap_indicators) for ex in negative_examples)
            self.stats.average_trap_indicators = total_traps / len(negative_examples)
    
    def get_stats(self) -> NegativeGenerationStats:
        """Get generation statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset generation statistics."""
        self.stats = NegativeGenerationStats()


def create_negative_generator(config_manager: ConfigManager) -> EnhancedNegativeGenerator:
    """Create a pre-configured enhanced negative generator."""
    return EnhancedNegativeGenerator(config_manager) 