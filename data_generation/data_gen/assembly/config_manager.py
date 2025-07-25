"""
Configuration Manager for Task Taxonomy

This module provides configuration management for the synthetic data generation pipeline,
loading and validating task specifications from YAML configuration files.
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging


@dataclass
class TaskConfig:
    """Configuration for a specific task type."""
    count: int
    focus_areas: List[str] = field(default_factory=list)
    templates: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)


@dataclass
class ComplexityConfig:
    """Configuration for a complexity level."""
    description: str
    qa_pairs: TaskConfig
    completion_tasks: TaskConfig
    debugging_tasks: TaskConfig
    refactoring_tasks: TaskConfig
    docstring_tasks: TaskConfig


@dataclass
class SymbolTypeConfig:
    """Configuration for a symbol type."""
    enabled: bool
    priority: int


@dataclass
class GlobalConfig:
    """Global configuration settings."""
    max_total_tasks_per_symbol: int
    min_total_tasks_per_symbol: int
    enable_grounding_verification: bool
    require_context_citations: bool


@dataclass
class QualityThresholds:
    """Quality threshold configuration."""
    groundedness_min_score: float
    specificity_min_score: float
    clarity_min_score: float
    usefulness_min_score: float


@dataclass
class NegativeExampleConfig:
    """Configuration for negative examples."""
    enabled: bool
    percentage_of_total: float
    types: List[str]
    expected_response: str


@dataclass
class TaskTaxonomyConfig:
    """Complete task taxonomy configuration."""
    global_config: GlobalConfig
    symbol_types: Dict[str, SymbolTypeConfig]
    complexity_levels: Dict[str, ComplexityConfig]
    task_configs: Dict[str, Dict[str, Any]]
    quality_thresholds: QualityThresholds
    negative_examples: NegativeExampleConfig


class ConfigManager:
    """Manages task taxonomy configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.logger = logging.getLogger(__name__)
        
        if config_path is None:
            # Default to the config file in the same package
            config_path = Path(__file__).parent / "../config/task_taxonomy.yaml"
        
        self.config_path = Path(config_path)
        self.config: Optional[TaskTaxonomyConfig] = None
        
        if self.config_path.exists():
            self.load_config()
        else:
            self.logger.warning(f"Config file not found: {self.config_path}")
    
    def load_config(self) -> TaskTaxonomyConfig:
        """Load and validate configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            self.config = self._parse_config(raw_config)
            self._validate_config()
            
            self.logger.info(f"Successfully loaded configuration from {self.config_path}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _parse_config(self, raw_config: Dict[str, Any]) -> TaskTaxonomyConfig:
        """Parse raw YAML config into structured configuration objects."""
        
        # Parse global config
        global_raw = raw_config.get('global', {})
        global_config = GlobalConfig(
            max_total_tasks_per_symbol=global_raw.get('max_total_tasks_per_symbol', 10),
            min_total_tasks_per_symbol=global_raw.get('min_total_tasks_per_symbol', 3),
            enable_grounding_verification=global_raw.get('enable_grounding_verification', True),
            require_context_citations=global_raw.get('require_context_citations', True)
        )
        
        # Parse symbol types
        symbol_types = {}
        for name, config in raw_config.get('symbol_types', {}).items():
            symbol_types[name] = SymbolTypeConfig(
                enabled=config.get('enabled', True),
                priority=config.get('priority', 5)
            )
        
        # Parse complexity levels
        complexity_levels = {}
        for level, config in raw_config.get('complexity_levels', {}).items():
            complexity_levels[level] = ComplexityConfig(
                description=config.get('description', ''),
                qa_pairs=self._parse_task_config(config.get('qa_pairs', {})),
                completion_tasks=self._parse_task_config(config.get('completion_tasks', {})),
                debugging_tasks=self._parse_task_config(config.get('debugging_tasks', {})),
                refactoring_tasks=self._parse_task_config(config.get('refactoring_tasks', {})),
                docstring_tasks=self._parse_task_config(config.get('docstring_tasks', {}))
            )
        
        # Parse task configs (keep as dict for flexibility)
        task_configs = raw_config.get('task_configs', {})
        
        # Parse quality thresholds
        thresholds_raw = raw_config.get('quality_thresholds', {})
        quality_thresholds = QualityThresholds(
            groundedness_min_score=thresholds_raw.get('groundedness_min_score', 0.7),
            specificity_min_score=thresholds_raw.get('specificity_min_score', 0.6),
            clarity_min_score=thresholds_raw.get('clarity_min_score', 0.6),
            usefulness_min_score=thresholds_raw.get('usefulness_min_score', 0.5)
        )
        
        # Parse negative examples
        neg_raw = raw_config.get('negative_examples', {})
        negative_examples = NegativeExampleConfig(
            enabled=neg_raw.get('enabled', True),
            percentage_of_total=neg_raw.get('percentage_of_total', 0.15),
            types=neg_raw.get('types', []),
            expected_response=neg_raw.get('expected_response', 'NOT_IN_CONTEXT')
        )
        
        return TaskTaxonomyConfig(
            global_config=global_config,
            symbol_types=symbol_types,
            complexity_levels=complexity_levels,
            task_configs=task_configs,
            quality_thresholds=quality_thresholds,
            negative_examples=negative_examples
        )
    
    def _parse_task_config(self, config: Dict[str, Any]) -> TaskConfig:
        """Parse a task configuration section."""
        return TaskConfig(
            count=config.get('count', 1),
            focus_areas=config.get('focus_areas', []),
            templates=config.get('templates', []),
            requirements=config.get('requirements', [])
        )
    
    def _validate_config(self):
        """Validate the loaded configuration."""
        if not self.config:
            raise ValueError("No configuration loaded")
        
        # Validate global config
        if self.config.global_config.max_total_tasks_per_symbol < self.config.global_config.min_total_tasks_per_symbol:
            raise ValueError("max_total_tasks_per_symbol must be >= min_total_tasks_per_symbol")
        
        # Validate complexity levels
        if not self.config.complexity_levels:
            raise ValueError("At least one complexity level must be defined")
        
        required_complexity_levels = {'simple', 'moderate', 'complex', 'very_complex'}
        missing_levels = required_complexity_levels - set(self.config.complexity_levels.keys())
        if missing_levels:
            self.logger.warning(f"Missing complexity levels: {missing_levels}")
        
        # Validate symbol types
        if not self.config.symbol_types:
            raise ValueError("At least one symbol type must be enabled")
        
        enabled_types = [name for name, config in self.config.symbol_types.items() if config.enabled]
        if not enabled_types:
            raise ValueError("At least one symbol type must be enabled")
        
        # Validate quality thresholds
        thresholds = self.config.quality_thresholds
        if not (0 <= thresholds.groundedness_min_score <= 1):
            raise ValueError("groundedness_min_score must be between 0 and 1")
        if not (0 <= thresholds.specificity_min_score <= 1):
            raise ValueError("specificity_min_score must be between 0 and 1")
        if not (0 <= thresholds.clarity_min_score <= 1):
            raise ValueError("clarity_min_score must be between 0 and 1")
        if not (0 <= thresholds.usefulness_min_score <= 1):
            raise ValueError("usefulness_min_score must be between 0 and 1")
        
        # Validate negative examples
        if self.config.negative_examples.enabled:
            if not (0 <= self.config.negative_examples.percentage_of_total <= 1):
                raise ValueError("negative_examples.percentage_of_total must be between 0 and 1")
    
    def get_task_distribution(self, symbol_type: str, complexity_level: str) -> Dict[str, int]:
        """
        Get the task distribution for a given symbol type and complexity level.
        
        Args:
            symbol_type: Type of symbol (function, class, method, etc.)
            complexity_level: Complexity level (simple, moderate, complex, very_complex)
            
        Returns:
            Dictionary mapping task types to counts
        """
        if not self.config:
            raise ValueError("Configuration not loaded")
        
        # Check if symbol type is enabled
        symbol_config = self.config.symbol_types.get(symbol_type)
        if not symbol_config or not symbol_config.enabled:
            return {}
        
        # Get complexity configuration
        complexity_config = self.config.complexity_levels.get(complexity_level)
        if not complexity_config:
            self.logger.warning(f"Unknown complexity level: {complexity_level}")
            return {}
        
        # Build task distribution
        distribution = {
            'qa_pairs': complexity_config.qa_pairs.count,
            'completion_tasks': complexity_config.completion_tasks.count,
            'debugging_tasks': complexity_config.debugging_tasks.count,
            'refactoring_tasks': complexity_config.refactoring_tasks.count,
            'docstring_tasks': complexity_config.docstring_tasks.count
        }
        
        # Apply global limits
        total_tasks = sum(distribution.values())
        max_tasks = self.config.global_config.max_total_tasks_per_symbol
        min_tasks = self.config.global_config.min_total_tasks_per_symbol
        
        if total_tasks > max_tasks:
            # Scale down proportionally
            scale_factor = max_tasks / total_tasks
            for task_type in distribution:
                distribution[task_type] = max(1, int(distribution[task_type] * scale_factor))
        
        elif total_tasks < min_tasks:
            # Scale up, preferring QA pairs
            additional_tasks = min_tasks - total_tasks
            distribution['qa_pairs'] += additional_tasks
        
        return distribution
    
    def get_task_config(self, task_type: str, complexity_level: str) -> TaskConfig:
        """
        Get the task configuration for a specific task type and complexity level.
        
        Args:
            task_type: Type of task (qa_pairs, completion_tasks, etc.)
            complexity_level: Complexity level
            
        Returns:
            TaskConfig object
        """
        if not self.config:
            raise ValueError("Configuration not loaded")
        
        complexity_config = self.config.complexity_levels.get(complexity_level)
        if not complexity_config:
            raise ValueError(f"Unknown complexity level: {complexity_level}")
        
        task_configs = {
            'qa_pairs': complexity_config.qa_pairs,
            'completion_tasks': complexity_config.completion_tasks,
            'debugging_tasks': complexity_config.debugging_tasks,
            'refactoring_tasks': complexity_config.refactoring_tasks,
            'docstring_tasks': complexity_config.docstring_tasks
        }
        
        if task_type not in task_configs:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return task_configs[task_type]
    
    def get_quality_thresholds(self) -> QualityThresholds:
        """Get quality thresholds configuration."""
        if not self.config:
            raise ValueError("Configuration not loaded")
        return self.config.quality_thresholds
    
    def get_negative_example_config(self) -> NegativeExampleConfig:
        """Get negative example configuration."""
        if not self.config:
            raise ValueError("Configuration not loaded")
        return self.config.negative_examples
    
    def should_generate_negative_examples(self) -> bool:
        """Check if negative examples should be generated."""
        if not self.config:
            return False
        return self.config.negative_examples.enabled
    
    def get_enabled_symbol_types(self) -> List[str]:
        """Get list of enabled symbol types."""
        if not self.config:
            return []
        
        return [
            name for name, config in self.config.symbol_types.items() 
            if config.enabled
        ]
    
    def get_symbol_type_priority(self, symbol_type: str) -> int:
        """Get priority for a symbol type."""
        if not self.config:
            return 5  # Default priority
        
        symbol_config = self.config.symbol_types.get(symbol_type)
        return symbol_config.priority if symbol_config else 5
    
    def get_task_specific_config(self, task_type: str) -> Dict[str, Any]:
        """Get task-specific configuration."""
        if not self.config:
            return {}
        
        return self.config.task_configs.get(task_type, {})
    
    def export_config(self, output_path: str):
        """Export current configuration to YAML file."""
        if not self.config:
            raise ValueError("No configuration to export")
        
        # Convert back to dictionary format
        config_dict = {
            'global': {
                'max_total_tasks_per_symbol': self.config.global_config.max_total_tasks_per_symbol,
                'min_total_tasks_per_symbol': self.config.global_config.min_total_tasks_per_symbol,
                'enable_grounding_verification': self.config.global_config.enable_grounding_verification,
                'require_context_citations': self.config.global_config.require_context_citations
            },
            'symbol_types': {
                name: {'enabled': config.enabled, 'priority': config.priority}
                for name, config in self.config.symbol_types.items()
            },
            'quality_thresholds': {
                'groundedness_min_score': self.config.quality_thresholds.groundedness_min_score,
                'specificity_min_score': self.config.quality_thresholds.specificity_min_score,
                'clarity_min_score': self.config.quality_thresholds.clarity_min_score,
                'usefulness_min_score': self.config.quality_thresholds.usefulness_min_score
            },
            'negative_examples': {
                'enabled': self.config.negative_examples.enabled,
                'percentage_of_total': self.config.negative_examples.percentage_of_total,
                'types': self.config.negative_examples.types,
                'expected_response': self.config.negative_examples.expected_response
            },
            'task_configs': self.config.task_configs
        }
        
        # Add complexity levels
        complexity_dict = {}
        for level, config in self.config.complexity_levels.items():
            complexity_dict[level] = {
                'description': config.description,
                'qa_pairs': {
                    'count': config.qa_pairs.count,
                    'focus_areas': config.qa_pairs.focus_areas,
                    'templates': config.qa_pairs.templates
                },
                'completion_tasks': {
                    'count': config.completion_tasks.count,
                    'focus_areas': config.completion_tasks.focus_areas
                },
                'debugging_tasks': {
                    'count': config.debugging_tasks.count,
                    'focus_areas': config.debugging_tasks.focus_areas
                },
                'refactoring_tasks': {
                    'count': config.refactoring_tasks.count,
                    'focus_areas': config.refactoring_tasks.focus_areas
                },
                'docstring_tasks': {
                    'count': config.docstring_tasks.count,
                    'requirements': config.docstring_tasks.requirements
                }
            }
        
        config_dict['complexity_levels'] = complexity_dict
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Configuration exported to {output_path}")


# Default configuration manager instance
_default_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the default configuration manager instance."""
    global _default_config_manager
    if _default_config_manager is None:
        _default_config_manager = ConfigManager()
    return _default_config_manager

def load_config(config_path: str) -> ConfigManager:
    """Load configuration from a specific path."""
    return ConfigManager(config_path) 