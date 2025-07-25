"""
Quality critique package for synthetic data evaluation.

This package provides quality assessment and deduplication tools
for evaluating and filtering generated training data.
"""

from .quality_critic import (
    QualityCritic,
    QualityScores,
    CriticFeedback,
    CriticStats
)

from .deduplicator import (
    QADeduplicator,
    SimilarityCluster,
    DeduplicationStats,
    create_deduplicator
)

__all__ = [
    'QualityCritic',
    'QualityScores',
    'CriticFeedback', 
    'CriticStats',
    'QADeduplicator',
    'SimilarityCluster',
    'DeduplicationStats',
    'create_deduplicator'
] 