"""
Domain Aggregates for Enactive Consciousness Framework.

This module defines the aggregate roots that serve as consistency boundaries
and entry points for domain operations, following DDD patterns.
"""

from .consciousness_aggregate import ConsciousnessAggregate
from .learning_aggregate import LearningAggregate

__all__ = [
    'ConsciousnessAggregate',
    'LearningAggregate'
]