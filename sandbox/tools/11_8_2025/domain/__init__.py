"""
Domain layer for the enactive consciousness framework.

Minimal working implementation focusing on core value objects
and basic domain entities.
"""

# Core value objects (most stable components)
from .value_objects import PhiValue, ConsciousnessState

__all__ = [
    'PhiValue',
    'ConsciousnessState'
]