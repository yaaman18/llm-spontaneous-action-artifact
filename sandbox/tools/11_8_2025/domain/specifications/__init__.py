"""
Domain Specifications for Enactive Consciousness Framework.

This module defines specifications that encapsulate complex business rules
and criteria for consciousness emergence, learning convergence, and
environmental coupling using the Specification pattern.
"""

from .consciousness_specifications import (
    ConsciousnessEmergenceSpecification,
    ConsciousnessStabilitySpecification,
    AttentionalCoherenceSpecification
)

from .learning_specifications import (
    LearningConvergenceSpecification,
    EnvironmentalCouplingSpecification,
    PredictionQualitySpecification
)

__all__ = [
    'ConsciousnessEmergenceSpecification',
    'ConsciousnessStabilitySpecification', 
    'AttentionalCoherenceSpecification',
    'LearningConvergenceSpecification',
    'EnvironmentalCouplingSpecification',
    'PredictionQualitySpecification'
]