"""
Domain Policies for Enactive Consciousness Framework.

This module defines domain policies that encapsulate complex business
rules and decision-making logic for consciousness emergence, learning
adaptation, and environmental coupling using the Policy pattern.
"""

from .consciousness_policies import (
    ConsciousnessEmergencePolicy,
    AttentionRegulationPolicy,
    MetacognitiveMonitoringPolicy
)

from .learning_policies import (
    AdaptiveLearningRatePolicy,
    EnvironmentalCouplingPolicy,
    PredictionErrorRegulationPolicy
)

__all__ = [
    'ConsciousnessEmergencePolicy',
    'AttentionRegulationPolicy',
    'MetacognitiveMonitoringPolicy',
    'AdaptiveLearningRatePolicy',
    'EnvironmentalCouplingPolicy',
    'PredictionErrorRegulationPolicy'
]