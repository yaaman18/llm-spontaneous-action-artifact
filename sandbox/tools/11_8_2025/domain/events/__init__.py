"""
Domain Events for Enactive Consciousness Framework.

This module defines domain events that represent significant occurrences
in the consciousness and learning domains, enabling decoupled communication
between bounded contexts.
"""

from .domain_events import (
    DomainEvent,
    ConsciousnessStateChanged,
    ConsciousnessEmergenceDetected,
    ConsciousnessFaded,
    AttentionFocusChanged,
    MetacognitiveInsightGained,
    LearningEpochCompleted,
    PredictionErrorThresholdCrossed,
    SelfOrganizationConverged,
    EnvironmentalCouplingStrengthened,
    AdaptiveLearningRateChanged
)

__all__ = [
    'DomainEvent',
    'ConsciousnessStateChanged',
    'ConsciousnessEmergenceDetected',
    'ConsciousnessFaded',
    'AttentionFocusChanged',
    'MetacognitiveInsightGained',
    'LearningEpochCompleted',
    'PredictionErrorThresholdCrossed',
    'SelfOrganizationConverged',
    'EnvironmentalCouplingStrengthened',
    'AdaptiveLearningRateChanged'
]