"""
Domain value objects for the enactive consciousness system.

Value objects are immutable data structures that represent domain concepts
without identity. They encapsulate related data and behavior while ensuring
immutability and value-based equality semantics.
"""

from .consciousness_state import ConsciousnessState
from .prediction_state import PredictionState
from .precision_weights import PrecisionWeights
from .som_topology import SOMTopology
from .learning_parameters import LearningParameters
from .phi_value import PhiValue
from .probability_distribution import ProbabilityDistribution

__all__ = [
    "ConsciousnessState",
    "PredictionState", 
    "PrecisionWeights",
    "SOMTopology",
    "LearningParameters",
    "PhiValue",
    "ProbabilityDistribution",
]