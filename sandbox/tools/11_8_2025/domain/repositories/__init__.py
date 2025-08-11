"""
Repository interfaces for the enactive consciousness system.

Repository interfaces define contracts for data persistence and retrieval
following the Repository pattern and Dependency Inversion Principle.
These interfaces are implemented in the infrastructure layer.
"""

from .consciousness_repository import ConsciousnessRepository
from .prediction_repository import PredictionRepository
from .learning_repository import LearningRepository

__all__ = [
    "ConsciousnessRepository",
    "PredictionRepository", 
    "LearningRepository",
]