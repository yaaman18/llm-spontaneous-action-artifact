"""
Bounded Contexts for Enactive Consciousness Framework.

This module organizes the domain into clear bounded contexts that represent
distinct areas of the business domain with their own ubiquitous language
and domain models.

Bounded Contexts:
- Consciousness Context: Core consciousness states, emergence, and awareness
- Learning Context: Predictive coding, SOM, and adaptive learning processes  
- Monitoring Context: Metacognitive monitoring and self-awareness
- Environmental Context: Environmental coupling and interaction dynamics
"""

from .consciousness_context import ConsciousnessContext
from .learning_context import LearningContext
from .monitoring_context import MonitoringContext
from .environmental_context import EnvironmentalContext

__all__ = [
    'ConsciousnessContext',
    'LearningContext', 
    'MonitoringContext',
    'EnvironmentalContext'
]