"""
Learning adaptation service for dynamic learning rate adjustment.

This service implements adaptive learning policies that adjust learning
parameters based on environmental coupling and prediction performance.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from ..value_objects.learning_parameters import LearningParameters
from ..value_objects.prediction_state import PredictionState


class LearningAdaptationService(ABC):
    """
    Abstract service for adaptive learning rate management.
    
    This service implements policies for adjusting learning parameters
    based on prediction performance and environmental coupling strength.
    """
    
    @abstractmethod
    def adapt_learning_rate(
        self,
        current_params: LearningParameters,
        prediction_state: PredictionState,
        coupling_strength: float
    ) -> LearningParameters:
        """
        Adapt learning parameters based on current performance.
        
        Args:
            current_params: Current learning parameters
            prediction_state: Current prediction performance state
            coupling_strength: Environmental coupling strength [0,1]
            
        Returns:
            Adapted learning parameters
        """
        pass
    
    @abstractmethod
    def should_adjust_precision(
        self,
        prediction_state: PredictionState
    ) -> bool:
        """
        Determine if precision weights should be adjusted.
        
        Args:
            prediction_state: Current prediction state
            
        Returns:
            True if precision adjustment is recommended
        """
        pass
    
    @abstractmethod
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """
        Get current adaptation metrics for monitoring.
        
        Returns:
            Dictionary of adaptation metrics
        """
        pass