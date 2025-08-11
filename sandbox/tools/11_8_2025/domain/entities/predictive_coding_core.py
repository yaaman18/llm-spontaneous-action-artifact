"""
Predictive Coding Core Entity.

Central domain entity implementing hierarchical predictive coding based on
the free energy principle. Follows Single Responsibility Principle by
focusing solely on prediction error minimization logic.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy.typing as npt
from ..value_objects.prediction_state import PredictionState
from ..value_objects.precision_weights import PrecisionWeights


class PredictiveCodingCore(ABC):
    """
    Abstract base class for hierarchical predictive coding system.
    
    This entity encapsulates the core business logic for prediction error
    minimization across multiple hierarchical levels. Implementation follows
    the Open/Closed Principle - open for extension (different algorithms)
    but closed for modification of the core interface.
    
    Responsibilities:
    - Hierarchical prediction generation
    - Prediction error computation 
    - Error propagation across levels
    - Precision-weighted learning
    """

    def __init__(self, hierarchy_levels: int, input_dimensions: int):
        """
        Initialize the predictive coding core.
        
        Args:
            hierarchy_levels: Number of hierarchical processing levels
            input_dimensions: Dimensionality of input data
            
        Raises:
            ValueError: If hierarchy_levels < 1 or input_dimensions < 1
        """
        if hierarchy_levels < 1:
            raise ValueError("Hierarchy levels must be positive")
        if input_dimensions < 1:
            raise ValueError("Input dimensions must be positive")
            
        self._hierarchy_levels = hierarchy_levels
        self._input_dimensions = input_dimensions
        self._current_state: Optional[PredictionState] = None

    @property
    def hierarchy_levels(self) -> int:
        """Number of hierarchical processing levels."""
        return self._hierarchy_levels

    @property
    def input_dimensions(self) -> int:
        """Dimensionality of input data."""
        return self._input_dimensions

    @property
    def current_state(self) -> Optional[PredictionState]:
        """Current prediction state of the system."""
        return self._current_state

    @abstractmethod
    def generate_predictions(
        self, 
        input_data: npt.NDArray,
        precision_weights: PrecisionWeights
    ) -> List[npt.NDArray]:
        """
        Generate hierarchical predictions from input data.
        
        Args:
            input_data: Input sensory data
            precision_weights: Precision weights for each level
            
        Returns:
            List of predictions for each hierarchical level
            
        Raises:
            ValueError: If input shape doesn't match expected dimensions
        """
        pass

    @abstractmethod
    def compute_prediction_errors(
        self,
        predictions: List[npt.NDArray],
        targets: List[npt.NDArray]
    ) -> List[npt.NDArray]:
        """
        Compute prediction errors across hierarchical levels.
        
        Args:
            predictions: Hierarchical predictions
            targets: Target values for each level
            
        Returns:
            List of prediction errors for each level
            
        Raises:
            ValueError: If predictions and targets don't match in structure
        """
        pass

    @abstractmethod
    def propagate_errors(
        self,
        prediction_errors: List[npt.NDArray],
        precision_weights: PrecisionWeights
    ) -> Tuple[List[npt.NDArray], PredictionState]:
        """
        Propagate errors through the hierarchy and update internal state.
        
        Args:
            prediction_errors: Errors from each hierarchical level
            precision_weights: Precision weights for error scaling
            
        Returns:
            Tuple of (updated_errors, new_prediction_state)
            
        Raises:
            ValueError: If error propagation fails
        """
        pass

    @abstractmethod
    def update_predictions(
        self,
        learning_rate: float,
        propagated_errors: List[npt.NDArray]
    ) -> None:
        """
        Update internal prediction models based on propagated errors.
        
        Args:
            learning_rate: Learning rate for parameter updates
            propagated_errors: Errors propagated through hierarchy
            
        Raises:
            ValueError: If learning_rate is not in valid range (0, 1]
        """
        pass

    def process_input(
        self,
        input_data: npt.NDArray,
        precision_weights: PrecisionWeights,
        learning_rate: float = 0.01
    ) -> PredictionState:
        """
        Complete processing cycle: predict -> error -> propagate -> update.
        
        Template method implementing the prediction-error cycle following
        the Free Energy Principle minimization algorithm.
        
        Args:
            input_data: Input sensory data
            precision_weights: Precision weights for each level  
            learning_rate: Learning rate for parameter updates
            
        Returns:
            Updated prediction state after processing
            
        Raises:
            ValueError: If any processing step fails
        """
        # Generate predictions (step 1): μ_i ← f(μ_{i+1})
        predictions = self.generate_predictions(input_data, precision_weights)
        
        # Create targets (step 2): s_i from sensory input and higher levels
        targets = self._create_targets_from_input(input_data, predictions)
        
        # Compute prediction errors (step 3): ε_i = s_i - μ_i
        prediction_errors = self.compute_prediction_errors(predictions, targets)
        
        # Compute free energy before update for tracking
        free_energy_before = self.compute_free_energy(predictions, targets, precision_weights)
        
        # Propagate errors through hierarchy (step 4): message passing
        propagated_errors, new_state = self.propagate_errors(
            prediction_errors, precision_weights
        )
        
        # Update internal models (step 5): Δθ ∝ -∇_θ F
        self.update_predictions(learning_rate, propagated_errors)
        
        # Update precisions (step 6): Π̇_i = γ(⟨ε_i²⟩ - Π_i^{-1})
        updated_precision_weights = self.update_precisions(propagated_errors, learning_rate)
        
        # Add free energy to state metadata
        enhanced_state = new_state.add_metadata('free_energy_before_update', free_energy_before)
        enhanced_state = enhanced_state.add_metadata('updated_precision_weights', updated_precision_weights.to_dict())
        
        # Update current state
        self._current_state = enhanced_state
        
        return enhanced_state

    @abstractmethod
    def _create_targets_from_input(
        self,
        input_data: npt.NDArray,
        predictions: List[npt.NDArray]
    ) -> List[npt.NDArray]:
        """
        Create target values from input data and current predictions.
        
        This method handles the creation of hierarchical targets based on
        the current input and existing predictions. Implementation details
        depend on the specific predictive coding algorithm used.
        
        Args:
            input_data: Current input data
            predictions: Current hierarchical predictions
            
        Returns:
            List of target values for each hierarchical level
        """
        pass

    @abstractmethod
    def compute_free_energy(
        self,
        predictions: List[npt.NDArray],
        targets: List[npt.NDArray],
        precision_weights: PrecisionWeights
    ) -> float:
        """
        Compute variational free energy F = D_KL[q||p] - E_q[ln p(s|θ)].
        
        The free energy quantifies the bound on model evidence and drives
        both perception (inference) and learning (parameter updates).
        
        Args:
            predictions: Hierarchical predictions μ_i
            targets: Target values for each level
            precision_weights: Precision matrices Π_i
            
        Returns:
            Scalar free energy value
            
        Raises:
            ValueError: If computation fails due to invalid inputs
        """
        pass

    @abstractmethod
    def update_precisions(
        self,
        prediction_errors: List[npt.NDArray],
        learning_rate: float = 0.01
    ) -> PrecisionWeights:
        """
        Update precision weights based on prediction error statistics.
        
        Precision (inverse variance) should increase when errors are 
        consistent and decrease when errors are highly variable.
        
        Update rule: Π̇_i = γ(⟨ε_i²⟩ - Π_i^{-1})
        
        Args:
            prediction_errors: Current prediction errors ε_i
            learning_rate: Precision adaptation rate γ
            
        Returns:
            Updated precision weights
        """
        pass

    def reset_state(self) -> None:
        """Reset the internal state of the predictive coding system."""
        self._current_state = None

    def get_total_prediction_error(self) -> float:
        """
        Calculate total prediction error across all hierarchical levels.
        
        Returns:
            Scalar value representing total system prediction error
            
        Raises:
            RuntimeError: If no current state exists
        """
        if self._current_state is None:
            raise RuntimeError("No current state available")
        
        return self._current_state.total_error