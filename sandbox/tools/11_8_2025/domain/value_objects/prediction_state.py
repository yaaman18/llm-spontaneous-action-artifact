"""
Prediction State Value Object.

Immutable representation of the hierarchical prediction state
in the predictive coding system. Encapsulates prediction errors,
hierarchical representations, and temporal dynamics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PredictionState:
    """
    Immutable representation of prediction system state.
    
    Value object encapsulating all aspects of the predictive coding
    system's current state including hierarchical errors, predictions,
    and quality metrics.
    """
    
    hierarchical_errors: List[float]
    hierarchical_predictions: List[npt.NDArray] = field(default_factory=list)
    precision_weighted_errors: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    convergence_status: str = field(default="not_converged")
    learning_iteration: int = field(default=0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate prediction state after initialization."""
        self._validate_hierarchical_consistency()
        self._validate_convergence_status()
        self._validate_learning_iteration()

    def _validate_hierarchical_consistency(self) -> None:
        """Validate consistency between hierarchical components."""
        if not self.hierarchical_errors:
            raise ValueError("Hierarchical errors cannot be empty")
            
        if self.hierarchical_predictions and len(self.hierarchical_predictions) != len(self.hierarchical_errors):
            raise ValueError("Predictions and errors must have same hierarchy levels")
            
        if self.precision_weighted_errors and len(self.precision_weighted_errors) != len(self.hierarchical_errors):
            raise ValueError("Precision weighted errors must match hierarchy levels")
            
        # Check for valid error values
        if any(not isinstance(error, (int, float)) or np.isnan(error) for error in self.hierarchical_errors):
            raise ValueError("All hierarchical errors must be valid numbers")

    def _validate_convergence_status(self) -> None:
        """Validate convergence status is valid."""
        valid_statuses = {"not_converged", "converging", "converged", "diverged"}
        if self.convergence_status not in valid_statuses:
            raise ValueError(f"Invalid convergence status: {self.convergence_status}")

    def _validate_learning_iteration(self) -> None:
        """Validate learning iteration is non-negative."""
        if self.learning_iteration < 0:
            raise ValueError("Learning iteration must be non-negative")

    @property
    def hierarchy_levels(self) -> int:
        """Number of hierarchical levels in the prediction system."""
        return len(self.hierarchical_errors)

    @property
    def total_error(self) -> float:
        """Total prediction error across all hierarchical levels."""
        return sum(abs(error) for error in self.hierarchical_errors)

    @property
    def mean_error(self) -> float:
        """Mean prediction error across hierarchical levels."""
        return self.total_error / len(self.hierarchical_errors)

    @property
    def error_variance(self) -> float:
        """Variance of prediction errors across levels."""
        if len(self.hierarchical_errors) <= 1:
            return 0.0
        
        mean_error = self.mean_error
        return sum((error - mean_error) ** 2 for error in self.hierarchical_errors) / len(self.hierarchical_errors)

    @property
    def is_converged(self) -> bool:
        """Check if the prediction system has converged."""
        return self.convergence_status == "converged"

    @property
    def is_stable(self) -> bool:
        """Check if the prediction errors are stable (low variance)."""
        return self.error_variance < 0.01  # Threshold for stability

    @property
    def prediction_quality(self) -> float:
        """
        Compute overall prediction quality [0, 1].
        
        Returns:
            Quality score where 1.0 is perfect prediction
        """
        # Convert error to quality (lower error = higher quality)
        max_reasonable_error = 10.0  # Domain-specific threshold
        normalized_error = min(self.total_error / max_reasonable_error, 1.0)
        return 1.0 - normalized_error

    def get_error_at_level(self, level: int) -> float:
        """
        Get prediction error at specific hierarchical level.
        
        Args:
            level: Hierarchical level (0-indexed)
            
        Returns:
            Prediction error at the specified level
            
        Raises:
            IndexError: If level is out of bounds
        """
        if not (0 <= level < len(self.hierarchical_errors)):
            raise IndexError(f"Level {level} out of bounds for {len(self.hierarchical_errors)} levels")
        
        return self.hierarchical_errors[level]

    def get_prediction_at_level(self, level: int) -> Optional[npt.NDArray]:
        """
        Get prediction at specific hierarchical level.
        
        Args:
            level: Hierarchical level (0-indexed)
            
        Returns:
            Prediction array at the specified level, or None if not available
            
        Raises:
            IndexError: If level is out of bounds
        """
        if not (0 <= level < len(self.hierarchical_errors)):
            raise IndexError(f"Level {level} out of bounds for {len(self.hierarchical_errors)} levels")
        
        if not self.hierarchical_predictions:
            return None
            
        return self.hierarchical_predictions[level]

    def with_updated_errors(self, new_errors: List[float]) -> 'PredictionState':
        """
        Create new prediction state with updated errors.
        
        Args:
            new_errors: New hierarchical error values
            
        Returns:
            New PredictionState instance with updated errors
        """
        return PredictionState(
            hierarchical_errors=new_errors,
            hierarchical_predictions=self.hierarchical_predictions,
            precision_weighted_errors=self.precision_weighted_errors,
            timestamp=datetime.now(),
            convergence_status=self._determine_convergence_status(new_errors),
            learning_iteration=self.learning_iteration + 1,
            metadata=self.metadata.copy()
        )

    def with_updated_predictions(self, new_predictions: List[npt.NDArray]) -> 'PredictionState':
        """
        Create new prediction state with updated predictions.
        
        Args:
            new_predictions: New hierarchical predictions
            
        Returns:
            New PredictionState instance with updated predictions
        """
        return PredictionState(
            hierarchical_errors=self.hierarchical_errors,
            hierarchical_predictions=new_predictions,
            precision_weighted_errors=self.precision_weighted_errors,
            timestamp=datetime.now(),
            convergence_status=self.convergence_status,
            learning_iteration=self.learning_iteration,
            metadata=self.metadata.copy()
        )

    def _determine_convergence_status(self, new_errors: List[float]) -> str:
        """Determine convergence status based on error trajectory."""
        current_total = sum(abs(error) for error in new_errors)
        previous_total = self.total_error
        
        # Simple convergence heuristics
        if current_total < 0.001:
            return "converged"
        elif current_total > previous_total * 1.1:
            return "diverged"
        elif current_total < previous_total * 0.99:
            return "converging"
        else:
            return "not_converged"

    def add_metadata(self, key: str, value: Any) -> 'PredictionState':
        """
        Add metadata to the prediction state.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            New PredictionState with added metadata
        """
        new_metadata = self.metadata.copy()
        new_metadata[key] = value
        
        return PredictionState(
            hierarchical_errors=self.hierarchical_errors,
            hierarchical_predictions=self.hierarchical_predictions,
            precision_weighted_errors=self.precision_weighted_errors,
            timestamp=self.timestamp,
            convergence_status=self.convergence_status,
            learning_iteration=self.learning_iteration,
            metadata=new_metadata
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert prediction state to dictionary representation.
        
        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "hierarchical_errors": self.hierarchical_errors,
            "hierarchical_predictions": [
                pred.tolist() if isinstance(pred, np.ndarray) else pred 
                for pred in self.hierarchical_predictions
            ],
            "precision_weighted_errors": self.precision_weighted_errors,
            "timestamp": self.timestamp.isoformat(),
            "convergence_status": self.convergence_status,
            "learning_iteration": self.learning_iteration,
            "metadata": self.metadata,
            "total_error": self.total_error,
            "mean_error": self.mean_error,
            "prediction_quality": self.prediction_quality,
            "is_converged": self.is_converged,
            "is_stable": self.is_stable
        }

    @classmethod
    def create_empty(cls, hierarchy_levels: int) -> 'PredictionState':
        """
        Create empty prediction state for initialization.
        
        Args:
            hierarchy_levels: Number of hierarchical levels
            
        Returns:
            Empty PredictionState instance
        """
        return cls(
            hierarchical_errors=[0.0] * hierarchy_levels,
            hierarchical_predictions=[],
            precision_weighted_errors=[],
            timestamp=datetime.now(),
            convergence_status="not_converged",
            learning_iteration=0,
            metadata={"initialized": True}
        )