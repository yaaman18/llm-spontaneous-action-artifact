"""
Precision Weights Value Object.

Immutable representation of precision weights used in predictive coding
for attention and error scaling across hierarchical levels.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PrecisionWeights:
    """
    Immutable representation of precision weights for predictive coding.
    
    Precision weights determine the relative importance of prediction errors
    at different hierarchical levels, implementing attention mechanisms
    and error scaling in the predictive processing framework.
    """
    
    weights: npt.NDArray
    normalization_method: str = field(default="softmax")
    temperature: float = field(default=1.0)
    adaptation_rate: float = field(default=0.01)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate precision weights after initialization."""
        self._validate_weights()
        self._validate_normalization_method()
        self._validate_temperature()
        self._validate_adaptation_rate()

    def _validate_weights(self) -> None:
        """Validate weight array properties."""
        if not isinstance(self.weights, np.ndarray):
            raise ValueError("Weights must be numpy array")
        if self.weights.ndim != 1:
            raise ValueError("Weights must be 1-dimensional array")
        if len(self.weights) == 0:
            raise ValueError("Weights array cannot be empty")
        if np.any(self.weights < 0):
            raise ValueError("All weights must be non-negative")
        if np.all(self.weights == 0):
            raise ValueError("At least one weight must be positive")

    def _validate_normalization_method(self) -> None:
        """Validate normalization method."""
        valid_methods = {"softmax", "sum_to_one", "max_normalize", "none"}
        if self.normalization_method not in valid_methods:
            raise ValueError(f"Invalid normalization method: {self.normalization_method}")

    def _validate_temperature(self) -> None:
        """Validate temperature parameter."""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")

    def _validate_adaptation_rate(self) -> None:
        """Validate adaptation rate."""
        if not (0.0 <= self.adaptation_rate <= 1.0):
            raise ValueError("Adaptation rate must be in [0, 1]")

    @property
    def hierarchy_levels(self) -> int:
        """Number of hierarchical levels."""
        return len(self.weights)

    @property
    def normalized_weights(self) -> npt.NDArray:
        """Get normalized weights according to the normalization method."""
        return self._apply_normalization(self.weights)

    @property
    def entropy(self) -> float:
        """Calculate entropy of the weight distribution."""
        normalized = self.normalized_weights
        return -np.sum(normalized * np.log(normalized + 1e-10))

    @property
    def max_entropy(self) -> float:
        """Maximum possible entropy for uniform distribution."""
        return np.log(self.hierarchy_levels)

    @property
    def attention_focus(self) -> float:
        """Attention focus measure (1 - normalized entropy)."""
        if self.max_entropy == 0:
            return 1.0
        return 1.0 - (self.entropy / self.max_entropy)

    @property
    def dominant_level(self) -> int:
        """Index of the hierarchical level with highest precision weight."""
        return int(np.argmax(self.weights))

    def _apply_normalization(self, weights: npt.NDArray) -> npt.NDArray:
        """Apply the specified normalization method."""
        if self.normalization_method == "softmax":
            exp_weights = np.exp(weights / self.temperature)
            return exp_weights / np.sum(exp_weights)
        elif self.normalization_method == "sum_to_one":
            return weights / np.sum(weights)
        elif self.normalization_method == "max_normalize":
            return weights / np.max(weights)
        elif self.normalization_method == "none":
            return weights.copy()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

    def get_weight_at_level(self, level: int) -> float:
        """
        Get precision weight at specific hierarchical level.
        
        Args:
            level: Hierarchical level (0-indexed)
            
        Returns:
            Precision weight at the specified level
            
        Raises:
            IndexError: If level is out of bounds
        """
        if not (0 <= level < len(self.weights)):
            raise IndexError(f"Level {level} out of bounds for {len(self.weights)} levels")
        
        return float(self.weights[level])

    def get_normalized_weight_at_level(self, level: int) -> float:
        """
        Get normalized precision weight at specific hierarchical level.
        
        Args:
            level: Hierarchical level (0-indexed)
            
        Returns:
            Normalized precision weight at the specified level
        """
        normalized = self.normalized_weights
        return float(normalized[level])

    def scale_errors(self, errors: List[float]) -> List[float]:
        """
        Scale prediction errors using precision weights.
        
        Args:
            errors: List of prediction errors for each level
            
        Returns:
            List of precision-weighted errors
            
        Raises:
            ValueError: If error count doesn't match weight count
        """
        if len(errors) != len(self.weights):
            raise ValueError(
                f"Error count ({len(errors)}) doesn't match weight count ({len(self.weights)})"
            )
        
        normalized = self.normalized_weights
        return [error * weight for error, weight in zip(errors, normalized)]

    def adapt_weights(
        self, 
        prediction_errors: List[float], 
        adaptation_strength: Optional[float] = None
    ) -> 'PrecisionWeights':
        """
        Adapt precision weights based on prediction errors.
        
        Args:
            prediction_errors: Current prediction errors
            adaptation_strength: Override adaptation rate if provided
            
        Returns:
            New PrecisionWeights with adapted weights
            
        Raises:
            ValueError: If prediction errors don't match hierarchy levels
        """
        if len(prediction_errors) != len(self.weights):
            raise ValueError("Prediction errors must match hierarchy levels")
        
        adaptation_rate = adaptation_strength or self.adaptation_rate
        
        # Adapt weights inversely to prediction errors (lower error = higher precision)
        error_array = np.array(prediction_errors)
        inverse_errors = 1.0 / (error_array + 1e-6)  # Avoid division by zero
        
        # Update weights using exponential moving average
        new_weights = (
            (1.0 - adaptation_rate) * self.weights + 
            adaptation_rate * inverse_errors
        )
        
        return PrecisionWeights(
            weights=new_weights,
            normalization_method=self.normalization_method,
            temperature=self.temperature,
            adaptation_rate=self.adaptation_rate,
            metadata=self.metadata.copy()
        )

    def with_temperature(self, new_temperature: float) -> 'PrecisionWeights':
        """
        Create new precision weights with different temperature.
        
        Args:
            new_temperature: New temperature parameter
            
        Returns:
            New PrecisionWeights with updated temperature
        """
        return PrecisionWeights(
            weights=self.weights.copy(),
            normalization_method=self.normalization_method,
            temperature=new_temperature,
            adaptation_rate=self.adaptation_rate,
            metadata=self.metadata.copy()
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert precision weights to dictionary representation.
        
        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "weights": self.weights.tolist(),
            "normalized_weights": self.normalized_weights.tolist(),
            "normalization_method": self.normalization_method,
            "temperature": self.temperature,
            "adaptation_rate": self.adaptation_rate,
            "metadata": self.metadata,
            "hierarchy_levels": self.hierarchy_levels,
            "entropy": self.entropy,
            "attention_focus": self.attention_focus,
            "dominant_level": self.dominant_level
        }

    @classmethod
    def create_uniform(cls, hierarchy_levels: int, temperature: float = 1.0) -> 'PrecisionWeights':
        """
        Create uniform precision weights for all hierarchical levels.
        
        Args:
            hierarchy_levels: Number of hierarchical levels
            temperature: Temperature for softmax normalization
            
        Returns:
            PrecisionWeights with uniform distribution
        """
        weights = np.ones(hierarchy_levels)
        return cls(
            weights=weights,
            normalization_method="softmax",
            temperature=temperature,
            adaptation_rate=0.01,
            metadata={"initialization": "uniform"}
        )

    @classmethod
    def create_focused(
        cls, 
        hierarchy_levels: int, 
        focus_level: int, 
        focus_strength: float = 5.0
    ) -> 'PrecisionWeights':
        """
        Create precision weights focused on a specific hierarchical level.
        
        Args:
            hierarchy_levels: Number of hierarchical levels
            focus_level: Level to focus attention on (0-indexed)
            focus_strength: Strength of focus (higher = more focused)
            
        Returns:
            PrecisionWeights with focused distribution
        """
        if not (0 <= focus_level < hierarchy_levels):
            raise ValueError(f"Focus level {focus_level} out of bounds")
        
        weights = np.ones(hierarchy_levels)
        weights[focus_level] = focus_strength
        
        return cls(
            weights=weights,
            normalization_method="softmax",
            temperature=1.0,
            adaptation_rate=0.01,
            metadata={"initialization": "focused", "focus_level": focus_level}
        )