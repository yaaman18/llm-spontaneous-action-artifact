"""
Probability Distribution Value Object.

Immutable representation of probability distributions used in Bayesian
inference and uncertainty quantification within the consciousness system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
import numpy.typing as npt
import math


@dataclass(frozen=True)
class ProbabilityDistribution:
    """
    Immutable representation of a probability distribution.
    
    Encapsulates probability values, uncertainty measures, and
    distribution characteristics for Bayesian reasoning.
    """
    
    probabilities: npt.NDArray
    support: Optional[npt.NDArray] = field(default=None)
    distribution_type: str = field(default="categorical")
    parameters: Dict[str, float] = field(default_factory=dict)
    confidence_level: float = field(default=0.95)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate probability distribution."""
        self._validate_probabilities()
        self._validate_support()
        self._validate_confidence_level()
        self._validate_consistency()

    def _validate_probabilities(self) -> None:
        """Validate probability array."""
        if not isinstance(self.probabilities, np.ndarray):
            raise ValueError("Probabilities must be numpy array")
        if self.probabilities.ndim != 1:
            raise ValueError("Probabilities must be 1-dimensional")
        if len(self.probabilities) == 0:
            raise ValueError("Probabilities array cannot be empty")
        if np.any(self.probabilities < 0):
            raise ValueError("All probabilities must be non-negative")
        
        # Check normalization (allow small numerical errors)
        prob_sum = float(np.sum(self.probabilities))
        if not (0.99 <= prob_sum <= 1.01):
            raise ValueError(f"Probabilities must sum to 1.0, got {prob_sum}")

    def _validate_support(self) -> None:
        """Validate support array if provided."""
        if self.support is not None:
            if not isinstance(self.support, np.ndarray):
                raise ValueError("Support must be numpy array")
            if len(self.support) != len(self.probabilities):
                raise ValueError("Support length must match probabilities length")

    def _validate_confidence_level(self) -> None:
        """Validate confidence level."""
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("Confidence level must be in (0, 1)")

    def _validate_consistency(self) -> None:
        """Validate internal consistency."""
        valid_types = {"categorical", "gaussian", "beta", "gamma", "exponential", "uniform"}
        if self.distribution_type not in valid_types:
            raise ValueError(f"Invalid distribution type: {self.distribution_type}")

    @property
    def size(self) -> int:
        """Number of elements in the distribution."""
        return len(self.probabilities)

    @property
    def entropy(self) -> float:
        """Shannon entropy of the distribution."""
        return float(-np.sum(self.probabilities * np.log(self.probabilities + 1e-10)))

    @property
    def entropy_normalized(self) -> float:
        """Normalized entropy [0, 1] where 1 is maximum entropy."""
        max_entropy = math.log(len(self.probabilities))
        return self.entropy / max_entropy if max_entropy > 0 else 0.0

    @property
    def max_probability(self) -> float:
        """Maximum probability value."""
        return float(np.max(self.probabilities))

    @property
    def min_probability(self) -> float:
        """Minimum probability value."""
        return float(np.min(self.probabilities))

    @property
    def mode_index(self) -> int:
        """Index of the mode (highest probability)."""
        return int(np.argmax(self.probabilities))

    @property
    def mode_value(self) -> Optional[float]:
        """Value at the mode (requires support)."""
        if self.support is None:
            return None
        return float(self.support[self.mode_index])

    @property
    def mean(self) -> Optional[float]:
        """Mean of the distribution (requires support)."""
        if self.support is None:
            return None
        return float(np.sum(self.probabilities * self.support))

    @property
    def variance(self) -> Optional[float]:
        """Variance of the distribution (requires support)."""
        if self.support is None or self.mean is None:
            return None
        
        mean_val = self.mean
        return float(np.sum(self.probabilities * (self.support - mean_val)**2))

    @property
    def standard_deviation(self) -> Optional[float]:
        """Standard deviation of the distribution."""
        if self.variance is None:
            return None
        return math.sqrt(self.variance)

    @property
    def is_uniform(self) -> bool:
        """Check if distribution is approximately uniform."""
        uniform_prob = 1.0 / len(self.probabilities)
        return bool(np.allclose(self.probabilities, uniform_prob, atol=1e-6))

    @property
    def is_deterministic(self) -> bool:
        """Check if distribution is deterministic (one probability = 1)."""
        return bool(np.any(self.probabilities > 0.99))

    @property
    def concentration(self) -> float:
        """Measure of concentration (inverse of entropy)."""
        return 1.0 - self.entropy_normalized

    def get_probability_at_index(self, index: int) -> float:
        """
        Get probability at specific index.
        
        Args:
            index: Index in the probability array
            
        Returns:
            Probability at the given index
            
        Raises:
            IndexError: If index is out of bounds
        """
        if not (0 <= index < len(self.probabilities)):
            raise IndexError(f"Index {index} out of bounds for {len(self.probabilities)} elements")
        
        return float(self.probabilities[index])

    def get_probability_at_value(self, value: float, tolerance: float = 1e-6) -> float:
        """
        Get probability at specific value (requires support).
        
        Args:
            value: Value to look up
            tolerance: Tolerance for value matching
            
        Returns:
            Probability at the given value
            
        Raises:
            ValueError: If support is not available or value not found
        """
        if self.support is None:
            raise ValueError("Support required for value lookup")
        
        # Find closest value within tolerance
        distances = np.abs(self.support - value)
        closest_idx = int(np.argmin(distances))
        
        if distances[closest_idx] > tolerance:
            raise ValueError(f"Value {value} not found within tolerance {tolerance}")
        
        return float(self.probabilities[closest_idx])

    def sample(self, n_samples: int = 1, random_state: Optional[int] = None) -> npt.NDArray:
        """
        Sample from the distribution.
        
        Args:
            n_samples: Number of samples to draw
            random_state: Random seed for reproducibility
            
        Returns:
            Array of sampled indices (or values if support is available)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        indices = np.random.choice(
            len(self.probabilities),
            size=n_samples,
            p=self.probabilities
        )
        
        if self.support is not None:
            return self.support[indices]
        else:
            return indices

    def compute_credible_interval(self) -> tuple[Optional[float], Optional[float]]:
        """
        Compute credible interval for the distribution.
        
        Returns:
            Tuple of (lower_bound, upper_bound) or (None, None) if no support
        """
        if self.support is None:
            return (None, None)
        
        # Sort indices by support values
        sorted_indices = np.argsort(self.support)
        sorted_probs = self.probabilities[sorted_indices]
        sorted_support = self.support[sorted_indices]
        
        # Compute cumulative distribution
        cumulative = np.cumsum(sorted_probs)
        
        # Find credible interval bounds
        alpha = 1.0 - self.confidence_level
        lower_bound_idx = np.searchsorted(cumulative, alpha / 2)
        upper_bound_idx = np.searchsorted(cumulative, 1.0 - alpha / 2)
        
        # Ensure bounds are within array limits
        lower_bound_idx = min(lower_bound_idx, len(sorted_support) - 1)
        upper_bound_idx = min(upper_bound_idx, len(sorted_support) - 1)
        
        return (
            float(sorted_support[lower_bound_idx]),
            float(sorted_support[upper_bound_idx])
        )

    def kl_divergence(self, other: 'ProbabilityDistribution') -> float:
        """
        Compute KL divergence between this and another distribution.
        
        Args:
            other: Another probability distribution
            
        Returns:
            KL divergence D(self || other)
            
        Raises:
            ValueError: If distributions have different sizes
        """
        if len(self.probabilities) != len(other.probabilities):
            raise ValueError("Distributions must have same size for KL divergence")
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        p = self.probabilities + epsilon
        q = other.probabilities + epsilon
        
        return float(np.sum(p * np.log(p / q)))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert distribution to dictionary representation.
        
        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "probabilities": self.probabilities.tolist(),
            "support": self.support.tolist() if self.support is not None else None,
            "distribution_type": self.distribution_type,
            "parameters": self.parameters,
            "confidence_level": self.confidence_level,
            "metadata": self.metadata,
            "size": self.size,
            "entropy": self.entropy,
            "entropy_normalized": self.entropy_normalized,
            "max_probability": self.max_probability,
            "mode_index": self.mode_index,
            "mode_value": self.mode_value,
            "mean": self.mean,
            "variance": self.variance,
            "is_uniform": self.is_uniform,
            "is_deterministic": self.is_deterministic,
            "concentration": self.concentration
        }

    @classmethod
    def uniform(cls, size: int) -> 'ProbabilityDistribution':
        """
        Create uniform distribution.
        
        Args:
            size: Number of elements
            
        Returns:
            Uniform probability distribution
        """
        probabilities = np.ones(size) / size
        return cls(
            probabilities=probabilities,
            distribution_type="uniform",
            metadata={"initialization": "uniform"}
        )

    @classmethod
    def categorical(cls, probabilities: List[float]) -> 'ProbabilityDistribution':
        """
        Create categorical distribution from probability list.
        
        Args:
            probabilities: List of probability values
            
        Returns:
            Categorical probability distribution
        """
        prob_array = np.array(probabilities)
        # Normalize to ensure sum = 1
        prob_array = prob_array / np.sum(prob_array)
        
        return cls(
            probabilities=prob_array,
            distribution_type="categorical",
            metadata={"initialization": "categorical"}
        )

    @classmethod
    def normal(cls, mean: float, variance: float, n_points: int = 50) -> 'ProbabilityDistribution':
        """
        Create normal (Gaussian) distribution.
        
        Args:
            mean: Mean of the distribution
            variance: Variance of the distribution
            n_points: Number of points to discretize the distribution
            
        Returns:
            Normal probability distribution (discretized)
        """
        if variance <= 0:
            raise ValueError("Variance must be positive")
        if n_points <= 0:
            raise ValueError("Number of points must be positive")
            
        std_dev = math.sqrt(variance)
        # Create support points around mean ± 3 standard deviations
        x_min = mean - 3 * std_dev
        x_max = mean + 3 * std_dev
        support = np.linspace(x_min, x_max, n_points)
        
        # Compute probability density
        pdf = (1 / (std_dev * math.sqrt(2 * math.pi))) * np.exp(
            -0.5 * ((support - mean) / std_dev) ** 2
        )
        
        # Normalize to create probabilities (discrete approximation)
        probabilities = pdf / np.sum(pdf)
        
        return cls(
            probabilities=probabilities,
            support=support,
            distribution_type="gaussian",
            parameters={"mean": mean, "variance": variance, "std_dev": std_dev},
            metadata={"initialization": "normal", "n_points": n_points}
        )

    @classmethod
    def from_samples(cls, samples: npt.NDArray, bins: Optional[int] = None) -> 'ProbabilityDistribution':
        """
        Create distribution from sample data.
        
        Args:
            samples: Sample data array
            bins: Number of bins for histogram (if None, uses unique values)
            
        Returns:
            Probability distribution estimated from samples
        """
        if bins is None:
            # Use unique values as bins
            unique_values, counts = np.unique(samples, return_counts=True)
            probabilities = counts / len(samples)
            support = unique_values
        else:
            # Create histogram
            counts, bin_edges = np.histogram(samples, bins=bins)
            probabilities = counts / len(samples)
            # Use bin centers as support
            support = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return cls(
            probabilities=probabilities,
            support=support,
            distribution_type="empirical",
            metadata={"initialization": "from_samples", "n_samples": len(samples)}
        )