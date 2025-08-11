"""
Consciousness State Value Object.

Immutable value object representing the overall state of consciousness
in the enactive system. Encapsulates integrated information and
phenomenological aspects of the conscious experience.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import numpy.typing as npt
from .phi_value import PhiValue
from .prediction_state import PredictionState
from .probability_distribution import ProbabilityDistribution
from .spatial_organization_state import SpatialOrganizationState


@dataclass(frozen=True)
class ConsciousnessState:
    """
    Immutable representation of the consciousness state.
    
    This value object encapsulates all relevant aspects of the current
    conscious state including integrated information (Φ), prediction
    quality, uncertainty levels, and temporal dynamics.
    
    Follows Value Object pattern principles:
    - Immutability (frozen=True)
    - Value-based equality
    - Rich domain behavior
    - Self-validation
    """
    
    phi_value: PhiValue
    prediction_state: PredictionState
    uncertainty_distribution: ProbabilityDistribution
    spatial_organization: SpatialOrganizationState
    timestamp: datetime = field(default_factory=datetime.now)
    metacognitive_confidence: float = field(default=0.0)
    attention_weights: Optional[npt.NDArray] = field(default=None)
    phenomenological_markers: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """
        Validate the consciousness state after initialization.
        
        Raises:
            ValueError: If any validation fails
        """
        self._validate_metacognitive_confidence()
        self._validate_attention_weights()
        self._validate_consistency()

    def _validate_metacognitive_confidence(self) -> None:
        """Validate metacognitive confidence is in valid range [0, 1]."""
        if not (0.0 <= self.metacognitive_confidence <= 1.0):
            raise ValueError(
                f"Metacognitive confidence must be in [0, 1], got {self.metacognitive_confidence}"
            )

    def _validate_attention_weights(self) -> None:
        """Validate attention weights sum to 1 if provided."""
        if self.attention_weights is not None:
            if not isinstance(self.attention_weights, np.ndarray):
                raise ValueError("Attention weights must be numpy array")
            
            weights_sum = float(self.attention_weights.sum())
            if not (0.99 <= weights_sum <= 1.01):  # Allow small numerical errors
                raise ValueError(
                    f"Attention weights must sum to 1.0, got {weights_sum}"
                )

    def _validate_consistency(self) -> None:
        """Validate internal consistency between components."""
        # Check that prediction state timestamp is consistent
        time_diff = abs((self.timestamp - self.prediction_state.timestamp).total_seconds())
        if time_diff > 1.0:  # Allow 1 second tolerance
            raise ValueError("Prediction state timestamp inconsistent with consciousness state")

    @property
    def is_conscious(self) -> bool:
        """
        Determine if the current state represents genuine consciousness.
        
        Based on integrated information theory and metacognitive awareness.
        
        Returns:
            True if the system exhibits conscious characteristics
        """
        return (
            self.phi_value.value > 0.0 and
            self.metacognitive_confidence > 0.1 and
            self.prediction_state.total_error < 2.0  # More lenient prediction quality threshold
        )

    @property
    def consciousness_level(self) -> float:
        """
        Compute a scalar consciousness level [0, 1].
        
        Integrates multiple indicators into a single measure.
        
        Returns:
            Scalar value representing consciousness intensity
        """
        if not self.is_conscious:
            return 0.0
            
        # Weight different components including spatial organization
        phi_component = min(self.phi_value.value / 10.0, 1.0)  # Normalize Φ
        meta_component = self.metacognitive_confidence
        prediction_component = max(0.0, 1.0 - self.prediction_state.total_error)
        uncertainty_component = 1.0 - self.uncertainty_distribution.entropy_normalized
        spatial_component = self.spatial_organization.consciousness_contribution
        
        # Weighted combination with spatial organization
        consciousness_level = (
            0.3 * phi_component +
            0.25 * meta_component +
            0.2 * prediction_component +
            0.1 * uncertainty_component +
            0.15 * spatial_component
        )
        
        return min(consciousness_level, 1.0)

    @property
    def attention_focus_strength(self) -> float:
        """
        Measure attention focus based on weight distribution.
        
        Returns:
            Scalar indicating how focused attention is (higher = more focused)
        """
        if self.attention_weights is None:
            return 0.0
            
        # Calculate entropy of attention distribution (lower entropy = more focused)
        attention_entropy = -sum(
            w * np.log(w + 1e-10) for w in self.attention_weights if w > 0
        )
        max_entropy = np.log(len(self.attention_weights))
        
        # Convert to focus strength (0 = completely unfocused, 1 = completely focused)
        return 1.0 - (attention_entropy / max_entropy) if max_entropy > 0 else 0.0

    def with_updated_phi(self, new_phi: PhiValue) -> 'ConsciousnessState':
        """
        Create new consciousness state with updated Φ value.
        
        Args:
            new_phi: New integrated information value
            
        Returns:
            New ConsciousnessState instance with updated Φ
        """
        return ConsciousnessState(
            phi_value=new_phi,
            prediction_state=self.prediction_state,
            uncertainty_distribution=self.uncertainty_distribution,
            spatial_organization=self.spatial_organization,
            timestamp=datetime.now(),
            metacognitive_confidence=self.metacognitive_confidence,
            attention_weights=self.attention_weights,
            phenomenological_markers=self.phenomenological_markers.copy()
        )

    def with_updated_prediction(self, new_prediction_state: PredictionState) -> 'ConsciousnessState':
        """
        Create new consciousness state with updated prediction state.
        
        Args:
            new_prediction_state: New prediction state
            
        Returns:
            New ConsciousnessState instance with updated prediction
        """
        return ConsciousnessState(
            phi_value=self.phi_value,
            prediction_state=new_prediction_state,
            uncertainty_distribution=self.uncertainty_distribution,
            spatial_organization=self.spatial_organization,
            timestamp=datetime.now(),
            metacognitive_confidence=self.metacognitive_confidence,
            attention_weights=self.attention_weights,
            phenomenological_markers=self.phenomenological_markers.copy()
        )

    def with_updated_spatial_organization(self, new_spatial_organization: SpatialOrganizationState) -> 'ConsciousnessState':
        """
        Create new consciousness state with updated spatial organization.
        
        Args:
            new_spatial_organization: New spatial organization state
            
        Returns:
            New ConsciousnessState instance with updated spatial organization
        """
        return ConsciousnessState(
            phi_value=self.phi_value,
            prediction_state=self.prediction_state,
            uncertainty_distribution=self.uncertainty_distribution,
            spatial_organization=new_spatial_organization,
            timestamp=datetime.now(),
            metacognitive_confidence=self.metacognitive_confidence,
            attention_weights=self.attention_weights,
            phenomenological_markers=self.phenomenological_markers.copy()
        )

    def add_phenomenological_marker(self, key: str, value: Any) -> 'ConsciousnessState':
        """
        Add a phenomenological marker to the consciousness state.
        
        Args:
            key: Marker identifier
            value: Marker value
            
        Returns:
            New ConsciousnessState with added marker
        """
        new_markers = self.phenomenological_markers.copy()
        new_markers[key] = value
        
        return ConsciousnessState(
            phi_value=self.phi_value,
            prediction_state=self.prediction_state,
            uncertainty_distribution=self.uncertainty_distribution,
            spatial_organization=self.spatial_organization,
            timestamp=self.timestamp,
            metacognitive_confidence=self.metacognitive_confidence,
            attention_weights=self.attention_weights,
            phenomenological_markers=new_markers
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert consciousness state to dictionary representation.
        
        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "phi_value": self.phi_value.to_dict(),
            "prediction_state": self.prediction_state.to_dict(),
            "uncertainty_distribution": self.uncertainty_distribution.to_dict(),
            "spatial_organization": self.spatial_organization.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "metacognitive_confidence": self.metacognitive_confidence,
            "attention_weights": self.attention_weights.tolist() if self.attention_weights is not None else None,
            "phenomenological_markers": self.phenomenological_markers,
            "is_conscious": self.is_conscious,
            "consciousness_level": self.consciousness_level,
            "attention_focus_strength": self.attention_focus_strength
        }

    @classmethod
    def create_minimal_consciousness(cls) -> 'ConsciousnessState':
        """
        Create a minimal consciousness state for testing or initialization.
        
        Returns:
            ConsciousnessState with minimal conscious characteristics
        """
        from .phi_value import PhiValue
        from .prediction_state import PredictionState
        from .probability_distribution import ProbabilityDistribution
        
        return cls(
            phi_value=PhiValue(0.5, complexity=1.0, integration=0.5),
            prediction_state=PredictionState.create_empty(hierarchy_levels=3),
            uncertainty_distribution=ProbabilityDistribution.uniform(10),
            spatial_organization=SpatialOrganizationState.create_initial(),
            metacognitive_confidence=0.2,
            attention_weights=None,
            phenomenological_markers={"minimal": True}
        )