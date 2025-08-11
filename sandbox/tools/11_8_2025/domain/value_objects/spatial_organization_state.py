"""
Spatial Organization State Value Object.

Immutable representation of the spatial organization state within
the consciousness system, based on Self-Organizing Map principles
but expressed in domain-specific terminology.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SpatialOrganizationState:
    """
    Immutable representation of spatial organization in consciousness.
    
    Encapsulates the spatial arrangement and organization of conscious
    representations, providing metrics for consciousness level calculation.
    
    Domain concepts:
    - OptimalRepresentation: Best matching representational position (BMU)
    - StructuralCoherence: Topological preservation quality
    - OrganizationQuality: Overall spatial organization measure
    """
    
    optimal_representation: Tuple[int, int]  # BMU position in spatial map
    structural_coherence: float  # Topological preservation [0, 1]
    organization_quality: float  # Overall organization measure [0, 1]
    phenomenological_quality: float  # Quality of phenomenological experience [0, 1]
    temporal_consistency: float  # Temporal coherence measure [0, 1]
    map_dimensions: Tuple[int, int] = field(default=(10, 10))
    quantization_error: float = field(default=0.0)
    activation_history: List[Tuple[int, int]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate spatial organization state after initialization."""
        self._validate_optimal_representation()
        self._validate_quality_metrics()
        self._validate_map_dimensions()
        self._validate_activation_history()

    def _validate_optimal_representation(self) -> None:
        """Validate optimal representation coordinates."""
        if not (0 <= self.optimal_representation[0] < self.map_dimensions[0]):
            raise ValueError(f"Invalid optimal representation row: {self.optimal_representation[0]}")
        if not (0 <= self.optimal_representation[1] < self.map_dimensions[1]):
            raise ValueError(f"Invalid optimal representation col: {self.optimal_representation[1]}")

    def _validate_quality_metrics(self) -> None:
        """Validate all quality metrics are in valid range [0, 1]."""
        metrics = [
            ("structural_coherence", self.structural_coherence),
            ("organization_quality", self.organization_quality),
            ("phenomenological_quality", self.phenomenological_quality),
            ("temporal_consistency", self.temporal_consistency)
        ]
        
        for name, value in metrics:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    def _validate_map_dimensions(self) -> None:
        """Validate map dimensions are positive."""
        if self.map_dimensions[0] <= 0 or self.map_dimensions[1] <= 0:
            raise ValueError(f"Map dimensions must be positive: {self.map_dimensions}")

    def _validate_activation_history(self) -> None:
        """Validate activation history contains valid coordinates."""
        for i, (row, col) in enumerate(self.activation_history):
            if not (0 <= row < self.map_dimensions[0] and 0 <= col < self.map_dimensions[1]):
                raise ValueError(f"Invalid activation history entry {i}: ({row}, {col})")

    @property
    def is_well_organized(self) -> bool:
        """
        Determine if the spatial organization is well-structured.
        
        Returns:
            True if organization quality exceeds threshold
        """
        return self.organization_quality >= 0.7

    @property
    def spatial_stability(self) -> float:
        """
        Measure spatial stability based on activation history.
        
        Returns:
            Stability measure [0, 1] where 1.0 is perfectly stable
        """
        if len(self.activation_history) < 2:
            return 0.0
        
        # Calculate average distance between consecutive activations
        distances = []
        for i in range(1, len(self.activation_history)):
            prev_pos = self.activation_history[i-1]
            curr_pos = self.activation_history[i]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                             (curr_pos[1] - prev_pos[1])**2)
            distances.append(distance)
        
        if not distances:
            return 0.0
        
        avg_distance = np.mean(distances)
        max_possible_distance = np.sqrt(self.map_dimensions[0]**2 + self.map_dimensions[1]**2)
        
        # Invert distance to get stability (closer positions = more stable)
        return max(0.0, 1.0 - (avg_distance / max_possible_distance))

    @property
    def consciousness_contribution(self) -> float:
        """
        Calculate contribution to overall consciousness level.
        
        Integrates all spatial organization metrics into single measure.
        
        Returns:
            Consciousness contribution [0, 1]
        """
        # Weight different components for consciousness calculation
        weights = {
            'organization': 0.3,
            'coherence': 0.25,
            'phenomenological': 0.25,
            'temporal': 0.2
        }
        
        contribution = (
            weights['organization'] * self.organization_quality +
            weights['coherence'] * self.structural_coherence +
            weights['phenomenological'] * self.phenomenological_quality +
            weights['temporal'] * self.temporal_consistency
        )
        
        return min(contribution, 1.0)

    @property
    def representational_precision(self) -> float:
        """
        Measure precision of representational mapping.
        
        Based on quantization error (lower error = higher precision).
        
        Returns:
            Precision measure [0, 1]
        """
        # Convert quantization error to precision measure
        max_reasonable_error = 2.0  # Domain-specific threshold
        normalized_error = min(self.quantization_error / max_reasonable_error, 1.0)
        return 1.0 - normalized_error

    def with_updated_position(self, new_position: Tuple[int, int]) -> 'SpatialOrganizationState':
        """
        Create new state with updated optimal representation position.
        
        Args:
            new_position: New optimal representation coordinates
            
        Returns:
            New SpatialOrganizationState with updated position
        """
        # Update activation history
        new_history = self.activation_history.copy()
        new_history.append(new_position)
        
        # Keep only recent history (last 20 activations)
        if len(new_history) > 20:
            new_history = new_history[-20:]
        
        return SpatialOrganizationState(
            optimal_representation=new_position,
            structural_coherence=self.structural_coherence,
            organization_quality=self.organization_quality,
            phenomenological_quality=self.phenomenological_quality,
            temporal_consistency=self._calculate_updated_temporal_consistency(new_history),
            map_dimensions=self.map_dimensions,
            quantization_error=self.quantization_error,
            activation_history=new_history,
            timestamp=datetime.now(),
            metadata=self.metadata.copy()
        )

    def with_updated_quality_metrics(self, 
                                   structural_coherence: Optional[float] = None,
                                   organization_quality: Optional[float] = None,
                                   phenomenological_quality: Optional[float] = None,
                                   quantization_error: Optional[float] = None) -> 'SpatialOrganizationState':
        """
        Create new state with updated quality metrics.
        
        Args:
            structural_coherence: New structural coherence value
            organization_quality: New organization quality value
            phenomenological_quality: New phenomenological quality value
            quantization_error: New quantization error value
            
        Returns:
            New SpatialOrganizationState with updated metrics
        """
        return SpatialOrganizationState(
            optimal_representation=self.optimal_representation,
            structural_coherence=structural_coherence if structural_coherence is not None else self.structural_coherence,
            organization_quality=organization_quality if organization_quality is not None else self.organization_quality,
            phenomenological_quality=phenomenological_quality if phenomenological_quality is not None else self.phenomenological_quality,
            temporal_consistency=self.temporal_consistency,
            map_dimensions=self.map_dimensions,
            quantization_error=quantization_error if quantization_error is not None else self.quantization_error,
            activation_history=self.activation_history,
            timestamp=datetime.now(),
            metadata=self.metadata.copy()
        )

    def _calculate_updated_temporal_consistency(self, new_history: List[Tuple[int, int]]) -> float:
        """Calculate temporal consistency from activation history."""
        if len(new_history) < 2:
            return 0.0
        
        # Calculate consistency based on activation pattern stability
        distances = []
        for i in range(1, len(new_history)):
            prev_pos = new_history[i-1]
            curr_pos = new_history[i]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                             (curr_pos[1] - prev_pos[1])**2)
            distances.append(distance)
        
        if not distances:
            return 0.0
        
        # Lower average distance indicates higher temporal consistency
        avg_distance = np.mean(distances)
        max_distance = np.sqrt(self.map_dimensions[0]**2 + self.map_dimensions[1]**2)
        
        return max(0.0, 1.0 - (avg_distance / max_distance))

    def add_metadata(self, key: str, value: Any) -> 'SpatialOrganizationState':
        """
        Add metadata to the spatial organization state.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            New SpatialOrganizationState with added metadata
        """
        new_metadata = self.metadata.copy()
        new_metadata[key] = value
        
        return SpatialOrganizationState(
            optimal_representation=self.optimal_representation,
            structural_coherence=self.structural_coherence,
            organization_quality=self.organization_quality,
            phenomenological_quality=self.phenomenological_quality,
            temporal_consistency=self.temporal_consistency,
            map_dimensions=self.map_dimensions,
            quantization_error=self.quantization_error,
            activation_history=self.activation_history,
            timestamp=self.timestamp,
            metadata=new_metadata
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert spatial organization state to dictionary representation.
        
        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "optimal_representation": self.optimal_representation,
            "structural_coherence": self.structural_coherence,
            "organization_quality": self.organization_quality,
            "phenomenological_quality": self.phenomenological_quality,
            "temporal_consistency": self.temporal_consistency,
            "map_dimensions": self.map_dimensions,
            "quantization_error": self.quantization_error,
            "activation_history": self.activation_history,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "is_well_organized": self.is_well_organized,
            "spatial_stability": self.spatial_stability,
            "consciousness_contribution": self.consciousness_contribution,
            "representational_precision": self.representational_precision
        }

    @classmethod
    def create_initial(cls, map_dimensions: Tuple[int, int] = (10, 10)) -> 'SpatialOrganizationState':
        """
        Create initial spatial organization state.
        
        Args:
            map_dimensions: Dimensions of the spatial organization map
            
        Returns:
            Initial SpatialOrganizationState
        """
        # Start at center of map
        center_position = (map_dimensions[0] // 2, map_dimensions[1] // 2)
        
        return cls(
            optimal_representation=center_position,
            structural_coherence=0.5,  # Neutral starting coherence
            organization_quality=0.1,  # Low initial organization
            phenomenological_quality=0.0,  # No initial phenomenological quality
            temporal_consistency=0.0,  # No history yet
            map_dimensions=map_dimensions,
            quantization_error=1.0,  # High initial error
            activation_history=[center_position],
            metadata={"initialized": True}
        )

    @classmethod
    def create_well_organized(cls, map_dimensions: Tuple[int, int] = (10, 10)) -> 'SpatialOrganizationState':
        """
        Create well-organized spatial state for testing.
        
        Args:
            map_dimensions: Dimensions of the spatial organization map
            
        Returns:
            Well-organized SpatialOrganizationState
        """
        optimal_position = (map_dimensions[0] // 2, map_dimensions[1] // 2)
        
        return cls(
            optimal_representation=optimal_position,
            structural_coherence=0.85,
            organization_quality=0.8,
            phenomenological_quality=0.75,
            temporal_consistency=0.7,
            map_dimensions=map_dimensions,
            quantization_error=0.2,
            activation_history=[optimal_position] * 5,  # Stable history
            metadata={"well_organized": True}
        )