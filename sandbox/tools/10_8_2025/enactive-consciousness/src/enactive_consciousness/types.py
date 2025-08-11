"""Type definitions and protocols for the enactive consciousness framework.

This module provides comprehensive type annotations following Python 3.9+ specifications,
with special attention to JAX array types and Equinox module integration.
"""

from __future__ import annotations

import abc
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)
from dataclasses import dataclass
from enum import Enum
import logging

import jax
import jax.numpy as jnp
import equinox as eqx
from pydantic import BaseModel, ConfigDict, Field, validator

# Configure module logger
logger = logging.getLogger(__name__)

# Type variables for generic programming
T = TypeVar('T')
StateT = TypeVar('StateT')
ObservationT = TypeVar('ObservationT')
ActionT = TypeVar('ActionT')

# JAX-specific type aliases
Array = jax.Array
ArrayLike = Union[Array, jnp.ndarray, List[float], Tuple[float, ...]]
PRNGKey = jax.random.PRNGKey
PyTree = Any  # JAX PyTree type

# Dimensions and shapes
Dim = int
Shape = Tuple[Dim, ...]

# Time and temporal structures
TimeStep = float
TemporalWindow = int


class ConsciousnessLevel(Enum):
    """Levels of consciousness processing."""
    MINIMAL = "minimal"
    BASIC = "basic"
    REFLECTIVE = "reflective"
    META_COGNITIVE = "meta_cognitive"


class CouplingStrength(Enum):
    """Strength levels for structural coupling."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    SYMBIOTIC = "symbiotic"


# Pydantic models for configuration and validation
class FrameworkConfig(BaseModel):
    """Configuration for the enactive consciousness framework."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='forbid',
    )
    
    # Temporal configuration
    retention_depth: int = Field(default=10, ge=1, le=100)
    protention_horizon: int = Field(default=5, ge=1, le=50)
    primal_impression_width: float = Field(default=0.1, gt=0.0, le=1.0)
    temporal_synthesis_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    
    # Body schema configuration
    proprioceptive_dim: int = Field(default=64, ge=16, le=512)
    motor_dim: int = Field(default=32, ge=8, le=256)
    body_map_size: Tuple[int, int] = Field(default=(20, 20))
    
    # Coupling configuration  
    coupling_strength: CouplingStrength = Field(default=CouplingStrength.MODERATE)
    environmental_dim: int = Field(default=128, ge=32, le=1024)
    
    # Consciousness configuration
    consciousness_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    integration_method: str = Field(default="weighted_sum")
    
    @validator('body_map_size')
    def validate_map_size(cls, v):
        if len(v) != 2 or any(dim <= 0 for dim in v):
            raise ValueError("body_map_size must be tuple of two positive integers")
        return v


@dataclass(frozen=True, slots=True)
class TemporalMoment:
    """Immutable temporal moment in phenomenological time."""
    
    timestamp: TimeStep
    retention: Array
    present_moment: Array
    protention: Array
    synthesis_weights: Array
    
    def __post_init__(self):
        """Validate temporal moment consistency."""
        if self.timestamp < 0:
            raise ValueError("Timestamp must be non-negative")
        
        # Ensure all arrays have compatible shapes
        shapes = [arr.shape for arr in [self.retention, self.present_moment, 
                                       self.protention, self.synthesis_weights]]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All arrays must have compatible shapes")


@dataclass(frozen=True, slots=True) 
class BodyState:
    """Current embodied state representation."""
    
    proprioception: Array
    motor_intention: Array
    boundary_signal: Array
    schema_confidence: float
    
    def __post_init__(self):
        """Validate body state consistency."""
        self._validate_confidence_range()
        self._validate_array_consistency()
    
    def _validate_confidence_range(self) -> None:
        """Extract method: Validate confidence is in valid range."""
        if not (0.0 <= self.schema_confidence <= 1.0):
            raise ValueError("schema_confidence must be in [0, 1]")
    
    def _validate_array_consistency(self) -> None:
        """Extract method: Validate array shapes and content."""
        arrays = [self.proprioception, self.motor_intention, self.boundary_signal]
        for i, arr in enumerate(arrays):
            if not jnp.all(jnp.isfinite(arr)):
                array_names = ['proprioception', 'motor_intention', 'boundary_signal']
                raise ValueError(f"{array_names[i]} contains non-finite values")


@dataclass(frozen=True, slots=True)
class CouplingState:
    """Structural coupling state between agent and environment."""
    
    agent_state: Array
    environmental_state: Array
    coupling_strength: float
    perturbation_history: Array
    stability_metric: float
    
    def __post_init__(self):
        """Validate coupling state consistency.""" 
        if not (0.0 <= self.coupling_strength <= 1.0):
            raise ValueError("coupling_strength must be in [0, 1]")
        if not (0.0 <= self.stability_metric <= 1.0):
            raise ValueError("stability_metric must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AffordanceVector:
    """Detected affordances in the environment."""
    
    affordance_strengths: Array
    action_potentials: Array
    contextual_relevance: Array
    detection_confidence: float
    
    def __post_init__(self):
        """Validate affordance vector consistency."""
        if not (0.0 <= self.detection_confidence <= 1.0):
            raise ValueError("detection_confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class MeaningStructure:
    """Emergent meaning structure from sense-making."""
    
    semantic_content: Array
    coherence_measure: float
    relevance_weight: Array
    emergence_timestamp: TimeStep
    
    def __post_init__(self):
        """Validate meaning structure consistency."""
        if not (0.0 <= self.coherence_measure <= 1.0):
            raise ValueError("coherence_measure must be in [0, 1]")
        if self.emergence_timestamp < 0:
            raise ValueError("emergence_timestamp must be non-negative")


@dataclass(frozen=True, slots=True)
class ExperienceRetentionState:
    """State of experience retention system."""
    
    retained_experiences_count: int
    sedimentation_depth: float
    temporal_continuity: float
    retention_quality: float
    last_retention_timestamp: TimeStep
    
    def __post_init__(self):
        """Validate experience retention state consistency."""
        if self.retained_experiences_count < 0:
            raise ValueError("retained_experiences_count must be non-negative")
        if not (0.0 <= self.retention_quality <= 1.0):
            raise ValueError("retention_quality must be in [0, 1]")
        if not (0.0 <= self.temporal_continuity <= 1.0):
            raise ValueError("temporal_continuity must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class EnactiveCouplingState:
    """Enhanced coupling state with enactive dynamics."""
    
    coupling_state: CouplingState
    circular_causality_strength: float
    meaning_emergence_level: float
    self_reference_autonomy: float
    enactive_quality_score: float
    
    def __post_init__(self):
        """Validate enactive coupling state consistency."""
        if not (0.0 <= self.circular_causality_strength <= 1.0):
            raise ValueError("circular_causality_strength must be in [0, 1]")
        if not (0.0 <= self.meaning_emergence_level <= 1.0):
            raise ValueError("meaning_emergence_level must be in [0, 1]")
        if not (0.0 <= self.self_reference_autonomy <= 1.0):
            raise ValueError("self_reference_autonomy must be in [0, 1]")
        if not (0.0 <= self.enactive_quality_score <= 1.0):
            raise ValueError("enactive_quality_score must be in [0, 1]")


# Protocol definitions for component interfaces
@runtime_checkable
class TemporalProcessor(Protocol):
    """Protocol for temporal consciousness processing."""
    
    def synthesize_temporal_moment(
        self,
        retention: Array,
        present_impression: Array,
        protention: Array,
    ) -> TemporalMoment:
        """Synthesize temporal moment from phenomenological components."""
        ...
    
    def update_temporal_flow(
        self,
        current_moment: TemporalMoment,
        new_impression: Array,
    ) -> TemporalMoment:
        """Update temporal flow with new impression."""
        ...


@runtime_checkable
class EmbodimentProcessor(Protocol):
    """Protocol for embodied processing."""
    
    def integrate_body_schema(
        self,
        proprioceptive_input: Array,
        motor_prediction: Array,
        tactile_feedback: Array,
    ) -> BodyState:
        """Integrate body schema from multi-modal input."""
        ...
    
    def generate_motor_intention(
        self,
        current_state: BodyState,
        goal_state: Array,
    ) -> Array:
        """Generate motor intention from current and goal states."""
        ...


@runtime_checkable  
class CouplingProcessor(Protocol):
    """Protocol for structural coupling processing."""
    
    def compute_coupling_dynamics(
        self,
        agent_state: Array,
        environmental_perturbation: Array,
    ) -> CouplingState:
        """Compute structural coupling dynamics."""
        ...
    
    def assess_coupling_stability(
        self,
        coupling_history: List[CouplingState],
    ) -> float:
        """Assess stability of coupling over time."""
        ...


@runtime_checkable
class AffordanceProcessor(Protocol):
    """Protocol for affordance perception."""
    
    def detect_affordances(
        self,
        perceptual_state: Array,
        action_capabilities: Array,
        contextual_factors: Array,
    ) -> AffordanceVector:
        """Detect affordances in current context."""
        ...
    
    def evaluate_affordance_relevance(
        self,
        affordances: AffordanceVector,
        current_intentions: Array,
    ) -> Array:
        """Evaluate relevance of detected affordances."""
        ...


@runtime_checkable
class SenseMakingProcessor(Protocol):
    """Protocol for sense-making processing."""
    
    def construct_meaning(
        self,
        experiential_input: Array,
        contextual_background: Array,
        current_concerns: Array,
    ) -> MeaningStructure:
        """Construct meaning from experiential input."""
        ...
    
    def evaluate_meaning_coherence(
        self,
        meaning_structure: MeaningStructure,
        historical_meanings: List[MeaningStructure],
    ) -> float:
        """Evaluate coherence of constructed meaning."""
        ...


# Composite protocols for higher-level functionality
@runtime_checkable
class EnactiveCouplingProcessor(Protocol):
    """Protocol for enactive coupling and circular causality processing."""
    
    def compute_coupling_dynamics(
        self,
        agent_state: Array,
        environmental_perturbation: Array,
        timestamp: Optional[TimeStep] = None,
    ) -> CouplingState:
        """Compute enactive coupling dynamics with circular causality."""
        ...
    
    def generate_emergent_meaning(
        self,
        coupling_state: CouplingState,
        coupling_history: Optional[Array] = None,
        timestamp: Optional[TimeStep] = None,
    ) -> MeaningStructure:
        """Generate emergent meaning from coupling dynamics."""
        ...
    
    def assess_enactive_quality(
        self,
        coupling_state: CouplingState,
    ) -> Dict[str, float]:
        """Assess quality of enactive processing."""
        ...


@runtime_checkable
class ExperienceRetentionProcessor(Protocol):
    """Protocol for experience retention and sedimentation processing."""
    
    def retain_experience(
        self,
        experience_content: Array,
        temporal_moment: TemporalMoment,
        meaning_structure: Optional[MeaningStructure] = None,
        timestamp: Optional[TimeStep] = None,
    ) -> 'ExperienceRetentionProcessor':
        """Retain experience across sedimentation, associative, and temporal systems."""
        ...
    
    def recall_related_experiences(
        self,
        query: Array,
        recall_mode: str = "associative",
        max_recalls: int = 5,
        timestamp: Optional[TimeStep] = None,
    ) -> List[Tuple[Any, float]]:
        """Recall experiences related to query using specified mode."""
        ...
    
    def assess_retention_quality(self) -> Dict[str, float]:
        """Assess quality of experience retention system."""
        ...


@runtime_checkable
class ConsciousnessIntegrator(Protocol):
    """Protocol for consciousness integration."""
    
    def integrate_conscious_moment(
        self,
        temporal_moment: TemporalMoment,
        body_state: BodyState,
        coupling_state: CouplingState,
        affordances: AffordanceVector,
        meaning_structure: MeaningStructure,
    ) -> Tuple[Array, Dict[str, Any]]:
        """Integrate all components into conscious moment."""
        ...
    
    def assess_consciousness_level(
        self,
        integrated_state: Array,
    ) -> ConsciousnessLevel:
        """Assess current level of consciousness."""
        ...


# Utility types for advanced functionality
class PerformanceMetrics(BaseModel):
    """Performance metrics for system evaluation."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    temporal_coherence: float = Field(ge=0.0, le=1.0)
    embodiment_stability: float = Field(ge=0.0, le=1.0)  
    coupling_effectiveness: float = Field(ge=0.0, le=1.0)
    affordance_detection_accuracy: float = Field(ge=0.0, le=1.0)
    meaning_construction_quality: float = Field(ge=0.0, le=1.0)
    overall_consciousness_score: float = Field(ge=0.0, le=1.0)
    
    processing_time_ms: float = Field(gt=0.0)
    memory_usage_mb: float = Field(gt=0.0)
    
    @property
    def _metric_weights(self) -> list[float]:
        """Replace temp with query: Get metric weights."""
        return [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]  # Consciousness gets extra weight
    
    @property 
    def _metric_scores(self) -> list[float]:
        """Replace temp with query: Get all metric scores."""
        return [
            self.temporal_coherence,
            self.embodiment_stability,
            self.coupling_effectiveness,
            self.affordance_detection_accuracy,
            self.meaning_construction_quality,
            self.overall_consciousness_score,
        ]
    
    def compute_overall_score(self) -> float:
        """Compute weighted overall performance score."""
        return sum(w * s for w, s in zip(self._metric_weights, self._metric_scores))
    
    def get_performance_summary(self) -> dict[str, float]:
        """Extract method: Get comprehensive performance summary."""
        return {
            'overall_score': self.compute_overall_score(),
            'cognitive_subscore': sum(w * s for w, s in zip(self._metric_weights[:3], self._metric_scores[:3])),
            'perceptual_subscore': sum(w * s for w, s in zip(self._metric_weights[3:5], self._metric_scores[3:5])),
            'consciousness_subscore': self._metric_weights[5] * self._metric_scores[5],
            'efficiency_ratio': self.compute_overall_score() / (self.processing_time_ms / 1000.0),
        }


# Exception types for domain-specific error handling
class EnactiveConsciousnessError(Exception):
    """Base exception for enactive consciousness framework."""
    pass


class TemporalSynthesisError(EnactiveConsciousnessError):
    """Error in temporal synthesis processing.""" 
    pass


class EmbodimentError(EnactiveConsciousnessError):
    """Error in embodiment processing."""
    pass


class CouplingError(EnactiveConsciousnessError):
    """Error in structural coupling."""
    pass


class AffordancePerceptionError(EnactiveConsciousnessError):
    """Error in affordance perception."""
    pass


class SenseMakingError(EnactiveConsciousnessError):
    """Error in sense-making processing."""
    pass


class ConsciousnessIntegrationError(EnactiveConsciousnessError):
    """Error in consciousness integration."""
    pass


# Type guards and validation functions
def is_valid_array_shape(arr: Array, expected_shape: Shape) -> bool:
    """Type guard for array shape validation."""
    return isinstance(arr, (jnp.ndarray, Array)) and arr.shape == expected_shape


def validate_temporal_consistency(
    retention: Array, 
    present: Array, 
    protention: Array
) -> bool:
    """Validate temporal moment consistency."""
    return _check_dimensional_consistency(retention, present, protention) and \
           _check_shape_consistency(retention, present, protention)


def _check_dimensional_consistency(retention: Array, present: Array, protention: Array) -> bool:
    """Extract method: Check dimensional consistency."""
    return retention.ndim == present.ndim == protention.ndim


def _check_shape_consistency(retention: Array, present: Array, protention: Array) -> bool:
    """Extract method: Check shape consistency."""
    return retention.shape[-1] == present.shape[-1] == protention.shape[-1]


def validate_consciousness_state(state: Array) -> bool:
    """Validate consciousness state array."""
    return (_is_valid_array_type(state) and 
            _has_valid_dimensions(state) and
            _has_finite_values(state) and
            _has_no_nan_values(state))


def _is_valid_array_type(state: Array) -> bool:
    """Extract method: Check if state is valid array type."""
    return isinstance(state, (jnp.ndarray, Array))


def _has_valid_dimensions(state: Array) -> bool:
    """Extract method: Check if state has valid dimensions."""
    return state.ndim >= 1


def _has_finite_values(state: Array) -> bool:
    """Extract method: Check if all values are finite."""
    return jnp.all(jnp.isfinite(state))


def _has_no_nan_values(state: Array) -> bool:
    """Extract method: Check if no values are NaN."""
    return not jnp.any(jnp.isnan(state))


# Factory functions for creating validated instances
def create_framework_config(**kwargs) -> FrameworkConfig:
    """Create validated framework configuration."""
    return FrameworkConfig(**kwargs)


def create_temporal_moment(
    timestamp: float,
    retention: ArrayLike,
    present_moment: ArrayLike, 
    protention: ArrayLike,
    synthesis_weights: ArrayLike,
) -> TemporalMoment:
    """Create validated temporal moment."""
    return TemporalMoment(
        timestamp=timestamp,
        retention=jnp.asarray(retention),
        present_moment=jnp.asarray(present_moment),
        protention=jnp.asarray(protention),
        synthesis_weights=jnp.asarray(synthesis_weights),
    )


# Export public API
__all__ = [
    # Core types
    'Array', 'ArrayLike', 'PRNGKey', 'PyTree',
    'Dim', 'Shape', 'TimeStep', 'TemporalWindow',
    
    # Enums
    'ConsciousnessLevel', 'CouplingStrength',
    
    # Configuration
    'FrameworkConfig',
    
    # Data structures
    'TemporalMoment', 'BodyState', 'CouplingState', 
    'AffordanceVector', 'MeaningStructure',
    'ExperienceRetentionState', 'EnactiveCouplingState',
    
    # Protocols
    'TemporalProcessor', 'EmbodimentProcessor', 'CouplingProcessor',
    'AffordanceProcessor', 'SenseMakingProcessor', 'ConsciousnessIntegrator',
    'EnactiveCouplingProcessor', 'ExperienceRetentionProcessor',
    
    # Metrics and validation
    'PerformanceMetrics', 
    'is_valid_array_shape', 'validate_temporal_consistency', 
    'validate_consciousness_state',
    
    # Factory functions
    'create_framework_config', 'create_temporal_moment',
    
    # Exceptions
    'EnactiveConsciousnessError', 'TemporalSynthesisError', 
    'EmbodimentError', 'CouplingError', 'AffordancePerceptionError',
    'SenseMakingError', 'ConsciousnessIntegrationError',
]