"""
Domain Events for Enactive Consciousness Framework.

Domain events capture significant occurrences within the consciousness and
learning domains, enabling decoupled communication between aggregates and
bounded contexts following event-driven architecture principles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid

from ..value_objects.consciousness_state import ConsciousnessState
from ..value_objects.prediction_state import PredictionState


class DomainEvent(ABC):
    """
    Base class for all domain events in the enactive consciousness system.
    
    Domain events represent significant business occurrences that other
    parts of the system need to know about. They enable loose coupling
    between aggregates and bounded contexts.
    """
    
    def __init__(self, aggregate_id: str, timestamp: Optional[datetime] = None):
        """
        Initialize domain event.
        
        Args:
            aggregate_id: ID of the aggregate that generated this event
            timestamp: When the event occurred (defaults to now)
        """
        self.event_id = str(uuid.uuid4())
        self.aggregate_id = aggregate_id
        self.timestamp = timestamp or datetime.now()
        self.event_version = 1
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        pass
    
    @property
    @abstractmethod
    def event_type(self) -> str:
        """String identifier for the event type."""
        pass


@dataclass
class ConsciousnessStateChanged(DomainEvent):
    """
    Event fired when consciousness state transitions occur.
    
    Represents changes in the overall consciousness state including
    Φ value changes, prediction state updates, and attention shifts.
    """
    
    previous_state: Optional[ConsciousnessState]
    new_state: ConsciousnessState
    consciousness_level_changed: bool
    
    def __init__(
        self,
        aggregate_id: str,
        previous_state: Optional[ConsciousnessState],
        new_state: ConsciousnessState,
        consciousness_level_changed: bool,
        timestamp: Optional[datetime] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.previous_state = previous_state
        self.new_state = new_state
        self.consciousness_level_changed = consciousness_level_changed
    
    @property
    def event_type(self) -> str:
        return "consciousness_state_changed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "previous_state": self.previous_state.to_dict() if self.previous_state else None,
            "new_state": self.new_state.to_dict(),
            "consciousness_level_changed": self.consciousness_level_changed,
            "phi_value_change": (
                self.new_state.phi_value.value - 
                (self.previous_state.phi_value.value if self.previous_state else 0)
            )
        }


@dataclass
class ConsciousnessEmergenceDetected(DomainEvent):
    """
    Event fired when consciousness emerges in the system.
    
    Triggered when the system transitions from unconscious to conscious
    state based on integrated information and metacognitive criteria.
    """
    
    phi_value: float
    consciousness_level: float
    emergence_factors: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(
        self,
        aggregate_id: str,
        phi_value: float,
        consciousness_level: float,
        timestamp: Optional[datetime] = None,
        emergence_factors: Optional[Dict[str, Any]] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.phi_value = phi_value
        self.consciousness_level = consciousness_level
        self.emergence_factors = emergence_factors or {}
    
    @property
    def event_type(self) -> str:
        return "consciousness_emergence_detected"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "phi_value": self.phi_value,
            "consciousness_level": self.consciousness_level,
            "emergence_factors": self.emergence_factors
        }


@dataclass
class ConsciousnessFaded(DomainEvent):
    """
    Event fired when consciousness fades from the system.
    
    Triggered when the system transitions from conscious to unconscious
    state, typically due to decreased Φ or environmental decoupling.
    """
    
    final_phi_value: float
    fade_duration_seconds: Optional[float] = None
    fade_causes: List[str] = field(default_factory=list)
    
    def __init__(
        self,
        aggregate_id: str,
        final_phi_value: float,
        timestamp: Optional[datetime] = None,
        fade_duration_seconds: Optional[float] = None,
        fade_causes: Optional[List[str]] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.final_phi_value = final_phi_value
        self.fade_duration_seconds = fade_duration_seconds
        self.fade_causes = fade_causes or []
    
    @property
    def event_type(self) -> str:
        return "consciousness_faded"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "final_phi_value": self.final_phi_value,
            "fade_duration_seconds": self.fade_duration_seconds,
            "fade_causes": self.fade_causes
        }


@dataclass
class AttentionFocusChanged(DomainEvent):
    """
    Event fired when attention focus patterns change.
    
    Represents shifts in attentional distribution and focus strength,
    which are crucial for consciousness quality and environmental coupling.
    """
    
    new_focus_strength: float
    attention_weights: List[float]
    focus_shift_magnitude: Optional[float] = None
    
    def __init__(
        self,
        aggregate_id: str,
        new_focus_strength: float,
        attention_weights: List[float],
        timestamp: Optional[datetime] = None,
        focus_shift_magnitude: Optional[float] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.new_focus_strength = new_focus_strength
        self.attention_weights = attention_weights
        self.focus_shift_magnitude = focus_shift_magnitude
    
    @property
    def event_type(self) -> str:
        return "attention_focus_changed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "new_focus_strength": self.new_focus_strength,
            "attention_weights": self.attention_weights,
            "focus_shift_magnitude": self.focus_shift_magnitude
        }


@dataclass
class MetacognitiveInsightGained(DomainEvent):
    """
    Event fired when metacognitive insights are gained.
    
    Represents moments of self-awareness, reflection, or meta-level
    understanding that indicate higher-order consciousness processes.
    """
    
    insight_type: str
    insight_content: Any
    consciousness_level: float
    confidence: float = 0.0
    
    def __init__(
        self,
        aggregate_id: str,
        insight_type: str,
        insight_content: Any,
        consciousness_level: float,
        timestamp: Optional[datetime] = None,
        confidence: float = 0.0
    ):
        super().__init__(aggregate_id, timestamp)
        self.insight_type = insight_type
        self.insight_content = insight_content
        self.consciousness_level = consciousness_level
        self.confidence = confidence
    
    @property
    def event_type(self) -> str:
        return "metacognitive_insight_gained"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "insight_type": self.insight_type,
            "insight_content": str(self.insight_content),
            "consciousness_level": self.consciousness_level,
            "confidence": self.confidence
        }


@dataclass
class LearningEpochCompleted(DomainEvent):
    """
    Event fired when a learning epoch completes.
    
    Represents completion of a learning cycle including predictive
    coding updates, SOM training, and environmental coupling assessment.
    """
    
    epoch_number: int
    prediction_error: float
    coupling_strength: float
    learning_rate: Optional[float] = None
    
    def __init__(
        self,
        aggregate_id: str,
        epoch_number: int,
        prediction_error: float,
        coupling_strength: float,
        timestamp: Optional[datetime] = None,
        learning_rate: Optional[float] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.epoch_number = epoch_number
        self.prediction_error = prediction_error
        self.coupling_strength = coupling_strength
        self.learning_rate = learning_rate
    
    @property
    def event_type(self) -> str:
        return "learning_epoch_completed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "epoch_number": self.epoch_number,
            "prediction_error": self.prediction_error,
            "coupling_strength": self.coupling_strength,
            "learning_rate": self.learning_rate
        }


@dataclass
class PredictionErrorThresholdCrossed(DomainEvent):
    """
    Event fired when prediction error crosses significant thresholds.
    
    Indicates important learning milestones or potential learning
    difficulties that require attention or intervention.
    """
    
    threshold_value: float
    previous_error: float
    new_error: float
    threshold_direction: str  # "crossed_below" or "crossed_above"
    
    def __init__(
        self,
        aggregate_id: str,
        threshold_value: float,
        previous_error: float,
        new_error: float,
        timestamp: Optional[datetime] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.threshold_value = threshold_value
        self.previous_error = previous_error
        self.new_error = new_error
        self.threshold_direction = (
            "crossed_below" if previous_error > threshold_value >= new_error
            else "crossed_above"
        )
    
    @property
    def event_type(self) -> str:
        return "prediction_error_threshold_crossed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "threshold_value": self.threshold_value,
            "previous_error": self.previous_error,
            "new_error": self.new_error,
            "threshold_direction": self.threshold_direction,
            "error_reduction": self.previous_error - self.new_error
        }


@dataclass
class SelfOrganizationConverged(DomainEvent):
    """
    Event fired when self-organizing processes reach convergence.
    
    Indicates that the SOM has reached a stable organization pattern
    that effectively represents the input space structure.
    """
    
    final_error: float
    epochs_to_convergence: int
    convergence_quality: Optional[float] = None
    
    def __init__(
        self,
        aggregate_id: str,
        final_error: float,
        epochs_to_convergence: int,
        timestamp: Optional[datetime] = None,
        convergence_quality: Optional[float] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.final_error = final_error
        self.epochs_to_convergence = epochs_to_convergence
        self.convergence_quality = convergence_quality
    
    @property
    def event_type(self) -> str:
        return "self_organization_converged"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "final_error": self.final_error,
            "epochs_to_convergence": self.epochs_to_convergence,
            "convergence_quality": self.convergence_quality
        }


@dataclass
class EnvironmentalCouplingStrengthened(DomainEvent):
    """
    Event fired when environmental coupling strength changes significantly.
    
    Represents changes in how strongly the system is coupled with its
    environment, reflecting enactivist principles of structural coupling.
    """
    
    coupling_strength: float
    adaptation_type: str
    coupling_change_magnitude: Optional[float] = None
    
    def __init__(
        self,
        aggregate_id: str,
        coupling_strength: float,
        adaptation_type: str,
        timestamp: Optional[datetime] = None,
        coupling_change_magnitude: Optional[float] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.coupling_strength = coupling_strength
        self.adaptation_type = adaptation_type
        self.coupling_change_magnitude = coupling_change_magnitude
    
    @property
    def event_type(self) -> str:
        return "environmental_coupling_strengthened"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "coupling_strength": self.coupling_strength,
            "adaptation_type": self.adaptation_type,
            "coupling_change_magnitude": self.coupling_change_magnitude
        }


@dataclass
class AdaptiveLearningRateChanged(DomainEvent):
    """
    Event fired when adaptive learning rate changes occur.
    
    Represents adjustments to learning rate based on environmental
    coupling, prediction quality, or other adaptive mechanisms.
    """
    
    new_learning_rate: float
    previous_learning_rate: float
    adaptation_reason: str
    adaptation_magnitude: float
    
    def __init__(
        self,
        aggregate_id: str,
        new_learning_rate: float,
        previous_learning_rate: float,
        adaptation_reason: str,
        timestamp: Optional[datetime] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.new_learning_rate = new_learning_rate
        self.previous_learning_rate = previous_learning_rate
        self.adaptation_reason = adaptation_reason
        self.adaptation_magnitude = abs(new_learning_rate - previous_learning_rate)
    
    @property
    def event_type(self) -> str:
        return "adaptive_learning_rate_changed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "new_learning_rate": self.new_learning_rate,
            "previous_learning_rate": self.previous_learning_rate,
            "adaptation_reason": self.adaptation_reason,
            "adaptation_magnitude": self.adaptation_magnitude
        }


@dataclass
class HierarchicalPredictionGenerated(DomainEvent):
    """
    Event fired when hierarchical predictions are generated.
    
    Represents the generation of predictions across all hierarchical levels
    in the predictive coding system.
    """
    
    hierarchy_levels: int
    prediction_magnitudes: List[float]
    attention_weights: List[float]
    free_energy_estimate: Optional[float] = None
    
    def __init__(
        self,
        aggregate_id: str,
        hierarchy_levels: int,
        prediction_magnitudes: List[float],
        attention_weights: List[float],
        timestamp: Optional[datetime] = None,
        free_energy_estimate: Optional[float] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.hierarchy_levels = hierarchy_levels
        self.prediction_magnitudes = prediction_magnitudes
        self.attention_weights = attention_weights
        self.free_energy_estimate = free_energy_estimate
    
    @property
    def event_type(self) -> str:
        return "hierarchical_prediction_generated"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "hierarchy_levels": self.hierarchy_levels,
            "prediction_magnitudes": self.prediction_magnitudes,
            "attention_weights": self.attention_weights,
            "free_energy_estimate": self.free_energy_estimate
        }


@dataclass
class PrecisionWeightsAdapted(DomainEvent):
    """
    Event fired when precision weights are adapted.
    
    Represents changes in precision weights based on prediction errors
    and environmental feedback, implementing attention mechanisms.
    """
    
    previous_weights: List[float]
    new_weights: List[float] 
    adaptation_magnitude: float
    attention_focus_change: float
    
    def __init__(
        self,
        aggregate_id: str,
        previous_weights: List[float],
        new_weights: List[float],
        timestamp: Optional[datetime] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.previous_weights = previous_weights
        self.new_weights = new_weights
        self.adaptation_magnitude = sum(
            abs(new - old) for new, old in zip(new_weights, previous_weights)
        )
        
        # Compute attention focus change (using entropy as proxy)
        prev_entropy = self._compute_entropy(previous_weights)
        new_entropy = self._compute_entropy(new_weights) 
        self.attention_focus_change = prev_entropy - new_entropy
    
    def _compute_entropy(self, weights: List[float]) -> float:
        """Compute entropy of weight distribution."""
        import math
        total = sum(weights)
        if total == 0:
            return 0.0
        normalized = [w / total for w in weights]
        return -sum(p * math.log(p + 1e-10) for p in normalized if p > 0)
    
    @property
    def event_type(self) -> str:
        return "precision_weights_adapted"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "previous_weights": self.previous_weights,
            "new_weights": self.new_weights,
            "adaptation_magnitude": self.adaptation_magnitude,
            "attention_focus_change": self.attention_focus_change
        }


@dataclass
class FreeEnergyMinimized(DomainEvent):
    """
    Event fired when free energy minimization step is completed.
    
    Represents completion of a variational free energy minimization
    step in the predictive coding system.
    """
    
    previous_free_energy: float
    new_free_energy: float
    energy_reduction: float
    optimization_method: str
    
    def __init__(
        self,
        aggregate_id: str,
        previous_free_energy: float,
        new_free_energy: float,
        optimization_method: str = "gradient_descent",
        timestamp: Optional[datetime] = None
    ):
        super().__init__(aggregate_id, timestamp)
        self.previous_free_energy = previous_free_energy
        self.new_free_energy = new_free_energy
        self.energy_reduction = previous_free_energy - new_free_energy
        self.optimization_method = optimization_method
    
    @property
    def event_type(self) -> str:
        return "free_energy_minimized"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "previous_free_energy": self.previous_free_energy,
            "new_free_energy": self.new_free_energy,
            "energy_reduction": self.energy_reduction,
            "optimization_method": self.optimization_method
        }


@dataclass
class ActiveInferenceEngaged(DomainEvent):
    """
    Event fired when active inference mechanisms are engaged.
    
    Represents activation of attention-based active inference where
    the system modulates its processing based on precision weights.
    """
    
    attention_modulation_strength: float
    dominant_hierarchical_level: int
    precision_entropy: float
    inference_type: str = "attention_modulated"
    
    def __init__(
        self,
        aggregate_id: str,
        attention_modulation_strength: float,
        dominant_hierarchical_level: int,
        precision_entropy: float,
        timestamp: Optional[datetime] = None,
        inference_type: str = "attention_modulated"
    ):
        super().__init__(aggregate_id, timestamp)
        self.attention_modulation_strength = attention_modulation_strength
        self.dominant_hierarchical_level = dominant_hierarchical_level
        self.precision_entropy = precision_entropy
        self.inference_type = inference_type
    
    @property
    def event_type(self) -> str:
        return "active_inference_engaged"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "attention_modulation_strength": self.attention_modulation_strength,
            "dominant_hierarchical_level": self.dominant_hierarchical_level,
            "precision_entropy": self.precision_entropy,
            "inference_type": self.inference_type
        }