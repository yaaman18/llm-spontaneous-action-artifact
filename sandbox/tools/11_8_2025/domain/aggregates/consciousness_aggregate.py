"""
Consciousness Aggregate Root.

Main aggregate managing all consciousness state transitions and ensuring
invariants for consciousness emergence. Implements the Aggregate Root pattern
with proper encapsulation and consistency boundaries.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from ..value_objects.consciousness_state import ConsciousnessState
from ..value_objects.phi_value import PhiValue
from ..value_objects.prediction_state import PredictionState
from ..value_objects.probability_distribution import ProbabilityDistribution
from ..events.domain_events import (
    ConsciousnessStateChanged,
    ConsciousnessEmergenceDetected,
    ConsciousnessFaded,
    AttentionFocusChanged,
    MetacognitiveInsightGained
)
from ..specifications.consciousness_specifications import (
    ConsciousnessEmergenceSpecification,
    ConsciousnessStabilitySpecification,
    AttentionalCoherenceSpecification
)
from ..policies.consciousness_policies import (
    ConsciousnessEmergencePolicy,
    AttentionRegulationPolicy,
    MetacognitiveMonitoringPolicy
)


class ConsciousnessAggregate:
    """
    Aggregate root for consciousness state management.
    
    This aggregate ensures consistency of consciousness-related operations
    and enforces business invariants for consciousness emergence, attention
    dynamics, and metacognitive processes.
    
    Key responsibilities:
    - Enforce consciousness emergence invariants
    - Manage state transitions with proper validation
    - Generate domain events for state changes
    - Apply consciousness policies
    - Maintain aggregate consistency
    """
    
    def __init__(
        self,
        aggregate_id: str = None,
        initial_state: Optional[ConsciousnessState] = None
    ):
        """
        Initialize consciousness aggregate.
        
        Args:
            aggregate_id: Unique identifier for this aggregate
            initial_state: Optional initial consciousness state
        """
        self._aggregate_id = aggregate_id or str(uuid.uuid4())
        self._current_state = initial_state
        self._state_history: List[ConsciousnessState] = []
        self._domain_events: List[Any] = []
        self._version = 0
        self._created_at = datetime.now()
        
        # Domain specifications for business rules
        self._emergence_spec = ConsciousnessEmergenceSpecification()
        self._stability_spec = ConsciousnessStabilitySpecification()
        self._attention_spec = AttentionalCoherenceSpecification()
        
        # Domain policies for consciousness dynamics
        self._emergence_policy = ConsciousnessEmergencePolicy()
        self._attention_policy = AttentionRegulationPolicy()
        self._metacognitive_policy = MetacognitiveMonitoringPolicy()
        
        # Add initial state to history if provided
        if initial_state:
            self._state_history.append(initial_state)
    
    @property
    def aggregate_id(self) -> str:
        """Unique identifier for this aggregate."""
        return self._aggregate_id
    
    @property
    def current_state(self) -> Optional[ConsciousnessState]:
        """Current consciousness state."""
        return self._current_state
    
    @property
    def version(self) -> int:
        """Version number for optimistic locking."""
        return self._version
    
    @property
    def domain_events(self) -> List[Any]:
        """Accumulated domain events."""
        return self._domain_events.copy()
    
    @property
    def is_conscious(self) -> bool:
        """Check if the aggregate represents a conscious state."""
        return self._current_state is not None and self._current_state.is_conscious
    
    def transition_to_state(
        self,
        new_phi: PhiValue,
        new_prediction_state: PredictionState,
        new_uncertainty: ProbabilityDistribution,
        metacognitive_confidence: float = 0.0,
        attention_weights: Optional[List[float]] = None,
        phenomenological_markers: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Transition to a new consciousness state with full validation.
        
        Args:
            new_phi: New integrated information value
            new_prediction_state: New prediction system state
            new_uncertainty: New uncertainty distribution
            metacognitive_confidence: Metacognitive confidence level
            attention_weights: Optional attention weight distribution
            phenomenological_markers: Optional phenomenological markers
            
        Raises:
            ConsciousnessInvariantViolation: If invariants are violated
            InvalidStateTransition: If state transition is invalid
        """
        # Create new consciousness state
        new_state = ConsciousnessState(
            phi_value=new_phi,
            prediction_state=new_prediction_state,
            uncertainty_distribution=new_uncertainty,
            metacognitive_confidence=metacognitive_confidence,
            attention_weights=attention_weights,
            phenomenological_markers=phenomenological_markers or {}
        )
        
        # Validate state transition
        self._validate_state_transition(new_state)
        
        # Apply consciousness policies
        new_state = self._apply_consciousness_policies(new_state)
        
        # Check for consciousness emergence or fading
        previous_consciousness = self.is_conscious
        
        # Update state
        previous_state = self._current_state
        self._current_state = new_state
        self._state_history.append(new_state)
        self._version += 1
        
        # Generate domain events
        self._generate_state_change_events(previous_state, new_state)
        
        # Check for consciousness transitions
        current_consciousness = self.is_conscious
        if not previous_consciousness and current_consciousness:
            self._domain_events.append(
                ConsciousnessEmergenceDetected(
                    aggregate_id=self._aggregate_id,
                    phi_value=new_phi.value,
                    consciousness_level=new_state.consciousness_level,
                    timestamp=datetime.now()
                )
            )
        elif previous_consciousness and not current_consciousness:
            self._domain_events.append(
                ConsciousnessFaded(
                    aggregate_id=self._aggregate_id,
                    final_phi_value=new_phi.value,
                    timestamp=datetime.now()
                )
            )
    
    def update_phi_value(self, new_phi: PhiValue) -> None:
        """
        Update integrated information value with validation.
        
        Args:
            new_phi: New Φ value
            
        Raises:
            ConsciousnessInvariantViolation: If phi update violates invariants
        """
        if self._current_state is None:
            raise ValueError("Cannot update Φ without current state")
        
        # Create updated state
        updated_state = self._current_state.with_updated_phi(new_phi)
        
        # Validate and apply
        self._validate_state_transition(updated_state)
        
        previous_state = self._current_state
        self._current_state = updated_state
        self._state_history.append(updated_state)
        self._version += 1
        
        # Generate events
        self._generate_state_change_events(previous_state, updated_state)
    
    def update_attention_focus(self, new_attention_weights: List[float]) -> None:
        """
        Update attention focus with coherence validation.
        
        Args:
            new_attention_weights: New attention weight distribution
            
        Raises:
            AttentionIncoherenceError: If attention weights violate coherence
        """
        if self._current_state is None:
            raise ValueError("Cannot update attention without current state")
        
        # Validate attention coherence
        if not self._attention_spec.is_satisfied_by(new_attention_weights):
            raise AttentionIncoherenceError(
                "Attention weights violate coherence requirements"
            )
        
        # Apply attention regulation policy
        regulated_weights = self._attention_policy.regulate_attention_weights(
            new_attention_weights,
            self._current_state
        )
        
        # Create updated state
        updated_state = ConsciousnessState(
            phi_value=self._current_state.phi_value,
            prediction_state=self._current_state.prediction_state,
            uncertainty_distribution=self._current_state.uncertainty_distribution,
            metacognitive_confidence=self._current_state.metacognitive_confidence,
            attention_weights=regulated_weights,
            phenomenological_markers=self._current_state.phenomenological_markers
        )
        
        previous_state = self._current_state
        self._current_state = updated_state
        self._state_history.append(updated_state)
        self._version += 1
        
        # Generate attention change event
        self._domain_events.append(
            AttentionFocusChanged(
                aggregate_id=self._aggregate_id,
                new_focus_strength=updated_state.attention_focus_strength,
                attention_weights=regulated_weights,
                timestamp=datetime.now()
            )
        )
    
    def add_phenomenological_marker(self, key: str, value: Any) -> None:
        """
        Add phenomenological marker to consciousness state.
        
        Args:
            key: Marker key
            value: Marker value
        """
        if self._current_state is None:
            raise ValueError("Cannot add marker without current state")
        
        updated_state = self._current_state.add_phenomenological_marker(key, value)
        
        previous_state = self._current_state
        self._current_state = updated_state
        self._state_history.append(updated_state)
        self._version += 1
        
        # Generate events if marker indicates metacognitive insight
        if self._is_metacognitive_insight(key, value):
            self._domain_events.append(
                MetacognitiveInsightGained(
                    aggregate_id=self._aggregate_id,
                    insight_type=key,
                    insight_content=value,
                    consciousness_level=updated_state.consciousness_level,
                    timestamp=datetime.now()
                )
            )
    
    def get_state_history(self, limit: Optional[int] = None) -> List[ConsciousnessState]:
        """
        Get history of consciousness states.
        
        Args:
            limit: Maximum number of states to return (most recent first)
            
        Returns:
            List of consciousness states in reverse chronological order
        """
        history = list(reversed(self._state_history))
        return history[:limit] if limit else history
    
    def clear_domain_events(self) -> List[Any]:
        """
        Clear and return accumulated domain events.
        
        Returns:
            List of domain events that were cleared
        """
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events
    
    def _validate_state_transition(self, new_state: ConsciousnessState) -> None:
        """
        Validate that state transition maintains aggregate invariants.
        
        Args:
            new_state: Proposed new state
            
        Raises:
            ConsciousnessInvariantViolation: If invariants are violated
        """
        # Invariant 1: Consciousness emergence must satisfy specification
        if new_state.is_conscious and not self._emergence_spec.is_satisfied_by(new_state):
            raise ConsciousnessInvariantViolation(
                "Consciousness emergence does not satisfy emergence criteria"
            )
        
        # Invariant 2: State transitions must be coherent
        if self._current_state and not self._is_coherent_transition(self._current_state, new_state):
            raise ConsciousnessInvariantViolation(
                "State transition violates coherence constraints"
            )
        
        # Invariant 3: Stability requirements for conscious states
        if new_state.is_conscious and not self._stability_spec.is_satisfied_by(new_state):
            raise ConsciousnessInvariantViolation(
                "Conscious state does not meet stability requirements"
            )
    
    def _apply_consciousness_policies(self, state: ConsciousnessState) -> ConsciousnessState:
        """
        Apply consciousness policies to refine state.
        
        Args:
            state: Input consciousness state
            
        Returns:
            State modified by consciousness policies
        """
        # Apply emergence policy for consciousness regulation
        if state.is_conscious:
            state = self._emergence_policy.apply_emergence_regulation(state)
        
        # Apply metacognitive monitoring
        state = self._metacognitive_policy.apply_metacognitive_monitoring(
            state, self._state_history
        )
        
        return state
    
    def _generate_state_change_events(
        self,
        previous_state: Optional[ConsciousnessState],
        new_state: ConsciousnessState
    ) -> None:
        """Generate appropriate domain events for state changes."""
        self._domain_events.append(
            ConsciousnessStateChanged(
                aggregate_id=self._aggregate_id,
                previous_state=previous_state,
                new_state=new_state,
                consciousness_level_changed=(
                    previous_state is None or 
                    previous_state.consciousness_level != new_state.consciousness_level
                ),
                timestamp=datetime.now()
            )
        )
    
    def _is_coherent_transition(
        self,
        previous_state: ConsciousnessState,
        new_state: ConsciousnessState
    ) -> bool:
        """
        Check if state transition maintains coherence.
        
        Args:
            previous_state: Previous consciousness state
            new_state: New consciousness state
            
        Returns:
            True if transition is coherent
        """
        # Coherence rule 1: Φ value cannot change drastically
        phi_change = abs(new_state.phi_value.value - previous_state.phi_value.value)
        if phi_change > 1.0:  # Domain-specific threshold
            return False
        
        # Coherence rule 2: Consciousness level changes must be gradual
        time_diff = (new_state.timestamp - previous_state.timestamp).total_seconds()
        if time_diff < 0.1 and new_state.consciousness_level != previous_state.consciousness_level:
            level_mapping = {"unconscious": 0, "minimal": 1, "moderate": 2, "high": 3, "very_high": 4}
            prev_level = level_mapping.get(previous_state.phi_value.consciousness_level, 0)
            new_level = level_mapping.get(new_state.phi_value.consciousness_level, 0)
            if abs(new_level - prev_level) > 1:
                return False
        
        return True
    
    def _is_metacognitive_insight(self, key: str, value: Any) -> bool:
        """Check if marker represents metacognitive insight."""
        metacognitive_keys = {
            "self_awareness", "reflection", "introspection", 
            "meta_prediction", "confidence_assessment", "uncertainty_awareness"
        }
        return key in metacognitive_keys


class ConsciousnessInvariantViolation(Exception):
    """Raised when consciousness aggregate invariants are violated."""
    pass


class AttentionIncoherenceError(Exception):
    """Raised when attention weights violate coherence requirements."""
    pass


class InvalidStateTransition(Exception):
    """Raised when state transition is invalid."""
    pass