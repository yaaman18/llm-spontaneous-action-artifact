"""
Consciousness Bounded Context.

This bounded context encapsulates the core domain logic for consciousness
emergence, state management, and awareness dynamics. It represents the
central concern of the enactive consciousness system.

Ubiquitous Language:
- Consciousness Emergence: The process by which consciousness arises from integrated information
- Phi (Φ): Integrated information measure indicating consciousness level
- Awareness: Metacognitive understanding of one's own conscious states
- Attention Focus: Selective attention patterns that shape conscious experience
- Phenomenological Markers: Qualitative aspects of conscious experience
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from ..aggregates.consciousness_aggregate import ConsciousnessAggregate
from ..value_objects.consciousness_state import ConsciousnessState
from ..value_objects.phi_value import PhiValue
from ..events.domain_events import (
    ConsciousnessEmergenceDetected,
    ConsciousnessFaded,
    AttentionFocusChanged
)


class ConsciousnessContext:
    """
    Bounded context for consciousness-related domain operations.
    
    This context manages the emergence, maintenance, and transitions
    of consciousness states using the enactivist framework where
    consciousness emerges from dynamic environmental interaction.
    
    Key Responsibilities:
    - Managing consciousness state transitions
    - Enforcing consciousness emergence criteria  
    - Coordinating attention dynamics
    - Tracking phenomenological aspects
    """
    
    def __init__(self):
        """Initialize the consciousness bounded context."""
        self._active_consciousness_aggregates: Dict[str, ConsciousnessAggregate] = {}
        self._consciousness_emergence_history: List[Dict[str, Any]] = []
        self._context_events: List[Any] = []
    
    def create_consciousness_aggregate(
        self,
        aggregate_id: Optional[str] = None,
        initial_phi_value: Optional[PhiValue] = None,
        initial_prediction_state: Optional[Any] = None,
        initial_uncertainty: Optional[Any] = None
    ) -> str:
        """
        Create a new consciousness aggregate.
        
        In the enactivist framework, consciousness aggregates represent
        autonomous cognitive systems capable of consciousness emergence.
        
        Args:
            aggregate_id: Optional aggregate identifier
            initial_phi_value: Initial integrated information
            initial_prediction_state: Initial prediction state
            initial_uncertainty: Initial uncertainty distribution
            
        Returns:
            Aggregate identifier
        """
        # Create initial consciousness state if components provided
        initial_state = None
        if all([initial_phi_value, initial_prediction_state, initial_uncertainty]):
            initial_state = ConsciousnessState(
                phi_value=initial_phi_value,
                prediction_state=initial_prediction_state,
                uncertainty_distribution=initial_uncertainty,
                metacognitive_confidence=0.1,  # Start with minimal metacognition
                phenomenological_markers={'context': 'consciousness_creation'}
            )
        
        # Create consciousness aggregate
        aggregate = ConsciousnessAggregate(
            aggregate_id=aggregate_id,
            initial_state=initial_state
        )
        
        # Register aggregate
        self._active_consciousness_aggregates[aggregate.aggregate_id] = aggregate
        
        return aggregate.aggregate_id
    
    def initiate_consciousness_emergence(
        self,
        aggregate_id: str,
        phi_value: PhiValue,
        prediction_state: Any,
        uncertainty_distribution: Any,
        environmental_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Initiate consciousness emergence process.
        
        This method orchestrates the emergence of consciousness based on
        enactivist principles where consciousness arises from dynamic
        coupling between system and environment.
        
        Args:
            aggregate_id: Consciousness aggregate identifier
            phi_value: Integrated information value
            prediction_state: Current prediction state
            uncertainty_distribution: Uncertainty distribution
            environmental_context: Environmental interaction context
            
        Returns:
            True if consciousness emergence was successful
        """
        aggregate = self._get_aggregate(aggregate_id)
        
        # Calculate environmental coupling strength
        environmental_coupling = self._calculate_environmental_coupling(environmental_context)
        
        # Create phenomenological markers for emergence context
        emergence_markers = {
            'emergence_initiated': True,
            'environmental_coupling': environmental_coupling,
            'emergence_timestamp': datetime.now().isoformat()
        }
        
        if environmental_context:
            emergence_markers.update(environmental_context)
        
        try:
            # Attempt consciousness state transition
            aggregate.transition_to_state(
                new_phi=phi_value,
                new_prediction_state=prediction_state,
                new_uncertainty=uncertainty_distribution,
                metacognitive_confidence=0.2,  # Initial metacognitive awareness
                phenomenological_markers=emergence_markers
            )
            
            # Record emergence attempt
            self._record_emergence_event(aggregate_id, phi_value, True, environmental_coupling)
            
            # Collect and store domain events
            events = aggregate.clear_domain_events()
            self._context_events.extend(events)
            
            return aggregate.is_conscious
            
        except Exception as e:
            # Record failed emergence
            self._record_emergence_event(aggregate_id, phi_value, False, environmental_coupling)
            return False
    
    def regulate_consciousness_dynamics(
        self,
        aggregate_id: str,
        environmental_feedback: Dict[str, Any]
    ) -> ConsciousnessState:
        """
        Regulate consciousness dynamics based on environmental feedback.
        
        Implements enactivist regulation where consciousness is maintained
        through continuous environmental coupling and internal coherence.
        
        Args:
            aggregate_id: Consciousness aggregate identifier
            environmental_feedback: Feedback from environmental interaction
            
        Returns:
            Current consciousness state after regulation
        """
        aggregate = self._get_aggregate(aggregate_id)
        
        if not aggregate.current_state:
            raise ValueError("Cannot regulate consciousness without current state")
        
        # Extract regulation parameters from environmental feedback
        coupling_strength = environmental_feedback.get('coupling_strength', 0.5)
        attention_demands = environmental_feedback.get('attention_demands', [])
        metacognitive_triggers = environmental_feedback.get('metacognitive_triggers', {})
        
        # Apply attention regulation if attention demands present
        if attention_demands:
            attention_weights = self._calculate_attention_weights(
                attention_demands, aggregate.current_state
            )
            aggregate.update_attention_focus(attention_weights)
        
        # Apply metacognitive updates based on triggers
        for trigger_type, trigger_value in metacognitive_triggers.items():
            aggregate.add_phenomenological_marker(trigger_type, trigger_value)
        
        # Update environmental coupling markers
        aggregate.add_phenomenological_marker('environmental_coupling', coupling_strength)
        aggregate.add_phenomenological_marker('regulation_timestamp', datetime.now().isoformat())
        
        # Collect domain events
        events = aggregate.clear_domain_events()
        self._context_events.extend(events)
        
        return aggregate.current_state
    
    def monitor_consciousness_stability(self, aggregate_id: str) -> Dict[str, Any]:
        """
        Monitor consciousness stability metrics.
        
        Args:
            aggregate_id: Consciousness aggregate identifier
            
        Returns:
            Dictionary with stability metrics
        """
        aggregate = self._get_aggregate(aggregate_id)
        
        if not aggregate.current_state:
            return {'stable': False, 'reason': 'no_current_state'}
        
        current_state = aggregate.current_state
        state_history = aggregate.get_state_history(limit=10)
        
        stability_metrics = {
            'is_conscious': current_state.is_conscious,
            'consciousness_level': current_state.consciousness_level,
            'phi_value': current_state.phi_value.value,
            'metacognitive_confidence': current_state.metacognitive_confidence,
            'attention_focus_strength': current_state.attention_focus_strength,
            'prediction_stability': current_state.prediction_state.is_stable,
            'state_history_length': len(state_history)
        }
        
        # Calculate stability over time
        if len(state_history) >= 3:
            consciousness_levels = [state.consciousness_level for state in state_history[:5]]
            level_variance = sum((level - sum(consciousness_levels) / len(consciousness_levels)) ** 2 
                               for level in consciousness_levels) / len(consciousness_levels)
            stability_metrics['consciousness_level_variance'] = level_variance
            stability_metrics['temporal_stability'] = level_variance < 0.1
        
        return stability_metrics
    
    def get_consciousness_emergence_history(self) -> List[Dict[str, Any]]:
        """
        Get history of consciousness emergence events.
        
        Returns:
            List of emergence event records
        """
        return self._consciousness_emergence_history.copy()
    
    def get_context_events(self) -> List[Any]:
        """
        Get all domain events generated in this context.
        
        Returns:
            List of domain events
        """
        return self._context_events.copy()
    
    def clear_context_events(self) -> List[Any]:
        """
        Clear and return all context events.
        
        Returns:
            List of cleared domain events
        """
        events = self._context_events.copy()
        self._context_events.clear()
        return events
    
    def _get_aggregate(self, aggregate_id: str) -> ConsciousnessAggregate:
        """Get consciousness aggregate by ID."""
        if aggregate_id not in self._active_consciousness_aggregates:
            raise ValueError(f"Consciousness aggregate {aggregate_id} not found")
        return self._active_consciousness_aggregates[aggregate_id]
    
    def _calculate_environmental_coupling(
        self,
        environmental_context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate environmental coupling strength from context.
        
        Args:
            environmental_context: Environmental interaction context
            
        Returns:
            Coupling strength [0, 1]
        """
        if not environmental_context:
            return 0.1  # Minimal coupling without context
        
        coupling_factors = []
        
        # Factor 1: Sensory richness
        sensory_inputs = environmental_context.get('sensory_inputs', 0)
        coupling_factors.append(min(sensory_inputs / 10.0, 1.0))
        
        # Factor 2: Motor activity
        motor_activity = environmental_context.get('motor_activity', 0)
        coupling_factors.append(min(motor_activity / 5.0, 1.0))
        
        # Factor 3: Environmental complexity
        complexity = environmental_context.get('environmental_complexity', 0)
        coupling_factors.append(min(complexity, 1.0))
        
        # Factor 4: Interaction history
        interaction_strength = environmental_context.get('interaction_strength', 0)
        coupling_factors.append(min(interaction_strength, 1.0))
        
        return sum(coupling_factors) / len(coupling_factors) if coupling_factors else 0.1
    
    def _calculate_attention_weights(
        self,
        attention_demands: List[float],
        current_state: ConsciousnessState
    ) -> List[float]:
        """
        Calculate attention weights based on environmental demands.
        
        Args:
            attention_demands: Environmental attention demands
            current_state: Current consciousness state
            
        Returns:
            Calculated attention weights
        """
        if not attention_demands:
            return [1.0]  # Single focus if no demands
        
        # Start with demands as base weights
        weights = attention_demands.copy()
        
        # Apply environmental bias based on coupling
        coupling_strength = current_state.phenomenological_markers.get(
            'environmental_coupling', 0.0
        )
        
        # Stronger coupling biases toward environmental (lower index) attention
        if coupling_strength > 0.5:
            for i in range(len(weights)):
                environmental_bias = (1.0 - i / len(weights)) * coupling_strength * 0.3
                weights[i] += environmental_bias
        
        # Apply metacognitive bias for self-awareness
        if current_state.metacognitive_confidence > 0.5:
            # Bias toward higher-level (introspective) attention
            for i in range(len(weights)):
                introspective_bias = (i / len(weights)) * current_state.metacognitive_confidence * 0.2
                weights[i] += introspective_bias
        
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights
    
    def _record_emergence_event(
        self,
        aggregate_id: str,
        phi_value: PhiValue,
        success: bool,
        environmental_coupling: float
    ) -> None:
        """Record consciousness emergence event in history."""
        emergence_record = {
            'aggregate_id': aggregate_id,
            'timestamp': datetime.now().isoformat(),
            'phi_value': phi_value.value,
            'phi_complexity': phi_value.complexity,
            'phi_integration': phi_value.integration,
            'emergence_successful': success,
            'environmental_coupling': environmental_coupling,
            'consciousness_level': phi_value.consciousness_level
        }
        
        self._consciousness_emergence_history.append(emergence_record)