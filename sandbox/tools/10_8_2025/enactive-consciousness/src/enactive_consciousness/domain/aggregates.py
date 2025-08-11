"""Domain Aggregates for Enactive Consciousness.

This module defines aggregates following Eric Evans' DDD methodology.
Aggregates are clusters of domain objects that can be treated as a single unit
for data changes, with one entity serving as the aggregate root.

Theoretical Foundations:
- Each aggregate represents a consistency boundary in consciousness
- Aggregate roots control access and maintain invariants
- Internal entities and value objects support aggregate functionality
- State management follows clean architecture patterns

Design Principles:
1. Aggregates maintain consistency boundaries
2. Aggregate roots are the only entities referenced from outside
3. Internal references use identity rather than direct object references  
4. Aggregates are designed around business use cases
5. Size is kept manageable to avoid performance issues
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import uuid
import time

import jax
import jax.numpy as jnp
from ..types import Array, TimeStep
from ..architecture.state_entities import ImmutableStateContainer, StateType, StateEvolutionType

from .value_objects import (
    RetentionMoment,
    PrimalImpression, 
    ProtentionalHorizon,
    TemporalSynthesisWeights,
    BodyBoundary,
    MotorIntention,
    ProprioceptiveField,
    TactileFeedback,
    CouplingStrength,
    AutopoeticProcess,
    StructuralCoupling,
    MeaningEmergence,
    ExperientialTrace,
    SedimentLayer,
    RecallContext,
    AssociativeLink,
    RecallMode,
)

from .entities import (
    TemporalMomentEntity,
    EmbodiedExperienceEntity,
    CircularCausalityEntity,
    ExperientialMemoryEntity,
    ConsciousnessState,
)


class AggregateState(Enum):
    """States of aggregate lifecycle."""
    INITIALIZING = "initializing"
    COHERENT = "coherent"
    PROCESSING = "processing"
    INTEGRATING = "integrating"
    DISSIPATING = "dissipating"


# ============================================================================
# TEMPORAL CONSCIOUSNESS AGGREGATE (Husserlian Phenomenology)
# ============================================================================

@dataclass
class TemporalConsciousnessAggregate:
    """Husserlian temporal consciousness aggregate.
    
    Aggregate root for temporal consciousness functionality, managing
    retention-present-protention synthesis and temporal flow evolution.
    This aggregate maintains the consistency boundary for all temporal
    consciousness operations following Husserlian phenomenology.
    
    Invariants:
    - At most one active temporal moment at any time
    - Retention buffer maintains temporal ordering
    - Protention projections must be temporally coherent
    - Temporal synthesis weights always sum to 1.0
    """
    
    # Aggregate Identity
    aggregate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: TimeStep = field(default_factory=time.time)
    
    # Aggregate Root Entity
    current_moment: Optional[TemporalMomentEntity] = None
    
    # Internal Entities and Value Objects
    retention_buffer: List[RetentionMoment] = field(default_factory=list)
    protention_projections: List[ProtentionalHorizon] = field(default_factory=list)
    temporal_flow_history: List[str] = field(default_factory=list)  # Moment IDs
    
    # Aggregate State
    aggregate_state: AggregateState = AggregateState.INITIALIZING
    temporal_coherence_threshold: float = 0.3
    max_retention_depth: int = 10
    max_protention_horizon: int = 5
    
    # Internal State Container Integration
    _state_container: Optional[ImmutableStateContainer] = field(default=None, init=False)
    _synthesis_cache: Dict[str, Array] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize aggregate and validate invariants."""
        self._validate_aggregate_invariants()
        if self.aggregate_state == AggregateState.INITIALIZING:
            self._initialize_temporal_system()
    
    def _validate_aggregate_invariants(self) -> None:
        """Validate core aggregate business invariants."""
        if not self.aggregate_id:
            raise ValueError("TemporalConsciousnessAggregate must have unique aggregate_id")
        
        if self.created_at < 0:
            raise ValueError("Aggregate created_at timestamp cannot be negative")
        
        if not (0.0 <= self.temporal_coherence_threshold <= 1.0):
            raise ValueError("Temporal coherence threshold must be in [0.0, 1.0]")
        
        if self.max_retention_depth < 1 or self.max_protention_horizon < 1:
            raise ValueError("Retention depth and protention horizon must be positive")
        
        # Validate retention buffer temporal ordering
        if len(self.retention_buffer) > 1:
            for i in range(1, len(self.retention_buffer)):
                if (self.retention_buffer[i].original_timestamp <= 
                    self.retention_buffer[i-1].original_timestamp):
                    raise ValueError("Retention buffer must maintain temporal ordering")
    
    def _initialize_temporal_system(self) -> None:
        """Initialize temporal consciousness system."""
        # Initialize with minimal temporal structure
        if not self.current_moment:
            initial_impression = PrimalImpression(
                impression_content=jnp.zeros(10),  # Default size
                impression_timestamp=self.created_at,
                phenomenological_vividness=0.5,
                intentional_directedness=jnp.zeros(10),
                synthesis_readiness=0.7
            )
            
            self.current_moment = TemporalMomentEntity(
                created_at=self.created_at,
                primal_impression=initial_impression
            )
        
        # Initialize state container
        if self._state_container is None:
            initial_state = self.current_moment.synthesize_temporal_moment()
            self._state_container = ImmutableStateContainer(
                initial_state=initial_state,
                state_type=StateType.TEMPORAL
            )
        
        self.aggregate_state = AggregateState.COHERENT
    
    # ========================================================================
    # CORE BUSINESS OPERATIONS
    # ========================================================================
    
    def synthesize_temporal_moment(self) -> Array:
        """Synthesize current temporal moment with retention-present-protention.
        
        Core business operation for Husserlian temporal synthesis.
        This is the primary operation of the temporal consciousness aggregate.
        
        Returns:
            Synthesized temporal moment array
        """
        if self.aggregate_state == AggregateState.DISSIPATING:
            raise ValueError("Cannot synthesize temporal moment in dissipating aggregate")
        
        if not self.current_moment:
            raise ValueError("Cannot synthesize without current temporal moment")
        
        self.aggregate_state = AggregateState.PROCESSING
        
        # Perform synthesis through aggregate root
        synthesized_moment = self.current_moment.synthesize_temporal_moment()
        
        # Cache synthesis result
        self._synthesis_cache[self.current_moment.moment_id] = synthesized_moment
        
        # Update state container
        if self._state_container:
            self._state_container = self._state_container.evolve_state(
                new_state=synthesized_moment,
                evolution_type=StateEvolutionType.SYNTHESIS_INTEGRATION,
                event_data={'operation': 'temporal_synthesis', 'moment_id': self.current_moment.moment_id}
            )
        
        # Validate temporal coherence
        temporal_coherence = self.current_moment.temporal_coherence
        if temporal_coherence < self.temporal_coherence_threshold:
            # Log warning but continue - phenomenology allows for incoherent moments
            pass
        
        self.aggregate_state = AggregateState.COHERENT
        return synthesized_moment
    
    def evolve_temporal_flow(self, new_impression: PrimalImpression) -> TemporalMomentEntity:
        """Evolve temporal flow with new primal impression.
        
        Business operation for temporal flow evolution, managing retention
        buffer updates and protention fulfillment/disappointment.
        
        Returns:
            New temporal moment entity representing evolved flow
        """
        if not isinstance(new_impression, PrimalImpression):
            raise ValueError("New impression must be PrimalImpression value object")
        
        if self.aggregate_state == AggregateState.DISSIPATING:
            raise ValueError("Cannot evolve temporal flow in dissipating aggregate")
        
        if not self.current_moment:
            raise ValueError("Cannot evolve temporal flow without current moment")
        
        self.aggregate_state = AggregateState.PROCESSING
        
        # Update retention buffer with current moment
        if self.current_moment.primal_impression:
            retention_moment = self.current_moment.primal_impression.transition_to_retention()
            self.retention_buffer.insert(0, retention_moment)
            
            # Maintain retention buffer size
            if len(self.retention_buffer) > self.max_retention_depth:
                # Apply decay to older retentions and remove weakest
                decayed_retentions = []
                for retention in self.retention_buffer:
                    decayed = retention.decay_retention()
                    if decayed.is_phenomenologically_accessible():
                        decayed_retentions.append(decayed)
                
                self.retention_buffer = decayed_retentions[:self.max_retention_depth]
        
        # Process protention fulfillment
        fulfilled_protentions = []
        for protention in self.protention_projections:
            fulfillment_score, updated_protention = protention.fulfill_expectation(new_impression)
            if updated_protention.is_expectationally_active():
                fulfilled_protentions.append(updated_protention)
        
        self.protention_projections = fulfilled_protentions
        
        # Evolve current moment
        evolved_moment = self.current_moment.evolve_temporal_flow(new_impression)
        
        # Add current moment to history
        self.temporal_flow_history.append(self.current_moment.moment_id)
        if len(self.temporal_flow_history) > 50:  # Limit history size
            self.temporal_flow_history.pop(0)
        
        # Update aggregate root
        previous_moment = self.current_moment
        self.current_moment = evolved_moment
        
        # Update state container
        if self._state_container:
            new_state = evolved_moment.synthesize_temporal_moment()
            self._state_container = self._state_container.evolve_state(
                new_state=new_state,
                evolution_type=StateEvolutionType.CONTINUOUS_FLOW,
                event_data={
                    'operation': 'temporal_evolution',
                    'previous_moment_id': previous_moment.moment_id,
                    'new_moment_id': evolved_moment.moment_id
                }
            )
        
        self.aggregate_state = AggregateState.COHERENT
        return evolved_moment
    
    def project_protentional_horizon(
        self, 
        anticipated_content: Array, 
        anticipatory_distance: float,
        expectation_strength: float = 0.7
    ) -> ProtentionalHorizon:
        """Project new protentional horizon for future anticipation.
        
        Business operation for protentional projection following
        Husserlian analysis of temporal anticipation.
        
        Returns:
            New protentional horizon
        """
        if not jnp.all(jnp.isfinite(anticipated_content)):
            raise ValueError("Anticipated content must contain finite values")
        
        if anticipatory_distance <= 0:
            raise ValueError("Anticipatory distance must be positive")
        
        if not (0.0 <= expectation_strength <= 1.0):
            raise ValueError("Expectation strength must be in [0.0, 1.0]")
        
        if not self.current_moment:
            raise ValueError("Cannot project protention without current moment")
        
        # Calculate expectation timestamp
        current_timestamp = self.current_moment.primal_impression.impression_timestamp
        expectation_timestamp = current_timestamp + anticipatory_distance
        
        # Create protentional horizon
        protention = ProtentionalHorizon(
            anticipated_content=anticipated_content,
            expectation_timestamp=expectation_timestamp,
            expectation_strength=expectation_strength,
            anticipatory_distance=anticipatory_distance,
            phenomenological_grip=expectation_strength * 0.8,
            expectational_specificity=0.5
        )
        
        # Add to protention projections
        self.protention_projections.append(protention)
        
        # Maintain protention horizon size
        if len(self.protention_projections) > self.max_protention_horizon:
            # Remove weakest protentions
            self.protention_projections.sort(key=lambda p: p.expectation_strength, reverse=True)
            self.protention_projections = self.protention_projections[:self.max_protention_horizon]
        
        return protention
    
    def assess_temporal_coherence(self) -> float:
        """Assess overall temporal coherence of the aggregate.
        
        Business operation for temporal coherence assessment across
        retention-present-protention synthesis structure.
        
        Returns:
            Temporal coherence measure [0.0, 1.0]
        """
        if not self.current_moment:
            return 0.0
        
        # Current moment coherence
        moment_coherence = self.current_moment.temporal_coherence
        
        # Retention buffer coherence
        if self.retention_buffer:
            retention_strengths = [r.retention_strength for r in self.retention_buffer]
            retention_coherence = jnp.mean(jnp.array(retention_strengths))
        else:
            retention_coherence = 0.0
        
        # Protention projection coherence
        if self.protention_projections:
            protention_strengths = [p.expectation_strength for p in self.protention_projections]
            protention_coherence = jnp.mean(jnp.array(protention_strengths))
        else:
            protention_coherence = 0.0
        
        # State container coherence (if available)
        if self._state_container:
            container_coherence = 1.0  # Assume container is coherent
        else:
            container_coherence = 0.0
        
        # Weighted temporal coherence
        overall_coherence = (
            0.4 * moment_coherence +
            0.2 * retention_coherence +
            0.2 * protention_coherence +
            0.2 * container_coherence
        )
        
        return float(jnp.clip(overall_coherence, 0.0, 1.0))
    
    # ========================================================================
    # AGGREGATE MANAGEMENT OPERATIONS
    # ========================================================================
    
    def integrate_with_aggregate(self, other_aggregate: TemporalConsciousnessAggregate) -> float:
        """Integrate with another temporal consciousness aggregate.
        
        Business operation for temporal consciousness integration across
        multiple temporal streams or agents.
        
        Returns:
            Integration strength achieved
        """
        if not isinstance(other_aggregate, TemporalConsciousnessAggregate):
            raise ValueError("Can only integrate with other TemporalConsciousnessAggregate")
        
        if (self.aggregate_state == AggregateState.DISSIPATING or 
            other_aggregate.aggregate_state == AggregateState.DISSIPATING):
            return 0.0
        
        if not (self.current_moment and other_aggregate.current_moment):
            return 0.0
        
        self.aggregate_state = AggregateState.INTEGRATING
        other_aggregate.aggregate_state = AggregateState.INTEGRATING
        
        # Integrate current moments
        integration_strength = self.current_moment.integrate_with_moment(
            other_aggregate.current_moment
        )
        
        # Share retention and protention if integration is strong
        if integration_strength > 0.5:
            # Share most accessible retentions
            accessible_retentions = [
                r for r in other_aggregate.retention_buffer 
                if r.is_phenomenologically_accessible()
            ]
            self.retention_buffer.extend(accessible_retentions[:3])  # Limit sharing
            
            # Share active protentions
            active_protentions = [
                p for p in other_aggregate.protention_projections
                if p.is_expectationally_active()
            ]
            self.protention_projections.extend(active_protentions[:2])  # Limit sharing
            
            # Maintain buffer sizes
            if len(self.retention_buffer) > self.max_retention_depth:
                self.retention_buffer = self.retention_buffer[:self.max_retention_depth]
            
            if len(self.protention_projections) > self.max_protention_horizon:
                self.protention_projections = self.protention_projections[:self.max_protention_horizon]
        
        self.aggregate_state = AggregateState.COHERENT
        other_aggregate.aggregate_state = AggregateState.COHERENT
        
        return integration_strength
    
    def get_temporal_state_snapshot(self) -> Dict[str, Any]:
        """Get comprehensive snapshot of temporal consciousness state.
        
        Returns:
            Dictionary containing all temporal state information
        """
        snapshot = {
            'aggregate_id': self.aggregate_id,
            'aggregate_state': self.aggregate_state.value,
            'temporal_coherence': self.assess_temporal_coherence(),
            'current_moment_id': self.current_moment.moment_id if self.current_moment else None,
            'retention_buffer_size': len(self.retention_buffer),
            'protention_projections_count': len(self.protention_projections),
            'temporal_flow_history_length': len(self.temporal_flow_history),
        }
        
        if self.current_moment:
            snapshot['current_moment_state'] = self.current_moment.consciousness_state.value
            snapshot['current_moment_coherence'] = self.current_moment.temporal_coherence
            snapshot['integration_readiness'] = self.current_moment.integration_readiness
        
        if self._state_container:
            snapshot['state_container_type'] = self._state_container.state_type.value
            snapshot['cached_syntheses'] = len(self._synthesis_cache)
        
        return snapshot
    
    def transition_aggregate_state(self, new_state: AggregateState) -> bool:
        """Transition aggregate state with validation."""
        valid_transitions = {
            AggregateState.INITIALIZING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.COHERENT: [AggregateState.PROCESSING, AggregateState.INTEGRATING, AggregateState.DISSIPATING],
            AggregateState.PROCESSING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.INTEGRATING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.DISSIPATING: []  # Terminal state
        }
        
        if new_state not in valid_transitions[self.aggregate_state]:
            return False
        
        self.aggregate_state = new_state
        return True


# ============================================================================
# EMBODIED EXPERIENCE AGGREGATE (Merleau-Pontian Phenomenology)
# ============================================================================

@dataclass
class EmbodiedExperienceAggregate:
    """Merleau-Pontian embodied experience aggregate.
    
    Aggregate root for embodied consciousness functionality, managing
    body schema integration, motor intentionality, and tool incorporation.
    This aggregate maintains consistency boundary for embodied interactions.
    
    Invariants:
    - Body schema must remain coherent across modifications
    - Motor intentions must be grounded in current proprioceptive state
    - Tool incorporations must respect body boundary extension capacity
    - Tactile feedback must be temporally consistent with actions
    """
    
    # Aggregate Identity
    aggregate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: TimeStep = field(default_factory=time.time)
    
    # Aggregate Root Entity
    current_experience: Optional[EmbodiedExperienceEntity] = None
    
    # Internal Components
    body_schema_history: List[Array] = field(default_factory=list)
    motor_action_queue: List[Dict[str, Any]] = field(default_factory=list)
    tool_incorporation_registry: Dict[str, Array] = field(default_factory=dict)
    
    # Aggregate State
    aggregate_state: AggregateState = AggregateState.INITIALIZING
    embodiment_coherence_threshold: float = 0.3
    max_schema_history: int = 20
    max_action_queue_size: int = 10
    
    # Internal State Container Integration
    _state_container: Optional[ImmutableStateContainer] = field(default=None, init=False)
    _integration_cache: Dict[str, Array] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize aggregate and validate invariants."""
        self._validate_aggregate_invariants()
        if self.aggregate_state == AggregateState.INITIALIZING:
            self._initialize_embodiment_system()
    
    def _validate_aggregate_invariants(self) -> None:
        """Validate core aggregate business invariants."""
        if not self.aggregate_id:
            raise ValueError("EmbodiedExperienceAggregate must have unique aggregate_id")
        
        if self.created_at < 0:
            raise ValueError("Aggregate created_at timestamp cannot be negative")
        
        if not (0.0 <= self.embodiment_coherence_threshold <= 1.0):
            raise ValueError("Embodiment coherence threshold must be in [0.0, 1.0]")
        
        if self.max_schema_history < 1 or self.max_action_queue_size < 1:
            raise ValueError("History and queue sizes must be positive")
    
    def _initialize_embodiment_system(self) -> None:
        """Initialize embodied experience system."""
        # Initialize with minimal embodied structure
        if not self.current_experience:
            initial_proprioception = ProprioceptiveField(
                proprioceptive_map=jnp.ones(15) * 0.5,  # Default body map
                postural_configuration=jnp.zeros(15),
                kinesthetic_flow=jnp.zeros(15),
                proprioceptive_clarity=0.7,
                postural_stability=0.8
            )
            
            self.current_experience = EmbodiedExperienceEntity(
                created_at=self.created_at,
                proprioceptive_field=initial_proprioception
            )
        
        # Initialize state container
        if self._state_container is None and self.current_experience:
            initial_schema = self.current_experience.integrate_body_schema(
                proprioceptive_input=self.current_experience.proprioceptive_field.proprioceptive_map,
                motor_prediction=jnp.zeros_like(self.current_experience.proprioceptive_field.proprioceptive_map),
                tactile_feedback=jnp.zeros_like(self.current_experience.proprioceptive_field.proprioceptive_map)
            )
            
            self._state_container = ImmutableStateContainer(
                initial_state=initial_schema,
                state_type=StateType.EMBODIMENT
            )
        
        self.aggregate_state = AggregateState.COHERENT
    
    # ========================================================================
    # CORE BUSINESS OPERATIONS
    # ========================================================================
    
    def integrate_multimodal_embodiment(
        self,
        proprioceptive_input: Array,
        motor_prediction: Array,
        tactile_feedback: Array
    ) -> Array:
        """Integrate multimodal embodied input into unified body schema.
        
        Core business operation for Merleau-Pontian body schema integration.
        
        Returns:
            Integrated body schema representation
        """
        if not self.current_experience:
            raise ValueError("Cannot integrate embodiment without current experience")
        
        if self.aggregate_state == AggregateState.DISSIPATING:
            raise ValueError("Cannot integrate embodiment in dissipating aggregate")
        
        self.aggregate_state = AggregateState.PROCESSING
        
        # Perform integration through aggregate root
        integrated_schema = self.current_experience.integrate_body_schema(
            proprioceptive_input, motor_prediction, tactile_feedback
        )
        
        # Cache integration result
        self._integration_cache[self.current_experience.experience_id] = integrated_schema
        
        # Update body schema history
        self.body_schema_history.append(integrated_schema)
        if len(self.body_schema_history) > self.max_schema_history:
            self.body_schema_history.pop(0)
        
        # Update state container
        if self._state_container:
            self._state_container = self._state_container.evolve_state(
                new_state=integrated_schema,
                evolution_type=StateEvolutionType.SYNTHESIS_INTEGRATION,
                event_data={
                    'operation': 'embodiment_integration',
                    'experience_id': self.current_experience.experience_id
                }
            )
        
        # Validate embodiment coherence
        embodiment_coherence = self.current_experience.embodiment_coherence
        if embodiment_coherence < self.embodiment_coherence_threshold:
            # Log warning but continue - embodied experience can be incoherent
            pass
        
        self.aggregate_state = AggregateState.COHERENT
        return integrated_schema
    
    def generate_motor_intention(
        self, 
        goal_state: Array, 
        contextual_affordances: Array,
        action_urgency: float = 0.5
    ) -> Array:
        """Generate motor intention for embodied action.
        
        Business operation for motor intentionality generation following
        Merleau-Pontian motor intentionality theory.
        
        Returns:
            Generated motor intention vector
        """
        if not self.current_experience:
            raise ValueError("Cannot generate motor intention without current experience")
        
        if not (0.0 <= action_urgency <= 1.0):
            raise ValueError("Action urgency must be in [0.0, 1.0]")
        
        self.aggregate_state = AggregateState.PROCESSING
        
        # Generate motor intention through aggregate root
        motor_intention = self.current_experience.generate_motor_intention(
            goal_state, contextual_affordances
        )
        
        # Queue motor action if urgency is high
        if action_urgency > 0.7:
            action_item = {
                'motor_intention': motor_intention,
                'goal_state': goal_state,
                'affordances': contextual_affordances,
                'urgency': action_urgency,
                'timestamp': time.time(),
                'experience_id': self.current_experience.experience_id
            }
            
            self.motor_action_queue.append(action_item)
            
            # Maintain queue size
            if len(self.motor_action_queue) > self.max_action_queue_size:
                # Remove oldest action
                self.motor_action_queue.pop(0)
        
        # Update state container
        if self._state_container:
            self._state_container = self._state_container.evolve_state(
                new_state=motor_intention,
                evolution_type=StateEvolutionType.DISCRETE_UPDATE,
                event_data={
                    'operation': 'motor_intention_generation',
                    'action_urgency': action_urgency
                }
            )
        
        self.aggregate_state = AggregateState.COHERENT
        return motor_intention
    
    def incorporate_tool_extension(
        self, 
        tool_id: str,
        tool_representation: Array,
        incorporation_strategy: str = "gradual"
    ) -> bool:
        """Incorporate tool into body schema (Merleau-Pontian tool incorporation).
        
        Business operation for body schema extension through tool use.
        
        Returns:
            True if tool incorporation was successful
        """
        if not self.current_experience:
            raise ValueError("Cannot incorporate tool without current experience")
        
        if tool_id in self.tool_incorporation_registry:
            # Tool already incorporated - check if update is needed
            existing_tool = self.tool_incorporation_registry[tool_id]
            if jnp.allclose(existing_tool, tool_representation, rtol=1e-5):
                return True  # No change needed
        
        self.aggregate_state = AggregateState.PROCESSING
        
        # Attempt tool incorporation through aggregate root
        success = self.current_experience.extend_body_schema_through_tool(tool_representation)
        
        if success:
            # Register tool incorporation
            self.tool_incorporation_registry[tool_id] = tool_representation
            
            # Update state container
            if self._state_container:
                # Get updated body schema
                updated_schema = self.current_experience.body_boundary.boundary_contour
                
                self._state_container = self._state_container.evolve_state(
                    new_state=updated_schema,
                    evolution_type=StateEvolutionType.DISCRETE_UPDATE,
                    event_data={
                        'operation': 'tool_incorporation',
                        'tool_id': tool_id,
                        'incorporation_strategy': incorporation_strategy
                    }
                )
        
        self.aggregate_state = AggregateState.COHERENT
        return success
    
    def assess_action_readiness(self) -> Dict[str, float]:
        """Assess readiness for various types of embodied action.
        
        Business operation for action readiness assessment across
        different embodied action modalities.
        
        Returns:
            Dictionary of readiness measures for different action types
        """
        if not self.current_experience:
            return {'overall': 0.0}
        
        # Overall action readiness from entity
        overall_readiness = self.current_experience.assess_action_readiness()
        
        # Motor queue readiness
        if self.motor_action_queue:
            queue_urgencies = [action['urgency'] for action in self.motor_action_queue]
            queue_readiness = jnp.mean(jnp.array(queue_urgencies))
        else:
            queue_readiness = 0.0
        
        # Tool incorporation readiness
        if self.tool_incorporation_registry:
            # More tools = potentially more action capabilities but less readiness
            tool_factor = 1.0 / (1.0 + 0.1 * len(self.tool_incorporation_registry))
        else:
            tool_factor = 1.0
        
        # Schema stability readiness (based on history)
        if len(self.body_schema_history) >= 2:
            recent_schemas = jnp.array(self.body_schema_history[-3:])
            schema_variance = jnp.var(recent_schemas, axis=0)
            stability_readiness = 1.0 / (1.0 + jnp.mean(schema_variance))
        else:
            stability_readiness = 0.5
        
        return {
            'overall': float(overall_readiness),
            'motor_queue': float(queue_readiness),
            'tool_enhanced': float(overall_readiness * tool_factor),
            'schema_stability': float(stability_readiness),
            'integrated': float(jnp.mean(jnp.array([
                overall_readiness,
                queue_readiness * 0.5,
                stability_readiness
            ])))
        }
    
    # ========================================================================
    # AGGREGATE MANAGEMENT OPERATIONS
    # ========================================================================
    
    def execute_motor_action_queue(self, max_actions: int = 3) -> List[Dict[str, Any]]:
        """Execute queued motor actions based on urgency and readiness.
        
        Returns:
            List of executed action results
        """
        if not self.motor_action_queue:
            return []
        
        # Sort actions by urgency
        sorted_actions = sorted(self.motor_action_queue, key=lambda x: x['urgency'], reverse=True)
        
        executed_actions = []
        actions_to_execute = min(max_actions, len(sorted_actions))
        
        for i in range(actions_to_execute):
            action = sorted_actions[i]
            
            # Execute action (simplified - would involve actual motor control)
            execution_result = {
                'action': action,
                'execution_timestamp': time.time(),
                'success': True,  # Simplified success
                'embodiment_state': self.current_experience.embodiment_coherence if self.current_experience else 0.0
            }
            
            executed_actions.append(execution_result)
            
            # Remove from queue
            if action in self.motor_action_queue:
                self.motor_action_queue.remove(action)
        
        return executed_actions
    
    def get_embodiment_state_snapshot(self) -> Dict[str, Any]:
        """Get comprehensive snapshot of embodied experience state."""
        snapshot = {
            'aggregate_id': self.aggregate_id,
            'aggregate_state': self.aggregate_state.value,
            'embodiment_coherence': self.current_experience.embodiment_coherence if self.current_experience else 0.0,
            'schema_history_length': len(self.body_schema_history),
            'motor_queue_size': len(self.motor_action_queue),
            'incorporated_tools': list(self.tool_incorporation_registry.keys()),
            'action_readiness': self.assess_action_readiness(),
        }
        
        if self.current_experience:
            snapshot['experience_id'] = self.current_experience.experience_id
            snapshot['experience_state'] = self.current_experience.consciousness_state.value
            snapshot['integration_readiness'] = self.current_experience.integration_readiness
            
            if self.current_experience.body_boundary:
                snapshot['body_boundary_confidence'] = self.current_experience.body_boundary.boundary_confidence
                snapshot['extension_capacity'] = self.current_experience.body_boundary.extension_capacity
        
        return snapshot
    
    def transition_aggregate_state(self, new_state: AggregateState) -> bool:
        """Transition aggregate state with validation."""
        valid_transitions = {
            AggregateState.INITIALIZING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.COHERENT: [AggregateState.PROCESSING, AggregateState.INTEGRATING, AggregateState.DISSIPATING],
            AggregateState.PROCESSING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.INTEGRATING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.DISSIPATING: []  # Terminal state
        }
        
        if new_state not in valid_transitions[self.aggregate_state]:
            return False
        
        self.aggregate_state = new_state
        return True


# ============================================================================
# CIRCULAR CAUSALITY AGGREGATE (Varela-Maturana Theory)
# ============================================================================

@dataclass  
class CircularCausalityAggregate:
    """Varela-Maturana circular causality aggregate.
    
    Aggregate root for circular causality functionality, managing
    agent-environment structural coupling, autopoietic processes,
    and meaning emergence through recurrent interactions.
    
    Invariants:
    - Agent and environment must maintain structural congruence
    - Autopoietic processes must maintain organizational closure
    - Coupling cycles must preserve circular causality
    - Meaning emergence must be grounded in coupling history
    """
    
    # Aggregate Identity
    aggregate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: TimeStep = field(default_factory=time.time)
    
    # Aggregate Root Entity
    current_causality: Optional[CircularCausalityEntity] = None
    
    # Internal Components
    coupling_cycle_history: List[Dict[str, Any]] = field(default_factory=list)
    meaning_emergence_events: List[Dict[str, Any]] = field(default_factory=list)
    perturbation_response_patterns: Dict[str, Array] = field(default_factory=dict)
    
    # Aggregate State
    aggregate_state: AggregateState = AggregateState.INITIALIZING
    coupling_stability_threshold: float = 0.4
    max_cycle_history: int = 30
    max_meaning_events: int = 15
    
    # Internal State Container Integration
    _state_container: Optional[ImmutableStateContainer] = field(default=None, init=False)
    _causality_cache: Dict[str, Tuple[Array, Array]] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize aggregate and validate invariants."""
        self._validate_aggregate_invariants()
        if self.aggregate_state == AggregateState.INITIALIZING:
            self._initialize_circular_causality_system()
    
    def _validate_aggregate_invariants(self) -> None:
        """Validate core aggregate business invariants."""
        if not self.aggregate_id:
            raise ValueError("CircularCausalityAggregate must have unique aggregate_id")
        
        if self.created_at < 0:
            raise ValueError("Aggregate created_at timestamp cannot be negative")
        
        if not (0.0 <= self.coupling_stability_threshold <= 1.0):
            raise ValueError("Coupling stability threshold must be in [0.0, 1.0]")
    
    def _initialize_circular_causality_system(self) -> None:
        """Initialize circular causality system."""
        # Initialize with minimal coupling structure
        if not self.current_causality:
            initial_coupling = StructuralCoupling(
                agent_structure=jnp.ones(12) * 0.5,
                environment_structure=jnp.ones(12) * 0.4,
                coupling_history=jnp.zeros(12),
                interaction_frequency=0.1,
                structural_drift=jnp.zeros(12),
                meaning_potential=0.3
            )
            
            self.current_causality = CircularCausalityEntity(
                created_at=self.created_at,
                structural_coupling=initial_coupling
            )
        
        # Initialize state container
        if self._state_container is None and self.current_causality:
            # Execute initial causality cycle to get state
            agent_state, env_state, meaning = self.current_causality.execute_circular_causality_cycle(
                agent_perturbation=jnp.zeros(12),
                environmental_perturbation=jnp.zeros(12)
            )
            
            # Combine agent and environment states
            initial_state = jnp.concatenate([agent_state, env_state])
            
            self._state_container = ImmutableStateContainer(
                initial_state=initial_state,
                state_type=StateType.COUPLING
            )
        
        self.aggregate_state = AggregateState.COHERENT
    
    # ========================================================================
    # CORE BUSINESS OPERATIONS
    # ========================================================================
    
    def execute_circular_causality_cycle(
        self,
        agent_perturbation: Array,
        environmental_perturbation: Array,
        cycle_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Array, Array, MeaningEmergence]:
        """Execute complete circular causality cycle.
        
        Core business operation for Varela-Maturana circular causality
        with agent-environment mutual specification.
        
        Returns:
            Tuple of (new_agent_state, new_env_state, emergent_meaning)
        """
        if not self.current_causality:
            raise ValueError("Cannot execute causality cycle without current causality entity")
        
        if self.aggregate_state == AggregateState.DISSIPATING:
            raise ValueError("Cannot execute causality cycle in dissipating aggregate")
        
        self.aggregate_state = AggregateState.PROCESSING
        
        # Execute cycle through aggregate root
        agent_state, env_state, meaning_emergence = self.current_causality.execute_circular_causality_cycle(
            agent_perturbation, environmental_perturbation
        )
        
        # Cache causality result
        cache_key = f"{self.current_causality.causality_id}_{len(self.coupling_cycle_history)}"
        self._causality_cache[cache_key] = (agent_state, env_state)
        
        # Record cycle in history
        cycle_record = {
            'cycle_id': len(self.coupling_cycle_history),
            'timestamp': time.time(),
            'agent_perturbation': agent_perturbation,
            'environmental_perturbation': environmental_perturbation,
            'agent_state': agent_state,
            'env_state': env_state,
            'coupling_strength': self.current_causality.coupling_strength.assess_coupling_quality(),
            'context': cycle_context or {}
        }
        
        self.coupling_cycle_history.append(cycle_record)
        
        # Maintain history size
        if len(self.coupling_cycle_history) > self.max_cycle_history:
            self.coupling_cycle_history.pop(0)
        
        # Record meaning emergence if significant
        if meaning_emergence.emergence_strength > 0.3:
            meaning_event = {
                'event_id': len(self.meaning_emergence_events),
                'timestamp': time.time(),
                'emergence_strength': meaning_emergence.emergence_strength,
                'semantic_content': meaning_emergence.semantic_content,
                'enactive_significance': meaning_emergence.enactive_significance,
                'meaning_coherence': meaning_emergence.meaning_coherence,
                'cycle_id': len(self.coupling_cycle_history) - 1
            }
            
            self.meaning_emergence_events.append(meaning_event)
            
            # Maintain meaning events size
            if len(self.meaning_emergence_events) > self.max_meaning_events:
                self.meaning_emergence_events.pop(0)
        
        # Update perturbation response patterns
        perturbation_signature = jnp.concatenate([agent_perturbation, environmental_perturbation])
        response_signature = jnp.concatenate([agent_state, env_state])
        
        pattern_key = f"pattern_{len(self.perturbation_response_patterns)}"
        self.perturbation_response_patterns[pattern_key] = jnp.concatenate([
            perturbation_signature, response_signature
        ])
        
        # Limit pattern registry size
        if len(self.perturbation_response_patterns) > 20:
            # Remove oldest pattern
            oldest_key = min(self.perturbation_response_patterns.keys())
            del self.perturbation_response_patterns[oldest_key]
        
        # Update state container
        if self._state_container:
            combined_state = jnp.concatenate([agent_state, env_state])
            
            self._state_container = self._state_container.evolve_state(
                new_state=combined_state,
                evolution_type=StateEvolutionType.COUPLING_DYNAMICS,
                event_data={
                    'operation': 'circular_causality_cycle',
                    'cycle_id': len(self.coupling_cycle_history) - 1,
                    'meaning_emerged': meaning_emergence.emergence_strength > 0.3
                }
            )
        
        self.aggregate_state = AggregateState.COHERENT
        return agent_state, env_state, meaning_emergence
    
    def assess_coupling_stability(self, assessment_window: int = 10) -> Dict[str, float]:
        """Assess stability of circular causality coupling.
        
        Business operation for coupling stability assessment across
        recent coupling cycle history.
        
        Returns:
            Dictionary of stability measures
        """
        if not self.current_causality:
            return {'overall_stability': 0.0}
        
        # Entity-level stability
        entity_stability = self.current_causality.assess_coupling_stability()
        
        # History-based stability
        if len(self.coupling_cycle_history) < 3:
            history_stability = 0.0
        else:
            recent_cycles = self.coupling_cycle_history[-assessment_window:]
            
            # Assess coupling strength consistency
            coupling_strengths = [cycle['coupling_strength'] for cycle in recent_cycles]
            strength_variance = jnp.var(jnp.array(coupling_strengths))
            strength_stability = jnp.exp(-strength_variance)
            
            # Assess state evolution consistency
            if len(recent_cycles) >= 2:
                agent_states = jnp.array([cycle['agent_state'] for cycle in recent_cycles])
                env_states = jnp.array([cycle['env_state'] for cycle in recent_cycles])
                
                agent_variance = jnp.mean(jnp.var(agent_states, axis=0))
                env_variance = jnp.mean(jnp.var(env_states, axis=0))
                
                evolution_stability = jnp.exp(-0.5 * (agent_variance + env_variance))
            else:
                evolution_stability = 0.5
            
            history_stability = float(0.6 * strength_stability + 0.4 * evolution_stability)
        
        # Meaning emergence stability
        if len(self.meaning_emergence_events) >= 2:
            recent_meanings = self.meaning_emergence_events[-5:]
            emergence_strengths = [event['emergence_strength'] for event in recent_meanings]
            
            meaning_consistency = 1.0 - jnp.std(jnp.array(emergence_strengths)) / (jnp.mean(jnp.array(emergence_strengths)) + 1e-6)
            meaning_stability = float(jnp.clip(meaning_consistency, 0.0, 1.0))
        else:
            meaning_stability = 0.0
        
        # Pattern stability
        if len(self.perturbation_response_patterns) >= 3:
            pattern_similarity = self._assess_pattern_similarity()
            pattern_stability = float(pattern_similarity)
        else:
            pattern_stability = 0.0
        
        overall_stability = float(jnp.mean(jnp.array([
            entity_stability,
            history_stability,
            meaning_stability * 0.5,  # Lower weight for meaning stability
            pattern_stability * 0.3   # Lower weight for pattern stability
        ])))
        
        return {
            'overall_stability': overall_stability,
            'entity_stability': float(entity_stability),
            'history_stability': history_stability,
            'meaning_stability': meaning_stability,
            'pattern_stability': pattern_stability
        }
    
    def assess_meaning_emergence_potential(self, agent_concerns: Array) -> Dict[str, float]:
        """Assess potential for meaning emergence given agent concerns.
        
        Returns:
            Dictionary of meaning emergence potential measures
        """
        if not self.current_causality:
            return {'overall_potential': 0.0}
        
        # Entity-level potential
        entity_potential = self.current_causality.assess_meaning_emergence_potential(agent_concerns)
        
        # History-based potential
        if self.meaning_emergence_events:
            recent_events = self.meaning_emergence_events[-3:]
            
            # Assess meaning development trend
            if len(recent_events) >= 2:
                emergence_trend = recent_events[-1]['emergence_strength'] - recent_events[0]['emergence_strength']
                trend_potential = float(jnp.clip(0.5 + emergence_trend, 0.0, 1.0))
            else:
                trend_potential = recent_events[-1]['emergence_strength']
            
            # Assess meaning coherence trajectory
            coherence_values = [event['meaning_coherence'] for event in recent_events]
            coherence_potential = float(jnp.mean(jnp.array(coherence_values)))
        else:
            trend_potential = 0.0
            coherence_potential = 0.0
        
        # Coupling maturity potential
        if self.current_causality.structural_coupling:
            coupling_maturity = self.current_causality.structural_coupling.assess_coupling_maturity()
            maturity_potential = float(coupling_maturity)
        else:
            maturity_potential = 0.0
        
        overall_potential = float(jnp.mean(jnp.array([
            entity_potential,
            trend_potential * 0.7,
            coherence_potential * 0.8,
            maturity_potential
        ])))
        
        return {
            'overall_potential': overall_potential,
            'entity_potential': float(entity_potential),
            'trend_potential': trend_potential,
            'coherence_potential': coherence_potential,
            'maturity_potential': maturity_potential
        }
    
    def _assess_pattern_similarity(self) -> float:
        """Assess similarity in perturbation-response patterns."""
        if len(self.perturbation_response_patterns) < 2:
            return 0.0
        
        patterns = list(self.perturbation_response_patterns.values())
        similarities = []
        
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                if patterns[i].shape == patterns[j].shape:
                    correlation = jnp.corrcoef(patterns[i], patterns[j])[0, 1]
                    if not jnp.isnan(correlation):
                        similarities.append(correlation)
        
        if similarities:
            return float(jnp.mean(jnp.array(similarities)))
        else:
            return 0.0
    
    # ========================================================================
    # AGGREGATE MANAGEMENT OPERATIONS
    # ========================================================================
    
    def get_causality_state_snapshot(self) -> Dict[str, Any]:
        """Get comprehensive snapshot of circular causality state."""
        snapshot = {
            'aggregate_id': self.aggregate_id,
            'aggregate_state': self.aggregate_state.value,
            'coupling_cycle_count': len(self.coupling_cycle_history),
            'meaning_emergence_events': len(self.meaning_emergence_events),
            'perturbation_patterns': len(self.perturbation_response_patterns),
            'stability_assessment': self.assess_coupling_stability(),
        }
        
        if self.current_causality:
            snapshot['causality_id'] = self.current_causality.causality_id
            snapshot['causality_state'] = self.current_causality.consciousness_state.value
            snapshot['coupling_coherence'] = self.current_causality.coupling_coherence
            snapshot['integration_readiness'] = self.current_causality.integration_readiness
            
            if self.current_causality.coupling_strength:
                snapshot['coupling_quality'] = self.current_causality.coupling_strength.assess_coupling_quality()
            
            if self.current_causality.meaning_emergence:
                snapshot['current_meaning_strength'] = self.current_causality.meaning_emergence.emergence_strength
                snapshot['meaning_coherence'] = self.current_causality.meaning_emergence.meaning_coherence
        
        return snapshot
    
    def transition_aggregate_state(self, new_state: AggregateState) -> bool:
        """Transition aggregate state with validation."""
        valid_transitions = {
            AggregateState.INITIALIZING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.COHERENT: [AggregateState.PROCESSING, AggregateState.INTEGRATING, AggregateState.DISSIPATING],
            AggregateState.PROCESSING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.INTEGRATING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.DISSIPATING: []  # Terminal state
        }
        
        if new_state not in valid_transitions[self.aggregate_state]:
            return False
        
        self.aggregate_state = new_state
        return True


# ============================================================================
# EXPERIENTIAL MEMORY AGGREGATE (Phenomenological Memory Theory)
# ============================================================================

@dataclass
class ExperientialMemoryAggregate:
    """Phenomenological experiential memory aggregate.
    
    Aggregate root for experiential memory functionality, managing
    experience retention, sedimentation, associative recall,
    and memory coherence across temporal development.
    
    Invariants:
    - Traces must maintain connection to original experiential context
    - Sediment layers must maintain depth ordering
    - Associative links must be bidirectionally consistent
    - Recall operations must respect phenomenological accessibility
    """
    
    # Aggregate Identity
    aggregate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: TimeStep = field(default_factory=time.time)
    
    # Aggregate Root Entity
    current_memory: Optional[ExperientialMemoryEntity] = None
    
    # Internal Components
    retention_operation_log: List[Dict[str, Any]] = field(default_factory=list)
    recall_operation_log: List[Dict[str, Any]] = field(default_factory=list)
    sedimentation_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Aggregate State
    aggregate_state: AggregateState = AggregateState.INITIALIZING
    memory_coherence_threshold: float = 0.3
    max_operation_log_size: int = 50
    max_sedimentation_events: int = 20
    
    # Internal State Container Integration
    _state_container: Optional[ImmutableStateContainer] = field(default=None, init=False)
    _recall_cache: Dict[str, List[Tuple[str, ExperientialTrace, float]]] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize aggregate and validate invariants."""
        self._validate_aggregate_invariants()
        if self.aggregate_state == AggregateState.INITIALIZING:
            self._initialize_memory_system()
    
    def _validate_aggregate_invariants(self) -> None:
        """Validate core aggregate business invariants."""
        if not self.aggregate_id:
            raise ValueError("ExperientialMemoryAggregate must have unique aggregate_id")
        
        if self.created_at < 0:
            raise ValueError("Aggregate created_at timestamp cannot be negative")
        
        if not (0.0 <= self.memory_coherence_threshold <= 1.0):
            raise ValueError("Memory coherence threshold must be in [0.0, 1.0]")
    
    def _initialize_memory_system(self) -> None:
        """Initialize experiential memory system."""
        # Initialize with empty memory entity
        if not self.current_memory:
            self.current_memory = ExperientialMemoryEntity(created_at=self.created_at)
        
        # Initialize state container
        if self._state_container is None:
            # Use memory coherence as initial state
            initial_state = jnp.array([self.current_memory.memory_coherence])
            
            self._state_container = ImmutableStateContainer(
                initial_state=initial_state,
                state_type=StateType.EXPERIENTIAL
            )
        
        self.aggregate_state = AggregateState.COHERENT
    
    # ========================================================================
    # CORE BUSINESS OPERATIONS
    # ========================================================================
    
    def retain_experiential_content(
        self,
        experience_content: Array,
        experiential_context: Array,
        affective_resonance: float = 0.5,
        retention_priority: str = "normal"
    ) -> str:
        """Retain experiential content in memory system.
        
        Core business operation for phenomenological experience retention
        with sedimentation and associative link formation.
        
        Returns:
            ID of created experiential trace
        """
        if not self.current_memory:
            raise ValueError("Cannot retain experience without current memory entity")
        
        if self.aggregate_state == AggregateState.DISSIPATING:
            raise ValueError("Cannot retain experience in dissipating aggregate")
        
        self.aggregate_state = AggregateState.PROCESSING
        
        # Retain experience through aggregate root
        trace_id = self.current_memory.retain_experience(
            experience_content=experience_content,
            timestamp=time.time(),
            experiential_context=experiential_context,
            affective_resonance=affective_resonance
        )
        
        # Log retention operation
        retention_record = {
            'operation_id': len(self.retention_operation_log),
            'timestamp': time.time(),
            'trace_id': trace_id,
            'content_shape': experience_content.shape,
            'context_shape': experiential_context.shape,
            'affective_resonance': affective_resonance,
            'retention_priority': retention_priority,
            'memory_coherence_after': self.current_memory.memory_coherence
        }
        
        self.retention_operation_log.append(retention_record)
        
        # Maintain log size
        if len(self.retention_operation_log) > self.max_operation_log_size:
            self.retention_operation_log.pop(0)
        
        # Check if sedimentation occurred
        if (len(self.current_memory.sediment_layers) > 1 or 
            (self.current_memory.sediment_layers and 
             self.current_memory.sediment_layers[0].experiential_density > 0.5)):
            
            sedimentation_event = {
                'event_id': len(self.sedimentation_events),
                'timestamp': time.time(),
                'trigger_trace_id': trace_id,
                'sediment_layers_count': len(self.current_memory.sediment_layers),
                'top_layer_density': self.current_memory.sediment_layers[0].experiential_density if self.current_memory.sediment_layers else 0.0
            }
            
            self.sedimentation_events.append(sedimentation_event)
            
            # Maintain sedimentation events size
            if len(self.sedimentation_events) > self.max_sedimentation_events:
                self.sedimentation_events.pop(0)
        
        # Update state container
        if self._state_container:
            # Use memory capacity metrics as state representation
            capacity_metrics = self.current_memory.assess_memory_capacity()
            memory_state = jnp.array([
                capacity_metrics['trace_count'] / 100.0,  # Normalize
                capacity_metrics['sediment_depth'] / 10.0,
                capacity_metrics['memory_coherence'],
                capacity_metrics['retention_efficiency']
            ])
            
            self._state_container = self._state_container.evolve_state(
                new_state=memory_state,
                evolution_type=StateEvolutionType.DISCRETE_UPDATE,
                event_data={
                    'operation': 'experience_retention',
                    'trace_id': trace_id,
                    'retention_priority': retention_priority
                }
            )
        
        self.aggregate_state = AggregateState.COHERENT
        return trace_id
    
    def recall_experiential_content(
        self,
        recall_cue: Array,
        contextual_factors: Array,
        recall_mode: RecallMode = RecallMode.ASSOCIATIVE,
        max_recalls: int = 5,
        recall_depth: str = "normal"
    ) -> List[Tuple[str, ExperientialTrace, float]]:
        """Recall experiential content based on cues and context.
        
        Core business operation for phenomenological experiential recall
        using associative, temporal, or sedimentary recall modes.
        
        Returns:
            List of (trace_id, trace, recall_strength) tuples
        """
        if not self.current_memory:
            raise ValueError("Cannot recall experiences without current memory entity")
        
        # Create recall context
        recall_context = RecallContext(
            recall_cue=recall_cue,
            contextual_factors=contextual_factors,
            recall_mode=recall_mode,
            affective_state=jnp.zeros_like(contextual_factors),  # Placeholder
            temporal_proximity=0.7,  # Default proximity
            contextual_coherence=0.6  # Default coherence
        )
        
        # Check cache first
        cache_key = f"{hash(recall_cue.tobytes())}_{recall_mode.value}_{max_recalls}"
        if cache_key in self._recall_cache:
            cached_results = self._recall_cache[cache_key]
            # Validate cached results are still relevant
            if len(cached_results) <= max_recalls:
                return cached_results
        
        self.aggregate_state = AggregateState.PROCESSING
        
        # Perform recall through aggregate root
        recall_results = self.current_memory.recall_experiences(
            recall_context=recall_context,
            max_recalls=max_recalls
        )
        
        # Cache results
        self._recall_cache[cache_key] = recall_results
        
        # Maintain cache size
        if len(self._recall_cache) > 20:
            # Remove oldest cache entry
            oldest_key = next(iter(self._recall_cache))
            del self._recall_cache[oldest_key]
        
        # Log recall operation
        recall_record = {
            'operation_id': len(self.recall_operation_log),
            'timestamp': time.time(),
            'recall_mode': recall_mode.value,
            'cue_shape': recall_cue.shape,
            'context_shape': contextual_factors.shape,
            'max_recalls': max_recalls,
            'recall_depth': recall_depth,
            'results_count': len(recall_results),
            'avg_recall_strength': jnp.mean(jnp.array([r[2] for r in recall_results])) if recall_results else 0.0
        }
        
        self.recall_operation_log.append(recall_record)
        
        # Maintain log size
        if len(self.recall_operation_log) > self.max_operation_log_size:
            self.recall_operation_log.pop(0)
        
        # Update state container
        if self._state_container:
            # Encode recall operation in state
            recall_operation_state = jnp.array([
                len(recall_results) / float(max_recalls),
                recall_record['avg_recall_strength'],
                self.current_memory.memory_coherence,
                len(self.current_memory.experiential_traces) / 100.0  # Normalize trace count
            ])
            
            self._state_container = self._state_container.evolve_state(
                new_state=recall_operation_state,
                evolution_type=StateEvolutionType.DISCRETE_UPDATE,
                event_data={
                    'operation': 'experiential_recall',
                    'recall_mode': recall_mode.value,
                    'results_count': len(recall_results)
                }
            )
        
        self.aggregate_state = AggregateState.COHERENT
        return recall_results
    
    def assess_memory_integrity(self) -> Dict[str, float]:
        """Assess integrity and health of memory system.
        
        Business operation for comprehensive memory system assessment
        including trace accessibility, sediment organization, and associative coherence.
        
        Returns:
            Dictionary of memory integrity measures
        """
        if not self.current_memory:
            return {'overall_integrity': 0.0}
        
        # Entity-level capacity assessment
        capacity_metrics = self.current_memory.assess_memory_capacity()
        
        # Trace accessibility assessment
        if self.current_memory.experiential_traces:
            trace_accessibilities = [
                trace.accessibility_index * trace.trace_strength
                for trace in self.current_memory.experiential_traces.values()
            ]
            avg_accessibility = float(jnp.mean(jnp.array(trace_accessibilities)))
            accessibility_variance = float(jnp.var(jnp.array(trace_accessibilities)))
        else:
            avg_accessibility = 0.0
            accessibility_variance = 0.0
        
        # Sediment organization assessment
        if self.current_memory.sediment_layers:
            layer_coherences = [layer.layer_coherence for layer in self.current_memory.sediment_layers]
            sediment_organization = float(jnp.mean(jnp.array(layer_coherences)))
            
            # Check depth ordering
            depth_ordering = all(
                self.current_memory.sediment_layers[i].sediment_depth <= 
                self.current_memory.sediment_layers[i+1].sediment_depth
                for i in range(len(self.current_memory.sediment_layers) - 1)
            ) if len(self.current_memory.sediment_layers) > 1 else True
            
            depth_ordering_score = 1.0 if depth_ordering else 0.0
        else:
            sediment_organization = 0.0
            depth_ordering_score = 0.0
        
        # Associative link integrity assessment
        if self.current_memory.associative_links:
            all_link_strengths = []
            bidirectional_consistency = 0.0
            bidirectional_count = 0
            
            for source_id, links in self.current_memory.associative_links.items():
                for link in links:
                    all_link_strengths.append(link.association_strength)
                    
                    # Check bidirectional consistency
                    if link.bidirectional and link.target_trace_id in self.current_memory.associative_links:
                        target_links = self.current_memory.associative_links[link.target_trace_id]
                        reverse_link_exists = any(
                            tl.target_trace_id == source_id for tl in target_links
                        )
                        if reverse_link_exists:
                            bidirectional_consistency += 1.0
                        bidirectional_count += 1
            
            if all_link_strengths:
                associative_strength = float(jnp.mean(jnp.array(all_link_strengths)))
            else:
                associative_strength = 0.0
            
            if bidirectional_count > 0:
                bidirectional_ratio = bidirectional_consistency / bidirectional_count
            else:
                bidirectional_ratio = 1.0
        else:
            associative_strength = 0.0
            bidirectional_ratio = 1.0
        
        # Operations efficiency assessment
        if self.retention_operation_log and self.recall_operation_log:
            avg_retention_coherence = jnp.mean(jnp.array([
                op['memory_coherence_after'] for op in self.retention_operation_log[-10:]
            ]))
            
            avg_recall_success = jnp.mean(jnp.array([
                1.0 if op['results_count'] > 0 else 0.0 for op in self.recall_operation_log[-10:]
            ]))
            
            operations_efficiency = float(0.5 * avg_retention_coherence + 0.5 * avg_recall_success)
        else:
            operations_efficiency = 0.0
        
        # Overall integrity calculation
        overall_integrity = float(jnp.mean(jnp.array([
            capacity_metrics['memory_coherence'],
            avg_accessibility,
            1.0 - jnp.clip(accessibility_variance, 0.0, 1.0),  # Lower variance is better
            sediment_organization,
            depth_ordering_score,
            associative_strength,
            bidirectional_ratio,
            operations_efficiency
        ])))
        
        return {
            'overall_integrity': overall_integrity,
            'trace_accessibility': avg_accessibility,
            'accessibility_consistency': 1.0 - accessibility_variance,
            'sediment_organization': sediment_organization,
            'depth_ordering': depth_ordering_score,
            'associative_strength': associative_strength,
            'bidirectional_consistency': bidirectional_ratio,
            'operations_efficiency': operations_efficiency,
            'capacity_metrics': capacity_metrics
        }
    
    # ========================================================================
    # AGGREGATE MANAGEMENT OPERATIONS
    # ========================================================================
    
    def get_memory_state_snapshot(self) -> Dict[str, Any]:
        """Get comprehensive snapshot of experiential memory state."""
        snapshot = {
            'aggregate_id': self.aggregate_id,
            'aggregate_state': self.aggregate_state.value,
            'retention_operations': len(self.retention_operation_log),
            'recall_operations': len(self.recall_operation_log),
            'sedimentation_events': len(self.sedimentation_events),
            'cached_recalls': len(self._recall_cache),
            'memory_integrity': self.assess_memory_integrity(),
        }
        
        if self.current_memory:
            snapshot['memory_id'] = self.current_memory.memory_id
            snapshot['memory_state'] = self.current_memory.consciousness_state.value
            snapshot['memory_coherence'] = self.current_memory.memory_coherence
            snapshot['integration_readiness'] = self.current_memory.integration_readiness
            snapshot['trace_count'] = len(self.current_memory.experiential_traces)
            snapshot['sediment_layers'] = len(self.current_memory.sediment_layers)
            snapshot['associative_links_total'] = sum(
                len(links) for links in self.current_memory.associative_links.values()
            )
        
        return snapshot
    
    def transition_aggregate_state(self, new_state: AggregateState) -> bool:
        """Transition aggregate state with validation."""
        valid_transitions = {
            AggregateState.INITIALIZING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.COHERENT: [AggregateState.PROCESSING, AggregateState.INTEGRATING, AggregateState.DISSIPATING],
            AggregateState.PROCESSING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.INTEGRATING: [AggregateState.COHERENT, AggregateState.DISSIPATING],
            AggregateState.DISSIPATING: []  # Terminal state
        }
        
        if new_state not in valid_transitions[self.aggregate_state]:
            return False
        
        self.aggregate_state = new_state
        return True