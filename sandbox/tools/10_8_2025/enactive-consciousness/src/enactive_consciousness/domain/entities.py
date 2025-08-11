"""Domain Entities for Enactive Consciousness.

This module defines the core domain entities following Eric Evans' DDD methodology.
Entities have identity, lifecycle, and behavior that embodies the essential 
business rules of consciousness phenomena.

Theoretical Foundations:
- Entities represent core consciousness phenomena with identity
- Each entity maintains state and behavior consistent with theory
- Identity is essential - entities are distinguished by ID, not attributes
- Lifecycle management follows phenomenological temporal structure

Design Principles:
1. Rich domain behavior embedded in entities
2. Identity-based equality and lifecycle management  
3. Business invariants enforced through entity behavior
4. Ubiquitous language reflected in entity interface
5. Proper encapsulation of entity state and rules
"""

from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

import jax
import jax.numpy as jnp
from ..types import Array, TimeStep

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
)


class ConsciousnessState(Enum):
    """States of consciousness entity."""
    DORMANT = "dormant"
    FORMING = "forming"  
    ACTIVE = "active"
    INTEGRATING = "integrating"
    DISSIPATING = "dissipating"


# ============================================================================
# TEMPORAL CONSCIOUSNESS ENTITY (Husserlian Phenomenology)
# ============================================================================

@dataclass
class TemporalMomentEntity:
    """Husserlian temporal moment as domain entity with identity and lifecycle.
    
    Represents a complete temporal moment with retention-present-protention
    synthesis, maintaining identity across temporal evolution while embodying
    the essential business rules of phenomenological time consciousness.
    
    Business Rules:
    - Each temporal moment has unique identity and temporal position
    - Must maintain phenomenological consistency across evolution
    - Temporal synthesis follows Husserlian retention-protention structure
    - Moments can be recalled, modified, and integrated with others
    """
    
    # Entity Identity
    moment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: TimeStep = field(default_factory=lambda: 0.0)
    
    # Temporal Components (Value Objects)
    retention: RetentionMoment = None
    primal_impression: PrimalImpression = None  
    protention: ProtentionalHorizon = None
    synthesis_weights: TemporalSynthesisWeights = None
    
    # Entity State
    consciousness_state: ConsciousnessState = ConsciousnessState.FORMING
    temporal_coherence: float = 0.0  # [0.0, 1.0]
    integration_readiness: float = 0.0  # [0.0, 1.0]
    
    # Entity Behavior State
    _synthesis_history: List[Array] = field(default_factory=list, init=False)
    _temporal_modifications: int = field(default=0, init=False)
    _associated_moments: Set[str] = field(default_factory=set, init=False)
    
    def __post_init__(self):
        """Initialize entity and validate business invariants."""
        self._validate_entity_invariants()
        if self.consciousness_state == ConsciousnessState.FORMING:
            self._initialize_temporal_synthesis()
    
    def _validate_entity_invariants(self) -> None:
        """Validate core entity business invariants."""
        if not self.moment_id:
            raise ValueError("TemporalMomentEntity must have unique moment_id")
        
        if self.created_at < 0:
            raise ValueError("Entity created_at timestamp cannot be negative")
        
        measures = [self.temporal_coherence, self.integration_readiness]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Entity coherence measures must be in [0.0, 1.0]")
    
    def _initialize_temporal_synthesis(self) -> None:
        """Initialize temporal synthesis when entity is forming."""
        if self.primal_impression is None:
            raise ValueError("Cannot initialize synthesis without primal impression")
        
        # Initialize retention and protention if not provided
        if self.retention is None:
            self.retention = RetentionMoment(
                original_timestamp=self.primal_impression.impression_timestamp - 1.0,
                retained_content=jnp.zeros_like(self.primal_impression.impression_content),
                retention_strength=0.0,
                temporal_distance=1.0,
                phenomenological_clarity=0.0
            )
        
        if self.protention is None:
            self.protention = ProtentionalHorizon(
                anticipated_content=self.primal_impression.impression_content,
                expectation_timestamp=self.primal_impression.impression_timestamp + 1.0,
                expectation_strength=0.5,
                anticipatory_distance=1.0,
                phenomenological_grip=0.5,
                expectational_specificity=0.3
            )
        
        if self.synthesis_weights is None:
            self.synthesis_weights = TemporalSynthesisWeights(
                retention_weight=0.3,
                present_weight=0.5,
                protention_weight=0.2,
                synthesis_coherence=0.7,
                attentional_focus=jnp.ones_like(self.primal_impression.impression_content)
            )
        
        self.consciousness_state = ConsciousnessState.ACTIVE
        self._compute_temporal_coherence()
    
    def synthesize_temporal_moment(self) -> Array:
        """Perform Husserlian temporal synthesis of retention-present-protention.
        
        Core business logic for temporal consciousness synthesis following
        Husserl's analysis of internal time consciousness.
        
        Returns:
            Synthesized temporal moment as unified consciousness content
        """
        if self.consciousness_state not in [ConsciousnessState.ACTIVE, ConsciousnessState.INTEGRATING]:
            raise ValueError("Cannot synthesize temporal moment in current state")
        
        if not all([self.retention, self.primal_impression, self.protention, self.synthesis_weights]):
            raise ValueError("All temporal components must be present for synthesis")
        
        # Apply temporal synthesis weights
        retained_contribution = (
            self.synthesis_weights.retention_weight * 
            self.retention.retention_strength * 
            self.retention.retained_content
        )
        
        present_contribution = (
            self.synthesis_weights.present_weight * 
            self.primal_impression.phenomenological_vividness * 
            self.primal_impression.impression_content
        )
        
        protentional_contribution = (
            self.synthesis_weights.protention_weight * 
            self.protention.expectation_strength * 
            self.protention.anticipated_content
        )
        
        # Synthesize with attentional focus
        synthesized_moment = (
            retained_contribution + 
            present_contribution + 
            protentional_contribution
        ) * self.synthesis_weights.attentional_focus
        
        # Record synthesis in history
        self._synthesis_history.append(synthesized_moment)
        if len(self._synthesis_history) > 10:  # Keep last 10 syntheses
            self._synthesis_history.pop(0)
        
        self._compute_temporal_coherence()
        return synthesized_moment
    
    def evolve_temporal_flow(self, new_impression: PrimalImpression) -> TemporalMomentEntity:
        """Evolve temporal flow with new primal impression.
        
        Business logic for temporal flow evolution following phenomenological
        time consciousness where present becomes past and future becomes present.
        
        Returns:
            New TemporalMomentEntity representing evolved temporal flow
        """
        if self.consciousness_state == ConsciousnessState.DISSIPATING:
            raise ValueError("Cannot evolve dissipating temporal moment")
        
        # Current impression becomes retention
        new_retention = self.primal_impression.transition_to_retention()
        
        # Process protentional fulfillment/disappointment  
        if self.protention:
            fulfillment_score, updated_protention = self.protention.fulfill_expectation(new_impression)
            # Update protention based on fulfillment
            new_protention = updated_protention
        else:
            new_protention = ProtentionalHorizon(
                anticipated_content=new_impression.impression_content,
                expectation_timestamp=new_impression.impression_timestamp + 1.0,
                expectation_strength=0.5,
                anticipatory_distance=1.0,
                phenomenological_grip=0.5,
                expectational_specificity=0.3
            )
        
        # Adjust synthesis weights based on temporal situation
        retention_demand = new_retention.retention_strength
        protention_demand = new_protention.expectation_strength
        new_synthesis_weights = self.synthesis_weights.rebalance_for_temporal_situation(
            retention_demand, protention_demand
        )
        
        # Create evolved temporal moment
        evolved_moment = TemporalMomentEntity(
            created_at=new_impression.impression_timestamp,
            retention=new_retention,
            primal_impression=new_impression,
            protention=new_protention,
            synthesis_weights=new_synthesis_weights,
            consciousness_state=ConsciousnessState.ACTIVE
        )
        
        # Link to this moment
        evolved_moment._associated_moments.add(self.moment_id)
        self._associated_moments.add(evolved_moment.moment_id)
        
        return evolved_moment
    
    def integrate_with_moment(self, other_moment: TemporalMomentEntity) -> float:
        """Integrate with another temporal moment.
        
        Business logic for temporal moment integration in consciousness unity.
        
        Returns:
            Integration strength achieved
        """
        if not isinstance(other_moment, TemporalMomentEntity):
            raise ValueError("Can only integrate with other TemporalMomentEntity")
        
        if other_moment.consciousness_state == ConsciousnessState.DISSIPATING:
            return 0.0
        
        self.consciousness_state = ConsciousnessState.INTEGRATING
        other_moment.consciousness_state = ConsciousnessState.INTEGRATING
        
        # Assess temporal compatibility
        temporal_distance = abs(
            self.primal_impression.impression_timestamp - 
            other_moment.primal_impression.impression_timestamp
        )
        
        temporal_compatibility = jnp.exp(-0.1 * temporal_distance)
        
        # Assess content similarity
        if (self.primal_impression.impression_content.shape == 
            other_moment.primal_impression.impression_content.shape):
            
            content_correlation = jnp.corrcoef(
                self.primal_impression.impression_content.flatten(),
                other_moment.primal_impression.impression_content.flatten()
            )[0, 1]
            
            if jnp.isnan(content_correlation):
                content_similarity = 0.0
            else:
                content_similarity = jnp.clip(content_correlation, 0.0, 1.0)
        else:
            content_similarity = 0.0
        
        # Compute integration strength
        integration_strength = float(
            temporal_compatibility * 
            content_similarity * 
            self.temporal_coherence * 
            other_moment.temporal_coherence
        )
        
        # Update integration readiness
        self.integration_readiness = 0.7 * self.integration_readiness + 0.3 * integration_strength
        other_moment.integration_readiness = 0.7 * other_moment.integration_readiness + 0.3 * integration_strength
        
        # Add to associated moments
        self._associated_moments.add(other_moment.moment_id)
        other_moment._associated_moments.add(self.moment_id)
        
        return integration_strength
    
    def _compute_temporal_coherence(self) -> None:
        """Compute temporal coherence based on synthesis quality."""
        if not all([self.retention, self.primal_impression, self.protention, self.synthesis_weights]):
            self.temporal_coherence = 0.0
            return
        
        # Assess coherence across temporal components
        retention_coherence = (
            self.retention.retention_strength * 
            self.retention.phenomenological_clarity
        )
        
        present_coherence = (
            self.primal_impression.phenomenological_vividness * 
            self.primal_impression.synthesis_readiness
        )
        
        protention_coherence = (
            self.protention.expectation_strength * 
            self.protention.phenomenological_grip
        )
        
        synthesis_coherence = self.synthesis_weights.synthesis_coherence
        
        # Weighted temporal coherence
        self.temporal_coherence = float(jnp.clip(
            0.25 * retention_coherence +
            0.4 * present_coherence +
            0.25 * protention_coherence +
            0.1 * synthesis_coherence,
            0.0, 1.0
        ))
    
    def assess_phenomenological_accessibility(self) -> float:
        """Assess how accessible this temporal moment is to consciousness."""
        if self.consciousness_state == ConsciousnessState.DISSIPATING:
            return 0.0
        
        state_accessibility = {
            ConsciousnessState.DORMANT: 0.1,
            ConsciousnessState.FORMING: 0.3,
            ConsciousnessState.ACTIVE: 1.0,
            ConsciousnessState.INTEGRATING: 0.8,
            ConsciousnessState.DISSIPATING: 0.0
        }
        
        base_accessibility = state_accessibility[self.consciousness_state]
        
        return float(base_accessibility * self.temporal_coherence * self.integration_readiness)
    
    def transition_state(self, new_state: ConsciousnessState) -> bool:
        """Transition entity state with business rule validation."""
        valid_transitions = {
            ConsciousnessState.DORMANT: [ConsciousnessState.FORMING],
            ConsciousnessState.FORMING: [ConsciousnessState.ACTIVE, ConsciousnessState.DISSIPATING],
            ConsciousnessState.ACTIVE: [ConsciousnessState.INTEGRATING, ConsciousnessState.DISSIPATING],
            ConsciousnessState.INTEGRATING: [ConsciousnessState.ACTIVE, ConsciousnessState.DISSIPATING],
            ConsciousnessState.DISSIPATING: []  # Terminal state
        }
        
        if new_state not in valid_transitions[self.consciousness_state]:
            return False
        
        self.consciousness_state = new_state
        return True
    
    def __eq__(self, other) -> bool:
        """Entity equality based on identity, not attributes."""
        if not isinstance(other, TemporalMomentEntity):
            return False
        return self.moment_id == other.moment_id
    
    def __hash__(self) -> int:
        """Hash based on entity identity."""
        return hash(self.moment_id)


# ============================================================================
# EMBODIED EXPERIENCE ENTITY (Merleau-Pontian Phenomenology)
# ============================================================================

@dataclass
class EmbodiedExperienceEntity:
    """Merleau-Pontian embodied experience as domain entity.
    
    Represents a complete embodied experience with body schema, motor
    intentionality, and proprioceptive integration, maintaining identity
    across embodied interactions while embodying phenomenological rules.
    
    Business Rules:
    - Embodied experience has identity and develops over interaction history  
    - Body schema must maintain coherence across modifications
    - Motor intentions must be grounded in current body schema
    - Proprioceptive integration provides foundational body awareness
    """
    
    # Entity Identity
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: TimeStep = field(default_factory=lambda: 0.0)
    
    # Embodiment Components (Value Objects)
    body_boundary: BodyBoundary = None
    motor_intention: MotorIntention = None
    proprioceptive_field: ProprioceptiveField = None
    tactile_feedback: TactileFeedback = None
    
    # Entity State
    consciousness_state: ConsciousnessState = ConsciousnessState.FORMING
    embodiment_coherence: float = 0.0  # [0.0, 1.0]
    integration_readiness: float = 0.0  # [0.0, 1.0]
    
    # Entity Behavior State
    _interaction_history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _schema_modifications: int = field(default=0, init=False)
    _associated_experiences: Set[str] = field(default_factory=set, init=False)
    
    def __post_init__(self):
        """Initialize entity and validate business invariants."""
        self._validate_entity_invariants()
        if self.consciousness_state == ConsciousnessState.FORMING:
            self._initialize_embodied_integration()
    
    def _validate_entity_invariants(self) -> None:
        """Validate core entity business invariants."""
        if not self.experience_id:
            raise ValueError("EmbodiedExperienceEntity must have unique experience_id")
        
        if self.created_at < 0:
            raise ValueError("Entity created_at timestamp cannot be negative")
        
        measures = [self.embodiment_coherence, self.integration_readiness]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Entity coherence measures must be in [0.0, 1.0]")
    
    def _initialize_embodied_integration(self) -> None:
        """Initialize embodied integration when entity is forming."""
        if self.proprioceptive_field is None:
            raise ValueError("Cannot initialize embodiment without proprioceptive field")
        
        # Initialize body boundary if not provided
        if self.body_boundary is None:
            self.body_boundary = BodyBoundary(
                boundary_contour=jnp.ones_like(self.proprioceptive_field.proprioceptive_map),
                boundary_confidence=0.5,
                permeability_index=0.3,
                extension_capacity=0.7,
                proprioceptive_grounding=self.proprioceptive_field.proprioceptive_map
            )
        
        # Initialize motor intention if not provided
        if self.motor_intention is None:
            self.motor_intention = MotorIntention(
                motor_vector=jnp.zeros_like(self.proprioceptive_field.proprioceptive_map),
                action_readiness=0.0,
                motor_schema_activation=jnp.zeros_like(self.proprioceptive_field.proprioceptive_map),
                intentional_directedness=jnp.zeros_like(self.proprioceptive_field.proprioceptive_map),
                embodied_confidence=0.5
            )
        
        self.consciousness_state = ConsciousnessState.ACTIVE
        self._compute_embodiment_coherence()
    
    def integrate_body_schema(
        self, 
        proprioceptive_input: Array,
        motor_prediction: Array,
        tactile_feedback: Array
    ) -> Array:
        """Integrate body schema from multi-modal embodied input.
        
        Core business logic for Merleau-Pontian body schema integration
        combining proprioception, motor prediction, and tactile feedback.
        
        Returns:
            Integrated body schema representation
        """
        if self.consciousness_state not in [ConsciousnessState.ACTIVE, ConsciousnessState.INTEGRATING]:
            raise ValueError("Cannot integrate body schema in current state")
        
        arrays = [proprioceptive_input, motor_prediction, tactile_feedback]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("All input arrays must contain finite values")
        
        # Update proprioceptive field
        proprioceptive_delta = proprioceptive_input - self.proprioceptive_field.proprioceptive_map
        self.proprioceptive_field = self.proprioceptive_field.integrate_kinesthetic_change(proprioceptive_delta)
        
        # Update tactile feedback
        self.tactile_feedback = TactileFeedback(
            tactile_pattern=tactile_feedback,
            contact_pressure=jnp.abs(tactile_feedback),  # Pressure from tactile intensity
            tactile_confidence=0.8,
            feedback_timeliness=0.9,
            environmental_texture=0.3 * tactile_feedback  # Derived texture
        )
        
        # Generate motor correction from tactile feedback
        if self.tactile_feedback:
            motor_correction = self.tactile_feedback.generate_motor_correction(motor_prediction)
            
            # Update motor intention with correction
            corrected_motor_vector = self.motor_intention.motor_vector + 0.1 * motor_correction
            
            self.motor_intention = MotorIntention(
                motor_vector=corrected_motor_vector,
                action_readiness=self.motor_intention.action_readiness * 1.05,
                motor_schema_activation=self.motor_intention.motor_schema_activation + 0.05 * motor_correction,
                intentional_directedness=self.motor_intention.intentional_directedness,
                embodied_confidence=self.motor_intention.embodied_confidence * 1.02
            )
        
        # Integrate into unified body schema
        proprioceptive_contribution = 0.4 * self.proprioceptive_field.proprioceptive_map
        motor_contribution = 0.3 * self.motor_intention.motor_vector
        tactile_contribution = 0.3 * tactile_feedback if self.tactile_feedback else jnp.zeros_like(proprioceptive_input)
        
        integrated_schema = proprioceptive_contribution + motor_contribution + tactile_contribution
        
        # Record integration in history
        self._interaction_history.append({
            'timestamp': self.created_at,
            'integration_type': 'body_schema',
            'inputs': {
                'proprioceptive': proprioceptive_input,
                'motor': motor_prediction, 
                'tactile': tactile_feedback
            },
            'result': integrated_schema
        })
        
        if len(self._interaction_history) > 20:  # Keep last 20 interactions
            self._interaction_history.pop(0)
        
        self._compute_embodiment_coherence()
        return integrated_schema
    
    def generate_motor_intention(self, goal_state: Array, contextual_affordances: Array) -> Array:
        """Generate motor intention from goal state and affordances.
        
        Business logic for Merleau-Pontian motor intentionality generation
        grounded in current embodied state and environmental affordances.
        
        Returns:
            Generated motor intention vector
        """
        if self.consciousness_state == ConsciousnessState.DISSIPATING:
            raise ValueError("Cannot generate motor intention in dissipating state")
        
        arrays = [goal_state, contextual_affordances]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Goal state and affordances must contain finite values")
        
        # Current proprioceptive state influences motor intention
        current_state = self.proprioceptive_field.proprioceptive_map
        
        # Compute goal direction
        goal_direction = goal_state - current_state
        goal_distance = jnp.linalg.norm(goal_direction)
        
        if goal_distance < 1e-6:
            return jnp.zeros_like(goal_state)
        
        # Normalize goal direction
        goal_direction = goal_direction / goal_distance
        
        # Modulate motor intention for contextual affordances
        modulation_strength = jnp.mean(contextual_affordances)
        self.motor_intention = self.motor_intention.modulate_for_context(
            contextual_affordances, 
            float(jnp.clip(modulation_strength, 0.0, 1.0))
        )
        
        # Generate motor intention vector
        intention_magnitude = jnp.tanh(goal_distance) * self.motor_intention.embodied_confidence
        intention_vector = intention_magnitude * goal_direction + 0.1 * self.motor_intention.motor_vector
        
        # Update motor intention state
        self.motor_intention = MotorIntention(
            motor_vector=intention_vector,
            action_readiness=self.motor_intention.action_readiness * 1.1,
            motor_schema_activation=self.motor_intention.motor_schema_activation + 0.1 * intention_vector,
            intentional_directedness=goal_direction,
            embodied_confidence=self.motor_intention.embodied_confidence
        )
        
        # Record motor intention generation
        self._interaction_history.append({
            'timestamp': self.created_at,
            'integration_type': 'motor_intention',
            'inputs': {
                'goal_state': goal_state,
                'affordances': contextual_affordances
            },
            'result': intention_vector
        })
        
        self._compute_embodiment_coherence()
        return intention_vector
    
    def extend_body_schema_through_tool(self, tool_representation: Array) -> bool:
        """Extend body schema through tool use (Merleau-Pontian tool incorporation).
        
        Business logic for body schema extension through tool incorporation,
        following Merleau-Ponty's analysis of tool use and body extension.
        
        Returns:
            True if tool incorporation was successful
        """
        if not jnp.all(jnp.isfinite(tool_representation)):
            raise ValueError("Tool representation must contain finite values")
        
        if self.body_boundary.extension_capacity < 0.1:
            return False  # Insufficient capacity for extension
        
        # Assess tool compatibility with current body schema
        tool_norm = jnp.linalg.norm(tool_representation)
        schema_norm = jnp.linalg.norm(self.proprioceptive_field.proprioceptive_map)
        
        if tool_norm < 1e-6 or schema_norm < 1e-6:
            return False
        
        compatibility = jnp.dot(
            tool_representation / tool_norm,
            self.proprioceptive_field.proprioceptive_map / schema_norm
        )
        
        extension_strength = float(jnp.clip(compatibility * 0.7, 0.0, 1.0))
        
        if extension_strength < 0.3:
            return False  # Tool not sufficiently compatible
        
        # Extend body boundary through tool
        self.body_boundary = self.body_boundary.extend_through_tool(
            tool_representation, extension_strength
        )
        
        self._schema_modifications += 1
        
        # Record tool incorporation
        self._interaction_history.append({
            'timestamp': self.created_at,
            'integration_type': 'tool_incorporation',
            'inputs': {
                'tool_representation': tool_representation,
                'extension_strength': extension_strength
            },
            'result': self.body_boundary.boundary_contour
        })
        
        self._compute_embodiment_coherence()
        return True
    
    def _compute_embodiment_coherence(self) -> None:
        """Compute embodiment coherence based on component integration."""
        if not all([self.body_boundary, self.proprioceptive_field]):
            self.embodiment_coherence = 0.0
            return
        
        # Boundary coherence
        boundary_coherence = self.body_boundary.assess_boundary_coherence()
        
        # Proprioceptive coherence
        proprioceptive_coherence = self.proprioceptive_field.assess_proprioceptive_coherence()
        
        # Motor coherence (if motor intention exists)
        if self.motor_intention:
            motor_coherence = self.motor_intention.assess_motor_coherence()
        else:
            motor_coherence = 0.5
        
        # Tactile coherence (if tactile feedback exists)  
        if self.tactile_feedback:
            tactile_coherence = self.tactile_feedback.assess_contact_quality()
        else:
            tactile_coherence = 0.5
        
        # Weighted embodiment coherence
        self.embodiment_coherence = float(jnp.clip(
            0.3 * boundary_coherence +
            0.3 * proprioceptive_coherence +
            0.2 * motor_coherence +
            0.2 * tactile_coherence,
            0.0, 1.0
        ))
    
    def assess_action_readiness(self) -> float:
        """Assess readiness for embodied action."""
        if self.consciousness_state == ConsciousnessState.DISSIPATING:
            return 0.0
        
        if not self.motor_intention:
            return 0.0
        
        motor_readiness = self.motor_intention.action_readiness * self.motor_intention.embodied_confidence
        schema_readiness = self.embodiment_coherence
        
        return float(motor_readiness * schema_readiness)
    
    def transition_state(self, new_state: ConsciousnessState) -> bool:
        """Transition entity state with business rule validation."""
        valid_transitions = {
            ConsciousnessState.DORMANT: [ConsciousnessState.FORMING],
            ConsciousnessState.FORMING: [ConsciousnessState.ACTIVE, ConsciousnessState.DISSIPATING],
            ConsciousnessState.ACTIVE: [ConsciousnessState.INTEGRATING, ConsciousnessState.DISSIPATING],
            ConsciousnessState.INTEGRATING: [ConsciousnessState.ACTIVE, ConsciousnessState.DISSIPATING],
            ConsciousnessState.DISSIPATING: []  # Terminal state
        }
        
        if new_state not in valid_transitions[self.consciousness_state]:
            return False
        
        self.consciousness_state = new_state
        return True
    
    def __eq__(self, other) -> bool:
        """Entity equality based on identity, not attributes."""
        if not isinstance(other, EmbodiedExperienceEntity):
            return False
        return self.experience_id == other.experience_id
    
    def __hash__(self) -> int:
        """Hash based on entity identity."""
        return hash(self.experience_id)


# ============================================================================
# CIRCULAR CAUSALITY ENTITY (Varela-Maturana Theory)
# ============================================================================

@dataclass
class CircularCausalityEntity:
    """Varela-Maturana circular causality as domain entity.
    
    Represents a complete circular causality cycle with agent-environment
    structural coupling, autopoietic processes, and meaning emergence,
    maintaining identity across coupling cycles.
    
    Business Rules:
    - Circular causality has identity and develops through coupling cycles
    - Agent and environment must maintain mutual specification
    - Autopoietic processes must maintain organizational closure
    - Meaning emerges from structural coupling history
    """
    
    # Entity Identity
    causality_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: TimeStep = field(default_factory=lambda: 0.0)
    
    # Circular Causality Components (Value Objects)
    coupling_strength: CouplingStrength = None
    autopoietic_process: AutopoeticProcess = None
    structural_coupling: StructuralCoupling = None
    meaning_emergence: MeaningEmergence = None
    
    # Entity State
    consciousness_state: ConsciousnessState = ConsciousnessState.FORMING
    coupling_coherence: float = 0.0  # [0.0, 1.0]
    integration_readiness: float = 0.0  # [0.0, 1.0]
    
    # Entity Behavior State
    _coupling_cycles: int = field(default=0, init=False)
    _perturbation_history: List[Array] = field(default_factory=list, init=False)
    _associated_causalities: Set[str] = field(default_factory=set, init=False)
    
    def __post_init__(self):
        """Initialize entity and validate business invariants."""
        self._validate_entity_invariants()
        if self.consciousness_state == ConsciousnessState.FORMING:
            self._initialize_circular_coupling()
    
    def _validate_entity_invariants(self) -> None:
        """Validate core entity business invariants."""
        if not self.causality_id:
            raise ValueError("CircularCausalityEntity must have unique causality_id")
        
        if self.created_at < 0:
            raise ValueError("Entity created_at timestamp cannot be negative")
        
        measures = [self.coupling_coherence, self.integration_readiness]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Entity coherence measures must be in [0.0, 1.0]")
    
    def _initialize_circular_coupling(self) -> None:
        """Initialize circular coupling when entity is forming."""
        if self.structural_coupling is None:
            raise ValueError("Cannot initialize circular causality without structural coupling")
        
        # Initialize coupling strength if not provided
        if self.coupling_strength is None:
            self.coupling_strength = CouplingStrength(
                coupling_intensity=0.5,
                mutual_specification=0.4,
                structural_congruence=0.5,
                coupling_stability=0.3,
                perturbation_sensitivity=jnp.ones(5) * 0.5  # 5 perturbation types
            )
        
        # Initialize autopoietic process if not provided  
        if self.autopoietic_process is None:
            from .value_objects import AutopoeticProcessType
            self.autopoietic_process = AutopoeticProcess(
                process_type=AutopoeticProcessType.ORGANIZATIONAL_CLOSURE,
                organizational_closure=0.6,
                self_production_rate=0.5,
                boundary_integrity=0.7,
                component_coherence=jnp.ones_like(self.structural_coupling.agent_structure) * 0.5,
                autonomy_measure=0.6
            )
        
        self.consciousness_state = ConsciousnessState.ACTIVE
        self._compute_coupling_coherence()
    
    def execute_circular_causality_cycle(
        self,
        agent_perturbation: Array,
        environmental_perturbation: Array
    ) -> Tuple[Array, Array, MeaningEmergence]:
        """Execute one complete circular causality cycle.
        
        Core business logic for Varela-Maturana circular causality where
        agent and environment mutually specify each other through
        recurrent structural coupling interactions.
        
        Returns:
            Tuple of (new_agent_state, new_env_state, emergent_meaning)
        """
        if self.consciousness_state not in [ConsciousnessState.ACTIVE, ConsciousnessState.INTEGRATING]:
            raise ValueError("Cannot execute circular causality cycle in current state")
        
        arrays = [agent_perturbation, environmental_perturbation]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Perturbations must contain finite values")
        
        # Update structural coupling through interaction
        self.structural_coupling = self.structural_coupling.update_coupling_interaction(
            agent_perturbation, environmental_perturbation
        )
        
        # Evolve autopoietic process in response to perturbations
        total_perturbation = 0.5 * (agent_perturbation + environmental_perturbation)
        self.autopoietic_process = self.autopoietic_process.evolve_autopoietic_cycle(total_perturbation)
        
        # Modulate coupling strength based on perturbation magnitude
        perturbation_magnitude = jnp.linalg.norm(total_perturbation)
        self.coupling_strength = self.coupling_strength.modulate_for_perturbation(
            float(perturbation_magnitude)
        )
        
        # Generate/update meaning emergence
        if self.meaning_emergence is None:
            # Initialize meaning emergence
            self.meaning_emergence = MeaningEmergence(
                semantic_content=0.1 * total_perturbation,
                emergence_strength=0.3,
                contextual_grounding=self.structural_coupling.coupling_history,
                temporal_development=total_perturbation,
                enactive_significance=0.4,
                meaning_coherence=0.5
            )
        else:
            # Develop existing meaning through temporal context
            self.meaning_emergence = self.meaning_emergence.develop_meaning_temporally(total_perturbation)
        
        # Compute new agent and environment states through circular causality
        coupling_influence = self.coupling_strength.coupling_intensity
        autopoietic_influence = self.autopoietic_process.self_production_rate
        
        # Agent state influenced by environment through coupling
        agent_coupling_effect = coupling_influence * environmental_perturbation
        agent_autopoietic_effect = autopoietic_influence * self.autopoietic_process.component_coherence
        new_agent_state = (
            self.structural_coupling.agent_structure + 
            0.3 * agent_coupling_effect +
            0.2 * agent_autopoietic_effect
        )
        
        # Environment state influenced by agent through coupling
        env_coupling_effect = coupling_influence * agent_perturbation
        new_env_state = (
            self.structural_coupling.environment_structure +
            0.2 * env_coupling_effect +
            0.1 * self.structural_coupling.structural_drift
        )
        
        # Record perturbation in history
        self._perturbation_history.append(total_perturbation)
        if len(self._perturbation_history) > 15:  # Keep last 15 perturbations
            self._perturbation_history.pop(0)
        
        self._coupling_cycles += 1
        self._compute_coupling_coherence()
        
        return new_agent_state, new_env_state, self.meaning_emergence
    
    def assess_coupling_stability(self) -> float:
        """Assess stability of circular causality coupling.
        
        Business logic for assessing coupling stability based on
        perturbation history and structural coupling maturity.
        
        Returns:
            Coupling stability measure [0.0, 1.0]
        """
        if len(self._perturbation_history) < 3:
            return 0.0  # Need history for stability assessment
        
        # Assess perturbation consistency
        recent_perturbations = jnp.array(self._perturbation_history[-5:])
        perturbation_variance = jnp.var(jnp.linalg.norm(recent_perturbations, axis=1))
        perturbation_consistency = jnp.exp(-perturbation_variance)
        
        # Assess structural coupling maturity
        coupling_maturity = self.structural_coupling.assess_coupling_maturity()
        
        # Assess autopoietic viability
        autopoietic_stability = self.autopoietic_process.assess_autopoietic_viability()
        
        # Combined stability measure
        stability = float(jnp.mean(jnp.array([
            perturbation_consistency,
            coupling_maturity,
            autopoietic_stability,
            self.coupling_strength.coupling_stability
        ])))
        
        return jnp.clip(stability, 0.0, 1.0)
    
    def assess_meaning_emergence_potential(self, agent_concerns: Array) -> float:
        """Assess potential for meaning emergence given agent concerns.
        
        Returns:
            Meaning emergence potential [0.0, 1.0]
        """
        if self.meaning_emergence is None:
            return self.structural_coupling.meaning_potential
        
        relevance = self.meaning_emergence.assess_enactive_relevance(agent_concerns)
        emergence_strength = self.meaning_emergence.emergence_strength
        coupling_quality = self.coupling_strength.assess_coupling_quality()
        
        return float(jnp.clip(
            relevance * emergence_strength * coupling_quality,
            0.0, 1.0
        ))
    
    def _compute_coupling_coherence(self) -> None:
        """Compute coupling coherence based on component integration."""
        if not all([self.coupling_strength, self.structural_coupling]):
            self.coupling_coherence = 0.0
            return
        
        # Coupling strength coherence
        strength_coherence = self.coupling_strength.assess_coupling_quality()
        
        # Structural coupling coherence
        coupling_maturity = self.structural_coupling.assess_coupling_maturity()
        
        # Autopoietic coherence
        if self.autopoietic_process:
            autopoietic_coherence = self.autopoietic_process.assess_autopoietic_viability()
        else:
            autopoietic_coherence = 0.5
        
        # Meaning coherence
        if self.meaning_emergence:
            meaning_coherence = self.meaning_emergence.meaning_coherence
        else:
            meaning_coherence = 0.5
        
        # Weighted coupling coherence
        self.coupling_coherence = float(jnp.clip(
            0.3 * strength_coherence +
            0.3 * coupling_maturity +
            0.2 * autopoietic_coherence +
            0.2 * meaning_coherence,
            0.0, 1.0
        ))
    
    def transition_state(self, new_state: ConsciousnessState) -> bool:
        """Transition entity state with business rule validation."""
        valid_transitions = {
            ConsciousnessState.DORMANT: [ConsciousnessState.FORMING],
            ConsciousnessState.FORMING: [ConsciousnessState.ACTIVE, ConsciousnessState.DISSIPATING],
            ConsciousnessState.ACTIVE: [ConsciousnessState.INTEGRATING, ConsciousnessState.DISSIPATING],
            ConsciousnessState.INTEGRATING: [ConsciousnessState.ACTIVE, ConsciousnessState.DISSIPATING],
            ConsciousnessState.DISSIPATING: []  # Terminal state
        }
        
        if new_state not in valid_transitions[self.consciousness_state]:
            return False
        
        self.consciousness_state = new_state
        return True
    
    def __eq__(self, other) -> bool:
        """Entity equality based on identity, not attributes."""
        if not isinstance(other, CircularCausalityEntity):
            return False
        return self.causality_id == other.causality_id
    
    def __hash__(self) -> int:
        """Hash based on entity identity."""
        return hash(self.causality_id)


# ============================================================================
# EXPERIENTIAL MEMORY ENTITY (Phenomenological Memory Theory)  
# ============================================================================

@dataclass
class ExperientialMemoryEntity:
    """Phenomenological experiential memory as domain entity.
    
    Represents a complete experiential memory system with traces,
    sedimentation, and associative recall capabilities, maintaining
    identity across memory operations and temporal evolution.
    
    Business Rules:
    - Memory entity has identity and develops through experience retention
    - Traces must maintain connection to original experiential context
    - Sedimentation layers provide background for meaning constitution
    - Associative links enable transitive recall networks
    """
    
    # Entity Identity
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: TimeStep = field(default_factory=lambda: 0.0)
    
    # Memory Components (Value Objects)
    experiential_traces: Dict[str, ExperientialTrace] = field(default_factory=dict)
    sediment_layers: List[SedimentLayer] = field(default_factory=list)
    associative_links: Dict[str, List[AssociativeLink]] = field(default_factory=dict)
    
    # Entity State
    consciousness_state: ConsciousnessState = ConsciousnessState.FORMING
    memory_coherence: float = 0.0  # [0.0, 1.0]
    integration_readiness: float = 0.0  # [0.0, 1.0]
    
    # Entity Behavior State
    _retention_operations: int = field(default=0, init=False)
    _recall_operations: int = field(default=0, init=False)
    _last_sedimentation: TimeStep = field(default=0.0, init=False)
    
    def __post_init__(self):
        """Initialize entity and validate business invariants."""
        self._validate_entity_invariants()
        if self.consciousness_state == ConsciousnessState.FORMING:
            self._initialize_memory_system()
    
    def _validate_entity_invariants(self) -> None:
        """Validate core entity business invariants."""
        if not self.memory_id:
            raise ValueError("ExperientialMemoryEntity must have unique memory_id")
        
        if self.created_at < 0:
            raise ValueError("Entity created_at timestamp cannot be negative")
        
        measures = [self.memory_coherence, self.integration_readiness]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Entity coherence measures must be in [0.0, 1.0]")
    
    def _initialize_memory_system(self) -> None:
        """Initialize memory system when entity is forming."""
        # Initialize first sediment layer
        if not self.sediment_layers:
            initial_layer = SedimentLayer(
                sedimented_content=jnp.zeros(10),  # Default size
                sediment_depth=0,
                consolidation_strength=0.1,
                experiential_density=0.0,
                background_influence=0.1,
                layer_coherence=1.0
            )
            self.sediment_layers.append(initial_layer)
        
        self.consciousness_state = ConsciousnessState.ACTIVE
        self._compute_memory_coherence()
    
    def retain_experience(
        self,
        experience_content: Array,
        timestamp: TimeStep,
        experiential_context: Array,
        affective_resonance: float = 0.5
    ) -> str:
        """Retain experience in memory system.
        
        Core business logic for experience retention following
        phenomenological theories of memory and sedimentation.
        
        Returns:
            ID of created experiential trace
        """
        if self.consciousness_state == ConsciousnessState.DISSIPATING:
            raise ValueError("Cannot retain experience in dissipating state")
        
        arrays = [experience_content, experiential_context]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Experience arrays must contain finite values")
        
        if not (0.0 <= affective_resonance <= 1.0):
            raise ValueError("Affective resonance must be in [0.0, 1.0]")
        
        # Create experiential trace
        trace_id = str(uuid.uuid4())
        experiential_trace = ExperientialTrace(
            trace_content=experience_content,
            original_timestamp=timestamp,
            trace_strength=1.0,  # Full strength when first retained
            experiential_context=experiential_context,
            affective_resonance=affective_resonance,
            accessibility_index=0.9  # High accessibility for new traces
        )
        
        self.experiential_traces[trace_id] = experiential_trace
        
        # Add to sediment layers
        self._sediment_experience(experience_content, affective_resonance)
        
        # Create associative links with existing traces
        self._create_associative_links(trace_id, experience_content, experiential_context)
        
        self._retention_operations += 1
        self._compute_memory_coherence()
        
        return trace_id
    
    def recall_experiences(
        self,
        recall_context: RecallContext,
        max_recalls: int = 5
    ) -> List[Tuple[str, ExperientialTrace, float]]:
        """Recall experiences based on recall context.
        
        Business logic for experiential recall using associative,
        temporal, or sedimentary recall modes.
        
        Returns:
            List of (trace_id, trace, recall_strength) tuples
        """
        if not self.experiential_traces:
            return []
        
        recall_results = []
        
        # Assess recall probability for each trace
        for trace_id, trace in self.experiential_traces.items():
            recall_probability = trace.assess_recall_probability(recall_context.recall_cue)
            
            # Modulate by recall context
            context_modulation = recall_context.assess_recall_strength()
            final_recall_strength = recall_probability * context_modulation
            
            if final_recall_strength > 0.1:  # Minimum threshold for recall
                recall_results.append((trace_id, trace, final_recall_strength))
        
        # Sort by recall strength and return top results
        recall_results.sort(key=lambda x: x[2], reverse=True)
        final_results = recall_results[:max_recalls]
        
        # Process associative recall for top results
        if recall_context.recall_mode == RecallMode.ASSOCIATIVE:
            final_results = self._process_associative_recall(final_results, recall_context)
        elif recall_context.recall_mode == RecallMode.SEDIMENTARY:
            final_results = self._process_sedimentary_recall(final_results, recall_context)
        
        self._recall_operations += 1
        return final_results
    
    def _sediment_experience(self, experience_content: Array, affective_weight: float) -> None:
        """Add experience to sediment layers."""
        # Add to most recent layer (depth 0)
        if self.sediment_layers:
            current_layer = self.sediment_layers[0]
            sediment_rate = 0.1 * (0.5 + 0.5 * affective_weight)  # Affect influences sedimentation
            
            updated_layer = current_layer.add_sediment(experience_content, sediment_rate)
            self.sediment_layers[0] = updated_layer
            
            # Check if layer is full and needs to deepen
            if updated_layer.experiential_density > 0.8:
                self._deepen_sedimentation()
        
        current_time = self.created_at  # Use entity creation time as proxy
        self._last_sedimentation = current_time
    
    def _deepen_sedimentation(self) -> None:
        """Deepen sedimentation by creating new layer and pushing existing layers down."""
        # Increment depth of existing layers
        for layer in self.sediment_layers:
            layer = SedimentLayer(
                sedimented_content=layer.sedimented_content,
                sediment_depth=layer.sediment_depth + 1,
                consolidation_strength=layer.consolidation_strength * 1.1,  # Deeper = more consolidated
                experiential_density=layer.experiential_density,
                background_influence=layer.background_influence * 0.9,  # Deeper = less influence
                layer_coherence=layer.layer_coherence
            )
        
        # Create new surface layer
        new_layer = SedimentLayer(
            sedimented_content=jnp.zeros_like(self.sediment_layers[0].sedimented_content),
            sediment_depth=0,
            consolidation_strength=0.1,
            experiential_density=0.0,
            background_influence=0.2,
            layer_coherence=1.0
        )
        
        self.sediment_layers.insert(0, new_layer)
        
        # Limit total sediment layers to prevent unbounded growth
        if len(self.sediment_layers) > 10:
            self.sediment_layers = self.sediment_layers[:10]
    
    def _create_associative_links(
        self, 
        new_trace_id: str, 
        experience_content: Array,
        experiential_context: Array
    ) -> None:
        """Create associative links with existing traces."""
        for existing_id, existing_trace in self.experiential_traces.items():
            if existing_id == new_trace_id:
                continue
            
            # Assess content similarity
            if experience_content.shape == existing_trace.trace_content.shape:
                content_similarity = jnp.corrcoef(
                    experience_content.flatten(),
                    existing_trace.trace_content.flatten()
                )[0, 1]
                
                if jnp.isnan(content_similarity):
                    content_similarity = 0.0
                else:
                    content_similarity = float(jnp.clip(content_similarity, 0.0, 1.0))
                
                # Assess context similarity  
                if experiential_context.shape == existing_trace.experiential_context.shape:
                    context_similarity = jnp.corrcoef(
                        experiential_context.flatten(),
                        existing_trace.experiential_context.flatten()
                    )[0, 1]
                    
                    if jnp.isnan(context_similarity):
                        context_similarity = 0.0
                    else:
                        context_similarity = float(jnp.clip(context_similarity, 0.0, 1.0))
                else:
                    context_similarity = 0.0
                
                # Create link if similarity exceeds threshold
                overall_similarity = 0.6 * content_similarity + 0.4 * context_similarity
                
                if overall_similarity > 0.3:
                    # Determine link type
                    if content_similarity > context_similarity:
                        link_type = "content_similarity"
                    else:
                        link_type = "context_similarity"
                    
                    # Create bidirectional link
                    link = AssociativeLink(
                        source_trace_id=new_trace_id,
                        target_trace_id=existing_id,
                        association_strength=overall_similarity,
                        link_type=link_type,
                        activation_history=jnp.array([overall_similarity]),
                        bidirectional=True
                    )
                    
                    # Add to associative links
                    if new_trace_id not in self.associative_links:
                        self.associative_links[new_trace_id] = []
                    self.associative_links[new_trace_id].append(link)
                    
                    if existing_id not in self.associative_links:
                        self.associative_links[existing_id] = []
                    
                    # Reverse link for bidirectionality
                    reverse_link = AssociativeLink(
                        source_trace_id=existing_id,
                        target_trace_id=new_trace_id,
                        association_strength=overall_similarity,
                        link_type=link_type,
                        activation_history=jnp.array([overall_similarity]),
                        bidirectional=True
                    )
                    self.associative_links[existing_id].append(reverse_link)
    
    def _process_associative_recall(
        self,
        base_results: List[Tuple[str, ExperientialTrace, float]],
        recall_context: RecallContext
    ) -> List[Tuple[str, ExperientialTrace, float]]:
        """Process associative recall to include associated traces."""
        extended_results = list(base_results)
        
        for trace_id, trace, strength in base_results:
            if trace_id in self.associative_links:
                for link in self.associative_links[trace_id]:
                    if link.association_strength > 0.4:  # Strong association threshold
                        target_id = link.target_trace_id
                        if target_id in self.experiential_traces:
                            target_trace = self.experiential_traces[target_id]
                            
                            # Check if not already in results
                            if not any(r[0] == target_id for r in extended_results):
                                associative_strength = strength * link.association_strength * 0.7
                                extended_results.append((target_id, target_trace, associative_strength))
        
        # Re-sort by strength
        extended_results.sort(key=lambda x: x[2], reverse=True)
        return extended_results[:len(base_results) + 2]  # Allow a few associative additions
    
    def _process_sedimentary_recall(
        self,
        base_results: List[Tuple[str, ExperientialTrace, float]],
        recall_context: RecallContext
    ) -> List[Tuple[str, ExperientialTrace, float]]:
        """Process sedimentary recall based on sediment layer access."""
        # For sedimentary recall, modulate results by sediment layer accessibility
        modulated_results = []
        
        for trace_id, trace, strength in base_results:
            # Find which sediment layer this trace would be in based on age
            trace_age = recall_context.temporal_proximity  # Proxy for trace age
            layer_depth = min(int(trace_age * 5), len(self.sediment_layers) - 1)
            
            if layer_depth < len(self.sediment_layers):
                layer = self.sediment_layers[layer_depth]
                layer_accessibility = layer.assess_accessibility()
                sedimentary_strength = strength * layer_accessibility
                
                modulated_results.append((trace_id, trace, sedimentary_strength))
            else:
                # Very old traces have minimal accessibility
                modulated_results.append((trace_id, trace, strength * 0.1))
        
        modulated_results.sort(key=lambda x: x[2], reverse=True)
        return modulated_results
    
    def _compute_memory_coherence(self) -> None:
        """Compute memory coherence based on trace and sediment organization."""
        if not self.experiential_traces and not self.sediment_layers:
            self.memory_coherence = 0.0
            return
        
        # Trace coherence based on accessibility
        if self.experiential_traces:
            trace_accessibilities = [
                trace.accessibility_index * trace.trace_strength 
                for trace in self.experiential_traces.values()
            ]
            trace_coherence = jnp.mean(jnp.array(trace_accessibilities))
        else:
            trace_coherence = 0.0
        
        # Sediment coherence based on layer organization
        if self.sediment_layers:
            layer_coherences = [layer.layer_coherence for layer in self.sediment_layers]
            sediment_coherence = jnp.mean(jnp.array(layer_coherences))
        else:
            sediment_coherence = 0.0
        
        # Associative coherence based on link strengths
        if self.associative_links:
            all_link_strengths = []
            for links in self.associative_links.values():
                for link in links:
                    all_link_strengths.append(link.association_strength)
            
            if all_link_strengths:
                associative_coherence = jnp.mean(jnp.array(all_link_strengths))
            else:
                associative_coherence = 0.0
        else:
            associative_coherence = 0.0
        
        # Weighted memory coherence
        self.memory_coherence = float(jnp.clip(
            0.4 * trace_coherence +
            0.3 * sediment_coherence +
            0.3 * associative_coherence,
            0.0, 1.0
        ))
    
    def assess_memory_capacity(self) -> Dict[str, float]:
        """Assess current memory system capacity and health."""
        return {
            'trace_count': len(self.experiential_traces),
            'sediment_depth': len(self.sediment_layers),
            'associative_links': sum(len(links) for links in self.associative_links.values()),
            'memory_coherence': self.memory_coherence,
            'retention_efficiency': self._retention_operations / max(1, len(self.experiential_traces)),
            'recall_efficiency': self._recall_operations / max(1, len(self.experiential_traces)),
        }
    
    def transition_state(self, new_state: ConsciousnessState) -> bool:
        """Transition entity state with business rule validation."""
        valid_transitions = {
            ConsciousnessState.DORMANT: [ConsciousnessState.FORMING],
            ConsciousnessState.FORMING: [ConsciousnessState.ACTIVE, ConsciousnessState.DISSIPATING],
            ConsciousnessState.ACTIVE: [ConsciousnessState.INTEGRATING, ConsciousnessState.DISSIPATING],
            ConsciousnessState.INTEGRATING: [ConsciousnessState.ACTIVE, ConsciousnessState.DISSIPATING],
            ConsciousnessState.DISSIPATING: []  # Terminal state
        }
        
        if new_state not in valid_transitions[self.consciousness_state]:
            return False
        
        self.consciousness_state = new_state
        return True
    
    def __eq__(self, other) -> bool:
        """Entity equality based on identity, not attributes."""
        if not isinstance(other, ExperientialMemoryEntity):
            return False
        return self.memory_id == other.memory_id
    
    def __hash__(self) -> int:
        """Hash based on entity identity."""
        return hash(self.memory_id)