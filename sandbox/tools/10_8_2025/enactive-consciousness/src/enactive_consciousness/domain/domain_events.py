"""Domain Events for Enactive Consciousness.

This module defines domain events following Eric Evans' DDD methodology.
Domain events represent significant business occurrences in the consciousness
domain that other parts of the system may need to react to.

Theoretical Foundations:
- Events represent phenomenologically significant state changes
- Follow Husserlian temporal structure and Merleau-Pontian embodied events
- Varela-Maturana circular causality completion events
- Experience sedimentation and memory formation events

Design Principles:
1. Events are immutable records of what happened
2. Named using past tense verbs reflecting business significance
3. Contain all information needed by event handlers
4. Follow ubiquitous language precisely
5. Support event sourcing and temporal reasoning
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid
import time

import jax
import jax.numpy as jnp
from ..types import Array, TimeStep


# ============================================================================
# BASE DOMAIN EVENT INFRASTRUCTURE
# ============================================================================

class EventSeverity(Enum):
    """Severity levels for domain events."""
    ROUTINE = "routine"
    SIGNIFICANT = "significant"  
    CRITICAL = "critical"
    SYSTEM = "system"


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events in enactive consciousness.
    
    Provides common structure and behavior for domain events
    while maintaining immutability and theoretical grounding.
    """
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_timestamp: TimeStep = field(default_factory=time.time)
    event_type: str = field(init=False)
    event_version: str = "1.0"
    
    aggregate_id: str = ""
    aggregate_type: str = ""
    
    severity: EventSeverity = EventSeverity.ROUTINE
    causality_chain: List[str] = field(default_factory=list)
    phenomenological_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize event type from class name."""
        object.__setattr__(self, 'event_type', self.__class__.__name__)
    
    def extend_causality_chain(self, parent_event_id: str) -> DomainEvent:
        """Create new event extending causality chain."""
        new_chain = self.causality_chain + [parent_event_id]
        return self.__class__(
            event_timestamp=self.event_timestamp,
            aggregate_id=self.aggregate_id,
            aggregate_type=self.aggregate_type,
            severity=self.severity,
            causality_chain=new_chain,
            phenomenological_context=self.phenomenological_context,
            **{k: v for k, v in self.__dict__.items() 
               if k not in ['event_id', 'event_timestamp', 'event_type', 'event_version',
                          'aggregate_id', 'aggregate_type', 'severity', 'causality_chain',
                          'phenomenological_context']}
        )
    
    def with_context(self, **context_updates) -> DomainEvent:
        """Create new event with updated phenomenological context."""
        new_context = {**self.phenomenological_context, **context_updates}
        return self.__class__(
            event_timestamp=self.event_timestamp,
            aggregate_id=self.aggregate_id,
            aggregate_type=self.aggregate_type,
            severity=self.severity,
            causality_chain=self.causality_chain,
            phenomenological_context=new_context,
            **{k: v for k, v in self.__dict__.items() 
               if k not in ['event_id', 'event_timestamp', 'event_type', 'event_version',
                          'aggregate_id', 'aggregate_type', 'severity', 'causality_chain',
                          'phenomenological_context']}
        )


# ============================================================================
# TEMPORAL CONSCIOUSNESS EVENTS (Husserlian Phenomenology)
# ============================================================================

@dataclass(frozen=True)
class TemporalSynthesisOccurred(DomainEvent):
    """Event fired when Husserlian temporal synthesis occurs.
    
    Represents the fundamental temporal consciousness operation where
    retention-present-protention are synthesized into unified temporal flow.
    """
    
    moment_id: str = ""
    retention_strength: float = 0.0
    present_vividness: float = 0.0
    protention_strength: float = 0.0
    synthesis_coherence: float = 0.0
    temporal_weights: Array = field(default_factory=lambda: jnp.array([0.3, 0.5, 0.2]))
    synthesized_content: Optional[Array] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Validate temporal synthesis parameters
        if not (0.0 <= self.retention_strength <= 1.0):
            raise ValueError("Retention strength must be in [0.0, 1.0]")
        if not (0.0 <= self.present_vividness <= 1.0):
            raise ValueError("Present vividness must be in [0.0, 1.0]")
        if not (0.0 <= self.protention_strength <= 1.0):
            raise ValueError("Protention strength must be in [0.0, 1.0]")
        if not (0.0 <= self.synthesis_coherence <= 1.0):
            raise ValueError("Synthesis coherence must be in [0.0, 1.0]")
        
        if not jnp.isclose(jnp.sum(self.temporal_weights), 1.0, rtol=1e-6):
            raise ValueError("Temporal weights must sum to 1.0")


@dataclass(frozen=True)
class RetentionUpdated(DomainEvent):
    """Event fired when retention buffer is updated.
    
    Represents the phenomenological process of present experiences
    transitioning into retention with associated decay dynamics.
    """
    
    moment_id: str = ""
    new_retention_id: str = ""
    original_timestamp: TimeStep = 0.0
    retention_strength: float = 0.0
    temporal_distance: float = 0.0
    phenomenological_clarity: float = 0.0
    buffer_size_after: int = 0
    decay_applied: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if not (0.0 <= self.retention_strength <= 1.0):
            raise ValueError("Retention strength must be in [0.0, 1.0]")
        if not (0.0 <= self.phenomenological_clarity <= 1.0):
            raise ValueError("Phenomenological clarity must be in [0.0, 1.0]")
        if self.temporal_distance < 0.0:
            raise ValueError("Temporal distance cannot be negative")
        if self.buffer_size_after < 0:
            raise ValueError("Buffer size cannot be negative")


@dataclass(frozen=True)
class ProtentionProjected(DomainEvent):
    """Event fired when protentional horizon is projected.
    
    Represents the phenomenological anticipation process where
    consciousness projects expected future content.
    """
    
    moment_id: str = ""
    protention_id: str = ""
    expectation_timestamp: TimeStep = 0.0
    expectation_strength: float = 0.0
    anticipatory_distance: float = 0.0
    phenomenological_grip: float = 0.0
    expectational_specificity: float = 0.0
    anticipated_content_summary: Optional[Array] = None
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.expectation_strength, self.phenomenological_grip, 
                   self.expectational_specificity]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Protention measures must be in [0.0, 1.0]")
        if self.anticipatory_distance <= 0.0:
            raise ValueError("Anticipatory distance must be positive")


@dataclass(frozen=True)
class PrimalImpressionFormed(DomainEvent):
    """Event fired when new primal impression is formed.
    
    Represents the source-point of temporal consciousness where
    new content originally impresses itself in awareness.
    """
    
    moment_id: str = ""
    impression_timestamp: TimeStep = 0.0
    phenomenological_vividness: float = 0.0
    synthesis_readiness: float = 0.0
    intentional_directedness_magnitude: float = 0.0
    content_dimensions: tuple = ()
    predecessor_moment_id: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.phenomenological_vividness, self.synthesis_readiness]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Primal impression measures must be in [0.0, 1.0]")
        if self.intentional_directedness_magnitude < 0.0:
            raise ValueError("Intentional directedness magnitude cannot be negative")


@dataclass(frozen=True)
class TemporalFlowEvolved(DomainEvent):
    """Event fired when temporal flow evolves to new moment.
    
    Represents the continuous temporal evolution where the temporal
    stream flows from one moment to the next through phenomenological time.
    """
    
    previous_moment_id: str = ""
    new_moment_id: str = ""
    flow_coherence: float = 0.0
    temporal_distance: float = 0.0
    retention_transitions: int = 0
    protention_fulfillments: int = 0
    protention_disappointments: int = 0
    flow_direction: str = "forward"  # forward, backward, circular
    
    def __post_init__(self):
        super().__post_init__()
        if not (0.0 <= self.flow_coherence <= 1.0):
            raise ValueError("Flow coherence must be in [0.0, 1.0]")
        if self.temporal_distance < 0.0:
            raise ValueError("Temporal distance cannot be negative")
        if self.flow_direction not in ["forward", "backward", "circular"]:
            raise ValueError("Flow direction must be forward, backward, or circular")


# ============================================================================
# EMBODIED EXPERIENCE EVENTS (Merleau-Pontian Phenomenology)
# ============================================================================

@dataclass(frozen=True)
class BodySchemaReconfigured(DomainEvent):
    """Event fired when body schema is reconfigured.
    
    Represents the Merleau-Pontian process where the pre-reflective
    body schema adapts to new sensorimotor experiences.
    """
    
    experience_id: str = ""
    proprioceptive_integration: bool = False
    motor_integration: bool = False
    tactile_integration: bool = False
    schema_coherence: float = 0.0
    boundary_confidence: float = 0.0
    extension_capacity_change: float = 0.0
    integration_modalities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.schema_coherence, self.boundary_confidence]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Body schema measures must be in [0.0, 1.0]")
        if not (-1.0 <= self.extension_capacity_change <= 1.0):
            raise ValueError("Extension capacity change must be in [-1.0, 1.0]")


@dataclass(frozen=True)
class MotorIntentionFormed(DomainEvent):
    """Event fired when motor intention is formed.
    
    Represents the pre-reflective motor intentionality that
    underlies embodied action in Merleau-Pontian phenomenology.
    """
    
    experience_id: str = ""
    motor_vector_magnitude: float = 0.0
    action_readiness: float = 0.0
    embodied_confidence: float = 0.0
    motor_schema_activation_count: int = 0
    goal_state_defined: bool = False
    contextual_affordances_count: int = 0
    intention_coherence: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.action_readiness, self.embodied_confidence, self.intention_coherence]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Motor intention measures must be in [0.0, 1.0]")
        if self.motor_vector_magnitude < 0.0:
            raise ValueError("Motor vector magnitude cannot be negative")


@dataclass(frozen=True)
class ProprioceptiveIntegrationCompleted(DomainEvent):
    """Event fired when proprioceptive integration completes.
    
    Represents the integration of proprioceptive information
    into the pre-reflective body awareness system.
    """
    
    experience_id: str = ""
    proprioceptive_clarity: float = 0.0
    postural_stability: float = 0.0
    kinesthetic_flow_magnitude: float = 0.0
    integration_coherence: float = 0.0
    postural_configuration_updated: bool = False
    kinesthetic_delta_applied: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.proprioceptive_clarity, self.postural_stability, 
                   self.integration_coherence]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Proprioceptive integration measures must be in [0.0, 1.0]")
        if self.kinesthetic_flow_magnitude < 0.0:
            raise ValueError("Kinesthetic flow magnitude cannot be negative")


@dataclass(frozen=True)
class TactileFeedbackProcessed(DomainEvent):
    """Event fired when tactile feedback is processed.
    
    Represents the processing of tactile information that
    informs ongoing embodied interaction with environment.
    """
    
    experience_id: str = ""
    tactile_confidence: float = 0.0
    feedback_timeliness: float = 0.0
    contact_quality: float = 0.0
    environmental_texture_detected: bool = False
    motor_correction_generated: bool = False
    correction_magnitude: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.tactile_confidence, self.feedback_timeliness, self.contact_quality]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Tactile feedback measures must be in [0.0, 1.0]")
        if self.correction_magnitude < 0.0:
            raise ValueError("Correction magnitude cannot be negative")


@dataclass(frozen=True)
class ToolIncorporationCompleted(DomainEvent):
    """Event fired when tool incorporation into body schema completes.
    
    Represents the Merleau-Pontian process of tool incorporation
    where tools become extensions of the lived body.
    """
    
    experience_id: str = ""
    tool_id: str = ""
    incorporation_success: bool = False
    extension_strength: float = 0.0
    boundary_extension_magnitude: float = 0.0
    schema_modifications_count: int = 0
    extension_capacity_after: float = 0.0
    tool_compatibility: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.extension_strength, self.extension_capacity_after, 
                   self.tool_compatibility]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Tool incorporation measures must be in [0.0, 1.0]")
        if self.boundary_extension_magnitude < 0.0:
            raise ValueError("Boundary extension magnitude cannot be negative")


# ============================================================================
# CIRCULAR CAUSALITY EVENTS (Varela-Maturana Theory)
# ============================================================================

@dataclass(frozen=True)
class CircularCausalityCompleted(DomainEvent):
    """Event fired when circular causality cycle completes.
    
    Represents the completion of agent-environment circular causality
    where mutual specification occurs through structural coupling.
    """
    
    causality_id: str = ""
    cycle_number: int = 0
    agent_perturbation_magnitude: float = 0.0
    environmental_perturbation_magnitude: float = 0.0
    coupling_strength: float = 0.0
    mutual_specification_degree: float = 0.0
    structural_congruence: float = 0.0
    meaning_emerged: bool = False
    meaning_emergence_strength: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.coupling_strength, self.mutual_specification_degree,
                   self.structural_congruence, self.meaning_emergence_strength]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Circular causality measures must be in [0.0, 1.0]")
        if any(m < 0.0 for m in [self.agent_perturbation_magnitude, 
                                self.environmental_perturbation_magnitude]):
            raise ValueError("Perturbation magnitudes cannot be negative")
        if self.cycle_number < 0:
            raise ValueError("Cycle number cannot be negative")


@dataclass(frozen=True)
class CouplingStrengthChanged(DomainEvent):
    """Event fired when structural coupling strength changes.
    
    Represents changes in the strength and quality of structural
    coupling between agent and environment.
    """
    
    causality_id: str = ""
    previous_coupling_strength: float = 0.0
    new_coupling_strength: float = 0.0
    strength_change: float = 0.0
    perturbation_influence: float = 0.0
    stability_impact: float = 0.0
    coupling_maturity: float = 0.0
    change_direction: str = "increase"  # increase, decrease, stable
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.previous_coupling_strength, self.new_coupling_strength, 
                   self.coupling_maturity]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Coupling strength measures must be in [0.0, 1.0]")
        if not (-1.0 <= self.strength_change <= 1.0):
            raise ValueError("Strength change must be in [-1.0, 1.0]")
        if self.change_direction not in ["increase", "decrease", "stable"]:
            raise ValueError("Change direction must be increase, decrease, or stable")


@dataclass(frozen=True)
class MeaningEmerged(DomainEvent):
    """Event fired when meaning emerges through enactive coupling.
    
    Represents the emergence of meaning through structural coupling
    and circular causality, following enactivist theories.
    """
    
    causality_id: str = ""
    meaning_id: str = ""
    emergence_strength: float = 0.0
    enactive_significance: float = 0.0
    meaning_coherence: float = 0.0
    contextual_grounding_strength: float = 0.0
    temporal_development_stage: str = "initial"  # initial, developing, mature
    semantic_content_dimensions: tuple = ()
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.emergence_strength, self.enactive_significance, 
                   self.meaning_coherence, self.contextual_grounding_strength]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Meaning emergence measures must be in [0.0, 1.0]")
        if self.temporal_development_stage not in ["initial", "developing", "mature"]:
            raise ValueError("Development stage must be initial, developing, or mature")


@dataclass(frozen=True)
class AutopoeticCycleCompleted(DomainEvent):
    """Event fired when autopoietic cycle completes.
    
    Represents the completion of autopoietic self-maintenance
    and organizational closure in living systems.
    """
    
    causality_id: str = ""
    process_type: str = ""  # From AutopoeticProcessType enum
    organizational_closure: float = 0.0
    self_production_rate: float = 0.0
    boundary_integrity: float = 0.0
    autonomy_measure: float = 0.0
    autopoietic_viability: float = 0.0
    component_coherence_avg: float = 0.0
    environmental_perturbation_magnitude: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.organizational_closure, self.self_production_rate,
                   self.boundary_integrity, self.autonomy_measure, 
                   self.autopoietic_viability, self.component_coherence_avg]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Autopoietic measures must be in [0.0, 1.0]")
        if self.environmental_perturbation_magnitude < 0.0:
            raise ValueError("Environmental perturbation magnitude cannot be negative")


@dataclass(frozen=True)
class StructuralCouplingUpdated(DomainEvent):
    """Event fired when structural coupling is updated.
    
    Represents updates to structural coupling through
    recurrent agent-environment interactions.
    """
    
    causality_id: str = ""
    interaction_frequency: float = 0.0
    meaning_potential_change: float = 0.0
    structural_drift_magnitude: float = 0.0
    coupling_maturity: float = 0.0
    agent_structure_updated: bool = False
    environment_structure_updated: bool = False
    coupling_history_length: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.interaction_frequency, self.coupling_maturity]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Structural coupling measures must be in [0.0, 1.0]")
        if not (-1.0 <= self.meaning_potential_change <= 1.0):
            raise ValueError("Meaning potential change must be in [-1.0, 1.0]")
        if self.structural_drift_magnitude < 0.0:
            raise ValueError("Structural drift magnitude cannot be negative")


# ============================================================================
# EXPERIENTIAL MEMORY EVENTS (Phenomenological Memory Theory)
# ============================================================================

@dataclass(frozen=True)
class ExperiencesSedimented(DomainEvent):
    """Event fired when experiences are sedimented into memory layers.
    
    Represents the phenomenological sedimentation process where
    experiences form accumulated background layers.
    """
    
    memory_id: str = ""
    trigger_trace_id: str = ""
    sediment_layer_depth: int = 0
    sedimentation_rate: float = 0.0
    experiential_density_after: float = 0.0
    consolidation_strength: float = 0.0
    background_influence: float = 0.0
    layer_coherence: float = 0.0
    new_layer_created: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.sedimentation_rate, self.experiential_density_after,
                   self.consolidation_strength, self.background_influence, 
                   self.layer_coherence]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Sedimentation measures must be in [0.0, 1.0]")
        if self.sediment_layer_depth < 0:
            raise ValueError("Sediment layer depth cannot be negative")


@dataclass(frozen=True)
class MemoryRecalled(DomainEvent):
    """Event fired when experiential memory is recalled.
    
    Represents the phenomenological recall process where past
    experiences are brought back into present consciousness.
    """
    
    memory_id: str = ""
    recall_mode: str = ""  # associative, temporal, sedimentary, contextual
    recall_cue_dimensions: tuple = ()
    contextual_factors_dimensions: tuple = ()
    recalled_traces_count: int = 0
    average_recall_strength: float = 0.0
    strongest_recall_strength: float = 0.0
    temporal_proximity_factor: float = 0.0
    contextual_coherence: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.average_recall_strength, self.strongest_recall_strength,
                   self.temporal_proximity_factor, self.contextual_coherence]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Memory recall measures must be in [0.0, 1.0]")
        if self.recalled_traces_count < 0:
            raise ValueError("Recalled traces count cannot be negative")


@dataclass(frozen=True)
class AssociativeLinkFormed(DomainEvent):
    """Event fired when associative link is formed between traces.
    
    Represents the formation of associative connections that
    enable transitive recall between related experiences.
    """
    
    memory_id: str = ""
    source_trace_id: str = ""
    target_trace_id: str = ""
    association_strength: float = 0.0
    link_type: str = ""  # content_similarity, context_similarity, temporal_contiguity
    bidirectional: bool = False
    content_similarity: float = 0.0
    context_similarity: float = 0.0
    overall_similarity: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.association_strength, self.content_similarity,
                   self.context_similarity, self.overall_similarity]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Associative link measures must be in [0.0, 1.0]")


@dataclass(frozen=True)
class SedimentLayerDeepened(DomainEvent):
    """Event fired when sediment layer structure deepens.
    
    Represents the deepening of experiential sedimentation
    where surface layers become background layers.
    """
    
    memory_id: str = ""
    new_surface_layer_created: bool = False
    total_layers_after: int = 0
    deepened_layers_count: int = 0
    surface_layer_density_trigger: float = 0.0
    consolidation_increase_avg: float = 0.0
    background_influence_decrease_avg: float = 0.0
    layer_pruning_occurred: bool = False
    max_depth_reached: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if self.total_layers_after < 0 or self.deepened_layers_count < 0:
            raise ValueError("Layer counts cannot be negative")
        measures = [self.surface_layer_density_trigger, self.consolidation_increase_avg,
                   self.background_influence_decrease_avg]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Layer deepening measures must be in [0.0, 1.0]")


@dataclass(frozen=True)
class ExperientialTraceDecayed(DomainEvent):
    """Event fired when experiential trace undergoes decay.
    
    Represents the natural decay process of experiential traces
    over time, affecting accessibility and strength.
    """
    
    memory_id: str = ""
    trace_id: str = ""
    original_trace_strength: float = 0.0
    decayed_trace_strength: float = 0.0
    original_accessibility: float = 0.0
    decayed_accessibility: float = 0.0
    original_clarity: float = 0.0
    decayed_clarity: float = 0.0
    decay_factor_applied: float = 0.0
    phenomenologically_accessible: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.original_trace_strength, self.decayed_trace_strength,
                   self.original_accessibility, self.decayed_accessibility,
                   self.original_clarity, self.decayed_clarity, self.decay_factor_applied]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Trace decay measures must be in [0.0, 1.0]")


# ============================================================================
# SYSTEM-LEVEL INTEGRATION EVENTS
# ============================================================================

@dataclass(frozen=True)
class ConsciousnessIntegrationCompleted(DomainEvent):
    """Event fired when consciousness components are integrated.
    
    Represents the integration of temporal, embodied, coupling,
    and memory components into unified conscious experience.
    """
    
    integration_id: str = ""
    temporal_coherence: float = 0.0
    embodiment_coherence: float = 0.0
    coupling_coherence: float = 0.0
    memory_coherence: float = 0.0
    overall_integration_strength: float = 0.0
    consciousness_level: str = "basic"  # minimal, basic, reflective, meta_cognitive
    integrated_components_count: int = 0
    integration_method: str = "weighted_sum"
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.temporal_coherence, self.embodiment_coherence,
                   self.coupling_coherence, self.memory_coherence, 
                   self.overall_integration_strength]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Integration measures must be in [0.0, 1.0]")
        if self.consciousness_level not in ["minimal", "basic", "reflective", "meta_cognitive"]:
            raise ValueError("Invalid consciousness level")
        if self.integrated_components_count < 0:
            raise ValueError("Components count cannot be negative")


@dataclass(frozen=True)
class AggregateStateTransitioned(DomainEvent):
    """Event fired when aggregate state transitions occur.
    
    Represents state transitions in consciousness aggregates
    following valid state machine transitions.
    """
    
    previous_state: str = ""
    new_state: str = ""
    transition_reason: str = ""
    transition_success: bool = True
    state_coherence_before: float = 0.0
    state_coherence_after: float = 0.0
    transition_validation_passed: bool = True
    affected_operations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        measures = [self.state_coherence_before, self.state_coherence_after]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("State coherence measures must be in [0.0, 1.0]")


@dataclass(frozen=True)
class StateContainerEvolved(DomainEvent):
    """Event fired when state container evolves.
    
    Represents evolution of state containers in the clean architecture
    state management system with immutable state evolution.
    """
    
    container_type: str = ""  # temporal, embodiment, coupling, experiential
    evolution_type: str = ""  # continuous_flow, discrete_update, etc.
    source_state_id: str = ""
    target_state_id: str = ""
    evolution_operation: str = ""
    state_magnitude_before: float = 0.0
    state_magnitude_after: float = 0.0
    evolution_rate: float = 0.0
    consistency_maintained: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if any(m < 0.0 for m in [self.state_magnitude_before, 
                                self.state_magnitude_after, self.evolution_rate]):
            raise ValueError("State magnitudes and evolution rate cannot be negative")


# ============================================================================
# EVENT FACTORY AND UTILITIES
# ============================================================================

class DomainEventFactory:
    """Factory for creating domain events with proper context."""
    
    @staticmethod
    def create_temporal_synthesis_event(
        moment_id: str,
        aggregate_id: str,
        retention_strength: float,
        present_vividness: float,
        protention_strength: float,
        synthesis_coherence: float,
        temporal_weights: Array,
        **context
    ) -> TemporalSynthesisOccurred:
        """Create temporal synthesis event with validation."""
        return TemporalSynthesisOccurred(
            aggregate_id=aggregate_id,
            aggregate_type="TemporalConsciousnessAggregate",
            severity=EventSeverity.SIGNIFICANT,
            moment_id=moment_id,
            retention_strength=retention_strength,
            present_vividness=present_vividness,
            protention_strength=protention_strength,
            synthesis_coherence=synthesis_coherence,
            temporal_weights=temporal_weights,
            phenomenological_context=context
        )
    
    @staticmethod  
    def create_circular_causality_event(
        causality_id: str,
        aggregate_id: str,
        cycle_number: int,
        agent_perturbation_magnitude: float,
        environmental_perturbation_magnitude: float,
        coupling_strength: float,
        meaning_emerged: bool,
        **context
    ) -> CircularCausalityCompleted:
        """Create circular causality completion event."""
        return CircularCausalityCompleted(
            aggregate_id=aggregate_id,
            aggregate_type="CircularCausalityAggregate",
            severity=EventSeverity.SIGNIFICANT if meaning_emerged else EventSeverity.ROUTINE,
            causality_id=causality_id,
            cycle_number=cycle_number,
            agent_perturbation_magnitude=agent_perturbation_magnitude,
            environmental_perturbation_magnitude=environmental_perturbation_magnitude,
            coupling_strength=coupling_strength,
            meaning_emerged=meaning_emerged,
            phenomenological_context=context
        )
    
    @staticmethod
    def create_memory_recall_event(
        memory_id: str,
        aggregate_id: str,
        recall_mode: str,
        recalled_traces_count: int,
        average_recall_strength: float,
        **context
    ) -> MemoryRecalled:
        """Create memory recall event."""
        return MemoryRecalled(
            aggregate_id=aggregate_id,
            aggregate_type="ExperientialMemoryAggregate",
            severity=EventSeverity.ROUTINE,
            memory_id=memory_id,
            recall_mode=recall_mode,
            recalled_traces_count=recalled_traces_count,
            average_recall_strength=average_recall_strength,
            phenomenological_context=context
        )