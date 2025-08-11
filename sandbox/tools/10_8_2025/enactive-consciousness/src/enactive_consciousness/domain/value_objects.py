"""Value Objects for Enactive Consciousness Domain.

This module defines immutable value objects that represent core concepts
from phenomenological and enactivist theory. Following Eric Evans' DDD,
these objects encapsulate domain knowledge and maintain theoretical integrity.

Theoretical Foundations:
- Husserl: Temporal consciousness structure (retention-present-protention)
- Merleau-Ponty: Embodied phenomenology (body schema, motor intentionality)  
- Varela-Maturana: Enactivism (autopoiesis, structural coupling, circular causality)
- Phenomenological memory: Sedimentation theory and associative recall

Design Principles:
1. Immutability preserves theoretical consistency
2. Value semantics ensure domain integrity
3. Rich behavior embedded in value objects
4. Ubiquitous language reflected precisely
5. Business invariants enforced through construction
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import time
import uuid

import jax
import jax.numpy as jnp
from ..types import Array, TimeStep
from ..architecture.state_interfaces import StateEvolutionType


# ============================================================================
# TEMPORAL CONSCIOUSNESS VALUE OBJECTS (Husserlian Phenomenology)
# ============================================================================

@dataclass(frozen=True, slots=True)
class RetentionMoment:
    """Husserlian retention moment in temporal consciousness.
    
    Represents a single retained temporal moment with decay characteristics
    following Husserl's analysis of retention as modified reproduction.
    
    Business Rules:
    - Retention strength decreases over temporal distance
    - Each moment maintains its original temporal character
    - Retention depth reflects phenomenological accessibility
    """
    
    original_timestamp: TimeStep
    retained_content: Array
    retention_strength: float  # [0.0, 1.0] - phenomenological accessibility
    temporal_distance: float   # Distance from present moment
    phenomenological_clarity: float  # [0.0, 1.0] - clarity of retained content
    
    def __post_init__(self):
        """Validate retention moment invariants."""
        if not (0.0 <= self.retention_strength <= 1.0):
            raise ValueError("Retention strength must be in [0.0, 1.0]")
        if not (0.0 <= self.phenomenological_clarity <= 1.0):
            raise ValueError("Phenomenological clarity must be in [0.0, 1.0]")
        if self.temporal_distance < 0.0:
            raise ValueError("Temporal distance cannot be negative")
        if not jnp.all(jnp.isfinite(self.retained_content)):
            raise ValueError("Retained content must contain finite values")
    
    def decay_retention(self, decay_factor: float = 0.95) -> RetentionMoment:
        """Apply temporal decay to retention following Husserlian theory."""
        if not (0.0 <= decay_factor <= 1.0):
            raise ValueError("Decay factor must be in [0.0, 1.0]")
            
        return RetentionMoment(
            original_timestamp=self.original_timestamp,
            retained_content=self.retained_content * decay_factor,
            retention_strength=self.retention_strength * decay_factor,
            temporal_distance=self.temporal_distance + 1.0,
            phenomenological_clarity=self.phenomenological_clarity * decay_factor
        )
    
    def is_phenomenologically_accessible(self, threshold: float = 0.1) -> bool:
        """Check if retention is still phenomenologically accessible."""
        return self.retention_strength >= threshold and self.phenomenological_clarity >= threshold


@dataclass(frozen=True, slots=True)
class PrimalImpression:
    """Husserlian primal impression - the now-moment of consciousness.
    
    Represents the source-point of temporal consciousness where new content
    originally impresses itself in consciousness.
    
    Business Rules:
    - Primal impression is the absolute present
    - Has maximum phenomenological vividness
    - Serves as origin for temporal synthesis
    """
    
    impression_content: Array
    impression_timestamp: TimeStep
    phenomenological_vividness: float  # [0.0, 1.0] - intensity of present awareness
    intentional_directedness: Array    # What consciousness is directed toward
    synthesis_readiness: float         # [0.0, 1.0] - readiness for temporal synthesis
    
    def __post_init__(self):
        """Validate primal impression invariants."""
        if not (0.0 <= self.phenomenological_vividness <= 1.0):
            raise ValueError("Phenomenological vividness must be in [0.0, 1.0]")
        if not (0.0 <= self.synthesis_readiness <= 1.0):
            raise ValueError("Synthesis readiness must be in [0.0, 1.0]")
        if self.impression_timestamp < 0:
            raise ValueError("Impression timestamp cannot be negative")
        if not jnp.all(jnp.isfinite(self.impression_content)):
            raise ValueError("Impression content must contain finite values")
        if not jnp.all(jnp.isfinite(self.intentional_directedness)):
            raise ValueError("Intentional directedness must contain finite values")
    
    def transition_to_retention(self) -> RetentionMoment:
        """Transform primal impression into retention moment."""
        return RetentionMoment(
            original_timestamp=self.impression_timestamp,
            retained_content=self.impression_content,
            retention_strength=1.0,  # Initial retention at full strength
            temporal_distance=0.0,   # Just transitioned from present
            phenomenological_clarity=self.phenomenological_vividness
        )
    
    def assess_intentional_fulfillment(self, expected_content: Array) -> float:
        """Assess how well impression fulfills intentional expectation."""
        if expected_content.shape != self.impression_content.shape:
            return 0.0
        
        correlation = jnp.corrcoef(
            self.impression_content.flatten(),
            expected_content.flatten()
        )[0, 1]
        
        # Handle NaN correlation (when one array is constant)
        if jnp.isnan(correlation):
            return 0.0
            
        return float(jnp.clip(correlation, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class ProtentionalHorizon:
    """Husserlian protention - anticipatory temporal horizon.
    
    Represents the anticipated future moments that give temporal depth
    to present consciousness through expectational directedness.
    
    Business Rules:
    - Protention provides temporal anticipation structure
    - Expectation strength decreases with temporal distance
    - Can be fulfilled or disappointed by actual impressions
    """
    
    anticipated_content: Array
    expectation_timestamp: TimeStep
    expectation_strength: float      # [0.0, 1.0] - strength of anticipation
    anticipatory_distance: float     # How far into future anticipated
    phenomenological_grip: float     # [0.0, 1.0] - how firmly grasped in consciousness
    expectational_specificity: float # [0.0, 1.0] - how specific the expectation
    
    def __post_init__(self):
        """Validate protentional horizon invariants."""
        if not (0.0 <= self.expectation_strength <= 1.0):
            raise ValueError("Expectation strength must be in [0.0, 1.0]")
        if not (0.0 <= self.phenomenological_grip <= 1.0):
            raise ValueError("Phenomenological grip must be in [0.0, 1.0]")
        if not (0.0 <= self.expectational_specificity <= 1.0):
            raise ValueError("Expectational specificity must be in [0.0, 1.0]")
        if self.anticipatory_distance < 0.0:
            raise ValueError("Anticipatory distance cannot be negative")
        if not jnp.all(jnp.isfinite(self.anticipated_content)):
            raise ValueError("Anticipated content must contain finite values")
    
    def fulfill_expectation(self, actual_impression: PrimalImpression) -> Tuple[float, ProtentionalHorizon]:
        """Process expectation fulfillment/disappointment."""
        fulfillment_score = actual_impression.assess_intentional_fulfillment(self.anticipated_content)
        
        # Update protention based on fulfillment
        new_strength = self.expectation_strength * (0.5 + 0.5 * fulfillment_score)
        new_grip = self.phenomenological_grip * (0.7 + 0.3 * fulfillment_score)
        
        updated_protention = ProtentionalHorizon(
            anticipated_content=self.anticipated_content,
            expectation_timestamp=self.expectation_timestamp,
            expectation_strength=new_strength,
            anticipatory_distance=self.anticipatory_distance,
            phenomenological_grip=new_grip,
            expectational_specificity=self.expectational_specificity * (0.8 + 0.2 * fulfillment_score)
        )
        
        return fulfillment_score, updated_protention
    
    def is_expectationally_active(self, threshold: float = 0.2) -> bool:
        """Check if protention is actively shaping present consciousness."""
        return (self.expectation_strength >= threshold and 
                self.phenomenological_grip >= threshold)


@dataclass(frozen=True, slots=True)
class TemporalSynthesisWeights:
    """Weights governing Husserlian temporal synthesis.
    
    Represents the phenomenological weights that determine how
    retention-present-protention are synthesized into unified temporal flow.
    
    Business Rules:
    - Weights must sum to 1.0 for conservation
    - Present weight typically dominant but not exclusive
    - Weights reflect phenomenological attention distribution
    """
    
    retention_weight: float      # [0.0, 1.0] - weight for retained moments
    present_weight: float        # [0.0, 1.0] - weight for primal impression
    protention_weight: float     # [0.0, 1.0] - weight for anticipated moments
    synthesis_coherence: float   # [0.0, 1.0] - overall temporal coherence
    attentional_focus: Array     # Spatial distribution of temporal attention
    
    def __post_init__(self):
        """Validate temporal synthesis weights invariants."""
        total_weight = self.retention_weight + self.present_weight + self.protention_weight
        if not jnp.isclose(total_weight, 1.0, rtol=1e-6):
            raise ValueError(f"Temporal weights must sum to 1.0, got {total_weight}")
        
        weights = [self.retention_weight, self.present_weight, self.protention_weight]
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("All temporal weights must be in [0.0, 1.0]")
            
        if not (0.0 <= self.synthesis_coherence <= 1.0):
            raise ValueError("Synthesis coherence must be in [0.0, 1.0]")
            
        if not jnp.all(jnp.isfinite(self.attentional_focus)):
            raise ValueError("Attentional focus must contain finite values")
    
    def rebalance_for_temporal_situation(
        self, 
        retention_demand: float, 
        protention_demand: float
    ) -> TemporalSynthesisWeights:
        """Rebalance weights based on temporal situational demands."""
        if not (0.0 <= retention_demand <= 1.0) or not (0.0 <= protention_demand <= 1.0):
            raise ValueError("Temporal demands must be in [0.0, 1.0]")
        
        # Adjust weights while maintaining conservation
        total_demand = retention_demand + protention_demand
        if total_demand > 1.0:
            # Normalize if demands exceed capacity
            retention_demand /= total_demand
            protention_demand /= total_demand
        
        new_retention_weight = 0.5 * self.retention_weight + 0.5 * retention_demand
        new_protention_weight = 0.5 * self.protention_weight + 0.5 * protention_demand
        new_present_weight = 1.0 - new_retention_weight - new_protention_weight
        
        return TemporalSynthesisWeights(
            retention_weight=new_retention_weight,
            present_weight=max(0.1, new_present_weight),  # Ensure present never fully disappears
            protention_weight=new_protention_weight,
            synthesis_coherence=self.synthesis_coherence * 0.95,  # Slight coherence cost for rebalancing
            attentional_focus=self.attentional_focus
        )


# ============================================================================
# EMBODIED CONSCIOUSNESS VALUE OBJECTS (Merleau-Pontian Phenomenology)  
# ============================================================================

@dataclass(frozen=True, slots=True)
class BodyBoundary:
    """Merleau-Pontian body boundary in embodied phenomenology.
    
    Represents the phenomenological boundary of the lived body as distinct
    from the objective body, constituting the margin of embodied experience.
    
    Business Rules:
    - Boundary is dynamic and contextually variable
    - Defines sphere of immediate bodily agency
    - Can extend through tool use (body schema extension)
    """
    
    boundary_contour: Array           # Spatial representation of body boundary
    boundary_confidence: float        # [0.0, 1.0] - confidence in boundary definition
    permeability_index: float         # [0.0, 1.0] - how permeable boundary is
    extension_capacity: float         # [0.0, 1.0] - capacity for extension through tools
    proprioceptive_grounding: Array   # Proprioceptive basis for boundary
    
    def __post_init__(self):
        """Validate body boundary invariants."""
        boundaries = [self.boundary_confidence, self.permeability_index, self.extension_capacity]
        if any(not (0.0 <= b <= 1.0) for b in boundaries):
            raise ValueError("Boundary measures must be in [0.0, 1.0]")
            
        arrays = [self.boundary_contour, self.proprioceptive_grounding]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Boundary arrays must contain finite values")
    
    def extend_through_tool(self, tool_representation: Array, extension_strength: float) -> BodyBoundary:
        """Extend body boundary through tool incorporation (Merleau-Ponty)."""
        if not (0.0 <= extension_strength <= 1.0):
            raise ValueError("Extension strength must be in [0.0, 1.0]")
        
        # Tool incorporation extends boundary based on extension capacity
        actual_extension = extension_strength * self.extension_capacity
        
        # Update boundary contour to include tool
        extended_contour = self.boundary_contour + actual_extension * tool_representation
        
        return BodyBoundary(
            boundary_contour=extended_contour,
            boundary_confidence=self.boundary_confidence * (0.8 + 0.2 * actual_extension),
            permeability_index=self.permeability_index * (1.0 - 0.3 * actual_extension),
            extension_capacity=self.extension_capacity * 0.95,  # Slight capacity cost
            proprioceptive_grounding=self.proprioceptive_grounding
        )
    
    def assess_boundary_coherence(self) -> float:
        """Assess phenomenological coherence of body boundary."""
        spatial_coherence = 1.0 - jnp.std(self.boundary_contour) / (jnp.mean(jnp.abs(self.boundary_contour)) + 1e-6)
        proprioceptive_coherence = 1.0 - jnp.std(self.proprioceptive_grounding) / (jnp.mean(jnp.abs(self.proprioceptive_grounding)) + 1e-6)
        
        return float(jnp.clip(
            self.boundary_confidence * spatial_coherence * proprioceptive_coherence,
            0.0, 1.0
        ))


@dataclass(frozen=True, slots=True)
class MotorIntention:
    """Merleau-Pontian motor intention in embodied action.
    
    Represents the motor intentionality that underlies embodied action,
    distinct from cognitive intention through its pre-reflective character.
    
    Business Rules:
    - Motor intention is pre-reflective and bodily directed
    - Manifests through motor schema activation
    - Links perception and action in embodied loops
    """
    
    motor_vector: Array              # Direction and magnitude of intended motion
    action_readiness: float          # [0.0, 1.0] - readiness for motor execution
    motor_schema_activation: Array   # Pattern of activated motor schemas
    intentional_directedness: Array  # What the intention is directed toward
    embodied_confidence: float       # [0.0, 1.0] - confidence in motor capability
    
    def __post_init__(self):
        """Validate motor intention invariants."""
        measures = [self.action_readiness, self.embodied_confidence]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Motor intention measures must be in [0.0, 1.0]")
            
        arrays = [self.motor_vector, self.motor_schema_activation, self.intentional_directedness]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Motor intention arrays must contain finite values")
    
    def modulate_for_context(self, contextual_affordances: Array, modulation_strength: float) -> MotorIntention:
        """Modulate motor intention based on perceived affordances."""
        if not (0.0 <= modulation_strength <= 1.0):
            raise ValueError("Modulation strength must be in [0.0, 1.0]")
        
        # Affordances modulate motor vector and schema activation
        affordance_influence = modulation_strength * contextual_affordances
        
        modulated_vector = self.motor_vector + 0.3 * affordance_influence
        modulated_schema = self.motor_schema_activation + 0.2 * affordance_influence
        
        return MotorIntention(
            motor_vector=modulated_vector,
            action_readiness=self.action_readiness * (0.7 + 0.3 * modulation_strength),
            motor_schema_activation=modulated_schema,
            intentional_directedness=self.intentional_directedness + 0.1 * affordance_influence,
            embodied_confidence=self.embodied_confidence * (0.8 + 0.2 * modulation_strength)
        )
    
    def assess_motor_coherence(self) -> float:
        """Assess coherence between motor vector and schema activation."""
        vector_norm = jnp.linalg.norm(self.motor_vector)
        schema_norm = jnp.linalg.norm(self.motor_schema_activation)
        
        if vector_norm < 1e-6 or schema_norm < 1e-6:
            return 0.0
        
        alignment = jnp.dot(
            self.motor_vector / vector_norm,
            self.motor_schema_activation / schema_norm
        )
        
        return float(jnp.clip(alignment * self.embodied_confidence, 0.0, 1.0))


@dataclass(frozen=True, slots=True)  
class ProprioceptiveField:
    """Merleau-Pontian proprioceptive field in embodied awareness.
    
    Represents the proprioceptive field that provides pre-reflective
    bodily awareness and grounds embodied spatial orientation.
    
    Business Rules:
    - Proprioception provides continuous bodily self-awareness
    - Grounds spatial orientation and body schema
    - Operates pre-reflectively in motor control
    """
    
    proprioceptive_map: Array        # Spatial map of proprioceptive sensations
    postural_configuration: Array   # Current postural state representation
    kinesthetic_flow: Array          # Flow of kinesthetic sensations
    proprioceptive_clarity: float    # [0.0, 1.0] - clarity of proprioceptive awareness
    postural_stability: float        # [0.0, 1.0] - stability of postural configuration
    
    def __post_init__(self):
        """Validate proprioceptive field invariants."""
        measures = [self.proprioceptive_clarity, self.postural_stability]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Proprioceptive measures must be in [0.0, 1.0]")
            
        arrays = [self.proprioceptive_map, self.postural_configuration, self.kinesthetic_flow]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Proprioceptive arrays must contain finite values")
    
    def integrate_kinesthetic_change(self, kinesthetic_delta: Array) -> ProprioceptiveField:
        """Integrate kinesthetic change into proprioceptive field."""
        if not jnp.all(jnp.isfinite(kinesthetic_delta)):
            raise ValueError("Kinesthetic delta must contain finite values")
        
        # Update kinesthetic flow and proprioceptive map
        new_flow = 0.7 * self.kinesthetic_flow + 0.3 * kinesthetic_delta
        new_map = self.proprioceptive_map + 0.1 * kinesthetic_delta
        
        # Assess impact on postural stability
        change_magnitude = jnp.linalg.norm(kinesthetic_delta)
        stability_impact = 1.0 - 0.2 * jnp.tanh(change_magnitude)
        
        return ProprioceptiveField(
            proprioceptive_map=new_map,
            postural_configuration=self.postural_configuration,
            kinesthetic_flow=new_flow,
            proprioceptive_clarity=self.proprioceptive_clarity * 0.98,  # Slight decay
            postural_stability=self.postural_stability * stability_impact
        )
    
    def assess_proprioceptive_coherence(self) -> float:
        """Assess overall coherence of proprioceptive field."""
        map_coherence = 1.0 - jnp.std(self.proprioceptive_map) / (jnp.mean(jnp.abs(self.proprioceptive_map)) + 1e-6)
        flow_coherence = 1.0 - jnp.std(self.kinesthetic_flow) / (jnp.mean(jnp.abs(self.kinesthetic_flow)) + 1e-6)
        
        return float(jnp.clip(
            self.proprioceptive_clarity * self.postural_stability * map_coherence * flow_coherence,
            0.0, 1.0
        ))


@dataclass(frozen=True, slots=True)
class TactileFeedback:
    """Merleau-Pontian tactile feedback in embodied interaction.
    
    Represents tactile feedback that informs embodied interaction with
    environment, providing crucial information for motor control.
    
    Business Rules:
    - Tactile feedback informs ongoing motor control
    - Provides information about environmental contact
    - Guides motor intention refinement
    """
    
    tactile_pattern: Array           # Spatial pattern of tactile stimulation
    contact_pressure: Array          # Pressure distribution from environmental contact
    tactile_confidence: float        # [0.0, 1.0] - confidence in tactile information
    feedback_timeliness: float       # [0.0, 1.0] - timeliness of feedback for control
    environmental_texture: Array     # Perceived environmental texture properties
    
    def __post_init__(self):
        """Validate tactile feedback invariants."""
        measures = [self.tactile_confidence, self.feedback_timeliness]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Tactile measures must be in [0.0, 1.0]")
            
        arrays = [self.tactile_pattern, self.contact_pressure, self.environmental_texture]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Tactile arrays must contain finite values")
    
    def assess_contact_quality(self) -> float:
        """Assess quality of environmental contact based on tactile feedback."""
        pressure_consistency = 1.0 - jnp.std(self.contact_pressure) / (jnp.mean(self.contact_pressure) + 1e-6)
        pattern_coherence = 1.0 - jnp.std(self.tactile_pattern) / (jnp.mean(jnp.abs(self.tactile_pattern)) + 1e-6)
        
        return float(jnp.clip(
            self.tactile_confidence * self.feedback_timeliness * pressure_consistency * pattern_coherence,
            0.0, 1.0
        ))
    
    def generate_motor_correction(self, target_contact: Array) -> Array:
        """Generate motor correction signal based on tactile feedback."""
        if not jnp.all(jnp.isfinite(target_contact)):
            raise ValueError("Target contact must contain finite values")
        
        contact_error = target_contact - self.contact_pressure
        correction_magnitude = self.tactile_confidence * self.feedback_timeliness
        
        return correction_magnitude * contact_error


# ============================================================================
# ENACTIVE COUPLING VALUE OBJECTS (Varela-Maturana Theory)
# ============================================================================

class AutopoeticProcessType(Enum):
    """Types of autopoietic processes."""
    SELF_MAINTENANCE = "self_maintenance"
    BOUNDARY_PRODUCTION = "boundary_production"
    COMPONENT_REGENERATION = "component_regeneration"
    ORGANIZATIONAL_CLOSURE = "organizational_closure"


@dataclass(frozen=True, slots=True)
class CouplingStrength:
    """Varela-Maturana structural coupling strength measure.
    
    Represents the strength and quality of structural coupling between
    agent and environment in enactive interaction.
    
    Business Rules:
    - Coupling strength affects meaning emergence potential
    - Strong coupling enables rich circular causality
    - Coupling stability reflects successful adaptation
    """
    
    coupling_intensity: float        # [0.0, 1.0] - intensity of agent-environment coupling
    mutual_specification: float     # [0.0, 1.0] - degree of mutual specification
    structural_congruence: float     # [0.0, 1.0] - structural match between agent/environment
    coupling_stability: float        # [0.0, 1.0] - stability of coupling over time
    perturbation_sensitivity: Array # Sensitivity to different perturbation types
    
    def __post_init__(self):
        """Validate coupling strength invariants."""
        measures = [self.coupling_intensity, self.mutual_specification, 
                   self.structural_congruence, self.coupling_stability]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Coupling strength measures must be in [0.0, 1.0]")
            
        if not jnp.all(jnp.isfinite(self.perturbation_sensitivity)):
            raise ValueError("Perturbation sensitivity must contain finite values")
    
    def assess_coupling_quality(self) -> float:
        """Assess overall quality of structural coupling."""
        return float(jnp.mean(jnp.array([
            self.coupling_intensity,
            self.mutual_specification,
            self.structural_congruence,
            self.coupling_stability
        ])))
    
    def modulate_for_perturbation(self, perturbation_magnitude: float) -> CouplingStrength:
        """Modulate coupling strength based on environmental perturbation."""
        if perturbation_magnitude < 0.0:
            raise ValueError("Perturbation magnitude cannot be negative")
        
        # Strong perturbations test coupling stability
        stability_impact = jnp.exp(-perturbation_magnitude)
        
        return CouplingStrength(
            coupling_intensity=self.coupling_intensity * (0.5 + 0.5 * stability_impact),
            mutual_specification=self.mutual_specification,
            structural_congruence=self.structural_congruence * stability_impact,
            coupling_stability=self.coupling_stability * stability_impact,
            perturbation_sensitivity=self.perturbation_sensitivity
        )


@dataclass(frozen=True, slots=True)
class AutopoeticProcess:
    """Varela-Maturana autopoietic process in living systems.
    
    Represents an autopoietic process that maintains system organization
    through continuous self-production and boundary maintenance.
    
    Business Rules:
    - Autopoiesis maintains organizational closure
    - Process must be self-sustaining and self-producing
    - Boundary production is essential for autonomy
    """
    
    process_type: AutopoeticProcessType
    organizational_closure: float    # [0.0, 1.0] - degree of organizational closure
    self_production_rate: float      # [0.0, 1.0] - rate of self-production
    boundary_integrity: float        # [0.0, 1.0] - integrity of system boundary
    component_coherence: Array       # Coherence of system components
    autonomy_measure: float          # [0.0, 1.0] - degree of system autonomy
    
    def __post_init__(self):
        """Validate autopoietic process invariants."""
        measures = [self.organizational_closure, self.self_production_rate, 
                   self.boundary_integrity, self.autonomy_measure]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Autopoietic measures must be in [0.0, 1.0]")
            
        if not jnp.all(jnp.isfinite(self.component_coherence)):
            raise ValueError("Component coherence must contain finite values")
    
    def assess_autopoietic_viability(self) -> float:
        """Assess viability of autopoietic process."""
        component_avg = jnp.mean(self.component_coherence)
        
        return float(jnp.clip(
            self.organizational_closure * 
            self.self_production_rate * 
            self.boundary_integrity * 
            component_avg * 
            self.autonomy_measure,
            0.0, 1.0
        ))
    
    def evolve_autopoietic_cycle(self, environmental_perturbation: Array) -> AutopoeticProcess:
        """Evolve autopoietic process through one cycle."""
        if not jnp.all(jnp.isfinite(environmental_perturbation)):
            raise ValueError("Environmental perturbation must contain finite values")
        
        # Autopoietic response to perturbation
        perturbation_impact = jnp.linalg.norm(environmental_perturbation)
        
        # Self-production responds to maintain organization
        new_production_rate = self.self_production_rate * (1.0 + 0.1 * jnp.tanh(perturbation_impact))
        new_closure = self.organizational_closure * (0.9 + 0.1 / (1.0 + perturbation_impact))
        new_boundary = self.boundary_integrity * jnp.exp(-0.1 * perturbation_impact)
        
        # Component coherence adapts
        new_coherence = self.component_coherence + 0.05 * environmental_perturbation
        
        return AutopoeticProcess(
            process_type=self.process_type,
            organizational_closure=float(jnp.clip(new_closure, 0.0, 1.0)),
            self_production_rate=float(jnp.clip(new_production_rate, 0.0, 1.0)),
            boundary_integrity=float(jnp.clip(new_boundary, 0.0, 1.0)),
            component_coherence=new_coherence,
            autonomy_measure=self.autonomy_measure * 0.99  # Slight autonomy cost
        )


@dataclass(frozen=True, slots=True)
class StructuralCoupling:
    """Varela-Maturana structural coupling between agent and environment.
    
    Represents the structural coupling that enables meaning emergence
    through history of recurrent interactions.
    
    Business Rules:
    - Coupling develops through recurrent interactions
    - History shapes current coupling structure
    - Enables bidirectional influence between agent and environment
    """
    
    agent_structure: Array           # Current structural configuration of agent
    environment_structure: Array    # Current structural configuration of environment
    coupling_history: Array         # History of coupling interactions
    interaction_frequency: float    # [0.0, 1.0] - frequency of meaningful interactions
    structural_drift: Array         # Drift in coupled structures over time
    meaning_potential: float        # [0.0, 1.0] - potential for meaning emergence
    
    def __post_init__(self):
        """Validate structural coupling invariants."""
        measures = [self.interaction_frequency, self.meaning_potential]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Coupling measures must be in [0.0, 1.0]")
            
        arrays = [self.agent_structure, self.environment_structure, 
                 self.coupling_history, self.structural_drift]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Coupling arrays must contain finite values")
    
    def update_coupling_interaction(
        self, 
        agent_perturbation: Array, 
        environment_perturbation: Array
    ) -> StructuralCoupling:
        """Update coupling through new interaction episode."""
        arrays = [agent_perturbation, environment_perturbation]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Perturbations must contain finite values")
        
        # Update structures through mutual perturbation
        new_agent_structure = self.agent_structure + 0.1 * environment_perturbation
        new_environment_structure = self.environment_structure + 0.1 * agent_perturbation
        
        # Update coupling history
        interaction_signature = 0.5 * (agent_perturbation + environment_perturbation)
        new_history = 0.9 * self.coupling_history + 0.1 * interaction_signature
        
        # Update structural drift
        agent_drift = new_agent_structure - self.agent_structure
        env_drift = new_environment_structure - self.environment_structure
        total_drift = 0.5 * (agent_drift + env_drift)
        new_drift = 0.8 * self.structural_drift + 0.2 * total_drift
        
        # Assess meaning potential change
        interaction_coherence = jnp.corrcoef(agent_perturbation.flatten(), environment_perturbation.flatten())[0, 1]
        if jnp.isnan(interaction_coherence):
            interaction_coherence = 0.0
        
        meaning_enhancement = 0.1 * jnp.clip(interaction_coherence, 0.0, 1.0)
        new_meaning_potential = jnp.clip(self.meaning_potential + meaning_enhancement, 0.0, 1.0)
        
        return StructuralCoupling(
            agent_structure=new_agent_structure,
            environment_structure=new_environment_structure,
            coupling_history=new_history,
            interaction_frequency=0.95 * self.interaction_frequency + 0.05,  # Boost frequency
            structural_drift=new_drift,
            meaning_potential=float(new_meaning_potential)
        )
    
    def assess_coupling_maturity(self) -> float:
        """Assess maturity of structural coupling."""
        history_depth = jnp.linalg.norm(self.coupling_history)
        drift_stability = 1.0 / (1.0 + jnp.linalg.norm(self.structural_drift))
        
        return float(jnp.clip(
            self.interaction_frequency * self.meaning_potential * history_depth * drift_stability,
            0.0, 1.0
        ))


@dataclass(frozen=True, slots=True)
class MeaningEmergence:
    """Varela-inspired meaning emergence through enactive coupling.
    
    Represents the emergence of meaning through structural coupling
    and circular causality in agent-environment interaction.
    
    Business Rules:
    - Meaning emerges from coupling history and context
    - Has temporal development and contextual grounding
    - Cannot be reduced to representational content
    """
    
    semantic_content: Array          # Emergent semantic content (not representational!)
    emergence_strength: float        # [0.0, 1.0] - strength of meaning emergence
    contextual_grounding: Array      # Contextual basis for meaning
    temporal_development: Array      # Temporal evolution of meaning
    enactive_significance: float     # [0.0, 1.0] - significance for enactive agent
    meaning_coherence: float         # [0.0, 1.0] - internal coherence of meaning
    
    def __post_init__(self):
        """Validate meaning emergence invariants."""
        measures = [self.emergence_strength, self.enactive_significance, self.meaning_coherence]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Meaning measures must be in [0.0, 1.0]")
            
        arrays = [self.semantic_content, self.contextual_grounding, self.temporal_development]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Meaning arrays must contain finite values")
    
    def develop_meaning_temporally(self, temporal_context: Array) -> MeaningEmergence:
        """Develop meaning through temporal context."""
        if not jnp.all(jnp.isfinite(temporal_context)):
            raise ValueError("Temporal context must contain finite values")
        
        # Meaning develops through temporal integration
        new_development = 0.7 * self.temporal_development + 0.3 * temporal_context
        
        # Context influences semantic content
        contextual_influence = 0.1 * temporal_context
        new_content = self.semantic_content + contextual_influence
        
        # Assess coherence development
        coherence_enhancement = jnp.corrcoef(
            self.temporal_development.flatten(), 
            temporal_context.flatten()
        )[0, 1]
        
        if jnp.isnan(coherence_enhancement):
            coherence_enhancement = 0.0
        else:
            coherence_enhancement = jnp.clip(coherence_enhancement * 0.1, 0.0, 0.1)
        
        return MeaningEmergence(
            semantic_content=new_content,
            emergence_strength=self.emergence_strength * 1.02,  # Gradual strengthening
            contextual_grounding=0.9 * self.contextual_grounding + 0.1 * temporal_context,
            temporal_development=new_development,
            enactive_significance=self.enactive_significance,
            meaning_coherence=float(jnp.clip(self.meaning_coherence + coherence_enhancement, 0.0, 1.0))
        )
    
    def assess_enactive_relevance(self, agent_concerns: Array) -> float:
        """Assess relevance of meaning for enactive agent's concerns."""
        if not jnp.all(jnp.isfinite(agent_concerns)):
            raise ValueError("Agent concerns must contain finite values")
        
        if agent_concerns.size != self.semantic_content.size:
            return 0.0
        
        relevance = jnp.corrcoef(
            self.semantic_content.flatten(),
            agent_concerns.flatten()
        )[0, 1]
        
        if jnp.isnan(relevance):
            return 0.0
        
        return float(jnp.clip(relevance * self.enactive_significance, 0.0, 1.0))


# ============================================================================
# EXPERIENTIAL MEMORY VALUE OBJECTS (Phenomenological Memory Theory)
# ============================================================================

class RecallMode(Enum):
    """Modes of experiential recall."""
    ASSOCIATIVE = "associative"
    TEMPORAL = "temporal"
    SEDIMENTARY = "sedimentary"
    CONTEXTUAL = "contextual"


@dataclass(frozen=True, slots=True)
class ExperientialTrace:
    """Phenomenological trace of past experience.
    
    Represents the trace left by experience in consciousness,
    following phenomenological theories of memory and retention.
    
    Business Rules:
    - Traces fade over time but can be reactivated
    - Maintain connection to original experiential context
    - Support associative and temporal recall
    """
    
    trace_content: Array             # Content of experiential trace
    original_timestamp: TimeStep     # When experience originally occurred
    trace_strength: float            # [0.0, 1.0] - current strength of trace
    experiential_context: Array      # Context in which experience occurred
    affective_resonance: float       # [0.0, 1.0] - emotional resonance of trace
    accessibility_index: float       # [0.0, 1.0] - how easily trace can be recalled
    
    def __post_init__(self):
        """Validate experiential trace invariants."""
        measures = [self.trace_strength, self.affective_resonance, self.accessibility_index]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Trace measures must be in [0.0, 1.0]")
            
        if self.original_timestamp < 0:
            raise ValueError("Original timestamp cannot be negative")
            
        arrays = [self.trace_content, self.experiential_context]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Trace arrays must contain finite values")
    
    def decay_trace(self, current_time: TimeStep, decay_rate: float = 0.01) -> ExperientialTrace:
        """Apply temporal decay to experiential trace."""
        if current_time < self.original_timestamp:
            raise ValueError("Current time cannot be before original timestamp")
        if not (0.0 <= decay_rate <= 1.0):
            raise ValueError("Decay rate must be in [0.0, 1.0]")
        
        time_elapsed = current_time - self.original_timestamp
        decay_factor = jnp.exp(-decay_rate * time_elapsed)
        
        return ExperientialTrace(
            trace_content=self.trace_content,
            original_timestamp=self.original_timestamp,
            trace_strength=float(self.trace_strength * decay_factor),
            experiential_context=self.experiential_context,
            affective_resonance=float(self.affective_resonance * (0.5 + 0.5 * decay_factor)),
            accessibility_index=float(self.accessibility_index * decay_factor)
        )
    
    def assess_recall_probability(self, recall_cue: Array) -> float:
        """Assess probability of successful recall given cue."""
        if not jnp.all(jnp.isfinite(recall_cue)):
            raise ValueError("Recall cue must contain finite values")
        
        if recall_cue.size != self.trace_content.size:
            return 0.0
        
        cue_match = jnp.corrcoef(
            self.trace_content.flatten(),
            recall_cue.flatten()
        )[0, 1]
        
        if jnp.isnan(cue_match):
            cue_match = 0.0
        else:
            cue_match = jnp.clip(cue_match, 0.0, 1.0)
        
        return float(cue_match * self.accessibility_index * self.trace_strength)


@dataclass(frozen=True, slots=True)
class SedimentLayer:
    """Phenomenological sediment layer in experiential memory.
    
    Represents a layer of sedimented experience that forms the
    background horizon of present experience.
    
    Business Rules:
    - Deeper layers are more stable but less accessible
    - Layers interact to form experiential background
    - Support meaning constitution through accumulated experience
    """
    
    sedimented_content: Array        # Content accumulated in this layer
    sediment_depth: int              # Depth of sediment layer (0 = most recent)
    consolidation_strength: float    # [0.0, 1.0] - how consolidated this layer is
    experiential_density: float      # [0.0, 1.0] - density of experiences in layer
    background_influence: float      # [0.0, 1.0] - influence on present experience
    layer_coherence: float          # [0.0, 1.0] - internal coherence of layer
    
    def __post_init__(self):
        """Validate sediment layer invariants."""
        if self.sediment_depth < 0:
            raise ValueError("Sediment depth cannot be negative")
            
        measures = [self.consolidation_strength, self.experiential_density, 
                   self.background_influence, self.layer_coherence]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Sediment measures must be in [0.0, 1.0]")
            
        if not jnp.all(jnp.isfinite(self.sedimented_content)):
            raise ValueError("Sedimented content must contain finite values")
    
    def add_sediment(self, new_content: Array, sediment_rate: float = 0.1) -> SedimentLayer:
        """Add new content to sediment layer."""
        if not jnp.all(jnp.isfinite(new_content)):
            raise ValueError("New content must contain finite values")
        if not (0.0 <= sediment_rate <= 1.0):
            raise ValueError("Sediment rate must be in [0.0, 1.0]")
        
        # Integrate new content into layer
        integration_strength = sediment_rate * (1.0 - self.experiential_density)
        new_sedimented_content = (
            (1.0 - integration_strength) * self.sedimented_content +
            integration_strength * new_content
        )
        
        # Update layer properties
        new_density = jnp.clip(self.experiential_density + 0.1 * sediment_rate, 0.0, 1.0)
        new_consolidation = jnp.clip(self.consolidation_strength + 0.05 * sediment_rate, 0.0, 1.0)
        
        return SedimentLayer(
            sedimented_content=new_sedimented_content,
            sediment_depth=self.sediment_depth,
            consolidation_strength=float(new_consolidation),
            experiential_density=float(new_density),
            background_influence=self.background_influence,
            layer_coherence=self.layer_coherence * 0.99  # Slight coherence cost
        )
    
    def assess_accessibility(self, depth_penalty: float = 0.1) -> float:
        """Assess accessibility of this sediment layer."""
        depth_factor = jnp.exp(-depth_penalty * self.sediment_depth)
        
        return float(
            self.consolidation_strength * 
            self.layer_coherence * 
            depth_factor * 
            (0.5 + 0.5 * self.background_influence)
        )


@dataclass(frozen=True, slots=True)
class RecallContext:
    """Context for experiential memory recall.
    
    Represents the contextual factors that influence recall
    of experiential memories.
    
    Business Rules:
    - Context shapes recall accessibility and content
    - Multiple contextual factors can combine
    - Recall context influences meaning constitution
    """
    
    recall_cue: Array               # Cue triggering recall
    contextual_factors: Array      # Additional contextual factors
    recall_mode: RecallMode         # Mode of recall being employed
    affective_state: Array         # Current affective state influencing recall
    temporal_proximity: float      # [0.0, 1.0] - temporal proximity to target memories
    contextual_coherence: float    # [0.0, 1.0] - coherence of recall context
    
    def __post_init__(self):
        """Validate recall context invariants."""
        measures = [self.temporal_proximity, self.contextual_coherence]
        if any(not (0.0 <= m <= 1.0) for m in measures):
            raise ValueError("Recall measures must be in [0.0, 1.0]")
            
        arrays = [self.recall_cue, self.contextual_factors, self.affective_state]
        for arr in arrays:
            if not jnp.all(jnp.isfinite(arr)):
                raise ValueError("Recall context arrays must contain finite values")
    
    def enhance_context(self, additional_context: Array) -> RecallContext:
        """Enhance recall context with additional contextual information."""
        if not jnp.all(jnp.isfinite(additional_context)):
            raise ValueError("Additional context must contain finite values")
        
        # Integrate additional context
        enhanced_factors = self.contextual_factors + 0.2 * additional_context
        
        # Assess coherence change
        context_correlation = jnp.corrcoef(
            self.contextual_factors.flatten(),
            additional_context.flatten()
        )[0, 1]
        
        if jnp.isnan(context_correlation):
            coherence_change = 0.0
        else:
            coherence_change = 0.1 * jnp.clip(context_correlation, -0.1, 0.1)
        
        return RecallContext(
            recall_cue=self.recall_cue,
            contextual_factors=enhanced_factors,
            recall_mode=self.recall_mode,
            affective_state=self.affective_state,
            temporal_proximity=self.temporal_proximity,
            contextual_coherence=float(jnp.clip(self.contextual_coherence + coherence_change, 0.0, 1.0))
        )
    
    def assess_recall_strength(self) -> float:
        """Assess strength of recall given this context."""
        cue_strength = jnp.linalg.norm(self.recall_cue) / (jnp.linalg.norm(self.recall_cue) + 1.0)
        context_strength = jnp.linalg.norm(self.contextual_factors) / (jnp.linalg.norm(self.contextual_factors) + 1.0)
        
        return float(jnp.clip(
            cue_strength * context_strength * self.temporal_proximity * self.contextual_coherence,
            0.0, 1.0
        ))


@dataclass(frozen=True, slots=True)
class AssociativeLink:
    """Associative link between experiential traces.
    
    Represents associative connections that enable one experience
    to trigger recall of related experiences.
    
    Business Rules:
    - Links strengthen with repeated activation
    - Enable transitive associative recall
    - Have directionality and strength measures
    """
    
    source_trace_id: str            # ID of source experiential trace
    target_trace_id: str            # ID of target experiential trace
    association_strength: float     # [0.0, 1.0] - strength of associative link
    link_type: str                  # Type of association (similarity, contiguity, etc.)
    activation_history: Array       # History of link activations
    bidirectional: bool            # Whether link works in both directions
    
    def __post_init__(self):
        """Validate associative link invariants."""
        if not (0.0 <= self.association_strength <= 1.0):
            raise ValueError("Association strength must be in [0.0, 1.0]")
            
        if not self.source_trace_id or not self.target_trace_id:
            raise ValueError("Source and target trace IDs cannot be empty")
            
        if not jnp.all(jnp.isfinite(self.activation_history)):
            raise ValueError("Activation history must contain finite values")
    
    def strengthen_link(self, activation_strength: float) -> AssociativeLink:
        """Strengthen associative link through activation."""
        if not (0.0 <= activation_strength <= 1.0):
            raise ValueError("Activation strength must be in [0.0, 1.0]")
        
        # Update activation history
        new_history = jnp.roll(self.activation_history, -1)
        new_history = new_history.at[-1].set(activation_strength)
        
        # Strengthen association based on activation
        strength_increment = 0.1 * activation_strength * (1.0 - self.association_strength)
        new_strength = jnp.clip(self.association_strength + strength_increment, 0.0, 1.0)
        
        return AssociativeLink(
            source_trace_id=self.source_trace_id,
            target_trace_id=self.target_trace_id,
            association_strength=float(new_strength),
            link_type=self.link_type,
            activation_history=new_history,
            bidirectional=self.bidirectional
        )
    
    def assess_link_stability(self) -> float:
        """Assess stability of associative link over time."""
        if self.activation_history.size == 0:
            return 0.0
        
        recent_activations = self.activation_history[-min(5, self.activation_history.size):]
        activation_consistency = 1.0 - jnp.std(recent_activations) / (jnp.mean(recent_activations) + 1e-6)
        
        return float(jnp.clip(
            self.association_strength * activation_consistency,
            0.0, 1.0
        ))