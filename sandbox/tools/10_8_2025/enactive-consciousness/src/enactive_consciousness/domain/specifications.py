"""Domain Specifications for Enactive Consciousness.

This module defines specifications following Eric Evans' DDD methodology.
Specifications encapsulate domain rules and business logic that can be
evaluated against domain objects, enabling flexible and reusable business rules.

Theoretical Foundations:
- Specifications test consciousness phenomena against theoretical requirements
- Follow phenomenological and enactivist criteria for validity
- Enable composition of complex domain rules through specification patterns
- Support runtime evaluation and validation of consciousness states

Design Principles:
1. Single Responsibility: Each specification tests one business rule
2. Composability: Specifications can be combined using logical operators
3. Expressiveness: Specifications use ubiquitous language
4. Testability: Business rules are isolated and testable
5. Theoretical Grounding: Rules reflect consciousness theory accurately
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Protocol, Union
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

from .entities import (
    TemporalMomentEntity,
    EmbodiedExperienceEntity,
    CircularCausalityEntity,
    ExperientialMemoryEntity,
)

from .aggregates import (
    TemporalConsciousnessAggregate,
    EmbodiedExperienceAggregate,
    CircularCausalityAggregate,
    ExperientialMemoryAggregate,
)


# ============================================================================
# BASE SPECIFICATION INFRASTRUCTURE
# ============================================================================

class SpecificationError(Exception):
    """Base exception for specification operations."""
    pass


class SpecificationEvaluationError(SpecificationError):
    """Raised when specification evaluation fails."""
    pass


T = Protocol  # Type variable for specification targets


class Specification(Protocol[T]):
    """Base specification interface for domain objects."""
    
    def is_satisfied_by(self, candidate: T) -> bool:
        """Test if candidate satisfies this specification."""
        ...
    
    def why_not_satisfied_by(self, candidate: T) -> List[str]:
        """Explain why candidate does not satisfy specification."""
        ...
    
    def and_(self, other: 'Specification[T]') -> 'CompositeSpecification[T]':
        """Combine specifications with AND logic."""
        ...
    
    def or_(self, other: 'Specification[T]') -> 'CompositeSpecification[T]':
        """Combine specifications with OR logic."""
        ...
    
    def not_(self) -> 'CompositeSpecification[T]':
        """Negate this specification."""
        ...


class CompositeSpecification:
    """Composite specification supporting logical operations."""
    
    def __init__(
        self,
        left: Specification,
        operator: str,
        right: Optional[Specification] = None
    ):
        self.left = left
        self.operator = operator  # 'AND', 'OR', 'NOT'
        self.right = right
    
    def is_satisfied_by(self, candidate) -> bool:
        """Evaluate composite specification."""
        if self.operator == 'AND':
            return self.left.is_satisfied_by(candidate) and self.right.is_satisfied_by(candidate)
        elif self.operator == 'OR':
            return self.left.is_satisfied_by(candidate) or self.right.is_satisfied_by(candidate)
        elif self.operator == 'NOT':
            return not self.left.is_satisfied_by(candidate)
        else:
            raise SpecificationError(f"Unknown operator: {self.operator}")
    
    def why_not_satisfied_by(self, candidate) -> List[str]:
        """Explain why composite specification is not satisfied."""
        reasons = []
        
        if self.operator == 'AND':
            if not self.left.is_satisfied_by(candidate):
                reasons.extend(self.left.why_not_satisfied_by(candidate))
            if not self.right.is_satisfied_by(candidate):
                reasons.extend(self.right.why_not_satisfied_by(candidate))
        elif self.operator == 'OR':
            if not (self.left.is_satisfied_by(candidate) or self.right.is_satisfied_by(candidate)):
                reasons.append("Neither OR condition satisfied:")
                reasons.extend(self.left.why_not_satisfied_by(candidate))
                reasons.extend(self.right.why_not_satisfied_by(candidate))
        elif self.operator == 'NOT':
            if self.left.is_satisfied_by(candidate):
                reasons.append("Negated condition was satisfied when it shouldn't be")
        
        return reasons
    
    def and_(self, other: Specification) -> 'CompositeSpecification':
        """Chain with AND logic."""
        return CompositeSpecification(self, 'AND', other)
    
    def or_(self, other: Specification) -> 'CompositeSpecification':
        """Chain with OR logic."""
        return CompositeSpecification(self, 'OR', other)
    
    def not_(self) -> 'CompositeSpecification':
        """Negate composite specification."""
        return CompositeSpecification(self, 'NOT')


# ============================================================================
# TEMPORAL CONSCIOUSNESS SPECIFICATIONS (Husserlian Phenomenology)
# ============================================================================

class TemporalConsistencySpecification:
    """Specification for temporal consciousness consistency following Husserl.
    
    Tests that temporal moments maintain proper retention-present-protention
    structure with phenomenologically valid relationships.
    
    Business Rules:
    - Temporal weights must sum to 1.0
    - Retention strength must decay with temporal distance
    - Protention must have reasonable anticipatory structure
    - Present moment must have sufficient phenomenological vividness
    """
    
    def __init__(
        self,
        min_present_vividness: float = 0.3,
        min_temporal_coherence: float = 0.2,
        max_temporal_distance: float = 50.0,
        weight_sum_tolerance: float = 1e-6
    ):
        self.min_present_vividness = min_present_vividness
        self.min_temporal_coherence = min_temporal_coherence
        self.max_temporal_distance = max_temporal_distance
        self.weight_sum_tolerance = weight_sum_tolerance
    
    def is_satisfied_by(self, candidate: TemporalMomentEntity) -> bool:
        """Test temporal consistency."""
        try:
            reasons = self.why_not_satisfied_by(candidate)
            return len(reasons) == 0
        except Exception:
            return False
    
    def why_not_satisfied_by(self, candidate: TemporalMomentEntity) -> List[str]:
        """Explain temporal consistency violations."""
        reasons = []
        
        # Check entity state
        if not candidate.primal_impression:
            reasons.append("Missing primal impression")
            return reasons
        
        # Check present moment vividness
        if candidate.primal_impression.phenomenological_vividness < self.min_present_vividness:
            reasons.append(
                f"Present vividness too low: {candidate.primal_impression.phenomenological_vividness:.3f} < {self.min_present_vividness}"
            )
        
        # Check temporal coherence
        if candidate.temporal_coherence < self.min_temporal_coherence:
            reasons.append(
                f"Temporal coherence too low: {candidate.temporal_coherence:.3f} < {self.min_temporal_coherence}"
            )
        
        # Check synthesis weights if present
        if candidate.synthesis_weights:
            weight_sum = (
                candidate.synthesis_weights.retention_weight + 
                candidate.synthesis_weights.present_weight +
                candidate.synthesis_weights.protention_weight
            )
            
            if abs(weight_sum - 1.0) > self.weight_sum_tolerance:
                reasons.append(
                    f"Synthesis weights don't sum to 1.0: {weight_sum:.6f}"
                )
            
            # Check synthesis coherence
            if candidate.synthesis_weights.synthesis_coherence < self.min_temporal_coherence:
                reasons.append(
                    f"Synthesis coherence too low: {candidate.synthesis_weights.synthesis_coherence:.3f}"
                )
        
        # Check retention validity if present
        if candidate.retention:
            if candidate.retention.temporal_distance > self.max_temporal_distance:
                reasons.append(
                    f"Retention temporal distance too large: {candidate.retention.temporal_distance:.1f} > {self.max_temporal_distance}"
                )
            
            if candidate.retention.retention_strength < 0.0 or candidate.retention.retention_strength > 1.0:
                reasons.append(
                    f"Retention strength out of bounds: {candidate.retention.retention_strength:.3f}"
                )
        
        # Check protention validity if present
        if candidate.protention:
            if candidate.protention.anticipatory_distance < 0.0:
                reasons.append(
                    f"Protention anticipatory distance negative: {candidate.protention.anticipatory_distance:.1f}"
                )
            
            if candidate.protention.expectation_strength < 0.0 or candidate.protention.expectation_strength > 1.0:
                reasons.append(
                    f"Protention expectation strength out of bounds: {candidate.protention.expectation_strength:.3f}"
                )
        
        return reasons
    
    def and_(self, other: Specification) -> CompositeSpecification:
        """Combine with another specification using AND logic."""
        return CompositeSpecification(self, 'AND', other)
    
    def or_(self, other: Specification) -> CompositeSpecification:
        """Combine with another specification using OR logic."""
        return CompositeSpecification(self, 'OR', other)
    
    def not_(self) -> CompositeSpecification:
        """Negate this specification."""
        return CompositeSpecification(self, 'NOT')


class TemporalFlowContinuitySpecification:
    """Specification for temporal flow continuity across moments.
    
    Tests that temporal flow maintains continuity following
    Husserlian analysis of temporal consciousness stream.
    """
    
    def __init__(
        self,
        max_coherence_gap: float = 0.5,
        max_temporal_gap: float = 10.0,
        min_content_correlation: float = 0.1
    ):
        self.max_coherence_gap = max_coherence_gap
        self.max_temporal_gap = max_temporal_gap
        self.min_content_correlation = min_content_correlation
    
    def is_satisfied_by(self, candidate: List[TemporalMomentEntity]) -> bool:
        """Test temporal flow continuity."""
        reasons = self.why_not_satisfied_by(candidate)
        return len(reasons) == 0
    
    def why_not_satisfied_by(self, candidate: List[TemporalMomentEntity]) -> List[str]:
        """Explain temporal flow continuity violations."""
        reasons = []
        
        if len(candidate) < 2:
            return reasons  # Single moment is trivially continuous
        
        for i in range(len(candidate) - 1):
            current = candidate[i]
            next_moment = candidate[i + 1]
            
            # Check coherence continuity
            coherence_gap = abs(current.temporal_coherence - next_moment.temporal_coherence)
            if coherence_gap > self.max_coherence_gap:
                reasons.append(
                    f"Coherence gap too large at position {i}: {coherence_gap:.3f} > {self.max_coherence_gap}"
                )
            
            # Check temporal continuity
            if current.primal_impression and next_moment.primal_impression:
                temporal_gap = abs(
                    next_moment.primal_impression.impression_timestamp - 
                    current.primal_impression.impression_timestamp
                )
                
                if temporal_gap > self.max_temporal_gap:
                    reasons.append(
                        f"Temporal gap too large at position {i}: {temporal_gap:.1f} > {self.max_temporal_gap}"
                    )
                
                # Check content continuity
                current_content = current.primal_impression.impression_content
                next_content = next_moment.primal_impression.impression_content
                
                if current_content.shape == next_content.shape:
                    correlation = jnp.corrcoef(
                        current_content.flatten(),
                        next_content.flatten()
                    )[0, 1]
                    
                    if not jnp.isnan(correlation) and correlation < self.min_content_correlation:
                        reasons.append(
                            f"Content correlation too low at position {i}: {correlation:.3f} < {self.min_content_correlation}"
                        )
        
        return reasons
    
    def and_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'AND', other)
    
    def or_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'OR', other)
    
    def not_(self) -> CompositeSpecification:
        return CompositeSpecification(self, 'NOT')


# ============================================================================
# EMBODIED CONSCIOUSNESS SPECIFICATIONS (Merleau-Pontian Phenomenology)
# ============================================================================

class EmbodiedCoherenceSpecification:
    """Specification for embodied experience coherence following Merleau-Ponty.
    
    Tests that embodied experiences maintain coherent body schema
    with proper motor intentionality and proprioceptive grounding.
    
    Business Rules:
    - Body boundary must have sufficient confidence
    - Motor intentions must be grounded in current embodied state
    - Proprioceptive field must provide stable postural configuration
    - Tool incorporations must respect extension capacity limits
    """
    
    def __init__(
        self,
        min_embodiment_coherence: float = 0.3,
        min_boundary_confidence: float = 0.2,
        min_proprioceptive_clarity: float = 0.2,
        min_postural_stability: float = 0.3,
        min_motor_coherence: float = 0.2
    ):
        self.min_embodiment_coherence = min_embodiment_coherence
        self.min_boundary_confidence = min_boundary_confidence
        self.min_proprioceptive_clarity = min_proprioceptive_clarity
        self.min_postural_stability = min_postural_stability
        self.min_motor_coherence = min_motor_coherence
    
    def is_satisfied_by(self, candidate: EmbodiedExperienceEntity) -> bool:
        """Test embodied coherence."""
        reasons = self.why_not_satisfied_by(candidate)
        return len(reasons) == 0
    
    def why_not_satisfied_by(self, candidate: EmbodiedExperienceEntity) -> List[str]:
        """Explain embodied coherence violations."""
        reasons = []
        
        # Check overall embodiment coherence
        if candidate.embodiment_coherence < self.min_embodiment_coherence:
            reasons.append(
                f"Embodiment coherence too low: {candidate.embodiment_coherence:.3f} < {self.min_embodiment_coherence}"
            )
        
        # Check body boundary
        if candidate.body_boundary:
            if candidate.body_boundary.boundary_confidence < self.min_boundary_confidence:
                reasons.append(
                    f"Body boundary confidence too low: {candidate.body_boundary.boundary_confidence:.3f} < {self.min_boundary_confidence}"
                )
            
            # Check boundary coherence
            boundary_coherence = candidate.body_boundary.assess_boundary_coherence()
            if boundary_coherence < self.min_boundary_confidence:
                reasons.append(
                    f"Body boundary coherence too low: {boundary_coherence:.3f} < {self.min_boundary_confidence}"
                )
        else:
            reasons.append("Missing body boundary")
        
        # Check proprioceptive field
        if candidate.proprioceptive_field:
            if candidate.proprioceptive_field.proprioceptive_clarity < self.min_proprioceptive_clarity:
                reasons.append(
                    f"Proprioceptive clarity too low: {candidate.proprioceptive_field.proprioceptive_clarity:.3f} < {self.min_proprioceptive_clarity}"
                )
            
            if candidate.proprioceptive_field.postural_stability < self.min_postural_stability:
                reasons.append(
                    f"Postural stability too low: {candidate.proprioceptive_field.postural_stability:.3f} < {self.min_postural_stability}"
                )
            
            # Check proprioceptive coherence
            prop_coherence = candidate.proprioceptive_field.assess_proprioceptive_coherence()
            if prop_coherence < self.min_proprioceptive_clarity:
                reasons.append(
                    f"Proprioceptive coherence too low: {prop_coherence:.3f} < {self.min_proprioceptive_clarity}"
                )
        else:
            reasons.append("Missing proprioceptive field")
        
        # Check motor intention
        if candidate.motor_intention:
            motor_coherence = candidate.motor_intention.assess_motor_coherence()
            if motor_coherence < self.min_motor_coherence:
                reasons.append(
                    f"Motor coherence too low: {motor_coherence:.3f} < {self.min_motor_coherence}"
                )
            
            if candidate.motor_intention.embodied_confidence < self.min_motor_coherence:
                reasons.append(
                    f"Motor embodied confidence too low: {candidate.motor_intention.embodied_confidence:.3f} < {self.min_motor_coherence}"
                )
        
        return reasons
    
    def and_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'AND', other)
    
    def or_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'OR', other)
    
    def not_(self) -> CompositeSpecification:
        return CompositeSpecification(self, 'NOT')


class MotorIntentionalitySpecification:
    """Specification for motor intentionality validity.
    
    Tests that motor intentions are properly grounded in embodied
    state and maintain phenomenological directedness.
    """
    
    def __init__(
        self,
        min_action_readiness: float = 0.2,
        min_embodied_confidence: float = 0.2,
        min_motor_coherence: float = 0.2,
        max_intention_magnitude: float = 10.0
    ):
        self.min_action_readiness = min_action_readiness
        self.min_embodied_confidence = min_embodied_confidence
        self.min_motor_coherence = min_motor_coherence
        self.max_intention_magnitude = max_intention_magnitude
    
    def is_satisfied_by(self, candidate: MotorIntention) -> bool:
        """Test motor intentionality."""
        reasons = self.why_not_satisfied_by(candidate)
        return len(reasons) == 0
    
    def why_not_satisfied_by(self, candidate: MotorIntention) -> List[str]:
        """Explain motor intentionality violations."""
        reasons = []
        
        # Check action readiness
        if candidate.action_readiness < self.min_action_readiness:
            reasons.append(
                f"Action readiness too low: {candidate.action_readiness:.3f} < {self.min_action_readiness}"
            )
        
        # Check embodied confidence
        if candidate.embodied_confidence < self.min_embodied_confidence:
            reasons.append(
                f"Embodied confidence too low: {candidate.embodied_confidence:.3f} < {self.min_embodied_confidence}"
            )
        
        # Check motor coherence
        motor_coherence = candidate.assess_motor_coherence()
        if motor_coherence < self.min_motor_coherence:
            reasons.append(
                f"Motor coherence too low: {motor_coherence:.3f} < {self.min_motor_coherence}"
            )
        
        # Check motor vector magnitude
        motor_magnitude = float(jnp.linalg.norm(candidate.motor_vector))
        if motor_magnitude > self.max_intention_magnitude:
            reasons.append(
                f"Motor intention magnitude too large: {motor_magnitude:.3f} > {self.max_intention_magnitude}"
            )
        
        # Check for finite values
        if not jnp.all(jnp.isfinite(candidate.motor_vector)):
            reasons.append("Motor vector contains non-finite values")
        
        if not jnp.all(jnp.isfinite(candidate.motor_schema_activation)):
            reasons.append("Motor schema activation contains non-finite values")
        
        return reasons
    
    def and_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'AND', other)
    
    def or_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'OR', other)
    
    def not_(self) -> CompositeSpecification:
        return CompositeSpecification(self, 'NOT')


# ============================================================================
# CIRCULAR CAUSALITY SPECIFICATIONS (Varela-Maturana Theory)
# ============================================================================

class CouplingStabilitySpecification:
    """Specification for circular causality coupling stability.
    
    Tests that structural coupling maintains stability and viability
    following Varela-Maturana autopoietic theory.
    
    Business Rules:
    - Coupling strength must maintain mutual specification
    - Autopoietic processes must preserve organizational closure
    - Agent-environment coupling must show structural congruence
    - Meaning emergence must be grounded in coupling history
    """
    
    def __init__(
        self,
        min_coupling_stability: float = 0.3,
        min_mutual_specification: float = 0.2,
        min_structural_congruence: float = 0.2,
        min_autopoietic_viability: float = 0.4,
        min_organizational_closure: float = 0.5
    ):
        self.min_coupling_stability = min_coupling_stability
        self.min_mutual_specification = min_mutual_specification
        self.min_structural_congruence = min_structural_congruence
        self.min_autopoietic_viability = min_autopoietic_viability
        self.min_organizational_closure = min_organizational_closure
    
    def is_satisfied_by(self, candidate: CircularCausalityEntity) -> bool:
        """Test coupling stability."""
        reasons = self.why_not_satisfied_by(candidate)
        return len(reasons) == 0
    
    def why_not_satisfied_by(self, candidate: CircularCausalityEntity) -> List[str]:
        """Explain coupling stability violations."""
        reasons = []
        
        # Check overall coupling coherence
        if candidate.coupling_coherence < self.min_coupling_stability:
            reasons.append(
                f"Coupling coherence too low: {candidate.coupling_coherence:.3f} < {self.min_coupling_stability}"
            )
        
        # Check coupling strength
        if candidate.coupling_strength:
            if candidate.coupling_strength.mutual_specification < self.min_mutual_specification:
                reasons.append(
                    f"Mutual specification too low: {candidate.coupling_strength.mutual_specification:.3f} < {self.min_mutual_specification}"
                )
            
            if candidate.coupling_strength.structural_congruence < self.min_structural_congruence:
                reasons.append(
                    f"Structural congruence too low: {candidate.coupling_strength.structural_congruence:.3f} < {self.min_structural_congruence}"
                )
            
            if candidate.coupling_strength.coupling_stability < self.min_coupling_stability:
                reasons.append(
                    f"Coupling stability too low: {candidate.coupling_strength.coupling_stability:.3f} < {self.min_coupling_stability}"
                )
        else:
            reasons.append("Missing coupling strength")
        
        # Check autopoietic process
        if candidate.autopoietic_process:
            autopoietic_viability = candidate.autopoietic_process.assess_autopoietic_viability()
            if autopoietic_viability < self.min_autopoietic_viability:
                reasons.append(
                    f"Autopoietic viability too low: {autopoietic_viability:.3f} < {self.min_autopoietic_viability}"
                )
            
            if candidate.autopoietic_process.organizational_closure < self.min_organizational_closure:
                reasons.append(
                    f"Organizational closure too low: {candidate.autopoietic_process.organizational_closure:.3f} < {self.min_organizational_closure}"
                )
        else:
            reasons.append("Missing autopoietic process")
        
        # Check structural coupling
        if candidate.structural_coupling:
            coupling_maturity = candidate.structural_coupling.assess_coupling_maturity()
            if coupling_maturity < self.min_coupling_stability:
                reasons.append(
                    f"Coupling maturity too low: {coupling_maturity:.3f} < {self.min_coupling_stability}"
                )
        else:
            reasons.append("Missing structural coupling")
        
        return reasons
    
    def and_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'AND', other)
    
    def or_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'OR', other)
    
    def not_(self) -> CompositeSpecification:
        return CompositeSpecification(self, 'NOT')


class MeaningEmergenceValiditySpecification:
    """Specification for valid meaning emergence.
    
    Tests that meaning emergence follows enactivist principles
    with proper contextual grounding and temporal development.
    """
    
    def __init__(
        self,
        min_emergence_strength: float = 0.3,
        min_enactive_significance: float = 0.2,
        min_meaning_coherence: float = 0.3,
        min_contextual_grounding: float = 0.2
    ):
        self.min_emergence_strength = min_emergence_strength
        self.min_enactive_significance = min_enactive_significance
        self.min_meaning_coherence = min_meaning_coherence
        self.min_contextual_grounding = min_contextual_grounding
    
    def is_satisfied_by(self, candidate: MeaningEmergence) -> bool:
        """Test meaning emergence validity."""
        reasons = self.why_not_satisfied_by(candidate)
        return len(reasons) == 0
    
    def why_not_satisfied_by(self, candidate: MeaningEmergence) -> List[str]:
        """Explain meaning emergence violations."""
        reasons = []
        
        # Check emergence strength
        if candidate.emergence_strength < self.min_emergence_strength:
            reasons.append(
                f"Emergence strength too low: {candidate.emergence_strength:.3f} < {self.min_emergence_strength}"
            )
        
        # Check enactive significance
        if candidate.enactive_significance < self.min_enactive_significance:
            reasons.append(
                f"Enactive significance too low: {candidate.enactive_significance:.3f} < {self.min_enactive_significance}"
            )
        
        # Check meaning coherence
        if candidate.meaning_coherence < self.min_meaning_coherence:
            reasons.append(
                f"Meaning coherence too low: {candidate.meaning_coherence:.3f} < {self.min_meaning_coherence}"
            )
        
        # Check contextual grounding strength
        contextual_strength = jnp.linalg.norm(candidate.contextual_grounding)
        normalized_strength = contextual_strength / (contextual_strength + 1.0)
        
        if normalized_strength < self.min_contextual_grounding:
            reasons.append(
                f"Contextual grounding too weak: {normalized_strength:.3f} < {self.min_contextual_grounding}"
            )
        
        # Check for finite values
        if not jnp.all(jnp.isfinite(candidate.semantic_content)):
            reasons.append("Semantic content contains non-finite values")
        
        if not jnp.all(jnp.isfinite(candidate.contextual_grounding)):
            reasons.append("Contextual grounding contains non-finite values")
        
        return reasons
    
    def and_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'AND', other)
    
    def or_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'OR', other)
    
    def not_(self) -> CompositeSpecification:
        return CompositeSpecification(self, 'NOT')


# ============================================================================
# EXPERIENTIAL MEMORY SPECIFICATIONS (Phenomenological Memory Theory)
# ============================================================================

class MemoryRetentionSpecification:
    """Specification for experiential memory retention quality.
    
    Tests that memory systems maintain proper trace accessibility,
    sedimentation organization, and associative coherence.
    
    Business Rules:
    - Experiential traces must maintain phenomenological accessibility
    - Sediment layers must be properly depth-ordered
    - Associative links must maintain bidirectional consistency
    - Memory coherence must support recall operations
    """
    
    def __init__(
        self,
        min_memory_coherence: float = 0.3,
        min_trace_accessibility: float = 0.1,
        min_associative_strength: float = 0.2,
        max_sediment_depth: int = 10,
        min_sediment_coherence: float = 0.4
    ):
        self.min_memory_coherence = min_memory_coherence
        self.min_trace_accessibility = min_trace_accessibility
        self.min_associative_strength = min_associative_strength
        self.max_sediment_depth = max_sediment_depth
        self.min_sediment_coherence = min_sediment_coherence
    
    def is_satisfied_by(self, candidate: ExperientialMemoryEntity) -> bool:
        """Test memory retention quality."""
        reasons = self.why_not_satisfied_by(candidate)
        return len(reasons) == 0
    
    def why_not_satisfied_by(self, candidate: ExperientialMemoryEntity) -> List[str]:
        """Explain memory retention violations."""
        reasons = []
        
        # Check overall memory coherence
        if candidate.memory_coherence < self.min_memory_coherence:
            reasons.append(
                f"Memory coherence too low: {candidate.memory_coherence:.3f} < {self.min_memory_coherence}"
            )
        
        # Check experiential traces
        if candidate.experiential_traces:
            inaccessible_traces = 0
            weak_traces = 0
            
            for trace_id, trace in candidate.experiential_traces.items():
                if trace.accessibility_index < self.min_trace_accessibility:
                    inaccessible_traces += 1
                
                if trace.trace_strength < self.min_trace_accessibility:
                    weak_traces += 1
            
            total_traces = len(candidate.experiential_traces)
            inaccessible_ratio = inaccessible_traces / total_traces
            weak_ratio = weak_traces / total_traces
            
            if inaccessible_ratio > 0.5:
                reasons.append(
                    f"Too many inaccessible traces: {inaccessible_ratio:.2%} > 50%"
                )
            
            if weak_ratio > 0.6:
                reasons.append(
                    f"Too many weak traces: {weak_ratio:.2%} > 60%"
                )
        else:
            reasons.append("No experiential traces present")
        
        # Check sediment layers
        if candidate.sediment_layers:
            # Check depth ordering
            for i in range(len(candidate.sediment_layers) - 1):
                current_layer = candidate.sediment_layers[i]
                next_layer = candidate.sediment_layers[i + 1]
                
                if current_layer.sediment_depth >= next_layer.sediment_depth:
                    reasons.append(
                        f"Sediment layers not properly ordered at position {i}: {current_layer.sediment_depth} >= {next_layer.sediment_depth}"
                    )
            
            # Check maximum depth
            max_depth = max(layer.sediment_depth for layer in candidate.sediment_layers)
            if max_depth > self.max_sediment_depth:
                reasons.append(
                    f"Sediment depth too large: {max_depth} > {self.max_sediment_depth}"
                )
            
            # Check sediment coherence
            low_coherence_layers = 0
            for layer in candidate.sediment_layers:
                if layer.layer_coherence < self.min_sediment_coherence:
                    low_coherence_layers += 1
            
            if low_coherence_layers > len(candidate.sediment_layers) * 0.3:
                reasons.append(
                    f"Too many low-coherence sediment layers: {low_coherence_layers}/{len(candidate.sediment_layers)}"
                )
        
        # Check associative links
        if candidate.associative_links:
            weak_links = 0
            total_links = 0
            
            for source_id, links in candidate.associative_links.items():
                for link in links:
                    total_links += 1
                    
                    if link.association_strength < self.min_associative_strength:
                        weak_links += 1
                    
                    # Check bidirectional consistency
                    if link.bidirectional:
                        target_id = link.target_trace_id
                        if target_id in candidate.associative_links:
                            reverse_links = candidate.associative_links[target_id]
                            reverse_exists = any(
                                rl.target_trace_id == source_id for rl in reverse_links
                            )
                            
                            if not reverse_exists:
                                reasons.append(
                                    f"Bidirectional link missing reverse: {source_id} -> {target_id}"
                                )
            
            if total_links > 0:
                weak_ratio = weak_links / total_links
                if weak_ratio > 0.4:
                    reasons.append(
                        f"Too many weak associative links: {weak_ratio:.2%} > 40%"
                    )
        
        return reasons
    
    def and_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'AND', other)
    
    def or_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'OR', other)
    
    def not_(self) -> CompositeSpecification:
        return CompositeSpecification(self, 'NOT')


class RecallQualitySpecification:
    """Specification for recall operation quality.
    
    Tests that recall operations maintain proper accessibility
    and contextual relevance following phenomenological principles.
    """
    
    def __init__(
        self,
        min_recall_strength: float = 0.2,
        min_contextual_coherence: float = 0.3,
        max_recall_count: int = 10,
        min_temporal_proximity: float = 0.1
    ):
        self.min_recall_strength = min_recall_strength
        self.min_contextual_coherence = min_contextual_coherence
        self.max_recall_count = max_recall_count
        self.min_temporal_proximity = min_temporal_proximity
    
    def is_satisfied_by(self, candidate: List[Tuple[str, ExperientialTrace, float]]) -> bool:
        """Test recall quality."""
        reasons = self.why_not_satisfied_by(candidate)
        return len(reasons) == 0
    
    def why_not_satisfied_by(self, candidate: List[Tuple[str, ExperientialTrace, float]]) -> List[str]:
        """Explain recall quality violations."""
        reasons = []
        
        if not candidate:
            reasons.append("No recall results")
            return reasons
        
        # Check recall count
        if len(candidate) > self.max_recall_count:
            reasons.append(
                f"Too many recall results: {len(candidate)} > {self.max_recall_count}"
            )
        
        # Check individual recall strengths
        weak_recalls = 0
        for trace_id, trace, strength in candidate:
            if strength < self.min_recall_strength:
                weak_recalls += 1
        
        if weak_recalls > len(candidate) * 0.5:
            reasons.append(
                f"Too many weak recall results: {weak_recalls}/{len(candidate)}"
            )
        
        # Check trace accessibility
        inaccessible_traces = 0
        for trace_id, trace, strength in candidate:
            if trace.accessibility_index < self.min_recall_strength:
                inaccessible_traces += 1
        
        if inaccessible_traces > len(candidate) * 0.3:
            reasons.append(
                f"Too many inaccessible traces recalled: {inaccessible_traces}/{len(candidate)}"
            )
        
        # Check recall strength distribution
        strengths = [strength for _, _, strength in candidate]
        if len(strengths) > 1:
            strength_variance = float(jnp.var(jnp.array(strengths)))
            if strength_variance > 0.5:
                reasons.append(
                    f"Recall strength variance too high: {strength_variance:.3f} > 0.5"
                )
        
        return reasons
    
    def and_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'AND', other)
    
    def or_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'OR', other)
    
    def not_(self) -> CompositeSpecification:
        return CompositeSpecification(self, 'NOT')


# ============================================================================
# CONSCIOUSNESS INTEGRATION SPECIFICATIONS
# ============================================================================

class ConsciousnessIntegrationReadinessSpecification:
    """Specification for consciousness integration readiness.
    
    Tests that consciousness components are ready for integration
    with sufficient coherence and compatibility across domains.
    """
    
    def __init__(
        self,
        min_temporal_coherence: float = 0.4,
        min_embodied_coherence: float = 0.3,
        min_coupling_coherence: float = 0.3,
        min_memory_coherence: float = 0.2,
        min_overall_readiness: float = 0.35
    ):
        self.min_temporal_coherence = min_temporal_coherence
        self.min_embodied_coherence = min_embodied_coherence
        self.min_coupling_coherence = min_coupling_coherence
        self.min_memory_coherence = min_memory_coherence
        self.min_overall_readiness = min_overall_readiness
    
    def is_satisfied_by(self, candidate: Dict[str, Any]) -> bool:
        """Test consciousness integration readiness."""
        reasons = self.why_not_satisfied_by(candidate)
        return len(reasons) == 0
    
    def why_not_satisfied_by(self, candidate: Dict[str, Any]) -> List[str]:
        """Explain consciousness integration readiness violations."""
        reasons = []
        
        # Check overall readiness
        overall_readiness = candidate.get('overall_readiness', 0.0)
        if overall_readiness < self.min_overall_readiness:
            reasons.append(
                f"Overall readiness too low: {overall_readiness:.3f} < {self.min_overall_readiness}"
            )
        
        # Check domain readiness
        domain_readiness = candidate.get('domain_readiness', {})
        
        temporal_readiness = domain_readiness.get('temporal', 0.0)
        if temporal_readiness < self.min_temporal_coherence:
            reasons.append(
                f"Temporal readiness too low: {temporal_readiness:.3f} < {self.min_temporal_coherence}"
            )
        
        embodied_readiness = domain_readiness.get('embodied', 0.0)
        if embodied_readiness < self.min_embodied_coherence:
            reasons.append(
                f"Embodied readiness too low: {embodied_readiness:.3f} < {self.min_embodied_coherence}"
            )
        
        coupling_readiness = domain_readiness.get('causality', 0.0)
        if coupling_readiness < self.min_coupling_coherence:
            reasons.append(
                f"Coupling readiness too low: {coupling_readiness:.3f} < {self.min_coupling_coherence}"
            )
        
        memory_readiness = domain_readiness.get('memory', 0.0)
        if memory_readiness < self.min_memory_coherence:
            reasons.append(
                f"Memory readiness too low: {memory_readiness:.3f} < {self.min_memory_coherence}"
            )
        
        # Check for integration blockers
        integration_blockers = candidate.get('integration_blockers', [])
        if integration_blockers:
            reasons.extend([f"Integration blocker: {blocker}" for blocker in integration_blockers])
        
        return reasons
    
    def and_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'AND', other)
    
    def or_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'OR', other)
    
    def not_(self) -> CompositeSpecification:
        return CompositeSpecification(self, 'NOT')


class ConsciousnessUnitySpecification:
    """Specification for consciousness unity across integrated components.
    
    Tests that integrated consciousness maintains unity while preserving
    the distinctiveness of component domains.
    """
    
    def __init__(
        self,
        min_integration_completeness: float = 0.5,
        min_cross_domain_coherence: float = 0.4,
        min_consciousness_unity: float = 0.4,
        min_phenomenological_grounding: float = 0.3
    ):
        self.min_integration_completeness = min_integration_completeness
        self.min_cross_domain_coherence = min_cross_domain_coherence
        self.min_consciousness_unity = min_consciousness_unity
        self.min_phenomenological_grounding = min_phenomenological_grounding
    
    def is_satisfied_by(self, candidate: Dict[str, float]) -> bool:
        """Test consciousness unity."""
        reasons = self.why_not_satisfied_by(candidate)
        return len(reasons) == 0
    
    def why_not_satisfied_by(self, candidate: Dict[str, float]) -> List[str]:
        """Explain consciousness unity violations."""
        reasons = []
        
        # Check integration completeness
        completeness = candidate.get('integration_completeness', 0.0)
        if completeness < self.min_integration_completeness:
            reasons.append(
                f"Integration completeness too low: {completeness:.3f} < {self.min_integration_completeness}"
            )
        
        # Check cross-domain coherence
        cross_domain_coherence = candidate.get('cross_domain_coherence', 0.0)
        if cross_domain_coherence < self.min_cross_domain_coherence:
            reasons.append(
                f"Cross-domain coherence too low: {cross_domain_coherence:.3f} < {self.min_cross_domain_coherence}"
            )
        
        # Check consciousness unity
        consciousness_unity = candidate.get('consciousness_unity', 0.0)
        if consciousness_unity < self.min_consciousness_unity:
            reasons.append(
                f"Consciousness unity too low: {consciousness_unity:.3f} < {self.min_consciousness_unity}"
            )
        
        # Check phenomenological grounding
        grounding = candidate.get('phenomenological_grounding', 0.0)
        if grounding < self.min_phenomenological_grounding:
            reasons.append(
                f"Phenomenological grounding too low: {grounding:.3f} < {self.min_phenomenological_grounding}"
            )
        
        # Check integration stability
        stability = candidate.get('integration_stability', 0.0)
        if stability < 0.3:  # Minimum stability threshold
            reasons.append(
                f"Integration stability too low: {stability:.3f} < 0.3"
            )
        
        return reasons
    
    def and_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'AND', other)
    
    def or_(self, other: Specification) -> CompositeSpecification:
        return CompositeSpecification(self, 'OR', other)
    
    def not_(self) -> CompositeSpecification:
        return CompositeSpecification(self, 'NOT')


# ============================================================================
# SPECIFICATION FACTORY AND UTILITIES
# ============================================================================

class SpecificationFactory:
    """Factory for creating domain specifications with standard configurations."""
    
    @staticmethod
    def create_temporal_consistency_specification(
        strictness: str = "normal"
    ) -> TemporalConsistencySpecification:
        """Create temporal consistency specification with preset strictness."""
        if strictness == "strict":
            return TemporalConsistencySpecification(
                min_present_vividness=0.5,
                min_temporal_coherence=0.4,
                max_temporal_distance=30.0
            )
        elif strictness == "relaxed":
            return TemporalConsistencySpecification(
                min_present_vividness=0.2,
                min_temporal_coherence=0.1,
                max_temporal_distance=100.0
            )
        else:  # normal
            return TemporalConsistencySpecification()
    
    @staticmethod
    def create_embodied_coherence_specification(
        strictness: str = "normal"
    ) -> EmbodiedCoherenceSpecification:
        """Create embodied coherence specification with preset strictness."""
        if strictness == "strict":
            return EmbodiedCoherenceSpecification(
                min_embodiment_coherence=0.5,
                min_boundary_confidence=0.4,
                min_proprioceptive_clarity=0.4,
                min_postural_stability=0.5
            )
        elif strictness == "relaxed":
            return EmbodiedCoherenceSpecification(
                min_embodiment_coherence=0.2,
                min_boundary_confidence=0.1,
                min_proprioceptive_clarity=0.1,
                min_postural_stability=0.2
            )
        else:  # normal
            return EmbodiedCoherenceSpecification()
    
    @staticmethod
    def create_coupling_stability_specification(
        strictness: str = "normal"
    ) -> CouplingStabilitySpecification:
        """Create coupling stability specification with preset strictness."""
        if strictness == "strict":
            return CouplingStabilitySpecification(
                min_coupling_stability=0.5,
                min_mutual_specification=0.4,
                min_structural_congruence=0.4,
                min_autopoietic_viability=0.6
            )
        elif strictness == "relaxed":
            return CouplingStabilitySpecification(
                min_coupling_stability=0.2,
                min_mutual_specification=0.1,
                min_structural_congruence=0.1,
                min_autopoietic_viability=0.3
            )
        else:  # normal
            return CouplingStabilitySpecification()
    
    @staticmethod
    def create_memory_retention_specification(
        strictness: str = "normal"
    ) -> MemoryRetentionSpecification:
        """Create memory retention specification with preset strictness."""
        if strictness == "strict":
            return MemoryRetentionSpecification(
                min_memory_coherence=0.5,
                min_trace_accessibility=0.2,
                min_associative_strength=0.4,
                max_sediment_depth=8
            )
        elif strictness == "relaxed":
            return MemoryRetentionSpecification(
                min_memory_coherence=0.2,
                min_trace_accessibility=0.05,
                min_associative_strength=0.1,
                max_sediment_depth=15
            )
        else:  # normal
            return MemoryRetentionSpecification()
    
    @staticmethod
    def create_consciousness_integration_specification(
        strictness: str = "normal"
    ) -> ConsciousnessIntegrationReadinessSpecification:
        """Create consciousness integration specification with preset strictness."""
        if strictness == "strict":
            return ConsciousnessIntegrationReadinessSpecification(
                min_temporal_coherence=0.6,
                min_embodied_coherence=0.5,
                min_coupling_coherence=0.5,
                min_memory_coherence=0.4,
                min_overall_readiness=0.5
            )
        elif strictness == "relaxed":
            return ConsciousnessIntegrationReadinessSpecification(
                min_temporal_coherence=0.3,
                min_embodied_coherence=0.2,
                min_coupling_coherence=0.2,
                min_memory_coherence=0.1,
                min_overall_readiness=0.25
            )
        else:  # normal
            return ConsciousnessIntegrationReadinessSpecification()


class SpecificationValidator:
    """Utility for validating objects against multiple specifications."""
    
    def __init__(self):
        self.specifications = []
    
    def add_specification(self, specification: Specification) -> None:
        """Add specification to validation set."""
        self.specifications.append(specification)
    
    def validate(self, candidate: Any) -> Tuple[bool, List[str]]:
        """Validate candidate against all specifications."""
        all_reasons = []
        all_satisfied = True
        
        for spec in self.specifications:
            try:
                satisfied = spec.is_satisfied_by(candidate)
                if not satisfied:
                    all_satisfied = False
                    reasons = spec.why_not_satisfied_by(candidate)
                    all_reasons.extend(reasons)
            except Exception as e:
                all_satisfied = False
                all_reasons.append(f"Specification error: {str(e)}")
        
        return all_satisfied, all_reasons
    
    def validate_with_details(self, candidate: Any) -> Dict[str, Any]:
        """Validate with detailed results per specification."""
        results = {
            'overall_valid': True,
            'specification_results': [],
            'total_violations': 0
        }
        
        for i, spec in enumerate(self.specifications):
            spec_result = {
                'specification_index': i,
                'specification_type': type(spec).__name__,
                'satisfied': False,
                'violations': []
            }
            
            try:
                satisfied = spec.is_satisfied_by(candidate)
                spec_result['satisfied'] = satisfied
                
                if not satisfied:
                    results['overall_valid'] = False
                    violations = spec.why_not_satisfied_by(candidate)
                    spec_result['violations'] = violations
                    results['total_violations'] += len(violations)
            
            except Exception as e:
                results['overall_valid'] = False
                spec_result['violations'] = [f"Evaluation error: {str(e)}"]
                results['total_violations'] += 1
            
            results['specification_results'].append(spec_result)
        
        return results