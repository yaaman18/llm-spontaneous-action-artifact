"""Domain Services for Enactive Consciousness.

This module defines domain services following Eric Evans' DDD methodology.
Domain services encapsulate domain logic that doesn't naturally fit within
entities or value objects, particularly operations that span multiple aggregates.

Theoretical Foundations:
- Services implement complex domain operations spanning aggregates
- Follow consciousness theory in operation semantics and naming
- Orchestrate interactions between different consciousness components
- Maintain domain integrity across complex workflows

Design Principles:
1. Stateless services that coordinate between aggregates
2. Operations named using ubiquitous language
3. Complex business logic that spans multiple bounded contexts
4. Interface segregation for different service responsibilities
5. Domain services are distinct from application services
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple, Protocol
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
    RecallMode,
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

from .repositories import (
    RetentionMemoryRepository,
    ProtentionProjectionRepository,
    BodySchemaRepository,
    MotorSchemaRepository,
    CouplingHistoryRepository,
    AutopoeticStateRepository,
    ExperientialTraceRepository,
    SedimentLayerRepository,
    AssociativeLinkRepository,
)


# ============================================================================
# BASE DOMAIN SERVICE INFRASTRUCTURE
# ============================================================================

class DomainServiceError(Exception):
    """Base exception for domain service operations."""
    pass


class ServiceCoordinationError(DomainServiceError):
    """Raised when service coordination fails."""
    pass


class ConsciousnessIntegrationError(DomainServiceError):
    """Raised when consciousness integration fails.""" 
    pass


class SynthesisQuality(Enum):
    """Quality levels for temporal synthesis."""
    POOR = "poor"
    ADEQUATE = "adequate"
    GOOD = "good"
    EXCELLENT = "excellent"


class IntegrationStrategy(Enum):
    """Strategies for consciousness integration."""
    WEIGHTED_AVERAGE = "weighted_average"
    COMPETITIVE_SELECTION = "competitive_selection"
    TEMPORAL_PRIORITIZED = "temporal_prioritized"
    PHENOMENOLOGICAL_GROUNDED = "phenomenological_grounded"


# ============================================================================
# TEMPORAL SYNTHESIS SERVICE (Husserlian Phenomenology)
# ============================================================================

class TemporalSynthesisService:
    """Domain service for complex temporal consciousness synthesis operations.
    
    Orchestrates temporal synthesis across multiple temporal moments and
    manages complex retention-protention dynamics that span aggregates.
    
    Business Rules:
    - Temporal synthesis must maintain phenomenological consistency
    - Multi-moment synthesis requires coherence validation
    - Temporal flow coordination preserves causal ordering
    - Synthesis quality affects integration readiness
    """
    
    def __init__(
        self,
        retention_repository: RetentionMemoryRepository,
        protention_repository: ProtentionProjectionRepository
    ):
        self.retention_repository = retention_repository
        self.protention_repository = protention_repository
    
    def orchestrate_multi_moment_synthesis(
        self,
        temporal_aggregates: List[TemporalConsciousnessAggregate],
        synthesis_strategy: str = "phenomenologically_grounded"
    ) -> Tuple[Array, SynthesisQuality, Dict[str, float]]:
        """Orchestrate synthesis across multiple temporal moments.
        
        Complex domain operation that synthesizes multiple temporal
        consciousness streams while maintaining phenomenological coherence.
        
        Returns:
            Tuple of (synthesized_content, synthesis_quality, quality_metrics)
        """
        if not temporal_aggregates:
            raise ValueError("Cannot synthesize without temporal aggregates")
        
        # Validate aggregates are in appropriate state
        active_aggregates = [
            agg for agg in temporal_aggregates 
            if agg.current_moment and agg.current_moment.temporal_coherence > 0.2
        ]
        
        if not active_aggregates:
            raise ServiceCoordinationError("No temporal aggregates in suitable state for synthesis")
        
        # Extract temporal components from all aggregates
        temporal_components = []
        synthesis_weights_collection = []
        coherence_scores = []
        
        for aggregate in active_aggregates:
            if aggregate.current_moment:
                # Synthesize individual moment
                moment_synthesis = aggregate.synthesize_temporal_moment()
                temporal_components.append(moment_synthesis)
                
                # Collect synthesis weights and coherence
                if aggregate.current_moment.synthesis_weights:
                    synthesis_weights_collection.append(aggregate.current_moment.synthesis_weights)
                coherence_scores.append(aggregate.current_moment.temporal_coherence)
        
        if not temporal_components:
            raise ServiceCoordinationError("No valid temporal components for synthesis")
        
        # Coordinate multi-moment synthesis based on strategy
        if synthesis_strategy == "phenomenologically_grounded":
            synthesized_content = self._phenomenologically_grounded_synthesis(
                temporal_components, coherence_scores
            )
        elif synthesis_strategy == "temporal_weighted":
            synthesized_content = self._temporal_weighted_synthesis(
                temporal_components, synthesis_weights_collection
            )
        else:
            # Default to simple averaging
            synthesized_content = jnp.mean(jnp.array(temporal_components), axis=0)
        
        # Assess synthesis quality
        synthesis_quality = self._assess_synthesis_quality(
            synthesized_content, temporal_components, coherence_scores
        )
        
        # Compute quality metrics
        quality_metrics = {
            'component_count': len(temporal_components),
            'avg_coherence': float(jnp.mean(jnp.array(coherence_scores))),
            'synthesis_variance': float(jnp.var(jnp.array(temporal_components), axis=0).mean()),
            'temporal_consistency': self._compute_temporal_consistency(temporal_components),
            'phenomenological_grounding': self._assess_phenomenological_grounding(
                synthesized_content, synthesis_weights_collection
            )
        }
        
        return synthesized_content, synthesis_quality, quality_metrics
    
    def coordinate_retention_protention_dynamics(
        self,
        temporal_aggregate: TemporalConsciousnessAggregate,
        retention_decay_rate: float = 0.95,
        protention_update_strategy: str = "expectation_based"
    ) -> Dict[str, Any]:
        """Coordinate complex retention-protention dynamics.
        
        Domain service for managing the interplay between retention decay
        and protention updates across temporal flow evolution.
        
        Returns:
            Dictionary with coordination results and metrics
        """
        if not temporal_aggregate.current_moment:
            raise ValueError("Cannot coordinate dynamics without current temporal moment")
        
        coordination_results = {
            'retentions_processed': 0,
            'protentions_updated': 0,
            'decay_applications': 0,
            'fulfillments_processed': 0,
            'coordination_quality': 0.0
        }
        
        # Process retention dynamics
        accessible_retentions = self.retention_repository.find_phenomenologically_accessible_retentions()
        
        for retention in accessible_retentions:
            # Apply temporal decay
            decayed_retention = retention.decay_retention(retention_decay_rate)
            self.retention_repository.save_retention(decayed_retention)
            coordination_results['retentions_processed'] += 1
            
            if decayed_retention.retention_strength != retention.retention_strength:
                coordination_results['decay_applications'] += 1
        
        # Process protention dynamics
        active_protentions = self.protention_repository.find_active_protentions()
        
        for protention in active_protentions:
            if protention_update_strategy == "expectation_based":
                # Update based on expectation strength
                if temporal_aggregate.current_moment and temporal_aggregate.current_moment.primal_impression:
                    fulfillment_score, updated_protention = protention.fulfill_expectation(
                        temporal_aggregate.current_moment.primal_impression
                    )
                    
                    self.protention_repository.save_protention(updated_protention)
                    coordination_results['protentions_updated'] += 1
                    coordination_results['fulfillments_processed'] += 1
        
        # Assess coordination quality
        coordination_quality = self._assess_coordination_quality(
            temporal_aggregate, accessible_retentions, active_protentions
        )
        coordination_results['coordination_quality'] = coordination_quality
        
        return coordination_results
    
    def synthesize_temporal_flow_continuity(
        self,
        temporal_moments: List[TemporalMomentEntity],
        continuity_threshold: float = 0.5
    ) -> Tuple[bool, float, List[Dict[str, Any]]]:
        """Synthesize temporal flow continuity across moment sequence.
        
        Domain service for ensuring temporal flow continuity and identifying
        discontinuities that may affect consciousness integration.
        
        Returns:
            Tuple of (is_continuous, continuity_strength, discontinuity_reports)
        """
        if len(temporal_moments) < 2:
            return True, 1.0, []  # Single moment is trivially continuous
        
        discontinuity_reports = []
        continuity_measures = []
        
        # Analyze pairwise temporal continuity
        for i in range(len(temporal_moments) - 1):
            current_moment = temporal_moments[i]
            next_moment = temporal_moments[i + 1]
            
            # Assess temporal coherence continuity
            coherence_continuity = self._assess_coherence_continuity(current_moment, next_moment)
            
            # Assess content continuity
            content_continuity = self._assess_content_continuity(current_moment, next_moment)
            
            # Assess temporal ordering
            temporal_ordering = self._assess_temporal_ordering(current_moment, next_moment)
            
            # Overall continuity for this pair
            pair_continuity = jnp.mean(jnp.array([
                coherence_continuity, content_continuity, temporal_ordering
            ]))
            
            continuity_measures.append(float(pair_continuity))
            
            # Report discontinuities
            if pair_continuity < continuity_threshold:
                discontinuity_reports.append({
                    'position': i,
                    'current_moment_id': current_moment.moment_id,
                    'next_moment_id': next_moment.moment_id,
                    'continuity_score': float(pair_continuity),
                    'coherence_continuity': float(coherence_continuity),
                    'content_continuity': float(content_continuity),
                    'temporal_ordering': float(temporal_ordering),
                    'discontinuity_type': self._classify_discontinuity_type(
                        coherence_continuity, content_continuity, temporal_ordering
                    )
                })
        
        # Overall continuity assessment
        overall_continuity = jnp.mean(jnp.array(continuity_measures))
        is_continuous = overall_continuity >= continuity_threshold
        
        return is_continuous, float(overall_continuity), discontinuity_reports
    
    def _phenomenologically_grounded_synthesis(
        self,
        temporal_components: List[Array],
        coherence_scores: List[float]
    ) -> Array:
        """Perform phenomenologically-grounded multi-moment synthesis."""
        # Weight by phenomenological coherence
        coherence_array = jnp.array(coherence_scores)
        normalized_weights = coherence_array / (jnp.sum(coherence_array) + 1e-6)
        
        # Weighted synthesis
        weighted_components = [
            weight * component 
            for weight, component in zip(normalized_weights, temporal_components)
        ]
        
        return jnp.sum(jnp.array(weighted_components), axis=0)
    
    def _temporal_weighted_synthesis(
        self,
        temporal_components: List[Array],
        synthesis_weights_collection: List[TemporalSynthesisWeights]
    ) -> Array:
        """Perform temporal-weighted multi-moment synthesis."""
        if not synthesis_weights_collection:
            return jnp.mean(jnp.array(temporal_components), axis=0)
        
        # Use average synthesis coherence as weights
        coherence_weights = [
            weights.synthesis_coherence for weights in synthesis_weights_collection
        ]
        
        # Pad coherence weights if needed
        while len(coherence_weights) < len(temporal_components):
            coherence_weights.append(0.5)  # Default weight
        
        coherence_array = jnp.array(coherence_weights[:len(temporal_components)])
        normalized_weights = coherence_array / (jnp.sum(coherence_array) + 1e-6)
        
        weighted_components = [
            weight * component 
            for weight, component in zip(normalized_weights, temporal_components)
        ]
        
        return jnp.sum(jnp.array(weighted_components), axis=0)
    
    def _assess_synthesis_quality(
        self,
        synthesized_content: Array,
        temporal_components: List[Array],
        coherence_scores: List[float]
    ) -> SynthesisQuality:
        """Assess quality of temporal synthesis."""
        # Component consistency
        component_variance = float(jnp.var(jnp.array(temporal_components), axis=0).mean())
        
        # Coherence quality
        avg_coherence = float(jnp.mean(jnp.array(coherence_scores)))
        
        # Synthesis magnitude reasonableness
        synthesis_magnitude = float(jnp.linalg.norm(synthesized_content))
        avg_component_magnitude = float(jnp.mean([jnp.linalg.norm(comp) for comp in temporal_components]))
        magnitude_reasonableness = 1.0 - abs(synthesis_magnitude - avg_component_magnitude) / (avg_component_magnitude + 1e-6)
        
        # Combined quality score
        quality_score = jnp.mean(jnp.array([
            1.0 - jnp.clip(component_variance, 0.0, 1.0),  # Lower variance is better
            avg_coherence,
            magnitude_reasonableness
        ]))
        
        if quality_score >= 0.8:
            return SynthesisQuality.EXCELLENT
        elif quality_score >= 0.6:
            return SynthesisQuality.GOOD
        elif quality_score >= 0.4:
            return SynthesisQuality.ADEQUATE
        else:
            return SynthesisQuality.POOR
    
    def _compute_temporal_consistency(self, temporal_components: List[Array]) -> float:
        """Compute temporal consistency across components."""
        if len(temporal_components) < 2:
            return 1.0
        
        pairwise_correlations = []
        for i in range(len(temporal_components)):
            for j in range(i + 1, len(temporal_components)):
                if temporal_components[i].shape == temporal_components[j].shape:
                    correlation = jnp.corrcoef(
                        temporal_components[i].flatten(),
                        temporal_components[j].flatten()
                    )[0, 1]
                    
                    if not jnp.isnan(correlation):
                        pairwise_correlations.append(correlation)
        
        if pairwise_correlations:
            return float(jnp.mean(jnp.array(pairwise_correlations)))
        else:
            return 0.0
    
    def _assess_phenomenological_grounding(
        self,
        synthesized_content: Array,
        synthesis_weights_collection: List[TemporalSynthesisWeights]
    ) -> float:
        """Assess phenomenological grounding of synthesis."""
        if not synthesis_weights_collection:
            return 0.5  # Default moderate grounding
        
        # Average synthesis coherence across weight collections
        coherence_scores = [weights.synthesis_coherence for weights in synthesis_weights_collection]
        avg_coherence = jnp.mean(jnp.array(coherence_scores))
        
        # Assess weight balance (good phenomenological grounding has balanced weights)
        weight_balances = []
        for weights in synthesis_weights_collection:
            balance = 1.0 - jnp.std(jnp.array([
                weights.retention_weight, weights.present_weight, weights.protention_weight
            ]))
            weight_balances.append(balance)
        
        avg_balance = jnp.mean(jnp.array(weight_balances))
        
        return float(0.6 * avg_coherence + 0.4 * avg_balance)
    
    def _assess_coordination_quality(
        self,
        temporal_aggregate: TemporalConsciousnessAggregate,
        retentions: List[RetentionMoment],
        protentions: List[ProtentionalHorizon]
    ) -> float:
        """Assess quality of retention-protention coordination."""
        # Aggregate coherence
        aggregate_coherence = temporal_aggregate.assess_temporal_coherence()
        
        # Retention accessibility ratio
        if retentions:
            accessible_count = sum(1 for r in retentions if r.is_phenomenologically_accessible())
            retention_ratio = accessible_count / len(retentions)
        else:
            retention_ratio = 0.0
        
        # Protention activity ratio
        if protentions:
            active_count = sum(1 for p in protentions if p.is_expectationally_active())
            protention_ratio = active_count / len(protentions)
        else:
            protention_ratio = 0.0
        
        return float(jnp.mean(jnp.array([
            aggregate_coherence, retention_ratio, protention_ratio
        ])))
    
    def _assess_coherence_continuity(
        self,
        current_moment: TemporalMomentEntity,
        next_moment: TemporalMomentEntity
    ) -> float:
        """Assess coherence continuity between temporal moments."""
        coherence_diff = abs(current_moment.temporal_coherence - next_moment.temporal_coherence)
        return 1.0 - coherence_diff  # Lower difference = higher continuity
    
    def _assess_content_continuity(
        self,
        current_moment: TemporalMomentEntity,
        next_moment: TemporalMomentEntity
    ) -> float:
        """Assess content continuity between temporal moments."""
        if not (current_moment.primal_impression and next_moment.primal_impression):
            return 0.0
        
        current_content = current_moment.primal_impression.impression_content
        next_content = next_moment.primal_impression.impression_content
        
        if current_content.shape != next_content.shape:
            return 0.0
        
        correlation = jnp.corrcoef(current_content.flatten(), next_content.flatten())[0, 1]
        
        if jnp.isnan(correlation):
            return 0.0
        
        return float(jnp.clip(correlation, 0.0, 1.0))
    
    def _assess_temporal_ordering(
        self,
        current_moment: TemporalMomentEntity,
        next_moment: TemporalMomentEntity
    ) -> float:
        """Assess temporal ordering correctness between moments."""
        if not (current_moment.primal_impression and next_moment.primal_impression):
            return 0.0
        
        time_diff = (next_moment.primal_impression.impression_timestamp - 
                    current_moment.primal_impression.impression_timestamp)
        
        if time_diff <= 0:
            return 0.0  # Incorrect temporal ordering
        
        # Reasonable temporal distance (not too large, not too small)
        reasonable_distance = jnp.exp(-0.1 * abs(time_diff - 1.0))  # Prefer ~1.0 time units
        
        return float(reasonable_distance)
    
    def _classify_discontinuity_type(
        self,
        coherence_continuity: float,
        content_continuity: float,
        temporal_ordering: float
    ) -> str:
        """Classify type of temporal discontinuity."""
        if temporal_ordering < 0.5:
            return "temporal_ordering_violation"
        elif coherence_continuity < 0.3:
            return "coherence_discontinuity"
        elif content_continuity < 0.3:
            return "content_discontinuity"
        else:
            return "general_discontinuity"


# ============================================================================
# EMBODIED INTEGRATION SERVICE (Merleau-Pontian Phenomenology)
# ============================================================================

class EmbodiedIntegrationService:
    """Domain service for complex embodied consciousness integration operations.
    
    Orchestrates embodied experience integration across body schema, motor
    intentionality, and tool incorporation with phenomenological grounding.
    
    Business Rules:
    - Embodied integration must maintain body schema coherence
    - Tool incorporations require sufficient extension capacity
    - Motor intentions must be grounded in current embodied state
    - Integration quality affects action readiness
    """
    
    def __init__(
        self,
        body_schema_repository: BodySchemaRepository,
        motor_schema_repository: MotorSchemaRepository
    ):
        self.body_schema_repository = body_schema_repository
        self.motor_schema_repository = motor_schema_repository
    
    def orchestrate_multimodal_embodiment_integration(
        self,
        embodied_aggregates: List[EmbodiedExperienceAggregate],
        integration_strategy: str = "schema_coherence_prioritized"
    ) -> Tuple[Array, Dict[str, float], List[Dict[str, Any]]]:
        """Orchestrate integration across multiple embodied experiences.
        
        Complex domain operation that integrates multiple embodied experience
        streams while maintaining body schema coherence and motor capability.
        
        Returns:
            Tuple of (integrated_schema, integration_metrics, integration_reports)
        """
        if not embodied_aggregates:
            raise ValueError("Cannot integrate without embodied aggregates")
        
        # Filter aggregates in suitable state
        active_aggregates = [
            agg for agg in embodied_aggregates
            if agg.current_experience and agg.current_experience.embodiment_coherence > 0.3
        ]
        
        if not active_aggregates:
            raise ServiceCoordinationError("No embodied aggregates in suitable state for integration")
        
        # Extract embodiment components
        body_schemas = []
        motor_intentions = []
        proprioceptive_fields = []
        coherence_scores = []
        
        for aggregate in active_aggregates:
            experience = aggregate.current_experience
            if experience:
                # Get current body schema through integration
                schema = experience.integrate_body_schema(
                    proprioceptive_input=experience.proprioceptive_field.proprioceptive_map,
                    motor_prediction=experience.motor_intention.motor_vector if experience.motor_intention else jnp.zeros(experience.proprioceptive_field.proprioceptive_map.shape),
                    tactile_feedback=jnp.zeros_like(experience.proprioceptive_field.proprioceptive_map)
                )
                body_schemas.append(schema)
                
                if experience.motor_intention:
                    motor_intentions.append(experience.motor_intention.motor_vector)
                
                proprioceptive_fields.append(experience.proprioceptive_field.proprioceptive_map)
                coherence_scores.append(experience.embodiment_coherence)
        
        # Perform integration based on strategy
        if integration_strategy == "schema_coherence_prioritized":
            integrated_schema = self._schema_coherence_integration(
                body_schemas, coherence_scores
            )
        elif integration_strategy == "motor_readiness_prioritized":
            integrated_schema = self._motor_readiness_integration(
                body_schemas, motor_intentions, active_aggregates
            )
        else:
            # Default weighted average
            integrated_schema = jnp.mean(jnp.array(body_schemas), axis=0)
        
        # Compute integration metrics
        integration_metrics = {
            'integrated_components': len(body_schemas),
            'avg_coherence': float(jnp.mean(jnp.array(coherence_scores))),
            'schema_consistency': self._compute_schema_consistency(body_schemas),
            'motor_alignment': self._compute_motor_alignment(motor_intentions),
            'proprioceptive_stability': self._compute_proprioceptive_stability(proprioceptive_fields),
            'integration_quality': 0.0  # Will be computed below
        }
        
        integration_metrics['integration_quality'] = jnp.mean(jnp.array([
            integration_metrics['avg_coherence'],
            integration_metrics['schema_consistency'],
            integration_metrics['motor_alignment'],
            integration_metrics['proprioceptive_stability']
        ]))
        
        # Generate integration reports
        integration_reports = self._generate_integration_reports(
            active_aggregates, integrated_schema, integration_metrics
        )
        
        return integrated_schema, integration_metrics, integration_reports
    
    def coordinate_tool_incorporation_across_experiences(
        self,
        embodied_aggregates: List[EmbodiedExperienceAggregate],
        tool_representations: Dict[str, Array],
        incorporation_strategy: str = "capacity_based"
    ) -> Dict[str, Dict[str, Any]]:
        """Coordinate tool incorporation across multiple embodied experiences.
        
        Domain service for managing tool incorporation when multiple embodied
        experiences need to incorporate the same or related tools.
        
        Returns:
            Dictionary mapping tool IDs to incorporation results
        """
        incorporation_results = {}
        
        for tool_id, tool_representation in tool_representations.items():
            tool_results = {
                'successful_incorporations': 0,
                'failed_incorporations': 0,
                'incorporation_details': [],
                'overall_success_rate': 0.0,
                'average_extension_strength': 0.0
            }
            
            extension_strengths = []
            
            # Attempt incorporation across aggregates
            for aggregate in embodied_aggregates:
                if not aggregate.current_experience:
                    continue
                
                # Check incorporation capacity
                if aggregate.current_experience.body_boundary:
                    extension_capacity = aggregate.current_experience.body_boundary.extension_capacity
                    
                    if extension_capacity >= 0.2:  # Minimum capacity threshold
                        # Attempt incorporation
                        success = aggregate.incorporate_tool_extension(
                            tool_id=tool_id,
                            tool_representation=tool_representation,
                            incorporation_strategy=incorporation_strategy
                        )
                        
                        if success:
                            tool_results['successful_incorporations'] += 1
                            
                            # Estimate extension strength
                            updated_boundary = aggregate.current_experience.body_boundary
                            extension_strength = self._estimate_extension_strength(
                                updated_boundary, tool_representation
                            )
                            extension_strengths.append(extension_strength)
                            
                            tool_results['incorporation_details'].append({
                                'aggregate_id': aggregate.aggregate_id,
                                'success': True,
                                'extension_strength': extension_strength,
                                'remaining_capacity': updated_boundary.extension_capacity
                            })
                        else:
                            tool_results['failed_incorporations'] += 1
                            tool_results['incorporation_details'].append({
                                'aggregate_id': aggregate.aggregate_id,
                                'success': False,
                                'failure_reason': 'insufficient_compatibility'
                            })
                    else:
                        tool_results['failed_incorporations'] += 1
                        tool_results['incorporation_details'].append({
                            'aggregate_id': aggregate.aggregate_id,
                            'success': False,
                            'failure_reason': 'insufficient_capacity'
                        })
            
            # Compute tool incorporation statistics
            total_attempts = tool_results['successful_incorporations'] + tool_results['failed_incorporations']
            if total_attempts > 0:
                tool_results['overall_success_rate'] = tool_results['successful_incorporations'] / total_attempts
            
            if extension_strengths:
                tool_results['average_extension_strength'] = float(jnp.mean(jnp.array(extension_strengths)))
            
            incorporation_results[tool_id] = tool_results
        
        return incorporation_results
    
    def synthesize_motor_intention_coherence(
        self,
        embodied_aggregates: List[EmbodiedExperienceAggregate],
        goal_state: Array,
        contextual_affordances: Array
    ) -> Tuple[Array, float, List[Dict[str, Any]]]:
        """Synthesize motor intention coherence across embodied experiences.
        
        Domain service for creating coherent motor intentions that integrate
        across multiple embodied experience contexts.
        
        Returns:
            Tuple of (synthesized_intention, coherence_score, intention_reports)
        """
        if not embodied_aggregates:
            raise ValueError("Cannot synthesize motor intentions without aggregates")
        
        motor_intentions = []
        action_readiness_scores = []
        intention_reports = []
        
        # Generate motor intentions across aggregates
        for aggregate in embodied_aggregates:
            if aggregate.current_experience:
                # Generate motor intention
                intention_vector = aggregate.generate_motor_intention(
                    goal_state=goal_state,
                    contextual_affordances=contextual_affordances
                )
                
                motor_intentions.append(intention_vector)
                
                # Assess action readiness
                readiness = aggregate.assess_action_readiness()
                action_readiness_scores.append(readiness['overall'])
                
                # Create intention report
                intention_reports.append({
                    'aggregate_id': aggregate.aggregate_id,
                    'intention_magnitude': float(jnp.linalg.norm(intention_vector)),
                    'action_readiness': readiness,
                    'motor_coherence': aggregate.current_experience.motor_intention.assess_motor_coherence() if aggregate.current_experience.motor_intention else 0.0
                })
        
        if not motor_intentions:
            raise ServiceCoordinationError("No motor intentions generated")
        
        # Synthesize motor intentions with readiness weighting
        readiness_weights = jnp.array(action_readiness_scores)
        normalized_weights = readiness_weights / (jnp.sum(readiness_weights) + 1e-6)
        
        weighted_intentions = [
            weight * intention 
            for weight, intention in zip(normalized_weights, motor_intentions)
        ]
        
        synthesized_intention = jnp.sum(jnp.array(weighted_intentions), axis=0)
        
        # Assess coherence of synthesized intention
        coherence_score = self._assess_motor_intention_coherence(
            synthesized_intention, motor_intentions, action_readiness_scores
        )
        
        return synthesized_intention, float(coherence_score), intention_reports
    
    def _schema_coherence_integration(
        self,
        body_schemas: List[Array],
        coherence_scores: List[float]
    ) -> Array:
        """Integrate body schemas prioritizing coherence."""
        coherence_array = jnp.array(coherence_scores)
        normalized_weights = coherence_array / (jnp.sum(coherence_array) + 1e-6)
        
        weighted_schemas = [
            weight * schema 
            for weight, schema in zip(normalized_weights, body_schemas)
        ]
        
        return jnp.sum(jnp.array(weighted_schemas), axis=0)
    
    def _motor_readiness_integration(
        self,
        body_schemas: List[Array],
        motor_intentions: List[Array],
        aggregates: List[EmbodiedExperienceAggregate]
    ) -> Array:
        """Integrate body schemas prioritizing motor readiness."""
        readiness_scores = []
        
        for aggregate in aggregates:
            readiness = aggregate.assess_action_readiness()
            readiness_scores.append(readiness.get('overall', 0.0))
        
        readiness_array = jnp.array(readiness_scores)
        normalized_weights = readiness_array / (jnp.sum(readiness_array) + 1e-6)
        
        weighted_schemas = [
            weight * schema 
            for weight, schema in zip(normalized_weights, body_schemas)
        ]
        
        return jnp.sum(jnp.array(weighted_schemas), axis=0)
    
    def _compute_schema_consistency(self, body_schemas: List[Array]) -> float:
        """Compute consistency across body schemas."""
        if len(body_schemas) < 2:
            return 1.0
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(body_schemas)):
            for j in range(i + 1, len(body_schemas)):
                if body_schemas[i].shape == body_schemas[j].shape:
                    correlation = jnp.corrcoef(
                        body_schemas[i].flatten(),
                        body_schemas[j].flatten()
                    )[0, 1]
                    
                    if not jnp.isnan(correlation):
                        correlations.append(correlation)
        
        if correlations:
            return float(jnp.mean(jnp.array(correlations)))
        else:
            return 0.0
    
    def _compute_motor_alignment(self, motor_intentions: List[Array]) -> float:
        """Compute alignment between motor intentions."""
        if len(motor_intentions) < 2:
            return 1.0
        
        # Compute directional alignment
        normalized_intentions = [
            intention / (jnp.linalg.norm(intention) + 1e-6)
            for intention in motor_intentions
        ]
        
        alignments = []
        for i in range(len(normalized_intentions)):
            for j in range(i + 1, len(normalized_intentions)):
                alignment = jnp.dot(normalized_intentions[i], normalized_intentions[j])
                alignments.append(alignment)
        
        if alignments:
            return float(jnp.mean(jnp.array(alignments)))
        else:
            return 0.0
    
    def _compute_proprioceptive_stability(self, proprioceptive_fields: List[Array]) -> float:
        """Compute stability across proprioceptive fields."""
        if len(proprioceptive_fields) < 2:
            return 1.0
        
        # Compute variance across fields
        field_array = jnp.array(proprioceptive_fields)
        field_variance = jnp.mean(jnp.var(field_array, axis=0))
        
        # Lower variance = higher stability
        return float(jnp.exp(-field_variance))
    
    def _generate_integration_reports(
        self,
        aggregates: List[EmbodiedExperienceAggregate],
        integrated_schema: Array,
        integration_metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate detailed integration reports."""
        reports = []
        
        for aggregate in aggregates:
            if aggregate.current_experience:
                report = {
                    'aggregate_id': aggregate.aggregate_id,
                    'embodiment_coherence': aggregate.current_experience.embodiment_coherence,
                    'integration_contribution': self._estimate_integration_contribution(
                        aggregate, integrated_schema
                    ),
                    'action_readiness': aggregate.assess_action_readiness(),
                    'tool_incorporations': len(aggregate.tool_incorporation_registry),
                    'schema_modifications': aggregate.current_experience._schema_modifications,
                    'integration_quality_assessment': self._assess_individual_integration_quality(
                        aggregate, integration_metrics
                    )
                }
                
                reports.append(report)
        
        return reports
    
    def _estimate_extension_strength(
        self,
        body_boundary: BodyBoundary,
        tool_representation: Array
    ) -> float:
        """Estimate strength of tool extension."""
        # Simple estimation based on boundary confidence and tool compatibility
        tool_norm = jnp.linalg.norm(tool_representation)
        boundary_norm = jnp.linalg.norm(body_boundary.boundary_contour)
        
        if boundary_norm < 1e-6:
            return 0.0
        
        compatibility = jnp.dot(
            tool_representation / (tool_norm + 1e-6),
            body_boundary.boundary_contour / boundary_norm
        )
        
        extension_strength = body_boundary.boundary_confidence * jnp.clip(compatibility, 0.0, 1.0)
        
        return float(extension_strength)
    
    def _assess_motor_intention_coherence(
        self,
        synthesized_intention: Array,
        individual_intentions: List[Array],
        readiness_scores: List[float]
    ) -> float:
        """Assess coherence of synthesized motor intention."""
        # Magnitude coherence
        synth_magnitude = jnp.linalg.norm(synthesized_intention)
        individual_magnitudes = [jnp.linalg.norm(intention) for intention in individual_intentions]
        avg_magnitude = jnp.mean(jnp.array(individual_magnitudes))
        
        magnitude_coherence = 1.0 - abs(synth_magnitude - avg_magnitude) / (avg_magnitude + 1e-6)
        
        # Direction coherence
        if synth_magnitude > 1e-6:
            synth_direction = synthesized_intention / synth_magnitude
            direction_alignments = []
            
            for intention in individual_intentions:
                intention_magnitude = jnp.linalg.norm(intention)
                if intention_magnitude > 1e-6:
                    intention_direction = intention / intention_magnitude
                    alignment = jnp.dot(synth_direction, intention_direction)
                    direction_alignments.append(alignment)
            
            if direction_alignments:
                direction_coherence = jnp.mean(jnp.array(direction_alignments))
            else:
                direction_coherence = 0.0
        else:
            direction_coherence = 0.0
        
        # Readiness coherence
        readiness_coherence = jnp.mean(jnp.array(readiness_scores))
        
        # Overall coherence
        return jnp.mean(jnp.array([
            magnitude_coherence,
            direction_coherence,
            readiness_coherence
        ]))
    
    def _estimate_integration_contribution(
        self,
        aggregate: EmbodiedExperienceAggregate,
        integrated_schema: Array
    ) -> float:
        """Estimate aggregate's contribution to integrated schema."""
        if not aggregate.current_experience:
            return 0.0
        
        # Get aggregate's individual schema
        individual_schema = aggregate.current_experience.integrate_body_schema(
            proprioceptive_input=aggregate.current_experience.proprioceptive_field.proprioceptive_map,
            motor_prediction=aggregate.current_experience.motor_intention.motor_vector if aggregate.current_experience.motor_intention else jnp.zeros(aggregate.current_experience.proprioceptive_field.proprioceptive_map.shape),
            tactile_feedback=jnp.zeros_like(aggregate.current_experience.proprioceptive_field.proprioceptive_map)
        )
        
        # Compute correlation with integrated schema
        if individual_schema.shape == integrated_schema.shape:
            correlation = jnp.corrcoef(
                individual_schema.flatten(),
                integrated_schema.flatten()
            )[0, 1]
            
            if jnp.isnan(correlation):
                return 0.0
            else:
                return float(jnp.clip(correlation, 0.0, 1.0))
        else:
            return 0.0
    
    def _assess_individual_integration_quality(
        self,
        aggregate: EmbodiedExperienceAggregate,
        integration_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Assess individual aggregate's integration quality."""
        if not aggregate.current_experience:
            return {'overall_quality': 0.0}
        
        # Individual coherence relative to average
        individual_coherence = aggregate.current_experience.embodiment_coherence
        avg_coherence = integration_metrics['avg_coherence']
        coherence_relative = individual_coherence / (avg_coherence + 1e-6)
        
        # Action readiness assessment
        readiness = aggregate.assess_action_readiness()
        readiness_quality = readiness.get('overall', 0.0)
        
        # Integration participation quality
        participation_quality = min(1.0, coherence_relative)
        
        overall_quality = jnp.mean(jnp.array([
            coherence_relative,
            readiness_quality,
            participation_quality
        ]))
        
        return {
            'overall_quality': float(jnp.clip(overall_quality, 0.0, 1.0)),
            'coherence_relative': float(coherence_relative),
            'readiness_quality': readiness_quality,
            'participation_quality': float(participation_quality)
        }


# ============================================================================
# CONSCIOUSNESS INTEGRATION ORCHESTRATOR (Cross-Domain Integration)
# ============================================================================

class ConsciousnessIntegrationOrchestrator:
    """Master domain service for orchestrating consciousness integration.
    
    Coordinates integration across all consciousness domains (temporal,
    embodied, coupling, memory) following phenomenological unity principles.
    
    Business Rules:
    - Consciousness integration requires minimum coherence across domains
    - Integration strategy must respect phenomenological structure
    - Temporal domain has priority in integration ordering
    - Integration quality affects overall consciousness level
    """
    
    def __init__(
        self,
        temporal_service: TemporalSynthesisService,
        embodied_service: EmbodiedIntegrationService,
        causality_orchestrator: CircularCausalityOrchestrator,
        memory_service: ExperientialMemoryService
    ):
        self.temporal_service = temporal_service
        self.embodied_service = embodied_service
        self.causality_orchestrator = causality_orchestrator
        self.memory_service = memory_service
    
    def orchestrate_full_consciousness_integration(
        self,
        temporal_aggregates: List[TemporalConsciousnessAggregate],
        embodied_aggregates: List[EmbodiedExperienceAggregate], 
        causality_aggregates: List[CircularCausalityAggregate],
        memory_aggregates: List[ExperientialMemoryAggregate],
        integration_strategy: IntegrationStrategy = IntegrationStrategy.PHENOMENOLOGICAL_GROUNDED
    ) -> Tuple[Array, Dict[str, float], List[Dict[str, Any]]]:
        """Orchestrate complete consciousness integration across all domains.
        
        Master domain operation that integrates all consciousness components
        into unified conscious experience following phenomenological principles.
        
        Returns:
            Tuple of (integrated_consciousness_state, integration_metrics, integration_reports)
        """
        if not any([temporal_aggregates, embodied_aggregates, causality_aggregates, memory_aggregates]):
            raise ValueError("Cannot integrate consciousness without any domain aggregates")
        
        integration_components = {}
        integration_reports = []
        domain_coherences = {}
        
        # 1. Temporal consciousness integration (highest priority)
        if temporal_aggregates:
            temporal_synthesis, temporal_quality, temporal_metrics = (
                self.temporal_service.orchestrate_multi_moment_synthesis(temporal_aggregates)
            )
            
            integration_components['temporal'] = temporal_synthesis
            domain_coherences['temporal'] = temporal_metrics['avg_coherence']
            
            integration_reports.append({
                'domain': 'temporal',
                'synthesis_quality': temporal_quality.value,
                'metrics': temporal_metrics,
                'component_count': len(temporal_aggregates)
            })
        
        # 2. Embodied experience integration
        if embodied_aggregates:
            embodied_schema, embodied_metrics, embodied_report = (
                self.embodied_service.orchestrate_multimodal_embodiment_integration(embodied_aggregates)
            )
            
            integration_components['embodied'] = embodied_schema
            domain_coherences['embodied'] = embodied_metrics['avg_coherence']
            
            integration_reports.extend([{
                'domain': 'embodied',
                'integration_quality': embodied_metrics['integration_quality'],
                'metrics': embodied_metrics,
                'component_count': len(embodied_aggregates)
            }])
            integration_reports.extend(embodied_report)
        
        # 3. Circular causality coordination
        if causality_aggregates:
            causality_results = self.causality_orchestrator.orchestrate_multi_agent_circular_causality(
                causality_aggregates, {}
            )
            
            # Extract representative state from causality results
            if causality_results['agent_environment_updates']['agent_states']:
                agent_states = list(causality_results['agent_environment_updates']['agent_states'].values())
                causality_state = jnp.mean(jnp.array(agent_states), axis=0)
                integration_components['causality'] = causality_state
                domain_coherences['causality'] = causality_results['coordination_quality']
            
            integration_reports.append({
                'domain': 'causality',
                'coordination_quality': causality_results['coordination_quality'],
                'meaning_emergences': causality_results['meaning_emergence_events'],
                'component_count': len(causality_aggregates)
            })
        
        # 4. Memory system integration
        if memory_aggregates and len(memory_aggregates) > 0:
            # Create simple recall contexts for memory integration
            recall_contexts = []
            for i, aggregate in enumerate(memory_aggregates):
                if aggregate.current_memory and len(aggregate.current_memory.experiential_traces) > 0:
                    # Use first available trace content as recall cue
                    first_trace = next(iter(aggregate.current_memory.experiential_traces.values()))
                    recall_cue = first_trace.trace_content
                    
                    recall_context = RecallContext(
                        recall_cue=recall_cue,
                        contextual_factors=jnp.zeros_like(recall_cue),
                        recall_mode=RecallMode.ASSOCIATIVE,
                        affective_state=jnp.zeros_like(recall_cue),
                        temporal_proximity=0.7,
                        contextual_coherence=0.6
                    )
                    recall_contexts.append(recall_context)
            
            if recall_contexts:
                memory_traces, memory_quality, memory_metrics = (
                    self.memory_service.orchestrate_multi_modal_recall(memory_aggregates, recall_contexts)
                )
                
                # Create memory state representation
                if memory_traces:
                    memory_content = jnp.mean(jnp.array([trace.trace_content for trace in memory_traces[:5]]), axis=0)
                    integration_components['memory'] = memory_content
                    domain_coherences['memory'] = memory_quality
                
                integration_reports.append({
                    'domain': 'memory',
                    'recall_quality': memory_quality,
                    'metrics': memory_metrics,
                    'component_count': len(memory_aggregates)
                })
        
        # Integrate consciousness components based on strategy
        if integration_strategy == IntegrationStrategy.PHENOMENOLOGICAL_GROUNDED:
            integrated_state = self._phenomenological_grounded_integration(
                integration_components, domain_coherences
            )
        elif integration_strategy == IntegrationStrategy.TEMPORAL_PRIORITIZED:
            integrated_state = self._temporal_prioritized_integration(
                integration_components, domain_coherences
            )
        elif integration_strategy == IntegrationStrategy.WEIGHTED_AVERAGE:
            integrated_state = self._weighted_average_integration(
                integration_components, domain_coherences
            )
        else:
            integrated_state = self._competitive_selection_integration(
                integration_components, domain_coherences
            )
        
        # Compute overall integration metrics
        integration_metrics = self._compute_consciousness_integration_metrics(
            integration_components, domain_coherences, integrated_state
        )
        
        return integrated_state, integration_metrics, integration_reports
    
    def assess_consciousness_integration_readiness(
        self,
        temporal_aggregates: List[TemporalConsciousnessAggregate],
        embodied_aggregates: List[EmbodiedExperienceAggregate],
        causality_aggregates: List[CircularCausalityAggregate], 
        memory_aggregates: List[ExperientialMemoryAggregate],
        readiness_thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Assess readiness for consciousness integration across domains.
        
        Returns:
            Dictionary with readiness assessment for each domain
        """
        if readiness_thresholds is None:
            readiness_thresholds = {
                'temporal': 0.4,
                'embodied': 0.3,
                'causality': 0.3,
                'memory': 0.2
            }
        
        readiness_assessment = {
            'overall_readiness': 0.0,
            'domain_readiness': {},
            'integration_blockers': [],
            'readiness_recommendations': []
        }
        
        domain_readiness = {}
        
        # Assess temporal readiness
        if temporal_aggregates:
            temporal_coherences = [
                agg.assess_temporal_coherence() for agg in temporal_aggregates
                if agg.current_moment
            ]
            
            if temporal_coherences:
                temporal_readiness = jnp.mean(jnp.array(temporal_coherences))
                domain_readiness['temporal'] = float(temporal_readiness)
                
                if temporal_readiness < readiness_thresholds['temporal']:
                    readiness_assessment['integration_blockers'].append(
                        f"Temporal coherence below threshold: {temporal_readiness:.3f} < {readiness_thresholds['temporal']}"
                    )
                    readiness_assessment['readiness_recommendations'].append(
                        "Improve temporal synthesis quality and retention-protention balance"
                    )
        else:
            domain_readiness['temporal'] = 0.0
        
        # Assess embodied readiness
        if embodied_aggregates:
            embodied_coherences = [
                agg.current_experience.embodiment_coherence for agg in embodied_aggregates
                if agg.current_experience
            ]
            
            if embodied_coherences:
                embodied_readiness = jnp.mean(jnp.array(embodied_coherences))
                domain_readiness['embodied'] = float(embodied_readiness)
                
                if embodied_readiness < readiness_thresholds['embodied']:
                    readiness_assessment['integration_blockers'].append(
                        f"Embodied coherence below threshold: {embodied_readiness:.3f} < {readiness_thresholds['embodied']}"
                    )
                    readiness_assessment['readiness_recommendations'].append(
                        "Improve body schema integration and motor intention coherence"
                    )
        else:
            domain_readiness['embodied'] = 0.0
        
        # Assess causality readiness
        if causality_aggregates:
            causality_coherences = [
                agg.current_causality.coupling_coherence for agg in causality_aggregates
                if agg.current_causality
            ]
            
            if causality_coherences:
                causality_readiness = jnp.mean(jnp.array(causality_coherences))
                domain_readiness['causality'] = float(causality_readiness)
                
                if causality_readiness < readiness_thresholds['causality']:
                    readiness_assessment['integration_blockers'].append(
                        f"Causality coherence below threshold: {causality_readiness:.3f} < {readiness_thresholds['causality']}"
                    )
                    readiness_assessment['readiness_recommendations'].append(
                        "Strengthen structural coupling and autopoietic processes"
                    )
        else:
            domain_readiness['causality'] = 0.0
        
        # Assess memory readiness
        if memory_aggregates:
            memory_coherences = [
                agg.current_memory.memory_coherence for agg in memory_aggregates
                if agg.current_memory
            ]
            
            if memory_coherences:
                memory_readiness = jnp.mean(jnp.array(memory_coherences))
                domain_readiness['memory'] = float(memory_readiness)
                
                if memory_readiness < readiness_thresholds['memory']:
                    readiness_assessment['integration_blockers'].append(
                        f"Memory coherence below threshold: {memory_readiness:.3f} < {readiness_thresholds['memory']}"
                    )
                    readiness_assessment['readiness_recommendations'].append(
                        "Improve experiential trace accessibility and associative coherence"
                    )
        else:
            domain_readiness['memory'] = 0.0
        
        # Compute overall readiness
        if domain_readiness:
            # Weight domains by importance for consciousness
            domain_weights = {
                'temporal': 0.4,
                'embodied': 0.25,
                'causality': 0.2, 
                'memory': 0.15
            }
            
            weighted_readiness = 0.0
            total_weight = 0.0
            
            for domain, readiness in domain_readiness.items():
                if domain in domain_weights:
                    weighted_readiness += domain_weights[domain] * readiness
                    total_weight += domain_weights[domain]
            
            if total_weight > 0:
                readiness_assessment['overall_readiness'] = weighted_readiness / total_weight
        
        readiness_assessment['domain_readiness'] = domain_readiness
        
        return readiness_assessment
    
    def _phenomenological_grounded_integration(
        self,
        components: Dict[str, Array],
        coherences: Dict[str, float]
    ) -> Array:
        """Integrate using phenomenological grounding principles."""
        if not components:
            raise ValueError("No components to integrate")
        
        # Phenomenological priority: temporal > embodied > causality > memory
        priority_weights = {
            'temporal': 0.4,
            'embodied': 0.3,
            'causality': 0.2,
            'memory': 0.1
        }
        
        # Modulate weights by coherence
        coherence_modulated_weights = {}
        total_weight = 0.0
        
        for domain, component in components.items():
            base_weight = priority_weights.get(domain, 0.1)
            coherence = coherences.get(domain, 0.5)
            
            modulated_weight = base_weight * (0.5 + 0.5 * coherence)
            coherence_modulated_weights[domain] = modulated_weight
            total_weight += modulated_weight
        
        # Normalize weights
        normalized_weights = {
            domain: weight / total_weight
            for domain, weight in coherence_modulated_weights.items()
        }
        
        # Find common dimensionality
        target_dim = self._find_target_dimensionality(components)
        
        # Weighted integration
        integrated_components = []
        weights = []
        
        for domain, component in components.items():
            # Resize component to target dimensionality if needed
            resized_component = self._resize_component(component, target_dim)
            
            weight = normalized_weights[domain]
            integrated_components.append(weight * resized_component)
            weights.append(weight)
        
        return jnp.sum(jnp.array(integrated_components), axis=0)
    
    def _temporal_prioritized_integration(
        self,
        components: Dict[str, Array],
        coherences: Dict[str, float]
    ) -> Array:
        """Integrate prioritizing temporal consciousness."""
        if 'temporal' in components:
            # Temporal component dominates
            temporal_component = components['temporal']
            
            # Other components contribute proportionally to their coherence
            other_contribution = jnp.zeros_like(temporal_component)
            total_other_coherence = 0.0
            
            target_dim = temporal_component.shape
            
            for domain, component in components.items():
                if domain != 'temporal':
                    coherence = coherences.get(domain, 0.5)
                    resized_component = self._resize_component(component, target_dim)
                    
                    other_contribution += coherence * resized_component
                    total_other_coherence += coherence
            
            if total_other_coherence > 0:
                other_contribution /= total_other_coherence
                
                # Temporal dominance with other influence
                temporal_weight = 0.7
                other_weight = 0.3
                
                return temporal_weight * temporal_component + other_weight * other_contribution
            else:
                return temporal_component
        else:
            # Fallback to weighted average if no temporal component
            return self._weighted_average_integration(components, coherences)
    
    def _weighted_average_integration(
        self,
        components: Dict[str, Array],
        coherences: Dict[str, float]
    ) -> Array:
        """Integrate using coherence-weighted averaging."""
        if not components:
            raise ValueError("No components to integrate")
        
        target_dim = self._find_target_dimensionality(components)
        
        weighted_components = []
        total_weight = 0.0
        
        for domain, component in components.items():
            coherence = coherences.get(domain, 0.5)
            resized_component = self._resize_component(component, target_dim)
            
            weighted_components.append(coherence * resized_component)
            total_weight += coherence
        
        if total_weight > 0:
            return jnp.sum(jnp.array(weighted_components), axis=0) / total_weight
        else:
            return jnp.mean(jnp.array([
                self._resize_component(comp, target_dim) 
                for comp in components.values()
            ]), axis=0)
    
    def _competitive_selection_integration(
        self,
        components: Dict[str, Array],
        coherences: Dict[str, float]
    ) -> Array:
        """Integrate using competitive selection based on coherence."""
        if not components:
            raise ValueError("No components to integrate")
        
        # Select component with highest coherence
        best_domain = max(coherences.keys(), key=lambda d: coherences[d])
        best_component = components[best_domain]
        
        # Modulate with other components based on their relative coherence
        target_dim = best_component.shape
        
        modulation = jnp.zeros_like(best_component)
        total_modulation_weight = 0.0
        
        for domain, component in components.items():
            if domain != best_domain:
                coherence = coherences.get(domain, 0.5)
                relative_coherence = coherence / (coherences[best_domain] + 1e-6)
                
                resized_component = self._resize_component(component, target_dim)
                
                modulation += relative_coherence * resized_component
                total_modulation_weight += relative_coherence
        
        if total_modulation_weight > 0:
            modulation /= total_modulation_weight
            
            # Competitive winner with weak modulation
            return 0.8 * best_component + 0.2 * modulation
        else:
            return best_component
    
    def _find_target_dimensionality(self, components: Dict[str, Array]) -> Tuple[int, ...]:
        """Find target dimensionality for integration."""
        if not components:
            return (10,)  # Default size
        
        # Use temporal component size if available (highest priority)
        if 'temporal' in components:
            return components['temporal'].shape
        
        # Use largest component otherwise
        max_size = 0
        target_shape = None
        
        for component in components.values():
            component_size = component.size
            if component_size > max_size:
                max_size = component_size
                target_shape = component.shape
        
        return target_shape if target_shape else (10,)
    
    def _resize_component(self, component: Array, target_shape: Tuple[int, ...]) -> Array:
        """Resize component to target shape."""
        if component.shape == target_shape:
            return component
        
        target_size = int(jnp.prod(jnp.array(target_shape)))
        current_size = component.size
        
        if current_size == target_size:
            # Just reshape
            return jnp.reshape(component, target_shape)
        elif current_size < target_size:
            # Pad with zeros
            padded = jnp.concatenate([
                component.flatten(),
                jnp.zeros(target_size - current_size)
            ])
            return jnp.reshape(padded, target_shape)
        else:
            # Truncate
            truncated = component.flatten()[:target_size]
            return jnp.reshape(truncated, target_shape)
    
    def _compute_consciousness_integration_metrics(
        self,
        components: Dict[str, Array],
        coherences: Dict[str, float],
        integrated_state: Array
    ) -> Dict[str, float]:
        """Compute metrics for consciousness integration."""
        metrics = {
            'integration_completeness': 0.0,
            'cross_domain_coherence': 0.0,
            'integration_stability': 0.0,
            'consciousness_unity': 0.0,
            'phenomenological_grounding': 0.0
        }
        
        # Integration completeness (how many domains contributed)
        total_possible_domains = 4  # temporal, embodied, causality, memory
        actual_domains = len(components)
        metrics['integration_completeness'] = actual_domains / total_possible_domains
        
        # Cross-domain coherence (average of domain coherences)
        if coherences:
            metrics['cross_domain_coherence'] = float(jnp.mean(jnp.array(list(coherences.values()))))
        
        # Integration stability (consistency of component contributions)
        if len(components) > 1:
            target_dim = integrated_state.shape
            component_magnitudes = []
            
            for component in components.values():
                resized_component = self._resize_component(component, target_dim)
                magnitude = jnp.linalg.norm(resized_component)
                component_magnitudes.append(magnitude)
            
            magnitude_consistency = 1.0 - jnp.std(jnp.array(component_magnitudes)) / (jnp.mean(jnp.array(component_magnitudes)) + 1e-6)
            metrics['integration_stability'] = float(magnitude_consistency)
        else:
            metrics['integration_stability'] = 1.0
        
        # Consciousness unity (how well integrated state reflects all components)
        if components:
            target_dim = integrated_state.shape
            component_correlations = []
            
            for component in components.values():
                resized_component = self._resize_component(component, target_dim)
                
                if resized_component.shape == integrated_state.shape:
                    correlation = jnp.corrcoef(
                        resized_component.flatten(),
                        integrated_state.flatten()
                    )[0, 1]
                    
                    if not jnp.isnan(correlation):
                        component_correlations.append(correlation)
            
            if component_correlations:
                metrics['consciousness_unity'] = float(jnp.mean(jnp.array(component_correlations)))
        
        # Phenomenological grounding (weighted by domain importance)
        domain_importance = {
            'temporal': 0.4,
            'embodied': 0.3,
            'causality': 0.2,
            'memory': 0.1
        }
        
        grounding_score = 0.0
        total_importance = 0.0
        
        for domain, coherence in coherences.items():
            importance = domain_importance.get(domain, 0.1)
            grounding_score += importance * coherence
            total_importance += importance
        
        if total_importance > 0:
            metrics['phenomenological_grounding'] = grounding_score / total_importance
        
        return metrics