"""Integrated consciousness system combining temporal, embodied, and experiential memory.

This module implements the full integration of temporal consciousness,
body schema, and experiential memory into a unified enactive consciousness system.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import equinox as eqx

from .types import (
    Array,
    ArrayLike,
    PRNGKey,
    TimeStep,
    TemporalMoment,
    BodyState,
    FrameworkConfig,
    ConsciousnessLevel,
    PerformanceMetrics,
    ConsciousnessIntegrator,
    EnactiveConsciousnessError,
)

from .temporal import (
    PhenomenologicalTemporalSynthesis,
    TemporalConsciousnessConfig,
)

from .embodiment import (
    BodySchemaIntegration,
    BodySchemaConfig,
)

from .experiential_memory import (
    IntegratedExperientialMemory,
    ExperientialTrace,
)


class ConsciousnessState(eqx.Module):
    """Unified consciousness state representation.
    
    Integrates temporal, embodied, and experiential aspects
    into a coherent consciousness state following enactivist principles.
    """
    
    temporal_moment: TemporalMoment
    body_state: BodyState
    experiential_context: Array
    consciousness_level: float
    integration_coherence: float
    circular_causality_strength: float
    timestamp: TimeStep
    
    def __init__(
        self,
        temporal_moment: TemporalMoment,
        body_state: BodyState,
        experiential_context: Array,
        consciousness_level: float,
        integration_coherence: float,
        circular_causality_strength: float,
        timestamp: TimeStep,
    ):
        self.temporal_moment = temporal_moment
        self.body_state = body_state
        self.experiential_context = experiential_context
        self.consciousness_level = consciousness_level
        self.integration_coherence = integration_coherence
        self.circular_causality_strength = circular_causality_strength
        self.timestamp = timestamp


class EnactiveConsciousnessSystem(eqx.Module):
    """Fully integrated enactive consciousness system.
    
    Implements the complete enactive consciousness framework including:
    - Husserlian temporal synthesis
    - Merleau-Ponty body schema
    - Varela-Maturana circular causality
    - Experiential memory and recall
    """
    
    # Core processing systems
    temporal_processor: PhenomenologicalTemporalSynthesis
    body_processor: BodySchemaIntegration
    experiential_memory: IntegratedExperientialMemory
    
    # Integration networks
    consciousness_integrator: eqx.nn.MLP
    level_assessor: eqx.nn.MLP
    coherence_evaluator: eqx.nn.MultiheadAttention
    
    # Configuration
    config: FrameworkConfig
    consciousness_threshold: float
    
    def __init__(
        self,
        config: FrameworkConfig,
        temporal_config: TemporalConsciousnessConfig,
        body_config: BodySchemaConfig,
        state_dim: int,
        environment_dim: int,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, 6)
        
        self.config = config
        self.consciousness_threshold = config.consciousness_threshold
        
        # Initialize core processors
        self.temporal_processor = PhenomenologicalTemporalSynthesis(
            temporal_config, state_dim, keys[0]
        )
        
        self.body_processor = BodySchemaIntegration(body_config, keys[1])
        
        context_dim = state_dim // 2
        self.experiential_memory = IntegratedExperientialMemory(
            state_dim, environment_dim, context_dim, key=keys[2]
        )
        
        # Integration networks
        integration_input_dim = (
            state_dim +  # temporal synthesis
            config.proprioceptive_dim +  # body schema
            state_dim  # experiential context
        )
        
        hidden_dim = max(128, integration_input_dim // 2)
        
        self.consciousness_integrator = eqx.nn.MLP(
            in_size=integration_input_dim,
            out_size=state_dim,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.gelu,
            key=keys[3],
        )
        
        self.level_assessor = eqx.nn.MLP(
            in_size=state_dim,
            out_size=1,
            width_size=hidden_dim // 2,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[4],
        )
        
        self.coherence_evaluator = eqx.nn.MultiheadAttention(
            num_heads=8,
            query_size=state_dim,
            key_size=state_dim,
            value_size=state_dim,
            output_size=state_dim,
            key=keys[5],
        )
    
    def integrate_conscious_moment(
        self,
        sensory_input: Array,
        proprioceptive_input: Array,
        motor_prediction: Array,
        environmental_state: Array,
        contextual_cues: Array,
        timestamp: Optional[TimeStep] = None,
    ) -> ConsciousnessState:
        """Integrate all systems into unified conscious moment."""
        
        try:
            current_time = timestamp if timestamp is not None else 0.0
            
            # 1. Temporal consciousness processing
            temporal_moment = self.temporal_processor.temporal_synthesis(
                primal_impression=sensory_input,
                environmental_context=environmental_state,
                timestamp=current_time,
            )
            
            # 2. Body schema integration
            tactile_feedback = sensory_input[:min(len(sensory_input), 24)]  # Extract tactile portion
            body_state = self.body_processor.integrate_body_schema(
                proprioceptive_input=proprioceptive_input,
                motor_prediction=motor_prediction,
                tactile_feedback=tactile_feedback,
            )
            
            # 3. Experiential memory processing
            current_experience = temporal_moment.present_moment
            experiential_context, exp_metadata = self.experiential_memory.process_experiential_moment(
                current_experience=current_experience,
                environmental_input=environmental_state,
                contextual_cues=contextual_cues,
                significance_weight=0.6,  # Default significance
            )
            
            # 4. Consciousness integration
            integration_input = jnp.concatenate([
                temporal_moment.present_moment,
                body_state.proprioception,
                experiential_context,
            ])
            
            integrated_consciousness = self.consciousness_integrator(integration_input)
            
            # 5. Consciousness level assessment
            consciousness_level = jax.nn.sigmoid(
                self.level_assessor(integrated_consciousness)
            ).squeeze()
            
            # 6. Coherence evaluation through self-attention
            coherence_input = integrated_consciousness[None, :]  # (1, feature_dim)
            
            # Handle attention mechanism call - it might only return attended values
            coherence_result = self.coherence_evaluator(
                coherence_input, coherence_input, coherence_input
            )
            
            # Extract coherent state and attention weights
            if isinstance(coherence_result, tuple) and len(coherence_result) == 2:
                coherent_state, attention_weights = coherence_result
            else:
                coherent_state = coherence_result
                attention_weights = jnp.ones((1, 1))  # Default attention
            
            # Compute integration coherence
            integration_coherence = float(jnp.mean(attention_weights))
            
            # Extract circular causality strength from experiential metadata
            circular_causality_strength = exp_metadata['circular_causality']['circular_coherence']
            
            # Create unified consciousness state
            consciousness_state = ConsciousnessState(
                temporal_moment=temporal_moment,
                body_state=body_state,
                experiential_context=experiential_context,
                consciousness_level=float(consciousness_level),
                integration_coherence=integration_coherence,
                circular_causality_strength=circular_causality_strength,
                timestamp=current_time,
            )
            
            return consciousness_state
            
        except Exception as e:
            raise EnactiveConsciousnessError(
                f"Failed to integrate conscious moment: {e}"
            )
    
    def assess_consciousness_level(
        self,
        consciousness_state: ConsciousnessState,
    ) -> ConsciousnessLevel:
        """Assess the level of consciousness based on integrated state."""
        
        level_score = consciousness_state.consciousness_level
        coherence = consciousness_state.integration_coherence
        causality = consciousness_state.circular_causality_strength
        
        # Weighted assessment considering multiple factors
        overall_score = (
            0.4 * level_score +
            0.3 * coherence +
            0.3 * causality
        )
        
        # Map to consciousness levels
        if overall_score < 0.3:
            return ConsciousnessLevel.MINIMAL
        elif overall_score < 0.6:
            return ConsciousnessLevel.BASIC
        elif overall_score < 0.8:
            return ConsciousnessLevel.REFLECTIVE
        else:
            return ConsciousnessLevel.META_COGNITIVE
    
    def compute_performance_metrics(
        self,
        consciousness_sequence: List[ConsciousnessState],
        processing_time_ms: float,
        memory_usage_mb: float,
    ) -> PerformanceMetrics:
        """Compute comprehensive performance metrics."""
        
        if not consciousness_sequence:
            raise ValueError("Empty consciousness sequence")
        
        # Extract metrics from sequence
        temporal_coherences = [
            jnp.corrcoef(
                state.temporal_moment.present_moment,
                state.temporal_moment.retention
            )[0, 1]
            for state in consciousness_sequence
            if jnp.var(state.temporal_moment.retention) > 1e-6
        ]
        
        embodiment_stabilities = [
            state.body_state.schema_confidence
            for state in consciousness_sequence
        ]
        
        coupling_effectivenesses = [
            state.circular_causality_strength
            for state in consciousness_sequence
        ]
        
        consciousness_scores = [
            state.consciousness_level
            for state in consciousness_sequence
        ]
        
        integration_coherences = [
            state.integration_coherence
            for state in consciousness_sequence
        ]
        
        # Compute averages (handle NaN values)
        temporal_coherence = float(jnp.nanmean(jnp.array(temporal_coherences))) if temporal_coherences else 0.0
        embodiment_stability = float(jnp.mean(jnp.array(embodiment_stabilities)))
        coupling_effectiveness = float(jnp.mean(jnp.array(coupling_effectivenesses)))
        
        # Simplified affordance detection (not yet implemented)
        affordance_detection_accuracy = 0.75  # Placeholder
        
        # Meaning construction quality based on experiential coherence
        meaning_construction_quality = float(jnp.mean(jnp.array(integration_coherences)))
        
        overall_consciousness_score = float(jnp.mean(jnp.array(consciousness_scores)))
        
        return PerformanceMetrics(
            temporal_coherence=max(0.0, min(1.0, temporal_coherence)),
            embodiment_stability=max(0.0, min(1.0, embodiment_stability)),
            coupling_effectiveness=max(0.0, min(1.0, coupling_effectiveness)),
            affordance_detection_accuracy=affordance_detection_accuracy,
            meaning_construction_quality=max(0.0, min(1.0, meaning_construction_quality)),
            overall_consciousness_score=max(0.0, min(1.0, overall_consciousness_score)),
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_usage_mb,
        )
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state information."""
        
        temporal_depth = self.temporal_processor.get_temporal_horizon_depth()
        memory_state = self.experiential_memory.get_memory_state()
        
        return {
            'temporal_horizon_depth': float(temporal_depth),
            'experiential_memory': memory_state,
            'consciousness_threshold': self.consciousness_threshold,
            'system_components': {
                'temporal_processor': True,
                'body_processor': True,
                'experiential_memory': True,
            },
            'configuration': {
                'retention_depth': self.config.retention_depth,
                'protention_horizon': self.config.protention_horizon,
                'proprioceptive_dim': self.config.proprioceptive_dim,
                'motor_dim': self.config.motor_dim,
            }
        }
    
    def reset_system_state(self, key: PRNGKey) -> 'EnactiveConsciousnessSystem':
        """Reset system state while preserving learned parameters."""
        keys = jax.random.split(key, 2)
        
        # Reset temporal processor
        new_temporal_processor = self.temporal_processor.reset_temporal_state(keys[0])
        
        # Create new experiential memory (preserving structure)
        new_experiential_memory = IntegratedExperientialMemory(
            self.experiential_memory.circular_engine.self_reference_network.in_size,
            self.experiential_memory.circular_engine.environment_coupling_network.in_size - 
            self.experiential_memory.circular_engine.self_reference_network.in_size,
            self.experiential_memory.recall_system.context_encoder.in_size,
            key=keys[1],
        )
        
        return eqx.tree_at(
            lambda x: (x.temporal_processor, x.experiential_memory),
            self,
            (new_temporal_processor, new_experiential_memory),
        )


# Factory function
def create_enactive_consciousness_system(
    config: FrameworkConfig,
    temporal_config: TemporalConsciousnessConfig,
    body_config: BodySchemaConfig,
    state_dim: int,
    environment_dim: int,
    key: PRNGKey,
) -> EnactiveConsciousnessSystem:
    """Factory function for creating consciousness system."""
    return EnactiveConsciousnessSystem(
        config, temporal_config, body_config,
        state_dim, environment_dim, key
    )


# Utility functions
def run_consciousness_sequence(
    system: EnactiveConsciousnessSystem,
    input_sequence: List[Dict[str, Array]],
    initial_timestamp: TimeStep = 0.0,
) -> List[ConsciousnessState]:
    """Run a sequence of consciousness processing steps."""
    
    consciousness_states = []
    current_timestamp = initial_timestamp
    
    for i, inputs in enumerate(input_sequence):
        consciousness_state = system.integrate_conscious_moment(
            sensory_input=inputs['sensory_input'],
            proprioceptive_input=inputs['proprioceptive_input'],
            motor_prediction=inputs['motor_prediction'],
            environmental_state=inputs['environmental_state'],
            contextual_cues=inputs['contextual_cues'],
            timestamp=current_timestamp,
        )
        
        consciousness_states.append(consciousness_state)
        current_timestamp += 0.1  # Default time step
    
    return consciousness_states


# Export public API
__all__ = [
    'ConsciousnessState',
    'EnactiveConsciousnessSystem',
    'create_enactive_consciousness_system',
    'run_consciousness_sequence',
]