"""Phenomenological temporal consciousness implementation.

This module implements Husserl's theory of internal time consciousness,
featuring retention-present-protention temporal synthesis with modern
JAX/Equinox implementation for high performance computing.
"""

from __future__ import annotations

import functools
import logging
from typing import Dict, List, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import equinox as eqx

from .core import (
    ProcessorBase,
    StateValidationMixin, 
    ConfigurableMixin,
    MemoryManager,
    MetricCollector,
    create_safe_jit_function,
    optimize_for_memory,
    GLOBAL_MEMORY_MANAGER,
    GLOBAL_METRICS,
)
from .jit_utils import (
    safe_jit,
    create_jit_factory,
    create_shape_safe_factory,
)

# Configure module logger
logger = logging.getLogger(__name__)

from .types import (
    Array,
    ArrayLike,
    PRNGKey,
    TimeStep,
    TemporalMoment,
    FrameworkConfig,
    TemporalProcessor,
    TemporalSynthesisError,
    create_temporal_moment,
    validate_temporal_consistency,
)


class TemporalConsciousnessConfig(eqx.Module):
    """Configuration for phenomenological temporal consciousness.
    
    Based on Husserl's analysis of internal time consciousness:
    - Retention: Primary memory of just-past moments
    - Present impression: Current now-moment awareness
    - Protention: Primary expectation of immediate future
    """
    
    retention_depth: int
    protention_horizon: int
    primal_impression_width: float
    temporal_synthesis_rate: float
    temporal_decay_factor: float
    
    def __init__(
        self,
        retention_depth: int = 10,
        protention_horizon: int = 5,
        primal_impression_width: float = 0.1,
        temporal_synthesis_rate: float = 0.05,
        temporal_decay_factor: float = 0.95,
    ):
        self.retention_depth = retention_depth
        self.protention_horizon = protention_horizon
        self.primal_impression_width = primal_impression_width
        self.temporal_synthesis_rate = temporal_synthesis_rate
        self.temporal_decay_factor = temporal_decay_factor


class RetentionMemory(eqx.Module):
    """Primary retention memory following Husserl's analysis.
    
    Implements the 'running-off' of now-moments into retained consciousness,
    maintaining temporal depth while allowing for phenomenological access
    to the immediate past.
    """
    
    memory_buffer: Array
    decay_weights: Array
    current_position: int
    
    def __init__(self, depth: int, state_dim: int, key: PRNGKey):
        self.memory_buffer = jnp.zeros((depth, state_dim))
        # Exponentially decaying weights for older retentions
        self.decay_weights = jnp.exp(-jnp.arange(depth) * 0.1)
        self.current_position = 0
    
    def update_retention(self, new_moment: Array) -> 'RetentionMemory':
        """Update retention memory with new present moment."""
        # Shift buffer and add new moment
        new_buffer = jnp.roll(self.memory_buffer, 1, axis=0)
        new_buffer = new_buffer.at[0].set(new_moment)
        
        return eqx.tree_at(
            lambda x: (x.memory_buffer, x.current_position),
            self,
            (new_buffer, (self.current_position + 1) % self.memory_buffer.shape[0])
        )
    
    def get_retained_synthesis(self) -> Array:
        """Compute weighted synthesis of retained moments."""
        weighted_retention = self.memory_buffer * self.decay_weights[:, None]
        return jnp.sum(weighted_retention, axis=0)


class ProtentionProjection(eqx.Module):
    """Primary protention following Husserl's temporal analysis.
    
    Implements forward-looking temporal consciousness, projecting
    anticipated future moments based on current dynamics and
    phenomenological temporal structure.
    """
    
    projection_weights: Array
    temporal_gradients: Array
    expectation_confidence: Array
    
    def __init__(self, horizon: int, state_dim: int, key: PRNGKey):
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Learned weights for temporal projection
        self.projection_weights = jax.random.normal(
            key1, (horizon, state_dim, state_dim)
        ) * 0.1
        
        # Gradient information for temporal dynamics
        self.temporal_gradients = jax.random.normal(
            key2, (horizon, state_dim)
        ) * 0.05
        
        # Confidence in protentional projections
        self.expectation_confidence = jax.nn.sigmoid(
            jax.random.normal(key3, (horizon,))
        )
    
    def project_protention(
        self, 
        current_moment: Array,
        temporal_context: Array,
    ) -> Array:
        """Project protentional consciousness into near future."""
        # Multi-step temporal projection
        projections = []
        current_state = current_moment
        
        for t in range(self.projection_weights.shape[0]):
            # Apply learned temporal transformation
            projected_state = jnp.tanh(
                self.projection_weights[t] @ current_state + 
                self.temporal_gradients[t] +
                temporal_context * 0.1
            )
            
            # Weight by expectation confidence
            projected_state = projected_state * self.expectation_confidence[t]
            projections.append(projected_state)
            current_state = projected_state
            
        return jnp.stack(projections)
    
    def get_protentional_synthesis(self, projections: Array) -> Array:
        """Synthesize protentional projections into unified expectation."""
        # Weight later projections less (temporal distance decay)
        distance_weights = jnp.exp(-jnp.arange(projections.shape[0]) * 0.2)
        weighted_projections = projections * distance_weights[:, None]
        return jnp.sum(weighted_projections, axis=0)


class PhenomenologicalTemporalSynthesis(ProcessorBase, StateValidationMixin, ConfigurableMixin):
    """Core implementation of Husserlian temporal consciousness.
    
    Integrates retention-present-protention structure into unified
    temporal flow, implementing the fundamental structure of
    time-consciousness as described in Husserl's phenomenology.
    
    Refactored following Martin Fowler's principles:
    - Extract Method for complex temporal synthesis
    - Replace Temp with Query for synthesis weights
    - Introduce Parameter Object for synthesis context
    """
    
    config: TemporalConsciousnessConfig
    retention_memory: RetentionMemory
    protention_projection: ProtentionProjection
    synthesis_network: eqx.nn.MLP
    temporal_attention: eqx.nn.MultiheadAttention
    
    def __init__(
        self, 
        config: TemporalConsciousnessConfig,
        state_dim: int,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, 4)
        
        self.config = config
        
        # Initialize temporal components with proper shape handling
        self.retention_memory = RetentionMemory(
            config.retention_depth, state_dim, keys[0]
        )
        
        self.protention_projection = ProtentionProjection(
            config.protention_horizon, state_dim, keys[1]
        )
        
        # Neural networks for temporal synthesis - ensure all sizes are concrete
        synthesis_input_size = state_dim * 3  # retention + present + protention
        synthesis_width_size = state_dim * 2
        
        self.synthesis_network = eqx.nn.MLP(
            in_size=synthesis_input_size,
            out_size=state_dim,
            width_size=synthesis_width_size,
            depth=2,
            activation=jax.nn.gelu,  # JAX function, not Python function
            key=keys[2],
        )
        
        # Multi-head attention for temporal relationships
        # Use fewer heads for better stability
        self.temporal_attention = eqx.nn.MultiheadAttention(
            num_heads=2,  # Reduced for stability
            query_size=state_dim,
            key_size=state_dim,
            value_size=state_dim,
            output_size=state_dim,
            key=keys[3],
        )
        
        # Note: ProcessorBase components will be initialized when needed
    
    @optimize_for_memory
    @functools.partial(jax.jit, static_argnames=['self'])
    def temporal_synthesis_jit(
        self,
        primal_impression: Array,
        environmental_context: Optional[Array] = None,
        timestamp: Optional[float] = None,
    ) -> TemporalMoment:
        """JIT-compiled version of temporal synthesis.
        
        This method is JIT-compiled for optimal performance but requires
        that the processor instance is treated as static.
        """
        return self._temporal_synthesis_impl(
            primal_impression, environmental_context, timestamp
        )
    
    @optimize_for_memory
    def temporal_synthesis(
        self,
        primal_impression: Array,
        environmental_context: Optional[Array] = None,
        timestamp: Optional[float] = None,
    ) -> TemporalMoment:
        """Non-JIT version of temporal synthesis for compatibility.
        
        Use this version when JIT compilation is not desired or fails.
        """
        return self._temporal_synthesis_impl(
            primal_impression, environmental_context, timestamp
        )
    
    def _temporal_synthesis_impl(
        self,
        primal_impression: Array,
        environmental_context: Optional[Array] = None,
        timestamp: Optional[float] = None,
    ) -> TemporalMoment:
        """Implementation of temporal synthesis (shared by JIT and non-JIT versions).
        
        Implements Husserl's temporal synthesis:
        1. Update retention with previous present
        2. Process current primal impression
        3. Project protentional horizon
        4. Synthesize into unified temporal moment
        
        Refactored into smaller methods for clarity.
        """
        with GLOBAL_MEMORY_MANAGER.track_memory("temporal_synthesis"):
            try:
                # Step 1: Validate and prepare inputs
                self._validate_primal_impression(primal_impression)
                context = self._prepare_environmental_context(environmental_context, primal_impression)
                
                # Step 2: Update temporal components with proper state threading
                retention_result = self._update_and_get_retention(primal_impression)
                if isinstance(retention_result, tuple) and len(retention_result) == 2:
                    updated_self, retained_synthesis = retention_result
                else:
                    # Fallback for incorrect tuple unpacking
                    updated_self = self
                    retained_synthesis = self.retention_memory.get_retained_synthesis()
                
                protentional_synthesis = updated_self._compute_protentional_synthesis(primal_impression, context)
                
                # Step 3: Apply temporal attention using updated self
                attended_present = updated_self._apply_temporal_attention(retained_synthesis, primal_impression, protentional_synthesis)
                
                # Step 4: Synthesize final temporal moment using updated self
                temporal_synthesis_result = updated_self._synthesize_temporal_components(retained_synthesis, attended_present, protentional_synthesis)
                
                # Step 5: Create and return temporal moment using updated self
                return updated_self._create_temporal_moment(
                    timestamp or 0.0,
                    retained_synthesis,
                    temporal_synthesis_result,
                    protentional_synthesis,
                    retained_synthesis, 
                    attended_present, 
                    protentional_synthesis
                )
                
            except Exception as e:
                raise TemporalSynthesisError(f"Failed to synthesize temporal moment: {e}")
    
    def _validate_primal_impression(self, primal_impression: Array) -> None:
        """Extract method: Validate primal impression input."""
        self.validate_input_state(primal_impression, "primal_impression")
    
    def _prepare_environmental_context(self, environmental_context: Optional[Array], primal_impression: Array) -> Array:
        """Extract method: Prepare environmental context."""
        if environmental_context is not None:
            # Ensure environmental context matches primal impression shape
            if environmental_context.shape[0] != primal_impression.shape[0]:
                # Pad or truncate to match primal impression size
                if environmental_context.shape[0] < primal_impression.shape[0]:
                    # Pad with zeros
                    padding_size = primal_impression.shape[0] - environmental_context.shape[0]
                    environmental_context = jnp.concatenate([
                        environmental_context,
                        jnp.zeros(padding_size)
                    ])
                else:
                    # Truncate to match
                    environmental_context = environmental_context[:primal_impression.shape[0]]
            return environmental_context
        else:
            return jnp.zeros_like(primal_impression)
    
    def _update_and_get_retention(self, primal_impression: Array) -> Tuple['PhenomenologicalTemporalSynthesis', Array]:
        """Extract method: Update retention memory and get synthesis with proper state threading."""
        updated_retention_memory = self.retention_memory.update_retention(primal_impression)
        retained_synthesis = updated_retention_memory.get_retained_synthesis()
        # Return updated self through eqx.tree_at for proper immutable state threading
        updated_self = eqx.tree_at(lambda x: x.retention_memory, self, updated_retention_memory)
        return updated_self, retained_synthesis
    
    def _compute_protentional_synthesis(self, primal_impression: Array, context: Array) -> Array:
        """Extract method: Compute protentional synthesis."""
        protentional_projections = self.protention_projection.project_protention(primal_impression, context)
        return self.protention_projection.get_protentional_synthesis(protentional_projections)
    
    def _apply_temporal_attention(self, retained_synthesis: Array, primal_impression: Array, protentional_synthesis: Array) -> Array:
        """Extract method: Apply attention to temporal sequence."""
        # Stack temporal components: (3, state_dim)
        temporal_sequence = jnp.stack([retained_synthesis, primal_impression, protentional_synthesis])
        
        # MultiheadAttention expects shapes: (seq_len, feature_dim)
        # Query: present moment only - shape (1, state_dim)
        query = primal_impression[None, :]  # Add sequence dimension
        # Key/Value: full temporal sequence - shape (3, state_dim) 
        key_value = temporal_sequence
        
        try:
            attended_temporal, _ = self.temporal_attention(
                query,      # (1, state_dim)
                key_value,  # (3, state_dim)  
                key_value,  # (3, state_dim)
            )
            return attended_temporal.squeeze(0)  # Remove sequence dimension
        except Exception as e:
            # Fallback to simple weighted combination if attention fails
            logger.debug(f"Attention failed: {e}, using simple combination")
            weights = jax.nn.softmax(jnp.array([0.3, 0.5, 0.2]))  # retention, present, protention
            return (
                weights[0] * retained_synthesis + 
                weights[1] * primal_impression + 
                weights[2] * protentional_synthesis
            )
    
    def _synthesize_temporal_components(self, retained_synthesis: Array, attended_present: Array, protentional_synthesis: Array) -> Array:
        """Extract method: Synthesize temporal components through MLP."""
        synthesis_input = jnp.concatenate([retained_synthesis, attended_present, protentional_synthesis])
        return self.synthesis_network(synthesis_input)
    
    def _create_synthesis_weights(self, retained_synthesis: Array, attended_present: Array, protentional_synthesis: Array) -> Array:
        """Replace temp with query: Compute synthesis weights.
        
        Returns weights with same shape as temporal state vectors for compatibility.
        """
        # Compute scalar weights based on vector norms
        raw_weights = jnp.array([
            jnp.linalg.norm(retained_synthesis),
            jnp.linalg.norm(attended_present), 
            jnp.linalg.norm(protentional_synthesis)
        ])
        scalar_weights = jax.nn.softmax(raw_weights)
        
        # Broadcast to match state vector dimensions for TemporalMoment compatibility
        # TemporalMoment expects all arrays to have the same shape
        state_dim = retained_synthesis.shape[-1]
        return jnp.broadcast_to(scalar_weights[:, None], (3, state_dim)).flatten()[:state_dim]
    
    def _create_temporal_moment(self, timestamp: float, retained_synthesis: Array, temporal_synthesis: Array, 
                               protentional_synthesis: Array, *weight_components: Array) -> TemporalMoment:
        """Extract method: Create validated temporal moment."""
        synthesis_weights = self._create_synthesis_weights(*weight_components)
        
        # Ensure all arrays have the same shape for TemporalMoment validation
        state_dim = temporal_synthesis.shape[-1]
        
        # Make sure all temporal components have consistent shapes
        retention_shaped = jnp.reshape(retained_synthesis, (state_dim,))
        present_shaped = jnp.reshape(temporal_synthesis, (state_dim,))
        protention_shaped = jnp.reshape(protentional_synthesis, (state_dim,))
        weights_shaped = jnp.reshape(synthesis_weights, (state_dim,))
        
        return create_temporal_moment(
            timestamp=timestamp,
            retention=retention_shaped,
            present_moment=present_shaped,
            protention=protention_shaped,
            synthesis_weights=weights_shaped,
        )
    
    def process(self, *args, **kwargs) -> Any:
        """Implementation of ProcessorBase abstract method."""
        return self.temporal_synthesis(*args, **kwargs)
    
    def update_temporal_flow(
        self, 
        current_moment: TemporalMoment,
        new_impression: Array,
    ) -> TemporalMoment:
        """Update temporal flow with new impression."""
        new_timestamp = self._compute_next_timestamp(current_moment.timestamp)
        return self.temporal_synthesis(
            primal_impression=new_impression,
            timestamp=new_timestamp,
        )
    
    def _compute_next_timestamp(self, current_timestamp: float) -> float:
        """Extract method: Compute next timestamp."""
        return current_timestamp + self.config.temporal_synthesis_rate
    
    def get_temporal_horizon_depth(self) -> float:
        """Compute current temporal horizon depth."""
        retention_strength = jnp.mean(jnp.abs(
            self.retention_memory.get_retained_synthesis()
        ))
        protention_strength = jnp.mean(jnp.abs(
            self.protention_projection.expectation_confidence
        ))
        return float(retention_strength + protention_strength)
    
    def reset_temporal_state(self, key: PRNGKey) -> 'PhenomenologicalTemporalSynthesis':
        """Reset temporal state while preserving learned parameters."""
        keys = jax.random.split(key, 2)
        
        new_retention = RetentionMemory(
            self.config.retention_depth,
            self.retention_memory.memory_buffer.shape[1],
            keys[0],
        )
        
        new_protention = eqx.tree_at(
            lambda x: x.temporal_gradients,
            self.protention_projection,
            jax.random.normal(keys[1], self.protention_projection.temporal_gradients.shape) * 0.05,
        )
        
        return eqx.tree_at(
            lambda x: (x.retention_memory, x.protention_projection),
            self,
            (new_retention, new_protention),
        )


# Base factory function (non-JIT)
def _create_temporal_processor_impl(
    config: TemporalConsciousnessConfig,
    state_dim: int,
    key: PRNGKey,
) -> PhenomenologicalTemporalSynthesis:
    """Base implementation for creating temporal processor.
    
    This is the core implementation shared by both JIT and non-JIT versions.
    """
    return PhenomenologicalTemporalSynthesis(config, state_dim, key)


# Create factory functions using JIT utilities
(
    create_temporal_processor,      # JIT version
    create_temporal_processor_no_jit,  # Non-JIT version
    create_temporal_processor_safe,    # Safe version with fallback
) = create_jit_factory(
    _create_temporal_processor_impl,
    static_argnames=['config', 'state_dim'],
    enable_jit=True,
    fallback=True,
)

# Add proper docstrings to the generated functions
create_temporal_processor.__doc__ = """Create JIT-compiled temporal processor.

Args:
    config: Temporal consciousness configuration (static)
    state_dim: Dimension of state vectors (static) 
    key: Random key for initialization

Returns:
    Initialized temporal processor
    
Note:
    Both config and state_dim are marked as static arguments to ensure
    JAX can compile this function properly. If JIT compilation fails,
    use create_temporal_processor_safe() instead.
"""

create_temporal_processor_no_jit.__doc__ = """Create temporal processor without JIT compilation.

Use this version if JIT compilation fails due to complex initialization
or when debugging.

Args:
    config: Temporal consciousness configuration
    state_dim: Dimension of state vectors
    key: Random key for initialization
    
Returns:
    Initialized temporal processor
"""

create_temporal_processor_safe.__doc__ = """Create temporal processor with automatic JIT/fallback selection.

This function attempts JIT compilation first, then falls back to non-JIT
if compilation fails.

Args:
    config: Temporal consciousness configuration
    state_dim: Dimension of state vectors
    key: Random key for initialization
    use_jit: Whether to attempt JIT compilation (default: True)
    
Returns:
    Initialized temporal processor
"""


# Utility functions for temporal analysis
def analyze_temporal_coherence(
    temporal_sequence: List[TemporalMoment],
) -> Dict[str, float]:
    """Analyze temporal coherence across sequence of moments."""
    if len(temporal_sequence) < 2:
        return {"coherence": 0.0, "stability": 0.0, "flow_continuity": 0.0}
    
    # Compute temporal coherence metrics
    present_moments = jnp.stack([moment.present_moment for moment in temporal_sequence])
    
    # Coherence: correlation between adjacent moments
    correlations = []
    for i in range(len(present_moments) - 1):
        corr = jnp.corrcoef(present_moments[i], present_moments[i+1])[0, 1]
        if jnp.isfinite(corr):
            correlations.append(corr)
    
    coherence = float(jnp.mean(jnp.array(correlations))) if correlations else 0.0
    
    # Stability: variance in temporal synthesis weights
    synthesis_weights = jnp.stack([moment.synthesis_weights for moment in temporal_sequence])
    stability = 1.0 - float(jnp.mean(jnp.var(synthesis_weights, axis=0)))
    
    # Flow continuity: smoothness of temporal progression
    timestamp_diffs = jnp.diff(jnp.array([moment.timestamp for moment in temporal_sequence]))
    flow_continuity = 1.0 / (1.0 + float(jnp.std(timestamp_diffs)))
    
    return {
        "coherence": max(0.0, min(1.0, coherence)),
        "stability": max(0.0, min(1.0, stability)),
        "flow_continuity": max(0.0, min(1.0, flow_continuity)),
    }


def create_temporal_test_sequence(
    processor: PhenomenologicalTemporalSynthesis,
    input_sequence: List[Array],
    key: PRNGKey,
) -> List[TemporalMoment]:
    """Create sequence of temporal moments for testing."""
    moments = []
    current_timestamp = 0.0
    
    for i, input_data in enumerate(input_sequence):
        moment = processor.temporal_synthesis(
            primal_impression=input_data,
            timestamp=current_timestamp,
        )
        moments.append(moment)
        current_timestamp += processor.config.temporal_synthesis_rate
    
    return moments


# Export public API
__all__ = [
    'TemporalConsciousnessConfig',
    'RetentionMemory', 
    'ProtentionProjection',
    'PhenomenologicalTemporalSynthesis',
    'create_temporal_processor',
    'create_temporal_processor_no_jit',
    'create_temporal_processor_safe',
    'analyze_temporal_coherence',
    'create_temporal_test_sequence',
]