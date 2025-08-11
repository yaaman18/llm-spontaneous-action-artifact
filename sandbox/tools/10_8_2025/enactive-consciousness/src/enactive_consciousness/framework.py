"""Main framework interface for enactive consciousness.

This module provides the unified interface following Martin Fowler's
Facade pattern and implements comprehensive error handling, monitoring,
and performance optimization.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx

from .core import (
    ProcessorBase,
    MemoryManager,
    MetricCollector,
    JITCompiler,
    ErrorHandler,
    create_safe_jit_function,
    optimize_for_memory,
    GLOBAL_MEMORY_MANAGER,
    GLOBAL_METRICS,
    ERROR_HANDLER,
)
from .config import UnifiedConfig, get_config
from .types import (
    Array,
    PRNGKey,
    TemporalMoment,
    BodyState,
    CouplingState,
    AffordanceVector,
    MeaningStructure,
    PerformanceMetrics,
    ConsciousnessLevel,
)
from .temporal import (
    PhenomenologicalTemporalSynthesis,
    TemporalConsciousnessConfig,
    create_temporal_processor,
)
from .embodiment import (
    BodySchemaIntegration,
    BodySchemaConfig,
    create_body_schema_processor,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsciousnessState:
    """Unified consciousness state combining all components."""
    
    temporal_moment: TemporalMoment
    body_state: BodyState
    coupling_state: Optional[CouplingState] = None
    affordances: Optional[AffordanceVector] = None
    meaning: Optional[MeaningStructure] = None
    consciousness_level: ConsciousnessLevel = ConsciousnessLevel.BASIC
    integration_confidence: float = 0.5
    timestamp: float = 0.0
    
    def __post_init__(self):
        """Validate consciousness state consistency."""
        if not (0.0 <= self.integration_confidence <= 1.0):
            raise ValueError("integration_confidence must be in [0, 1]")


@dataclass(frozen=True)
class ProcessingContext:
    """Context for consciousness processing operations."""
    
    prng_key: PRNGKey
    environmental_context: Optional[Array] = None
    processing_mode: str = "normal"  # normal, fast, accurate
    debug_mode: bool = False
    
    def split_key(self, num: int = 2) -> List[PRNGKey]:
        """Split PRNG key for multiple operations."""
        return list(jax.random.split(self.prng_key, num))


class ConsciousnessFramework(eqx.Module):
    """Main framework for enactive consciousness processing.
    
    This class implements the Facade pattern to provide a unified interface
    to all consciousness processing components, with comprehensive error
    handling, performance monitoring, and memory management.
    """
    
    config: UnifiedConfig
    temporal_processor: PhenomenologicalTemporalSynthesis
    embodiment_processor: BodySchemaIntegration
    jit_compiler: JITCompiler
    
    def __init__(
        self,
        config: Optional[UnifiedConfig] = None,
        key: Optional[PRNGKey] = None,
    ):
        """Initialize the consciousness framework.
        
        Args:
            config: Framework configuration. If None, uses global config.
            key: PRNG key for initialization. If None, creates new key.
        """
        if config is None:
            config = get_config()
        if key is None:
            key = jax.random.PRNGKey(42)
        
        self.config = config
        
        # Split keys for component initialization
        keys = jax.random.split(key, 4)
        
        # Initialize components
        temporal_config = TemporalConsciousnessConfig(
            retention_depth=config.temporal.retention_depth,
            protention_horizon=config.temporal.protention_horizon,
            primal_impression_width=config.temporal.primal_impression_width,
            temporal_synthesis_rate=config.temporal.temporal_synthesis_rate,
        )
        
        self.temporal_processor = create_temporal_processor(
            temporal_config,
            config.temporal.state_dim,
            keys[0],
        )
        
        embodiment_config = BodySchemaConfig(
            proprioceptive_dim=config.embodiment.proprioceptive_dim,
            motor_dim=config.embodiment.motor_dim,
            body_map_resolution=config.embodiment.body_map_resolution,
            boundary_sensitivity=config.embodiment.boundary_sensitivity,
            schema_adaptation_rate=config.embodiment.schema_adaptation_rate,
            motor_intention_strength=config.embodiment.motor_intention_strength,
        )
        
        self.embodiment_processor = create_body_schema_processor(
            embodiment_config,
            keys[1],
        )
        
        # Initialize performance optimization
        from .core import PerformanceConfig
        perf_config = PerformanceConfig(
            enable_jit_compilation=config.system.enable_jit,
            enable_memory_optimization=config.system.enable_memory_optimization,
            max_memory_usage_mb=config.system.max_memory_mb,
        )
        self.jit_compiler = JITCompiler(perf_config)
        
        logger.info("ConsciousnessFramework initialized successfully")
    
    @optimize_for_memory
    def process_consciousness_moment(
        self,
        sensory_input: Array,
        motor_prediction: Array,
        tactile_feedback: Array,
        context: ProcessingContext,
    ) -> ConsciousnessState:
        """Process a single consciousness moment.
        
        This is the main processing method that integrates all components
        to produce a unified consciousness state.
        
        Args:
            sensory_input: Current sensory input array
            motor_prediction: Predicted motor actions
            tactile_feedback: Tactile sensory feedback
            context: Processing context with PRNG keys and options
            
        Returns:
            Unified consciousness state
        """
        with GLOBAL_MEMORY_MANAGER.track_memory("consciousness_processing"):
            start_time = time.time()
            
            try:
                # Validate inputs
                self._validate_processing_inputs(
                    sensory_input, motor_prediction, tactile_feedback
                )
                
                # Split keys for component processing
                keys = context.split_key(4)
                
                # Process temporal consciousness
                temporal_moment = self._process_temporal_moment(
                    sensory_input, context.environmental_context, keys[0]
                )
                
                # Process embodied consciousness
                body_state = self._process_body_state(
                    sensory_input, motor_prediction, tactile_feedback, keys[1]
                )
                
                # Integrate consciousness components
                consciousness_state = self._integrate_consciousness_components(
                    temporal_moment, body_state, context
                )
                
                # Record performance metrics
                processing_time = time.time() - start_time
                self._record_performance_metrics(
                    consciousness_state, processing_time
                )
                
                return consciousness_state
                
            except Exception as e:
                return ERROR_HANDLER.handle_processing_error(
                    e, "consciousness_moment_processing",
                    self._create_fallback_state(sensory_input.shape[0])
                )
    
    def _validate_processing_inputs(
        self,
        sensory_input: Array,
        motor_prediction: Array, 
        tactile_feedback: Array,
    ) -> None:
        """Validate all processing inputs."""
        inputs = {
            'sensory_input': sensory_input,
            'motor_prediction': motor_prediction,
            'tactile_feedback': tactile_feedback,
        }
        
        for name, array in inputs.items():
            if not jnp.all(jnp.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")
            if array.size == 0:
                raise ValueError(f"{name} is empty")
    
    def _process_temporal_moment(
        self,
        sensory_input: Array,
        environmental_context: Optional[Array],
        key: PRNGKey,
    ) -> TemporalMoment:
        """Process temporal consciousness component."""
        # Use JIT compilation if enabled
        if self.config.system.enable_jit:
            temporal_fn = self.jit_compiler.compile_if_enabled(
                self.temporal_processor.temporal_synthesis
            )
        else:
            temporal_fn = self.temporal_processor.temporal_synthesis
        
        return temporal_fn(
            primal_impression=sensory_input,
            environmental_context=environmental_context,
            timestamp=time.time(),
        )
    
    def _process_body_state(
        self,
        sensory_input: Array,
        motor_prediction: Array,
        tactile_feedback: Array,
        key: PRNGKey,
    ) -> BodyState:
        """Process embodied consciousness component."""
        # Use JIT compilation if enabled
        if self.config.system.enable_jit:
            embodiment_fn = self.jit_compiler.compile_if_enabled(
                self.embodiment_processor.integrate_body_schema
            )
        else:
            embodiment_fn = self.embodiment_processor.integrate_body_schema
        
        return embodiment_fn(
            proprioceptive_input=sensory_input,
            motor_prediction=motor_prediction,
            tactile_feedback=tactile_feedback,
        )
    
    def _integrate_consciousness_components(
        self,
        temporal_moment: TemporalMoment,
        body_state: BodyState,
        context: ProcessingContext,
    ) -> ConsciousnessState:
        """Integrate all consciousness components into unified state."""
        # Compute integration confidence based on component qualities
        integration_confidence = self._compute_integration_confidence(
            temporal_moment, body_state
        )
        
        # Determine consciousness level based on integration quality
        consciousness_level = self._assess_consciousness_level(
            integration_confidence, body_state.schema_confidence
        )
        
        return ConsciousnessState(
            temporal_moment=temporal_moment,
            body_state=body_state,
            consciousness_level=consciousness_level,
            integration_confidence=integration_confidence,
            timestamp=time.time(),
        )
    
    def _compute_integration_confidence(
        self,
        temporal_moment: TemporalMoment,
        body_state: BodyState,
    ) -> float:
        """Compute confidence in consciousness integration."""
        # Temporal coherence measure
        temporal_coherence = float(jnp.mean(temporal_moment.synthesis_weights))
        
        # Body schema confidence
        body_confidence = body_state.schema_confidence
        
        # Combined confidence with weighting
        return float(jnp.clip(
            0.6 * temporal_coherence + 0.4 * body_confidence,
            0.0, 1.0
        ))
    
    def _assess_consciousness_level(
        self,
        integration_confidence: float,
        schema_confidence: float,
    ) -> ConsciousnessLevel:
        """Assess current consciousness level."""
        average_confidence = (integration_confidence + schema_confidence) / 2.0
        
        if average_confidence < 0.3:
            return ConsciousnessLevel.MINIMAL
        elif average_confidence < 0.6:
            return ConsciousnessLevel.BASIC
        elif average_confidence < 0.8:
            return ConsciousnessLevel.REFLECTIVE
        else:
            return ConsciousnessLevel.META_COGNITIVE
    
    def _record_performance_metrics(
        self,
        consciousness_state: ConsciousnessState,
        processing_time: float,
    ) -> None:
        """Record performance metrics for monitoring."""
        GLOBAL_METRICS.record_metric(
            "consciousness_processing_time",
            processing_time * 1000  # Convert to ms
        )
        
        GLOBAL_METRICS.record_metric(
            "integration_confidence",
            consciousness_state.integration_confidence
        )
        
        GLOBAL_METRICS.record_metric(
            "body_schema_confidence",
            consciousness_state.body_state.schema_confidence
        )
    
    def _create_fallback_state(self, input_dim: int) -> ConsciousnessState:
        """Create fallback consciousness state for error recovery."""
        from .types import create_temporal_moment
        
        # Create minimal temporal moment
        temporal_moment = create_temporal_moment(
            timestamp=time.time(),
            retention=jnp.zeros(input_dim),
            present_moment=jnp.zeros(input_dim),
            protention=jnp.zeros(input_dim),
            synthesis_weights=jnp.ones(3) / 3.0,
        )
        
        # Create minimal body state
        body_state = BodyState(
            proprioception=jnp.zeros(input_dim),
            motor_intention=jnp.zeros(self.config.embodiment.motor_dim),
            boundary_signal=jnp.zeros(1),
            schema_confidence=0.1,
        )
        
        return ConsciousnessState(
            temporal_moment=temporal_moment,
            body_state=body_state,
            consciousness_level=ConsciousnessLevel.MINIMAL,
            integration_confidence=0.1,
            timestamp=time.time(),
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        temporal_stats = GLOBAL_METRICS.get_metric_statistics("consciousness_processing_time")
        confidence_stats = GLOBAL_METRICS.get_metric_statistics("integration_confidence")
        memory_stats = GLOBAL_MEMORY_MANAGER.get_memory_stats()
        
        return {
            'processing_time_ms': temporal_stats,
            'integration_confidence': confidence_stats,
            'memory_usage': memory_stats,
            'jit_compilation_enabled': self.config.system.enable_jit,
            'memory_optimization_enabled': self.config.system.enable_memory_optimization,
        }
    
    @contextmanager
    def debug_mode(self):
        """Context manager for debug mode processing."""
        original_debug = self.config.system.debug_mode
        
        try:
            # Enable debug mode
            self.config.system.debug_mode = True
            logger.info("Debug mode enabled")
            yield
        finally:
            # Restore original debug mode
            self.config.system.debug_mode = original_debug
            logger.info("Debug mode restored")
    
    def reset_processors(self, key: PRNGKey) -> None:
        """Reset all processors while preserving learned parameters."""
        keys = jax.random.split(key, 2)
        
        # Reset temporal processor
        self.temporal_processor = self.temporal_processor.reset_temporal_state(keys[0])
        
        logger.info("All processors reset successfully")
    
    def update_configuration(self, **kwargs) -> None:
        """Update framework configuration at runtime."""
        from .config import config_manager
        
        # Update global configuration
        new_config = config_manager.update_config(kwargs)
        
        # Update local config reference
        self.config = new_config
        
        logger.info(f"Configuration updated: {kwargs}")


# Factory function for easy framework creation
def create_consciousness_framework(
    config: Optional[UnifiedConfig] = None,
    key: Optional[PRNGKey] = None,
) -> ConsciousnessFramework:
    """Factory function to create consciousness framework."""
    return ConsciousnessFramework(config=config, key=key)


# Convenience function for single consciousness processing
@optimize_for_memory
def process_single_moment(
    sensory_input: Array,
    motor_prediction: Array,
    tactile_feedback: Array,
    environmental_context: Optional[Array] = None,
    key: Optional[PRNGKey] = None,
) -> ConsciousnessState:
    """Process a single consciousness moment with default framework."""
    if key is None:
        key = jax.random.PRNGKey(int(time.time()) % 2**32)
    
    framework = create_consciousness_framework(key=key)
    context = ProcessingContext(
        prng_key=key,
        environmental_context=environmental_context,
    )
    
    return framework.process_consciousness_moment(
        sensory_input, motor_prediction, tactile_feedback, context
    )


# Export public API
__all__ = [
    'ConsciousnessState',
    'ProcessingContext', 
    'ConsciousnessFramework',
    'create_consciousness_framework',
    'process_single_moment',
]