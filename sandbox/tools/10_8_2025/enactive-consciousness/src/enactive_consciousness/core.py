"""Core utilities and base classes for the enactive consciousness framework.

This module provides common functionality, performance optimizations,
and base classes following Martin Fowler's refactoring principles.
"""

from __future__ import annotations

import abc
import functools
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar, Union
from contextlib import contextmanager
import gc
import time

import jax
import jax.numpy as jnp
import equinox as eqx
from pydantic import BaseModel, ConfigDict, Field

from .types import (
    Array, 
    ArrayLike, 
    PRNGKey,
    EnactiveConsciousnessError,
    validate_consciousness_state,
)

# Type variables
T = TypeVar('T')
StateT = TypeVar('StateT')

# Configure logging
logger = logging.getLogger(__name__)


class PerformanceConfig(BaseModel):
    """Configuration for performance optimizations."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    
    enable_jit_compilation: bool = Field(default=True)
    enable_memory_optimization: bool = Field(default=True)
    enable_gradient_checkpointing: bool = Field(default=False)
    max_memory_usage_mb: float = Field(default=1024.0, gt=0.0)
    jit_static_argnums: Tuple[int, ...] = Field(default=())


class ArrayValidator:
    """Utility class for array validation with enhanced error messages."""
    
    @staticmethod
    def validate_shape(arr: Array, expected_shape: Tuple[int, ...], name: str = "array") -> None:
        """Validate array shape with descriptive error."""
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {arr.shape}, expected {expected_shape}"
            )
    
    @staticmethod
    def validate_finite(arr: Array, name: str = "array") -> None:
        """Validate that array contains only finite values."""
        if not jnp.all(jnp.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    
    @staticmethod
    def validate_range(arr: Array, min_val: float = None, max_val: float = None, 
                      name: str = "array") -> None:
        """Validate array values are within specified range."""
        if min_val is not None and jnp.any(arr < min_val):
            raise ValueError(f"{name} contains values below {min_val}")
        if max_val is not None and jnp.any(arr > max_val):
            raise ValueError(f"{name} contains values above {max_val}")
    
    @staticmethod
    def validate_normalized(arr: Array, axis: int = -1, name: str = "array") -> None:
        """Validate that array is normalized along specified axis."""
        norms = jnp.linalg.norm(arr, axis=axis, keepdims=True)
        if not jnp.allclose(norms, 1.0, atol=1e-6):
            raise ValueError(f"{name} is not normalized along axis {axis}")


class MemoryManager:
    """Memory management utilities for JAX operations."""
    
    def __init__(self, max_memory_mb: float = 1024.0):
        self.max_memory_mb = max_memory_mb
        self._memory_tracking = {}
    
    @contextmanager
    def track_memory(self, operation_name: str):
        """Context manager for tracking memory usage."""
        start_time = time.time()
        gc.collect()  # Clean up before measurement
        
        try:
            yield
        finally:
            gc.collect()  # Clean up after operation
            end_time = time.time()
            
            # Log performance metrics
            duration = end_time - start_time
            logger.debug(f"{operation_name} completed in {duration:.3f}s")
            
            self._memory_tracking[operation_name] = {
                'duration': duration,
                'timestamp': end_time,
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'tracked_operations': len(self._memory_tracking),
            'recent_operations': list(self._memory_tracking.keys())[-5:],
            'max_memory_mb': self.max_memory_mb,
        }
    
    @staticmethod 
    def optimize_array_memory(arr: Array) -> Array:
        """Optimize array memory usage."""
        # Convert to appropriate dtype if possible
        if jnp.all(arr == jnp.asarray(arr, dtype=jnp.float32)):
            return jnp.asarray(arr, dtype=jnp.float32)
        return arr


class JITCompiler:
    """Utility for managing JIT compilation with configuration."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._compiled_functions = {}
    
    def compile_if_enabled(self, func, static_argnums=None, donate_argnums=None):
        """Conditionally compile function based on configuration."""
        if not self.config.enable_jit_compilation:
            return func
            
        key = (func.__name__, static_argnums, donate_argnums)
        if key not in self._compiled_functions:
            compiled = jax.jit(
                func,
                static_argnums=static_argnums or self.config.jit_static_argnums,
                donate_argnums=donate_argnums,
            )
            self._compiled_functions[key] = compiled
            logger.debug(f"JIT compiled function: {func.__name__}")
        
        return self._compiled_functions[key]
    
    def clear_cache(self) -> None:
        """Clear compiled function cache."""
        self._compiled_functions.clear()
        logger.debug("JIT compilation cache cleared")


class StateValidationMixin:
    """Mixin providing common state validation functionality."""
    
    def validate_input_state(self, state: Array, name: str) -> None:
        """Validate input state array."""
        if not validate_consciousness_state(state):
            raise EnactiveConsciousnessError(f"Invalid {name} state")
        
        ArrayValidator.validate_finite(state, name)
    
    def validate_output_state(self, state: Array, name: str) -> Array:
        """Validate and clean output state."""
        ArrayValidator.validate_finite(state, name)
        
        # Clip extreme values
        state = jnp.clip(state, -10.0, 10.0)
        
        return state


class ConfigurableMixin:
    """Mixin for classes with configurable behavior."""
    
    def update_config(self, **kwargs) -> 'ConfigurableMixin':
        """Update configuration parameters."""
        if hasattr(self, 'config'):
            # Create new config with updated values
            config_dict = self.config.model_dump()
            config_dict.update(kwargs)
            new_config = type(self.config)(**config_dict)
            
            # Return new instance with updated config
            return eqx.tree_at(lambda x: x.config, self, new_config)
        
        raise AttributeError("No config attribute found")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        if hasattr(self, 'config'):
            return self.config.model_dump()
        return {}


class MetricCollector:
    """Collect and aggregate performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def record_metric(self, name: str, value: float, timestamp: float = None) -> None:
        """Record a metric value."""
        if timestamp is None:
            timestamp = time.time()
            
        if name not in self.history:
            self.history[name] = []
        
        self.history[name].append({'value': value, 'timestamp': timestamp})
        self.metrics[name] = value
    
    def get_metric_statistics(self, name: str, window_size: int = 100) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.history:
            return {}
        
        values = [entry['value'] for entry in self.history[name][-window_size:]]
        values_array = jnp.array(values)
        
        return {
            'mean': float(jnp.mean(values_array)),
            'std': float(jnp.std(values_array)),
            'min': float(jnp.min(values_array)),
            'max': float(jnp.max(values_array)),
            'latest': float(values_array[-1]) if len(values) > 0 else 0.0,
            'count': len(values),
        }
    
    def clear_history(self, name: str = None) -> None:
        """Clear metric history."""
        if name is None:
            self.history.clear()
        elif name in self.history:
            del self.history[name]


class ProcessorBase(eqx.Module):
    """Abstract base class for all consciousness processors."""
    
    config: Any = eqx.field(static=True)
    memory_manager: MemoryManager = eqx.field(static=True, default=None)
    metrics: MetricCollector = eqx.field(static=True, default=None) 
    
    def process(self, *args, **kwargs) -> Any:
        """Main processing method - should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process() method")
    
    def validate_processing_inputs(self, inputs: Dict[str, Array]) -> None:
        """Validate inputs for processing."""
        for name, array in inputs.items():
            ArrayValidator.validate_finite(array, name)
    
    def record_processing_time(self, operation_name: str, duration: float) -> None:
        """Record processing time metric."""
        self.metrics.record_metric(f"{operation_name}_time_ms", duration * 1000)


class ErrorHandler:
    """Centralized error handling with context."""
    
    @staticmethod
    def handle_processing_error(error: Exception, context: str, 
                               fallback_value: Any = None) -> Any:
        """Handle processing errors with context information."""
        error_msg = f"Error in {context}: {str(error)}"
        logger.error(error_msg, exc_info=True)
        
        if fallback_value is not None:
            logger.warning(f"Returning fallback value for {context}")
            return fallback_value
        
        # Re-raise with enhanced context
        raise EnactiveConsciousnessError(error_msg) from error
    
    @contextmanager
    def error_context(self, context: str, fallback_value: Any = None):
        """Context manager for error handling."""
        try:
            yield
        except Exception as e:
            if fallback_value is not None:
                yield fallback_value
            else:
                self.handle_processing_error(e, context)


def create_safe_jit_function(func, static_argnums=None, error_context: str = None):
    """Create a JIT-compiled function with error handling."""
    
    @functools.wraps(func)
    def safe_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = error_context or f"JIT function {func.__name__}"
            ErrorHandler.handle_processing_error(e, context)
    
    if static_argnums is not None:
        return jax.jit(safe_wrapper, static_argnums=static_argnums)
    else:
        return jax.jit(safe_wrapper)


def optimize_for_memory(func):
    """Decorator for memory optimization."""
    
    @functools.wraps(func)
    def memory_optimized(*args, **kwargs):
        # Pre-process: optimize input arrays
        optimized_args = []
        for arg in args:
            if isinstance(arg, jnp.ndarray):
                optimized_args.append(MemoryManager.optimize_array_memory(arg))
            else:
                optimized_args.append(arg)
        
        # Execute with memory tracking
        with MemoryManager().track_memory(func.__name__):
            result = func(*optimized_args, **kwargs)
        
        # Post-process: optimize output
        if isinstance(result, jnp.ndarray):
            result = MemoryManager.optimize_array_memory(result)
        elif isinstance(result, (tuple, list)):
            result = type(result)(
                MemoryManager.optimize_array_memory(item) if isinstance(item, jnp.ndarray) else item
                for item in result
            )
        
        return result
    
    return memory_optimized


# Factory functions
def create_memory_manager(max_memory_mb: float = 1024.0) -> MemoryManager:
    """Factory for memory manager."""
    return MemoryManager(max_memory_mb)


def create_metric_collector() -> MetricCollector:
    """Factory for metric collector."""
    return MetricCollector()


def create_performance_config(**kwargs) -> PerformanceConfig:
    """Factory for performance configuration."""
    return PerformanceConfig(**kwargs)


# Global utilities
GLOBAL_MEMORY_MANAGER = create_memory_manager()
GLOBAL_METRICS = create_metric_collector()
ERROR_HANDLER = ErrorHandler()


# Export public API
__all__ = [
    'PerformanceConfig',
    'ArrayValidator', 
    'MemoryManager',
    'JITCompiler',
    'StateValidationMixin',
    'ConfigurableMixin',
    'MetricCollector',
    'ProcessorBase',
    'ErrorHandler',
    'create_safe_jit_function',
    'optimize_for_memory',
    'create_memory_manager',
    'create_metric_collector',
    'create_performance_config',
    'GLOBAL_MEMORY_MANAGER',
    'GLOBAL_METRICS', 
    'ERROR_HANDLER',
]