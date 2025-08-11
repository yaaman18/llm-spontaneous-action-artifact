"""JIT compilation utilities and strategies for the enactive consciousness framework.

This module provides utilities for managing JIT compilation, handling static arguments,
and providing fallback strategies when JIT compilation fails.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import warnings

import jax
import jax.numpy as jnp
import equinox as eqx

# Type variables
F = TypeVar('F', bound=Callable[..., Any])

# Configure logging
logger = logging.getLogger(__name__)


class JITCompilationError(Exception):
    """Exception raised when JIT compilation fails."""
    pass


class JITStrategy:
    """Strategy for JIT compilation with fallback options."""
    
    def __init__(
        self,
        enable_jit: bool = True,
        static_argnames: Optional[Union[str, List[str]]] = None,
        static_argnums: Optional[Union[int, List[int]]] = None,
        device: Optional[str] = None,
        backend: Optional[str] = None,
        donate_argnums: Optional[Union[int, List[int]]] = None,
    ):
        self.enable_jit = enable_jit
        self.static_argnames = static_argnames or []
        self.static_argnums = static_argnums or []
        self.device = device
        self.backend = backend
        self.donate_argnums = donate_argnums
        
        # Normalize to lists
        if isinstance(self.static_argnames, str):
            self.static_argnames = [self.static_argnames]
        if isinstance(self.static_argnums, int):
            self.static_argnums = [self.static_argnums]
        if isinstance(self.donate_argnums, int):
            self.donate_argnums = [self.donate_argnums]
    
    def compile(self, func: F) -> F:
        """Compile function with JIT using current strategy."""
        if not self.enable_jit:
            return func
            
        jit_kwargs = {}
        
        if self.static_argnames:
            jit_kwargs['static_argnames'] = self.static_argnames
        if self.static_argnums:
            jit_kwargs['static_argnums'] = self.static_argnums
        if self.device:
            jit_kwargs['device'] = self.device
        if self.backend:
            jit_kwargs['backend'] = self.backend
        if self.donate_argnums:
            jit_kwargs['donate_argnums'] = self.donate_argnums
            
        return jax.jit(func, **jit_kwargs)


def safe_jit(
    func: Optional[F] = None,
    *,
    static_argnames: Optional[Union[str, List[str]]] = None,
    static_argnums: Optional[Union[int, List[int]]] = None,
    device: Optional[str] = None,
    backend: Optional[str] = None,
    donate_argnums: Optional[Union[int, List[int]]] = None,
    fallback_on_error: bool = True,
    warn_on_fallback: bool = True,
) -> Union[F, Callable[[F], F]]:
    """Safe JIT decorator that falls back to non-JIT on compilation errors.
    
    Args:
        func: Function to compile (if used as decorator without parameters)
        static_argnames: Argument names to treat as static
        static_argnums: Argument positions to treat as static
        device: Device to compile for
        backend: Backend to use for compilation
        donate_argnums: Arguments to donate (memory optimization)
        fallback_on_error: Whether to fallback to non-JIT on error
        warn_on_fallback: Whether to emit warning on fallback
        
    Returns:
        JIT-compiled function or original function if compilation fails
        
    Example:
        @safe_jit(static_argnames=['config', 'state_dim'])
        def factory_function(config, state_dim, key):
            return Module(config, state_dim, key)
    """
    
    def decorator(f: F) -> F:
        strategy = JITStrategy(
            enable_jit=True,
            static_argnames=static_argnames,
            static_argnums=static_argnums,
            device=device,
            backend=backend,
            donate_argnums=donate_argnums,
        )
        
        try:
            jit_func = strategy.compile(f)
            
            # Test compilation with dummy inputs (if possible)
            # This is a simple check - for complex functions, 
            # compilation errors might only appear at call time
            
            @functools.wraps(f)
            def safe_wrapper(*args, **kwargs):
                try:
                    return jit_func(*args, **kwargs)
                except Exception as e:
                    if fallback_on_error:
                        if warn_on_fallback:
                            print(f"JIT compilation failed: {e}. Falling back to non-JIT version.")
                        logger.debug(f"JIT fallback for {f.__name__}: {e}")
                        return f(*args, **kwargs)
                    else:
                        raise JITCompilationError(f"JIT compilation failed: {e}") from e
            
            return safe_wrapper
            
        except Exception as e:
            if fallback_on_error:
                if warn_on_fallback:
                    warnings.warn(
                        f"JIT compilation failed for {f.__name__}: {e}. "
                        f"Using non-JIT version.",
                        UserWarning,
                        stacklevel=2
                    )
                logger.debug(f"JIT setup failed for {f.__name__}: {e}")
                return f
            else:
                raise JITCompilationError(f"JIT setup failed: {e}") from e
    
    if func is None:
        # Used as @safe_jit(...)
        return decorator
    else:
        # Used as @safe_jit
        return decorator(func)


def create_jit_factory(
    factory_func: Callable,
    static_argnames: Optional[List[str]] = None,
    enable_jit: bool = True,
    fallback: bool = True,
) -> Tuple[Callable, Callable, Callable]:
    """Create JIT and non-JIT versions of a factory function.
    
    Args:
        factory_func: Original factory function
        static_argnames: Names of arguments to treat as static
        enable_jit: Whether JIT is enabled by default
        fallback: Whether to provide fallback functionality
        
    Returns:
        Tuple of (jit_version, no_jit_version, safe_version)
        
    Example:
        jit_factory, no_jit_factory, safe_factory = create_jit_factory(
            create_processor,
            static_argnames=['config', 'state_dim']
        )
    """
    
    # JIT version
    if enable_jit and static_argnames:
        jit_version = safe_jit(
            factory_func,
            static_argnames=static_argnames,
            fallback_on_error=False,
        )
    else:
        jit_version = factory_func
    
    # Non-JIT version
    no_jit_version = factory_func
    
    # Safe version with fallback
    @functools.wraps(factory_func)
    def safe_version(*args, use_jit: bool = True, **kwargs):
        """Safe factory with JIT/no-JIT selection and fallback."""
        try:
            if use_jit and enable_jit:
                return jit_version(*args, **kwargs)
            else:
                return no_jit_version(*args, **kwargs)
        except Exception as e:
            if fallback and use_jit:
                logger.warning(
                    f"JIT factory failed: {e}. Falling back to non-JIT version."
                )
                return no_jit_version(*args, **kwargs)
            else:
                raise
    
    return jit_version, no_jit_version, safe_version


def validate_static_args(func: Callable, static_argnames: List[str]) -> bool:
    """Validate that static argument names exist in function signature.
    
    Args:
        func: Function to validate
        static_argnames: List of static argument names
        
    Returns:
        True if all static arguments are valid
        
    Raises:
        ValueError: If any static argument name is invalid
    """
    import inspect
    
    try:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        invalid_args = [name for name in static_argnames if name not in param_names]
        
        if invalid_args:
            raise ValueError(
                f"Invalid static argument names: {invalid_args}. "
                f"Available parameters: {param_names}"
            )
        
        return True
        
    except Exception as e:
        logger.warning(f"Could not validate static arguments for {func.__name__}: {e}")
        return False


def get_jit_info() -> Dict[str, Any]:
    """Get information about JAX JIT compilation environment.
    
    Returns:
        Dictionary with JIT environment information
    """
    return {
        'jax_version': jax.__version__,
        'available_backends': jax.lib.xla_bridge.get_backend().platform,
        'devices': [str(device) for device in jax.devices()],
        'jit_enabled': True,  # JAX JIT is always available
        'xla_flags': jax.config.FLAGS.jax_xla_backend,
    }


def create_shape_safe_factory(
    module_class: type,
    config_param: str = 'config',
    shape_params: Optional[List[str]] = None,
) -> Callable:
    """Create a factory function that safely handles shape parameters in JIT.
    
    Args:
        module_class: The Equinox module class to instantiate
        config_param: Name of the configuration parameter
        shape_params: List of shape parameter names (e.g., 'state_dim')
        
    Returns:
        Factory function with proper JIT support
        
    Example:
        create_processor = create_shape_safe_factory(
            ProcessorClass,
            config_param='config',
            shape_params=['state_dim', 'hidden_dim']
        )
    """
    shape_params = shape_params or []
    all_static_params = [config_param] + shape_params
    
    @safe_jit(static_argnames=all_static_params)
    def factory(*args, **kwargs):
        return module_class(*args, **kwargs)
    
    # Add metadata
    factory.__doc__ = f"""JIT-safe factory for {module_class.__name__}.
    
    Static parameters: {all_static_params}
    Shape parameters: {shape_params}
    """
    
    return factory


# Global JIT strategy instance
DEFAULT_JIT_STRATEGY = JITStrategy(
    enable_jit=True,
    static_argnames=[],
    static_argnums=[],
)


# Export public API
__all__ = [
    'JITStrategy',
    'JITCompilationError',
    'safe_jit',
    'create_jit_factory',
    'validate_static_args',
    'get_jit_info',
    'create_shape_safe_factory',
    'DEFAULT_JIT_STRATEGY',
]