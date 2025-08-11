"""Equinox State Framework - Concrete implementation layer.

This module implements the Framework layer of Clean Architecture, providing
concrete implementations using JAX and Equinox for state management.

Key Design Principles:
- Framework-specific implementation details isolated from business logic
- Dependency Inversion: Framework depends on abstractions from inner layers
- Single Responsibility: Each class handles one framework-specific concern
- Adapter Pattern: Translate between framework and architecture abstractions

This layer contains the "dirty details" of working with Equinox, JAX trees,
and low-level state management operations.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional, Generic, TypeVar, Union, Tuple, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax

from .state_entities import (
    StateSnapshot,
    StateEvolutionEvent,
    StateEvolutionType,
    StateType,
    ImmutableStateContainer,
    StateConsistencyRules
)
from .state_interfaces import (
    StateManager,
    StateEvolutionStrategy,
    StateValidator,
    StateThreadingCoordinator,
    ReadableState,
    WritableState,
    StateChangeObserver,
    StatePublisher
)
from ..types import Array, TimeStep, PRNGKey


logger = logging.getLogger(__name__)

StateT = TypeVar('StateT')


@dataclass
class EquinoxStateConfiguration:
    """Configuration for Equinox state management."""
    
    enable_jit: bool = True
    enable_checkpointing: bool = True
    max_checkpoints: int = 10
    validation_mode: str = "strict"  # strict, lenient, disabled
    threading_mode: str = "sequential"  # sequential, parallel, async
    memory_optimization: bool = True
    state_history_limit: int = 100
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_checkpoints < 1:
            raise ValueError("max_checkpoints must be at least 1")
        if self.state_history_limit < 1:
            raise ValueError("state_history_limit must be at least 1")
        if self.validation_mode not in ["strict", "lenient", "disabled"]:
            raise ValueError("validation_mode must be 'strict', 'lenient', or 'disabled'")


class EquinoxStateHolder(eqx.Module):
    """Equinox module for holding state with proper PyTree structure.
    
    This class wraps state data in an Equinox module to ensure proper
    JAX PyTree handling and enable JIT compilation of state operations.
    """
    
    state_data: Array
    state_metadata: Dict[str, Any] = eqx.field(static=True)
    creation_timestamp: float = eqx.field(static=True)
    
    def __init__(
        self,
        initial_state: Array,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.state_data = initial_state
        self.state_metadata = metadata or {}
        self.creation_timestamp = time.time()
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get shape of state data."""
        return self.state_data.shape
    
    @property
    def size(self) -> int:
        """Get size of state data."""
        return self.state_data.size
    
    def with_updated_data(self, new_data: Array) -> EquinoxStateHolder:
        """Create new holder with updated data (immutable update)."""
        return EquinoxStateHolder(
            new_data,
            self.state_metadata
        )
    
    def with_metadata(self, **kwargs) -> EquinoxStateHolder:
        """Create new holder with updated metadata."""
        new_metadata = {**self.state_metadata, **kwargs}
        return EquinoxStateHolder(
            self.state_data,
            new_metadata
        )


class EquinoxStateEvolutionEngine:
    """Engine for applying state evolution strategies using Equinox patterns.
    
    This class handles the low-level details of applying state transformations
    while maintaining Equinox PyTree structure and enabling JIT compilation.
    """
    
    def __init__(self, config: EquinoxStateConfiguration):
        self.config = config
        self._compiled_evolution_functions: Dict[str, Callable] = {}
        self._evolution_statistics: Dict[str, List[float]] = {}
    
    @jax.jit
    def _jit_evolve_state(
        self,
        state_holder: EquinoxStateHolder,
        evolution_fn: Callable[[Array], Array],
        evolution_params: Dict[str, Any]
    ) -> EquinoxStateHolder:
        """JIT-compiled state evolution function."""
        new_state_data = evolution_fn(state_holder.state_data)
        return state_holder.with_updated_data(new_state_data)
    
    def evolve_state_with_strategy(
        self,
        state_holder: EquinoxStateHolder,
        strategy: StateEvolutionStrategy[Array],
        evolution_input: Any,
        context: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> EquinoxStateHolder:
        """Apply evolution strategy to state holder."""
        start_time = time.time()
        
        try:
            # Apply strategy to get new state
            new_state_data = strategy.evolve(
                state_holder.state_data,
                evolution_input,
                context,
                key
            )
            
            # Create new state holder
            if self.config.enable_jit:
                # Use JIT compilation for performance
                evolution_fn = lambda x: strategy.evolve(x, evolution_input, context, key)
                new_holder = self._jit_evolve_state(
                    state_holder,
                    evolution_fn,
                    context
                )
            else:
                new_holder = state_holder.with_updated_data(new_state_data)
            
            # Record performance statistics
            evolution_time = (time.time() - start_time) * 1000
            strategy_name = strategy.__class__.__name__
            if strategy_name not in self._evolution_statistics:
                self._evolution_statistics[strategy_name] = []
            self._evolution_statistics[strategy_name].append(evolution_time)
            
            return new_holder
            
        except Exception as e:
            logger.error(f"State evolution failed: {str(e)}")
            raise
    
    @jax.jit
    def batch_evolve_states(
        self,
        state_holders: List[EquinoxStateHolder],
        evolution_fn: Callable[[Array], Array]
    ) -> List[EquinoxStateHolder]:
        """JIT-compiled batch state evolution."""
        # Stack state data for vectorized processing
        stacked_data = jnp.stack([holder.state_data for holder in state_holders])
        
        # Apply evolution function to all states
        evolved_data = jax.vmap(evolution_fn)(stacked_data)
        
        # Create new holders with evolved data
        return [
            holder.with_updated_data(evolved_data[i])
            for i, holder in enumerate(state_holders)
        ]
    
    def get_evolution_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for evolution strategies."""
        stats = {}
        for strategy_name, times in self._evolution_statistics.items():
            if times:
                stats[strategy_name] = {
                    'mean_time_ms': jnp.mean(jnp.array(times)),
                    'max_time_ms': jnp.max(jnp.array(times)),
                    'min_time_ms': jnp.min(jnp.array(times)),
                    'total_calls': len(times)
                }
        return stats


class EquinoxStateManager:
    """Complete state manager implementation using Equinox framework.
    
    This class provides the full state management implementation following
    Clean Architecture principles while leveraging Equinox for performance.
    """
    
    def __init__(
        self,
        initial_state: Array,
        state_type: StateType,
        config: Optional[EquinoxStateConfiguration] = None,
        consistency_rules: Optional[StateConsistencyRules] = None,
        evolution_strategy: Optional[StateEvolutionStrategy[Array]] = None
    ):
        self.config = config or EquinoxStateConfiguration()
        self.state_type = state_type
        
        # Initialize core components
        self._state_holder = EquinoxStateHolder(initial_state)
        self._evolution_engine = EquinoxStateEvolutionEngine(self.config)
        self._consistency_rules = consistency_rules or StateConsistencyRules()
        self._evolution_strategy = evolution_strategy
        
        # Initialize state history and checkpoints
        self._state_history: List[EquinoxStateHolder] = [self._state_holder]
        self._checkpoints: Dict[str, EquinoxStateHolder] = {}
        self._evolution_events: List[StateEvolutionEvent] = []
        
        # Initialize observers and validation
        self._observers: List[StateChangeObserver] = []
        self._validator = EquinoxStateValidator(self._consistency_rules)
        
        # Threading support
        self._state_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2) if self.config.threading_mode == "async" else None
    
    def get_readable_state(self) -> ReadableState[Array]:
        """Get read-only interface to state."""
        return EquinoxReadableStateImpl(self._state_holder, self.state_type)
    
    def get_writable_state(self) -> WritableState[Array]:
        """Get writable interface to state."""
        return EquinoxWritableStateImpl(
            self,
            self._evolution_engine,
            self._consistency_rules
        )
    
    def get_validator(self) -> StateValidator:
        """Get state validator."""
        return self._validator
    
    def set_evolution_strategy(self, strategy: StateEvolutionStrategy[Array]) -> None:
        """Set strategy for state evolution."""
        self._evolution_strategy = strategy
    
    def get_evolution_history(self) -> List[StateEvolutionEvent]:
        """Get history of state evolution events."""
        with self._state_lock:
            return list(self._evolution_events)  # Defensive copy
    
    @contextmanager
    def transaction_context(self):
        """Context manager for transactional state operations."""
        checkpoint_id = self.create_checkpoint()
        
        try:
            yield
        except Exception as e:
            self.restore_checkpoint(checkpoint_id)
            logger.warning(f"Transaction rolled back: {str(e)}")
            raise
        finally:
            # Cleanup checkpoint
            if checkpoint_id in self._checkpoints:
                del self._checkpoints[checkpoint_id]
    
    def create_checkpoint(self) -> str:
        """Create state checkpoint."""
        with self._state_lock:
            checkpoint_id = str(uuid.uuid4())
            
            # Create deep copy of current state
            if self.config.memory_optimization:
                # Use JAX tree_map for efficient copying
                self._checkpoints[checkpoint_id] = jax.tree_util.tree_map(
                    jnp.copy, self._state_holder
                )
            else:
                self._checkpoints[checkpoint_id] = self._state_holder
            
            # Manage checkpoint limit
            if len(self._checkpoints) > self.config.max_checkpoints:
                oldest_checkpoint = min(self._checkpoints.keys())
                del self._checkpoints[oldest_checkpoint]
            
            return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from checkpoint."""
        with self._state_lock:
            if checkpoint_id not in self._checkpoints:
                return False
            
            try:
                # Restore state holder
                previous_state = self._state_holder.state_data
                self._state_holder = self._checkpoints[checkpoint_id]
                
                # Record evolution event
                evolution_event = StateEvolutionEvent(
                    evolution_type=StateEvolutionType.DISCRETE_UPDATE,
                    source_state_id="checkpoint",
                    target_state_id=checkpoint_id,
                    event_data=f"restored_from_{checkpoint_id}"
                )
                self._evolution_events.append(evolution_event)
                
                # Notify observers
                self._notify_observers(
                    previous_state,
                    self._state_holder.state_data,
                    evolution_event
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Checkpoint restore failed: {str(e)}")
                return False
    
    def _update_state_internal(
        self,
        new_state_data: Array,
        evolution_type: StateEvolutionType,
        event_data: Any
    ) -> None:
        """Internal method for updating state with proper synchronization."""
        with self._state_lock:
            # Store previous state for notification
            previous_state = self._state_holder.state_data
            
            # Create new state holder
            new_state_holder = self._state_holder.with_updated_data(new_state_data)
            
            # Update state holder
            self._state_holder = new_state_holder
            
            # Update history
            self._state_history.append(new_state_holder)
            if len(self._state_history) > self.config.state_history_limit:
                self._state_history.pop(0)
            
            # Record evolution event
            evolution_event = StateEvolutionEvent(
                evolution_type=evolution_type,
                source_state_id=str(len(self._evolution_events)),
                target_state_id=str(len(self._evolution_events) + 1),
                event_data=event_data
            )
            self._evolution_events.append(evolution_event)
            
            # Notify observers
            self._notify_observers(previous_state, new_state_data, evolution_event)
    
    def _notify_observers(
        self,
        previous_state: Array,
        new_state: Array,
        evolution_event: StateEvolutionEvent
    ) -> None:
        """Notify all observers of state change."""
        for observer in self._observers:
            try:
                observer.on_state_changed(previous_state, new_state, evolution_event)
            except Exception as e:
                logger.warning(f"Observer notification failed: {str(e)}")
    
    def subscribe_observer(self, observer: StateChangeObserver) -> None:
        """Subscribe observer to state changes."""
        with self._state_lock:
            if observer not in self._observers:
                self._observers.append(observer)
    
    def unsubscribe_observer(self, observer: StateChangeObserver) -> None:
        """Unsubscribe observer from state changes."""
        with self._state_lock:
            if observer in self._observers:
                self._observers.remove(observer)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this state manager."""
        return {
            'state_history_length': len(self._state_history),
            'evolution_events_count': len(self._evolution_events),
            'checkpoints_count': len(self._checkpoints),
            'observers_count': len(self._observers),
            'evolution_statistics': self._evolution_engine.get_evolution_statistics(),
            'memory_optimization_enabled': self.config.memory_optimization,
            'jit_enabled': self.config.enable_jit
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        self._checkpoints.clear()
        self._observers.clear()


class EquinoxReadableStateImpl:
    """Concrete implementation of ReadableState interface for Equinox."""
    
    def __init__(self, state_holder: EquinoxStateHolder, state_type: StateType):
        self._state_holder = state_holder
        self._state_type = state_type
    
    def get_current_state(self) -> Array:
        """Get current state data."""
        return self._state_holder.state_data
    
    def get_state_snapshot(self) -> StateSnapshot[Array]:
        """Get current state as immutable snapshot."""
        return StateSnapshot(
            state_type=self._state_type,
            data=self._state_holder.state_data,
            timestamp=self._state_holder.creation_timestamp,
            metadata=self._state_holder.state_metadata
        )
    
    def get_state_type(self) -> StateType:
        """Get the type of state managed."""
        return self._state_type
    
    def get_state_at_time(self, timestamp: TimeStep) -> Optional[Array]:
        """Get state at specific timestamp if available."""
        # For now, only return current state if timestamp matches
        if abs(self._state_holder.creation_timestamp - timestamp) < 0.1:
            return self._state_holder.state_data
        return None


class EquinoxWritableStateImpl:
    """Concrete implementation of WritableState interface for Equinox."""
    
    def __init__(
        self,
        state_manager: EquinoxStateManager,
        evolution_engine: EquinoxStateEvolutionEngine,
        consistency_rules: StateConsistencyRules
    ):
        self._state_manager = state_manager
        self._evolution_engine = evolution_engine
        self._consistency_rules = consistency_rules
    
    def update_state(
        self,
        new_state: Array,
        evolution_type: StateEvolutionType = StateEvolutionType.DISCRETE_UPDATE,
        event_data: Any = None
    ) -> None:
        """Update state with new value."""
        # Validate new state
        if not self._consistency_rules.validate_state_bounds(new_state):
            raise ValueError("New state violates consistency rules")
        
        # Update state through manager
        self._state_manager._update_state_internal(new_state, evolution_type, event_data)
    
    def evolve_state_with_function(
        self,
        evolution_fn: Callable[[Array], Array],
        evolution_type: StateEvolutionType = StateEvolutionType.CONTINUOUS_FLOW,
        event_data: Any = None
    ) -> None:
        """Evolve state using provided function."""
        current_state = self._state_manager._state_holder.state_data
        new_state = evolution_fn(current_state)
        self.update_state(new_state, evolution_type, event_data)


class EquinoxStateValidator:
    """Concrete state validator implementation for Equinox states."""
    
    def __init__(self, consistency_rules: StateConsistencyRules):
        self._rules = consistency_rules
    
    def validate_state(self, state: Array) -> Tuple[bool, List[str]]:
        """Validate state data."""
        errors = []
        
        # Check for finite values
        if not jnp.all(jnp.isfinite(state)):
            errors.append("State contains non-finite values")
        
        # Check bounds
        if not self._rules.validate_state_bounds(state):
            errors.append(f"State magnitude exceeds bounds ({self._rules.max_state_magnitude})")
        
        return len(errors) == 0, errors
    
    def validate_evolution(
        self,
        from_state: Array,
        to_state: Array,
        time_delta: float
    ) -> Tuple[bool, List[str]]:
        """Validate state evolution step."""
        errors = []
        
        # Validate individual states
        from_valid, from_errors = self.validate_state(from_state)
        to_valid, to_errors = self.validate_state(to_state)
        
        errors.extend(from_errors)
        errors.extend(to_errors)
        
        # Validate evolution rate if both states are valid
        if from_valid and to_valid:
            if not self._rules.validate_evolution_rate(from_state, to_state, time_delta):
                errors.append(f"Evolution rate exceeds maximum allowed ({self._rules.max_evolution_rate})")
        
        # Validate temporal ordering
        if time_delta <= 0:
            errors.append("Time delta must be positive")
        elif time_delta < self._rules.min_temporal_interval:
            errors.append(f"Time delta below minimum interval ({self._rules.min_temporal_interval})")
        
        return len(errors) == 0, errors
    
    def get_consistency_rules(self) -> StateConsistencyRules:
        """Get current consistency rules."""
        return self._rules


class EquinoxStateThreadingCoordinator:
    """Threading coordinator implementation using Equinox and JAX."""
    
    def __init__(self, config: EquinoxStateConfiguration):
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def thread_state_through_pipeline(
        self,
        initial_state: Any,
        processing_steps: List[Callable],
        context: Dict[str, Any]
    ) -> Tuple[Any, List[StateEvolutionEvent]]:
        """Thread state through processing pipeline."""
        current_state = initial_state
        evolution_events = []
        
        if self.config.threading_mode == "sequential":
            # Sequential processing
            for i, step in enumerate(processing_steps):
                try:
                    previous_state = current_state
                    current_state = step(current_state)
                    
                    # Record evolution event
                    event = StateEvolutionEvent(
                        evolution_type=StateEvolutionType.CONTINUOUS_FLOW,
                        source_state_id=f"step_{i}",
                        target_state_id=f"step_{i+1}",
                        event_data={'step_index': i, 'step_name': step.__name__}
                    )
                    evolution_events.append(event)
                    
                except Exception as e:
                    self._logger.error(f"Pipeline step {i} failed: {str(e)}")
                    raise
        
        elif self.config.threading_mode == "parallel":
            # Parallel processing where possible
            # This is a simplified implementation - in practice would need
            # dependency analysis to determine which steps can be parallelized
            current_state = self._parallel_pipeline_processing(
                initial_state, processing_steps, context
            )
            
            # Create single evolution event for parallel processing
            event = StateEvolutionEvent(
                evolution_type=StateEvolutionType.SYNTHESIS_INTEGRATION,
                source_state_id="parallel_start",
                target_state_id="parallel_end",
                event_data={'parallel_steps': len(processing_steps)}
            )
            evolution_events.append(event)
        
        return current_state, evolution_events
    
    def _parallel_pipeline_processing(
        self,
        initial_state: Any,
        processing_steps: List[Callable],
        context: Dict[str, Any]
    ) -> Any:
        """Execute pipeline steps in parallel where possible."""
        # This is a simplified implementation
        # In practice, would need sophisticated dependency analysis
        
        # For now, apply all steps to copies of the initial state
        # and then merge results
        if isinstance(initial_state, jnp.ndarray):
            # Apply each step to the state
            results = []
            for step in processing_steps:
                result = step(initial_state)
                results.append(result)
            
            # Simple merge strategy: average all results
            if results:
                stacked_results = jnp.stack(results)
                merged_state = jnp.mean(stacked_results, axis=0)
                return merged_state
        
        return initial_state
    
    def parallel_state_evolution(
        self,
        state_managers: List[StateManager],
        evolution_inputs: List[Any],
        synchronization_points: Optional[List[int]] = None
    ) -> List[StateEvolutionEvent]:
        """Coordinate parallel state evolution across managers."""
        evolution_events = []
        
        # Simple parallel evolution - apply inputs to corresponding managers
        for i, (manager, input_data) in enumerate(zip(state_managers, evolution_inputs)):
            try:
                writable = manager.get_writable_state()
                writable.update_state(
                    input_data,
                    StateEvolutionType.DISCRETE_UPDATE,
                    f"parallel_evolution_{i}"
                )
                
                # Record event
                event = StateEvolutionEvent(
                    evolution_type=StateEvolutionType.DISCRETE_UPDATE,
                    source_state_id=f"manager_{i}_before",
                    target_state_id=f"manager_{i}_after",
                    event_data=f"parallel_evolution_{i}"
                )
                evolution_events.append(event)
                
            except Exception as e:
                self._logger.error(f"Parallel evolution failed for manager {i}: {str(e)}")
        
        return evolution_events
    
    def merge_state_branches(
        self,
        branch_states: List[Any],
        merge_strategy: str = "weighted_average"
    ) -> Any:
        """Merge multiple state branches into unified state."""
        if not branch_states:
            raise ValueError("No branch states to merge")
        
        if len(branch_states) == 1:
            return branch_states[0]
        
        # Convert to arrays if possible
        array_states = []
        for state in branch_states:
            if isinstance(state, jnp.ndarray):
                array_states.append(state)
            else:
                # Try to convert to array
                try:
                    array_states.append(jnp.asarray(state))
                except:
                    raise ValueError("Cannot convert state to array for merging")
        
        if merge_strategy == "weighted_average":
            # Simple uniform weighting for now
            stacked_states = jnp.stack(array_states)
            return jnp.mean(stacked_states, axis=0)
        
        elif merge_strategy == "max_pooling":
            stacked_states = jnp.stack(array_states)
            return jnp.max(stacked_states, axis=0)
        
        elif merge_strategy == "concatenate":
            return jnp.concatenate(array_states, axis=-1)
        
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")


# Factory Functions

def create_equinox_state_manager(
    initial_state: Array,
    state_type: StateType,
    config: Optional[EquinoxStateConfiguration] = None,
    consistency_rules: Optional[StateConsistencyRules] = None,
    evolution_strategy: Optional[StateEvolutionStrategy[Array]] = None
) -> EquinoxStateManager:
    """Factory function to create Equinox state manager."""
    return EquinoxStateManager(
        initial_state=initial_state,
        state_type=state_type,
        config=config,
        consistency_rules=consistency_rules,
        evolution_strategy=evolution_strategy
    )


def create_equinox_threading_coordinator(
    config: Optional[EquinoxStateConfiguration] = None
) -> EquinoxStateThreadingCoordinator:
    """Factory function to create Equinox threading coordinator."""
    return EquinoxStateThreadingCoordinator(
        config or EquinoxStateConfiguration()
    )