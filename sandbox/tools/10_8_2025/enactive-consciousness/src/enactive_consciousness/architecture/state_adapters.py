"""State Adapters - Interface adapters following Clean Architecture.

This module implements the Interface Adapters layer, providing concrete
implementations that adapt between the framework-agnostic use cases and
the specific Equinox/JAX framework implementation.

Key Design Principles:
- Adapter Pattern: Convert interface of a class into another interface clients expect
- Dependency Inversion: High-level modules depend on abstractions
- Single Responsibility: Each adapter handles one specific adaptation concern
- Open/Closed: New adapters can be added without modifying existing code

Interface Adapters sit between Use Cases and Frameworks, translating
data formats and orchestrating framework-specific operations.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional, Generic, TypeVar, Union, Tuple
from contextlib import contextmanager
import time
import uuid

import jax
import jax.numpy as jnp
import equinox as eqx

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
    ReadableState,
    WritableState,
    StateEvolutionStrategy,
    StateValidator,
    TemporalStateManager,
    EmbodimentStateManager,
    CouplingStateManager,
    StateChangeObserver,
    StatePublisher
)
from ..types import Array, TimeStep, PRNGKey, TemporalMoment, BodyState, CouplingState


logger = logging.getLogger(__name__)

StateT = TypeVar('StateT')


class BaseStateAdapter(Generic[StateT]):
    """Base adapter providing common functionality for all state adapters.
    
    Implements common patterns shared across all state adapters while
    allowing specialization for specific state types.
    """
    
    def __init__(
        self,
        state_type: StateType,
        consistency_rules: Optional[StateConsistencyRules] = None
    ):
        self._state_type = state_type
        self._consistency_rules = consistency_rules or StateConsistencyRules()
        self._observers: List[StateChangeObserver] = []
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def _validate_state_data(self, state_data: StateT) -> Tuple[bool, List[str]]:
        """Validate state data against consistency rules."""
        errors = []
        
        if isinstance(state_data, jnp.ndarray):
            if not self._consistency_rules.validate_state_bounds(state_data):
                errors.append("State data violates magnitude bounds")
        
        return len(errors) == 0, errors
    
    def _notify_observers(
        self,
        previous_state: StateT,
        new_state: StateT,
        evolution_event: StateEvolutionEvent
    ) -> None:
        """Notify registered observers of state changes."""
        for observer in self._observers:
            try:
                observer.on_state_changed(previous_state, new_state, evolution_event)
            except Exception as e:
                self._logger.warning(f"Observer notification failed: {str(e)}")
    
    def subscribe_observer(self, observer: StateChangeObserver) -> None:
        """Subscribe observer to state changes."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def unsubscribe_observer(self, observer: StateChangeObserver) -> None:
        """Unsubscribe observer from state changes."""
        if observer in self._observers:
            self._observers.remove(observer)


class EquinoxStateAdapter(BaseStateAdapter[Array]):
    """Adapter for integrating Equinox state management with Clean Architecture.
    
    This adapter translates between Clean Architecture state abstractions
    and Equinox's specific state management patterns (eqx.nn.State, tree_at, etc.).
    """
    
    def __init__(
        self,
        initial_state: Array,
        state_type: StateType,
        consistency_rules: Optional[StateConsistencyRules] = None
    ):
        super().__init__(state_type, consistency_rules)
        
        # Store state directly (Equinox State API has changed in v0.13)
        self._current_state = initial_state
        self._state_container = ImmutableStateContainer(
            initial_state, state_type, consistency_rules
        )
        self._checkpoints: Dict[str, eqx.nn.State] = {}
    
    def get_readable_state(self) -> ReadableState[Array]:
        """Get read-only interface to state."""
        return EquinoxReadableState(self._current_state, self._state_container)
    
    def get_writable_state(self) -> WritableState[Array]:
        """Get writable interface to state."""
        return EquinoxWritableState(
            self,  # Pass the adapter itself
            self._state_container,
            self._consistency_rules,
            self._notify_observers
        )
    
    def get_validator(self) -> StateValidator:
        """Get state validator."""
        return EquinoxStateValidator(self._consistency_rules)
    
    def get_evolution_history(self) -> List[StateEvolutionEvent]:
        """Get evolution history."""
        return self._state_container.evolution_history
    
    def set_evolution_strategy(self, strategy: StateEvolutionStrategy[Array]) -> None:
        """Set evolution strategy (delegated to writable state)."""
        writable = self.get_writable_state()
        if hasattr(writable, 'set_evolution_strategy'):
            writable.set_evolution_strategy(strategy)
    
    @contextmanager
    def transaction_context(self):
        """Context manager for transactional state operations."""
        # Create checkpoint before transaction
        checkpoint_id = self.create_checkpoint()
        
        try:
            yield
        except Exception as e:
            # Rollback on exception
            self.restore_checkpoint(checkpoint_id)
            self._logger.warning(f"Transaction rolled back due to: {str(e)}")
            raise
        finally:
            # Cleanup checkpoint (optional - could keep for audit trail)
            if checkpoint_id in self._checkpoints:
                del self._checkpoints[checkpoint_id]
    
    def create_checkpoint(self) -> str:
        """Create state checkpoint."""
        checkpoint_id = str(uuid.uuid4())
        # Create deep copy of equinox state
        self._checkpoints[checkpoint_id] = jax.tree_util.tree_map(
            jnp.copy, self._equinox_state
        )
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from checkpoint."""
        if checkpoint_id not in self._checkpoints:
            return False
        
        try:
            # Restore equinox state
            self._equinox_state = self._checkpoints[checkpoint_id]
            
            # Update state container
            restored_data = self._equinox_state.get()
            self._state_container = self._state_container.evolve_state(
                restored_data,
                StateEvolutionType.DISCRETE_UPDATE,
                f"restored_from_checkpoint_{checkpoint_id}"
            )
            
            return True
            
        except Exception as e:
            self._logger.error(f"Checkpoint restore failed: {str(e)}")
            return False


class EquinoxReadableState:
    """Read-only interface to Equinox state."""
    
    def __init__(self, current_state: Array, container: ImmutableStateContainer):
        self._current_state = current_state
        self._container = container
    
    def get_current_state(self) -> Array:
        """Get current state data."""
        return self._current_state
    
    def get_state_snapshot(self) -> StateSnapshot[Array]:
        """Get current state as immutable snapshot."""
        return self._container.current_snapshot
    
    def get_state_type(self) -> StateType:
        """Get the type of state managed."""
        return self._container.state_type
    
    def get_state_at_time(self, timestamp: TimeStep) -> Optional[Array]:
        """Get state at specific timestamp if available."""
        snapshot = self._container.snapshot_at_time(timestamp)
        return snapshot.data if snapshot else None


class EquinoxWritableState:
    """Writable interface to Equinox state with immutable updates."""
    
    def __init__(
        self,
        adapter: EquinoxStateAdapter,
        container: ImmutableStateContainer,
        consistency_rules: StateConsistencyRules,
        notify_callback: callable
    ):
        self._adapter = adapter
        self._container = container
        self._consistency_rules = consistency_rules
        self._notify_callback = notify_callback
        self._evolution_strategy: Optional[StateEvolutionStrategy] = None
    
    def update_state(
        self,
        new_state: Array,
        evolution_type: StateEvolutionType = StateEvolutionType.DISCRETE_UPDATE,
        event_data: Any = None
    ) -> None:
        """Update state with new value using Equinox patterns."""
        # Validate new state
        if not self._consistency_rules.validate_state_bounds(new_state):
            raise ValueError("New state violates consistency rules")
        
        # Get previous state for notification
        previous_state = self._adapter._current_state
        
        # Update state in adapter
        self._adapter._current_state = new_state
        
        # Update container
        self._container = self._container.evolve_state(
            new_state, evolution_type, event_data
        )
        
        # Create evolution event for notification
        evolution_event = StateEvolutionEvent(
            evolution_type=evolution_type,
            source_state_id=self._container.evolution_history[-2].target_state_id if len(self._container.evolution_history) > 1 else "initial",
            target_state_id=self._container.current_snapshot.snapshot_id,
            event_data=event_data
        )
        
        # Notify observers
        self._notify_callback(previous_state, new_state, evolution_event)
    
    def evolve_state_with_function(
        self,
        evolution_fn: callable[[Array], Array],
        evolution_type: StateEvolutionType = StateEvolutionType.CONTINUOUS_FLOW,
        event_data: Any = None
    ) -> None:
        """Evolve state using provided function."""
        current_state = self._adapter._current_state
        new_state = evolution_fn(current_state)
        self.update_state(new_state, evolution_type, event_data)
    
    def set_evolution_strategy(self, strategy: StateEvolutionStrategy) -> None:
        """Set evolution strategy for this writable state."""
        self._evolution_strategy = strategy


class EquinoxStateValidator:
    """State validator for Equinox-based states."""
    
    def __init__(self, consistency_rules: StateConsistencyRules):
        self._rules = consistency_rules
    
    def validate_state(self, state: Array) -> Tuple[bool, List[str]]:
        """Validate state data."""
        errors = []
        
        if not jnp.all(jnp.isfinite(state)):
            errors.append("State contains non-finite values")
        
        if not self._rules.validate_state_bounds(state):
            errors.append("State magnitude exceeds bounds")
        
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
        
        # Validate evolution rate
        if from_valid and to_valid:
            if not self._rules.validate_evolution_rate(from_state, to_state, time_delta):
                errors.append("Evolution rate exceeds maximum allowed")
        
        return len(errors) == 0, errors
    
    def get_consistency_rules(self) -> StateConsistencyRules:
        """Get current consistency rules."""
        return self._rules


# Specialized Adapters for Different Consciousness Components

class TemporalStateAdapter(EquinoxStateAdapter):
    """Specialized adapter for temporal consciousness state management.
    
    Adapts temporal-specific operations to the general state management interface
    while providing access to temporal-specific functionality.
    """
    
    def __init__(
        self,
        initial_temporal_state: TemporalMoment,
        consistency_rules: Optional[StateConsistencyRules] = None
    ):
        # Convert TemporalMoment to Array for base adapter
        temporal_array = self._temporal_moment_to_array(initial_temporal_state)
        super().__init__(temporal_array, StateType.TEMPORAL, consistency_rules)
        
        self._current_temporal_moment = initial_temporal_state
        self._retention_buffer = initial_temporal_state.retention
        self._protention_weights = initial_temporal_state.synthesis_weights
    
    def get_retention_buffer(self) -> Array:
        """Get current retention buffer."""
        return self._retention_buffer
    
    def get_protention_horizon(self) -> Array:
        """Get protention horizon."""
        return self._current_temporal_moment.protention
    
    def get_temporal_synthesis_weights(self) -> Array:
        """Get weights for temporal synthesis."""
        return self._protention_weights
    
    def update_temporal_moment(
        self,
        new_moment: TemporalMoment,
        evolution_type: StateEvolutionType = StateEvolutionType.CONTINUOUS_FLOW
    ) -> None:
        """Update with new temporal moment."""
        # Update internal temporal tracking
        self._current_temporal_moment = new_moment
        self._retention_buffer = new_moment.retention
        self._protention_weights = new_moment.synthesis_weights
        
        # Convert to array and update base state
        temporal_array = self._temporal_moment_to_array(new_moment)
        writable_state = self.get_writable_state()
        writable_state.update_state(temporal_array, evolution_type, new_moment)
    
    def _temporal_moment_to_array(self, moment: TemporalMoment) -> Array:
        """Convert TemporalMoment to Array representation."""
        # Just use the present moment as the main state representation
        return moment.present_moment
    
    def _array_to_temporal_moment(self, array: Array) -> TemporalMoment:
        """Convert Array back to TemporalMoment structure."""
        # This would need to know the original shapes to reconstruct properly
        # For now, return current moment (in practice, would store shape metadata)
        return self._current_temporal_moment


class EmbodimentStateAdapter(EquinoxStateAdapter):
    """Specialized adapter for embodiment state management."""
    
    def __init__(
        self,
        initial_body_state: BodyState,
        consistency_rules: Optional[StateConsistencyRules] = None
    ):
        # Convert BodyState to Array
        body_array = self._body_state_to_array(initial_body_state)
        super().__init__(body_array, StateType.EMBODIMENT, consistency_rules)
        
        self._current_body_state = initial_body_state
    
    def get_proprioceptive_state(self) -> Array:
        """Get proprioceptive state."""
        return self._current_body_state.proprioception
    
    def get_motor_intentions(self) -> Array:
        """Get current motor intentions."""
        return self._current_body_state.motor_intention
    
    def get_body_schema_confidence(self) -> float:
        """Get body schema confidence level."""
        return self._current_body_state.schema_confidence
    
    def update_body_state(
        self,
        new_body_state: BodyState,
        evolution_type: StateEvolutionType = StateEvolutionType.DISCRETE_UPDATE
    ) -> None:
        """Update with new body state."""
        self._current_body_state = new_body_state
        
        body_array = self._body_state_to_array(new_body_state)
        writable_state = self.get_writable_state()
        writable_state.update_state(body_array, evolution_type, new_body_state)
    
    def _body_state_to_array(self, body_state: BodyState) -> Array:
        """Convert BodyState to Array representation."""
        return jnp.concatenate([
            body_state.proprioception.flatten(),
            body_state.motor_intention.flatten(),
            body_state.boundary_signal.flatten(),
            jnp.array([body_state.schema_confidence])
        ])


class CouplingStateAdapter(EquinoxStateAdapter):
    """Specialized adapter for structural coupling state management."""
    
    def __init__(
        self,
        initial_coupling_state: CouplingState,
        consistency_rules: Optional[StateConsistencyRules] = None
    ):
        # Convert CouplingState to Array
        coupling_array = self._coupling_state_to_array(initial_coupling_state)
        super().__init__(coupling_array, StateType.COUPLING, consistency_rules)
        
        self._current_coupling_state = initial_coupling_state
    
    def get_agent_state(self) -> Array:
        """Get agent internal state."""
        return self._current_coupling_state.agent_state
    
    def get_environmental_state(self) -> Array:
        """Get environmental state representation."""
        return self._current_coupling_state.environmental_state
    
    def get_coupling_strength(self) -> float:
        """Get current coupling strength."""
        return self._current_coupling_state.coupling_strength
    
    def update_coupling_state(
        self,
        new_coupling_state: CouplingState,
        evolution_type: StateEvolutionType = StateEvolutionType.COUPLING_DYNAMICS
    ) -> None:
        """Update with new coupling state."""
        self._current_coupling_state = new_coupling_state
        
        coupling_array = self._coupling_state_to_array(new_coupling_state)
        writable_state = self.get_writable_state()
        writable_state.update_state(coupling_array, evolution_type, new_coupling_state)
    
    def _coupling_state_to_array(self, coupling_state: CouplingState) -> Array:
        """Convert CouplingState to Array representation."""
        return jnp.concatenate([
            coupling_state.agent_state.flatten(),
            coupling_state.environmental_state.flatten(),
            coupling_state.perturbation_history.flatten(),
            jnp.array([coupling_state.coupling_strength, coupling_state.stability_metric])
        ])


class IntegratedStateAdapter:
    """Adapter for managing integrated consciousness state across multiple components.
    
    This adapter coordinates between specialized adapters to provide unified
    state management for the complete consciousness system.
    """
    
    def __init__(
        self,
        temporal_adapter: TemporalStateAdapter,
        embodiment_adapter: EmbodimentStateAdapter,
        coupling_adapter: CouplingStateAdapter
    ):
        self._temporal_adapter = temporal_adapter
        self._embodiment_adapter = embodiment_adapter
        self._coupling_adapter = coupling_adapter
        
        self._observers: List[StateChangeObserver] = []
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def get_temporal_manager(self) -> TemporalStateAdapter:
        """Get temporal state manager."""
        return self._temporal_adapter
    
    def get_embodiment_manager(self) -> EmbodimentStateAdapter:
        """Get embodiment state manager."""
        return self._embodiment_adapter
    
    def get_coupling_manager(self) -> CouplingStateAdapter:
        """Get coupling state manager."""
        return self._coupling_adapter
    
    def integrate_states(self) -> Tuple[Array, Dict[str, Any]]:
        """Integrate all component states into unified representation."""
        # Get current states from all components
        temporal_state = self._temporal_adapter.get_readable_state().get_current_state()
        embodiment_state = self._embodiment_adapter.get_readable_state().get_current_state()
        coupling_state = self._coupling_adapter.get_readable_state().get_current_state()
        
        # Simple integration strategy: concatenate all states
        integrated_state = jnp.concatenate([
            temporal_state,
            embodiment_state,
            coupling_state
        ])
        
        # Collect metadata from all components
        metadata = {
            'temporal_confidence': self._temporal_adapter._current_temporal_moment.synthesis_weights.mean(),
            'embodiment_confidence': self._embodiment_adapter.get_body_schema_confidence(),
            'coupling_strength': self._coupling_adapter.get_coupling_strength(),
            'integration_timestamp': time.time()
        }
        
        return integrated_state, metadata
    
    def synchronize_managers(self) -> None:
        """Synchronize all component state managers."""
        # This could implement cross-component consistency checks
        # and synchronization operations
        self._logger.debug("Synchronizing component state managers")
        
        # For now, just log the synchronization
        # In practice, this might involve:
        # - Cross-validation of states
        # - Temporal alignment of updates
        # - Consistency checking across components
    
    def subscribe_observer(self, observer: StateChangeObserver) -> None:
        """Subscribe observer to integrated state changes."""
        if observer not in self._observers:
            self._observers.append(observer)
            
            # Also subscribe to component adapters
            self._temporal_adapter.subscribe_observer(observer)
            self._embodiment_adapter.subscribe_observer(observer)
            self._coupling_adapter.subscribe_observer(observer)
    
    def unsubscribe_observer(self, observer: StateChangeObserver) -> None:
        """Unsubscribe observer from integrated state changes."""
        if observer in self._observers:
            self._observers.remove(observer)
            
            # Also unsubscribe from component adapters
            self._temporal_adapter.unsubscribe_observer(observer)
            self._embodiment_adapter.unsubscribe_observer(observer)
            self._coupling_adapter.unsubscribe_observer(observer)