"""State Management Interfaces - Abstract interfaces following ISP and DIP.

This module defines the interface layer of Clean Architecture, implementing
the Interface Segregation Principle (ISP) and Dependency Inversion Principle (DIP).

Key Design Principles:
- Interface Segregation: No client forced to depend on unused interfaces
- Dependency Inversion: High-level modules don't depend on low-level modules
- Single Responsibility: Each interface has one reason to change
- Open/Closed: Extensible without modification through strategy patterns
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, Union
from contextlib import contextmanager

from .state_entities import (
    StateSnapshot,
    StateEvolutionEvent, 
    StateEvolutionType,
    StateType,
    ImmutableStateContainer,
    StateConsistencyRules
)
from ..types import Array, TimeStep, PRNGKey


StateT = TypeVar('StateT')
EventT = TypeVar('EventT')
ResultT = TypeVar('ResultT')


# Core State Access Interfaces (Interface Segregation Principle)

class ReadableState(Protocol, Generic[StateT]):
    """Interface for read-only access to state.
    
    Following ISP: Clients that only need to read state are not forced
    to depend on mutation methods.
    """
    
    def get_current_state(self) -> StateT:
        """Get current state data."""
        ...
    
    def get_state_snapshot(self) -> StateSnapshot[StateT]:
        """Get current state as immutable snapshot."""
        ...
    
    def get_state_type(self) -> StateType:
        """Get the type of state managed."""
        ...
    
    def get_state_at_time(self, timestamp: TimeStep) -> Optional[StateT]:
        """Get state at specific timestamp if available."""
        ...


class WritableState(Protocol, Generic[StateT]):
    """Interface for state mutation operations.
    
    Following ISP: Separate mutation interface prevents read-only
    clients from accidentally mutating state.
    """
    
    def update_state(
        self, 
        new_state: StateT,
        evolution_type: StateEvolutionType = StateEvolutionType.DISCRETE_UPDATE,
        event_data: Any = None
    ) -> None:
        """Update state with new value."""
        ...
    
    def evolve_state_with_function(
        self,
        evolution_fn: callable[[StateT], StateT],
        evolution_type: StateEvolutionType = StateEvolutionType.CONTINUOUS_FLOW,
        event_data: Any = None
    ) -> None:
        """Evolve state using provided function."""
        ...


class StateValidator(Protocol):
    """Interface for state validation operations.
    
    Following SRP: Validation logic is separated from storage logic.
    """
    
    def validate_state(self, state: Any) -> tuple[bool, List[str]]:
        """Validate state data returns (is_valid, error_messages)."""
        ...
    
    def validate_evolution(
        self,
        from_state: Any,
        to_state: Any,
        time_delta: float
    ) -> tuple[bool, List[str]]:
        """Validate state evolution step."""
        ...
    
    def get_consistency_rules(self) -> StateConsistencyRules:
        """Get current consistency rules."""
        ...


# State Evolution Strategy Interfaces (Strategy Pattern + OCP)

class StateEvolutionStrategy(Protocol, Generic[StateT]):
    """Strategy interface for different types of state evolution.
    
    Following OCP: New evolution strategies can be added without
    modifying existing code.
    """
    
    def can_handle_evolution_type(self, evolution_type: StateEvolutionType) -> bool:
        """Check if strategy can handle given evolution type."""
        ...
    
    def evolve(
        self,
        current_state: StateT,
        evolution_input: Any,
        context: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> StateT:
        """Apply evolution strategy to current state."""
        ...
    
    def estimate_evolution_cost(
        self,
        current_state: StateT,
        evolution_input: Any
    ) -> float:
        """Estimate computational cost of evolution."""
        ...


class TemporalEvolutionStrategy(StateEvolutionStrategy[Array]):
    """Strategy for temporal consciousness state evolution."""
    
    @abc.abstractmethod
    def synthesize_temporal_moment(
        self,
        retention: Array,
        present_impression: Array,
        protention: Array,
        synthesis_weights: Array
    ) -> Array:
        """Synthesize temporal moment from components."""
        ...
    
    @abc.abstractmethod
    def update_retention_buffer(
        self,
        current_buffer: Array,
        new_experience: Array,
        decay_rate: float = 0.95
    ) -> Array:
        """Update retention buffer with new experience."""
        ...


class EmbodimentEvolutionStrategy(StateEvolutionStrategy[Array]):
    """Strategy for embodiment state evolution."""
    
    @abc.abstractmethod
    def integrate_body_schema(
        self,
        proprioceptive_input: Array,
        motor_prediction: Array,
        tactile_feedback: Array
    ) -> Array:
        """Integrate body schema from multi-modal input."""
        ...
    
    @abc.abstractmethod
    def update_motor_intentions(
        self,
        current_intentions: Array,
        sensory_feedback: Array,
        goal_state: Array
    ) -> Array:
        """Update motor intentions based on feedback."""
        ...


class CouplingEvolutionStrategy(StateEvolutionStrategy[Array]):
    """Strategy for coupling dynamics evolution."""
    
    @abc.abstractmethod
    def compute_coupling_dynamics(
        self,
        agent_state: Array,
        environmental_perturbation: Array,
        coupling_strength: float
    ) -> Array:
        """Compute coupling dynamics between agent and environment."""
        ...
    
    @abc.abstractmethod
    def assess_coupling_stability(
        self,
        coupling_history: List[Array],
        stability_window: int = 10
    ) -> float:
        """Assess stability of coupling over time."""
        ...


# State Management Coordination Interfaces

class StateManager(Protocol, Generic[StateT]):
    """Main interface for state management operations.
    
    Following DIP: High-level state coordination depends on
    abstractions rather than concrete implementations.
    """
    
    def get_readable_state(self) -> ReadableState[StateT]:
        """Get read-only interface to state."""
        ...
    
    def get_writable_state(self) -> WritableState[StateT]:
        """Get writable interface to state."""
        ...
    
    def get_validator(self) -> StateValidator:
        """Get state validator."""
        ...
    
    def set_evolution_strategy(self, strategy: StateEvolutionStrategy[StateT]) -> None:
        """Set strategy for state evolution."""
        ...
    
    def get_evolution_history(self) -> List[StateEvolutionEvent]:
        """Get history of state evolution events."""
        ...
    
    @contextmanager
    def transaction_context(self):
        """Context manager for transactional state operations."""
        ...
    
    def create_checkpoint(self) -> str:
        """Create state checkpoint, return checkpoint ID."""
        ...
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from checkpoint."""
        ...


class StateThreadingCoordinator(Protocol):
    """Interface for coordinating state flow through processing pipelines.
    
    Following SRP: Threading coordination is separated from state management.
    """
    
    def thread_state_through_pipeline(
        self,
        initial_state: Any,
        processing_steps: List[callable],
        context: Dict[str, Any]
    ) -> tuple[Any, List[StateEvolutionEvent]]:
        """Thread state through processing pipeline."""
        ...
    
    def parallel_state_evolution(
        self,
        state_managers: List[StateManager],
        evolution_inputs: List[Any],
        synchronization_points: List[int] = None
    ) -> List[StateEvolutionEvent]:
        """Coordinate parallel state evolution across managers."""
        ...
    
    def merge_state_branches(
        self,
        branch_states: List[Any],
        merge_strategy: str = "weighted_average"
    ) -> Any:
        """Merge multiple state branches into unified state."""
        ...


# Specialized State Interfaces for Different Consciousness Components

class TemporalStateManager(StateManager[Array]):
    """Specialized interface for temporal consciousness state."""
    
    @abc.abstractmethod
    def get_retention_buffer(self) -> Array:
        """Get current retention buffer."""
        ...
    
    @abc.abstractmethod
    def get_protention_horizon(self) -> Array:
        """Get protention horizon."""
        ...
    
    @abc.abstractmethod
    def get_temporal_synthesis_weights(self) -> Array:
        """Get weights for temporal synthesis."""
        ...


class EmbodimentStateManager(StateManager[Array]):
    """Specialized interface for embodiment state."""
    
    @abc.abstractmethod
    def get_proprioceptive_state(self) -> Array:
        """Get proprioceptive state."""
        ...
    
    @abc.abstractmethod
    def get_motor_intentions(self) -> Array:
        """Get current motor intentions."""
        ...
    
    @abc.abstractmethod
    def get_body_schema_confidence(self) -> float:
        """Get body schema confidence level."""
        ...


class CouplingStateManager(StateManager[Array]):
    """Specialized interface for structural coupling state."""
    
    @abc.abstractmethod
    def get_agent_state(self) -> Array:
        """Get agent internal state."""
        ...
    
    @abc.abstractmethod
    def get_environmental_state(self) -> Array:
        """Get environmental state representation."""
        ...
    
    @abc.abstractmethod
    def get_coupling_strength(self) -> float:
        """Get current coupling strength."""
        ...


class IntegratedStateManager(Protocol):
    """Interface for managing integrated consciousness state.
    
    This coordinates multiple specialized state managers.
    """
    
    def get_temporal_manager(self) -> TemporalStateManager:
        """Get temporal state manager."""
        ...
    
    def get_embodiment_manager(self) -> EmbodimentStateManager:
        """Get embodiment state manager."""
        ...
    
    def get_coupling_manager(self) -> CouplingStateManager:
        """Get coupling state manager."""
        ...
    
    def integrate_states(self) -> tuple[Array, Dict[str, Any]]:
        """Integrate all component states into unified representation."""
        ...
    
    def synchronize_managers(self) -> None:
        """Synchronize all component state managers."""
        ...


# Factory Interfaces (Abstract Factory Pattern)

class StateManagerFactory(Protocol):
    """Factory interface for creating state managers.
    
    Following DIP: Clients depend on factory abstraction,
    not concrete factory implementations.
    """
    
    def create_temporal_manager(
        self,
        initial_state: Array,
        config: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> TemporalStateManager:
        """Create temporal state manager."""
        ...
    
    def create_embodiment_manager(
        self,
        initial_state: Array,
        config: Dict[str, Any], 
        key: Optional[PRNGKey] = None
    ) -> EmbodimentStateManager:
        """Create embodiment state manager."""
        ...
    
    def create_coupling_manager(
        self,
        initial_agent_state: Array,
        initial_env_state: Array,
        config: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> CouplingStateManager:
        """Create coupling state manager."""
        ...
    
    def create_integrated_manager(
        self,
        managers: Dict[str, StateManager],
        integration_config: Dict[str, Any]
    ) -> IntegratedStateManager:
        """Create integrated state manager from component managers."""
        ...


# Observer Interfaces (Observer Pattern)

class StateChangeObserver(Protocol):
    """Observer interface for state change notifications.
    
    Following OCP: New observers can be added without modifying
    existing state management code.
    """
    
    def on_state_changed(
        self,
        previous_state: Any,
        new_state: Any, 
        evolution_event: StateEvolutionEvent
    ) -> None:
        """Called when state changes."""
        ...
    
    def on_state_validation_failed(
        self,
        invalid_state: Any,
        validation_errors: List[str]
    ) -> None:
        """Called when state validation fails."""
        ...


class StatePublisher(Protocol):
    """Interface for publishing state changes.
    
    Following SRP: Publishing concerns separated from state management.
    """
    
    def subscribe_observer(self, observer: StateChangeObserver) -> None:
        """Subscribe observer to state changes."""
        ...
    
    def unsubscribe_observer(self, observer: StateChangeObserver) -> None:
        """Unsubscribe observer from state changes."""
        ...
    
    def notify_state_change(
        self,
        previous_state: Any,
        new_state: Any,
        evolution_event: StateEvolutionEvent
    ) -> None:
        """Notify all observers of state change."""
        ...


# Configuration and Monitoring Interfaces

class StateMonitor(Protocol):
    """Interface for monitoring state system health."""
    
    def get_state_metrics(self) -> Dict[str, float]:
        """Get current state system metrics."""
        ...
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get statistics about state evolution patterns."""
        ...
    
    def get_validation_summary(self) -> Dict[str, int]:
        """Get summary of state validation results."""
        ...


class StateConfiguration(Protocol):
    """Interface for runtime state system configuration."""
    
    def update_consistency_rules(self, rules: StateConsistencyRules) -> None:
        """Update consistency rules at runtime."""
        ...
    
    def set_evolution_strategy(
        self, 
        state_type: StateType,
        strategy: StateEvolutionStrategy
    ) -> None:
        """Set evolution strategy for specific state type."""
        ...
    
    def configure_threading_behavior(self, config: Dict[str, Any]) -> None:
        """Configure state threading behavior."""
        ...