"""State Management Use Cases - Business logic orchestration layer.

This module implements the Use Cases layer of Clean Architecture, containing
application-specific business rules for state management operations.

Key Design Principles:
- Single Responsibility: Each use case handles one specific business operation
- Dependency Inversion: Use cases depend on interfaces, not implementations
- Open/Closed: New use cases can be added without modifying existing ones
- Interface Segregation: Use cases only depend on interfaces they need

Use Cases represent the intention of the application and orchestrate
the flow of data to and from entities while containing application-specific
business rules.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional, Generic, TypeVar, Union
from dataclasses import dataclass
from contextlib import contextmanager
import time

from .state_entities import (
    StateSnapshot,
    StateEvolutionEvent,
    StateEvolutionType,
    StateType,
    ImmutableStateContainer,
    StateConsistencyRules,
    StateConsistencyValidator
)
from .state_interfaces import (
    StateManager,
    ReadableState,
    WritableState,
    StateEvolutionStrategy,
    StateValidator,
    StateThreadingCoordinator,
    StateChangeObserver,
    StatePublisher
)
from ..types import Array, TimeStep, PRNGKey


logger = logging.getLogger(__name__)

StateT = TypeVar('StateT')
ResultT = TypeVar('ResultT')


@dataclass
class StateOperationResult(Generic[ResultT]):
    """Result of a state operation with metadata."""
    
    success: bool
    result: Optional[ResultT] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    validation_errors: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.validation_errors is None:
            self.validation_errors = []
        if self.metadata is None:
            self.metadata = {}
    
    def is_valid(self) -> bool:
        """Check if result is valid (success with no validation errors)."""
        return self.success and len(self.validation_errors) == 0
    
    def with_metadata(self, **kwargs) -> StateOperationResult[ResultT]:
        """Add metadata to result."""
        new_metadata = {**self.metadata, **kwargs}
        return StateOperationResult(
            success=self.success,
            result=self.result,
            error_message=self.error_message,
            execution_time_ms=self.execution_time_ms,
            validation_errors=self.validation_errors,
            metadata=new_metadata
        )


class BaseStateUseCase(Generic[StateT]):
    """Base class for state use cases providing common functionality.
    
    Implements Template Method pattern for consistent error handling
    and logging across all use cases.
    """
    
    def __init__(
        self,
        state_manager: StateManager[StateT],
        validator: Optional[StateValidator] = None,
        publisher: Optional[StatePublisher] = None
    ):
        self._state_manager = state_manager
        self._validator = validator or state_manager.get_validator()
        self._publisher = publisher
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @contextmanager
    def _execution_context(self, operation_name: str):
        """Context manager for use case execution with timing and error handling."""
        start_time = time.time()
        self._logger.debug(f"Starting {operation_name}")
        
        try:
            yield
        except Exception as e:
            self._logger.error(f"Error in {operation_name}: {str(e)}", exc_info=True)
            raise
        finally:
            execution_time = (time.time() - start_time) * 1000
            self._logger.debug(f"Completed {operation_name} in {execution_time:.2f}ms")
    
    def _validate_and_notify(
        self,
        previous_state: StateT,
        new_state: StateT,
        evolution_event: StateEvolutionEvent
    ) -> List[str]:
        """Validate state change and notify observers."""
        # Validate the new state
        is_valid, validation_errors = self._validator.validate_state(new_state)
        
        if not is_valid:
            self._logger.warning(f"State validation failed: {validation_errors}")
            if self._publisher:
                self._publisher.notify_state_change(
                    previous_state, new_state, evolution_event
                )
            return validation_errors
        
        # Notify observers of successful state change
        if self._publisher:
            self._publisher.notify_state_change(
                previous_state, new_state, evolution_event
            )
        
        return []


class StateEvolutionUseCase(BaseStateUseCase[StateT]):
    """Use case for evolving state through various strategies.
    
    This use case orchestrates the business logic for applying
    state evolution while maintaining consistency and validation.
    """
    
    def __init__(
        self,
        state_manager: StateManager[StateT],
        evolution_strategy: StateEvolutionStrategy[StateT],
        validator: Optional[StateValidator] = None,
        publisher: Optional[StatePublisher] = None
    ):
        super().__init__(state_manager, validator, publisher)
        self._evolution_strategy = evolution_strategy
    
    def evolve_with_input(
        self,
        evolution_input: Any,
        evolution_type: StateEvolutionType = StateEvolutionType.DISCRETE_UPDATE,
        context: Optional[Dict[str, Any]] = None,
        key: Optional[PRNGKey] = None
    ) -> StateOperationResult[StateT]:
        """Evolve state with given input using configured strategy."""
        with self._execution_context("evolve_with_input"):
            try:
                # Get current state
                readable_state = self._state_manager.get_readable_state()
                current_state = readable_state.get_current_state()
                
                # Validate that strategy can handle evolution type
                if not self._evolution_strategy.can_handle_evolution_type(evolution_type):
                    return StateOperationResult(
                        success=False,
                        error_message=f"Strategy cannot handle evolution type: {evolution_type}"
                    )
                
                # Apply evolution strategy
                context = context or {}
                new_state = self._evolution_strategy.evolve(
                    current_state,
                    evolution_input, 
                    context,
                    key
                )
                
                # Validate evolution step
                time_delta = context.get('time_delta', 0.1)
                is_valid, validation_errors = self._validator.validate_evolution(
                    current_state, new_state, time_delta
                )
                
                if not is_valid:
                    return StateOperationResult(
                        success=False,
                        error_message="Evolution validation failed",
                        validation_errors=validation_errors
                    )
                
                # Update state through manager
                writable_state = self._state_manager.get_writable_state()
                writable_state.update_state(new_state, evolution_type, evolution_input)
                
                # Validate and notify
                validation_errors = self._validate_and_notify(
                    current_state,
                    new_state,
                    StateEvolutionEvent(
                        evolution_type=evolution_type,
                        source_state_id="current",
                        target_state_id="new",
                        event_data=evolution_input
                    )
                )
                
                return StateOperationResult(
                    success=True,
                    result=new_state,
                    validation_errors=validation_errors,
                    metadata={
                        'evolution_type': evolution_type,
                        'strategy_cost': self._evolution_strategy.estimate_evolution_cost(
                            current_state, evolution_input
                        )
                    }
                )
                
            except Exception as e:
                return StateOperationResult(
                    success=False,
                    error_message=str(e)
                )
    
    def evolve_with_function(
        self,
        evolution_function: callable[[StateT], StateT],
        evolution_type: StateEvolutionType = StateEvolutionType.CONTINUOUS_FLOW,
        context: Optional[Dict[str, Any]] = None
    ) -> StateOperationResult[StateT]:
        """Evolve state using provided function."""
        with self._execution_context("evolve_with_function"):
            try:
                # Get current state
                readable_state = self._state_manager.get_readable_state()
                current_state = readable_state.get_current_state()
                
                # Apply evolution function
                new_state = evolution_function(current_state)
                
                # Validate evolution
                time_delta = context.get('time_delta', 0.1) if context else 0.1
                is_valid, validation_errors = self._validator.validate_evolution(
                    current_state, new_state, time_delta
                )
                
                if not is_valid:
                    return StateOperationResult(
                        success=False,
                        error_message="Function evolution validation failed",
                        validation_errors=validation_errors
                    )
                
                # Update state
                writable_state = self._state_manager.get_writable_state()
                writable_state.evolve_state_with_function(
                    evolution_function, evolution_type, context
                )
                
                return StateOperationResult(
                    success=True,
                    result=new_state,
                    validation_errors=validation_errors,
                    metadata={'evolution_type': evolution_type}
                )
                
            except Exception as e:
                return StateOperationResult(
                    success=False,
                    error_message=str(e)
                )


class StateConsistencyUseCase(BaseStateUseCase[StateT]):
    """Use case for maintaining and validating state consistency.
    
    This use case implements business rules around state consistency
    and provides operations for consistency checking and repair.
    """
    
    def __init__(
        self,
        state_manager: StateManager[StateT],
        consistency_validator: StateConsistencyValidator,
        validator: Optional[StateValidator] = None
    ):
        super().__init__(state_manager, validator)
        self._consistency_validator = consistency_validator
    
    def validate_current_state(self) -> StateOperationResult[Dict[str, Any]]:
        """Validate current state against consistency rules."""
        with self._execution_context("validate_current_state"):
            try:
                readable_state = self._state_manager.get_readable_state()
                current_state = readable_state.get_current_state()
                
                # Validate state data
                is_valid, state_errors = self._validator.validate_state(current_state)
                
                # Validate evolution history if available
                evolution_history = self._state_manager.get_evolution_history()
                history_valid, history_errors = self._consistency_validator.validate_evolution_sequence(
                    evolution_history
                )
                
                all_errors = state_errors + history_errors
                overall_valid = is_valid and history_valid
                
                validation_report = {
                    'state_valid': is_valid,
                    'history_valid': history_valid,
                    'overall_valid': overall_valid,
                    'error_count': len(all_errors),
                    'state_errors': state_errors,
                    'history_errors': history_errors
                }
                
                return StateOperationResult(
                    success=True,
                    result=validation_report,
                    validation_errors=all_errors
                )
                
            except Exception as e:
                return StateOperationResult(
                    success=False,
                    error_message=str(e)
                )
    
    def repair_inconsistent_state(
        self,
        repair_strategy: str = "reset_to_valid"
    ) -> StateOperationResult[StateT]:
        """Attempt to repair inconsistent state."""
        with self._execution_context("repair_inconsistent_state"):
            try:
                # First validate current state
                validation_result = self.validate_current_state()
                
                if validation_result.result['overall_valid']:
                    return StateOperationResult(
                        success=True,
                        result=self._state_manager.get_readable_state().get_current_state(),
                        error_message="State is already consistent"
                    )
                
                # Apply repair strategy
                if repair_strategy == "reset_to_valid":
                    # Reset to a known valid state
                    rules = self._validator.get_consistency_rules()
                    valid_state = self._create_minimal_valid_state(rules)
                    
                    writable_state = self._state_manager.get_writable_state()
                    writable_state.update_state(
                        valid_state,
                        StateEvolutionType.DISCRETE_UPDATE,
                        "consistency_repair"
                    )
                    
                    return StateOperationResult(
                        success=True,
                        result=valid_state,
                        metadata={'repair_strategy': repair_strategy}
                    )
                
                else:
                    return StateOperationResult(
                        success=False,
                        error_message=f"Unknown repair strategy: {repair_strategy}"
                    )
                    
            except Exception as e:
                return StateOperationResult(
                    success=False,
                    error_message=str(e)
                )
    
    def _create_minimal_valid_state(self, rules: StateConsistencyRules) -> StateT:
        """Create minimal valid state according to consistency rules."""
        # This is a simplified implementation
        # In practice, this would depend on the specific state type
        import jax.numpy as jnp
        return jnp.zeros(10)  # Placeholder implementation


class StateSnapshotUseCase(BaseStateUseCase[StateT]):
    """Use case for creating and managing state snapshots.
    
    This use case implements business logic for state checkpointing,
    allowing for state recovery and temporal analysis.
    """
    
    def __init__(
        self,
        state_manager: StateManager[StateT],
        max_snapshots: int = 10,
        validator: Optional[StateValidator] = None
    ):
        super().__init__(state_manager, validator)
        self._max_snapshots = max_snapshots
        self._snapshots: Dict[str, StateSnapshot[StateT]] = {}
    
    def create_snapshot(
        self, 
        snapshot_name: Optional[str] = None
    ) -> StateOperationResult[str]:
        """Create snapshot of current state."""
        with self._execution_context("create_snapshot"):
            try:
                readable_state = self._state_manager.get_readable_state()
                snapshot = readable_state.get_state_snapshot()
                
                # Generate snapshot name if not provided
                if snapshot_name is None:
                    snapshot_name = f"snapshot_{int(time.time())}"
                
                # Store snapshot (implement LRU eviction if needed)
                if len(self._snapshots) >= self._max_snapshots:
                    oldest_key = min(self._snapshots.keys(), 
                                   key=lambda k: self._snapshots[k].timestamp)
                    del self._snapshots[oldest_key]
                
                self._snapshots[snapshot_name] = snapshot
                
                return StateOperationResult(
                    success=True,
                    result=snapshot_name,
                    metadata={
                        'snapshot_id': snapshot.snapshot_id,
                        'timestamp': snapshot.timestamp,
                        'total_snapshots': len(self._snapshots)
                    }
                )
                
            except Exception as e:
                return StateOperationResult(
                    success=False,
                    error_message=str(e)
                )
    
    def restore_snapshot(self, snapshot_name: str) -> StateOperationResult[StateT]:
        """Restore state from snapshot."""
        with self._execution_context("restore_snapshot"):
            try:
                if snapshot_name not in self._snapshots:
                    return StateOperationResult(
                        success=False,
                        error_message=f"Snapshot '{snapshot_name}' not found"
                    )
                
                snapshot = self._snapshots[snapshot_name]
                
                # Validate snapshot data before restoring
                is_valid, validation_errors = self._validator.validate_state(snapshot.data)
                
                if not is_valid:
                    return StateOperationResult(
                        success=False,
                        error_message="Snapshot data is invalid",
                        validation_errors=validation_errors
                    )
                
                # Restore state
                writable_state = self._state_manager.get_writable_state()
                writable_state.update_state(
                    snapshot.data,
                    StateEvolutionType.DISCRETE_UPDATE,
                    f"restored_from_{snapshot_name}"
                )
                
                return StateOperationResult(
                    success=True,
                    result=snapshot.data,
                    metadata={
                        'restored_from': snapshot_name,
                        'snapshot_timestamp': snapshot.timestamp
                    }
                )
                
            except Exception as e:
                return StateOperationResult(
                    success=False,
                    error_message=str(e)
                )
    
    def list_snapshots(self) -> StateOperationResult[List[Dict[str, Any]]]:
        """List all available snapshots."""
        snapshot_info = []
        for name, snapshot in self._snapshots.items():
            snapshot_info.append({
                'name': name,
                'snapshot_id': snapshot.snapshot_id,
                'timestamp': snapshot.timestamp,
                'state_type': snapshot.state_type.value,
                'metadata_keys': list(snapshot.metadata.keys())
            })
        
        return StateOperationResult(
            success=True,
            result=snapshot_info
        )


class StateRecoveryUseCase(BaseStateUseCase[StateT]):
    """Use case for state recovery and error handling.
    
    This use case implements business logic for recovering from
    state corruption or invalid state transitions.
    """
    
    def __init__(
        self,
        state_manager: StateManager[StateT],
        fallback_state_factory: callable[[], StateT],
        validator: Optional[StateValidator] = None
    ):
        super().__init__(state_manager, validator)
        self._fallback_state_factory = fallback_state_factory
        self._recovery_history: List[Dict[str, Any]] = []
    
    def attempt_automatic_recovery(self) -> StateOperationResult[StateT]:
        """Attempt automatic recovery from invalid state."""
        with self._execution_context("attempt_automatic_recovery"):
            try:
                # Validate current state
                readable_state = self._state_manager.get_readable_state()
                current_state = readable_state.get_current_state()
                is_valid, validation_errors = self._validator.validate_state(current_state)
                
                if is_valid:
                    return StateOperationResult(
                        success=True,
                        result=current_state,
                        error_message="State is already valid"
                    )
                
                recovery_attempts = []
                
                # Attempt 1: Try to restore from most recent checkpoint
                checkpoint_result = self._try_checkpoint_recovery()
                recovery_attempts.append(checkpoint_result)
                
                if checkpoint_result.success:
                    return checkpoint_result
                
                # Attempt 2: Try to repair current state
                repair_result = self._try_state_repair(current_state, validation_errors)
                recovery_attempts.append(repair_result)
                
                if repair_result.success:
                    return repair_result
                
                # Attempt 3: Fall back to known valid state
                fallback_result = self._try_fallback_recovery()
                recovery_attempts.append(fallback_result)
                
                # Log recovery attempt
                self._recovery_history.append({
                    'timestamp': time.time(),
                    'original_errors': validation_errors,
                    'attempts': len(recovery_attempts),
                    'successful': fallback_result.success
                })
                
                return fallback_result
                
            except Exception as e:
                return StateOperationResult(
                    success=False,
                    error_message=str(e)
                )
    
    def _try_checkpoint_recovery(self) -> StateOperationResult[StateT]:
        """Try to recover using state manager checkpoints."""
        try:
            # This would interact with the state manager's checkpoint system
            # For now, return failure as checkpoints aren't implemented
            return StateOperationResult(
                success=False,
                error_message="No valid checkpoints available"
            )
        except Exception as e:
            return StateOperationResult(
                success=False,
                error_message=f"Checkpoint recovery failed: {str(e)}"
            )
    
    def _try_state_repair(
        self,
        invalid_state: StateT,
        validation_errors: List[str]
    ) -> StateOperationResult[StateT]:
        """Try to repair the invalid state."""
        try:
            # Simple repair strategy: clip values to valid range
            import jax.numpy as jnp
            
            if isinstance(invalid_state, jnp.ndarray):
                # Clip to reasonable range and replace non-finite values
                repaired_state = jnp.where(
                    jnp.isfinite(invalid_state),
                    jnp.clip(invalid_state, -10.0, 10.0),
                    0.0
                )
                
                # Validate repaired state
                is_valid, _ = self._validator.validate_state(repaired_state)
                
                if is_valid:
                    writable_state = self._state_manager.get_writable_state()
                    writable_state.update_state(
                        repaired_state,
                        StateEvolutionType.DISCRETE_UPDATE,
                        "automatic_repair"
                    )
                    
                    return StateOperationResult(
                        success=True,
                        result=repaired_state,
                        metadata={'repair_method': 'clip_and_replace_nan'}
                    )
            
            return StateOperationResult(
                success=False,
                error_message="Unable to repair state"
            )
            
        except Exception as e:
            return StateOperationResult(
                success=False,
                error_message=f"State repair failed: {str(e)}"
            )
    
    def _try_fallback_recovery(self) -> StateOperationResult[StateT]:
        """Try to recover using fallback state."""
        try:
            fallback_state = self._fallback_state_factory()
            
            # Validate fallback state
            is_valid, validation_errors = self._validator.validate_state(fallback_state)
            
            if not is_valid:
                return StateOperationResult(
                    success=False,
                    error_message="Fallback state is also invalid",
                    validation_errors=validation_errors
                )
            
            # Apply fallback state
            writable_state = self._state_manager.get_writable_state()
            writable_state.update_state(
                fallback_state,
                StateEvolutionType.DISCRETE_UPDATE,
                "fallback_recovery"
            )
            
            return StateOperationResult(
                success=True,
                result=fallback_state,
                metadata={'recovery_method': 'fallback'}
            )
            
        except Exception as e:
            return StateOperationResult(
                success=False,
                error_message=f"Fallback recovery failed: {str(e)}"
            )
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """Get history of recovery attempts."""
        return list(self._recovery_history)  # Defensive copy


class StateThreadingUseCase:
    """Use case for coordinating state flow through processing pipelines.
    
    This use case orchestrates the business logic for threading state
    through complex processing pipelines while maintaining consistency.
    """
    
    def __init__(
        self,
        threading_coordinator: StateThreadingCoordinator,
        validator: StateValidator
    ):
        self._coordinator = threading_coordinator
        self._validator = validator
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def process_sequential_pipeline(
        self,
        initial_state: Any,
        processing_steps: List[callable],
        step_contexts: Optional[List[Dict[str, Any]]] = None
    ) -> StateOperationResult[tuple[Any, List[StateEvolutionEvent]]]:
        """Process state through sequential pipeline."""
        try:
            if step_contexts is None:
                step_contexts = [{}] * len(processing_steps)
            
            # Thread state through pipeline
            final_state, evolution_events = self._coordinator.thread_state_through_pipeline(
                initial_state,
                processing_steps,
                {'step_contexts': step_contexts}
            )
            
            # Validate final state
            is_valid, validation_errors = self._validator.validate_state(final_state)
            
            return StateOperationResult(
                success=is_valid,
                result=(final_state, evolution_events),
                validation_errors=validation_errors,
                metadata={
                    'steps_processed': len(processing_steps),
                    'events_generated': len(evolution_events)
                }
            )
            
        except Exception as e:
            return StateOperationResult(
                success=False,
                error_message=str(e)
            )