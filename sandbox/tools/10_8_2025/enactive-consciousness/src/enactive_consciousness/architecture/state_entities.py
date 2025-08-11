"""State Entities - Core domain objects for state management.

This module defines the core entities representing state in the enactive consciousness
system, following Uncle Bob's Entity layer principles. These are the most stable
parts of the architecture and contain enterprise business rules.

Design Principles:
- Entities encapsulate the most general and high-level rules
- They are least likely to change when external factors change
- They know nothing about databases, UIs, or frameworks
- They embody critical business rules that would exist regardless of automation
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar, Union, List
from enum import Enum
import time
import uuid

import jax
import jax.numpy as jnp
from ..types import Array, TimeStep


StateT = TypeVar('StateT')
EventT = TypeVar('EventT')


class StateType(Enum):
    """Types of state in the consciousness system."""
    TEMPORAL = "temporal"
    EMBODIMENT = "embodiment" 
    COUPLING = "coupling"
    EXPERIENTIAL = "experiential"
    INTEGRATED = "integrated"


class StateEvolutionType(Enum):
    """Types of state evolution patterns."""
    CONTINUOUS_FLOW = "continuous_flow"
    DISCRETE_UPDATE = "discrete_update"
    RETENTION_DECAY = "retention_decay"
    COUPLING_DYNAMICS = "coupling_dynamics"
    SYNTHESIS_INTEGRATION = "synthesis_integration"


@dataclass(frozen=True)
class StateSnapshot(Generic[StateT]):
    """Immutable snapshot of system state at a specific moment.
    
    This entity represents a point-in-time capture of state data,
    following the principle that business entities should be
    immutable to prevent accidental state corruption.
    """
    
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: TimeStep = field(default_factory=time.time)
    state_type: StateType = StateType.INTEGRATED
    data: StateT = None
    consistency_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate snapshot integrity."""
        if self.data is None:
            raise ValueError("StateSnapshot cannot have None data")
        
        if self.timestamp < 0:
            raise ValueError("StateSnapshot timestamp must be non-negative")
    
    def with_metadata(self, **kwargs) -> StateSnapshot[StateT]:
        """Create new snapshot with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return StateSnapshot(
            snapshot_id=self.snapshot_id,
            timestamp=self.timestamp,
            state_type=self.state_type,
            data=self.data,
            consistency_hash=self.consistency_hash,
            metadata=new_metadata
        )
    
    def is_consistent_with(self, other: StateSnapshot[StateT]) -> bool:
        """Check if this snapshot is consistent with another."""
        if self.consistency_hash is None or other.consistency_hash is None:
            return True  # Cannot verify consistency without hashes
        
        return (
            self.state_type == other.state_type and
            self.consistency_hash == other.consistency_hash
        )


@dataclass(frozen=True)
class StateEvolutionEvent(Generic[EventT]):
    """Event representing a state evolution step.
    
    This entity captures the business rules around how state changes
    occur in the consciousness system. It represents the "what happened"
    rather than the "how it's implemented".
    """
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: TimeStep = field(default_factory=time.time)
    evolution_type: StateEvolutionType = StateEvolutionType.DISCRETE_UPDATE
    source_state_id: str = ""
    target_state_id: str = ""
    event_data: EventT = None
    causality_chain: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate event consistency."""
        if not self.source_state_id:
            raise ValueError("StateEvolutionEvent must have source_state_id")
        
        if not self.target_state_id:
            raise ValueError("StateEvolutionEvent must have target_state_id")
        
        if self.event_data is None:
            raise ValueError("StateEvolutionEvent cannot have None event_data")
    
    def extends_causality_chain(self, previous_event_id: str) -> StateEvolutionEvent[EventT]:
        """Create new event extending the causality chain."""
        new_chain = self.causality_chain + [previous_event_id]
        return StateEvolutionEvent(
            event_id=self.event_id,
            timestamp=self.timestamp,
            evolution_type=self.evolution_type,
            source_state_id=self.source_state_id,
            target_state_id=self.target_state_id,
            event_data=self.event_data,
            causality_chain=new_chain
        )
    
    def is_causal_successor(self, other: StateEvolutionEvent[EventT]) -> bool:
        """Check if this event is a causal successor of another."""
        return (
            other.target_state_id == self.source_state_id and
            other.timestamp <= self.timestamp
        )


@dataclass(frozen=True)
class StateConsistencyRules:
    """Business rules for state consistency validation.
    
    This entity encapsulates the invariants that must be maintained
    across all state transitions in the consciousness system.
    """
    
    max_state_magnitude: float = 1000.0
    min_temporal_interval: float = 0.001  # Minimum time between updates
    max_evolution_rate: float = 10.0  # Maximum rate of change
    require_finite_values: bool = True
    require_causal_ordering: bool = True
    consistency_tolerance: float = 1e-6
    
    def validate_state_bounds(self, state_data: Array) -> bool:
        """Validate state data is within acceptable bounds."""
        if self.require_finite_values and not jnp.all(jnp.isfinite(state_data)):
            return False
        
        magnitude = jnp.max(jnp.abs(state_data))
        return magnitude <= self.max_state_magnitude
    
    def validate_temporal_ordering(
        self, 
        previous_timestamp: TimeStep, 
        current_timestamp: TimeStep
    ) -> bool:
        """Validate temporal ordering constraints."""
        time_diff = current_timestamp - previous_timestamp
        
        if self.require_causal_ordering and time_diff < 0:
            return False
        
        return time_diff >= self.min_temporal_interval
    
    def validate_evolution_rate(
        self,
        previous_state: Array,
        current_state: Array, 
        time_delta: float
    ) -> bool:
        """Validate rate of state evolution."""
        if time_delta <= 0:
            return False
        
        state_change = jnp.linalg.norm(current_state - previous_state)
        evolution_rate = state_change / time_delta
        
        return evolution_rate <= self.max_evolution_rate
    
    def with_updated_bounds(self, **kwargs) -> StateConsistencyRules:
        """Create new rules with updated bounds."""
        current_dict = {
            'max_state_magnitude': self.max_state_magnitude,
            'min_temporal_interval': self.min_temporal_interval,
            'max_evolution_rate': self.max_evolution_rate,
            'require_finite_values': self.require_finite_values,
            'require_causal_ordering': self.require_causal_ordering,
            'consistency_tolerance': self.consistency_tolerance,
        }
        current_dict.update(kwargs)
        return StateConsistencyRules(**current_dict)


class ImmutableStateContainer(Generic[StateT]):
    """Immutable container for state data with evolution tracking.
    
    This entity provides the core abstraction for state storage
    while maintaining immutability and tracking state evolution.
    It embodies the business rule that consciousness state should
    be append-only and never destructively modified.
    """
    
    def __init__(
        self,
        initial_state: StateT,
        state_type: StateType,
        consistency_rules: Optional[StateConsistencyRules] = None
    ):
        """Initialize immutable state container."""
        self._current_snapshot = StateSnapshot(
            state_type=state_type,
            data=initial_state,
            timestamp=time.time()
        )
        self._evolution_history: List[StateEvolutionEvent] = []
        self._consistency_rules = consistency_rules or StateConsistencyRules()
        
        # Validate initial state
        if isinstance(initial_state, jnp.ndarray):
            if not self._consistency_rules.validate_state_bounds(initial_state):
                raise ValueError("Initial state violates consistency rules")
    
    @property
    def current_state(self) -> StateT:
        """Get current state data (read-only access)."""
        return self._current_snapshot.data
    
    @property 
    def current_snapshot(self) -> StateSnapshot[StateT]:
        """Get current state snapshot."""
        return self._current_snapshot
    
    @property
    def state_type(self) -> StateType:
        """Get the type of state this container holds."""
        return self._current_snapshot.state_type
    
    @property
    def evolution_history(self) -> List[StateEvolutionEvent]:
        """Get read-only view of evolution history."""
        return list(self._evolution_history)  # Defensive copy
    
    def evolve_state(
        self, 
        new_state: StateT,
        evolution_type: StateEvolutionType,
        event_data: Any = None
    ) -> ImmutableStateContainer[StateT]:
        """Create new container with evolved state.
        
        This method embodies the business rule that state evolution
        must be immutable and traceable.
        """
        current_time = time.time()
        
        # Validate evolution against business rules
        if isinstance(new_state, jnp.ndarray):
            if not self._consistency_rules.validate_state_bounds(new_state):
                raise ValueError("New state violates consistency bounds")
            
            if isinstance(self.current_state, jnp.ndarray):
                time_delta = current_time - self._current_snapshot.timestamp
                if not self._consistency_rules.validate_evolution_rate(
                    self.current_state, new_state, time_delta
                ):
                    raise ValueError("State evolution rate exceeds maximum allowed")
        
        # Create new snapshot
        new_snapshot = StateSnapshot(
            state_type=self._current_snapshot.state_type,
            data=new_state,
            timestamp=current_time
        )
        
        # Create evolution event
        evolution_event = StateEvolutionEvent(
            evolution_type=evolution_type,
            source_state_id=self._current_snapshot.snapshot_id,
            target_state_id=new_snapshot.snapshot_id,
            event_data=event_data,
            timestamp=current_time
        )
        
        # Create new container with evolved state
        new_container = ImmutableStateContainer(
            new_state,
            self._current_snapshot.state_type,
            self._consistency_rules
        )
        new_container._current_snapshot = new_snapshot
        new_container._evolution_history = self._evolution_history + [evolution_event]
        
        return new_container
    
    def snapshot_at_time(self, target_time: TimeStep) -> Optional[StateSnapshot[StateT]]:
        """Get snapshot closest to target time."""
        # For now, return current snapshot
        # In a full implementation, this would search history
        if abs(self._current_snapshot.timestamp - target_time) < 0.1:
            return self._current_snapshot
        return None
    
    def can_evolve_to(self, new_state: StateT) -> bool:
        """Check if evolution to new state is valid."""
        if isinstance(new_state, jnp.ndarray):
            return self._consistency_rules.validate_state_bounds(new_state)
        return True
    
    def with_consistency_rules(
        self, 
        rules: StateConsistencyRules
    ) -> ImmutableStateContainer[StateT]:
        """Create new container with different consistency rules."""
        new_container = ImmutableStateContainer(
            self.current_state,
            self.state_type,
            rules
        )
        new_container._current_snapshot = self._current_snapshot
        new_container._evolution_history = self._evolution_history
        return new_container


# Domain Services for Entity Operations

class StateConsistencyValidator:
    """Domain service for validating state consistency.
    
    This service encapsulates the business logic for validating
    that state transitions follow the rules of consciousness.
    """
    
    def __init__(self, rules: Optional[StateConsistencyRules] = None):
        self.rules = rules or StateConsistencyRules()
    
    def validate_state_container(
        self, 
        container: ImmutableStateContainer
    ) -> tuple[bool, List[str]]:
        """Validate entire state container for consistency."""
        errors = []
        
        # Validate current state
        if isinstance(container.current_state, jnp.ndarray):
            if not self.rules.validate_state_bounds(container.current_state):
                errors.append("Current state violates magnitude bounds")
        
        # Validate evolution history
        for i, event in enumerate(container.evolution_history):
            if i > 0:
                prev_event = container.evolution_history[i-1]
                if not self.rules.validate_temporal_ordering(
                    prev_event.timestamp, event.timestamp
                ):
                    errors.append(f"Event {event.event_id} violates temporal ordering")
        
        return len(errors) == 0, errors
    
    def validate_evolution_sequence(
        self, 
        events: List[StateEvolutionEvent]
    ) -> tuple[bool, List[str]]:
        """Validate sequence of evolution events."""
        errors = []
        
        for i in range(1, len(events)):
            current = events[i]
            previous = events[i-1]
            
            if not current.is_causal_successor(previous):
                errors.append(f"Event {current.event_id} breaks causal chain")
        
        return len(errors) == 0, errors