"""Clean Architecture State Management for Enactive Consciousness.

This module implements Uncle Bob's Clean Architecture principles and SOLID design
patterns for state management in the enactive consciousness system.

Architecture Layers:
    Entities: Core state representations (state_entities.py)
    Use Cases: State evolution business rules (state_use_cases.py)  
    Interface Adapters: Framework-specific adapters (state_adapters.py)
    Frameworks: JAX/Equinox implementation details (equinox_state_framework.py)

Key Design Principles:
    - Single Responsibility: Each class has one reason to change
    - Open/Closed: Extensible without modification
    - Liskov Substitution: Interchangeable implementations
    - Interface Segregation: Focused, client-specific interfaces
    - Dependency Inversion: Depend on abstractions, not concretions

State Management Patterns:
    - Immutable state holders with functional updates
    - Strategy pattern for different state evolution types
    - Dependency injection for state managers
    - Clean separation of stateful/stateless operations
"""

from .state_entities import (
    StateSnapshot,
    StateEvolutionEvent,
    StateConsistencyRules,
    ImmutableStateContainer,
    StateType,
    StateEvolutionType,
    StateConsistencyValidator,
)

from .state_interfaces import (
    StateManager,
    ReadableState,
    WritableState, 
    StateEvolutionStrategy,
    StateValidator,
    StateThreadingCoordinator,
)

from .state_use_cases import (
    StateEvolutionUseCase,
    StateConsistencyUseCase,
    StateSnapshotUseCase,
    StateRecoveryUseCase,
)

from .state_adapters import (
    EquinoxStateAdapter,
    TemporalStateAdapter,
    EmbodimentStateAdapter,
    CouplingStateAdapter,
)

from .equinox_state_framework import (
    EquinoxStateManager,
    EquinoxStateHolder,
    EquinoxStateEvolutionEngine,
    create_equinox_state_manager,
)

from .state_factory import (
    StateManagerFactory,
    create_temporal_state_manager,
    create_embodiment_state_manager,
    create_coupling_state_manager,
    create_integrated_state_manager,
)

__all__ = [
    # Entities
    'StateSnapshot',
    'StateEvolutionEvent', 
    'StateConsistencyRules',
    'ImmutableStateContainer',
    'StateType',
    'StateEvolutionType', 
    'StateConsistencyValidator',
    
    # Interfaces
    'StateManager',
    'ReadableState',
    'WritableState',
    'StateEvolutionStrategy', 
    'StateValidator',
    'StateThreadingCoordinator',
    
    # Use Cases
    'StateEvolutionUseCase',
    'StateConsistencyUseCase',
    'StateSnapshotUseCase',
    'StateRecoveryUseCase',
    
    # Adapters
    'EquinoxStateAdapter',
    'TemporalStateAdapter',
    'EmbodimentStateAdapter', 
    'CouplingStateAdapter',
    
    # Framework
    'EquinoxStateManager',
    'EquinoxStateHolder',
    'EquinoxStateEvolutionEngine',
    'create_equinox_state_manager',
    
    # Factory
    'StateManagerFactory',
    'create_temporal_state_manager',
    'create_embodiment_state_manager', 
    'create_coupling_state_manager',
    'create_integrated_state_manager',
]