"""State Factory - Factory implementations following Abstract Factory pattern.

This module implements factory classes and functions for creating state managers
following Clean Architecture principles and SOLID design patterns.

Key Design Principles:
- Abstract Factory: Create families of related objects without specifying concrete classes
- Dependency Inversion: Clients depend on factory abstractions, not implementations
- Single Responsibility: Each factory creates one type of state manager
- Open/Closed: New factory implementations can be added without modifying existing code

The Factory layer provides a clean interface for creating properly configured
state management components while hiding construction complexity.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional, Generic, TypeVar, Union, Protocol
from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp

from .state_entities import (
    StateType,
    StateConsistencyRules,
    StateEvolutionType
)
from .state_interfaces import (
    StateManager,
    StateEvolutionStrategy,
    TemporalStateManager,
    EmbodimentStateManager,
    CouplingStateManager,
    IntegratedStateManager,
    StateManagerFactory as StateManagerFactoryProtocol
)
from .state_adapters import (
    TemporalStateAdapter,
    EmbodimentStateAdapter,
    CouplingStateAdapter,
    IntegratedStateAdapter
)
from .equinox_state_framework import (
    EquinoxStateManager,
    EquinoxStateConfiguration,
    create_equinox_state_manager
)
from ..types import Array, PRNGKey, TemporalMoment, BodyState, CouplingState


logger = logging.getLogger(__name__)


@dataclass
class StateManagerCreationConfig:
    """Configuration for state manager creation."""
    
    # Base configuration
    state_dim: int = 64
    enable_jit: bool = True
    enable_checkpointing: bool = True
    max_checkpoints: int = 10
    validation_mode: str = "strict"
    threading_mode: str = "sequential"
    
    # Consistency rules configuration
    max_state_magnitude: float = 1000.0
    min_temporal_interval: float = 0.001
    max_evolution_rate: float = 10.0
    require_finite_values: bool = True
    
    # Component-specific configuration
    temporal_config: Optional[Dict[str, Any]] = None
    embodiment_config: Optional[Dict[str, Any]] = None
    coupling_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default configurations for components."""
        if self.temporal_config is None:
            self.temporal_config = {
                'retention_depth': 10,
                'protention_horizon': 5,
                'synthesis_rate': 0.05
            }
        
        if self.embodiment_config is None:
            self.embodiment_config = {
                'proprioceptive_dim': 32,
                'motor_dim': 16,
                'body_map_resolution': (20, 20),
                'boundary_sensitivity': 0.1
            }
        
        if self.coupling_config is None:
            self.coupling_config = {
                'environmental_dim': 64,
                'coupling_strength': 0.5,
                'stability_window': 10
            }
    
    def create_equinox_config(self) -> EquinoxStateConfiguration:
        """Create Equinox configuration from this config."""
        return EquinoxStateConfiguration(
            enable_jit=self.enable_jit,
            enable_checkpointing=self.enable_checkpointing,
            max_checkpoints=self.max_checkpoints,
            validation_mode=self.validation_mode,
            threading_mode=self.threading_mode
        )
    
    def create_consistency_rules(self) -> StateConsistencyRules:
        """Create consistency rules from this config."""
        return StateConsistencyRules(
            max_state_magnitude=self.max_state_magnitude,
            min_temporal_interval=self.min_temporal_interval,
            max_evolution_rate=self.max_evolution_rate,
            require_finite_values=self.require_finite_values
        )


# Evolution Strategy Factories

class TemporalEvolutionStrategyFactory:
    """Factory for creating temporal evolution strategies."""
    
    @staticmethod
    def create_retention_decay_strategy(
        decay_rate: float = 0.95,
        retention_depth: int = 10
    ) -> StateEvolutionStrategy[Array]:
        """Create retention decay evolution strategy."""
        return RetentionDecayStrategy(decay_rate, retention_depth)
    
    @staticmethod
    def create_temporal_synthesis_strategy(
        synthesis_weights: Optional[Array] = None,
        protention_strength: float = 0.3
    ) -> StateEvolutionStrategy[Array]:
        """Create temporal synthesis evolution strategy."""
        return TemporalSynthesisStrategy(synthesis_weights, protention_strength)


class EmbodimentEvolutionStrategyFactory:
    """Factory for creating embodiment evolution strategies."""
    
    @staticmethod
    def create_body_schema_integration_strategy(
        integration_rate: float = 0.1,
        boundary_sensitivity: float = 0.1
    ) -> StateEvolutionStrategy[Array]:
        """Create body schema integration strategy."""
        return BodySchemaIntegrationStrategy(integration_rate, boundary_sensitivity)
    
    @staticmethod
    def create_motor_intention_strategy(
        prediction_horizon: int = 5,
        adaptation_rate: float = 0.05
    ) -> StateEvolutionStrategy[Array]:
        """Create motor intention evolution strategy."""
        return MotorIntentionStrategy(prediction_horizon, adaptation_rate)


class CouplingEvolutionStrategyFactory:
    """Factory for creating coupling evolution strategies."""
    
    @staticmethod
    def create_structural_coupling_strategy(
        coupling_strength: float = 0.5,
        stability_threshold: float = 0.8
    ) -> StateEvolutionStrategy[Array]:
        """Create structural coupling evolution strategy."""
        return StructuralCouplingStrategy(coupling_strength, stability_threshold)
    
    @staticmethod
    def create_circular_causality_strategy(
        causality_strength: float = 0.3,
        feedback_delay: int = 1
    ) -> StateEvolutionStrategy[Array]:
        """Create circular causality evolution strategy."""
        return CircularCausalityStrategy(causality_strength, feedback_delay)


# Concrete Evolution Strategies

class RetentionDecayStrategy:
    """Concrete strategy for retention buffer decay."""
    
    def __init__(self, decay_rate: float, retention_depth: int):
        self.decay_rate = decay_rate
        self.retention_depth = retention_depth
    
    def can_handle_evolution_type(self, evolution_type: StateEvolutionType) -> bool:
        """Check if strategy can handle evolution type."""
        return evolution_type == StateEvolutionType.RETENTION_DECAY
    
    def evolve(
        self,
        current_state: Array,
        evolution_input: Any,
        context: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> Array:
        """Apply retention decay to current state."""
        # Apply exponential decay to retention buffer
        decay_factors = jnp.power(self.decay_rate, jnp.arange(current_state.shape[0]))
        decayed_state = current_state * decay_factors[:, None]
        
        # Add new experience if provided
        if evolution_input is not None:
            new_experience = jnp.asarray(evolution_input)
            # Roll buffer and add new experience at the end
            rolled_state = jnp.roll(decayed_state, -1, axis=0)
            rolled_state = rolled_state.at[-1].set(new_experience)
            return rolled_state
        
        return decayed_state
    
    def estimate_evolution_cost(
        self,
        current_state: Array,
        evolution_input: Any
    ) -> float:
        """Estimate computational cost."""
        return current_state.size * 2.0  # Decay + roll operations


class TemporalSynthesisStrategy:
    """Concrete strategy for temporal moment synthesis."""
    
    def __init__(self, synthesis_weights: Optional[Array], protention_strength: float):
        self.synthesis_weights = synthesis_weights
        self.protention_strength = protention_strength
    
    def can_handle_evolution_type(self, evolution_type: StateEvolutionType) -> bool:
        """Check if strategy can handle evolution type."""
        return evolution_type == StateEvolutionType.SYNTHESIS_INTEGRATION
    
    def evolve(
        self,
        current_state: Array,
        evolution_input: Any,
        context: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> Array:
        """Apply temporal synthesis to current state."""
        # Simple temporal synthesis implementation
        if self.synthesis_weights is None:
            # Create default weights favoring present moment
            weights = jnp.array([0.3, 0.5, 0.2])  # retention, present, protention
        else:
            weights = self.synthesis_weights
        
        # Split state into temporal components
        state_dim = current_state.shape[-1] // 3  # Assume state is [retention, present, protention]
        retention = current_state[:state_dim]
        present = current_state[state_dim:2*state_dim]
        protention = current_state[2*state_dim:]
        
        # Synthesize temporal moment
        synthesized = (
            weights[0] * retention +
            weights[1] * present +
            weights[2] * protention
        )
        
        # Return updated state with synthesized present moment
        return jnp.concatenate([retention, synthesized, protention])
    
    def estimate_evolution_cost(
        self,
        current_state: Array,
        evolution_input: Any
    ) -> float:
        """Estimate computational cost."""
        return current_state.size * 3.0  # Split + synthesis + concatenate


class BodySchemaIntegrationStrategy:
    """Concrete strategy for body schema integration."""
    
    def __init__(self, integration_rate: float, boundary_sensitivity: float):
        self.integration_rate = integration_rate
        self.boundary_sensitivity = boundary_sensitivity
    
    def can_handle_evolution_type(self, evolution_type: StateEvolutionType) -> bool:
        """Check if strategy can handle evolution type."""
        return evolution_type == StateEvolutionType.DISCRETE_UPDATE
    
    def evolve(
        self,
        current_state: Array,
        evolution_input: Any,
        context: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> Array:
        """Apply body schema integration."""
        if evolution_input is None:
            return current_state
        
        # Integrate new sensory input with current body state
        sensory_input = jnp.asarray(evolution_input)
        
        # Simple integration with learning rate
        updated_state = (
            (1 - self.integration_rate) * current_state +
            self.integration_rate * sensory_input
        )
        
        return updated_state
    
    def estimate_evolution_cost(
        self,
        current_state: Array,
        evolution_input: Any
    ) -> float:
        """Estimate computational cost."""
        return current_state.size * 2.0  # Weighted sum operations


class MotorIntentionStrategy:
    """Concrete strategy for motor intention evolution."""
    
    def __init__(self, prediction_horizon: int, adaptation_rate: float):
        self.prediction_horizon = prediction_horizon
        self.adaptation_rate = adaptation_rate
    
    def can_handle_evolution_type(self, evolution_type: StateEvolutionType) -> bool:
        """Check if strategy can handle evolution type."""
        return evolution_type == StateEvolutionType.CONTINUOUS_FLOW
    
    def evolve(
        self,
        current_state: Array,
        evolution_input: Any,
        context: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> Array:
        """Apply motor intention evolution."""
        # Simple predictive motor control
        if evolution_input is not None:
            goal_state = jnp.asarray(evolution_input)
            # Move towards goal state
            motor_update = self.adaptation_rate * (goal_state - current_state)
            return current_state + motor_update
        
        # Apply momentum if no goal specified
        momentum = context.get('momentum', 0.9)
        return momentum * current_state
    
    def estimate_evolution_cost(
        self,
        current_state: Array,
        evolution_input: Any
    ) -> float:
        """Estimate computational cost."""
        return current_state.size * 1.5


class StructuralCouplingStrategy:
    """Concrete strategy for structural coupling dynamics."""
    
    def __init__(self, coupling_strength: float, stability_threshold: float):
        self.coupling_strength = coupling_strength
        self.stability_threshold = stability_threshold
    
    def can_handle_evolution_type(self, evolution_type: StateEvolutionType) -> bool:
        """Check if strategy can handle evolution type."""
        return evolution_type == StateEvolutionType.COUPLING_DYNAMICS
    
    def evolve(
        self,
        current_state: Array,
        evolution_input: Any,
        context: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> Array:
        """Apply structural coupling dynamics."""
        if evolution_input is None:
            return current_state
        
        environmental_perturbation = jnp.asarray(evolution_input)
        
        # Apply coupling dynamics
        coupling_effect = self.coupling_strength * environmental_perturbation
        updated_state = current_state + coupling_effect
        
        # Apply stability constraint
        state_norm = jnp.linalg.norm(updated_state)
        if state_norm > self.stability_threshold:
            updated_state = updated_state * (self.stability_threshold / state_norm)
        
        return updated_state
    
    def estimate_evolution_cost(
        self,
        current_state: Array,
        evolution_input: Any
    ) -> float:
        """Estimate computational cost."""
        return current_state.size * 2.5  # Coupling + normalization


class CircularCausalityStrategy:
    """Concrete strategy for circular causality dynamics."""
    
    def __init__(self, causality_strength: float, feedback_delay: int):
        self.causality_strength = causality_strength
        self.feedback_delay = feedback_delay
    
    def can_handle_evolution_type(self, evolution_type: StateEvolutionType) -> bool:
        """Check if strategy can handle evolution type."""
        return evolution_type == StateEvolutionType.COUPLING_DYNAMICS
    
    def evolve(
        self,
        current_state: Array,
        evolution_input: Any,
        context: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> Array:
        """Apply circular causality dynamics."""
        # Simple circular causality: state influences itself through environment
        self_influence = self.causality_strength * jnp.tanh(current_state)
        
        if evolution_input is not None:
            external_input = jnp.asarray(evolution_input)
            return current_state + self_influence + external_input
        
        return current_state + self_influence
    
    def estimate_evolution_cost(
        self,
        current_state: Array,
        evolution_input: Any
    ) -> float:
        """Estimate computational cost."""
        return current_state.size * 2.0  # Tanh + addition


# Main State Manager Factory

class StateManagerFactory:
    """Main factory for creating different types of state managers.
    
    Implements the Abstract Factory pattern to provide a unified interface
    for creating families of related state management objects.
    """
    
    def __init__(self, config: Optional[StateManagerCreationConfig] = None):
        self.config = config or StateManagerCreationConfig()
        self._strategy_factories = {
            'temporal': TemporalEvolutionStrategyFactory(),
            'embodiment': EmbodimentEvolutionStrategyFactory(),
            'coupling': CouplingEvolutionStrategyFactory()
        }
    
    def create_temporal_manager(
        self,
        initial_state: Array,
        config: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> TemporalStateAdapter:
        """Create temporal state manager."""
        try:
            # Create temporal moment from initial state
            temporal_moment = self._create_temporal_moment_from_array(initial_state, config)
            
            # Create consistency rules
            consistency_rules = self.config.create_consistency_rules()
            
            # Create temporal adapter
            adapter = TemporalStateAdapter(temporal_moment, consistency_rules)
            
            # Set evolution strategy
            strategy = self._strategy_factories['temporal'].create_retention_decay_strategy(
                decay_rate=config.get('decay_rate', 0.95),
                retention_depth=config.get('retention_depth', 10)
            )
            adapter.set_evolution_strategy(strategy)
            
            logger.info("Created temporal state manager")
            return adapter
            
        except Exception as e:
            logger.error(f"Failed to create temporal manager: {str(e)}")
            raise
    
    def create_embodiment_manager(
        self,
        initial_state: Array,
        config: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> EmbodimentStateAdapter:
        """Create embodiment state manager."""
        try:
            # Create body state from initial state
            body_state = self._create_body_state_from_array(initial_state, config)
            
            # Create consistency rules
            consistency_rules = self.config.create_consistency_rules()
            
            # Create embodiment adapter
            adapter = EmbodimentStateAdapter(body_state, consistency_rules)
            
            # Set evolution strategy
            strategy = self._strategy_factories['embodiment'].create_body_schema_integration_strategy(
                integration_rate=config.get('integration_rate', 0.1),
                boundary_sensitivity=config.get('boundary_sensitivity', 0.1)
            )
            adapter.set_evolution_strategy(strategy)
            
            logger.info("Created embodiment state manager")
            return adapter
            
        except Exception as e:
            logger.error(f"Failed to create embodiment manager: {str(e)}")
            raise
    
    def create_coupling_manager(
        self,
        initial_agent_state: Array,
        initial_env_state: Array,
        config: Dict[str, Any],
        key: Optional[PRNGKey] = None
    ) -> CouplingStateAdapter:
        """Create coupling state manager."""
        try:
            # Create coupling state
            coupling_state = self._create_coupling_state_from_arrays(
                initial_agent_state, initial_env_state, config
            )
            
            # Create consistency rules
            consistency_rules = self.config.create_consistency_rules()
            
            # Create coupling adapter
            adapter = CouplingStateAdapter(coupling_state, consistency_rules)
            
            # Set evolution strategy
            strategy = self._strategy_factories['coupling'].create_structural_coupling_strategy(
                coupling_strength=config.get('coupling_strength', 0.5),
                stability_threshold=config.get('stability_threshold', 0.8)
            )
            adapter.set_evolution_strategy(strategy)
            
            logger.info("Created coupling state manager")
            return adapter
            
        except Exception as e:
            logger.error(f"Failed to create coupling manager: {str(e)}")
            raise
    
    def create_integrated_manager(
        self,
        managers: Dict[str, StateManager],
        integration_config: Dict[str, Any]
    ) -> IntegratedStateAdapter:
        """Create integrated state manager from component managers."""
        try:
            # Extract specialized adapters
            temporal_adapter = managers.get('temporal')
            embodiment_adapter = managers.get('embodiment')
            coupling_adapter = managers.get('coupling')
            
            if not all([temporal_adapter, embodiment_adapter, coupling_adapter]):
                raise ValueError("All component managers (temporal, embodiment, coupling) required")
            
            # Create integrated adapter
            integrated_adapter = IntegratedStateAdapter(
                temporal_adapter,
                embodiment_adapter,
                coupling_adapter
            )
            
            logger.info("Created integrated state manager")
            return integrated_adapter
            
        except Exception as e:
            logger.error(f"Failed to create integrated manager: {str(e)}")
            raise
    
    def _create_temporal_moment_from_array(
        self,
        state_array: Array,
        config: Dict[str, Any]
    ) -> TemporalMoment:
        """Create TemporalMoment from array."""
        from ..types import create_temporal_moment
        
        state_dim = state_array.shape[-1]
        
        return create_temporal_moment(
            timestamp=time.time(),
            retention=jnp.zeros((state_dim,)),
            present_moment=state_array,
            protention=jnp.zeros((state_dim,)),
            synthesis_weights=jnp.ones((state_dim,)) * 0.5  # Same shape as other arrays
        )
    
    def _create_body_state_from_array(
        self,
        state_array: Array,
        config: Dict[str, Any]
    ) -> BodyState:
        """Create BodyState from array."""
        proprioceptive_dim = config.get('proprioceptive_dim', state_array.shape[-1] // 2)
        motor_dim = config.get('motor_dim', state_array.shape[-1] // 4)
        
        return BodyState(
            proprioception=state_array[:proprioceptive_dim],
            motor_intention=jnp.zeros((motor_dim,)),
            boundary_signal=jnp.zeros((1,)),
            schema_confidence=0.5
        )
    
    def _create_coupling_state_from_arrays(
        self,
        agent_state: Array,
        env_state: Array,
        config: Dict[str, Any]
    ) -> CouplingState:
        """Create CouplingState from arrays."""
        return CouplingState(
            agent_state=agent_state,
            environmental_state=env_state,
            coupling_strength=config.get('coupling_strength', 0.5),
            perturbation_history=jnp.zeros((10, env_state.shape[-1])),
            stability_metric=0.5
        )


# Convenience Factory Functions

def create_temporal_state_manager(
    initial_state: Array,
    retention_depth: int = 10,
    decay_rate: float = 0.95,
    key: Optional[PRNGKey] = None
) -> TemporalStateAdapter:
    """Convenience function to create temporal state manager."""
    factory = StateManagerFactory()
    config = {
        'retention_depth': retention_depth,
        'decay_rate': decay_rate
    }
    return factory.create_temporal_manager(initial_state, config, key)


def create_embodiment_state_manager(
    initial_state: Array,
    proprioceptive_dim: int = 32,
    motor_dim: int = 16,
    integration_rate: float = 0.1,
    key: Optional[PRNGKey] = None
) -> EmbodimentStateAdapter:
    """Convenience function to create embodiment state manager."""
    factory = StateManagerFactory()
    config = {
        'proprioceptive_dim': proprioceptive_dim,
        'motor_dim': motor_dim,
        'integration_rate': integration_rate
    }
    return factory.create_embodiment_manager(initial_state, config, key)


def create_coupling_state_manager(
    initial_agent_state: Array,
    initial_env_state: Array,
    coupling_strength: float = 0.5,
    stability_threshold: float = 0.8,
    key: Optional[PRNGKey] = None
) -> CouplingStateAdapter:
    """Convenience function to create coupling state manager."""
    factory = StateManagerFactory()
    config = {
        'coupling_strength': coupling_strength,
        'stability_threshold': stability_threshold
    }
    return factory.create_coupling_manager(
        initial_agent_state, initial_env_state, config, key
    )


def create_integrated_state_manager(
    temporal_manager: TemporalStateAdapter,
    embodiment_manager: EmbodimentStateAdapter,
    coupling_manager: CouplingStateAdapter,
    integration_config: Optional[Dict[str, Any]] = None
) -> IntegratedStateAdapter:
    """Convenience function to create integrated state manager."""
    factory = StateManagerFactory()
    managers = {
        'temporal': temporal_manager,
        'embodiment': embodiment_manager,
        'coupling': coupling_manager
    }
    return factory.create_integrated_manager(
        managers, integration_config or {}
    )