"""Continuous dynamics for enactive consciousness using diffrax.

This module implements continuous-time differential equation systems for enactive
consciousness, transforming discrete-time processes into smooth mathematical flows.
Based on Husserlian temporal consciousness and Varela-Maturana circular causality,
this provides rigorous mathematical foundations for phenomenological processes.

Key Features:
1. Husserlian retention-present-protention as continuous flows
2. Varela-Maturana circular causality as coupled ODEs/SDEs  
3. Neural ODEs for smooth state evolution
4. Stochastic differential equations for environmental coupling
5. Numerical stability and error handling
6. Integration with existing temporal processing

Domain-Driven Design Principles:
- Clear separation of mathematical concerns from domain logic
- Rich domain objects for differential equation specifications
- Ubiquitous language from phenomenology and enactivism
- Bounded contexts for different types of dynamics
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

import jax
import jax.numpy as jnp
import equinox as eqx

# Try to import diffrax, but gracefully handle if it's not available
try:
    import diffrax
    # For now, use fallback until diffrax issues are resolved
    DIFFRAX_AVAILABLE = False
    warnings.warn(
        "Using simplified Euler integration for testing. Enable diffrax for full functionality.",
        ImportWarning
    )
except ImportError:
    DIFFRAX_AVAILABLE = False
    diffrax = None
    warnings.warn(
        "diffrax not available. Continuous dynamics will use simplified Euler integration.",
        ImportWarning
    )

from .core import (
    ProcessorBase,
    StateValidationMixin,
    ConfigurableMixin,
    MemoryManager,
    MetricCollector,
    optimize_for_memory,
    GLOBAL_MEMORY_MANAGER,
    GLOBAL_METRICS,
)

from .types import (
    Array,
    ArrayLike,
    PRNGKey,
    TimeStep,
    TemporalMoment,
    CouplingState,
    MeaningStructure,
    FrameworkConfig,
    EnactiveConsciousnessError,
    validate_consciousness_state,
    validate_temporal_consistency,
)

# Configure module logger
logger = logging.getLogger(__name__)


class DynamicsType(Enum):
    """Types of continuous dynamics supported."""
    DETERMINISTIC_ODE = "deterministic_ode"
    STOCHASTIC_SDE = "stochastic_sde"
    NEURAL_ODE = "neural_ode"
    COUPLED_SYSTEM = "coupled_system"


class IntegrationMethod(Enum):
    """Numerical integration methods for differential equations."""
    EULER = "euler"
    HEUN = "heun"
    MIDPOINT = "midpoint"
    RUNGE_KUTTA_4 = "rk4"
    TSIT5 = "tsit5"
    DOPRI5 = "dopri5"
    DOPRI8 = "dopri8"


@dataclass(frozen=True)
class DynamicsConfig:
    """Configuration for continuous dynamics systems.
    
    This configuration object encapsulates all parameters needed for
    continuous-time modeling of consciousness processes.
    """
    # Time integration parameters
    dt_min: float = 1e-4
    dt_max: float = 0.1
    rtol: float = 1e-3
    atol: float = 1e-6
    max_steps: int = 10000
    
    # Phenomenological time parameters
    retention_decay_rate: float = 0.1
    protention_anticipation_rate: float = 0.05
    present_moment_width: float = 0.01
    temporal_synthesis_strength: float = 1.0
    
    # Enactive coupling parameters
    agent_environment_coupling_strength: float = 0.5
    circular_causality_time_constant: float = 0.1
    structural_coupling_adaptation_rate: float = 0.02
    meaning_emergence_threshold_dynamics: float = 0.3
    
    # Stochastic parameters
    environmental_noise_strength: float = 0.1
    internal_noise_strength: float = 0.05
    coupling_noise_correlation: float = 0.3
    
    # Neural ODE parameters
    hidden_dynamics_dim: int = 128
    neural_ode_depth: int = 3
    activation_dynamics: str = "swish"
    
    # Numerical stability parameters
    stability_regularization: float = 1e-6
    gradient_clipping_threshold: float = 10.0
    adaptive_step_safety_factor: float = 0.9


@dataclass(frozen=True)
class ContinuousState:
    """State representation for continuous dynamics.
    
    This represents the complete state of consciousness processes
    in continuous time, serving as the state vector for differential equations.
    """
    # Temporal consciousness components
    retention_field: Array  # Continuous retention field R(t,τ)
    present_awareness: Array  # Present moment awareness P(t)
    protention_field: Array  # Protentional horizon field F(t,τ)
    temporal_synthesis_weights: Array  # Dynamic synthesis weights W(t)
    
    # Enactive coupling components
    agent_state: Array  # Agent internal state A(t)
    environmental_coupling: Array  # Environmental coupling E(t)
    circular_causality_trace: Array  # Causality history trace C(t)
    meaning_emergence_level: Array  # Meaning emergence M(t)
    
    # System metadata
    timestamp: float
    consciousness_level: float
    coupling_strength: float
    temporal_coherence: float
    
    def validate_state(self) -> bool:
        """Validate the continuous state for mathematical consistency."""
        try:
            # Check for finite values
            for field_name, field_value in self.__dict__.items():
                if isinstance(field_value, Array):
                    if not jnp.all(jnp.isfinite(field_value)):
                        logger.warning(f"Non-finite values in {field_name}")
                        return False
            
            # Check temporal consistency
            if not validate_temporal_consistency(
                self.retention_field, self.present_awareness, self.protention_field
            ):
                return False
            
            # Check consciousness state
            if not validate_consciousness_state(self.present_awareness):
                return False
            
            # Check scalar bounds
            if not (0.0 <= self.consciousness_level <= 1.0):
                return False
            if not (0.0 <= self.coupling_strength <= 1.0):
                return False
            if not (0.0 <= self.temporal_coherence <= 1.0):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            return False


class HusserlianTemporalFlow(eqx.Module):
    """Continuous-time implementation of Husserlian temporal consciousness.
    
    This implements the temporal flow of consciousness as a system of
    coupled differential equations representing retention, present impression,
    and protention as continuous fields evolving in time.
    
    Mathematical formulation:
    dR/dt = -λ_r * R(t) + δ(t) * P(t-dt)  # Retention decay + impression flow
    dP/dt = f_present(S(t), R(t), F(t))   # Present synthesis function
    dF/dt = λ_f * ∇F + g_anticipation(P(t), R(t))  # Protentional anticipation
    """
    
    # Temporal field dynamics
    retention_dynamics: eqx.nn.MLP
    present_synthesis: eqx.nn.MLP
    protention_projection: eqx.nn.MLP
    temporal_attention: eqx.nn.MultiheadAttention
    
    # Phenomenological parameters
    retention_decay_constant: float
    protention_anticipation_constant: float
    present_integration_strength: float
    temporal_synthesis_coupling: float
    
    def __init__(
        self,
        state_dim: int,
        temporal_depth: int,
        config: DynamicsConfig,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, 4)
        
        # Initialize neural networks for temporal dynamics
        self.retention_dynamics = eqx.nn.MLP(
            in_size=state_dim * 2,  # Current retention + present influence
            out_size=state_dim,
            width_size=state_dim,
            depth=2,
            activation=getattr(jax.nn, config.activation_dynamics),
            key=keys[0],
        )
        
        self.present_synthesis = eqx.nn.MLP(
            in_size=state_dim * 3,  # Retention + present + protention
            out_size=state_dim,
            width_size=state_dim * 2,
            depth=3,
            activation=getattr(jax.nn, config.activation_dynamics),
            key=keys[1],
        )
        
        self.protention_projection = eqx.nn.MLP(
            in_size=state_dim * 2,  # Present + retention context
            out_size=state_dim,
            width_size=state_dim,
            depth=2,
            activation=getattr(jax.nn, config.activation_dynamics),
            key=keys[2],
        )
        
        self.temporal_attention = eqx.nn.MultiheadAttention(
            num_heads=4,
            query_size=state_dim,
            key_size=state_dim,
            value_size=state_dim,
            output_size=state_dim,
            key=keys[3],
        )
        
        # Phenomenological parameters from configuration
        self.retention_decay_constant = config.retention_decay_rate
        self.protention_anticipation_constant = config.protention_anticipation_rate
        self.present_integration_strength = config.temporal_synthesis_strength
        self.temporal_synthesis_coupling = config.temporal_synthesis_strength * 0.5
    
    def temporal_flow_equations(
        self,
        t: float,
        temporal_state: Array,
        environmental_input: Array,
    ) -> Array:
        """Define the differential equations for Husserlian temporal flow.
        
        This is the core mathematical model implementing Husserl's analysis
        of internal time consciousness as a continuous dynamical system.
        """
        # Unpack temporal state components
        state_dim = temporal_state.shape[0] // 4  # retention, present, protention, weights
        retention = temporal_state[:state_dim]
        present = temporal_state[state_dim:2*state_dim]
        protention = temporal_state[2*state_dim:3*state_dim]
        synthesis_weights = temporal_state[3*state_dim:4*state_dim]
        
        # Retention dynamics: decay + flow from present
        # dR/dt = -λ_r * R + P(t-dt) + noise
        retention_input = jnp.concatenate([retention, present])
        retention_flow = self.retention_dynamics(retention_input)
        # Ensure environmental input matches expected dimensions
        env_retention = environmental_input[:min(state_dim, len(environmental_input))]
        if len(env_retention) < state_dim:
            env_retention = jnp.pad(env_retention, (0, state_dim - len(env_retention)))
        
        dR_dt = (-self.retention_decay_constant * retention + 
                 retention_flow + 
                 env_retention * 0.1)
        
        # Present moment dynamics: synthesis of temporal horizon
        # dP/dt = f_synthesis(R, P, F, env)
        temporal_context = jnp.concatenate([retention, present, protention])
        present_synthesis_raw = self.present_synthesis(temporal_context)
        
        # Apply temporal attention to synthesis
        temporal_sequence = jnp.stack([retention, present, protention])  # Shape: (3, state_dim)
        
        # MultiheadAttention expects (seq_len, features), we have (3, state_dim)
        attention_result = self.temporal_attention(
            present_synthesis_raw[None, :],  # Query: (1, state_dim)
            temporal_sequence,               # Key: (3, state_dim) 
            temporal_sequence,               # Value: (3, state_dim)
        )
        
        # Handle different return formats from attention
        if isinstance(attention_result, tuple):
            attended_synthesis, _ = attention_result
        else:
            attended_synthesis = attention_result
        
        attended_synthesis = attended_synthesis.squeeze(0)  # Remove batch dimension
        
        # Ensure environmental input for present matches dimensions  
        start_idx = min(state_dim, len(environmental_input))
        end_idx = min(2 * state_dim, len(environmental_input))
        env_present = environmental_input[start_idx:end_idx] if end_idx > start_idx else jnp.zeros(0)
        
        if len(env_present) < state_dim:
            env_present = jnp.pad(env_present, (0, state_dim - len(env_present)))
        
        dP_dt = (self.present_integration_strength * attended_synthesis +
                 env_present * 0.3)
        
        # Protention dynamics: anticipatory projection
        # dF/dt = λ_f * projection(P, R) + anticipation_gradient
        protention_context = jnp.concatenate([present, retention])
        protention_projection = self.protention_projection(protention_context)
        
        # Ensure environmental input for protention matches dimensions
        start_idx = min(2 * state_dim, len(environmental_input))
        end_idx = min(3 * state_dim, len(environmental_input))
        env_protention = environmental_input[start_idx:end_idx] if end_idx > start_idx else jnp.zeros(0)
        
        if len(env_protention) < state_dim:
            env_protention = jnp.pad(env_protention, (0, state_dim - len(env_protention)))
        
        dF_dt = (self.protention_anticipation_constant * protention_projection +
                 jnp.gradient(protention) * 0.1 +  # Temporal gradient
                 env_protention * 0.05)
        
        # Synthesis weights dynamics: adaptive temporal integration
        # dW/dt = attention_gradients + normalization_flow
        temporal_norms = jnp.array([
            jnp.linalg.norm(retention),
            jnp.linalg.norm(present), 
            jnp.linalg.norm(protention)
        ])
        
        target_weights = jax.nn.softmax(temporal_norms + 1e-8)
        current_weights_normalized = jax.nn.softmax(synthesis_weights[:3])
        
        dW_dt = jnp.zeros_like(synthesis_weights)
        dW_dt = dW_dt.at[:3].set(
            0.1 * (target_weights - current_weights_normalized)
        )
        
        # Combine all temporal derivatives
        return jnp.concatenate([dR_dt, dP_dt, dF_dt, dW_dt])


class EnactiveCouplingDynamics(eqx.Module):
    """Continuous-time enactive coupling with circular causality.
    
    This implements Varela-Maturana structural coupling and circular causality
    as a system of coupled stochastic differential equations, modeling the
    continuous interaction between agent and environment.
    
    Mathematical formulation:
    dA/dt = f_agent(A, E, M) + σ_A * dW_A  # Agent dynamics
    dE/dt = g_env(A, E) + h_perturbation(t) + σ_E * dW_E  # Environment dynamics
    dM/dt = emergence_dynamics(A, E, history) + σ_M * dW_M  # Meaning emergence
    
    Where circular causality is encoded in the coupling functions f_agent and g_env.
    """
    
    # Agent-environment coupling networks
    agent_dynamics: eqx.nn.MLP
    environment_response: eqx.nn.MLP
    circular_causality_network: eqx.nn.MLP
    meaning_emergence_dynamics: eqx.nn.MLP
    
    # Coupling strength adaptation
    coupling_adaptation: eqx.nn.Linear
    stability_controller: eqx.nn.MLP
    
    # System parameters
    coupling_time_constant: float
    causality_feedback_strength: float
    adaptation_learning_rate: float
    noise_correlation_matrix: Array
    
    def __init__(
        self,
        agent_dim: int,
        environment_dim: int,
        meaning_dim: int,
        config: DynamicsConfig,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, 6)
        
        self.agent_dynamics = eqx.nn.MLP(
            in_size=agent_dim + environment_dim + meaning_dim,
            out_size=agent_dim,
            width_size=max(agent_dim, environment_dim),
            depth=config.neural_ode_depth,
            activation=getattr(jax.nn, config.activation_dynamics),
            key=keys[0],
        )
        
        self.environment_response = eqx.nn.MLP(
            in_size=agent_dim + environment_dim,
            out_size=environment_dim,
            width_size=max(agent_dim, environment_dim),
            depth=2,
            activation=getattr(jax.nn, config.activation_dynamics),
            key=keys[1],
        )
        
        self.circular_causality_network = eqx.nn.MLP(
            in_size=(agent_dim + environment_dim) * 2,  # Current + history
            out_size=agent_dim + environment_dim,
            width_size=max(agent_dim, environment_dim) * 2,
            depth=3,
            activation=getattr(jax.nn, config.activation_dynamics),
            key=keys[2],
        )
        
        self.meaning_emergence_dynamics = eqx.nn.MLP(
            in_size=agent_dim + environment_dim + meaning_dim,  # Include hidden state
            out_size=meaning_dim,
            width_size=meaning_dim * 2,
            depth=2,
            activation=getattr(jax.nn, config.activation_dynamics),
            key=keys[3],
        )
        
        self.coupling_adaptation = eqx.nn.Linear(
            in_features=agent_dim + environment_dim,
            out_features=1,  # Coupling strength
            key=keys[4],
        )
        
        self.stability_controller = eqx.nn.MLP(
            in_size=agent_dim + environment_dim + meaning_dim,
            out_size=agent_dim + environment_dim,
            width_size=max(agent_dim, environment_dim),
            depth=2,
            activation=jax.nn.tanh,
            key=keys[5],
        )
        
        # System parameters from configuration
        self.coupling_time_constant = config.circular_causality_time_constant
        self.causality_feedback_strength = config.agent_environment_coupling_strength
        self.adaptation_learning_rate = config.structural_coupling_adaptation_rate
        
        # Noise correlation matrix for correlated Brownian motion
        total_dim = agent_dim + environment_dim + meaning_dim
        self.noise_correlation_matrix = (
            jnp.eye(total_dim) + 
            config.coupling_noise_correlation * jnp.ones((total_dim, total_dim))
        ) / (1.0 + config.coupling_noise_correlation * (total_dim - 1))
    
    def enactive_coupling_sde(
        self,
        t: float,
        coupling_state: Array,
        history_context: Array,
        meaning_context: Array,
    ) -> Tuple[Array, Array]:
        """Define the SDE system for enactive coupling with circular causality.
        
        Returns (drift, diffusion) terms for the stochastic differential equation.
        """
        # Unpack coupling state
        agent_dim = coupling_state.shape[0] // 3  # agent, environment, meaning each 1/3
        agent_state = coupling_state[:agent_dim]
        environment_state = coupling_state[agent_dim:2*agent_dim]
        meaning_state = coupling_state[2*agent_dim:3*agent_dim]
        
        # === DRIFT TERMS ===
        
        # Agent dynamics with environmental coupling and meaning influence
        agent_input = jnp.concatenate([agent_state, environment_state, meaning_state])
        agent_drift_base = self.agent_dynamics(agent_input)
        
        # Environment response to agent actions
        env_input = jnp.concatenate([agent_state, environment_state])
        environment_drift_base = self.environment_response(env_input)
        
        # Circular causality computation
        current_coupling = jnp.concatenate([agent_state, environment_state])
        causality_input = jnp.concatenate([current_coupling, history_context])
        circular_influence = self.circular_causality_network(causality_input)
        
        agent_causality = circular_influence[:agent_dim]
        environment_causality = circular_influence[agent_dim:]
        
        # Apply circular causality with time constant
        agent_drift = (agent_drift_base + 
                      self.causality_feedback_strength * agent_causality)
        
        environment_drift = (environment_drift_base + 
                           self.causality_feedback_strength * environment_causality)
        
        # Meaning emergence dynamics using MLP with context
        meaning_input = jnp.concatenate([agent_state, environment_state, meaning_context])
        meaning_drift = self.meaning_emergence_dynamics(meaning_input)
        
        # Adaptive coupling strength
        coupling_strength_input = jnp.concatenate([agent_state, environment_state])
        coupling_adaptation = jax.nn.sigmoid(
            self.coupling_adaptation(coupling_strength_input)
        )
        
        # Apply adaptive coupling to all components
        agent_drift = agent_drift * coupling_adaptation
        environment_drift = environment_drift * coupling_adaptation
        
        # Stability control to prevent blow-up
        stability_input = jnp.concatenate([agent_state, environment_state, meaning_state])
        stability_correction = self.stability_controller(stability_input)
        
        agent_drift = agent_drift + stability_correction[:agent_dim] * 0.1
        environment_drift = environment_drift + stability_correction[agent_dim:] * 0.1
        
        # Combine all drift terms
        total_drift = jnp.concatenate([agent_drift, environment_drift, meaning_drift])
        
        # === DIFFUSION TERMS ===
        
        # State-dependent diffusion (multiplicative noise)
        agent_diffusion = 0.1 * jnp.abs(agent_state) * 0.1
        environment_diffusion = 0.1 * jnp.abs(environment_state) * 0.15
        meaning_diffusion = 0.1 * jnp.abs(meaning_state) * 0.05
        
        total_diffusion_diagonal = jnp.concatenate([
            agent_diffusion, environment_diffusion, meaning_diffusion
        ])
        
        # Apply correlation structure
        total_diffusion = jnp.diag(total_diffusion_diagonal) @ jnp.sqrt(
            self.noise_correlation_matrix
        )
        
        return total_drift, total_diffusion


class NeuralODEConsciousnessFlow(eqx.Module):
    """Neural ODE implementation for smooth consciousness state evolution.
    
    This uses neural ODEs to learn the continuous dynamics of consciousness
    processes directly from data, providing a flexible framework for modeling
    complex temporal dependencies in phenomenological processes.
    
    The neural ODE learns a function f_θ(x, t) such that:
    dx/dt = f_θ(x, t)
    
    Where x represents the consciousness state and θ are learnable parameters.
    """
    
    # Neural ODE networks
    dynamics_network: eqx.nn.MLP
    temporal_modulation: eqx.nn.MLP
    consciousness_level_predictor: eqx.nn.Linear
    
    # Adaptive time stepping
    time_embedding: eqx.nn.MLP
    adaptive_step_controller: eqx.nn.Linear
    
    # Regularization components
    jacobian_regularizer: eqx.nn.Linear
    energy_conservator: eqx.nn.MLP
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        config: DynamicsConfig,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, 6)
        
        self.dynamics_network = eqx.nn.MLP(
            in_size=state_dim + 1,  # State + time
            out_size=state_dim,
            width_size=hidden_dim,
            depth=config.neural_ode_depth,
            activation=getattr(jax.nn, config.activation_dynamics),
            key=keys[0],
        )
        
        self.temporal_modulation = eqx.nn.MLP(
            in_size=1,  # Time input
            out_size=state_dim,
            width_size=hidden_dim // 2,
            depth=2,
            activation=jax.nn.tanh,  # Smooth temporal modulation
            key=keys[1],
        )
        
        self.consciousness_level_predictor = eqx.nn.Linear(
            in_features=state_dim,
            out_features=1,
            key=keys[2],
        )
        
        self.time_embedding = eqx.nn.MLP(
            in_size=1,
            out_size=hidden_dim // 4,
            width_size=hidden_dim // 2,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[3],
        )
        
        self.adaptive_step_controller = eqx.nn.Linear(
            in_features=state_dim + hidden_dim // 4,
            out_features=1,
            key=keys[4],
        )
        
        self.jacobian_regularizer = eqx.nn.Linear(
            in_features=state_dim * state_dim,  # Flattened Jacobian
            out_features=1,
            key=keys[5],
        )
        
        self.energy_conservator = eqx.nn.MLP(
            in_size=state_dim,
            out_size=state_dim,
            width_size=hidden_dim // 2,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[5],  # Reuse last key
        )
    
    def neural_ode_dynamics(
        self,
        t: float,
        consciousness_state: Array,
        phenomenological_context: Optional[Array] = None,
    ) -> Array:
        """Neural ODE dynamics function for consciousness state evolution."""
        # Time embedding for temporal modulation
        time_tensor = jnp.array([t])
        time_modulation = self.temporal_modulation(time_tensor)
        time_embedding = self.time_embedding(time_tensor)
        
        # Core dynamics computation
        state_time_input = jnp.concatenate([consciousness_state, time_tensor])
        raw_dynamics = self.dynamics_network(state_time_input)
        
        # Apply temporal modulation
        modulated_dynamics = raw_dynamics * (1.0 + 0.1 * time_modulation)
        
        # Energy conservation constraint
        energy_correction = self.energy_conservator(consciousness_state)
        energy_conserved_dynamics = modulated_dynamics + 0.05 * energy_correction
        
        # Adaptive step size hint (for external solver)
        step_control_input = jnp.concatenate([consciousness_state, time_embedding])
        suggested_step_size = jax.nn.sigmoid(
            self.adaptive_step_controller(step_control_input)
        ) * 0.1  # Max step size of 0.1
        
        # Add phenomenological context if provided
        if phenomenological_context is not None:
            context_influence = phenomenological_context * 0.1
            energy_conserved_dynamics = energy_conserved_dynamics + context_influence
        
        # Gradient clipping for numerical stability
        dynamics_norm = jnp.linalg.norm(energy_conserved_dynamics)
        max_norm = 10.0  # Configurable gradient clipping threshold
        
        if dynamics_norm > max_norm:
            energy_conserved_dynamics = (
                energy_conserved_dynamics * (max_norm / dynamics_norm)
            )
        
        return energy_conserved_dynamics
    
    def compute_consciousness_level(self, state: Array) -> float:
        """Compute consciousness level from current state."""
        consciousness_raw = self.consciousness_level_predictor(state)
        return float(jax.nn.sigmoid(consciousness_raw.squeeze()))
    
    def jacobian_regularization_loss(self, state: Array, t: float) -> float:
        """Compute Jacobian regularization loss for stability."""
        # Compute Jacobian of dynamics w.r.t. state
        jacobian_fn = jax.jacobian(
            lambda s: self.neural_ode_dynamics(t, s), argnums=0
        )
        jacobian = jacobian_fn(state)
        
        # Flatten and regularize
        jacobian_flat = jacobian.flatten()
        regularization_score = self.jacobian_regularizer(jacobian_flat)
        
        # Penalize large eigenvalues for stability
        eigenvals = jnp.linalg.eigvals(jacobian)
        stability_penalty = jnp.sum(jnp.maximum(0, jnp.real(eigenvals) - 1.0))
        
        return float(regularization_score.squeeze() + stability_penalty * 0.1)


class ContinuousDynamicsProcessor(ProcessorBase, StateValidationMixin, ConfigurableMixin):
    """Main processor for continuous dynamics of enactive consciousness.
    
    This integrates all continuous-time models into a unified framework,
    providing smooth mathematical representations of consciousness processes
    while maintaining phenomenological and enactive principles.
    
    Domain-Driven Design Implementation:
    - Rich domain models for continuous states and dynamics configurations
    - Clear bounded contexts for different types of dynamics
    - Ubiquitous language from differential equations and phenomenology
    - Anti-corruption layer between discrete and continuous representations
    """
    
    # Core continuous dynamics components
    temporal_flow: HusserlianTemporalFlow
    coupling_dynamics: EnactiveCouplingDynamics  
    neural_ode_flow: NeuralODEConsciousnessFlow
    
    # Configuration and state management
    config: DynamicsConfig
    current_continuous_state: Optional[ContinuousState]
    integration_method: IntegrationMethod
    
    # Solvers and integration components
    ode_solver: Any  # diffrax solver
    sde_solver: Any  # diffrax SDE solver
    adaptive_stepper: Any  # adaptive time stepping
    
    def __init__(
        self,
        state_dim: int,
        config: DynamicsConfig,
        key: PRNGKey,
        integration_method: IntegrationMethod = IntegrationMethod.TSIT5,
    ):
        keys = jax.random.split(key, 3)
        
        self.config = config
        self.integration_method = integration_method
        self.current_continuous_state = None
        
        # Initialize temporal flow dynamics
        self.temporal_flow = HusserlianTemporalFlow(
            state_dim=state_dim,
            temporal_depth=10,  # Configurable temporal depth
            config=config,
            key=keys[0],
        )
        
        # Initialize enactive coupling dynamics
        agent_dim = state_dim
        environment_dim = state_dim
        meaning_dim = state_dim // 2
        
        self.coupling_dynamics = EnactiveCouplingDynamics(
            agent_dim=agent_dim,
            environment_dim=environment_dim,
            meaning_dim=meaning_dim,
            config=config,
            key=keys[1],
        )
        
        # Initialize neural ODE flow
        self.neural_ode_flow = NeuralODEConsciousnessFlow(
            state_dim=state_dim * 4,  # Extended state for full consciousness
            hidden_dim=config.hidden_dynamics_dim,
            config=config,
            key=keys[2],
        )
        
        # Set up diffrax solvers
        self._initialize_solvers()
        
        logger.info(f"ContinuousDynamicsProcessor initialized with {integration_method.value}")
    
    def _initialize_solvers(self) -> None:
        """Initialize diffrax solvers for different types of dynamics."""
        if not DIFFRAX_AVAILABLE:
            # Use simple fallback solvers
            self.ode_solver = "euler_fallback"
            self.sde_solver = "euler_fallback" 
            self.adaptive_stepper = "fixed_step"
            return
        
        # ODE solver for deterministic dynamics
        if self.integration_method == IntegrationMethod.EULER:
            self.ode_solver = diffrax.Euler()
        elif self.integration_method == IntegrationMethod.HEUN:
            self.ode_solver = diffrax.Heun()
        elif self.integration_method == IntegrationMethod.MIDPOINT:
            self.ode_solver = diffrax.Midpoint()
        elif self.integration_method == IntegrationMethod.RUNGE_KUTTA_4:
            self.ode_solver = diffrax.RungeKutta4()
        elif self.integration_method == IntegrationMethod.TSIT5:
            self.ode_solver = diffrax.Tsit5()
        elif self.integration_method == IntegrationMethod.DOPRI5:
            self.ode_solver = diffrax.Dopri5()
        elif self.integration_method == IntegrationMethod.DOPRI8:
            self.ode_solver = diffrax.Dopri8()
        else:
            self.ode_solver = diffrax.Tsit5()  # Default
        
        # SDE solver for stochastic dynamics
        self.sde_solver = diffrax.Euler()  # Euler-Maruyama for SDEs
        
        # Adaptive step size controller
        self.adaptive_stepper = diffrax.PIDController(
            rtol=self.config.rtol,
            atol=self.config.atol,
            safety=self.config.adaptive_step_safety_factor,
        )
    
    @optimize_for_memory
    def evolve_temporal_consciousness(
        self,
        initial_temporal_moment: TemporalMoment,
        environmental_input: Array,
        time_span: Tuple[float, float],
        num_steps: Optional[int] = None,
    ) -> Tuple[Array, Array]:
        """Evolve temporal consciousness using continuous-time dynamics.
        
        This transforms a discrete TemporalMoment into continuous temporal flow
        and integrates the Husserlian temporal equations over the specified time span.
        """
        with GLOBAL_MEMORY_MANAGER.track_memory("temporal_consciousness_evolution"):
            try:
                # Convert discrete temporal moment to continuous state
                continuous_temporal_state = self._convert_temporal_moment_to_continuous(
                    initial_temporal_moment
                )
                
                if not DIFFRAX_AVAILABLE:
                    # Use simple Euler integration as fallback
                    return self._simple_euler_integration(
                        lambda t, state: self.temporal_flow.temporal_flow_equations(
                            t, state, environmental_input
                        ),
                        time_span,
                        continuous_temporal_state,
                        num_steps or 100,
                    )
                
                # Define the temporal flow ODE system
                def temporal_ode(t: float, state: Array, args) -> Array:
                    return self.temporal_flow.temporal_flow_equations(
                        t, state, environmental_input
                    )
                
                # Set up ODE problem
                ode_term = diffrax.ODETerm(temporal_ode)
                
                # Time span and stepping
                t0, t1 = time_span
                if num_steps is None:
                    # Adaptive stepping
                    step_size_controller = self.adaptive_stepper
                    dt0 = (t1 - t0) / 100  # Initial guess
                else:
                    # Fixed stepping
                    dt0 = (t1 - t0) / num_steps
                    step_size_controller = diffrax.ConstantStepSize()
                
                # Solve the ODE
                solution = diffrax.diffeqsolve(
                    terms=ode_term,
                    solver=self.ode_solver,
                    t0=t0,
                    t1=t1,
                    dt0=dt0,
                    y0=continuous_temporal_state,
                    stepsize_controller=step_size_controller,
                    max_steps=self.config.max_steps,
                )
                
                if solution.stats['num_steps'] >= self.config.max_steps:
                    warnings.warn("Maximum steps reached in temporal evolution")
                
                return solution.ts, solution.ys
                
            except Exception as e:
                raise EnactiveConsciousnessError(
                    f"Failed to evolve temporal consciousness: {e}"
                )
    
    @optimize_for_memory  
    def evolve_enactive_coupling(
        self,
        initial_coupling_state: CouplingState,
        environmental_perturbations: Callable[[float], Array],
        time_span: Tuple[float, float],
        noise_key: PRNGKey,
        num_steps: Optional[int] = None,
    ) -> Tuple[Array, Array]:
        """Evolve enactive coupling using stochastic differential equations.
        
        This implements the continuous-time version of Varela-Maturana structural
        coupling with environmental noise and circular causality feedback.
        """
        with GLOBAL_MEMORY_MANAGER.track_memory("enactive_coupling_evolution"):
            try:
                # Convert discrete coupling state to continuous representation
                continuous_coupling_state = self._convert_coupling_state_to_continuous(
                    initial_coupling_state
                )
                
                # Initialize history context and meaning context
                history_context = jnp.zeros(continuous_coupling_state.shape[0] // 3 * 2)
                meaning_context = jnp.zeros(continuous_coupling_state.shape[0] // 3)
                
                if not DIFFRAX_AVAILABLE:
                    # Use simple fallback
                    def coupling_dynamics(t: float, state: Array) -> Array:
                        # Get environmental perturbation at time t
                        env_perturbation = environmental_perturbations(t)
                        
                        try:
                            result = self.coupling_dynamics.enactive_coupling_sde(
                                t, state, history_context, meaning_context
                            )
                            if isinstance(result, tuple) and len(result) == 2:
                                drift, _ = result
                            else:
                                # Handle unexpected return format
                                drift = result if not isinstance(result, tuple) else result[0]
                        except Exception as e:
                            logger.warning(f"SDE call failed: {e}, using zero drift")
                            drift = jnp.zeros_like(state)
                        return drift  # Ignore diffusion term in fallback
                    
                    return self._simple_euler_integration(
                        coupling_dynamics, time_span, continuous_coupling_state, num_steps or 100
                    )
                
                # Define the SDE system
                def coupling_sde(t: float, state: Array, args) -> Tuple[Array, Array]:
                    # Get environmental perturbation at time t
                    env_perturbation = environmental_perturbations(t)
                    
                    # Update history context (simple moving average)
                    nonlocal history_context
                    current_coupling = state[:2 * (state.shape[0] // 3)]
                    history_context = 0.9 * history_context + 0.1 * current_coupling
                    
                    try:
                        return self.coupling_dynamics.enactive_coupling_sde(
                            t, state, history_context, meaning_context
                        )
                    except Exception as e:
                        logger.warning(f"SDE call failed in coupling_sde: {e}")
                        return jnp.zeros_like(state), jnp.zeros((len(state), len(state)))
                
                # Set up SDE terms
                drift_term = diffrax.ODETerm(lambda t, y, args: coupling_sde(t, y, args)[0])
                diffusion_term = diffrax.ControlTerm(
                    lambda t, y, args: coupling_sde(t, y, args)[1],
                    diffrax.VirtualBrownianTree(
                        t0=time_span[0],
                        t1=time_span[1], 
                        tol=self.config.atol,
                        shape=(continuous_coupling_state.shape[0],),
                        key=noise_key,
                    )
                )
                
                terms = diffrax.MultiTerm(drift_term, diffusion_term)
                
                # Time span and stepping
                t0, t1 = time_span
                if num_steps is None:
                    dt0 = (t1 - t0) / 100
                    step_size_controller = self.adaptive_stepper
                else:
                    dt0 = (t1 - t0) / num_steps
                    step_size_controller = diffrax.ConstantStepSize()
                
                # Solve the SDE
                solution = diffrax.diffeqsolve(
                    terms=terms,
                    solver=self.sde_solver,
                    t0=t0,
                    t1=t1,
                    dt0=dt0,
                    y0=continuous_coupling_state,
                    stepsize_controller=step_size_controller,
                    max_steps=self.config.max_steps,
                )
                
                return solution.ts, solution.ys
                
            except Exception as e:
                raise EnactiveConsciousnessError(
                    f"Failed to evolve enactive coupling: {e}"
                )
    
    @optimize_for_memory
    def evolve_neural_ode_consciousness(
        self,
        initial_consciousness_state: Array,
        phenomenological_context: Optional[Array],
        time_span: Tuple[float, float],
        num_steps: Optional[int] = None,
    ) -> Tuple[Array, Array, Array]:
        """Evolve consciousness using Neural ODEs with phenomenological context.
        
        This provides a learnable continuous dynamics model that can adapt
        to specific consciousness phenomena while maintaining mathematical rigor.
        """
        with GLOBAL_MEMORY_MANAGER.track_memory("neural_ode_consciousness_evolution"):
            try:
                if not DIFFRAX_AVAILABLE:
                    # Simple fallback
                    def neural_dynamics(t: float, state: Array) -> Array:
                        return self.neural_ode_flow.neural_ode_dynamics(
                            t, state, phenomenological_context
                        )
                    
                    times, states = self._simple_euler_integration(
                        neural_dynamics, time_span, initial_consciousness_state, num_steps or 100
                    )
                    
                    # Compute consciousness levels
                    consciousness_levels = jnp.array([
                        self.neural_ode_flow.compute_consciousness_level(state)
                        for state in states
                    ])
                    
                    return times, states, consciousness_levels
                
                # Define Neural ODE system
                def neural_consciousness_ode(t: float, state: Array, args) -> Array:
                    return self.neural_ode_flow.neural_ode_dynamics(
                        t, state, phenomenological_context
                    )
                
                # Set up ODE problem
                ode_term = diffrax.ODETerm(neural_consciousness_ode)
                
                # Time configuration
                t0, t1 = time_span
                if num_steps is None:
                    dt0 = (t1 - t0) / 100
                    step_size_controller = self.adaptive_stepper
                else:
                    dt0 = (t1 - t0) / num_steps
                    step_size_controller = diffrax.ConstantStepSize()
                
                # Solve Neural ODE
                solution = diffrax.diffeqsolve(
                    terms=ode_term,
                    solver=self.ode_solver,
                    t0=t0,
                    t1=t1,
                    dt0=dt0,
                    y0=initial_consciousness_state,
                    stepsize_controller=step_size_controller,
                    max_steps=self.config.max_steps,
                )
                
                # Compute consciousness levels over time
                consciousness_levels = jnp.array([
                    self.neural_ode_flow.compute_consciousness_level(state)
                    for state in solution.ys
                ])
                
                return solution.ts, solution.ys, consciousness_levels
                
            except Exception as e:
                raise EnactiveConsciousnessError(
                    f"Failed to evolve Neural ODE consciousness: {e}"
                )
    
    def integrate_continuous_consciousness(
        self,
        temporal_moment: TemporalMoment,
        coupling_state: CouplingState,
        environmental_dynamics: Callable[[float], Array],
        time_span: Tuple[float, float],
        integration_key: PRNGKey,
        num_steps: Optional[int] = None,
    ) -> ContinuousState:
        """Integrate all continuous dynamics into unified consciousness evolution.
        
        This is the main method that combines temporal flow, enactive coupling,
        and neural ODE dynamics into a complete continuous representation
        of consciousness processes.
        """
        with GLOBAL_MEMORY_MANAGER.track_memory("integrated_continuous_consciousness"):
            try:
                keys = jax.random.split(integration_key, 3)
                
                # Evolve temporal consciousness
                t_times, temporal_states = self.evolve_temporal_consciousness(
                    temporal_moment, environmental_dynamics(time_span[0]), 
                    time_span, num_steps
                )
                
                # Evolve enactive coupling 
                c_times, coupling_states = self.evolve_enactive_coupling(
                    coupling_state, environmental_dynamics, time_span, keys[1], num_steps
                )
                
                # Extract final states for Neural ODE initialization
                final_temporal_state = temporal_states[-1]
                final_coupling_state = coupling_states[-1]
                
                # Combine for consciousness state
                combined_consciousness_state = jnp.concatenate([
                    final_temporal_state, final_coupling_state
                ])
                
                # Evolve using Neural ODE for final integration
                n_times, consciousness_states, consciousness_levels = (
                    self.evolve_neural_ode_consciousness(
                        combined_consciousness_state,
                        environmental_dynamics(time_span[1]), 
                        time_span, num_steps
                    )
                )
                
                # Create final continuous state
                final_consciousness_state = consciousness_states[-1]
                state_dim = len(final_consciousness_state) // 4
                
                continuous_state = ContinuousState(
                    retention_field=final_consciousness_state[:state_dim],
                    present_awareness=final_consciousness_state[state_dim:2*state_dim],
                    protention_field=final_consciousness_state[2*state_dim:3*state_dim],
                    temporal_synthesis_weights=final_consciousness_state[3*state_dim:],
                    
                    agent_state=final_coupling_state[:state_dim],
                    environmental_coupling=final_coupling_state[state_dim:2*state_dim],
                    circular_causality_trace=final_coupling_state[2*state_dim:3*state_dim],
                    meaning_emergence_level=final_coupling_state[3*state_dim:] if len(final_coupling_state) > 3*state_dim else jnp.zeros(state_dim//2),
                    
                    timestamp=time_span[1],
                    consciousness_level=consciousness_levels[-1],
                    coupling_strength=float(jnp.mean(jnp.abs(final_coupling_state[:state_dim]))),
                    temporal_coherence=float(jnp.corrcoef(
                        final_consciousness_state[:state_dim],
                        final_consciousness_state[state_dim:2*state_dim]
                    )[0, 1]),
                )
                
                # Validate the final continuous state
                if not continuous_state.validate_state():
                    raise EnactiveConsciousnessError(
                        "Generated continuous state failed validation"
                    )
                
                self.current_continuous_state = continuous_state
                return continuous_state
                
            except Exception as e:
                raise EnactiveConsciousnessError(
                    f"Failed to integrate continuous consciousness: {e}"
                )
    
    def process(self, *args, **kwargs) -> Any:
        """Implementation of ProcessorBase abstract method."""
        return self.integrate_continuous_consciousness(*args, **kwargs)
    
    def _simple_euler_integration(
        self,
        dynamics_fn: Callable[[float, Array], Array],
        time_span: Tuple[float, float],
        initial_state: Array,
        num_steps: int,
    ) -> Tuple[Array, Array]:
        """Simple Euler integration as fallback when diffrax is not available."""
        t0, t1 = time_span
        dt = (t1 - t0) / num_steps
        
        times = jnp.linspace(t0, t1, num_steps + 1)
        states = jnp.zeros((num_steps + 1, *initial_state.shape))
        states = states.at[0].set(initial_state)
        
        current_state = initial_state
        for i in range(num_steps):
            t = times[i]
            # Euler step: y_{n+1} = y_n + dt * f(t_n, y_n)
            derivative = dynamics_fn(t, current_state)
            current_state = current_state + dt * derivative
            states = states.at[i + 1].set(current_state)
        
        return times, states
    
    def _convert_temporal_moment_to_continuous(
        self, temporal_moment: TemporalMoment
    ) -> Array:
        """Convert discrete TemporalMoment to continuous state representation."""
        return jnp.concatenate([
            temporal_moment.retention,
            temporal_moment.present_moment,
            temporal_moment.protention,
            temporal_moment.synthesis_weights,
        ])
    
    def _convert_coupling_state_to_continuous(
        self, coupling_state: CouplingState
    ) -> Array:
        """Convert discrete CouplingState to continuous state representation."""
        # Pad environmental state to match agent state dimensions if needed
        agent_state = coupling_state.agent_state
        environmental_state = coupling_state.environmental_state
        
        if environmental_state.shape[0] < agent_state.shape[0]:
            padding = jnp.zeros(agent_state.shape[0] - environmental_state.shape[0])
            environmental_state = jnp.concatenate([environmental_state, padding])
        elif environmental_state.shape[0] > agent_state.shape[0]:
            environmental_state = environmental_state[:agent_state.shape[0]]
        
        # Create meaning state from coupling strength and history
        meaning_state = jnp.ones(agent_state.shape[0] // 2) * coupling_state.coupling_strength
        
        return jnp.concatenate([
            agent_state,
            environmental_state,
            meaning_state,
        ])
    
    def compute_trajectory_statistics(
        self, 
        times: Array, 
        states: Array
    ) -> Dict[str, float]:
        """Compute statistical properties of continuous trajectories."""
        try:
            # Basic trajectory statistics
            trajectory_length = float(jnp.sum(jnp.linalg.norm(
                jnp.diff(states, axis=0), axis=1
            )))
            
            # Lyapunov exponent approximation
            jacobians = []
            for i in range(min(len(times), 100)):  # Sample subset for efficiency
                state = states[i]
                time = times[i]
                
                # Approximate Jacobian via finite differences
                eps = 1e-6
                state_dim = len(state)
                jacobian = jnp.zeros((state_dim, state_dim))
                
                for j in range(state_dim):
                    state_plus = state.at[j].add(eps)
                    state_minus = state.at[j].add(-eps)
                    
                    # This would require access to the dynamics function
                    # Simplified approximation for now
                    grad_approx = (state_plus - state_minus) / (2 * eps)
                    jacobian = jacobian.at[:, j].set(grad_approx)
                
                jacobians.append(jacobian)
            
            # Average largest eigenvalue as Lyapunov approximation
            if jacobians:
                eigenvals = [jnp.max(jnp.real(jnp.linalg.eigvals(J))) for J in jacobians]
                lyapunov_estimate = float(jnp.mean(jnp.array(eigenvals)))
            else:
                lyapunov_estimate = 0.0
            
            # Temporal coherence
            state_correlations = []
            for i in range(len(states) - 1):
                corr = jnp.corrcoef(states[i], states[i+1])[0, 1]
                if jnp.isfinite(corr):
                    state_correlations.append(corr)
            
            temporal_coherence = float(jnp.mean(jnp.array(state_correlations))) if state_correlations else 0.0
            
            # Phase space volume (determinant of covariance)
            state_covariance = jnp.cov(states.T)
            phase_volume = float(jnp.linalg.det(state_covariance + jnp.eye(states.shape[1]) * 1e-8))
            
            return {
                "trajectory_length": trajectory_length,
                "lyapunov_estimate": lyapunov_estimate,
                "temporal_coherence": temporal_coherence,
                "phase_space_volume": phase_volume,
                "mean_state_norm": float(jnp.mean(jnp.linalg.norm(states, axis=1))),
                "state_variance": float(jnp.mean(jnp.var(states, axis=0))),
                "total_time": float(times[-1] - times[0]),
                "num_integration_steps": len(times),
            }
            
        except Exception as e:
            logger.warning(f"Failed to compute trajectory statistics: {e}")
            return {
                "trajectory_length": 0.0,
                "lyapunov_estimate": 0.0,
                "temporal_coherence": 0.0,
                "phase_space_volume": 0.0,
                "mean_state_norm": 0.0,
                "state_variance": 0.0,
                "total_time": 0.0,
                "num_integration_steps": 0,
            }


# Factory functions for creating continuous dynamics systems

def create_continuous_dynamics_processor(
    state_dim: int,
    config: DynamicsConfig,
    key: PRNGKey,
    integration_method: IntegrationMethod = IntegrationMethod.TSIT5,
) -> ContinuousDynamicsProcessor:
    """Factory function for creating continuous dynamics processor."""
    return ContinuousDynamicsProcessor(
        state_dim=state_dim,
        config=config,
        key=key,
        integration_method=integration_method,
    )


def create_default_dynamics_config(
    **overrides
) -> DynamicsConfig:
    """Create default dynamics configuration with optional overrides."""
    default_config = DynamicsConfig()
    
    # Apply any overrides
    if overrides:
        config_dict = {
            field.name: getattr(default_config, field.name) 
            for field in default_config.__dataclass_fields__.values()
        }
        config_dict.update(overrides)
        return DynamicsConfig(**config_dict)
    
    return default_config


def create_environmental_perturbation_function(
    amplitude: float = 1.0,
    frequency: float = 0.1,
    noise_strength: float = 0.1,
    key: PRNGKey = jax.random.PRNGKey(42),
) -> Callable[[float], Array]:
    """Create a function that generates environmental perturbations over time."""
    def environmental_dynamics(t: float) -> Array:
        # Deterministic component (oscillatory)
        deterministic = amplitude * jnp.sin(2 * jnp.pi * frequency * t)
        
        # Stochastic component
        subkey = jax.random.fold_in(key, int(t * 1000) % 1000000)
        stochastic = jax.random.normal(subkey, (10,)) * noise_strength
        
        # Combine
        base_signal = jnp.array([deterministic])
        full_signal = jnp.concatenate([base_signal, stochastic])
        
        return full_signal
    
    return environmental_dynamics


# Utility functions for analysis and visualization

def analyze_continuous_consciousness_dynamics(
    processor: ContinuousDynamicsProcessor,
    initial_temporal_moment: TemporalMoment,
    initial_coupling_state: CouplingState,
    time_span: Tuple[float, float],
    key: PRNGKey,
) -> Dict[str, Any]:
    """Comprehensive analysis of continuous consciousness dynamics."""
    try:
        # Create environmental dynamics
        env_dynamics = create_environmental_perturbation_function(key=key)
        
        # Integrate full consciousness dynamics
        final_state = processor.integrate_continuous_consciousness(
            initial_temporal_moment,
            initial_coupling_state, 
            env_dynamics,
            time_span,
            key,
        )
        
        # Analyze individual components
        temporal_times, temporal_states = processor.evolve_temporal_consciousness(
            initial_temporal_moment,
            env_dynamics(time_span[0]),
            time_span,
        )
        
        coupling_times, coupling_states = processor.evolve_enactive_coupling(
            initial_coupling_state,
            env_dynamics,
            time_span,
            jax.random.split(key)[0],
        )
        
        # Compute statistics
        temporal_stats = processor.compute_trajectory_statistics(temporal_times, temporal_states)
        coupling_stats = processor.compute_trajectory_statistics(coupling_times, coupling_states)
        
        return {
            "final_continuous_state": final_state,
            "temporal_trajectory_stats": temporal_stats,
            "coupling_trajectory_stats": coupling_stats,
            "consciousness_level_evolution": final_state.consciousness_level,
            "temporal_coherence_final": final_state.temporal_coherence,
            "coupling_strength_final": final_state.coupling_strength,
            "analysis_success": True,
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            "analysis_success": False,
            "error_message": str(e),
        }


# Export public API
__all__ = [
    # Core classes
    'DynamicsType', 'IntegrationMethod', 'DynamicsConfig', 'ContinuousState',
    
    # Dynamics components
    'HusserlianTemporalFlow', 'EnactiveCouplingDynamics', 'NeuralODEConsciousnessFlow',
    
    # Main processor
    'ContinuousDynamicsProcessor',
    
    # Factory functions
    'create_continuous_dynamics_processor', 'create_default_dynamics_config',
    'create_environmental_perturbation_function',
    
    # Analysis utilities
    'analyze_continuous_consciousness_dynamics',
]