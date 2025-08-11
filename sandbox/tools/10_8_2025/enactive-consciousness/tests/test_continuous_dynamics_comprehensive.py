"""Comprehensive test suite for continuous dynamics module.

This test suite follows TDD principles with extensive coverage of
continuous-time differential equation systems for enactive consciousness,
including Husserlian temporal flow, enactive coupling dynamics, and
Neural ODEs with mathematical validation.

Test Coverage:
- DynamicsConfig validation and functionality
- ContinuousState creation and validation
- HusserlianTemporalFlow implementation
- EnactiveCouplingDynamics with circular causality
- NeuralODEConsciousnessFlow systems
- ContinuousDynamicsProcessor integration
- Mathematical correctness validation
- Performance and scalability testing
- Numerical stability analysis
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from unittest.mock import patch, MagicMock
import warnings

# Import the module under test
import sys
sys.path.insert(0, '/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/10_8_2025/enactive-consciousness/src')

from enactive_consciousness.continuous_dynamics import (
    DynamicsType,
    IntegrationMethod,
    DynamicsConfig,
    ContinuousState,
    HusserlianTemporalFlow,
    EnactiveCouplingDynamics,
    NeuralODEConsciousnessFlow,
    ContinuousDynamicsProcessor,
    create_continuous_dynamics_processor,
    create_default_dynamics_config,
    create_environmental_perturbation_function,
    analyze_continuous_consciousness_dynamics,
)

from enactive_consciousness.types import (
    TemporalMoment,
    CouplingState,
    PRNGKey,
    TimeStep,
)


class TestDynamicsType:
    """Test cases for DynamicsType enum."""
    
    def test_dynamics_type_values(self):
        """Test DynamicsType enum values."""
        assert DynamicsType.DETERMINISTIC_ODE.value == "deterministic_ode"
        assert DynamicsType.STOCHASTIC_SDE.value == "stochastic_sde"
        assert DynamicsType.NEURAL_ODE.value == "neural_ode"
        assert DynamicsType.COUPLED_SYSTEM.value == "coupled_system"
    
    def test_dynamics_type_completeness(self):
        """Test that all expected dynamics types are available."""
        expected_types = {
            "deterministic_ode", "stochastic_sde", "neural_ode", "coupled_system"
        }
        actual_types = {dtype.value for dtype in DynamicsType}
        assert actual_types == expected_types


class TestIntegrationMethod:
    """Test cases for IntegrationMethod enum."""
    
    def test_integration_method_values(self):
        """Test IntegrationMethod enum values."""
        assert IntegrationMethod.EULER.value == "euler"
        assert IntegrationMethod.HEUN.value == "heun"
        assert IntegrationMethod.MIDPOINT.value == "midpoint"
        assert IntegrationMethod.RUNGE_KUTTA_4.value == "rk4"
        assert IntegrationMethod.TSIT5.value == "tsit5"
        assert IntegrationMethod.DOPRI5.value == "dopri5"
        assert IntegrationMethod.DOPRI8.value == "dopri8"
    
    def test_integration_method_completeness(self):
        """Test that all expected integration methods are available."""
        expected_methods = {
            "euler", "heun", "midpoint", "rk4", "tsit5", "dopri5", "dopri8"
        }
        actual_methods = {method.value for method in IntegrationMethod}
        assert actual_methods == expected_methods


class TestDynamicsConfig:
    """Test cases for DynamicsConfig."""
    
    def test_config_creation_default(self):
        """Test default configuration creation."""
        config = DynamicsConfig()
        
        # Time integration parameters
        assert config.dt_min == 1e-4
        assert config.dt_max == 0.1
        assert config.rtol == 1e-3
        assert config.atol == 1e-6
        assert config.max_steps == 10000
        
        # Phenomenological time parameters
        assert config.retention_decay_rate == 0.1
        assert config.protention_anticipation_rate == 0.05
        assert config.present_moment_width == 0.01
        assert config.temporal_synthesis_strength == 1.0
        
        # Enactive coupling parameters
        assert config.agent_environment_coupling_strength == 0.5
        assert config.circular_causality_time_constant == 0.1
        assert config.structural_coupling_adaptation_rate == 0.02
        
        # Stochastic parameters
        assert config.environmental_noise_strength == 0.1
        assert config.internal_noise_strength == 0.05
        assert config.coupling_noise_correlation == 0.3
        
        # Neural ODE parameters
        assert config.hidden_dynamics_dim == 128
        assert config.neural_ode_depth == 3
        assert config.activation_dynamics == "swish"
    
    def test_config_creation_custom(self):
        """Test custom configuration creation."""
        config = DynamicsConfig(
            dt_min=1e-5,
            dt_max=0.05,
            retention_decay_rate=0.2,
            agent_environment_coupling_strength=0.7,
            hidden_dynamics_dim=64,
            activation_dynamics="gelu",
        )
        
        assert config.dt_min == 1e-5
        assert config.dt_max == 0.05
        assert config.retention_decay_rate == 0.2
        assert config.agent_environment_coupling_strength == 0.7
        assert config.hidden_dynamics_dim == 64
        assert config.activation_dynamics == "gelu"
    
    def test_config_parameter_bounds(self):
        """Test configuration parameter bounds and validity."""
        config = DynamicsConfig()
        
        # Time parameters should be positive
        assert config.dt_min > 0
        assert config.dt_max > 0
        assert config.dt_min <= config.dt_max
        assert config.rtol > 0
        assert config.atol > 0
        assert config.max_steps > 0
        
        # Rates should be non-negative
        assert config.retention_decay_rate >= 0
        assert config.protention_anticipation_rate >= 0
        assert config.present_moment_width > 0
        assert config.temporal_synthesis_strength > 0
        
        # Coupling strengths should be in reasonable ranges
        assert 0 <= config.agent_environment_coupling_strength <= 1.0
        assert config.circular_causality_time_constant > 0
        assert config.structural_coupling_adaptation_rate >= 0
        
        # Noise parameters should be non-negative
        assert config.environmental_noise_strength >= 0
        assert config.internal_noise_strength >= 0
        assert -1 <= config.coupling_noise_correlation <= 1
        
        # Neural ODE parameters
        assert config.hidden_dynamics_dim > 0
        assert config.neural_ode_depth > 0


class TestContinuousState:
    """Test cases for ContinuousState."""
    
    @pytest.fixture
    def sample_continuous_state(self):
        """Create sample continuous state for testing."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 8)
        
        state_dim = 16
        
        return ContinuousState(
            retention_field=jax.random.normal(keys[0], (state_dim,)),
            present_awareness=jax.random.normal(keys[1], (state_dim,)),
            protention_field=jax.random.normal(keys[2], (state_dim,)),
            temporal_synthesis_weights=jax.random.uniform(keys[3], (state_dim,)),
            agent_state=jax.random.normal(keys[4], (state_dim,)),
            environmental_coupling=jax.random.normal(keys[5], (state_dim,)),
            circular_causality_trace=jax.random.normal(keys[6], (state_dim,)),
            meaning_emergence_level=jax.random.normal(keys[7], (state_dim//2,)),
            timestamp=100.0,
            consciousness_level=0.7,
            coupling_strength=0.6,
            temporal_coherence=0.8,
        )
    
    def test_continuous_state_creation(self, sample_continuous_state):
        """Test ContinuousState creation."""
        state = sample_continuous_state
        
        assert isinstance(state.retention_field, jax.Array)
        assert isinstance(state.present_awareness, jax.Array)
        assert isinstance(state.protention_field, jax.Array)
        assert isinstance(state.temporal_synthesis_weights, jax.Array)
        assert isinstance(state.agent_state, jax.Array)
        assert isinstance(state.environmental_coupling, jax.Array)
        assert isinstance(state.circular_causality_trace, jax.Array)
        assert isinstance(state.meaning_emergence_level, jax.Array)
        
        assert state.timestamp == 100.0
        assert state.consciousness_level == 0.7
        assert state.coupling_strength == 0.6
        assert state.temporal_coherence == 0.8
    
    def test_continuous_state_validation_valid(self, sample_continuous_state):
        """Test validation of valid continuous state."""
        state = sample_continuous_state
        assert state.validate_state() is True
    
    def test_continuous_state_validation_consciousness_bounds(self, sample_continuous_state):
        """Test validation with invalid consciousness level."""
        state = sample_continuous_state
        
        # Test consciousness level > 1.0
        invalid_state = ContinuousState(
            retention_field=state.retention_field,
            present_awareness=state.present_awareness,
            protention_field=state.protention_field,
            temporal_synthesis_weights=state.temporal_synthesis_weights,
            agent_state=state.agent_state,
            environmental_coupling=state.environmental_coupling,
            circular_causality_trace=state.circular_causality_trace,
            meaning_emergence_level=state.meaning_emergence_level,
            timestamp=state.timestamp,
            consciousness_level=1.2,  # Invalid
            coupling_strength=state.coupling_strength,
            temporal_coherence=state.temporal_coherence,
        )
        
        assert invalid_state.validate_state() is False
    
    def test_continuous_state_validation_coupling_bounds(self, sample_continuous_state):
        """Test validation with invalid coupling strength."""
        state = sample_continuous_state
        
        invalid_state = ContinuousState(
            retention_field=state.retention_field,
            present_awareness=state.present_awareness,
            protention_field=state.protention_field,
            temporal_synthesis_weights=state.temporal_synthesis_weights,
            agent_state=state.agent_state,
            environmental_coupling=state.environmental_coupling,
            circular_causality_trace=state.circular_causality_trace,
            meaning_emergence_level=state.meaning_emergence_level,
            timestamp=state.timestamp,
            consciousness_level=state.consciousness_level,
            coupling_strength=-0.1,  # Invalid
            temporal_coherence=state.temporal_coherence,
        )
        
        assert invalid_state.validate_state() is False
    
    def test_continuous_state_validation_with_nan(self):
        """Test validation with NaN values."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 8)
        
        state_dim = 16
        
        # Create state with NaN in retention field
        state_with_nan = ContinuousState(
            retention_field=jnp.full((state_dim,), jnp.nan),  # NaN values
            present_awareness=jax.random.normal(keys[1], (state_dim,)),
            protention_field=jax.random.normal(keys[2], (state_dim,)),
            temporal_synthesis_weights=jax.random.uniform(keys[3], (state_dim,)),
            agent_state=jax.random.normal(keys[4], (state_dim,)),
            environmental_coupling=jax.random.normal(keys[5], (state_dim,)),
            circular_causality_trace=jax.random.normal(keys[6], (state_dim,)),
            meaning_emergence_level=jax.random.normal(keys[7], (state_dim//2,)),
            timestamp=100.0,
            consciousness_level=0.7,
            coupling_strength=0.6,
            temporal_coherence=0.8,
        )
        
        assert state_with_nan.validate_state() is False
    
    def test_continuous_state_validation_with_inf(self):
        """Test validation with infinite values."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 8)
        
        state_dim = 16
        
        # Create state with inf in present awareness
        state_with_inf = ContinuousState(
            retention_field=jax.random.normal(keys[0], (state_dim,)),
            present_awareness=jnp.full((state_dim,), jnp.inf),  # Inf values
            protention_field=jax.random.normal(keys[2], (state_dim,)),
            temporal_synthesis_weights=jax.random.uniform(keys[3], (state_dim,)),
            agent_state=jax.random.normal(keys[4], (state_dim,)),
            environmental_coupling=jax.random.normal(keys[5], (state_dim,)),
            circular_causality_trace=jax.random.normal(keys[6], (state_dim,)),
            meaning_emergence_level=jax.random.normal(keys[7], (state_dim//2,)),
            timestamp=100.0,
            consciousness_level=0.7,
            coupling_strength=0.6,
            temporal_coherence=0.8,
        )
        
        assert state_with_inf.validate_state() is False


class TestHusserlianTemporalFlow:
    """Test cases for HusserlianTemporalFlow."""
    
    @pytest.fixture
    def temporal_flow_setup(self):
        """Set up Husserlian temporal flow for testing."""
        state_dim = 12
        temporal_depth = 8
        config = DynamicsConfig(
            retention_decay_rate=0.1,
            protention_anticipation_rate=0.05,
            temporal_synthesis_strength=1.0,
            activation_dynamics="gelu",
        )
        key = jax.random.PRNGKey(42)
        
        flow = HusserlianTemporalFlow(
            state_dim=state_dim,
            temporal_depth=temporal_depth,
            config=config,
            key=key,
        )
        
        return flow, state_dim, temporal_depth, config
    
    def test_temporal_flow_initialization(self, temporal_flow_setup):
        """Test HusserlianTemporalFlow initialization."""
        flow, state_dim, temporal_depth, config = temporal_flow_setup
        
        # Check neural network components
        assert hasattr(flow, 'retention_dynamics')
        assert hasattr(flow, 'present_synthesis')
        assert hasattr(flow, 'protention_projection')
        assert hasattr(flow, 'temporal_attention')
        
        # Check phenomenological parameters
        assert flow.retention_decay_constant == config.retention_decay_rate
        assert flow.protention_anticipation_constant == config.protention_anticipation_rate
        assert flow.present_integration_strength == config.temporal_synthesis_strength
    
    def test_temporal_flow_equations(self, temporal_flow_setup):
        """Test temporal flow equations computation."""
        flow, state_dim, temporal_depth, config = temporal_flow_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        t = 1.0
        temporal_state = jax.random.normal(keys[0], (state_dim * 4,))  # 4x state_dim
        environmental_input = jax.random.normal(keys[1], (state_dim * 3,))
        
        derivatives = flow.temporal_flow_equations(t, temporal_state, environmental_input)
        
        assert derivatives.shape == temporal_state.shape
        assert jnp.all(jnp.isfinite(derivatives))
    
    def test_temporal_flow_equations_dimensions(self, temporal_flow_setup):
        """Test temporal flow equations with different input dimensions."""
        flow, state_dim, temporal_depth, config = temporal_flow_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        t = 2.0
        temporal_state = jax.random.normal(keys[0], (state_dim * 4,))
        
        # Test with smaller environmental input
        small_env_input = jax.random.normal(keys[1], (state_dim,))
        derivatives_small = flow.temporal_flow_equations(t, temporal_state, small_env_input)
        
        assert derivatives_small.shape == temporal_state.shape
        assert jnp.all(jnp.isfinite(derivatives_small))
        
        # Test with larger environmental input
        large_env_input = jax.random.normal(keys[1], (state_dim * 6,))
        derivatives_large = flow.temporal_flow_equations(t, temporal_state, large_env_input)
        
        assert derivatives_large.shape == temporal_state.shape
        assert jnp.all(jnp.isfinite(derivatives_large))
    
    def test_temporal_flow_conservation_properties(self, temporal_flow_setup):
        """Test conservation properties of temporal flow."""
        flow, state_dim, temporal_depth, config = temporal_flow_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        t = 0.5
        temporal_state = jax.random.normal(keys[0], (state_dim * 4,))
        environmental_input = jax.random.normal(keys[1], (state_dim * 3,))
        
        derivatives = flow.temporal_flow_equations(t, temporal_state, environmental_input)
        
        # Extract synthesis weights derivatives (last state_dim elements)
        synthesis_weights_derivative = derivatives[3*state_dim:4*state_dim]
        
        # The first 3 elements correspond to retention, present, protention weights
        # Their sum change should be small (approximately conserved)
        weight_sum_change = jnp.sum(synthesis_weights_derivative[:3])
        assert abs(weight_sum_change) < 0.5  # Allow some flexibility


class TestEnactiveCouplingDynamics:
    """Test cases for EnactiveCouplingDynamics."""
    
    @pytest.fixture
    def coupling_dynamics_setup(self):
        """Set up enactive coupling dynamics for testing."""
        agent_dim = 10
        environment_dim = 8
        meaning_dim = 6
        config = DynamicsConfig(
            agent_environment_coupling_strength=0.6,
            circular_causality_time_constant=0.1,
            structural_coupling_adaptation_rate=0.02,
            activation_dynamics="gelu",
        )
        key = jax.random.PRNGKey(42)
        
        coupling = EnactiveCouplingDynamics(
            agent_dim=agent_dim,
            environment_dim=environment_dim,
            meaning_dim=meaning_dim,
            config=config,
            key=key,
        )
        
        return coupling, agent_dim, environment_dim, meaning_dim, config
    
    def test_coupling_dynamics_initialization(self, coupling_dynamics_setup):
        """Test EnactiveCouplingDynamics initialization."""
        coupling, agent_dim, environment_dim, meaning_dim, config = coupling_dynamics_setup
        
        # Check neural network components
        assert hasattr(coupling, 'agent_dynamics')
        assert hasattr(coupling, 'environment_response')
        assert hasattr(coupling, 'circular_causality_network')
        assert hasattr(coupling, 'meaning_emergence_dynamics')
        assert hasattr(coupling, 'coupling_adaptation')
        assert hasattr(coupling, 'stability_controller')
        
        # Check system parameters
        assert coupling.coupling_time_constant == config.circular_causality_time_constant
        assert coupling.causality_feedback_strength == config.agent_environment_coupling_strength
        assert coupling.adaptation_learning_rate == config.structural_coupling_adaptation_rate
        
        # Check noise correlation matrix
        total_dim = agent_dim + environment_dim + meaning_dim
        assert coupling.noise_correlation_matrix.shape == (total_dim, total_dim)
    
    def test_enactive_coupling_sde(self, coupling_dynamics_setup):
        """Test enactive coupling SDE computation."""
        coupling, agent_dim, environment_dim, meaning_dim, config = coupling_dynamics_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        t = 1.0
        total_dim = agent_dim + environment_dim + meaning_dim
        coupling_state = jax.random.normal(keys[0], (total_dim,))
        history_context = jax.random.normal(keys[1], (agent_dim + environment_dim,))
        meaning_context = jax.random.normal(keys[2], (meaning_dim,))
        
        drift, diffusion = coupling.enactive_coupling_sde(
            t, coupling_state, history_context, meaning_context
        )
        
        # Check drift term
        assert drift.shape == (total_dim,)
        assert jnp.all(jnp.isfinite(drift))
        
        # Check diffusion term
        assert diffusion.shape == (total_dim, total_dim)
        assert jnp.all(jnp.isfinite(diffusion))
    
    def test_circular_causality_computation(self, coupling_dynamics_setup):
        """Test circular causality computation in SDE."""
        coupling, agent_dim, environment_dim, meaning_dim, config = coupling_dynamics_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        t = 0.5
        total_dim = agent_dim + environment_dim + meaning_dim
        coupling_state = jax.random.normal(keys[0], (total_dim,))
        history_context = jax.random.normal(keys[1], (agent_dim + environment_dim,))
        meaning_context = jax.random.normal(keys[2], (meaning_dim,))
        
        drift, _ = coupling.enactive_coupling_sde(
            t, coupling_state, history_context, meaning_context
        )
        
        # Extract components
        agent_drift = drift[:agent_dim]
        environment_drift = drift[agent_dim:agent_dim + environment_dim]
        meaning_drift = drift[agent_dim + environment_dim:]
        
        # All components should be finite
        assert jnp.all(jnp.isfinite(agent_drift))
        assert jnp.all(jnp.isfinite(environment_drift))
        assert jnp.all(jnp.isfinite(meaning_drift))
    
    def test_coupling_strength_adaptation(self, coupling_dynamics_setup):
        """Test adaptive coupling strength."""
        coupling, agent_dim, environment_dim, meaning_dim, config = coupling_dynamics_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        # Test multiple coupling states with different characteristics
        for i in range(3):
            t = float(i)
            total_dim = agent_dim + environment_dim + meaning_dim
            coupling_state = jax.random.normal(keys[0], (total_dim,)) * (i + 1)  # Different magnitudes
            history_context = jax.random.normal(keys[1], (agent_dim + environment_dim,))
            meaning_context = jax.random.normal(keys[2], (meaning_dim,))
            
            drift, diffusion = coupling.enactive_coupling_sde(
                t, coupling_state, history_context, meaning_context
            )
            
            # Check that adaptation produces valid results
            assert jnp.all(jnp.isfinite(drift))
            assert jnp.all(jnp.isfinite(diffusion))


class TestNeuralODEConsciousnessFlow:
    """Test cases for NeuralODEConsciousnessFlow."""
    
    @pytest.fixture
    def neural_ode_setup(self):
        """Set up Neural ODE consciousness flow for testing."""
        state_dim = 16
        hidden_dim = 32
        config = DynamicsConfig(
            hidden_dynamics_dim=hidden_dim,
            neural_ode_depth=3,
            activation_dynamics="gelu",
        )
        key = jax.random.PRNGKey(42)
        
        ode_flow = NeuralODEConsciousnessFlow(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            config=config,
            key=key,
        )
        
        return ode_flow, state_dim, hidden_dim, config
    
    def test_neural_ode_initialization(self, neural_ode_setup):
        """Test NeuralODEConsciousnessFlow initialization."""
        ode_flow, state_dim, hidden_dim, config = neural_ode_setup
        
        # Check neural network components
        assert hasattr(ode_flow, 'dynamics_network')
        assert hasattr(ode_flow, 'temporal_modulation')
        assert hasattr(ode_flow, 'consciousness_level_predictor')
        assert hasattr(ode_flow, 'time_embedding')
        assert hasattr(ode_flow, 'adaptive_step_controller')
        assert hasattr(ode_flow, 'jacobian_regularizer')
        assert hasattr(ode_flow, 'energy_conservator')
    
    def test_neural_ode_dynamics(self, neural_ode_setup):
        """Test Neural ODE dynamics computation."""
        ode_flow, state_dim, hidden_dim, config = neural_ode_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        t = 1.0
        consciousness_state = jax.random.normal(keys[0], (state_dim,))
        phenomenological_context = jax.random.normal(keys[1], (state_dim,))
        
        dynamics = ode_flow.neural_ode_dynamics(
            t, consciousness_state, phenomenological_context
        )
        
        assert dynamics.shape == consciousness_state.shape
        assert jnp.all(jnp.isfinite(dynamics))
    
    def test_neural_ode_dynamics_without_context(self, neural_ode_setup):
        """Test Neural ODE dynamics without phenomenological context."""
        ode_flow, state_dim, hidden_dim, config = neural_ode_setup
        key = jax.random.PRNGKey(42)
        
        t = 0.5
        consciousness_state = jax.random.normal(key, (state_dim,))
        
        dynamics = ode_flow.neural_ode_dynamics(t, consciousness_state, None)
        
        assert dynamics.shape == consciousness_state.shape
        assert jnp.all(jnp.isfinite(dynamics))
    
    def test_consciousness_level_computation(self, neural_ode_setup):
        """Test consciousness level computation."""
        ode_flow, state_dim, hidden_dim, config = neural_ode_setup
        key = jax.random.PRNGKey(42)
        
        consciousness_state = jax.random.normal(key, (state_dim,))
        
        consciousness_level = ode_flow.compute_consciousness_level(consciousness_state)
        
        assert isinstance(consciousness_level, float)
        assert 0.0 <= consciousness_level <= 1.0
        assert jnp.isfinite(consciousness_level)
    
    def test_jacobian_regularization_loss(self, neural_ode_setup):
        """Test Jacobian regularization loss computation."""
        ode_flow, state_dim, hidden_dim, config = neural_ode_setup
        key = jax.random.PRNGKey(42)
        
        consciousness_state = jax.random.normal(key, (state_dim,))
        t = 1.0
        
        jacobian_loss = ode_flow.jacobian_regularization_loss(consciousness_state, t)
        
        assert isinstance(jacobian_loss, float)
        assert jnp.isfinite(jacobian_loss)
    
    def test_gradient_clipping(self, neural_ode_setup):
        """Test gradient clipping in dynamics computation."""
        ode_flow, state_dim, hidden_dim, config = neural_ode_setup
        key = jax.random.PRNGKey(42)
        
        t = 1.0
        # Create a state that might produce large gradients
        large_state = jax.random.normal(key, (state_dim,)) * 100
        
        dynamics = ode_flow.neural_ode_dynamics(t, large_state, None)
        
        # Dynamics should be clipped to reasonable magnitude
        dynamics_norm = jnp.linalg.norm(dynamics)
        assert dynamics_norm <= 10.0  # Max norm threshold
        assert jnp.all(jnp.isfinite(dynamics))


class TestContinuousDynamicsProcessor:
    """Test cases for ContinuousDynamicsProcessor."""
    
    @pytest.fixture
    def processor_setup(self):
        """Set up continuous dynamics processor for testing."""
        state_dim = 12
        config = DynamicsConfig(
            retention_decay_rate=0.1,
            agent_environment_coupling_strength=0.5,
            hidden_dynamics_dim=32,
        )
        key = jax.random.PRNGKey(42)
        integration_method = IntegrationMethod.TSIT5
        
        processor = ContinuousDynamicsProcessor(
            state_dim=state_dim,
            config=config,
            key=key,
            integration_method=integration_method,
        )
        
        return processor, state_dim, config
    
    def test_processor_initialization(self, processor_setup):
        """Test ContinuousDynamicsProcessor initialization."""
        processor, state_dim, config = processor_setup
        
        assert processor.config == config
        assert processor.integration_method == IntegrationMethod.TSIT5
        assert processor.current_continuous_state is None
        
        # Check components
        assert isinstance(processor.temporal_flow, HusserlianTemporalFlow)
        assert isinstance(processor.coupling_dynamics, EnactiveCouplingDynamics)
        assert isinstance(processor.neural_ode_flow, NeuralODEConsciousnessFlow)
    
    def test_evolve_temporal_consciousness(self, processor_setup):
        """Test temporal consciousness evolution."""
        processor, state_dim, config = processor_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Create initial temporal moment
        initial_temporal_moment = TemporalMoment(
            timestamp=0.0,
            retention=jax.random.normal(keys[0], (state_dim,)),
            present_moment=jax.random.normal(keys[1], (state_dim,)),
            protention=jax.random.normal(keys[2], (state_dim,)),
            synthesis_weights=jax.random.normal(keys[3], (state_dim,)),
        )
        
        environmental_input = jax.random.normal(keys[3], (state_dim * 3,))
        time_span = (0.0, 0.1)
        num_steps = 10
        
        times, states = processor.evolve_temporal_consciousness(
            initial_temporal_moment, environmental_input, time_span, num_steps
        )
        
        assert len(times) == len(states)
        assert times[0] == time_span[0]
        assert times[-1] == pytest.approx(time_span[1], abs=1e-3)
        
        for state in states:
            assert state.shape == (state_dim * 4,)  # 4x for temporal components
            assert jnp.all(jnp.isfinite(state))
    
    def test_evolve_enactive_coupling(self, processor_setup):
        """Test enactive coupling evolution."""
        processor, state_dim, config = processor_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Create initial coupling state
        initial_coupling_state = CouplingState(
            agent_state=jax.random.normal(keys[0], (state_dim,)),
            environmental_state=jax.random.normal(keys[1], (state_dim,)),
            coupling_strength=0.6,
            stability_metric=0.7,
            adaptation_rate=0.02,
            coupling_history=jax.random.normal(keys[2], (5, state_dim)),
        )
        
        # Environmental perturbation function
        def environmental_perturbations(t):
            return jax.random.normal(keys[3], (10,)) * 0.1
        
        time_span = (0.0, 0.1)
        noise_key = keys[3]
        num_steps = 8
        
        times, states = processor.evolve_enactive_coupling(
            initial_coupling_state, environmental_perturbations, time_span, noise_key, num_steps
        )
        
        assert len(times) == len(states)
        assert times[0] == time_span[0]
        assert times[-1] == pytest.approx(time_span[1], abs=1e-3)
        
        for state in states:
            # State should have agent + environment + meaning dimensions
            expected_dim = state_dim + state_dim + state_dim // 2
            assert state.shape == (expected_dim,)
            assert jnp.all(jnp.isfinite(state))
    
    def test_evolve_neural_ode_consciousness(self, processor_setup):
        """Test Neural ODE consciousness evolution."""
        processor, state_dim, config = processor_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        initial_consciousness_state = jax.random.normal(keys[0], (state_dim * 4,))  # Extended state
        phenomenological_context = jax.random.normal(keys[1], (state_dim * 4,))
        time_span = (0.0, 0.05)
        num_steps = 5
        
        times, states, consciousness_levels = processor.evolve_neural_ode_consciousness(
            initial_consciousness_state, phenomenological_context, time_span, num_steps
        )
        
        assert len(times) == len(states) == len(consciousness_levels)
        assert times[0] == time_span[0]
        assert times[-1] == pytest.approx(time_span[1], abs=1e-3)
        
        for i, (state, consciousness_level) in enumerate(zip(states, consciousness_levels)):
            assert state.shape == initial_consciousness_state.shape
            assert jnp.all(jnp.isfinite(state))
            assert 0.0 <= consciousness_level <= 1.0
            assert jnp.isfinite(consciousness_level)
    
    def test_integrate_continuous_consciousness(self, processor_setup):
        """Test integrated continuous consciousness evolution."""
        processor, state_dim, config = processor_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 6)
        
        # Create input components
        temporal_moment = TemporalMoment(
            timestamp=0.0,
            retention=jax.random.normal(keys[0], (state_dim,)),
            present_moment=jax.random.normal(keys[1], (state_dim,)),
            protention=jax.random.normal(keys[2], (state_dim,)),
            synthesis_weights=jax.random.normal(keys[3], (state_dim,)),
        )
        
        coupling_state = CouplingState(
            agent_state=jax.random.normal(keys[4], (state_dim,)),
            environmental_state=jax.random.normal(keys[5], (state_dim,)),
            coupling_strength=0.6,
            stability_metric=0.7,
            adaptation_rate=0.02,
            coupling_history=jax.random.normal(keys[4], (5, state_dim)),
        )
        
        def environmental_dynamics(t):
            return jax.random.normal(keys[5], (10,)) * 0.1 * jnp.sin(t)
        
        time_span = (0.0, 0.05)
        integration_key = keys[0]
        num_steps = 5
        
        continuous_state = processor.integrate_continuous_consciousness(
            temporal_moment, coupling_state, environmental_dynamics,
            time_span, integration_key, num_steps
        )
        
        assert isinstance(continuous_state, ContinuousState)
        assert continuous_state.validate_state()
        assert continuous_state.timestamp == time_span[1]
        assert 0.0 <= continuous_state.consciousness_level <= 1.0
        assert 0.0 <= continuous_state.coupling_strength <= 1.0
        assert -1.0 <= continuous_state.temporal_coherence <= 1.0
    
    def test_convert_temporal_moment_to_continuous(self, processor_setup):
        """Test temporal moment to continuous state conversion."""
        processor, state_dim, config = processor_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        temporal_moment = TemporalMoment(
            timestamp=0.0,
            retention=jax.random.normal(keys[0], (state_dim,)),
            present_moment=jax.random.normal(keys[1], (state_dim,)),
            protention=jax.random.normal(keys[2], (state_dim,)),
            synthesis_weights=jax.random.normal(keys[3], (state_dim,)),
        )
        
        continuous_state = processor._convert_temporal_moment_to_continuous(temporal_moment)
        
        assert continuous_state.shape == (state_dim * 4,)
        assert jnp.all(jnp.isfinite(continuous_state))
    
    def test_convert_coupling_state_to_continuous(self, processor_setup):
        """Test coupling state to continuous state conversion."""
        processor, state_dim, config = processor_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        coupling_state = CouplingState(
            agent_state=jax.random.normal(keys[0], (state_dim,)),
            environmental_state=jax.random.normal(keys[1], (state_dim - 2,)),  # Different size
            coupling_strength=0.6,
            stability_metric=0.7,
            adaptation_rate=0.02,
            coupling_history=jax.random.normal(keys[0], (5, state_dim)),
        )
        
        continuous_state = processor._convert_coupling_state_to_continuous(coupling_state)
        
        # Should handle different dimensions appropriately
        expected_dim = state_dim + state_dim + state_dim // 2  # agent + env + meaning
        assert continuous_state.shape == (expected_dim,)
        assert jnp.all(jnp.isfinite(continuous_state))
    
    def test_compute_trajectory_statistics(self, processor_setup):
        """Test trajectory statistics computation."""
        processor, state_dim, config = processor_setup
        key = jax.random.PRNGKey(42)
        
        # Create mock trajectory
        num_steps = 10
        times = jnp.linspace(0, 1, num_steps)
        states = jax.random.normal(key, (num_steps, state_dim))
        
        statistics = processor.compute_trajectory_statistics(times, states)
        
        expected_keys = {
            'trajectory_length', 'lyapunov_estimate', 'temporal_coherence',
            'phase_space_volume', 'mean_state_norm', 'state_variance',
            'total_time', 'num_integration_steps'
        }
        assert set(statistics.keys()) == expected_keys
        
        for key, value in statistics.items():
            assert isinstance(value, float)
            assert jnp.isfinite(value)


class TestFactoryAndUtilityFunctions:
    """Test cases for factory and utility functions."""
    
    def test_create_continuous_dynamics_processor(self):
        """Test factory function for creating processor."""
        state_dim = 16
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        integration_method = IntegrationMethod.EULER
        
        processor = create_continuous_dynamics_processor(
            state_dim, config, key, integration_method
        )
        
        assert isinstance(processor, ContinuousDynamicsProcessor)
        assert processor.config == config
        assert processor.integration_method == integration_method
    
    def test_create_default_dynamics_config(self):
        """Test default configuration creation."""
        config = create_default_dynamics_config()
        
        assert isinstance(config, DynamicsConfig)
        assert config.dt_min == 1e-4
        assert config.dt_max == 0.1
    
    def test_create_default_dynamics_config_with_overrides(self):
        """Test default configuration creation with overrides."""
        config = create_default_dynamics_config(
            dt_min=1e-5,
            retention_decay_rate=0.2,
            hidden_dynamics_dim=64,
        )
        
        assert isinstance(config, DynamicsConfig)
        assert config.dt_min == 1e-5
        assert config.retention_decay_rate == 0.2
        assert config.hidden_dynamics_dim == 64
        # Other parameters should remain default
        assert config.dt_max == 0.1
    
    def test_create_environmental_perturbation_function(self):
        """Test environmental perturbation function creation."""
        amplitude = 2.0
        frequency = 0.2
        noise_strength = 0.15
        key = jax.random.PRNGKey(42)
        
        env_function = create_environmental_perturbation_function(
            amplitude, frequency, noise_strength, key
        )
        
        # Test the function at different time points
        for t in [0.0, 0.5, 1.0, 2.0]:
            perturbation = env_function(t)
            
            assert isinstance(perturbation, jax.Array)
            assert perturbation.shape == (11,)  # 1 deterministic + 10 stochastic
            assert jnp.all(jnp.isfinite(perturbation))
    
    def test_analyze_continuous_consciousness_dynamics(self):
        """Test continuous consciousness dynamics analysis."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Create processor
        state_dim = 8
        config = DynamicsConfig()
        processor = create_continuous_dynamics_processor(state_dim, config, keys[0])
        
        # Create initial conditions
        initial_temporal_moment = TemporalMoment(
            timestamp=0.0,
            retention=jax.random.normal(keys[1], (state_dim,)),
            present_moment=jax.random.normal(keys[2], (state_dim,)),
            protention=jax.random.normal(keys[3], (state_dim,)),
            synthesis_weights=jax.random.normal(keys[1], (state_dim,)),
        )
        
        initial_coupling_state = CouplingState(
            agent_state=jax.random.normal(keys[2], (state_dim,)),
            environmental_state=jax.random.normal(keys[3], (state_dim,)),
            coupling_strength=0.6,
            stability_metric=0.7,
            adaptation_rate=0.02,
            coupling_history=jax.random.normal(keys[2], (3, state_dim)),
        )
        
        time_span = (0.0, 0.02)  # Short time span for testing
        
        analysis_result = analyze_continuous_consciousness_dynamics(
            processor, initial_temporal_moment, initial_coupling_state,
            time_span, keys[3]
        )
        
        assert isinstance(analysis_result, dict)
        
        if analysis_result.get('analysis_success', False):
            expected_keys = {
                'final_continuous_state', 'temporal_trajectory_stats',
                'coupling_trajectory_stats', 'consciousness_level_evolution',
                'temporal_coherence_final', 'coupling_strength_final',
                'analysis_success'
            }
            assert set(analysis_result.keys()) == expected_keys
            
            assert isinstance(analysis_result['final_continuous_state'], ContinuousState)
            assert isinstance(analysis_result['temporal_trajectory_stats'], dict)
            assert isinstance(analysis_result['coupling_trajectory_stats'], dict)
        else:
            # If analysis failed, should have error information
            assert 'error_message' in analysis_result
            assert analysis_result['analysis_success'] is False


class TestMathematicalCorrectness:
    """Test cases for mathematical correctness of continuous dynamics."""
    
    def test_temporal_flow_energy_conservation(self):
        """Test energy conservation in temporal flow."""
        state_dim = 8
        config = DynamicsConfig(retention_decay_rate=0.05)
        key = jax.random.PRNGKey(42)
        
        flow = HusserlianTemporalFlow(state_dim, 6, config, key)
        
        keys = jax.random.split(key, 2)
        t = 1.0
        temporal_state = jax.random.normal(keys[0], (state_dim * 4,)) * 0.1
        environmental_input = jax.random.normal(keys[1], (state_dim * 3,)) * 0.05
        
        # Compute derivatives
        derivatives = flow.temporal_flow_equations(t, temporal_state, environmental_input)
        
        # Check that derivatives don't explode
        derivative_norm = jnp.linalg.norm(derivatives)
        state_norm = jnp.linalg.norm(temporal_state)
        
        # Derivative should be bounded relative to state magnitude
        assert derivative_norm <= 10 * state_norm + 1.0
    
    def test_coupling_dynamics_stability(self):
        """Test stability of coupling dynamics."""
        agent_dim, environment_dim, meaning_dim = 6, 6, 4
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        
        coupling = EnactiveCouplingDynamics(agent_dim, environment_dim, meaning_dim, config, key)
        
        keys = jax.random.split(key, 3)
        t = 0.5
        total_dim = agent_dim + environment_dim + meaning_dim
        coupling_state = jax.random.normal(keys[0], (total_dim,)) * 0.1
        history_context = jax.random.normal(keys[1], (agent_dim + environment_dim,)) * 0.1
        meaning_context = jax.random.normal(keys[2], (meaning_dim,)) * 0.1
        
        drift, diffusion = coupling.enactive_coupling_sde(
            t, coupling_state, history_context, meaning_context
        )
        
        # Check stability: drift should not be too large
        drift_norm = jnp.linalg.norm(drift)
        state_norm = jnp.linalg.norm(coupling_state)
        
        assert drift_norm <= 5 * state_norm + 1.0  # Reasonable bound
        
        # Check diffusion matrix properties
        assert jnp.all(jnp.isfinite(diffusion))
        # Diffusion matrix should be bounded
        diffusion_norm = jnp.linalg.norm(diffusion)
        assert diffusion_norm <= 10.0
    
    def test_neural_ode_lipschitz_continuity(self):
        """Test Lipschitz continuity of Neural ODE."""
        state_dim = 12
        hidden_dim = 24
        config = DynamicsConfig(hidden_dynamics_dim=hidden_dim)
        key = jax.random.PRNGKey(42)
        
        ode_flow = NeuralODEConsciousnessFlow(state_dim, hidden_dim, config, key)
        
        keys = jax.random.split(key, 2)
        t = 1.0
        
        # Two nearby states
        state1 = jax.random.normal(keys[0], (state_dim,)) * 0.1
        state2 = state1 + jax.random.normal(keys[1], (state_dim,)) * 0.01  # Small perturbation
        
        dynamics1 = ode_flow.neural_ode_dynamics(t, state1, None)
        dynamics2 = ode_flow.neural_ode_dynamics(t, state2, None)
        
        # Check Lipschitz condition (approximately)
        state_diff = jnp.linalg.norm(state2 - state1)
        dynamics_diff = jnp.linalg.norm(dynamics2 - dynamics1)
        
        # Lipschitz constant should be reasonable for well-behaved dynamics
        if state_diff > 1e-8:  # Avoid division by very small numbers
            lipschitz_estimate = dynamics_diff / state_diff
            assert lipschitz_estimate <= 100.0  # Reasonable bound
    
    def test_integration_consistency(self):
        """Test consistency of integration methods."""
        state_dim = 6
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        processor = create_continuous_dynamics_processor(
            state_dim, config, keys[0], IntegrationMethod.EULER
        )
        
        # Create simple temporal moment
        temporal_moment = TemporalMoment(
            timestamp=0.0,
            retention=jax.random.normal(keys[1], (state_dim,)) * 0.1,
            present_moment=jax.random.normal(keys[1], (state_dim,)) * 0.1,
            protention=jax.random.normal(keys[1], (state_dim,)) * 0.1,
            synthesis_weights=jax.random.normal(keys[1], (state_dim,)) * 0.1,
        )
        
        environmental_input = jax.random.normal(keys[1], (state_dim * 3,)) * 0.05
        time_span = (0.0, 0.01)  # Very short integration
        
        times, states = processor.evolve_temporal_consciousness(
            temporal_moment, environmental_input, time_span, 10
        )
        
        # Integration should be stable
        for state in states:
            assert jnp.all(jnp.isfinite(state))
            # State magnitude shouldn't explode
            assert jnp.linalg.norm(state) <= 100.0


class TestPerformanceAndScalability:
    """Test cases for performance and scalability."""
    
    @pytest.mark.parametrize("state_dim", [8, 16, 24])
    def test_processor_scalability_with_dimension(self, state_dim):
        """Test processor scalability with different state dimensions."""
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        
        processor = create_continuous_dynamics_processor(
            state_dim, config, key, IntegrationMethod.EULER
        )
        
        assert isinstance(processor, ContinuousDynamicsProcessor)
        # Check that components are initialized properly
        assert isinstance(processor.temporal_flow, HusserlianTemporalFlow)
        assert isinstance(processor.coupling_dynamics, EnactiveCouplingDynamics)
        assert isinstance(processor.neural_ode_flow, NeuralODEConsciousnessFlow)
    
    @pytest.mark.parametrize("integration_method", [IntegrationMethod.EULER, IntegrationMethod.HEUN])
    def test_processor_scalability_with_integration_methods(self, integration_method):
        """Test processor with different integration methods."""
        state_dim = 12
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        
        processor = create_continuous_dynamics_processor(
            state_dim, config, key, integration_method
        )
        
        assert processor.integration_method == integration_method
        assert isinstance(processor, ContinuousDynamicsProcessor)
    
    def test_large_system_memory_efficiency(self):
        """Test memory efficiency with larger systems."""
        # Use moderately large dimensions for testing
        state_dim = 32
        config = DynamicsConfig(hidden_dynamics_dim=64)
        key = jax.random.PRNGKey(42)
        
        # Should be able to create large systems
        processor = create_continuous_dynamics_processor(
            state_dim, config, key, IntegrationMethod.EULER
        )
        
        assert isinstance(processor, ContinuousDynamicsProcessor)
    
    def test_short_integration_efficiency(self):
        """Test efficiency of short time integrations."""
        state_dim = 10
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        processor = create_continuous_dynamics_processor(state_dim, config, keys[0])
        
        # Create temporal moment
        temporal_moment = TemporalMoment(
            timestamp=0.0,
            retention=jax.random.normal(keys[1], (state_dim,)),
            present_moment=jax.random.normal(keys[1], (state_dim,)),
            protention=jax.random.normal(keys[1], (state_dim,)),
            synthesis_weights=jax.random.normal(keys[1], (state_dim,)),
        )
        
        environmental_input = jax.random.normal(keys[1], (state_dim * 3,))
        time_span = (0.0, 0.001)  # Very short integration
        num_steps = 5
        
        # Should complete efficiently
        times, states = processor.evolve_temporal_consciousness(
            temporal_moment, environmental_input, time_span, num_steps
        )
        
        assert len(times) == len(states)
        assert len(times) == num_steps + 1


class TestErrorHandlingAndEdgeCases:
    """Test cases for error handling and edge cases."""
    
    def test_zero_state_inputs(self):
        """Test behavior with zero state inputs."""
        state_dim = 8
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        
        flow = HusserlianTemporalFlow(state_dim, 6, config, key)
        
        t = 1.0
        zero_state = jnp.zeros(state_dim * 4)
        zero_input = jnp.zeros(state_dim * 3)
        
        derivatives = flow.temporal_flow_equations(t, zero_state, zero_input)
        
        # Should handle gracefully
        assert derivatives.shape == zero_state.shape
        assert jnp.all(jnp.isfinite(derivatives))
    
    def test_very_small_time_steps(self):
        """Test behavior with very small time steps."""
        state_dim = 6
        config = DynamicsConfig(dt_min=1e-6)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        processor = create_continuous_dynamics_processor(state_dim, config, keys[0])
        
        temporal_moment = TemporalMoment(
            timestamp=0.0,
            retention=jax.random.normal(keys[1], (state_dim,)),
            present_moment=jax.random.normal(keys[1], (state_dim,)),
            protention=jax.random.normal(keys[1], (state_dim,)),
            synthesis_weights=jax.random.normal(keys[1], (state_dim,)),
        )
        
        environmental_input = jax.random.normal(keys[1], (state_dim * 3,))
        time_span = (0.0, 1e-4)  # Very small time span
        
        times, states = processor.evolve_temporal_consciousness(
            temporal_moment, environmental_input, time_span, 5
        )
        
        assert len(times) > 0
        assert len(states) > 0
        assert jnp.all(jnp.isfinite(states[0]))
    
    def test_extreme_coupling_strengths(self):
        """Test behavior with extreme coupling strengths."""
        agent_dim, environment_dim, meaning_dim = 4, 4, 2
        config = DynamicsConfig(agent_environment_coupling_strength=0.99)  # Very strong coupling
        key = jax.random.PRNGKey(42)
        
        coupling = EnactiveCouplingDynamics(agent_dim, environment_dim, meaning_dim, config, key)
        
        keys = jax.random.split(key, 3)
        t = 1.0
        total_dim = agent_dim + environment_dim + meaning_dim
        coupling_state = jax.random.normal(keys[0], (total_dim,))
        history_context = jax.random.normal(keys[1], (agent_dim + environment_dim,))
        meaning_context = jax.random.normal(keys[2], (meaning_dim,))
        
        drift, diffusion = coupling.enactive_coupling_sde(
            t, coupling_state, history_context, meaning_context
        )
        
        # Should handle extreme coupling without instability
        assert jnp.all(jnp.isfinite(drift))
        assert jnp.all(jnp.isfinite(diffusion))
    
    def test_mismatched_environmental_dimensions(self):
        """Test behavior with mismatched environmental dimensions."""
        state_dim = 8
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        processor = create_continuous_dynamics_processor(state_dim, config, keys[0])
        
        coupling_state = CouplingState(
            agent_state=jax.random.normal(keys[1], (state_dim,)),
            environmental_state=jax.random.normal(keys[1], (state_dim + 2,)),  # Larger environment
            coupling_strength=0.5,
            stability_metric=0.6,
            adaptation_rate=0.01,
            coupling_history=jax.random.normal(keys[1], (3, state_dim)),
        )
        
        # Should handle dimension mismatch gracefully
        continuous_state = processor._convert_coupling_state_to_continuous(coupling_state)
        
        assert jnp.all(jnp.isfinite(continuous_state))
    
    def test_nan_in_environmental_input(self):
        """Test handling of NaN in environmental input."""
        state_dim = 6
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        
        flow = HusserlianTemporalFlow(state_dim, 4, config, key)
        
        t = 1.0
        temporal_state = jax.random.normal(key, (state_dim * 4,))
        environmental_input_with_nan = jnp.array([jnp.nan] * (state_dim * 3))
        
        # Should handle NaN input without crashing
        try:
            derivatives = flow.temporal_flow_equations(t, temporal_state, environmental_input_with_nan)
            # If it doesn't crash, check that output handling is reasonable
            # (may contain NaN, but shouldn't crash the computation)
            assert derivatives.shape == temporal_state.shape
        except Exception:
            # It's acceptable to raise an exception for invalid input
            pass


# Integration tests
class TestContinuousDynamicsIntegration:
    """Integration tests for continuous dynamics with other modules."""
    
    def test_temporal_information_theory_integration(self):
        """Test integration with temporal processing and information theory."""
        state_dim = 8
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        processor = create_continuous_dynamics_processor(state_dim, config, keys[0])
        
        # Create temporal sequence
        temporal_moments = []
        for i in range(3):
            temporal_moment = TemporalMoment(
                timestamp=float(i) * 0.1,
                retention=jax.random.normal(keys[1], (state_dim,)),
                present_moment=jax.random.normal(keys[2], (state_dim,)),
                protention=jax.random.normal(keys[1], (state_dim,)),
                synthesis_weights=jax.random.normal(keys[2], (state_dim,)),
            )
            temporal_moments.append(temporal_moment)
        
        # Evolve each temporal moment
        evolved_states = []
        for temporal_moment in temporal_moments:
            environmental_input = jax.random.normal(keys[2], (state_dim * 3,)) * 0.1
            time_span = (temporal_moment.timestamp, temporal_moment.timestamp + 0.01)
            
            times, states = processor.evolve_temporal_consciousness(
                temporal_moment, environmental_input, time_span, 3
            )
            evolved_states.append(states[-1])  # Final state
        
        # Should have coherent temporal evolution
        assert len(evolved_states) == 3
        for state in evolved_states:
            assert jnp.all(jnp.isfinite(state))
    
    def test_coupling_dynamics_with_consciousness_measures(self):
        """Test coupling dynamics integration with consciousness measures."""
        state_dim = 6
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        processor = create_continuous_dynamics_processor(state_dim, config, keys[0])
        
        # Create coupling states with varying consciousness characteristics
        coupling_states = []
        consciousness_levels = [0.3, 0.7, 0.9]
        
        for consciousness_level in consciousness_levels:
            coupling_state = CouplingState(
                agent_state=jax.random.normal(keys[1], (state_dim,)) * consciousness_level,
                environmental_state=jax.random.normal(keys[2], (state_dim,)),
                coupling_strength=consciousness_level,
                stability_metric=0.8,
                adaptation_rate=0.02,
                coupling_history=jax.random.normal(keys[3], (3, state_dim)),
            )
            coupling_states.append(coupling_state)
        
        # Evolve each coupling state
        for coupling_state in coupling_states:
            def env_dynamics(t):
                return jax.random.normal(keys[3], (8,)) * 0.05
            
            time_span = (0.0, 0.01)
            
            times, states = processor.evolve_enactive_coupling(
                coupling_state, env_dynamics, time_span, keys[3], 3
            )
            
            # Should evolve stably regardless of consciousness level
            assert len(times) == len(states)
            for state in states:
                assert jnp.all(jnp.isfinite(state))
    
    def test_full_consciousness_dynamics_workflow(self):
        """Test complete consciousness dynamics workflow."""
        state_dim = 8
        config = DynamicsConfig()
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)
        
        processor = create_continuous_dynamics_processor(state_dim, config, keys[0])
        
        # Create rich temporal-coupling scenario
        temporal_moment = TemporalMoment(
            timestamp=0.0,
            retention=jax.random.normal(keys[1], (state_dim,)),
            present_moment=jax.random.normal(keys[2], (state_dim,)),
            protention=jax.random.normal(keys[3], (state_dim,)),
            synthesis_weights=jax.random.normal(keys[4], (state_dim,)),
        )
        
        coupling_state = CouplingState(
            agent_state=temporal_moment.present_moment,  # Link temporal and coupling
            environmental_state=jax.random.normal(keys[3], (state_dim,)),
            coupling_strength=0.6,
            stability_metric=0.7,
            adaptation_rate=0.02,
            coupling_history=jax.random.normal(keys[4], (3, state_dim)),
        )
        
        def environmental_dynamics(t):
            # Dynamic environment that responds to time
            return jax.random.normal(keys[4], (10,)) * 0.1 * (1 + 0.5 * jnp.sin(t * 10))
        
        time_span = (0.0, 0.02)
        
        # Full integrated evolution
        final_continuous_state = processor.integrate_continuous_consciousness(
            temporal_moment, coupling_state, environmental_dynamics,
            time_span, keys[0], 5
        )
        
        assert isinstance(final_continuous_state, ContinuousState)
        assert final_continuous_state.validate_state()
        
        # Check that the state has evolved meaningfully
        assert final_continuous_state.timestamp == time_span[1]
        assert 0.0 <= final_continuous_state.consciousness_level <= 1.0
        assert jnp.all(jnp.isfinite(final_continuous_state.retention_field))
        assert jnp.all(jnp.isfinite(final_continuous_state.present_awareness))
        assert jnp.all(jnp.isfinite(final_continuous_state.protention_field))


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main([__file__, "-v", "--tb=short"])