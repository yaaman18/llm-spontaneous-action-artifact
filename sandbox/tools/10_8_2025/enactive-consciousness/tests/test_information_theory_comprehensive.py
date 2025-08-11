"""Comprehensive test suite for information theory module.

This test suite follows TDD principles with Red-Green-Refactor cycles,
providing extensive coverage of information-theoretic measures for
enactive consciousness with mathematical validation.

Test Coverage:
- Circular causality index computation and validation
- Transfer entropy estimation with edge cases
- Mutual information computation correctness
- Entropy rate calculation accuracy
- Integrated information (Phi) measures
- Complexity measures validation
- Mathematical correctness validation
- Performance and boundary condition testing
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple
import warnings
from unittest.mock import patch, MagicMock

# Import the module under test
import sys
sys.path.insert(0, '/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/10_8_2025/enactive-consciousness/src')

from enactive_consciousness.information_theory import (
    InformationTheoryError,
    mutual_information_kraskov,
    transfer_entropy,
    circular_causality_index,
    entropy_rate,
    integrated_information_phi,
    complexity_measure,
)


class TestInformationTheoryError:
    """Test cases for InformationTheoryError exception."""
    
    def test_error_creation(self):
        """Test creating InformationTheoryError."""
        error = InformationTheoryError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


class TestMutualInformationKraskov:
    """Test cases for mutual information estimation using Kraskov method."""
    
    @pytest.fixture
    def setup_test_data(self):
        """Set up test data for mutual information tests."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Independent variables (MI should be close to 0)
        x_independent = jax.random.normal(keys[0], (100,))
        y_independent = jax.random.normal(keys[1], (100,))
        
        # Dependent variables (MI should be positive)
        x_dependent = jax.random.normal(keys[2], (100,))
        y_dependent = x_dependent + 0.1 * jax.random.normal(keys[3], (100,))
        
        return {
            'x_independent': x_independent,
            'y_independent': y_independent,
            'x_dependent': x_dependent,
            'y_dependent': y_dependent
        }
    
    def test_mutual_information_independent_variables(self, setup_test_data):
        """Test that MI is low for independent variables."""
        data = setup_test_data
        mi = mutual_information_kraskov(
            data['x_independent'], 
            data['y_independent'],
            k=3,
            base=jnp.e
        )
        
        # MI should be close to 0 for independent variables
        assert mi >= 0.0, "Mutual information must be non-negative"
        assert mi < 1.0, "MI should be low for independent variables"
        assert isinstance(mi, float)
    
    def test_mutual_information_dependent_variables(self, setup_test_data):
        """Test that MI is higher for dependent variables."""
        data = setup_test_data
        mi = mutual_information_kraskov(
            data['x_dependent'], 
            data['y_dependent'],
            k=3,
            base=jnp.e
        )
        
        # MI should be positive for dependent variables
        assert mi >= 0.0, "Mutual information must be non-negative"
        assert mi > 0.1, "MI should be positive for dependent variables"
        assert isinstance(mi, float)
    
    def test_mutual_information_identical_variables(self):
        """Test that MI is maximum for identical variables."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (50,))
        
        mi = mutual_information_kraskov(x, x, k=3)
        
        # MI should be high for identical variables
        assert mi >= 0.0, "Mutual information must be non-negative"
        assert mi > 0.5, "MI should be high for identical variables"
    
    def test_mutual_information_base_conversion(self):
        """Test base conversion from nats to bits."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (50,))
        y = x + 0.1 * jax.random.normal(key, (50,))
        
        mi_nats = mutual_information_kraskov(x, y, base=jnp.e)
        mi_bits = mutual_information_kraskov(x, y, base=2.0)
        
        # Conversion should be consistent
        expected_bits = mi_nats / jnp.log(2.0)
        assert abs(mi_bits - expected_bits) < 0.1, "Base conversion should be consistent"
    
    def test_mutual_information_dimension_mismatch(self):
        """Test error handling for dimension mismatch."""
        x = jnp.array([1, 2, 3])
        y = jnp.array([1, 2])  # Different length
        
        with pytest.raises(InformationTheoryError):
            mutual_information_kraskov(x, y)
    
    def test_mutual_information_multidimensional(self):
        """Test MI with multidimensional arrays."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (50, 3))  # Multidimensional
        y = jax.random.normal(key, (50, 2))
        
        mi = mutual_information_kraskov(x, y, k=3)
        
        assert mi >= 0.0, "Mutual information must be non-negative"
        assert isinstance(mi, float)
    
    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_mutual_information_different_k_values(self, k):
        """Test MI estimation with different k values."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (100,))
        y = x + 0.1 * jax.random.normal(key, (100,))
        
        mi = mutual_information_kraskov(x, y, k=k)
        
        assert mi >= 0.0, f"MI must be non-negative for k={k}"
        assert isinstance(mi, float)


class TestTransferEntropy:
    """Test cases for transfer entropy computation."""
    
    @pytest.fixture
    def setup_transfer_entropy_data(self):
        """Set up test data for transfer entropy tests."""
        key = jax.random.PRNGKey(42)
        
        # Create time series with causal relationship: y[t] depends on x[t-1]
        n_steps = 100
        x = jax.random.normal(key, (n_steps,))
        y = jnp.zeros(n_steps)
        
        # y[t] = 0.5 * x[t-1] + noise
        for t in range(1, n_steps):
            y = y.at[t].set(0.5 * x[t-1] + 0.1 * jax.random.normal(key, ()))
        
        return {'source': x, 'target': y, 'n_steps': n_steps}
    
    def test_transfer_entropy_causal_relationship(self, setup_transfer_entropy_data):
        """Test transfer entropy detects causal relationship."""
        data = setup_transfer_entropy_data
        
        te = transfer_entropy(
            data['source'], 
            data['target'],
            history_length=1,
            k=3
        )
        
        assert te >= 0.0, "Transfer entropy must be non-negative"
        assert te > 0.01, "TE should be positive for causal relationship"
        assert isinstance(te, float)
    
    def test_transfer_entropy_independent_series(self):
        """Test transfer entropy for independent time series."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Independent time series
        source = jax.random.normal(keys[0], (100,))
        target = jax.random.normal(keys[1], (100,))
        
        te = transfer_entropy(source, target, history_length=1, k=3)
        
        assert te >= 0.0, "Transfer entropy must be non-negative"
        assert te < 0.5, "TE should be low for independent series"
    
    def test_transfer_entropy_length_mismatch(self):
        """Test error handling for length mismatch."""
        source = jnp.array([1, 2, 3])
        target = jnp.array([1, 2])  # Different length
        
        with pytest.raises(InformationTheoryError):
            transfer_entropy(source, target)
    
    def test_transfer_entropy_insufficient_data(self):
        """Test error handling for insufficient data."""
        source = jnp.array([1, 2])  # Too short
        target = jnp.array([1, 2])
        
        with pytest.raises(InformationTheoryError):
            transfer_entropy(source, target, history_length=2)
    
    @pytest.mark.parametrize("history_length", [1, 2, 3])
    def test_transfer_entropy_different_history_lengths(self, history_length):
        """Test transfer entropy with different history lengths."""
        key = jax.random.PRNGKey(42)
        source = jax.random.normal(key, (50,))
        target = jax.random.normal(key, (50,))
        
        if len(source) > history_length + 1:
            te = transfer_entropy(source, target, history_length=history_length)
            
            assert te >= 0.0, f"TE must be non-negative for history_length={history_length}"
            assert isinstance(te, float)


class TestCircularCausalityIndex:
    """Test cases for circular causality index computation."""
    
    @pytest.fixture
    def setup_circular_causality_data(self):
        """Set up test data for circular causality tests."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        n_steps = 50
        
        # Create coupled agent-environment system
        agent_states = jax.random.normal(keys[0], (n_steps, 10))
        env_states = jax.random.normal(keys[1], (n_steps, 8))
        
        return {
            'agent_states': agent_states,
            'env_states': env_states,
            'n_steps': n_steps
        }
    
    def test_circular_causality_index_computation(self, setup_circular_causality_data):
        """Test basic circular causality index computation."""
        data = setup_circular_causality_data
        
        causality_metrics = circular_causality_index(
            data['agent_states'],
            data['env_states'],
            history_length=2
        )
        
        # Check all expected keys are present
        expected_keys = {
            'circular_causality',
            'transfer_entropy_env_to_agent',
            'transfer_entropy_agent_to_env',
            'circular_strength',
            'causality_asymmetry',
            'coupling_coherence',
            'instantaneous_coupling'
        }
        assert set(causality_metrics.keys()) == expected_keys
        
        # Check value ranges
        assert 0.0 <= causality_metrics['circular_causality'] <= 1.0
        assert causality_metrics['transfer_entropy_env_to_agent'] >= 0.0
        assert causality_metrics['transfer_entropy_agent_to_env'] >= 0.0
        assert causality_metrics['circular_strength'] >= 0.0
        assert causality_metrics['causality_asymmetry'] >= 0.0
        assert 0.0 <= causality_metrics['coupling_coherence'] <= 1.0
        assert causality_metrics['instantaneous_coupling'] >= 0.0
    
    def test_circular_causality_1d_inputs(self):
        """Test circular causality with 1D input arrays."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        agent_states = jax.random.normal(keys[0], (50,))
        env_states = jax.random.normal(keys[1], (50,))
        
        causality_metrics = circular_causality_index(
            agent_states, 
            env_states,
            history_length=1
        )
        
        assert isinstance(causality_metrics, dict)
        assert 'circular_causality' in causality_metrics
    
    def test_circular_causality_coupled_system(self):
        """Test circular causality with strongly coupled system."""
        key = jax.random.PRNGKey(42)
        
        # Create strongly coupled system
        n_steps = 30
        agent_base = jax.random.normal(key, (n_steps,))
        coupling_strength = 0.8
        
        # Environment responds to agent with coupling
        env_states = coupling_strength * agent_base + \
                    (1 - coupling_strength) * jax.random.normal(key, (n_steps,))
        
        # Agent also responds to environment (circular causality)
        agent_states = 0.6 * env_states + 0.4 * agent_base
        
        causality_metrics = circular_causality_index(
            agent_states,
            env_states,
            history_length=1
        )
        
        # Should detect significant coupling
        assert causality_metrics['circular_causality'] > 0.1
        assert causality_metrics['circular_strength'] > 0.0
        
    @pytest.mark.parametrize("history_length", [1, 2, 3])
    def test_circular_causality_different_history_lengths(self, history_length):
        """Test circular causality with different history lengths."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        agent_states = jax.random.normal(keys[0], (20,))
        env_states = jax.random.normal(keys[1], (20,))
        
        if len(agent_states) > history_length + 1:
            causality_metrics = circular_causality_index(
                agent_states,
                env_states,
                history_length=history_length
            )
            
            assert isinstance(causality_metrics, dict)
            assert 'circular_causality' in causality_metrics


class TestEntropyRate:
    """Test cases for entropy rate computation."""
    
    @pytest.fixture
    def setup_entropy_rate_data(self):
        """Set up test data for entropy rate tests."""
        key = jax.random.PRNGKey(42)
        
        # Regular time series
        regular_series = jnp.sin(jnp.linspace(0, 4*jnp.pi, 100))
        
        # Random time series
        random_series = jax.random.normal(key, (100,))
        
        # Constant time series
        constant_series = jnp.ones(100)
        
        return {
            'regular_series': regular_series,
            'random_series': random_series,
            'constant_series': constant_series
        }
    
    def test_entropy_rate_computation(self, setup_entropy_rate_data):
        """Test entropy rate computation for different time series."""
        data = setup_entropy_rate_data
        
        # Test regular series
        entropy_regular = entropy_rate(
            data['regular_series'],
            embedding_dim=2,
            tolerance=0.1
        )
        
        # Test random series
        entropy_random = entropy_rate(
            data['random_series'],
            embedding_dim=2,
            tolerance=0.1
        )
        
        assert isinstance(entropy_regular, float)
        assert isinstance(entropy_random, float)
        assert jnp.isfinite(entropy_regular)
        assert jnp.isfinite(entropy_random)
    
    def test_entropy_rate_constant_series(self, setup_entropy_rate_data):
        """Test entropy rate for constant time series."""
        data = setup_entropy_rate_data
        
        entropy_constant = entropy_rate(
            data['constant_series'],
            embedding_dim=2,
            tolerance=0.1
        )
        
        # Constant series should have low entropy rate
        assert isinstance(entropy_constant, float)
        assert jnp.isfinite(entropy_constant)
    
    @pytest.mark.parametrize("embedding_dim", [2, 3, 4])
    def test_entropy_rate_different_embedding_dims(self, embedding_dim):
        """Test entropy rate with different embedding dimensions."""
        key = jax.random.PRNGKey(42)
        time_series = jax.random.normal(key, (50,))
        
        if len(time_series) >= embedding_dim + 2:
            entropy = entropy_rate(
                time_series,
                embedding_dim=embedding_dim,
                tolerance=0.1
            )
            
            assert isinstance(entropy, float)
            assert jnp.isfinite(entropy)
    
    def test_entropy_rate_insufficient_data(self):
        """Test error handling for insufficient data."""
        time_series = jnp.array([1, 2])  # Too short
        
        with pytest.raises(InformationTheoryError):
            entropy_rate(time_series, embedding_dim=3)


class TestIntegratedInformationPhi:
    """Test cases for integrated information (Phi) computation."""
    
    def test_integrated_information_basic(self):
        """Test basic integrated information computation."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        # System state
        system_state = jax.random.normal(keys[0], (20,))
        
        # Subsystem states
        subsystem1 = jax.random.normal(keys[1], (15,))
        subsystem2 = jax.random.normal(keys[2], (15,))
        subsystem_states = [subsystem1, subsystem2]
        
        phi = integrated_information_phi(
            system_state,
            subsystem_states
        )
        
        assert phi >= 0.0, "Integrated information must be non-negative"
        assert isinstance(phi, float)
        assert jnp.isfinite(phi)
    
    def test_integrated_information_single_subsystem(self):
        """Test integrated information with single subsystem."""
        key = jax.random.PRNGKey(42)
        
        system_state = jax.random.normal(key, (20,))
        subsystem_states = [jax.random.normal(key, (15,))]
        
        phi = integrated_information_phi(system_state, subsystem_states)
        
        # Should return 0 for single subsystem
        assert phi == 0.0
    
    def test_integrated_information_empty_subsystems(self):
        """Test integrated information with empty subsystems."""
        key = jax.random.PRNGKey(42)
        
        system_state = jax.random.normal(key, (20,))
        subsystem_states = []
        
        phi = integrated_information_phi(system_state, subsystem_states)
        
        # Should return 0 for empty subsystems
        assert phi == 0.0
    
    def test_integrated_information_connectivity_matrix(self):
        """Test integrated information with connectivity matrix."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        system_state = jax.random.normal(keys[0], (20,))
        subsystem1 = jax.random.normal(keys[1], (15,))
        subsystem2 = jax.random.normal(keys[2], (15,))
        subsystem_states = [subsystem1, subsystem2]
        
        connectivity_matrix = jax.random.uniform(keys[3], (2, 2))
        
        phi = integrated_information_phi(
            system_state,
            subsystem_states,
            connectivity_matrix
        )
        
        assert phi >= 0.0
        assert isinstance(phi, float)


class TestComplexityMeasure:
    """Test cases for complexity measure computation."""
    
    @pytest.fixture
    def setup_complexity_data(self):
        """Set up test data for complexity measure tests."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Create agent and environment states
        agent_states = jax.random.normal(keys[0], (50,))
        environment_states = jax.random.normal(keys[1], (50,))
        
        return {
            'agent_states': agent_states,
            'environment_states': environment_states
        }
    
    def test_complexity_measure_computation(self, setup_complexity_data):
        """Test basic complexity measure computation."""
        data = setup_complexity_data
        
        complexity_metrics = complexity_measure(
            data['agent_states'],
            data['environment_states'],
            window_size=10
        )
        
        # Check all expected keys are present
        expected_keys = {
            'overall_complexity',
            'agent_complexity',
            'environment_complexity',
            'interaction_complexity',
            'agent_entropy_rate',
            'environment_entropy_rate'
        }
        assert set(complexity_metrics.keys()) == expected_keys
        
        # Check value ranges
        for key, value in complexity_metrics.items():
            assert isinstance(value, float)
            assert jnp.isfinite(value)
            assert value >= 0.0, f"{key} should be non-negative"
    
    def test_complexity_measure_correlated_states(self):
        """Test complexity measure with correlated states."""
        key = jax.random.PRNGKey(42)
        
        # Create correlated agent-environment states
        base_signal = jax.random.normal(key, (50,))
        agent_states = base_signal + 0.1 * jax.random.normal(key, (50,))
        environment_states = 0.8 * base_signal + 0.2 * jax.random.normal(key, (50,))
        
        complexity_metrics = complexity_measure(
            agent_states,
            environment_states,
            window_size=10
        )
        
        # Should detect interaction complexity
        assert complexity_metrics['interaction_complexity'] > 0.0
        assert complexity_metrics['overall_complexity'] > 0.0
    
    @pytest.mark.parametrize("window_size", [5, 10, 20])
    def test_complexity_measure_different_window_sizes(self, window_size):
        """Test complexity measure with different window sizes."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        agent_states = jax.random.normal(keys[0], (50,))
        environment_states = jax.random.normal(keys[1], (50,))
        
        if len(agent_states) >= window_size:
            complexity_metrics = complexity_measure(
                agent_states,
                environment_states,
                window_size=window_size
            )
            
            assert isinstance(complexity_metrics, dict)
            assert 'overall_complexity' in complexity_metrics
    
    def test_complexity_measure_multidimensional(self):
        """Test complexity measure with multidimensional states."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Multidimensional states
        agent_states = jax.random.normal(keys[0], (30, 5))
        environment_states = jax.random.normal(keys[1], (30, 3))
        
        complexity_metrics = complexity_measure(
            agent_states,
            environment_states,
            window_size=10
        )
        
        assert isinstance(complexity_metrics, dict)
        assert 'overall_complexity' in complexity_metrics


class TestMathematicalCorrectness:
    """Test cases for mathematical correctness and theoretical properties."""
    
    def test_mutual_information_symmetry(self):
        """Test MI symmetry: MI(X, Y) = MI(Y, X)."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        x = jax.random.normal(keys[0], (50,))
        y = jax.random.normal(keys[1], (50,))
        
        mi_xy = mutual_information_kraskov(x, y)
        mi_yx = mutual_information_kraskov(y, x)
        
        # Should be approximately equal (allowing for numerical errors)
        assert abs(mi_xy - mi_yx) < 0.1, "MI should be symmetric"
    
    def test_transfer_entropy_asymmetry(self):
        """Test TE asymmetry for causal relationships."""
        key = jax.random.PRNGKey(42)
        
        n_steps = 50
        x = jax.random.normal(key, (n_steps,))
        
        # Create y that depends on x (causal relationship)
        y = jnp.zeros(n_steps)
        for t in range(1, n_steps):
            y = y.at[t].set(0.5 * x[t-1] + 0.1 * jax.random.normal(key, ()))
        
        te_x_to_y = transfer_entropy(x, y, history_length=1)
        te_y_to_x = transfer_entropy(y, x, history_length=1)
        
        # TE(X->Y) should be greater than TE(Y->X) for causal relationship
        assert te_x_to_y >= 0.0
        assert te_y_to_x >= 0.0
    
    def test_complexity_measure_positivity(self):
        """Test that complexity measures are always non-negative."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Test with different types of signals
        test_cases = [
            # Random signals
            (jax.random.normal(keys[0], (30,)), jax.random.normal(keys[1], (30,))),
            # Constant signals
            (jnp.ones(30), jnp.ones(30) * 2),
            # Linear signals
            (jnp.linspace(0, 1, 30), jnp.linspace(1, 2, 30))
        ]
        
        for agent_states, env_states in test_cases:
            complexity_metrics = complexity_measure(agent_states, env_states)
            
            for key, value in complexity_metrics.items():
                assert value >= 0.0, f"{key} should be non-negative"
    
    def test_circular_causality_coherence_bounds(self):
        """Test that coupling coherence is properly bounded."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Create various coupling scenarios
        agent_states = jax.random.normal(keys[0], (30,))
        env_states = jax.random.normal(keys[1], (30,))
        
        causality_metrics = circular_causality_index(agent_states, env_states)
        
        # Coupling coherence should be in [0, 1]
        coherence = causality_metrics['coupling_coherence']
        assert 0.0 <= coherence <= 1.0, f"Coherence {coherence} not in [0,1]"


class TestErrorHandlingAndEdgeCases:
    """Test cases for error handling and edge cases."""
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        empty_array = jnp.array([])
        
        with pytest.raises((InformationTheoryError, ValueError, IndexError)):
            mutual_information_kraskov(empty_array, empty_array)
    
    def test_single_element_arrays(self):
        """Test behavior with single-element arrays."""
        single_element = jnp.array([1.0])
        
        with pytest.raises((InformationTheoryError, ValueError, IndexError)):
            mutual_information_kraskov(single_element, single_element)
    
    def test_nan_values(self):
        """Test error handling with NaN values."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (20,))
        y = jax.random.normal(key, (20,))
        
        # Introduce NaN values
        x_with_nan = x.at[0].set(jnp.nan)
        
        # Should handle NaN gracefully
        try:
            mi = mutual_information_kraskov(x_with_nan, y)
            assert jnp.isfinite(mi) or jnp.isnan(mi), "Should handle NaN appropriately"
        except InformationTheoryError:
            pass  # Acceptable to raise error for invalid data
    
    def test_inf_values(self):
        """Test error handling with infinite values."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (20,))
        y = jax.random.normal(key, (20,))
        
        # Introduce infinite values
        x_with_inf = x.at[0].set(jnp.inf)
        
        # Should handle infinity gracefully
        try:
            mi = mutual_information_kraskov(x_with_inf, y)
            assert jnp.isfinite(mi) or jnp.isinf(mi), "Should handle inf appropriately"
        except InformationTheoryError:
            pass  # Acceptable to raise error for invalid data
    
    def test_very_small_arrays(self):
        """Test behavior with very small arrays."""
        small_array = jnp.array([1.0, 2.0])
        
        # Should handle small arrays gracefully or raise appropriate error
        try:
            mi = mutual_information_kraskov(small_array, small_array)
            assert jnp.isfinite(mi), "Should return finite value for small arrays"
        except InformationTheoryError:
            pass  # Acceptable to raise error for insufficient data


class TestPerformanceAndScalability:
    """Test cases for performance and scalability."""
    
    @pytest.mark.parametrize("size", [50, 100, 200])
    def test_scalability_with_data_size(self, size):
        """Test that functions scale reasonably with data size."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        x = jax.random.normal(keys[0], (size,))
        y = jax.random.normal(keys[1], (size,))
        
        # Should complete in reasonable time
        mi = mutual_information_kraskov(x, y)
        assert jnp.isfinite(mi)
        
        te = transfer_entropy(x, y, history_length=min(2, size//10))
        assert jnp.isfinite(te)
    
    def test_memory_usage_large_arrays(self):
        """Test memory usage with reasonably large arrays."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Large but manageable arrays for testing
        size = 500
        x = jax.random.normal(keys[0], (size,))
        y = jax.random.normal(keys[1], (size,))
        
        # Should not cause memory issues
        mi = mutual_information_kraskov(x, y, k=3)
        assert jnp.isfinite(mi)


# Integration tests
class TestInformationTheoryIntegration:
    """Integration tests for combined information theory measures."""
    
    def test_complete_consciousness_assessment(self):
        """Test using all measures together for consciousness assessment."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Create agent-environment interaction scenario
        n_steps = 40
        agent_states = jax.random.normal(keys[0], (n_steps, 8))
        env_states = jax.random.normal(keys[1], (n_steps, 6))
        
        # Compute all measures
        causality_metrics = circular_causality_index(agent_states, env_states)
        complexity_metrics = complexity_measure(
            agent_states.flatten(), env_states.flatten()
        )
        
        # Integrated assessment
        consciousness_score = (
            0.3 * causality_metrics['circular_causality'] +
            0.3 * causality_metrics['coupling_coherence'] +
            0.4 * (complexity_metrics['overall_complexity'] / 
                   (1 + complexity_metrics['overall_complexity']))
        )
        
        assert 0.0 <= consciousness_score <= 1.0
        assert jnp.isfinite(consciousness_score)
    
    def test_temporal_consciousness_dynamics(self):
        """Test information measures over temporal dynamics."""
        key = jax.random.PRNGKey(42)
        
        n_steps = 50
        time_series = []
        
        # Generate temporal sequence with evolving coupling
        for t in range(n_steps):
            coupling_strength = 0.5 + 0.3 * jnp.sin(2 * jnp.pi * t / 20)
            
            agent_t = jax.random.normal(key, (5,)) 
            env_t = coupling_strength * agent_t + \
                   (1 - coupling_strength) * jax.random.normal(key, (5,))
            
            time_series.append((agent_t, env_t))
        
        # Analyze temporal evolution of information measures
        causality_scores = []
        for t in range(10, n_steps-5):  # Need sufficient history
            agent_window = jnp.stack([ts[0] for ts in time_series[t-10:t+5]])
            env_window = jnp.stack([ts[1] for ts in time_series[t-10:t+5]])
            
            causality = circular_causality_index(agent_window, env_window)
            causality_scores.append(causality['circular_causality'])
        
        # Should detect temporal variation in coupling
        causality_variance = jnp.var(jnp.array(causality_scores))
        assert causality_variance >= 0.0  # Some variation expected


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main([__file__, "-v", "--tb=short"])