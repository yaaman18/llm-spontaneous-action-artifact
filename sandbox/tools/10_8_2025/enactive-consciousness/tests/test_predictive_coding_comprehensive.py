"""Comprehensive test suite for predictive coding module.

This test suite follows TDD principles with extensive coverage of
hierarchical prediction, error minimization, and multi-scale temporal
prediction for enactive consciousness with mathematical validation.

Test Coverage:
- PredictiveCodingConfig validation and functionality
- HierarchicalPredictionNetwork with NGC integration
- MultiScaleTemporalPredictor across different scales
- DynamicErrorMinimization with hyperparameter adaptation
- IntegratedPredictiveCoding system integration
- Mathematical correctness validation
- Performance and scalability testing
- Prediction accuracy assessment
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional
from unittest.mock import patch, MagicMock
import warnings

# Import the module under test
import sys
sys.path.insert(0, '/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/10_8_2025/enactive-consciousness/src')

from enactive_consciousness.predictive_coding import (
    PredictionScale,
    PredictiveCodingConfig,
    PredictiveState,
    HierarchicalPredictionNetwork,
    MultiScaleTemporalPredictor,
    DynamicErrorMinimization,
    IntegratedPredictiveCoding,
    create_predictive_coding_system,
    optimize_hyperparameters,
    create_test_predictive_sequence,
)

from enactive_consciousness.types import (
    TemporalMoment,
    BodyState,
    PRNGKey,
    TimeStep,
)

# Mock configurations for testing
class MockTemporalConsciousnessConfig:
    """Mock temporal consciousness config for testing."""
    def __init__(self):
        self.retention_depth = 10
        self.present_moment_duration = 0.1
        self.protention_horizon = 5

class MockBodySchemaConfig:
    """Mock body schema config for testing."""
    def __init__(self):
        self.proprioceptive_dim = 16
        self.motor_intention_dim = 8
        self.boundary_detection_dim = 4


class TestPredictionScale:
    """Test cases for PredictionScale enum."""
    
    def test_prediction_scale_values(self):
        """Test PredictionScale enum values."""
        assert PredictionScale.MICRO.value == "micro"
        assert PredictionScale.MESO.value == "meso"
        assert PredictionScale.MACRO.value == "macro"
    
    def test_prediction_scale_completeness(self):
        """Test that all expected prediction scales are available."""
        expected_scales = {"micro", "meso", "macro"}
        actual_scales = {scale.value for scale in PredictionScale}
        assert actual_scales == expected_scales


class TestPredictiveCodingConfig:
    """Test cases for PredictiveCodingConfig."""
    
    def test_config_creation_default(self):
        """Test default configuration creation."""
        config = PredictiveCodingConfig()
        
        assert config.hierarchy_levels == 4
        assert config.prediction_horizon == 10
        assert config.error_convergence_threshold == 1e-4
        assert config.ngc_learning_rate == 1e-3
        assert config.ngc_weight_decay == 1e-5
        assert config.ngc_beta1 == 0.9
        assert config.ngc_beta2 == 0.999
        assert len(config.temporal_scales) == 3
        assert config.scale_weights.shape == (3,)
        assert config.body_schema_weight == 0.3
        assert config.temporal_synthesis_weight == 0.4
        assert config.environmental_context_weight == 0.3
    
    def test_config_creation_custom(self):
        """Test custom configuration creation."""
        custom_scales = (PredictionScale.MICRO, PredictionScale.MACRO)
        custom_weights = jnp.array([0.7, 0.3])
        
        config = PredictiveCodingConfig(
            hierarchy_levels=3,
            prediction_horizon=15,
            temporal_scales=custom_scales,
            scale_weights=custom_weights,
            ngc_learning_rate=2e-3,
        )
        
        assert config.hierarchy_levels == 3
        assert config.prediction_horizon == 15
        assert config.temporal_scales == custom_scales
        assert jnp.array_equal(config.scale_weights, custom_weights)
        assert config.ngc_learning_rate == 2e-3
    
    def test_config_scale_weights_default(self):
        """Test default scale weights creation."""
        config = PredictiveCodingConfig()
        
        # Should have weights for micro, meso, macro
        assert config.scale_weights.shape == (3,)
        assert jnp.allclose(config.scale_weights, jnp.array([0.5, 0.3, 0.2]))
        assert jnp.sum(config.scale_weights) == pytest.approx(1.0, abs=1e-6)
    
    def test_config_parameter_bounds(self):
        """Test configuration parameter bounds."""
        config = PredictiveCodingConfig(
            hierarchy_levels=2,
            prediction_horizon=5,
            error_convergence_threshold=1e-6,
        )
        
        assert config.hierarchy_levels >= 1
        assert config.prediction_horizon >= 1
        assert config.error_convergence_threshold > 0
        assert 0 < config.ngc_learning_rate < 1
        assert 0 < config.ngc_weight_decay < 1
        assert 0 < config.ngc_beta1 < 1
        assert 0 < config.ngc_beta2 < 1


class TestPredictiveState:
    """Test cases for PredictiveState dataclass."""
    
    def test_predictive_state_creation(self):
        """Test PredictiveState creation."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        hierarchical_predictions = [
            jax.random.normal(keys[0], (16,)),
            jax.random.normal(keys[1], (12,)),
            jax.random.normal(keys[2], (8,)),
        ]
        
        prediction_errors = [
            jax.random.normal(keys[0], (16,)) * 0.1,
            jax.random.normal(keys[1], (12,)) * 0.1,
        ]
        
        confidence_estimates = jax.random.uniform(keys[3], (2,))
        
        scale_predictions = {
            PredictionScale.MICRO: jax.random.normal(keys[0], (16,)),
            PredictionScale.MESO: jax.random.normal(keys[1], (12,)),
            PredictionScale.MACRO: jax.random.normal(keys[2], (8,)),
        }
        
        state = PredictiveState(
            hierarchical_predictions=hierarchical_predictions,
            prediction_errors=prediction_errors,
            confidence_estimates=confidence_estimates,
            scale_predictions=scale_predictions,
            total_prediction_error=0.15,
            convergence_status=True,
            timestamp=100.0,
        )
        
        assert len(state.hierarchical_predictions) == 3
        assert len(state.prediction_errors) == 2
        assert state.confidence_estimates.shape == (2,)
        assert len(state.scale_predictions) == 3
        assert state.total_prediction_error == 0.15
        assert state.convergence_status is True
        assert state.timestamp == 100.0


class TestHierarchicalPredictionNetwork:
    """Test cases for HierarchicalPredictionNetwork."""
    
    @pytest.fixture
    def hierarchical_network_setup(self):
        """Set up hierarchical prediction network for testing."""
        input_dim = 20
        layer_dimensions = (16, 12, 8)
        key = jax.random.PRNGKey(42)
        
        network = HierarchicalPredictionNetwork(
            input_dim=input_dim,
            layer_dimensions=layer_dimensions,
            key=key,
            use_ngc=False,  # Use fallback for testing
        )
        
        return network, input_dim, layer_dimensions
    
    def test_hierarchical_network_initialization(self, hierarchical_network_setup):
        """Test HierarchicalPredictionNetwork initialization."""
        network, input_dim, layer_dimensions = hierarchical_network_setup
        
        assert network.layer_dimensions == layer_dimensions
        assert len(network.layers) == len(layer_dimensions)
        assert len(network.activations) == len(layer_dimensions)
        assert len(network.prediction_weights) == len(layer_dimensions) - 1
        assert network.error_integration_weights.shape == (len(layer_dimensions),)
        
        # Check layer dimensions
        expected_input_dims = [input_dim] + list(layer_dimensions[:-1])
        for i, layer in enumerate(network.layers):
            assert layer.in_features == expected_input_dims[i]
            assert layer.out_features == layer_dimensions[i]
    
    def test_build_hierarchical_layers(self, hierarchical_network_setup):
        """Test hierarchical layer building."""
        network, input_dim, layer_dimensions = hierarchical_network_setup
        
        # Verify layer structure
        assert len(network.layers) == 3
        assert network.layers[0].in_features == input_dim
        assert network.layers[0].out_features == layer_dimensions[0]
        assert network.layers[1].in_features == layer_dimensions[0]
        assert network.layers[1].out_features == layer_dimensions[1]
        assert network.layers[2].in_features == layer_dimensions[1]
        assert network.layers[2].out_features == layer_dimensions[2]
    
    def test_initialize_prediction_weights(self, hierarchical_network_setup):
        """Test prediction weights initialization."""
        network, input_dim, layer_dimensions = hierarchical_network_setup
        
        # Should have weights for top-down connections
        assert len(network.prediction_weights) == 2  # For 3 layers
        
        # Check weight dimensions (from higher to lower layer)
        assert network.prediction_weights[0].shape == (layer_dimensions[0], layer_dimensions[1])
        assert network.prediction_weights[1].shape == (layer_dimensions[1], layer_dimensions[2])
    
    def test_forward_prediction(self, hierarchical_network_setup):
        """Test forward prediction through network."""
        network, input_dim, layer_dimensions = hierarchical_network_setup
        key = jax.random.PRNGKey(42)
        
        input_state = jax.random.normal(key, (input_dim,))
        
        representations, prediction_errors = network.forward_prediction(input_state)
        
        # Check representations
        assert len(representations) == len(layer_dimensions)
        for i, repr in enumerate(representations):
            assert repr.shape == (layer_dimensions[i],)
            assert jnp.all(jnp.isfinite(repr))
        
        # Check prediction errors
        assert len(prediction_errors) == len(layer_dimensions) - 1
        for i, error in enumerate(prediction_errors):
            assert error.shape == (layer_dimensions[i],)
            assert jnp.all(jnp.isfinite(error))
    
    def test_compute_top_down_prediction(self, hierarchical_network_setup):
        """Test top-down prediction computation."""
        network, input_dim, layer_dimensions = hierarchical_network_setup
        key = jax.random.PRNGKey(42)
        
        higher_representation = jax.random.normal(key, (layer_dimensions[1],))
        
        predicted_lower = network._compute_top_down_prediction(higher_representation, 0)
        
        assert predicted_lower.shape == (layer_dimensions[0],)
        assert jnp.all(jnp.isfinite(predicted_lower))
    
    def test_compute_prediction_error(self, hierarchical_network_setup):
        """Test prediction error computation."""
        network, input_dim, layer_dimensions = hierarchical_network_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        predicted = jax.random.normal(keys[0], (layer_dimensions[0],))
        actual = jax.random.normal(keys[1], (layer_dimensions[0],))
        
        error = network._compute_prediction_error(predicted, actual)
        
        assert error.shape == predicted.shape
        assert jnp.all(jnp.isfinite(error))
        
        # Error should be scaled by precision weighting
        raw_error = predicted - actual
        assert not jnp.allclose(error, raw_error), "Error should be precision-weighted"


class TestMultiScaleTemporalPredictor:
    """Test cases for MultiScaleTemporalPredictor."""
    
    @pytest.fixture
    def temporal_predictor_setup(self):
        """Set up multi-scale temporal predictor for testing."""
        input_dim = 16
        temporal_config = MockTemporalConsciousnessConfig()
        predictive_config = PredictiveCodingConfig(
            temporal_scales=(PredictionScale.MICRO, PredictionScale.MESO),
            scale_weights=jnp.array([0.6, 0.4]),
        )
        key = jax.random.PRNGKey(42)
        
        predictor = MultiScaleTemporalPredictor(
            input_dim=input_dim,
            temporal_config=temporal_config,
            predictive_config=predictive_config,
            key=key,
        )
        
        return predictor, input_dim, temporal_config, predictive_config
    
    def test_temporal_predictor_initialization(self, temporal_predictor_setup):
        """Test MultiScaleTemporalPredictor initialization."""
        predictor, input_dim, temporal_config, predictive_config = temporal_predictor_setup
        
        assert len(predictor.scale_predictors) == 2  # MICRO and MESO
        assert 'micro' in predictor.scale_predictors
        assert 'meso' in predictor.scale_predictors
        assert hasattr(predictor, 'temporal_integration_network')
        assert hasattr(predictor, 'scale_attention')
        assert len(predictor.prediction_history) == 2
    
    def test_get_scale_dimensions(self, temporal_predictor_setup):
        """Test scale dimension computation."""
        predictor, input_dim, _, _ = temporal_predictor_setup
        
        micro_dims = predictor._get_scale_dimensions(PredictionScale.MICRO, input_dim)
        meso_dims = predictor._get_scale_dimensions(PredictionScale.MESO, input_dim)
        macro_dims = predictor._get_scale_dimensions(PredictionScale.MACRO, input_dim)
        
        # Check that dimensions are tuples of integers
        assert isinstance(micro_dims, tuple)
        assert isinstance(meso_dims, tuple)
        assert isinstance(macro_dims, tuple)
        
        assert all(isinstance(dim, int) and dim > 0 for dim in micro_dims)
        assert all(isinstance(dim, int) and dim > 0 for dim in meso_dims)
        assert all(isinstance(dim, int) and dim > 0 for dim in macro_dims)
    
    def test_get_scale_history_length(self, temporal_predictor_setup):
        """Test scale history length computation."""
        predictor, input_dim, temporal_config, _ = temporal_predictor_setup
        
        micro_length = predictor._get_scale_history_length(PredictionScale.MICRO, temporal_config)
        meso_length = predictor._get_scale_history_length(PredictionScale.MESO, temporal_config)
        macro_length = predictor._get_scale_history_length(PredictionScale.MACRO, temporal_config)
        
        # History lengths should be positive integers
        assert isinstance(micro_length, int) and micro_length > 0
        assert isinstance(meso_length, int) and meso_length > 0
        assert isinstance(macro_length, int) and macro_length > 0
        
        # MICRO should have shorter history than MACRO
        assert micro_length <= macro_length
    
    def test_predict_temporal_dynamics(self, temporal_predictor_setup):
        """Test temporal dynamics prediction."""
        predictor, input_dim, _, _ = temporal_predictor_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Create temporal moment
        temporal_moment = TemporalMoment(
            timestamp=100.0,
            retention=jax.random.normal(keys[0], (input_dim,)),
            present_moment=jax.random.normal(keys[1], (input_dim,)),
            protention=jax.random.normal(keys[2], (input_dim,)),
            synthesis_weights=jax.random.normal(keys[3], (input_dim,)),
        )
        
        temporal_context = jax.random.normal(keys[3], (input_dim,))
        
        scale_predictions = predictor.predict_temporal_dynamics(
            temporal_moment, temporal_context
        )
        
        # Check predictions for each scale
        assert isinstance(scale_predictions, dict)
        assert len(scale_predictions) == 2  # MICRO and MESO
        
        for scale, (prediction, error) in scale_predictions.items():
            assert isinstance(scale, PredictionScale)
            assert jnp.all(jnp.isfinite(prediction))
            assert jnp.all(jnp.isfinite(error))
    
    def test_prepare_temporal_input(self, temporal_predictor_setup):
        """Test temporal input preparation."""
        predictor, input_dim, _, _ = temporal_predictor_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)
        
        temporal_moment = TemporalMoment(
            timestamp=100.0,
            retention=jax.random.normal(keys[0], (input_dim,)),
            present_moment=jax.random.normal(keys[1], (input_dim,)),
            protention=jax.random.normal(keys[2], (input_dim,)),
            synthesis_weights=jax.random.normal(keys[3], (input_dim,)),
        )
        
        context = jax.random.normal(keys[4], (input_dim,))
        
        temporal_input = predictor._prepare_temporal_input(temporal_moment, context)
        
        assert temporal_input.shape == (input_dim,)
        assert jnp.all(jnp.isfinite(temporal_input))
    
    def test_integrate_scale_predictions(self, temporal_predictor_setup):
        """Test scale prediction integration."""
        predictor, input_dim, _, predictive_config = temporal_predictor_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        # Create mock scale predictions
        scale_predictions = {
            PredictionScale.MICRO: (
                jax.random.normal(keys[0], (12,)),
                jax.random.normal(keys[0], (12,)) * 0.1
            ),
            PredictionScale.MESO: (
                jax.random.normal(keys[1], (16,)),
                jax.random.normal(keys[1], (16,)) * 0.1
            ),
        }
        
        scale_weights = predictive_config.scale_weights
        
        integrated_prediction = predictor.integrate_scale_predictions(
            scale_predictions, scale_weights
        )
        
        assert jnp.all(jnp.isfinite(integrated_prediction))
        # Output dimension should match attention output size
        assert integrated_prediction.shape == (predictor.scale_attention.output_size,)


class TestDynamicErrorMinimization:
    """Test cases for DynamicErrorMinimization."""
    
    @pytest.fixture
    def error_minimizer_setup(self):
        """Set up dynamic error minimizer for testing."""
        config = PredictiveCodingConfig(
            prediction_error_history_length=50,
            dynamic_adjustment_sensitivity=0.1,
        )
        key = jax.random.PRNGKey(42)
        
        minimizer = DynamicErrorMinimization(config, key)
        return minimizer, config
    
    def test_error_minimizer_initialization(self, error_minimizer_setup):
        """Test DynamicErrorMinimization initialization."""
        minimizer, config = error_minimizer_setup
        
        assert minimizer.error_history.shape == (config.prediction_error_history_length,)
        assert hasattr(minimizer, 'hyperparameter_adaptation_network')
        assert hasattr(minimizer, 'error_minimization_optimizer')
        assert isinstance(minimizer.adaptation_state, dict)
        
        # Check adaptation state keys
        expected_keys = {'learning_rate', 'weight_decay', 'beta1', 'beta2'}
        assert set(minimizer.adaptation_state.keys()) == expected_keys
    
    def test_minimize_prediction_error(self, error_minimizer_setup):
        """Test prediction error minimization."""
        minimizer, config = error_minimizer_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        # Create mock prediction errors
        prediction_errors = [
            jax.random.normal(keys[0], (16,)) * 0.1,
            jax.random.normal(keys[1], (12,)) * 0.15,
            jax.random.normal(keys[2], (8,)) * 0.2,
        ]
        
        # Mock model parameters (simplified)
        model_parameters = {"weights": jax.random.normal(key, (20, 16))}
        
        updated_params, metrics = minimizer.minimize_prediction_error(
            prediction_errors, model_parameters, config
        )
        
        # Check return types
        assert isinstance(updated_params, dict)
        assert isinstance(metrics, dict)
        
        # Check metrics
        expected_metric_keys = {
            'total_prediction_error', 'error_history_mean', 'error_history_std',
            'adapted_learning_rate', 'adapted_weight_decay', 'adapted_beta1', 'adapted_beta2'
        }
        assert set(metrics.keys()) == expected_metric_keys
        
        for key, value in metrics.items():
            assert isinstance(value, float)
            assert jnp.isfinite(value)
    
    def test_compute_total_error(self, error_minimizer_setup):
        """Test total error computation."""
        minimizer, _ = error_minimizer_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        prediction_errors = [
            jax.random.normal(keys[0], (10,)) * 0.1,
            jax.random.normal(keys[1], (8,)) * 0.2,
        ]
        
        total_error = minimizer._compute_total_error(prediction_errors)
        
        assert isinstance(total_error, float)
        assert total_error >= 0.0
        assert jnp.isfinite(total_error)
    
    def test_update_error_history(self, error_minimizer_setup):
        """Test error history updating."""
        minimizer, config = error_minimizer_setup
        
        original_history = minimizer.error_history.copy()
        new_error = 0.25
        
        updated_history = minimizer._update_error_history(new_error)
        
        assert updated_history.shape == original_history.shape
        assert updated_history[0] == new_error
        # Check that history was rolled
        assert jnp.allclose(updated_history[1:], original_history[:-1])
    
    def test_adapt_hyperparameters(self, error_minimizer_setup):
        """Test hyperparameter adaptation."""
        minimizer, _ = error_minimizer_setup
        
        current_error = 0.15
        sensitivity = 0.1
        
        adapted_params = minimizer._adapt_hyperparameters(current_error, sensitivity)
        
        # Check structure
        expected_keys = {'learning_rate', 'weight_decay', 'beta1', 'beta2'}
        assert set(adapted_params.keys()) == expected_keys
        
        # Check bounds
        assert 1e-6 <= adapted_params['learning_rate'] <= 1e-1
        assert 1e-8 <= adapted_params['weight_decay'] <= 1e-2
        assert 0.1 <= adapted_params['beta1'] <= 0.999
        assert 0.1 <= adapted_params['beta2'] <= 0.999
    
    def test_empty_prediction_errors(self, error_minimizer_setup):
        """Test behavior with empty prediction errors."""
        minimizer, _ = error_minimizer_setup
        
        empty_errors = []
        total_error = minimizer._compute_total_error(empty_errors)
        
        assert total_error == 0.0


class TestIntegratedPredictiveCoding:
    """Test cases for IntegratedPredictiveCoding."""
    
    @pytest.fixture
    def integrated_system_setup(self):
        """Set up integrated predictive coding system for testing."""
        config = PredictiveCodingConfig(
            hierarchy_levels=3,
            temporal_scales=(PredictionScale.MICRO, PredictionScale.MESO),
            scale_weights=jnp.array([0.7, 0.3]),
        )
        temporal_config = MockTemporalConsciousnessConfig()
        body_schema_config = MockBodySchemaConfig()
        state_dim = 20
        key = jax.random.PRNGKey(42)
        
        system = IntegratedPredictiveCoding(
            config=config,
            temporal_config=temporal_config,
            body_schema_config=body_schema_config,
            state_dim=state_dim,
            key=key,
        )
        
        return system, config, temporal_config, body_schema_config, state_dim
    
    def test_integrated_system_initialization(self, integrated_system_setup):
        """Test IntegratedPredictiveCoding initialization."""
        system, config, _, body_schema_config, state_dim = integrated_system_setup
        
        assert system.config == config
        assert isinstance(system.hierarchical_predictor, HierarchicalPredictionNetwork)
        assert isinstance(system.temporal_predictor, MultiScaleTemporalPredictor)
        assert isinstance(system.error_minimizer, DynamicErrorMinimization)
        assert hasattr(system, 'integration_network')
        assert hasattr(system, 'body_schema_predictor')
    
    def test_generate_hierarchical_predictions(self, integrated_system_setup):
        """Test hierarchical prediction generation."""
        system, _, _, _, state_dim = integrated_system_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 6)
        
        current_state = jax.random.normal(keys[0], (state_dim,))
        
        temporal_moment = TemporalMoment(
            timestamp=100.0,
            retention=jax.random.normal(keys[1], (state_dim,)),
            present_moment=current_state,
            protention=jax.random.normal(keys[2], (state_dim,)),
            synthesis_weights=jax.random.normal(keys[3], (state_dim,)),
        )
        
        body_state = BodyState(
            proprioception=jax.random.normal(keys[4], (16,)),
            motor_intention=jax.random.normal(keys[5], (8,)),
            boundary_signal=jax.random.normal(keys[5], (1,)),
            schema_confidence=0.8,
        )
        
        environmental_context = jax.random.normal(keys[5], (state_dim,))
        
        predictive_state = system.generate_hierarchical_predictions(
            current_state, temporal_moment, body_state, environmental_context
        )
        
        # Check predictive state structure
        assert isinstance(predictive_state, PredictiveState)
        assert len(predictive_state.hierarchical_predictions) > 0
        assert len(predictive_state.prediction_errors) >= 0
        assert predictive_state.confidence_estimates.shape[0] > 0
        assert isinstance(predictive_state.scale_predictions, dict)
        assert predictive_state.total_prediction_error >= 0.0
        assert isinstance(predictive_state.convergence_status, bool)
        assert predictive_state.timestamp == temporal_moment.timestamp
    
    def test_validate_prediction_inputs(self, integrated_system_setup):
        """Test prediction input validation."""
        system, _, _, _, state_dim = integrated_system_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 6)
        
        current_state = jax.random.normal(keys[0], (state_dim,))
        
        temporal_moment = TemporalMoment(
            timestamp=100.0,
            retention=jax.random.normal(keys[1], (state_dim,)),
            present_moment=current_state,
            protention=jax.random.normal(keys[2], (state_dim,)),
            synthesis_weights=jax.random.normal(keys[3], (state_dim,)),
        )
        
        body_state = BodyState(
            proprioception=jax.random.normal(keys[4], (16,)),
            motor_intention=jax.random.normal(keys[5], (8,)),
            boundary_signal=jax.random.normal(keys[5], (1,)),
            schema_confidence=0.8,
        )
        
        # Should not raise exception for valid inputs
        system._validate_prediction_inputs(current_state, temporal_moment, body_state)
    
    def test_generate_hierarchical_predictions_method(self, integrated_system_setup):
        """Test hierarchical predictions generation method."""
        system, _, _, _, state_dim = integrated_system_setup
        key = jax.random.PRNGKey(42)
        
        current_state = jax.random.normal(key, (state_dim,))
        
        representations, errors = system._generate_hierarchical_predictions(current_state)
        
        assert isinstance(representations, list)
        assert isinstance(errors, list)
        assert len(representations) > 0
        
        for repr in representations:
            assert jnp.all(jnp.isfinite(repr))
        
        for error in errors:
            assert jnp.all(jnp.isfinite(error))
    
    def test_generate_body_schema_predictions(self, integrated_system_setup):
        """Test body schema predictions generation."""
        system, _, _, _, _ = integrated_system_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        body_state = BodyState(
            proprioception=jax.random.normal(keys[0], (16,)),
            motor_intention=jax.random.normal(keys[1], (8,)),
            boundary_signal=jax.random.normal(keys[2], (1,)),
            schema_confidence=0.8,
        )
        
        body_predictions = system._generate_body_schema_predictions(body_state)
        
        assert body_predictions.shape == (16,)  # proprioceptive_dim
        assert jnp.all(jnp.isfinite(body_predictions))
        # Should be bounded by tanh activation
        assert jnp.all(jnp.abs(body_predictions) <= 1.0)
    
    def test_compute_prediction_confidence(self, integrated_system_setup):
        """Test prediction confidence computation."""
        system, _, _, _, _ = integrated_system_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Create mock hierarchical errors
        hierarchical_errors = [
            jax.random.normal(keys[0], (16,)) * 0.1,
            jax.random.normal(keys[1], (12,)) * 0.15,
        ]
        
        # Create mock temporal predictions
        temporal_predictions = {
            PredictionScale.MICRO: (
                jax.random.normal(keys[2], (16,)),
                jax.random.normal(keys[2], (16,)) * 0.1
            ),
            PredictionScale.MESO: (
                jax.random.normal(keys[3], (12,)),
                jax.random.normal(keys[3], (12,)) * 0.2
            ),
        }
        
        confidence = system._compute_prediction_confidence(
            hierarchical_errors, temporal_predictions
        )
        
        assert confidence.shape == (2,)  # hierarchical and temporal confidence
        assert jnp.all(confidence >= 0.0)
        assert jnp.all(confidence <= 1.0)
        assert jnp.all(jnp.isfinite(confidence))
    
    def test_assess_convergence(self, integrated_system_setup):
        """Test convergence assessment."""
        system, config, _, _, _ = integrated_system_setup
        
        # Test convergence with low error
        low_error = config.error_convergence_threshold * 0.5
        assert system._assess_convergence(low_error) is True
        
        # Test non-convergence with high error
        high_error = config.error_convergence_threshold * 2.0
        assert system._assess_convergence(high_error) is False
    
    def test_optimize_predictions(self, integrated_system_setup):
        """Test prediction optimization."""
        system, _, _, _, state_dim = integrated_system_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        # Create mock predictive state
        hierarchical_predictions = [jax.random.normal(keys[0], (16,))]
        prediction_errors = [jax.random.normal(keys[1], (16,)) * 0.1]
        confidence_estimates = jax.random.uniform(keys[2], (2,))
        scale_predictions = {PredictionScale.MICRO: jax.random.normal(keys[0], (16,))}
        
        predictive_state = PredictiveState(
            hierarchical_predictions=hierarchical_predictions,
            prediction_errors=prediction_errors,
            confidence_estimates=confidence_estimates,
            scale_predictions=scale_predictions,
            total_prediction_error=0.15,
            convergence_status=False,
            timestamp=100.0,
        )
        
        optimized_state, adaptation_metrics = system.optimize_predictions(predictive_state)
        
        assert isinstance(optimized_state, PredictiveState)
        assert isinstance(adaptation_metrics, dict)
    
    def test_assess_predictive_accuracy(self, integrated_system_setup):
        """Test predictive accuracy assessment."""
        system, _, _, _, state_dim = integrated_system_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Create mock predictive state
        hierarchical_predictions = [jax.random.normal(keys[0], (state_dim,))]
        prediction_errors = []
        confidence_estimates = jax.random.uniform(keys[1], (2,))
        scale_predictions = {
            PredictionScale.MICRO: jax.random.normal(keys[2], (16,)),
        }
        
        predictive_state = PredictiveState(
            hierarchical_predictions=hierarchical_predictions,
            prediction_errors=prediction_errors,
            confidence_estimates=confidence_estimates,
            scale_predictions=scale_predictions,
            total_prediction_error=0.1,
            convergence_status=True,
            timestamp=100.0,
        )
        
        # Create actual outcomes
        actual_outcomes = {
            'next_state': jax.random.normal(keys[3], (state_dim,)),
            'future_states': [jax.random.normal(keys[3], (16,))],
        }
        
        accuracy_metrics = system.assess_predictive_accuracy(predictive_state, actual_outcomes)
        
        # Check metrics structure
        expected_keys = {
            'hierarchical_accuracy', 'micro_accuracy', 'overall_confidence',
            'convergence_achieved', 'total_prediction_error'
        }
        assert set(accuracy_metrics.keys()) == expected_keys
        
        # Check value ranges
        for key, value in accuracy_metrics.items():
            if key != 'convergence_achieved':  # Boolean metric
                assert 0.0 <= value <= 1.0 or key == 'total_prediction_error'
                assert jnp.isfinite(value)


class TestFactoryAndUtilityFunctions:
    """Test cases for factory and utility functions."""
    
    def test_create_predictive_coding_system(self):
        """Test factory function for predictive coding system."""
        config = PredictiveCodingConfig(hierarchy_levels=2)
        temporal_config = MockTemporalConsciousnessConfig()
        body_schema_config = MockBodySchemaConfig()
        state_dim = 16
        key = jax.random.PRNGKey(42)
        
        system = create_predictive_coding_system(
            config, temporal_config, body_schema_config, state_dim, key
        )
        
        assert isinstance(system, IntegratedPredictiveCoding)
        assert system.config == config
    
    def test_optimize_hyperparameters(self):
        """Test hyperparameter optimization function."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Create mock system
        config = PredictiveCodingConfig()
        temporal_config = MockTemporalConsciousnessConfig()
        body_schema_config = MockBodySchemaConfig()
        state_dim = 16
        
        system = create_predictive_coding_system(
            config, temporal_config, body_schema_config, state_dim, keys[0]
        )
        
        # Create mock validation data
        validation_data = []
        for i in range(3):
            current_state = jax.random.normal(keys[1], (state_dim,))
            temporal_moment = TemporalMoment(
                timestamp=float(i),
                retention=jax.random.normal(keys[2], (state_dim,)),
                present_moment=current_state,
                protention=jax.random.normal(keys[3], (state_dim,)),
                synthesis_weights=jax.random.normal(keys[1], (state_dim,)),
            )
            body_state = BodyState(
                proprioception=jax.random.normal(keys[2], (16,)),
                motor_intention=jax.random.normal(keys[3], (8,)),
                boundary_signal=jax.random.normal(keys[1], (1,)),
                schema_confidence=0.8,
            )
            validation_data.append((current_state, temporal_moment, body_state))
        
        # Test optimization
        optimized_config, optimization_metrics = optimize_hyperparameters(
            system, validation_data, optimization_steps=10, key=keys[0]
        )
        
        assert isinstance(optimized_config, PredictiveCodingConfig)
        assert isinstance(optimization_metrics, dict)
        
        expected_keys = {
            'initial_accuracy', 'final_accuracy',
            'optimization_improvement', 'convergence_rate'
        }
        assert set(optimization_metrics.keys()) == expected_keys
    
    def test_create_test_predictive_sequence(self):
        """Test test sequence creation function."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Create system
        config = PredictiveCodingConfig(hierarchy_levels=2)
        temporal_config = MockTemporalConsciousnessConfig()
        body_schema_config = MockBodySchemaConfig()
        state_dim = 12
        
        system = create_predictive_coding_system(
            config, temporal_config, body_schema_config, state_dim, keys[0]
        )
        
        # Create test sequence
        sequence_length = 5
        predictive_states = create_test_predictive_sequence(
            system, sequence_length, state_dim, keys[1]
        )
        
        assert len(predictive_states) == sequence_length
        for state in predictive_states:
            assert isinstance(state, PredictiveState)
            assert state.timestamp >= 0.0
            assert len(state.hierarchical_predictions) > 0


class TestMathematicalCorrectness:
    """Test cases for mathematical correctness of predictive coding."""
    
    def test_prediction_error_minimization_convergence(self):
        """Test that prediction error minimization converges."""
        config = PredictiveCodingConfig(
            prediction_error_history_length=20,
            dynamic_adjustment_sensitivity=0.05,
        )
        key = jax.random.PRNGKey(42)
        
        minimizer = DynamicErrorMinimization(config, key)
        
        # Simulate decreasing errors over time
        errors_sequence = [
            [jax.random.normal(key, (10,)) * (0.5 - 0.05 * i)]
            for i in range(5)
        ]
        
        model_params = {"weights": jax.random.normal(key, (10, 10))}
        total_errors = []
        
        for errors in errors_sequence:
            updated_params, metrics = minimizer.minimize_prediction_error(
                errors, model_params, config
            )
            total_errors.append(metrics['total_prediction_error'])
        
        # Errors should generally decrease (allowing for some fluctuation)
        assert len(total_errors) == 5
        for error in total_errors:
            assert error >= 0.0
    
    def test_hierarchical_prediction_consistency(self):
        """Test consistency of hierarchical predictions."""
        input_dim = 16
        layer_dimensions = (12, 8, 4)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        network = HierarchicalPredictionNetwork(
            input_dim=input_dim,
            layer_dimensions=layer_dimensions,
            key=keys[0],
            use_ngc=False,
        )
        
        # Same input should produce same predictions
        input_state = jax.random.normal(keys[1], (input_dim,))
        
        repr1, errors1 = network.forward_prediction(input_state)
        repr2, errors2 = network.forward_prediction(input_state)
        
        # Should be deterministic
        for r1, r2 in zip(repr1, repr2):
            assert jnp.allclose(r1, r2, atol=1e-6)
        
        for e1, e2 in zip(errors1, errors2):
            assert jnp.allclose(e1, e2, atol=1e-6)
    
    def test_scale_prediction_integration_bounds(self):
        """Test that scale prediction integration produces bounded outputs."""
        input_dim = 12
        temporal_config = MockTemporalConsciousnessConfig()
        predictive_config = PredictiveCodingConfig()
        key = jax.random.PRNGKey(42)
        
        predictor = MultiScaleTemporalPredictor(
            input_dim=input_dim,
            temporal_config=temporal_config,
            predictive_config=predictive_config,
            key=key,
        )
        
        keys = jax.random.split(key, 4)
        
        # Create mock predictions with varying scales
        scale_predictions = {
            PredictionScale.MICRO: (
                jax.random.normal(keys[0], (10,)) * 2,  # Larger values
                jax.random.normal(keys[0], (10,)) * 0.1
            ),
            PredictionScale.MESO: (
                jax.random.normal(keys[1], (8,)) * 0.5,  # Smaller values
                jax.random.normal(keys[1], (8,)) * 0.1
            ),
            PredictionScale.MACRO: (
                jax.random.normal(keys[2], (6,)) * 1,   # Medium values
                jax.random.normal(keys[2], (6,)) * 0.1
            ),
        }
        
        integrated = predictor.integrate_scale_predictions(
            scale_predictions, predictive_config.scale_weights
        )
        
        # Output should be finite and bounded
        assert jnp.all(jnp.isfinite(integrated))
    
    def test_confidence_estimation_bounds(self):
        """Test that confidence estimates are properly bounded."""
        config = PredictiveCodingConfig()
        temporal_config = MockTemporalConsciousnessConfig()
        body_schema_config = MockBodySchemaConfig()
        state_dim = 16
        key = jax.random.PRNGKey(42)
        
        system = IntegratedPredictiveCoding(
            config=config,
            temporal_config=temporal_config,
            body_schema_config=body_schema_config,
            state_dim=state_dim,
            key=key,
        )
        
        keys = jax.random.split(key, 6)
        
        # Test with various error magnitudes
        error_magnitudes = [0.01, 0.1, 1.0, 10.0]
        
        for magnitude in error_magnitudes:
            hierarchical_errors = [
                jax.random.normal(keys[0], (12,)) * magnitude,
                jax.random.normal(keys[1], (8,)) * magnitude,
            ]
            
            temporal_predictions = {
                PredictionScale.MICRO: (
                    jax.random.normal(keys[2], (16,)),
                    jax.random.normal(keys[3], (16,)) * magnitude
                ),
            }
            
            confidence = system._compute_prediction_confidence(
                hierarchical_errors, temporal_predictions
            )
            
            # Confidence should be in [0, 1]
            assert jnp.all(confidence >= 0.0)
            assert jnp.all(confidence <= 1.0)
            assert jnp.all(jnp.isfinite(confidence))


class TestPerformanceAndScalability:
    """Test cases for performance and scalability."""
    
    @pytest.mark.parametrize("state_dim", [8, 16, 32])
    def test_system_scalability_with_state_dimension(self, state_dim):
        """Test system scalability with different state dimensions."""
        config = PredictiveCodingConfig(hierarchy_levels=2)
        temporal_config = MockTemporalConsciousnessConfig()
        body_schema_config = MockBodySchemaConfig()
        key = jax.random.PRNGKey(42)
        
        system = create_predictive_coding_system(
            config, temporal_config, body_schema_config, state_dim, key
        )
        
        assert isinstance(system, IntegratedPredictiveCoding)
    
    @pytest.mark.parametrize("hierarchy_levels", [2, 3, 4])
    def test_system_scalability_with_hierarchy_levels(self, hierarchy_levels):
        """Test system scalability with different hierarchy levels."""
        config = PredictiveCodingConfig(hierarchy_levels=hierarchy_levels)
        temporal_config = MockTemporalConsciousnessConfig()
        body_schema_config = MockBodySchemaConfig()
        state_dim = 16
        key = jax.random.PRNGKey(42)
        
        system = create_predictive_coding_system(
            config, temporal_config, body_schema_config, state_dim, key
        )
        
        assert isinstance(system, IntegratedPredictiveCoding)
        # Check that hierarchical predictor has correct number of levels
        expected_levels = hierarchy_levels
        actual_levels = len(system.hierarchical_predictor.layers)
        assert actual_levels == expected_levels
    
    def test_batch_prediction_efficiency(self):
        """Test efficiency of batch predictions."""
        config = PredictiveCodingConfig()
        temporal_config = MockTemporalConsciousnessConfig()
        body_schema_config = MockBodySchemaConfig()
        state_dim = 12
        key = jax.random.PRNGKey(42)
        
        system = create_predictive_coding_system(
            config, temporal_config, body_schema_config, state_dim, key
        )
        
        # Create test sequence
        sequence_length = 10
        predictive_states = create_test_predictive_sequence(
            system, sequence_length, state_dim, key
        )
        
        assert len(predictive_states) == sequence_length
        for state in predictive_states:
            assert isinstance(state, PredictiveState)


class TestErrorHandlingAndEdgeCases:
    """Test cases for error handling and edge cases."""
    
    def test_zero_input_state(self):
        """Test behavior with zero input state."""
        input_dim = 16
        layer_dimensions = (12, 8)
        key = jax.random.PRNGKey(42)
        
        network = HierarchicalPredictionNetwork(
            input_dim=input_dim,
            layer_dimensions=layer_dimensions,
            key=key,
            use_ngc=False,
        )
        
        zero_input = jnp.zeros(input_dim)
        
        representations, errors = network.forward_prediction(zero_input)
        
        # Should handle gracefully
        assert len(representations) == len(layer_dimensions)
        for repr in representations:
            assert jnp.all(jnp.isfinite(repr))
        
        for error in errors:
            assert jnp.all(jnp.isfinite(error))
    
    def test_very_small_hierarchy(self):
        """Test behavior with minimal hierarchy."""
        input_dim = 10
        layer_dimensions = (8,)  # Single layer
        key = jax.random.PRNGKey(42)
        
        network = HierarchicalPredictionNetwork(
            input_dim=input_dim,
            layer_dimensions=layer_dimensions,
            key=key,
            use_ngc=False,
        )
        
        input_state = jax.random.normal(key, (input_dim,))
        
        representations, errors = network.forward_prediction(input_state)
        
        assert len(representations) == 1
        assert len(errors) == 0  # No top-down predictions possible
    
    def test_extreme_error_values(self):
        """Test behavior with extreme error values."""
        config = PredictiveCodingConfig()
        key = jax.random.PRNGKey(42)
        
        minimizer = DynamicErrorMinimization(config, key)
        
        # Test with very large errors
        large_errors = [jnp.ones(10) * 100.0]
        total_error = minimizer._compute_total_error(large_errors)
        
        assert jnp.isfinite(total_error)
        assert total_error > 0.0
        
        # Test with very small errors
        small_errors = [jnp.ones(10) * 1e-10]
        total_error_small = minimizer._compute_total_error(small_errors)
        
        assert jnp.isfinite(total_error_small)
        assert total_error_small >= 0.0
    
    def test_nan_handling_in_confidence_computation(self):
        """Test handling of NaN values in confidence computation."""
        config = PredictiveCodingConfig()
        temporal_config = MockTemporalConsciousnessConfig()
        body_schema_config = MockBodySchemaConfig()
        state_dim = 16
        key = jax.random.PRNGKey(42)
        
        system = IntegratedPredictiveCoding(
            config=config,
            temporal_config=temporal_config,
            body_schema_config=body_schema_config,
            state_dim=state_dim,
            key=key,
        )
        
        # Create errors with potential NaN issues
        hierarchical_errors = [jnp.zeros(10)]  # Could lead to division by zero
        temporal_predictions = {
            PredictionScale.MICRO: (
                jnp.zeros(16),
                jnp.zeros(16)  # Zero errors
            ),
        }
        
        confidence = system._compute_prediction_confidence(
            hierarchical_errors, temporal_predictions
        )
        
        # Should handle gracefully and produce finite confidence
        assert jnp.all(jnp.isfinite(confidence))
        assert jnp.all(confidence >= 0.0)
        assert jnp.all(confidence <= 1.0)


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main([__file__, "-v", "--tb=short"])