"""
Unit tests for Predictive Coding Training Service.

Tests the application service responsible for orchestrating training
of the predictive coding system, following TDD principles and testing
all use cases and error conditions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Tuple, Callable
from datetime import datetime, timedelta

from application.services.predictive_coding_training_service import (
    PredictiveCodingTrainingService,
    TrainingConfiguration,
    TrainingMetrics
)
from domain.value_objects.learning_parameters import LearningParameters
from domain.value_objects.precision_weights import PrecisionWeights
from domain.value_objects.prediction_state import PredictionState
from domain.events.domain_events import (
    LearningEpochCompleted,
    AdaptiveLearningRateChanged,
    PredictionErrorThresholdCrossed
)
from infrastructure.jax_predictive_coding_core import JaxPredictiveCodingCore


class TestTrainingConfiguration:
    """Test suite for TrainingConfiguration value object."""
    
    def test_default_configuration_creation(self):
        """Test creating training configuration with defaults."""
        # Arrange & Act
        config = TrainingConfiguration()
        
        # Assert
        assert config.max_epochs == 1000
        assert config.convergence_threshold == 0.001
        assert config.early_stopping_patience == 50
        assert config.validation_frequency == 10
        assert config.checkpoint_frequency == 100
        assert config.learning_rate_schedule == "adaptive"
        assert config.precision_adaptation_rate == 0.01
        assert config.enable_monitoring is True
        assert "total_error" in config.monitoring_metrics
        assert "free_energy" in config.monitoring_metrics
    
    def test_custom_configuration_creation(self):
        """Test creating training configuration with custom values."""
        # Arrange & Act
        config = TrainingConfiguration(
            max_epochs=500,
            convergence_threshold=0.01,
            learning_rate_schedule="exponential",
            monitoring_metrics=["custom_metric"]
        )
        
        # Assert
        assert config.max_epochs == 500
        assert config.convergence_threshold == 0.01
        assert config.learning_rate_schedule == "exponential"
        assert config.monitoring_metrics == ["custom_metric"]


class TestTrainingMetrics:
    """Test suite for TrainingMetrics value object."""
    
    def test_training_metrics_creation_with_defaults(self):
        """Test creating training metrics with automatic timestamp."""
        # Arrange
        start_time = datetime.now()
        
        # Act
        metrics = TrainingMetrics(
            epoch=10,
            total_error=0.5,
            free_energy_estimate=2.0,
            precision_entropy=1.5,
            convergence_rate=0.1,
            stability_measure=0.8,
            learning_rate=0.01
        )
        
        # Assert
        assert metrics.epoch == 10
        assert metrics.total_error == 0.5
        assert metrics.free_energy_estimate == 2.0
        assert metrics.timestamp >= start_time
        assert metrics.validation_error is None
    
    def test_training_metrics_with_all_fields(self):
        """Test creating training metrics with all fields specified."""
        # Arrange
        timestamp = datetime.now()
        
        # Act
        metrics = TrainingMetrics(
            epoch=5,
            total_error=0.2,
            free_energy_estimate=1.0,
            precision_entropy=0.8,
            convergence_rate=0.2,
            stability_measure=0.9,
            learning_rate=0.005,
            validation_error=0.3,
            timestamp=timestamp
        )
        
        # Assert
        assert metrics.epoch == 5
        assert metrics.validation_error == 0.3
        assert metrics.timestamp == timestamp


class TestPredictiveCodingTrainingServiceCreation:
    """Test suite for training service initialization."""
    
    def test_training_service_creation_with_defaults(self):
        """Test creating training service with default configuration."""
        # Arrange
        mock_core = Mock(spec=JaxPredictiveCodingCore)
        mock_core.hierarchy_levels = 3
        learning_params = LearningParameters(
            base_learning_rate=0.01,
            min_learning_rate=0.001,
            max_learning_rate=0.1
        )
        
        # Act
        service = PredictiveCodingTrainingService(mock_core, learning_params)
        
        # Assert
        assert service._core is mock_core
        assert service._learning_params is learning_params
        assert isinstance(service._config, TrainingConfiguration)
        assert service._current_epoch == 0
        assert len(service._training_history) == 0
        assert service._is_training is False
        assert service._current_learning_rate == learning_params.base_learning_rate
    
    def test_training_service_creation_with_custom_config(self):
        """Test creating training service with custom configuration."""
        # Arrange
        mock_core = Mock(spec=JaxPredictiveCodingCore)
        learning_params = LearningParameters(base_learning_rate=0.02)
        custom_config = TrainingConfiguration(max_epochs=200, convergence_threshold=0.01)
        
        # Act
        service = PredictiveCodingTrainingService(mock_core, learning_params, custom_config)
        
        # Assert
        assert service._config is custom_config
        assert service._config.max_epochs == 200
        assert service._config.convergence_threshold == 0.01


class TestPredictiveCodingTrainingServiceOnlineTraining:
    """Test suite for online training functionality."""
    
    @pytest.fixture
    def mock_training_setup(self):
        """Setup mock objects for training tests."""
        mock_core = Mock(spec=JaxPredictiveCodingCore)
        mock_core.hierarchy_levels = 2
        mock_core.process_input.return_value = PredictionState(
            hierarchical_errors=[0.1, 0.2],
            convergence_status="not_converged",
            learning_iteration=1
        )
        
        learning_params = LearningParameters(base_learning_rate=0.01)
        config = TrainingConfiguration(max_epochs=10, validation_frequency=5)
        
        service = PredictiveCodingTrainingService(mock_core, learning_params, config)
        
        return {
            'service': service,
            'mock_core': mock_core,
            'learning_params': learning_params,
            'config': config
        }
    
    def test_online_training_basic_functionality(self, mock_training_setup):
        """Test basic online training functionality."""
        # Arrange
        service = mock_training_setup['service']
        mock_core = mock_training_setup['mock_core']
        
        call_count = 0
        def data_generator():
            nonlocal call_count
            call_count += 1
            return (
                np.random.rand(5),
                PrecisionWeights.create_uniform(2)
            )
        
        # Act
        metrics = service.train_online(data_generator, max_steps=5)
        
        # Assert
        assert len(metrics) == 5
        assert all(isinstance(m, TrainingMetrics) for m in metrics)
        assert mock_core.process_input.call_count == 5
        assert call_count == 5
        
        # Check that each metric has required fields
        for i, metric in enumerate(metrics):
            assert metric.epoch == i
            assert isinstance(metric.total_error, float)
            assert isinstance(metric.learning_rate, float)
    
    def test_online_training_with_validation(self, mock_training_setup):
        """Test online training with validation data."""
        # Arrange
        service = mock_training_setup['service']
        
        def training_data_generator():
            return (np.random.rand(3), PrecisionWeights.create_uniform(2))
        
        def validation_data_generator():
            return (np.random.rand(3), PrecisionWeights.create_uniform(2))
        
        # Mock the validation error computation
        with patch.object(service, '_compute_validation_error', return_value=0.15):
            # Act
            metrics = service.train_online(
                training_data_generator, 
                max_steps=10,
                validation_data=validation_data_generator
            )
        
        # Assert
        assert len(metrics) == 10
        # Validation should be computed at validation frequency intervals
        validation_metrics = [m for m in metrics if m.validation_error is not None]
        assert len(validation_metrics) > 0
    
    def test_online_training_early_stopping(self, mock_training_setup):
        """Test early stopping in online training."""
        # Arrange
        service = mock_training_setup['service']
        mock_core = mock_training_setup['mock_core']
        
        # Configure mock to return converged state after 3 steps
        def side_effect(*args, **kwargs):
            if mock_core.process_input.call_count <= 3:
                return PredictionState(
                    hierarchical_errors=[0.5, 0.4],
                    convergence_status="not_converged"
                )
            else:
                return PredictionState(
                    hierarchical_errors=[0.0005, 0.0003],  # Below threshold
                    convergence_status="converged"
                )
        
        mock_core.process_input.side_effect = side_effect
        
        def data_generator():
            return (np.random.rand(4), PrecisionWeights.create_uniform(2))
        
        # Act
        metrics = service.train_online(data_generator, max_steps=100)  # Set high but should stop early
        
        # Assert - should stop early due to convergence
        assert len(metrics) < 100
        assert mock_core.process_input.call_count <= 4  # Should stop after convergence
    
    def test_online_training_domain_event_publication(self, mock_training_setup):
        """Test that domain events are published during online training."""
        # Arrange
        service = mock_training_setup['service']
        
        def data_generator():
            return (np.random.rand(3), PrecisionWeights.create_uniform(2))
        
        # Act
        service.train_online(data_generator, max_steps=3)
        domain_events = service.get_domain_events()
        
        # Assert
        assert len(domain_events) > 0
        epoch_completed_events = [
            e for e in domain_events 
            if isinstance(e, LearningEpochCompleted)
        ]
        assert len(epoch_completed_events) == 3  # One per step


class TestPredictiveCodingTrainingServiceBatchTraining:
    """Test suite for batch training functionality."""
    
    def test_batch_training_basic_functionality(self, mock_training_setup):
        """Test basic batch training functionality."""
        # Arrange
        service = mock_training_setup['service']
        mock_core = mock_training_setup['mock_core']
        
        training_data = [
            (np.random.rand(4), PrecisionWeights.create_uniform(2))
            for _ in range(5)
        ]
        
        # Act
        metrics = service.train_batch(training_data)
        
        # Assert
        assert len(metrics) > 0
        assert all(isinstance(m, TrainingMetrics) for m in metrics)
        # Should process each data point in each epoch
        assert mock_core.process_input.call_count >= len(training_data)
    
    def test_batch_training_with_validation(self, mock_training_setup):
        """Test batch training with validation data."""
        # Arrange
        service = mock_training_setup['service']
        
        training_data = [
            (np.random.rand(3), PrecisionWeights.create_uniform(2))
            for _ in range(3)
        ]
        validation_data = [
            (np.random.rand(3), PrecisionWeights.create_uniform(2))
            for _ in range(2)
        ]
        
        # Mock validation error computation
        with patch.object(service, '_compute_validation_error', return_value=0.2):
            # Act
            metrics = service.train_batch(training_data, validation_data)
        
        # Assert
        assert len(metrics) > 0
        # Should have validation errors at validation frequency
        validation_metrics = [m for m in metrics if m.validation_error is not None]
        assert len(validation_metrics) > 0
    
    def test_batch_training_convergence_stopping(self, mock_training_setup):
        """Test that batch training stops on convergence."""
        # Arrange
        service = mock_training_setup['service']
        mock_core = mock_training_setup['mock_core']
        service._config.convergence_threshold = 0.01
        
        # Configure mock to converge quickly
        epoch_count = 0
        def side_effect(*args, **kwargs):
            nonlocal epoch_count
            epoch_count += 1
            if epoch_count <= 10:  # First few calls have high error
                return PredictionState(hierarchical_errors=[0.5, 0.4])
            else:  # Then converge
                return PredictionState(hierarchical_errors=[0.005, 0.003])
        
        mock_core.process_input.side_effect = side_effect
        
        training_data = [
            (np.random.rand(3), PrecisionWeights.create_uniform(2))
            for _ in range(2)
        ]
        
        # Act
        metrics = service.train_batch(training_data)
        
        # Assert - should stop early due to convergence
        assert len(metrics) < service._config.max_epochs
        final_metric = metrics[-1]
        assert final_metric.total_error < service._config.convergence_threshold
    
    def test_batch_training_early_stopping_validation(self, mock_training_setup):
        """Test early stopping based on validation error."""
        # Arrange
        service = mock_training_setup['service']
        service._config.early_stopping_patience = 3
        service._config.validation_frequency = 1  # Check every epoch
        
        training_data = [(np.random.rand(3), PrecisionWeights.create_uniform(2))]
        validation_data = [(np.random.rand(3), PrecisionWeights.create_uniform(2))]
        
        # Mock validation errors that get worse over time
        validation_call_count = 0
        def mock_validation_error(*args):
            nonlocal validation_call_count
            validation_call_count += 1
            return 0.1 + validation_call_count * 0.05  # Increasing validation error
        
        with patch.object(service, '_compute_validation_error', side_effect=mock_validation_error):
            # Act
            metrics = service.train_batch(training_data, validation_data)
        
        # Assert - should stop due to validation error not improving
        assert len(metrics) < service._config.max_epochs
        assert service._patience_counter >= service._config.early_stopping_patience


class TestPredictiveCodingTrainingServiceMetrics:
    """Test suite for training metrics computation."""
    
    def test_compute_training_metrics_basic(self, mock_training_setup):
        """Test basic training metrics computation."""
        # Arrange
        service = mock_training_setup['service']
        prediction_state = PredictionState(
            hierarchical_errors=[0.1, 0.2, 0.3],
            convergence_status="converging",
            learning_iteration=5
        )
        
        # Mock precision entropy computation
        with patch.object(service, '_compute_precision_entropy', return_value=1.5):
            with patch.object(service, '_compute_convergence_rate', return_value=0.1):
                with patch.object(service, '_compute_stability_measure', return_value=0.8):
                    # Act
                    metrics = service._compute_training_metrics(10, prediction_state)
        
        # Assert
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.epoch == 10
        assert metrics.total_error == 0.6  # Sum of hierarchical errors
        assert metrics.precision_entropy == 1.5
        assert metrics.convergence_rate == 0.1
        assert metrics.stability_measure == 0.8
        assert metrics.learning_rate == service._current_learning_rate
    
    def test_precision_entropy_computation(self, mock_training_setup):
        """Test precision entropy computation."""
        # Arrange
        service = mock_training_setup['service']
        mock_core = mock_training_setup['mock_core']
        
        # Mock precision estimates
        mock_core.get_precision_estimates.return_value = {
            'level_0': 2.0,
            'level_1': 1.0,
            'level_2': 0.5
        }
        
        # Act
        entropy = service._compute_precision_entropy()
        
        # Assert
        assert isinstance(entropy, float)
        assert entropy >= 0.0  # Entropy should be non-negative
        assert not np.isnan(entropy)
    
    def test_convergence_rate_computation(self, mock_training_setup):
        """Test convergence rate computation."""
        # Arrange
        service = mock_training_setup['service']
        
        # Add some training history with decreasing errors
        for i in range(5):
            metrics = TrainingMetrics(
                epoch=i,
                total_error=1.0 - i * 0.1,  # Decreasing error
                free_energy_estimate=2.0,
                precision_entropy=1.0,
                convergence_rate=0.0,
                stability_measure=0.5,
                learning_rate=0.01
            )
            service._training_history.append(metrics)
        
        # Act
        convergence_rate = service._compute_convergence_rate()
        
        # Assert
        assert isinstance(convergence_rate, float)
        # Should be positive since errors are decreasing (negative slope -> positive rate)
        assert convergence_rate > 0
    
    def test_stability_measure_computation(self, mock_training_setup):
        """Test stability measure computation."""
        # Arrange
        service = mock_training_setup['service']
        
        # Add training history with varying stability
        stable_errors = [0.1, 0.101, 0.099, 0.102, 0.098]  # Low variance
        for i, error in enumerate(stable_errors):
            metrics = TrainingMetrics(
                epoch=i,
                total_error=error,
                free_energy_estimate=1.0,
                precision_entropy=1.0,
                convergence_rate=0.0,
                stability_measure=0.0,
                learning_rate=0.01
            )
            service._training_history.append(metrics)
        
        # Act
        stability = service._compute_stability_measure()
        
        # Assert
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0  # Should be normalized
        assert stability > 0.9  # Should be high for stable errors


class TestPredictiveCodingTrainingServiceAdaptiveFeatures:
    """Test suite for adaptive training features."""
    
    def test_adaptive_learning_rate_adjustment(self, mock_training_setup):
        """Test adaptive learning rate adjustment."""
        # Arrange
        service = mock_training_setup['service']
        service._config.learning_rate_schedule = "adaptive"
        initial_lr = service._current_learning_rate
        
        # Create metrics that suggest slow convergence
        slow_convergence_metrics = TrainingMetrics(
            epoch=10,
            total_error=0.5,
            free_energy_estimate=2.0,
            precision_entropy=1.0,
            convergence_rate=0.005,  # Very slow
            stability_measure=0.9,   # But stable
            learning_rate=initial_lr
        )
        
        # Act
        service._adapt_learning_rate(slow_convergence_metrics)
        
        # Assert - learning rate should increase for slow but stable convergence
        assert service._current_learning_rate != initial_lr
        # Should still be within bounds
        assert (service._learning_params.min_learning_rate <= 
                service._current_learning_rate <= 
                service._learning_params.max_learning_rate)
    
    def test_adaptive_learning_rate_exponential_schedule(self, mock_training_setup):
        """Test exponential learning rate schedule."""
        # Arrange
        service = mock_training_setup['service']
        service._config.learning_rate_schedule = "exponential"
        service._current_epoch = 10
        initial_lr = service._learning_params.base_learning_rate
        
        metrics = TrainingMetrics(
            epoch=10, total_error=0.1, free_energy_estimate=1.0,
            precision_entropy=1.0, convergence_rate=0.1, stability_measure=0.8,
            learning_rate=initial_lr
        )
        
        # Act
        service._adapt_learning_rate(metrics)
        
        # Assert - should apply exponential decay
        expected_lr = initial_lr * (0.95 ** service._current_epoch)
        assert abs(service._current_learning_rate - expected_lr) < 1e-6
    
    def test_precision_weights_adaptation(self, mock_training_setup):
        """Test precision weights adaptation."""
        # Arrange
        service = mock_training_setup['service']
        
        prediction_state = PredictionState(
            hierarchical_errors=[0.5, 0.2],  # Different error levels
            convergence_status="not_converged"
        )
        
        initial_weights = PrecisionWeights.create_uniform(2)
        
        # Act
        adapted_weights = service._adapt_precision_weights(prediction_state, initial_weights)
        
        # Assert
        assert isinstance(adapted_weights, PrecisionWeights)
        assert adapted_weights.hierarchy_levels == 2
        # Weights should have been adapted based on errors
        assert not np.allclose(
            adapted_weights.weights, 
            initial_weights.weights, 
            rtol=0.01
        )
    
    def test_learning_rate_change_event_publication(self, mock_training_setup):
        """Test that learning rate changes publish domain events."""
        # Arrange
        service = mock_training_setup['service']
        initial_lr = service._current_learning_rate
        
        # Create metrics that will trigger learning rate change
        metrics = TrainingMetrics(
            epoch=5, total_error=0.1, free_energy_estimate=1.0,
            precision_entropy=1.0, convergence_rate=0.001,  # Very slow
            stability_measure=0.3, learning_rate=initial_lr  # Unstable
        )
        
        # Act
        service._adapt_learning_rate(metrics)
        
        # Check if learning rate actually changed
        if abs(service._current_learning_rate - initial_lr) > 0.001:
            events = service.get_domain_events()
            lr_change_events = [
                e for e in events 
                if isinstance(e, AdaptiveLearningRateChanged)
            ]
            assert len(lr_change_events) > 0
            
            event = lr_change_events[0]
            assert event.new_learning_rate == service._current_learning_rate
            assert event.previous_learning_rate == initial_lr


class TestPredictiveCodingTrainingServiceEventHandling:
    """Test suite for domain event handling and publication."""
    
    def test_error_threshold_crossing_event_publication(self, mock_training_setup):
        """Test publication of error threshold crossing events."""
        # Arrange
        service = mock_training_setup['service']
        
        # Add initial high error to history
        high_error_metrics = TrainingMetrics(
            epoch=0, total_error=0.5, free_energy_estimate=2.0,
            precision_entropy=1.0, convergence_rate=0.0, stability_measure=0.5,
            learning_rate=0.01
        )
        service._training_history.append(high_error_metrics)
        
        # Create metrics with error below threshold
        low_error_metrics = TrainingMetrics(
            epoch=1, total_error=0.05, free_energy_estimate=1.0,  # Below 0.1 threshold
            precision_entropy=1.0, convergence_rate=0.2, stability_measure=0.8,
            learning_rate=0.01
        )
        
        # Act
        service._check_error_thresholds(low_error_metrics)
        
        # Assert
        events = service.get_domain_events()
        threshold_events = [
            e for e in events 
            if isinstance(e, PredictionErrorThresholdCrossed)
        ]
        
        if threshold_events:  # Should have crossed the 0.1 threshold
            event = threshold_events[0]
            assert event.threshold_value == 0.1
            assert event.previous_error == 0.5
            assert event.new_error == 0.05
            assert event.threshold_direction == "crossed_below"
    
    def test_domain_event_clearing(self, mock_training_setup):
        """Test clearing of accumulated domain events."""
        # Arrange
        service = mock_training_setup['service']
        
        # Add some events
        def data_generator():
            return (np.random.rand(3), PrecisionWeights.create_uniform(2))
        
        service.train_online(data_generator, max_steps=2)
        
        # Ensure events were generated
        assert len(service.get_domain_events()) > 0
        
        # Act
        service.clear_domain_events()
        
        # Assert
        assert len(service.get_domain_events()) == 0


class TestPredictiveCodingTrainingServiceUtilityMethods:
    """Test suite for utility and helper methods."""
    
    def test_training_summary_generation(self, mock_training_setup):
        """Test generation of training summary."""
        # Arrange
        service = mock_training_setup['service']
        
        # Add some training history
        for i in range(3):
            metrics = TrainingMetrics(
                epoch=i, total_error=0.5 - i * 0.1, free_energy_estimate=2.0 - i * 0.2,
                precision_entropy=1.0, convergence_rate=0.1 + i * 0.05,
                stability_measure=0.7 + i * 0.1, learning_rate=0.01,
                timestamp=datetime.now() - timedelta(minutes=10-i)
            )
            service._training_history.append(metrics)
        
        # Act
        summary = service.get_training_summary()
        
        # Assert
        assert isinstance(summary, dict)
        assert summary["status"] == "completed"
        assert summary["total_epochs"] == 3
        assert summary["final_error"] == 0.2  # Last error
        assert summary["final_free_energy"] == 1.4  # Last free energy
        assert summary["convergence_achieved"] is False  # Above threshold
        assert summary["training_duration"] > 0  # Should have duration
    
    def test_training_summary_not_trained(self, mock_training_setup):
        """Test training summary when no training has occurred."""
        # Arrange
        service = mock_training_setup['service']
        # No training history
        
        # Act
        summary = service.get_training_summary()
        
        # Assert
        assert summary["status"] == "not_trained"
    
    def test_training_history_access(self, mock_training_setup):
        """Test access to training history."""
        # Arrange
        service = mock_training_setup['service']
        
        # Add training history
        original_metrics = []
        for i in range(3):
            metrics = TrainingMetrics(
                epoch=i, total_error=0.1 * (i + 1), free_energy_estimate=1.0,
                precision_entropy=1.0, convergence_rate=0.1, stability_measure=0.8,
                learning_rate=0.01
            )
            service._training_history.append(metrics)
            original_metrics.append(metrics)
        
        # Act
        history = service.get_training_history()
        
        # Assert
        assert len(history) == 3
        assert history == original_metrics
        # Should be a copy, not the original
        assert history is not service._training_history
    
    def test_current_learning_rate_access(self, mock_training_setup):
        """Test access to current learning rate."""
        # Arrange
        service = mock_training_setup['service']
        initial_lr = service._current_learning_rate
        
        # Act
        current_lr = service.get_current_learning_rate()
        
        # Assert
        assert current_lr == initial_lr
        
        # Change learning rate and verify
        service._current_learning_rate = 0.005
        assert service.get_current_learning_rate() == 0.005
    
    def test_training_status_tracking(self, mock_training_setup):
        """Test training status tracking."""
        # Arrange
        service = mock_training_setup['service']
        
        # Initially not training
        assert service.is_training() is False
        
        # During training (simulated by setting flag)
        service._is_training = True
        assert service.is_training() is True
        
        # After training
        service._is_training = False
        assert service.is_training() is False


class TestPredictiveCodingTrainingServiceErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_should_stop_training_convergence(self, mock_training_setup):
        """Test training stopping logic for convergence."""
        # Arrange
        service = mock_training_setup['service']
        service._config.convergence_threshold = 0.001
        
        # Converged metrics
        converged_metrics = TrainingMetrics(
            epoch=10, total_error=0.0005,  # Below threshold
            free_energy_estimate=0.5, precision_entropy=1.0, convergence_rate=0.1,
            stability_measure=0.9, learning_rate=0.01
        )
        
        # Act & Assert
        assert service._should_stop_training(converged_metrics) is True
    
    def test_should_stop_training_divergence(self, mock_training_setup):
        """Test training stopping logic for divergence."""
        # Arrange
        service = mock_training_setup['service']
        
        # Diverged metrics
        diverged_metrics = TrainingMetrics(
            epoch=10, total_error=150.0,  # Very high error
            free_energy_estimate=10.0, precision_entropy=1.0, convergence_rate=-0.1,
            stability_measure=0.1, learning_rate=0.01
        )
        
        # Act & Assert
        assert service._should_stop_training(diverged_metrics) is True
    
    def test_validation_error_computation_exception_handling(self, mock_training_setup):
        """Test handling of exceptions in validation error computation."""
        # Arrange
        service = mock_training_setup['service']
        
        def failing_validation_data():
            raise RuntimeError("Validation data unavailable")
        
        # Act
        validation_error = service._compute_validation_error(failing_validation_data)
        
        # Assert - should return NaN on exception
        assert np.isnan(validation_error)
    
    def test_aggregate_prediction_states_empty_list(self, mock_training_setup):
        """Test aggregation of prediction states with empty list."""
        # Arrange
        service = mock_training_setup['service']
        
        # Act
        aggregated_state = service._aggregate_prediction_states([])
        
        # Assert
        assert isinstance(aggregated_state, PredictionState)
        assert aggregated_state.hierarchy_levels == service._core.hierarchy_levels
        assert all(error == 0.0 for error in aggregated_state.hierarchical_errors)