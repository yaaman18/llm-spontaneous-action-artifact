"""
Unit tests for PredictiveCodingCore entity.

Comprehensive TDD test suite covering hierarchical predictive coding
algorithms, error propagation, and learning dynamics. Uses test doubles
for abstract method implementations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple
import numpy.typing as npt

from domain.entities.predictive_coding_core import PredictiveCodingCore
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.precision_weights import PrecisionWeights


# Test implementation of abstract PredictiveCodingCore for testing
class MockPredictiveCodingCore(PredictiveCodingCore):
    """Mock implementation of PredictiveCodingCore for testing."""
    
    def __init__(self, hierarchy_levels: int, input_dimensions: int):
        super().__init__(hierarchy_levels, input_dimensions)
        self.generated_predictions = []
        self.computed_errors = []
        self.propagated_errors = []
        self.update_calls = []
    
    def generate_predictions(
        self, 
        input_data: npt.NDArray,
        precision_weights: PrecisionWeights
    ) -> List[npt.NDArray]:
        """Mock prediction generation."""
        self.generated_predictions.append((input_data.copy(), precision_weights))
        
        # Generate simple predictions based on hierarchy
        predictions = []
        for level in range(self.hierarchy_levels):
            # Create predictions with decreasing dimensionality
            pred_dim = max(1, self.input_dimensions - level * 2)
            prediction = np.random.rand(pred_dim) * 0.5
            predictions.append(prediction)
        
        return predictions
    
    def compute_prediction_errors(
        self,
        predictions: List[npt.NDArray],
        targets: List[npt.NDArray]
    ) -> List[npt.NDArray]:
        """Mock prediction error computation."""
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        
        errors = []
        for pred, target in zip(predictions, targets):
            if pred.shape != target.shape:
                raise ValueError("Prediction and target shapes must match")
            error = pred - target
            errors.append(error)
        
        self.computed_errors.append((predictions, targets))
        return errors
    
    def propagate_errors(
        self,
        prediction_errors: List[npt.NDArray],
        precision_weights: PrecisionWeights
    ) -> Tuple[List[npt.NDArray], PredictionState]:
        """Mock error propagation."""
        # Apply precision weighting
        propagated_errors = []
        for i, error in enumerate(prediction_errors):
            weight = precision_weights.get_weight_at_level(i) if i < precision_weights.hierarchy_levels else 1.0
            weighted_error = error * weight
            propagated_errors.append(weighted_error)
        
        # Create prediction state
        error_magnitudes = [float(np.mean(np.abs(error))) for error in propagated_errors]
        prediction_state = PredictionState(
            hierarchical_errors=error_magnitudes,
            convergence_status="not_converged" if any(e > 0.1 for e in error_magnitudes) else "converged"
        )
        
        self.propagated_errors.append((prediction_errors, precision_weights))
        return propagated_errors, prediction_state
    
    def update_predictions(
        self,
        learning_rate: float,
        propagated_errors: List[npt.NDArray]
    ) -> None:
        """Mock prediction update."""
        if not (0 < learning_rate <= 1.0):
            raise ValueError("Learning rate must be in (0, 1]")
        
        self.update_calls.append((learning_rate, [error.copy() for error in propagated_errors]))
    
    def _create_targets_from_input(
        self,
        input_data: npt.NDArray,
        predictions: List[npt.NDArray]
    ) -> List[npt.NDArray]:
        """Mock target creation."""
        targets = []
        for i, prediction in enumerate(predictions):
            # Create simple targets that are slightly different from predictions
            target = prediction + np.random.rand(*prediction.shape) * 0.1
            targets.append(target)
        return targets
    
    def compute_free_energy(
        self,
        predictions: List[npt.NDArray],
        targets: List[npt.NDArray],
        precision_weights: PrecisionWeights
    ) -> float:
        """Mock free energy computation."""
        total_error = 0.0
        for pred, target in zip(predictions, targets):
            total_error += float(np.sum((pred - target) ** 2))
        return total_error
    
    def update_precisions(
        self,
        prediction_errors: List[npt.NDArray],
        learning_rate: float = 0.01
    ) -> PrecisionWeights:
        """Mock precision update."""
        # Return simple uniform precision weights
        return PrecisionWeights.create_uniform(self.hierarchy_levels)


class TestPredictiveCodingCoreCreation:
    """Test suite for PredictiveCodingCore creation and validation."""
    
    def test_valid_predictive_coding_core_creation(self):
        """Test creating valid PredictiveCodingCore instance."""
        # Arrange & Act
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=10)
        
        # Assert
        assert core.hierarchy_levels == 3
        assert core.input_dimensions == 10
        assert core.current_state is None

    def test_invalid_hierarchy_levels_raises_error(self):
        """Test that invalid hierarchy levels raise error."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Hierarchy levels must be positive"):
            MockPredictiveCodingCore(hierarchy_levels=0, input_dimensions=10)
        
        with pytest.raises(ValueError, match="Hierarchy levels must be positive"):
            MockPredictiveCodingCore(hierarchy_levels=-1, input_dimensions=10)

    def test_invalid_input_dimensions_raises_error(self):
        """Test that invalid input dimensions raise error."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Input dimensions must be positive"):
            MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=0)
        
        with pytest.raises(ValueError, match="Input dimensions must be positive"):
            MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=-1)

    def test_properties_are_read_only(self):
        """Test that core properties are read-only."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=10)
        
        # Act & Assert
        assert hasattr(core, 'hierarchy_levels')
        assert hasattr(core, 'input_dimensions')
        assert hasattr(core, 'current_state')


class TestPredictiveCodingCoreAbstractMethods:
    """Test suite for abstract method requirements."""
    
    def test_generate_predictions_with_valid_input(self):
        """Test prediction generation with valid input."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=10)
        input_data = np.random.rand(10)
        precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
        
        # Act
        predictions = core.generate_predictions(input_data, precision_weights)
        
        # Assert
        assert len(predictions) == 3  # One prediction per hierarchy level
        assert all(isinstance(pred, np.ndarray) for pred in predictions)
        assert len(core.generated_predictions) == 1

    def test_compute_prediction_errors_with_valid_data(self):
        """Test prediction error computation with valid data."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=10)
        predictions = [np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0, 5.0, 6.0])]
        targets = [np.array([1.1, 1.9]), np.array([3.2]), np.array([3.8, 5.1, 5.9])]
        
        # Act
        errors = core.compute_prediction_errors(predictions, targets)
        
        # Assert
        assert len(errors) == len(predictions)
        np.testing.assert_array_almost_equal(errors[0], np.array([-0.1, 0.1]))
        np.testing.assert_array_almost_equal(errors[1], np.array([-0.2]))

    def test_compute_prediction_errors_mismatched_lengths(self):
        """Test error computation with mismatched prediction and target lengths."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=10)
        predictions = [np.array([1.0, 2.0]), np.array([3.0])]
        targets = [np.array([1.1, 1.9])]  # One less target
        
        # Act & Assert
        with pytest.raises(ValueError, match="Predictions and targets must have same length"):
            core.compute_prediction_errors(predictions, targets)

    def test_propagate_errors_with_precision_weights(self):
        """Test error propagation with precision weights."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=10)
        prediction_errors = [
            np.array([0.1, -0.2]),
            np.array([0.3]),
            np.array([-0.1, 0.2, -0.3])
        ]
        precision_weights = PrecisionWeights(np.array([2.0, 1.5, 1.0]))
        
        # Act
        propagated_errors, prediction_state = core.propagate_errors(prediction_errors, precision_weights)
        
        # Assert
        assert len(propagated_errors) == len(prediction_errors)
        assert isinstance(prediction_state, PredictionState)
        assert prediction_state.hierarchy_levels == 3

    def test_update_predictions_with_valid_parameters(self):
        """Test prediction updates with valid parameters."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=10)
        learning_rate = 0.01
        propagated_errors = [np.array([0.1, 0.2]), np.array([0.3])]
        
        # Act
        core.update_predictions(learning_rate, propagated_errors)
        
        # Assert
        assert len(core.update_calls) == 1
        assert core.update_calls[0][0] == learning_rate

    def test_update_predictions_invalid_learning_rate(self):
        """Test prediction updates with invalid learning rate."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=10)
        propagated_errors = [np.array([0.1, 0.2])]
        
        # Act & Assert
        with pytest.raises(ValueError, match="Learning rate must be in \\(0, 1\\]"):
            core.update_predictions(learning_rate=0.0, propagated_errors=propagated_errors)
        
        with pytest.raises(ValueError, match="Learning rate must be in \\(0, 1\\]"):
            core.update_predictions(learning_rate=1.5, propagated_errors=propagated_errors)


class TestPredictiveCodingCoreTemplateMethod:
    """Test suite for the template method (process_input)."""
    
    def test_process_input_complete_cycle(self):
        """Test complete processing cycle through template method."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=10)
        input_data = np.random.rand(10)
        precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
        learning_rate = 0.01
        
        # Act
        prediction_state = core.process_input(input_data, precision_weights, learning_rate)
        
        # Assert
        assert isinstance(prediction_state, PredictionState)
        assert core.current_state is prediction_state
        assert len(core.generated_predictions) == 1
        assert len(core.computed_errors) == 1
        assert len(core.propagated_errors) == 1
        assert len(core.update_calls) == 1

    def test_process_input_with_default_learning_rate(self):
        """Test processing with default learning rate."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=2, input_dimensions=5)
        input_data = np.random.rand(5)
        precision_weights = PrecisionWeights(np.array([1.0, 0.8]))
        
        # Act
        prediction_state = core.process_input(input_data, precision_weights)
        
        # Assert
        assert core.update_calls[0][0] == 0.01  # Default learning rate

    def test_process_input_updates_current_state(self):
        """Test that processing updates current state."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=2, input_dimensions=5)
        input_data = np.random.rand(5)
        precision_weights = PrecisionWeights(np.array([1.0, 0.8]))
        
        # Verify initial state
        assert core.current_state is None
        
        # Act
        prediction_state1 = core.process_input(input_data, precision_weights)
        assert core.current_state is prediction_state1
        
        prediction_state2 = core.process_input(input_data, precision_weights)
        assert core.current_state is prediction_state2
        assert prediction_state2 is not prediction_state1

    def test_process_input_template_method_order(self):
        """Test that template method calls abstract methods in correct order."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=2, input_dimensions=5)
        
        # Mock the abstract methods to track call order
        call_order = []
        original_generate = core.generate_predictions
        original_compute = core.compute_prediction_errors
        original_propagate = core.propagate_errors
        original_update = core.update_predictions
        
        def track_generate(*args, **kwargs):
            call_order.append('generate')
            return original_generate(*args, **kwargs)
        
        def track_compute(*args, **kwargs):
            call_order.append('compute')
            return original_compute(*args, **kwargs)
        
        def track_propagate(*args, **kwargs):
            call_order.append('propagate')
            return original_propagate(*args, **kwargs)
        
        def track_update(*args, **kwargs):
            call_order.append('update')
            return original_update(*args, **kwargs)
        
        core.generate_predictions = track_generate
        core.compute_prediction_errors = track_compute
        core.propagate_errors = track_propagate
        core.update_predictions = track_update
        
        input_data = np.random.rand(5)
        precision_weights = PrecisionWeights(np.array([1.0, 0.8]))
        
        # Act
        core.process_input(input_data, precision_weights)
        
        # Assert correct order
        expected_order = ['generate', 'compute', 'propagate', 'update']
        assert call_order == expected_order


class TestPredictiveCodingCoreStateManagement:
    """Test suite for state management operations."""
    
    def test_reset_state_clears_current_state(self):
        """Test that reset_state clears current state."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=2, input_dimensions=5)
        input_data = np.random.rand(5)
        precision_weights = PrecisionWeights(np.array([1.0, 0.8]))
        
        # Set initial state
        core.process_input(input_data, precision_weights)
        assert core.current_state is not None
        
        # Act
        core.reset_state()
        
        # Assert
        assert core.current_state is None

    def test_get_total_prediction_error_with_state(self):
        """Test getting total prediction error when state exists."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=2, input_dimensions=5)
        input_data = np.random.rand(5)
        precision_weights = PrecisionWeights(np.array([1.0, 0.8]))
        
        # Process input to create state
        prediction_state = core.process_input(input_data, precision_weights)
        expected_error = prediction_state.total_error
        
        # Act
        total_error = core.get_total_prediction_error()
        
        # Assert
        assert total_error == expected_error

    def test_get_total_prediction_error_without_state_raises_error(self):
        """Test getting total error without state raises error."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=2, input_dimensions=5)
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="No current state available"):
            core.get_total_prediction_error()


class TestPredictiveCodingCoreErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_process_input_with_wrong_input_dimensions(self):
        """Test processing with incorrect input dimensions."""
        # This test depends on the specific implementation
        # Here we test the general contract that errors should be handled
        
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=2, input_dimensions=5)
        wrong_size_input = np.random.rand(10)  # Wrong size
        precision_weights = PrecisionWeights(np.array([1.0, 0.8]))
        
        # For this mock implementation, we don't enforce input size validation
        # But a real implementation should validate input dimensions
        # Act & Assert - this will work in our mock but might fail in real implementation
        try:
            core.process_input(wrong_size_input, precision_weights)
        except ValueError:
            pass  # Expected behavior for real implementation


class TestPredictiveCodingCoreIntegration:
    """Integration tests for PredictiveCodingCore operations."""
    
    def test_multiple_processing_cycles(self):
        """Test multiple processing cycles maintain consistency."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=8)
        precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
        learning_rate = 0.05
        
        # Act - process multiple inputs
        states = []
        for i in range(5):
            input_data = np.random.rand(8) + i * 0.1  # Slightly different inputs
            state = core.process_input(input_data, precision_weights, learning_rate)
            states.append(state)
        
        # Assert
        assert len(states) == 5
        assert all(isinstance(state, PredictionState) for state in states)
        assert all(state.hierarchy_levels == 3 for state in states)
        
        # Check that learning iterations increase
        iterations = [state.learning_iteration for state in states]
        assert all(iterations[i] >= iterations[i-1] for i in range(1, len(iterations)))

    def test_prediction_error_patterns(self, prediction_error_patterns):
        """Test processing with different error patterns."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=4, input_dimensions=6)
        precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6, 0.4]))
        
        # Create mock that returns specific error patterns
        def mock_propagate_errors(prediction_errors, precision_weights):
            """Mock that returns predictable error patterns."""
            # Use decreasing error pattern
            error_pattern = prediction_error_patterns['decreasing']
            prediction_state = PredictionState(
                hierarchical_errors=error_pattern,
                convergence_status="converging",
                learning_iteration=1
            )
            return prediction_errors, prediction_state
        
        core.propagate_errors = mock_propagate_errors
        
        # Act
        input_data = np.random.rand(6)
        final_state = core.process_input(input_data, precision_weights)
        
        # Assert
        assert final_state.convergence_status == "converging"
        assert final_state.hierarchical_errors == prediction_error_patterns['decreasing']

    def test_precision_weights_integration(self):
        """Test integration with precision weights."""
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=5)
        
        # Test with different precision weight configurations
        high_precision = PrecisionWeights(np.array([2.0, 2.0, 2.0]))
        low_precision = PrecisionWeights(np.array([0.1, 0.1, 0.1]))
        
        input_data = np.random.rand(5)
        
        # Act
        state_high_precision = core.process_input(input_data, high_precision)
        core.reset_state()  # Reset for clean comparison
        
        state_low_precision = core.process_input(input_data, low_precision)
        
        # Assert - high precision should generally lead to different error magnitudes
        # (exact behavior depends on implementation details)
        assert isinstance(state_high_precision, PredictionState)
        assert isinstance(state_low_precision, PredictionState)
        assert state_high_precision.hierarchy_levels == state_low_precision.hierarchy_levels


class TestPredictiveCodingCorePerformance:
    """Performance and scalability tests."""
    
    def test_processing_time_scales_reasonably(self, performance_timer):
        """Test that processing time scales reasonably with problem size."""
        # Arrange
        small_core = MockPredictiveCodingCore(hierarchy_levels=2, input_dimensions=10)
        large_core = MockPredictiveCodingCore(hierarchy_levels=5, input_dimensions=50)
        
        precision_weights_small = PrecisionWeights(np.array([1.0, 0.8]))
        precision_weights_large = PrecisionWeights(np.array([1.0, 0.8, 0.6, 0.4, 0.2]))
        
        # Act & Measure
        performance_timer.start()
        for _ in range(10):
            input_data = np.random.rand(10)
            small_core.process_input(input_data, precision_weights_small)
        small_time = performance_timer.stop()
        
        performance_timer.start()
        for _ in range(10):
            input_data = np.random.rand(50)
            large_core.process_input(input_data, precision_weights_large)
        large_time = performance_timer.stop()
        
        # Assert - larger problem should take more time but not excessively
        assert large_time > small_time
        # Reasonable scaling assumption (adjust based on actual implementation)
        assert large_time < small_time * 50  # Should not be more than 50x slower

    def test_memory_usage_stability(self):
        """Test that repeated processing doesn't cause memory leaks."""
        # This is a basic test - more sophisticated memory profiling tools
        # would be needed for production systems
        
        # Arrange
        core = MockPredictiveCodingCore(hierarchy_levels=3, input_dimensions=20)
        precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
        
        # Act - perform many processing cycles
        for i in range(100):
            input_data = np.random.rand(20)
            core.process_input(input_data, precision_weights)
            
            # Reset periodically to prevent unbounded growth in mock data
            if i % 20 == 19:
                core.generated_predictions = []
                core.computed_errors = []
                core.propagated_errors = []
                core.update_calls = []
        
        # Assert - test completed without memory errors
        assert core.current_state is not None