"""
Unit tests for PredictionState value object.

Comprehensive TDD test suite covering prediction state management,
hierarchical error tracking, and convergence analysis. Tests include
property-based testing for prediction error patterns.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from hypothesis import given, assume, strategies as st

from domain.value_objects.prediction_state import PredictionState


class TestPredictionStateCreation:
    """Test suite for PredictionState creation and validation."""
    
    def test_minimal_prediction_state_creation(self):
        """Test creating PredictionState with minimal required parameters."""
        # Arrange
        hierarchical_errors = [0.1, 0.2, 0.3]
        
        # Act
        state = PredictionState(hierarchical_errors=hierarchical_errors)
        
        # Assert
        assert state.hierarchical_errors == hierarchical_errors
        assert state.hierarchy_levels == 3
        assert state.convergence_status == "not_converged"
        assert state.learning_iteration == 0
        assert isinstance(state.timestamp, datetime)

    def test_full_prediction_state_creation(self):
        """Test creating PredictionState with all parameters."""
        # Arrange
        hierarchical_errors = [0.1, 0.2, 0.3]
        hierarchical_predictions = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        precision_weighted_errors = [0.05, 0.1, 0.15]
        timestamp = datetime.now()
        convergence_status = "converged"
        learning_iteration = 100
        metadata = {"test": "data"}
        
        # Act
        state = PredictionState(
            hierarchical_errors=hierarchical_errors,
            hierarchical_predictions=hierarchical_predictions,
            precision_weighted_errors=precision_weighted_errors,
            timestamp=timestamp,
            convergence_status=convergence_status,
            learning_iteration=learning_iteration,
            metadata=metadata
        )
        
        # Assert
        assert state.hierarchical_errors == hierarchical_errors
        assert len(state.hierarchical_predictions) == 3
        assert state.precision_weighted_errors == precision_weighted_errors
        assert state.timestamp == timestamp
        assert state.convergence_status == convergence_status
        assert state.learning_iteration == learning_iteration
        assert state.metadata == metadata

    def test_empty_hierarchical_errors_raises_error(self):
        """Test that empty hierarchical errors list raises error."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Hierarchical errors cannot be empty"):
            PredictionState(hierarchical_errors=[])

    def test_mismatched_predictions_and_errors_raises_error(self):
        """Test that mismatched predictions and errors raises error."""
        # Arrange
        hierarchical_errors = [0.1, 0.2, 0.3]
        hierarchical_predictions = [np.array([1, 2]), np.array([3, 4])]  # Only 2 predictions
        
        # Act & Assert
        with pytest.raises(ValueError, match="Predictions and errors must have same hierarchy levels"):
            PredictionState(
                hierarchical_errors=hierarchical_errors,
                hierarchical_predictions=hierarchical_predictions
            )

    def test_mismatched_precision_weights_and_errors_raises_error(self):
        """Test that mismatched precision weights and errors raises error."""
        # Arrange
        hierarchical_errors = [0.1, 0.2, 0.3]
        precision_weighted_errors = [0.1, 0.2]  # Only 2 weights
        
        # Act & Assert
        with pytest.raises(ValueError, match="Precision weighted errors must match hierarchy levels"):
            PredictionState(
                hierarchical_errors=hierarchical_errors,
                precision_weighted_errors=precision_weighted_errors
            )

    def test_invalid_error_values_raise_error(self):
        """Test that NaN error values raise error."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="All hierarchical errors must be valid numbers"):
            PredictionState(hierarchical_errors=[0.1, float('nan'), 0.3])

    def test_invalid_convergence_status_raises_error(self):
        """Test that invalid convergence status raises error."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Invalid convergence status"):
            PredictionState(
                hierarchical_errors=[0.1, 0.2, 0.3],
                convergence_status="invalid_status"
            )

    def test_negative_learning_iteration_raises_error(self):
        """Test that negative learning iteration raises error."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Learning iteration must be non-negative"):
            PredictionState(
                hierarchical_errors=[0.1, 0.2, 0.3],
                learning_iteration=-1
            )


class TestPredictionStateProperties:
    """Test suite for PredictionState computed properties."""
    
    def test_hierarchy_levels_property(self):
        """Test hierarchy levels property calculation."""
        # Arrange
        errors = [0.1, 0.2, 0.3, 0.4, 0.5]
        state = PredictionState(hierarchical_errors=errors)
        
        # Act & Assert
        assert state.hierarchy_levels == 5

    def test_total_error_calculation(self):
        """Test total error calculation."""
        # Arrange
        errors = [0.1, -0.2, 0.3, -0.4]  # Mix of positive and negative
        state = PredictionState(hierarchical_errors=errors)
        
        # Act
        total_error = state.total_error
        
        # Assert
        assert total_error == 1.0  # |0.1| + |-0.2| + |0.3| + |-0.4|

    def test_mean_error_calculation(self):
        """Test mean error calculation."""
        # Arrange
        errors = [0.2, 0.4, 0.6]
        state = PredictionState(hierarchical_errors=errors)
        
        # Act
        mean_error = state.mean_error
        
        # Assert
        assert abs(mean_error - 0.4) < 1e-10  # (0.2 + 0.4 + 0.6) / 3 with floating point tolerance

    def test_error_variance_calculation(self):
        """Test error variance calculation."""
        # Arrange
        errors = [1.0, 2.0, 3.0]  # Mean = 2.0
        state = PredictionState(hierarchical_errors=errors)
        
        # Act
        variance = state.error_variance
        
        # Assert
        expected_variance = ((1.0 - 2.0)**2 + (2.0 - 2.0)**2 + (3.0 - 2.0)**2) / 3
        assert abs(variance - expected_variance) < 1e-10

    def test_error_variance_single_error(self):
        """Test error variance with single error."""
        # Arrange
        state = PredictionState(hierarchical_errors=[0.5])
        
        # Act & Assert
        assert state.error_variance == 0.0

    def test_is_converged_property(self):
        """Test is_converged property."""
        # Arrange
        converged_state = PredictionState(
            hierarchical_errors=[0.1, 0.2],
            convergence_status="converged"
        )
        not_converged_state = PredictionState(
            hierarchical_errors=[0.1, 0.2],
            convergence_status="not_converged"
        )
        
        # Act & Assert
        assert converged_state.is_converged
        assert not not_converged_state.is_converged

    def test_is_stable_property(self):
        """Test is_stable property based on error variance."""
        # Arrange
        stable_state = PredictionState(hierarchical_errors=[0.1, 0.1, 0.1])  # Low variance
        unstable_state = PredictionState(hierarchical_errors=[0.1, 1.0, 0.1])  # High variance
        
        # Act & Assert
        assert stable_state.is_stable
        assert not unstable_state.is_stable

    def test_prediction_quality_calculation(self):
        """Test prediction quality calculation."""
        # Arrange
        low_error_state = PredictionState(hierarchical_errors=[0.1, 0.1, 0.1])
        high_error_state = PredictionState(hierarchical_errors=[5.0, 5.0, 5.0])
        
        # Act
        low_error_quality = low_error_state.prediction_quality
        high_error_quality = high_error_state.prediction_quality
        
        # Assert
        assert 0.0 <= low_error_quality <= 1.0
        assert 0.0 <= high_error_quality <= 1.0
        assert low_error_quality > high_error_quality


class TestPredictionStateAccessors:
    """Test suite for PredictionState accessor methods."""
    
    def test_get_error_at_level_valid_index(self):
        """Test getting error at valid hierarchical level."""
        # Arrange
        errors = [0.1, 0.2, 0.3]
        state = PredictionState(hierarchical_errors=errors)
        
        # Act & Assert
        assert state.get_error_at_level(0) == 0.1
        assert state.get_error_at_level(1) == 0.2
        assert state.get_error_at_level(2) == 0.3

    def test_get_error_at_level_invalid_index(self):
        """Test getting error at invalid hierarchical level."""
        # Arrange
        errors = [0.1, 0.2, 0.3]
        state = PredictionState(hierarchical_errors=errors)
        
        # Act & Assert
        with pytest.raises(IndexError, match="Level .* out of bounds"):
            state.get_error_at_level(3)
        
        with pytest.raises(IndexError, match="Level .* out of bounds"):
            state.get_error_at_level(-1)

    def test_get_prediction_at_level_valid_index(self):
        """Test getting prediction at valid hierarchical level."""
        # Arrange
        errors = [0.1, 0.2, 0.3]
        predictions = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        state = PredictionState(
            hierarchical_errors=errors,
            hierarchical_predictions=predictions
        )
        
        # Act & Assert
        np.testing.assert_array_equal(state.get_prediction_at_level(0), np.array([1, 2]))
        np.testing.assert_array_equal(state.get_prediction_at_level(1), np.array([3, 4]))
        np.testing.assert_array_equal(state.get_prediction_at_level(2), np.array([5, 6]))

    def test_get_prediction_at_level_no_predictions(self):
        """Test getting prediction when no predictions are stored."""
        # Arrange
        state = PredictionState(hierarchical_errors=[0.1, 0.2, 0.3])
        
        # Act & Assert
        assert state.get_prediction_at_level(0) is None

    def test_get_prediction_at_level_invalid_index(self):
        """Test getting prediction at invalid hierarchical level."""
        # Arrange
        state = PredictionState(hierarchical_errors=[0.1, 0.2, 0.3])
        
        # Act & Assert
        with pytest.raises(IndexError, match="Level .* out of bounds"):
            state.get_prediction_at_level(3)


class TestPredictionStateImmutability:
    """Test suite for PredictionState immutability and value object behavior."""
    
    def test_prediction_state_is_immutable(self):
        """Test that PredictionState instances are immutable."""
        # Arrange
        state = PredictionState(hierarchical_errors=[0.1, 0.2, 0.3])
        
        # Act & Assert
        with pytest.raises(AttributeError):
            state.hierarchical_errors = [0.2, 0.3, 0.4]

    def test_with_updated_errors_creates_new_instance(self):
        """Test that updating errors creates new instance."""
        # Arrange
        original_state = PredictionState(
            hierarchical_errors=[0.1, 0.2, 0.3],
            learning_iteration=5
        )
        
        # Act
        updated_state = original_state.with_updated_errors([0.05, 0.1, 0.15])
        
        # Assert
        assert updated_state.hierarchical_errors == [0.05, 0.1, 0.15]
        assert original_state.hierarchical_errors == [0.1, 0.2, 0.3]  # Original unchanged
        assert updated_state.learning_iteration == 6  # Incremented
        assert updated_state is not original_state  # Different instances

    def test_with_updated_predictions_creates_new_instance(self):
        """Test that updating predictions creates new instance."""
        # Arrange
        original_predictions = [np.array([1, 2]), np.array([3, 4])]
        original_state = PredictionState(
            hierarchical_errors=[0.1, 0.2],
            hierarchical_predictions=original_predictions
        )
        
        new_predictions = [np.array([5, 6]), np.array([7, 8])]
        
        # Act
        updated_state = original_state.with_updated_predictions(new_predictions)
        
        # Assert
        assert len(updated_state.hierarchical_predictions) == 2
        np.testing.assert_array_equal(updated_state.hierarchical_predictions[0], np.array([5, 6]))
        np.testing.assert_array_equal(original_state.hierarchical_predictions[0], np.array([1, 2]))

    def test_add_metadata_creates_new_instance(self):
        """Test that adding metadata creates new instance."""
        # Arrange
        original_state = PredictionState(hierarchical_errors=[0.1, 0.2, 0.3])
        
        # Act
        updated_state = original_state.add_metadata("test_key", "test_value")
        
        # Assert
        assert updated_state.metadata["test_key"] == "test_value"
        assert "test_key" not in original_state.metadata
        assert updated_state is not original_state


class TestPredictionStateConvergenceLogic:
    """Test suite for convergence status determination logic."""
    
    def test_convergence_status_determination_converged(self, prediction_error_patterns):
        """Test convergence detection for converging patterns."""
        # Arrange
        original_state = PredictionState(
            hierarchical_errors=prediction_error_patterns['stable_low']
        )
        very_low_errors = [0.0001, 0.0001, 0.0001, 0.0001]
        
        # Act
        updated_state = original_state.with_updated_errors(very_low_errors)
        
        # Assert
        assert updated_state.convergence_status == "converged"

    def test_convergence_status_determination_diverged(self):
        """Test divergence detection for increasing errors."""
        # Arrange
        original_state = PredictionState(hierarchical_errors=[0.1, 0.1, 0.1])
        much_higher_errors = [1.5, 1.5, 1.5]  # >10% increase
        
        # Act
        updated_state = original_state.with_updated_errors(much_higher_errors)
        
        # Assert
        assert updated_state.convergence_status == "diverged"

    def test_convergence_status_determination_converging(self):
        """Test converging detection for decreasing errors."""
        # Arrange
        original_state = PredictionState(hierarchical_errors=[1.0, 1.0, 1.0])
        lower_errors = [0.8, 0.8, 0.8]  # >1% decrease
        
        # Act
        updated_state = original_state.with_updated_errors(lower_errors)
        
        # Assert
        assert updated_state.convergence_status == "converging"

    def test_convergence_status_determination_not_converged(self):
        """Test not converged status for stable errors."""
        # Arrange
        original_state = PredictionState(hierarchical_errors=[0.5, 0.5, 0.5])
        similar_errors = [0.505, 0.505, 0.505]  # Minimal change
        
        # Act
        updated_state = original_state.with_updated_errors(similar_errors)
        
        # Assert
        assert updated_state.convergence_status == "not_converged"


class TestPredictionStateSerialization:
    """Test suite for PredictionState serialization and factory methods."""
    
    def test_to_dict_includes_all_properties(self):
        """Test dictionary conversion includes all properties."""
        # Arrange
        predictions = [np.array([1, 2]), np.array([3, 4])]
        state = PredictionState(
            hierarchical_errors=[0.1, 0.2],
            hierarchical_predictions=predictions,
            precision_weighted_errors=[0.05, 0.1],
            convergence_status="converged",
            learning_iteration=50
        )
        
        # Act
        state_dict = state.to_dict()
        
        # Assert
        required_keys = [
            'hierarchical_errors', 'hierarchical_predictions',
            'precision_weighted_errors', 'timestamp', 'convergence_status',
            'learning_iteration', 'metadata', 'total_error', 'mean_error',
            'prediction_quality', 'is_converged', 'is_stable'
        ]
        
        for key in required_keys:
            assert key in state_dict
        
        # Check numpy array serialization
        assert isinstance(state_dict['hierarchical_predictions'], list)
        assert isinstance(state_dict['hierarchical_predictions'][0], list)

    def test_create_empty_factory(self):
        """Test empty prediction state factory method."""
        # Act
        empty_state = PredictionState.create_empty(hierarchy_levels=4)
        
        # Assert
        assert len(empty_state.hierarchical_errors) == 4
        assert all(error == 0.0 for error in empty_state.hierarchical_errors)
        assert empty_state.convergence_status == "not_converged"
        assert empty_state.learning_iteration == 0
        assert empty_state.metadata["initialized"] is True


class TestPredictionStatePropertyBased:
    """Property-based tests for PredictionState mathematical properties."""
    
    @given(
        errors=st.lists(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=10
        )
    )
    def test_prediction_state_invariants(self, errors):
        """Test PredictionState invariants hold for all valid inputs."""
        # Arrange & Act
        state = PredictionState(hierarchical_errors=errors)
        
        # Assert invariants
        assert state.hierarchy_levels == len(errors)
        assert state.total_error >= 0.0  # Total error is always non-negative
        assert state.mean_error >= 0.0  # Mean error is always non-negative
        assert state.error_variance >= 0.0  # Variance is always non-negative
        assert 0.0 <= state.prediction_quality <= 1.0  # Quality is bounded
        
        # Test accessor methods don't raise errors
        for i in range(len(errors)):
            assert state.get_error_at_level(i) == errors[i]

    @given(
        errors=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=5
        )
    )
    def test_error_statistics_relationships(self, errors):
        """Test relationships between error statistics."""
        # Arrange & Act
        state = PredictionState(hierarchical_errors=errors)
        
        # Assert relationships
        assert state.total_error == sum(abs(e) for e in errors)
        assert state.mean_error == state.total_error / len(errors)
        
        # If all errors are the same, variance should be zero
        if len(set(errors)) == 1:
            assert abs(state.error_variance) < 1e-10

    @given(
        initial_errors=st.lists(
            st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=5
        ),
        error_multiplier=st.floats(min_value=0.01, max_value=3.0, allow_nan=False, allow_infinity=False)
    )
    def test_convergence_status_consistency(self, initial_errors, error_multiplier):
        """Test convergence status determination consistency."""
        # Arrange
        original_state = PredictionState(hierarchical_errors=initial_errors)
        new_errors = [e * error_multiplier for e in initial_errors]
        
        # Act
        updated_state = original_state.with_updated_errors(new_errors)
        
        # Assert consistency based on error change
        original_total = sum(abs(e) for e in initial_errors)
        new_total = sum(abs(e) for e in new_errors)
        
        if new_total < 0.001:
            assert updated_state.convergence_status == "converged"
        elif new_total > original_total * 1.1:
            assert updated_state.convergence_status == "diverged"
        elif new_total < original_total * 0.99:
            assert updated_state.convergence_status == "converging"
        else:
            assert updated_state.convergence_status == "not_converged"


class TestPredictionStateEdgeCases:
    """Test suite for PredictionState edge cases and boundary conditions."""
    
    def test_single_error_state(self):
        """Test prediction state with single hierarchical level."""
        # Arrange
        state = PredictionState(hierarchical_errors=[0.5])
        
        # Act & Assert
        assert state.hierarchy_levels == 1
        assert state.total_error == 0.5
        assert state.mean_error == 0.5
        assert state.error_variance == 0.0
        assert state.is_stable  # Single error is always "stable"

    def test_very_large_errors(self):
        """Test handling of very large prediction errors."""
        # Arrange
        large_errors = [1e6, 1e6, 1e6]
        state = PredictionState(hierarchical_errors=large_errors)
        
        # Act & Assert
        assert not np.isinf(state.total_error)
        assert not np.isnan(state.mean_error)
        assert 0.0 <= state.prediction_quality <= 1.0

    def test_zero_errors(self):
        """Test handling of zero prediction errors."""
        # Arrange
        zero_errors = [0.0, 0.0, 0.0]
        state = PredictionState(hierarchical_errors=zero_errors)
        
        # Act & Assert
        assert state.total_error == 0.0
        assert state.mean_error == 0.0
        assert state.error_variance == 0.0
        assert state.prediction_quality == 1.0  # Perfect predictions
        assert state.is_stable

    def test_mixed_sign_errors(self):
        """Test handling of errors with mixed signs."""
        # Arrange
        mixed_errors = [-0.5, 0.3, -0.2, 0.4]
        state = PredictionState(hierarchical_errors=mixed_errors)
        
        # Act & Assert
        assert state.total_error == 1.4  # Sum of absolute values
        assert state.mean_error == 0.35
        # Variance calculation should handle mixed signs correctly
        assert state.error_variance >= 0.0

    def test_prediction_state_with_very_high_iteration(self):
        """Test prediction state with very high learning iteration."""
        # Arrange
        high_iteration = 1000000
        state = PredictionState(
            hierarchical_errors=[0.1, 0.2],
            learning_iteration=high_iteration
        )
        
        # Act
        updated_state = state.with_updated_errors([0.05, 0.1])
        
        # Assert
        assert updated_state.learning_iteration == high_iteration + 1

    def test_timestamp_progression(self):
        """Test that timestamps progress correctly on updates."""
        # Arrange
        original_state = PredictionState(hierarchical_errors=[0.1, 0.2])
        original_time = original_state.timestamp
        
        # Add small delay to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        # Act
        updated_state = original_state.with_updated_errors([0.05, 0.1])
        
        # Assert
        assert updated_state.timestamp > original_time