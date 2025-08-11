#!/usr/bin/env python3
"""Test-driven development approach to identify and fix dimension mismatch errors.

Following TDD principles:
1. RED: Write failing tests that reproduce the dimension mismatch issue
2. GREEN: Write minimal code to pass the tests
3. REFACTOR: Improve the code while keeping tests passing

Focus on the body schema integration dimension errors reported.
"""

import jax
import jax.numpy as jnp
import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from enactive_consciousness.embodiment import (
    BodySchemaConfig,
    create_body_schema_processor_safe,
    MotorSchemaNetwork,
    BodyBoundaryDetector,
    BodySchemaIntegration,
)
from enactive_consciousness.types import EmbodimentError


class TestDimensionConsistency:
    """Test suite focusing on dimension consistency in body schema integration."""
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration with known dimensions."""
        return BodySchemaConfig(
            proprioceptive_dim=64,
            motor_dim=32,
            body_map_resolution=(8, 8),  # Small resolution for testing
            schema_adaptation_rate=0.02,
        )
    
    @pytest.fixture
    def test_key(self):
        """Create a test PRNG key."""
        return jax.random.PRNGKey(42)
    
    @pytest.fixture
    def test_inputs(self, test_config, test_key):
        """Create test inputs with known dimensions."""
        keys = jax.random.split(test_key, 3)
        
        proprioceptive_input = jax.random.normal(keys[0], (test_config.proprioceptive_dim,))
        motor_prediction = jax.random.normal(keys[1], (test_config.motor_dim,))
        tactile_feedback = jax.random.normal(keys[2], (32,))  # Fixed 32 dimensions
        
        return proprioceptive_input, motor_prediction, tactile_feedback
    
    def test_motor_schema_dimension_consistency(self, test_config, test_key):
        """RED: Test that motor schema network handles dimensions consistently."""
        hidden_dim = max(64, test_config.motor_dim * 2)
        motor_schema = MotorSchemaNetwork(test_config.motor_dim, hidden_dim, test_key)
        
        # Test single processing step
        motor_input = jax.random.normal(test_key, (test_config.motor_dim,))
        motor_command, new_state, confidence = motor_schema.process_motor_intention(motor_input)
        
        # Verify output dimensions
        assert motor_command.shape == (test_config.motor_dim,), f"Motor command shape mismatch: {motor_command.shape}"
        assert new_state.shape == (hidden_dim,), f"Hidden state shape mismatch: {new_state.shape}"
        assert isinstance(confidence, float), f"Confidence should be float, got: {type(confidence)}"
        
        # Test state threading
        motor_input_2 = jax.random.normal(jax.random.PRNGKey(43), (test_config.motor_dim,))
        motor_command_2, new_state_2, confidence_2 = motor_schema.process_motor_intention(
            motor_input_2, previous_state=new_state
        )
        
        # Verify consistency after state threading
        assert motor_command_2.shape == (test_config.motor_dim,), "Second motor command shape mismatch"
        assert new_state_2.shape == (hidden_dim,), "Second hidden state shape mismatch"
    
    def test_boundary_detector_input_consistency(self, test_config, test_key):
        """RED: Test that boundary detector handles varying input dimensions consistently."""
        tactile_dim = 32  # Fixed tactile dimension from basic_demo.py
        total_sensory_dim = test_config.proprioceptive_dim + tactile_dim
        hidden_dim = max(64, test_config.motor_dim * 2)
        
        boundary_detector = BodyBoundaryDetector(
            total_sensory_dim, test_config.motor_dim, hidden_dim, test_key
        )
        
        # Test with correct dimensions
        proprioceptive_input = jax.random.normal(test_key, (test_config.proprioceptive_dim,))
        tactile_input = jax.random.normal(jax.random.PRNGKey(43), (tactile_dim,))
        motor_state = jax.random.normal(jax.random.PRNGKey(44), (test_config.motor_dim,))
        
        boundary_signal, boundary_confidence = boundary_detector.detect_body_boundary(
            proprioceptive_input, tactile_input, motor_state
        )
        
        assert isinstance(boundary_signal, float), "Boundary signal should be float"
        assert isinstance(boundary_confidence, float), "Boundary confidence should be float"
        assert 0.0 <= boundary_confidence <= 1.0, f"Invalid boundary confidence: {boundary_confidence}"
        
    def test_dimension_mismatch_reproduction(self, test_config, test_key):
        """RED: Try to reproduce the dimension mismatch error mentioned in the issue."""
        processor = create_body_schema_processor_safe(test_config, test_key, use_jit=False)
        
        # Create inputs with exact dimensions from basic_demo.py
        proprioceptive_input = jax.random.normal(test_key, (64,))  # 64-dim
        motor_prediction = jax.random.normal(jax.random.PRNGKey(43), (32,))  # 32-dim
        tactile_feedback = jax.random.normal(jax.random.PRNGKey(44), (32,))  # 32-dim
        
        # This should work without dimension errors
        try:
            body_state = processor.integrate_body_schema(
                proprioceptive_input=proprioceptive_input,
                motor_prediction=motor_prediction,
                tactile_feedback=tactile_feedback,
            )
            
            # Verify the output state has expected structure
            assert hasattr(body_state, 'proprioception'), "Missing proprioception field"
            assert hasattr(body_state, 'motor_intention'), "Missing motor_intention field"
            assert hasattr(body_state, 'boundary_signal'), "Missing boundary_signal field"
            assert hasattr(body_state, 'schema_confidence'), "Missing schema_confidence field"
            
            # Verify dimensions
            assert body_state.proprioception.shape == (test_config.proprioceptive_dim,), \
                f"Proprioception dimension mismatch: {body_state.proprioception.shape}"
            assert body_state.motor_intention.shape == (test_config.motor_dim,), \
                f"Motor intention dimension mismatch: {body_state.motor_intention.shape}"
            assert body_state.boundary_signal.shape == (1,), \
                f"Boundary signal dimension mismatch: {body_state.boundary_signal.shape}"
            assert isinstance(body_state.schema_confidence, float), \
                f"Schema confidence should be float, got: {type(body_state.schema_confidence)}"
                
        except Exception as e:
            pytest.fail(f"Integration failed with error: {e}")
    
    def test_sequential_processing_dimension_consistency(self, test_config, test_key):
        """RED: Test dimension consistency over multiple sequential processing steps."""
        processor = create_body_schema_processor_safe(test_config, test_key, use_jit=False)
        
        # Process multiple steps with varying inputs (mimicking integrated processing)
        states = []
        previous_motor_state = None
        
        for step in range(5):
            step_key = jax.random.PRNGKey(100 + step)
            keys = jax.random.split(step_key, 3)
            
            # Create varying but dimensionally consistent inputs
            proprioceptive_input = jax.random.normal(keys[0], (64,)) * (0.5 + step * 0.1)
            motor_prediction = jax.random.normal(keys[1], (32,)) * (0.3 + step * 0.2)
            tactile_feedback = jax.random.normal(keys[2], (32,)) * 0.1
            
            try:
                body_state = processor.integrate_body_schema(
                    proprioceptive_input=proprioceptive_input,
                    motor_prediction=motor_prediction,
                    tactile_feedback=tactile_feedback,
                    previous_motor_state=previous_motor_state,
                )
                
                states.append(body_state)
                
                # Extract motor state for next iteration (if available)
                if hasattr(body_state, 'motor_intention'):
                    previous_motor_state = body_state.motor_intention
                    
                # Verify dimension consistency across steps
                assert body_state.proprioception.shape == (test_config.proprioceptive_dim,), \
                    f"Step {step}: Proprioception dimension changed: {body_state.proprioception.shape}"
                assert body_state.motor_intention.shape == (test_config.motor_dim,), \
                    f"Step {step}: Motor intention dimension changed: {body_state.motor_intention.shape}"
                    
            except Exception as e:
                pytest.fail(f"Sequential processing failed at step {step} with error: {e}")
        
        # Verify we processed all steps successfully
        assert len(states) == 5, f"Expected 5 states, got {len(states)}"
    
    def test_varying_tactile_input_dimensions(self, test_config, test_key):
        """RED: Test what happens when tactile input dimensions vary (potential cause of the issue)."""
        processor = create_body_schema_processor_safe(test_config, test_key, use_jit=False)
        
        # Test with different tactile input sizes
        tactile_dimensions = [16, 32, 48, 64]  # Different sizes that might occur
        
        for tactile_dim in tactile_dimensions:
            step_key = jax.random.PRNGKey(200 + tactile_dim)
            keys = jax.random.split(step_key, 3)
            
            proprioceptive_input = jax.random.normal(keys[0], (64,))
            motor_prediction = jax.random.normal(keys[1], (32,))
            tactile_feedback = jax.random.normal(keys[2], (tactile_dim,))
            
            # This test is expected to fail for non-32 dimensions
            # based on the current BodyBoundaryDetector implementation
            if tactile_dim == 32:
                try:
                    body_state = processor.integrate_body_schema(
                        proprioceptive_input=proprioceptive_input,
                        motor_prediction=motor_prediction,
                        tactile_feedback=tactile_feedback,
                    )
                    # Should succeed for 32-dim tactile input
                    assert body_state is not None, "Body state should not be None for 32-dim tactile input"
                except Exception as e:
                    pytest.fail(f"Should succeed for 32-dim tactile input, but failed: {e}")
            else:
                # Should handle dimension mismatch gracefully or fail predictably
                with pytest.raises((ValueError, EmbodimentError, Exception)):
                    processor.integrate_body_schema(
                        proprioceptive_input=proprioceptive_input,
                        motor_prediction=motor_prediction,
                        tactile_feedback=tactile_feedback,
                    )
    
    def test_motor_schema_gru_state_consistency(self, test_config, test_key):
        """RED: Test GRU state consistency that may cause dot_general errors."""
        hidden_dim = max(64, test_config.motor_dim * 2)
        motor_schema = MotorSchemaNetwork(test_config.motor_dim, hidden_dim, test_key)
        
        # Test state consistency over multiple calls
        motor_inputs = [
            jax.random.normal(jax.random.PRNGKey(300 + i), (test_config.motor_dim,))
            for i in range(10)
        ]
        
        current_state = None
        for i, motor_input in enumerate(motor_inputs):
            try:
                motor_command, new_state, confidence = motor_schema.process_motor_intention(
                    motor_input, previous_state=current_state
                )
                
                # Verify state dimensions remain consistent
                assert new_state.shape == (hidden_dim,), \
                    f"Iteration {i}: GRU state dimension changed: {new_state.shape}"
                
                # Verify no NaN or infinite values (can cause dot_general errors)
                assert jnp.all(jnp.isfinite(new_state)), \
                    f"Iteration {i}: GRU state contains non-finite values"
                assert jnp.all(jnp.isfinite(motor_command)), \
                    f"Iteration {i}: Motor command contains non-finite values"
                
                current_state = new_state
                
            except Exception as e:
                pytest.fail(f"GRU processing failed at iteration {i}: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])