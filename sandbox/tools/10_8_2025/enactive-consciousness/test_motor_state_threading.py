#!/usr/bin/env python3
"""TDD test for motor state threading issue.

RED: Test that exposes the state threading bug
GREEN: Fix to make the test pass
REFACTOR: Clean up the solution
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
    MotorSchemaNetwork,
)


def test_motor_state_threading_bug():
    """RED: Expose the bug where motor intention is confused with GRU hidden state."""
    config = BodySchemaConfig(motor_dim=32)
    key = jax.random.PRNGKey(42)
    hidden_dim = max(64, config.motor_dim * 2)  # Should be 64
    
    motor_schema = MotorSchemaNetwork(config.motor_dim, hidden_dim, key)
    
    # First processing step - should work fine
    motor_input_1 = jax.random.normal(key, (config.motor_dim,))
    motor_command_1, hidden_state_1, confidence_1 = motor_schema.process_motor_intention(motor_input_1)
    
    print(f"First step:")
    print(f"  motor_input_1.shape: {motor_input_1.shape}")
    print(f"  hidden_state_1.shape: {hidden_state_1.shape}")
    print(f"  motor_command_1.shape: {motor_command_1.shape}")
    
    # The bug: passing motor_command_1 (32-dim) as previous_state to GRU expecting 64-dim
    motor_input_2 = jax.random.normal(jax.random.PRNGKey(43), (config.motor_dim,))
    
    print(f"\nSecond step (reproducing bug):")
    print(f"  motor_input_2.shape: {motor_input_2.shape}")
    print(f"  Passing motor_command_1 (32-dim) as hidden state to GRU expecting 64-dim")
    
    # This should fail with dimension mismatch
    try:
        motor_command_2, hidden_state_2, confidence_2 = motor_schema.process_motor_intention(
            motor_input_2, previous_state=motor_command_1  # BUG: wrong state type
        )
        pytest.fail("Expected dimension mismatch error, but got success")
    except Exception as e:
        print(f"  Expected error: {e}")
        assert "dot_general" in str(e) or "shape" in str(e), f"Unexpected error: {e}"
    
    # Correct usage: passing hidden_state_1 instead
    print(f"\nSecond step (correct usage):")
    print(f"  Passing hidden_state_1 (64-dim) to GRU")
    motor_command_2, hidden_state_2, confidence_2 = motor_schema.process_motor_intention(
        motor_input_2, previous_state=hidden_state_1  # CORRECT: proper hidden state
    )
    
    print(f"  Success! motor_command_2.shape: {motor_command_2.shape}")
    print(f"  hidden_state_2.shape: {hidden_state_2.shape}")
    
    assert motor_command_2.shape == (config.motor_dim,)
    assert hidden_state_2.shape == (hidden_dim,)


if __name__ == "__main__":
    test_motor_state_threading_bug()
    print("\n✅ Test completed - bug reproduced and solution verified!")