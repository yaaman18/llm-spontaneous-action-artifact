#!/usr/bin/env python3
"""Corrected TDD test suite for Equinox state management in enactive consciousness.

This test suite follows TDD methodology with the correct Equinox v0.13+ API
for state management patterns including proper stateful layer usage,
eqx.tree_at operations, and state threading patterns.

Test Categories:
1. Stateful layer integration (GRU, Linear with state)
2. eqx.tree_at for immutable PyTree updates  
3. State threading through temporal sequences
4. Memory buffer management patterns
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


class TemporalMemoryState(eqx.Module):
    """Memory state container for temporal consciousness (as eqx.Module for tree_at)."""
    retention_buffer: jax.Array
    protention_weights: jax.Array
    coupling_matrix: jax.Array
    iteration_count: int


class EnactiveTemporalProcessor(eqx.Module):
    """Processor implementing enactive temporal dynamics with proper state management."""
    
    gru_cell: eqx.nn.GRUCell
    memory_projector: eqx.nn.Linear
    coupling_network: eqx.nn.Linear
    
    hidden_dim: int
    buffer_depth: int
    
    def __init__(self, input_dim: int, hidden_dim: int, buffer_depth: int, key: jax.Array):
        keys = jax.random.split(key, 3)
        
        self.gru_cell = eqx.nn.GRUCell(input_dim, hidden_dim, key=keys[0])
        self.memory_projector = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1])
        self.coupling_network = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[2])
        
        self.hidden_dim = hidden_dim
        self.buffer_depth = buffer_depth


class TestEquinoxStatefulLayers:
    """Test proper Equinox stateful layer integration."""
    
    def setup_method(self):
        """Setup test fixtures with correct Equinox patterns."""
        self.key = jax.random.PRNGKey(42)
        self.input_dim = 32
        self.hidden_dim = 64
        self.buffer_depth = 15
        self.batch_size = 2
        
        self.processor = EnactiveTemporalProcessor(
            self.input_dim,
            self.hidden_dim,
            self.buffer_depth,
            self.key
        )
        
        # Test data
        self.test_input = jax.random.normal(
            self.key,
            (self.batch_size, self.input_dim)
        )
        self.initial_hidden = jnp.zeros((self.batch_size, self.hidden_dim))
    
    def test_gru_cell_stateful_operation_red(self):
        """Test RED: GRU cell should handle stateful operations correctly."""
        # This test drives proper stateful layer usage
        
        # Arrange: GRU cell and states
        gru_cell = self.processor.gru_cell
        
        # Act: Single GRU step
        new_hidden = jax.vmap(gru_cell)(self.test_input, self.initial_hidden)
        
        # Assert: Proper stateful evolution  
        assert new_hidden.shape == (self.batch_size, self.hidden_dim), "Hidden state shape preserved"
        assert not jnp.allclose(new_hidden, self.initial_hidden), "State should evolve"
        assert jnp.all(jnp.isfinite(new_hidden)), "All values should be finite"
        
        # GRU uses tanh activation, so outputs should be bounded
        assert jnp.all(new_hidden >= -1.0) and jnp.all(new_hidden <= 1.0), "GRU outputs should be bounded"
    
    def test_stateful_sequence_processing_green(self):
        """Test GREEN: Stateful processing through sequences should maintain consistency."""
        
        # Arrange: Sequence data
        sequence_length = 10
        sequence = jax.random.normal(
            self.key,
            (sequence_length, self.batch_size, self.input_dim)
        )
        
        # Act: Process sequence maintaining state
        def process_stateful_sequence(
            processor: EnactiveTemporalProcessor,
            sequence: jax.Array,
            initial_hidden: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            """Process sequence with proper state threading."""
            
            def step_fn(hidden_state: jax.Array, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
                # GRU step
                new_hidden = jax.vmap(processor.gru_cell)(x, hidden_state)
                
                # Memory projection
                memory_out = jax.vmap(processor.memory_projector)(new_hidden)
                
                return new_hidden, memory_out
            
            final_hidden, outputs = jax.lax.scan(step_fn, initial_hidden, sequence)
            return outputs, final_hidden
        
        outputs, final_hidden = process_stateful_sequence(
            self.processor,
            sequence,
            self.initial_hidden
        )
        
        # Assert: Proper sequence processing
        assert outputs.shape == (sequence_length, self.batch_size, self.hidden_dim), "Output sequence shape"
        assert final_hidden.shape == (self.batch_size, self.hidden_dim), "Final hidden shape"
        assert not jnp.allclose(self.initial_hidden, final_hidden), "Hidden state should evolve"
        assert jnp.all(jnp.isfinite(outputs)), "All outputs should be finite"
    
    def test_gradient_flow_through_stateful_operations_green(self):
        """Test GREEN: Gradients should flow properly through stateful operations."""
        
        # Arrange: Loss function for gradient testing
        def temporal_loss_function(
            processor: EnactiveTemporalProcessor,
            sequence: jax.Array,
            targets: jax.Array,
            initial_hidden: jax.Array
        ) -> jax.Array:
            """Compute loss for temporal prediction task."""
            
            def forward_pass(sequence: jax.Array) -> jax.Array:
                def step_fn(hidden: jax.Array, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
                    new_hidden = jax.vmap(processor.gru_cell)(x, hidden)
                    output = jax.vmap(processor.memory_projector)(new_hidden)
                    return new_hidden, output
                
                _, predictions = jax.lax.scan(step_fn, initial_hidden, sequence)
                return predictions
            
            predictions = forward_pass(sequence)
            return jnp.mean((predictions - targets) ** 2)
        
        # Test data
        sequence_length = 8
        sequence = jax.random.normal(
            self.key,
            (sequence_length, self.batch_size, self.input_dim)
        )
        targets = jax.random.normal(
            self.key,
            (sequence_length, self.batch_size, self.hidden_dim)
        )
        
        # Act: Compute gradients
        loss_fn = lambda proc: temporal_loss_function(
            proc, sequence, targets, self.initial_hidden
        )
        loss_value, grads = jax.value_and_grad(loss_fn)(self.processor)
        
        # Assert: Gradient flow
        assert jnp.isfinite(loss_value), "Loss should be finite"
        
        # Check GRU gradients
        gru_grads = grads.gru_cell
        assert jnp.all(jnp.isfinite(gru_grads.weight_ih)), "GRU input-hidden gradients finite"
        assert jnp.all(jnp.isfinite(gru_grads.weight_hh)), "GRU hidden-hidden gradients finite"
        assert jnp.all(jnp.isfinite(gru_grads.bias)), "GRU bias gradients finite"
        
        # Gradients should be non-zero (indicating flow)
        assert jnp.sum(jnp.abs(gru_grads.weight_ih)) > 1e-6, "GRU gradients should be non-zero"


class TestEquinoxTreeAtOperations:
    """Test eqx.tree_at for immutable PyTree operations."""
    
    def setup_method(self):
        """Setup fixtures for tree operations."""
        self.key = jax.random.PRNGKey(24)
        
        # Complex temporal memory state
        self.temporal_state = TemporalMemoryState(
            retention_buffer=jnp.zeros((15, 64)),  # 15 time steps, 64 dim
            protention_weights=jnp.ones((8,)) * 0.5,  # 8 future steps
            coupling_matrix=jnp.eye(64) * 0.1,  # Coupling between dimensions
            iteration_count=0
        )
    
    def test_tree_at_memory_buffer_update_red(self):
        """Test RED: tree_at should handle memory buffer updates immutably."""
        # This test drives proper immutable update patterns
        
        # Arrange: New experience to add to buffer
        new_experience = jax.random.normal(self.key, (64,))
        
        # Act: Update buffer using eqx.tree_at (correct pattern)
        # First, update just the last element of the buffer
        updated_state = eqx.tree_at(
            lambda state: state.retention_buffer,
            self.temporal_state,
            self.temporal_state.retention_buffer.at[-1].set(new_experience)
        )
        
        # For full buffer shift operation
        shifted_buffer = jnp.roll(self.temporal_state.retention_buffer, -1, axis=0)
        shifted_buffer = shifted_buffer.at[-1].set(new_experience)
        
        fully_updated_state = eqx.tree_at(
            lambda s: (s.retention_buffer, s.iteration_count),
            self.temporal_state,
            (shifted_buffer, self.temporal_state.iteration_count + 1)
        )
        
        # Assert: Proper immutable updates
        assert jnp.array_equal(
            updated_state.retention_buffer[-1],
            new_experience
        ), "Experience should be stored"
        
        assert fully_updated_state.iteration_count == 1, "Iteration count should increment"
        
        # Original state unchanged (immutability)
        assert self.temporal_state.iteration_count == 0, "Original state preserved"
        assert not jnp.array_equal(
            self.temporal_state.retention_buffer,
            fully_updated_state.retention_buffer
        ), "Buffer should change"
    
    def test_tree_at_coupling_matrix_dynamics_green(self):
        """Test GREEN: tree_at should handle coupling matrix dynamics."""
        
        # Arrange: Coupling update based on activity
        activity_pattern = jax.random.normal(self.key, (64,)) * 0.1
        
        # Hebbian-like coupling update
        coupling_update = jnp.outer(activity_pattern, activity_pattern)
        learning_rate = 0.01
        
        # Act: Update coupling matrix
        new_coupling = (
            self.temporal_state.coupling_matrix + 
            learning_rate * coupling_update
        )
        
        # Ensure symmetry
        symmetric_coupling = (new_coupling + new_coupling.T) / 2
        
        updated_state = eqx.tree_at(
            lambda s: s.coupling_matrix,
            self.temporal_state,
            symmetric_coupling
        )
        
        # Assert: Proper coupling dynamics
        assert not jnp.array_equal(
            self.temporal_state.coupling_matrix,
            updated_state.coupling_matrix
        ), "Coupling should change"
        
        # Check symmetry preservation
        coupling = updated_state.coupling_matrix
        assert jnp.allclose(coupling, coupling.T), "Coupling should be symmetric"
        
        # Check activity influence
        activity_magnitude = jnp.linalg.norm(activity_pattern)
        coupling_change = jnp.linalg.norm(
            updated_state.coupling_matrix - self.temporal_state.coupling_matrix
        )
        assert coupling_change > 0, "Coupling should change based on activity"
    
    def test_tree_at_simultaneous_updates_green(self):
        """Test GREEN: tree_at should handle simultaneous updates correctly."""
        
        # Arrange: Multiple simultaneous updates
        new_buffer = jax.random.normal(self.key, (15, 64)) * 0.1
        new_weights = jax.random.uniform(self.key, (8,))
        new_count = 5
        
        # Act: Apply simultaneous updates (correct pattern for multiple fields)
        updated_state = eqx.tree_at(
            lambda s: (s.retention_buffer, s.protention_weights, s.iteration_count),
            self.temporal_state,
            (new_buffer, new_weights, new_count)
        )
        
        # Assert: All updates applied correctly
        assert jnp.array_equal(
            updated_state.retention_buffer,
            new_buffer
        ), "Buffer should update"
        
        assert jnp.array_equal(
            updated_state.protention_weights,
            new_weights
        ), "Weights should update"
        
        assert updated_state.iteration_count == new_count, "Count should update"
        
        # Original unchanged
        assert jnp.array_equal(
            self.temporal_state.retention_buffer,
            jnp.zeros((15, 64))
        ), "Original buffer unchanged"


class TestAdvancedStatefulPatterns:
    """Test advanced stateful computation patterns."""
    
    def test_memory_sedimentation_pattern_refactor(self):
        """Test REFACTOR: Advanced memory sedimentation through state evolution."""
        
        # Arrange: Long-term memory evolution simulation
        key = jax.random.PRNGKey(42)
        processor = EnactiveTemporalProcessor(32, 64, 20, key)
        
        def simulate_memory_sedimentation(
            processor: EnactiveTemporalProcessor,
            num_experiences: int,
            key: jax.Array
        ) -> List[TemporalMemoryState]:
            """Simulate long-term memory sedimentation process."""
            
            keys = jax.random.split(key, num_experiences + 1)
            
            # Initial memory state
            current_state = TemporalMemoryState(
                retention_buffer=jnp.zeros((processor.buffer_depth, processor.hidden_dim)),
                protention_weights=jnp.ones((8,)) * 0.3,
                coupling_matrix=jnp.eye(processor.hidden_dim) * 0.05,
                iteration_count=0
            )
            
            states_history = [current_state]
            
            for i in range(num_experiences):
                # Generate new experience
                experience = jax.random.normal(keys[i+1], (1, 32))  # Single experience
                
                # Process through network
                hidden = jnp.zeros((1, processor.hidden_dim))
                processed_exp = jax.vmap(processor.gru_cell)(experience, hidden)
                processed_exp = processed_exp.squeeze()  # Remove batch dim
                
                # Update memory buffer (sedimentation)
                shifted_buffer = jnp.roll(current_state.retention_buffer, -1, axis=0)
                shifted_buffer = shifted_buffer.at[-1].set(processed_exp)
                
                # Update coupling based on experience
                coupling_influence = jnp.outer(processed_exp, processed_exp) * 0.001
                updated_coupling = current_state.coupling_matrix + coupling_influence
                updated_coupling = (updated_coupling + updated_coupling.T) / 2
                
                # Decay older protention weights, strengthen based on coupling
                coupling_strength = jnp.trace(updated_coupling) / processor.hidden_dim
                weight_adjustment = current_state.protention_weights * (0.99 + coupling_strength * 0.02)
                
                # Create new state
                current_state = TemporalMemoryState(
                    retention_buffer=shifted_buffer,
                    protention_weights=weight_adjustment,
                    coupling_matrix=updated_coupling,
                    iteration_count=current_state.iteration_count + 1
                )
                
                states_history.append(current_state)
            
            return states_history
        
        # Act: Run sedimentation simulation
        num_experiences = 30
        memory_evolution = simulate_memory_sedimentation(
            processor,
            num_experiences,
            key
        )
        
        # Assert: Sedimentation properties
        assert len(memory_evolution) == num_experiences + 1, "Should track all states"
        
        # Memory complexity should increase
        initial_coupling_norm = jnp.linalg.norm(memory_evolution[0].coupling_matrix)
        final_coupling_norm = jnp.linalg.norm(memory_evolution[-1].coupling_matrix)
        assert final_coupling_norm > initial_coupling_norm, "Coupling should strengthen"
        
        # Buffer should be filled with experiences
        final_buffer_activity = jnp.mean(jnp.abs(memory_evolution[-1].retention_buffer))
        assert final_buffer_activity > 0.01, "Buffer should contain meaningful experiences"
        
        # Weights should adapt
        initial_weight_var = jnp.var(memory_evolution[0].protention_weights)
        final_weight_var = jnp.var(memory_evolution[-1].protention_weights)
        # Some adaptation should occur
        assert not jnp.allclose(
            memory_evolution[0].protention_weights,
            memory_evolution[-1].protention_weights
        ), "Weights should adapt over time"


if __name__ == "__main__":
    # Run with pytest for proper test discovery and reporting
    pytest.main([__file__, "-v", "--tb=short"])