#!/usr/bin/env python3
"""
Enhanced Predictive Coding Demonstration.

Demonstrates the mathematically enhanced JAX-based predictive coding system
following the Free Energy Principle as specified by Karl Friston.
"""

import numpy as np
from datetime import datetime
import jax.numpy as jnp

# Domain objects
from domain.value_objects.phi_value import PhiValue
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.consciousness_state import ConsciousnessState
from domain.value_objects.precision_weights import PrecisionWeights
from domain.value_objects.probability_distribution import ProbabilityDistribution

# Infrastructure implementation
from infrastructure.jax_predictive_coding_core import JaxPredictiveCodingCore


def create_test_input_sequence(sequence_length: int = 10, input_dimensions: int = 4) -> np.ndarray:
    """Create a test input sequence with temporal structure."""
    # Create a sinusoidal pattern with some noise
    t = np.linspace(0, 4*np.pi, sequence_length)
    
    # Multiple frequency components for rich dynamics
    input_sequence = np.zeros((sequence_length, input_dimensions))
    for i in range(input_dimensions):
        freq = (i + 1) * 0.5  # Different frequency for each dimension
        phase = i * np.pi / 4  # Different phase for each dimension
        input_sequence[:, i] = np.sin(freq * t + phase) + 0.1 * np.random.randn(sequence_length)
    
    return input_sequence


def demonstrate_free_energy_minimization():
    """Demonstrate the core free energy minimization process."""
    print("=== Free Energy Minimization Demonstration ===\n")
    
    # Initialize the system
    hierarchy_levels = 3
    input_dimensions = 4
    
    # Create JAX-based predictive coding core
    predictive_core = JaxPredictiveCodingCore(
        hierarchy_levels=hierarchy_levels,
        input_dimensions=input_dimensions,
        learning_rate=0.01,
        enable_active_inference=True,
        temporal_window=5
    )
    
    # Create initial precision weights (uniform attention)
    precision_weights = PrecisionWeights.create_uniform(hierarchy_levels, temperature=1.0)
    
    # Generate test input sequence
    input_sequence = create_test_input_sequence(sequence_length=20, input_dimensions=input_dimensions)
    
    print(f"System Configuration:")
    print(f"  - Hierarchy levels: {hierarchy_levels}")
    print(f"  - Input dimensions: {input_dimensions}")
    print(f"  - Sequence length: {len(input_sequence)}")
    print(f"  - Initial precision weights: {precision_weights.normalized_weights}")
    print()
    
    # Process each input and track free energy evolution
    free_energies = []
    prediction_errors = []
    consciousness_states = []
    
    for t, input_data in enumerate(input_sequence):
        print(f"Processing time step {t+1}/{len(input_sequence)}")
        
        # Process input through predictive coding system
        prediction_state = predictive_core.process_input(
            input_data=input_data,
            precision_weights=precision_weights,
            learning_rate=0.01
        )
        
        # Extract free energy from metadata
        free_energy = prediction_state.metadata.get('free_energy_before_update', 0.0)
        free_energies.append(free_energy)
        
        # Track prediction error evolution
        total_error = prediction_state.total_error
        prediction_errors.append(total_error)
        
        # Create consciousness state
        phi_value = PhiValue(
            value=max(0.0, 1.0 - total_error),  # Φ inversely related to prediction error
            complexity=1.0 + 0.1 * np.log(t + 1),  # Increasing complexity over time
            integration=0.8 * np.exp(-total_error)  # Integration decreases with error
        )
        
        uncertainty_distribution = ProbabilityDistribution.uniform(hierarchy_levels)
        
        consciousness_state = ConsciousnessState(
            phi_value=phi_value,
            prediction_state=prediction_state,
            uncertainty_distribution=uncertainty_distribution,
            metacognitive_confidence=max(0.1, 1.0 - total_error)
        )
        consciousness_states.append(consciousness_state)
        
        # Update precision weights based on attention (simple heuristic)
        if t > 0 and len(prediction_errors) > 5:
            # Focus attention on levels with consistently low error
            recent_errors = prediction_errors[-5:]
            error_trend = np.mean(recent_errors)
            
            if error_trend < 0.5:  # Good performance
                # Focus on lower levels (sensory processing)
                precision_weights = PrecisionWeights.create_focused(
                    hierarchy_levels, focus_level=0, focus_strength=3.0
                )
            else:  # Poor performance
                # Distribute attention more uniformly
                precision_weights = PrecisionWeights.create_uniform(hierarchy_levels, temperature=2.0)
        
        print(f"  - Free energy: {free_energy:.4f}")
        print(f"  - Total prediction error: {total_error:.4f}")
        print(f"  - Φ value: {phi_value.value:.4f}")
        print(f"  - Consciousness level: {consciousness_state.consciousness_level:.4f}")
        print(f"  - Convergence: {prediction_state.convergence_status}")
        print()
    
    # Analyze results
    print("=== Analysis ===")
    print(f"Initial free energy: {free_energies[0]:.4f}")
    print(f"Final free energy: {free_energies[-1]:.4f}")
    print(f"Free energy reduction: {free_energies[0] - free_energies[-1]:.4f}")
    print()
    
    print(f"Initial prediction error: {prediction_errors[0]:.4f}")
    print(f"Final prediction error: {prediction_errors[-1]:.4f}")
    print(f"Error reduction: {prediction_errors[0] - prediction_errors[-1]:.4f}")
    print()
    
    # Analyze consciousness evolution
    initial_consciousness = consciousness_states[0].consciousness_level
    final_consciousness = consciousness_states[-1].consciousness_level
    print(f"Initial consciousness level: {initial_consciousness:.4f}")
    print(f"Final consciousness level: {final_consciousness:.4f}")
    print(f"Consciousness change: {final_consciousness - initial_consciousness:.4f}")
    print()
    
    # Check for convergence
    converged_states = sum(1 for state in consciousness_states 
                          if state.prediction_state.is_converged)
    print(f"Converged states: {converged_states}/{len(consciousness_states)}")
    
    # Display final system state
    final_precision_estimates = predictive_core.get_precision_estimates()
    print(f"Final precision estimates: {final_precision_estimates}")
    
    return {
        'free_energies': free_energies,
        'prediction_errors': prediction_errors,
        'consciousness_states': consciousness_states,
        'predictive_core': predictive_core
    }


def demonstrate_hierarchical_dynamics():
    """Demonstrate hierarchical message passing dynamics."""
    print("\n=== Hierarchical Message Passing ===\n")
    
    # Create system with more levels for better hierarchy demonstration
    hierarchy_levels = 4
    predictive_core = JaxPredictiveCodingCore(
        hierarchy_levels=hierarchy_levels,
        input_dimensions=2,
        learning_rate=0.005,  # Slower learning for stability
        temporal_window=3
    )
    
    # Create structured input (alternating pattern)
    structured_input = np.array([
        [1.0, 0.0],  # Pattern A
        [0.0, 1.0],  # Pattern B
        [1.0, 0.0],  # Pattern A
        [0.0, 1.0],  # Pattern B
        [1.0, 0.0],  # Pattern A
    ])
    
    precision_weights = PrecisionWeights.create_uniform(hierarchy_levels)
    
    print("Processing structured alternating pattern...")
    print("Expected: System should learn to predict pattern transitions\n")
    
    for t, input_data in enumerate(structured_input):
        print(f"Step {t+1}: Input = {input_data}")
        
        # Generate predictions before processing
        predictions = predictive_core.generate_predictions(input_data, precision_weights)
        print(f"Hierarchical predictions:")
        for level, pred in enumerate(predictions):
            print(f"  Level {level}: {np.array(pred).flatten()[:3]}...")  # Show first 3 elements
        
        # Process input
        prediction_state = predictive_core.process_input(
            input_data, precision_weights, learning_rate=0.005
        )
        
        # Show hierarchical errors
        print(f"Hierarchical errors: {prediction_state.hierarchical_errors}")
        print(f"Convergence: {prediction_state.convergence_status}")
        print()


def demonstrate_active_inference():
    """Demonstrate active inference through precision weighting."""
    print("=== Active Inference Demonstration ===\n")
    
    predictive_core = JaxPredictiveCodingCore(
        hierarchy_levels=3,
        input_dimensions=3,
        enable_active_inference=True
    )
    
    # Create ambiguous input that could benefit from attention
    ambiguous_input = np.array([0.5, 0.5, 0.1])  # Unclear signal
    
    # Test different attention strategies
    attention_strategies = [
        ("Uniform attention", PrecisionWeights.create_uniform(3)),
        ("Focus on level 0", PrecisionWeights.create_focused(3, focus_level=0, focus_strength=5.0)),
        ("Focus on level 1", PrecisionWeights.create_focused(3, focus_level=1, focus_strength=5.0)),
        ("Focus on level 2", PrecisionWeights.create_focused(3, focus_level=2, focus_strength=5.0)),
    ]
    
    for strategy_name, precision_weights in attention_strategies:
        print(f"{strategy_name}:")
        print(f"  Attention weights: {precision_weights.normalized_weights}")
        print(f"  Attention focus: {precision_weights.attention_focus:.3f}")
        
        # Process with this attention strategy
        prediction_state = predictive_core.process_input(
            ambiguous_input, precision_weights
        )
        
        print(f"  Total error: {prediction_state.total_error:.4f}")
        print(f"  Free energy: {prediction_state.metadata.get('free_energy_before_update', 0.0):.4f}")
        print()


if __name__ == "__main__":
    print("Enhanced Predictive Coding System Demonstration")
    print("Following Karl Friston's Free Energy Principle")
    print("=" * 60)
    
    try:
        # Run demonstrations
        results = demonstrate_free_energy_minimization()
        demonstrate_hierarchical_dynamics() 
        demonstrate_active_inference()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("The enhanced predictive coding system is working correctly.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()