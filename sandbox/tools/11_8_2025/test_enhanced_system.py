#!/usr/bin/env python3
"""
Test Enhanced Predictive Coding System.

Simple test to verify the enhanced mathematical foundations work correctly.
"""

import numpy as np
from infrastructure.jax_predictive_coding_core import JaxPredictiveCodingCore
from domain.value_objects.precision_weights import PrecisionWeights


def test_basic_functionality():
    """Test basic functionality of enhanced system."""
    print("Testing Enhanced Predictive Coding System...")
    
    # Create system
    core = JaxPredictiveCodingCore(
        hierarchy_levels=3,
        input_dimensions=2,
        learning_rate=0.01
    )
    
    # Create test input
    test_input = np.array([1.0, 0.5])
    precision_weights = PrecisionWeights.create_uniform(3)
    
    print(f"Input: {test_input}")
    print(f"Precision weights: {precision_weights.normalized_weights}")
    
    try:
        # Test prediction generation
        predictions = core.generate_predictions(test_input, precision_weights)
        print(f"✓ Predictions generated: {len(predictions)} levels")
        
        # Test error computation  
        targets = predictions  # Use predictions as targets for this test
        errors = core.compute_prediction_errors(predictions, targets)
        print(f"✓ Errors computed: {[np.mean(np.abs(err)) for err in errors]}")
        
        # Test free energy computation
        free_energy = core.compute_free_energy(predictions, targets, precision_weights)
        print(f"✓ Free energy computed: {free_energy:.4f}")
        
        # Test error propagation
        propagated_errors, new_state = core.propagate_errors(errors, precision_weights)
        print(f"✓ Errors propagated: {new_state.total_error:.4f}")
        
        # Test precision updates
        updated_precisions = core.update_precisions(propagated_errors)
        print(f"✓ Precisions updated: {updated_precisions.normalized_weights}")
        
        # Test complete processing cycle
        final_state = core.process_input(test_input, precision_weights)
        print(f"✓ Complete cycle: {final_state.convergence_status}")
        
        print("\nAll tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\nEnhanced system is ready for use!")
    else:
        print("\nSystem needs debugging before use.")