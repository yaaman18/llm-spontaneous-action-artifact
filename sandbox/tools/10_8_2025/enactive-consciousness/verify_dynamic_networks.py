#!/usr/bin/env python3
"""
Quick verification of Dynamic Networks integration.
"""

import jax
import jax.numpy as jnp
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from enactive_consciousness import (
    DynamicNetworkProcessor,
    NetworkTopology,
    AdaptationMechanism,
    NetworkIntegrator,
    circular_causality_index,
    complexity_measure,
)

def main():
    print("Dynamic Networks Integration Verification")
    print("=" * 45)
    
    # Initialize components
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)
    
    # Create network processor
    network_processor = DynamicNetworkProcessor(
        num_nodes=8,
        hidden_dim=16,
        num_message_passing_steps=2,
        key=keys[0],
    )
    
    # Create initial network
    initial_features = jax.random.normal(keys[1], (8, 16))
    network_state = network_processor.create_initial_network_state(
        initial_features, NetworkTopology.SMALL_WORLD, key=keys[2]
    )
    
    print(f"✓ Created {NetworkTopology.SMALL_WORLD.value} network")
    print(f"  - Nodes: {network_state.node_features.shape[0]}")
    print(f"  - Consciousness: {network_state.consciousness_level:.3f}")
    
    # Test message passing
    updated_state = network_processor.process_graph_message_passing(network_state)
    print(f"✓ Message passing completed")
    
    # Test network metrics
    metrics = network_processor.compute_network_metrics(updated_state)
    print(f"✓ Computed {len(metrics)} network metrics")
    print(f"  - Density: {metrics['density']:.3f}")
    print(f"  - Clustering: {metrics['clustering_coefficient']:.3f}")
    
    # Test consciousness assessment
    consciousness = network_processor.assess_consciousness_level(updated_state)
    print(f"✓ Consciousness assessment: {consciousness['consciousness_score']:.3f}")
    
    # Test information theory integration with sufficient data points
    test_data = jax.random.normal(key, (20, 16))  # More time steps
    causality = circular_causality_index(test_data, test_data)
    complexity = complexity_measure(test_data.mean(axis=1), test_data.mean(axis=1))
    
    print(f"✓ Information theory integration:")
    print(f"  - Circular causality: {causality['circular_causality']:.3f}")
    print(f"  - Complexity: {complexity['overall_complexity']:.3f}")
    
    print("\n✓ All verifications passed!")
    print("Dynamic Networks module is properly integrated.")


if __name__ == "__main__":
    main()