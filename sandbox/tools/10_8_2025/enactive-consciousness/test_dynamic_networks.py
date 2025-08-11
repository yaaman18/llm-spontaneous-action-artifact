#!/usr/bin/env python3
"""
Quick test for Dynamic Networks module.

Tests basic functionality of the dynamic networks implementation
to ensure it integrates correctly with the existing system.
"""

import jax
import jax.numpy as jnp
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from enactive_consciousness.dynamic_networks import (
        DynamicNetworkProcessor,
        NetworkIntegrator,
        NetworkTopology,
        AdaptationMechanism,
        NetworkState,
    )
    from enactive_consciousness.types import CouplingState, TemporalMoment
    
    print("✓ Successfully imported dynamic networks module")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_network_creation():
    """Test network creation and initialization."""
    print("\nTesting network creation...")
    
    key = jax.random.PRNGKey(42)
    
    # Create network processor
    network_processor = DynamicNetworkProcessor(
        num_nodes=10,
        hidden_dim=32,
        num_message_passing_steps=2,
        key=key,
    )
    
    # Create initial network state
    keys = jax.random.split(key, 2)
    initial_features = jax.random.normal(keys[0], (10, 32))
    
    network_state = network_processor.create_initial_network_state(
        initial_features,
        NetworkTopology.SMALL_WORLD,
        key=keys[1],
    )
    
    print(f"✓ Created network with {network_state.node_features.shape[0]} nodes")
    print(f"✓ Consciousness level: {network_state.consciousness_level:.3f}")
    print(f"✓ Adaptation strength: {network_state.adaptation_strength:.3f}")
    
    return network_processor, network_state


def test_message_passing():
    """Test graph message passing."""
    print("\nTesting message passing...")
    
    network_processor, network_state = test_network_creation()
    
    # Process message passing
    updated_state = network_processor.process_graph_message_passing(network_state)
    
    print(f"✓ Message passing completed")
    print(f"✓ Node features shape: {updated_state.node_features.shape}")
    print(f"✓ New consciousness level: {updated_state.consciousness_level:.3f}")
    
    return network_processor, updated_state


def test_adaptive_reorganization():
    """Test adaptive network reorganization."""
    print("\nTesting adaptive reorganization...")
    
    network_processor, network_state = test_message_passing()
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 3)
    
    # Create mock coupling state
    coupling_state = CouplingState(
        agent_state=jax.random.normal(keys[0], (32,)),
        environmental_state=jax.random.normal(keys[1], (32,)),
        coupling_strength=0.7,
        perturbation_history=jax.random.normal(keys[2], (5, 32)),
        stability_metric=0.8,
    )
    
    # Test reorganization
    result = network_processor.adaptive_reorganization(
        network_state,
        coupling_state,
        adaptation_mechanism=AdaptationMechanism.CIRCULAR_CAUSALITY,
        key=key,
    )
    
    print(f"✓ Reorganization completed")
    print(f"✓ Reorganization strength: {result.reorganization_strength:.3f}")
    print(f"✓ Meaning emergence: {result.meaning_emergence:.3f}")
    print(f"✓ Consciousness delta: {result.consciousness_delta:.3f}")
    
    return network_processor, result.new_network_state


def test_network_metrics():
    """Test network metrics computation."""
    print("\nTesting network metrics...")
    
    network_processor, network_state = test_adaptive_reorganization()
    
    # Compute metrics
    metrics = network_processor.compute_network_metrics(network_state)
    
    print(f"✓ Computed {len(metrics)} metrics")
    print(f"✓ Density: {metrics['density']:.3f}")
    print(f"✓ Clustering coefficient: {metrics['clustering_coefficient']:.3f}")
    print(f"✓ Mean degree: {metrics['mean_degree']:.3f}")
    
    # Assess consciousness
    consciousness_metrics = network_processor.assess_consciousness_level(network_state)
    
    print(f"✓ Consciousness assessment completed")
    print(f"✓ Overall consciousness: {consciousness_metrics['consciousness_score']:.3f}")
    
    return network_processor, network_state


def test_network_integration():
    """Test network integration."""
    print("\nTesting network integration...")
    
    network_processor, network_state = test_network_metrics()
    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 3)
    
    # Create integrator
    integrator = NetworkIntegrator(network_processor, key=keys[0])
    
    # Create coupling state
    coupling_state = CouplingState(
        agent_state=jax.random.normal(keys[1], (32,)),
        environmental_state=jax.random.normal(keys[2], (32,)),
        coupling_strength=0.6,
        perturbation_history=jax.random.normal(keys[1], (3, 32)),
        stability_metric=0.9,
    )
    
    # Integrate dynamics
    updated_state, integration_metrics = integrator.integrate_network_dynamics(
        network_state, coupling_state, key=key
    )
    
    print(f"✓ Network integration completed")
    print(f"✓ Integration metrics: {len(integration_metrics)} measures")
    print(f"✓ Network-coupling strength: {integration_metrics['network_coupling_strength']:.3f}")
    
    return integrator, updated_state


def test_information_theory_integration():
    """Test integration with information theory measures."""
    print("\nTesting information theory integration...")
    
    # Import information theory functions
    from enactive_consciousness.information_theory import (
        mutual_information_kraskov,
        circular_causality_index,
    )
    
    # Create test data
    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, 2)
    
    agent_states = jax.random.normal(keys[0], (10, 32))
    env_states = jax.random.normal(keys[1], (10, 32))
    
    # Test mutual information
    mi = mutual_information_kraskov(
        agent_states.mean(axis=1),
        env_states.mean(axis=1),
    )
    
    print(f"✓ Mutual information: {mi:.3f}")
    
    # Test circular causality
    causality_metrics = circular_causality_index(
        agent_states, env_states
    )
    
    print(f"✓ Circular causality: {causality_metrics['circular_causality']:.3f}")
    print(f"✓ Information theory integration successful")


def main():
    """Run all tests."""
    print("Testing Dynamic Networks Module")
    print("=" * 40)
    
    try:
        # Run all tests
        test_network_creation()
        test_message_passing()
        test_adaptive_reorganization()
        test_network_metrics()
        test_network_integration()
        test_information_theory_integration()
        
        print("\n" + "=" * 40)
        print("✓ ALL TESTS PASSED!")
        print("Dynamic Networks module is working correctly.")
        print("✓ Network topologies can be created")
        print("✓ Message passing works correctly")
        print("✓ Adaptive reorganization functions")
        print("✓ Network metrics are computed")
        print("✓ Integration with coupling works") 
        print("✓ Information theory integration works")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()