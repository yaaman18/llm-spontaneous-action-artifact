#!/usr/bin/env python3
"""Basic Integration Test to Validate Core System Functionality.

This simplified test ensures the core integration test architecture works
before running the comprehensive test suite.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import jax
import jax.numpy as jnp

from enactive_consciousness import (
    create_framework_config,
    TemporalConsciousnessConfig,
    BodySchemaConfig,
)

from enactive_consciousness.integrated_consciousness import (
    create_enactive_consciousness_system,
    run_consciousness_sequence,
)


def test_basic_integration():
    """Test basic integration with properly configured dimensions."""
    
    print("🧪 Testing Basic Integration...")
    
    # Configure JAX
    jax.config.update('jax_platform_name', 'cpu')
    
    # Create key
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)
    
    # Use consistent dimensions throughout
    state_dim = 64
    proprioceptive_dim = 64  # Match state_dim
    motor_dim = 24
    environment_dim = 32
    context_dim = 48
    
    # Create configurations with matching dimensions
    config = create_framework_config(
        retention_depth=6,
        protention_horizon=3,
        consciousness_threshold=0.4,
        proprioceptive_dim=proprioceptive_dim,
        motor_dim=motor_dim,
    )
    
    temporal_config = TemporalConsciousnessConfig(
        retention_depth=6,
        protention_horizon=3,
        temporal_synthesis_rate=0.1,
        temporal_decay_factor=0.92,
    )
    
    body_config = BodySchemaConfig(
        proprioceptive_dim=proprioceptive_dim,
        motor_dim=motor_dim,
        body_map_resolution=(8, 8),  # Smaller for testing
        boundary_sensitivity=0.7,
        schema_adaptation_rate=0.015,
    )
    
    print(f"   State dim: {state_dim}")
    print(f"   Proprioceptive dim: {proprioceptive_dim}")
    print(f"   Motor dim: {motor_dim}")
    print(f"   Environment dim: {environment_dim}")
    
    # Create consciousness system
    print("   Creating consciousness system...")
    
    system = create_enactive_consciousness_system(
        config=config,
        temporal_config=temporal_config,
        body_config=body_config,
        state_dim=state_dim,
        environment_dim=environment_dim,
        key=keys[0],
    )
    
    print("   ✅ System created successfully")
    
    # Create test sequence
    sequence_length = 5
    test_sequence = []
    
    for t in range(sequence_length):
        # Create properly dimensioned inputs
        test_input = {
            'sensory_input': jax.random.normal(keys[1 + t], (state_dim,)) * 0.5,
            'proprioceptive_input': jax.random.normal(keys[1 + t], (proprioceptive_dim,)) * 0.4,
            'motor_prediction': jax.random.normal(keys[1 + t], (motor_dim,)) * 0.3,
            'environmental_state': jax.random.normal(keys[1 + t], (environment_dim,)) * 0.4,
            'contextual_cues': jax.random.normal(keys[1 + t], (context_dim,)) * 0.3,
        }
        test_sequence.append(test_input)
    
    print(f"   Generated test sequence with {sequence_length} steps")
    
    # Test single consciousness moment
    print("   Testing single consciousness moment...")
    
    consciousness_state = system.integrate_conscious_moment(
        **test_sequence[0], timestamp=1.0
    )
    
    print(f"   ✅ Single moment integration successful")
    print(f"      Consciousness level: {consciousness_state.consciousness_level:.3f}")
    print(f"      Integration coherence: {consciousness_state.integration_coherence:.3f}")
    print(f"      Circular causality: {consciousness_state.circular_causality_strength:.3f}")
    
    # Test sequence processing
    print("   Testing sequence processing...")
    
    consciousness_states = run_consciousness_sequence(system, test_sequence)
    
    print(f"   ✅ Sequence processing successful")
    print(f"      Processed {len(consciousness_states)} states")
    
    # Validate results
    assert len(consciousness_states) == sequence_length
    
    for i, state in enumerate(consciousness_states):
        assert hasattr(state, 'consciousness_level')
        assert hasattr(state, 'integration_coherence')
        assert hasattr(state, 'circular_causality_strength')
        assert 0.0 <= state.consciousness_level <= 1.0
        assert 0.0 <= state.integration_coherence <= 1.0
        assert 0.0 <= state.circular_causality_strength <= 1.0
    
    # Test system performance metrics
    print("   Testing performance metrics...")
    
    performance_metrics = system.compute_performance_metrics(
        consciousness_states, 100.0, 50.0
    )
    
    print(f"   ✅ Performance metrics computed")
    print(f"      Temporal coherence: {performance_metrics.temporal_coherence:.3f}")
    print(f"      Embodiment stability: {performance_metrics.embodiment_stability:.3f}")
    print(f"      Coupling effectiveness: {performance_metrics.coupling_effectiveness:.3f}")
    print(f"      Overall score: {performance_metrics.overall_consciousness_score:.3f}")
    
    # Compute basic validation score
    base_score = (
        0.25 * performance_metrics.temporal_coherence +
        0.25 * performance_metrics.embodiment_stability +
        0.25 * performance_metrics.coupling_effectiveness +
        0.25 * performance_metrics.overall_consciousness_score
    )
    
    print(f"\n🎯 Basic Integration Test Results:")
    print(f"   Base validation score: {base_score:.3f}")
    print(f"   Test sequence length: {len(consciousness_states)}")
    print(f"   All dimensions consistent: ✅")
    print(f"   Core functionality working: ✅")
    
    return {
        'success': True,
        'base_score': float(base_score),
        'sequence_length': len(consciousness_states),
        'performance_metrics': performance_metrics,
        'consciousness_states': consciousness_states,
    }


def test_module_availability():
    """Test availability of advanced modules."""
    
    print("\n🔍 Testing Module Availability...")
    
    module_status = {}
    
    # Test core modules
    core_modules = ['jax', 'equinox', 'numpy']
    for module in core_modules:
        try:
            __import__(module)
            module_status[module] = True
            print(f"   ✅ {module}")
        except ImportError:
            module_status[module] = False
            print(f"   ❌ {module}")
    
    # Test advanced modules
    advanced_modules = [
        'information_theory',
        'dynamic_networks', 
        'sparse_representations',
        'predictive_coding',
        'continuous_dynamics',
    ]
    
    for module in advanced_modules:
        try:
            __import__(f'enactive_consciousness.{module}')
            module_status[f'enactive_consciousness.{module}'] = True
            print(f"   ✅ enactive_consciousness.{module}")
        except ImportError:
            module_status[f'enactive_consciousness.{module}'] = False
            print(f"   ❌ enactive_consciousness.{module}")
    
    available_count = sum(1 for available in module_status.values() if available)
    total_count = len(module_status)
    
    print(f"\n   Module availability: {available_count}/{total_count}")
    
    return module_status


if __name__ == '__main__':
    """Run basic integration test."""
    
    print("🚀 Basic Integration Test for Enactive Consciousness Framework")
    print("=" * 70)
    
    try:
        # Test module availability first
        module_status = test_module_availability()
        
        # Run basic integration test
        results = test_basic_integration()
        
        print(f"\n✅ Basic Integration Test Completed Successfully!")
        print(f"   Validation Score: {results['base_score']:.3f}")
        print(f"   System Ready for Comprehensive Testing: ✅")
        
        # Check if comprehensive tests can be run
        core_available = all(module_status[mod] for mod in ['jax', 'equinox', 'numpy'])
        
        if core_available:
            print(f"\n🎯 Integration Test Suite Status:")
            print(f"   Core functionality: ✅ Ready")
            print(f"   Basic integration: ✅ Working") 
            print(f"   Comprehensive tests: ✅ Can be run")
            print(f"   Performance tests: ✅ Can be run")
        else:
            print(f"\n⚠️  Some core modules missing - limited functionality")
        
        print(f"\n🧠 Next Steps:")
        print(f"   1. Run comprehensive integration tests:")
        print(f"      python run_integration_tests.py --suite basic")
        print(f"   2. Run with coverage:")
        print(f"      python run_integration_tests.py --coverage")
        print(f"   3. Run performance tests:")
        print(f"      python run_integration_tests.py --suite performance")
        
    except Exception as e:
        print(f"\n❌ Basic Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)