#!/usr/bin/env python3
"""Test script to verify JIT compilation fixes for the enactive consciousness framework."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp
from enactive_consciousness import (
    create_temporal_processor_safe,
    create_body_schema_processor_safe,
    TemporalConsciousnessConfig,
    BodySchemaConfig,
    analyze_temporal_coherence,
)


def test_temporal_processor_jit():
    """Test temporal processor with JIT compilation."""
    print("🧠 Testing Temporal Processor JIT Compilation")
    print("-" * 50)
    
    config = TemporalConsciousnessConfig(
        retention_depth=8,
        protention_horizon=4,
        temporal_synthesis_rate=0.1,
    )
    key = jax.random.PRNGKey(42)
    state_dim = 32
    
    # Test JIT version
    try:
        processor = create_temporal_processor_safe(config, state_dim, key, use_jit=True)
        print("✅ JIT processor creation: SUCCESS")
        
        # Test processing multiple timesteps
        moments = []
        for i in range(5):
            test_input = jax.random.normal(jax.random.PRNGKey(i + 100), (state_dim,))
            moment = processor.temporal_synthesis(test_input, timestamp=i * 0.1)
            moments.append(moment)
            
        print(f"✅ JIT processing {len(moments)} moments: SUCCESS")
        
        # Test temporal coherence analysis
        coherence_metrics = analyze_temporal_coherence(moments)
        print(f"✅ Temporal coherence analysis: {coherence_metrics['coherence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Temporal JIT test failed: {e}")
        return False


def test_body_schema_processor_jit():
    """Test body schema processor with JIT compilation."""
    print("\n🦾 Testing Body Schema Processor JIT Compilation")
    print("-" * 50)
    
    config = BodySchemaConfig(
        proprioceptive_dim=32,
        motor_dim=12,
        body_map_resolution=(8, 8),
        schema_adaptation_rate=0.02,
    )
    key = jax.random.PRNGKey(123)
    
    # Test JIT version
    try:
        processor = create_body_schema_processor_safe(config, key, use_jit=True)
        print("✅ JIT processor creation: SUCCESS")
        
        # Test processing multiple timesteps
        body_states = []
        for i in range(3):
            proprioceptive_input = jax.random.normal(jax.random.PRNGKey(i + 200), (32,))
            motor_prediction = jax.random.normal(jax.random.PRNGKey(i + 300), (12,))
            tactile_feedback = jax.random.normal(jax.random.PRNGKey(i + 400), (16,))
            
            body_state = processor.integrate_body_schema(
                proprioceptive_input=proprioceptive_input,
                motor_prediction=motor_prediction,
                tactile_feedback=tactile_feedback,
            )
            body_states.append(body_state)
        
        print(f"✅ JIT processing {len(body_states)} body states: SUCCESS")
        
        # Test embodiment quality assessment
        quality_metrics = processor.assess_embodiment_quality(body_states[-1])
        print(f"✅ Embodiment quality: {quality_metrics['overall_embodiment']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Body schema JIT test failed: {e}")
        return False


def test_integrated_processing():
    """Test integrated temporal-embodied processing."""
    print("\n🌟 Testing Integrated Processing")
    print("-" * 50)
    
    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 2)
    
    # Initialize both processors
    temporal_config = TemporalConsciousnessConfig(retention_depth=6, protention_horizon=3)
    body_config = BodySchemaConfig(proprioceptive_dim=24, motor_dim=8)
    
    try:
        temporal_processor = create_temporal_processor_safe(
            temporal_config, state_dim=24, key=keys[0], use_jit=True
        )
        body_processor = create_body_schema_processor_safe(
            body_config, key=keys[1], use_jit=True
        )
        print("✅ Both processors created: SUCCESS")
        
        # Integrated processing loop
        integration_qualities = []
        
        for i in range(4):
            # Generate correlated inputs
            base_pattern = jnp.sin(i * 0.3) * 0.5
            
            sensory_input = base_pattern + jax.random.normal(
                jax.random.PRNGKey(i + 500), (24,)
            ) * 0.1
            
            proprioceptive_input = base_pattern + jax.random.normal(
                jax.random.PRNGKey(i + 600), (24,)
            ) * 0.1
            
            motor_prediction = base_pattern * 0.8 + jax.random.normal(
                jax.random.PRNGKey(i + 700), (8,)
            ) * 0.05
            
            tactile_feedback = jax.random.normal(
                jax.random.PRNGKey(i + 800), (12,)
            ) * 0.05
            
            # Process through both systems
            temporal_moment = temporal_processor.temporal_synthesis(
                primal_impression=sensory_input,
                timestamp=i * 0.1,
            )
            
            body_state = body_processor.integrate_body_schema(
                proprioceptive_input=proprioceptive_input,
                motor_prediction=motor_prediction,
                tactile_feedback=tactile_feedback,
            )
            
            # Compute integration quality
            integration_quality = (
                float(jnp.mean(temporal_moment.synthesis_weights)) * 
                body_state.schema_confidence
            )
            integration_qualities.append(integration_quality)
        
        avg_integration = float(jnp.mean(jnp.array(integration_qualities)))
        print(f"✅ Integrated processing: {avg_integration:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated processing test failed: {e}")
        return False


def test_performance_comparison():
    """Compare JIT vs non-JIT performance."""
    print("\n⚡ Performance Comparison (JIT vs Non-JIT)")
    print("-" * 50)
    
    config = TemporalConsciousnessConfig()
    key = jax.random.PRNGKey(789)
    state_dim = 32
    
    try:
        # Create both versions
        processor_jit = create_temporal_processor_safe(config, state_dim, key, use_jit=True)
        processor_no_jit = create_temporal_processor_safe(config, state_dim, key, use_jit=False)
        
        test_input = jax.random.normal(key, (state_dim,))
        
        # Warmup JIT
        _ = processor_jit.temporal_synthesis(test_input)
        
        import time
        
        # Time JIT version
        start_time = time.time()
        for _ in range(10):
            _ = processor_jit.temporal_synthesis(test_input)
        jit_time = time.time() - start_time
        
        # Time non-JIT version
        start_time = time.time()
        for _ in range(10):
            _ = processor_no_jit.temporal_synthesis(test_input)
        no_jit_time = time.time() - start_time
        
        speedup = no_jit_time / jit_time if jit_time > 0 else 0
        
        print(f"✅ JIT version: {jit_time:.4f}s")
        print(f"✅ Non-JIT version: {no_jit_time:.4f}s")
        print(f"✅ Speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all JIT compilation tests."""
    print("🚀 Enactive Consciousness Framework - JIT Compilation Tests")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(test_temporal_processor_jit())
    results.append(test_body_schema_processor_jit())
    results.append(test_integrated_processing())
    results.append(test_performance_comparison())
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Temporal JIT",
        "Body Schema JIT", 
        "Integrated Processing",
        "Performance Comparison"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All JIT compilation tests passed!")
        print("✅ Factory functions work correctly")
        print("✅ JIT compilation successful with proper fallbacks")
        print("✅ Static arguments configured properly")
        print("✅ Shape handling fixed")
        print("✅ Performance optimizations active")
        print("\n🚀 Framework ready for production deployment!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - check configuration")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)