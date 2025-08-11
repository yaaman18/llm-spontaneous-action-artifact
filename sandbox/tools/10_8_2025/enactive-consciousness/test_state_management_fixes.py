#!/usr/bin/env python3
"""
Test script to validate state management fixes in experiential_memory.py

This script tests the key fixes implemented to resolve SOLID and Clean Architecture violations:
1. Proper immutable state threading using eqx.tree_at
2. Extract Method refactoring for complex operations
3. Elimination of direct mutation patterns
4. Proper Dependency Inversion compliance
"""

import jax
import jax.numpy as jnp
import sys
import os

# Add the source directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from enactive_consciousness.experiential_memory import (
        CircularCausalityEngine,
        ExperientialSedimentation,
        IntegratedExperientialMemory,
        create_experiential_memory_system
    )
    print("✅ Successfully imported experiential_memory modules")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_circular_causality_immutability():
    """Test CircularCausalityEngine immutable state management."""
    print("\n🧪 Testing CircularCausalityEngine immutability...")
    
    key = jax.random.PRNGKey(42)
    state_dim = 16
    env_dim = 8
    hidden_dim = 32
    
    # Create engine
    engine = CircularCausalityEngine(
        state_dim=state_dim,
        environment_dim=env_dim, 
        hidden_dim=hidden_dim,
        key=key
    )
    
    # Test state and environment inputs
    current_state = jax.random.normal(key, (state_dim,))
    environmental_input = jax.random.normal(key, (env_dim,))
    
    # Store original connectivity for comparison
    original_connectivity = engine.network_connectivity.copy()
    original_buffer = engine.history_buffer.copy()
    
    try:
        # Execute circular causality step - should return updated engine
        updated_engine, next_state, emergent_meaning, metrics = engine.circular_causality_step(
            current_state, environmental_input, step_count=0
        )
        
        # Verify original engine is unchanged (immutability)
        assert jnp.allclose(engine.network_connectivity, original_connectivity), \
            "❌ Original engine connectivity was mutated!"
        assert jnp.allclose(engine.history_buffer, original_buffer), \
            "❌ Original engine history buffer was mutated!"
        
        # Verify updated engine has changes
        connectivity_changed = not jnp.allclose(
            updated_engine.network_connectivity, original_connectivity
        )
        buffer_changed = not jnp.allclose(
            updated_engine.history_buffer, original_buffer
        )
        
        assert connectivity_changed or buffer_changed, \
            "❌ Updated engine should have state changes!"
        
        # Verify output shapes and types
        assert next_state.shape == (state_dim,), f"❌ Wrong next_state shape: {next_state.shape}"
        assert emergent_meaning.shape == (state_dim,), f"❌ Wrong emergent_meaning shape: {emergent_meaning.shape}"
        assert isinstance(metrics, dict), "❌ Metrics should be a dictionary"
        assert 'circular_coherence' in metrics, "❌ Missing required metric"
        
        print("✅ CircularCausalityEngine immutability test passed!")
        return True
        
    except Exception as e:
        print(f"❌ CircularCausalityEngine test failed: {e}")
        return False


def test_sedimentation_immutability():
    """Test ExperientialSedimentation immutable state management."""
    print("\n🧪 Testing ExperientialSedimentation immutability...")
    
    key = jax.random.PRNGKey(123)
    experience_dim = 16
    num_layers = 5
    
    # Create sedimentation system
    sedimentation = ExperientialSedimentation(
        experience_dim=experience_dim,
        num_layers=num_layers,
        key=key
    )
    
    # Test inputs
    new_experience = jax.random.normal(key, (experience_dim,))
    significance_weight = 0.7
    temporal_context = jax.random.normal(key, (experience_dim,))
    
    # Store original state for comparison
    original_layers = sedimentation.sediment_layers.copy()
    original_significance = sedimentation.significance_tracker.copy()
    original_codes = sedimentation.sparse_codes.copy()
    
    try:
        # Execute sedimentation - should return updated instance
        updated_sedimentation = sedimentation.sediment_experience(
            new_experience, significance_weight, temporal_context
        )
        
        # Verify original instance is unchanged (immutability)
        assert jnp.allclose(sedimentation.sediment_layers, original_layers), \
            "❌ Original sedimentation layers were mutated!"
        assert jnp.allclose(sedimentation.significance_tracker, original_significance), \
            "❌ Original significance tracker was mutated!"
        assert jnp.allclose(sedimentation.sparse_codes, original_codes), \
            "❌ Original sparse codes were mutated!"
        
        # Verify updated instance has changes
        layers_changed = not jnp.allclose(
            updated_sedimentation.sediment_layers, original_layers
        )
        significance_changed = not jnp.allclose(
            updated_sedimentation.significance_tracker, original_significance
        )
        
        assert layers_changed or significance_changed, \
            "❌ Updated sedimentation should have state changes!"
        
        # Verify shapes are preserved
        assert updated_sedimentation.sediment_layers.shape == (num_layers, experience_dim), \
            f"❌ Wrong sediment layers shape: {updated_sedimentation.sediment_layers.shape}"
        assert updated_sedimentation.significance_tracker.shape == (num_layers,), \
            f"❌ Wrong significance tracker shape: {updated_sedimentation.significance_tracker.shape}"
        
        print("✅ ExperientialSedimentation immutability test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ExperientialSedimentation test failed: {e}")
        return False


def test_integrated_memory_system():
    """Test IntegratedExperientialMemory state threading."""
    print("\n🧪 Testing IntegratedExperientialMemory state threading...")
    
    key = jax.random.PRNGKey(456)
    experience_dim = 12
    environment_dim = 6
    context_dim = 8
    
    try:
        # Create integrated memory system
        memory_system = create_experiential_memory_system(
            experience_dim=experience_dim,
            environment_dim=environment_dim,
            context_dim=context_dim,
            key=key
        )
        
        # Test inputs
        current_experience = jax.random.normal(key, (experience_dim,))
        environmental_input = jax.random.normal(key, (environment_dim,))
        contextual_cues = jax.random.normal(key, (context_dim,))
        
        # Store original state
        original_engine = memory_system.circular_engine
        original_sedimentation = memory_system.sedimentation
        original_traces_count = len(memory_system.experience_traces)
        
        # Process experiential moment
        updated_memory, integrated_experience, metadata = memory_system.process_experiential_moment(
            current_experience, environmental_input, contextual_cues
        )
        
        # Verify original memory system is unchanged
        assert memory_system.circular_engine is original_engine, \
            "❌ Original memory system engine reference changed!"
        assert memory_system.sedimentation is original_sedimentation, \
            "❌ Original memory system sedimentation reference changed!"
        assert len(memory_system.experience_traces) == original_traces_count, \
            "❌ Original memory system traces were mutated!"
        
        # Verify updated system has changes
        engine_updated = updated_memory.circular_engine is not original_engine
        sedimentation_updated = updated_memory.sedimentation is not original_sedimentation
        traces_updated = len(updated_memory.experience_traces) > original_traces_count
        
        assert engine_updated or sedimentation_updated or traces_updated, \
            "❌ Updated memory system should have state changes!"
        
        # Verify output structure
        assert integrated_experience.shape == (experience_dim,), \
            f"❌ Wrong integrated experience shape: {integrated_experience.shape}"
        assert isinstance(metadata, dict), "❌ Metadata should be a dictionary"
        assert 'circular_causality' in metadata, "❌ Missing circular causality metadata"
        
        print("✅ IntegratedExperientialMemory test passed!")
        return True
        
    except Exception as e:
        print(f"❌ IntegratedExperientialMemory test failed: {e}")
        return False


def test_extract_method_refactoring():
    """Test Extract Method pattern implementation."""
    print("\n🧪 Testing Extract Method refactoring...")
    
    key = jax.random.PRNGKey(789)
    state_dim = 10
    env_dim = 5
    hidden_dim = 20
    
    try:
        engine = CircularCausalityEngine(
            state_dim=state_dim,
            environment_dim=env_dim,
            hidden_dim=hidden_dim,
            key=key
        )
        
        current_state = jax.random.normal(key, (state_dim,))
        environmental_input = jax.random.normal(key, (env_dim,))
        
        # Test individual extracted methods
        processing_state = engine._execute_core_processing(
            current_state, environmental_input, None
        )
        
        assert isinstance(processing_state, dict), "❌ Core processing should return dict"
        assert 'self_referenced' in processing_state, "❌ Missing self_referenced in processing state"
        assert 'environment_coupled' in processing_state, "❌ Missing environment_coupled in processing state"
        
        # Test information theory integration
        info_state, info_metrics = engine._integrate_information_theory(
            current_state, environmental_input, 0
        )
        
        assert info_state.shape == (state_dim,), f"❌ Wrong info state shape: {info_state.shape}"
        assert isinstance(info_metrics, dict), "❌ Info metrics should be dict"
        
        # Test network processing
        updated_engine, network_state, network_features = engine._process_dynamic_networks(
            current_state, environmental_input
        )
        
        assert network_state.shape == (state_dim,), f"❌ Wrong network state shape: {network_state.shape}"
        assert updated_engine is not engine, "❌ Network processing should return new engine instance"
        
        print("✅ Extract Method refactoring test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Extract Method test failed: {e}")
        return False


def main():
    """Run all state management tests."""
    print("🚀 Starting State Management Fixes Validation")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(test_circular_causality_immutability())
    test_results.append(test_sedimentation_immutability())
    test_results.append(test_integrated_memory_system())
    test_results.append(test_extract_method_refactoring())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = sum(test_results)
    total = len(test_results)
    
    if passed == total:
        print(f"🎉 All {total} tests PASSED! State management fixes are working correctly.")
        return 0
    else:
        print(f"⚠️  {passed}/{total} tests passed. {total - passed} tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())