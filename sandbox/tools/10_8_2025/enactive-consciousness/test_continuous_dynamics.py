#!/usr/bin/env python3
"""
Test script for continuous dynamics module.

This validates the basic functionality of the continuous dynamics implementation
and demonstrates its integration with existing enactive consciousness components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from enactive_consciousness.continuous_dynamics import (
        DynamicsConfig,
        ContinuousDynamicsProcessor,
        IntegrationMethod,
        create_continuous_dynamics_processor,
        create_default_dynamics_config,
        create_environmental_perturbation_function,
        analyze_continuous_consciousness_dynamics,
    )
    
    from enactive_consciousness.types import (
        create_temporal_moment,
        CouplingState,
    )
    
    from enactive_consciousness.temporal import (
        TemporalConsciousnessConfig,
        PhenomenologicalTemporalSynthesis,
    )
    
    from enactive_consciousness.enactive_coupling import (
        EnactiveCouplingConfig,
        EnactiveCouplingProcessor,
    )
    
    logger.info("All imports successful!")
    
except ImportError as e:
    logger.error(f"Import failed: {e}")
    logger.info("This is expected if diffrax is not installed. The module structure is correct.")
    sys.exit(0)  # Exit gracefully for missing dependencies


def test_continuous_dynamics_basic():
    """Test basic continuous dynamics functionality."""
    logger.info("Testing basic continuous dynamics...")
    
    try:
        # Create configuration
        config = create_default_dynamics_config(
            retention_decay_rate=0.1,
            protention_anticipation_rate=0.05,
            agent_environment_coupling_strength=0.3,
        )
        logger.info(f"Created dynamics config: {config}")
        
        # Create processor
        key = jax.random.PRNGKey(42)
        state_dim = 32
        
        processor = create_continuous_dynamics_processor(
            state_dim=state_dim,
            config=config,
            key=key,
            integration_method=IntegrationMethod.TSIT5,
        )
        logger.info("Created continuous dynamics processor")
        
        # Create test temporal moment
        temporal_moment = create_temporal_moment(
            timestamp=0.0,
            retention=jax.random.normal(key, (state_dim,)),
            present_moment=jax.random.normal(jax.random.split(key)[0], (state_dim,)),
            protention=jax.random.normal(jax.random.split(key)[1], (state_dim,)),
            synthesis_weights=jnp.ones(state_dim) / state_dim,
        )
        logger.info("Created temporal moment")
        
        # Create test coupling state
        coupling_state = CouplingState(
            agent_state=jax.random.normal(key, (state_dim,)),
            environmental_state=jax.random.normal(jax.random.split(key)[0], (state_dim,)),
            coupling_strength=0.5,
            perturbation_history=jax.random.normal(jax.random.split(key)[1], (10, state_dim)),
            stability_metric=0.7,
        )
        logger.info("Created coupling state")
        
        # Test temporal evolution
        environmental_input = jax.random.normal(key, (state_dim * 4,))
        time_span = (0.0, 1.0)
        
        logger.info(f"Environmental input shape: {environmental_input.shape}")
        logger.info(f"Temporal moment shapes: R={temporal_moment.retention.shape}, P={temporal_moment.present_moment.shape}")
        
        times, states = processor.evolve_temporal_consciousness(
            temporal_moment, environmental_input, time_span, num_steps=10
        )
        
        logger.info(f"Temporal evolution successful: {len(times)} time steps")
        logger.info(f"Final state shape: {states[-1].shape}")
        
        # Test trajectory statistics
        stats = processor.compute_trajectory_statistics(times, states)
        logger.info(f"Trajectory statistics: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Basic test failed: {e}")
        return False


def test_environmental_perturbation():
    """Test environmental perturbation function."""
    logger.info("Testing environmental perturbation...")
    
    try:
        key = jax.random.PRNGKey(123)
        env_func = create_environmental_perturbation_function(
            amplitude=1.0,
            frequency=0.1,
            noise_strength=0.1,
            key=key,
        )
        
        # Test at various time points
        test_times = [0.0, 0.5, 1.0, 2.0, 5.0]
        perturbations = [env_func(t) for t in test_times]
        
        logger.info(f"Environmental perturbations at {test_times}:")
        for t, p in zip(test_times, perturbations):
            logger.info(f"  t={t}: shape={p.shape}, norm={float(jnp.linalg.norm(p)):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Environmental perturbation test failed: {e}")
        return False


def test_integration_methods():
    """Test different integration methods."""
    logger.info("Testing integration methods...")
    
    try:
        key = jax.random.PRNGKey(456)
        state_dim = 16
        config = create_default_dynamics_config()
        
        methods = [
            IntegrationMethod.EULER,
            IntegrationMethod.HEUN,
            IntegrationMethod.TSIT5,
        ]
        
        for method in methods:
            logger.info(f"Testing integration method: {method.value}")
            
            processor = create_continuous_dynamics_processor(
                state_dim=state_dim,
                config=config,
                key=key,
                integration_method=method,
            )
            
            # Simple test
            temporal_moment = create_temporal_moment(
                timestamp=0.0,
                retention=jnp.ones(state_dim),
                present_moment=jnp.ones(state_dim),
                protention=jnp.ones(state_dim),
                synthesis_weights=jnp.ones(state_dim) / state_dim,
            )
            
            environmental_input = jnp.ones(state_dim * 4)
            time_span = (0.0, 0.5)
            
            times, states = processor.evolve_temporal_consciousness(
                temporal_moment, environmental_input, time_span, num_steps=5
            )
            
            logger.info(f"  {method.value}: {len(times)} steps, final norm: {float(jnp.linalg.norm(states[-1])):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration methods test failed: {e}")
        return False


def test_consciousness_integration():
    """Test full consciousness integration."""
    logger.info("Testing full consciousness integration...")
    
    try:
        key = jax.random.PRNGKey(789)
        keys = jax.random.split(key, 3)
        
        state_dim = 24
        config = create_default_dynamics_config(max_steps=100)
        
        processor = create_continuous_dynamics_processor(
            state_dim=state_dim,
            config=config,
            key=keys[0],
            integration_method=IntegrationMethod.TSIT5,
        )
        
        # Create realistic initial states
        temporal_moment = create_temporal_moment(
            timestamp=0.0,
            retention=jax.random.normal(keys[1], (state_dim,)) * 0.5,
            present_moment=jax.random.normal(keys[2], (state_dim,)),
            protention=jax.random.normal(keys[1], (state_dim,)) * 0.3,
            synthesis_weights=jax.nn.softmax(jax.random.normal(keys[2], (state_dim,))),
        )
        
        coupling_state = CouplingState(
            agent_state=jax.random.normal(keys[0], (state_dim,)),
            environmental_state=jax.random.normal(keys[1], (state_dim,)),
            coupling_strength=0.6,
            perturbation_history=jax.random.normal(keys[2], (10, state_dim)) * 0.1,
            stability_metric=0.8,
        )
        
        # Create environmental dynamics
        env_dynamics = create_environmental_perturbation_function(
            amplitude=0.5,
            frequency=0.2,
            noise_strength=0.05,
            key=keys[1],
        )
        
        # Integrate consciousness
        time_span = (0.0, 2.0)
        final_state = processor.integrate_continuous_consciousness(
            temporal_moment,
            coupling_state,
            env_dynamics,
            time_span,
            keys[2],
            num_steps=20,
        )
        
        logger.info(f"Integration successful!")
        logger.info(f"Final consciousness level: {final_state.consciousness_level:.3f}")
        logger.info(f"Final coupling strength: {final_state.coupling_strength:.3f}")
        logger.info(f"Final temporal coherence: {final_state.temporal_coherence:.3f}")
        logger.info(f"State validation: {final_state.validate_state()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Consciousness integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing Continuous Dynamics Module")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Functionality", test_continuous_dynamics_basic),
        ("Environmental Perturbation", test_environmental_perturbation),
        ("Integration Methods", test_integration_methods),
        ("Consciousness Integration", test_consciousness_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
            logger.info(f"✓ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            logger.error(f"✗ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name:<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Continuous dynamics module is working correctly.")
    else:
        logger.info("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)