#!/usr/bin/env python3
"""Debug script to isolate the unpacking error."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from enactive_consciousness.continuous_dynamics import (
        HusserlianTemporalFlow,
        DynamicsConfig,
        create_default_dynamics_config,
    )
    
    # Test just the temporal flow in isolation
    config = create_default_dynamics_config()
    key = jax.random.PRNGKey(42)
    
    temporal_flow = HusserlianTemporalFlow(
        state_dim=8,  # Small dimension for testing
        temporal_depth=5,
        config=config,
        key=key,
    )
    
    logger.info("Created HusserlianTemporalFlow")
    
    # Create a test temporal state
    state_dim = 8
    temporal_state = jax.random.normal(key, (state_dim * 4,))  # 4 components
    environmental_input = jax.random.normal(jax.random.split(key)[0], (state_dim * 4,))
    
    logger.info(f"Temporal state shape: {temporal_state.shape}")
    logger.info(f"Environmental input shape: {environmental_input.shape}")
    
    # Test the temporal flow equations
    try:
        result = temporal_flow.temporal_flow_equations(0.0, temporal_state, environmental_input)
        logger.info(f"Temporal flow result shape: {result.shape}")
        logger.info("✓ Temporal flow equations work!")
    except Exception as e:
        logger.error(f"✗ Temporal flow equations failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    logger.error(f"Import or setup failed: {e}")
    import traceback
    traceback.print_exc()