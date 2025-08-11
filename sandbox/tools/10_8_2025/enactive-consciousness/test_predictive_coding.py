"""Simple test script for predictive coding functionality.

This script provides a quick validation of the predictive coding module
without requiring the full test suite infrastructure.
"""

import logging
import traceback
from typing import Dict, Any

import jax
import jax.numpy as jnp

# Configure logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports() -> Dict[str, bool]:
    """Test basic imports of predictive coding components."""
    
    results = {}
    
    try:
        from src.enactive_consciousness.predictive_coding import (
            PredictionScale,
            PredictiveCodingConfig,
            PredictiveState,
            HierarchicalPredictionNetwork,
            MultiScaleTemporalPredictor,
            DynamicErrorMinimization,
            IntegratedPredictiveCoding,
            create_predictive_coding_system,
        )
        results['imports'] = True
        logger.info("✓ Basic imports successful")
    except Exception as e:
        results['imports'] = False
        logger.error(f"✗ Import failed: {e}")
        traceback.print_exc()
    
    return results


def test_configuration_creation() -> Dict[str, bool]:
    """Test configuration object creation."""
    
    results = {}
    
    try:
        from src.enactive_consciousness.predictive_coding import PredictiveCodingConfig, PredictionScale
        
        # Test default configuration
        config = PredictiveCodingConfig()
        results['config_default'] = True
        logger.info("✓ Default configuration creation successful")
        
        # Test custom configuration
        custom_config = PredictiveCodingConfig(
            hierarchy_levels=3,
            prediction_horizon=8,
            temporal_scales=(PredictionScale.MICRO, PredictionScale.MESO),
            scale_weights=jnp.array([0.6, 0.4]),
        )
        results['config_custom'] = True
        logger.info("✓ Custom configuration creation successful")
        
        # Validate configuration values
        assert config.hierarchy_levels == 4
        assert len(custom_config.temporal_scales) == 2
        results['config_validation'] = True
        logger.info("✓ Configuration validation successful")
        
    except Exception as e:
        results['config_default'] = False
        results['config_custom'] = False
        results['config_validation'] = False
        logger.error(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
    
    return results


def test_hierarchical_network() -> Dict[str, bool]:
    """Test hierarchical prediction network creation and forward pass."""
    
    results = {}
    
    try:
        from src.enactive_consciousness.predictive_coding import HierarchicalPredictionNetwork
        
        # Create network
        key = jax.random.PRNGKey(42)
        input_dim = 32
        layer_dims = (24, 16, 8)
        
        network = HierarchicalPredictionNetwork(
            input_dim=input_dim,
            layer_dimensions=layer_dims,
            key=key,
            use_ngc=False  # Use fallback for testing
        )
        
        results['network_creation'] = True
        logger.info("✓ Hierarchical network creation successful")
        
        # Test forward pass
        test_input = jax.random.normal(key, (input_dim,))
        representations, errors = network.forward_prediction(test_input)
        
        # Validate output shapes
        assert len(representations) == len(layer_dims)
        assert representations[0].shape == (layer_dims[0],)
        assert len(errors) == len(layer_dims) - 1
        
        results['network_forward'] = True
        logger.info("✓ Network forward pass successful")
        
    except Exception as e:
        results['network_creation'] = False
        results['network_forward'] = False
        logger.error(f"✗ Hierarchical network test failed: {e}")
        traceback.print_exc()
    
    return results


def test_temporal_predictor() -> Dict[str, bool]:
    """Test multi-scale temporal predictor."""
    
    results = {}
    
    try:
        from src.enactive_consciousness.predictive_coding import MultiScaleTemporalPredictor, PredictiveCodingConfig
        from src.enactive_consciousness.temporal import TemporalConsciousnessConfig
        from src.enactive_consciousness.types import create_temporal_moment
        
        # Create configurations
        temporal_config = TemporalConsciousnessConfig()
        predictive_config = PredictiveCodingConfig()
        
        # Create predictor
        key = jax.random.PRNGKey(42)
        input_dim = 32
        
        predictor = MultiScaleTemporalPredictor(
            input_dim=input_dim,
            temporal_config=temporal_config,
            predictive_config=predictive_config,
            key=key
        )
        
        results['temporal_creation'] = True
        logger.info("✓ Temporal predictor creation successful")
        
        # Test prediction
        temporal_moment = create_temporal_moment(
            timestamp=0.0,
            retention=jax.random.normal(key, (input_dim,)),
            present_moment=jax.random.normal(key, (input_dim,)),
            protention=jax.random.normal(key, (input_dim,)),
            synthesis_weights=jax.random.normal(key, (input_dim,))
        )
        
        predictions = predictor.predict_temporal_dynamics(temporal_moment)
        
        # Validate predictions
        assert len(predictions) == len(predictive_config.temporal_scales)
        
        results['temporal_prediction'] = True
        logger.info("✓ Temporal prediction successful")
        
    except Exception as e:
        results['temporal_creation'] = False
        results['temporal_prediction'] = False
        logger.error(f"✗ Temporal predictor test failed: {e}")
        traceback.print_exc()
    
    return results


def test_integrated_system() -> Dict[str, bool]:
    """Test integrated predictive coding system."""
    
    results = {}
    
    try:
        from src.enactive_consciousness.predictive_coding import create_predictive_coding_system, PredictiveCodingConfig
        from src.enactive_consciousness.temporal import TemporalConsciousnessConfig  
        from src.enactive_consciousness.embodiment import BodySchemaConfig
        from src.enactive_consciousness.types import create_temporal_moment, BodyState
        
        # Create configurations
        predictive_config = PredictiveCodingConfig()
        temporal_config = TemporalConsciousnessConfig()
        body_config = BodySchemaConfig()
        
        # Create integrated system
        key = jax.random.PRNGKey(42)
        state_dim = 32
        
        system = create_predictive_coding_system(
            config=predictive_config,
            temporal_config=temporal_config,
            body_schema_config=body_config,
            state_dim=state_dim,
            key=key
        )
        
        results['system_creation'] = True
        logger.info("✓ Integrated system creation successful")
        
        # Test system processing
        keys = jax.random.split(key, 4)
        
        current_state = jax.random.normal(keys[0], (state_dim,))
        temporal_moment = create_temporal_moment(
            timestamp=0.0,
            retention=jax.random.normal(keys[1], (state_dim,)),
            present_moment=current_state,
            protention=jax.random.normal(keys[2], (state_dim,)),
            synthesis_weights=jax.random.normal(keys[3], (state_dim,))
        )
        
        body_state = BodyState(
            proprioception=jax.random.normal(keys[0], (64,)),
            motor_intention=jax.random.normal(keys[1], (32,)),
            boundary_signal=jnp.array([0.8]),
            schema_confidence=0.7
        )
        
        predictive_state = system.generate_hierarchical_predictions(
            current_state=current_state,
            temporal_moment=temporal_moment,
            body_state=body_state
        )
        
        # Validate predictive state
        assert hasattr(predictive_state, 'hierarchical_predictions')
        assert hasattr(predictive_state, 'total_prediction_error')
        assert hasattr(predictive_state, 'convergence_status')
        
        results['system_processing'] = True
        logger.info("✓ System processing successful")
        logger.info(f"  Total prediction error: {predictive_state.total_prediction_error:.4f}")
        logger.info(f"  Convergence status: {predictive_state.convergence_status}")
        
    except Exception as e:
        results['system_creation'] = False
        results['system_processing'] = False
        logger.error(f"✗ Integrated system test failed: {e}")
        traceback.print_exc()
    
    return results


def run_all_tests() -> None:
    """Run all basic tests for predictive coding functionality."""
    
    logger.info("="*60)
    logger.info("PREDICTIVE CODING MODULE TESTS")
    logger.info("="*60)
    
    all_results = {}
    
    # Run tests
    logger.info("Testing basic imports...")
    all_results.update(test_basic_imports())
    
    logger.info("\nTesting configuration creation...")
    all_results.update(test_configuration_creation())
    
    logger.info("\nTesting hierarchical network...")
    all_results.update(test_hierarchical_network())
    
    logger.info("\nTesting temporal predictor...")
    all_results.update(test_temporal_predictor())
    
    logger.info("\nTesting integrated system...")
    all_results.update(test_integrated_system())
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for result in all_results.values() if result)
    total = len(all_results)
    
    logger.info(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.warning("⚠️  Some tests failed. Check output above.")
        
        # Show failed tests
        failed_tests = [test for test, result in all_results.items() if not result]
        logger.warning(f"Failed tests: {', '.join(failed_tests)}")
    
    logger.info("="*60)


if __name__ == "__main__":
    run_all_tests()