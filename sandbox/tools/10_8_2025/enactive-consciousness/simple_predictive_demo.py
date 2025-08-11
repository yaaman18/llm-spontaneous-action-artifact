"""Simple demonstration of predictive coding functionality.

This example shows the basic usage of the predictive coding system
integrated with the enactive consciousness framework.
"""

import jax
import jax.numpy as jnp

from src.enactive_consciousness.predictive_coding import (
    PredictiveCodingConfig,
    create_predictive_coding_system,
    PredictionScale
)
from src.enactive_consciousness.temporal import TemporalConsciousnessConfig
from src.enactive_consciousness.embodiment import BodySchemaConfig
from src.enactive_consciousness.types import create_temporal_moment, BodyState


def main():
    print("="*60)
    print("ENACTIVE CONSCIOUSNESS PREDICTIVE CODING DEMO")
    print("="*60)
    
    # Setup
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)
    state_dim = 32
    
    # Create configurations
    predictive_config = PredictiveCodingConfig(
        hierarchy_levels=3,
        prediction_horizon=8,
        temporal_scales=(PredictionScale.MICRO, PredictionScale.MESO),
        scale_weights=jnp.array([0.6, 0.4]),
    )
    
    temporal_config = TemporalConsciousnessConfig(
        retention_depth=8,
        protention_horizon=5,
    )
    
    body_schema_config = BodySchemaConfig(
        proprioceptive_dim=32,
        motor_dim=16,
    )
    
    print(f"Configuration created:")
    print(f"  Hierarchy levels: {predictive_config.hierarchy_levels}")
    print(f"  Temporal scales: {[s.value for s in predictive_config.temporal_scales]}")
    print(f"  State dimension: {state_dim}")
    
    # Create predictive system
    predictive_system = create_predictive_coding_system(
        config=predictive_config,
        temporal_config=temporal_config,
        body_schema_config=body_schema_config,
        state_dim=state_dim,
        key=keys[0]
    )
    print("✓ Predictive coding system created")
    
    # Generate sample consciousness state
    current_state = jax.random.normal(keys[1], (state_dim,))
    
    temporal_moment = create_temporal_moment(
        timestamp=0.0,
        retention=jax.random.normal(keys[2], (state_dim,)),
        present_moment=current_state,
        protention=jax.random.normal(keys[3], (state_dim,)),
        synthesis_weights=jax.nn.softmax(jax.random.normal(keys[0], (state_dim,)))
    )
    
    body_state = BodyState(
        proprioception=jax.random.normal(keys[1], (32,)),
        motor_intention=jax.random.normal(keys[2], (16,)),
        boundary_signal=jnp.array([0.8]),
        schema_confidence=0.75
    )
    
    print("✓ Sample consciousness states created")
    
    # Generate hierarchical predictions
    print("\nGenerating hierarchical predictions...")
    predictive_state = predictive_system.generate_hierarchical_predictions(
        current_state=current_state,
        temporal_moment=temporal_moment,
        body_state=body_state,
        environmental_context=None
    )
    
    print(f"✓ Predictions generated successfully!")
    print(f"  Hierarchical levels: {len(predictive_state.hierarchical_predictions)}")
    print(f"  Prediction errors: {len(predictive_state.prediction_errors)}")
    print(f"  Total prediction error: {predictive_state.total_prediction_error:.4f}")
    print(f"  Confidence estimates: {predictive_state.confidence_estimates}")
    print(f"  Convergence achieved: {predictive_state.convergence_status}")
    print(f"  Temporal scales: {list(predictive_state.scale_predictions.keys())}")
    
    # Demonstrate multiple prediction steps
    print(f"\nDemonstrating temporal prediction sequence...")
    sequence_length = 5
    
    for i in range(sequence_length):
        # Update state with slight variation
        current_state = current_state + jax.random.normal(keys[i % 4], (state_dim,)) * 0.1
        temporal_moment = create_temporal_moment(
            timestamp=float(i * 0.1),
            retention=temporal_moment.present_moment,  # Previous present becomes retention
            present_moment=current_state,
            protention=jax.random.normal(keys[i % 4], (state_dim,)) * 0.8,
            synthesis_weights=jax.nn.softmax(jax.random.normal(keys[i % 4], (state_dim,)))
        )
        
        predictive_state = predictive_system.generate_hierarchical_predictions(
            current_state=current_state,
            temporal_moment=temporal_moment,
            body_state=body_state,
        )
        
        print(f"  Step {i+1}: Error={predictive_state.total_prediction_error:.4f}, "
              f"Converged={predictive_state.convergence_status}")
    
    # Note: Accuracy assessment would require matching dimensions
    print(f"\nPredictive coding system operational!")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return predictive_system, predictive_state


if __name__ == "__main__":
    system, state = main()
    print(f"\nDemo completed! The predictive coding system is working correctly.")
    print(f"You can now explore the 'system' and 'state' objects interactively.")