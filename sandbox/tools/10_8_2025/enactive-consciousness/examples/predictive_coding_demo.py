"""Demonstration of advanced predictive coding capabilities.

This example showcases the integrated predictive coding system with:
- NGC-based hierarchical prediction networks
- Multi-scale temporal predictions
- Dynamic error minimization
- Integration with body schema and temporal consciousness
- Hyperparameter optimization

Run this script to see the predictive coding system in action.
"""

import logging
from typing import List, Tuple
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from enactive_consciousness import (
    PredictiveCodingConfig,
    PredictiveState,
    IntegratedPredictiveCoding,
    create_predictive_coding_system,
    optimize_hyperparameters,
    TemporalConsciousnessConfig,
    BodySchemaConfig,
    TemporalMoment,
    BodyState,
    PredictionScale,
    create_temporal_moment,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demonstration_config() -> Tuple[PredictiveCodingConfig, TemporalConsciousnessConfig, BodySchemaConfig]:
    """Create configuration for demonstration."""
    
    # Predictive coding configuration
    predictive_config = PredictiveCodingConfig(
        hierarchy_levels=4,
        prediction_horizon=15,
        error_convergence_threshold=1e-4,
        ngc_learning_rate=1e-3,
        temporal_scales=(
            PredictionScale.MICRO,
            PredictionScale.MESO,
            PredictionScale.MACRO
        ),
        scale_weights=jnp.array([0.5, 0.3, 0.2]),
        body_schema_weight=0.3,
        temporal_synthesis_weight=0.4,
        environmental_context_weight=0.3,
        hyperparameter_adaptation_rate=1e-4,
        prediction_error_history_length=100,
        dynamic_adjustment_sensitivity=0.1,
    )
    
    # Temporal consciousness configuration
    temporal_config = TemporalConsciousnessConfig(
        retention_depth=12,
        protention_horizon=8,
        primal_impression_width=0.15,
        temporal_synthesis_rate=0.05,
        temporal_decay_factor=0.95,
    )
    
    # Body schema configuration
    body_schema_config = BodySchemaConfig(
        proprioceptive_dim=64,
        motor_dim=32,
        body_map_resolution=(20, 20),
        boundary_sensitivity=0.7,
        schema_adaptation_rate=0.01,
        motor_intention_strength=0.5,
    )
    
    return predictive_config, temporal_config, body_schema_config


def generate_synthetic_consciousness_data(
    sequence_length: int,
    state_dim: int,
    key: jax.random.PRNGKey,
) -> List[Tuple[jnp.ndarray, TemporalMoment, BodyState]]:
    """Generate synthetic data representing consciousness states over time."""
    
    keys = jax.random.split(key, sequence_length)
    data_sequence = []
    
    for i, step_key in enumerate(keys):
        sub_keys = jax.random.split(step_key, 5)
        
        # Current consciousness state
        current_state = jax.random.normal(sub_keys[0], (state_dim,))
        
        # Add temporal dynamics (slight drift and cyclic patterns)
        temporal_drift = jnp.sin(i * 0.1) * 0.2
        current_state = current_state + temporal_drift
        
        # Create temporal moment with realistic temporal structure
        temporal_moment = create_temporal_moment(
            timestamp=float(i * 0.05),  # 50ms time steps
            retention=jax.random.normal(sub_keys[1], (state_dim,)) * 0.8,
            present_moment=current_state,
            protention=jax.random.normal(sub_keys[2], (state_dim,)) * 0.6,
            synthesis_weights=jax.nn.softmax(jax.random.normal(sub_keys[3], (state_dim,)))
        )
        
        # Create body state with realistic proprioceptive patterns
        body_state = BodyState(
            proprioception=jax.random.normal(sub_keys[4], (64,)) * 0.7,
            motor_intention=jax.random.normal(sub_keys[4], (32,)) * 0.5,
            boundary_signal=jnp.array([jnp.sin(i * 0.05) * 0.3 + 0.7]),
            schema_confidence=float(0.6 + 0.3 * jnp.sin(i * 0.02))
        )
        
        data_sequence.append((current_state, temporal_moment, body_state))
    
    return data_sequence


def demonstrate_hierarchical_predictions(
    predictive_system: IntegratedPredictiveCoding,
    data_sequence: List[Tuple[jnp.ndarray, TemporalMoment, BodyState]],
) -> List[PredictiveState]:
    """Demonstrate hierarchical prediction generation."""
    
    logger.info("Generating hierarchical predictions across consciousness sequence...")
    
    prediction_states = []
    start_time = time.time()
    
    for i, (current_state, temporal_moment, body_state) in enumerate(data_sequence):
        # Generate comprehensive predictions
        predictive_state = predictive_system.generate_hierarchical_predictions(
            current_state=current_state,
            temporal_moment=temporal_moment,
            body_state=body_state,
            environmental_context=None  # No external context in this demo
        )
        
        prediction_states.append(predictive_state)
        
        if i % 10 == 0:
            logger.info(f"  Step {i:3d}: Error={predictive_state.total_prediction_error:.4f}, "
                       f"Confidence={jnp.mean(predictive_state.confidence_estimates):.3f}, "
                       f"Converged={predictive_state.convergence_status}")
    
    processing_time = time.time() - start_time
    logger.info(f"Completed {len(data_sequence)} predictions in {processing_time:.2f}s "
               f"({processing_time/len(data_sequence)*1000:.1f}ms per step)")
    
    return prediction_states


def demonstrate_multi_scale_predictions(
    prediction_states: List[PredictiveState],
) -> None:
    """Demonstrate multi-scale temporal prediction analysis."""
    
    logger.info("Analyzing multi-scale temporal predictions...")
    
    # Analyze scale-specific predictions
    scale_errors = {scale: [] for scale in PredictionScale}
    
    for state in prediction_states:
        for scale, prediction in state.scale_predictions.items():
            # Compute scale-specific error (using magnitude as proxy)
            scale_error = float(jnp.mean(jnp.abs(prediction)))
            scale_errors[scale].append(scale_error)
    
    # Report scale analysis
    for scale, errors in scale_errors.items():
        if errors:
            mean_error = jnp.mean(jnp.array(errors))
            std_error = jnp.std(jnp.array(errors))
            logger.info(f"  {scale.value.capitalize()} scale: "
                       f"Mean error={mean_error:.4f} (±{std_error:.4f})")
    
    # Analyze convergence patterns
    convergence_rate = sum(state.convergence_status for state in prediction_states) / len(prediction_states)
    avg_prediction_error = jnp.mean(jnp.array([state.total_prediction_error for state in prediction_states]))
    avg_confidence = jnp.mean(jnp.array([
        jnp.mean(state.confidence_estimates) for state in prediction_states
    ]))
    
    logger.info(f"Overall prediction quality:")
    logger.info(f"  Convergence rate: {convergence_rate:.1%}")
    logger.info(f"  Average prediction error: {avg_prediction_error:.4f}")
    logger.info(f"  Average confidence: {avg_confidence:.3f}")


def demonstrate_dynamic_error_minimization(
    predictive_system: IntegratedPredictiveCoding,
    prediction_states: List[PredictiveState],
) -> None:
    """Demonstrate dynamic error minimization and adaptation."""
    
    logger.info("Demonstrating dynamic error minimization...")
    
    # Track error reduction over optimization steps
    error_history = []
    adaptation_metrics_history = []
    
    for i, predictive_state in enumerate(prediction_states[::5]):  # Sample every 5th state
        # Optimize predictions
        optimized_state, adaptation_metrics = predictive_system.optimize_predictions(
            predictive_state,
            learning_rate_adjustment=0.1 if predictive_state.total_prediction_error > 0.1 else None
        )
        
        error_history.append(optimized_state.total_prediction_error)
        adaptation_metrics_history.append(adaptation_metrics)
        
        if i % 4 == 0 and adaptation_metrics:
            logger.info(f"  Optimization step {i:2d}: "
                       f"Error={adaptation_metrics.get('total_prediction_error', 0.0):.4f}, "
                       f"LR={adaptation_metrics.get('adapted_learning_rate', 0.0):.2e}")
    
    if adaptation_metrics_history and adaptation_metrics_history[0]:
        # Report adaptation summary
        final_metrics = adaptation_metrics_history[-1]
        logger.info(f"Error minimization summary:")
        logger.info(f"  Initial error: {error_history[0]:.4f}")
        logger.info(f"  Final error: {error_history[-1]:.4f}")
        logger.info(f"  Improvement: {(error_history[0] - error_history[-1])/error_history[0]:.1%}")
        logger.info(f"  Final learning rate: {final_metrics.get('adapted_learning_rate', 0.0):.2e}")


def demonstrate_prediction_accuracy_assessment(
    predictive_system: IntegratedPredictiveCoding,
    data_sequence: List[Tuple[jnp.ndarray, TemporalMoment, BodyState]],
    prediction_states: List[PredictiveState],
) -> None:
    """Demonstrate prediction accuracy assessment."""
    
    logger.info("Assessing prediction accuracy...")
    
    accuracy_scores = []
    
    for i in range(len(prediction_states) - 1):
        predictive_state = prediction_states[i]
        actual_next_state = data_sequence[i + 1][0]  # Next consciousness state
        
        # Create actual outcomes dictionary
        actual_outcomes = {
            'next_state': actual_next_state,
            'future_states': [data_sequence[j][0] for j in range(i + 1, min(i + 4, len(data_sequence)))]
        }
        
        # Assess accuracy
        accuracy_metrics = predictive_system.assess_predictive_accuracy(
            predictive_state, actual_outcomes
        )
        
        accuracy_scores.append(accuracy_metrics)
    
    if accuracy_scores:
        # Aggregate accuracy metrics
        hierarchical_acc = jnp.mean(jnp.array([
            score.get('hierarchical_accuracy', 0.0) for score in accuracy_scores
        ]))
        
        overall_confidence = jnp.mean(jnp.array([
            score.get('overall_confidence', 0.0) for score in accuracy_scores
        ]))
        
        convergence_rate = jnp.mean(jnp.array([
            float(score.get('convergence_achieved', False)) for score in accuracy_scores
        ]))
        
        logger.info(f"Prediction accuracy assessment:")
        logger.info(f"  Hierarchical prediction accuracy: {hierarchical_acc:.3f}")
        logger.info(f"  Overall confidence: {overall_confidence:.3f}")
        logger.info(f"  Convergence rate: {convergence_rate:.1%}")
        
        # Report temporal scale accuracies
        for scale in PredictionScale:
            scale_key = f'{scale.value}_accuracy'
            scale_accuracies = [
                score.get(scale_key, 0.0) for score in accuracy_scores 
                if scale_key in score
            ]
            if scale_accuracies:
                mean_acc = jnp.mean(jnp.array(scale_accuracies))
                logger.info(f"  {scale.value.capitalize()} scale accuracy: {mean_acc:.3f}")


def create_visualization(
    prediction_states: List[PredictiveState],
    save_path: str = "predictive_coding_demo_results_jp.png"
) -> None:
    """予測コーディング結果の可視化を作成する。"""
    
    logger.info("可視化を作成中...")
    
    # 簡略化されたフォント設定
    try:
        plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
    except:
        pass
    
    # プロット用データの抽出
    timestamps = [state.timestamp for state in prediction_states]
    prediction_errors = [state.total_prediction_error for state in prediction_states]
    confidence_estimates = [jnp.mean(state.confidence_estimates) for state in prediction_states]
    convergence_status = [float(state.convergence_status) for state in prediction_states]
    
    # サブプロット図の作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('階層予測コーディングシステム - デモンストレーション結果\\n（ベイズ脳理論とアクティブインファレンス統合）', fontsize=14)
    
    # プロット1: 予測誤差の時間発展
    axes[0, 0].plot(timestamps, prediction_errors, 'b-', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('予測誤差の進化', fontsize=11)
    axes[0, 0].set_xlabel('時間', fontsize=10)
    axes[0, 0].set_ylabel('総予測誤差', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.02, 0.98, '【予測誤差】\\n知覚予測と\\n実際の感覚入力\\nとの差分', 
                   transform=axes[0, 0].transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # プロット2: 信頼度推定
    axes[0, 1].plot(timestamps, confidence_estimates, 'g-', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('予測信頼度', fontsize=11)
    axes[0, 1].set_xlabel('時間', fontsize=10) 
    axes[0, 1].set_ylabel('平均信頼度', fontsize=10)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.02, 0.98, '【信頼度】\\nベイズ推論による\\n予測の確実性\\n評価', 
                   transform=axes[0, 1].transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # プロット3: 収束状況
    axes[1, 0].fill_between(timestamps, convergence_status, alpha=0.6, color='orange')
    axes[1, 0].set_title('予測収束状況', fontsize=11)
    axes[1, 0].set_xlabel('時間', fontsize=10)
    axes[1, 0].set_ylabel('収束率', fontsize=10)
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0.02, 0.98, '【収束】\\n階層間での\\n予測モデルの\\n安定化度合い', 
                   transform=axes[1, 0].transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # プロット4: 誤差vs信頼度の相関
    axes[1, 1].scatter(prediction_errors, confidence_estimates, alpha=0.6, color='purple')
    axes[1, 1].set_title('誤差-信頼度相関', fontsize=11)
    axes[1, 1].set_xlabel('予測誤差', fontsize=10)
    axes[1, 1].set_ylabel('信頼度', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.02, 0.98, '【相関分析】\\n予測精度と\\n主観的確実性\\nの関係性', 
                   transform=axes[1, 1].transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"可視化を保存: {save_path}")
    
    # ノンブロッキング表示
    plt.show(block=False)
    plt.pause(0.1)


def main():
    """Run the complete predictive coding demonstration."""
    
    logger.info("=" * 60)
    logger.info("ENACTIVE CONSCIOUSNESS PREDICTIVE CODING DEMONSTRATION")
    logger.info("=" * 60)
    
    # Setup
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)
    state_dim = 64
    sequence_length = 50
    
    # Create configurations
    logger.info("Setting up predictive coding system...")
    predictive_config, temporal_config, body_schema_config = create_demonstration_config()
    
    # Create predictive system
    predictive_system = create_predictive_coding_system(
        config=predictive_config,
        temporal_config=temporal_config,
        body_schema_config=body_schema_config,
        state_dim=state_dim,
        key=keys[0]
    )
    
    logger.info(f"Predictive system initialized:")
    logger.info(f"  Hierarchy levels: {predictive_config.hierarchy_levels}")
    logger.info(f"  Temporal scales: {[s.value for s in predictive_config.temporal_scales]}")
    logger.info(f"  State dimension: {state_dim}")
    
    # Generate synthetic consciousness data
    logger.info("Generating synthetic consciousness sequence...")
    consciousness_data = generate_synthetic_consciousness_data(
        sequence_length=sequence_length,
        state_dim=state_dim,
        key=keys[1]
    )
    
    # Demonstrate hierarchical predictions
    prediction_states = demonstrate_hierarchical_predictions(
        predictive_system, consciousness_data
    )
    
    # Demonstrate multi-scale analysis
    demonstrate_multi_scale_predictions(prediction_states)
    
    # Demonstrate error minimization
    demonstrate_dynamic_error_minimization(predictive_system, prediction_states)
    
    # Assess prediction accuracy
    demonstrate_prediction_accuracy_assessment(
        predictive_system, consciousness_data, prediction_states
    )
    
    # Create visualization
    try:
        create_visualization(prediction_states)
    except Exception as e:
        logger.warning(f"Visualization creation failed: {e}")
    
    # Demonstrate hyperparameter optimization
    logger.info("Demonstrating hyperparameter optimization...")
    try:
        validation_data = consciousness_data[:20]  # Use subset for validation
        optimized_config, optimization_metrics = optimize_hyperparameters(
            predictive_system, validation_data, optimization_steps=50, key=keys[2]
        )
        
        logger.info(f"Hyperparameter optimization completed:")
        for metric, value in optimization_metrics.items():
            logger.info(f"  {metric}: {value:.3f}")
            
    except Exception as e:
        logger.warning(f"Hyperparameter optimization failed: {e}")
    
    logger.info("=" * 60)
    logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    return predictive_system, prediction_states


if __name__ == "__main__":
    # Run demonstration
    system, states = main()
    
    # Additional analysis if running interactively
    print(f"\nDemo completed! Generated {len(states)} predictive states.")
    print("You can now explore the 'system' and 'states' objects interactively.")