"""
Predictive Coding System Demonstration.

Comprehensive demonstration of the hierarchical predictive coding system
implementation based on the Free Energy Principle. Shows training,
inference, and analysis capabilities with both synthetic and real-world
inspired data patterns.

Usage:
    python examples/predictive_coding_demonstration.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import List, Tuple
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.jax_predictive_coding_core import JaxPredictiveCodingCore
from application.services.predictive_coding_training_service import (
    PredictiveCodingTrainingService,
    TrainingConfiguration
)
from application.services.predictive_coding_inference_service import (
    PredictiveCodingInferenceService
)
from domain.value_objects.precision_weights import PrecisionWeights
from domain.value_objects.learning_parameters import LearningParameters
from domain.events.domain_events import *


def create_synthetic_data_patterns() -> dict:
    """Create various synthetic data patterns for demonstration."""
    
    # Pattern 1: Sinusoidal waves with noise
    time_steps = 200
    t = np.linspace(0, 4*np.pi, time_steps)
    
    sinusoidal_data = []
    for i in range(time_steps):
        # Multi-frequency sinusoidal pattern
        pattern = np.array([
            np.sin(t[i]) + 0.3 * np.sin(3*t[i]),
            np.cos(t[i]) + 0.2 * np.cos(2*t[i]),
            0.5 * np.sin(0.5*t[i]),
            np.sin(t[i] + np.pi/4) * 0.8
        ])
        # Add small amount of noise
        pattern += np.random.normal(0, 0.05, size=pattern.shape)
        sinusoidal_data.append(pattern)
    
    # Pattern 2: Step function transitions
    step_data = []
    for i in range(time_steps):
        step_value = int(i // (time_steps // 4))  # 4 distinct steps
        pattern = np.array([
            step_value * 0.5,
            (3 - step_value) * 0.3,
            step_value % 2 * 0.8,
            (step_value + 1) * 0.2
        ])
        pattern += np.random.normal(0, 0.02, size=pattern.shape)
        step_data.append(pattern)
    
    # Pattern 3: Random walk with drift
    random_walk_data = []
    current_state = np.array([0.0, 0.0, 0.0, 0.0])
    drift = np.array([0.01, -0.005, 0.002, -0.008])
    
    for i in range(time_steps):
        current_state += drift + np.random.normal(0, 0.1, size=4)
        current_state = np.clip(current_state, -2.0, 2.0)  # Bounded random walk
        random_walk_data.append(current_state.copy())
    
    # Pattern 4: Hierarchical structure (higher-level pattern controls lower levels)
    hierarchical_data = []
    for i in range(time_steps):
        # High-level oscillation
        high_level = np.sin(t[i] / 4)
        
        # Mid-level modulated by high-level
        mid_level = high_level * np.cos(t[i])
        
        # Low-level details
        pattern = np.array([
            high_level,
            mid_level,
            mid_level * 0.5 + np.sin(2*t[i]) * 0.3,
            high_level * 0.2 + np.random.normal(0, 0.05)
        ])
        hierarchical_data.append(pattern)
    
    return {
        'sinusoidal': np.array(sinusoidal_data),
        'step_function': np.array(step_data),
        'random_walk': np.array(random_walk_data),
        'hierarchical': np.array(hierarchical_data)
    }


def demonstrate_basic_functionality():
    """Demonstrate basic predictive coding functionality."""
    print("=== 基本機能のデモンストレーション ===")
    print("(Basic Functionality Demonstration)")
    
    # Create predictive coding core
    core = JaxPredictiveCodingCore(
        hierarchy_levels=3,
        input_dimensions=4,
        learning_rate=0.02,
        temporal_window=10,
        enable_active_inference=True
    )
    
    # Create precision weights
    uniform_weights = PrecisionWeights.create_uniform(3)
    focused_weights = PrecisionWeights.create_focused(3, focus_level=1, focus_strength=3.0)
    
    print(f"階層レベル数: {core.hierarchy_levels}")
    print(f"入力次元数: {core.input_dimensions}")
    print(f"隠れ層次元: {core._hidden_dimensions}")
    print(f"注意機構: {'有効' if core._enable_active_inference else '無効'}")
    print()
    
    # Test prediction generation
    test_input = np.array([0.5, -0.3, 0.8, -0.2])
    
    print("予測生成テスト:")
    predictions_uniform = core.generate_predictions(test_input, uniform_weights)
    predictions_focused = core.generate_predictions(test_input, focused_weights)
    
    for level in range(core.hierarchy_levels):
        print(f"  レベル {level}:")
        print(f"    均等注意: 形状 {predictions_uniform[level].shape}, 平均 {np.mean(predictions_uniform[level]):.4f}")
        print(f"    集中注意: 形状 {predictions_focused[level].shape}, 平均 {np.mean(predictions_focused[level]):.4f}")
    
    # Test complete processing cycle
    print("\n完全処理サイクルテスト:")
    prediction_state = core.process_input(test_input, uniform_weights, learning_rate=0.01)
    
    print(f"  総予測誤差: {prediction_state.total_error:.6f}")
    print(f"  収束状態: {prediction_state.convergence_status}")
    print(f"  学習反復: {prediction_state.learning_iteration}")
    print(f"  予測品質: {prediction_state.prediction_quality:.4f}")
    
    # Test free energy and precision estimates
    free_energy = core.get_free_energy_estimate()
    precision_estimates = core.get_precision_estimates()
    
    print(f"  自由エネルギー推定: {free_energy:.6f}")
    print("  精度推定:")
    for level, precision in precision_estimates.items():
        print(f"    {level}: {precision:.4f}")
    
    print()
    return core


def demonstrate_training_capabilities():
    """Demonstrate training capabilities with different data patterns."""
    print("=== 学習機能のデモンストレーション ===")
    print("(Training Capabilities Demonstration)")
    
    # Create system components
    core = JaxPredictiveCodingCore(
        hierarchy_levels=3,
        input_dimensions=4,
        learning_rate=0.01,
        enable_active_inference=True
    )
    
    learning_params = LearningParameters(
        base_learning_rate=0.01,
        min_learning_rate=0.001,
        max_learning_rate=0.1,
        adaptation_rate=0.05
    )
    
    training_config = TrainingConfiguration(
        max_epochs=50,
        convergence_threshold=0.01,
        early_stopping_patience=10,
        validation_frequency=5,
        learning_rate_schedule="adaptive",
        enable_monitoring=True
    )
    
    training_service = PredictiveCodingTrainingService(
        core, learning_params, training_config
    )
    
    # Generate training data
    print("学習データの生成中...")
    data_patterns = create_synthetic_data_patterns()
    
    # Use sinusoidal pattern for training
    training_data = []
    validation_data = []
    
    sinusoidal_data = data_patterns['sinusoidal']
    split_point = int(0.8 * len(sinusoidal_data))
    
    for i in range(split_point):
        precision_weights = PrecisionWeights.create_uniform(3)
        # Occasionally use focused attention
        if i % 10 == 0:
            precision_weights = PrecisionWeights.create_focused(
                3, focus_level=np.random.randint(0, 3), focus_strength=2.0
            )
        training_data.append((sinusoidal_data[i], precision_weights))
    
    for i in range(split_point, len(sinusoidal_data)):
        precision_weights = PrecisionWeights.create_uniform(3)
        validation_data.append((sinusoidal_data[i], precision_weights))
    
    print(f"学習データ数: {len(training_data)}")
    print(f"検証データ数: {len(validation_data)}")
    
    # Train the model
    print("\nバッチ学習開始...")
    start_time = datetime.now()
    training_metrics = training_service.train_batch(training_data, validation_data)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"学習完了 (時間: {training_time:.2f}秒)")
    print(f"学習エポック数: {len(training_metrics)}")
    
    # Analyze training results
    if training_metrics:
        final_metrics = training_metrics[-1]
        initial_metrics = training_metrics[0]
        
        print(f"\n学習結果:")
        print(f"  初期誤差: {initial_metrics.total_error:.6f}")
        print(f"  最終誤差: {final_metrics.total_error:.6f}")
        print(f"  誤差削減: {(initial_metrics.total_error - final_metrics.total_error):.6f}")
        print(f"  最終自由エネルギー: {final_metrics.free_energy_estimate:.6f}")
        print(f"  収束率: {final_metrics.convergence_rate:.6f}")
        print(f"  安定性: {final_metrics.stability_measure:.4f}")
        
        if final_metrics.validation_error:
            print(f"  検証誤差: {final_metrics.validation_error:.6f}")
    
    # Analyze domain events
    domain_events = training_service.get_domain_events()
    event_types = {}
    for event in domain_events:
        event_type = event.event_type
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    print(f"\n生成されたドメインイベント:")
    for event_type, count in event_types.items():
        print(f"  {event_type}: {count}件")
    
    # Get training summary
    training_summary = training_service.get_training_summary()
    print(f"\n学習サマリー:")
    print(f"  状態: {training_summary['status']}")
    print(f"  収束達成: {'はい' if training_summary['convergence_achieved'] else 'いいえ'}")
    print(f"  最終学習率: {training_summary['final_learning_rate']:.6f}")
    
    return core, training_service, training_metrics


def demonstrate_inference_and_analysis(trained_core, training_service):
    """Demonstrate inference and analysis capabilities."""
    print("\n=== 推論・解析機能のデモンストレーション ===")
    print("(Inference and Analysis Demonstration)")
    
    # Create inference service
    inference_service = PredictiveCodingInferenceService(trained_core)
    
    # Generate test data patterns
    data_patterns = create_synthetic_data_patterns()
    
    print("異なるデータパターンでの推論テスト:")
    
    pattern_results = {}
    
    for pattern_name, pattern_data in data_patterns.items():
        print(f"\n{pattern_name.upper()}パターン:")
        
        # Test on first few samples
        test_samples = pattern_data[:5]
        pattern_results[pattern_name] = []
        
        for i, test_input in enumerate(test_samples):
            # Use different attention strategies
            if i % 2 == 0:
                precision_weights = PrecisionWeights.create_uniform(3)
                attention_type = "uniform"
            else:
                precision_weights = PrecisionWeights.create_focused(
                    3, focus_level=1, focus_strength=3.0
                )
                attention_type = "focused"
            
            result = inference_service.predict(
                test_input,
                precision_weights,
                return_uncertainty=True,
                return_representations=True
            )
            
            pattern_results[pattern_name].append(result)
            
            print(f"  サンプル {i+1} ({attention_type}):")
            print(f"    予測誤差: {[f'{err:.4f}' for err in result.prediction_errors]}")
            print(f"    信頼度: {result.confidence:.4f}")
            print(f"    自由エネルギー: {result.free_energy_estimate:.4f}")
            if result.temporal_consistency:
                print(f"    時間的一貫性: {result.temporal_consistency:.4f}")
    
    # Comprehensive system analysis
    print("\n\n=== システム状態解析 ===")
    analysis_result = inference_service.analyze_system_state()
    
    print("システム状態:")
    for key, value in analysis_result.system_state.items():
        print(f"  {key}: {value}")
    
    print("\n階層別性能分析:")
    for level, metrics in analysis_result.hierarchical_metrics.items():
        print(f"  レベル {level}:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                print(f"    {metric_name}: {metric_value:.4f}")
            else:
                print(f"    {metric_name}: {metric_value}")
    
    print("\n精度動力学分析:")
    for key, value in analysis_result.precision_analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n収束分析:")
    for key, value in analysis_result.convergence_analysis.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    if analysis_result.recommendations:
        print("\nシステム推奨事項:")
        for recommendation in analysis_result.recommendations:
            print(f"  • {recommendation}")
    
    # Uncertainty analysis
    print("\n\n=== 不確実性分析 ===")
    test_input = data_patterns['sinusoidal'][0]
    
    uncertainty_analysis = inference_service.compute_prediction_uncertainty(
        test_input, num_samples=20, precision_noise_scale=0.1
    )
    
    print("モンテカルロ不確実性推定:")
    print(f"  総認識論的不確実性: {uncertainty_analysis['total_epistemic_uncertainty']:.6f}")
    print(f"  最大不確実性レベル: {uncertainty_analysis['max_uncertainty_level']}")
    print("  レベル別不確実性:")
    for level, uncertainty in enumerate(uncertainty_analysis['level_uncertainties']):
        print(f"    レベル {level}: {uncertainty:.6f}")
    
    return inference_service, analysis_result


def demonstrate_attention_mechanisms(core):
    """Demonstrate attention and active inference mechanisms."""
    print("\n=== 注意機構・能動推論のデモンストレーション ===")
    print("(Attention Mechanisms and Active Inference Demonstration)")
    
    # Create test input
    test_input = np.array([0.8, -0.4, 0.2, -0.6])
    
    # Different attention configurations
    attention_configs = [
        ("均等注意", PrecisionWeights.create_uniform(3)),
        ("レベル0集中", PrecisionWeights.create_focused(3, 0, 4.0)),
        ("レベル1集中", PrecisionWeights.create_focused(3, 1, 4.0)),
        ("レベル2集中", PrecisionWeights.create_focused(3, 2, 4.0)),
    ]
    
    print("異なる注意戦略の比較:")
    
    attention_results = {}
    
    for name, precision_weights in attention_configs:
        print(f"\n{name}:")
        
        # Reset core state for fair comparison
        core.reset_state()
        
        # Generate predictions
        predictions = core.generate_predictions(test_input, precision_weights)
        
        # Process input to get full state
        prediction_state = core.process_input(test_input, precision_weights, 0.02)
        
        attention_results[name] = {
            'predictions': predictions,
            'prediction_state': prediction_state,
            'precision_weights': precision_weights
        }
        
        print(f"  注意集中度: {precision_weights.attention_focus:.4f}")
        print(f"  エントロピー: {precision_weights.entropy:.4f}")
        print(f"  支配レベル: {precision_weights.dominant_level}")
        print(f"  総予測誤差: {prediction_state.total_error:.6f}")
        print(f"  自由エネルギー: {core.get_free_energy_estimate():.6f}")
        
        print("  正規化重み:", [f"{w:.3f}" for w in precision_weights.normalized_weights])
    
    # Demonstrate attention adaptation
    print("\n\n注意重みの適応:")
    
    # Start with uniform attention
    adaptive_weights = PrecisionWeights.create_uniform(3)
    
    # Simulate errors that should trigger adaptation
    simulation_errors = [
        [0.5, 0.1, 0.2],  # High error at level 0
        [0.3, 0.05, 0.15],  # Decreasing errors
        [0.1, 0.02, 0.08],  # Further decrease
        [0.05, 0.01, 0.04]  # Final low errors
    ]
    
    print("適応プロセス:")
    for step, errors in enumerate(simulation_errors):
        print(f"\nステップ {step + 1}:")
        print(f"  誤差: {errors}")
        print(f"  適応前重み: {[f'{w:.3f}' for w in adaptive_weights.weights]}")
        
        # Adapt weights
        adaptive_weights = adaptive_weights.adapt_weights(errors, adaptation_strength=0.1)
        
        print(f"  適応後重み: {[f'{w:.3f}' for w in adaptive_weights.weights]}")
        print(f"  注意集中度: {adaptive_weights.attention_focus:.4f}")
        print(f"  支配レベル: {adaptive_weights.dominant_level}")
    
    return attention_results


def visualize_results(training_metrics, pattern_results):
    """Visualize training and inference results."""
    print("\n=== 結果の可視化 ===")
    print("(Results Visualization)")
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Predictive Coding System Analysis', fontsize=16)
        
        # Plot 1: Training metrics over time
        if training_metrics:
            epochs = [m.epoch for m in training_metrics]
            errors = [m.total_error for m in training_metrics]
            free_energies = [m.free_energy_estimate for m in training_metrics]
            learning_rates = [m.learning_rate for m in training_metrics]
            
            ax1 = axes[0, 0]
            ax1.plot(epochs, errors, 'b-', label='Total Error', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Prediction Error', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_title('Training Progress')
            
            # Secondary axis for free energy
            ax1_twin = ax1.twinx()
            ax1_twin.plot(epochs, free_energies, 'r--', label='Free Energy', alpha=0.7)
            ax1_twin.set_ylabel('Free Energy', color='r')
            ax1_twin.tick_params(axis='y', labelcolor='r')
            
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learning rate adaptation
        if training_metrics:
            ax2 = axes[0, 1]
            ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Adaptive Learning Rate')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        # Plot 3: Confidence across different patterns
        ax3 = axes[1, 0]
        pattern_names = list(pattern_results.keys())
        avg_confidences = []
        
        for pattern_name in pattern_names:
            results = pattern_results[pattern_name]
            confidences = [r.confidence for r in results]
            avg_confidence = np.mean(confidences)
            avg_confidences.append(avg_confidence)
        
        bars = ax3.bar(range(len(pattern_names)), avg_confidences, 
                      color=['blue', 'green', 'orange', 'red'])
        ax3.set_xlabel('Data Pattern')
        ax3.set_ylabel('Average Confidence')
        ax3.set_title('Prediction Confidence by Pattern')
        ax3.set_xticks(range(len(pattern_names)))
        ax3.set_xticklabels(pattern_names, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, conf in zip(bars, avg_confidences):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom')
        
        # Plot 4: Error distribution across hierarchy levels
        ax4 = axes[1, 1]
        
        # Collect errors from all patterns
        all_level_errors = {0: [], 1: [], 2: []}
        
        for pattern_results_list in pattern_results.values():
            for result in pattern_results_list:
                for level, error in enumerate(result.prediction_errors):
                    if level in all_level_errors:
                        all_level_errors[level].append(error)
        
        levels = list(all_level_errors.keys())
        level_means = [np.mean(all_level_errors[level]) for level in levels]
        level_stds = [np.std(all_level_errors[level]) for level in levels]
        
        ax4.bar(levels, level_means, yerr=level_stds, capsize=5,
               color=['lightblue', 'lightgreen', 'lightcoral'])
        ax4.set_xlabel('Hierarchy Level')
        ax4.set_ylabel('Mean Prediction Error')
        ax4.set_title('Error Distribution Across Hierarchy')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'predictive_coding_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可視化結果を保存しました: {output_path}")
        
        # Try to display if running interactively
        try:
            plt.show()
        except:
            print("対話的環境ではないため、プロット表示をスキップしました。")
        
    except ImportError:
        print("matplotlib が利用できません。可視化をスキップします。")
    except Exception as e:
        print(f"可視化エラー: {str(e)}")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("階層的予測符号化システム デモンストレーション")
    print("Hierarchical Predictive Coding System Demonstration")
    print("Based on the Free Energy Principle")
    print("=" * 60)
    print()
    
    try:
        # 1. Basic functionality demonstration
        core = demonstrate_basic_functionality()
        
        # 2. Training capabilities demonstration
        trained_core, training_service, training_metrics = demonstrate_training_capabilities()
        
        # 3. Inference and analysis demonstration
        inference_service, analysis_result = demonstrate_inference_and_analysis(
            trained_core, training_service
        )
        
        # 4. Attention mechanisms demonstration
        attention_results = demonstrate_attention_mechanisms(trained_core)
        
        # 5. Collect pattern results for visualization
        data_patterns = create_synthetic_data_patterns()
        pattern_results = {}
        
        for pattern_name, pattern_data in data_patterns.items():
            pattern_results[pattern_name] = []
            for i in range(min(3, len(pattern_data))):
                precision_weights = PrecisionWeights.create_uniform(3)
                result = inference_service.predict(pattern_data[i], precision_weights)
                pattern_results[pattern_name].append(result)
        
        # 6. Visualize results
        visualize_results(training_metrics, pattern_results)
        
        print("\n" + "=" * 60)
        print("デモンストレーション完了")
        print("Demonstration Complete")
        print("=" * 60)
        
        # Final summary
        print(f"\n最終統計:")
        print(f"• 学習エポック数: {len(training_metrics) if training_metrics else 0}")
        print(f"• 推論実行回数: {len(inference_service.get_inference_history())}")
        print(f"• ドメインイベント数: {len(training_service.get_domain_events())}")
        
        final_summary = training_service.get_training_summary()
        if final_summary['status'] != 'not_trained':
            print(f"• 最終予測誤差: {final_summary['final_error']:.6f}")
            print(f"• 収束達成: {'はい' if final_summary['convergence_achieved'] else 'いいえ'}")
        
        print(f"• システム推奨事項数: {len(analysis_result.recommendations)}")
        
        return {
            'core': trained_core,
            'training_service': training_service,
            'inference_service': inference_service,
            'training_metrics': training_metrics,
            'analysis_result': analysis_result,
            'attention_results': attention_results,
            'pattern_results': pattern_results
        }
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()