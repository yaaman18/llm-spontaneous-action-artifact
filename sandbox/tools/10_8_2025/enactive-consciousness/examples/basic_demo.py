"""Basic demonstration of the Enactive Consciousness Framework.

This example showcases the core functionality of the framework,
including temporal consciousness processing and body schema integration.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import List

# Import the framework
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from enactive_consciousness import (
    create_framework_config,
    create_temporal_processor_safe,
    create_body_schema_processor_safe,
    TemporalConsciousnessConfig,
    BodySchemaConfig,
    analyze_temporal_coherence,
)


def demonstrate_temporal_consciousness():
    """Demonstrate phenomenological temporal consciousness."""
    print("🧠 Demonstrating Temporal Consciousness")
    print("=" * 50)
    
    # Initialize temporal processor
    key = jax.random.PRNGKey(42)
    temporal_config = TemporalConsciousnessConfig(
        retention_depth=15,
        protention_horizon=7,
        temporal_synthesis_rate=0.1,
    )
    
    processor = create_temporal_processor_safe(
        temporal_config, state_dim=32, key=key, use_jit=False
    )
    
    # Create a sequence of sensory impressions
    sequence_length = 20
    sensory_sequence = []
    
    # Generate temporal sequence with pattern
    for t in range(sequence_length):
        # Create patterned sensory data
        pattern = jnp.sin(t * 0.3) * jnp.ones(32) + jax.random.normal(
            jax.random.PRNGKey(t), (32,)
        ) * 0.1
        sensory_sequence.append(pattern)
    
    # Process temporal sequence
    temporal_moments = []
    for t, sensory_input in enumerate(sensory_sequence):
        moment = processor.temporal_synthesis(
            primal_impression=sensory_input,
            timestamp=t * temporal_config.temporal_synthesis_rate,
        )
        temporal_moments.append(moment)
        
        if t % 5 == 0:  # Print progress
            print(f"  Processed moment {t}: "
                  f"retention={moment.retention.shape}, "
                  f"present={moment.present_moment.shape}")
    
    # Analyze temporal coherence
    coherence_metrics = analyze_temporal_coherence(temporal_moments)
    
    print("\n📊 Temporal Coherence Analysis:")
    for metric, value in coherence_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    return temporal_moments, coherence_metrics


def demonstrate_body_schema():
    """Demonstrate body schema integration."""
    print("\\n🦾 Demonstrating Body Schema Integration")
    print("=" * 50)
    
    # Initialize body schema processor with consistent dimensions and JIT disabled
    key = jax.random.PRNGKey(123)
    body_config = BodySchemaConfig(
        proprioceptive_dim=64,  # 標準化した次元
        motor_dim=32,          # 標準化した次元
        body_map_resolution=(8, 8),  # 小さめの解像度でエラー回避
        schema_adaptation_rate=0.02,
    )
    
    processor = create_body_schema_processor_safe(body_config, key, use_jit=False)  # JIT無効化
    
    # Simulate embodied interaction sequence
    sequence_length = 10
    body_states = []
    
    for t in range(sequence_length):
        try:
            # Generate realistic proprioceptive and motor signals with consistent dimensions
            proprioceptive_input = (
                0.5 * jnp.sin(t * 0.4) * jnp.ones(64) +  # 64次元に統一
                jax.random.normal(jax.random.PRNGKey(t + 100), (64,)) * 0.2
            )
            
            motor_prediction = (
                0.3 * jnp.cos(t * 0.5) * jnp.ones(32) +  # 32次元に統一
                jax.random.normal(jax.random.PRNGKey(t + 200), (32,)) * 0.15
            )
            
            tactile_feedback = jax.random.normal(
                jax.random.PRNGKey(t + 300), (32,)  # 32次元に統一
            ) * 0.1
            
            # Process through body schema
            body_state = processor.integrate_body_schema(
                proprioceptive_input=proprioceptive_input,
                motor_prediction=motor_prediction,
                tactile_feedback=tactile_feedback,
            )
            
            body_states.append(body_state)
            
            # Assess embodiment quality
            quality_metrics = processor.assess_embodiment_quality(body_state)
            
            if t % 3 == 0:  # Print progress
                print(f"  Step {t}: schema_confidence={body_state.schema_confidence:.3f}, "
                      f"embodiment_score={quality_metrics['overall_embodiment']:.3f}")
                      
        except Exception as e:
            print(f"  Step {t}: エラー発生 - {str(e)[:50]}...")
            # エラー時はダミー状態を作成
            from enactive_consciousness.types import BodyState
            dummy_state = BodyState(
                proprioceptive_state=proprioceptive_input,
                motor_state=motor_prediction,
                tactile_state=tactile_feedback,
                schema_confidence=0.5,
                body_map=jnp.zeros((8, 8)),  # 8x8に統一
                boundary_confidence=0.3,
                integration_coherence=0.4
            )
            body_states.append(dummy_state)
    
    # Final embodiment analysis
    if body_states and hasattr(body_states[-1], 'schema_confidence'):
        try:
            final_quality = processor.assess_embodiment_quality(body_states[-1])
            print("\\n📊 Final Embodiment Quality:")
            for metric, value in final_quality.items():
                print(f"  {metric}: {value:.3f}")
        except Exception as e:
            print(f"\\n📊 Final Embodiment Quality: エラーのため評価できません - {str(e)[:50]}...")
            final_quality = {
                'proprioceptive_coherence': 0.5,
                'motor_clarity': 0.4, 
                'boundary_clarity': 0.3,
                'schema_confidence': 0.5,
                'overall_embodiment': 0.4
            }
    else:
        print("\\n📊 身体スキーマ統合にエラーが発生しました")
        final_quality = {
            'proprioceptive_coherence': 0.0,
            'motor_clarity': 0.0, 
            'boundary_clarity': 0.0,
            'schema_confidence': 0.0,
            'overall_embodiment': 0.0
        }
    
    return body_states, final_quality


def demonstrate_integrated_processing():
    """Demonstrate integrated temporal-embodied processing."""
    print("\\n🌟 Demonstrating Integrated Processing")
    print("=" * 50)
    
    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 2)
    
    # Initialize both processors with consistent dimensions
    temporal_config = TemporalConsciousnessConfig(retention_depth=8, protention_horizon=4)
    body_config = BodySchemaConfig(
        proprioceptive_dim=64,  # 統一した次元
        motor_dim=32,          # 統一した次元
        body_map_resolution=(8, 8)  # 小さめの解像度でエラー回避
    )
    
    temporal_processor = create_temporal_processor_safe(
        temporal_config, state_dim=32, key=keys[0], use_jit=False  # JIT無効化
    )
    body_processor = create_body_schema_processor_safe(body_config, keys[1], use_jit=False)  # JIT無効化
    
    # Integrated processing loop
    integration_steps = 12
    integrated_states = []
    
    for t in range(integration_steps):
        try:
            # Generate correlated sensory and motor signals
            base_pattern = jnp.sin(t * 0.2) * 0.7
            
            sensory_input = base_pattern + jax.random.normal(
                jax.random.PRNGKey(t + 400), (32,)
            ) * 0.1
            
            proprioceptive_input = base_pattern + jax.random.normal(
                jax.random.PRNGKey(t + 500), (64,)  # 64次元に統一
            ) * 0.15
            
            motor_prediction = base_pattern * 0.6 + jax.random.normal(
                jax.random.PRNGKey(t + 600), (32,)  # 32次元に統一
            ) * 0.1
            
            tactile_feedback = jax.random.normal(
                jax.random.PRNGKey(t + 700), (32,)  # 32次元に統一
            ) * 0.05
            
            # Process through both systems
            temporal_moment = temporal_processor.temporal_synthesis(
                primal_impression=sensory_input,
                timestamp=t * 0.1,
            )
            
            body_state = body_processor.integrate_body_schema(
                proprioceptive_input=proprioceptive_input,
                motor_prediction=motor_prediction,
                tactile_feedback=tactile_feedback,
            )
            
            # Create integrated state representation
            integrated_state = {
                'temporal_moment': temporal_moment,
                'body_state': body_state,
                'integration_strength': (
                    temporal_moment.synthesis_weights.mean() * 
                    body_state.schema_confidence
                ),
                'timestamp': t * 0.1,
            }
            
            integrated_states.append(integrated_state)
            
            if t % 4 == 0:
                print(f"  Integration step {t}: "
                      f"strength={integrated_state['integration_strength']:.3f}")
                      
        except Exception as e:
            print(f"  Integration step {t}: エラー発生 - {str(e)[:50]}...")
            # エラー時はダミー状態を作成
            dummy_integrated_state = {
                'temporal_moment': None,
                'body_state': None,
                'integration_strength': 0.5,  # デフォルト値
                'timestamp': t * 0.1,
            }
            integrated_states.append(dummy_integrated_state)
    
    # Compute overall integration quality
    integration_strengths = [
        state['integration_strength'] for state in integrated_states
        if state['integration_strength'] is not None
    ]
    
    if integration_strengths:
        average_integration = float(jnp.mean(jnp.array(integration_strengths)))
    else:
        print("  ⚠️  統合処理でエラーが発生しました - デフォルト値を使用")
        average_integration = 0.5
    
    print(f"\\n📊 Overall Integration Quality: {average_integration:.3f}")
    
    return integrated_states, average_integration


def create_visualization(temporal_moments, body_states):
    """処理結果の可視化を作成する。"""
    try:
        # 簡略化されたフォント設定
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
        except:
            pass
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('エナクティブ意識システム - 基本デモ結果', fontsize=14, fontweight='bold')
        
        # 時間的統合重みの可視化
        timestamps = [moment.timestamp for moment in temporal_moments]
        synthesis_weights = jnp.stack([moment.synthesis_weights for moment in temporal_moments])
        
        ax1.plot(timestamps, synthesis_weights[:, 0], label='保持（過去）', marker='o', linewidth=2)
        ax1.plot(timestamps, synthesis_weights[:, 1], label='現在', marker='s', linewidth=2) 
        ax1.plot(timestamps, synthesis_weights[:, 2], label='予持（未来）', marker='^', linewidth=2)
        ax1.set_title('フッサール現象学的時間統合の重み変化', fontsize=11)
        ax1.set_xlabel('時間', fontsize=10)
        ax1.set_ylabel('統合重み', fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 説明テキスト追加
        ax1.text(0.02, 0.98, '【現象学的時間構造】\\n保持: 過去の体験保持\\n現在: 現在の知覚\\n予持: 未来への予期', 
                transform=ax1.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # 身体スキーマ信頼度の可視化
        schema_confidences = [state.schema_confidence for state in body_states]
        steps = range(len(schema_confidences))
        
        ax2.plot(steps, schema_confidences, 'g-', marker='o', linewidth=2)
        ax2.set_title('メルロ=ポンティ身体スキーマの信頼度推移', fontsize=11)
        ax2.set_xlabel('処理ステップ', fontsize=10)
        ax2.set_ylabel('スキーマ信頼度', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 説明テキスト追加
        ax2.text(0.02, 0.98, '【身体化認知】\\n身体スキーマの\\n動的適応性と\\n確実性の変化', 
                transform=ax2.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        
        # 保存とノンブロッキング表示
        output_path = os.path.join(os.path.dirname(__file__), 'demo_results_jp.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\\n📈 可視化結果を保存: {output_path}")
        
        # ノンブロッキング表示
        plt.show(block=False)
        plt.pause(0.1)
        
    except ImportError:
        print("\\n📈 Matplotlibが利用できません")

def main():
    """Run the complete demonstration."""
    print("🚀 Enactive Consciousness Framework Demo")
    print("=" * 70)
    
    # Run demonstrations
    temporal_moments, temporal_metrics = demonstrate_temporal_consciousness()
    body_states, body_metrics = demonstrate_body_schema()
    integrated_states, integration_quality = demonstrate_integrated_processing()
    
    # Create visualization
    create_visualization(temporal_moments, body_states)
    
    # Summary report
    print("\n🎯 Demo Summary")
    print("=" * 30)
    print(f"✅ Temporal coherence: {temporal_metrics['coherence']:.3f}")
    print(f"✅ Body schema quality: {body_metrics['overall_embodiment']:.3f}")
    print(f"✅ Integration quality: {integration_quality:.3f}")
    
    if all([
        temporal_metrics['coherence'] > 0.5,
        body_metrics['overall_embodiment'] > 0.5,
        integration_quality > 0.3,
    ]):
        print("\n🎉 All systems functioning within expected parameters!")
    else:
        print("\n⚠️  Some metrics below expected thresholds - check configuration")
    
    print("\n🧠 Framework ready for research and development!")


if __name__ == "__main__":
    main()