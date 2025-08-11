"""Demonstration of experiential memory and integrated consciousness system.

This example showcases the advanced experiential memory capabilities including:
- Circular causality dynamics
- Experiential sedimentation
- Associative recall
- Integrated consciousness processing
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import List, Dict
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from enactive_consciousness import (
    create_framework_config,
    TemporalConsciousnessConfig,
    BodySchemaConfig,
    ConsciousnessLevel,
)

from enactive_consciousness.integrated_consciousness import (
    create_enactive_consciousness_system,
    run_consciousness_sequence,
)

from enactive_consciousness.experiential_memory import (
    create_experiential_memory_system,
)


def demonstrate_experiential_memory():
    """Demonstrate core experiential memory functions."""
    print("🧠 Demonstrating Experiential Memory System")
    print("=" * 60)
    
    # Initialize system
    key = jax.random.PRNGKey(42)
    experience_dim = 64
    environment_dim = 32
    context_dim = 16
    
    memory_system = create_experiential_memory_system(
        experience_dim, environment_dim, context_dim, key
    )
    
    print(f"  Initialized memory system:")
    print(f"    Experience dimension: {experience_dim}")
    print(f"    Environment dimension: {environment_dim}")
    print(f"    Context dimension: {context_dim}")
    
    # Create sequence of experiences
    sequence_length = 15
    experiences = []
    
    print(f"\n📚 Processing {sequence_length} experiential moments...")
    
    for t in range(sequence_length):
        # Create varied experiential inputs
        base_pattern = jnp.sin(t * 0.4) * 0.7
        noise_key = jax.random.PRNGKey(t + 100)
        
        current_experience = (
            base_pattern * jnp.ones(experience_dim) +
            jax.random.normal(noise_key, (experience_dim,)) * 0.2
        )
        
        environmental_input = jax.random.normal(
            jax.random.PRNGKey(t + 200), (environment_dim,)
        ) * 0.3
        
        contextual_cues = jax.random.normal(
            jax.random.PRNGKey(t + 300), (context_dim,)
        ) * 0.5
        
        # Vary significance based on pattern
        significance_weight = 0.3 + 0.4 * abs(jnp.sin(t * 0.2))
        
        # Process through experiential memory
        processed_experience, metadata = memory_system.process_experiential_moment(
            current_experience=current_experience,
            environmental_input=environmental_input,
            contextual_cues=contextual_cues,
            significance_weight=float(significance_weight),
        )
        
        experiences.append({
            'original': current_experience,
            'processed': processed_experience,
            'metadata': metadata,
            'significance': float(significance_weight),
            'timestamp': t,
        })
        
        if t % 5 == 0:
            print(f"    Step {t}: "
                  f"Recalls={metadata['num_recalls']}, "
                  f"Significance={significance_weight:.3f}, "
                  f"Circular coherence={metadata['circular_causality']['circular_coherence']:.3f}")
    
    # Analyze experiential memory development
    print(f"\n📊 Experiential Memory Analysis:")
    
    memory_state = memory_system.get_memory_state()
    print(f"  Total traces: {memory_state['num_traces']}")
    print(f"  Sediment layers: {memory_state['sediment_layers']}")
    print(f"  Average significance: {memory_state['average_significance']:.3f}")
    print(f"  Total sedimentation: {memory_state['total_sedimentation']:.2f}")
    
    # Analyze recall patterns
    recall_counts = [exp['metadata']['num_recalls'] for exp in experiences]
    avg_recalls = jnp.mean(jnp.array(recall_counts))
    print(f"  Average recalls per moment: {avg_recalls:.2f}")
    
    return experiences, memory_system


def demonstrate_circular_causality():
    """Demonstrate circular causality dynamics in isolation."""
    print("\n🔄 Demonstrating Circular Causality Engine")
    print("=" * 60)
    
    from enactive_consciousness.experiential_memory import CircularCausalityEngine
    
    key = jax.random.PRNGKey(123)
    state_dim = 32
    environment_dim = 24
    hidden_dim = 64
    
    engine = CircularCausalityEngine(state_dim, environment_dim, hidden_dim, key)
    
    print(f"  Initialized circular causality engine:")
    print(f"    State dimension: {state_dim}")
    print(f"    Environment dimension: {environment_dim}")
    
    # Run circular causality dynamics
    current_state = jax.random.normal(jax.random.PRNGKey(1), (state_dim,))
    
    causality_sequence = []
    
    print(f"\n⚡ Running circular causality dynamics...")
    
    for step in range(20):
        # Create environmental input with some pattern
        env_input = (
            jnp.sin(step * 0.3) * jnp.ones(environment_dim) +
            jax.random.normal(jax.random.PRNGKey(step + 50), (environment_dim,)) * 0.1
        )
        
        # Execute circular causality step
        next_state, emergent_meaning, metrics = engine.circular_causality_step(
            current_state=current_state,
            environmental_input=env_input,
            previous_meaning=None if step == 0 else emergent_meaning,
        )
        
        causality_sequence.append({
            'state': next_state,
            'meaning': emergent_meaning,
            'metrics': metrics,
            'step': step,
        })
        
        current_state = next_state
        
        if step % 5 == 0:
            print(f"    Step {step}: "
                  f"Self-ref={metrics['self_reference_strength']:.3f}, "
                  f"Coupling={metrics['coupling_strength']:.3f}, "
                  f"Meaning={metrics['meaning_emergence']:.3f}")
    
    # Analyze circular causality development
    print(f"\n📊 Circular Causality Analysis:")
    
    coherence_values = [seq['metrics']['circular_coherence'] for seq in causality_sequence]
    avg_coherence = jnp.mean(jnp.array(coherence_values))
    
    meaning_strengths = [jnp.linalg.norm(seq['meaning']) for seq in causality_sequence]
    avg_meaning_strength = jnp.mean(jnp.array(meaning_strengths))
    
    print(f"  Average circular coherence: {avg_coherence:.3f}")
    print(f"  Average meaning emergence: {avg_meaning_strength:.3f}")
    print(f"  System convergence: {'Yes' if avg_coherence > 0.5 else 'No'}")
    
    return causality_sequence, engine


def demonstrate_integrated_consciousness():
    """Demonstrate full integrated consciousness system."""
    print("\n🌟 Demonstrating Integrated Consciousness System")
    print("=" * 60)
    
    # System configuration
    key = jax.random.PRNGKey(456)
    
    config = create_framework_config(
        retention_depth=12,
        protention_horizon=6,
        consciousness_threshold=0.55,
        proprioceptive_dim=48,
        motor_dim=16,
    )
    
    temporal_config = TemporalConsciousnessConfig(
        retention_depth=12,
        protention_horizon=6,
        temporal_synthesis_rate=0.08,
    )
    
    body_config = BodySchemaConfig(
        proprioceptive_dim=48,
        motor_dim=16,
        body_map_resolution=(12, 12),
    )
    
    state_dim = 64
    environment_dim = 32
    
    # Create integrated system
    consciousness_system = create_enactive_consciousness_system(
        config=config,
        temporal_config=temporal_config,
        body_config=body_config,
        state_dim=state_dim,
        environment_dim=environment_dim,
        key=key,
    )
    
    print(f"  Initialized integrated consciousness system:")
    print(f"    State dimension: {state_dim}")
    print(f"    Consciousness threshold: {config.consciousness_threshold}")
    
    # Create input sequence for consciousness processing
    sequence_length = 25
    input_sequence = []
    
    for t in range(sequence_length):
        # Create correlated multi-modal inputs
        base_pattern = jnp.sin(t * 0.15) * 0.8
        noise_scale = 0.05 + 0.1 * abs(jnp.cos(t * 0.3))
        
        inputs = {
            'sensory_input': (
                base_pattern * jnp.ones(state_dim) +
                jax.random.normal(jax.random.PRNGKey(t + 400), (state_dim,)) * noise_scale
            ),
            'proprioceptive_input': (
                base_pattern * 0.7 * jnp.ones(48) +
                jax.random.normal(jax.random.PRNGKey(t + 500), (48,)) * noise_scale
            ),
            'motor_prediction': (
                base_pattern * 0.5 * jnp.ones(16) +
                jax.random.normal(jax.random.PRNGKey(t + 600), (16,)) * noise_scale
            ),
            'environmental_state': (
                jax.random.normal(jax.random.PRNGKey(t + 700), (environment_dim,)) * 0.3
            ),
            'contextual_cues': (
                jax.random.normal(jax.random.PRNGKey(t + 800), (32,)) * 0.4
            ),
        }
        
        input_sequence.append(inputs)
    
    print(f"\n🧠 Processing {sequence_length} integrated consciousness moments...")
    
    # Process consciousness sequence
    consciousness_states = run_consciousness_sequence(
        consciousness_system, input_sequence, initial_timestamp=0.0
    )
    
    # Analyze consciousness development
    print(f"\n📊 Consciousness Analysis:")
    
    # Consciousness levels over time
    consciousness_levels = [state.consciousness_level for state in consciousness_states]
    avg_consciousness = jnp.mean(jnp.array(consciousness_levels))
    max_consciousness = jnp.max(jnp.array(consciousness_levels))
    
    # Integration coherence
    coherence_values = [state.integration_coherence for state in consciousness_states]
    avg_coherence = jnp.mean(jnp.array(coherence_values))
    
    # Circular causality strength
    causality_strengths = [state.circular_causality_strength for state in consciousness_states]
    avg_causality = jnp.mean(jnp.array(causality_strengths))
    
    print(f"  Average consciousness level: {avg_consciousness:.3f}")
    print(f"  Maximum consciousness level: {max_consciousness:.3f}")
    print(f"  Average integration coherence: {avg_coherence:.3f}")
    print(f"  Average circular causality: {avg_causality:.3f}")
    
    # Assess consciousness levels
    consciousness_assessments = [
        consciousness_system.assess_consciousness_level(state)
        for state in consciousness_states
    ]
    
    level_counts = {}
    for level in consciousness_assessments:
        level_counts[level.value] = level_counts.get(level.value, 0) + 1
    
    print(f"\n  Consciousness level distribution:")
    for level, count in level_counts.items():
        percentage = (count / len(consciousness_assessments)) * 100
        print(f"    {level}: {count} ({percentage:.1f}%)")
    
    # Performance metrics
    import time
    start_time = time.time()
    
    # Run a performance test sequence
    test_inputs = input_sequence[:5]  # Use first 5 for timing
    _ = run_consciousness_sequence(consciousness_system, test_inputs)
    
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    performance_metrics = consciousness_system.compute_performance_metrics(
        consciousness_sequence=consciousness_states[:5],
        processing_time_ms=processing_time,
        memory_usage_mb=50.0,  # Estimated
    )
    
    print(f"\n📈 Performance Metrics:")
    print(f"  Temporal coherence: {performance_metrics.temporal_coherence:.3f}")
    print(f"  Embodiment stability: {performance_metrics.embodiment_stability:.3f}")
    print(f"  Coupling effectiveness: {performance_metrics.coupling_effectiveness:.3f}")
    print(f"  Overall consciousness score: {performance_metrics.overall_consciousness_score:.3f}")
    print(f"  Processing time: {performance_metrics.processing_time_ms:.1f}ms")
    
    return consciousness_states, consciousness_system, performance_metrics


def create_visualization(experiences, causality_sequence, consciousness_states):
    """包括的な結果可視化を作成する。"""
    try:
        # 簡略化されたフォント設定
        try:
            plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
        except:
            pass
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('体験記憶と円環的因果性システム - 統合分析\\n（現象学・エナクティヴィズム・統合情報理論の融合）', fontsize=13)
        
        # プロット1: 体験記憶の重要度の時間変化
        significances = [exp['significance'] for exp in experiences]
        recalls = [exp['metadata']['num_recalls'] for exp in experiences]
        timestamps = [exp['timestamp'] for exp in experiences]
        
        ax1.plot(timestamps, significances, 'b-', marker='o', label='重要度重み', linewidth=2)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(timestamps, recalls, 'r-', marker='s', label='想起回数', linewidth=2)
        
        ax1.set_title('体験記憶の発達過程', fontsize=11)
        ax1.set_xlabel('時間ステップ', fontsize=10)
        ax1.set_ylabel('重要度重み', color='b', fontsize=10)
        ax1_twin.set_ylabel('想起回数', color='r', fontsize=10)
        ax1.legend(loc='upper left', fontsize=9)
        ax1_twin.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 説明テキスト
        ax1.text(0.02, 0.98, '【体験記憶】\\n経験の意味的重要度と\\n想起頻度の相関関係', 
                transform=ax1.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # プロット2: 円環的因果性動力学
        steps = [seq['step'] for seq in causality_sequence]
        coherence = [seq['metrics']['circular_coherence'] for seq in causality_sequence]
        meaning_strength = [jnp.linalg.norm(seq['meaning']) for seq in causality_sequence]
        
        ax2.plot(steps, coherence, 'g-', marker='o', label='円環的一貫性', linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(steps, meaning_strength, 'm-', marker='^', label='意味強度', linewidth=2)
        
        ax2.set_title('円環的因果性の動力学', fontsize=11)
        ax2.set_xlabel('処理ステップ', fontsize=10)
        ax2.set_ylabel('円環的一貫性', color='g', fontsize=10)
        ax2_twin.set_ylabel('意味強度', color='m', fontsize=10)
        ax2.legend(loc='upper left', fontsize=9)
        ax2_twin.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 説明テキスト
        ax2.text(0.02, 0.98, '【円環的因果性】\\nバレラの自己組織化\\nによる意味創発', 
                transform=ax2.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # プロット3: 意識レベルの時間変化
        consciousness_levels = [state.consciousness_level for state in consciousness_states]
        integration_coherence = [state.integration_coherence for state in consciousness_states]
        consciousness_timestamps = [state.timestamp for state in consciousness_states]
        
        ax3.plot(consciousness_timestamps, consciousness_levels, 'purple', marker='o', linewidth=2, label='意識レベル')
        ax3.plot(consciousness_timestamps, integration_coherence, 'orange', marker='s', linewidth=2, label='統合一貫性')
        
        ax3.axhline(y=0.55, color='red', linestyle='--', alpha=0.7, label='意識閾値')
        ax3.set_title('統合意識の発達', fontsize=11)
        ax3.set_xlabel('時間', fontsize=10)
        ax3.set_ylabel('レベル/一貫性', fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 説明テキスト
        ax3.text(0.02, 0.98, '【統合意識】\\nIITのΦ値と\\nエナクティブ結合\\nの統合指標', 
                transform=ax3.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        # プロット4: システム性能概観
        causality_strengths = [state.circular_causality_strength for state in consciousness_states]
        
        # 性能レーダー風プロット
        performance_categories = ['意識\\nレベル', '統合\\n一貫性', '円環的\\n因果性']
        performance_values = [
            jnp.mean(jnp.array(consciousness_levels)),
            jnp.mean(jnp.array(integration_coherence)),
            jnp.mean(jnp.array(causality_strengths))
        ]
        
        colors = ['purple', 'orange', 'green']
        bars = ax4.bar(performance_categories, performance_values, 
                      color=colors, alpha=0.7)
        
        # バーに数値ラベルを追加
        for bar, value in zip(bars, performance_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax4.set_title('システム統合性能', fontsize=11)
        ax4.set_ylabel('平均スコア', fontsize=10)
        ax4.set_ylim(0, 1.0)
        ax4.grid(True, alpha=0.3)
        
        # 説明テキスト
        ax4.text(0.02, 0.98, '【統合性能】\\n各サブシステムの\\n相乗効果による\\n全体性能指標', 
                transform=ax4.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
        
        plt.tight_layout()
        
        # 可視化の保存
        output_path = os.path.join(os.path.dirname(__file__), 'experiential_memory_results_jp.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\\n📊 可視化結果を保存: {output_path}")
        
        # ノンブロッキング表示
        plt.show(block=False)
        plt.pause(0.1)
        
    except ImportError:
        print("\\n📊 Matplotlibが利用できません")

def main():
    """Run the complete experiential memory demonstration."""
    print("🚀 Experiential Memory & Integrated Consciousness Demo")
    print("=" * 70)
    
    # Run demonstrations
    print("\nPhase 1: Core experiential memory functions")
    experiences, memory_system = demonstrate_experiential_memory()
    
    print("\nPhase 2: Circular causality dynamics")
    causality_sequence, causality_engine = demonstrate_circular_causality()
    
    print("\nPhase 3: Integrated consciousness system")
    consciousness_states, consciousness_system, performance_metrics = demonstrate_integrated_consciousness()
    
    # Create comprehensive visualization
    create_visualization(experiences, causality_sequence, consciousness_states)
    
    # Final summary
    print("\n🎯 Demo Summary")
    print("=" * 40)
    print(f"✅ Experiential memory traces: {len(experiences)}")
    print(f"✅ Circular causality steps: {len(causality_sequence)}")
    print(f"✅ Consciousness moments: {len(consciousness_states)}")
    print(f"✅ Overall system performance: {performance_metrics.overall_consciousness_score:.3f}")
    
    # System readiness assessment
    readiness_score = (
        0.3 * (len(experiences) / 15) +  # Experiential coverage
        0.3 * performance_metrics.overall_consciousness_score +  # Performance
        0.4 * (jnp.mean(jnp.array([state.consciousness_level for state in consciousness_states])))  # Consciousness
    )
    
    print(f"\n🎉 System Readiness Score: {readiness_score:.3f}")
    
    if readiness_score > 0.7:
        print("🌟 System demonstrates strong enactive consciousness capabilities!")
    elif readiness_score > 0.5:
        print("✨ System shows promising enactive consciousness development!")
    else:
        print("⚠️  System requires further tuning for optimal performance.")
    
    print("\n🧠 Experiential memory system ready for advanced research!")


if __name__ == "__main__":
    main()