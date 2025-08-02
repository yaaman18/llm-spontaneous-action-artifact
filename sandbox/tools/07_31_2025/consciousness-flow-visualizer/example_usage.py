"""
意識の流れビジュアライザー - 使用例
研究者のための実践的なサンプルコード
"""

import asyncio
import time
import numpy as np
from consciousness_flow import ConsciousnessStream, PhenomenalAnalyzer


async def simulate_problem_solving_consciousness():
    """問題解決中の意識の流れをシミュレート"""
    stream = ConsciousnessStream()
    
    print("🧠 問題解決タスクのシミュレーションを開始...")
    
    # フェーズ1: 問題認識
    for i in range(50):
        state = {
            'attention': {
                'problem_recognition': 0.8 + 0.2 * np.sin(i * 0.1),
                'memory_search': 0.3 + 0.1 * np.random.random()
            },
            'integration': 0.4 + 0.1 * np.sin(i * 0.05),
            'phenomenal_properties': {
                'clarity': 0.3 + 0.2 * (i / 50),
                'confusion': 0.7 - 0.3 * (i / 50),
                'curiosity': 0.6
            },
            'cognitive_load': 0.6 + 0.2 * np.sin(i * 0.1),
            'meta_awareness': 0.5,
            'flow_vector': (np.sin(i * 0.1), 0, np.cos(i * 0.1))
        }
        stream.add_state(state)
        await asyncio.sleep(0.1)
    
    # フェーズ2: 洞察の瞬間
    print("💡 洞察の瞬間をシミュレート...")
    for i in range(20):
        state = {
            'attention': {
                'insight': 0.9,
                'pattern_recognition': 0.8 + 0.1 * np.sin(i * 0.3)
            },
            'integration': 0.8 + 0.2 * np.exp(-i * 0.1),  # 急激な統合
            'phenomenal_properties': {
                'clarity': 0.9,
                'eureka_feeling': 0.8 * np.exp(-i * 0.2),
                'coherence': 0.85
            },
            'cognitive_load': 0.3,  # 負荷が急減
            'meta_awareness': 0.9,  # 高いメタ認知
            'flow_vector': (0, 2 * np.exp(-i * 0.1), 0)  # 上向きの爆発的な流れ
        }
        stream.add_state(state)
        await asyncio.sleep(0.05)
    
    # フェーズ3: 統合と理解
    print("🌊 理解の統合フェーズ...")
    for i in range(50):
        state = {
            'attention': {
                'integration': 0.7,
                'understanding': 0.8 + 0.1 * np.sin(i * 0.1),
                'planning': 0.4 + 0.4 * (i / 50)
            },
            'integration': 0.75 + 0.05 * np.sin(i * 0.05),
            'phenomenal_properties': {
                'clarity': 0.8,
                'satisfaction': 0.6 + 0.2 * (i / 50),
                'coherence': 0.9
            },
            'cognitive_load': 0.4,
            'meta_awareness': 0.7,
            'flow_vector': (
                np.cos(i * 0.05) * 0.5,
                0.2,
                np.sin(i * 0.05) * 0.5
            )
        }
        stream.add_state(state)
        await asyncio.sleep(0.1)
    
    # 分析結果を表示
    print("\n📊 意識フローの分析結果:")
    dynamics = stream.get_flow_dynamics(window_size=20)
    for key, value in dynamics.items():
        print(f"  {key}: {value:.3f}")
    
    # 現象的遷移を検出
    transitions = PhenomenalAnalyzer.detect_phenomenal_transitions(stream)
    print(f"\n🔄 検出された現象的遷移: {len(transitions)}件")
    for t in transitions[:5]:
        print(f"  - {t['type']} at {t['timestamp']:.2f}s (強度: {t['magnitude']:.2f})")


async def simulate_meditation_consciousness():
    """瞑想中の意識の流れをシミュレート"""
    stream = ConsciousnessStream()
    
    print("\n🧘 瞑想状態のシミュレーションを開始...")
    
    for i in range(100):
        # 瞑想の深まりに応じて変化
        depth = min(1.0, i / 50)
        
        state = {
            'attention': {
                'breath_awareness': 0.7 + 0.2 * np.sin(i * 0.02),  # 呼吸のリズム
                'present_moment': 0.5 + 0.4 * depth,
                'wandering_thoughts': 0.5 * (1 - depth) * (1 + np.random.random() * 0.5)
            },
            'integration': 0.6 + 0.3 * depth,
            'phenomenal_properties': {
                'tranquility': 0.4 + 0.5 * depth,
                'spaciousness': 0.3 + 0.6 * depth,
                'equanimity': 0.5 + 0.4 * depth,
                'bliss': 0.2 + 0.3 * depth * np.sin(i * 0.01)
            },
            'cognitive_load': 0.5 * (1 - depth),
            'meta_awareness': 0.6 + 0.3 * depth,
            'flow_vector': (
                0.1 * np.sin(i * 0.02),  # 穏やかな揺らぎ
                -0.2 * depth,  # 深まりとともに下降
                0.1 * np.cos(i * 0.02)
            )
        }
        stream.add_state(state)
        await asyncio.sleep(0.1)
    
    print("瞑想シミュレーション完了")
    
    # 最終状態の現象学的分析
    if stream.current_state:
        qualia = PhenomenalAnalyzer.analyze_qualia_structure(stream.current_state)
        print("\n🎨 最終状態のクオリア構造:")
        print(f"  強度: {qualia['intensity']:.3f}")
        print(f"  複雑性: {qualia['complexity']:.3f}")
        print(f"  現象的統一性: {qualia['phenomenal_unity']:.3f}")


async def simulate_creative_flow():
    """創造的フロー状態をシミュレート"""
    stream = ConsciousnessStream()
    
    print("\n🎨 創造的フロー状態のシミュレーションを開始...")
    
    for i in range(80):
        phase = i * 0.1
        
        # フロー状態の特徴的なパターン
        state = {
            'attention': {
                'creative_focus': 0.9,
                'idea_generation': 0.5 + 0.4 * np.sin(phase * 0.5),
                'evaluation': 0.3 + 0.3 * np.cos(phase * 0.7),
                'time_perception': 0.2  # 時間感覚の消失
            },
            'integration': 0.8 + 0.15 * np.sin(phase * 0.3),
            'phenomenal_properties': {
                'flow': 0.85,
                'effortlessness': 0.8,
                'joy': 0.7 + 0.2 * np.sin(phase * 0.4),
                'novelty': 0.6 + 0.3 * np.random.random(),
                'absorption': 0.9
            },
            'cognitive_load': 0.6,  # 最適な負荷レベル
            'meta_awareness': 0.3,  # 低いメタ認知（没入状態）
            'flow_vector': (
                2 * np.sin(phase),
                0.5 * np.sin(phase * 2),
                2 * np.cos(phase)
            )
        }
        stream.add_state(state)
        await asyncio.sleep(0.1)
    
    print("創造的フロー シミュレーション完了")


def demonstrate_analysis_capabilities():
    """分析機能のデモンストレーション"""
    print("\n🔬 意識流分析機能のデモンストレーション")
    
    # テスト用の意識状態を作成
    test_state = ConsciousnessState(
        timestamp=time.time(),
        attention={'reading': 0.8, 'understanding': 0.7},
        integration=0.75,
        phenomenal_properties={
            'clarity': 0.8,
            'interest': 0.9,
            'comprehension': 0.7
        },
        cognitive_load=0.5,
        meta_awareness=0.6,
        flow_vector=(1.0, 0.0, 0.5)
    )
    
    # クオリア構造の分析
    qualia_analysis = PhenomenalAnalyzer.analyze_qualia_structure(test_state)
    
    print("\nテスト状態のクオリア分析:")
    print(f"  強度: {qualia_analysis['intensity']:.3f}")
    print(f"  複雑性: {qualia_analysis['complexity']:.3f}")
    print(f"  現象的統一性: {qualia_analysis['phenomenal_unity']:.3f}")
    print(f"  支配的な質:")
    for quality, value in qualia_analysis['dominant_qualities']:
        print(f"    - {quality}: {value:.3f}")


async def main():
    """メイン実行関数"""
    print("=== 意識の流れビジュアライザー デモンストレーション ===\n")
    
    # 各種シミュレーションを実行
    await simulate_problem_solving_consciousness()
    await simulate_meditation_consciousness()
    await simulate_creative_flow()
    
    # 分析機能のデモ
    demonstrate_analysis_capabilities()
    
    print("\n✨ デモンストレーション完了!")
    print("ブラウザで http://localhost:8080 にアクセスして、")
    print("リアルタイムの意識の流れを観察してください。")


if __name__ == "__main__":
    asyncio.run(main())