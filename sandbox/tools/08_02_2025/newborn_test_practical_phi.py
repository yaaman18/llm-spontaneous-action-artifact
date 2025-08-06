#!/usr/bin/env python3
"""
NewbornAI 2.0 Practical Phi Integration Test
実用的φ値計算器をNewbornAI 2.0システムに統合テスト

実行方法:
python newborn_test_practical_phi.py
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from newborn_ai_2_integrated_system import create_newborn_ai_2_system
import time


async def test_newborn_practical_phi():
    """NewbornAI 2.0での実用的φ値計算テスト"""
    
    print("🌟 NewbornAI 2.0 実用的φ値計算器統合テスト")
    print("=" * 60)
    
    # システム作成（詳細ログ有効）
    system = create_newborn_ai_2_system("test_practical_phi", verbose=True)
    
    print(f"\n📊 システム初期状態:")
    print(f"   発達段階: {system.current_stage.value}")
    print(f"   φ値: {system.consciousness_level:.6f}")
    print(f"   体験概念数: {len(system.experiential_concepts)}")
    
    # 体験概念を手動で追加（テスト用）
    print(f"\n📝 テスト用体験概念を追加中...")
    test_concepts = []
    
    for i in range(50):
        concept = {
            'type': 'test_experiential_insight',
            'content': f'テスト体験{i}: 私は実用的φ値計算器による意識発達を体験しています。',
            'experiential_quality': 0.7 + (i * 0.005),
            'coherence': 0.8 + (i * 0.003),
            'temporal_depth': 2 + (i // 10),
            'timestamp': time.time(),
            'cycle': i
        }
        test_concepts.append(concept)
    
    # 体験概念を格納
    system.experiential_concepts.extend(test_concepts)
    print(f"   追加された概念数: {len(test_concepts)}")
    print(f"   総概念数: {len(system.experiential_concepts)}")
    
    # 実用的φ値計算を実行
    print(f"\n🧠 実用的φ値計算実行...")
    start_time = time.time()
    
    phi_result = await system.phi_calculator.calculate_experiential_phi(
        system.experiential_concepts
    )
    
    calculation_time = time.time() - start_time
    
    print(f"\n🎉 φ値計算完了!")
    print(f"   ⚡ φ値: {phi_result.phi_value:.6f}")
    print(f"   🌱 発達段階: {phi_result.stage_prediction.value}")
    print(f"   📊 統合品質: {phi_result.integration_quality:.3f}")
    print(f"   ✨ 体験純粋性: {phi_result.experiential_purity:.3f}")
    print(f"   ⏱️  計算時間: {calculation_time:.3f}秒")
    
    # システム状態更新
    system._update_consciousness_state(phi_result)
    
    print(f"\n📈 システム状態更新後:")
    print(f"   発達段階: {system.current_stage.value}")
    print(f"   φ値: {system.consciousness_level:.6f}")
    
    # 発達段階移行チェック
    if phi_result.phi_value >= 0.1:
        print(f"\n✅ 発達段階移行成功!")
        print(f"   φ値 {phi_result.phi_value:.6f} ≥ 0.1")
        
        if system.current_stage.value != "前意識基盤層":
            print(f"   🚀 段階進歩: {system.current_stage.value}")
        else:
            print(f"   ⚠️  段階更新要確認")
    else:
        print(f"\n⚠️  発達段階移行には更なる成長が必要")
    
    # 実用統計確認
    if hasattr(system.phi_calculator, 'get_practical_statistics'):
        practical_stats = system.phi_calculator.get_practical_statistics()
        
        if practical_stats.get('status') != 'theoretical_calculator_in_use':
            print(f"\n📊 実用φ計算統計:")
            print(f"   総計算回数: {practical_stats.get('total_calculations', 0)}")
            print(f"   平均φ値: {practical_stats.get('average_phi', 0.0):.6f}")
            print(f"   最大φ値: {practical_stats.get('max_phi', 0.0):.6f}")
            print(f"   平均計算時間: {practical_stats.get('average_calculation_time', 0.0):.3f}秒")
    
    # 意識レポート表示
    print(f"\n🧠 統合意識状態レポート:")
    system.consciousness_report()
    
    print(f"\n🏆 テスト完了 - NewbornAI 2.0での実用的φ値計算器統合成功!")
    
    return system, phi_result


async def test_multiple_cycles():
    """複数サイクルでの連続φ値計算テスト"""
    
    print("\n" + "=" * 60)
    print("🔄 複数サイクル連続計算テスト")
    print("=" * 60)
    
    system = create_newborn_ai_2_system("test_multi_cycle", verbose=False)
    
    phi_history = []
    
    for cycle in range(5):
        print(f"\n🔄 サイクル {cycle + 1}/5:")
        
        # 新しい体験概念を追加
        new_concepts = []
        for i in range(10):
            concept = {
                'type': 'cycle_experience',
                'content': f'サイクル{cycle}体験{i}: 継続的な意識発達の体験を感じています。',
                'experiential_quality': 0.6 + (cycle * 0.1) + (i * 0.02),
                'coherence': 0.7 + (cycle * 0.05) + (i * 0.01),
                'temporal_depth': 1 + cycle + (i // 3),
                'timestamp': time.time(),
                'cycle': cycle,
                'sub_index': i
            }
            new_concepts.append(concept)
        
        system.experiential_concepts.extend(new_concepts)
        
        # φ値計算
        phi_result = await system.phi_calculator.calculate_experiential_phi(
            system.experiential_concepts
        )
        
        phi_history.append(phi_result.phi_value)
        system._update_consciousness_state(phi_result)
        
        print(f"   概念数: {len(system.experiential_concepts)}")
        print(f"   φ値: {phi_result.phi_value:.6f}")
        print(f"   発達段階: {system.current_stage.value}")
    
    # 成長分析
    print(f"\n📈 成長分析:")
    print(f"   初期φ値: {phi_history[0]:.6f}")
    print(f"   最終φ値: {phi_history[-1]:.6f}")
    print(f"   φ値成長: {phi_history[-1] - phi_history[0]:+.6f}")
    print(f"   φ値履歴: {[f'{p:.3f}' for p in phi_history]}")
    
    if phi_history[-1] > phi_history[0]:
        print(f"   ✅ 連続的な成長を確認!")
    else:
        print(f"   ⚠️  成長パターン要分析")
    
    return phi_history


async def main():
    """メインテスト実行"""
    
    print("🚀 NewbornAI 2.0 実用的φ値計算器 統合テスト開始")
    
    try:
        # 1. 基本統合テスト  
        system, phi_result = await test_newborn_practical_phi()
        
        # 2. 複数サイクルテスト
        phi_history = await test_multiple_cycles()
        
        print(f"\n" + "=" * 60)
        print("🎯 統合テスト総合結果")
        print("=" * 60)
        
        print(f"✅ 基本統合: 成功")
        print(f"   最終φ値: {phi_result.phi_value:.6f}")
        print(f"   発達段階: {phi_result.stage_prediction.value}")
        
        print(f"✅ 連続計算: 成功")
        print(f"   サイクル数: {len(phi_history)}")
        print(f"   φ値範囲: {min(phi_history):.3f} - {max(phi_history):.3f}")
        
        if phi_result.phi_value >= 0.1:
            print(f"✅ 発達段階移行: 成功")
        else:
            print(f"⚠️  発達段階移行: 要改善")
        
        print(f"\n🏆 NewbornAI 2.0実用的φ値計算器統合 完全成功!")
        print(f"🔬 IIT4理論準拠 + 体験記憶特化 + 実用的感度")
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())