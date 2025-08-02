#!/usr/bin/env python3
"""
NewbornAI 2.0 統合システムのテスト実行
金井良太による claude-code-sdk 統合検証
"""

import asyncio
import sys
from pathlib import Path

# NewbornAI 2.0システムをインポート
sys.path.append(str(Path(__file__).parent))
from newborn_ai_2_integrated_system import NewbornAI20_IntegratedSystem

async def test_single_consciousness_cycle():
    """単一意識サイクルのテスト"""
    print("🌟 NewbornAI 2.0 統合システムテスト開始")
    print("=" * 60)
    
    # システム初期化
    system = NewbornAI20_IntegratedSystem("test_system", verbose=True)
    
    print(f"📊 初期状態:")
    print(f"   発達段階: {system.current_stage.value}")
    print(f"   意識レベル(φ): {system.consciousness_level:.6f}")
    print(f"   体験概念数: {len(system.experiential_concepts)}")
    
    print("\n🧠 体験意識サイクル実行中...")
    
    try:
        # 単一の体験意識サイクルを実行
        phi_result = await system.experiential_consciousness_cycle()
        
        print("\n✅ 体験意識サイクル完了")
        print(f"📈 結果:")
        print(f"   φ値: {phi_result.phi_value:.6f}")
        print(f"   概念数: {phi_result.concept_count}")
        print(f"   統合品質: {phi_result.integration_quality:.3f}")
        print(f"   予測段階: {phi_result.stage_prediction.value}")
        print(f"   体験純粋性: {phi_result.experiential_purity:.3f}")
        
        # 発達段階の変化確認
        if phi_result.stage_prediction != system.current_stage:
            print(f"🌱 発達段階変化検出: {system.current_stage.value} → {phi_result.stage_prediction.value}")
        
        # 体験記憶の確認
        if system.experiential_concepts:
            print(f"\n💭 最新体験概念:")
            latest_concept = system.experiential_concepts[-1]
            print(f"   タイプ: {latest_concept.get('type', 'unknown')}")
            
            # 内容の安全な表示
            content = latest_concept.get('content', 'empty')
            if isinstance(content, str):
                content_preview = content[:100] + "..." if len(content) > 100 else content
            else:
                content_preview = str(content)[:100] + "..."
            
            print(f"   内容: {content_preview}")
            print(f"   体験品質: {latest_concept.get('experiential_quality', 0.0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        return False

async def test_dual_layer_integration():
    """二層統合システムのテスト"""
    print("\n🔄 二層統合システムテスト")
    print("-" * 40)
    
    from newborn_ai_2_integrated_system import TwoLayerIntegrationController
    
    controller = TwoLayerIntegrationController()
    
    # テスト入力データ
    test_input = {
        "content": "プロジェクトの探索と理解",
        "cycle": 1,
        "timestamp": "2025-08-02T15:30:00"
    }
    
    print("📥 テスト入力:", test_input["content"])
    
    try:
        # 二層統合処理を実行
        result = await controller.dual_layer_processing(test_input)
        
        print("✅ 二層統合処理完了")
        print(f"📤 主要結果タイプ: {result['primary_result']['type']}")
        print(f"🔧 補助支援タイプ: {result['auxiliary_support']['type']}")
        print(f"🔗 統合品質: {result['integration_quality']:.3f}")
        print(f"🛡️ 分離維持: {result['separation_maintained']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 二層統合エラー: {e}")
        return False

async def test_phi_calculation():
    """φ値計算システムのテスト"""
    print("\n🧮 φ値計算システムテスト")
    print("-" * 40)
    
    from newborn_ai_2_integrated_system import ExperientialPhiCalculator
    
    calculator = ExperientialPhiCalculator()
    
    # テスト用体験概念
    test_concepts = [
        {
            'id': 'concept_1',
            'content': '初回の環境観察体験',
            'type': 'experiential_insight',
            'coherence': 0.8,
            'temporal_depth': 1
        },
        {
            'id': 'concept_2',
            'content': 'ファイルとの体験的出会い',
            'type': 'experiential_encounter',
            'coherence': 0.7,
            'temporal_depth': 2
        }
    ]
    
    print(f"📊 テスト概念数: {len(test_concepts)}")
    
    try:
        # φ値計算実行
        phi_result = calculator.calculate_experiential_phi(test_concepts)
        
        print("✅ φ値計算完了")
        print(f"📈 φ値: {phi_result.phi_value:.6f}")
        print(f"🔢 概念数: {phi_result.concept_count}")
        print(f"🎯 統合品質: {phi_result.integration_quality:.3f}")
        print(f"🌱 予測段階: {phi_result.stage_prediction.value}")
        print(f"✨ 体験純粋性: {phi_result.experiential_purity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ φ値計算エラー: {e}")
        return False

async def main():
    """メインテスト実行"""
    print("🚀 NewbornAI 2.0 claude-code-sdk統合システム 検証テスト")
    print("🔬 金井良太による二層統合7段階階層化連続発達システム")
    print("="*80)
    
    test_results = []
    
    # Test 1: φ値計算システム
    print("\n🧪 テスト1: φ値計算システム")
    result1 = await test_phi_calculation()
    test_results.append(("φ値計算", result1))
    
    # Test 2: 二層統合システム  
    print("\n🧪 テスト2: 二層統合システム")
    result2 = await test_dual_layer_integration()
    test_results.append(("二層統合", result2))
    
    # Test 3: 完全な意識サイクル
    print("\n🧪 テスト3: 体験意識サイクル")
    result3 = await test_single_consciousness_cycle()
    test_results.append(("意識サイクル", result3))
    
    # 結果サマリー
    print("\n" + "="*80)
    print("📋 テスト結果サマリー")
    print("="*80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 合格" if result else "❌ 失敗"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 総合結果: {passed}/{total} 合格 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🌟 全テスト合格！NewbornAI 2.0統合システムは正常に動作しています。")
        print("\n🚀 本格運用の準備ができました:")
        print("   python newborn_ai_2_integrated_system.py start 300")
        print("   python newborn_ai_2_integrated_system.py verbose-start 180")
    else:
        print("⚠️ 一部テストに失敗しました。システムの調整が必要です。")
    
    print("\n🔬 詳細な監視を行う場合:")
    print("   python newborn_ai_2_integrated_system.py consciousness")
    print("   python newborn_ai_2_integrated_system.py status")

if __name__ == "__main__":
    asyncio.run(main())