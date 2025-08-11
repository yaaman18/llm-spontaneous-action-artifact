#!/usr/bin/env python3
"""
エナクティブ意識フレームワーク V3.0 - 基本デモ

このデモスクリプトは、エナクティブ意識フレームワークの
基本的な使用方法を示します。
"""

import numpy as np
import time
from typing import List

# コアシステムのインポート  
import sys
sys.path.append('..')
from ngc_learn_adapter import HybridPredictiveCodingAdapter
from domain.value_objects.precision_weights import PrecisionWeights
from domain.value_objects.consciousness_state import ConsciousnessState
from domain.value_objects.phi_value import PhiValue
from domain.factories.consciousness_factory import ConsciousnessFactory
from infrastructure.basic_som import BasicSOM
from domain.value_objects.som_topology import SOMTopology
from domain.value_objects.learning_parameters import LearningParameters


def print_header():
    """ヘッダー表示"""
    print("=" * 60)
    print("  エナクティブ意識フレームワーク V3.0 - 基本デモ")
    print("  Enactive Consciousness Framework V3.0 - Basic Demo")
    print("=" * 60)
    print()


def test_core_systems():
    """コアシステムの動作確認"""
    print("【1. コアシステム動作確認】")
    print("-" * 40)
    
    results = []
    
    # 1. Predictive Coding Core
    try:
        from domain.entities.predictive_coding_core import PredictiveCodingCore
        print("✅ Predictive Coding Core: 正常にインポート")
        results.append(True)
    except Exception as e:
        print(f"❌ Predictive Coding Core: {e}")
        results.append(False)
    
    # 2. NGC-Learn Adapter
    try:
        adapter = HybridPredictiveCodingAdapter(3, 10)
        engine_type = adapter.engine_type
        print(f"✅ Hybrid Adapter: {engine_type}エンジンで動作中")
        results.append(True)
    except Exception as e:
        print(f"❌ Hybrid Adapter: {e}")
        results.append(False)
    
    # 3. SOM System
    try:
        topology = SOMTopology.create_rectangular()
        som = BasicSOM(
            map_dimensions=(5, 5),
            input_dimensions=3,
            topology=topology
        )
        print("✅ Self-Organizing Map: 正常に初期化")
        results.append(True)
    except Exception as e:
        print(f"❌ Self-Organizing Map: {e}")
        results.append(False)
    
    print()
    return all(results)


def demo_predictive_processing():
    """予測符号化処理のデモ"""
    print("【2. 予測符号化処理デモ】")
    print("-" * 40)
    
    # システム初期化
    hierarchy_levels = 3
    input_dimensions = 10
    
    print(f"階層レベル: {hierarchy_levels}")
    print(f"入力次元数: {input_dimensions}")
    
    adapter = HybridPredictiveCodingAdapter(hierarchy_levels, input_dimensions)
    
    # 精度重みの設定（numpy配列として正しく初期化）
    precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
    
    # 5ステップの処理実行
    print("\n処理実行中...")
    errors = []
    
    for step in range(5):
        # ランダム入力生成（環境からの感覚入力をシミュレート）
        input_data = np.random.rand(input_dimensions) + 0.1 * np.sin(step * 0.5)
        
        # 予測処理実行
        prediction_state = adapter.process_input(input_data, precision_weights)
        
        # 結果記録
        errors.append(prediction_state.total_error)
        
        print(f"  ステップ {step+1}: エラー={prediction_state.total_error:.4f}, "
              f"状態={prediction_state.convergence_status}")
        
        time.sleep(0.1)  # 短い待機
    
    # 統計表示
    print(f"\n平均エラー: {np.mean(errors):.4f}")
    print(f"最小エラー: {np.min(errors):.4f}")
    print(f"最大エラー: {np.max(errors):.4f}")
    print()
    
    return np.mean(errors) < 10.0  # 妥当なエラー範囲内か確認


def demo_consciousness_states():
    """意識状態生成のデモ"""
    print("【3. 意識状態生成デモ】")
    print("-" * 40)
    
    # 意識ファクトリーの使用
    factory = ConsciousnessFactory()
    
    # 異なる条件での意識状態生成
    conditions = [
        ("低複雑性", 5, 0.3, 0.2),
        ("中複雑性", 10, 0.5, 0.5),
        ("高複雑性", 20, 0.8, 0.8),
    ]
    
    for name, complexity, env_richness, potential in conditions:
        aggregate = factory.create_consciousness_aggregate(
            system_complexity=complexity,
            environmental_richness=env_richness,
            consciousness_potential=potential
        )
        
        state = aggregate.current_state
        print(f"\n{name}環境:")
        print(f"  Φ値: {state.phi_value.value:.3f}")
        print(f"  意識レベル: {state.consciousness_level}")
        print(f"  予測品質: {state.prediction_state.prediction_quality:.3f}")
        print(f"  メタ認知信頼度: {state.metacognitive_confidence:.3f}")
    
    print()
    return True


def demo_som_training():
    """SOM訓練のデモ"""
    print("【4. 自己組織化マップ訓練デモ】")
    print("-" * 40)
    
    # SOM初期化
    topology = SOMTopology.create_rectangular()
    som = BasicSOM(
        map_dimensions=(5, 5),
        input_dimensions=3,
        topology=topology,
        random_seed=42  # 再現性のため
    )
    
    # 学習パラメータ
    learning_params = LearningParameters(
        initial_learning_rate=0.1,
        final_learning_rate=0.01,
        initial_radius=2.0,
        final_radius=0.5,
        max_iterations=50
    )
    
    # 訓練データ生成（3つのクラスター）
    print("訓練データ生成中...")
    training_data = []
    np.random.seed(42)  # 再現性のため
    for _ in range(30):
        # クラスター1
        training_data.append(np.random.randn(3) * 0.2 + np.array([1.0, 0.0, 0.0]))
        # クラスター2
        training_data.append(np.random.randn(3) * 0.2 + np.array([0.0, 1.0, 0.0]))
        # クラスター3
        training_data.append(np.random.randn(3) * 0.2 + np.array([0.0, 0.0, 1.0]))
    
    # 訓練実行
    print("SOM訓練実行中...")
    som.train(training_data, learning_params)
    
    # 結果確認
    test_samples = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]
    
    print("\nBMU（最適合ユニット）検索結果:")
    for i, sample in enumerate(test_samples):
        bmu = som.find_bmu(sample)
        print(f"  サンプル{i+1}: BMU位置 = {bmu}")
    
    # 量子化誤差の計算
    quantization_error = som.compute_quantization_error(test_samples)
    print(f"\n量子化誤差: {quantization_error:.4f}")
    
    print()
    return True


def demo_integrated_system():
    """統合システムのデモ"""
    print("【5. 統合システム動作デモ】")
    print("-" * 40)
    
    # 全コンポーネントの協調動作
    print("予測符号化とSOMの統合処理...")
    
    # 予測符号化システム
    predictor = HybridPredictiveCodingAdapter(3, 5)
    precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
    
    # SOMシステム
    topology = SOMTopology.create_rectangular()
    som = BasicSOM(
        map_dimensions=(4, 4),
        input_dimensions=5,
        topology=topology,
        random_seed=42
    )
    
    # 統合処理
    integrated_errors = []
    for i in range(10):
        # 共通入力データ
        input_data = np.random.rand(5)
        
        # 予測処理
        pred_state = predictor.process_input(input_data, precision_weights)
        
        # SOM更新
        bmu = som.find_bmu(input_data)
        
        # 統合エラー計算
        integrated_error = pred_state.total_error * 0.8 + np.linalg.norm(bmu) * 0.2
        integrated_errors.append(integrated_error)
    
    print(f"統合処理完了:")
    print(f"  平均統合エラー: {np.mean(integrated_errors):.4f}")
    print(f"  処理の安定性: {1.0 / (1.0 + np.std(integrated_errors)):.3f}")
    print()
    
    return True


def main():
    """メイン実行関数"""
    print_header()
    
    # タイマー開始
    start_time = time.time()
    
    # 各デモ実行
    results = []
    
    # 1. コアシステム確認
    results.append(test_core_systems())
    
    # 2. 予測符号化
    results.append(demo_predictive_processing())
    
    # 3. 意識状態
    results.append(demo_consciousness_states())
    
    # 4. SOM訓練
    results.append(demo_som_training())
    
    # 5. 統合システム
    results.append(demo_integrated_system())
    
    # 実行時間計算
    elapsed_time = time.time() - start_time
    
    # 最終結果表示
    print("=" * 60)
    print("【実行結果サマリー】")
    print("-" * 40)
    
    if all(results):
        print("✅ 全テスト成功")
        print(f"✅ 実行時間: {elapsed_time:.2f}秒")
        print("✅ システム準備完了")
        print("\n🧠✨ エナクティブ意識フレームワーク V3.0")
        print("     研究開発での使用準備が整いました")
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("   詳細は上記のログを確認してください")
    
    print("=" * 60)
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    exit(main())