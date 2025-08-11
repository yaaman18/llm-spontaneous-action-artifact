#!/usr/bin/env python3
"""
最小限のシステムテスト。

基本的な値オブジェクトの作成と動作確認を行います。
"""

from datetime import datetime
import numpy as np
from domain.value_objects.phi_value import PhiValue
from domain.value_objects.prediction_state import PredictionState  
from domain.value_objects.probability_distribution import ProbabilityDistribution
from domain.value_objects.consciousness_state import ConsciousnessState


def create_minimal_prediction_state() -> PredictionState:
    """最小限の予測状態を作成"""
    return PredictionState(
        hierarchical_errors=[0.1, 0.05, 0.02],
        precision_weighted_errors=[0.1, 0.04, 0.012],
        convergence_status="converging",
        learning_iteration=10,
        timestamp=datetime.now()
    )


def create_minimal_probability_distribution() -> ProbabilityDistribution:
    """最小限の確率分布を作成"""
    # 正規化された3要素のカテゴリ分布
    probabilities = np.array([0.5, 0.3, 0.2])
    return ProbabilityDistribution(
        probabilities=probabilities,
        distribution_type="categorical"
    )


def create_minimal_consciousness_state() -> ConsciousnessState:
    """最小限の意識状態を作成"""
    phi_value = PhiValue(value=0.7, complexity=1.0, integration=0.7)
    prediction_state = create_minimal_prediction_state()
    uncertainty_dist = create_minimal_probability_distribution()
    
    return ConsciousnessState(
        phi_value=phi_value,
        prediction_state=prediction_state,
        uncertainty_distribution=uncertainty_dist,
        metacognitive_confidence=0.6
    )


def main():
    """メインテスト関数"""
    print("=== エナクティブ意識フレームワーク最小システムテスト ===\n")
    
    # 1. Φ値のテスト
    print("1. Φ値テスト:")
    phi = PhiValue(value=0.5, complexity=1.0, integration=0.5)
    print(f"   Φ値: {phi.value}")
    print(f"   意識状態: {'有' if phi.is_conscious else '無'}")
    print(f"   複雑性: {phi.complexity}")
    print(f"   統合性: {phi.integration}\n")
    
    # 2. 予測状態のテスト
    print("2. 予測状態テスト:")
    pred_state = create_minimal_prediction_state()
    print(f"   総誤差: {pred_state.total_error:.3f}")
    print(f"   収束状態: {'収束' if pred_state.is_converged else '学習中'}")
    print(f"   階層誤差: {pred_state.hierarchical_errors}\n")
    
    # 3. 確率分布のテスト
    print("3. 確率分布テスト:")
    prob_dist = create_minimal_probability_distribution()
    print(f"   確率値: {prob_dist.probabilities}")
    print(f"   エントロピー: {prob_dist.entropy:.3f}")
    print(f"   分布型: {prob_dist.distribution_type}\n")
    
    # 4. 意識状態のテスト
    print("4. 意識状態テスト:")
    try:
        consciousness = create_minimal_consciousness_state()
        print(f"   意識レベル: {consciousness.consciousness_level:.3f}")
        print(f"   意識状態: {'有意識' if consciousness.is_conscious else '無意識'}")
        print(f"   Φ値: {consciousness.phi_value.value}")
        print(f"   メタ認知信頼度: {consciousness.metacognitive_confidence}")
        print("   ✓ 意識状態の作成に成功しました")
    except Exception as e:
        print(f"   ✗ 意識状態の作成に失敗: {e}")
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    main()