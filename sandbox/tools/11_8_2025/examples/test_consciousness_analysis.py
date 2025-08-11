#!/usr/bin/env python3
"""
意識分析システムのテスト実行スクリプト

コマンドライン引数でテキストを受け取り、意識分析を実行する。
"""

import sys
import numpy as np
sys.path.append('..')

from text_consciousness_features import PhenomenologicalTextAnalyzer, analyze_text_consciousness
from domain.factories.consciousness_factory import ConsciousnessFactory
from ngc_learn_adapter import HybridPredictiveCodingAdapter
from domain.value_objects.precision_weights import PrecisionWeights


def process_single_text(text: str):
    """単一テキストの意識分析"""
    print(f"🧠 エナクティブ意識フレームワーク V3.0")
    print(f"   テキスト意識分析システム")
    print("=" * 50)
    
    # 現象学的特徴量分析
    analyzer = PhenomenologicalTextAnalyzer()
    features, feature_dict, interpretation = analyze_text_consciousness(text, verbose=False)
    
    # 意識状態生成
    factory = ConsciousnessFactory()
    
    # 予測誤差と結合強度の計算
    variance = np.var(features)
    error_base = max(0.01, variance * 2)
    prediction_errors = [error_base * 1.5, error_base * 1.0, error_base * 0.5]
    
    active_features = np.sum(features > 0.1)
    coupling_strength = 0.3 + (active_features / 10) * 0.7
    coupling_strength = min(coupling_strength, 1.0)
    
    consciousness_state = factory.create_emergent_consciousness_state(
        environmental_input=features,
        prediction_errors=prediction_errors,
        coupling_strength=coupling_strength
    )
    
    # 予測処理
    adapter = HybridPredictiveCodingAdapter(3, 10)
    precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
    prediction_state = adapter.process_input(features, precision_weights)
    
    # 意味生成
    phi = consciousness_state.phi_value.value
    dominant_feature_idx = np.argmax(features)
    
    meaning_categories = [
        "志向的思考", "時間的省察", "身体的体験", "社会的交流", "直感的理解",
        "創造的思考", "日常的体験", "自然な流れ", "文脈的理解", "個人的体験"
    ]
    
    base_meaning = meaning_categories[dominant_feature_idx]
    
    if phi > 1.0:
        depth = "高次意識"
    elif phi > 0.5:
        depth = "中程度意識"
    elif phi > 0.2:
        depth = "基本意識"
    else:
        depth = "前意識"
    
    complexity = "複雑" if len(text.split()) > 10 else "シンプル"
    generated_meaning = f"{depth}的{base_meaning}（{complexity}な表現）"
    
    # 結果表示
    print(f"📝 入力テキスト:")
    print(f"   '{text}'")
    print()
    
    print(f"🧠 意識分析結果:")
    print(f"   Φ値 (統合情報): {phi:.4f}")
    print(f"   意識レベル: {consciousness_state.consciousness_level:.3f}")
    print(f"   予測誤差: {prediction_state.total_error:.4f}")
    print(f"   結合強度: {coupling_strength:.3f}")
    print()
    
    print(f"🔍 現象学的解釈:")
    print(f"   {interpretation}")
    print()
    
    print(f"💭 生成された意味:")
    print(f"   {generated_meaning}")
    print()
    
    print(f"📊 特徴量詳細:")
    explanations = analyzer.get_feature_explanations()
    for i, (name, value) in enumerate(feature_dict.items()):
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"   {name:<12}: {bar} {value:.3f}")
    print()
    
    print("=" * 50)
    print("✅ 意識分析完了")
    
    return {
        'phi_value': phi,
        'consciousness_level': consciousness_state.consciousness_level,
        'generated_meaning': generated_meaning,
        'features': features,
        'interpretation': interpretation
    }


def main():
    """メイン実行"""
    if len(sys.argv) < 2:
        print("使用方法: python test_consciousness_analysis.py \"テキスト\"")
        print("例: python test_consciousness_analysis.py \"私は今日、美しい夕日を見ています。\"")
        return
    
    input_text = sys.argv[1]
    result = process_single_text(input_text)


if __name__ == "__main__":
    main()