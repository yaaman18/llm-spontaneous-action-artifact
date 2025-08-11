#!/usr/bin/env python3
"""
エナクティブ意識フレームワーク V3.0 - クイックスタート

最小限のコードでシステムを起動・実行するサンプル
"""

import numpy as np
from ngc_learn_adapter import HybridPredictiveCodingAdapter
from domain.value_objects.precision_weights import PrecisionWeights


def main():
    """最小限の実行例"""
    
    print("エナクティブ意識フレームワーク V3.0 - クイックスタート")
    print("=" * 50)
    
    # 1. システム初期化（3階層、10次元入力）
    adapter = HybridPredictiveCodingAdapter(
        hierarchy_levels=3,
        input_dimensions=10
    )
    print(f"✅ システム初期化完了: {adapter.engine_type}エンジン使用")
    
    # 2. 精度重みの設定（numpy配列として正しく初期化）
    precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
    print("✅ 精度重み設定完了")
    
    # 3. テスト入力データ生成
    input_data = np.random.rand(10)
    print("✅ 入力データ生成完了")
    
    # 4. 予測処理実行
    print("\n処理実行中...")
    prediction_state = adapter.process_input(input_data, precision_weights)
    
    # 5. 結果表示
    print("\n【実行結果】")
    print("-" * 30)
    print(f"予測エラー: {prediction_state.total_error:.4f}")
    print(f"収束状態: {prediction_state.convergence_status}")
    print(f"学習イテレーション: {prediction_state.learning_iteration}")
    print(f"階層数: {prediction_state.hierarchy_levels}")
    
    # 階層別エラー表示
    print("\n階層別エラー:")
    for i, error in enumerate(prediction_state.hierarchical_errors):
        print(f"  レベル{i}: {error:.4f}")
    
    # 成功判定
    if prediction_state.total_error < 100:
        print("\n✅ 処理成功！システムは正常に動作しています。")
    else:
        print("\n⚠️ エラーが大きいです。パラメータ調整が必要かもしれません。")
    
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit(main())