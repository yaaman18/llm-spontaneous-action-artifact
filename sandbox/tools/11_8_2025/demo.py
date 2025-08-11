#!/usr/bin/env python3
"""
エナクティブ意識フレームワーク デモンストレーション

このスクリプトは、実装されたエナクティブ意識フレームワークの
主要機能を段階的にデモンストレーションします。

実行方法:
    python demo.py
    python demo.py --gui  # GUIモニター付き
"""

import sys
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# MatplotlibバックエンドをCLI用に設定（GUIなし）
import matplotlib
matplotlib.use('Agg')  # GUIを使わないバックエンド
import matplotlib.pyplot as plt

# 日本語フォントの設定（単一フォントで警告回避）
try:
    # 利用可能な日本語フォントを使用（macOSはHiragino Sansが確実）
    if sys.platform == "darwin":  # macOS
        plt.rcParams['font.family'] = ['Hiragino Sans']
    elif sys.platform == "win32":  # Windows  
        plt.rcParams['font.family'] = ['Yu Gothic']
    else:  # Linux
        plt.rcParams['font.family'] = ['DejaVu Sans']  # フォールバック
    
    # 日本語の負の値表示を修正
    plt.rcParams['axes.unicode_minus'] = False
    print("日本語フォント設定完了（CLI用バックエンド）")
    
except Exception as e:
    print(f"日本語フォント設定警告: {e}")
    plt.rcParams['axes.unicode_minus'] = False

# パスの追加
sys.path.append(str(Path(__file__).parent))

# ドメインモジュールのインポート
from domain.value_objects.phi_value import PhiValue
from domain.value_objects.consciousness_state import ConsciousnessState
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.probability_distribution import ProbabilityDistribution


def print_header(title: str):
    """セクションヘッダーを表示"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")


def demo_phi_values():
    """Φ値のデモンストレーション"""
    print_header("1. Φ値（統合情報理論）のデモ")
    
    # 異なるΦ値での意識状態
    phi_scenarios = [
        (0.0, 0.5, 0.0, "無意識状態"),
        (0.1, 0.8, 0.125, "微弱な意識"),
        (0.5, 1.0, 0.5, "明確な意識"),
        (1.2, 1.5, 0.8, "高次意識"),
    ]
    
    print(f"{'Φ値':<8} {'複雑性':<8} {'統合性':<8} {'状態':<12} {'意識判定'}")
    print("-" * 50)
    
    phi_objects = []
    for phi_val, complexity, integration, description in phi_scenarios:
        phi = PhiValue(value=phi_val, complexity=complexity, integration=integration)
        phi_objects.append(phi)
        consciousness_status = "有意識" if phi.is_conscious else "無意識"
        print(f"{phi_val:<8.1f} {complexity:<8.1f} {integration:<8.1f} {description:<12} {consciousness_status}")
    
    return phi_objects


def demo_prediction_states():
    """予測状態のデモンストレーション"""
    print_header("2. 階層的予測符号化のデモ")
    
    # 異なる学習段階での予測状態
    prediction_scenarios = [
        ([1.5, 0.8, 0.3], [1.5, 0.8, 0.3], "学習初期"),
        ([0.8, 0.4, 0.1], [0.8, 0.4, 0.1], "学習中期"), 
        ([0.2, 0.1, 0.05], [0.2, 0.1, 0.05], "学習後期"),
        ([0.05, 0.02, 0.01], [0.05, 0.02, 0.01], "収束状態"),
    ]
    
    print(f"{'段階':<12} {'総誤差':<10} {'平均誤差':<10} {'収束状態':<8} {'品質'}")
    print("-" * 55)
    
    prediction_objects = []
    for errors, precision_errors, stage in prediction_scenarios:
        pred_state = PredictionState(
            hierarchical_errors=errors,
            precision_weighted_errors=precision_errors,
            convergence_status="converged" if max(errors) < 0.1 else "converging",
            learning_iteration=np.random.randint(1, 100)
        )
        prediction_objects.append(pred_state)
        
        convergence = "収束" if pred_state.is_converged else "学習中"
        print(f"{stage:<12} {pred_state.total_error:<10.3f} {pred_state.mean_error:<10.3f} {convergence:<8} {pred_state.prediction_quality:<6.3f}")
    
    return prediction_objects


def demo_consciousness_states(phi_objects, prediction_objects):
    """意識状態の統合デモ"""
    print_header("3. 統合された意識状態のデモ")
    
    # 確率分布の作成
    prob_distributions = [
        ProbabilityDistribution(probabilities=np.array([0.7, 0.2, 0.1]), distribution_type="categorical"),
        ProbabilityDistribution(probabilities=np.array([0.5, 0.3, 0.2]), distribution_type="categorical"),
        ProbabilityDistribution(probabilities=np.array([0.4, 0.4, 0.2]), distribution_type="categorical"),
        ProbabilityDistribution(probabilities=np.array([0.33, 0.33, 0.34]), distribution_type="categorical"),
    ]
    
    print(f"{'シナリオ':<12} {'Φ値':<8} {'意識レベル':<12} {'メタ認知':<10} {'状態'}")
    print("-" * 55)
    
    consciousness_states = []
    scenarios = ["初期状態", "学習中", "安定期", "高次統合"]
    
    for i, scenario in enumerate(scenarios):
        # メタ認知信頼度を段階的に向上
        metacognitive_confidence = 0.2 + (i * 0.2)
        
        consciousness_state = ConsciousnessState(
            phi_value=phi_objects[i],
            prediction_state=prediction_objects[i], 
            uncertainty_distribution=prob_distributions[i],
            metacognitive_confidence=metacognitive_confidence
        )
        consciousness_states.append(consciousness_state)
        
        status = "有意識" if consciousness_state.is_conscious else "無意識"
        print(f"{scenario:<12} {consciousness_state.phi_value.value:<8.2f} {consciousness_state.consciousness_level:<12.3f} {metacognitive_confidence:<10.2f} {status}")
    
    return consciousness_states


def demo_jax_predictive_coding():
    """JAX予測符号化のデモ"""
    print_header("4. JAX予測符号化エンジンのデモ")
    
    try:
        from infrastructure.jax_predictive_coding_core import JaxPredictiveCodingCore
        import jax.random as random
        
        print("JAX予測符号化コアを初期化中...")
        
        # 予測符号化コアの初期化
        key = random.PRNGKey(42)
        layer_dims = [8, 6, 4]  # 3層ネットワーク
        
        core = JaxPredictiveCodingCore(
            layer_dimensions=layer_dims,
            learning_rate=0.01,
            precision_adaptation_rate=0.001,
            random_key=key
        )
        
        print(f"初期化完了: {len(layer_dims)}層ネットワーク ({layer_dims})")
        
        # 学習プロセスのシミュレーション
        print("\n学習プロセスをシミュレーション中...")
        print(f"{'エポック':<8} {'自由エネルギー':<15} {'総誤差':<10} {'平均精度'}")
        print("-" * 45)
        
        free_energies = []
        errors = []
        
        for epoch in range(5):
            # ランダム入力データ
            input_data = np.random.randn(8) * 0.5 + 0.5
            
            # 予測符号化処理
            prediction_state = core.process_input(input_data)
            
            # メトリクス取得
            free_energy = prediction_state.metadata.get('free_energy', 0.0)
            total_error = prediction_state.total_error
            
            free_energies.append(float(free_energy))
            errors.append(total_error)
            
            # 精度重みの平均
            precision_weights = prediction_state.metadata.get('precision_weights', [1.0])
            avg_precision = np.mean(precision_weights) if precision_weights else 1.0
            
            print(f"{epoch+1:<8} {free_energy:<15.6f} {total_error:<10.6f} {avg_precision:<10.6f}")
        
        # 学習曲線の表示
        if len(free_energies) > 1:
            print(f"\n学習進捗:")
            print(f"  初期自由エネルギー: {free_energies[0]:.6f}")
            print(f"  最終自由エネルギー: {free_energies[-1]:.6f}")
            print(f"  誤差改善: {errors[0]:.6f} → {errors[-1]:.6f}")
            improvement = ((errors[0] - errors[-1]) / errors[0]) * 100 if errors[0] > 0 else 0
            print(f"  改善率: {improvement:.1f}%")
        
        return True
        
    except ImportError as e:
        print(f"JAXライブラリが見つかりません: {e}")
        print("requirements.txtを確認してJAXをインストールしてください")
        return False
    except Exception as e:
        print(f"JAX予測符号化デモでエラーが発生: {e}")
        return False


def demo_visualization(consciousness_states):
    """意識状態の可視化"""
    print_header("5. 意識状態の可視化")
    
    # データ抽出
    phi_values = [cs.phi_value.value for cs in consciousness_states]
    consciousness_levels = [cs.consciousness_level for cs in consciousness_states]
    metacognitive_conf = [cs.metacognitive_confidence for cs in consciousness_states]
    scenarios = ["初期状態", "学習中", "安定期", "高次統合"]
    
    # プロット作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('エナクティブ意識フレームワーク - 状態変化', fontsize=16)
    
    # Φ値の変化
    ax1.plot(range(len(phi_values)), phi_values, 'b-o', linewidth=2, markersize=8)
    ax1.set_title('Φ値の変化')
    ax1.set_ylabel('Φ値')
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels(scenarios, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 意識レベルの変化
    ax2.plot(range(len(consciousness_levels)), consciousness_levels, 'g-o', linewidth=2, markersize=8)
    ax2.set_title('意識レベルの変化')
    ax2.set_ylabel('意識レベル')
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels(scenarios, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # メタ認知信頼度
    ax3.bar(range(len(metacognitive_conf)), metacognitive_conf, color='purple', alpha=0.7)
    ax3.set_title('メタ認知信頼度')
    ax3.set_ylabel('信頼度')
    ax3.set_xticks(range(len(scenarios)))
    ax3.set_xticklabels(scenarios, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 統合指標（レーダーチャート風）
    categories = ['Φ値', '意識レベル', 'メタ認知', '予測品質']
    final_state = consciousness_states[-1]
    values = [
        min(final_state.phi_value.value / 2.0, 1.0),  # Φ値正規化
        final_state.consciousness_level,
        final_state.metacognitive_confidence,
        final_state.prediction_state.prediction_quality
    ]
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # 円を閉じる
    angles += angles[:1]
    
    ax4.plot(angles, values, 'ro-', linewidth=2)
    ax4.fill(angles, values, alpha=0.25, color='red')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_title('最終状態の統合指標')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # 保存（GUIは表示しない）
    plt.savefig('consciousness_demo_results.png', dpi=150, bbox_inches='tight')
    print("可視化結果を 'consciousness_demo_results.png' に保存しました")
    plt.close()  # メモリ解放
    
    # CLIモードではファイル保存のみ
    print("CLIモードのため、グラフはファイルとして保存されました")


def main():
    """メインデモ実行"""
    print_header("エナクティブ意識フレームワーク デモンストレーション")
    print("このデモは実装された主要機能を段階的に紹介します\n")
    
    start_time = time.time()
    
    # 1. Φ値のデモ
    phi_objects = demo_phi_values()
    
    # 2. 予測状態のデモ
    prediction_objects = demo_prediction_states()
    
    # 3. 意識状態統合のデモ
    consciousness_states = demo_consciousness_states(phi_objects, prediction_objects)
    
    # 4. JAX予測符号化のデモ
    jax_success = demo_jax_predictive_coding()
    
    # 5. 可視化デモ
    try:
        demo_visualization(consciousness_states)
    except Exception as e:
        print(f"可視化デモでエラー: {e}")
    
    # 実行時間とサマリー
    execution_time = time.time() - start_time
    
    print_header("デモ実行サマリー")
    print(f"実行時間: {execution_time:.2f}秒")
    print(f"Φ値シナリオ: {len(phi_objects)}個")
    print(f"予測状態: {len(prediction_objects)}個") 
    print(f"意識状態: {len(consciousness_states)}個")
    print(f"JAX予測符号化: {'✓ 成功' if jax_success else '✗ 失敗'}")
    
    # 最終状態の詳細
    if consciousness_states:
        final_state = consciousness_states[-1]
        print(f"\n最終意識状態の詳細:")
        print(f"  - Φ値: {final_state.phi_value.value:.3f}")
        print(f"  - 意識レベル: {final_state.consciousness_level:.3f}")
        print(f"  - 意識状態: {'有意識' if final_state.is_conscious else '無意識'}")
        print(f"  - メタ認知信頼度: {final_state.metacognitive_confidence:.3f}")
        print(f"  - 予測品質: {final_state.prediction_state.prediction_quality:.3f}")
    
    print_header("デモ完了")
    print("実装されたエナクティブ意識フレームワークの機能確認が完了しました")
    print("詳細な動作確認にはGUIモニターを使用してください:")
    print("  python main.py --gui")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="エナクティブ意識フレームワーク デモ")
    parser.add_argument("--gui", action="store_true", help="デモ後にGUIモニターを起動")
    
    args = parser.parse_args()
    
    # メインデモ実行
    main()
    
    # GUI起動（オプション）
    if args.gui:
        try:
            from gui.consciousness_monitor import ConsciousnessMonitor
            print("\n" + "="*60)
            print("GUIモニターを起動します...")
            app = ConsciousnessMonitor()
            app.run()
        except Exception as e:
            print(f"GUI起動エラー: {e}")
            print("GUIモニターは別途起動してください: python main.py --gui")