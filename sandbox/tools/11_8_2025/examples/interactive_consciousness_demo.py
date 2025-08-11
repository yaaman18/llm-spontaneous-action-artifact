#!/usr/bin/env python3
"""
対話的意識生成システム

自然言語入力から現象学的特徴量を抽出し、
エナクティブ意識状態を生成する対話システム。
"""

import sys
import numpy as np
from typing import Dict, List, Optional
sys.path.append('..')

from text_consciousness_features import PhenomenologicalTextAnalyzer, analyze_text_consciousness
from domain.factories.consciousness_factory import ConsciousnessFactory
from ngc_learn_adapter import HybridPredictiveCodingAdapter
from domain.value_objects.precision_weights import PrecisionWeights


class InteractiveConsciousnessSystem:
    """対話的意識生成システム"""
    
    def __init__(self):
        """システム初期化"""
        self.text_analyzer = PhenomenologicalTextAnalyzer()
        self.consciousness_factory = ConsciousnessFactory()
        self.adapter = HybridPredictiveCodingAdapter(3, 10)
        self.precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
        
        # 対話履歴
        self.dialogue_history = []
        self.consciousness_history = []
        
        print("🧠 対話的意識生成システム V3.0")
        print("=" * 50)
        print("現象学的テキスト分析による意識状態創発")
        print("フッサール＆メルロ＝ポンティ理論に基づく")
        print("=" * 50)
        print()
        print("使用方法:")
        print("- 任意のテキストを入力してください")
        print("- 'quit' または 'exit' で終了")
        print("- 'history' で対話履歴表示")
        print("- 'explain' で特徴量の詳細説明")
        print()
    
    def process_text_input(self, text: str) -> Dict:
        """
        テキスト入力から意識状態を生成
        
        Args:
            text: 入力テキスト
            
        Returns:
            処理結果辞書
        """
        # 1. 現象学的特徴量抽出
        features, feature_dict, interpretation = analyze_text_consciousness(text, verbose=False)
        
        # 2. 予測誤差の計算（特徴量から推定）
        prediction_errors = self._calculate_prediction_errors(features)
        
        # 3. 結合強度の計算
        coupling_strength = self._calculate_coupling_strength(features)
        
        # 4. 意識状態生成
        consciousness_state = self.consciousness_factory.create_emergent_consciousness_state(
            environmental_input=features,
            prediction_errors=prediction_errors,
            coupling_strength=coupling_strength
        )
        
        # 5. 予測処理実行
        prediction_state = self.adapter.process_input(features, self.precision_weights)
        
        # 6. 意味生成
        meaning = self._generate_meaning_from_consciousness(
            consciousness_state, features, text
        )
        
        # 7. 結果集約
        result = {
            'input_text': text,
            'features': features,
            'feature_dict': feature_dict,
            'interpretation': interpretation,
            'phi_value': consciousness_state.phi_value.value,
            'consciousness_level': consciousness_state.consciousness_level,
            'prediction_error': prediction_state.total_error,
            'generated_meaning': meaning,
            'coupling_strength': coupling_strength,
            'consciousness_state': consciousness_state
        }
        
        return result
    
    def _calculate_prediction_errors(self, features: np.ndarray) -> List[float]:
        """特徴量から予測誤差を推定"""
        # 特徴量の分散から誤差を計算
        variance = np.var(features)
        mean_feature = np.mean(features)
        
        # 階層的誤差の生成
        error_base = max(0.01, variance * 2)
        errors = [
            error_base * 1.5,  # 低次階層
            error_base * 1.0,  # 中次階層
            error_base * 0.5   # 高次階層
        ]
        
        return errors
    
    def _calculate_coupling_strength(self, features: np.ndarray) -> float:
        """特徴量から結合強度を計算"""
        # 複数の特徴量が活性化している場合、結合が強い
        active_features = np.sum(features > 0.1)
        max_coupling = 1.0
        min_coupling = 0.3
        
        # 活性化特徴数に基づく結合強度
        coupling = min_coupling + (active_features / 10) * (max_coupling - min_coupling)
        return min(coupling, max_coupling)
    
    def _generate_meaning_from_consciousness(
        self, 
        consciousness_state, 
        features: np.ndarray, 
        text: str
    ) -> str:
        """意識状態から意味を生成"""
        phi = consciousness_state.phi_value.value
        dominant_feature_idx = np.argmax(features)
        
        # 特徴量に基づく意味カテゴリ
        meaning_categories = {
            0: "志向的思考",  # 志向的方向性
            1: "時間的省察",  # 時間意識統合
            2: "身体的体験",  # 身体化認知
            3: "社会的交流",  # 間主観的共鳴
            4: "直感的理解",  # 前反省的気づき
            5: "創造的思考",  # 意味構成
            6: "日常的体験",  # 生活世界連関
            7: "自然な流れ",  # 受動的総合
            8: "文脈的理解",  # 地平構造
            9: "個人的体験"   # 一人称的視点
        }
        
        base_meaning = meaning_categories[dominant_feature_idx]
        
        # Φ値による意識の深度
        if phi > 1.0:
            depth = "高次意識"
        elif phi > 0.5:
            depth = "中程度意識"
        elif phi > 0.2:
            depth = "基本意識"
        else:
            depth = "前意識"
        
        # テキスト長による複雑性
        complexity = "複雑" if len(text.split()) > 10 else "シンプル"
        
        return f"{depth}的{base_meaning}（{complexity}な表現）"
    
    def display_result(self, result: Dict):
        """結果の詳細表示"""
        print(f"📝 入力: '{result['input_text']}'")
        print()
        print("🧠 意識分析結果:")
        print(f"  Φ値 (統合情報): {result['phi_value']:.4f}")
        print(f"  意識レベル: {result['consciousness_level']}")
        print(f"  予測誤差: {result['prediction_error']:.4f}")
        print(f"  結合強度: {result['coupling_strength']:.3f}")
        print()
        print("🔍 現象学的解釈:")
        print(f"  {result['interpretation']}")
        print()
        print("💭 生成された意味:")
        print(f"  {result['generated_meaning']}")
        print()
        print("📊 特徴量詳細:")
        for name, value in result['feature_dict'].items():
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {name:<12}: {bar} {value:.3f}")
        print()
    
    def show_feature_explanations(self):
        """特徴量の詳細説明表示"""
        explanations = self.text_analyzer.get_feature_explanations()
        print("📚 現象学的特徴量の詳細説明:")
        print("=" * 60)
        for i, explanation in explanations.items():
            print(f"{i+1:2d}. {explanation}")
        print("=" * 60)
        print()
    
    def show_history(self):
        """対話履歴の表示"""
        if not self.dialogue_history:
            print("対話履歴はありません。")
            return
        
        print("📜 対話履歴:")
        print("-" * 50)
        for i, (text, result) in enumerate(self.dialogue_history[-5:], 1):  # 最新5件
            print(f"{i}. '{text}' → Φ={result['phi_value']:.3f}, {result['generated_meaning']}")
        print("-" * 50)
        print()
    
    def run_interactive_session(self):
        """対話セッション実行"""
        while True:
            try:
                print("💬 ", end="")
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                # 特別コマンドの処理
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("対話を終了します。ありがとうございました！")
                    break
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'explain':
                    self.show_feature_explanations()
                    continue
                
                # テキスト処理
                print("🔄 意識状態を生成中...")
                result = self.process_text_input(user_input)
                
                # 履歴保存
                self.dialogue_history.append((user_input, result))
                self.consciousness_history.append(result['consciousness_state'])
                
                # 結果表示
                self.display_result(result)
                
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n対話を中断しました。")
                break
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                continue
    
    def batch_analysis(self, texts: List[str]):
        """バッチ分析（複数テキストの一括処理）"""
        print(f"📊 バッチ分析開始（{len(texts)}件のテキスト）")
        print("=" * 60)
        
        results = []
        for i, text in enumerate(texts, 1):
            print(f"処理中 {i}/{len(texts)}: '{text[:30]}...'")
            result = self.process_text_input(text)
            results.append(result)
            
            # 簡易結果表示
            print(f"  → Φ={result['phi_value']:.3f}, {result['generated_meaning']}")
            print()
        
        # 統計サマリー
        phi_values = [r['phi_value'] for r in results]
        print("📈 統計サマリー:")
        print(f"  平均Φ値: {np.mean(phi_values):.4f}")
        print(f"  最大Φ値: {np.max(phi_values):.4f}")
        print(f"  最小Φ値: {np.min(phi_values):.4f}")
        print(f"  標準偏差: {np.std(phi_values):.4f}")
        print("=" * 60)
        
        return results


def main():
    """メイン実行関数"""
    system = InteractiveConsciousnessSystem()
    
    # サンプルテキストでのデモ
    sample_texts = [
        "私は今日、とても美しい夕日を見て深く感動しました。",
        "We need to think carefully about our future together.",
        "なんとなく、この音楽には特別な意味があるような気がします。"
    ]
    
    print("🚀 サンプルテキストでのデモ:")
    print("-" * 40)
    for text in sample_texts:
        result = system.process_text_input(text)
        print(f"'{text}' → Φ={result['phi_value']:.3f}, {result['generated_meaning']}")
    print("-" * 40)
    print()
    
    # 対話セッション開始
    print("対話的セッションを開始します。")
    system.run_interactive_session()


if __name__ == "__main__":
    main()