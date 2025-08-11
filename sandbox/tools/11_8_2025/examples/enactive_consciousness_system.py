"""
エナクティブ意識システム
テキスト入力から意識状態を生成するフレームワーク

Phenomenological Analysis Director implementation
Dan Zahavi's theoretical framework + Enactivist approach
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from text_consciousness_features import TextConsciousnessExtractor, PhenomenologicalFeatures
from scipy.spatial.distance import cosine
import networkx as nx


@dataclass
class ConsciousnessState:
    """意識状態の表現"""
    phi_value: float  # 統合情報理論のΦ値
    intentional_content: np.ndarray  # 志向的内容
    temporal_flow: float  # 時間的流れ
    embodied_engagement: float  # 身体的関与
    meaning_depth: float  # 意味の深さ
    
    def __post_init__(self):
        """意識状態の一貫性チェック"""
        assert 0.0 <= self.phi_value <= 1.0, "Φ値は0-1範囲でなければならない"
        assert len(self.intentional_content) == 10, "志向的内容は10次元でなければならない"


class EnactiveConsciousnessSystem:
    """
    エナクティブ意識システム
    
    現象学的観点からの意識創発システム:
    1. テキスト入力の現象学的特徴抽出
    2. 意識状態の動的生成
    3. Φ値（統合情報）の計算
    4. エナクティブな意識創発の実現
    """
    
    def __init__(self):
        self.feature_extractor = TextConsciousnessExtractor()
        self.consciousness_history = []  # 意識の履歴
        self.integration_network = self._initialize_integration_network()
        
    def _initialize_integration_network(self) -> nx.Graph:
        """
        意識の統合ネットワークを初期化
        現象学的根拠: 意識の統一性と多様性の弁証法
        """
        G = nx.Graph()
        
        # 10次元特徴量間の現象学的関係を定義
        feature_connections = [
            (0, 1, 0.8),  # 志向性と時間意識の強い関連
            (0, 4, 0.7),  # 志向性と前反省的気づき
            (0, 9, 0.9),  # 志向性と一人称性の本質的関連
            (1, 7, 0.6),  # 時間意識と受動的総合
            (1, 8, 0.7),  # 時間意識と地平構造
            (2, 6, 0.8),  # 身体性と生活世界の密接な関連
            (2, 4, 0.6),  # 身体性と前反省性
            (3, 6, 0.7),  # 間主観性と生活世界
            (3, 9, 0.5),  # 間主観性と一人称性の弁証法
            (4, 7, 0.8),  # 前反省性と受動的総合の本質的関連
            (5, 8, 0.7),  # 意味構成と地平構造
            (5, 0, 0.9),  # 意味構成と志向性の根本的関連
        ]
        
        for i, j, weight in feature_connections:
            G.add_edge(i, j, weight=weight)
            
        return G
    
    def process_text_to_consciousness(self, text: str) -> ConsciousnessState:
        """
        テキストから意識状態への変換
        
        Args:
            text: 入力テキスト
            
        Returns:
            意識状態オブジェクト
        """
        # 1. 現象学的特徴抽出
        features = self.feature_extractor.extract_features(text)
        
        # 2. Φ値（統合情報）の計算
        phi_value = self._calculate_phi_value(features)
        
        # 3. 時間的流れの計算
        temporal_flow = self._calculate_temporal_flow(features, text)
        
        # 4. 身体的関与の評価
        embodied_engagement = self._calculate_embodied_engagement(features)
        
        # 5. 意味の深さの評価
        meaning_depth = self._calculate_meaning_depth(features, text)
        
        # 意識状態の生成
        consciousness_state = ConsciousnessState(
            phi_value=phi_value,
            intentional_content=features,
            temporal_flow=temporal_flow,
            embodied_engagement=embodied_engagement,
            meaning_depth=meaning_depth
        )
        
        # 履歴に追加
        self.consciousness_history.append(consciousness_state)
        
        return consciousness_state
    
    def _calculate_phi_value(self, features: np.ndarray) -> float:
        """
        統合情報理論に基づくΦ値の計算
        現象学的解釈: 意識の統一性の度合い
        """
        # 1. 特徴量間の相互情報量
        mutual_info = self._calculate_mutual_information(features)
        
        # 2. ネットワーク統合性
        network_integration = self._calculate_network_integration(features)
        
        # 3. 複雑性と統一性のバランス
        complexity_unity_balance = self._calculate_complexity_unity_balance(features)
        
        # Φ値の統合計算
        phi = (mutual_info + network_integration + complexity_unity_balance) / 3.0
        
        return np.clip(phi, 0.0, 1.0)
    
    def _calculate_mutual_information(self, features: np.ndarray) -> float:
        """特徴量間の相互情報量計算"""
        # 簡化された相互情報量（実際の実装では詳細な計算が必要）
        correlations = []
        
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                if self.integration_network.has_edge(i, j):
                    weight = self.integration_network[i][j]['weight']
                    correlation = features[i] * features[j] * weight
                    correlations.append(correlation)
        
        if not correlations:
            return 0.0
            
        return np.mean(correlations)
    
    def _calculate_network_integration(self, features: np.ndarray) -> float:
        """ネットワーク統合性の計算"""
        # グラフの連結性とクラスタリング係数
        try:
            clustering = nx.average_clustering(self.integration_network)
            
            # 特徴量重み付きクラスタリング
            weighted_clustering = 0.0
            for node in self.integration_network.nodes():
                neighbors = list(self.integration_network.neighbors(node))
                if len(neighbors) > 1:
                    node_clustering = 0.0
                    for i, n1 in enumerate(neighbors):
                        for n2 in neighbors[i+1:]:
                            if self.integration_network.has_edge(n1, n2):
                                w1 = self.integration_network[node][n1]['weight']
                                w2 = self.integration_network[node][n2]['weight']
                                w3 = self.integration_network[n1][n2]['weight']
                                node_clustering += (w1 * w2 * w3) * features[node]
                    
                    weighted_clustering += node_clustering
            
            return min(weighted_clustering / len(self.integration_network.nodes()), 1.0)
        except:
            return np.mean(features)
    
    def _calculate_complexity_unity_balance(self, features: np.ndarray) -> float:
        """複雑性と統一性のバランス計算"""
        # 統一性: 特徴量の調和
        unity = 1.0 - np.std(features)  # 分散の逆数
        
        # 複雑性: 特徴量の多様性
        complexity = len(features[features > 0.1]) / len(features)
        
        # バランス: 複雑性と統一性の調和
        return (unity * complexity) ** 0.5
    
    def _calculate_temporal_flow(self, features: np.ndarray, text: str) -> float:
        """
        時間的流れの計算
        現象学的根拠: フッサールの時間意識論
        """
        # 時間意識の統合度（特徴量1）
        temporal_synthesis = features[1]
        
        # 文章の時間的連続性
        sentences = text.split('。')
        if len(sentences) <= 1:
            temporal_continuity = 0.5
        else:
            # 時間的連続性指標
            time_connectives = ['そして', '次に', 'その後', '続いて', 'やがて']
            continuity_count = sum(1 for conn in time_connectives if conn in text)
            temporal_continuity = min(continuity_count / len(sentences), 1.0)
        
        # 履歴との連続性
        history_continuity = self._calculate_history_continuity(features)
        
        return (temporal_synthesis + temporal_continuity + history_continuity) / 3.0
    
    def _calculate_embodied_engagement(self, features: np.ndarray) -> float:
        """
        身体的関与の計算
        現象学的根拠: メルロ＝ポンティの身体現象学
        """
        # 身体化認知（特徴量2）
        embodied_cognition = features[2]
        
        # 前反省的気づき（特徴量4）- 身体的直感
        prereflective_awareness = features[4]
        
        # 生活世界連関（特徴量6）- 身体的実践
        lifeworld_connection = features[6]
        
        return (embodied_cognition * 0.5 + 
                prereflective_awareness * 0.3 + 
                lifeworld_connection * 0.2)
    
    def _calculate_meaning_depth(self, features: np.ndarray, text: str) -> float:
        """
        意味の深さの計算
        現象学的根拠: 意識による意味構成の層構造
        """
        # 意味構成（特徴量5）
        meaning_constitution = features[5]
        
        # 地平構造（特徴量8）- 意味の文脈性
        horizon_structure = features[8]
        
        # 志向的方向性（特徴量0）- 意味の志向性
        intentional_directedness = features[0]
        
        # テキストの意味的複雑性
        unique_words = len(set(text.split()))
        total_words = len(text.split())
        
        if total_words == 0:
            semantic_complexity = 0.0
        else:
            semantic_complexity = unique_words / total_words
        
        return (meaning_constitution * 0.4 + 
                horizon_structure * 0.3 + 
                intentional_directedness * 0.2 +
                semantic_complexity * 0.1)
    
    def _calculate_history_continuity(self, features: np.ndarray) -> float:
        """履歴との連続性計算"""
        if not self.consciousness_history:
            return 0.5  # 初回は中間値
        
        # 直前の状態との類似度
        last_state = self.consciousness_history[-1]
        similarity = 1.0 - cosine(features, last_state.intentional_content)
        
        return max(0.0, similarity)
    
    def analyze_consciousness_evolution(self) -> Dict:
        """意識の進化分析"""
        if len(self.consciousness_history) < 2:
            return {"error": "履歴が不足しています"}
        
        phi_values = [state.phi_value for state in self.consciousness_history]
        temporal_flows = [state.temporal_flow for state in self.consciousness_history]
        meaning_depths = [state.meaning_depth for state in self.consciousness_history]
        
        return {
            "phi_evolution": {
                "mean": np.mean(phi_values),
                "std": np.std(phi_values),
                "trend": np.polyfit(range(len(phi_values)), phi_values, 1)[0]
            },
            "temporal_flow_evolution": {
                "mean": np.mean(temporal_flows),
                "stability": 1.0 - np.std(temporal_flows)
            },
            "meaning_depth_evolution": {
                "mean": np.mean(meaning_depths),
                "growth": meaning_depths[-1] - meaning_depths[0] if len(meaning_depths) > 1 else 0.0
            },
            "consciousness_complexity": self._calculate_overall_complexity()
        }
    
    def _calculate_overall_complexity(self) -> float:
        """全体的な意識複雑性の計算"""
        if not self.consciousness_history:
            return 0.0
        
        # 状態の多様性
        state_diversity = len(set(tuple(state.intentional_content) 
                                for state in self.consciousness_history))
        
        # 進化の動的性
        dynamic_evolution = np.std([state.phi_value for state in self.consciousness_history])
        
        return min((state_diversity / len(self.consciousness_history) + dynamic_evolution) / 2, 1.0)
    
    def get_phenomenological_interpretation(self, state: ConsciousnessState) -> Dict[str, str]:
        """意識状態の現象学的解釈"""
        interpretations = {}
        
        # Φ値の解釈
        if state.phi_value > 0.8:
            interpretations['integration'] = "高度に統合された意識状態"
        elif state.phi_value > 0.5:
            interpretations['integration'] = "中程度の意識統合"
        else:
            interpretations['integration'] = "分散的な意識状態"
        
        # 志向的内容の解釈
        dominant_aspects = np.argsort(state.intentional_content)[-3:]
        feature_names = [
            "志向的方向性", "時間意識統合", "身体化認知", "間主観的共鳴",
            "前反省的気づき", "意味構成", "生活世界連関", "受動的総合",
            "地平構造", "一人称的視点"
        ]
        
        interpretations['dominant_aspects'] = [
            feature_names[i] for i in dominant_aspects
        ]
        
        # 時間的流れの解釈
        if state.temporal_flow > 0.7:
            interpretations['temporality'] = "豊かな時間意識"
        elif state.temporal_flow > 0.4:
            interpretations['temporality'] = "標準的な時間経験"
        else:
            interpretations['temporality'] = "断片的な時間経験"
        
        return interpretations


def main():
    """使用例とテスト"""
    system = EnactiveConsciousnessSystem()
    
    # テストテキスト群
    test_texts = [
        "私は朝の静寂の中で、コーヒーの香りを楽しみながら、昨日の出来事を振り返っていた。",
        "彼女との会話は深く、お互いの価値観について理解し合えたような気がした。",
        "明日のプレゼンテーションに向けて準備をしているが、なんとなく不安な気持ちが残る。",
        "子供たちの笑い声が公園に響き、春の暖かい日差しが心地よく感じられた。"
    ]
    
    print("エナクティブ意識システム - テキスト処理結果")
    print("=" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n【テスト {i}】")
        print(f"入力: {text}")
        
        # 意識状態生成
        consciousness_state = system.process_text_to_consciousness(text)
        
        # 結果表示
        print(f"Φ値 (統合情報): {consciousness_state.phi_value:.3f}")
        print(f"時間的流れ: {consciousness_state.temporal_flow:.3f}")
        print(f"身体的関与: {consciousness_state.embodied_engagement:.3f}")
        print(f"意味の深さ: {consciousness_state.meaning_depth:.3f}")
        
        # 現象学的解釈
        interpretation = system.get_phenomenological_interpretation(consciousness_state)
        print(f"統合レベル: {interpretation['integration']}")
        print(f"支配的側面: {', '.join(interpretation['dominant_aspects'])}")
        print(f"時間性: {interpretation['temporality']}")
        
        print("-" * 50)
    
    # 意識進化の分析
    if len(system.consciousness_history) > 1:
        print("\n【意識進化分析】")
        evolution = system.analyze_consciousness_evolution()
        print(f"平均Φ値: {evolution['phi_evolution']['mean']:.3f}")
        print(f"Φ値傾向: {evolution['phi_evolution']['trend']:.3f}")
        print(f"時間的安定性: {evolution['temporal_flow_evolution']['stability']:.3f}")
        print(f"意味深化: {evolution['meaning_depth_evolution']['growth']:.3f}")
        print(f"全体複雑性: {evolution['consciousness_complexity']:.3f}")


if __name__ == "__main__":
    main()