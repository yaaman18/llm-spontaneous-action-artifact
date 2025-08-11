"""
エナクティブ意識フレームワーク用SOM（自己組織化マップ）実装

理論的基盤：
1. エナクティビズム: 認知は環境との構造的カップリングによる自己生成
2. 現象学的基盤: 体験の構造的組織化としてのSOM
3. 予測符号化統合: 空間的予測表象の生成機構
4. 意識レベル統合: 組織化度・分化性・動的安定性の定量化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

class ConsciousnessLevel(Enum):
    """意識レベルの定義"""
    MINIMAL = 1      # 最小意識
    BASIC = 2        # 基本意識  
    INTEGRATED = 3   # 統合意識
    REFLECTIVE = 4   # 反省的意識

@dataclass
class EnactiveState:
    """エナクティブ状態の表現"""
    sensory_input: np.ndarray
    prediction: np.ndarray
    action_tendency: np.ndarray
    phenomenological_quality: float
    temporal_coherence: float

@dataclass
class ConsciousnessMetrics:
    """意識レベル計算用メトリクス"""
    integration: float      # 統合性
    differentiation: float  # 分化性
    information_flow: float # 情報流動性
    temporal_coherence: float # 時間的一貫性
    som_organization: float   # SOM組織化度
    
    def compute_level(self) -> Tuple[ConsciousnessLevel, float]:
        """5つの指標から意識レベルを計算"""
        score = (self.integration + self.differentiation + 
                self.information_flow + self.temporal_coherence + 
                self.som_organization) / 5.0
        
        if score >= 0.8:
            return ConsciousnessLevel.REFLECTIVE, score
        elif score >= 0.6:
            return ConsciousnessLevel.INTEGRATED, score
        elif score >= 0.4:
            return ConsciousnessLevel.BASIC, score
        else:
            return ConsciousnessLevel.MINIMAL, score

class EnactiveSOM:
    """
    エナクティブ意識統合型自己組織化マップ
    
    特徴:
    - 予測符号化との統合
    - 現象学的品質の保持
    - 動的適応機構
    - 意識レベルへの寄与計算
    """
    
    def __init__(self, 
                 map_size: Tuple[int, int] = (10, 10),
                 input_dim: int = 100,
                 learning_rate: float = 0.1,
                 sigma_initial: float = 2.0,
                 sigma_decay: float = 0.99,
                 predictive_weight: float = 0.3):
        
        self.map_size = map_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma_initial
        self.sigma_decay = sigma_decay
        self.predictive_weight = predictive_weight
        
        # SOMの重みマトリックス初期化
        self.weights = np.random.randn(map_size[0], map_size[1], input_dim) * 0.1
        
        # エナクティブ拡張
        self.phenomenological_map = np.zeros(map_size)  # 現象学的品質マップ
        self.prediction_errors = np.zeros(map_size)     # 予測誤差マップ
        self.activation_history = []                    # 活性化履歴
        self.temporal_coherence = 0.0                   # 時間的一貫性
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
    def find_bmu(self, input_vector: np.ndarray) -> Tuple[int, int]:
        """最良適合ユニット（BMU）を見つける"""
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), self.map_size)
        return bmu_idx
    
    def _gaussian_kernel(self, center: Tuple[int, int], sigma: float) -> np.ndarray:
        """ガウシアン近傍関数"""
        x, y = np.meshgrid(range(self.map_size[1]), range(self.map_size[0]))
        d_squared = (x - center[1])**2 + (y - center[0])**2
        return np.exp(-d_squared / (2 * sigma**2))
    
    def update_with_prediction(self, 
                             sensory_input: np.ndarray,
                             prediction: np.ndarray,
                             prediction_error: float) -> Tuple[int, int]:
        """
        予測符号化統合型更新
        
        Args:
            sensory_input: 感覚入力
            prediction: トップダウン予測
            prediction_error: 予測誤差
        """
        # エナクティブ統合入力の生成
        integrated_input = (1 - self.predictive_weight) * sensory_input + \
                          self.predictive_weight * prediction
        
        # BMU発見
        bmu = self.find_bmu(integrated_input)
        
        # 近傍関数計算
        neighborhood = self._gaussian_kernel(bmu, self.sigma)
        
        # 重み更新
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                learning_influence = self.learning_rate * neighborhood[i, j]
                self.weights[i, j] += learning_influence * (integrated_input - self.weights[i, j])
        
        # エナクティブ属性更新
        self.prediction_errors[bmu] = prediction_error
        self._update_phenomenological_quality(bmu, sensory_input, prediction)
        self._update_temporal_coherence(bmu)
        
        # パラメータ減衰
        self.sigma *= self.sigma_decay
        
        return bmu
    
    def _update_phenomenological_quality(self, 
                                       bmu: Tuple[int, int],
                                       sensory_input: np.ndarray,
                                       prediction: np.ndarray):
        """現象学的品質の更新"""
        # 感覚-予測の一致度を現象学的品質として計算
        coherence = 1.0 - np.linalg.norm(sensory_input - prediction) / \
                   (np.linalg.norm(sensory_input) + np.linalg.norm(prediction) + 1e-8)
        
        self.phenomenological_map[bmu] = 0.9 * self.phenomenological_map[bmu] + \
                                       0.1 * coherence
    
    def _update_temporal_coherence(self, bmu: Tuple[int, int]):
        """時間的一貫性の更新"""
        self.activation_history.append(bmu)
        
        # 直近の活性化パターンから一貫性を計算
        if len(self.activation_history) > 10:
            self.activation_history = self.activation_history[-10:]
            
        if len(self.activation_history) > 1:
            distances = []
            for i in range(1, len(self.activation_history)):
                prev_bmu = self.activation_history[i-1]
                curr_bmu = self.activation_history[i]
                dist = np.sqrt((curr_bmu[0] - prev_bmu[0])**2 + 
                             (curr_bmu[1] - prev_bmu[1])**2)
                distances.append(dist)
            
            # 距離の逆数で一貫性を計算（近い位置ほど一貫性が高い）
            avg_distance = np.mean(distances)
            self.temporal_coherence = 1.0 / (1.0 + avg_distance)
    
    def compute_organization_metrics(self) -> Dict[str, float]:
        """SOM組織化メトリクスの計算"""
        # トポロジカル保存性
        topological_error = self._compute_topological_error()
        
        # 量子化誤差
        quantization_error = self._compute_quantization_error()
        
        # 現象学的一貫性
        phenomenological_coherence = np.mean(self.phenomenological_map)
        
        # 組織化度（3つの指標の統合）
        organization_score = (1.0 - topological_error) * 0.4 + \
                           (1.0 - quantization_error) * 0.3 + \
                           phenomenological_coherence * 0.3
        
        return {
            'topological_error': topological_error,
            'quantization_error': quantization_error,
            'phenomenological_coherence': phenomenological_coherence,
            'organization_score': organization_score,
            'temporal_coherence': self.temporal_coherence
        }
    
    def _compute_topological_error(self) -> float:
        """トポロジカル誤差の計算"""
        if len(self.activation_history) < 2:
            return 0.0
            
        errors = 0
        for i in range(1, len(self.activation_history)):
            curr_bmu = self.activation_history[i]
            prev_bmu = self.activation_history[i-1]
            
            # 隣接していない場合はエラー
            manhattan_dist = abs(curr_bmu[0] - prev_bmu[0]) + abs(curr_bmu[1] - prev_bmu[1])
            if manhattan_dist > 2:  # 2次近傍まで許容
                errors += 1
        
        return errors / (len(self.activation_history) - 1)
    
    def _compute_quantization_error(self) -> float:
        """量子化誤差の計算"""
        if not hasattr(self, '_last_inputs') or len(self._last_inputs) == 0:
            return 0.0
            
        total_error = 0.0
        for input_vec in self._last_inputs:
            bmu = self.find_bmu(input_vec)
            error = np.linalg.norm(input_vec - self.weights[bmu])
            total_error += error
        
        return total_error / len(self._last_inputs)
    
    def get_consciousness_contribution(self) -> float:
        """意識レベル計算への寄与度を返す"""
        metrics = self.compute_organization_metrics()
        return metrics['organization_score']
    
    def visualize_state(self) -> Dict[str, np.ndarray]:
        """現在の状態を可視化用に返す"""
        return {
            'weights': self.weights,
            'phenomenological_map': self.phenomenological_map,
            'prediction_errors': self.prediction_errors,
            'activation_history': np.array(self.activation_history) if self.activation_history else np.array([])
        }

class EnactiveConsciousnessFramework:
    """
    エナクティブ意識フレームワーク統合クラス
    SOM、予測符号化、現象学的基盤を統合
    """
    
    def __init__(self,
                 som_config: Optional[Dict] = None,
                 consciousness_threshold: float = 0.5):
        
        # SOM初期化
        som_config = som_config or {}
        self.som = EnactiveSOM(**som_config)
        
        self.consciousness_threshold = consciousness_threshold
        self.current_state = None
        self.consciousness_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def process_enactive_experience(self, 
                                  sensory_input: np.ndarray,
                                  prediction: np.ndarray,
                                  action_tendency: np.ndarray) -> EnactiveState:
        """
        エナクティブ体験の処理
        
        Args:
            sensory_input: 感覚入力
            prediction: トップダウン予測
            action_tendency: 行動傾向
        """
        # 予測誤差計算
        prediction_error = np.linalg.norm(sensory_input - prediction)
        
        # SOM更新
        bmu = self.som.update_with_prediction(sensory_input, prediction, prediction_error)
        
        # 現象学的品質計算
        phenomenological_quality = self.som.phenomenological_map[bmu]
        
        # 時間的一貫性
        temporal_coherence = self.som.temporal_coherence
        
        # エナクティブ状態構築
        state = EnactiveState(
            sensory_input=sensory_input,
            prediction=prediction,
            action_tendency=action_tendency,
            phenomenological_quality=phenomenological_quality,
            temporal_coherence=temporal_coherence
        )
        
        self.current_state = state
        return state
    
    def compute_consciousness_level(self) -> Tuple[ConsciousnessLevel, float]:
        """意識レベルの計算"""
        if self.current_state is None:
            return ConsciousnessLevel.MINIMAL, 0.0
        
        # SOM組織化メトリクス
        som_metrics = self.som.compute_organization_metrics()
        
        # 意識メトリクス構築
        metrics = ConsciousnessMetrics(
            integration=self._compute_integration(),
            differentiation=self._compute_differentiation(),
            information_flow=self._compute_information_flow(),
            temporal_coherence=self.current_state.temporal_coherence,
            som_organization=som_metrics['organization_score']
        )
        
        level, score = metrics.compute_level()
        self.consciousness_history.append((level, score))
        
        self.logger.info(f"Consciousness Level: {level.name}, Score: {score:.3f}")
        return level, score
    
    def _compute_integration(self) -> float:
        """統合性の計算"""
        if self.current_state is None:
            return 0.0
        
        # 感覚-予測-行動の統合度
        sensory_norm = np.linalg.norm(self.current_state.sensory_input)
        prediction_norm = np.linalg.norm(self.current_state.prediction)
        action_norm = np.linalg.norm(self.current_state.action_tendency)
        
        total_norm = sensory_norm + prediction_norm + action_norm
        if total_norm == 0:
            return 0.0
        
        # バランスの取れた統合を評価
        balance = 1.0 - np.std([sensory_norm, prediction_norm, action_norm]) / (total_norm / 3 + 1e-8)
        return max(0.0, balance)
    
    def _compute_differentiation(self) -> float:
        """分化性の計算"""
        som_metrics = self.som.compute_organization_metrics()
        return 1.0 - som_metrics['topological_error']
    
    def _compute_information_flow(self) -> float:
        """情報流動性の計算"""
        if len(self.consciousness_history) < 2:
            return 0.5
        
        # 意識レベルの変動から情報流動性を推定
        recent_scores = [score for _, score in self.consciousness_history[-10:]]
        if len(recent_scores) > 1:
            variability = np.std(recent_scores)
            # 適度な変動が情報流動性を示す
            return min(1.0, variability * 2.0)
        
        return 0.5

# 使用例とテスト
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # フレームワーク初期化
    framework = EnactiveConsciousnessFramework()
    
    # シミュレーション実行
    for step in range(100):
        # ランダムな入力生成（実際の使用では外部から供給）
        sensory_input = np.random.randn(100)
        prediction = sensory_input + np.random.randn(100) * 0.1  # 小さなノイズ付き予測
        action_tendency = np.random.randn(100) * 0.5
        
        # エナクティブ体験処理
        state = framework.process_enactive_experience(
            sensory_input, prediction, action_tendency
        )
        
        # 意識レベル計算
        level, score = framework.compute_consciousness_level()
        
        if step % 20 == 0:
            print(f"Step {step}: Level={level.name}, Score={score:.3f}, "
                  f"Phenomenological Quality={state.phenomenological_quality:.3f}")
    
    print("\n=== Final SOM State ===")
    som_metrics = framework.som.compute_organization_metrics()
    for key, value in som_metrics.items():
        print(f"{key}: {value:.3f}")