"""
エナクティビズム観点での自己組織化マップ実装
吉田正俊のエナクティブ神経科学とTaguchi Shigeruの現象学的基盤に基づく

コア原理:
1. 能動的知覚: SOMは環境との相互作用を通じて形成される
2. 身体化された認知: 運動-感覚統合による意味の創発
3. 予測的処理: 概念空間は予測誤差最小化構造として組織化
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jit

from .value_objects import PhiValue
from .entities import ConsciousnessState


@dataclass
class EnactivePattern:
    """エナクティブパターン：行為-知覚の統合単位"""
    sensory_state: jnp.ndarray
    motor_intention: jnp.ndarray
    prediction_error: float
    affordance_weight: float
    embodiment_context: Dict[str, Any]


@dataclass
class ConceptualAffordance:
    """概念的アフォーダンス：行為可能性の認知的拡張"""
    concept_vector: jnp.ndarray
    action_potential: jnp.ndarray
    contextual_relevance: float
    temporal_horizon: float


class EnactiveSOM(ABC):
    """
    エナクティブ自己組織化マップの抽象基底クラス
    
    従来のSOMと異なり：
    - 静的マッピングではなく動的行為-知覚統合
    - 受動的学習ではなく能動的環境探索
    - 表現の獲得ではなく意味の創発
    """
    
    def __init__(self, 
                 map_dimensions: Tuple[int, int],
                 input_dim: int,
                 embodiment_constraints: Dict[str, float],
                 temporal_horizon: int = 10):
        """
        Args:
            map_dimensions: (height, width) のマップサイズ  
            input_dim: 入力次元数
            embodiment_constraints: 身体的制約パラメータ
            temporal_horizon: 予測時間窓
        """
        self.map_dimensions = map_dimensions
        self.input_dim = input_dim
        self.embodiment_constraints = embodiment_constraints
        self.temporal_horizon = temporal_horizon
        
        # JAXキーの初期化
        self.key = jax.random.PRNGKey(42)
        
        # マップの初期化
        self.key, subkey = jax.random.split(self.key)
        self.weight_map = jax.random.normal(
            subkey, 
            (map_dimensions[0], map_dimensions[1], input_dim)
        )
        
        # エナクティブ拡張
        self.affordance_map = jnp.zeros(map_dimensions + (4,))  # 4方向の行為可能性
        self.prediction_map = jnp.zeros(map_dimensions + (input_dim,))
        self.embodiment_map = jnp.ones(map_dimensions)  # 身体化度
        
    @abstractmethod
    def enactive_update(self, 
                       sensory_input: jnp.ndarray,
                       motor_command: jnp.ndarray,
                       prediction_error: jnp.ndarray) -> 'EnactiveSOM':
        """
        エナクティブ更新：感覚-運動統合による学習
        
        Args:
            sensory_input: 感覚入力
            motor_command: 運動指令
            prediction_error: 予測誤差
            
        Returns:
            更新されたSOMインスタンス
        """
        pass
    
    @abstractmethod
    def compute_affordances(self, 
                          current_state: jnp.ndarray,
                          context: Dict[str, Any]) -> List[ConceptualAffordance]:
        """
        概念的アフォーダンスの計算
        
        Args:
            current_state: 現在の状態ベクトル
            context: 文脈情報
            
        Returns:
            利用可能なアフォーダンスのリスト
        """
        pass
    
    def consciousness_contribution(self, 
                                 current_state: ConsciousnessState) -> PhiValue:
        """
        SOMの意識への寄与度を計算
        
        エナクティビズム観点での意識寄与：
        1. 概念空間の動的再編成度
        2. 行為-知覚ループの統合度  
        3. 予測誤差による構造変化
        """
        # SOMの動的変化率
        plasticity_measure = self._compute_map_plasticity()
        
        # 概念クラスターの結合度
        cluster_coherence = self._compute_cluster_coherence()
        
        # 予測統合度
        prediction_integration = self._compute_prediction_integration()
        
        # 身体化度
        embodiment_factor = jnp.mean(self.embodiment_map)
        
        # エナクティブ意識寄与の計算
        enactive_phi = (
            plasticity_measure * 0.3 + 
            cluster_coherence * 0.25 + 
            prediction_integration * 0.25 +
            embodiment_factor * 0.2
        )
        
        return PhiValue(float(enactive_phi))
    
    @jit
    def _compute_map_plasticity(self) -> float:
        """マップの可塑性（動的変化度）を計算"""
        # 隣接ニューロン間の重み差分の総和
        horizontal_diff = jnp.sum(jnp.abs(
            self.weight_map[:-1, :, :] - self.weight_map[1:, :, :]
        ))
        vertical_diff = jnp.sum(jnp.abs(
            self.weight_map[:, :-1, :] - self.weight_map[:, 1:, :]
        ))
        
        total_variation = horizontal_diff + vertical_diff
        normalized_plasticity = total_variation / (
            self.map_dimensions[0] * self.map_dimensions[1] * self.input_dim
        )
        
        return float(normalized_plasticity)
    
    @jit  
    def _compute_cluster_coherence(self) -> float:
        """概念クラスターの結合度を計算"""
        # k-means風のクラスター内分散の逆数
        flat_weights = self.weight_map.reshape(-1, self.input_dim)
        
        # 簡易クラスタリング（5クラスター固定）
        n_clusters = 5
        cluster_centers = flat_weights[::len(flat_weights)//n_clusters][:n_clusters]
        
        # 各点の最近傍クラスターとの距離
        distances = vmap(lambda x: jnp.min(vmap(
            lambda c: jnp.linalg.norm(x - c)
        )(cluster_centers)))(flat_weights)
        
        # 結合度 = 1 / (平均距離 + ε)
        coherence = 1.0 / (jnp.mean(distances) + 1e-6)
        return float(coherence)
    
    @jit
    def _compute_prediction_integration(self) -> float:
        """予測統合度を計算"""
        # 予測マップと重みマップの一致度
        prediction_alignment = jnp.mean(jnp.abs(
            self.prediction_map - self.weight_map
        ))
        
        # 統合度 = 1 / (予測誤差 + ε)  
        integration = 1.0 / (prediction_alignment + 1e-6)
        return float(integration)
    
    def active_inference_step(self, 
                            observation: jnp.ndarray,
                            action_space: List[jnp.ndarray]) -> Tuple[jnp.ndarray, float]:
        """
        能動推論ステップ：予測誤差最小化による行動選択
        
        Args:
            observation: 現在の観測
            action_space: 可能な行動の集合
            
        Returns:
            (選択された行動, 予想される自由エネルギー減少)
        """
        best_action = None
        min_free_energy = float('inf')
        
        for action in action_space:
            # 行動による予測状態の計算
            predicted_observation = self._predict_next_observation(observation, action)
            
            # 自由エネルギーの計算
            surprise = self._compute_surprise(predicted_observation)
            complexity = self._compute_complexity(action)
            free_energy = surprise + complexity
            
            if free_energy < min_free_energy:
                min_free_energy = free_energy
                best_action = action
                
        energy_reduction = self._current_free_energy - min_free_energy
        return best_action, energy_reduction
    
    @abstractmethod
    def _predict_next_observation(self, 
                                current_obs: jnp.ndarray, 
                                action: jnp.ndarray) -> jnp.ndarray:
        """次の観測を予測"""
        pass
    
    @abstractmethod  
    def _compute_surprise(self, observation: jnp.ndarray) -> float:
        """観測のサプライズ（負の対数尤度）を計算"""
        pass
        
    @abstractmethod
    def _compute_complexity(self, action: jnp.ndarray) -> float:
        """行動の複雑性（KLダイバージェンス）を計算"""
        pass
    
    def phenomenological_mapping(self, 
                               conscious_state: ConsciousnessState) -> Dict[str, Any]:
        """
        現象学的マッピング：意識状態の質的構造をSOMに投影
        
        TaguchI Shigeru の現象学的アプローチに基づき、
        意識の時間的構造（把持-原印象-予持）をSOMに統合
        """
        current_phi = float(conscious_state.phi_value.value)
        
        # 時間的地平の構造化
        retention_horizon = self._map_retention_structure(conscious_state)
        primal_impression = self._map_primal_impression(conscious_state) 
        protention_horizon = self._map_protention_structure(conscious_state)
        
        # 現象学的マッピングの統合
        phenomenological_structure = {
            'temporal_synthesis': {
                'retention': retention_horizon,
                'primal_impression': primal_impression, 
                'protention': protention_horizon
            },
            'intentional_arc': self._compute_intentional_arc(),
            'embodied_schema': self._extract_embodied_schema(),
            'intercorporeal_field': self._map_intercorporeal_relations()
        }
        
        return phenomenological_structure
    
    def _map_retention_structure(self, state: ConsciousnessState) -> jnp.ndarray:
        """把持（過去の意識）構造のマッピング"""
        # 過去の状態履歴をSOM上に投影
        history_weight = 0.8  # 過去重み
        retention_map = self.weight_map * history_weight
        return retention_map
        
    def _map_primal_impression(self, state: ConsciousnessState) -> jnp.ndarray:
        """原印象（現在の意識）のマッピング"""
        # 現在のΦ値を強度として原印象を構成
        phi_intensity = float(state.phi_value.value)
        impression_map = self.weight_map * phi_intensity
        return impression_map
        
    def _map_protention_structure(self, state: ConsciousnessState) -> jnp.ndarray:
        """予持（未来への意識）構造のマッピング"""
        # 予測マップから未来予期を構成
        anticipation_weight = 0.6
        protention_map = self.prediction_map * anticipation_weight
        return protention_map
    
    def _compute_intentional_arc(self) -> float:
        """志向的弧の計算"""
        # SOMの最大活性化から意図方向を推定
        max_activation = jnp.max(self.weight_map, axis=2)
        activation_gradient = jnp.gradient(max_activation)
        intentional_strength = jnp.linalg.norm(jnp.stack(activation_gradient))
        return float(intentional_strength)
    
    def _extract_embodied_schema(self) -> Dict[str, float]:
        """身体図式の抽出"""
        return {
            'motor_readiness': float(jnp.mean(self.affordance_map)),
            'spatial_orientation': float(jnp.std(self.embodiment_map)),
            'temporal_flow': float(jnp.mean(jnp.abs(
                self.prediction_map - self.weight_map
            )))
        }
    
    def _map_intercorporeal_relations(self) -> float:
        """間身体的関係のマッピング"""
        # SOMの対称性から間主観性を推定
        map_symmetry = jnp.mean(jnp.abs(
            self.weight_map - jnp.flip(self.weight_map, axis=[0, 1])
        ))
        intercorporeal_resonance = 1.0 / (1.0 + map_symmetry)
        return float(intercorporeal_resonance)