"""
身体化記憶特徴値オブジェクト

エナクティブ認知理論に基づく身体感覚統合による記憶特徴の拡張。
固有受容感覚・前庭感覚・内受容感覚による記憶錨定効果を実現し、
42%不一致率問題の解決を目指す。
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import numpy as np
from enum import Enum

from .visual_feature import VisualFeature


class BodyReference(Enum):
    """身体基準点の定義"""
    HEAD = "head"
    CHEST = "chest"
    DOMINANT_HAND = "dominant_hand"
    NON_DOMINANT_HAND = "non_dominant_hand"
    WAIST = "waist"
    FEET = "feet"


@dataclass(frozen=True)
class ProprioceptiveFeature:
    """固有受容感覚特徴"""
    
    body_relative_position: Tuple[float, float, float]
    """身体中心からの相対位置（x, y, z）メートル単位"""
    
    reaching_distance: float
    """手の届く範囲内か（0.0-1.0、1.0が完全に届く）"""
    
    manipulation_affordance: float
    """操作可能性（0.0-1.0、身体サイズとの関係）"""
    
    body_schema_activation: Dict[BodyReference, float]
    """身体図式の活性化度（各身体部位）"""
    
    proprioceptive_confidence: float
    """固有受容感覚の信頼度"""


@dataclass(frozen=True)
class VestibularFeature:
    """前庭感覚特徴"""
    
    gravitational_reference: Tuple[float, float, float]
    """重力方向ベクトル（身体座標系）"""
    
    head_orientation: Tuple[float, float, float]
    """頭部の向き（ピッチ、ヨー、ロール）"""
    
    balance_state: float
    """平衡状態（0.0-1.0、1.0が完全な平衡）"""
    
    spatial_stability: float
    """空間的安定性（記憶錨定の強度）"""
    
    temporal_coherence: float
    """時間的整合性（前庭感覚による時系列記憶の信頼度）"""


@dataclass(frozen=True)
class InteroceptiveFeature:
    """内受容感覚特徴"""
    
    autonomic_response: Dict[str, float]
    """自律神経系反応（心拍、呼吸、皮膚電導度など）"""
    
    emotional_valence: float
    """情動価（-1.0から1.0、負が嫌悪、正が接近）"""
    
    arousal_level: float
    """覚醒度（0.0-1.0）"""
    
    visceral_memory_strength: float
    """内臓感覚記憶の強度（記憶の身体的定着度）"""
    
    interoceptive_awareness: float
    """内受容感覚の意識的認識度"""


@dataclass(frozen=True)
class EmbodiedMemoryFeature:
    """
    身体化記憶特徴の統合表現
    
    エナクティブ認知理論に基づき、視覚特徴に身体感覚を統合し、
    記憶の身体的錨定効果を実現する。42%不一致率の改善を目指す。
    """
    
    visual_feature: VisualFeature
    """基底となる視覚特徴"""
    
    proprioceptive_feature: ProprioceptiveFeature
    """固有受容感覚特徴"""
    
    vestibular_feature: VestibularFeature
    """前庭感覚特徴"""
    
    interoceptive_feature: InteroceptiveFeature
    """内受容感覚特徴"""
    
    embodiment_timestamp: datetime
    """身体化時刻"""
    
    embodiment_confidence: float
    """身体化信頼度（0.0-1.0）"""
    
    sensorimotor_coupling_strength: float
    """感覚運動結合の強度"""
    
    def __post_init__(self):
        """身体化記憶特徴の不変条件検証"""
        if not (0.0 <= self.embodiment_confidence <= 1.0):
            raise ValueError(f"Embodiment confidence must be between 0.0 and 1.0, got {self.embodiment_confidence}")
        
        if not (0.0 <= self.sensorimotor_coupling_strength <= 1.0):
            raise ValueError(f"Sensorimotor coupling strength must be between 0.0 and 1.0, got {self.sensorimotor_coupling_strength}")
    
    def get_embodied_feature_vector(self) -> np.ndarray:
        """
        身体化統合特徴ベクトルの生成
        
        視覚特徴に身体感覚を統合し、身体的コンテクストを含む
        統一的な特徴表現を生成する。
        """
        feature_vectors = []
        
        # 基底視覚特徴
        visual_vector = self.visual_feature.get_unified_feature_vector()
        feature_vectors.append(visual_vector)
        
        # 固有受容感覚ベクトル
        proprioceptive_vector = self._vectorize_proprioceptive_feature()
        feature_vectors.append(proprioceptive_vector)
        
        # 前庭感覚ベクトル
        vestibular_vector = self._vectorize_vestibular_feature()
        feature_vectors.append(vestibular_vector)
        
        # 内受容感覚ベクトル
        interoceptive_vector = self._vectorize_interoceptive_feature()
        feature_vectors.append(interoceptive_vector)
        
        if not feature_vectors:
            return np.array([])
        
        # 身体感覚による重み付け統合
        weighted_vectors = self._apply_embodied_weighting(feature_vectors)
        embodied_vector = np.concatenate(weighted_vectors)
        
        # 身体図式による正規化
        normalized_vector = self._apply_body_schema_normalization(embodied_vector)
        
        return normalized_vector
    
    def _vectorize_proprioceptive_feature(self) -> np.ndarray:
        """固有受容感覚特徴のベクトル化"""
        prop_vector = []
        
        # 身体相対位置
        prop_vector.extend(self.proprioceptive_feature.body_relative_position)
        
        # 手の届く範囲
        prop_vector.append(self.proprioceptive_feature.reaching_distance)
        
        # 操作可能性
        prop_vector.append(self.proprioceptive_feature.manipulation_affordance)
        
        # 身体図式活性化
        body_schema_values = list(self.proprioceptive_feature.body_schema_activation.values())
        prop_vector.extend(body_schema_values)
        
        # 固有受容感覚信頼度
        prop_vector.append(self.proprioceptive_feature.proprioceptive_confidence)
        
        return np.array(prop_vector)
    
    def _vectorize_vestibular_feature(self) -> np.ndarray:
        """前庭感覚特徴のベクトル化"""
        vest_vector = []
        
        # 重力基準ベクトル
        vest_vector.extend(self.vestibular_feature.gravitational_reference)
        
        # 頭部向き
        vest_vector.extend(self.vestibular_feature.head_orientation)
        
        # 平衡・安定性・時間整合性
        vest_vector.append(self.vestibular_feature.balance_state)
        vest_vector.append(self.vestibular_feature.spatial_stability)
        vest_vector.append(self.vestibular_feature.temporal_coherence)
        
        return np.array(vest_vector)
    
    def _vectorize_interoceptive_feature(self) -> np.ndarray:
        """内受容感覚特徴のベクトル化"""
        intero_vector = []
        
        # 自律神経系反応
        autonomic_values = list(self.interoceptive_feature.autonomic_response.values())
        intero_vector.extend(autonomic_values)
        
        # 情動価・覚醒度
        intero_vector.append(self.interoceptive_feature.emotional_valence)
        intero_vector.append(self.interoceptive_feature.arousal_level)
        
        # 内臓記憶強度・内受容認識度
        intero_vector.append(self.interoceptive_feature.visceral_memory_strength)
        intero_vector.append(self.interoceptive_feature.interoceptive_awareness)
        
        return np.array(intero_vector)
    
    def _apply_embodied_weighting(self, feature_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """身体感覚による重み付け統合"""
        if len(feature_vectors) != 4:
            return feature_vectors
        
        visual_vec, prop_vec, vest_vec, intero_vec = feature_vectors
        
        # 身体感覚の信頼度による重み計算
        prop_weight = self.proprioceptive_feature.proprioceptive_confidence
        vest_weight = self.vestibular_feature.spatial_stability * 0.8  # 前庭感覚の重み調整
        intero_weight = self.interoceptive_feature.interoceptive_awareness * 0.6  # 内受容感覚の重み調整
        
        # 感覚運動結合強度による全体調整
        coupling_factor = self.sensorimotor_coupling_strength
        
        # 重み付け適用
        weighted_visual = visual_vec * (1.0 + coupling_factor * 0.2)  # 視覚の基本重み維持
        weighted_prop = prop_vec * prop_weight * coupling_factor
        weighted_vest = vest_vec * vest_weight * coupling_factor
        weighted_intero = intero_vec * intero_weight * coupling_factor
        
        return [weighted_visual, weighted_prop, weighted_vest, weighted_intero]
    
    def _apply_body_schema_normalization(self, vector: np.ndarray) -> np.ndarray:
        """身体図式による正規化"""
        if vector.size == 0:
            return vector
        
        # 身体図式の活性化度による正規化係数
        body_activation = np.mean(list(self.proprioceptive_feature.body_schema_activation.values()))
        
        # 前庭感覚による空間的正規化
        spatial_normalization = self.vestibular_feature.spatial_stability
        
        # 総合正規化係数
        normalization_factor = np.sqrt(body_activation * spatial_normalization + 1e-6)
        
        # L2正規化に身体的要素を組み込み
        l2_norm = np.linalg.norm(vector)
        if l2_norm > 0:
            embodied_normalized = vector / l2_norm * normalization_factor
        else:
            embodied_normalized = vector
        
        return embodied_normalized
    
    def calculate_embodied_similarity(self, other: 'EmbodiedMemoryFeature') -> float:
        """
        身体化記憶特徴間の類似度計算
        
        エナクティブ認知理論に基づき、身体感覚を含む
        統合的類似度を計算する。
        """
        if not isinstance(other, EmbodiedMemoryFeature):
            raise TypeError("Comparison target must be EmbodiedMemoryFeature")
        
        similarities = []
        
        # 視覚特徴類似度（基底）
        visual_sim = self.visual_feature.calculate_similarity(other.visual_feature)
        similarities.append(('visual', visual_sim, 0.4))  # 重み0.4
        
        # 固有受容感覚類似度
        prop_sim = self._calculate_proprioceptive_similarity(other)
        similarities.append(('proprioceptive', prop_sim, 0.25))  # 重み0.25
        
        # 前庭感覚類似度
        vest_sim = self._calculate_vestibular_similarity(other)
        similarities.append(('vestibular', vest_sim, 0.2))  # 重み0.2
        
        # 内受容感覚類似度
        intero_sim = self._calculate_interoceptive_similarity(other)
        similarities.append(('interoceptive', intero_sim, 0.15))  # 重み0.15
        
        # 重み付き平均による統合類似度
        weighted_sum = sum(sim * weight for _, sim, weight in similarities)
        total_weight = sum(weight for _, _, weight in similarities)
        
        integrated_similarity = weighted_sum / total_weight
        
        # 身体化信頼度による調整
        confidence_factor = (self.embodiment_confidence + other.embodiment_confidence) / 2.0
        
        return integrated_similarity * confidence_factor
    
    def _calculate_proprioceptive_similarity(self, other: 'EmbodiedMemoryFeature') -> float:
        """固有受容感覚類似度計算"""
        similarities = []
        
        # 身体相対位置の類似度
        pos_diff = np.linalg.norm(
            np.array(self.proprioceptive_feature.body_relative_position) - 
            np.array(other.proprioceptive_feature.body_relative_position)
        )
        pos_sim = max(0.0, 1.0 - pos_diff / 2.0)  # 2m以内で正規化
        similarities.append(pos_sim)
        
        # 手の届く範囲類似度
        reach_sim = 1.0 - abs(
            self.proprioceptive_feature.reaching_distance - 
            other.proprioceptive_feature.reaching_distance
        )
        similarities.append(reach_sim)
        
        # 操作可能性類似度
        manip_sim = 1.0 - abs(
            self.proprioceptive_feature.manipulation_affordance - 
            other.proprioceptive_feature.manipulation_affordance
        )
        similarities.append(manip_sim)
        
        return np.mean(similarities)
    
    def _calculate_vestibular_similarity(self, other: 'EmbodiedMemoryFeature') -> float:
        """前庭感覚類似度計算"""
        similarities = []
        
        # 重力方向類似度（コサイン類似度）
        grav1 = np.array(self.vestibular_feature.gravitational_reference)
        grav2 = np.array(other.vestibular_feature.gravitational_reference)
        if np.linalg.norm(grav1) > 0 and np.linalg.norm(grav2) > 0:
            grav_sim = np.dot(grav1, grav2) / (np.linalg.norm(grav1) * np.linalg.norm(grav2))
            grav_sim = max(0.0, grav_sim)
        else:
            grav_sim = 0.0
        similarities.append(grav_sim)
        
        # 空間安定性類似度
        stability_sim = 1.0 - abs(
            self.vestibular_feature.spatial_stability - 
            other.vestibular_feature.spatial_stability
        )
        similarities.append(stability_sim)
        
        return np.mean(similarities)
    
    def _calculate_interoceptive_similarity(self, other: 'EmbodiedMemoryFeature') -> float:
        """内受容感覚類似度計算"""
        similarities = []
        
        # 情動価類似度
        valence_sim = 1.0 - abs(
            self.interoceptive_feature.emotional_valence - 
            other.interoceptive_feature.emotional_valence
        ) / 2.0  # -1〜1の範囲なので2で除算
        similarities.append(valence_sim)
        
        # 覚醒度類似度
        arousal_sim = 1.0 - abs(
            self.interoceptive_feature.arousal_level - 
            other.interoceptive_feature.arousal_level
        )
        similarities.append(arousal_sim)
        
        # 内臓記憶強度類似度
        visceral_sim = 1.0 - abs(
            self.interoceptive_feature.visceral_memory_strength - 
            other.interoceptive_feature.visceral_memory_strength
        )
        similarities.append(visceral_sim)
        
        return np.mean(similarities)
    
    def get_memory_anchoring_strength(self) -> float:
        """
        記憶錨定強度の計算
        
        身体感覚による記憶の定着度を定量化し、
        42%不一致率改善の指標とする。
        """
        # 固有受容感覚による空間錨定
        spatial_anchoring = (
            self.proprioceptive_feature.body_schema_activation.get(BodyReference.CHEST, 0.0) * 0.3 +
            self.proprioceptive_feature.reaching_distance * 0.4 +
            self.proprioceptive_feature.proprioceptive_confidence * 0.3
        )
        
        # 前庭感覚による時間錨定
        temporal_anchoring = (
            self.vestibular_feature.temporal_coherence * 0.5 +
            self.vestibular_feature.spatial_stability * 0.5
        )
        
        # 内受容感覚による情動錨定
        emotional_anchoring = (
            abs(self.interoceptive_feature.emotional_valence) * 0.4 +  # 情動の強度
            self.interoceptive_feature.visceral_memory_strength * 0.6
        )
        
        # 感覚運動統合による総合錨定
        integrated_anchoring = (
            spatial_anchoring * 0.4 +
            temporal_anchoring * 0.3 +
            emotional_anchoring * 0.3
        ) * self.sensorimotor_coupling_strength
        
        return min(1.0, integrated_anchoring)
    
    def predict_memory_consistency(self) -> float:
        """
        記憶一致性の予測
        
        身体感覚による記憶錨定強度から、
        記憶の一致性を予測する。目標：42%不一致率の改善
        """
        anchoring_strength = self.get_memory_anchoring_strength()
        
        # 身体化信頼度との相互作用
        embodiment_factor = self.embodiment_confidence
        
        # 予測一致率の計算
        # 基底一致率58% + 身体感覚による改善効果
        base_consistency = 0.58
        embodied_improvement = anchoring_strength * embodiment_factor * 0.35  # 最大35%改善
        
        predicted_consistency = base_consistency + embodied_improvement
        
        return min(1.0, predicted_consistency)
    
    @classmethod
    def create_from_iot_sensors(
        cls,
        visual_feature: VisualFeature,
        imu_data: Dict[str, float],
        biometric_data: Dict[str, float],
        spatial_context: Dict[str, float]
    ) -> 'EmbodiedMemoryFeature':
        """
        IoTセンサーデータから身体化記憶特徴を生成
        
        実際のハードウェアセンサーから身体感覚を再構成し、
        仮想的な身体化認知を実現する。
        """
        # 固有受容感覚の再構成
        proprioceptive = ProprioceptiveFeature(
            body_relative_position=(
                spatial_context.get('relative_x', 0.0),
                spatial_context.get('relative_y', 0.0),
                spatial_context.get('relative_z', 0.0)
            ),
            reaching_distance=spatial_context.get('reaching_distance', 0.5),
            manipulation_affordance=spatial_context.get('manipulation_affordance', 0.5),
            body_schema_activation={
                BodyReference.HEAD: imu_data.get('head_activation', 0.5),
                BodyReference.CHEST: imu_data.get('chest_activation', 0.5),
                BodyReference.DOMINANT_HAND: imu_data.get('hand_activation', 0.5),
                BodyReference.WAIST: imu_data.get('waist_activation', 0.5),
                BodyReference.FEET: imu_data.get('feet_activation', 0.5)
            },
            proprioceptive_confidence=imu_data.get('imu_confidence', 0.7)
        )
        
        # 前庭感覚の再構成
        vestibular = VestibularFeature(
            gravitational_reference=(
                imu_data.get('gravity_x', 0.0),
                imu_data.get('gravity_y', 0.0),
                imu_data.get('gravity_z', -9.8)
            ),
            head_orientation=(
                imu_data.get('pitch', 0.0),
                imu_data.get('yaw', 0.0),
                imu_data.get('roll', 0.0)
            ),
            balance_state=imu_data.get('balance_state', 0.8),
            spatial_stability=imu_data.get('spatial_stability', 0.8),
            temporal_coherence=imu_data.get('temporal_coherence', 0.7)
        )
        
        # 内受容感覚の再構成
        interoceptive = InteroceptiveFeature(
            autonomic_response={
                'heart_rate': biometric_data.get('heart_rate', 70.0),
                'respiratory_rate': biometric_data.get('respiratory_rate', 16.0),
                'skin_conductance': biometric_data.get('skin_conductance', 0.5)
            },
            emotional_valence=biometric_data.get('emotional_valence', 0.0),
            arousal_level=biometric_data.get('arousal_level', 0.5),
            visceral_memory_strength=biometric_data.get('visceral_strength', 0.6),
            interoceptive_awareness=biometric_data.get('interoceptive_awareness', 0.6)
        )
        
        # 感覚運動結合強度の計算
        sensorimotor_coupling = np.mean([
            proprioceptive.proprioceptive_confidence,
            vestibular.spatial_stability,
            interoceptive.interoceptive_awareness
        ])
        
        return cls(
            visual_feature=visual_feature,
            proprioceptive_feature=proprioceptive,
            vestibular_feature=vestibular,
            interoceptive_feature=interoceptive,
            embodiment_timestamp=datetime.now(),
            embodiment_confidence=0.8,  # IoTセンサーベースの信頼度
            sensorimotor_coupling_strength=sensorimotor_coupling
        )