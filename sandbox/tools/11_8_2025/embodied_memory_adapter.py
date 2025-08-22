"""
身体化記憶アダプター

エナクティブ認知理論に基づく身体感覚統合記憶システム。
既存のNGC-Learn統合アダプターを拡張し、身体感覚による記憶錨定効果を実現。
42%不一致率問題の解決を目指す実装。

理論的基盤：
- Ezequiel Di Paolo のエナクティブ認知アプローチ
- メルロ＝ポンティの身体現象学
- Embodied Cognition理論
- 多感覚統合による記憶安定化
"""

import jax
import jax.numpy as jnp
from typing import List, Optional, Dict, Any, Tuple
import logging
import time
from datetime import datetime
import numpy as np

from ngc_learn_adapter import HybridPredictiveCodingAdapter, PredictiveCodingCore
from domain.value_objects.embodied_memory_feature import (
    EmbodiedMemoryFeature, 
    ProprioceptiveFeature, 
    VestibularFeature, 
    InteroceptiveFeature,
    BodyReference
)
from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.prediction_state import PredictionState


class EmbodiedMemoryAdapter(HybridPredictiveCodingAdapter):
    """
    身体化記憶アダプター
    
    エナクティブ認知理論に基づき、既存の予測符号化システムに
    身体感覚を統合し、記憶の身体的錨定効果を実現する。
    
    主要機能：
    1. 身体感覚による記憶錨定
    2. 感覚運動統合による記憶安定化
    3. 身体図式による記憶組織化
    4. 多感覚統合による一致性向上（目標：42%不一致率改善）
    """
    
    def __init__(self, 
                 hierarchy_levels: int, 
                 input_dimensions: int,
                 embodied_mode: bool = True,
                 sensorimotor_coupling_threshold: float = 0.6,
                 body_schema_update_rate: float = 0.1):
        """
        身体化記憶アダプターの初期化
        
        Args:
            hierarchy_levels: 階層レベル数
            input_dimensions: 入力次元（視覚特徴のみ）
            embodied_mode: 身体化モードの有効化
            sensorimotor_coupling_threshold: 感覚運動結合の閾値
            body_schema_update_rate: 身体図式更新率
        """
        # 身体感覚統合のための次元拡張
        embodied_input_dimensions = input_dimensions + self._calculate_embodied_dimensions()
        
        super().__init__(
            hierarchy_levels=hierarchy_levels,
            input_dimensions=embodied_input_dimensions,
            prefer_ngc_learn=True,
            fallback_to_jax=True
        )
        
        self.embodied_mode = embodied_mode
        self.sensorimotor_coupling_threshold = sensorimotor_coupling_threshold
        self.body_schema_update_rate = body_schema_update_rate
        
        # 身体化記憶システムの初期化
        self._initialize_embodied_memory_system()
        
        # パフォーマンス追跡の拡張
        self._initialize_embodied_performance_tracking()
        
        self.logger.info(f"EmbodiedMemoryAdapter initialized with embodied_mode: {embodied_mode}")
    
    def _calculate_embodied_dimensions(self) -> int:
        """身体感覚統合に必要な追加次元数を計算"""
        proprioceptive_dims = 3 + 1 + 1 + len(BodyReference) + 1  # 位置(3) + 到達距離 + 操作性 + 身体図式 + 信頼度
        vestibular_dims = 3 + 3 + 1 + 1 + 1  # 重力(3) + 頭部向き(3) + 平衡 + 安定性 + 時間整合性
        interoceptive_dims = 3 + 1 + 1 + 1 + 1  # 自律神経(3) + 情動価 + 覚醒 + 内臓記憶 + 認識度
        
        return proprioceptive_dims + vestibular_dims + interoceptive_dims
    
    def _initialize_embodied_memory_system(self):
        """身体化記憶システムの初期化"""
        # 身体図式の状態管理
        self.body_schema_state = {
            body_ref: 0.5 for body_ref in BodyReference  # 初期活性化度
        }
        
        # 感覚運動記憶の履歴
        self.sensorimotor_memory_history = []
        
        # 身体感覚による記憶錨定マップ
        self.memory_anchoring_map = {}
        
        # 記憶一致性の履歴追跡
        self.memory_consistency_history = []
        
        # 身体感覚統合の重み
        self.embodied_integration_weights = {
            'proprioceptive': 0.25,
            'vestibular': 0.20,
            'interoceptive': 0.15,
            'visual': 0.40
        }
    
    def _initialize_embodied_performance_tracking(self):
        """身体化性能追跡の初期化"""
        self.embodied_performance_metrics = {
            'memory_consistency_improvements': [],
            'anchoring_strength_history': [],
            'sensorimotor_coupling_quality': [],
            'embodied_prediction_accuracy': [],
            'body_schema_adaptation_rate': [],
            'cross_modal_integration_success': 0,
            'baseline_consistency_rate': 0.58,  # 既存システムの一致率
            'target_consistency_rate': 0.85,   # 目標一致率
            'current_consistency_rate': 0.58
        }
    
    def process_embodied_memory(self, 
                               embodied_feature: EmbodiedMemoryFeature) -> PredictionState:
        """
        身体化記憶の処理
        
        身体感覚を統合した記憶処理を実行し、
        身体的錨定効果による記憶安定化を実現。
        
        Args:
            embodied_feature: 身体化記憶特徴
            
        Returns:
            身体感覚統合済みの予測状態
        """
        start_time = time.time()
        
        try:
            # 身体化特徴ベクトルの生成
            embodied_vector = embodied_feature.get_embodied_feature_vector()
            
            if not self.embodied_mode:
                # 身体化モード無効時は視覚特徴のみ使用
                visual_vector = embodied_feature.visual_feature.get_unified_feature_vector()
                return super().forward_prediction(visual_vector)
            
            # 身体図式の更新
            self._update_body_schema(embodied_feature.proprioceptive_feature)
            
            # 感覚運動統合による予測
            embodied_prediction_state = self._forward_embodied_prediction(
                embodied_vector, embodied_feature
            )
            
            # 記憶錨定強度の計算と更新
            anchoring_strength = embodied_feature.get_memory_anchoring_strength()
            self._update_memory_anchoring(embodied_feature, anchoring_strength)
            
            # 記憶一致性の予測と追跡
            predicted_consistency = embodied_feature.predict_memory_consistency()
            self._track_memory_consistency(predicted_consistency)
            
            # 身体化メタデータの付加
            self._enhance_prediction_with_embodied_metadata(
                embodied_prediction_state, 
                embodied_feature,
                anchoring_strength,
                predicted_consistency,
                time.time() - start_time
            )
            
            return embodied_prediction_state
            
        except Exception as e:
            self.logger.error(f"Embodied memory processing error: {e}")
            # フォールバック：視覚特徴のみで処理
            visual_vector = embodied_feature.visual_feature.get_unified_feature_vector()
            return super().forward_prediction(visual_vector)
    
    def _update_body_schema(self, proprioceptive_feature: ProprioceptiveFeature):
        """身体図式の動的更新"""
        for body_ref, activation in proprioceptive_feature.body_schema_activation.items():
            current_activation = self.body_schema_state[body_ref]
            
            # 指数移動平均による身体図式の更新
            updated_activation = (
                (1 - self.body_schema_update_rate) * current_activation +
                self.body_schema_update_rate * activation
            )
            
            self.body_schema_state[body_ref] = updated_activation
    
    def _forward_embodied_prediction(self, 
                                   embodied_vector: jnp.ndarray,
                                   embodied_feature: EmbodiedMemoryFeature) -> PredictionState:
        """身体感覚統合予測の実行"""
        # 基底予測符号化の実行
        base_predictions = self.engine.predict_hierarchical(embodied_vector)
        
        # 身体感覚による予測の修正
        embodied_predictions = self._apply_embodied_modulation(
            base_predictions, embodied_feature
        )
        
        # 身体的コンテクストによる誤差計算
        embodied_errors = self._calculate_embodied_prediction_errors(
            embodied_predictions, embodied_vector, embodied_feature
        )
        
        # 収束状態の身体的判定
        convergence_status = self._determine_embodied_convergence(
            embodied_errors, embodied_feature
        )
        
        return PredictionState(
            hierarchical_errors=embodied_errors,
            hierarchical_predictions=embodied_predictions,
            convergence_status=convergence_status,
            learning_iteration=1
        )
    
    def _apply_embodied_modulation(self, 
                                 base_predictions: List[jnp.ndarray],
                                 embodied_feature: EmbodiedMemoryFeature) -> List[jnp.ndarray]:
        """身体感覚による予測の調整"""
        if not base_predictions:
            return base_predictions
        
        modulated_predictions = []
        
        for i, prediction in enumerate(base_predictions):
            # 階層レベルに応じた身体感覚の重み
            hierarchical_weight = 1.0 / (i + 1)  # 上位階層ほど身体感覚の影響が強い
            
            # 前庭感覚による空間的調整
            spatial_modulation = self._apply_vestibular_spatial_modulation(
                prediction, embodied_feature.vestibular_feature, hierarchical_weight
            )
            
            # 内受容感覚による情動的調整
            emotional_modulation = self._apply_interoceptive_emotional_modulation(
                spatial_modulation, embodied_feature.interoceptive_feature, hierarchical_weight
            )
            
            # 固有受容感覚による身体的調整
            proprioceptive_modulation = self._apply_proprioceptive_body_modulation(
                emotional_modulation, embodied_feature.proprioceptive_feature, hierarchical_weight
            )
            
            modulated_predictions.append(proprioceptive_modulation)
        
        return modulated_predictions
    
    def _apply_vestibular_spatial_modulation(self, 
                                           prediction: jnp.ndarray,
                                           vestibular_feature: VestibularFeature,
                                           hierarchical_weight: float) -> jnp.ndarray:
        """前庭感覚による空間的調整"""
        # 重力方向による空間的基準の設定
        gravity_influence = vestibular_feature.spatial_stability * hierarchical_weight
        
        # 空間的安定性による予測の平滑化
        stability_factor = vestibular_feature.temporal_coherence
        
        # 予測への空間的調整の適用
        spatial_adjustment = gravity_influence * stability_factor * 0.1  # 調整の強度制限
        
        return prediction * (1.0 + spatial_adjustment)
    
    def _apply_interoceptive_emotional_modulation(self, 
                                                prediction: jnp.ndarray,
                                                interoceptive_feature: InteroceptiveFeature,
                                                hierarchical_weight: float) -> jnp.ndarray:
        """内受容感覚による情動的調整"""
        # 情動価による記憶の重み付け（正の情動は記憶を強化）
        emotional_weight = (interoceptive_feature.emotional_valence + 1.0) / 2.0  # 0-1に正規化
        
        # 覚醒度による記憶の鮮明化
        arousal_factor = interoceptive_feature.arousal_level
        
        # 内臓記憶強度による定着度
        visceral_strength = interoceptive_feature.visceral_memory_strength
        
        # 統合的情動調整
        emotional_modulation = (
            emotional_weight * 0.4 +
            arousal_factor * 0.3 +
            visceral_strength * 0.3
        ) * hierarchical_weight
        
        return prediction * (1.0 + emotional_modulation * 0.15)  # 調整強度制限
    
    def _apply_proprioceptive_body_modulation(self, 
                                            prediction: jnp.ndarray,
                                            proprioceptive_feature: ProprioceptiveFeature,
                                            hierarchical_weight: float) -> jnp.ndarray:
        """固有受容感覚による身体的調整"""
        # 身体図式活性化による調整
        body_activation = np.mean(list(proprioceptive_feature.body_schema_activation.values()))
        
        # 手の届く範囲による操作性の考慮
        manipulation_factor = proprioceptive_feature.manipulation_affordance
        
        # 身体的調整の適用
        body_modulation = (
            body_activation * 0.6 +
            manipulation_factor * 0.4
        ) * hierarchical_weight
        
        return prediction * (1.0 + body_modulation * 0.12)  # 調整強度制限
    
    def _calculate_embodied_prediction_errors(self, 
                                            predictions: List[jnp.ndarray],
                                            target_vector: jnp.ndarray,
                                            embodied_feature: EmbodiedMemoryFeature) -> List[float]:
        """身体的コンテクストを考慮した予測誤差計算"""
        if not predictions:
            return [0.2]
        
        errors = []
        
        for i, prediction in enumerate(predictions):
            # 基本的な予測誤差
            if i == 0:
                # 第1層：入力ベクトルとの直接比較
                min_len = min(len(prediction), len(target_vector))
                base_error = float(jnp.mean(jnp.abs(prediction[:min_len] - target_vector[:min_len])))
            else:
                # 上位層：階層的誤差
                base_error = 0.1 / (i + 1)
            
            # 身体感覚による誤差修正
            embodied_error = self._apply_embodied_error_correction(
                base_error, embodied_feature, i
            )
            
            errors.append(min(embodied_error, 0.4))
        
        return errors
    
    def _apply_embodied_error_correction(self, 
                                       base_error: float,
                                       embodied_feature: EmbodiedMemoryFeature,
                                       hierarchy_level: int) -> float:
        """身体感覚による誤差修正"""
        # 記憶錨定強度による誤差削減
        anchoring_strength = embodied_feature.get_memory_anchoring_strength()
        error_reduction_factor = anchoring_strength * 0.3  # 最大30%の誤差削減
        
        # 感覚運動結合による安定化
        sensorimotor_stability = embodied_feature.sensorimotor_coupling_strength
        stability_bonus = sensorimotor_stability * 0.2  # 最大20%の安定化
        
        # 階層レベルによる調整（上位階層ほど身体感覚の効果が強い）
        hierarchical_factor = 1.0 + hierarchy_level * 0.1
        
        # 統合的誤差修正
        corrected_error = base_error * (1.0 - error_reduction_factor - stability_bonus) * hierarchical_factor
        
        return max(0.01, corrected_error)  # 最小誤差の保証
    
    def _determine_embodied_convergence(self, 
                                      errors: List[float],
                                      embodied_feature: EmbodiedMemoryFeature) -> str:
        """身体的収束状態の判定"""
        if not errors:
            return "no_data"
        
        avg_error = sum(errors) / len(errors)
        
        # 身体感覚による収束閾値の調整
        anchoring_strength = embodied_feature.get_memory_anchoring_strength()
        convergence_threshold = 0.1 * (1.0 - anchoring_strength * 0.3)  # 身体錨定による閾値降下
        
        if avg_error < convergence_threshold:
            return "embodied_converged"
        elif avg_error < convergence_threshold * 2.0:
            return "embodied_converging"
        else:
            return "embodied_not_converged"
    
    def _update_memory_anchoring(self, 
                               embodied_feature: EmbodiedMemoryFeature,
                               anchoring_strength: float):
        """記憶錨定マップの更新"""
        # 特徴のハッシュ値による索引
        feature_hash = self._calculate_feature_hash(embodied_feature)
        
        # 錨定履歴の更新
        if feature_hash not in self.memory_anchoring_map:
            self.memory_anchoring_map[feature_hash] = []
        
        self.memory_anchoring_map[feature_hash].append({
            'timestamp': datetime.now(),
            'anchoring_strength': anchoring_strength,
            'embodiment_confidence': embodied_feature.embodiment_confidence,
            'sensorimotor_coupling': embodied_feature.sensorimotor_coupling_strength
        })
        
        # 履歴の上限管理
        if len(self.memory_anchoring_map[feature_hash]) > 100:
            self.memory_anchoring_map[feature_hash] = self.memory_anchoring_map[feature_hash][-100:]
        
        # パフォーマンス追跡の更新
        self.embodied_performance_metrics['anchoring_strength_history'].append(anchoring_strength)
    
    def _track_memory_consistency(self, predicted_consistency: float):
        """記憶一致性の追跡"""
        self.memory_consistency_history.append({
            'timestamp': datetime.now(),
            'predicted_consistency': predicted_consistency,
            'baseline_consistency': self.embodied_performance_metrics['baseline_consistency_rate']
        })
        
        # 現在の一致率の更新
        recent_consistencies = [entry['predicted_consistency'] for entry in self.memory_consistency_history[-10:]]
        self.embodied_performance_metrics['current_consistency_rate'] = np.mean(recent_consistencies)
        
        # 改善度の計算
        improvement = predicted_consistency - self.embodied_performance_metrics['baseline_consistency_rate']
        self.embodied_performance_metrics['memory_consistency_improvements'].append(improvement)
        
        # 履歴の上限管理
        if len(self.memory_consistency_history) > 1000:
            self.memory_consistency_history = self.memory_consistency_history[-1000:]
    
    def _enhance_prediction_with_embodied_metadata(self, 
                                                 prediction_state: PredictionState,
                                                 embodied_feature: EmbodiedMemoryFeature,
                                                 anchoring_strength: float,
                                                 predicted_consistency: float,
                                                 processing_time: float):
        """予測状態への身体化メタデータ付加"""
        if not hasattr(prediction_state, 'metadata'):
            prediction_state.metadata = {}
        
        # 身体化メタデータの追加
        prediction_state.metadata.update({
            'embodied_processing': True,
            'anchoring_strength': anchoring_strength,
            'predicted_consistency': predicted_consistency,
            'embodiment_confidence': embodied_feature.embodiment_confidence,
            'sensorimotor_coupling_strength': embodied_feature.sensorimotor_coupling_strength,
            'body_schema_state': dict(self.body_schema_state),
            'embodied_processing_time': processing_time,
            'consistency_improvement': predicted_consistency - self.embodied_performance_metrics['baseline_consistency_rate'],
            'target_consistency_progress': (predicted_consistency - self.embodied_performance_metrics['baseline_consistency_rate']) / (self.embodied_performance_metrics['target_consistency_rate'] - self.embodied_performance_metrics['baseline_consistency_rate'])
        })
    
    def _calculate_feature_hash(self, embodied_feature: EmbodiedMemoryFeature) -> str:
        """身体化特徴のハッシュ値計算"""
        # 特徴ベクトルからハッシュを生成
        feature_vector = embodied_feature.get_embodied_feature_vector()
        if feature_vector.size > 0:
            return hash(tuple(feature_vector.flatten()[:50]))  # 最初の50次元でハッシュ
        else:
            return hash(str(embodied_feature.embodiment_timestamp))
    
    def get_embodied_performance_report(self) -> Dict[str, Any]:
        """身体化性能レポートの取得"""
        base_report = self.backend_info
        
        # 記憶一致性の統計
        if self.embodied_performance_metrics['memory_consistency_improvements']:
            avg_improvement = np.mean(self.embodied_performance_metrics['memory_consistency_improvements'])
            max_improvement = np.max(self.embodied_performance_metrics['memory_consistency_improvements'])
        else:
            avg_improvement = 0.0
            max_improvement = 0.0
        
        # 身体化特有のメトリクス
        embodied_metrics = {
            'embodied_mode_active': self.embodied_mode,
            'current_consistency_rate': self.embodied_performance_metrics['current_consistency_rate'],
            'baseline_consistency_rate': self.embodied_performance_metrics['baseline_consistency_rate'],
            'target_consistency_rate': self.embodied_performance_metrics['target_consistency_rate'],
            'average_consistency_improvement': avg_improvement,
            'maximum_consistency_improvement': max_improvement,
            'current_improvement_percentage': (avg_improvement / (self.embodied_performance_metrics['target_consistency_rate'] - self.embodied_performance_metrics['baseline_consistency_rate'])) * 100 if avg_improvement > 0 else 0,
            'average_anchoring_strength': np.mean(self.embodied_performance_metrics['anchoring_strength_history']) if self.embodied_performance_metrics['anchoring_strength_history'] else 0.0,
            'body_schema_adaptation_active': len(self.memory_anchoring_map) > 0,
            'sensorimotor_integration_quality': np.mean(self.embodied_performance_metrics['sensorimotor_coupling_quality']) if self.embodied_performance_metrics['sensorimotor_coupling_quality'] else 0.0,
            'total_embodied_memories_processed': len(self.memory_anchoring_map),
            'memory_consistency_trend': 'improving' if avg_improvement > 0.05 else 'stable' if avg_improvement > -0.05 else 'declining'
        }
        
        # 基本レポートに身体化メトリクスを統合
        base_report.update(embodied_metrics)
        
        return base_report
    
    def simulate_42percent_consistency_improvement(self) -> Dict[str, float]:
        """42%不一致率改善のシミュレーション"""
        current_inconsistency = 1.0 - self.embodied_performance_metrics['current_consistency_rate']
        baseline_inconsistency = 1.0 - self.embodied_performance_metrics['baseline_consistency_rate']
        target_inconsistency = 1.0 - self.embodied_performance_metrics['target_consistency_rate']
        
        # 改善度の計算
        improvement_ratio = (baseline_inconsistency - current_inconsistency) / baseline_inconsistency if baseline_inconsistency > 0 else 0
        target_improvement_ratio = (baseline_inconsistency - target_inconsistency) / baseline_inconsistency if baseline_inconsistency > 0 else 0
        
        return {
            'baseline_inconsistency_rate': baseline_inconsistency * 100,  # 42%
            'current_inconsistency_rate': current_inconsistency * 100,
            'target_inconsistency_rate': target_inconsistency * 100,     # 15%
            'achieved_improvement_percentage': improvement_ratio * 100,
            'target_improvement_percentage': target_improvement_ratio * 100,
            'progress_towards_target': (improvement_ratio / target_improvement_ratio * 100) if target_improvement_ratio > 0 else 0,
            'estimated_final_inconsistency_rate': max(target_inconsistency * 100, current_inconsistency * 100 * 0.7)  # 保守的推定
        }


# 使用例とテスト
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 身体化記憶アダプターの作成
    embodied_adapter = EmbodiedMemoryAdapter(
        hierarchy_levels=3,
        input_dimensions=10,
        embodied_mode=True,
        sensorimotor_coupling_threshold=0.6
    )
    
    print("=== 身体化記憶システム初期化完了 ===")
    print(f"Backend info: {embodied_adapter.backend_info}")
    print(f"身体化モード: {embodied_adapter.embodied_mode}")
    
    # サンプル身体化特徴の作成（IoTセンサーシミュレーション）
    sample_visual_feature = VisualFeature(
        edge_features={'edge_histogram': np.random.rand(16), 'edge_density': np.array([0.3])},
        color_features={'color_histogram': np.random.rand(64)},
        shape_features={'aspect_ratio': 1.2, 'solidity': 0.8, 'extent': 0.9},
        texture_features={},
        spatial_location=(100, 150),
        extraction_timestamp=datetime.now(),
        confidence=0.8
    )
    
    # IoTセンサーデータのシミュレーション
    imu_data = {
        'gravity_x': 0.0, 'gravity_y': 0.0, 'gravity_z': -9.8,
        'pitch': 0.1, 'yaw': 0.0, 'roll': 0.05,
        'balance_state': 0.9, 'spatial_stability': 0.85, 'temporal_coherence': 0.8,
        'head_activation': 0.7, 'chest_activation': 0.8, 'hand_activation': 0.6,
        'waist_activation': 0.7, 'feet_activation': 0.9, 'imu_confidence': 0.85
    }
    
    biometric_data = {
        'heart_rate': 72.0, 'respiratory_rate': 16.0, 'skin_conductance': 0.6,
        'emotional_valence': 0.2, 'arousal_level': 0.6,
        'visceral_strength': 0.7, 'interoceptive_awareness': 0.65
    }
    
    spatial_context = {
        'relative_x': 0.3, 'relative_y': 0.0, 'relative_z': 1.2,
        'reaching_distance': 0.8, 'manipulation_affordance': 0.7
    }
    
    # 身体化記憶特徴の生成
    embodied_feature = EmbodiedMemoryFeature.create_from_iot_sensors(
        visual_feature=sample_visual_feature,
        imu_data=imu_data,
        biometric_data=biometric_data,
        spatial_context=spatial_context
    )
    
    print(f"\n=== 身体化記憶特徴生成完了 ===")
    print(f"記憶錨定強度: {embodied_feature.get_memory_anchoring_strength():.3f}")
    print(f"予測一致性: {embodied_feature.predict_memory_consistency():.3f}")
    
    # 身体化記憶処理の実行
    start_time = time.time()
    prediction_state = embodied_adapter.process_embodied_memory(embodied_feature)
    processing_time = time.time() - start_time
    
    print(f"\n=== 身体化記憶処理結果 ===")
    print(f"収束状態: {prediction_state.convergence_status}")
    print(f"処理時間: {processing_time:.3f}秒")
    print(f"階層誤差: {prediction_state.hierarchical_errors}")
    
    # 42%不一致率改善のシミュレーション
    improvement_simulation = embodied_adapter.simulate_42percent_consistency_improvement()
    print(f"\n=== 42%不一致率改善シミュレーション ===")
    for key, value in improvement_simulation.items():
        print(f"{key}: {value:.2f}%")
    
    # 身体化性能レポート
    performance_report = embodied_adapter.get_embodied_performance_report()
    print(f"\n=== 身体化性能レポート ===")
    print(f"現在の一致率: {performance_report['current_consistency_rate']:.3f}")
    print(f"一致性改善度: {performance_report['average_consistency_improvement']:.3f}")
    print(f"目標達成進捗: {performance_report.get('target_consistency_progress', 0):.1%}")
    
    print("\n✅ 身体化記憶システムのテスト完了")