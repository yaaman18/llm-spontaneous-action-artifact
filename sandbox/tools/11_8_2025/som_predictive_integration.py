"""
SOM-予測符号化統合モジュール - TDD GREEN Phase

TDD専門家の指針に従い、最小限の実装でテストを通すことを目標とする。

実装方針（Clean Architecture Engineer指針）:
- Optional Injection Pattern による段階的統合
- 既存システムへの非破壊的統合
- インフラ層でのJAX実装分離

実装内容（DDD Engineer指針）:
- SOM状態と意識状態の集約関係
- ドメインサービスによる組織化メトリクス計算
"""

import numpy as np
import jax.numpy as jnp
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

# 既存システムのインポート
from domain.value_objects.phi_value import PhiValue
from domain.value_objects.consciousness_state import ConsciousnessState
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.spatial_organization_state import SpatialOrganizationState
from som_bmu import find_bmu_jax


@dataclass
class SOMIntegrationState:
    """SOM統合状態 - 最小実装"""
    som_contribution: float
    spatial_organization: float
    bmu_coordinates: Tuple[int, int]
    integration_confidence: float = 1.0


class SOMPredictiveIntegration:
    """
    SOM-予測符号化統合クラス - GREEN Phase 最小実装
    
    目的:
    - 予測符号化システムとSOMの基本的な統合
    - 既存意識レベル計算への寄与
    - Clean Architecture準拠の非破壊的統合
    """
    
    def __init__(self, 
                 som_map_size: Tuple[int, int] = (10, 10),
                 som_input_dim: int = 64):
        self.som_map_size = som_map_size
        self.som_input_dim = som_input_dim
        
        # SOM重みの簡易初期化
        self.som_weights = jnp.ones((*som_map_size, som_input_dim)) * 0.1
        
        self.integration_history = []
        self.logger = logging.getLogger(__name__)
    
    def integrate_prediction_state(self, 
                                 prediction_state: PredictionState) -> SOMIntegrationState:
        """
        予測状態をSOM空間に統合 - 最小実装
        
        Args:
            prediction_state: 既存の予測状態
        
        Returns:
            SOM統合状態
        """
        # 予測誤差をSOM入力ベクトルに変換（簡易版）
        som_input = self._convert_prediction_to_som_input(prediction_state)
        
        # BMU検索
        bmu_coords = find_bmu_jax(self.som_weights, som_input)
        bmu_row, bmu_col = int(bmu_coords[0]), int(bmu_coords[1])
        
        # 空間組織化度計算（簡易版）
        spatial_organization = self._compute_spatial_organization(som_input, (bmu_row, bmu_col))
        
        # SOM寄与度計算
        som_contribution = self._compute_som_contribution(spatial_organization, prediction_state)
        
        # 統合状態の構築
        integration_state = SOMIntegrationState(
            som_contribution=som_contribution,
            spatial_organization=spatial_organization,
            bmu_coordinates=(bmu_row, bmu_col),
            integration_confidence=0.8  # 初期実装での固定値
        )
        
        self.integration_history.append(integration_state)
        return integration_state
    
    def enhance_consciousness_state(self, 
                                  consciousness_state: ConsciousnessState,
                                  integration_state: SOMIntegrationState) -> ConsciousnessState:
        """
        意識状態にSOM情報を統合して拡張
        
        Args:
            consciousness_state: 既存の意識状態
            integration_state: SOM統合状態
        
        Returns:
            SOM情報で拡張された新しい意識状態
        """
        # 既存のΦ値にSOM寄与度を反映
        enhanced_phi = self._enhance_phi_value_with_som(
            consciousness_state.phi_value, 
            integration_state
        )
        
        # SOM統合状態を用いて新しい空間組織化状態を作成
        enhanced_spatial_org = consciousness_state.spatial_organization.with_updated_position(
            integration_state.bmu_coordinates
        ).with_updated_quality_metrics(
            organization_quality=integration_state.spatial_organization,
            phenomenological_quality=integration_state.spatial_organization * 0.8
        )
        
        # 新しい意識状態を構築（既存状態を基本として）
        enhanced_consciousness_state = ConsciousnessState(
            phi_value=enhanced_phi,
            prediction_state=consciousness_state.prediction_state,
            uncertainty_distribution=consciousness_state.uncertainty_distribution,
            spatial_organization=enhanced_spatial_org,
            metacognitive_confidence=consciousness_state.metacognitive_confidence,
            attention_weights=consciousness_state.attention_weights
        )
        
        return enhanced_consciousness_state
    
    def _convert_prediction_to_som_input(self, prediction_state: PredictionState) -> jnp.ndarray:
        """予測状態をSOM入力ベクトルに変換"""
        # 階層誤差を固定長ベクトルに変換（最小実装）
        errors = prediction_state.hierarchical_errors[:self.som_input_dim]
        
        # 足りない次元はゼロパディング
        if len(errors) < self.som_input_dim:
            padding = [0.0] * (self.som_input_dim - len(errors))
            errors.extend(padding)
        
        return jnp.array(errors[:self.som_input_dim])
    
    def _compute_spatial_organization(self, 
                                    som_input: jnp.ndarray, 
                                    bmu_coords: Tuple[int, int]) -> float:
        """空間組織化度の計算"""
        # BMU重みとの類似度を組織化度とする（簡易版）
        bmu_weights = self.som_weights[bmu_coords[0], bmu_coords[1]]
        similarity = 1.0 - jnp.linalg.norm(som_input - bmu_weights)
        
        # [0, 1] の範囲にクランプ
        return float(jnp.clip(similarity, 0.0, 1.0))
    
    def _compute_som_contribution(self, 
                                spatial_organization: float, 
                                prediction_state: PredictionState) -> float:
        """SOMの意識レベルへの寄与度計算"""
        # 空間組織化度と予測精度の組み合わせ
        prediction_accuracy = 1.0 - prediction_state.mean_error
        som_contribution = spatial_organization * prediction_accuracy * 0.15  # 15%の重み
        
        return float(jnp.clip(som_contribution, 0.0, 1.0))
    
    def _enhance_phi_value_with_som(self, 
                                  original_phi: PhiValue,
                                  integration_state: SOMIntegrationState) -> PhiValue:
        """Φ値にSOM情報を統合"""
        # SOM寄与を統合して新しいΦ値を計算
        enhanced_value = original_phi.value + integration_state.som_contribution
        enhanced_integration = min(original_phi.integration + integration_state.spatial_organization * 0.1, 1.0)
        
        # 新しいPhiValueインスタンスを作成（不変オブジェクト）
        enhanced_phi = PhiValue(
            value=enhanced_value,
            complexity=original_phi.complexity,
            integration=enhanced_integration,
            system_size=original_phi.system_size,
            computation_method=original_phi.computation_method,
            confidence=min(original_phi.confidence, integration_state.integration_confidence),
            metadata={**original_phi.metadata, 'som_enhanced': True}
        )
        
        return enhanced_phi
    
    def integrate_prediction_with_som(self, prediction_state, som_state):
        """
        予測状態とSOM状態を統合 - 未実装
        
        TDD RED Phase: このメソッドはまだ実装されていません
        """
        raise NotImplementedError("integrate_prediction_with_som method not yet implemented")
    
    def advanced_som_mapping(self, prediction_state):
        """
        高度なSOMマッピング機能 - 未実装
        
        TDD RED Phase: このメソッドはまだ実装されていません
        """
        raise NotImplementedError("advanced_som_mapping method not yet implemented")
    
    def deep_learning_integration(self, neural_network, training_data):
        """
        深層学習統合機能 - 未実装
        
        TDD RED Phase: このメソッドはまだ実装されていません
        """
        raise NotImplementedError("deep_learning_integration method not yet implemented")
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """
        統合メトリクスの取得
        
        SOM統合システムの性能・品質指標を収集して返す。
        統合回数、平均寄与度、空間組織化度、成功率等を含む。
        
        Returns:
            統合メトリクスの辞書
        """
        if not self.integration_history:
            return {
                'integration_count': 0,
                'avg_som_contribution': 0.0,
                'avg_spatial_organization': 0.0,
                'success_rate': 0.0
            }
        
        return {
            'integration_count': len(self.integration_history),
            'avg_som_contribution': np.mean([s.som_contribution for s in self.integration_history]),
            'avg_spatial_organization': np.mean([s.spatial_organization for s in self.integration_history]),
            'success_rate': sum(1 for s in self.integration_history if s.integration_confidence > 0.5) / len(self.integration_history)
        }


# ユーティリティ関数（テスト用）
def map_prediction_to_som(prediction_state: PredictionState, 
                         som_integration: SOMPredictiveIntegration) -> SOMIntegrationState:
    """予測状態をSOMにマッピング - ユーティリティ関数"""
    return som_integration.integrate_prediction_state(prediction_state)


# 使用例とテスト
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # サンプル予測状態の作成
    sample_prediction_state = PredictionState(
        hierarchical_errors=[0.1, 0.2, 0.3, 0.15],
        convergence_status="converging",
        learning_iteration=10
    )
    
    # SOM統合システムの作成
    som_integration = SOMPredictiveIntegration()
    
    # 統合実行
    integration_state = som_integration.integrate_prediction_state(sample_prediction_state)
    
    print(f"SOM統合完了:")
    print(f"  BMU座標: {integration_state.bmu_coordinates}")
    print(f"  空間組織化度: {integration_state.spatial_organization:.3f}")
    print(f"  SOM寄与度: {integration_state.som_contribution:.3f}")
    
    # 統合メトリクス
    metrics = som_integration.get_integration_metrics()
    print(f"\n統合メトリクス: {metrics}")