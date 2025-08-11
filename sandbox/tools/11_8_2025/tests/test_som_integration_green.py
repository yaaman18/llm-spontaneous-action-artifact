"""
SOM統合のGREENフェーズ・テスト

TDD GREEN Phase: 実装した統合コードが正しく動作することを確認
- SOM-予測符号化統合の正確性
- 既存システムとの非破壊的統合
- 意識レベル計算への適切な寄与
"""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, Any

# 既存システムのインポート
from domain.value_objects.phi_value import PhiValue
from domain.value_objects.consciousness_state import ConsciousnessState
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.probability_distribution import ProbabilityDistribution
from domain.value_objects.spatial_organization_state import SpatialOrganizationState

# SOM統合システムのインポート
from som_predictive_integration import (
    SOMPredictiveIntegration, 
    SOMIntegrationState,
    map_prediction_to_som
)


class TestSOMIntegrationGreen:
    """GREEN Phase: SOM統合実装の正確性確認"""
    
    def test_som_predictive_integration_class_creation(self):
        """SOM統合クラスが正しく作成される"""
        integration = SOMPredictiveIntegration(som_map_size=(8, 8), som_input_dim=32)
        
        assert integration.som_map_size == (8, 8)
        assert integration.som_input_dim == 32
        assert integration.som_weights.shape == (8, 8, 32)
        assert len(integration.integration_history) == 0
    
    def test_basic_prediction_state_integration(self):
        """基本的な予測状態統合が動作する"""
        # Arrange
        integration = SOMPredictiveIntegration()
        prediction_state = PredictionState(
            hierarchical_errors=[0.1, 0.2, 0.15, 0.05],
            convergence_status="converging",
            learning_iteration=5
        )
        
        # Act
        integration_state = integration.integrate_prediction_state(prediction_state)
        
        # Assert
        assert isinstance(integration_state, SOMIntegrationState)
        assert 0.0 <= integration_state.som_contribution <= 1.0
        assert 0.0 <= integration_state.spatial_organization <= 1.0
        assert isinstance(integration_state.bmu_coordinates, tuple)
        assert len(integration_state.bmu_coordinates) == 2
        assert 0 <= integration_state.bmu_coordinates[0] < 10  # default map size
        assert 0 <= integration_state.bmu_coordinates[1] < 10
    
    def test_consciousness_state_enhancement(self):
        """意識状態の拡張が正しく動作する"""
        # Arrange
        integration = SOMPredictiveIntegration()
        
        original_phi = PhiValue(value=1.0, complexity=2.0, integration=0.5)
        original_consciousness = ConsciousnessState(
            phi_value=original_phi,
            prediction_state=PredictionState(hierarchical_errors=[0.1, 0.2]),
            uncertainty_distribution=ProbabilityDistribution.uniform(3),
            spatial_organization=SpatialOrganizationState.create_initial(),
            metacognitive_confidence=0.7
        )
        
        som_integration_state = SOMIntegrationState(
            som_contribution=0.2,
            spatial_organization=0.8,
            bmu_coordinates=(3, 4),
            integration_confidence=0.9
        )
        
        # Act
        enhanced_consciousness = integration.enhance_consciousness_state(
            original_consciousness, som_integration_state
        )
        
        # Assert
        assert enhanced_consciousness.phi_value.value > original_phi.value
        assert enhanced_consciousness.phi_value.integration >= original_phi.integration
        assert enhanced_consciousness.phi_value.metadata.get('som_enhanced') is True
        
        # 他の属性が保持されることを確認
        assert enhanced_consciousness.prediction_state == original_consciousness.prediction_state
        assert enhanced_consciousness.uncertainty_distribution == original_consciousness.uncertainty_distribution
    
    def test_prediction_to_som_mapping_function(self):
        """予測状態のSOMマッピング関数が動作する"""
        # Arrange
        integration = SOMPredictiveIntegration(som_input_dim=8)
        prediction_state = PredictionState(
            hierarchical_errors=[0.3, 0.1, 0.25, 0.05, 0.2, 0.15],
            convergence_status="not_converged"
        )
        
        # Act
        som_state = map_prediction_to_som(prediction_state, integration)
        
        # Assert
        assert isinstance(som_state, SOMIntegrationState)
        assert som_state.som_contribution >= 0.0
        assert som_state.spatial_organization >= 0.0
    
    def test_different_prediction_patterns(self):
        """異なる予測パターンでの統合テスト"""
        integration = SOMPredictiveIntegration()
        
        test_patterns = {
            'converged': [0.01, 0.02, 0.005],
            'diverged': [0.1, 0.3, 0.7, 1.0],
            'converging': [0.2, 0.05, 0.4, 0.1, 0.3]
        }
        
        results = {}
        
        for pattern_name, errors in test_patterns.items():
            prediction_state = PredictionState(
                hierarchical_errors=errors,
                convergence_status=pattern_name
            )
            
            som_state = integration.integrate_prediction_state(prediction_state)
            results[pattern_name] = som_state
        
        # 収束したパターンがより良い組織化を示すことを確認
        assert results['converged'].spatial_organization >= results['diverged'].spatial_organization
        assert results['converged'].som_contribution >= results['diverged'].som_contribution
    
    def test_integration_metrics_tracking(self):
        """統合メトリクスの追跡機能"""
        # Arrange
        integration = SOMPredictiveIntegration()
        
        # 複数回の統合実行
        for i in range(5):
            prediction_state = PredictionState(
                hierarchical_errors=[0.1 + i*0.05, 0.2 - i*0.02],
                learning_iteration=i
            )
            integration.integrate_prediction_state(prediction_state)
        
        # Act
        metrics = integration.get_integration_metrics()
        
        # Assert
        assert metrics['integration_count'] == 5
        assert 0.0 <= metrics['avg_som_contribution'] <= 1.0
        assert 0.0 <= metrics['avg_spatial_organization'] <= 1.0
        assert 0.0 <= metrics['success_rate'] <= 1.0
    
    def test_edge_cases_handling(self):
        """境界条件の処理"""
        integration = SOMPredictiveIntegration()
        
        # 最小限の誤差リスト
        minimal_prediction = PredictionState(hierarchical_errors=[0.001])
        som_state = integration.integrate_prediction_state(minimal_prediction)
        assert som_state is not None
        
        # 非常に大きな誤差
        large_error_prediction = PredictionState(hierarchical_errors=[10.0, 50.0, 100.0])
        som_state = integration.integrate_prediction_state(large_error_prediction)
        assert 0.0 <= som_state.som_contribution <= 1.0
        
        # 非常に小さな誤差
        tiny_error_prediction = PredictionState(hierarchical_errors=[1e-10, 1e-9])
        som_state = integration.integrate_prediction_state(tiny_error_prediction)
        assert som_state.spatial_organization >= 0.0
    
    def test_consistency_across_multiple_runs(self):
        """複数回実行での一貫性"""
        integration = SOMPredictiveIntegration()
        
        # 同じ入力での複数回実行
        prediction_state = PredictionState(
            hierarchical_errors=[0.1, 0.2, 0.15],
            convergence_status="converged"
        )
        
        results = []
        for _ in range(3):
            som_state = integration.integrate_prediction_state(prediction_state)
            results.append(som_state)
        
        # 結果の一貫性を確認
        for i in range(1, len(results)):
            assert results[i].bmu_coordinates == results[0].bmu_coordinates
            assert abs(results[i].spatial_organization - results[0].spatial_organization) < 1e-10
    
    def test_integration_confidence_calculation(self):
        """統合信頼度の計算"""
        integration = SOMPredictiveIntegration()
        
        # 高品質な予測状態
        good_prediction = PredictionState(
            hierarchical_errors=[0.01, 0.02],
            convergence_status="converged"
        )
        
        good_som_state = integration.integrate_prediction_state(good_prediction)
        
        # 低品質な予測状態
        poor_prediction = PredictionState(
            hierarchical_errors=[1.0, 2.0, 3.0],
            convergence_status="diverged"
        )
        
        poor_som_state = integration.integrate_prediction_state(poor_prediction)
        
        # 品質に応じた信頼度の違いを確認
        assert good_som_state.integration_confidence >= poor_som_state.integration_confidence


class TestSOMIntegrationCompatibility:
    """既存システムとの互換性テスト"""
    
    def test_phi_value_enhancement_preserves_original_properties(self):
        """Φ値拡張が元の属性を保持する"""
        integration = SOMPredictiveIntegration()
        
        original_phi = PhiValue(
            value=2.5,
            complexity=3.0,
            integration=0.8,
            system_size=5,
            computation_method="exact",
            confidence=0.95,
            metadata={'test': 'value'}
        )
        
        som_state = SOMIntegrationState(
            som_contribution=0.1,
            spatial_organization=0.6,
            bmu_coordinates=(2, 3)
        )
        
        enhanced_phi = integration._enhance_phi_value_with_som(original_phi, som_state)
        
        # 重要な属性が保持されることを確認
        assert enhanced_phi.complexity == original_phi.complexity
        assert enhanced_phi.system_size == original_phi.system_size
        assert enhanced_phi.computation_method == original_phi.computation_method
        assert enhanced_phi.metadata['test'] == 'value'
        assert enhanced_phi.metadata['som_enhanced'] is True
    
    def test_consciousness_state_structure_preservation(self):
        """意識状態構造の保持"""
        integration = SOMPredictiveIntegration()
        
        original_consciousness = ConsciousnessState(
            phi_value=PhiValue(value=1.0, complexity=1.5, integration=0.6),
            prediction_state=PredictionState(hierarchical_errors=[0.1, 0.2]),
            uncertainty_distribution=ProbabilityDistribution.uniform(4),
            spatial_organization=SpatialOrganizationState.create_initial(),
            metacognitive_confidence=0.6,
            attention_weights=np.array([0.3, 0.7])
        )
        
        som_state = SOMIntegrationState(
            som_contribution=0.05,
            spatial_organization=0.4,
            bmu_coordinates=(1, 1)
        )
        
        enhanced = integration.enhance_consciousness_state(original_consciousness, som_state)
        
        # 構造の保持を確認
        assert type(enhanced) == ConsciousnessState
        assert enhanced.prediction_state == original_consciousness.prediction_state
        assert enhanced.uncertainty_distribution == original_consciousness.uncertainty_distribution
        assert enhanced.metacognitive_confidence == original_consciousness.metacognitive_confidence
        assert np.array_equal(enhanced.attention_weights, original_consciousness.attention_weights)
    
    def test_non_destructive_integration(self):
        """非破壊的統合の確認"""
        integration = SOMPredictiveIntegration()
        
        # 元のオブジェクトを作成
        original_prediction = PredictionState(hierarchical_errors=[0.1, 0.2, 0.3])
        original_phi = PhiValue(value=1.5, complexity=2.0, integration=0.75)
        original_consciousness = ConsciousnessState(
            phi_value=original_phi,
            prediction_state=original_prediction,
            uncertainty_distribution=ProbabilityDistribution.uniform(3),
            spatial_organization=SpatialOrganizationState.create_initial()
        )
        
        # 統合処理実行
        som_state = integration.integrate_prediction_state(original_prediction)
        enhanced_consciousness = integration.enhance_consciousness_state(original_consciousness, som_state)
        
        # 元のオブジェクトが変更されていないことを確認
        assert original_phi.value == 1.5
        assert original_phi.metadata == {}
        assert original_consciousness.phi_value == original_phi
        
        # 新しいオブジェクトが作成されていることを確認
        assert enhanced_consciousness != original_consciousness
        assert enhanced_consciousness.phi_value != original_phi


if __name__ == "__main__":
    # GREEN Phaseテストの実行
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # 最初の失敗で停止
    ])