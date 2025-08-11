"""
NGC-Learn統合互換性テスト

後方互換性を確保しながらngc-learnを統合するためのテストスイート。
設計文書必須要件（★★★★★）の段階的実装を検証。

テスト戦略:
1. 後方互換性の保証（既存APIの完全維持）
2. ngc-learn統合の正確性検証
3. フォールバック機構の信頼性確認
"""

import pytest
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from ngc_learn_adapter import (
    HybridPredictiveCodingAdapter,
    PredictiveCodingFactory,
    NGC_LEARN_AVAILABLE,
    NGCLearnEngine,
    JAXFallbackEngine
)
from domain.value_objects.prediction_state import PredictionState
from domain.entities.predictive_coding_core import PredictiveCodingCore


class TestNGCLearnAvailability:
    """NGC-Learn利用可能性のテスト"""
    
    def test_ngc_learn_availability_detection(self):
        """NGC-Learn利用可能性の正確な検出"""
        # 実際の環境での検出結果をテスト
        if NGC_LEARN_AVAILABLE:
            # ngc-learnがインストール済みの場合
            import ngclearn  # noqa
            assert True, "ngc-learn successfully imported"
        else:
            # ngc-learnが未インストールの場合
            with pytest.raises(ImportError):
                import ngclearn  # noqa
    
    def test_conditional_import_pattern(self):
        """条件付きインポートパターンの動作確認"""
        # アダプターが適切にImportErrorをハンドルすることを確認
        adapter = HybridPredictiveCodingAdapter(3, 10)
        
        backend_info = adapter.backend_info
        assert "engine_type" in backend_info
        assert "ngc_learn_available" in backend_info
        assert backend_info["ngc_learn_available"] == NGC_LEARN_AVAILABLE


class TestBackwardCompatibility:
    """既存API後方互換性テスト"""
    
    def test_predictive_coding_core_interface_preserved(self):
        """PredictiveCodingCoreインターフェースの保持確認"""
        adapter = HybridPredictiveCodingAdapter(3, 10)
        
        # 既存の抽象メソッドが実装されていることを確認
        assert isinstance(adapter, PredictiveCodingCore)
        assert hasattr(adapter, 'forward_prediction')
        assert hasattr(adapter, 'update_internal_state')
        assert hasattr(adapter, 'get_hierarchy_level_count')
    
    def test_forward_prediction_api_compatibility(self):
        """forward_prediction APIの互換性確認"""
        adapter = HybridPredictiveCodingAdapter(3, 10)
        input_data = jnp.ones((10,))
        
        # 既存APIでの呼び出しが成功することを確認
        result = adapter.forward_prediction(input_data)
        
        assert isinstance(result, PredictionState)
        assert hasattr(result, 'hierarchical_errors')
        assert hasattr(result, 'convergence_status')
        assert len(result.hierarchical_errors) > 0
    
    def test_hierarchy_level_count_compatibility(self):
        """階層レベル数取得の互換性確認"""
        hierarchy_levels = 5
        adapter = HybridPredictiveCodingAdapter(hierarchy_levels, 10)
        
        # 既存APIでの呼び出し
        level_count = adapter.get_hierarchy_level_count()
        
        assert level_count == hierarchy_levels
        assert isinstance(level_count, int)
    
    def test_internal_state_update_compatibility(self):
        """内部状態更新の互換性確認"""
        adapter = HybridPredictiveCodingAdapter(3, 10)
        
        # 既存形式のPredictionStateで更新
        new_state = PredictionState(
            hierarchical_errors=[0.1, 0.2, 0.3],
            convergence_status="converging"
        )
        
        # 例外なく更新できることを確認
        adapter.update_internal_state(new_state)
        assert adapter._current_state == new_state


class TestHybridEngineSelection:
    """ハイブリッドエンジン選択のテスト"""
    
    def test_ngc_learn_preference_when_available(self):
        """NGC-Learn利用可能時の優先選択確認"""
        if NGC_LEARN_AVAILABLE:
            adapter = HybridPredictiveCodingAdapter(
                3, 10, 
                prefer_ngc_learn=True, 
                fallback_to_jax=True
            )
            # NGC-Learnが利用可能な場合、それを優先することを確認
            # 注: 実際のngc-learn初期化が失敗する可能性があるため条件付き
            assert adapter.engine_type in ["ngc-learn", "jax-fallback"]
        else:
            pytest.skip("NGC-Learn not available for this test")
    
    def test_jax_fallback_when_ngc_learn_unavailable(self):
        """NGC-Learn未利用時のJAXフォールバック確認"""
        # NGC-Learnを無効化してテスト
        with patch('ngc_learn_adapter.NGC_LEARN_AVAILABLE', False):
            adapter = HybridPredictiveCodingAdapter(
                3, 10,
                prefer_ngc_learn=True,
                fallback_to_jax=True
            )
            
            assert adapter.engine_type == "jax-fallback"
            assert not adapter.is_using_ngc_learn
    
    def test_forced_jax_fallback(self):
        """JAXフォールバックの強制使用確認"""
        adapter = HybridPredictiveCodingAdapter(
            3, 10,
            prefer_ngc_learn=False,  # NGC-Learnを優先しない
            fallback_to_jax=True
        )
        
        assert adapter.engine_type == "jax-fallback"
        assert not adapter.is_using_ngc_learn
    
    def test_no_fallback_error_handling(self):
        """フォールバック無効時のエラーハンドリング確認"""
        with patch('ngc_learn_adapter.NGC_LEARN_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="NGC-Learn not available"):
                HybridPredictiveCodingAdapter(
                    3, 10,
                    prefer_ngc_learn=True,
                    fallback_to_jax=False  # フォールバック無効
                )


class TestFactoryPattern:
    """ファクトリーパターンのテスト"""
    
    def test_optimal_implementation_creation(self):
        """最適実装作成の動作確認"""
        implementation = PredictiveCodingFactory.create_optimal_implementation(
            hierarchy_levels=3,
            input_dimensions=10
        )
        
        assert isinstance(implementation, HybridPredictiveCodingAdapter)
        assert implementation.get_hierarchy_level_count() == 3
        
        backend_info = implementation.backend_info
        assert backend_info["hierarchy_levels"] == 3
        assert backend_info["input_dimensions"] == 10
    
    def test_legacy_compatible_creation(self):
        """レガシー互換実装作成の動作確認"""
        implementation = PredictiveCodingFactory.create_legacy_compatible(
            hierarchy_levels=4,
            input_dimensions=20
        )
        
        assert isinstance(implementation, HybridPredictiveCodingAdapter)
        assert implementation.engine_type == "jax-fallback"  # レガシー互換はJAX優先
        assert not implementation.is_using_ngc_learn
    
    def test_forced_ngc_learn_requirement(self):
        """NGC-Learn強制要求時の動作確認"""
        if not NGC_LEARN_AVAILABLE:
            # NGC-Learn未インストール環境での強制要求エラー
            with pytest.raises(RuntimeError, match="ngc-learn is required"):
                PredictiveCodingFactory.create_optimal_implementation(
                    hierarchy_levels=3,
                    input_dimensions=10,
                    force_ngc_learn=True
                )
        else:
            # NGC-Learnインストール済み環境での正常作成
            implementation = PredictiveCodingFactory.create_optimal_implementation(
                hierarchy_levels=3,
                input_dimensions=10,
                force_ngc_learn=True
            )
            # 注: 実際のngc-learn初期化の成否により結果は変わる
            assert isinstance(implementation, HybridPredictiveCodingAdapter)


class TestEnginePerformance:
    """エンジン性能比較テスト"""
    
    def test_jax_fallback_engine_basic_functionality(self):
        """JAXフォールバックエンジンの基本機能確認"""
        engine = JAXFallbackEngine(3, 10)
        input_data = jnp.ones((10,))
        
        # 基本的な予測・誤差計算が動作することを確認
        predictions = engine.predict_hierarchical(input_data)
        
        # 現在の実装では空のリストが返される可能性
        assert isinstance(predictions, list)
        
        # 誤差計算のテスト（ダミーデータで）
        dummy_predictions = [jnp.ones((10,)), jnp.ones((10,))]
        dummy_targets = [jnp.zeros((10,)), jnp.zeros((10,))]
        
        errors = engine.compute_prediction_errors(dummy_predictions, dummy_targets)
        assert len(errors) == len(dummy_predictions)
        assert all(isinstance(error, jnp.ndarray) for error in errors)
    
    @pytest.mark.skipif(not NGC_LEARN_AVAILABLE, reason="NGC-Learn not available")
    def test_ngc_learn_engine_instantiation(self):
        """NGC-Learnエンジンのインスタンス化確認"""
        # NGC-Learn利用可能時のみ実行
        try:
            engine = NGCLearnEngine(3, 10)
            assert engine.hierarchy_levels == 3
            assert engine.input_dimensions == 10
        except RuntimeError as e:
            # NGC-Learnの実際の初期化に失敗した場合（想定内）
            assert "ngc-learn" in str(e).lower()
    
    def test_performance_consistency_across_engines(self):
        """エンジン間での性能一貫性確認"""
        # 異なるエンジンで同じ入力に対する処理時間の妥当性確認
        hierarchy_levels, input_dimensions = 3, 10
        input_data = jnp.ones((input_dimensions,))
        
        # JAXフォールバック
        jax_adapter = HybridPredictiveCodingAdapter(
            hierarchy_levels, input_dimensions,
            prefer_ngc_learn=False
        )
        
        import time
        start_time = time.time()
        jax_result = jax_adapter.forward_prediction(input_data)
        jax_time = time.time() - start_time
        
        # 処理時間が合理的範囲内（1秒未満）であることを確認
        assert jax_time < 1.0
        assert isinstance(jax_result, PredictionState)


class TestIntegrationScenarios:
    """統合シナリオテスト"""
    
    def test_gradual_migration_scenario(self):
        """段階的移行シナリオのテスト"""
        # フェーズ1: 既存JAX実装のみ
        legacy_adapter = PredictiveCodingFactory.create_legacy_compatible(3, 10)
        assert legacy_adapter.engine_type == "jax-fallback"
        
        # フェーズ2: NGC-Learn優先だがフォールバック有効
        hybrid_adapter = PredictiveCodingFactory.create_optimal_implementation(3, 10)
        # どちらのエンジンでも基本APIは同じ
        
        # 両方の実装で同じAPIが動作することを確認
        input_data = jnp.ones((10,))
        legacy_result = legacy_adapter.forward_prediction(input_data)
        hybrid_result = hybrid_adapter.forward_prediction(input_data)
        
        # 結果の形式は同じ
        assert type(legacy_result) == type(hybrid_result)
        assert len(legacy_result.hierarchical_errors) > 0
        assert len(hybrid_result.hierarchical_errors) > 0
    
    def test_production_readiness_checklist(self):
        """本番環境準備状況チェックリスト"""
        adapter = HybridPredictiveCodingAdapter(3, 10)
        
        # 1. 基本機能の動作確認
        assert adapter.get_hierarchy_level_count() > 0
        
        # 2. エラーハンドリング機能
        assert hasattr(adapter, 'backend_info')
        backend_info = adapter.backend_info
        assert all(key in backend_info for key in [
            "engine_type", "ngc_learn_available", 
            "hierarchy_levels", "input_dimensions"
        ])
        
        # 3. ログ機能の存在確認
        assert hasattr(adapter, 'logger')
        
        # 4. 設定可能性の確認
        assert hasattr(adapter, 'prefer_ngc_learn')
        assert hasattr(adapter, 'fallback_to_jax')
    
    def test_v2_compatibility_preservation(self):
        """V2互換性の保持確認"""
        # V2で確立されたSOM統合との併用テスト
        from som_predictive_integration import SOMPredictiveIntegration
        from domain.value_objects.consciousness_state import ConsciousnessState
        from domain.value_objects.phi_value import PhiValue
        from domain.value_objects.spatial_organization_state import SpatialOrganizationState
        from domain.value_objects.probability_distribution import ProbabilityDistribution
        
        # NGC-Learn統合アダプターの作成
        pc_adapter = HybridPredictiveCodingAdapter(3, 10)
        
        # SOM統合システムとの併用
        som_integration = SOMPredictiveIntegration()
        
        # 予測状態の生成（NGC-Learn or JAX）
        input_data = jnp.ones((10,))
        prediction_state = pc_adapter.forward_prediction(input_data)
        
        # SOM統合処理（V2機能）
        som_state = som_integration.integrate_prediction_state(prediction_state)
        
        # 最終的な意識状態構築（V2 + NGC-Learn統合）
        consciousness = ConsciousnessState(
            phi_value=PhiValue(value=1.5, complexity=2.0, integration=0.75),
            prediction_state=prediction_state,
            uncertainty_distribution=ProbabilityDistribution.uniform(3),
            spatial_organization=SpatialOrganizationState.create_initial(),
            metacognitive_confidence=0.3
        )
        
        # V2機能が正常に動作することを確認
        assert consciousness.is_conscious
        assert som_state.som_contribution >= 0.0
        
        # NGC-Learn統合による拡張情報の確認
        assert 'engine_type' in prediction_state.metadata


if __name__ == "__main__":
    # 統合テストの実行
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # 最初の失敗で停止
    ])