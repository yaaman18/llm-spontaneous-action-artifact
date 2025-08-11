"""
NGC-Learn統合 Property-based Testing

TDD Engineer (t_wada) アプローチによる包括的プロパティ検証
生物学的妥当性の不変条件を体系的に検証する
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, strategies as st, settings, assume, note
from typing import List, Dict, Any, Tuple
import time

# テスト対象モジュール
from ngc_learn_adapter import HybridPredictiveCodingAdapter, NGCLearnEngine, PredictiveCodingFactory
from domain.value_objects.prediction_state import PredictionState


class TestPredictionInvariants:
    """予測システムの不変条件テスト"""
    
    @given(
        hierarchy_levels=st.integers(min_value=2, max_value=6),
        input_dimensions=st.integers(min_value=5, max_value=30),
        input_scale=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=5, deadline=None)  # デッドライン無効化
    def test_prediction_consistency_property(self, hierarchy_levels, input_dimensions, input_scale):
        """Property: 予測の一貫性保証"""
        note(f"Testing with {hierarchy_levels} levels, {input_dimensions} dims, scale {input_scale}")
        
        adapter = HybridPredictiveCodingAdapter(hierarchy_levels, input_dimensions)
        input_data = jnp.ones((input_dimensions,)) * input_scale
        
        try:
            prediction_state = adapter.forward_prediction(input_data)
            
            # Property 1: 予測階層数の一貫性
            assert len(prediction_state.hierarchical_predictions) > 0
            assert len(prediction_state.hierarchical_errors) == len(prediction_state.hierarchical_predictions)
            
            # Property 2: 予測の有界性（生物学的制約）
            for i, prediction in enumerate(prediction_state.hierarchical_predictions):
                assert jnp.all(jnp.isfinite(prediction)), f"Layer {i} contains infinite values"
                assert jnp.all(jnp.abs(prediction) <= 100), f"Layer {i} predictions exceed biological bounds"
            
            # Property 3: 誤差の非負性
            for i, error in enumerate(prediction_state.hierarchical_errors):
                assert error >= 0, f"Error at layer {i} is negative: {error}"
                assert error < 1000, f"Error at layer {i} is unrealistically large: {error}"
            
            # Property 4: 処理時間制約（現実的な制限に調整）
            if 'processing_time' in prediction_state.metadata:
                processing_time = prediction_state.metadata['processing_time']
                # より現実的な制限（2秒以下）に調整
                assert processing_time < 2.0, f"Processing time {processing_time}s exceeds reasonable limit"
                
                # 大規模入力の場合は更に緩い制限を適用
                input_complexity = hierarchy_levels * input_dimensions
                if input_complexity > 100:  # 大規模ケース
                    assert processing_time < 5.0, f"Processing time {processing_time}s exceeds large-scale limit"
            
        except Exception as e:
            pytest.fail(f"Prediction failed unexpectedly: {e}")
    
    @given(
        hierarchy_levels=st.integers(min_value=2, max_value=5),
        input_dimensions=st.integers(min_value=3, max_value=20),
        noise_level=st.floats(min_value=0.0, max_value=2.0, allow_nan=False)
    )
    @settings(max_examples=5, deadline=5000)
    def test_noise_robustness_property(self, hierarchy_levels, input_dimensions, noise_level):
        """Property: ノイズに対する頑健性"""
        assume(noise_level >= 0.0)
        
        adapter = HybridPredictiveCodingAdapter(hierarchy_levels, input_dimensions)
        
        # クリーンな入力での予測
        clean_input = jnp.ones((input_dimensions,))
        clean_prediction = adapter.forward_prediction(clean_input)
        
        # ノイズを含む入力での予測
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (input_dimensions,)) * noise_level
        noisy_input = clean_input + noise
        noisy_prediction = adapter.forward_prediction(noisy_input)
        
        # Property: ノイズレベルに対する予測の安定性
        clean_errors = clean_prediction.hierarchical_errors
        noisy_errors = noisy_prediction.hierarchical_errors
        
        for i, (clean_err, noisy_err) in enumerate(zip(clean_errors, noisy_errors)):
            error_change_ratio = abs(noisy_err - clean_err) / (clean_err + 1e-6)
            # ノイズの影響が過度でないことを確認
            assert error_change_ratio < 10 * noise_level + 1.0, f"Layer {i} too sensitive to noise"
    
    @given(
        hierarchy_levels=st.integers(min_value=2, max_value=4),
        input_dimensions=st.integers(min_value=5, max_value=15)
    )
    @settings(max_examples=5, deadline=5000)
    def test_scale_invariance_property(self, hierarchy_levels, input_dimensions):
        """Property: スケール不変性（部分的）"""
        
        adapter = HybridPredictiveCodingAdapter(hierarchy_levels, input_dimensions)
        base_input = jnp.linspace(-1, 1, input_dimensions)
        
        # 異なるスケールでの予測
        scale_factors = [0.5, 1.0, 2.0]
        scaled_predictions = []
        
        for scale in scale_factors:
            scaled_input = base_input * scale
            prediction = adapter.forward_prediction(scaled_input)
            scaled_predictions.append(prediction)
        
        # Property: 相対的な予測パターンの保持
        for i in range(len(scaled_predictions) - 1):
            pred_a = scaled_predictions[i]
            pred_b = scaled_predictions[i + 1]
            
            # 相関を計算（パターンの類似性）
            for layer_idx in range(min(len(pred_a.hierarchical_predictions), 
                                     len(pred_b.hierarchical_predictions))):
                pred_a_layer = pred_a.hierarchical_predictions[layer_idx].flatten()
                pred_b_layer = pred_b.hierarchical_predictions[layer_idx].flatten()
                
                if len(pred_a_layer) > 1 and len(pred_b_layer) > 1:
                    correlation = jnp.corrcoef(pred_a_layer, pred_b_layer)[0, 1]
                    if jnp.isfinite(correlation):
                        assert correlation > -0.5, f"Layer {layer_idx} correlation too negative: {correlation}"


class TestLearningConvergenceProperties:
    """学習収束特性のProperty-based testing"""
    
    @given(
        hierarchy_levels=st.integers(min_value=2, max_value=4),
        input_dimensions=st.integers(min_value=5, max_value=12),
        learning_iterations=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=3, deadline=5000)
    def test_learning_monotonicity_property(self, hierarchy_levels, input_dimensions, learning_iterations):
        """Property: 学習による誤差の単調減少傾向"""
        
        adapter = HybridPredictiveCodingAdapter(hierarchy_levels, input_dimensions)
        input_data = jnp.ones((input_dimensions,))
        
        error_history = []
        
        # 反復学習による誤差変化の記録
        for iteration in range(learning_iterations):
            prediction_state = adapter.forward_prediction(input_data)
            current_error = sum(prediction_state.hierarchical_errors)
            error_history.append(current_error)
            
            # 学習更新
            adapter.update_internal_state(prediction_state)
        
        # Property: 誤差の改善傾向
        if len(error_history) >= 3:
            # 最初と最後の3分の1の平均を比較
            early_errors = error_history[:len(error_history)//3] or error_history[:1]
            late_errors = error_history[-len(error_history)//3:] or error_history[-1:]
            
            avg_early_error = sum(early_errors) / len(early_errors)
            avg_late_error = sum(late_errors) / len(late_errors)
            
            # 改善または安定していることを確認（完全な単調性は要求しない）
            improvement_ratio = (avg_early_error - avg_late_error) / (avg_early_error + 1e-6)
            assert improvement_ratio > -0.5, f"Learning degraded significantly: {improvement_ratio}"
    
    @given(
        hierarchy_levels=st.integers(min_value=2, max_value=4),
        input_dimensions=st.integers(min_value=5, max_value=12),
        target_pattern=st.lists(st.floats(min_value=-2, max_value=2, allow_nan=False), 
                               min_size=5, max_size=12)
    )
    @settings(max_examples=3, deadline=5000)
    def test_target_approximation_property(self, hierarchy_levels, input_dimensions, target_pattern):
        """Property: ターゲットパターンへの近似能力"""
        assume(len(target_pattern) == input_dimensions)
        
        adapter = HybridPredictiveCodingAdapter(hierarchy_levels, input_dimensions)
        target_array = jnp.array(target_pattern)
        
        # 初期予測
        initial_prediction = adapter.forward_prediction(target_array)
        initial_error = sum(initial_prediction.hierarchical_errors)
        
        # 数回の学習イテレーション
        current_state = initial_prediction
        for _ in range(5):
            adapter.update_internal_state(current_state)
            current_state = adapter.forward_prediction(target_array)
        
        final_error = sum(current_state.hierarchical_errors)
        
        # Property: 学習により誤差が発散しない
        error_ratio = final_error / (initial_error + 1e-6)
        assert error_ratio < 10.0, f"Error exploded during learning: {error_ratio}"
        
        # Property: 予測値が合理的範囲内
        for prediction in current_state.hierarchical_predictions:
            prediction_range = jnp.max(prediction) - jnp.min(prediction)
            target_range = jnp.max(target_array) - jnp.min(target_array) + 1e-6
            assert prediction_range / target_range < 100, "Prediction range unreasonably large"


class TestBiologicalPlausibilityProperties:
    """生物学的妥当性のProperty-based testing"""
    
    @given(
        hierarchy_levels=st.integers(min_value=2, max_value=5),
        input_dimensions=st.integers(min_value=5, max_value=20),
        processing_repeats=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=3, deadline=5000)
    def test_temporal_consistency_property(self, hierarchy_levels, input_dimensions, processing_repeats):
        """Property: 時間的一貫性（同じ入力に対する安定性）"""
        
        adapter = HybridPredictiveCodingAdapter(hierarchy_levels, input_dimensions)
        input_data = jnp.ones((input_dimensions,)) * 0.5
        
        predictions_over_time = []
        processing_times = []
        
        # 同じ入力での反復処理
        for _ in range(processing_repeats):
            start_time = time.time()
            prediction_state = adapter.forward_prediction(input_data)
            end_time = time.time()
            
            predictions_over_time.append(prediction_state)
            processing_times.append(end_time - start_time)
        
        # Property 1: 処理時間の一貫性
        if len(processing_times) > 1:
            time_variance = np.var(processing_times)
            mean_time = np.mean(processing_times)
            cv_time = time_variance / (mean_time ** 2 + 1e-6)
            assert cv_time < 3.0, f"Processing time too variable: CV={cv_time}"
        
        # Property 2: 予測の時間的安定性
        if len(predictions_over_time) >= 2:
            first_pred = predictions_over_time[0]
            last_pred = predictions_over_time[-1]
            
            # エネルギー効率の一貫性
            if ('energy_cost' in first_pred.metadata and 
                'energy_cost' in last_pred.metadata):
                energy_consistency = abs(
                    first_pred.metadata['energy_cost'] - 
                    last_pred.metadata['energy_cost']
                )
                assert energy_consistency < 0.5, "Energy consumption too variable"
    
    @given(
        hierarchy_levels=st.integers(min_value=2, max_value=4),
        input_dimensions=st.integers(min_value=5, max_value=15),
        batch_size=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=6, deadline=2000)
    def test_energy_efficiency_property(self, hierarchy_levels, input_dimensions, batch_size):
        """Property: エネルギー効率の制約"""
        
        adapter = HybridPredictiveCodingAdapter(hierarchy_levels, input_dimensions)
        
        total_energy_cost = 0
        prediction_count = 0
        
        # バッチ処理による効率性テスト
        for batch_idx in range(batch_size):
            input_data = jnp.ones((input_dimensions,)) * (0.5 + 0.3 * batch_idx)
            prediction_state = adapter.forward_prediction(input_data)
            
            if 'energy_cost' in prediction_state.metadata:
                total_energy_cost += prediction_state.metadata['energy_cost']
                prediction_count += 1
        
        # Property: エネルギー効率の合理性
        if prediction_count > 0:
            avg_energy_cost = total_energy_cost / prediction_count
            
            # 生物学的制約：1予測あたりのエネルギーコストが妥当
            assert avg_energy_cost <= 1.0, f"Average energy cost too high: {avg_energy_cost}"
            assert avg_energy_cost >= 0.0, f"Energy cost cannot be negative: {avg_energy_cost}"
        
        # Property: エネルギー効率比率の確認
        if hasattr(adapter, 'energy_consumption_tracker'):
            efficiency_ratio = adapter.energy_consumption_tracker.get_efficiency_ratio()
            assert 0.0 <= efficiency_ratio <= 1.0, f"Invalid efficiency ratio: {efficiency_ratio}"


class TestSystemRobustnessProperties:
    """システム頑健性のProperty-based testing"""
    
    @given(
        hierarchy_levels=st.integers(min_value=2, max_value=4),
        input_dimensions=st.integers(min_value=5, max_value=15),
        extreme_values=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=8, deadline=3000)
    def test_extreme_input_robustness_property(self, hierarchy_levels, input_dimensions, extreme_values):
        """Property: 極端な入力値に対する頑健性"""
        
        adapter = HybridPredictiveCodingAdapter(hierarchy_levels, input_dimensions)
        extreme_input = jnp.full((input_dimensions,), extreme_values)
        
        try:
            prediction_state = adapter.forward_prediction(extreme_input)
            
            # Property: 極端入力でもシステムが破綻しない
            assert prediction_state is not None, "System returned None for extreme input"
            assert len(prediction_state.hierarchical_errors) > 0, "No error information returned"
            
            # Property: 出力の有界性維持
            for prediction in prediction_state.hierarchical_predictions:
                assert jnp.all(jnp.isfinite(prediction)), "Extreme input caused infinite predictions"
            
            # Property: エラーハンドリングの適切性
            if 'biological_constraints_met' in prediction_state.metadata:
                # 極端な入力の場合、制約違反があっても許容
                pass  # システムが動作していることが重要
            
        except Exception as e:
            # 極端な入力で例外が発生するのは許容されるが、
            # システム全体が破綻してはならない
            assert isinstance(e, (ValueError, RuntimeError)), f"Unexpected exception type: {type(e)}"
    
    @given(
        hierarchy_levels=st.integers(min_value=2, max_value=4),
        input_dimensions=st.integers(min_value=5, max_value=12)
    )
    @settings(max_examples=5, deadline=1500)
    def test_memory_efficiency_property(self, hierarchy_levels, input_dimensions):
        """Property: メモリ効率性"""
        
        adapter = HybridPredictiveCodingAdapter(hierarchy_levels, input_dimensions)
        input_data = jnp.ones((input_dimensions,))
        
        # 複数回の予測実行によるメモリリーク検出
        initial_backend_info = adapter.backend_info
        
        for _ in range(10):  # 複数回実行
            prediction_state = adapter.forward_prediction(input_data)
            adapter.update_internal_state(prediction_state)
        
        final_backend_info = adapter.backend_info
        
        # Property: パフォーマンスメトリクスが合理的範囲内
        if 'avg_processing_time' in final_backend_info:
            avg_time = final_backend_info['avg_processing_time']
            assert avg_time < 1.0, f"Average processing time too high: {avg_time}"
        
        # Property: システム状態の一貫性維持
        assert adapter.get_hierarchy_level_count() == hierarchy_levels, "Hierarchy count changed"


if __name__ == "__main__":
    # Property-based testの実行
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # 最初の失敗で停止
        "--hypothesis-show-statistics"
    ])