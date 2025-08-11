"""
SOM BMU計算のGREENフェーズ・テスト

TDD GREEN Phase: 実装したコードが正しく動作することを確認
- 基本的なBMU検索の正確性
- 距離計算の正確性
- 境界条件の処理
- JAX互換性
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from hypothesis import given, strategies as st, settings

from som_bmu import find_bmu_jax, find_bmu_vectorized, find_bmu_jit_optimized, SOMBMUCalculator


class TestBMUGreenPhase:
    """GREEN Phase: BMU実装の正確性確認"""
    
    def test_bmu_calculator_class_exists(self):
        """BMU Calculatorクラスが正しく作成される"""
        calculator = SOMBMUCalculator()
        assert calculator.name == "SOM BMU Calculator"
    
    def test_find_bmu_basic_functionality(self):
        """基本的なBMU検索が正しく動作する"""
        # 3x3のSOMマップ、4次元入力
        weights = jnp.array([
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5], [1.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0], [0.3, 0.3, 0.3, 0.1], [0.8, 0.2, 0.0, 0.0]]
        ])
        
        # 入力ベクトル [1.0, 0.0, 0.0, 0.0] に最も近いのは weights[0,0]
        input_vector = jnp.array([1.0, 0.0, 0.0, 0.0])
        row, col = find_bmu_jax(weights, input_vector)
        row, col = int(row), int(col)
        
        assert row == 0 and col == 0, f"Expected BMU at (0,0), got ({row},{col})"
    
    def test_find_bmu_distance_calculation_accuracy(self):
        """距離計算の正確性テスト"""
        # 既知の距離を持つ単純なケース
        weights = jnp.array([
            [[0.0, 0.0], [3.0, 4.0]],  # (0,1)までの距離は5.0
            [[1.0, 1.0], [0.0, 0.0]]   # (1,0)までの距離は√2 ≈ 1.414
        ])
        
        input_vector = jnp.array([0.0, 0.0])
        row, col = find_bmu_jax(weights, input_vector)
        row, col = int(row), int(col)
        
        # 最も近いのは (0,0) または (1,1) (どちらも距離0)
        # JAXの実装では最初に見つかった最小値を返すので (0,0) が期待される
        assert (row == 0 and col == 0) or (row == 1 and col == 1)
    
    def test_find_bmu_edge_cases(self):
        """境界条件テスト"""
        # 1x1マップでのBMU検索
        weights_1x1 = jnp.array([[[1.0, 2.0, 3.0]]])
        input_vector = jnp.array([0.0, 0.0, 0.0])
        row, col = find_bmu_jax(weights_1x1, input_vector)
        row, col = int(row), int(col)
        assert row == 0 and col == 0, "1x1マップではBMUは(0,0)であるべき"
        
        # 零ベクトル入力の処理
        weights_zero = jnp.zeros((2, 2, 3))
        input_zero = jnp.zeros((3,))
        row, col = find_bmu_jax(weights_zero, input_zero)
        row, col = int(row), int(col)
        assert 0 <= row < 2 and 0 <= col < 2, "零ベクトル入力でも有効な座標を返すべき"
    
    def test_find_bmu_vectorized_consistency(self):
        """ベクトル化実装と基本実装の一貫性"""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        weights = jax.random.uniform(key1, (5, 5, 10))
        input_vector = jax.random.uniform(key2, (10,))
        
        result_basic = find_bmu_jax(weights, input_vector)
        result_vectorized = find_bmu_vectorized(weights, input_vector)
        
        basic_row, basic_col = int(result_basic[0]), int(result_basic[1])
        vec_row, vec_col = int(result_vectorized[0]), int(result_vectorized[1])
        
        assert (basic_row, basic_col) == (vec_row, vec_col), \
            f"基本実装とベクトル化実装の結果が一致しない: ({basic_row}, {basic_col}) vs ({vec_row}, {vec_col})"
    
    def test_jit_compilation_functionality(self):
        """JIT最適化版の基本動作テスト"""
        key = jax.random.PRNGKey(123)
        key1, key2 = jax.random.split(key)
        weights = jax.random.uniform(key1, (4, 4, 8))
        input_vector = jax.random.uniform(key2, (8,))
        
        # JIT版の実行
        result_jit = find_bmu_jit_optimized(weights, input_vector)
        result_basic = find_bmu_jax(weights, input_vector)
        
        # JAX arrays to Python ints for comparison
        jit_row, jit_col = int(result_jit[0]), int(result_jit[1])
        basic_row, basic_col = int(result_basic[0]), int(result_basic[1])
        
        assert (jit_row, jit_col) == (basic_row, basic_col), \
            f"JIT版と基本版の結果が一致しない: ({jit_row}, {jit_col}) vs ({basic_row}, {basic_col})"
    
    @given(
        map_height=st.integers(min_value=2, max_value=5),
        map_width=st.integers(min_value=2, max_value=5),
        input_dim=st.integers(min_value=2, max_value=10)
    )
    @settings(deadline=None, max_examples=10)
    def test_bmu_property_based(self, map_height, map_width, input_dim):
        """Property-based testing: ランダムな入力に対する基本的性質の確認"""
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        weights = jnp.abs(jax.random.normal(key1, (map_height, map_width, input_dim)))
        input_vector = jnp.abs(jax.random.normal(key2, (input_dim,)))
        
        row, col = find_bmu_jax(weights, input_vector)
        row, col = int(row), int(col)
        
        # 基本的性質：BMUは有効な範囲内にある
        assert 0 <= row < map_height, f"BMU row {row} が範囲外 [0, {map_height})"
        assert 0 <= col < map_width, f"BMU col {col} が範囲外 [0, {map_width})"
        
        # BMUが実際に最小距離を持つことを確認
        bmu_distance = jnp.linalg.norm(weights[row, col] - input_vector)
        
        # 他の全てのユニットとの距離を計算
        all_distances = jnp.linalg.norm(weights - input_vector[None, None, :], axis=2)
        min_distance = jnp.min(all_distances)
        
        # BMUの距離が最小であることを確認（数値誤差を考慮）
        assert jnp.allclose(bmu_distance, min_distance, rtol=1e-6), \
            f"BMU距離 {bmu_distance} が最小距離 {min_distance} と一致しない"
    
    def test_numerical_stability(self):
        """数値安定性テスト"""
        # 極端に大きな値
        weights_large = jnp.ones((3, 3, 2)) * 1e6
        input_large = jnp.array([1e6, 1e6])
        row, col = find_bmu_jax(weights_large, input_large)
        row, col = int(row), int(col)
        assert 0 <= row < 3 and 0 <= col < 3
        
        # 極端に小さな値
        weights_small = jnp.ones((3, 3, 2)) * 1e-6
        input_small = jnp.array([1e-6, 1e-6])
        row, col = find_bmu_jax(weights_small, input_small)
        row, col = int(row), int(col)
        assert 0 <= row < 3 and 0 <= col < 3
    
    def test_different_data_types(self):
        """異なるデータ型でのテスト"""
        # float32
        weights_f32 = jnp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.float32)
        input_f32 = jnp.array([1.0, 2.0], dtype=jnp.float32)
        row, col = find_bmu_jax(weights_f32, input_f32)
        row, col = int(row), int(col)
        assert row == 0 and col == 0


class TestBMUPerformanceBasics:
    """基本的な性能確認（REFACTORフェーズで詳細測定）"""
    
    def test_reasonable_execution_time(self):
        """実行時間が合理的範囲内であることを確認"""
        import time
        
        key = jax.random.PRNGKey(456)
        key1, key2 = jax.random.split(key)
        weights = jax.random.uniform(key1, (20, 20, 64))
        input_vector = jax.random.uniform(key2, (64,))
        
        # 性能測定（簡易版）
        start_time = time.perf_counter()
        for _ in range(100):
            _ = find_bmu_jax(weights, input_vector)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 100
        
        # 1実行あたり100ms以下が目標（寛容な基準）
        assert avg_time < 0.1, f"実行時間 {avg_time:.4f}秒 が目標100msを超過"


if __name__ == "__main__":
    # GREEN Phaseテストの実行
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # 最初の失敗で停止
    ])