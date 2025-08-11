"""
SOM BMU (Best Matching Unit) 計算 - TDD Green Phase 最小実装

TDD原則：
- Red: 失敗テストを確認済み
- Green: テストを通す最小限の実装
- Refactor: 後で最適化予定

実装方針：
- 基本的な距離計算とBMU検索
- JAX互換性確保
- 性能最適化は後のRefactorフェーズで実施
"""

import jax
import jax.numpy as jnp
from typing import Tuple


class SOMBMUCalculator:
    """SOM BMU計算クラス - 最小実装"""
    
    def __init__(self):
        self.name = "SOM BMU Calculator"
    
    def find_bmu_with_distance_matrix(self, weights, distance_matrix):
        """
        距離行列を用いたBMU検索 - 未実装
        
        TDD RED Phase: このメソッドはまだ実装されていません
        """
        raise NotImplementedError("find_bmu_with_distance_matrix method not yet implemented")
    
    def find_bmu_with_custom_metric(self, weights, input_vector, metric):
        """
        カスタム距離メトリクスを用いたBMU検索 - 未実装
        
        TDD RED Phase: このメソッドはまだ実装されていません
        """
        raise NotImplementedError("find_bmu_with_custom_metric method not yet implemented")
    
    def property_based_bmu_search(self, weights, input_vector):
        """
        プロパティベースBMU検索 - 未実装
        
        TDD RED Phase: このメソッドはまだ実装されていません
        """
        raise NotImplementedError("property_based_bmu_search method not yet implemented")
    
    def jit_optimized_batch_bmu_search(self, weights, input_vectors):
        """
        JIT最適化バッチBMU検索 - 未実装
        
        TDD RED Phase: このメソッドはまだ実装されていません
        """
        raise NotImplementedError("jit_optimized_batch_bmu_search method not yet implemented")
    

def find_bmu_jax(weights: jnp.ndarray, input_vector: jnp.ndarray) -> Tuple[int, int]:
    """
    JAX実装のBMU検索 - 基本版
    
    Args:
        weights: SOM重みマトリクス (height, width, input_dim)
        input_vector: 入力ベクトル (input_dim,)
    
    Returns:
        BMUの座標 (row, col)
    """
    # ユークリッド距離計算
    distances = jnp.linalg.norm(weights - input_vector, axis=2)
    
    # 最小距離のインデックス取得
    bmu_idx = jnp.argmin(distances)
    
    # 2D座標に変換
    row, col = jnp.divmod(bmu_idx, weights.shape[1])
    
    return row, col


def find_bmu_vectorized(weights: jnp.ndarray, input_vector: jnp.ndarray) -> Tuple[int, int]:
    """
    ベクトル化BMU検索 - property-based testing用
    
    Args:
        weights: SOM重みマトリクス (height, width, input_dim)
        input_vector: 入力ベクトル (input_dim,)
    
    Returns:
        BMUの座標 (row, col)
    """
    # find_bmu_jaxと同じ実装（最小実装段階では）
    return find_bmu_jax(weights, input_vector)


# JIT最適化版（基本実装）
find_bmu_jit_optimized = jax.jit(find_bmu_jax)