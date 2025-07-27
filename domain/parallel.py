"""
並列処理によるパフォーマンス最適化
Martin Fowlerのリファクタリング原則に基づく設計
"""
from typing import List, Tuple, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
import multiprocessing
import logging

from .value_objects import PhiValue


@dataclass
class ParallelExecutionConfig:
    """並列実行設定"""
    max_workers: Optional[int] = None
    use_processes: bool = False
    chunk_size: int = 1
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = multiprocessing.cpu_count()


class ParallelPhiCalculator:
    """並列化されたΦ値計算器"""
    
    def __init__(self, 
                 base_calculator: Callable,
                 config: Optional[ParallelExecutionConfig] = None):
        """
        Args:
            base_calculator: 基本となる計算関数
            config: 並列実行設定
        """
        self.base_calculator = base_calculator
        self.config = config or ParallelExecutionConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_batch(self, 
                       subsystems: List[Tuple[np.ndarray, np.ndarray]]) -> List[PhiValue]:
        """
        複数のサブシステムのΦ値を並列計算
        
        Args:
            subsystems: (connectivity, state) のタプルのリスト
            
        Returns:
            Φ値のリスト
        """
        if len(subsystems) <= 1:
            # 並列化のオーバーヘッドを避ける
            return [self.base_calculator(conn, state) for conn, state in subsystems]
        
        # 適切なエグゼキューターを選択
        executor_class = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor
        
        results = []
        with executor_class(max_workers=self.config.max_workers) as executor:
            # タスクを投入
            future_to_index = {}
            for i, (connectivity, state) in enumerate(subsystems):
                future = executor.submit(self.base_calculator, connectivity, state)
                future_to_index[future] = i
            
            # 結果を収集（順序を保持）
            results = [None] * len(subsystems)
            for future in as_completed(future_to_index, timeout=self.config.timeout):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Parallel calculation failed for index {index}: {e}")
                    results[index] = PhiValue(0.0)
        
        return results


class OptimizedBoundaryDetector:
    """最適化された境界検出器"""
    
    def __init__(self, 
                 base_detector: Any,
                 parallel_config: Optional[ParallelExecutionConfig] = None):
        """
        Args:
            base_detector: 基本となる境界検出器
            parallel_config: 並列実行設定
        """
        self.base_detector = base_detector
        self.parallel_config = parallel_config or ParallelExecutionConfig()
        self.parallel_calculator = ParallelPhiCalculator(
            base_calculator=base_detector.phi_strategy.calculate,
            config=self.parallel_config
        )
    
    def detect_boundaries_parallel(self,
                                 connectivity_matrix: np.ndarray,
                                 state_vector: np.ndarray) -> List[Tuple[int, int]]:
        """
        並列化された境界検出
        
        Args:
            connectivity_matrix: ノード間の接続行列
            state_vector: 現在の状態ベクトル
            
        Returns:
            検出された境界のリスト
        """
        n_nodes = len(connectivity_matrix)
        max_size = self.base_detector.max_subsystem_size or n_nodes
        
        # 全てのサブシステム候補を準備
        subsystem_candidates = []
        subsystem_info = []  # (start, end) の情報を保持
        
        for size in range(self.base_detector.min_subsystem_size, min(max_size + 1, n_nodes + 1)):
            for start in range(n_nodes - size + 1):
                end = start + size
                sub_connectivity = connectivity_matrix[start:end, start:end]
                sub_state = state_vector[start:end]
                
                subsystem_candidates.append((sub_connectivity, sub_state))
                subsystem_info.append((start, end))
        
        # 並列でΦ値を計算
        phi_values = self.parallel_calculator.calculate_batch(subsystem_candidates)
        
        # 結果を整理
        candidate_boundaries = []
        for (start, end), phi in zip(subsystem_info, phi_values):
            if phi.indicates_consciousness(self.base_detector.phi_threshold):
                candidate_boundaries.append((start, end, phi))
        
        # 最適な境界を選択
        return self.base_detector._select_optimal_boundaries(candidate_boundaries)


def create_optimized_detector(base_detector: Any, 
                            use_parallel: bool = True,
                            use_processes: bool = False) -> Any:
    """
    最適化された検出器を作成
    
    Args:
        base_detector: 基本となる検出器
        use_parallel: 並列化を使用するか
        use_processes: プロセスベースの並列化を使用するか
        
    Returns:
        最適化された検出器
    """
    if not use_parallel:
        return base_detector
    
    config = ParallelExecutionConfig(use_processes=use_processes)
    optimized = OptimizedBoundaryDetector(base_detector, config)
    
    # 元の検出器のメソッドを置き換え
    base_detector.detect_boundaries = optimized.detect_boundaries_parallel
    
    return base_detector