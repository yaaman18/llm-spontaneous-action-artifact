"""
基本的な自己組織化マップ（SOM）の具体実装

抽象クラスSelfOrganizingMapの具体的な実装を提供
"""

import numpy as np
from typing import Tuple, List, Optional
import numpy.typing as npt

from domain.entities.self_organizing_map import SelfOrganizingMap
from domain.value_objects.som_topology import SOMTopology
from domain.value_objects.learning_parameters import LearningParameters


class BasicSOM(SelfOrganizingMap):
    """基本的なSOM実装"""
    
    def __init__(
        self,
        map_dimensions: Tuple[int, int],
        input_dimensions: int,
        topology: SOMTopology,
        random_seed: Optional[int] = None
    ):
        """
        BasicSOMの初期化
        
        Args:
            map_dimensions: マップのサイズ (width, height)
            input_dimensions: 入力ベクトルの次元数
            topology: SOMトポロジー設定
            random_seed: ランダムシード（再現性のため）
        """
        super().__init__(map_dimensions, input_dimensions, topology)
        self._random_state = np.random.RandomState(random_seed)
        self._weights = None
        self._initialize_weights()
    
    def initialize_weights(self, initialization_method: str = "random") -> None:
        """
        重みベクトルの初期化
        
        Args:
            initialization_method: 初期化方法 ("random", "pca", "sample")
        """
        self._initialize_weights(initialization_method)
    
    def _initialize_weights(self, initialization_method: str = "random") -> None:
        """
        内部的な重み初期化
        
        Args:
            initialization_method: 初期化方法
        """
        width, height = self._map_dimensions
        
        if initialization_method == "random":
            # ランダム初期化（-1から1の範囲）
            self._weights = self._random_state.uniform(
                -1, 1, 
                (width, height, self._input_dimensions)
            )
        elif initialization_method == "zeros":
            # ゼロ初期化
            self._weights = np.zeros((width, height, self._input_dimensions))
        else:
            # デフォルトはランダム
            self._weights = self._random_state.uniform(
                -1, 1,
                (width, height, self._input_dimensions)
            )
    
    def find_best_matching_unit(self, input_vector: npt.NDArray) -> Tuple[int, int]:
        """
        最適合ユニット（BMU）を見つける
        
        Args:
            input_vector: 入力ベクトル
            
        Returns:
            BMUの座標 (x, y)
        """
        if self._weights is None:
            self._initialize_weights()
        
        # 各ユニットとの距離を計算
        distances = np.linalg.norm(self._weights - input_vector, axis=2)
        
        # 最小距離のユニットを見つける
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        
        return bmu_idx
    
    def find_bmu(self, input_vector: npt.NDArray) -> Tuple[int, int]:
        """
        BMU検索（エイリアス）
        
        Args:
            input_vector: 入力ベクトル
            
        Returns:
            BMUの座標
        """
        return self.find_best_matching_unit(input_vector)
    
    def update_weights(
        self,
        input_vector: npt.NDArray,
        bmu: Tuple[int, int],
        learning_rate: float,
        neighborhood_radius: float
    ) -> None:
        """
        重みベクトルの更新
        
        Args:
            input_vector: 入力ベクトル
            bmu: BMUの座標
            learning_rate: 学習率
            neighborhood_radius: 近傍半径
        """
        width, height = self._map_dimensions
        
        for x in range(width):
            for y in range(height):
                # BMUからの距離を計算
                distance = self._topology.calculate_grid_distance(
                    bmu, (x, y), self._map_dimensions
                )
                
                # 近傍関数の値を計算
                if distance <= neighborhood_radius:
                    influence = self.compute_neighborhood_function(
                        distance, neighborhood_radius
                    )
                    
                    # 重みを更新
                    self._weights[x, y] += (
                        learning_rate * influence * 
                        (input_vector - self._weights[x, y])
                    )
    
    def compute_neighborhood_function(
        self,
        distance: float,
        radius: float,
        learning_iteration: int = 0
    ) -> float:
        """
        近傍関数の計算（ガウシアン）
        
        Args:
            distance: BMUからの距離
            radius: 現在の近傍半径
            learning_iteration: 学習イテレーション（未使用）
            
        Returns:
            近傍関数の値 [0, 1]
        """
        if distance > radius:
            return 0.0
        
        # ガウシアン近傍関数
        return np.exp(-(distance ** 2) / (2 * (radius ** 2)))
    
    def compute_quantization_error(
        self,
        data: List[npt.NDArray]
    ) -> float:
        """
        量子化誤差の計算
        
        Args:
            data: データセット
            
        Returns:
            平均量子化誤差
        """
        if not data:
            return 0.0
        
        total_error = 0.0
        for vector in data:
            bmu = self.find_best_matching_unit(vector)
            bmu_weight = self._weights[bmu[0], bmu[1]]
            error = np.linalg.norm(vector - bmu_weight)
            total_error += error
        
        return total_error / len(data)
    
    def compute_topographic_error(
        self,
        data: List[npt.NDArray]
    ) -> float:
        """
        トポグラフィック誤差の計算
        
        Args:
            data: データセット
            
        Returns:
            トポグラフィック誤差率
        """
        if not data:
            return 0.0
        
        errors = 0
        for vector in data:
            # 最も近い2つのユニットを見つける
            distances = np.linalg.norm(self._weights - vector, axis=2)
            flat_distances = distances.flatten()
            sorted_indices = np.argsort(flat_distances)
            
            # 最も近い2つのユニットの座標
            bmu1_flat = sorted_indices[0]
            bmu2_flat = sorted_indices[1]
            
            bmu1 = np.unravel_index(bmu1_flat, distances.shape)
            bmu2 = np.unravel_index(bmu2_flat, distances.shape)
            
            # 隣接していなければエラー
            grid_distance = self._topology.calculate_grid_distance(
                bmu1, bmu2, self._map_dimensions
            )
            if grid_distance > 1.5:  # 隣接していない
                errors += 1
        
        return errors / len(data)
    
    def train(
        self,
        training_data: List[npt.NDArray],
        learning_params: LearningParameters,
        epochs: int = 1
    ) -> None:
        """
        SOMの訓練
        
        Args:
            training_data: 訓練データ
            learning_params: 学習パラメータ
            epochs: エポック数
        """
        if self._weights is None:
            self._initialize_weights()
        
        total_iterations = learning_params.max_iterations * epochs
        current_iteration = 0
        
        for epoch in range(epochs):
            # データをシャッフル
            shuffled_data = training_data.copy()
            self._random_state.shuffle(shuffled_data)
            
            for data_point in shuffled_data:
                if current_iteration >= total_iterations:
                    break
                
                # 現在の学習率と近傍半径を計算
                current_lr = learning_params.current_learning_rate(current_iteration)
                current_radius = learning_params.current_radius(current_iteration)
                
                # BMUを見つける
                bmu = self.find_best_matching_unit(data_point)
                
                # 重みを更新
                self.update_weights(data_point, bmu, current_lr, current_radius)
                
                current_iteration += 1
    
    def get_weight_vectors(self) -> npt.NDArray:
        """
        重みベクトルを取得
        
        Returns:
            重みベクトル配列
        """
        if self._weights is None:
            self._initialize_weights()
        return self._weights.copy()
    
    def get_unified_distance_matrix(self) -> npt.NDArray:
        """
        U-Matrix（統一距離行列）を計算
        
        Returns:
            U-Matrix
        """
        width, height = self._map_dimensions
        u_matrix = np.zeros((width, height))
        
        for x in range(width):
            for y in range(height):
                # 隣接ユニットとの平均距離を計算
                neighbors = []
                
                # 8近傍をチェック
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            dist = np.linalg.norm(
                                self._weights[x, y] - self._weights[nx, ny]
                            )
                            neighbors.append(dist)
                
                if neighbors:
                    u_matrix[x, y] = np.mean(neighbors)
        
        return u_matrix