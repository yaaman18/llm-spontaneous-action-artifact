"""
意識計算のためのストラテジーパターン実装
Martin Fowlerのリファクタリング原則に基づく設計
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
from dataclasses import dataclass

from .value_objects import PhiValue


class PhiCalculationStrategy(ABC):
    """Φ値計算ストラテジーの抽象基底クラス"""
    
    @abstractmethod
    def calculate(self, connectivity: np.ndarray, state: np.ndarray) -> PhiValue:
        """Φ値を計算する抽象メソッド"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """ストラテジー名を返す"""
        pass


class StandardPhiStrategy(PhiCalculationStrategy):
    """標準的なΦ値計算ストラテジー"""
    
    def calculate(self, connectivity: np.ndarray, state: np.ndarray) -> PhiValue:
        """IITに基づく標準的なΦ値計算"""
        if connectivity.size == 0 or state.size == 0:
            return PhiValue(0.0)
        
        integration = self._calculate_integration(connectivity)
        differentiation = self._calculate_differentiation(state)
        causal_efficacy = self._calculate_causal_efficacy(connectivity, state)
        
        phi_value = integration * differentiation * causal_efficacy
        return PhiValue(float(phi_value))
    
    def _calculate_integration(self, connectivity: np.ndarray) -> float:
        """統合度の計算"""
        density = np.sum(connectivity) / (connectivity.size - len(connectivity) + 1e-10)
        
        eigenvalues = np.linalg.eigvals(connectivity)
        if len(eigenvalues) > 1:
            sorted_eigenvalues = np.sort(np.real(eigenvalues))
            fiedler_value = sorted_eigenvalues[1]
            return density * (1 + abs(fiedler_value))
        
        return density
    
    def _calculate_differentiation(self, state: np.ndarray) -> float:
        """差異化度の計算"""
        if len(state) < 2:
            return 0.0
        
        unique_states = len(np.unique(state))
        differentiation = unique_states / len(state)
        
        if unique_states > 1:
            _, counts = np.unique(state, return_counts=True)
            probabilities = counts / len(state)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            differentiation *= (1 + entropy)
        
        return float(differentiation)
    
    def _calculate_causal_efficacy(self, connectivity: np.ndarray, state: np.ndarray) -> float:
        """因果的有効性の計算"""
        if connectivity.size == 0:
            return 0.0
        
        causal_power = np.sum(connectivity * np.outer(state, state))
        normalized_causal = causal_power / (connectivity.size + 1)
        
        return float(1 + normalized_causal)
    
    def get_name(self) -> str:
        return "standard"


class FastPhiStrategy(PhiCalculationStrategy):
    """高速化されたΦ値計算ストラテジー"""
    
    def __init__(self, approximation_level: float = 0.9):
        """
        Args:
            approximation_level: 近似レベル (0.0-1.0)
        """
        self.approximation_level = approximation_level
    
    def calculate(self, connectivity: np.ndarray, state: np.ndarray) -> PhiValue:
        """近似を使用した高速Φ値計算"""
        if connectivity.size == 0 or state.size == 0:
            return PhiValue(0.0)
        
        # 簡略化された計算
        avg_connectivity = np.mean(connectivity)
        state_variance = np.var(state)
        
        # 近似式によるΦ値計算
        phi_approximation = avg_connectivity * (1 + state_variance) * self.approximation_level
        
        return PhiValue(float(phi_approximation))
    
    def get_name(self) -> str:
        return f"fast_approximation_{self.approximation_level}"


class AdaptivePhiStrategy(PhiCalculationStrategy):
    """適応的Φ値計算ストラテジー"""
    
    def __init__(self, base_strategy: Optional[PhiCalculationStrategy] = None):
        """
        Args:
            base_strategy: 基底となるストラテジー
        """
        self.base_strategy = base_strategy or StandardPhiStrategy()
        self.adaptation_history = []
    
    def calculate(self, connectivity: np.ndarray, state: np.ndarray) -> PhiValue:
        """システムの特性に適応したΦ値計算"""
        base_phi = self.base_strategy.calculate(connectivity, state)
        
        # システムサイズに基づく適応係数
        system_size = len(connectivity)
        size_factor = 1.0 + np.log(system_size + 1) / 10.0
        
        # 接続密度に基づく適応係数
        density = np.sum(connectivity) / (connectivity.size + 1e-10)
        density_factor = 1.0 + density
        
        # 適応的Φ値
        adapted_phi = float(base_phi.value) * size_factor * density_factor
        
        # 適応履歴の記録
        self.adaptation_history.append({
            'base_phi': float(base_phi.value),
            'adapted_phi': adapted_phi,
            'size_factor': size_factor,
            'density_factor': density_factor
        })
        
        return PhiValue(adapted_phi)
    
    def get_name(self) -> str:
        return f"adaptive_{self.base_strategy.get_name()}"


@dataclass
class PhiCalculationContext:
    """Φ値計算のコンテキスト"""
    strategy: PhiCalculationStrategy
    cache_enabled: bool = True
    parallel_enabled: bool = False
    
    def calculate_phi(self, connectivity: np.ndarray, state: np.ndarray) -> PhiValue:
        """コンテキストに応じたΦ値計算"""
        # TODO: キャッシングと並列化の実装
        return self.strategy.calculate(connectivity, state)
    
    def switch_strategy(self, new_strategy: PhiCalculationStrategy) -> None:
        """ストラテジーの切り替え"""
        self.strategy = new_strategy