"""
流暢なAPI（Fluent API）の実装
Martin Fowlerのリファクタリング原則に基づく設計
"""
from typing import Optional, List, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .consciousness_core import DynamicPhiBoundaryDetector, ConsciousnessMetrics
from .strategies import PhiCalculationStrategy, StandardPhiStrategy, FastPhiStrategy, AdaptivePhiStrategy
from .observers import ConsciousnessObserver, ConsciousnessSubject
from .caching import PhiCalculationCache
from .parallel import ParallelExecutionConfig, create_optimized_detector
from .value_objects import PhiValue
from .entities import ConsciousnessState


class ConsciousnessSystemBuilder:
    """意識システムの流暢なビルダー"""
    
    def __init__(self):
        self._min_subsystem_size = 3
        self._max_subsystem_size = None
        self._phi_threshold = 3.0
        self._phi_strategy = None
        self._cache_config = {'enabled': True, 'max_size': 1000}
        self._parallel_config = {'enabled': False, 'use_processes': False}
        self._observers = []
    
    def with_subsystem_size(self, min_size: int, max_size: Optional[int] = None) -> 'ConsciousnessSystemBuilder':
        """サブシステムサイズを設定"""
        self._min_subsystem_size = min_size
        self._max_subsystem_size = max_size
        return self
    
    def with_phi_threshold(self, threshold: float) -> 'ConsciousnessSystemBuilder':
        """Φ値閾値を設定"""
        self._phi_threshold = threshold
        return self
    
    def with_standard_strategy(self) -> 'ConsciousnessSystemBuilder':
        """標準計算ストラテジーを使用"""
        self._phi_strategy = StandardPhiStrategy()
        return self
    
    def with_fast_strategy(self, approximation_level: float = 0.9) -> 'ConsciousnessSystemBuilder':
        """高速計算ストラテジーを使用"""
        self._phi_strategy = FastPhiStrategy(approximation_level)
        return self
    
    def with_adaptive_strategy(self) -> 'ConsciousnessSystemBuilder':
        """適応的計算ストラテジーを使用"""
        self._phi_strategy = AdaptivePhiStrategy()
        return self
    
    def with_caching(self, enabled: bool = True, max_size: int = 1000) -> 'ConsciousnessSystemBuilder':
        """キャッシングを設定"""
        self._cache_config = {'enabled': enabled, 'max_size': max_size}
        return self
    
    def with_parallel_processing(self, enabled: bool = True, use_processes: bool = False) -> 'ConsciousnessSystemBuilder':
        """並列処理を設定"""
        self._parallel_config = {'enabled': enabled, 'use_processes': use_processes}
        return self
    
    def with_observer(self, observer: ConsciousnessObserver) -> 'ConsciousnessSystemBuilder':
        """オブザーバーを追加"""
        self._observers.append(observer)
        return self
    
    def build(self) -> 'FluentConsciousnessSystem':
        """意識システムを構築"""
        # デフォルトストラテジーの設定
        if self._phi_strategy is None:
            self._phi_strategy = StandardPhiStrategy()
        
        # 検出器の作成
        detector = DynamicPhiBoundaryDetector(
            min_subsystem_size=self._min_subsystem_size,
            max_subsystem_size=self._max_subsystem_size,
            phi_threshold=self._phi_threshold,
            phi_strategy=self._phi_strategy
        )
        
        # 並列処理の適用
        if self._parallel_config['enabled']:
            detector = create_optimized_detector(
                detector,
                use_parallel=True,
                use_processes=self._parallel_config['use_processes']
            )
        
        # サブジェクトの作成とオブザーバーの登録
        subject = ConsciousnessSubject()
        for observer in self._observers:
            subject.attach(observer)
        
        return FluentConsciousnessSystem(detector, subject)


class FluentConsciousnessSystem:
    """流暢なインターフェースを持つ意識システム"""
    
    def __init__(self, detector: DynamicPhiBoundaryDetector, subject: ConsciousnessSubject):
        self._detector = detector
        self._subject = subject
        self._current_state = None
        self._connectivity = None
        self._state_vector = None
    
    def analyze(self, connectivity: np.ndarray, state: np.ndarray) -> 'FluentConsciousnessSystem':
        """システムを分析"""
        self._connectivity = connectivity
        self._state_vector = state
        return self
    
    def detect_boundaries(self) -> 'FluentConsciousnessSystem':
        """境界を検出"""
        if self._connectivity is None or self._state_vector is None:
            raise ValueError("Must call analyze() first")
        
        boundaries = self._detector.detect_boundaries(self._connectivity, self._state_vector)
        
        # メトリクスの作成
        phi_value = self._calculate_system_phi()
        metrics = ConsciousnessMetrics(
            phi_value=phi_value,
            timestamp=datetime.now(),
            subsystem_boundaries=boundaries,
            intrinsic_existence_score=0.0,  # 後で計算
            temporal_coherence=1.0,  # 後で計算
            metadata={'boundaries_count': len(boundaries)}
        )
        
        # 状態の更新と通知
        if self._current_state:
            self._subject.notify_state_transition(self._current_state, metrics)
        else:
            self._subject.notify(metrics, 'initial_analysis')
        
        self._current_state = metrics
        return self
    
    def _calculate_system_phi(self) -> PhiValue:
        """システム全体のΦ値を計算"""
        return self._detector.phi_strategy.calculate(self._connectivity, self._state_vector)
    
    def with_threshold(self, threshold: float) -> 'FluentConsciousnessSystem':
        """閾値を動的に変更"""
        old_threshold = self._detector.phi_threshold
        self._detector.phi_threshold = threshold
        
        if self._current_state:
            self._subject.notify(
                self._current_state,
                'threshold_changed',
                {'old_threshold': old_threshold, 'new_threshold': threshold}
            )
        
        return self
    
    def get_metrics(self) -> Optional[ConsciousnessMetrics]:
        """現在のメトリクスを取得"""
        return self._current_state
    
    def get_cache_stats(self) -> dict:
        """キャッシュ統計を取得"""
        return self._detector._cache.get_stats()
    
    def clear_cache(self) -> 'FluentConsciousnessSystem':
        """キャッシュをクリア"""
        self._detector._cache.clear()
        return self


# 使用例を示すヘルパー関数
def create_standard_system() -> FluentConsciousnessSystem:
    """標準的な意識システムを作成"""
    return (ConsciousnessSystemBuilder()
            .with_standard_strategy()
            .with_caching(enabled=True)
            .with_subsystem_size(min_size=3, max_size=10)
            .build())


def create_fast_system() -> FluentConsciousnessSystem:
    """高速な意識システムを作成"""
    return (ConsciousnessSystemBuilder()
            .with_fast_strategy(approximation_level=0.8)
            .with_parallel_processing(enabled=True)
            .with_caching(enabled=True, max_size=2000)
            .build())


def create_adaptive_system() -> FluentConsciousnessSystem:
    """適応的な意識システムを作成"""
    return (ConsciousnessSystemBuilder()
            .with_adaptive_strategy()
            .with_caching(enabled=True)
            .with_subsystem_size(min_size=2, max_size=15)
            .with_phi_threshold(2.5)
            .build())