"""
ドメインサービスの実装
エリック・エヴァンスのDDDに基づくサービス層
"""
from typing import List, Optional, Dict, Any, Callable
from collections import deque
from dataclasses import dataclass
import numpy as np
from datetime import datetime

from .value_objects import PhiValue
from .entities import ConsciousnessState


@dataclass
class ConsciousnessClassification:
    """意識状態の分類結果"""
    base_state: str
    threshold_used: float
    confidence: float
    phi_value: PhiValue
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PhiBoundaryChanged:
    """Φ境界変更イベント"""
    old_threshold: float
    new_threshold: float
    trigger_reason: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DynamicPhiBoundaryDetector:
    """
    動的Φ境界検出サービス
    
    観測されたΦ値に基づいて、意識判定の閾値を
    動的に調整する。
    """
    
    def __init__(self,
                 initial_threshold: float = 3.0,
                 window_size: int = 100,
                 adaptation_rate: float = 0.1,
                 minimum_threshold: float = 1.0,
                 use_statistical_method: bool = False,
                 rapid_adaptation_enabled: bool = False,
                 event_bus: Optional[Any] = None):
        """
        Args:
            initial_threshold: 初期閾値
            window_size: 観測ウィンドウサイズ
            adaptation_rate: 適応率 (0.0-1.0)
            minimum_threshold: 最小許容閾値
            use_statistical_method: 統計的手法を使用するか
            rapid_adaptation_enabled: 急速適応モードを有効にするか
            event_bus: イベントバス（オプション）
        """
        self.current_threshold = initial_threshold
        self.observation_window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.minimum_threshold = minimum_threshold
        self.use_statistical_method = use_statistical_method
        self.rapid_adaptation_enabled = rapid_adaptation_enabled
        self.event_bus = event_bus
        
        # 観測履歴
        self.observations: deque = deque(maxlen=window_size)
        self.adaptation_history: List[Dict[str, Any]] = []
        self.rapid_adaptation_triggered = False
        
        # 統計情報のキャッシュ
        self._statistics_cache: Optional[Dict[str, float]] = None
        self._cache_timestamp: Optional[datetime] = None
        
    def observe(self, phi_value: PhiValue) -> None:
        """
        Φ値を観測し、必要に応じて閾値を調整
        
        Args:
            phi_value: 観測されたΦ値
        """
        old_threshold = self.current_threshold
        self.observations.append(phi_value)
        
        # キャッシュをクリア
        self._statistics_cache = None
        
        # 十分な観測データがある場合のみ適応
        # 小さいウィンドウサイズの場合は最小3つの観測で適応開始
        min_observations = min(3, self.observation_window_size // 2)
        if len(self.observations) >= min_observations:
            self._adapt_threshold()
            
            # 閾値が変更された場合、イベントを発火
            if abs(self.current_threshold - old_threshold) > 0.01:
                self._emit_boundary_changed_event(old_threshold)
    
    def _adapt_threshold(self) -> None:
        """閾値を現在の観測に基づいて適応"""
        phi_values = [float(obs.value) for obs in self.observations]
        
        if self.use_statistical_method:
            # 統計的手法による閾値計算
            new_threshold = self._calculate_statistical_threshold(phi_values)
        else:
            # 移動平均ベースの適応
            new_threshold = self._calculate_adaptive_threshold(phi_values)
        
        # 急速適応の検出
        adaptation_rate = self.adaptation_rate
        if self.rapid_adaptation_enabled:
            if self._detect_rapid_change(phi_values):
                self.rapid_adaptation_triggered = True
                adaptation_rate = min(0.5, self.adaptation_rate * 3)
        
        # 閾値の更新（適応率を考慮）
        self.current_threshold = (
            (1 - adaptation_rate) * self.current_threshold +
            adaptation_rate * new_threshold
        )
        
        # 最小閾値の強制
        self.current_threshold = max(self.minimum_threshold, self.current_threshold)
        
        # 適応履歴の記録
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'new_threshold': self.current_threshold,
            'observation_count': len(self.observations),
            'rapid_adaptation': self.rapid_adaptation_triggered
        })
        
        # 急速適応フラグをリセット
        if self.rapid_adaptation_triggered:
            self.adaptation_rate = 0.1  # デフォルトに戻す
            self.rapid_adaptation_triggered = False
    
    def _calculate_statistical_threshold(self, values: List[float]) -> float:
        """統計的手法による閾値計算"""
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        
        # 75パーセンタイルを基準に使用
        percentile_75 = np.percentile(arr, 75)
        
        # 外れ値の影響を減らすため、中央値も考慮
        median = np.median(arr)
        
        # 複合的な閾値計算
        statistical_threshold = (
            0.3 * mean +
            0.3 * median +
            0.4 * (percentile_75 - 0.5 * std)
        )
        
        return float(statistical_threshold)
    
    def _calculate_adaptive_threshold(self, values: List[float]) -> float:
        """移動平均ベースの適応的閾値計算"""
        recent_values = values[-20:]  # 最近の20個の値を重視
        
        # 加重移動平均
        weights = np.exp(np.linspace(-1, 0, len(recent_values)))
        weights /= weights.sum()
        
        weighted_mean = np.average(recent_values, weights=weights)
        
        # 変動性を考慮
        volatility = np.std(recent_values)
        
        # 高い変動性の場合は保守的に
        if volatility > 1.0:
            return float(weighted_mean - 0.5 * volatility)
        else:
            return float(weighted_mean)
    
    def _detect_rapid_change(self, values: List[float]) -> bool:
        """急激な変化を検出"""
        if len(values) < 2:
            return False
            
        # 最新値と直前の平均値の差
        recent_value = values[-1]
        previous_mean = np.mean(values[:-1])
        
        change_ratio = abs(recent_value - previous_mean) / (previous_mean + 1e-6)
        
        return change_ratio > 0.5  # 50%以上の変化
    
    def _emit_boundary_changed_event(self, old_threshold: float) -> None:
        """境界変更イベントを発火"""
        if self.event_bus is None:
            return
            
        event = PhiBoundaryChanged(
            old_threshold=old_threshold,
            new_threshold=self.current_threshold,
            trigger_reason='adaptive_adjustment'
        )
        
        self.event_bus.publish(event)
    
    def classify_consciousness(self, phi_value: PhiValue) -> ConsciousnessClassification:
        """
        現在の動的閾値に基づいて意識状態を分類
        
        Args:
            phi_value: 分類対象のΦ値
            
        Returns:
            分類結果
        """
        value = float(phi_value.value)
        
        # 基本的な状態分類
        if value < 1.0:
            base_state = 'dormant'
        elif value < 3.0:
            base_state = 'emerging'
        elif value < 6.0:
            base_state = 'conscious'
        else:
            base_state = 'highly_conscious'
        
        # 動的閾値との相対的な位置に基づく信頼度
        distance_from_threshold = abs(value - self.current_threshold)
        confidence = 1.0 / (1.0 + np.exp(-distance_from_threshold))
        
        return ConsciousnessClassification(
            base_state=base_state,
            threshold_used=self.current_threshold,
            confidence=confidence,
            phi_value=phi_value
        )
    
    def get_statistics(self) -> Dict[str, float]:
        """
        現在の観測統計を取得
        
        Returns:
            統計情報の辞書
        """
        # キャッシュチェック
        if self._statistics_cache is not None:
            cache_age = (datetime.now() - self._cache_timestamp).seconds
            if cache_age < 60:  # 1分間キャッシュ
                return self._statistics_cache
        
        if not self.observations:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'percentile_25': 0.0,
                'percentile_50': 0.0,
                'percentile_75': 0.0
            }
        
        values = [float(obs.value) for obs in self.observations]
        arr = np.array(values)
        
        statistics = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'percentile_25': float(np.percentile(arr, 25)),
            'percentile_50': float(np.percentile(arr, 50)),
            'percentile_75': float(np.percentile(arr, 75))
        }
        
        # キャッシュ更新
        self._statistics_cache = statistics
        self._cache_timestamp = datetime.now()
        
        return statistics