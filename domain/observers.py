"""
意識状態監視のためのオブザーバーパターン実装
Martin Fowlerのリファクタリング原則に基づく設計
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .entities import ConsciousnessState
from .value_objects import PhiValue, StateType


class ConsciousnessObserver(ABC):
    """意識状態観察者の抽象基底クラス"""
    
    @abstractmethod
    def update(self, state: ConsciousnessState, event_type: str, metadata: Dict[str, Any]) -> None:
        """状態変化の通知を受け取る"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """オブザーバー名を返す"""
        pass


class LoggingObserver(ConsciousnessObserver):
    """ログ記録オブザーバー"""
    
    def __init__(self, logger_name: str = "consciousness"):
        self.logger = logging.getLogger(logger_name)
    
    def update(self, state: ConsciousnessState, event_type: str, metadata: Dict[str, Any]) -> None:
        """状態変化をログに記録"""
        self.logger.info(
            f"Consciousness event: {event_type} | "
            f"State: {state.type.value} | "
            f"Phi: {state.phi_value.value:.4f} | "
            f"Metadata: {metadata}"
        )
    
    def get_name(self) -> str:
        return "logging_observer"


class MetricsCollectorObserver(ConsciousnessObserver):
    """メトリクス収集オブザーバー"""
    
    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
        self.state_transitions: List[Tuple[StateType, StateType, datetime]] = []
    
    def update(self, state: ConsciousnessState, event_type: str, metadata: Dict[str, Any]) -> None:
        """メトリクスを収集"""
        metric = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'state_type': state.type.value,
            'phi_value': float(state.phi_value.value),
            'metadata': metadata.copy()
        }
        self.metrics.append(metric)
        
        # 状態遷移の追跡
        if event_type == 'state_transition' and 'previous_state' in metadata:
            previous_state = metadata['previous_state']
            self.state_transitions.append(
                (previous_state.type, state.type, datetime.now())
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """収集したメトリクスの統計情報を返す"""
        if not self.metrics:
            return {}
        
        phi_values = [m['phi_value'] for m in self.metrics]
        
        return {
            'total_events': len(self.metrics),
            'avg_phi': sum(phi_values) / len(phi_values),
            'max_phi': max(phi_values),
            'min_phi': min(phi_values),
            'state_transitions': len(self.state_transitions)
        }
    
    def get_name(self) -> str:
        return "metrics_collector"


class ThresholdAlertObserver(ConsciousnessObserver):
    """閾値アラートオブザーバー"""
    
    def __init__(self, phi_threshold: float = 3.0, callback: Optional[Callable] = None):
        """
        Args:
            phi_threshold: アラートを発生させるΦ値の閾値
            callback: 閾値超過時に呼び出されるコールバック関数
        """
        self.phi_threshold = phi_threshold
        self.callback = callback
        self.alerts: List[Dict[str, Any]] = []
    
    def update(self, state: ConsciousnessState, event_type: str, metadata: Dict[str, Any]) -> None:
        """閾値超過を監視"""
        phi_value = float(state.phi_value.value)
        
        if phi_value >= self.phi_threshold:
            alert = {
                'timestamp': datetime.now(),
                'phi_value': phi_value,
                'threshold': self.phi_threshold,
                'state_type': state.type.value,
                'event_type': event_type
            }
            self.alerts.append(alert)
            
            if self.callback:
                self.callback(state, alert)
    
    def get_name(self) -> str:
        return f"threshold_alert_{self.phi_threshold}"


@dataclass
class ConsciousnessSubject:
    """意識状態の主体（観察対象）"""
    observers: List[ConsciousnessObserver] = field(default_factory=list)
    _state_history: List[ConsciousnessState] = field(default_factory=list, init=False)
    
    def attach(self, observer: ConsciousnessObserver) -> None:
        """オブザーバーを追加"""
        if observer not in self.observers:
            self.observers.append(observer)
    
    def detach(self, observer: ConsciousnessObserver) -> None:
        """オブザーバーを削除"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify(self, state: ConsciousnessState, event_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """全てのオブザーバーに通知"""
        metadata = metadata or {}
        self._state_history.append(state)
        
        for observer in self.observers:
            try:
                observer.update(state, event_type, metadata)
            except Exception as e:
                logging.error(f"Observer {observer.get_name()} failed: {e}")
    
    def notify_state_transition(self, old_state: ConsciousnessState, new_state: ConsciousnessState) -> None:
        """状態遷移を通知"""
        metadata = {
            'previous_state': old_state,
            'transition_time': datetime.now(),
            'phi_change': float(new_state.phi_value.value) - float(old_state.phi_value.value)
        }
        self.notify(new_state, 'state_transition', metadata)
    
    def notify_phi_update(self, state: ConsciousnessState, old_phi: PhiValue) -> None:
        """Φ値更新を通知"""
        metadata = {
            'old_phi': float(old_phi.value),
            'new_phi': float(state.phi_value.value),
            'change_percentage': ((float(state.phi_value.value) - float(old_phi.value)) / float(old_phi.value) * 100) if old_phi.value != 0 else 0
        }
        self.notify(state, 'phi_update', metadata)
    
    def get_observer_count(self) -> int:
        """登録されているオブザーバー数を返す"""
        return len(self.observers)
    
    def get_state_history(self) -> List[ConsciousnessState]:
        """状態履歴を返す"""
        return self._state_history.copy()