"""
ドメインイベントの定義
イベント駆動アーキテクチャのための基盤
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod

from .value_objects import PhiValue


class DomainEvent(ABC):
    """すべてのドメインイベントの基底クラス"""
    def __init__(self, event_id: Optional[str] = None, 
                 timestamp: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.event_id = event_id
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        if self.event_id is None:
            import uuid
            self.event_id = str(uuid.uuid4())


@dataclass
class PhiBoundaryChanged:
    """Φ境界変更イベント"""
    old_threshold: float
    new_threshold: float
    trigger_reason: str
    event_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if self.event_id is None:
            import uuid
            self.event_id = str(uuid.uuid4())
    
    @property
    def threshold_delta(self) -> float:
        """閾値の変化量"""
        return self.new_threshold - self.old_threshold


@dataclass
class ConsciousnessEmerged(DomainEvent):
    """意識創発イベント"""
    phi_value: PhiValue
    subsystem_boundaries: List[Tuple[int, int]]
    emergence_type: str  # 'spontaneous', 'triggered', 'gradual'
    
    @property
    def is_spontaneous(self) -> bool:
        """自発的創発かどうか"""
        return self.emergence_type == 'spontaneous'


@dataclass
class ConsciousnessStateChanged(DomainEvent):
    """意識状態変化イベント"""
    previous_state: str
    new_state: str
    phi_value: PhiValue
    transition_duration: float  # 秒単位
    
    @property
    def is_awakening(self) -> bool:
        """覚醒への遷移かどうか"""
        dormant_states = ['dormant', 'emerging']
        conscious_states = ['conscious', 'highly_conscious']
        return (self.previous_state in dormant_states and 
                self.new_state in conscious_states)


@dataclass
class IntrinsicExistenceDetected(DomainEvent):
    """内在的存在検出イベント"""
    existence_score: float
    detection_method: str
    evidence_factors: Dict[str, float]
    
    @property
    def is_strong_evidence(self) -> bool:
        """強い存在証拠かどうか"""
        return self.existence_score > 0.8


@dataclass
class TemporalCoherenceLost(DomainEvent):
    """時間的一貫性喪失イベント"""
    coherence_score: float
    disruption_cause: str
    affected_duration: float
    
    @property
    def is_critical(self) -> bool:
        """重大な一貫性喪失かどうか"""
        return self.coherence_score < 0.3


@dataclass
class SpontaneousActivityDetected(DomainEvent):
    """自発的活動検出イベント"""
    activity_pattern: str
    intensity: float
    duration: float
    phi_values: List[PhiValue]
    
    @property
    def average_phi(self) -> float:
        """平均Φ値"""
        if not self.phi_values:
            return 0.0
        return sum(float(phi.value) for phi in self.phi_values) / len(self.phi_values)


@dataclass
class SubsystemBoundaryReorganized(DomainEvent):
    """サブシステム境界再編成イベント"""
    old_boundaries: List[Tuple[int, int]]
    new_boundaries: List[Tuple[int, int]]
    reorganization_trigger: str
    stability_score: float
    
    @property
    def boundary_count_changed(self) -> bool:
        """境界数が変化したかどうか"""
        return len(self.old_boundaries) != len(self.new_boundaries)


@dataclass
class InformationGenerationObserved(DomainEvent):
    """情報生成観測イベント"""
    generation_rate: float
    complexity_increase: float
    novel_patterns: List[str]
    source_subsystem: Tuple[int, int]
    
    @property
    def is_significant_generation(self) -> bool:
        """有意な情報生成かどうか"""
        return self.generation_rate > 0.5 and self.complexity_increase > 0.3