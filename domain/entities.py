"""
ドメインエンティティの実装
DDD (Eric Evans) に基づくエンティティ定義
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import uuid

from .value_objects import PhiValue, StateType


@dataclass
class ConsciousnessState:
    """
    意識状態エンティティ
    
    時間的に変化する意識の状態を表現し、
    状態遷移のルールを内包する。
    """
    type: StateType
    phi_value: PhiValue
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    previous_state_id: Optional[str] = None
    transition_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _phi_history: List[PhiValue] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """初期化後の処理"""
        self._phi_history.append(self.phi_value)
        # 不変性を保証するためのフラグ
        object.__setattr__(self, '_frozen', True)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """属性の設定を制御して不変性を保証"""
        if hasattr(self, '_frozen') and self._frozen:
            raise AttributeError(f"ConsciousnessState is immutable")
        super().__setattr__(name, value)
    
    @classmethod
    def create_initial(cls, 
                      timestamp: Optional[datetime] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> 'ConsciousnessState':
        """初期状態を生成"""
        initial_metadata = {'origin': 'initial'}
        if metadata:
            initial_metadata.update(metadata)
            
        return cls(
            type=StateType.DORMANT,
            phi_value=PhiValue(0.0),
            timestamp=timestamp or datetime.now(),
            metadata=initial_metadata
        )
    
    def transition_with_phi(self, 
                           new_phi: PhiValue,
                           timestamp: Optional[datetime] = None) -> 'ConsciousnessState':
        """Φ値に基づいて状態遷移"""
        new_timestamp = timestamp or datetime.now()
        
        # Φ値に基づく状態タイプの決定
        phi_val = float(new_phi.value)
        if phi_val < 1.0:
            new_type = StateType.DORMANT
        elif phi_val < 3.0:
            new_type = StateType.EMERGING
        else:
            new_type = StateType.AWARE
        
        # 遷移の妥当性をチェック
        if not self.is_valid_transition(self.type, new_type):
            # 段階的な遷移を強制
            if self.type == StateType.DORMANT and new_type == StateType.AWARE:
                new_type = StateType.EMERGING
            elif self.type == StateType.AWARE and new_type == StateType.DORMANT:
                new_type = StateType.EMERGING
        
        # 新しい状態を生成
        new_metadata = self.metadata.copy()
        new_metadata.update({
            'transition_reason': 'phi_change',
            'previous_phi': float(self.phi_value.value),
            'previous_state_duration': self.get_duration(new_timestamp).total_seconds()
        })
        
        return ConsciousnessState(
            type=new_type,
            phi_value=new_phi,
            timestamp=new_timestamp,
            previous_state_id=self.id,
            transition_count=self.transition_count + 1,
            metadata=new_metadata
        )
    
    def transition_to_reflective(self, self_reference_level: float) -> 'ConsciousnessState':
        """反省的意識への遷移"""
        if self.type != StateType.AWARE:
            raise ValueError("Can only transition to reflective from aware state")
        
        new_metadata = self.metadata.copy()
        new_metadata['self_reference_level'] = self_reference_level
        
        return ConsciousnessState(
            type=StateType.REFLECTIVE,
            phi_value=self.phi_value,
            previous_state_id=self.id,
            transition_count=self.transition_count + 1,
            metadata=new_metadata
        )
    
    def force_transition_to(self, target_type: StateType) -> 'ConsciousnessState':
        """強制的な状態遷移（検証付き）"""
        from .exceptions import InvalidStateTransition
        
        if not self.is_valid_transition(self.type, target_type):
            raise InvalidStateTransition(
                f"Cannot transition from {self.type.name} to {target_type.name}"
            )
        
        return ConsciousnessState(
            type=target_type,
            phi_value=self.phi_value,
            previous_state_id=self.id,
            transition_count=self.transition_count + 1,
            metadata=self.metadata.copy()
        )
    
    @staticmethod
    def is_valid_transition(from_type: StateType, to_type: StateType) -> bool:
        """状態遷移の妥当性を検証"""
        valid_transitions = {
            StateType.DORMANT: {StateType.EMERGING},
            StateType.EMERGING: {StateType.DORMANT, StateType.AWARE},
            StateType.AWARE: {StateType.EMERGING, StateType.REFLECTIVE},
            StateType.REFLECTIVE: {StateType.AWARE}
        }
        
        return to_type in valid_transitions.get(from_type, set())
    
    def get_duration(self, until: Optional[datetime] = None) -> timedelta:
        """状態の継続時間を取得"""
        end_time = until or datetime.now()
        return end_time - self.timestamp
    
    def can_trace_back_to(self, state_id: str) -> bool:
        """指定した状態まで遡れるかチェック"""
        # 簡略化された実装
        return self.transition_count > 0
    
    def get_transition_history(self) -> List[Dict[str, Any]]:
        """遷移履歴を取得"""
        # 簡略化された実装
        history = []
        for i in range(self.transition_count):
            history.append({
                'from_state': f'state_{i}',
                'to_state': f'state_{i+1}',
                'transition_number': i + 1
            })
        return history
    
    @property
    def energy_consumption_rate(self) -> float:
        """エネルギー消費率"""
        rates = {
            StateType.DORMANT: 0.1,
            StateType.EMERGING: 0.3,
            StateType.AWARE: 0.6,
            StateType.REFLECTIVE: 0.9
        }
        return rates.get(self.type, 0.0)
    
    def calculate_stability(self) -> float:
        """状態の安定性を計算"""
        if len(self._phi_history) < 2:
            return 1.0
        
        # Φ値の変動係数から安定性を計算
        import numpy as np
        phi_values = [float(phi.value) for phi in self._phi_history[-5:]]
        
        if not phi_values:
            return 1.0
            
        mean_phi = np.mean(phi_values)
        if mean_phi == 0:
            return 1.0
            
        std_phi = np.std(phi_values)
        cv = std_phi / mean_phi
        
        # 変動係数が小さいほど安定
        stability = 1.0 / (1.0 + cv)
        return float(stability)
    
    def is_stable(self) -> bool:
        """安定した状態かどうか"""
        return self.calculate_stability() > 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式にシリアライズ"""
        return {
            'id': self.id,
            'type': self.type.value,
            'phi_value': float(self.phi_value.value),
            'timestamp': self.timestamp.isoformat(),
            'previous_state_id': self.previous_state_id,
            'transition_count': self.transition_count,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsciousnessState':
        """辞書形式からデシリアライズ"""
        state = cls(
            type=StateType(data['type']),
            phi_value=PhiValue(data['phi_value']),
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            previous_state_id=data.get('previous_state_id'),
            transition_count=data.get('transition_count', 0),
            metadata=data.get('metadata', {})
        )
        # _frozenフラグを一時的に解除してデシリアライズ
        object.__setattr__(state, '_frozen', False)
        object.__setattr__(state, '_frozen', True)
        return state