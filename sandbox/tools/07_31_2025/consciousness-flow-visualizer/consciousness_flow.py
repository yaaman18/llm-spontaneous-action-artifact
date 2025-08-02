"""
Consciousness Flow Visualizer - Core Module
意識の流れを記録・分析・可視化するためのコアモジュール
"""

import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading
import asyncio


@dataclass
class ConsciousnessState:
    """意識の瞬間的状態を表現するデータクラス"""
    timestamp: float
    attention: Dict[str, float]  # 注意の焦点と強度
    integration: float  # 情報統合度 (0-1)
    phenomenal_properties: Dict[str, float]  # 現象的性質
    cognitive_load: float  # 認知負荷 (0-1)
    meta_awareness: float  # メタ認知レベル (0-1)
    flow_vector: Tuple[float, float, float]  # 3D空間での流れの方向
    
    def to_dict(self):
        return asdict(self)


class ConsciousnessStream:
    """意識の流れを記録・管理するクラス"""
    
    def __init__(self, max_history: int = 1000):
        self.states: deque = deque(maxlen=max_history)
        self.current_state: Optional[ConsciousnessState] = None
        self.recording = False
        self._lock = threading.Lock()
        self.observers = []
        
    def add_state(self, state_data: Dict):
        """新しい意識状態を追加"""
        with self._lock:
            state = ConsciousnessState(
                timestamp=state_data.get('timestamp', time.time()),
                attention=state_data.get('attention', {}),
                integration=state_data.get('integration', 0.5),
                phenomenal_properties=state_data.get('phenomenal_properties', {}),
                cognitive_load=state_data.get('cognitive_load', 0.5),
                meta_awareness=state_data.get('meta_awareness', 0.5),
                flow_vector=state_data.get('flow_vector', (0, 0, 0))
            )
            self.states.append(state)
            self.current_state = state
            self._notify_observers(state)
            
    def _notify_observers(self, state: ConsciousnessState):
        """観察者に状態変化を通知"""
        for observer in self.observers:
            observer(state)
            
    def get_flow_dynamics(self, window_size: int = 10) -> Dict:
        """意識の流れのダイナミクスを分析"""
        if len(self.states) < window_size:
            return {}
            
        recent_states = list(self.states)[-window_size:]
        
        # 統合度の変化率
        integration_values = [s.integration for s in recent_states]
        integration_gradient = np.gradient(integration_values)
        
        # 注意の安定性
        attention_stability = self._calculate_attention_stability(recent_states)
        
        # 認知負荷の平均
        avg_cognitive_load = np.mean([s.cognitive_load for s in recent_states])
        
        # フロー速度
        flow_velocity = self._calculate_flow_velocity(recent_states)
        
        return {
            'integration_trend': float(np.mean(integration_gradient)),
            'attention_stability': attention_stability,
            'average_cognitive_load': avg_cognitive_load,
            'flow_velocity': flow_velocity,
            'meta_awareness_level': float(np.mean([s.meta_awareness for s in recent_states]))
        }
        
    def _calculate_attention_stability(self, states: List[ConsciousnessState]) -> float:
        """注意の安定性を計算"""
        if len(states) < 2:
            return 1.0
            
        focus_changes = []
        for i in range(1, len(states)):
            prev_focus = set(states[i-1].attention.keys())
            curr_focus = set(states[i].attention.keys())
            stability = len(prev_focus & curr_focus) / max(len(prev_focus | curr_focus), 1)
            focus_changes.append(stability)
            
        return float(np.mean(focus_changes))
        
    def _calculate_flow_velocity(self, states: List[ConsciousnessState]) -> float:
        """意識の流れの速度を計算"""
        if len(states) < 2:
            return 0.0
            
        velocities = []
        for i in range(1, len(states)):
            v1 = np.array(states[i-1].flow_vector)
            v2 = np.array(states[i].flow_vector)
            dt = states[i].timestamp - states[i-1].timestamp
            if dt > 0:
                velocity = np.linalg.norm(v2 - v1) / dt
                velocities.append(velocity)
                
        return float(np.mean(velocities)) if velocities else 0.0
        
    def export_for_visualization(self, last_n: Optional[int] = None) -> List[Dict]:
        """可視化用にデータをエクスポート"""
        states_to_export = list(self.states)
        if last_n:
            states_to_export = states_to_export[-last_n:]
            
        return [state.to_dict() for state in states_to_export]


class PhenomenalAnalyzer:
    """現象学的分析を行うクラス"""
    
    @staticmethod
    def analyze_qualia_structure(state: ConsciousnessState) -> Dict:
        """クオリアの構造を分析"""
        properties = state.phenomenal_properties
        
        # 現象的特性の次元削減（仮想的な実装）
        intensity = sum(properties.values()) / max(len(properties), 1)
        complexity = len(properties) * np.std(list(properties.values()))
        
        return {
            'intensity': intensity,
            'complexity': complexity,
            'dominant_qualities': sorted(properties.items(), key=lambda x: x[1], reverse=True)[:3],
            'phenomenal_unity': state.integration * state.meta_awareness
        }
        
    @staticmethod
    def detect_phenomenal_transitions(stream: ConsciousnessStream, threshold: float = 0.3) -> List[Dict]:
        """現象的な遷移を検出"""
        transitions = []
        states = list(stream.states)
        
        for i in range(1, len(states)):
            prev_state = states[i-1]
            curr_state = states[i]
            
            # 統合度の急激な変化
            integration_change = abs(curr_state.integration - prev_state.integration)
            
            # 注意の大きな移動
            prev_focus = set(prev_state.attention.keys())
            curr_focus = set(curr_state.attention.keys())
            attention_shift = 1 - (len(prev_focus & curr_focus) / max(len(prev_focus | curr_focus), 1))
            
            if integration_change > threshold or attention_shift > threshold:
                transitions.append({
                    'timestamp': curr_state.timestamp,
                    'type': 'integration_shift' if integration_change > attention_shift else 'attention_shift',
                    'magnitude': max(integration_change, attention_shift)
                })
                
        return transitions


class ConsciousnessFlowGenerator:
    """テスト用の意識フロー生成器"""
    
    def __init__(self, stream: ConsciousnessStream):
        self.stream = stream
        self.running = False
        
    async def start_generation(self):
        """リアルタイムで意識の流れを生成"""
        self.running = True
        phase = 0
        
        while self.running:
            phase += 0.1
            
            # 注意の焦点をシミュレート
            attention = {}
            if np.sin(phase) > 0:
                attention['problem_solving'] = abs(np.sin(phase))
            if np.cos(phase * 0.7) > 0:
                attention['memory_retrieval'] = abs(np.cos(phase * 0.7))
            if np.sin(phase * 1.3) > 0:
                attention['sensory_processing'] = abs(np.sin(phase * 1.3))
                
            # 現象的性質をシミュレート
            phenomenal = {
                'clarity': 0.5 + 0.5 * np.sin(phase * 0.3),
                'vividness': 0.5 + 0.5 * np.cos(phase * 0.5),
                'coherence': 0.5 + 0.5 * np.sin(phase * 0.8)
            }
            
            # フローベクトルを生成
            flow_vector = (
                np.sin(phase) * 2,
                np.cos(phase * 0.7) * 2,
                np.sin(phase * 1.2) * 1
            )
            
            state_data = {
                'attention': attention,
                'integration': 0.5 + 0.3 * np.sin(phase * 0.2),
                'phenomenal_properties': phenomenal,
                'cognitive_load': abs(np.sin(phase * 0.4)),
                'meta_awareness': 0.3 + 0.4 * abs(np.cos(phase * 0.15)),
                'flow_vector': flow_vector
            }
            
            self.stream.add_state(state_data)
            await asyncio.sleep(0.1)  # 100ms間隔で更新
            
    def stop_generation(self):
        """生成を停止"""
        self.running = False