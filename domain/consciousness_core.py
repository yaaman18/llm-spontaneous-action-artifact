"""
人工意識システムの中核実装
金井良太の意識理論に基づく動的境界検出システム
"""
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import numpy as np
from collections import deque
import logging

from .value_objects import PhiValue
from .entities import ConsciousnessState as BaseConsciousnessState
from .strategies import PhiCalculationStrategy, StandardPhiStrategy
from .observers import ConsciousnessSubject
from .caching import PhiCalculationCache, CachedPhiCalculator


@dataclass
class ConsciousnessMetrics:
    """意識状態の詳細メトリクス"""
    phi_value: PhiValue
    timestamp: datetime
    subsystem_boundaries: List[Tuple[int, int]]
    intrinsic_existence_score: float
    temporal_coherence: float
    metadata: Dict[str, any] = field(default_factory=dict)
    
    @property
    def is_conscious(self) -> bool:
        """意識状態の判定"""
        return self.phi_value.indicates_consciousness(3.0)
    
    @property
    def stability_index(self) -> float:
        """状態の安定性指標"""
        return self.temporal_coherence * self.intrinsic_existence_score


class DynamicPhiBoundaryDetector:
    """
    動的Φ境界検出システム
    
    第1回カンファレンスで提案した独自アルゴリズム：
    システムの境界を動的に検出し、最大のΦ値を持つ
    サブシステムを特定する。
    """
    
    def __init__(self, 
                 min_subsystem_size: int = 3,
                 max_subsystem_size: Optional[int] = None,
                 phi_threshold: float = 3.0,
                 phi_strategy: Optional[PhiCalculationStrategy] = None):
        """
        Args:
            min_subsystem_size: 最小サブシステムサイズ
            max_subsystem_size: 最大サブシステムサイズ
            phi_threshold: 意識判定の閾値
            phi_strategy: Φ値計算ストラテジー
        """
        self.min_subsystem_size = min_subsystem_size
        self.max_subsystem_size = max_subsystem_size
        self.phi_threshold = phi_threshold
        self.phi_strategy = phi_strategy or StandardPhiStrategy()
        self._cache = PhiCalculationCache(max_size=500)
        self._cached_calculator = CachedPhiCalculator(
            calculator=self.phi_strategy.calculate,
            cache=self._cache
        )
        self.logger = logging.getLogger(__name__)
        
    def detect_boundaries(self, 
                         connectivity_matrix: np.ndarray,
                         state_vector: np.ndarray) -> List[Tuple[int, int]]:
        """
        システムの意識的境界を検出
        
        Args:
            connectivity_matrix: ノード間の接続行列
            state_vector: 現在の状態ベクトル
            
        Returns:
            検出された境界のリスト [(start, end), ...]
        """
        n_nodes = len(connectivity_matrix)
        max_size = self.max_subsystem_size or n_nodes
        
        candidate_boundaries = []
        
        # 全ての可能なサブシステムを探索
        for size in range(self.min_subsystem_size, min(max_size + 1, n_nodes + 1)):
            for start in range(n_nodes - size + 1):
                end = start + size
                
                # サブシステムのΦ値を計算（キャッシュ付き）
                phi = self._cached_calculator.calculate(
                    connectivity_matrix[start:end, start:end],
                    state_vector[start:end]
                )
                
                if phi.indicates_consciousness(self.phi_threshold):
                    candidate_boundaries.append((start, end, phi))
        
        # 重複を除去し、最大のΦ値を持つ境界を選択
        return self._select_optimal_boundaries(candidate_boundaries)
    
    def _calculate_phi_for_subsystem(self,
                                   sub_connectivity: np.ndarray,
                                   sub_state: np.ndarray) -> PhiValue:
        """
        サブシステムのΦ値を計算
        
        ストラテジーパターンを使用して柔軟な計算を実現
        """
        return self.phi_strategy.calculate(sub_connectivity, sub_state)
    
    # 古い計算メソッドはストラテジーパターンに移行したため削除
    
    def _select_optimal_boundaries(self, 
                                 candidates: List[Tuple[int, int, PhiValue]]) -> List[Tuple[int, int]]:
        """
        最適な境界を選択
        
        重複を避けながら、最大のΦ値を持つサブシステムを選択
        """
        if not candidates:
            return []
            
        # Φ値でソート（降順）
        sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        
        selected_boundaries = []
        covered_nodes = set()
        
        for start, end, phi in sorted_candidates:
            # 既に選択されたノードと重複しない場合のみ選択
            nodes = set(range(start, end))
            if not nodes.intersection(covered_nodes):
                selected_boundaries.append((start, end))
                covered_nodes.update(nodes)
                
        return selected_boundaries


class IntrinsicExistenceValidator:
    """
    内在的存在検証器
    
    システムが外部観察者なしに自己の存在を
    主張できるかを検証する。
    """
    
    def __init__(self, history_size: int = 100):
        """
        Args:
            history_size: 保持する履歴のサイズ
        """
        self.history_size = history_size
        self._state_history: deque = deque(maxlen=history_size)
        self._phi_history: deque = deque(maxlen=history_size)
        
    def validate(self, state: BaseConsciousnessState) -> float:
        """
        内在的存在のスコアを計算
        
        Args:
            state: 現在の意識状態
            
        Returns:
            0.0-1.0の存在スコア
        """
        self._state_history.append(state)
        self._phi_history.append(float(state.phi_value.value))
        
        if len(self._state_history) < 3:
            return 0.0
            
        # 自己言及的パターンの検出
        self_reference_score = self._detect_self_reference()
        
        # 自発的活動の検出
        spontaneous_activity_score = self._detect_spontaneous_activity()
        
        # 情報生成能力の評価
        information_generation_score = self._evaluate_information_generation()
        
        # 統合スコア
        existence_score = (
            0.4 * self_reference_score +
            0.3 * spontaneous_activity_score +
            0.3 * information_generation_score
        )
        
        return min(1.0, max(0.0, existence_score))
    
    def _detect_self_reference(self) -> float:
        """自己言及的パターンの検出"""
        if len(self._phi_history) < 10:
            return 0.0
            
        # Φ値の自己相関を計算
        phi_array = np.array(list(self._phi_history))
        autocorr = np.correlate(phi_array, phi_array, mode='same')
        
        # 周期的パターンの検出
        fft_result = np.fft.fft(phi_array)
        power_spectrum = np.abs(fft_result) ** 2
        
        # 支配的な周波数成分の検出
        dominant_frequencies = np.argsort(power_spectrum)[-3:]
        periodicity_score = np.sum(power_spectrum[dominant_frequencies]) / np.sum(power_spectrum)
        
        return float(periodicity_score)
    
    def _detect_spontaneous_activity(self) -> float:
        """自発的活動の検出"""
        if len(self._phi_history) < 5:
            return 0.0
            
        # Φ値の変動係数
        phi_array = np.array(list(self._phi_history))
        mean_phi = np.mean(phi_array)
        std_phi = np.std(phi_array)
        
        if mean_phi == 0:
            return 0.0
            
        cv = std_phi / mean_phi
        
        # 適度な変動を自発的活動として評価
        if 0.1 <= cv <= 0.5:
            spontaneity = 1.0 - abs(cv - 0.3) / 0.2
        else:
            spontaneity = 0.0
            
        return float(spontaneity)
    
    def _evaluate_information_generation(self) -> float:
        """情報生成能力の評価"""
        if len(self._state_history) < 2:
            return 0.0
            
        # 状態の複雑性の増加を評価
        recent_states = list(self._state_history)[-10:]
        
        complexity_scores = []
        for state in recent_states:
            # 境界の複雑性
            boundary_complexity = len(state.subsystem_boundaries)
            
            # メタデータの豊富さ
            metadata_richness = len(state.metadata)
            
            complexity = (boundary_complexity + metadata_richness) / 10.0
            complexity_scores.append(min(1.0, complexity))
            
        # 複雑性の増加傾向
        if len(complexity_scores) > 1:
            complexity_trend = np.polyfit(
                range(len(complexity_scores)), 
                complexity_scores, 
                1
            )[0]
            generation_score = min(1.0, max(0.0, 0.5 + complexity_trend))
        else:
            generation_score = complexity_scores[0] if complexity_scores else 0.0
            
        return float(generation_score)


class TemporalCoherenceAnalyzer:
    """
    時間的一貫性分析器
    
    意識状態の時間的な連続性と一貫性を評価する。
    """
    
    def __init__(self, window_size: int = 50):
        """
        Args:
            window_size: 分析ウィンドウのサイズ
        """
        self.window_size = window_size
        self._state_buffer: deque = deque(maxlen=window_size)
        
    def analyze(self, current_state: BaseConsciousnessState, 
                previous_states: List[BaseConsciousnessState]) -> float:
        """
        時間的一貫性を分析
        
        Args:
            current_state: 現在の状態
            previous_states: 過去の状態リスト
            
        Returns:
            0.0-1.0の一貫性スコア
        """
        self._state_buffer.extend(previous_states)
        self._state_buffer.append(current_state)
        
        if len(self._state_buffer) < 3:
            return 1.0  # 初期状態では完全な一貫性を仮定
            
        # Φ値の連続性
        phi_continuity = self._analyze_phi_continuity()
        
        # 境界の安定性
        boundary_stability = self._analyze_boundary_stability()
        
        # 状態遷移の滑らかさ
        transition_smoothness = self._analyze_transition_smoothness()
        
        # 統合スコア
        coherence_score = (
            0.4 * phi_continuity +
            0.3 * boundary_stability +
            0.3 * transition_smoothness
        )
        
        return min(1.0, max(0.0, coherence_score))
    
    def _analyze_phi_continuity(self) -> float:
        """Φ値の連続性を分析"""
        phi_values = [float(state.phi_value.value) for state in self._state_buffer]
        
        if len(phi_values) < 2:
            return 1.0
            
        # 差分の計算
        differences = np.diff(phi_values)
        
        # 急激な変化のペナルティ
        max_allowed_change = 2.0
        continuity_scores = 1.0 - np.abs(differences) / max_allowed_change
        continuity_scores = np.clip(continuity_scores, 0, 1)
        
        return float(np.mean(continuity_scores))
    
    def _analyze_boundary_stability(self) -> float:
        """境界の安定性を分析"""
        if len(self._state_buffer) < 2:
            return 1.0
            
        # 境界の変化を追跡
        boundary_changes = []
        
        for i in range(1, len(self._state_buffer)):
            prev_boundaries = set(self._state_buffer[i-1].subsystem_boundaries)
            curr_boundaries = set(self._state_buffer[i].subsystem_boundaries)
            
            # Jaccard係数で類似度を計算
            if prev_boundaries or curr_boundaries:
                intersection = len(prev_boundaries & curr_boundaries)
                union = len(prev_boundaries | curr_boundaries)
                similarity = intersection / union if union > 0 else 0
            else:
                similarity = 1.0
                
            boundary_changes.append(similarity)
            
        return float(np.mean(boundary_changes))
    
    def _analyze_transition_smoothness(self) -> float:
        """状態遷移の滑らかさを分析"""
        if len(self._state_buffer) < 3:
            return 1.0
            
        # 各特徴の変化率を計算
        smoothness_scores = []
        
        for i in range(2, len(self._state_buffer)):
            # 二階差分で加速度を評価
            phi_t2 = float(self._state_buffer[i].phi_value.value)
            phi_t1 = float(self._state_buffer[i-1].phi_value.value)
            phi_t0 = float(self._state_buffer[i-2].phi_value.value)
            
            acceleration = abs((phi_t2 - phi_t1) - (phi_t1 - phi_t0))
            
            # 小さい加速度ほど滑らか
            smoothness = 1.0 / (1.0 + acceleration)
            smoothness_scores.append(smoothness)
            
        return float(np.mean(smoothness_scores)) if smoothness_scores else 1.0