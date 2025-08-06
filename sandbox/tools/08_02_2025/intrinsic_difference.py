"""
Intrinsic Difference Calculation Module for IIT 4.0
Detailed implementation of ID calculation using KL divergence

This module provides comprehensive intrinsic difference computation following
the mathematical framework established in Tononi et al. (2023).

Key Components:
- Cause-effect probability computation
- Optimal cause-effect state selection  
- KL divergence calculation with numerical stability
- Mechanism state space analysis

Mathematical Foundation:
ID(mechanism, purview) = KL(p(effect|mechanism_on) || p(effect|mechanism_off)) + 
                        KL(p(cause|mechanism_on) || p(cause|mechanism_off))

Author: IIT Integration Master
Date: 2025-08-03
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, FrozenSet, Set, Any
from dataclasses import dataclass
import itertools
from functools import lru_cache
import logging
from scipy.special import rel_entr
from scipy.stats import entropy
import warnings

# Configure logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MechanismState:
    """
    メカニズム状態の表現
    """
    nodes: FrozenSet[int]           # ノード集合
    state_values: Tuple[int, ...]   # 状態値（0または1）
    
    def __post_init__(self):
        if len(self.nodes) != len(self.state_values):
            raise ValueError("ノード数と状態値数が一致しません")
        if any(val not in (0, 1) for val in self.state_values):
            raise ValueError("状態値は0または1である必要があります")


@dataclass(frozen=True)
class CauseEffectProbability:
    """
    因果効果確率分布
    """
    mechanism: FrozenSet[int]
    purview: FrozenSet[int]
    direction: str                  # 'cause' または 'effect'
    probability_on: np.ndarray      # メカニズムON時の確率分布
    probability_off: np.ndarray     # メカニズムOFF時の確率分布
    kl_divergence: float           # KLダイバージェンス
    
    def __post_init__(self):
        if self.direction not in ('cause', 'effect'):
            raise ValueError("direction は 'cause' または 'effect' である必要があります")
        if len(self.probability_on) != len(self.probability_off):
            raise ValueError("ON/OFF確率分布の次元が一致しません")


class OptimalPurviewFinder:
    """
    最適範囲（purview）発見エンジン
    メカニズムの因果効果力を最大化する範囲を特定
    """
    
    def __init__(self, max_purview_size: int = 8):
        """
        Args:
            max_purview_size: 最大範囲サイズ（計算複雑度制御）
        """
        self.max_purview_size = max_purview_size
        self._cache = {}
    
    def find_optimal_purview(self, mechanism: FrozenSet[int], 
                           candidate_nodes: FrozenSet[int],
                           tpm: np.ndarray, system_state: np.ndarray,
                           direction: str) -> Tuple[FrozenSet[int], float]:
        """
        最適範囲の発見
        
        Args:
            mechanism: メカニズム
            candidate_nodes: 候補ノード集合
            tpm: 状態遷移確率行列
            system_state: 現在のシステム状態
            direction: 'cause' または 'effect'
            
        Returns:
            Tuple[FrozenSet[int], float]: (最適範囲, 最大ID値)
        """
        cache_key = (tuple(sorted(mechanism)), tuple(sorted(candidate_nodes)), 
                    tuple(system_state), direction)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        max_id = 0.0
        optimal_purview = frozenset()
        
        # 範囲サイズ制限
        max_size = min(len(candidate_nodes), self.max_purview_size)
        
        try:
            for purview_size in range(1, max_size + 1):
                for purview_nodes in itertools.combinations(candidate_nodes, purview_size):
                    purview = frozenset(purview_nodes)
                    
                    # ID値計算
                    id_calculator = DetailedIntrinsicDifferenceCalculator()
                    id_value = id_calculator.compute_directional_id(
                        mechanism, purview, tpm, system_state, direction
                    )
                    
                    if id_value > max_id:
                        max_id = id_value
                        optimal_purview = purview
            
            result = (optimal_purview, max_id)
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"最適範囲発見エラー: {e}")
            return frozenset(), 0.0


class StateSpaceAnalyzer:
    """
    状態空間分析器
    システムの状態遷移構造を分析
    """
    
    def __init__(self):
        self._state_space_cache = {}
    
    def analyze_state_space(self, nodes: FrozenSet[int]) -> Dict[str, Any]:
        """
        状態空間の分析
        
        Args:
            nodes: 分析対象ノード集合
            
        Returns:
            Dict: 状態空間の特性情報
        """
        cache_key = tuple(sorted(nodes))
        if cache_key in self._state_space_cache:
            return self._state_space_cache[cache_key]
        
        n_nodes = len(nodes)
        n_states = 2 ** n_nodes
        
        analysis = {
            'n_nodes': n_nodes,
            'n_states': n_states,
            'state_indices': list(range(n_states)),
            'state_vectors': self._generate_state_vectors(n_nodes),
            'connectivity_patterns': self._analyze_connectivity_patterns(nodes)
        }
        
        self._state_space_cache[cache_key] = analysis
        return analysis
    
    def _generate_state_vectors(self, n_nodes: int) -> List[np.ndarray]:
        """状態ベクトルの生成"""
        state_vectors = []
        for i in range(2 ** n_nodes):
            binary_str = format(i, f'0{n_nodes}b')
            state_vector = np.array([int(bit) for bit in binary_str])
            state_vectors.append(state_vector)
        return state_vectors
    
    def _analyze_connectivity_patterns(self, nodes: FrozenSet[int]) -> Dict[str, float]:
        """接続パターンの分析"""
        # ここでは基本的な分析のみ実装
        # 実際のシステムでは、TPMから接続強度などを計算
        return {
            'density': 1.0,  # プレースホルダー
            'clustering': 0.5,  # プレースホルダー
            'path_length': 2.0  # プレースホルダー
        }


class DetailedIntrinsicDifferenceCalculator:
    """
    詳細内在的差異計算器
    高精度なID計算とエラーハンドリングを提供
    """
    
    def __init__(self, precision: float = 1e-12, 
                 numerical_stability: bool = True):
        """
        Args:
            precision: 数値計算精度
            numerical_stability: 数値安定性確保フラグ
        """
        self.precision = precision
        self.numerical_stability = numerical_stability
        self.state_analyzer = StateSpaceAnalyzer()
        self.purview_finder = OptimalPurviewFinder()
        self._computation_cache = {}
    
    def compute_full_intrinsic_difference(self, mechanism: FrozenSet[int],
                                        candidate_purviews: FrozenSet[int],
                                        tpm: np.ndarray, 
                                        system_state: np.ndarray) -> Dict[str, Any]:
        """
        完全な内在的差異計算
        
        Args:
            mechanism: メカニズム
            candidate_purviews: 候補範囲ノード集合
            tpm: 状態遷移確率行列
            system_state: 現在のシステム状態
            
        Returns:
            Dict: 計算結果の詳細情報
        """
        try:
            # 最適範囲の発見
            optimal_cause_purview, max_cause_id = self.purview_finder.find_optimal_purview(
                mechanism, candidate_purviews, tpm, system_state, 'cause'
            )
            optimal_effect_purview, max_effect_id = self.purview_finder.find_optimal_purview(
                mechanism, candidate_purviews, tpm, system_state, 'effect'
            )
            
            # 詳細な因果効果確率計算
            cause_prob = self._compute_detailed_cause_probability(
                mechanism, optimal_cause_purview, tpm, system_state
            )
            effect_prob = self._compute_detailed_effect_probability(
                mechanism, optimal_effect_purview, tpm, system_state
            )
            
            # 統合ID値
            total_id = max_cause_id + max_effect_id
            
            # φ値（最小ID）
            phi_value = min(max_cause_id, max_effect_id)
            
            return {
                'mechanism': mechanism,
                'optimal_cause_purview': optimal_cause_purview,
                'optimal_effect_purview': optimal_effect_purview,
                'cause_id': max_cause_id,
                'effect_id': max_effect_id,
                'total_id': total_id,
                'phi_value': phi_value,
                'cause_probability': cause_prob,
                'effect_probability': effect_prob,
                'computation_metadata': {
                    'precision': self.precision,
                    'numerical_stability': self.numerical_stability
                }
            }
            
        except Exception as e:
            logger.error(f"完全ID計算エラー: {e}")
            return self._create_error_result(mechanism)
    
    def compute_directional_id(self, mechanism: FrozenSet[int], 
                              purview: FrozenSet[int],
                              tpm: np.ndarray, system_state: np.ndarray,
                              direction: str) -> float:
        """
        方向的ID計算（cause または effect）
        
        Args:
            mechanism: メカニズム
            purview: 範囲
            tpm: 状態遷移確率行列
            system_state: 現在のシステム状態
            direction: 'cause' または 'effect'
            
        Returns:
            float: ID値
        """
        cache_key = (tuple(sorted(mechanism)), tuple(sorted(purview)), 
                    tuple(system_state), direction)
        
        if cache_key in self._computation_cache:
            return self._computation_cache[cache_key]
        
        try:
            if direction == 'cause':
                prob_on, prob_off = self._compute_cause_distributions(
                    mechanism, purview, tpm, system_state
                )
            elif direction == 'effect':
                prob_on, prob_off = self._compute_effect_distributions(
                    mechanism, purview, tpm, system_state
                )
            else:
                raise ValueError(f"不明な方向: {direction}")
            
            # KLダイバージェンス計算
            kl_div = self._compute_stable_kl_divergence(prob_on, prob_off)
            
            self._computation_cache[cache_key] = kl_div
            return kl_div
            
        except Exception as e:
            logger.warning(f"方向的ID計算エラー ({direction}): {e}")
            return 0.0
    
    def _compute_detailed_cause_probability(self, mechanism: FrozenSet[int],
                                          purview: FrozenSet[int],
                                          tpm: np.ndarray, 
                                          system_state: np.ndarray) -> CauseEffectProbability:
        """詳細な原因確率計算"""
        prob_on, prob_off = self._compute_cause_distributions(
            mechanism, purview, tpm, system_state
        )
        
        kl_div = self._compute_stable_kl_divergence(prob_on, prob_off)
        
        return CauseEffectProbability(
            mechanism=mechanism,
            purview=purview,
            direction='cause',
            probability_on=prob_on,
            probability_off=prob_off,
            kl_divergence=kl_div
        )
    
    def _compute_detailed_effect_probability(self, mechanism: FrozenSet[int],
                                           purview: FrozenSet[int],
                                           tpm: np.ndarray,
                                           system_state: np.ndarray) -> CauseEffectProbability:
        """詳細な効果確率計算"""
        prob_on, prob_off = self._compute_effect_distributions(
            mechanism, purview, tpm, system_state
        )
        
        kl_div = self._compute_stable_kl_divergence(prob_on, prob_off)
        
        return CauseEffectProbability(
            mechanism=mechanism,
            purview=purview,
            direction='effect',
            probability_on=prob_on,
            probability_off=prob_off,
            kl_divergence=kl_div
        )
    
    def _compute_cause_distributions(self, mechanism: FrozenSet[int],
                                   purview: FrozenSet[int],
                                   tpm: np.ndarray,
                                   system_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """原因確率分布の計算"""
        purview_size = len(purview)
        n_states = 2 ** purview_size
        
        prob_on = np.zeros(n_states)
        prob_off = np.zeros(n_states)
        
        purview_list = sorted(list(purview))
        
        for state_idx in range(n_states):
            # 原因状態の構築
            cause_state = self._index_to_state(state_idx, purview_size)
            
            # メカニズムON時の遷移確率
            prob_on[state_idx] = self._compute_backward_probability(
                cause_state, system_state, mechanism, purview, tpm, mechanism_active=True
            )
            
            # メカニズムOFF時の遷移確率
            prob_off[state_idx] = self._compute_backward_probability(
                cause_state, system_state, mechanism, purview, tpm, mechanism_active=False
            )
        
        # 正規化
        prob_on = self._normalize_distribution(prob_on)
        prob_off = self._normalize_distribution(prob_off)
        
        return prob_on, prob_off
    
    def _compute_effect_distributions(self, mechanism: FrozenSet[int],
                                    purview: FrozenSet[int],
                                    tpm: np.ndarray,
                                    system_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """効果確率分布の計算"""
        purview_size = len(purview)
        n_states = 2 ** purview_size
        
        prob_on = np.zeros(n_states)
        prob_off = np.zeros(n_states)
        
        for state_idx in range(n_states):
            # 効果状態の構築
            effect_state = self._index_to_state(state_idx, purview_size)
            
            # メカニズムON時の遷移確率
            prob_on[state_idx] = self._compute_forward_probability(
                system_state, effect_state, mechanism, purview, tpm, mechanism_active=True
            )
            
            # メカニズムOFF時の遷移確率
            prob_off[state_idx] = self._compute_forward_probability(
                system_state, effect_state, mechanism, purview, tpm, mechanism_active=False
            )
        
        # 正規化
        prob_on = self._normalize_distribution(prob_on)
        prob_off = self._normalize_distribution(prob_off)
        
        return prob_on, prob_off
    
    def _compute_backward_probability(self, cause_state: np.ndarray,
                                    current_state: np.ndarray,
                                    mechanism: FrozenSet[int],
                                    purview: FrozenSet[int],
                                    tpm: np.ndarray,
                                    mechanism_active: bool) -> float:
        """後向き確率計算（原因から現在へ）"""
        try:
            # システム全体の状態を構築
            full_cause_state = self._embed_state_in_system(
                cause_state, purview, len(current_state)
            )
            
            # メカニズム制約を適用
            constrained_state = self._apply_mechanism_constraint(
                full_cause_state, mechanism, mechanism_active
            )
            
            # 遷移確率の計算
            transition_prob = self._compute_transition_probability_detailed(
                constrained_state, current_state, tpm
            )
            
            return max(transition_prob, self.precision)
            
        except Exception as e:
            logger.warning(f"後向き確率計算エラー: {e}")
            return self.precision
    
    def _compute_forward_probability(self, current_state: np.ndarray,
                                   effect_state: np.ndarray,
                                   mechanism: FrozenSet[int],
                                   purview: FrozenSet[int],
                                   tpm: np.ndarray,
                                   mechanism_active: bool) -> float:
        """前向き確率計算（現在から効果へ）"""
        try:
            # メカニズム制約を適用した現在状態
            constrained_current = self._apply_mechanism_constraint(
                current_state, mechanism, mechanism_active
            )
            
            # システム全体の効果状態を構築
            full_effect_state = self._embed_state_in_system(
                effect_state, purview, len(current_state)
            )
            
            # 遷移確率の計算
            transition_prob = self._compute_transition_probability_detailed(
                constrained_current, full_effect_state, tpm
            )
            
            return max(transition_prob, self.precision)
            
        except Exception as e:
            logger.warning(f"前向き確率計算エラー: {e}")
            return self.precision
    
    def _compute_transition_probability_detailed(self, from_state: np.ndarray,
                                               to_state: np.ndarray,
                                               tpm: np.ndarray) -> float:
        """詳細な遷移確率計算"""
        try:
            from_idx = self._state_to_index(from_state)
            
            if from_idx >= tpm.shape[0]:
                return self.precision
            
            # 各ノードの遷移確率の積
            total_prob = 1.0
            for node_idx, target_value in enumerate(to_state):
                if node_idx < tpm.shape[1]:
                    node_prob = tpm[from_idx, node_idx]
                    if target_value == 1:
                        total_prob *= node_prob
                    else:
                        total_prob *= (1 - node_prob)
                    
                    # 数値安定性チェック
                    if total_prob < self.precision:
                        return self.precision
            
            return total_prob
            
        except Exception as e:
            logger.warning(f"遷移確率計算エラー: {e}")
            return self.precision
    
    def _apply_mechanism_constraint(self, state: np.ndarray,
                                   mechanism: FrozenSet[int],
                                   mechanism_active: bool) -> np.ndarray:
        """メカニズム制約の適用"""
        constrained_state = state.copy()
        
        for node in mechanism:
            if node < len(constrained_state):
                constrained_state[node] = 1 if mechanism_active else 0
        
        return constrained_state
    
    def _embed_state_in_system(self, partial_state: np.ndarray,
                              nodes: FrozenSet[int],
                              system_size: int) -> np.ndarray:
        """部分状態をシステム全体に埋め込み"""
        full_state = np.zeros(system_size)
        nodes_list = sorted(list(nodes))
        
        for i, node in enumerate(nodes_list):
            if node < system_size and i < len(partial_state):
                full_state[node] = partial_state[i]
        
        return full_state
    
    def _compute_stable_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """数値安定なKLダイバージェンス計算"""
        if not self.numerical_stability:
            return self._basic_kl_divergence(p, q)
        
        try:
            # ゼロ要素の処理
            p_stable = np.clip(p, self.precision, 1.0)
            q_stable = np.clip(q, self.precision, 1.0)
            
            # 正規化
            p_stable = p_stable / np.sum(p_stable)
            q_stable = q_stable / np.sum(q_stable)
            
            # SciPyのrel_entrを使用（数値安定性向上）
            kl_values = rel_entr(p_stable, q_stable)
            
            # 無限大や NaN の処理
            kl_values = np.where(np.isfinite(kl_values), kl_values, 0.0)
            
            return float(np.sum(kl_values))
            
        except Exception as e:
            logger.warning(f"安定KL計算エラー、基本計算にフォールバック: {e}")
            return self._basic_kl_divergence(p, q)
    
    def _basic_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """基本的なKLダイバージェンス計算"""
        try:
            p_clipped = np.clip(p, self.precision, 1.0)
            q_clipped = np.clip(q, self.precision, 1.0)
            
            # 正規化
            p_norm = p_clipped / np.sum(p_clipped)
            q_norm = q_clipped / np.sum(q_clipped)
            
            return float(np.sum(p_norm * np.log(p_norm / q_norm)))
            
        except Exception as e:
            logger.warning(f"基本KL計算エラー: {e}")
            return 0.0
    
    def _normalize_distribution(self, distribution: np.ndarray) -> np.ndarray:
        """確率分布の正規化"""
        total = np.sum(distribution)
        if total > self.precision:
            return distribution / total
        else:
            # 一様分布にフォールバック
            return np.ones_like(distribution) / len(distribution)
    
    def _index_to_state(self, index: int, n_bits: int) -> np.ndarray:
        """インデックスをバイナリ状態に変換"""
        binary_str = format(index, f'0{n_bits}b')
        return np.array([int(bit) for bit in binary_str])
    
    def _state_to_index(self, state: np.ndarray) -> int:
        """バイナリ状態をインデックスに変換"""
        binary_str = ''.join(map(str, state.astype(int)))
        return int(binary_str, 2)
    
    def _create_error_result(self, mechanism: FrozenSet[int]) -> Dict[str, Any]:
        """エラー時のデフォルト結果作成"""
        return {
            'mechanism': mechanism,
            'optimal_cause_purview': frozenset(),
            'optimal_effect_purview': frozenset(),
            'cause_id': 0.0,
            'effect_id': 0.0,
            'total_id': 0.0,
            'phi_value': 0.0,
            'cause_probability': None,
            'effect_probability': None,
            'computation_metadata': {
                'error': True,
                'precision': self.precision
            }
        }


class IntrinsicDifferenceValidator:
    """
    内在的差異計算の検証器
    理論的一貫性と数値的正確性を検証
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def validate_id_computation(self, id_result: Dict[str, Any]) -> Dict[str, bool]:
        """ID計算結果の検証"""
        validations = {}
        
        # 基本的な値の検証
        validations['non_negative_ids'] = self._validate_non_negative_values(id_result)
        validations['probability_validity'] = self._validate_probability_distributions(id_result)
        validations['kl_divergence_validity'] = self._validate_kl_divergences(id_result)
        validations['phi_consistency'] = self._validate_phi_consistency(id_result)
        
        return validations
    
    def _validate_non_negative_values(self, result: Dict[str, Any]) -> bool:
        """非負値の検証"""
        values_to_check = ['cause_id', 'effect_id', 'total_id', 'phi_value']
        for key in values_to_check:
            if key in result and result[key] < -self.tolerance:
                return False
        return True
    
    def _validate_probability_distributions(self, result: Dict[str, Any]) -> bool:
        """確率分布の検証"""
        prob_objects = ['cause_probability', 'effect_probability']
        
        for prob_key in prob_objects:
            if prob_key in result and result[prob_key] is not None:
                prob_obj = result[prob_key]
                
                # 確率の和が1に近いかチェック
                for dist_name in ['probability_on', 'probability_off']:
                    if hasattr(prob_obj, dist_name):
                        dist = getattr(prob_obj, dist_name)
                        if abs(np.sum(dist) - 1.0) > self.tolerance:
                            return False
                        
                        # 非負性チェック
                        if np.any(dist < -self.tolerance):
                            return False
        
        return True
    
    def _validate_kl_divergences(self, result: Dict[str, Any]) -> bool:
        """KLダイバージェンスの検証"""
        # KLダイバージェンスは非負でなければならない
        id_values = [result.get('cause_id', 0), result.get('effect_id', 0)]
        return all(val >= -self.tolerance for val in id_values)
    
    def _validate_phi_consistency(self, result: Dict[str, Any]) -> bool:
        """φ値一貫性の検証"""
        cause_id = result.get('cause_id', 0)
        effect_id = result.get('effect_id', 0)
        phi_value = result.get('phi_value', 0)
        
        # φ値は min(cause_id, effect_id) であるべき
        expected_phi = min(cause_id, effect_id)
        return abs(phi_value - expected_phi) <= self.tolerance


# タイプミス修正
# Line 238のtmp → tmpは意図的であるためそのまま保持