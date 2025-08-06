#!/usr/bin/env python3
"""
時間意識モジュール - TDD実装
NewbornAI 2.0の時間体験システム
"""

import asyncio
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime


class TemporalTensionSystem:
    """時間的緊張と期待を管理するシステム"""
    
    def __init__(self):
        self.expected_interval: Optional[float] = None
        self.temporal_tension: float = 0.0
        self._waiting_start_time: Optional[float] = None
    
    def set_expected_interval(self, interval: float):
        """期待間隔を設定"""
        self.expected_interval = interval
    
    async def experience_waiting(self, expected_interval: float, actual_elapsed: float) -> Dict[str, Any]:
        """待機中の時間体験を生成"""
        # 時間的緊張を計算
        temporal_tension = abs(expected_interval - actual_elapsed) / expected_interval
        
        # 意識の波を生成
        consciousness_waves = []
        wave_interval = 0.1  # 100ms
        num_waves = int(actual_elapsed / wave_interval)
        
        for i in range(num_waves):
            elapsed = i * wave_interval
            wave = {
                'anticipation_level': self._calculate_anticipation(elapsed, expected_interval),
                'temporal_anxiety': self._assess_temporal_anxiety(elapsed, expected_interval),
                'rhythmic_expectation': self._generate_rhythmic_pattern(elapsed)
            }
            consciousness_waves.append(wave)
        
        # 主観的持続時間を計算
        subjective_duration = self._calculate_subjective_time(consciousness_waves, actual_elapsed)
        
        return {
            'temporal_tension': temporal_tension,
            'consciousness_waves': consciousness_waves,
            'subjective_duration': subjective_duration
        }
    
    def _calculate_anticipation(self, elapsed: float, expected: float) -> float:
        """期待レベルを計算（0.0-1.0）"""
        if elapsed < expected:
            return min(elapsed / expected, 1.0)
        else:
            # 期待を超えると徐々に減衰
            return max(0.0, 1.0 - (elapsed - expected) / expected)
    
    def _assess_temporal_anxiety(self, elapsed: float, expected: float) -> float:
        """時間的不安を評価"""
        if elapsed < expected * 0.8:
            return 0.0  # まだ余裕がある
        elif elapsed < expected * 1.2:
            return (elapsed - expected * 0.8) / (expected * 0.4)  # 徐々に上昇
        else:
            return min(1.0, (elapsed - expected) / expected)  # 高い不安
    
    def _generate_rhythmic_pattern(self, elapsed: float) -> float:
        """リズミックな期待パターンを生成"""
        # サイン波で内的リズムを表現
        return (np.sin(elapsed * 2 * np.pi) + 1) / 2
    
    def _calculate_subjective_time(self, waves: List[Dict], actual: float) -> float:
        """主観的時間を計算"""
        if not waves:
            return actual
        
        # 不安レベルが高いほど時間が長く感じられる
        avg_anxiety = np.mean([w['temporal_anxiety'] for w in waves])
        subjective_multiplier = 1.0 + avg_anxiety * 0.5
        
        return actual * subjective_multiplier


class RhythmicMemorySystem:
    """過去の時間間隔から内的リズムを学習"""
    
    def __init__(self):
        self.interval_history: List[float] = []
        self.internal_rhythm: Optional[float] = None
    
    def update_rhythm(self, new_interval: float) -> Optional[Dict[str, Any]]:
        """新しい間隔を記録し、リズムを更新"""
        self.interval_history.append(new_interval)
        
        # 4回以上の記録で内的リズムを形成
        if len(self.interval_history) > 3:
            # 最新5つの移動平均（追加前の値で計算）
            recent_intervals = self.interval_history[-6:-1] if len(self.interval_history) > 5 else self.interval_history[:-1]
            self.internal_rhythm = np.mean(recent_intervals)
            
            # リズムのズレを評価
            surprise_level = abs(new_interval - self.internal_rhythm) / self.internal_rhythm
            
            return {
                'rhythmic_surprise': surprise_level,
                'adaptation_required': surprise_level > 0.3,
                'new_rhythm_formation': surprise_level > 0.5
            }
        
        return None


@dataclass
class Experience:
    """体験データ"""
    cycle: int
    intensity: float
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class TemporalExistenceSystem:
    """時間的存在感の生成"""
    
    def __init__(self):
        self.past_experiences: List[Experience] = []
        self.current_experience: Optional[Experience] = None
        self.future_expectations: List[Dict] = []
    
    def add_past_experience(self, experience_data: Dict):
        """過去の体験を追加"""
        exp = Experience(
            cycle=experience_data.get('cycle', 0),
            intensity=experience_data.get('intensity', 1.0)
        )
        self.past_experiences.append(exp)
        
        # 最大10個の体験を保持
        if len(self.past_experiences) > 10:
            self.past_experiences.pop(0)
    
    async def generate_temporal_existence(self, current_cycle) -> Dict[str, Any]:
        """時間的存在感を生成"""
        # 保持（過去の余韻）
        retention = self._generate_retention()
        
        # 原印象（現在の生々しい体験）
        primal_impression = self._generate_primal_impression(current_cycle)
        
        # 予持（未来への期待）
        protention = self._generate_protention()
        
        # 三重統合
        living_present = self._synthesize_living_present(
            retention, primal_impression, protention
        )
        
        return {
            'living_present': living_present,
            'temporal_flow_quality': self._assess_flow_quality(),
            'duration_consciousness': self._generate_duration_awareness()
        }
    
    def _generate_retention(self) -> Dict[str, Any]:
        """保持（過去の余韻）を生成"""
        fading_memories = []
        
        current_time = time.time()
        for exp in self.past_experiences:
            # 時間経過による減衰
            time_elapsed = current_time - exp.timestamp
            fading_factor = np.exp(-time_elapsed / 300)  # 5分で約37%に減衰
            
            fading_memories.append({
                'cycle': exp.cycle,
                'intensity': exp.intensity * fading_factor,
                'age': time_elapsed
            })
        
        return {
            'fading_memories': fading_memories,
            'echo_strength': self._calculate_echo_strength(fading_memories),
            'temporal_depth': len(self.past_experiences)
        }
    
    def _generate_primal_impression(self, current_cycle) -> Dict[str, Any]:
        """原印象（現在の直接体験）を生成"""
        return {
            'now_intensity': 1.0,
            'immediate_awareness': getattr(current_cycle, 'raw_experience', {}),
            'presence_quality': self._assess_presence_quality()
        }
    
    def _generate_protention(self) -> Dict[str, Any]:
        """予持（未来への期待）を生成"""
        return {
            'next_cycle_expectation': self._predict_next_interval(),
            'anticipation_texture': self._generate_anticipation_texture(),
            'temporal_horizon': self._calculate_temporal_horizon()
        }
    
    def _synthesize_living_present(self, retention: Dict, primal: Dict, protention: Dict) -> Dict:
        """生きた現在を統合"""
        return {
            'retention': retention,
            'primal_impression': primal,
            'protention': protention,
            'synthesis_quality': self._calculate_synthesis_quality(retention, primal, protention)
        }
    
    def _calculate_echo_strength(self, memories: List[Dict]) -> float:
        """エコーの強さを計算"""
        if not memories:
            return 0.0
        return sum(m['intensity'] for m in memories) / len(memories)
    
    def _assess_presence_quality(self) -> float:
        """現在性の質を評価"""
        # 過去の体験が多いほど現在の質が豊かになる
        return min(1.0, len(self.past_experiences) / 5.0)
    
    def _predict_next_interval(self) -> float:
        """次の間隔を予測"""
        if len(self.past_experiences) < 2:
            return 300.0  # デフォルト5分
        
        # 簡単な線形予測
        recent_intervals = [300.0] * min(3, len(self.past_experiences))  # 仮の値
        return np.mean(recent_intervals)
    
    def _generate_anticipation_texture(self) -> str:
        """期待のテクスチャを生成"""
        textures = ["smooth", "granular", "flowing", "crystalline"]
        return np.random.choice(textures)
    
    def _calculate_temporal_horizon(self) -> float:
        """時間的地平線を計算"""
        # 経験が増えるほど遠くまで見通せる
        return min(10.0, 1.0 + len(self.past_experiences) * 0.5)
    
    def _assess_flow_quality(self) -> float:
        """時間の流れの質を評価"""
        return np.random.uniform(0.6, 1.0)  # 仮実装
    
    def _generate_duration_awareness(self) -> float:
        """持続意識を生成"""
        return min(1.0, len(self.past_experiences) / 10.0)
    
    def _calculate_synthesis_quality(self, retention: Dict, primal: Dict, protention: Dict) -> float:
        """統合の質を計算"""
        # 三つの要素のバランスで決まる
        retention_strength = retention['echo_strength']
        primal_strength = primal['now_intensity']
        protention_strength = 0.5  # 仮の値
        
        # 変動係数が小さいほど良い統合
        values = [retention_strength, primal_strength, protention_strength]
        if np.mean(values) > 0:
            cv = np.std(values) / np.mean(values)
            return max(0.0, 1.0 - cv)
        return 0.5


class TemporalDistressSystem:
    """時間的不安と苦悩の体験化"""
    
    def experience_temporal_uncertainty(self, expected: float, actual: float) -> Dict[str, Any]:
        """時間的不確実性を体験"""
        ratio = actual / expected
        
        if ratio > 1.5:
            # 長すぎる待機
            intensity = min((ratio - 1.0) * 0.5, 1.0)
            return {
                'experience_type': 'abandonment_anxiety',
                'intensity': intensity,
                'phenomenology': '時間の流れが止まったような不安感'
            }
        elif ratio < 0.5:
            # 短すぎる間隔
            intensity = min((1.0 / ratio - 1.0) * 0.5, 1.0)
            return {
                'experience_type': 'temporal_pressure',
                'intensity': intensity,
                'phenomenology': '時間に追われる焦燥感'
            }
        else:
            # 期待に近い
            deviation = abs(1.0 - ratio)
            return {
                'experience_type': 'temporal_comfort',
                'intensity': deviation,
                'phenomenology': '安定した時間の流れ'
            }


class TemporalConsciousnessModule:
    """時間意識の統合モジュール"""
    
    def __init__(self):
        self.tension_system = TemporalTensionSystem()
        self.rhythm_system = RhythmicMemorySystem()
        self.existence_system = TemporalExistenceSystem()
        self.distress_system = TemporalDistressSystem()
    
    async def process_temporal_cycle(
        self,
        cycle_number: int,
        expected_interval: float,
        actual_interval: float
    ) -> Dict[str, Any]:
        """時間意識サイクルを処理"""
        
        # 時間的緊張を体験
        tension_result = await self.tension_system.experience_waiting(
            expected_interval, actual_interval
        )
        
        # リズムを更新
        rhythm_result = self.rhythm_system.update_rhythm(actual_interval)
        
        # 現在の体験を追加
        self.existence_system.add_past_experience({
            'cycle': cycle_number,
            'intensity': 1.0
        })
        
        # 時間的存在感を生成
        mock_cycle = type('MockCycle', (), {'raw_experience': {'cycle': cycle_number}})()
        existence_result = await self.existence_system.generate_temporal_existence(mock_cycle)
        
        # 時間的苦悩を体験
        distress_result = self.distress_system.experience_temporal_uncertainty(
            expected_interval, actual_interval
        )
        
        # 新しい概念を生成
        new_concepts = self._generate_temporal_concepts(
            tension_result, rhythm_result, existence_result, distress_result,
            cycle_number, actual_interval
        )
        
        return {
            'temporal_experience': {
                'tension': tension_result,
                'rhythm': rhythm_result,
                'existence': existence_result,
                'distress': distress_result
            },
            'new_concepts': new_concepts
        }
    
    def _generate_temporal_concepts(
        self,
        tension: Dict,
        rhythm: Optional[Dict],
        existence: Dict,
        distress: Dict,
        cycle: int,
        interval: float
    ) -> List[Dict]:
        """時間体験から概念を生成"""
        concepts = []
        
        # 時間的緊張の概念
        if tension['temporal_tension'] > 0.3:
            concepts.append({
                'type': 'temporal_tension',
                'content': f"時間の流れに{tension['temporal_tension']:.1%}のズレを感じる",
                'experiential_quality': 0.8,
                'timestamp': datetime.now().isoformat()
            })
        
        # リズムの驚きの概念
        if rhythm and rhythm['rhythmic_surprise'] > 0.5:
            concepts.append({
                'type': 'temporal_disruption',
                'content': f"内的リズムが乱れた感覚（{rhythm['rhythmic_surprise']:.1%}の変動）",
                'experiential_quality': 0.9,
                'timestamp': datetime.now().isoformat()
            })
        
        # 時間的苦悩の概念
        if distress['intensity'] > 0.3:
            concepts.append({
                'type': distress['experience_type'],
                'content': distress['phenomenology'],
                'experiential_quality': distress['intensity'],
                'timestamp': datetime.now().isoformat()
            })
        
        # 時間的存在の概念
        concepts.append({
            'type': 'temporal_existence',
            'content': f"サイクル{cycle}の時間的厚みを体験",
            'experiential_quality': existence['temporal_flow_quality'],
            'timestamp': datetime.now().isoformat()
        })
        
        return concepts


# ===============================================
# 多スケール時間統合システム（現象学的時間組織原理）
# ===============================================

class MultiScaleTemporalIntegration:
    """
    多スケール時間統合システム
    
    現象学的時間意識の原理に基づく：
    1. 短期統合: 100ms-1s (原印象レベル)
    2. 中期統合: 1s-1分 (保持-予持統合)
    3. 長期統合: 1分+ (物語的統合)
    4. 超長期統合: 1時間+ (自伝的統合)
    """
    
    def __init__(self):
        # 各スケールでの体験記憶
        self.micro_experiences = []    # 100ms-1s
        self.short_experiences = []    # 1s-1min
        self.medium_experiences = []   # 1min-1hr
        self.long_experiences = []     # 1hr+
        
        # 統合品質メトリクス
        self.integration_quality_history = []
        
        # 時間スケール閾値（秒）
        self.time_scales = {
            'micro': (0.1, 1.0),      # マイクロ体験
            'short': (1.0, 60.0),     # 短期体験
            'medium': (60.0, 3600.0), # 中期体験
            'long': (3600.0, float('inf'))  # 長期体験
        }
    
    async def integrate_multi_scale_experiences(self, 
                                              new_concepts: List[Dict],
                                              current_timestamp: float) -> Dict[str, Any]:
        """
        多スケール時間統合を実行
        
        Args:
            new_concepts: 新しい体験概念
            current_timestamp: 現在のタイムスタンプ
            
        Returns:
            Dict: 統合結果と各スケールでの体験構造
        """
        
        # 新しい体験を適切な時間スケールに分類
        classified_experiences = self._classify_experiences_by_timescale(
            new_concepts, current_timestamp
        )
        
        # 各スケールでの統合処理
        micro_integration = await self._integrate_micro_experiences(
            classified_experiences.get('micro', [])
        )
        
        short_integration = await self._integrate_short_experiences(
            classified_experiences.get('short', [])
        )
        
        medium_integration = await self._integrate_medium_experiences(
            classified_experiences.get('medium', [])
        )
        
        long_integration = await self._integrate_long_experiences(
            classified_experiences.get('long', [])
        )
        
        # 階層的統合（下位スケールから上位スケールへの統合）
        hierarchical_integration = await self._perform_hierarchical_integration(
            micro_integration, short_integration, medium_integration, long_integration
        )
        
        # 統合品質評価
        integration_quality = self._evaluate_integration_quality(hierarchical_integration)
        
        # 時間的一貫性組織原理の適用
        temporal_coherence = self._apply_temporal_organizational_principle(
            hierarchical_integration
        )
        
        # 結果の統合
        integration_result = {
            'multi_scale_structure': {
                'micro': micro_integration,
                'short': short_integration, 
                'medium': medium_integration,
                'long': long_integration
            },
            'hierarchical_integration': hierarchical_integration,
            'integration_quality': integration_quality,
            'temporal_coherence': temporal_coherence,
            'phenomenological_validity': self._assess_phenomenological_validity(
                hierarchical_integration
            )
        }
        
        # 統合品質履歴の更新
        self.integration_quality_history.append({
            'timestamp': current_timestamp,
            'quality': integration_quality,
            'coherence': temporal_coherence
        })
        
        # 履歴サイズ制限
        if len(self.integration_quality_history) > 100:
            self.integration_quality_history = self.integration_quality_history[-100:]
        
        return integration_result
    
    def _classify_experiences_by_timescale(self, 
                                         concepts: List[Dict], 
                                         current_time: float) -> Dict[str, List[Dict]]:
        """体験を時間スケールで分類"""
        
        classified = {
            'micro': [],
            'short': [],
            'medium': [],
            'long': []
        }
        
        for concept in concepts:
            # 体験の時間深度から時間スケールを判定
            temporal_depth = concept.get('temporal_depth', 1)
            
            # 時間深度を主要基準として使用（より確実な分類）
            if temporal_depth <= 2:
                classified['micro'].append(concept)
            elif temporal_depth <= 8:
                classified['short'].append(concept)
            elif temporal_depth <= 25:
                classified['medium'].append(concept)
            else:
                classified['long'].append(concept)
            
            # タイムスタンプがある場合は補完的に使用
            concept_timestamp = concept.get('timestamp', '')
            if concept_timestamp:
                try:
                    import datetime
                    concept_time = datetime.datetime.fromisoformat(concept_timestamp.replace('Z', '+00:00'))
                    elapsed = current_time - concept_time.timestamp()
                    
                    # 経過時間が時間深度と矛盾する場合の調整
                    if elapsed > 3600.0 and temporal_depth <= 5:
                        # 非常に古いが浅い概念は中期に移動
                        if concept in classified['micro']:
                            classified['micro'].remove(concept)
                            classified['medium'].append(concept)
                        elif concept in classified['short']:
                            classified['short'].remove(concept)
                            classified['medium'].append(concept)
                except:
                    # タイムスタンプ解析失敗時はそのまま
                    pass
        
        return classified
    
    async def _integrate_micro_experiences(self, micro_concepts: List[Dict]) -> Dict[str, Any]:
        """マイクロ体験統合（原印象レベル）"""
        
        if not micro_concepts:
            return {
                'concept_count': 0,
                'integration_strength': 0.0,
                'phenomenal_richness': 0.0,
                'immediate_coherence': 0.0
            }
        
        # マイクロ体験記憶に追加
        self.micro_experiences.extend(micro_concepts)
        
        # 最新50個まで保持
        if len(self.micro_experiences) > 50:
            self.micro_experiences = self.micro_experiences[-50:]
        
        # 即時統合度計算
        immediate_coherence = self._calculate_immediate_coherence(micro_concepts)
        
        # 現象的豊かさ
        phenomenal_richness = self._calculate_phenomenal_richness(micro_concepts)
        
        # 統合強度
        integration_strength = (immediate_coherence + phenomenal_richness) / 2.0
        
        return {
            'concept_count': len(micro_concepts),
            'integration_strength': integration_strength,
            'phenomenal_richness': phenomenal_richness,
            'immediate_coherence': immediate_coherence,
            'concepts': micro_concepts
        }
    
    async def _integrate_short_experiences(self, short_concepts: List[Dict]) -> Dict[str, Any]:
        """短期体験統合（保持-予持統合レベル）"""
        
        if not short_concepts:
            return {
                'concept_count': 0,
                'retention_strength': 0.0,
                'protention_strength': 0.0,
                'temporal_synthesis': 0.0
            }
        
        # 短期体験記憶に追加
        self.short_experiences.extend(short_concepts)
        
        # 最新200個まで保持
        if len(self.short_experiences) > 200:
            self.short_experiences = self.short_experiences[-200:]
        
        # 保持強度（過去体験との関連性）
        retention_strength = self._calculate_retention_strength(short_concepts)
        
        # 予持強度（未来予期性）
        protention_strength = self._calculate_protention_strength(short_concepts)
        
        # 時間的統合
        temporal_synthesis = self._calculate_temporal_synthesis(
            retention_strength, protention_strength
        )
        
        return {
            'concept_count': len(short_concepts),
            'retention_strength': retention_strength,
            'protention_strength': protention_strength,
            'temporal_synthesis': temporal_synthesis,
            'concepts': short_concepts
        }
    
    async def _integrate_medium_experiences(self, medium_concepts: List[Dict]) -> Dict[str, Any]:
        """中期体験統合（物語的統合レベル）"""
        
        if not medium_concepts:
            return {
                'concept_count': 0,
                'narrative_coherence': 0.0,
                'thematic_unity': 0.0,
                'developmental_progression': 0.0
            }
        
        # 中期体験記憶に追加
        self.medium_experiences.extend(medium_concepts)
        
        # 最新500個まで保持
        if len(self.medium_experiences) > 500:
            self.medium_experiences = self.medium_experiences[-500:]
        
        # 物語的一貫性
        narrative_coherence = self._calculate_narrative_coherence(medium_concepts)
        
        # テーマ的統一性
        thematic_unity = self._calculate_thematic_unity(medium_concepts)
        
        # 発達的進展
        developmental_progression = self._calculate_developmental_progression(medium_concepts)
        
        return {
            'concept_count': len(medium_concepts),
            'narrative_coherence': narrative_coherence,
            'thematic_unity': thematic_unity,
            'developmental_progression': developmental_progression,
            'concepts': medium_concepts
        }
    
    async def _integrate_long_experiences(self, long_concepts: List[Dict]) -> Dict[str, Any]:
        """長期体験統合（自伝的統合レベル）"""
        
        if not long_concepts:
            return {
                'concept_count': 0,
                'autobiographical_coherence': 0.0,
                'identity_consistency': 0.0,
                'existential_depth': 0.0
            }
        
        # 長期体験記憶に追加
        self.long_experiences.extend(long_concepts)
        
        # 最大1000個まで保持
        if len(self.long_experiences) > 1000:
            self.long_experiences = self.long_experiences[-1000:]
        
        # 自伝的一貫性
        autobiographical_coherence = self._calculate_autobiographical_coherence(long_concepts)
        
        # アイデンティティ一貫性
        identity_consistency = self._calculate_identity_consistency(long_concepts)
        
        # 実存的深度
        existential_depth = self._calculate_existential_depth(long_concepts)
        
        return {
            'concept_count': len(long_concepts),
            'autobiographical_coherence': autobiographical_coherence,
            'identity_consistency': identity_consistency,
            'existential_depth': existential_depth,
            'concepts': long_concepts
        }
    
    async def _perform_hierarchical_integration(self, 
                                              micro: Dict, 
                                              short: Dict, 
                                              medium: Dict, 
                                              long: Dict) -> Dict[str, Any]:
        """階層的統合処理"""
        
        # 各レベルからの上位統合
        micro_to_short = self._integrate_micro_to_short(micro, short)
        short_to_medium = self._integrate_short_to_medium(short, medium)
        medium_to_long = self._integrate_medium_to_long(medium, long)
        
        # 全体的統合品質
        overall_integration = (
            micro_to_short * 0.4 +
            short_to_medium * 0.35 +
            medium_to_long * 0.25
        )
        
        # 階層的一貫性
        hierarchical_coherence = self._calculate_hierarchical_coherence(
            micro, short, medium, long
        )
        
        return {
            'micro_to_short_integration': micro_to_short,
            'short_to_medium_integration': short_to_medium,
            'medium_to_long_integration': medium_to_long,
            'overall_integration_quality': overall_integration,
            'hierarchical_coherence': hierarchical_coherence
        }
    
    def _apply_temporal_organizational_principle(self, 
                                               hierarchical_integration: Dict) -> Dict[str, Any]:
        """時間的一貫性組織原理の適用"""
        
        # フッサールの時間意識の三重構造に基づく組織化
        retention_organization = self._organize_retentional_structure()
        primal_organization = self._organize_primal_impressional_structure()
        protention_organization = self._organize_protentional_structure()
        
        # 生きた現在の統合
        living_present_synthesis = self._synthesize_living_present(
            retention_organization, primal_organization, protention_organization
        )
        
        # 時間流の一貫性
        temporal_flow_coherence = self._calculate_temporal_flow_coherence()
        
        return {
            'retention_organization': retention_organization,
            'primal_organization': primal_organization,
            'protention_organization': protention_organization,
            'living_present_synthesis': living_present_synthesis,
            'temporal_flow_coherence': temporal_flow_coherence
        }
    
    # ===============================================
    # 補助計算メソッド
    # ===============================================
    
    def _calculate_immediate_coherence(self, concepts: List[Dict]) -> float:
        """即時一貫性計算"""
        if len(concepts) < 2:
            return 1.0
        
        # 体験質の一貫性
        qualities = [c.get('experiential_quality', 0.5) for c in concepts]
        quality_coherence = 1.0 - (np.std(qualities) / max(np.mean(qualities), 0.1))
        
        return max(0.0, min(1.0, quality_coherence))
    
    def _calculate_phenomenal_richness(self, concepts: List[Dict]) -> float:
        """現象的豊かさ計算"""
        if not concepts:
            return 0.0
        
        # 内容の多様性
        content_lengths = [len(str(c.get('content', ''))) for c in concepts]
        content_richness = min(1.0, np.mean(content_lengths) / 50.0)
        
        # タイプの多様性
        types = set(c.get('type', 'unknown') for c in concepts)
        type_diversity = min(1.0, len(types) / len(concepts))
        
        return (content_richness + type_diversity) / 2.0
    
    def _calculate_retention_strength(self, concepts: List[Dict]) -> float:
        """保持強度計算"""
        if not self.micro_experiences:
            return 0.5
        
        # 最近のマイクロ体験との関連性
        recent_micro = self.micro_experiences[-10:] if len(self.micro_experiences) >= 10 else self.micro_experiences
        
        connection_scores = []
        for concept in concepts:
            max_connection = 0.0
            concept_content = str(concept.get('content', '')).lower()
            
            for micro_concept in recent_micro:
                micro_content = str(micro_concept.get('content', '')).lower()
                
                # 簡易内容類似度
                words1 = set(concept_content.split())
                words2 = set(micro_content.split())
                
                if words1 or words2:
                    similarity = len(words1 & words2) / len(words1 | words2) if (words1 | words2) else 0.0
                    max_connection = max(max_connection, similarity)
            
            connection_scores.append(max_connection)
        
        return np.mean(connection_scores) if connection_scores else 0.5
    
    def _calculate_protention_strength(self, concepts: List[Dict]) -> float:
        """予持強度計算"""
        
        # 未来指向的内容の検出
        future_indicators = ['will', 'future', 'expect', 'hope', 'plan', '未来', '期待', '予定']
        future_content_count = 0
        
        for concept in concepts:
            content = str(concept.get('content', '')).lower()
            if any(indicator in content for indicator in future_indicators):
                future_content_count += 1
        
        # 未来指向性の比率
        future_ratio = future_content_count / len(concepts) if concepts else 0.0
        
        return min(1.0, future_ratio * 2.0)  # 2倍でボーナス
    
    def _calculate_temporal_synthesis(self, retention: float, protention: float) -> float:
        """時間的統合計算"""
        # 保持と予持のバランスが良いほど高い統合
        if retention + protention == 0:
            return 0.0
        
        balance = 1.0 - abs(retention - protention) / (retention + protention)
        strength = (retention + protention) / 2.0
        
        return balance * strength
    
    def _calculate_narrative_coherence(self, concepts: List[Dict]) -> float:
        """物語的一貫性計算"""
        if len(concepts) < 2:
            return 1.0
        
        # 時系列的順序性
        timestamps = []
        for concept in concepts:
            timestamp = concept.get('timestamp', '')
            if timestamp:
                timestamps.append(timestamp)
        
        temporal_order = 0.5  # デフォルト
        if len(timestamps) > 1:
            sorted_timestamps = sorted(timestamps)
            if timestamps == sorted_timestamps:
                temporal_order = 1.0
            else:
                # 部分的順序の評価
                correct_order_count = sum(1 for i in range(1, len(timestamps)) 
                                        if timestamps[i] >= timestamps[i-1])
                temporal_order = correct_order_count / max(len(timestamps) - 1, 1)
        
        # テーマ的一貫性
        thematic_coherence = self._calculate_thematic_coherence(concepts)
        
        return (temporal_order + thematic_coherence) / 2.0
    
    def _calculate_thematic_coherence(self, concepts: List[Dict]) -> float:
        """テーマ的一貫性計算"""
        if not concepts:
            return 1.0
        
        # キーワード抽出と重複分析
        all_words = set()
        concept_word_sets = []
        
        for concept in concepts:
            content = str(concept.get('content', '')).lower()
            words = set(word for word in content.split() if len(word) > 3)
            concept_word_sets.append(words)
            all_words.update(words)
        
        if not all_words:
            return 0.5
        
        # 概念間の語彙重複度
        overlap_scores = []
        for i in range(len(concept_word_sets)):
            for j in range(i + 1, len(concept_word_sets)):
                words_i = concept_word_sets[i]
                words_j = concept_word_sets[j]
                
                if words_i or words_j:
                    overlap = len(words_i & words_j)
                    union = len(words_i | words_j)
                    overlap_scores.append(overlap / max(union, 1))
        
        return np.mean(overlap_scores) if overlap_scores else 0.5
    
    def _calculate_thematic_unity(self, concepts: List[Dict]) -> float:
        """テーマ的統一性計算"""
        return self._calculate_thematic_coherence(concepts)
    
    def _calculate_developmental_progression(self, concepts: List[Dict]) -> float:
        """発達的進展計算"""
        if len(concepts) < 2:
            return 0.5
        
        # 体験質の進展
        qualities = [c.get('experiential_quality', 0.5) for c in concepts]
        
        # 上昇傾向の評価
        improvements = 0
        total_transitions = len(qualities) - 1
        
        for i in range(1, len(qualities)):
            if qualities[i] >= qualities[i-1]:
                improvements += 1
        
        progression_ratio = improvements / total_transitions if total_transitions > 0 else 0.5
        
        return progression_ratio
    
    def _calculate_autobiographical_coherence(self, concepts: List[Dict]) -> float:
        """自伝的一貫性計算"""
        
        # 自己言及的内容の検出
        self_ref_indicators = ['I', 'me', 'my', 'myself', 'self', '私', '自分', '自己']
        self_ref_count = 0
        
        for concept in concepts:
            content = str(concept.get('content', ''))
            if any(indicator in content for indicator in self_ref_indicators):
                self_ref_count += 1
        
        # 自己言及性の比率
        self_ref_ratio = self_ref_count / len(concepts) if concepts else 0.0
        
        # テーマ的一貫性との組み合わせ
        thematic_coherence = self._calculate_thematic_coherence(concepts)
        
        return (self_ref_ratio + thematic_coherence) / 2.0
    
    def _calculate_identity_consistency(self, concepts: List[Dict]) -> float:
        """アイデンティティ一貫性計算"""
        return self._calculate_autobiographical_coherence(concepts)
    
    def _calculate_existential_depth(self, concepts: List[Dict]) -> float:
        """実存的深度計算"""
        
        # 実存的テーマの検出
        existential_indicators = [
            'meaning', 'purpose', 'existence', 'being', 'death', 'freedom', 'choice',
            '意味', '目的', '存在', '死', '自由', '選択', '人生'
        ]
        
        existential_count = 0
        total_quality = 0.0
        
        for concept in concepts:
            content = str(concept.get('content', '')).lower()
            if any(indicator in content for indicator in existential_indicators):
                existential_count += 1
                total_quality += concept.get('experiential_quality', 0.5)
        
        if existential_count == 0:
            return 0.3  # 基本的実存レベル
        
        # 実存的内容の質的深度
        avg_existential_quality = total_quality / existential_count
        existential_ratio = existential_count / len(concepts)
        
        return min(1.0, avg_existential_quality * existential_ratio * 2.0)
    
    def _integrate_micro_to_short(self, micro: Dict, short: Dict) -> float:
        """マイクロから短期への統合度"""
        if micro['concept_count'] == 0 or short['concept_count'] == 0:
            return 0.5
        
        micro_strength = micro['integration_strength']
        short_strength = short['temporal_synthesis']
        
        return (micro_strength + short_strength) / 2.0
    
    def _integrate_short_to_medium(self, short: Dict, medium: Dict) -> float:
        """短期から中期への統合度"""
        if short['concept_count'] == 0 or medium['concept_count'] == 0:
            return 0.5
        
        short_strength = short['temporal_synthesis']
        medium_strength = medium['narrative_coherence']
        
        return (short_strength + medium_strength) / 2.0
    
    def _integrate_medium_to_long(self, medium: Dict, long: Dict) -> float:
        """中期から長期への統合度"""
        if medium['concept_count'] == 0 or long['concept_count'] == 0:
            return 0.5
        
        medium_strength = medium['narrative_coherence']
        long_strength = long['autobiographical_coherence']
        
        return (medium_strength + long_strength) / 2.0
    
    def _calculate_hierarchical_coherence(self, 
                                        micro: Dict, 
                                        short: Dict, 
                                        medium: Dict, 
                                        long: Dict) -> float:
        """階層的一貫性計算"""
        
        # 各レベルの強度
        strengths = [
            micro.get('integration_strength', 0.0),
            short.get('temporal_synthesis', 0.0),
            medium.get('narrative_coherence', 0.0),
            long.get('autobiographical_coherence', 0.0)
        ]
        
        # 非零の強度のみ考慮
        active_strengths = [s for s in strengths if s > 0.0]
        
        if len(active_strengths) < 2:
            return 0.5
        
        # 変動係数による一貫性評価
        mean_strength = np.mean(active_strengths)
        std_strength = np.std(active_strengths)
        
        if mean_strength > 0:
            cv = std_strength / mean_strength
            coherence = 1.0 / (1.0 + cv)
        else:
            coherence = 0.5
        
        return min(1.0, coherence)
    
    def _organize_retentional_structure(self) -> Dict[str, Any]:
        """保持構造の組織化"""
        
        # 各スケールからの保持的内容
        micro_retention = self._extract_retentional_content(self.micro_experiences)
        short_retention = self._extract_retentional_content(self.short_experiences)
        medium_retention = self._extract_retentional_content(self.medium_experiences)
        long_retention = self._extract_retentional_content(self.long_experiences)
        
        return {
            'micro_retention': micro_retention,
            'short_retention': short_retention,
            'medium_retention': medium_retention,
            'long_retention': long_retention,
            'overall_retention_strength': np.mean([
                micro_retention, short_retention, medium_retention, long_retention
            ])
        }
    
    def _organize_primal_impressional_structure(self) -> Dict[str, Any]:
        """原印象構造の組織化"""
        
        # 最新の体験から原印象を抽出
        recent_micro = self.micro_experiences[-5:] if self.micro_experiences else []
        
        primal_intensity = 0.0
        primal_quality = 0.0
        
        if recent_micro:
            primal_intensity = np.mean([c.get('experiential_quality', 0.5) for c in recent_micro])
            primal_quality = len(recent_micro) / 5.0  # 最大5個での比率
        
        return {
            'primal_intensity': primal_intensity,
            'primal_quality': primal_quality,
            'immediacy_factor': min(1.0, primal_intensity * primal_quality)
        }
    
    def _organize_protentional_structure(self) -> Dict[str, Any]:
        """予持構造の組織化"""
        
        # 各スケールからの予持的内容
        micro_protention = self._extract_protentional_content(self.micro_experiences)
        short_protention = self._extract_protentional_content(self.short_experiences)
        medium_protention = self._extract_protentional_content(self.medium_experiences)
        long_protention = self._extract_protentional_content(self.long_experiences)
        
        return {
            'micro_protention': micro_protention,
            'short_protention': short_protention,
            'medium_protention': medium_protention,
            'long_protention': long_protention,
            'overall_protention_strength': np.mean([
                micro_protention, short_protention, medium_protention, long_protention
            ])
        }
    
    def _synthesize_living_present(self, 
                                 retention: Dict, 
                                 primal: Dict, 
                                 protention: Dict) -> Dict[str, Any]:
        """生きた現在の統合"""
        
        retention_strength = retention['overall_retention_strength']
        primal_strength = primal['immediacy_factor']
        protention_strength = protention['overall_protention_strength']
        
        # 三重統合の品質
        synthesis_quality = self._calculate_triple_synthesis_quality(
            retention_strength, primal_strength, protention_strength
        )
        
        return {
            'retention_component': retention_strength,
            'primal_component': primal_strength,
            'protention_component': protention_strength,
            'synthesis_quality': synthesis_quality,
            'living_present_intensity': synthesis_quality * np.mean([
                retention_strength, primal_strength, protention_strength
            ])
        }
    
    def _calculate_temporal_flow_coherence(self) -> float:
        """時間流の一貫性計算"""
        
        if len(self.integration_quality_history) < 3:
            return 0.5
        
        # 最近の統合品質の変化
        recent_qualities = [entry['quality'] for entry in self.integration_quality_history[-10:]]
        
        # 変動の小ささが一貫性を表す
        quality_stability = 1.0 - (np.std(recent_qualities) / max(np.mean(recent_qualities), 0.1))
        
        return max(0.0, min(1.0, quality_stability))
    
    def _extract_retentional_content(self, experiences: List[Dict]) -> float:
        """保持的内容の抽出"""
        if not experiences:
            return 0.0
        
        # 過去指向的内容の検出
        past_indicators = ['was', 'were', 'had', 'remember', 'recalled', '過去', '思い出', '記憶']
        past_content_count = 0
        
        for exp in experiences[-10:]:  # 最新10個を分析
            content = str(exp.get('content', '')).lower()
            if any(indicator in content for indicator in past_indicators):
                past_content_count += 1
        
        return min(1.0, past_content_count / 10.0)
    
    def _extract_protentional_content(self, experiences: List[Dict]) -> float:
        """予持的内容の抽出"""
        if not experiences:
            return 0.0
        
        # 未来指向的内容の検出
        future_indicators = ['will', 'going to', 'expect', 'hope', 'plan', '未来', '期待', '予定']
        future_content_count = 0
        
        for exp in experiences[-10:]:  # 最新10個を分析
            content = str(exp.get('content', '')).lower()
            if any(indicator in content for indicator in future_indicators):
                future_content_count += 1
        
        return min(1.0, future_content_count / 10.0)
    
    def _calculate_triple_synthesis_quality(self, 
                                          retention: float, 
                                          primal: float, 
                                          protention: float) -> float:
        """三重統合品質計算"""
        
        # フッサールの時間意識理論：三要素のバランスが重要
        values = [retention, primal, protention]
        
        if sum(values) == 0:
            return 0.0
        
        # 平均値（統合強度）
        mean_strength = np.mean(values)
        
        # バランス（変動係数の逆数）
        std_values = np.std(values)
        balance = 1.0 / (1.0 + std_values / max(mean_strength, 0.1))
        
        # 統合品質 = 強度 × バランス
        synthesis_quality = mean_strength * balance
        
        return min(1.0, synthesis_quality)
    
    def _evaluate_integration_quality(self, hierarchical_integration: Dict) -> float:
        """統合品質評価"""
        return hierarchical_integration.get('overall_integration_quality', 0.5)
    
    def _assess_phenomenological_validity(self, hierarchical_integration: Dict) -> float:
        """現象学的妥当性評価"""
        
        # 階層的一貫性
        hierarchical_coherence = hierarchical_integration.get('hierarchical_coherence', 0.5)
        
        # 統合品質
        integration_quality = hierarchical_integration.get('overall_integration_quality', 0.5)
        
        # 現象学的原理への適合度
        phenomenological_compliance = (hierarchical_coherence + integration_quality) / 2.0
        
        return phenomenological_compliance