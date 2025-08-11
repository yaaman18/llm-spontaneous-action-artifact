"""
体験保持システム理論的検証モジュール
Experience Retention System Theoretical Validation

検証対象システム:
1. RetentionMemory（把持記憶）
2. ProprioceptiveMap（固有感覚マップ）  
3. 質的体験マップ（提案）
4. 意味創出履歴（提案）

現象学的・エナクティブ理論に基づく厳密な評価
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
import logging
from phenomenological_foundations import (
    ExperientialContent, TemporalStructure, 
    IntegratedExperientialMemory, PhenomenologicalValidator
)


class ValidationCriteria(Enum):
    """検証基準"""
    HUSSERLIAN_CONSISTENCY = "フッサール現象学との整合性"
    MERLEAU_PONTY_EMBODIMENT = "メルロ=ポンティ身体性理論との適合性"
    VARELA_STRUCTURAL_COUPLING = "バレラ構造的カップリングとの統合"
    ENACTIVE_CIRCULARITY = "エナクティブ循環因果性の実現"
    FUNCTIONAL_COHERENCE = "機能的一貫性の維持"


@dataclass
class ValidationResult:
    """検証結果構造"""
    criterion: ValidationCriteria
    score: float  # 0.0-1.0
    detailed_analysis: Dict[str, Any]
    theoretical_justification: str
    implementation_recommendations: List[str]


class RetentionMemoryValidator:
    """把持記憶システムの理論的検証"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_retention_structure(self, system: IntegratedExperientialMemory) -> ValidationResult:
        """把持記憶構造の現象学的妥当性検証
        
        フッサール『内的時間意識の現象学』に基づく検証:
        - 把持の時間的階層性
        - 受動的統合の自動性  
        - 時間的フェーディング効果
        - 志向的構造の保持
        """
        
        retention_chain = system.husserlian_retention.retention_chain
        passive_syntheses = system.husserlian_retention.passive_syntheses
        
        # 1. 時間的階層性の検証
        temporal_hierarchy_score = self._evaluate_temporal_hierarchy(retention_chain)
        
        # 2. 受動的統合の検証
        passive_synthesis_score = self._evaluate_passive_synthesis(passive_syntheses)
        
        # 3. 時間的フェーディングの検証  
        temporal_fading_score = self._evaluate_temporal_fading(retention_chain)
        
        # 4. 志向的構造保持の検証
        intentional_preservation_score = self._evaluate_intentional_preservation(retention_chain)
        
        overall_score = np.mean([
            temporal_hierarchy_score,
            passive_synthesis_score, 
            temporal_fading_score,
            intentional_preservation_score
        ])
        
        return ValidationResult(
            criterion=ValidationCriteria.HUSSERLIAN_CONSISTENCY,
            score=overall_score,
            detailed_analysis={
                'temporal_hierarchy': temporal_hierarchy_score,
                'passive_synthesis': passive_synthesis_score,
                'temporal_fading': temporal_fading_score,
                'intentional_preservation': intentional_preservation_score,
                'retention_depth': len(retention_chain),
                'synthesis_clusters': len(passive_syntheses)
            },
            theoretical_justification="""
            フッサール現象学における把持（Retention）は、意識の時間的統合を可能にする
            根本構造です。実装において以下の現象学的要件を満たす必要があります：
            
            1. 把持は「準現在」として現在意識に含まれる
            2. 時間的距離に応じた「明確性の勾配」を持つ
            3. 受動的統合により類似内容の自動的グルーピングが生じる
            4. 志向的構造（ノエマ-ノエシス構造）が保持される
            
            現在の実装は基本的構造を満たしていますが、時間的厚みの
            より精密な現象学的実装が必要です。
            """,
            implementation_recommendations=[
                "時間的距離による明確性勾配の精密化",
                "受動的統合の類型別実装（類似性・対比性・因果性）",
                "志向的構造の詳細保持メカニズム強化",
                "把持の「準現在」性質の機能的表現"
            ]
        )
    
    def _evaluate_temporal_hierarchy(self, retention_chain: List[ExperientialContent]) -> float:
        """時間的階層性の評価"""
        if len(retention_chain) < 2:
            return 0.0
        
        # 時間的厚みの単調減少性チェック
        thickness_values = [content.temporal_thickness for content in retention_chain]
        is_monotonic = all(thickness_values[i] >= thickness_values[i+1] 
                          for i in range(len(thickness_values)-1))
        
        # 階層の深さ評価
        depth_score = min(1.0, len(retention_chain) / 10.0)  # 理想的な把持深度は10程度
        
        return 0.7 * (1.0 if is_monotonic else 0.3) + 0.3 * depth_score
    
    def _evaluate_passive_synthesis(self, passive_syntheses: Dict[str, List[ExperientialContent]]) -> float:
        """受動的統合の評価"""
        if not passive_syntheses:
            return 0.0
        
        # クラスター形成の質的評価
        cluster_quality = 0.0
        for key, contents in passive_syntheses.items():
            if len(contents) > 1:  # 複数要素のクラスターのみ評価
                cluster_quality += len(contents) / 10.0  # 理想的クラスターサイズで正規化
        
        cluster_count_score = min(1.0, len(passive_syntheses) / 5.0)  # 理想的クラスター数
        average_cluster_quality = cluster_quality / len(passive_syntheses) if passive_syntheses else 0.0
        
        return 0.6 * cluster_count_score + 0.4 * min(1.0, average_cluster_quality)
    
    def _evaluate_temporal_fading(self, retention_chain: List[ExperientialContent]) -> float:
        """時間的フェーディングの評価"""
        if len(retention_chain) < 2:
            return 0.0
        
        # フェーディング効果の確認
        initial_thickness = retention_chain[0].temporal_thickness if retention_chain else 0.0
        final_thickness = retention_chain[-1].temporal_thickness if retention_chain else 0.0
        
        if initial_thickness <= 0:
            return 0.0
        
        fading_ratio = final_thickness / initial_thickness
        # 理想的なフェーディングは0.1-0.5の範囲
        if 0.1 <= fading_ratio <= 0.5:
            return 1.0
        elif fading_ratio < 0.1:
            return 0.7  # 強すぎるフェーディング
        else:
            return 0.3  # 不十分なフェーディング
    
    def _evaluate_intentional_preservation(self, retention_chain: List[ExperientialContent]) -> float:
        """志向的構造保持の評価"""
        if not retention_chain:
            return 0.0
        
        # 志向的内容の保持率
        content_preservation = 0.0
        for content in retention_chain:
            if content.intentional_content and len(content.intentional_content) > 0:
                content_preservation += 1.0
        
        return content_preservation / len(retention_chain) if retention_chain else 0.0


class ProprioceptiveMapValidator:
    """固有感覚マップの理論的検証"""
    
    def validate_bodily_schema(self, system: IntegratedExperientialMemory) -> ValidationResult:
        """メルロ=ポンティ身体図式理論に基づく検証
        
        『知覚の現象学』での身体図式概念:
        - 身体的習慣の堆積構造
        - 運動的志向性の保持
        - 固有感覚の統合的組織化
        - 触覚的記憶の空間的配置
        """
        
        bodily_memory = system.bodily_memory
        
        # 1. 運動習慣の形成評価
        motor_habits_score = self._evaluate_motor_habits(bodily_memory.motor_habits)
        
        # 2. 固有感覚図式の評価  
        proprioceptive_score = self._evaluate_proprioceptive_schema(bodily_memory.proprioceptive_schema)
        
        # 3. 身体的記憶統合の評価
        integration_score = self._evaluate_bodily_integration(bodily_memory)
        
        # 4. 空間的組織化の評価
        spatial_organization_score = self._evaluate_spatial_organization(bodily_memory.proprioceptive_schema)
        
        overall_score = np.mean([
            motor_habits_score,
            proprioceptive_score,
            integration_score,
            spatial_organization_score
        ])
        
        return ValidationResult(
            criterion=ValidationCriteria.MERLEAU_PONTY_EMBODIMENT,
            score=overall_score,
            detailed_analysis={
                'motor_habits_formation': motor_habits_score,
                'proprioceptive_integration': proprioceptive_score,
                'bodily_memory_integration': integration_score,
                'spatial_organization': spatial_organization_score,
                'habit_count': len(bodily_memory.motor_habits),
                'schema_activation_level': np.mean(bodily_memory.proprioceptive_schema)
            },
            theoretical_justification="""
            メルロ=ポンティの身体現象学では、身体図式は「運動的志向性」の
            基盤として機能します。身体は単なる物体ではなく、世界への
            志向的な開かれを持つ「生きられた身体」です。
            
            固有感覚マップは以下の現象学的要件を満たす必要があります：
            1. 習慣的運動パターンの「堆積的記憶」
            2. 身体部位間の「統合的組織化」  
            3. 空間的志向性の「図式的保持」
            4. 触覚的記憶の「身体的定位」
            
            実装において運動習慣と固有感覚の統合メカニズムの
            さらなる現象学的精密化が必要です。
            """,
            implementation_recommendations=[
                "運動的志向性の方向性保持メカニズム強化",
                "身体部位間の統合的関連性の詳細実装",
                "習慣的運動パターンの階層的組織化",
                "触覚的記憶の空間的配置最適化"
            ]
        )
    
    def _evaluate_motor_habits(self, motor_habits: Dict[str, float]) -> float:
        """運動習慣の評価"""
        if not motor_habits:
            return 0.0
        
        # 習慣強度の分布評価
        habit_values = list(motor_habits.values())
        habit_diversity = len(set(np.round(habit_values, 1))) / len(habit_values) if habit_values else 0.0
        habit_strength = np.mean(habit_values) if habit_values else 0.0
        
        return 0.6 * min(1.0, habit_strength) + 0.4 * habit_diversity
    
    def _evaluate_proprioceptive_schema(self, schema: np.ndarray) -> float:
        """固有感覚図式の評価"""
        if schema.size == 0:
            return 0.0
        
        # 図式の活性化度合い
        activation_level = np.mean(schema)
        
        # 図式の構造化度（局所的パターンの存在）
        local_variance = np.mean([
            np.var(schema[i:i+5, j:j+5]) 
            for i in range(0, schema.shape[0]-5, 5)
            for j in range(0, schema.shape[1]-5, 5)
        ])
        
        structure_score = min(1.0, local_variance / 0.1)  # 正規化
        activation_score = min(1.0, activation_level / 0.5)  # 理想的活性化レベル
        
        return 0.7 * activation_score + 0.3 * structure_score
    
    def _evaluate_bodily_integration(self, bodily_memory) -> float:
        """身体的記憶統合の評価"""
        # 運動習慣と触覚記憶の統合度
        motor_count = len(bodily_memory.motor_habits)
        tactile_count = len(bodily_memory.tactile_memory)
        
        if motor_count == 0 and tactile_count == 0:
            return 0.0
        
        # バランスの取れた統合を評価
        balance_score = 1.0 - abs(motor_count - tactile_count) / (motor_count + tactile_count + 1)
        coverage_score = min(1.0, (motor_count + tactile_count) / 20.0)  # 理想的カバレッジ
        
        return 0.6 * balance_score + 0.4 * coverage_score
    
    def _evaluate_spatial_organization(self, schema: np.ndarray) -> float:
        """空間的組織化の評価"""
        if schema.size == 0:
            return 0.0
        
        # 空間的連続性の評価（隣接する領域の相関）
        correlations = []
        for i in range(schema.shape[0] - 1):
            for j in range(schema.shape[1] - 1):
                local_patch = schema[i:i+2, j:j+2].flatten()
                if len(set(local_patch)) > 1:  # 変化がある場合のみ
                    correlations.append(np.corrcoef(local_patch[:-1], local_patch[1:])[0,1])
        
        if not correlations:
            return 0.5  # デフォルト値
        
        # NaNを除去して平均相関を計算
        valid_correlations = [c for c in correlations if not np.isnan(c)]
        return np.mean(valid_correlations) if valid_correlations else 0.5


class QualitativeExperienceMapValidator:
    """質的体験マップの理論的検証（提案システム）"""
    
    def validate_qualitative_structure(self, system: IntegratedExperientialMemory) -> ValidationResult:
        """質的体験の現象学的構造検証
        
        注意: クオリア問題は回避し、機能的・行動的側面に焦点
        - 体験内容の質的差異の機能的表現
        - 感情的共鳴の構造的保持
        - 体験の「質的厚み」の時間的変化
        - 質的関連性による連想構造
        """
        
        # 既存システムから質的側面を抽出・評価
        retention_chain = system.husserlian_retention.retention_chain
        
        # 1. 質的差異の機能的表現評価
        qualitative_differentiation_score = self._evaluate_qualitative_differentiation(retention_chain)
        
        # 2. 感情的共鳴構造の評価
        emotional_resonance_score = self._evaluate_emotional_resonance_structure(retention_chain)
        
        # 3. 質的厚みの時間的変化評価
        qualitative_temporal_thickness_score = self._evaluate_qualitative_temporal_thickness(retention_chain)
        
        # 4. 質的連想構造の評価
        qualitative_association_score = self._evaluate_qualitative_associations(system)
        
        overall_score = np.mean([
            qualitative_differentiation_score,
            emotional_resonance_score,
            qualitative_temporal_thickness_score,
            qualitative_association_score
        ])
        
        return ValidationResult(
            criterion=ValidationCriteria.FUNCTIONAL_COHERENCE,
            score=overall_score,
            detailed_analysis={
                'qualitative_differentiation': qualitative_differentiation_score,
                'emotional_resonance_structure': emotional_resonance_score,
                'temporal_thickness_quality': qualitative_temporal_thickness_score,
                'qualitative_associations': qualitative_association_score,
                'experience_diversity': self._calculate_experience_diversity(retention_chain)
            },
            theoretical_justification="""
            質的体験マップは、クオリア問題を回避しつつ、体験の質的側面の
            機能的・構造的特徴を保持することを目的とします。
            
            現象学的根拠：
            1. フッサールの「質的差異」概念：体験内容の区別可能性
            2. 感情的共鳴：メルロ=ポンティの「感情的身体」概念
            3. 時間的質感：体験の「濃度」や「密度」の現象学的記述
            4. 質的連想：類似する質感による記憶の呼び起こし
            
            実装においては、質的側面を定量的指標に還元せず、
            構造的・関係的特徴として保持することが重要です。
            """,
            implementation_recommendations=[
                "質的差異の多次元的表現システム構築",
                "感情的共鳴パターンの詳細分類",
                "質的厚みの非線形時間変化モデル導入",
                "質的連想の現象学的類型化実装"
            ]
        )
    
    def _evaluate_qualitative_differentiation(self, retention_chain: List[ExperientialContent]) -> float:
        """質的差異の機能的表現評価"""
        if len(retention_chain) < 2:
            return 0.0
        
        # 体験内容の多様性評価（志向的内容の構造的差異）
        content_signatures = []
        for content in retention_chain:
            signature = tuple(sorted(content.intentional_content.keys())) if content.intentional_content else ()
            content_signatures.append(signature)
        
        unique_signatures = len(set(content_signatures))
        diversity_score = unique_signatures / len(retention_chain) if retention_chain else 0.0
        
        # 身体共鳴の差異評価
        resonance_values = [content.bodily_resonance for content in retention_chain]
        resonance_variance = np.var(resonance_values) if resonance_values else 0.0
        variance_score = min(1.0, resonance_variance / 0.25)  # 正規化
        
        return 0.6 * diversity_score + 0.4 * variance_score
    
    def _evaluate_emotional_resonance_structure(self, retention_chain: List[ExperientialContent]) -> float:
        """感情的共鳴構造の評価"""
        if not retention_chain:
            return 0.0
        
        # 身体共鳴値の分布と構造的パターン評価
        resonance_values = [content.bodily_resonance for content in retention_chain]
        
        # 共鳴レベルの階層性評価
        high_resonance = sum(1 for r in resonance_values if r > 0.7)
        medium_resonance = sum(1 for r in resonance_values if 0.3 <= r <= 0.7)
        low_resonance = sum(1 for r in resonance_values if r < 0.3)
        
        total = len(resonance_values)
        hierarchy_balance = 1.0 - abs(high_resonance/total - 0.3) - abs(medium_resonance/total - 0.4) - abs(low_resonance/total - 0.3)
        
        # 共鳴の時間的連続性評価
        continuity_score = self._evaluate_resonance_continuity(resonance_values)
        
        return 0.6 * max(0.0, hierarchy_balance) + 0.4 * continuity_score
    
    def _evaluate_qualitative_temporal_thickness(self, retention_chain: List[ExperientialContent]) -> float:
        """質的厚みの時間的変化評価"""
        if len(retention_chain) < 2:
            return 0.0
        
        # 時間的厚みの質的変化パターン
        thickness_values = [content.temporal_thickness for content in retention_chain]
        
        # 非線形減衰パターンの評価（現象学的により適切）
        expected_pattern = [1.0 * (0.9 ** i) for i in range(len(thickness_values))]
        actual_pattern = thickness_values
        
        pattern_similarity = 1.0 - np.mean([
            abs(actual - expected) for actual, expected in zip(actual_pattern, expected_pattern)
        ])
        
        return max(0.0, pattern_similarity)
    
    def _evaluate_qualitative_associations(self, system: IntegratedExperientialMemory) -> float:
        """質的連想構造の評価"""
        # 受動的統合における質的類似性の評価
        passive_syntheses = system.husserlian_retention.passive_syntheses
        
        if not passive_syntheses:
            return 0.0
        
        # 質的類似性に基づくクラスター形成の評価
        qualitative_clusters = 0
        for key, contents in passive_syntheses.items():
            if len(contents) > 1:
                # 身体共鳴の類似性でクラスターの質的妥当性を評価
                resonances = [c.bodily_resonance for c in contents]
                if len(set(np.round(resonances, 1))) <= len(resonances) // 2:  # 類似する共鳴値
                    qualitative_clusters += 1
        
        return min(1.0, qualitative_clusters / max(1, len(passive_syntheses) // 2))
    
    def _calculate_experience_diversity(self, retention_chain: List[ExperientialContent]) -> float:
        """体験多様性の計算"""
        if not retention_chain:
            return 0.0
        
        # 複数の次元での多様性評価
        temporal_phases = [content.temporal_phase for content in retention_chain]
        habit_layers = [content.habit_layer for content in retention_chain]
        
        phase_diversity = len(set(temporal_phases)) / len(TemporalStructure)
        layer_diversity = len(set(habit_layers)) / len(retention_chain) if retention_chain else 0.0
        
        return (phase_diversity + layer_diversity) / 2.0
    
    def _evaluate_resonance_continuity(self, resonance_values: List[float]) -> float:
        """共鳴の時間的連続性評価"""
        if len(resonance_values) < 2:
            return 0.0
        
        # 隣接する体験間の共鳴値の滑らかな変化を評価
        differences = [abs(resonance_values[i] - resonance_values[i+1]) 
                      for i in range(len(resonance_values) - 1)]
        
        # 急激な変化が少ない場合に高スコア
        smooth_changes = sum(1 for diff in differences if diff < 0.3)
        return smooth_changes / len(differences) if differences else 0.0


class MeaningCreationHistoryValidator:
    """意味創出履歴の理論的検証（提案システム）"""
    
    def validate_meaning_creation_structure(self, system: IntegratedExperientialMemory) -> ValidationResult:
        """意味創出プロセスの現象学的・エナクティブ検証
        
        理論的基盤：
        - バレラの「意味創出（Meaning-making）」概念
        - エナクティブ認知における循環因果性
        - 意味の創発的性質と歴史性
        - 構造的カップリングによる意味生成
        """
        
        enactive_memory = system.enactive_memory
        
        # 1. 意味創出の循環因果性評価
        circular_causality_score = self._evaluate_circular_causality(enactive_memory)
        
        # 2. 意味の創発的生成評価
        emergent_meaning_score = self._evaluate_emergent_meaning_generation(enactive_memory)
        
        # 3. 構造的カップリングの歴史性評価
        coupling_historicity_score = self._evaluate_coupling_historicity(enactive_memory)
        
        # 4. 意味の堆積的蓄積評価
        meaning_sedimentation_score = self._evaluate_meaning_sedimentation(enactive_memory)
        
        overall_score = np.mean([
            circular_causality_score,
            emergent_meaning_score,
            coupling_historicity_score,
            meaning_sedimentation_score
        ])
        
        return ValidationResult(
            criterion=ValidationCriteria.VARELA_STRUCTURAL_COUPLING,
            score=overall_score,
            detailed_analysis={
                'circular_causality': circular_causality_score,
                'emergent_meaning': emergent_meaning_score,
                'coupling_historicity': coupling_historicity_score,
                'meaning_sedimentation': meaning_sedimentation_score,
                'coupling_events': len(enactive_memory.structural_coupling_history),
                'autopoietic_patterns': len(enactive_memory.autopoietic_patterns)
            },
            theoretical_justification="""
            バレラのエナクティブ理論では、意味は主体と環境の構造的カップリングを
            通じて創発的に生成されます。意味創出履歴システムは以下の要件を満たす必要があります：
            
            1. 循環因果性：主体の行為が環境を変化させ、変化した環境が主体に影響を与える
            2. 創発性：個別の相互作用から予測できない新しい意味パターンの出現  
            3. 歴史性：過去の相互作用履歴が現在の意味創出に影響を与える
            4. 堆積性：反復される意味パターンの「習慣化」による定着
            
            実装において、単なる情報処理を超えた真の意味創出の
            循環的・創発的性質の実現が課題です。
            """,
            implementation_recommendations=[
                "循環因果性の明示的モデリング強化",
                "創発的パターン検出アルゴリズムの精密化",
                "歴史的文脈の意味生成への統合",
                "習慣化による意味の堆積メカニズム実装"
            ]
        )
    
    def _evaluate_circular_causality(self, enactive_memory) -> float:
        """循環因果性の評価"""
        coupling_history = enactive_memory.structural_coupling_history
        circular_traces = enactive_memory.circular_causality_traces
        
        if len(coupling_history) < 2:
            return 0.0
        
        # 環境への作用とその結果の循環的関係を評価
        circular_events = 0
        for i in range(len(coupling_history) - 1):
            current_response = coupling_history[i].get('response', {})
            next_environment = coupling_history[i + 1].get('environment', {})
            
            # 応答が次の環境状態に影響を与えているかチェック
            if self._detect_response_to_environment_influence(current_response, next_environment):
                circular_events += 1
        
        return circular_events / max(1, len(coupling_history) - 1)
    
    def _evaluate_emergent_meaning_generation(self, enactive_memory) -> float:
        """創発的意味生成の評価"""
        autopoietic_patterns = enactive_memory.autopoietic_patterns
        
        if not autopoietic_patterns:
            return 0.0
        
        # パターンの複雑性と新規性を評価
        pattern_complexity = 0
        for pattern_key, pattern_values in autopoietic_patterns.items():
            if len(pattern_values) > 3:  # 十分なデータがある場合
                # パターンの変化傾向（新しい意味の創発を示唆）
                recent_values = pattern_values[-3:]
                early_values = pattern_values[:3] if len(pattern_values) >= 6 else pattern_values[:-3]
                
                if early_values and recent_values:
                    trend_change = abs(np.mean(recent_values) - np.mean(early_values))
                    pattern_complexity += min(1.0, trend_change)
        
        return pattern_complexity / len(autopoietic_patterns) if autopoietic_patterns else 0.0
    
    def _evaluate_coupling_historicity(self, enactive_memory) -> float:
        """構造的カップリングの歴史性評価"""
        coupling_history = enactive_memory.structural_coupling_history
        
        if len(coupling_history) < 3:
            return 0.0
        
        # 過去の経験が現在の相互作用に与える影響を評価
        historical_influence = 0
        for i in range(2, len(coupling_history)):
            current_coupling = coupling_history[i]
            past_couplings = coupling_history[:i]
            
            # 類似する過去の状況とその影響を検出
            similar_past_events = [
                past for past in past_couplings 
                if self._calculate_environmental_similarity(
                    current_coupling.get('environment', {}),
                    past.get('environment', {})
                ) > 0.7
            ]
            
            if similar_past_events:
                historical_influence += 1
        
        return historical_influence / max(1, len(coupling_history) - 2)
    
    def _evaluate_meaning_sedimentation(self, enactive_memory) -> float:
        """意味の堆積的蓄積評価"""
        autopoietic_patterns = enactive_memory.autopoietic_patterns
        
        if not autopoietic_patterns:
            return 0.0
        
        # 反復による意味パターンの定着度評価
        sedimented_patterns = 0
        for pattern_key, pattern_values in autopoietic_patterns.items():
            if len(pattern_values) >= 5:
                # パターンの安定性（堆積度）を評価
                stability = 1.0 - np.var(pattern_values) / (np.mean(pattern_values) + 1e-6)
                if stability > 0.7:  # 安定したパターン
                    sedimented_patterns += 1
        
        return sedimented_patterns / len(autopoietic_patterns) if autopoietic_patterns else 0.0
    
    def _detect_response_to_environment_influence(self, response: Dict[str, Any], 
                                                 next_environment: Dict[str, Any]) -> bool:
        """応答の環境への影響検出"""
        # 簡単な実装：共通キーでの値の変化を検出
        common_keys = set(response.keys()) & set(next_environment.keys())
        
        for key in common_keys:
            resp_val = response[key]
            env_val = next_environment[key]
            
            if isinstance(resp_val, (int, float)) and isinstance(env_val, (int, float)):
                if abs(resp_val - env_val) < 0.1:  # 影響を示唆する類似性
                    return True
        
        return False
    
    def _calculate_environmental_similarity(self, env1: Dict[str, Any], env2: Dict[str, Any]) -> float:
        """環境状態の類似性計算（再利用）"""
        common_keys = set(env1.keys()) & set(env2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            if isinstance(env1[key], (int, float)) and isinstance(env2[key], (int, float)):
                sim = 1.0 - abs(env1[key] - env2[key]) / (abs(env1[key]) + abs(env2[key]) + 1e-6)
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0


class ComprehensiveSystemValidator:
    """統合的システム検証"""
    
    def __init__(self):
        self.retention_validator = RetentionMemoryValidator()
        self.proprioceptive_validator = ProprioceptiveMapValidator()
        self.qualitative_validator = QualitativeExperienceMapValidator()
        self.meaning_validator = MeaningCreationHistoryValidator()
    
    def perform_comprehensive_validation(self, system: IntegratedExperientialMemory) -> Dict[str, ValidationResult]:
        """包括的システム検証の実行"""
        
        results = {}
        
        # 各サブシステムの検証
        results['retention_memory'] = self.retention_validator.validate_retention_structure(system)
        results['proprioceptive_map'] = self.proprioceptive_validator.validate_bodily_schema(system)
        results['qualitative_experience'] = self.qualitative_validator.validate_qualitative_structure(system)
        results['meaning_creation'] = self.meaning_validator.validate_meaning_creation_structure(system)
        
        return results
    
    def generate_comprehensive_report(self, validation_results: Dict[str, ValidationResult]) -> str:
        """包括的検証レポートの生成"""
        
        report = "=== 体験的感覚保持システム：理論的妥当性検証レポート ===\n"
        report += "Experience Retention System: Theoretical Validity Validation Report\n\n"
        
        overall_scores = []
        
        for subsystem, result in validation_results.items():
            overall_scores.append(result.score)
            
            report += f"## {subsystem.upper()}\n"
            report += f"総合スコア: {result.score:.3f}/1.000\n"
            report += f"検証基準: {result.criterion.value}\n\n"
            
            report += "詳細分析:\n"
            for metric, value in result.detailed_analysis.items():
                if isinstance(value, float):
                    report += f"  - {metric}: {value:.3f}\n"
                else:
                    report += f"  - {metric}: {value}\n"
            
            report += f"\n理論的根拠:\n{result.theoretical_justification}\n"
            
            report += "実装推奨事項:\n"
            for recommendation in result.implementation_recommendations:
                report += f"  • {recommendation}\n"
            
            report += "\n" + "="*60 + "\n\n"
        
        # 全体評価
        overall_score = np.mean(overall_scores)
        report += f"## 全体的理論的妥当性\n"
        report += f"総合評価スコア: {overall_score:.3f}/1.000\n\n"
        
        if overall_score >= 0.8:
            report += "評価: 優秀 - 現象学的・エナクティブ理論との高い整合性\n"
        elif overall_score >= 0.6:
            report += "評価: 良好 - 基本的理論要件を満たし、改善の余地あり\n"
        elif overall_score >= 0.4:
            report += "評価: 要改善 - 重要な理論的課題の解決が必要\n"
        else:
            report += "評価: 不十分 - 根本的な理論的再設計が必要\n"
        
        # 最重要課題の特定
        lowest_score_subsystem = min(validation_results.keys(), 
                                   key=lambda k: validation_results[k].score)
        report += f"\n最優先改善領域: {lowest_score_subsystem}\n"
        report += f"スコア: {validation_results[lowest_score_subsystem].score:.3f}\n"
        
        return report


if __name__ == "__main__":
    # システム検証のデモンストレーション
    print("体験的感覚保持システムの理論的妥当性検証を開始します...")
    
    # テストシステムの構築
    system = IntegratedExperientialMemory()
    
    # テストデータの追加
    for i in range(5):
        test_content = ExperientialContent(
            temporal_phase=TemporalStructure.RETENTION,
            intentional_content={
                'sensory_input': np.random.random(),
                'motor_pattern': {'action': np.random.random()},
                'environment_state': {'parameter': np.random.random()}
            },
            bodily_resonance=np.random.random(),
            temporal_thickness=1.0 - i * 0.1,
            associative_potential=['sensory', 'motor'],
            habit_layer='passive' if i % 2 == 0 else 'active'
        )
        system.retain_experience(test_content)
    
    # 包括的検証の実行
    validator = ComprehensiveSystemValidator()
    results = validator.perform_comprehensive_validation(system)
    
    # レポート生成と出力
    report = validator.generate_comprehensive_report(results)
    print(report)