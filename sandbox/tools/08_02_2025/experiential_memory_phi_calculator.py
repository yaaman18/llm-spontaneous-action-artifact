"""
Experiential Memory Phi Calculator - 体験記憶特化φ値計算器
IIT Integration Master による体験記憶システム専用の実用的φ値実装

設計哲学:
- IIT4の5つの公理を体験記憶の性質に適応
- 理論的厳密性と実用的感度のバランス
- 概念数と質的豊かさを適切に反映
- 発達段階移行を可能にする感度設計

体験記憶の特性:
1. 存在: 体験概念の活性化パターン  
2. 内在性: 自己参照的性質
3. 情報: 体験の質的差異化
4. 統合: 体験間の不可分な関係
5. 排他性: 最大φ値による境界確定
"""

import numpy as np
import asyncio
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExperientialAxiom(Enum):
    """体験記憶用IIT公理"""
    EXISTENCE = "体験存在"      # 体験概念の活性化
    INTRINSIC = "体験内在性"    # 自己参照的体験
    INFORMATION = "体験情報"    # 質的差異化
    INTEGRATION = "体験統合"    # 不可分な体験関係
    EXCLUSION = "体験排他性"    # 最大φ境界


@dataclass
class ExperientialPhiResult:
    """体験記憶φ計算結果"""
    phi_value: float
    concept_count: int
    integration_quality: float
    experiential_purity: float
    temporal_depth: float
    self_reference_strength: float
    consciousness_level: float
    development_stage_prediction: str
    
    # 詳細メトリクス
    existence_score: float = 0.0
    intrinsic_score: float = 0.0
    information_score: float = 0.0
    integration_score: float = 0.0
    exclusion_score: float = 0.0
    
    # 実行統計
    calculation_time: float = 0.0
    complexity_level: str = "LOW"


class ExperientialMemoryPhiCalculator:
    """
    体験記憶特化φ値計算器
    
    理論的基盤:
    - IIT4の公理を体験記憶の特性に適応
    - 複雑なKL発散を体験間関係度に簡略化
    - TPM構築を体験概念活性化マップに変更
    - 発達段階促進のための感度調整
    """
    
    def __init__(self, sensitivity_factor: float = 2.0):
        """
        Args:
            sensitivity_factor: 感度係数（発達促進用）
        """
        self.sensitivity_factor = sensitivity_factor
        self.calculation_history = []
        
        # 発達段階閾値（実用的調整済み）
        self.stage_thresholds = {
            'STAGE_0_PRE_CONSCIOUS': 0.0,
            'STAGE_1_EXPERIENTIAL_EMERGENCE': 0.1,    # 0.5 → 0.1に調整
            'STAGE_2_TEMPORAL_INTEGRATION': 0.5,      # 2.0 → 0.5に調整
            'STAGE_3_RELATIONAL_FORMATION': 2.0,      # 8.0 → 2.0に調整
            'STAGE_4_SELF_ESTABLISHMENT': 8.0,        # 30.0 → 8.0に調整
            'STAGE_5_REFLECTIVE_OPERATION': 25.0,     # 100.0 → 25.0に調整
            'STAGE_6_NARRATIVE_INTEGRATION': 75.0     # 300.0 → 75.0に調整
        }
        
        logger.info(f"体験記憶φ計算器初期化 (感度係数: {sensitivity_factor})")
    
    async def calculate_experiential_phi(self, experiential_concepts: List[Dict]) -> ExperientialPhiResult:
        """
        体験記憶からφ値を計算
        
        Args:
            experiential_concepts: 体験概念リスト
            
        Returns:
            ExperientialPhiResult: 包括的φ計算結果
        """
        start_time = time.time()
        
        if not experiential_concepts:
            return self._create_empty_result(start_time)
        
        # IIT公理に基づく各スコア計算
        existence_score = self._calculate_existence_axiom(experiential_concepts)
        intrinsic_score = self._calculate_intrinsic_axiom(experiential_concepts)
        information_score = self._calculate_information_axiom(experiential_concepts)
        integration_score = await self._calculate_integration_axiom(experiential_concepts)
        exclusion_score = self._calculate_exclusion_axiom(experiential_concepts)
        
        # 体験記憶特化メトリクス
        experiential_metrics = self._calculate_experiential_metrics(experiential_concepts)
        
        # φ値統合計算
        phi_value = self._integrate_phi_components(
            existence_score, intrinsic_score, information_score,
            integration_score, exclusion_score, experiential_metrics
        )
        
        # 発達段階予測
        stage_prediction = self._predict_development_stage(phi_value, experiential_metrics)
        
        # 意識レベル計算
        consciousness_level = self._calculate_consciousness_level(phi_value, experiential_metrics)
        
        calculation_time = time.time() - start_time
        complexity_level = self._determine_complexity_level(len(experiential_concepts))
        
        result = ExperientialPhiResult(
            phi_value=phi_value,
            concept_count=len(experiential_concepts),
            integration_quality=experiential_metrics['integration_quality'],
            experiential_purity=experiential_metrics['experiential_purity'],
            temporal_depth=experiential_metrics['temporal_depth'],
            self_reference_strength=experiential_metrics['self_reference_strength'],
            consciousness_level=consciousness_level,
            development_stage_prediction=stage_prediction,
            existence_score=existence_score,
            intrinsic_score=intrinsic_score,
            information_score=information_score,
            integration_score=integration_score,
            exclusion_score=exclusion_score,
            calculation_time=calculation_time,
            complexity_level=complexity_level
        )
        
        # 履歴記録
        self.calculation_history.append({
            'timestamp': start_time,
            'result': result,
            'concept_count': len(experiential_concepts)
        })
        
        # 履歴サイズ制限
        if len(self.calculation_history) > 50:
            self.calculation_history = self.calculation_history[-50:]
        
        logger.info(f"φ計算完了: φ={phi_value:.6f}, 概念数={len(experiential_concepts)}, "
                   f"時間={calculation_time:.3f}秒, 段階={stage_prediction}")
        
        return result
    
    def _create_empty_result(self, start_time: float) -> ExperientialPhiResult:
        """空の結果を生成"""
        return ExperientialPhiResult(
            phi_value=0.0,
            concept_count=0,
            integration_quality=0.0,
            experiential_purity=1.0,
            temporal_depth=0.0,
            self_reference_strength=0.0,
            consciousness_level=0.0,
            development_stage_prediction='STAGE_0_PRE_CONSCIOUS',
            calculation_time=time.time() - start_time,
            complexity_level="EMPTY"
        )
    
    def _calculate_existence_axiom(self, concepts: List[Dict]) -> float:
        """
        存在公理: 体験概念の活性化パターン
        A experience exists - 体験が存在する
        """
        if not concepts:
            return 0.0
        
        # 活性化レベルの計算
        activation_levels = []
        for concept in concepts:
            # 体験質 × 一貫性 で基本活性化
            quality = concept.get('experiential_quality', 0.5)
            coherence = concept.get('coherence', 0.5)
            temporal_depth = concept.get('temporal_depth', 1)
            
            # 時間深度による強化（体験記憶の蓄積効果）
            temporal_boost = min(math.log(temporal_depth + 1) / 3.0, 1.0)
            
            # コンテンツの豊かさ
            content_richness = min(len(str(concept.get('content', ''))) / 50.0, 2.0)
            
            activation = quality * coherence * (1.0 + temporal_boost) * (1.0 + content_richness * 0.5)
            activation_levels.append(activation)
        
        # 集合的存在スコア
        mean_activation = np.mean(activation_levels)
        activation_consistency = 1.0 - np.std(activation_levels) / max(mean_activation, 0.1)
        concept_count_factor = min(math.log(len(concepts) + 1) / 5.0, 1.0)
        
        existence_score = mean_activation * activation_consistency * (1.0 + concept_count_factor)
        
        return min(2.0, existence_score * self.sensitivity_factor)
    
    def _calculate_intrinsic_axiom(self, concepts: List[Dict]) -> float:
        """
        内在性公理: 自己参照的体験特性
        An experience exists intrinsically - 体験は内在的に存在する
        """
        if not concepts:
            return 0.0
        
        # 自己参照指標
        self_ref_indicators = ['I', 'me', 'my', 'myself', 'self', '私', '自分', '自己', '体験']
        self_ref_count = 0
        quality_sum = 0.0
        
        for concept in concepts:
            content = str(concept.get('content', '')).lower()
            
            # 自己参照の検出
            if any(indicator in content for indicator in self_ref_indicators):
                self_ref_count += 1
            
            # 内在的体験の質
            quality = concept.get('experiential_quality', 0.5)
            quality_sum += quality
        
        # 自己参照率
        self_ref_ratio = self_ref_count / len(concepts)
        
        # 平均体験質
        mean_quality = quality_sum / len(concepts)
        
        # 内在性の深度（体験の自己完結性）
        intrinsic_depth = self_ref_ratio * mean_quality
        
        # 体験記憶の内在的統合性
        concept_diversity = len(set(str(c.get('type', '')) for c in concepts))
        diversity_factor = min(concept_diversity / max(len(concepts), 1), 1.0)
        
        intrinsic_score = intrinsic_depth * (1.0 + diversity_factor)
        
        return min(2.0, intrinsic_score * self.sensitivity_factor * 1.5)  # 内在性を重視
    
    def _calculate_information_axiom(self, concepts: List[Dict]) -> float:
        """
        情報公理: 体験の質的差異化
        An experience is specific - 体験は特定的である
        """
        if not concepts:
            return 0.0
        
        # 体験の質的多様性
        quality_values = [c.get('experiential_quality', 0.5) for c in concepts]
        coherence_values = [c.get('coherence', 0.5) for c in concepts]
        
        # 質的差異の計算
        quality_variance = np.var(quality_values) if len(quality_values) > 1 else 0.0
        coherence_variance = np.var(coherence_values) if len(coherence_values) > 1 else 0.0
        
        # コンテンツの情報豊富性
        content_lengths = [len(str(c.get('content', ''))) for c in concepts]
        content_diversity = np.std(content_lengths) / max(np.mean(content_lengths), 1.0)
        
        # 時間的情報構造
        temporal_depths = [c.get('temporal_depth', 1) for c in concepts]
        temporal_information = math.log(max(temporal_depths) + 1) / 3.0
        
        # 体験タイプの多様性
        types = [str(c.get('type', 'default')) for c in concepts]
        type_diversity = len(set(types)) / len(concepts)
        
        # 情報スコア統合
        information_score = (
            quality_variance * 2.0 +
            coherence_variance * 1.5 +
            content_diversity * 1.0 +
            temporal_information * 1.0 +
            type_diversity * 1.5
        ) / 5.0
        
        # 概念数によるボーナス（情報豊富性）
        concept_count_bonus = min(math.log(len(concepts) + 1) / 4.0, 1.0)
        
        final_score = information_score * (1.0 + concept_count_bonus)
        
        return min(2.0, final_score * self.sensitivity_factor)
    
    async def _calculate_integration_axiom(self, concepts: List[Dict]) -> float:
        """
        統合公理: 体験間の不可分な関係
        An experience is unitary - 体験は統一的である
        """
        if len(concepts) < 2:
            return 0.0
        
        # 体験間関係強度の計算
        relationship_strengths = []
        
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                concept_i = concepts[i]
                concept_j = concepts[j]
                
                # 時間的関係
                temporal_rel = self._calculate_temporal_relationship(concept_i, concept_j)
                
                # 質的関係
                quality_rel = self._calculate_quality_relationship(concept_i, concept_j)
                
                # 内容的関係
                content_rel = self._calculate_content_relationship(concept_i, concept_j)
                
                # 統合関係強度
                integration_strength = (temporal_rel + quality_rel + content_rel) / 3.0
                relationship_strengths.append(integration_strength)
        
        if not relationship_strengths:
            return 0.0
        
        # 統合測定
        mean_integration = np.mean(relationship_strengths)
        integration_consistency = 1.0 - np.std(relationship_strengths) / max(mean_integration, 0.1)
        
        # ネットワーク密度（全体統合性）
        n_concepts = len(concepts)
        max_relationships = n_concepts * (n_concepts - 1) / 2
        network_density = len(relationship_strengths) / max(max_relationships, 1)
        
        # 最小カット近似（簡略化）
        min_cut_approximation = min(relationship_strengths) if relationship_strengths else 0.0
        
        # 統合スコア
        integration_score = (
            mean_integration * 0.4 +
            integration_consistency * 0.3 +
            network_density * 0.2 +
            min_cut_approximation * 0.1
        )
        
        return min(2.0, integration_score * self.sensitivity_factor * 1.2)  # 統合を重視
    
    def _calculate_exclusion_axiom(self, concepts: List[Dict]) -> float:
        """
        排他性公理: 最大φ値による境界確定
        An experience is definite - 体験は明確である
        """
        if not concepts:
            return 0.0
        
        # 体験境界の明確性
        boundary_clarity = 0.0
        
        # 体験質の一貫性（内部統一性）
        qualities = [c.get('experiential_quality', 0.5) for c in concepts]
        quality_coherence = 1.0 / (1.0 + np.std(qualities)) if len(qualities) > 1 else 1.0
        
        # 時間的境界の明確性
        temporal_depths = [c.get('temporal_depth', 1) for c in concepts]
        temporal_range = max(temporal_depths) - min(temporal_depths) if len(temporal_depths) > 1 else 0
        temporal_definiteness = 1.0 / (1.0 + temporal_range * 0.1)
        
        # コンテンツの一貫性
        content_coherence = self._calculate_content_coherence(concepts)
        
        # 体験タイプの統一性
        types = [str(c.get('type', 'default')) for c in concepts]
        type_consistency = 1.0 - (len(set(types)) - 1) / max(len(concepts), 1)
        
        # 排他性スコア
        exclusion_score = (
            quality_coherence * 0.3 +
            temporal_definiteness * 0.3 +
            content_coherence * 0.2 +
            type_consistency * 0.2
        )
        
        # 概念数による調整（大きすぎると境界が曖昧）
        size_penalty = 1.0 / (1.0 + len(concepts) * 0.01)
        
        final_score = exclusion_score * size_penalty
        
        return min(1.5, final_score * self.sensitivity_factor)
    
    def _calculate_temporal_relationship(self, concept_i: Dict, concept_j: Dict) -> float:
        """時間的関係強度を計算"""
        depth_i = concept_i.get('temporal_depth', 1)
        depth_j = concept_j.get('temporal_depth', 1)
        
        # 時間深度の類似性
        depth_similarity = 1.0 - abs(depth_i - depth_j) / max(depth_i + depth_j, 1)
        
        # タイムスタンプの近接性
        time_i = concept_i.get('timestamp', '')
        time_j = concept_j.get('timestamp', '')
        
        temporal_proximity = 0.5  # デフォルト
        if time_i and time_j:
            try:
                # 簡単な時間近接性計算
                hash_diff = abs(hash(time_i) - hash(time_j)) % 100000
                temporal_proximity = 1.0 / (1.0 + hash_diff / 10000.0)
            except:
                temporal_proximity = 0.5
        
        return (depth_similarity + temporal_proximity) / 2.0
    
    def _calculate_quality_relationship(self, concept_i: Dict, concept_j: Dict) -> float:
        """質的関係強度を計算"""
        quality_i = concept_i.get('experiential_quality', 0.5)
        quality_j = concept_j.get('experiential_quality', 0.5)
        
        coherence_i = concept_i.get('coherence', 0.5)
        coherence_j = concept_j.get('coherence', 0.5)
        
        # 質的類似性
        quality_similarity = 1.0 - abs(quality_i - quality_j)
        coherence_similarity = 1.0 - abs(coherence_i - coherence_j)
        
        return (quality_similarity + coherence_similarity) / 2.0
    
    def _calculate_content_relationship(self, concept_i: Dict, concept_j: Dict) -> float:
        """内容的関係強度を計算"""
        content_i = str(concept_i.get('content', '')).lower()
        content_j = str(concept_j.get('content', '')).lower()
        
        # 単語レベルの類似性
        words_i = set(content_i.split())
        words_j = set(content_j.split())
        
        if not words_i and not words_j:
            return 0.5
        
        intersection = len(words_i & words_j)
        union = len(words_i | words_j)
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        return jaccard_similarity
    
    def _calculate_content_coherence(self, concepts: List[Dict]) -> float:
        """コンテンツ一貫性を計算"""
        if len(concepts) < 2:
            return 1.0
        
        # 全体的な語彙の一貫性
        all_words = set()
        concept_word_sets = []
        
        for concept in concepts:
            content = str(concept.get('content', '')).lower()
            words = set(word for word in content.split() if len(word) > 2)
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
    
    def _calculate_experiential_metrics(self, concepts: List[Dict]) -> Dict[str, float]:
        """体験記憶特化メトリクス計算"""
        if not concepts:
            return {
                'integration_quality': 0.0,
                'experiential_purity': 1.0,
                'temporal_depth': 0.0,
                'self_reference_strength': 0.0
            }
        
        # 統合品質
        integration_quality = min(1.0, len(concepts) / 100.0)  # 概念数に基づく統合品質
        
        # 体験純粋性
        experiential_purity = self._calculate_experiential_purity(concepts)
        
        # 時間深度
        temporal_depths = [c.get('temporal_depth', 1) for c in concepts]
        temporal_depth = min(1.0, np.mean(temporal_depths) / 10.0)
        
        # 自己参照強度
        self_reference_strength = self._calculate_self_reference_strength(concepts)
        
        return {
            'integration_quality': integration_quality,
            'experiential_purity': experiential_purity,
            'temporal_depth': temporal_depth,
            'self_reference_strength': self_reference_strength
        }
    
    def _calculate_experiential_purity(self, concepts: List[Dict]) -> float:
        """体験純粋性を計算"""
        if not concepts:
            return 1.0
        
        # LLM混入指標
        llm_indicators = [
            'general_knowledge', 'learned_fact', 'training_data',
            'based on', 'according to', 'research shows'
        ]
        
        # 体験指標
        exp_indicators = [
            'feel', 'experience', 'sense', '感じ', '体験', '気づ'
        ]
        
        purity_scores = []
        for concept in concepts:
            content = str(concept.get('content', '')).lower()
            
            # LLM混入の検出
            llm_count = sum(1 for ind in llm_indicators if ind in content)
            exp_count = sum(1 for ind in exp_indicators if ind in content)
            
            # 純粋性スコア
            if llm_count > 0:
                purity = max(0.0, 1.0 - llm_count * 0.3)
            else:
                purity = 1.0
            
            # 体験指標ボーナス
            purity += min(exp_count * 0.1, 0.3)
            
            purity_scores.append(min(1.0, purity))
        
        return np.mean(purity_scores)
    
    def _calculate_self_reference_strength(self, concepts: List[Dict]) -> float:
        """自己参照強度を計算"""
        if not concepts:
            return 0.0
        
        self_ref_indicators = [
            'I', 'me', 'my', 'myself', 'self', '私', '自分', '自己'
        ]
        
        self_ref_count = 0
        for concept in concepts:
            content = str(concept.get('content', ''))
            if any(indicator in content for indicator in self_ref_indicators):
                self_ref_count += 1
        
        return min(1.0, self_ref_count / len(concepts) * 2.0)
    
    def _integrate_phi_components(self, existence: float, intrinsic: float, 
                                information: float, integration: float, 
                                exclusion: float, metrics: Dict[str, float]) -> float:
        """φ構成要素を統合"""
        
        # IIT公理の重み付け統合
        axiom_phi = (
            existence * 0.25 +      # 存在
            intrinsic * 0.20 +      # 内在性  
            information * 0.20 +    # 情報
            integration * 0.25 +    # 統合（重要）
            exclusion * 0.10        # 排他性
        )
        
        # 体験記憶特化ボーナス
        experiential_bonus = (
            metrics['experiential_purity'] * 0.3 +
            metrics['temporal_depth'] * 0.2 +
            metrics['self_reference_strength'] * 0.2 +
            metrics['integration_quality'] * 0.3
        )
        
        # 最終φ値
        phi_value = axiom_phi * (1.0 + experiential_bonus)
        
        return phi_value
    
    def _predict_development_stage(self, phi_value: float, metrics: Dict[str, float]) -> str:
        """発達段階を予測"""
        # 統合品質による調整
        adjusted_phi = phi_value * (0.5 + metrics['integration_quality'])
        
        # 段階判定
        for stage, threshold in reversed(list(self.stage_thresholds.items())):
            if adjusted_phi >= threshold:
                return stage
        
        return 'STAGE_0_PRE_CONSCIOUS'
    
    def _calculate_consciousness_level(self, phi_value: float, metrics: Dict[str, float]) -> float:
        """意識レベルを計算"""
        # φ値の正規化
        phi_component = min(phi_value / 10.0, 1.0)
        
        # メトリクス統合
        metrics_component = (
            metrics['integration_quality'] * 0.3 +
            metrics['experiential_purity'] * 0.3 +
            metrics['temporal_depth'] * 0.2 +
            metrics['self_reference_strength'] * 0.2
        )
        
        consciousness_level = (phi_component + metrics_component) / 2.0
        
        return min(1.0, consciousness_level)
    
    def _determine_complexity_level(self, concept_count: int) -> str:
        """計算複雑度レベルを判定"""
        if concept_count == 0:
            return "EMPTY"
        elif concept_count < 10:
            return "LOW"
        elif concept_count < 50:
            return "MEDIUM"
        elif concept_count < 200:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def get_calculation_statistics(self) -> Dict:
        """計算統計を取得"""
        if not self.calculation_history:
            return {'status': 'no_calculations'}
        
        phi_values = [entry['result'].phi_value for entry in self.calculation_history]
        concept_counts = [entry['concept_count'] for entry in self.calculation_history]
        calculation_times = [entry['result'].calculation_time for entry in self.calculation_history]
        
        return {
            'total_calculations': len(self.calculation_history),
            'average_phi': np.mean(phi_values),
            'max_phi': max(phi_values),
            'average_concepts': np.mean(concept_counts),
            'max_concepts': max(concept_counts),
            'average_calculation_time': np.mean(calculation_times),
            'phi_growth_rate': phi_values[-1] - phi_values[0] if len(phi_values) > 1 else 0.0,
            'latest_stage': self.calculation_history[-1]['result'].development_stage_prediction
        }
    
    # ===============================================
    # 動的クラスタリング機能（現象学的に健全な実装）
    # ===============================================
    
    def perform_dynamic_experiential_clustering(self, concepts: List[Dict]) -> Dict[str, Any]:
        """
        体験質ベースの動的クラスタリング実装
        
        現象学的原理に基づく：
        1. 志向的行為タイプによる分類
        2. 時間的流れ構造の保持
        3. 体験の質的特性による自然なクラスタリング
        
        Args:
            concepts: 体験概念リスト
            
        Returns:
            Dict: 動的クラスター構造と統計
        """
        if not concepts:
            return {'clusters': {}, 'quality_distribution': {}, 'temporal_coherence': 0.0}
        
        # 志向的行為タイプによる基本分類
        intentional_clusters = self._cluster_by_intentional_acts(concepts)
        
        # 体験質による細分化
        quality_refined_clusters = self._refine_by_experiential_quality(intentional_clusters)
        
        # 時間的一貫性による統合
        temporally_integrated_clusters = self._integrate_temporal_coherence(quality_refined_clusters)
        
        # 動的境界調整
        dynamic_clusters = self._adjust_dynamic_boundaries(temporally_integrated_clusters)
        
        # クラスター統計計算
        cluster_statistics = self._calculate_cluster_statistics(dynamic_clusters)
        
        return {
            'clusters': dynamic_clusters,
            'intentional_structure': intentional_clusters,
            'quality_distribution': cluster_statistics['quality_distribution'],
            'temporal_coherence': cluster_statistics['temporal_coherence'],
            'boundary_flexibility': cluster_statistics['boundary_flexibility'],
            'phenomenological_validity': cluster_statistics['phenomenological_validity']
        }
    
    def _cluster_by_intentional_acts(self, concepts: List[Dict]) -> Dict[str, List[Dict]]:
        """志向的行為タイプによる分類"""
        
        # 志向的行為の基本タイプ（より具体的で重複のない指標）
        intentional_act_types = {
            'perceiving': ['see', 'hear', 'feel', 'sense', 'notice', 'perceive', 'observe', '見る', '聞く', '感じる'],
            'remembering': ['remember', 'recall', 'recollect', 'reminisce', 'think back', '思い出', '記憶', '回想'],
            'anticipating': ['expect', 'anticipate', 'await', 'hope', 'look forward', 'tomorrow', '期待', '予想', '未来'],
            'judging': ['think', 'believe', 'consider', 'judge', 'decide', 'opinion', '思う', '考える', '判断'],
            'valuing': ['love', 'hate', 'prefer', 'value', 'appreciate', 'beautiful', '好き', '嫌い', '価値'],
            'willing': ['want', 'desire', 'intend', 'choose', 'wish', 'goal', '欲しい', '意図', '選択']
        }
        
        clusters = {act_type: [] for act_type in intentional_act_types.keys()}
        clusters['mixed_intentional'] = []  # 複数の志向性を持つ概念
        clusters['pure_experiential'] = []  # 純粋体験（特定の志向性に分類困難）
        
        for concept in concepts:
            content = str(concept.get('content', '')).lower()
            detected_acts = []
            act_scores = {}
            
            # 各志向的行為タイプの検出（より詳細な分析）
            for act_type, indicators in intentional_act_types.items():
                score = 0
                words = content.split()
                for indicator in indicators:
                    if indicator in content:
                        # 単語境界を考慮した検出
                        if indicator in words or any(indicator in word for word in words):
                            score += 1
                
                if score > 0:
                    detected_acts.append(act_type)
                    act_scores[act_type] = score
            
            # 分類（最も強いシグナルを持つものを優先）
            if len(detected_acts) == 0:
                clusters['pure_experiential'].append(concept)
            elif len(detected_acts) == 1:
                clusters[detected_acts[0]].append(concept)
            else:
                # 複数検出された場合は最も強いシグナルを優先
                if max(act_scores.values()) >= 2:  # 強いシグナルがある場合
                    best_act = max(act_scores.items(), key=lambda x: x[1])[0]
                    clusters[best_act].append(concept)
                else:
                    clusters['mixed_intentional'].append(concept)
        
        return clusters
    
    def _refine_by_experiential_quality(self, intentional_clusters: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """体験質による細分化"""
        
        refined_clusters = {}
        
        for cluster_name, concepts in intentional_clusters.items():
            if not concepts:
                continue
                
            # 体験質による細分化
            quality_levels = self._analyze_experiential_quality_distribution(concepts)
            
            if len(quality_levels) <= 1:
                # 細分化不要
                refined_clusters[cluster_name] = concepts
            else:
                # 質的レベルによる細分化
                for i, quality_group in enumerate(quality_levels):
                    refined_cluster_name = f"{cluster_name}_quality_{i+1}"
                    refined_clusters[refined_cluster_name] = quality_group
        
        return refined_clusters
    
    def _analyze_experiential_quality_distribution(self, concepts: List[Dict]) -> List[List[Dict]]:
        """体験質分布分析とグループ化"""
        
        if len(concepts) <= 3:
            return [concepts]  # 小さなグループは分割しない
        
        # 体験質の抽出
        qualities = []
        for concept in concepts:
            quality = concept.get('experiential_quality', 0.5)
            coherence = concept.get('coherence', 0.5)
            temporal_depth = concept.get('temporal_depth', 1)
            
            # 複合的な質的指標
            composite_quality = quality * 0.5 + coherence * 0.3 + min(temporal_depth / 5.0, 1.0) * 0.2
            qualities.append(composite_quality)
        
        # K-means風の簡易クラスタリング
        return self._simple_quality_clustering(concepts, qualities)
    
    def _simple_quality_clustering(self, concepts: List[Dict], qualities: List[float]) -> List[List[Dict]]:
        """簡易質的クラスタリング"""
        
        if len(concepts) <= 3:
            return [concepts]
        
        # 質的差異に基づく分割点検出
        sorted_indices = sorted(range(len(qualities)), key=lambda i: qualities[i])
        
        # 最大ギャップ検出
        gaps = []
        for i in range(1, len(sorted_indices)):
            gap = qualities[sorted_indices[i]] - qualities[sorted_indices[i-1]]
            gaps.append((gap, i))
        
        # 最大ギャップで分割（最大2分割）
        if gaps:
            max_gap, split_point = max(gaps)
            if max_gap > 0.2:  # 有意なギャップがある場合のみ分割
                group1_indices = sorted_indices[:split_point]
                group2_indices = sorted_indices[split_point:]
                
                group1 = [concepts[i] for i in group1_indices]
                group2 = [concepts[i] for i in group2_indices]
                
                return [group1, group2]
        
        return [concepts]
    
    def _integrate_temporal_coherence(self, quality_clusters: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """時間的一貫性による統合"""
        
        integrated_clusters = {}
        
        for cluster_name, concepts in quality_clusters.items():
            if not concepts:
                continue
            
            # 時間的一貫性分析
            temporal_coherence = self._calculate_temporal_coherence_score(concepts)
            
            if temporal_coherence > 0.7:
                # 高い時間的一貫性 - そのまま保持
                integrated_clusters[cluster_name] = concepts
            elif temporal_coherence > 0.4:
                # 中程度の一貫性 - 時間的統合を試行
                temporally_integrated = self._attempt_temporal_integration(concepts)
                integrated_clusters[cluster_name] = temporally_integrated
            else:
                # 低い一貫性 - 時間的分割を考慮
                temporal_subgroups = self._split_by_temporal_patterns(concepts)
                for i, subgroup in enumerate(temporal_subgroups):
                    integrated_clusters[f"{cluster_name}_temporal_{i+1}"] = subgroup
        
        return integrated_clusters
    
    def _calculate_temporal_coherence_score(self, concepts: List[Dict]) -> float:
        """時間的一貫性スコア計算"""
        
        if len(concepts) < 2:
            return 1.0
        
        # 時間深度の一貫性
        temporal_depths = [concept.get('temporal_depth', 1) for concept in concepts]
        depth_consistency = 1.0 - (np.std(temporal_depths) / max(np.mean(temporal_depths), 1.0))
        
        # タイムスタンプの連続性（利用可能な場合）
        timestamps = [concept.get('timestamp', '') for concept in concepts if concept.get('timestamp')]
        timestamp_coherence = 0.5  # デフォルト
        
        if len(timestamps) > 1:
            # 簡易的な時間順序一貫性チェック
            timestamp_hashes = [hash(ts) % 1000000 for ts in timestamps]
            if timestamp_hashes == sorted(timestamp_hashes):
                timestamp_coherence = 1.0
            else:
                # 部分的順序の評価
                correct_order_count = sum(1 for i in range(1, len(timestamp_hashes)) 
                                        if timestamp_hashes[i] >= timestamp_hashes[i-1])
                timestamp_coherence = correct_order_count / max(len(timestamp_hashes) - 1, 1)
        
        # 統合的時間一貫性
        temporal_coherence = (depth_consistency * 0.6 + timestamp_coherence * 0.4)
        
        return min(1.0, temporal_coherence)
    
    def _attempt_temporal_integration(self, concepts: List[Dict]) -> List[Dict]:
        """時間的統合の試行"""
        
        # 時間深度による重み付け統合
        integrated_concepts = []
        
        # 類似した時間深度の概念をグループ化
        depth_groups = {}
        for concept in concepts:
            depth = concept.get('temporal_depth', 1)
            depth_key = round(depth)  # 整数に丸める
            
            if depth_key not in depth_groups:
                depth_groups[depth_key] = []
            depth_groups[depth_key].append(concept)
        
        # 各深度グループの代表的概念を選択または統合
        for depth, group in depth_groups.items():
            if len(group) == 1:
                integrated_concepts.extend(group)
            else:
                # 複数概念がある場合は質的に最も豊かな概念を選択
                best_concept = max(group, key=lambda c: c.get('experiential_quality', 0.5))
                integrated_concepts.append(best_concept)
        
        return integrated_concepts
    
    def _split_by_temporal_patterns(self, concepts: List[Dict]) -> List[List[Dict]]:
        """時間的パターンによる分割"""
        
        if len(concepts) <= 2:
            return [concepts]
        
        # 時間深度による分割
        temporal_depths = [concept.get('temporal_depth', 1) for concept in concepts]
        median_depth = np.median(temporal_depths)
        
        short_term_concepts = [c for c in concepts if c.get('temporal_depth', 1) <= median_depth]
        long_term_concepts = [c for c in concepts if c.get('temporal_depth', 1) > median_depth]
        
        # 空のグループを避ける
        result = []
        if short_term_concepts:
            result.append(short_term_concepts)
        if long_term_concepts:
            result.append(long_term_concepts)
        
        return result if result else [concepts]
    
    def _adjust_dynamic_boundaries(self, temporal_clusters: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """動的境界調整"""
        
        # 小さすぎるクラスターの統合
        min_cluster_size = 2
        large_clusters = {}
        small_concepts = []
        
        for cluster_name, concepts in temporal_clusters.items():
            if len(concepts) >= min_cluster_size:
                large_clusters[cluster_name] = concepts
            else:
                small_concepts.extend(concepts)
        
        # 小さな概念を適切なクラスターに統合
        if small_concepts:
            for concept in small_concepts:
                best_cluster = self._find_best_matching_cluster(concept, large_clusters)
                if best_cluster:
                    large_clusters[best_cluster].append(concept)
                else:
                    # 新しいクラスターを作成
                    cluster_name = "integrated_small_concepts"
                    if cluster_name not in large_clusters:
                        large_clusters[cluster_name] = []
                    large_clusters[cluster_name].append(concept)
        
        return large_clusters
    
    def _find_best_matching_cluster(self, concept: Dict, clusters: Dict[str, List[Dict]]) -> Optional[str]:
        """概念に最も適合するクラスターを検索"""
        
        best_cluster = None
        best_similarity = 0.0
        
        concept_quality = concept.get('experiential_quality', 0.5)
        concept_coherence = concept.get('coherence', 0.5)
        concept_content = str(concept.get('content', '')).lower()
        
        for cluster_name, cluster_concepts in clusters.items():
            if not cluster_concepts:
                continue
            
            # クラスターとの類似度計算
            similarities = []
            
            for cluster_concept in cluster_concepts:
                # 質的類似度
                quality_sim = 1.0 - abs(concept_quality - cluster_concept.get('experiential_quality', 0.5))
                coherence_sim = 1.0 - abs(concept_coherence - cluster_concept.get('coherence', 0.5))
                
                # 内容類似度
                content_sim = self._calculate_content_similarity(concept_content, 
                                                              str(cluster_concept.get('content', '')).lower())
                
                # 統合類似度
                total_sim = (quality_sim * 0.3 + coherence_sim * 0.3 + content_sim * 0.4)
                similarities.append(total_sim)
            
            cluster_similarity = np.mean(similarities)
            
            if cluster_similarity > best_similarity:
                best_similarity = cluster_similarity
                best_cluster = cluster_name
        
        # 閾値以上の類似度を持つクラスターのみ返す
        return best_cluster if best_similarity > 0.5 else None
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """内容類似度計算"""
        
        words1 = set(word for word in content1.split() if len(word) > 2)
        words2 = set(word for word in content2.split() if len(word) > 2)
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_cluster_statistics(self, clusters: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """クラスター統計計算"""
        
        total_concepts = sum(len(concepts) for concepts in clusters.values())
        
        if total_concepts == 0:
            return {
                'quality_distribution': {},
                'temporal_coherence': 0.0,
                'boundary_flexibility': 0.0,
                'phenomenological_validity': 0.0
            }
        
        # 質的分布分析
        quality_distribution = {}
        all_qualities = []
        
        for cluster_name, concepts in clusters.items():
            if concepts:
                cluster_qualities = [c.get('experiential_quality', 0.5) for c in concepts]
                quality_distribution[cluster_name] = {
                    'mean_quality': np.mean(cluster_qualities),
                    'quality_variance': np.var(cluster_qualities),
                    'concept_count': len(concepts)
                }
                all_qualities.extend(cluster_qualities)
        
        # 全体的時間的一貫性
        temporal_coherence_scores = []
        for concepts in clusters.values():
            if len(concepts) > 1:
                coherence = self._calculate_temporal_coherence_score(concepts)
                temporal_coherence_scores.append(coherence)
        
        overall_temporal_coherence = np.mean(temporal_coherence_scores) if temporal_coherence_scores else 0.0
        
        # 境界柔軟性（クラスター間の質的重複度）
        boundary_flexibility = self._calculate_boundary_flexibility(clusters)
        
        # 現象学的妥当性スコア
        phenomenological_validity = self._calculate_phenomenological_validity(clusters)
        
        return {
            'quality_distribution': quality_distribution,
            'temporal_coherence': overall_temporal_coherence,
            'boundary_flexibility': boundary_flexibility,
            'phenomenological_validity': phenomenological_validity,
            'total_concepts': total_concepts,
            'cluster_count': len(clusters),
            'average_cluster_size': total_concepts / len(clusters) if clusters else 0
        }
    
    def _calculate_boundary_flexibility(self, clusters: Dict[str, List[Dict]]) -> float:
        """境界柔軟性計算"""
        
        cluster_list = list(clusters.items())
        if len(cluster_list) < 2:
            return 1.0  # 単一クラスターは最大柔軟性
        
        flexibility_scores = []
        
        # クラスター間の重複度分析
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                cluster1_name, cluster1_concepts = cluster_list[i]
                cluster2_name, cluster2_concepts = cluster_list[j]
                
                if not cluster1_concepts or not cluster2_concepts:
                    continue
                
                # 質的重複度
                q1_values = [c.get('experiential_quality', 0.5) for c in cluster1_concepts]
                q2_values = [c.get('experiential_quality', 0.5) for c in cluster2_concepts]
                
                q1_range = (min(q1_values), max(q1_values))
                q2_range = (min(q2_values), max(q2_values))
                
                # 範囲重複計算
                overlap_start = max(q1_range[0], q2_range[0])
                overlap_end = min(q1_range[1], q2_range[1])
                
                if overlap_end > overlap_start:
                    overlap_size = overlap_end - overlap_start
                    total_range = max(q1_range[1], q2_range[1]) - min(q1_range[0], q2_range[0])
                    flexibility = overlap_size / total_range if total_range > 0 else 0.0
                    flexibility_scores.append(flexibility)
        
        return np.mean(flexibility_scores) if flexibility_scores else 0.5
    
    def _calculate_phenomenological_validity(self, clusters: Dict[str, List[Dict]]) -> float:
        """現象学的妥当性スコア計算"""
        
        validity_components = []
        
        # 1. 志向的構造の保持
        intentional_preservation = self._assess_intentional_structure_preservation(clusters)
        validity_components.append(intentional_preservation)
        
        # 2. 時間的統一性
        temporal_unity = self._assess_temporal_unity(clusters)
        validity_components.append(temporal_unity)
        
        # 3. 体験的純粋性
        experiential_purity = self._assess_experiential_purity_preservation(clusters)
        validity_components.append(experiential_purity)
        
        # 4. 動的適応性
        dynamic_adaptability = self._assess_dynamic_adaptability(clusters)
        validity_components.append(dynamic_adaptability)
        
        # 重み付け統合
        validity_score = (
            intentional_preservation * 0.3 +
            temporal_unity * 0.3 +
            experiential_purity * 0.2 +
            dynamic_adaptability * 0.2
        )
        
        return min(1.0, validity_score)
    
    def _assess_intentional_structure_preservation(self, clusters: Dict[str, List[Dict]]) -> float:
        """志向的構造保持評価"""
        
        # 志向的行為タイプクラスターの存在確認
        intentional_cluster_indicators = [
            'perceiving', 'remembering', 'anticipating', 'judging', 'valuing', 'willing',
            'mixed_intentional', 'pure_experiential'
        ]
        
        preserved_structures = 0
        for cluster_name in clusters.keys():
            if any(indicator in cluster_name for indicator in intentional_cluster_indicators):
                preserved_structures += 1
        
        # 全体クラスター数に対する志向的構造クラスターの割合
        preservation_ratio = preserved_structures / max(len(clusters), 1)
        
        return min(1.0, preservation_ratio * 1.2)  # 1.2倍でボーナス
    
    def _assess_temporal_unity(self, clusters: Dict[str, List[Dict]]) -> float:
        """時間的統一性評価"""
        
        temporal_scores = []
        
        for concepts in clusters.values():
            if len(concepts) > 1:
                temporal_coherence = self._calculate_temporal_coherence_score(concepts)
                temporal_scores.append(temporal_coherence)
        
        return np.mean(temporal_scores) if temporal_scores else 0.8
    
    def _assess_experiential_purity_preservation(self, clusters: Dict[str, List[Dict]]) -> float:
        """体験的純粋性保持評価"""
        
        purity_scores = []
        
        for concepts in clusters.values():
            if concepts:
                cluster_purity = self._calculate_experiential_purity(concepts)
                purity_scores.append(cluster_purity)
        
        return np.mean(purity_scores) if purity_scores else 0.8
    
    def _assess_dynamic_adaptability(self, clusters: Dict[str, List[Dict]]) -> float:
        """動的適応性評価"""
        
        # クラスターサイズの多様性（固定サイズでない）
        cluster_sizes = [len(concepts) for concepts in clusters.values() if concepts]
        
        if len(cluster_sizes) < 2:
            return 0.5
        
        size_variance = np.var(cluster_sizes)
        size_mean = np.mean(cluster_sizes)
        
        # 適度な多様性を評価（完全に均一でも完全に不均一でもない）
        diversity_score = min(1.0, size_variance / max(size_mean, 1))
        optimal_diversity = 0.5  # 最適多様性レベル
        
        adaptability = 1.0 - abs(diversity_score - optimal_diversity)
        
        return adaptability