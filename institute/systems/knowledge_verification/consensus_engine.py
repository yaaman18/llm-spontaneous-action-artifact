"""
専門家コンセンサス形成エンジン
複数の専門家による合意形成と知識統合システム
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict, Counter

from .domain_specialists import (
    BaseDomainSpecialist, DomainSpecialistFactory, 
    DomainVerificationResult, VerificationLevel
)

class ConsensusType(Enum):
    UNANIMOUS = "unanimous"          # 全員一致
    STRONG_MAJORITY = "strong_majority"  # 強い多数決 (80%+)
    SIMPLE_MAJORITY = "simple_majority"  # 単純多数決 (50%+)
    PLURALITY = "plurality"          # 相対多数
    NO_CONSENSUS = "no_consensus"    # 合意なし
    EXPERT_OVERRIDE = "expert_override"  # 専門家優先

class DisagreementType(Enum):
    METHODOLOGICAL = "methodological"    # 方法論的相違
    THEORETICAL = "theoretical"          # 理論的相違
    EMPIRICAL = "empirical"             # 実証的相違
    DEFINITIONAL = "definitional"       # 定義的相違
    CONTEXTUAL = "contextual"           # 文脈的相違

@dataclass
class ExpertOpinion:
    expert_name: str
    domain: str
    verification_result: DomainVerificationResult
    weight: float  # 専門性による重み
    confidence: float
    reasoning: str
    supporting_evidence: List[str]
    dissenting_points: List[str]

@dataclass
class ConsensusResult:
    statement: str
    consensus_type: ConsensusType
    overall_validity: bool
    confidence_score: float
    participating_experts: List[str]
    agreeing_experts: List[str]
    dissenting_experts: List[str]
    key_points_of_agreement: List[str]
    points_of_disagreement: List[str]
    disagreement_types: List[DisagreementType]
    synthesized_conclusion: str
    recommendations: List[str]
    minority_opinions: List[str]
    timestamp: datetime
    
    # 詳細分析
    domain_breakdown: Dict[str, Any]
    expertise_weighted_score: float
    uncertainty_factors: List[str]

class ExpertWeightingSystem:
    """専門家重み付けシステム"""
    
    def __init__(self):
        # ドメインごとの専門性スコア
        self.domain_expertise = {
            'consciousness': {
                'tononi-koch': 0.95,      # IIT創始者
                'zahavi': 0.90,           # 現象学権威
                'kanai': 0.85,            # 実装専門家
                'chalmers': 0.92,         # 意識哲学権威
                'baars': 0.88             # GWT提唱者
            },
            'philosophy': {
                'zahavi': 0.88,
                'chalmers': 0.85,
                'dennett': 0.90,
                'nagel': 0.87
            },
            'mathematics': {
                'tononi-koch': 0.80,      # 数理モデル専門
                'computational-expert': 0.95
            }
        }
        
        # 分野横断的影響力
        self.cross_domain_influence = {
            'tononi-koch': 0.9,   # 理論統合力
            'zahavi': 0.8,        # 哲学的基盤
            'kanai': 0.85         # 実装橋渡し
        }
    
    def calculate_expert_weight(self, 
                              expert_name: str, 
                              primary_domain: str,
                              statement_domains: List[str]) -> float:
        """専門家の重みを計算"""
        
        # 主要分野での専門性
        primary_expertise = self.domain_expertise.get(primary_domain, {}).get(expert_name, 0.5)
        
        # 関連分野での専門性
        related_expertise = []
        for domain in statement_domains:
            if domain != primary_domain:
                expertise = self.domain_expertise.get(domain, {}).get(expert_name, 0.3)
                related_expertise.append(expertise)
        
        # 分野横断影響力
        cross_influence = self.cross_domain_influence.get(expert_name, 0.5)
        
        # 総合重み計算
        if related_expertise:
            avg_related = np.mean(related_expertise)
            total_weight = (
                primary_expertise * 0.6 + 
                avg_related * 0.2 + 
                cross_influence * 0.2
            )
        else:
            total_weight = primary_expertise * 0.8 + cross_influence * 0.2
        
        return float(np.clip(total_weight, 0.1, 1.0))

class DisagreementAnalyzer:
    """意見相違分析システム"""
    
    def __init__(self):
        self.disagreement_patterns = {
            'theoretical': [
                'theory', 'model', 'framework', 'paradigm',
                'approach', 'perspective', 'interpretation'
            ],
            'methodological': [
                'method', 'methodology', 'approach', 'technique',
                'procedure', 'protocol', 'measurement'
            ],
            'empirical': [
                'evidence', 'data', 'research', 'study',
                'experiment', 'observation', 'finding'
            ],
            'definitional': [
                'definition', 'meaning', 'concept', 'term',
                'notion', 'understanding', 'interpretation'
            ]
        }
    
    def analyze_disagreements(self, 
                            expert_opinions: List[ExpertOpinion]) -> Tuple[List[str], List[DisagreementType]]:
        """意見相違を分析"""
        
        # 意見の分類
        agreeing = [op for op in expert_opinions if op.verification_result.is_valid]
        dissenting = [op for op in expert_opinions if not op.verification_result.is_valid]
        
        disagreement_points = []
        disagreement_types = []
        
        if len(agreeing) > 0 and len(dissenting) > 0:
            # 具体的な相違点を抽出
            for dissenter in dissenting:
                for finding in dissenter.verification_result.findings:
                    disagreement_points.append(f"{dissenter.expert_name}: {finding}")
                    
                    # 相違タイプを分類
                    disagreement_type = self._classify_disagreement_type(finding)
                    if disagreement_type not in disagreement_types:
                        disagreement_types.append(disagreement_type)
        
        return disagreement_points, disagreement_types
    
    def _classify_disagreement_type(self, disagreement_text: str) -> DisagreementType:
        """相違タイプを分類"""
        text_lower = disagreement_text.lower()
        
        # パターンマッチング
        for disagreement_type, keywords in self.disagreement_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return DisagreementType(disagreement_type)
        
        return DisagreementType.THEORETICAL  # デフォルト

class ConsensusEngine:
    """専門家コンセンサス形成エンジン"""
    
    def __init__(self):
        self.weighting_system = ExpertWeightingSystem()
        self.disagreement_analyzer = DisagreementAnalyzer()
        self.consensus_history: List[ConsensusResult] = []
        
        # コンセンサス閾値
        self.consensus_thresholds = {
            ConsensusType.UNANIMOUS: 1.0,
            ConsensusType.STRONG_MAJORITY: 0.8,
            ConsensusType.SIMPLE_MAJORITY: 0.5,
            ConsensusType.PLURALITY: 0.0  # 最大得票
        }
    
    async def form_consensus(self, 
                           statement: str,
                           expert_opinions: List[ExpertOpinion],
                           context: str = None) -> ConsensusResult:
        """専門家コンセンサスを形成"""
        
        if not expert_opinions:
            return self._create_no_consensus_result(statement, "No expert opinions provided")
        
        # 1. 専門家重み計算
        weighted_opinions = await self._calculate_expert_weights(expert_opinions, statement)
        
        # 2. 基本統計計算
        stats = self._calculate_consensus_statistics(weighted_opinions)
        
        # 3. コンセンサスタイプ決定
        consensus_type = self._determine_consensus_type(stats)
        
        # 4. 意見相違分析
        disagreement_points, disagreement_types = self.disagreement_analyzer.analyze_disagreements(
            expert_opinions
        )
        
        # 5. 専門分野別分析
        domain_breakdown = self._analyze_by_domain(expert_opinions)
        
        # 6. 総合結論生成
        synthesized_conclusion = await self._synthesize_conclusion(
            weighted_opinions, consensus_type, stats
        )
        
        # 7. 推奨事項生成
        recommendations = await self._generate_recommendations(
            weighted_opinions, consensus_type, disagreement_types
        )
        
        # 8. 結果構築
        consensus_result = ConsensusResult(
            statement=statement,
            consensus_type=consensus_type,
            overall_validity=stats['weighted_validity'] > 0.6,
            confidence_score=stats['confidence_score'],
            participating_experts=[op.expert_name for op in expert_opinions],
            agreeing_experts=[op.expert_name for op in expert_opinions if op.verification_result.is_valid],
            dissenting_experts=[op.expert_name for op in expert_opinions if not op.verification_result.is_valid],
            key_points_of_agreement=self._extract_agreement_points(expert_opinions),
            points_of_disagreement=disagreement_points,
            disagreement_types=disagreement_types,
            synthesized_conclusion=synthesized_conclusion,
            recommendations=recommendations,
            minority_opinions=self._extract_minority_opinions(expert_opinions),
            timestamp=datetime.now(),
            domain_breakdown=domain_breakdown,
            expertise_weighted_score=stats['weighted_validity'],
            uncertainty_factors=self._identify_uncertainty_factors(expert_opinions)
        )
        
        # 履歴記録
        self.consensus_history.append(consensus_result)
        
        return consensus_result
    
    async def _calculate_expert_weights(self, 
                                      expert_opinions: List[ExpertOpinion],
                                      statement: str) -> List[Tuple[ExpertOpinion, float]]:
        """専門家の重みを計算"""
        
        # 文に関連するドメインを特定
        statement_domains = self._identify_statement_domains(statement, expert_opinions)
        
        weighted_opinions = []
        for opinion in expert_opinions:
            weight = self.weighting_system.calculate_expert_weight(
                opinion.expert_name,
                opinion.domain,
                statement_domains
            )
            # 信頼度による重み調整
            adjusted_weight = weight * opinion.confidence
            weighted_opinions.append((opinion, adjusted_weight))
        
        return weighted_opinions
    
    def _identify_statement_domains(self, 
                                  statement: str, 
                                  expert_opinions: List[ExpertOpinion]) -> List[str]:
        """文に関連するドメインを特定"""
        domains = set()
        for opinion in expert_opinions:
            domains.add(opinion.domain)
        return list(domains)
    
    def _calculate_consensus_statistics(self, 
                                      weighted_opinions: List[Tuple[ExpertOpinion, float]]) -> Dict[str, float]:
        """コンセンサス統計を計算"""
        
        if not weighted_opinions:
            return {'weighted_validity': 0.0, 'confidence_score': 0.0}
        
        total_weight = sum(weight for _, weight in weighted_opinions)
        if total_weight == 0:
            return {'weighted_validity': 0.0, 'confidence_score': 0.0}
        
        # 重み付き妥当性スコア
        weighted_validity_sum = sum(
            weight * (1.0 if opinion.verification_result.is_valid else 0.0)
            for opinion, weight in weighted_opinions
        )
        weighted_validity = weighted_validity_sum / total_weight
        
        # 重み付き信頼度スコア
        weighted_confidence_sum = sum(
            weight * opinion.verification_result.confidence_score
            for opinion, weight in weighted_opinions
        )
        weighted_confidence = weighted_confidence_sum / total_weight
        
        # 合意度計算
        agreement_ratio = len([op for op, _ in weighted_opinions if op.verification_result.is_valid]) / len(weighted_opinions)
        
        return {
            'weighted_validity': weighted_validity,
            'confidence_score': weighted_confidence,
            'agreement_ratio': agreement_ratio,
            'total_experts': len(weighted_opinions)
        }
    
    def _determine_consensus_type(self, stats: Dict[str, float]) -> ConsensusType:
        """コンセンサスタイプを決定"""
        
        agreement_ratio = stats['agreement_ratio']
        
        if agreement_ratio == 1.0:
            return ConsensusType.UNANIMOUS
        elif agreement_ratio >= 0.8:
            return ConsensusType.STRONG_MAJORITY
        elif agreement_ratio >= 0.5:
            return ConsensusType.SIMPLE_MAJORITY
        elif agreement_ratio > 0:
            return ConsensusType.PLURALITY
        else:
            return ConsensusType.NO_CONSENSUS
    
    def _analyze_by_domain(self, expert_opinions: List[ExpertOpinion]) -> Dict[str, Any]:
        """分野別分析"""
        
        domain_analysis = defaultdict(lambda: {
            'experts': [],
            'validity_scores': [],
            'confidence_scores': [],
            'agreement_ratio': 0.0
        })
        
        for opinion in expert_opinions:
            domain = opinion.domain
            domain_analysis[domain]['experts'].append(opinion.expert_name)
            domain_analysis[domain]['validity_scores'].append(
                1.0 if opinion.verification_result.is_valid else 0.0
            )
            domain_analysis[domain]['confidence_scores'].append(
                opinion.verification_result.confidence_score
            )
        
        # 各分野の合意度を計算
        for domain, data in domain_analysis.items():
            if data['validity_scores']:
                data['agreement_ratio'] = np.mean(data['validity_scores'])
                data['avg_confidence'] = np.mean(data['confidence_scores'])
        
        return dict(domain_analysis)
    
    async def _synthesize_conclusion(self, 
                                   weighted_opinions: List[Tuple[ExpertOpinion, float]],
                                   consensus_type: ConsensusType,
                                   stats: Dict[str, float]) -> str:
        """総合結論を生成"""
        
        conclusion_parts = []
        
        # コンセンサスタイプに基づく基本結論
        if consensus_type == ConsensusType.UNANIMOUS:
            conclusion_parts.append("全専門家が一致して")
            if stats['weighted_validity'] > 0.5:
                conclusion_parts.append("この文は妥当であると判断しています。")
            else:
                conclusion_parts.append("この文は妥当でないと判断しています。")
        
        elif consensus_type == ConsensusType.STRONG_MAJORITY:
            conclusion_parts.append("強い多数決により")
            if stats['weighted_validity'] > 0.5:
                conclusion_parts.append("この文は概ね妥当と考えられます。")
            else:
                conclusion_parts.append("この文は概ね妥当でないと考えられます。")
        
        elif consensus_type == ConsensusType.SIMPLE_MAJORITY:
            conclusion_parts.append("過半数の専門家により")
            if stats['weighted_validity'] > 0.5:
                conclusion_parts.append("この文は妥当とされていますが、異論もあります。")
            else:
                conclusion_parts.append("この文は妥当でないとされていますが、支持する意見もあります。")
        
        else:
            conclusion_parts.append("専門家の間で明確な合意は形成されていません。")
        
        # 信頼度情報追加
        confidence_level = "高い" if stats['confidence_score'] > 0.8 else "中程度" if stats['confidence_score'] > 0.6 else "低い"
        conclusion_parts.append(f"この判断の信頼度は{confidence_level}です。")
        
        return " ".join(conclusion_parts)
    
    async def _generate_recommendations(self, 
                                      weighted_opinions: List[Tuple[ExpertOpinion, float]],
                                      consensus_type: ConsensusType,
                                      disagreement_types: List[DisagreementType]) -> List[str]:
        """推奨事項を生成"""
        
        recommendations = []
        
        # コンセンサスタイプに基づく推奨
        if consensus_type == ConsensusType.NO_CONSENSUS:
            recommendations.append("さらなる専門家の意見収集を推奨します")
            recommendations.append("文の表現をより明確にすることを検討してください")
        
        elif consensus_type in [ConsensusType.SIMPLE_MAJORITY, ConsensusType.PLURALITY]:
            recommendations.append("少数意見も考慮した文の修正を検討してください")
        
        # 相違タイプに基づく推奨
        if DisagreementType.DEFINITIONAL in disagreement_types:
            recommendations.append("用語の定義をより明確にしてください")
        
        if DisagreementType.METHODOLOGICAL in disagreement_types:
            recommendations.append("方法論的前提を明示することを推奨します")
        
        if DisagreementType.EMPIRICAL in disagreement_types:
            recommendations.append("実証的根拠の追加検証が必要です")
        
        # 高信頼度専門家の特別推奨
        high_weight_opinions = [op for op, weight in weighted_opinions if weight > 0.8]
        if high_weight_opinions:
            for opinion in high_weight_opinions:
                if opinion.verification_result.corrections:
                    recommendations.extend([
                        f"高信頼度専門家({opinion.expert_name})の修正提案: {correction}"
                        for correction in opinion.verification_result.corrections[:2]
                    ])
        
        return recommendations[:5]  # 最大5つの推奨事項
    
    def _extract_agreement_points(self, expert_opinions: List[ExpertOpinion]) -> List[str]:
        """合意点を抽出"""
        
        # 共通の指摘事項を特定
        all_findings = []
        for opinion in expert_opinions:
            all_findings.extend(opinion.verification_result.findings)
        
        # 頻出する指摘事項を合意点とみなす
        finding_counts = Counter(all_findings)
        common_findings = [
            finding for finding, count in finding_counts.items()
            if count >= len(expert_opinions) * 0.5  # 半数以上が指摘
        ]
        
        return common_findings[:3]  # 上位3つ
    
    def _extract_minority_opinions(self, expert_opinions: List[ExpertOpinion]) -> List[str]:
        """少数意見を抽出"""
        
        # 妥当性判断の少数派を特定
        validity_votes = [op.verification_result.is_valid for op in expert_opinions]
        majority_vote = Counter(validity_votes).most_common(1)[0][0]
        
        minority_opinions = []
        for opinion in expert_opinions:
            if opinion.verification_result.is_valid != majority_vote:
                minority_opinions.append(
                    f"{opinion.expert_name}: {opinion.reasoning}"
                )
        
        return minority_opinions
    
    def _identify_uncertainty_factors(self, expert_opinions: List[ExpertOpinion]) -> List[str]:
        """不確実性要因を特定"""
        
        uncertainty_factors = []
        
        # 信頼度のばらつき
        confidences = [op.verification_result.confidence_score for op in expert_opinions]
        if len(confidences) > 1:
            confidence_std = np.std(confidences)
            if confidence_std > 0.2:
                uncertainty_factors.append("専門家間で信頼度に大きなばらつきがあります")
        
        # 検証レベルの不統一
        verification_levels = [op.verification_result.verification_level for op in expert_opinions]
        if len(set(verification_levels)) > 1:
            uncertainty_factors.append("検証レベルが専門家間で異なります")
        
        # 分野横断的な複雑性
        domains = set(op.domain for op in expert_opinions)
        if len(domains) > 2:
            uncertainty_factors.append("複数分野にまたがる複雑な内容です")
        
        return uncertainty_factors
    
    def _create_no_consensus_result(self, statement: str, reason: str) -> ConsensusResult:
        """合意なし結果を作成"""
        return ConsensusResult(
            statement=statement,
            consensus_type=ConsensusType.NO_CONSENSUS,
            overall_validity=False,
            confidence_score=0.0,
            participating_experts=[],
            agreeing_experts=[],
            dissenting_experts=[],
            key_points_of_agreement=[],
            points_of_disagreement=[reason],
            disagreement_types=[],
            synthesized_conclusion=f"コンセンサス形成不可: {reason}",
            recommendations=["専門家の意見を収集してください"],
            minority_opinions=[],
            timestamp=datetime.now(),
            domain_breakdown={},
            expertise_weighted_score=0.0,
            uncertainty_factors=[reason]
        )
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """コンセンサス統計を取得"""
        if not self.consensus_history:
            return {"total_consensus_attempts": 0}
        
        total = len(self.consensus_history)
        consensus_types = Counter(result.consensus_type for result in self.consensus_history)
        
        avg_confidence = np.mean([result.confidence_score for result in self.consensus_history])
        avg_experts_per_consensus = np.mean([len(result.participating_experts) for result in self.consensus_history])
        
        return {
            "total_consensus_attempts": total,
            "consensus_type_distribution": dict(consensus_types),
            "average_confidence": float(avg_confidence),
            "average_experts_per_consensus": float(avg_experts_per_consensus),
            "unanimous_rate": consensus_types[ConsensusType.UNANIMOUS] / total if total > 0 else 0
        }

# 使用例
async def main():
    """コンセンサスエンジンテスト"""
    
    # テスト用の専門家意見を作成
    consciousness_specialist = DomainSpecialistFactory.create_specialist('consciousness')
    philosophy_specialist = DomainSpecialistFactory.create_specialist('philosophy')
    
    test_statement = "意識は脳の電気活動によって完全に説明できる物理現象である"
    
    # 各専門家の意見を取得
    consciousness_result = await consciousness_specialist.verify_statement(test_statement)
    philosophy_result = await philosophy_specialist.verify_statement(test_statement)
    
    # 専門家意見リストを構築
    expert_opinions = [
        ExpertOpinion(
            expert_name="consciousness_specialist",
            domain="consciousness",
            verification_result=consciousness_result,
            weight=0.9,
            confidence=consciousness_result.confidence_score,
            reasoning="意識研究の観点から検証",
            supporting_evidence=[],
            dissenting_points=[]
        ),
        ExpertOpinion(
            expert_name="philosophy_specialist", 
            domain="philosophy",
            verification_result=philosophy_result,
            weight=0.85,
            confidence=philosophy_result.confidence_score,
            reasoning="哲学的観点から検証",
            supporting_evidence=[],
            dissenting_points=[]
        )
    ]
    
    # コンセンサス形成
    consensus_engine = ConsensusEngine()
    consensus_result = await consensus_engine.form_consensus(test_statement, expert_opinions)
    
    print(f"Statement: {consensus_result.statement}")
    print(f"Consensus Type: {consensus_result.consensus_type.value}")
    print(f"Overall Validity: {consensus_result.overall_validity}")
    print(f"Confidence: {consensus_result.confidence_score:.2f}")
    print(f"Conclusion: {consensus_result.synthesized_conclusion}")
    print(f"Recommendations: {consensus_result.recommendations}")

if __name__ == "__main__":
    asyncio.run(main())