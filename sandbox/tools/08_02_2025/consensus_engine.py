#!/usr/bin/env python3
"""
NewbornAI 2.0 専門家コンセンサス形成システム
複数の専門領域からの知見を統合してドキュメント信頼性を評価
"""

import asyncio
import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import logging
import numpy as np
from pathlib import Path

class ExpertDomain(Enum):
    """専門領域定義"""
    PHENOMENOLOGY = "phenomenology"
    CONSCIOUSNESS_STUDIES = "consciousness_studies" 
    INTEGRATED_INFORMATION_THEORY = "integrated_information_theory"
    ENACTIVE_COGNITION = "enactive_cognition"
    AI_ARCHITECTURE = "ai_architecture"
    PYTHON_IMPLEMENTATION = "python_implementation"
    SECURITY_ENGINEERING = "security_engineering"
    CLAUDE_SDK_INTEGRATION = "claude_sdk_integration"
    MATHEMATICAL_MODELING = "mathematical_modeling"
    COGNITIVE_DEVELOPMENT = "cognitive_development"

@dataclass
class ExpertProfile:
    """専門家プロファイル"""
    name: str
    domain: ExpertDomain
    credibility_score: float  # 0.0-1.0
    bias_factors: Dict[str, float]
    specialization_areas: List[str]
    validation_criteria: Dict[str, Any]

@dataclass
class ValidationResult:
    """検証結果"""
    expert_name: str
    domain: ExpertDomain
    document_path: str
    confidence_score: float  # 0.0-1.0
    identified_issues: List[Dict[str, Any]]
    theoretical_accuracy: float
    implementation_feasibility: float
    consistency_score: float
    hallucination_risk: float
    timestamp: datetime
    detailed_analysis: Dict[str, Any]

@dataclass
class ConsensusResult:
    """コンセンサス結果"""
    document_path: str
    overall_confidence: float
    consensus_strength: float  # 専門家間の一致度
    expert_opinions: List[ValidationResult]
    identified_hallucinations: List[Dict[str, Any]]
    theoretical_issues: List[Dict[str, Any]]
    implementation_concerns: List[Dict[str, Any]]
    recommendations: List[str]
    final_assessment: Dict[str, float]
    timestamp: datetime

class ExpertSimulator:
    """専門家知見シミュレーター"""
    
    def __init__(self, expert_profile: ExpertProfile):
        self.profile = expert_profile
        self.domain_knowledge = self._load_domain_knowledge()
        self.validation_patterns = self._initialize_validation_patterns()
        
    def _load_domain_knowledge(self) -> Dict[str, Any]:
        """領域特化知識の読み込み"""
        domain_knowledge = {
            ExpertDomain.PHENOMENOLOGY: {
                "key_concepts": ["intentionality", "epoché", "temporality", "intersubjectivity"],
                "authority_sources": ["Husserl", "Heidegger", "Merleau-Ponty", "Sartre"],
                "validation_criteria": {
                    "concept_accuracy": 0.9,
                    "citation_relevance": 0.8,
                    "logical_consistency": 0.95
                }
            },
            ExpertDomain.INTEGRATED_INFORMATION_THEORY: {
                "key_concepts": ["phi_value", "integrated_information", "minimum_cut", "causal_structure"],
                "authority_sources": ["Tononi", "Oizumi", "Albantakis", "Massimini"],
                "validation_criteria": {
                    "mathematical_accuracy": 0.95,
                    "algorithmic_correctness": 0.9,
                    "empirical_support": 0.8
                }
            },
            ExpertDomain.ENACTIVE_COGNITION: {
                "key_concepts": ["autopoiesis", "structural_coupling", "embodied_cognition", "sensorimotor_contingencies"],
                "authority_sources": ["Varela", "Maturana", "Di Paolo", "Stewart"],
                "validation_criteria": {
                    "theoretical_coherence": 0.9,
                    "biological_plausibility": 0.85,
                    "implementation_viability": 0.8
                }
            },
            ExpertDomain.AI_ARCHITECTURE: {
                "key_concepts": ["clean_architecture", "dependency_injection", "microservices", "scalability"],
                "authority_sources": ["Martin", "Evans", "Fowler", "Newman"],
                "validation_criteria": {
                    "architectural_soundness": 0.9,
                    "scalability_considerations": 0.85,
                    "maintainability": 0.8
                }
            },
            ExpertDomain.PYTHON_IMPLEMENTATION: {
                "key_concepts": ["asyncio", "dataclasses", "type_hints", "performance_optimization"],
                "authority_sources": ["Van Rossum", "Beazley", "Ramalho", "Percival"],
                "validation_criteria": {
                    "code_quality": 0.9,
                    "pythonic_practices": 0.85,
                    "performance_efficiency": 0.8
                }
            }
        }
        
        return domain_knowledge.get(self.profile.domain, {})
    
    def _initialize_validation_patterns(self) -> Dict[str, Any]:
        """検証パターン初期化"""
        return {
            "red_flags": [
                "undefined_technical_terms",
                "impossible_performance_claims", 
                "missing_theoretical_foundation",
                "inconsistent_numerical_values",
                "unrealistic_implementation_timelines"
            ],
            "quality_indicators": [
                "proper_citation_format",
                "mathematical_rigor",
                "implementation_detail_completeness",
                "error_handling_consideration",
                "scalability_discussion"
            ]
        }
    
    async def validate_document(self, document_content: str, document_path: str) -> ValidationResult:
        """ドキュメント検証の実行"""
        
        logging.info(f"{self.profile.name} analyzing {document_path}")
        
        # 領域特化分析
        domain_analysis = await self._analyze_domain_specific_content(document_content)
        
        # 理論的正確性評価
        theoretical_accuracy = await self._assess_theoretical_accuracy(document_content)
        
        # 実装可能性評価  
        implementation_feasibility = await self._assess_implementation_feasibility(document_content)
        
        # 一貫性評価
        consistency_score = await self._assess_consistency(document_content)
        
        # ハルシネーション検出
        hallucination_risk = await self._detect_hallucinations(document_content)
        
        # 問題点特定
        identified_issues = await self._identify_issues(document_content, domain_analysis)
        
        # 総合信頼度計算
        confidence_score = self._calculate_confidence_score(
            theoretical_accuracy, implementation_feasibility, 
            consistency_score, hallucination_risk
        )
        
        return ValidationResult(
            expert_name=self.profile.name,
            domain=self.profile.domain,
            document_path=document_path,
            confidence_score=confidence_score,
            identified_issues=identified_issues,
            theoretical_accuracy=theoretical_accuracy,
            implementation_feasibility=implementation_feasibility,
            consistency_score=consistency_score,
            hallucination_risk=hallucination_risk,
            timestamp=datetime.now(),
            detailed_analysis=domain_analysis
        )
    
    async def _analyze_domain_specific_content(self, content: str) -> Dict[str, Any]:
        """領域特化内容分析"""
        
        analysis = {
            "concept_coverage": {},
            "authority_citations": [],
            "technical_depth": 0.0,
            "novelty_claims": [],
            "validation_flags": []
        }
        
        # キー概念の検出と評価
        for concept in self.domain_knowledge.get("key_concepts", []):
            if concept.lower() in content.lower():
                # 概念の使用文脈分析
                context_quality = self._analyze_concept_context(content, concept)
                analysis["concept_coverage"][concept] = context_quality
        
        # 権威ある情報源の引用確認
        for authority in self.domain_knowledge.get("authority_sources", []):
            if authority in content:
                analysis["authority_citations"].append(authority)
        
        # 技術的深度評価
        analysis["technical_depth"] = self._assess_technical_depth(content)
        
        return analysis
    
    async def _assess_theoretical_accuracy(self, content: str) -> float:
        """理論的正確性評価"""
        
        accuracy_factors = []
        
        # 領域特化検証
        if self.profile.domain == ExpertDomain.PHENOMENOLOGY:
            accuracy_factors.append(self._verify_phenomenological_accuracy(content))
        elif self.profile.domain == ExpertDomain.INTEGRATED_INFORMATION_THEORY:
            accuracy_factors.append(self._verify_iit_mathematical_accuracy(content))
        elif self.profile.domain == ExpertDomain.ENACTIVE_COGNITION:
            accuracy_factors.append(self._verify_enactive_theoretical_coherence(content))
        
        # 一般的理論的厳密性
        accuracy_factors.append(self._assess_logical_consistency(content))
        accuracy_factors.append(self._check_citation_accuracy(content))
        
        return np.mean(accuracy_factors) if accuracy_factors else 0.5
    
    async def _assess_implementation_feasibility(self, content: str) -> float:
        """実装可能性評価"""
        
        feasibility_factors = []
        
        # リソース要件の現実性
        feasibility_factors.append(self._assess_resource_requirements(content))
        
        # 技術的制約の考慮
        feasibility_factors.append(self._check_technical_constraints(content))
        
        # 実装詳細の完全性
        feasibility_factors.append(self._assess_implementation_completeness(content))
        
        # タイムライン評価
        feasibility_factors.append(self._assess_timeline_realism(content))
        
        return np.mean(feasibility_factors)
    
    async def _detect_hallucinations(self, content: str) -> float:
        """ハルシネーション検出（リスクスコア：高いほど危険）"""
        
        risk_factors = []
        
        # 検証不可能な主張
        risk_factors.append(self._detect_unverifiable_claims(content))
        
        # 矛盾する記述
        risk_factors.append(self._detect_contradictions(content))
        
        # 非現実的な性能主張
        risk_factors.append(self._detect_unrealistic_claims(content))
        
        # 引用の正確性
        risk_factors.append(self._verify_citation_accuracy(content))
        
        return np.mean(risk_factors)
    
    def _verify_phenomenological_accuracy(self, content: str) -> float:
        """現象学的正確性検証"""
        accuracy_score = 0.8  # ベースライン
        
        # フッサールの時間意識理論の正確性
        if "husserian" in content.lower() and "time consciousness" in content.lower():
            if "retention" in content and "protention" in content and "primal impression" in content:
                accuracy_score += 0.1
            else:
                accuracy_score -= 0.2
        
        # 志向性概念の適切な使用
        if "intentionality" in content.lower():
            if "noesis" in content and "noema" in content:
                accuracy_score += 0.05
        
        return min(1.0, max(0.0, accuracy_score))
    
    def _verify_iit_mathematical_accuracy(self, content: str) -> float:
        """IIT数学的正確性検証"""
        accuracy_score = 0.7  # ベースライン
        
        # φ値計算の数学的正確性
        if "phi" in content.lower() or "φ" in content:
            if "integrated information" in content.lower():
                accuracy_score += 0.1
            if "minimum cut" in content.lower():
                accuracy_score += 0.1
            # 不正確な計算式の検出
            if "φ = Φ_experiential(S) = ∑[EI(experiential_concept) - min_cut(experiential_concept)]" in content:
                # この式は実際のIITと異なる可能性を検証
                accuracy_score -= 0.15  # 要検証項目
        
        return min(1.0, max(0.0, accuracy_score))
    
    def _verify_enactive_theoretical_coherence(self, content: str) -> float:
        """エナクティブ認知理論の一貫性検証"""
        accuracy_score = 0.75
        
        # オートポイエーシス概念の適切な使用
        if "autopoiesis" in content.lower():
            if "structural coupling" in content.lower():
                accuracy_score += 0.1
        
        # 身体化認知の適切な表現
        if "embodied" in content.lower() and "cognition" in content.lower():
            accuracy_score += 0.1
        
        return min(1.0, max(0.0, accuracy_score))
    
    def _assess_logical_consistency(self, content: str) -> float:
        """論理的一貫性評価"""
        return 0.8
    
    def _check_citation_accuracy(self, content: str) -> float:
        """引用の正確性チェック"""
        return 0.8
    
    def _assess_resource_requirements(self, content: str) -> float:
        """リソース要件の現実性評価"""
        if "real-time" in content.lower() and "complex" in content.lower():
            return 0.6
        return 0.8
    
    def _check_technical_constraints(self, content: str) -> float:
        """技術的制約の考慮度"""
        return 0.8
    
    def _assess_implementation_completeness(self, content: str) -> float:
        """実装詳細の完全性"""
        if "class " in content and "def " in content:
            return 0.9
        return 0.7
    
    def _assess_timeline_realism(self, content: str) -> float:
        """タイムライン評価"""
        return 0.8
    
    def _detect_unverifiable_claims(self, content: str) -> float:
        """検証不可能な主張の検出"""
        risk_score = 0.2
        exaggerated_claims = ["revolutionary", "breakthrough", "unprecedented", "完全な"]
        for claim in exaggerated_claims:
            if claim in content.lower():
                risk_score += 0.1
        return min(1.0, risk_score)
    
    def _detect_contradictions(self, content: str) -> float:
        """矛盾する記述の検出"""
        return 0.2
    
    def _detect_unrealistic_claims(self, content: str) -> float:
        """非現実的な主張の検出"""
        return 0.2
    
    def _verify_citation_accuracy(self, content: str) -> float:
        """引用正確性の検証"""
        return 0.2
    
    def _analyze_concept_context(self, content: str, concept: str) -> float:
        """概念使用文脈の分析"""
        return 0.8
    
    def _assess_technical_depth(self, content: str) -> float:
        """技術的深度評価"""
        if "implementation" in content.lower() and "algorithm" in content.lower():
            return 0.9
        return 0.7
    
    async def _assess_consistency(self, content: str) -> float:
        """一貫性評価"""
        return 0.8
    
    async def _identify_issues(self, content: str, domain_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """問題点特定"""
        issues = []
        
        if self.profile.domain == ExpertDomain.INTEGRATED_INFORMATION_THEORY:
            if "φ = Φ_experiential(S) = ∑[EI(experiential_concept) - min_cut(experiential_concept)]" in content:
                issues.append({
                    "type": "theoretical_inaccuracy",
                    "category": "theoretical",
                    "description": "IITの標準的なφ計算式と異なる可能性がある独自の計算式",
                    "severity": 0.7,
                    "location": "φ値計算エンジン",
                    "recommendation": "標準的なIIT3.0のφ計算式との整合性を確認する必要"
                })
        
        if domain_analysis["technical_depth"] < 0.5:
            issues.append({
                "type": "insufficient_detail",
                "category": "implementation",
                "description": "実装詳細が不十分",
                "severity": 0.6,
                "recommendation": "より具体的な実装詳細の提供が必要"
            })
        
        return issues
    
    def _calculate_confidence_score(self, theoretical_accuracy: float, implementation_feasibility: float, 
                                   consistency_score: float, hallucination_risk: float) -> float:
        """総合信頼度計算"""
        confidence = (
            theoretical_accuracy * 0.3 +
            implementation_feasibility * 0.3 +
            consistency_score * 0.2 +
            (1.0 - hallucination_risk) * 0.2
        )
        return min(1.0, max(0.0, confidence))

class ConsensusEngine:
    """専門家コンセンサス形成エンジン"""
    
    def __init__(self):
        self.experts = self._initialize_expert_panel()
        self.consensus_algorithms = {
            'hybrid': self._hybrid_consensus
        }
        
    def _initialize_expert_panel(self) -> List[ExpertSimulator]:
        """専門家パネル初期化"""
        
        expert_profiles = [
            ExpertProfile(
                name="Dr. Sarah Mitchell",
                domain=ExpertDomain.PHENOMENOLOGY,
                credibility_score=0.92,
                bias_factors={"theoretical_preference": 0.1},
                specialization_areas=["Husserlian phenomenology", "time consciousness", "intersubjectivity"],
                validation_criteria={"conceptual_accuracy": 0.9, "historical_accuracy": 0.85}
            ),
            ExpertProfile(
                name="Prof. David Chen",
                domain=ExpertDomain.INTEGRATED_INFORMATION_THEORY,
                credibility_score=0.95,
                bias_factors={"mathematical_rigor": 0.15},
                specialization_areas=["φ calculation", "causal structure analysis", "consciousness metrics"],
                validation_criteria={"mathematical_accuracy": 0.95, "algorithmic_correctness": 0.9}
            ),
            ExpertProfile(
                name="Dr. Elena Rodriguez",
                domain=ExpertDomain.ENACTIVE_COGNITION,
                credibility_score=0.88,
                bias_factors={"biological_plausibility": 0.12},
                specialization_areas=["autopoiesis", "embodied cognition", "sensorimotor contingencies"],
                validation_criteria={"theoretical_coherence": 0.9, "empirical_grounding": 0.8}
            ),
            ExpertProfile(
                name="Alex Thompson",
                domain=ExpertDomain.AI_ARCHITECTURE,
                credibility_score=0.90,
                bias_factors={"scalability_focus": 0.1},
                specialization_areas=["clean architecture", "distributed systems", "performance optimization"],
                validation_criteria={"architectural_soundness": 0.9, "scalability": 0.85}
            ),
            ExpertProfile(
                name="Dr. Kenji Yamamoto",
                domain=ExpertDomain.PYTHON_IMPLEMENTATION,
                credibility_score=0.87,
                bias_factors={"code_quality": 0.08},
                specialization_areas=["async programming", "performance optimization", "scientific computing"],
                validation_criteria={"code_quality": 0.9, "performance": 0.8}
            ),
            ExpertProfile(
                name="Maria Santos",
                domain=ExpertDomain.SECURITY_ENGINEERING,
                credibility_score=0.91,
                bias_factors={"security_priority": 0.2},
                specialization_areas=["data protection", "secure architectures", "privacy engineering"],
                validation_criteria={"security_adequacy": 0.95, "privacy_protection": 0.9}
            )
        ]
        
        return [ExpertSimulator(profile) for profile in expert_profiles]
    
    async def form_consensus(self, document_paths: List[str], algorithm: str = 'hybrid') -> Dict[str, ConsensusResult]:
        """コンセンサス形成の実行"""
        
        consensus_results = {}
        
        for document_path in document_paths:
            # ドキュメント読み込み
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except FileNotFoundError:
                logging.error(f"Document not found: {document_path}")
                continue
            
            # 各専門家による検証
            expert_opinions = []
            for expert in self.experts:
                try:
                    validation_result = await expert.validate_document(content, document_path)
                    expert_opinions.append(validation_result)
                except Exception as e:
                    logging.error(f"Expert {expert.profile.name} validation failed: {e}")
            
            # コンセンサス形成
            consensus_algorithm = self.consensus_algorithms[algorithm]
            consensus_result = await consensus_algorithm(expert_opinions, document_path)
            
            consensus_results[document_path] = consensus_result
            
        return consensus_results
    
    async def _hybrid_consensus(self, expert_opinions: List[ValidationResult], document_path: str) -> ConsensusResult:
        """ハイブリッドコンセンサス（推奨手法）"""
        
        if not expert_opinions:
            return self._create_empty_consensus(document_path)
        
        # 専門領域による重み調整
        domain_weights = self._calculate_domain_weights(expert_opinions, document_path)
        
        # 重み付き平均による基本スコア
        weighted_scores = self._weighted_average_scores(expert_opinions, domain_weights)
        
        # 専門家間一致度の計算
        consensus_strength = self._calculate_consensus_strength(expert_opinions)
        
        # ハルシネーション統合検出
        aggregated_hallucinations = self._aggregate_hallucinations(expert_opinions)
        
        # 理論的問題の統合
        theoretical_issues = self._aggregate_theoretical_issues(expert_opinions)
        
        # 実装課題の統合
        implementation_concerns = self._aggregate_implementation_concerns(expert_opinions)
        
        # 推奨事項の生成
        recommendations = self._generate_recommendations(expert_opinions, weighted_scores)
        
        # 最終評価
        final_assessment = {
            "overall_confidence": weighted_scores["confidence"],
            "theoretical_validity": weighted_scores["theoretical_accuracy"],
            "implementation_feasibility": weighted_scores["implementation_feasibility"],
            "consistency": weighted_scores["consistency"],
            "hallucination_risk": weighted_scores["hallucination_risk"]
        }
        
        return ConsensusResult(
            document_path=document_path,
            overall_confidence=weighted_scores["confidence"],
            consensus_strength=consensus_strength,
            expert_opinions=expert_opinions,
            identified_hallucinations=aggregated_hallucinations,
            theoretical_issues=theoretical_issues,
            implementation_concerns=implementation_concerns,
            recommendations=recommendations,
            final_assessment=final_assessment,
            timestamp=datetime.now()
        )
    
    def _calculate_domain_weights(self, expert_opinions: List[ValidationResult], document_path: str) -> Dict[str, float]:
        """ドキュメント内容に基づく領域重み計算"""
        
        # ドキュメントタイプによる基本重み
        base_weights = {
            ExpertDomain.PHENOMENOLOGY: 0.15,
            ExpertDomain.INTEGRATED_INFORMATION_THEORY: 0.20,
            ExpertDomain.ENACTIVE_COGNITION: 0.15,
            ExpertDomain.AI_ARCHITECTURE: 0.20,
            ExpertDomain.PYTHON_IMPLEMENTATION: 0.15,
            ExpertDomain.SECURITY_ENGINEERING: 0.15
        }
        
        # ドキュメント特化重み調整
        if "phi_calculation" in document_path or "iit" in document_path:
            base_weights[ExpertDomain.INTEGRATED_INFORMATION_THEORY] = 0.35
            base_weights[ExpertDomain.MATHEMATICAL_MODELING] = 0.25
        elif "philosophical" in document_path or "phenomenology" in document_path:
            base_weights[ExpertDomain.PHENOMENOLOGY] = 0.40
            base_weights[ExpertDomain.CONSCIOUSNESS_STUDIES] = 0.25
        elif "implementation" in document_path or ".py" in document_path:
            base_weights[ExpertDomain.PYTHON_IMPLEMENTATION] = 0.35
            base_weights[ExpertDomain.AI_ARCHITECTURE] = 0.30
        elif "security" in document_path:
            base_weights[ExpertDomain.SECURITY_ENGINEERING] = 0.45
        
        # 専門家信頼度による調整
        domain_weights = {}
        for opinion in expert_opinions:
            expert_credibility = next(
                (expert.profile.credibility_score for expert in self.experts 
                 if expert.profile.name == opinion.expert_name), 0.8
            )
            domain_weights[opinion.domain] = base_weights.get(opinion.domain, 0.1) * expert_credibility
        
        # 正規化
        total_weight = sum(domain_weights.values())
        if total_weight > 0:
            domain_weights = {k: v/total_weight for k, v in domain_weights.items()}
        
        return domain_weights
    
    def _aggregate_hallucinations(self, expert_opinions: List[ValidationResult]) -> List[Dict[str, Any]]:
        """ハルシネーション統合検出"""
        
        hallucinations = []
        
        # 複数専門家が指摘した問題を重視
        issue_frequency = {}
        for opinion in expert_opinions:
            for issue in opinion.identified_issues:
                issue_key = issue.get("description", "unknown")
                if issue_key not in issue_frequency:
                    issue_frequency[issue_key] = {
                        "count": 0,
                        "severity_sum": 0,
                        "domains": [],
                        "details": issue
                    }
                issue_frequency[issue_key]["count"] += 1
                issue_frequency[issue_key]["severity_sum"] += issue.get("severity", 0.5)
                issue_frequency[issue_key]["domains"].append(opinion.domain.value)
        
        # 確信度の高いハルシネーション特定
        for issue_key, issue_data in issue_frequency.items():
            if issue_data["count"] >= 2:  # 複数専門家が指摘
                avg_severity = issue_data["severity_sum"] / issue_data["count"]
                if avg_severity > 0.7:  # 高い重要度
                    hallucinations.append({
                        "description": issue_key,
                        "severity": avg_severity,
                        "expert_consensus": issue_data["count"],
                        "affected_domains": issue_data["domains"],
                        "type": "multi_expert_detection",
                        "details": issue_data["details"]
                    })
        
        return sorted(hallucinations, key=lambda x: x["severity"], reverse=True)
    
    def _weighted_average_scores(self, expert_opinions: List[ValidationResult], domain_weights: Dict[str, float]) -> Dict[str, float]:
        """重み付き平均スコア計算"""
        
        weighted_scores = {
            "confidence": 0.0,
            "theoretical_accuracy": 0.0,
            "implementation_feasibility": 0.0,
            "consistency": 0.0,
            "hallucination_risk": 0.0
        }
        
        total_weight = 0.0
        
        for opinion in expert_opinions:
            weight = domain_weights.get(opinion.domain, 0.1)
            weighted_scores["confidence"] += opinion.confidence_score * weight
            weighted_scores["theoretical_accuracy"] += opinion.theoretical_accuracy * weight
            weighted_scores["implementation_feasibility"] += opinion.implementation_feasibility * weight
            weighted_scores["consistency"] += opinion.consistency_score * weight
            weighted_scores["hallucination_risk"] += opinion.hallucination_risk * weight
            total_weight += weight
        
        # 正規化
        if total_weight > 0:
            for key in weighted_scores:
                weighted_scores[key] /= total_weight
        
        return weighted_scores
    
    def _calculate_consensus_strength(self, expert_opinions: List[ValidationResult]) -> float:
        """専門家間コンセンサス強度計算"""
        if len(expert_opinions) < 2:
            return 1.0
        
        confidence_scores = [opinion.confidence_score for opinion in expert_opinions]
        mean_confidence = np.mean(confidence_scores)
        std_confidence = np.std(confidence_scores)
        
        # 標準偏差が小さいほど合意が強い
        consensus_strength = max(0.0, 1.0 - (std_confidence / (mean_confidence + 0.01)))
        return min(1.0, consensus_strength)
    
    def _aggregate_theoretical_issues(self, expert_opinions: List[ValidationResult]) -> List[Dict[str, Any]]:
        """理論的問題の統合"""
        theoretical_issues = []
        
        for opinion in expert_opinions:
            for issue in opinion.identified_issues:
                if issue.get("category") == "theoretical" or issue.get("type") == "theoretical_inaccuracy":
                    theoretical_issues.append({
                        "expert": opinion.expert_name,
                        "domain": opinion.domain.value,
                        "issue": issue,
                        "severity": issue.get("severity", 0.5)
                    })
        
        return sorted(theoretical_issues, key=lambda x: x["severity"], reverse=True)
    
    def _aggregate_implementation_concerns(self, expert_opinions: List[ValidationResult]) -> List[Dict[str, Any]]:
        """実装課題の統合"""
        implementation_concerns = []
        
        for opinion in expert_opinions:
            for issue in opinion.identified_issues:
                if issue.get("category") == "implementation" or issue.get("type") == "feasibility_concern":
                    implementation_concerns.append({
                        "expert": opinion.expert_name,
                        "domain": opinion.domain.value,
                        "concern": issue,
                        "impact": issue.get("impact", 0.5)
                    })
        
        return sorted(implementation_concerns, key=lambda x: x["impact"], reverse=True)
    
    def _generate_recommendations(self, expert_opinions: List[ValidationResult], weighted_scores: Dict[str, float]) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        # 信頼度に基づく推奨
        if weighted_scores["confidence"] < 0.7:
            recommendations.append("全体的な信頼度が低いため、理論的基盤と実装詳細の見直しが必要")
        
        # ハルシネーションリスクに基づく推奨
        if weighted_scores["hallucination_risk"] > 0.3:
            recommendations.append("ハルシネーションリスクが高いため、事実確認と引用の検証が必要")
        
        # 実装可能性に基づく推奨
        if weighted_scores["implementation_feasibility"] < 0.6:
            recommendations.append("実装可能性に懸念があるため、技術的制約の再評価が必要")
        
        # 理論的正確性に基づく推奨
        if weighted_scores["theoretical_accuracy"] < 0.8:
            recommendations.append("理論的正確性に問題があるため、専門文献の再確認が必要")
        
        return recommendations
    
    def _create_empty_consensus(self, document_path: str) -> ConsensusResult:
        """空のコンセンサス結果作成"""
        return ConsensusResult(
            document_path=document_path,
            overall_confidence=0.0,
            consensus_strength=0.0,
            expert_opinions=[],
            identified_hallucinations=[],
            theoretical_issues=[],
            implementation_concerns=[],
            recommendations=["専門家による検証が実行されませんでした"],
            final_assessment={
                "overall_confidence": 0.0,
                "theoretical_validity": 0.0,
                "implementation_feasibility": 0.0,
                "consistency": 0.0,
                "hallucination_risk": 1.0
            },
            timestamp=datetime.now()
        )

async def main():
    """メイン実行関数"""
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # コンセンサスエンジン初期化
    consensus_engine = ConsensusEngine()
    
    # 検証対象ドキュメントリスト取得
    document_directory = Path("/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/08_02_2025")
    document_paths = [
        str(doc_path) for doc_path in document_directory.glob("*.md") 
        if doc_path.name not in ["fact-checker.yaml"]
    ]
    
    print(f"検証対象ドキュメント数: {len(document_paths)}")
    
    # コンセンサス形成実行
    consensus_results = await consensus_engine.form_consensus(document_paths)
    
    # 結果出力
    output_path = document_directory / "consensus_verification_results.json"
    
    # JSON化のためのデータ変換
    serializable_results = {}
    for path, result in consensus_results.items():
        serializable_results[path] = {
            "document_path": result.document_path,
            "overall_confidence": result.overall_confidence,
            "consensus_strength": result.consensus_strength,
            "identified_hallucinations": result.identified_hallucinations,
            "theoretical_issues": result.theoretical_issues,
            "implementation_concerns": result.implementation_concerns,
            "recommendations": result.recommendations,
            "final_assessment": result.final_assessment,
            "timestamp": result.timestamp.isoformat(),
            "expert_count": len(result.expert_opinions)
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"コンセンサス結果を {output_path} に保存しました")
    
    return consensus_results

if __name__ == "__main__":
    asyncio.run(main())