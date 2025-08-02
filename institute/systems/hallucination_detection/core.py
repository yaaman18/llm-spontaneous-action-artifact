"""
多層ハルシネーション検出システム
Semantic Entropy法とMulti-Agent検証を統合した高精度検出システム
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from enum import Enum
import hashlib

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class HallucinationType(Enum):
    FACTUAL_ERROR = "factual_error"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    DOMAIN_MISMATCH = "domain_mismatch"
    CONFABULATION = "confabulation"
    SOURCE_HALLUCINATION = "source_hallucination"

@dataclass
class DetectionResult:
    statement_id: str
    original_text: str
    is_hallucination: bool
    confidence_score: float
    hallucination_type: Optional[HallucinationType]
    evidence: List[str]
    corrected_text: Optional[str]
    sources: List[str]
    timestamp: datetime
    detector_agents: List[str]
    semantic_entropy: float
    consensus_score: float

@dataclass
class AgentVerification:
    agent_name: str
    specialist_domain: str
    verification_score: float
    findings: str
    sources_checked: List[str]
    confidence: ConfidenceLevel

class SemanticEntropyDetector:
    """セマンティックエントロピー法による幻覚検出"""
    
    def __init__(self):
        self.threshold = 0.5  # 研究に基づく閾値
        
    async def calculate_semantic_entropy(self, 
                                       statement: str, 
                                       context: str = None) -> float:
        """セマンティックエントロピーを計算"""
        
        # 複数の表現バリエーションを生成
        variations = await self._generate_semantic_variations(statement)
        
        # 各バリエーションの意味的一貫性を評価
        consistency_scores = []
        for variation in variations:
            score = await self._evaluate_semantic_consistency(
                statement, variation, context
            )
            consistency_scores.append(score)
        
        # エントロピー計算
        entropy = self._calculate_entropy(consistency_scores)
        return entropy
    
    async def _generate_semantic_variations(self, statement: str) -> List[str]:
        """意味を保持した複数のバリエーションを生成"""
        # 実装では複数のLLMを使用してバリエーションを生成
        variations = [
            statement,  # 元の文
            # 以下、実際の実装では複数のパラフレーズを生成
        ]
        return variations
    
    async def _evaluate_semantic_consistency(self, 
                                           original: str, 
                                           variation: str, 
                                           context: str = None) -> float:
        """意味的一貫性を評価"""
        # 実装: embedding距離、論理的整合性など
        return 0.8  # プレースホルダー
    
    def _calculate_entropy(self, scores: List[float]) -> float:
        """情報エントロピーを計算"""
        if not scores:
            return 1.0
        
        # スコアを確率分布に変換
        scores_array = np.array(scores)
        if np.sum(scores_array) == 0:
            return 1.0
            
        probabilities = scores_array / np.sum(scores_array)
        
        # エントロピー計算
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)

class MultiAgentVerifier:
    """複数の専門エージェントによる検証システム"""
    
    def __init__(self, agents_config: Dict[str, Dict]):
        self.agents = agents_config
        self.verification_threshold = 0.7
        
    async def verify_statement(self, 
                             statement: str, 
                             context: str = None,
                             domain_hint: str = None) -> List[AgentVerification]:
        """複数エージェントによる並列検証"""
        
        # 関連する専門エージェントを選択
        relevant_agents = self._select_relevant_agents(statement, domain_hint)
        
        # 並列検証実行
        verification_tasks = []
        for agent_name in relevant_agents:
            task = self._verify_with_agent(agent_name, statement, context)
            verification_tasks.append(task)
        
        verifications = await asyncio.gather(*verification_tasks)
        return verifications
    
    def _select_relevant_agents(self, 
                              statement: str, 
                              domain_hint: str = None) -> List[str]:
        """文に関連する専門エージェントを選択"""
        
        # ドメイン分析
        detected_domains = self._analyze_domain(statement)
        if domain_hint:
            detected_domains.append(domain_hint)
        
        # 関連エージェント選択
        relevant = []
        for agent_name, config in self.agents.items():
            agent_domains = config.get('expertise', {}).get('primary', [])
            if any(domain in agent_domains for domain in detected_domains):
                relevant.append(agent_name)
        
        return relevant[:5]  # 最大5エージェント
    
    def _analyze_domain(self, statement: str) -> List[str]:
        """文の専門分野を分析"""
        # キーワードベースの簡易ドメイン分析
        domain_keywords = {
            'consciousness': ['意識', '現象学', 'consciousness', 'phenomenology'],
            'neuroscience': ['神経', 'neural', 'brain', '脳'],
            'philosophy': ['哲学', 'philosophy', '存在', '実在'],
            'mathematics': ['数学', '計算', 'mathematical', 'computation'],
            'physics': ['物理', 'physics', '量子', 'quantum']
        }
        
        detected = []
        for domain, keywords in domain_keywords.items():
            if any(keyword in statement.lower() for keyword in keywords):
                detected.append(domain)
        
        return detected or ['general']
    
    async def _verify_with_agent(self, 
                                agent_name: str, 
                                statement: str, 
                                context: str = None) -> AgentVerification:
        """特定エージェントによる検証"""
        
        agent_config = self.agents[agent_name]
        
        # エージェント固有の検証プロセス
        verification_score = await self._agent_specific_verification(
            agent_name, statement, context, agent_config
        )
        
        # 検証結果構築
        return AgentVerification(
            agent_name=agent_name,
            specialist_domain=agent_config.get('department', 'unknown'),
            verification_score=verification_score,
            findings=f"Agent {agent_name} verification completed",
            sources_checked=[],  # 実装で追加
            confidence=self._score_to_confidence(verification_score)
        )
    
    async def _agent_specific_verification(self, 
                                         agent_name: str, 
                                         statement: str, 
                                         context: str, 
                                         config: Dict) -> float:
        """エージェント固有の検証ロジック"""
        # 実装: 各エージェントの専門性に基づく検証
        return 0.8  # プレースホルダー
    
    def _score_to_confidence(self, score: float) -> ConfidenceLevel:
        """スコアを信頼度レベルに変換"""
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN

class HallucinationDetectionEngine:
    """統合ハルシネーション検出エンジン"""
    
    def __init__(self, agents_config: Dict[str, Dict]):
        self.semantic_detector = SemanticEntropyDetector()
        self.multi_agent_verifier = MultiAgentVerifier(agents_config)
        self.detection_history: List[DetectionResult] = []
        
    async def detect_hallucination(self, 
                                 statement: str, 
                                 context: str = None,
                                 domain_hint: str = None) -> DetectionResult:
        """包括的ハルシネーション検出"""
        
        statement_id = self._generate_statement_id(statement)
        
        # 並列検出実行
        semantic_task = self.semantic_detector.calculate_semantic_entropy(
            statement, context
        )
        verification_task = self.multi_agent_verifier.verify_statement(
            statement, context, domain_hint
        )
        
        semantic_entropy, verifications = await asyncio.gather(
            semantic_task, verification_task
        )
        
        # 結果統合
        detection_result = self._integrate_results(
            statement_id, statement, semantic_entropy, verifications
        )
        
        # 履歴記録
        self.detection_history.append(detection_result)
        
        return detection_result
    
    def _generate_statement_id(self, statement: str) -> str:
        """文のユニークIDを生成"""
        return hashlib.md5(statement.encode()).hexdigest()[:12]
    
    def _integrate_results(self, 
                         statement_id: str, 
                         statement: str, 
                         semantic_entropy: float, 
                         verifications: List[AgentVerification]) -> DetectionResult:
        """検出結果を統合"""
        
        # コンセンサススコア計算
        if verifications:
            consensus_score = np.mean([v.verification_score for v in verifications])
        else:
            consensus_score = 0.5
        
        # 総合判定
        is_hallucination = self._make_final_decision(
            semantic_entropy, consensus_score, verifications
        )
        
        # 信頼度スコア計算
        confidence_score = self._calculate_confidence_score(
            semantic_entropy, consensus_score, len(verifications)
        )
        
        # 幻覚タイプ特定
        hallucination_type = self._identify_hallucination_type(
            verifications, semantic_entropy
        ) if is_hallucination else None
        
        return DetectionResult(
            statement_id=statement_id,
            original_text=statement,
            is_hallucination=is_hallucination,
            confidence_score=confidence_score,
            hallucination_type=hallucination_type,
            evidence=self._collect_evidence(verifications),
            corrected_text=None,  # 後で修正システムで実装
            sources=self._collect_sources(verifications),
            timestamp=datetime.now(),
            detector_agents=[v.agent_name for v in verifications],
            semantic_entropy=semantic_entropy,
            consensus_score=consensus_score
        )
    
    def _make_final_decision(self, 
                           semantic_entropy: float, 
                           consensus_score: float, 
                           verifications: List[AgentVerification]) -> bool:
        """最終的なハルシネーション判定"""
        
        # セマンティックエントロピーによる判定
        entropy_indicates_hallucination = semantic_entropy > 0.5
        
        # コンセンサスによる判定
        consensus_indicates_accuracy = consensus_score > 0.7
        
        # 高信頼度エージェントの意見を重視
        high_confidence_verifications = [
            v for v in verifications 
            if v.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
        ]
        
        if len(high_confidence_verifications) >= 2:
            expert_consensus = np.mean([
                v.verification_score for v in high_confidence_verifications
            ])
            if expert_consensus < 0.4:  # 専門家が疑問視
                return True
        
        # 総合判定
        return entropy_indicates_hallucination and not consensus_indicates_accuracy
    
    def _calculate_confidence_score(self, 
                                  semantic_entropy: float, 
                                  consensus_score: float, 
                                  num_verifiers: int) -> float:
        """信頼度スコアを計算"""
        
        # セマンティック信頼度
        semantic_confidence = 1.0 - semantic_entropy
        
        # コンセンサス信頼度
        consensus_confidence = consensus_score
        
        # 検証者数による重み
        verifier_weight = min(1.0, num_verifiers / 3.0)
        
        # 総合信頼度
        overall_confidence = (
            semantic_confidence * 0.4 + 
            consensus_confidence * 0.4 + 
            verifier_weight * 0.2
        )
        
        return float(np.clip(overall_confidence, 0.0, 1.0))
    
    def _identify_hallucination_type(self, 
                                   verifications: List[AgentVerification], 
                                   semantic_entropy: float) -> HallucinationType:
        """幻覚のタイプを特定"""
        
        # 簡易実装 - 実際にはより詳細な分析が必要
        if semantic_entropy > 0.8:
            return HallucinationType.CONFABULATION
        
        # エージェントの指摘内容から判定
        for verification in verifications:
            if "factual" in verification.findings.lower():
                return HallucinationType.FACTUAL_ERROR
            elif "logical" in verification.findings.lower():
                return HallucinationType.LOGICAL_INCONSISTENCY
        
        return HallucinationType.FACTUAL_ERROR
    
    def _collect_evidence(self, verifications: List[AgentVerification]) -> List[str]:
        """検証証拠を収集"""
        evidence = []
        for verification in verifications:
            if verification.findings:
                evidence.append(f"{verification.agent_name}: {verification.findings}")
        return evidence
    
    def _collect_sources(self, verifications: List[AgentVerification]) -> List[str]:
        """情報源を収集"""
        sources = []
        for verification in verifications:
            sources.extend(verification.sources_checked)
        return list(set(sources))  # 重複除去
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """検出結果のサマリーを取得"""
        if not self.detection_history:
            return {"total_checks": 0, "hallucination_rate": 0.0}
        
        total_checks = len(self.detection_history)
        hallucinations = sum(1 for r in self.detection_history if r.is_hallucination)
        
        return {
            "total_checks": total_checks,
            "hallucination_rate": hallucinations / total_checks,
            "average_confidence": np.mean([r.confidence_score for r in self.detection_history]),
            "most_common_type": self._get_most_common_hallucination_type()
        }
    
    def _get_most_common_hallucination_type(self) -> Optional[str]:
        """最も多い幻覚タイプを取得"""
        types = [r.hallucination_type for r in self.detection_history if r.hallucination_type]
        if not types:
            return None
        
        from collections import Counter
        return Counter(types).most_common(1)[0][0].value

# 使用例とテスト
async def main():
    """システムテスト用メイン関数"""
    
    # エージェント設定（実際のYAMLから読み込み）
    agents_config = {
        "tononi-koch": {
            "department": "theoretical-foundations",
            "expertise": {"primary": ["consciousness", "mathematics"]}
        },
        "zahavi": {
            "department": "philosophy-ethics", 
            "expertise": {"primary": ["philosophy", "phenomenology"]}
        }
    }
    
    # 検出エンジン初期化
    detector = HallucinationDetectionEngine(agents_config)
    
    # テスト文
    test_statement = "統合情報理論によると、意識は量子もつれによって生まれる"
    
    # 検出実行
    result = await detector.detect_hallucination(
        test_statement, 
        context="意識理論の議論中",
        domain_hint="consciousness"
    )
    
    print(f"Statement: {result.original_text}")
    print(f"Is Hallucination: {result.is_hallucination}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Semantic Entropy: {result.semantic_entropy:.2f}")
    print(f"Evidence: {result.evidence}")

if __name__ == "__main__":
    asyncio.run(main())