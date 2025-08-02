"""
分野特化知識検証システム
各専門分野の深層知識を検証する専門チェッカー
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import re
import numpy as np

class VerificationLevel(Enum):
    SURFACE = "surface"      # 表面的検証
    SHALLOW = "shallow"      # 浅い検証  
    MODERATE = "moderate"    # 中程度検証
    DEEP = "deep"           # 深い検証
    EXPERT = "expert"       # 専門家レベル

class KnowledgeType(Enum):
    FACTUAL = "factual"           # 事実
    THEORETICAL = "theoretical"   # 理論
    METHODOLOGICAL = "methodological"  # 方法論
    EMPIRICAL = "empirical"       # 実証的
    PHILOSOPHICAL = "philosophical"  # 哲学的
    MATHEMATICAL = "mathematical"  # 数学的

@dataclass
class VerificationCriteria:
    domain: str
    knowledge_type: KnowledgeType
    verification_level: VerificationLevel
    required_sources: List[str]
    critical_concepts: List[str]
    common_misconceptions: List[str]
    validation_rules: List[str]

@dataclass
class DomainVerificationResult:
    domain: str
    statement: str
    is_valid: bool
    confidence_score: float
    verification_level: VerificationLevel
    findings: List[str]
    corrections: List[str]
    supporting_references: List[str]
    red_flags: List[str]
    specialist_notes: str
    timestamp: datetime

class BaseDomainSpecialist(ABC):
    """分野専門家基底クラス"""
    
    def __init__(self, domain: str, expertise_config: Dict[str, Any]):
        self.domain = domain
        self.expertise_config = expertise_config
        self.verification_history: List[DomainVerificationResult] = []
        
        # 専門知識データベース
        self.core_concepts = expertise_config.get('core_concepts', [])
        self.key_researchers = expertise_config.get('key_researchers', [])
        self.foundational_papers = expertise_config.get('foundational_papers', [])
        self.common_errors = expertise_config.get('common_errors', [])
        self.validation_patterns = expertise_config.get('validation_patterns', [])
    
    @abstractmethod
    async def verify_statement(self, 
                             statement: str, 
                             context: str = None,
                             verification_level: VerificationLevel = VerificationLevel.MODERATE) -> DomainVerificationResult:
        """文の専門的検証を実行"""
        pass
    
    @abstractmethod
    def get_domain_keywords(self) -> List[str]:
        """ドメイン固有キーワードを取得"""
        pass
    
    @abstractmethod
    def check_conceptual_accuracy(self, statement: str) -> Tuple[bool, List[str]]:
        """概念的正確性をチェック"""
        pass
    
    def detect_common_misconceptions(self, statement: str) -> List[str]:
        """よくある誤解を検出"""
        detected_errors = []
        
        for error_pattern in self.common_errors:
            if self._matches_error_pattern(statement, error_pattern):
                detected_errors.append(error_pattern['description'])
        
        return detected_errors
    
    def _matches_error_pattern(self, statement: str, pattern: Dict[str, Any]) -> bool:
        """エラーパターンとのマッチング"""
        keywords = pattern.get('keywords', [])
        return any(keyword.lower() in statement.lower() for keyword in keywords)
    
    def calculate_domain_relevance(self, statement: str) -> float:
        """ドメイン関連度を計算"""
        domain_keywords = self.get_domain_keywords()
        statement_words = set(statement.lower().split())
        
        matches = sum(1 for keyword in domain_keywords 
                     if keyword.lower() in statement_words)
        
        return min(1.0, matches / max(1, len(domain_keywords) * 0.3))

class ConsciousnessSpecialist(BaseDomainSpecialist):
    """意識研究専門家"""
    
    def __init__(self):
        expertise_config = {
            'core_concepts': [
                'integrated information theory', 'IIT', 'phi value', 'Φ',
                'global workspace theory', 'GWT', 'consciousness',
                'phenomenology', 'qualia', 'hard problem',
                'temporal consciousness', 'intentionality', 'awareness'
            ],
            'key_researchers': [
                'Giulio Tononi', 'Christof Koch', 'Bernard Baars',
                'David Chalmers', 'Dan Zahavi', 'Edmund Husserl',
                'Thomas Nagel', 'Stanislas Dehaene'
            ],
            'foundational_papers': [
                'IIT 4.0', 'Consciousness and Complexity',
                'The Global Workspace Theory', 'Facing Up to the Hard Problem'
            ],
            'common_errors': [
                {
                    'description': 'Confusing consciousness with intelligence',
                    'keywords': ['conscious AI', 'intelligent therefore conscious']
                },
                {
                    'description': 'Misunderstanding phi value calculation',
                    'keywords': ['phi equals', 'consciousness level', 'simple addition']
                },
                {
                    'description': 'Quantum consciousness misconception', 
                    'keywords': ['quantum consciousness', 'quantum entanglement consciousness']
                }
            ],
            'validation_patterns': [
                'IIT theoretical consistency',
                'Phenomenological accuracy',
                'Neuroscientific grounding'
            ]
        }
        super().__init__('consciousness', expertise_config)
    
    async def verify_statement(self, 
                             statement: str, 
                             context: str = None,
                             verification_level: VerificationLevel = VerificationLevel.MODERATE) -> DomainVerificationResult:
        """意識研究文の検証"""
        
        findings = []
        corrections = []
        red_flags = []
        supporting_refs = []
        
        # 1. 基本的概念チェック
        is_conceptually_accurate, concept_issues = self.check_conceptual_accuracy(statement)
        if not is_conceptually_accurate:
            findings.extend(concept_issues)
        
        # 2. よくある誤解検出
        misconceptions = self.detect_common_misconceptions(statement)
        red_flags.extend(misconceptions)
        
        # 3. IIT特化検証
        iit_validation = await self._validate_iit_claims(statement)
        findings.extend(iit_validation['findings'])
        if iit_validation['corrections']:
            corrections.extend(iit_validation['corrections'])
        
        # 4. 現象学的妥当性チェック
        if verification_level in [VerificationLevel.DEEP, VerificationLevel.EXPERT]:
            phenom_check = await self._check_phenomenological_consistency(statement)
            findings.extend(phenom_check)
        
        # 5. 参考文献の妥当性
        if verification_level == VerificationLevel.EXPERT:
            ref_validation = await self._validate_references(statement)
            supporting_refs.extend(ref_validation)
        
        # 総合判定
        is_valid = self._make_consciousness_validity_decision(
            is_conceptually_accurate, misconceptions, iit_validation
        )
        
        confidence_score = self._calculate_consciousness_confidence(
            is_conceptually_accurate, len(misconceptions), 
            len(findings), verification_level
        )
        
        specialist_notes = self._generate_consciousness_notes(
            statement, findings, corrections, red_flags
        )
        
        return DomainVerificationResult(
            domain='consciousness',
            statement=statement,
            is_valid=is_valid,
            confidence_score=confidence_score,
            verification_level=verification_level,
            findings=findings,
            corrections=corrections,
            supporting_references=supporting_refs,
            red_flags=red_flags,
            specialist_notes=specialist_notes,
            timestamp=datetime.now()
        )
    
    def get_domain_keywords(self) -> List[str]:
        return self.core_concepts
    
    def check_conceptual_accuracy(self, statement: str) -> Tuple[bool, List[str]]:
        """意識研究の概念的正確性チェック"""
        issues = []
        
        # IIT関連チェック
        if 'phi' in statement.lower() or 'φ' in statement:
            if 'quantum' in statement.lower():
                issues.append("IITのΦ値は量子効果とは直接関係ありません")
        
        # GWT関連チェック
        if 'global workspace' in statement.lower():
            if 'unconscious' not in statement.lower():
                issues.append("GWTは意識と無意識の区別が重要です")
        
        # 現象学関連チェック
        if 'phenomenology' in statement.lower():
            if 'qualia' not in statement.lower() and 'intentionality' not in statement.lower():
                issues.append("現象学には質感や志向性の概念が重要です")
        
        return len(issues) == 0, issues
    
    async def _validate_iit_claims(self, statement: str) -> Dict[str, Any]:
        """IIT関連主張の検証"""
        findings = []
        corrections = []
        
        # Φ値に関する主張
        if re.search(r'phi|φ', statement, re.IGNORECASE):
            phi_patterns = [
                (r'phi.*equals.*(\d+)', 'Φ値は具体的な数値ではなく、システムに依存します'),
                (r'consciousness.*level.*phi', 'Φ値は意識レベルを直接表すものではありません'),
                (r'phi.*simple.*addition', 'Φ値の計算は単純な加算ではありません')
            ]
            
            for pattern, correction in phi_patterns:
                if re.search(pattern, statement, re.IGNORECASE):
                    findings.append(f"IIT理解の問題: {correction}")
                    corrections.append(correction)
        
        # 統合情報に関する主張
        if 'integrated information' in statement.lower():
            if 'complexity' in statement.lower() and 'simple' in statement.lower():
                findings.append("統合情報は複雑性だけでなく統合性が重要です")
        
        return {'findings': findings, 'corrections': corrections}
    
    async def _check_phenomenological_consistency(self, statement: str) -> List[str]:
        """現象学的一貫性チェック"""
        findings = []
        
        phenom_concepts = ['intentionality', 'temporal consciousness', 'intersubjectivity']
        
        for concept in phenom_concepts:
            if concept in statement.lower():
                # より詳細な検証ロジック
                if concept == 'temporal consciousness':
                    if not any(term in statement.lower() 
                             for term in ['retention', 'protention', 'present']):
                        findings.append("時間意識には把持・予持・現在の三重構造が重要です")
        
        return findings
    
    async def _validate_references(self, statement: str) -> List[str]:
        """参考文献の妥当性検証"""
        references = []
        
        # 研究者名の検出と対応論文の提案
        for researcher in self.key_researchers:
            if researcher.lower() in statement.lower():
                if researcher == 'Giulio Tononi':
                    references.append("Tononi, G. (2008). Consciousness and complexity")
                elif researcher == 'David Chalmers':
                    references.append("Chalmers, D. (1995). Facing up to the hard problem")
        
        return references
    
    def _make_consciousness_validity_decision(self, 
                                            conceptual_accuracy: bool,
                                            misconceptions: List[str],
                                            iit_validation: Dict[str, Any]) -> bool:
        """意識研究文の妥当性判定"""
        
        # 重大な概念エラーがある場合は無効
        if not conceptual_accuracy:
            return False
        
        # よくある誤解が多い場合は無効
        if len(misconceptions) >= 2:
            return False
        
        # IIT関連で重大なエラーがある場合
        if len(iit_validation['findings']) >= 3:
            return False
        
        return True
    
    def _calculate_consciousness_confidence(self, 
                                          conceptual_accuracy: bool,
                                          num_misconceptions: int,
                                          num_findings: int,
                                          verification_level: VerificationLevel) -> float:
        """意識研究の信頼度計算"""
        
        base_confidence = 0.8 if conceptual_accuracy else 0.3
        
        # 誤解によるペナルティ
        misconception_penalty = num_misconceptions * 0.2
        
        # 検証レベルボーナス
        level_bonus = {
            VerificationLevel.SURFACE: 0.0,
            VerificationLevel.SHALLOW: 0.05,
            VerificationLevel.MODERATE: 0.1,
            VerificationLevel.DEEP: 0.15,
            VerificationLevel.EXPERT: 0.2
        }.get(verification_level, 0.0)
        
        # 指摘事項によるペナルティ
        findings_penalty = min(0.4, num_findings * 0.1)
        
        confidence = base_confidence - misconception_penalty + level_bonus - findings_penalty
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _generate_consciousness_notes(self, 
                                    statement: str,
                                    findings: List[str],
                                    corrections: List[str],
                                    red_flags: List[str]) -> str:
        """意識研究専門家ノートを生成"""
        
        notes = []
        
        if red_flags:
            notes.append("⚠️ 検出された問題:")
            notes.extend([f"  - {flag}" for flag in red_flags])
        
        if corrections:
            notes.append("\n✓ 推奨修正:")
            notes.extend([f"  - {correction}" for correction in corrections])
        
        if findings:
            notes.append("\n📝 専門的所見:")
            notes.extend([f"  - {finding}" for finding in findings])
        
        # 改善提案
        notes.append("\n💡 改善提案:")
        if 'IIT' in statement or 'phi' in statement.lower():
            notes.append("  - IIT 4.0の最新理論を参照することを推奨")
        if 'phenomenology' in statement.lower():
            notes.append("  - フッサールの時間意識分析を参考に")
        
        return '\n'.join(notes)

class PhilosophySpecialist(BaseDomainSpecialist):
    """哲学専門家"""
    
    def __init__(self):
        expertise_config = {
            'core_concepts': [
                'ontology', 'epistemology', 'metaphysics', 'logic',
                'existence', 'being', 'reality', 'truth', 'knowledge',
                'mind-body problem', 'free will', 'personal identity'
            ],
            'key_researchers': [
                'Aristotle', 'Immanuel Kant', 'Edmund Husserl',
                'Martin Heidegger', 'Ludwig Wittgenstein', 'John Searle',
                'Daniel Dennett', 'Thomas Nagel'
            ],
            'common_errors': [
                {
                    'description': 'Category mistake in mind-body discussion',
                    'keywords': ['mind is brain', 'brain creates mind']
                },
                {
                    'description': 'Logical fallacy in argumentation',
                    'keywords': ['therefore proves', 'obviously true']
                }
            ]
        }
        super().__init__('philosophy', expertise_config)
    
    async def verify_statement(self, 
                             statement: str, 
                             context: str = None,
                             verification_level: VerificationLevel = VerificationLevel.MODERATE) -> DomainVerificationResult:
        """哲学的文の検証"""
        
        findings = []
        corrections = []
        red_flags = []
        
        # 論理的一貫性チェック
        logical_check = await self._check_logical_consistency(statement)
        findings.extend(logical_check)
        
        # 概念的明確性チェック
        is_conceptually_clear, clarity_issues = self.check_conceptual_accuracy(statement)
        findings.extend(clarity_issues)
        
        # 論証構造チェック
        if verification_level in [VerificationLevel.DEEP, VerificationLevel.EXPERT]:
            argument_analysis = await self._analyze_argument_structure(statement)
            findings.extend(argument_analysis)
        
        is_valid = is_conceptually_clear and len(findings) <= 2
        confidence_score = max(0.1, 0.9 - len(findings) * 0.15)
        
        return DomainVerificationResult(
            domain='philosophy',
            statement=statement,
            is_valid=is_valid,
            confidence_score=confidence_score,
            verification_level=verification_level,
            findings=findings,
            corrections=corrections,
            supporting_references=[],
            red_flags=red_flags,
            specialist_notes=f"哲学的分析完了: {len(findings)}件の指摘事項",
            timestamp=datetime.now()
        )
    
    def get_domain_keywords(self) -> List[str]:
        return self.core_concepts
    
    def check_conceptual_accuracy(self, statement: str) -> Tuple[bool, List[str]]:
        """哲学的概念の正確性チェック"""
        issues = []
        
        # 存在論的チェック
        if 'existence' in statement.lower() or 'being' in statement.lower():
            if 'physical' in statement.lower() and 'only' in statement.lower():
                issues.append("存在論は物理的存在に限定されません")
        
        return len(issues) == 0, issues
    
    async def _check_logical_consistency(self, statement: str) -> List[str]:
        """論理的一貫性チェック"""
        issues = []
        
        # 矛盾検出
        contradictory_pairs = [
            (['all', 'every'], ['some', 'not all']),
            (['necessary'], ['contingent', 'might not']),
            (['impossible'], ['possible'])
        ]
        
        statement_lower = statement.lower()
        for pos_terms, neg_terms in contradictory_pairs:
            has_positive = any(term in statement_lower for term in pos_terms)
            has_negative = any(term in statement_lower for term in neg_terms)
            
            if has_positive and has_negative:
                issues.append("論理的矛盾の可能性があります")
        
        return issues
    
    async def _analyze_argument_structure(self, statement: str) -> List[str]:
        """論証構造分析"""
        analysis = []
        
        # 前提と結論の識別
        if 'therefore' in statement.lower() or 'thus' in statement.lower():
            if 'because' not in statement.lower() and 'since' not in statement.lower():
                analysis.append("結論はありますが、明確な前提が不足しています")
        
        return analysis

class MathematicsSpecialist(BaseDomainSpecialist):
    """数学専門家"""
    
    def __init__(self):
        expertise_config = {
            'core_concepts': [
                'theorem', 'proof', 'axiom', 'lemma', 'corollary',
                'set theory', 'topology', 'algebra', 'analysis',
                'probability', 'statistics', 'logic', 'computation'
            ],
            'common_errors': [
                {
                    'description': 'Division by zero',
                    'keywords': ['divide by zero', '/0']
                },
                {
                    'description': 'Correlation implies causation',
                    'keywords': ['correlation proves', 'statistics show causation']
                }
            ]
        }
        super().__init__('mathematics', expertise_config)
    
    async def verify_statement(self, 
                             statement: str, 
                             context: str = None,
                             verification_level: VerificationLevel = VerificationLevel.MODERATE) -> DomainVerificationResult:
        """数学的文の検証"""
        
        findings = []
        corrections = []
        
        # 数学的記法チェック
        notation_check = await self._check_mathematical_notation(statement)
        findings.extend(notation_check)
        
        # 証明構造チェック
        if verification_level >= VerificationLevel.DEEP:
            proof_analysis = await self._analyze_proof_structure(statement)
            findings.extend(proof_analysis)
        
        is_valid = len(findings) <= 1
        confidence_score = max(0.1, 0.95 - len(findings) * 0.2)
        
        return DomainVerificationResult(
            domain='mathematics',
            statement=statement,
            is_valid=is_valid,
            confidence_score=confidence_score,
            verification_level=verification_level,
            findings=findings,
            corrections=corrections,
            supporting_references=[],
            red_flags=[],
            specialist_notes=f"数学的検証完了: 信頼度 {confidence_score:.2f}",
            timestamp=datetime.now()
        )
    
    def get_domain_keywords(self) -> List[str]:
        return self.core_concepts
    
    def check_conceptual_accuracy(self, statement: str) -> Tuple[bool, List[str]]:
        """数学的概念の正確性チェック"""
        issues = []
        
        # 除算ゼロチェック
        if '/0' in statement or 'divide by zero' in statement.lower():
            issues.append("ゼロ除算は数学的に未定義です")
        
        return len(issues) == 0, issues
    
    async def _check_mathematical_notation(self, statement: str) -> List[str]:
        """数学的記法チェック"""
        issues = []
        
        # 不適切な等号使用
        if '=' in statement:
            # 簡易チェック: 等号の前後に数式があるか
            equals_contexts = re.findall(r'.{5}=.{5}', statement)
            for context in equals_contexts:
                if not re.search(r'\d|[a-zA-Z]', context):
                    issues.append("等号の使用に問題がある可能性があります")
        
        return issues
    
    async def _analyze_proof_structure(self, statement: str) -> List[str]:
        """証明構造分析"""
        analysis = []
        
        proof_keywords = ['theorem', 'proof', 'qed', 'lemma']
        has_proof_structure = any(keyword in statement.lower() for keyword in proof_keywords)
        
        if has_proof_structure:
            if 'proof:' not in statement.lower() and 'proof.' not in statement.lower():
                analysis.append("定理の記述がありますが、証明が不明確です")
        
        return analysis

class DomainSpecialistFactory:
    """分野専門家ファクトリー"""
    
    _specialists = {
        'consciousness': ConsciousnessSpecialist,
        'philosophy': PhilosophySpecialist,
        'mathematics': MathematicsSpecialist,
    }
    
    @classmethod
    def create_specialist(cls, domain: str) -> BaseDomainSpecialist:
        """専門家インスタンスを作成"""
        specialist_class = cls._specialists.get(domain)
        if not specialist_class:
            raise ValueError(f"Unknown domain: {domain}")
        return specialist_class()
    
    @classmethod
    def get_available_domains(cls) -> List[str]:
        """利用可能な分野を取得"""
        return list(cls._specialists.keys())

# 使用例
async def main():
    """分野専門家システムテスト"""
    
    # 意識研究専門家のテスト
    consciousness_expert = DomainSpecialistFactory.create_specialist('consciousness')
    
    test_statement = "統合情報理論では、意識のレベルはΦ値で測定され、Φ=5なら高い意識レベルを示す"
    
    result = await consciousness_expert.verify_statement(
        test_statement, 
        verification_level=VerificationLevel.DEEP
    )
    
    print(f"Domain: {result.domain}")
    print(f"Valid: {result.is_valid}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Findings: {result.findings}")
    print(f"Specialist Notes:\n{result.specialist_notes}")

if __name__ == "__main__":
    asyncio.run(main())