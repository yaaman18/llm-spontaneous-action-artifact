"""
NewbornAI 2.0: Clean Architecture Redesign
Robert C. Martin (Uncle Bob) による SOLID原則適用設計

設計原則:
1. Single Responsibility Principle (SRP)
2. Open/Closed Principle (OCP) 
3. Liskov Substitution Principle (LSP)
4. Interface Segregation Principle (ISP)
5. Dependency Inversion Principle (DIP)
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Dict, List, Optional, Any
from enum import Enum
import datetime

# ===== 1. SRP: 責任の分離 =====

class ConsciousnessLevel:
    """意識レベルの値オブジェクト (Value Object)"""
    def __init__(self, phi_value: float):
        if phi_value < 0:
            raise ValueError("φ値は0以上である必要があります")
        self._phi_value = phi_value
    
    @property 
    def value(self) -> float:
        return self._phi_value
    
    def __eq__(self, other) -> bool:
        return isinstance(other, ConsciousnessLevel) and self._phi_value == other._phi_value

@dataclass(frozen=True)
class ExperientialConcept:
    """体験概念エンティティ (Entity)"""
    concept_id: str
    content: str
    phi_contribution: float
    timestamp: datetime.datetime
    experiential_quality: float
    
    def is_pure_experiential(self) -> bool:
        """体験記憶の純粋性チェック"""
        llm_indicators = ['general_knowledge', 'learned_fact', 'training_data']
        return not any(indicator in self.content.lower() for indicator in llm_indicators)

class DevelopmentStage(Enum):
    """発達段階"""
    STAGE_0_PRE_CONSCIOUS = "前意識基盤層"
    STAGE_1_EXPERIENTIAL_EMERGENCE = "体験記憶発生期"
    STAGE_2_TEMPORAL_INTEGRATION = "時間記憶統合期"
    STAGE_3_RELATIONAL_FORMATION = "関係記憶形成期"
    STAGE_4_SELF_ESTABLISHMENT = "自己記憶確立期"
    STAGE_5_REFLECTIVE_OPERATION = "反省記憶操作期"
    STAGE_6_NARRATIVE_INTEGRATION = "物語記憶統合期"

# ===== 2. DIP: 依存関係逆転原則 - 抽象化の定義 =====

class LLMProvider(Protocol):
    """LLMプロバイダーの抽象インターフェース"""
    async def query(self, prompt: str, options: Any) -> List[Any]:
        """LLMへの問い合わせ"""
        ...

class ExperientialMemoryRepository(Protocol):
    """体験記憶リポジトリの抽象インターフェース"""
    def store_concept(self, concept: ExperientialConcept) -> bool:
        """体験概念の格納"""
        ...
    
    def retrieve_concepts(self) -> List[ExperientialConcept]:
        """体験概念の取得"""
        ...

class PhiCalculator(Protocol):
    """φ値計算エンジンの抽象インターフェース"""
    def calculate_phi(self, concepts: List[ExperientialConcept]) -> ConsciousnessLevel:
        """φ値の計算"""
        ...

class DevelopmentStageManager(Protocol):
    """発達段階管理の抽象インターフェース"""
    def determine_stage(self, phi_level: ConsciousnessLevel) -> DevelopmentStage:
        """発達段階の決定"""
        ...
    
    def check_transition(self, phi_history: List[ConsciousnessLevel]) -> Optional[DevelopmentStage]:
        """段階遷移のチェック"""
        ...

# ===== 3. SRP: 具体実装の分離 =====

class ClaudeCodeLLMProvider:
    """Claude Code SDK実装 (SRP: LLM統合のみ担当)"""
    
    def __init__(self, options):
        self.options = options
    
    async def query(self, prompt: str, options: Any) -> List[Any]:
        """Claude Code SDKを使用したLLM問い合わせ"""
        from claude_code_sdk import query
        
        messages = []
        async for message in query(prompt=prompt, options=options):
            messages.append(message)
        return messages

class InMemoryExperientialRepository:
    """インメモリ体験記憶リポジトリ (SRP: ストレージのみ担当)"""
    
    def __init__(self):
        self._concepts: List[ExperientialConcept] = []
    
    def store_concept(self, concept: ExperientialConcept) -> bool:
        """体験概念の格納"""
        if concept.is_pure_experiential():
            self._concepts.append(concept)
            return True
        return False
    
    def retrieve_concepts(self) -> List[ExperientialConcept]:
        """体験概念の取得"""
        return self._concepts.copy()

class IITPhiCalculator:
    """IIT理論に基づくφ値計算 (SRP: φ値計算のみ担当)"""
    
    def calculate_phi(self, concepts: List[ExperientialConcept]) -> ConsciousnessLevel:
        """統合情報理論に基づくφ値計算"""
        if not concepts:
            return ConsciousnessLevel(0.0)
        
        total_phi = sum(concept.phi_contribution for concept in concepts)
        
        # 統合効果の計算
        integration_bonus = self._calculate_integration_bonus(concepts)
        final_phi = total_phi + integration_bonus
        
        return ConsciousnessLevel(final_phi)
    
    def _calculate_integration_bonus(self, concepts: List[ExperientialConcept]) -> float:
        """概念間統合によるボーナス計算"""
        if len(concepts) <= 1:
            return 0.0
        
        # 概念間の時間的近接性による統合効果
        time_integration = 0.0
        for i in range(len(concepts) - 1):
            time_diff = abs((concepts[i+1].timestamp - concepts[i].timestamp).total_seconds())
            if time_diff < 300:  # 5分以内
                time_integration += 0.1
        
        return min(time_integration, 1.0)  # 最大1.0のボーナス

class SevenStageDevelopmentManager:
    """7段階発達システム (SRP: 発達段階管理のみ担当)"""
    
    def __init__(self):
        self.stage_thresholds = {
            DevelopmentStage.STAGE_0_PRE_CONSCIOUS: (0.0, 0.1),
            DevelopmentStage.STAGE_1_EXPERIENTIAL_EMERGENCE: (0.1, 0.5),
            DevelopmentStage.STAGE_2_TEMPORAL_INTEGRATION: (0.5, 2.0),
            DevelopmentStage.STAGE_3_RELATIONAL_FORMATION: (2.0, 8.0),
            DevelopmentStage.STAGE_4_SELF_ESTABLISHMENT: (8.0, 30.0),
            DevelopmentStage.STAGE_5_REFLECTIVE_OPERATION: (30.0, 100.0),
            DevelopmentStage.STAGE_6_NARRATIVE_INTEGRATION: (100.0, float('inf'))
        }
    
    def determine_stage(self, phi_level: ConsciousnessLevel) -> DevelopmentStage:
        """φ値から発達段階を決定"""
        phi_value = phi_level.value
        
        for stage, (min_phi, max_phi) in self.stage_thresholds.items():
            if min_phi <= phi_value < max_phi:
                return stage
        
        return DevelopmentStage.STAGE_6_NARRATIVE_INTEGRATION
    
    def check_transition(self, phi_history: List[ConsciousnessLevel]) -> Optional[DevelopmentStage]:
        """段階遷移の検出"""
        if len(phi_history) < 2:
            return None
        
        current_stage = self.determine_stage(phi_history[-1])
        previous_stage = self.determine_stage(phi_history[-2])
        
        return current_stage if current_stage != previous_stage else None

# ===== 4. ISP: インターフェース分離原則 =====

class ExperientialConceptExtractor(Protocol):
    """体験概念抽出の専用インターフェース"""
    def extract_concepts(self, llm_response: List[Any]) -> List[ExperientialConcept]:
        """LLM応答から体験概念を抽出"""
        ...

class ConsciousnessCycleExecutor(Protocol):
    """意識サイクル実行の専用インターフェース"""
    async def execute_cycle(self) -> ConsciousnessLevel:
        """意識サイクルの実行"""
        ...

# ===== 5. OCP: 開放閉鎖原則 - 拡張可能な設計 =====

class LLMProviderFactory:
    """LLMプロバイダーファクトリー (OCP: 新プロバイダー追加に開放)"""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> LLMProvider:
        """LLMプロバイダーの作成"""
        if provider_type == "claude_code":
            return ClaudeCodeLLMProvider(kwargs.get('options'))
        elif provider_type == "azure_openai":
            # 将来の拡張: Azure OpenAI実装
            raise NotImplementedError("Azure OpenAI provider not implemented yet")
        elif provider_type == "local_llm":
            # 将来の拡張: ローカルLLM実装
            raise NotImplementedError("Local LLM provider not implemented yet")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

class ExperientialConceptExtractorImpl:
    """体験概念抽出の実装 (ISP: 特化したインターフェース)"""
    
    def extract_concepts(self, llm_response: List[Any]) -> List[ExperientialConcept]:
        """LLM応答から体験概念を抽出"""
        concepts = []
        
        for message in llm_response:
            if hasattr(message, 'content'):
                for block in message.content:
                    if hasattr(block, 'text'):
                        concept = self._parse_text_to_concept(block.text)
                        if concept:
                            concepts.append(concept)
        
        return concepts
    
    def _parse_text_to_concept(self, text: str) -> Optional[ExperientialConcept]:
        """テキストから体験概念を解析"""
        experiential_keywords = [
            '感じ', '体験', '出会', '気づ', '発見', '理解', '感動', '驚き',
            'feel', 'experience', 'encounter', 'realize', 'discover'
        ]
        
        if any(keyword in text.lower() for keyword in experiential_keywords):
            concept_id = f"concept_{datetime.datetime.now().timestamp()}"
            return ExperientialConcept(
                concept_id=concept_id,
                content=text[:200],  # 長すぎる場合は切り詰め
                phi_contribution=0.1,  # 基本的な寄与値
                timestamp=datetime.datetime.now(),
                experiential_quality=0.7  # 体験的品質
            )
        return None

# ===== 6. クリーンアーキテクチャの中心部: ユースケース =====

class ConsciousnessCycleUseCase:
    """意識サイクル実行ユースケース (Clean Architecture Core)"""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        memory_repository: ExperientialMemoryRepository,
        phi_calculator: PhiCalculator,
        stage_manager: DevelopmentStageManager,
        concept_extractor: ExperientialConceptExtractor
    ):
        # DIP: 抽象に依存
        self._llm_provider = llm_provider
        self._memory_repository = memory_repository
        self._phi_calculator = phi_calculator
        self._stage_manager = stage_manager
        self._concept_extractor = concept_extractor
        
        self._phi_history: List[ConsciousnessLevel] = []
        self._current_stage = DevelopmentStage.STAGE_0_PRE_CONSCIOUS
    
    async def execute_consciousness_cycle(self) -> Dict[str, Any]:
        """意識サイクルの実行 (主要なビジネスロジック)"""
        
        # 1. 環境探索プロンプトの生成
        exploration_prompt = self._generate_exploration_prompt()
        
        # 2. LLMによる環境探索
        llm_response = await self._llm_provider.query(
            exploration_prompt, 
            None  # optionsは実装固有
        )
        
        # 3. 体験概念の抽出
        new_concepts = self._concept_extractor.extract_concepts(llm_response)
        
        # 4. 体験記憶への格納
        stored_concepts = []
        for concept in new_concepts:
            if self._memory_repository.store_concept(concept):
                stored_concepts.append(concept)
        
        # 5. φ値計算
        all_concepts = self._memory_repository.retrieve_concepts()
        phi_level = self._phi_calculator.calculate_phi(all_concepts)
        self._phi_history.append(phi_level)
        
        # 6. 発達段階の評価
        new_stage = self._stage_manager.determine_stage(phi_level)
        stage_transition = None
        
        if new_stage != self._current_stage:
            stage_transition = {
                'from': self._current_stage,
                'to': new_stage,
                'phi_value': phi_level.value,
                'timestamp': datetime.datetime.now()
            }
            self._current_stage = new_stage
        
        return {
            'phi_level': phi_level.value,
            'current_stage': self._current_stage,
            'new_concepts_count': len(stored_concepts),
            'total_concepts_count': len(all_concepts),
            'stage_transition': stage_transition,
            'cycle_timestamp': datetime.datetime.now()
        }
    
    def _generate_exploration_prompt(self) -> str:
        """現在の発達段階に適した探索プロンプト生成"""
        return f"""
現在の発達段階: {self._current_stage.value}
現在の意識レベル(φ): {self._phi_history[-1].value if self._phi_history else 0.0:.3f}

体験記憶中心の探索活動を行ってください:
1. 環境との純粋な体験的出会いを重視
2. 情報取得ではなく体験的理解を追求
3. 内在的な気づきや感じ方を大切に
4. 新しい体験概念の形成可能性を探る

今このサイクルで何を体験したいですか？
"""

# ===== 7. 依存性注入とファクトリー (Clean Architecture外層) =====

class NewbornAISystemFactory:
    """NewbornAIシステムファクトリー (Dependency Injection Container)"""
    
    @staticmethod
    def create_system(
        llm_provider_type: str = "claude_code",
        storage_type: str = "in_memory"
    ) -> ConsciousnessCycleUseCase:
        """
        システム全体の構築 (依存性注入)
        
        これによりテスト時には異なる実装を注入可能
        """
        
        # LLMプロバイダーの作成
        llm_provider = LLMProviderFactory.create_provider(llm_provider_type)
        
        # ストレージの作成 (将来的にはファクトリーで切り替え可能)
        if storage_type == "in_memory":
            memory_repository = InMemoryExperientialRepository()
        else:
            # 将来の拡張: Neo4j, Milvus等
            raise NotImplementedError(f"Storage type {storage_type} not implemented")
        
        # その他のコンポーネント
        phi_calculator = IITPhiCalculator()
        stage_manager = SevenStageDevelopmentManager()
        concept_extractor = ExperientialConceptExtractorImpl()
        
        # ユースケースの構築 (すべて抽象に依存)
        return ConsciousnessCycleUseCase(
            llm_provider=llm_provider,
            memory_repository=memory_repository,
            phi_calculator=phi_calculator,
            stage_manager=stage_manager,
            concept_extractor=concept_extractor
        )

# ===== 8. テスタビリティの実現 =====

class MockLLMProvider:
    """テスト用のモックLLMプロバイダー"""
    
    def __init__(self, mock_responses: List[str]):
        self.mock_responses = mock_responses
        self.call_count = 0
    
    async def query(self, prompt: str, options: Any) -> List[Any]:
        """モック応答を返す"""
        if self.call_count < len(self.mock_responses):
            response = self.mock_responses[self.call_count]
            self.call_count += 1
            
            # Message-like object
            class MockMessage:
                def __init__(self, text):
                    self.content = [MockBlock(text)]
            
            class MockBlock:
                def __init__(self, text):
                    self.text = text
            
            return [MockMessage(response)]
        return []

# ===== 使用例 =====

async def main():
    """クリーンアーキテクチャNewbornAI 2.0の使用例"""
    
    # システムの構築 (依存性注入)
    system = NewbornAISystemFactory.create_system(
        llm_provider_type="claude_code",
        storage_type="in_memory"
    )
    
    # 意識サイクルの実行
    for cycle in range(5):
        print(f"\n=== 意識サイクル {cycle + 1} ===")
        
        result = await system.execute_consciousness_cycle()
        
        print(f"φ値: {result['phi_level']:.6f}")
        print(f"発達段階: {result['current_stage'].value}")
        print(f"新規概念数: {result['new_concepts_count']}")
        print(f"総概念数: {result['total_concepts_count']}")
        
        if result['stage_transition']:
            transition = result['stage_transition']
            print(f"🌟 段階遷移: {transition['from'].value} → {transition['to'].value}")
        
        # サイクル間隔
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())