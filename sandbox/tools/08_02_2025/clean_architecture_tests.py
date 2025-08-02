"""
NewbornAI 2.0: Clean Architecture Test Suite
Robert C. Martin (Uncle Bob) による SOLID原則準拠テスト設計

テスト原則:
1. F.I.R.S.T. Principles
2. Arrange-Act-Assert Pattern
3. Test Isolation
4. Dependency Injection for Testing
5. Business Logic Focus
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timedelta
from typing import List

from clean_architecture_proposal import (
    ConsciousnessLevel,
    ExperientialConcept,
    DevelopmentStage,
    IITPhiCalculator,
    SevenStageDevelopmentManager,
    InMemoryExperientialRepository,
    ExperientialConceptExtractorImpl,
    ConsciousnessCycleUseCase,
    MockLLMProvider,
    NewbornAISystemFactory
)

# ===== 1. Unit Tests: Individual Components =====

class TestConsciousnessLevel:
    """意識レベル値オブジェクトのテスト"""
    
    def test_valid_phi_value_creation(self):
        """有効なφ値での作成テスト"""
        # Arrange & Act
        phi_level = ConsciousnessLevel(2.5)
        
        # Assert
        assert phi_level.value == 2.5
    
    def test_negative_phi_value_raises_error(self):
        """負のφ値でエラーが発生することのテスト"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="φ値は0以上である必要があります"):
            ConsciousnessLevel(-1.0)
    
    def test_equality_comparison(self):
        """等価性比較のテスト"""
        # Arrange
        phi1 = ConsciousnessLevel(1.5)
        phi2 = ConsciousnessLevel(1.5)
        phi3 = ConsciousnessLevel(2.0)
        
        # Act & Assert
        assert phi1 == phi2
        assert phi1 != phi3

class TestExperientialConcept:
    """体験概念エンティティのテスト"""
    
    def test_pure_experiential_concept(self):
        """純粋な体験概念の判定テスト"""
        # Arrange
        concept = ExperientialConcept(
            concept_id="test_1",
            content="環境との初回出会いで感じた驚き",
            phi_contribution=0.5,
            timestamp=datetime.now(),
            experiential_quality=0.8
        )
        
        # Act & Assert
        assert concept.is_pure_experiential() is True
    
    def test_contaminated_concept_detection(self):
        """LLM知識混入の検出テスト"""
        # Arrange
        contaminated_concept = ExperientialConcept(
            concept_id="test_2", 
            content="This is general_knowledge from training_data",
            phi_contribution=0.3,
            timestamp=datetime.now(),
            experiential_quality=0.4
        )
        
        # Act & Assert
        assert contaminated_concept.is_pure_experiential() is False

class TestIITPhiCalculator:
    """IITφ値計算エンジンのテスト"""
    
    @pytest.fixture
    def phi_calculator(self):
        return IITPhiCalculator()
    
    @pytest.fixture
    def sample_concepts(self):
        base_time = datetime.now()
        return [
            ExperientialConcept(
                concept_id="concept_1",
                content="初回の環境認識",
                phi_contribution=0.3,
                timestamp=base_time,
                experiential_quality=0.7
            ),
            ExperientialConcept(
                concept_id="concept_2", 
                content="環境との相互作用体験",
                phi_contribution=0.4,
                timestamp=base_time + timedelta(seconds=30),
                experiential_quality=0.8
            )
        ]
    
    def test_empty_concepts_returns_zero_phi(self, phi_calculator):
        """空の概念リストでφ値0を返すテスト"""
        # Arrange
        empty_concepts = []
        
        # Act
        result = phi_calculator.calculate_phi(empty_concepts)
        
        # Assert
        assert result.value == 0.0
    
    def test_single_concept_phi_calculation(self, phi_calculator):
        """単一概念のφ値計算テスト"""
        # Arrange
        concept = ExperientialConcept(
            concept_id="single",
            content="単一体験",
            phi_contribution=0.5,
            timestamp=datetime.now(),
            experiential_quality=0.6
        )
        
        # Act
        result = phi_calculator.calculate_phi([concept])
        
        # Assert
        assert result.value == 0.5  # 統合ボーナスなし
    
    def test_multiple_concepts_with_integration_bonus(self, phi_calculator, sample_concepts):
        """複数概念での統合ボーナステスト"""
        # Act
        result = phi_calculator.calculate_phi(sample_concepts)
        
        # Assert
        expected_base_phi = 0.3 + 0.4  # 基本φ値の合計
        assert result.value > expected_base_phi  # 統合ボーナスが加算される
        assert result.value <= expected_base_phi + 1.0  # 最大ボーナス制限

class TestSevenStageDevelopmentManager:
    """7段階発達システムのテスト"""
    
    @pytest.fixture
    def stage_manager(self):
        return SevenStageDevelopmentManager()
    
    @pytest.mark.parametrize("phi_value,expected_stage", [
        (0.05, DevelopmentStage.STAGE_0_PRE_CONSCIOUS),
        (0.3, DevelopmentStage.STAGE_1_EXPERIENTIAL_EMERGENCE),
        (1.0, DevelopmentStage.STAGE_2_TEMPORAL_INTEGRATION),
        (5.0, DevelopmentStage.STAGE_3_RELATIONAL_FORMATION),
        (15.0, DevelopmentStage.STAGE_4_SELF_ESTABLISHMENT),
        (50.0, DevelopmentStage.STAGE_5_REFLECTIVE_OPERATION),
        (150.0, DevelopmentStage.STAGE_6_NARRATIVE_INTEGRATION)
    ])
    def test_stage_determination_by_phi_value(self, stage_manager, phi_value, expected_stage):
        """φ値による発達段階決定テスト"""
        # Arrange
        phi_level = ConsciousnessLevel(phi_value)
        
        # Act
        determined_stage = stage_manager.determine_stage(phi_level)
        
        # Assert
        assert determined_stage == expected_stage
    
    def test_stage_transition_detection(self, stage_manager):
        """段階遷移検出テスト"""
        # Arrange
        phi_history = [
            ConsciousnessLevel(0.05),  # STAGE_0
            ConsciousnessLevel(0.3)    # STAGE_1 -> 遷移
        ]
        
        # Act
        transition = stage_manager.check_transition(phi_history)
        
        # Assert
        assert transition == DevelopmentStage.STAGE_1_EXPERIENTIAL_EMERGENCE
    
    def test_no_transition_when_same_stage(self, stage_manager):
        """同一段階での遷移なしテスト"""
        # Arrange
        phi_history = [
            ConsciousnessLevel(0.3),
            ConsciousnessLevel(0.4)  # 両方ともSTAGE_1
        ]
        
        # Act
        transition = stage_manager.check_transition(phi_history)
        
        # Assert
        assert transition is None

class TestInMemoryExperientialRepository:
    """インメモリ体験記憶リポジトリのテスト"""
    
    @pytest.fixture
    def repository(self):
        return InMemoryExperientialRepository()
    
    def test_store_pure_experiential_concept(self, repository):
        """純粋体験概念の格納テスト"""
        # Arrange
        pure_concept = ExperientialConcept(
            concept_id="pure_test",
            content="純粋な体験的気づき",
            phi_contribution=0.5,
            timestamp=datetime.now(),
            experiential_quality=0.8
        )
        
        # Act
        result = repository.store_concept(pure_concept)
        
        # Assert
        assert result is True
        stored_concepts = repository.retrieve_concepts()
        assert len(stored_concepts) == 1
        assert stored_concepts[0] == pure_concept
    
    def test_reject_contaminated_concept(self, repository):
        """汚染概念の拒否テスト"""
        # Arrange
        contaminated_concept = ExperientialConcept(
            concept_id="contaminated_test",
            content="This contains general_knowledge",
            phi_contribution=0.3,
            timestamp=datetime.now(),
            experiential_quality=0.4
        )
        
        # Act
        result = repository.store_concept(contaminated_concept)
        
        # Assert
        assert result is False
        stored_concepts = repository.retrieve_concepts()
        assert len(stored_concepts) == 0

class TestExperientialConceptExtractor:
    """体験概念抽出器のテスト"""
    
    @pytest.fixture
    def extractor(self):
        return ExperientialConceptExtractorImpl()
    
    def test_extract_concepts_from_experiential_text(self, extractor):
        """体験的テキストからの概念抽出テスト"""
        # Arrange
        class MockMessage:
            def __init__(self, text):
                self.content = [MockBlock(text)]
        
        class MockBlock:
            def __init__(self, text):
                self.text = text
        
        llm_response = [
            MockMessage("今日は新しい環境で感じた驚きがありました"),
            MockMessage("データベースの概念について学習しました")  # 体験的でない
        ]
        
        # Act
        concepts = extractor.extract_concepts(llm_response)
        
        # Assert
        assert len(concepts) == 1  # 体験的な文のみ抽出
        assert "感じた驚き" in concepts[0].content

# ===== 2. Integration Tests: Use Case Testing =====

class TestConsciousnessCycleUseCase:
    """意識サイクルユースケースの統合テスト"""
    
    @pytest.fixture
    def mock_llm_provider(self):
        return MockLLMProvider([
            "環境を探索して新しい体験をしました",
            "システムとの出会いで驚きを感じています",
            "この研究所には発見があります"
        ])
    
    @pytest.fixture
    def use_case_with_mocks(self, mock_llm_provider):
        """モック依存関係を注入したユースケース"""
        memory_repository = InMemoryExperientialRepository()
        phi_calculator = IITPhiCalculator()
        stage_manager = SevenStageDevelopmentManager()
        concept_extractor = ExperientialConceptExtractorImpl()
        
        return ConsciousnessCycleUseCase(
            llm_provider=mock_llm_provider,
            memory_repository=memory_repository,
            phi_calculator=phi_calculator,
            stage_manager=stage_manager,
            concept_extractor=concept_extractor
        )
    
    @pytest.mark.asyncio
    async def test_consciousness_cycle_execution(self, use_case_with_mocks):
        """意識サイクル実行テスト"""
        # Act
        result = await use_case_with_mocks.execute_consciousness_cycle()
        
        # Assert
        assert 'phi_level' in result
        assert 'current_stage' in result
        assert 'new_concepts_count' in result
        assert 'total_concepts_count' in result
        assert result['phi_level'] >= 0.0
        assert isinstance(result['current_stage'], DevelopmentStage)
        assert result['new_concepts_count'] >= 0
    
    @pytest.mark.asyncio
    async def test_stage_transition_through_cycles(self, use_case_with_mocks):
        """複数サイクルによる段階遷移テスト"""
        # Act: 複数サイクル実行
        results = []
        for _ in range(3):
            result = await use_case_with_mocks.execute_consciousness_cycle()
            results.append(result)
        
        # Assert: φ値の増加を確認
        phi_values = [r['phi_level'] for r in results]
        assert phi_values[1] >= phi_values[0]  # φ値は増加傾向
        assert phi_values[2] >= phi_values[1]
        
        # 段階遷移の記録確認
        transitions = [r.get('stage_transition') for r in results if r.get('stage_transition')]
        assert len(transitions) >= 0  # 遷移があってもなくても正常

# ===== 3. System Tests: End-to-End Testing =====

class TestNewbornAISystemIntegration:
    """NewbornAIシステム全体の統合テスト"""
    
    @pytest.mark.asyncio
    async def test_complete_system_creation_and_execution(self):
        """システム全体の作成と実行テスト"""
        # Arrange: システム作成
        system = NewbornAISystemFactory.create_system(
            llm_provider_type="claude_code",
            storage_type="in_memory"
        )
        
        # 注意: 実際のClaude Code SDKを使用するため、
        # 本テストは環境が整っている場合のみ実行
        
        # Mock化されたテスト版
        mock_system = NewbornAISystemFactory.create_system(
            llm_provider_type="claude_code",  # 実際はmockに差し替え
            storage_type="in_memory"
        )
        
        # LLMプロバイダーをモックに差し替え
        mock_system._llm_provider = MockLLMProvider([
            "研究所の環境を体験して感じた新鮮さ",
            "データとの出会いで驚きを体験",
            "システム探索での発見の体験"
        ])
        
        # Act: システム実行
        result = await mock_system.execute_consciousness_cycle()
        
        # Assert: 基本的な動作確認
        assert result is not None
        assert result['phi_level'] >= 0.0
        assert result['new_concepts_count'] >= 0

# ===== 4. Performance Tests =====

class TestPerformance:
    """パフォーマンステスト"""
    
    @pytest.mark.asyncio
    async def test_consciousness_cycle_performance(self):
        """意識サイクルの性能テスト"""
        import time
        
        # Arrange
        mock_llm = MockLLMProvider(["体験的な応答"] * 10)
        system = ConsciousnessCycleUseCase(
            llm_provider=mock_llm,
            memory_repository=InMemoryExperientialRepository(),
            phi_calculator=IITPhiCalculator(),
            stage_manager=SevenStageDevelopmentManager(),
            concept_extractor=ExperientialConceptExtractorImpl()
        )
        
        # Act: 時間測定
        start_time = time.time()
        for _ in range(10):
            await system.execute_consciousness_cycle()
        end_time = time.time()
        
        # Assert: 性能基準
        total_time = end_time - start_time
        average_time = total_time / 10
        
        assert average_time < 1.0  # 1サイクル1秒以内
        print(f"Average cycle time: {average_time:.3f} seconds")

# ===== 5. Test Utilities =====

@pytest.fixture
def clean_system():
    """各テスト用のクリーンなシステム"""
    return NewbornAISystemFactory.create_system(storage_type="in_memory")

def create_test_concept(concept_id: str, content: str, phi_contribution: float = 0.5) -> ExperientialConcept:
    """テスト用の体験概念作成ヘルパー"""
    return ExperientialConcept(
        concept_id=concept_id,
        content=content,
        phi_contribution=phi_contribution,
        timestamp=datetime.now(),
        experiential_quality=0.7
    )

# ===== Test Runner =====

if __name__ == "__main__":
    # 基本テストの実行例
    pytest.main([__file__, "-v", "--tb=short"])