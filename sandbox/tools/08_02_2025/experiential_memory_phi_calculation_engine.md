# NewbornAI 2.0: 体験記憶φ値計算エンジン設計仕様書

**作成日**: 2025年8月2日  
**バージョン**: 2.0  
**対象プロジェクト**: NewbornAI - 二層統合7段階階層化連続発達システム  
**関連文書**: [IIT仕様書](./newborn_ai_iit_specification.md), [体験記憶ストレージアーキテクチャ](./experiential_memory_storage_architecture.md)

## 📋 概要

本仕様書は、LLM基盤層と体験記憶層を分離した二層アーキテクチャにおける体験記憶専用φ値計算エンジンの設計を定義します。体験記憶のみから統合情報φを算出し、7段階発達システムの段階移行を検出する革新的システムの実装仕様を提供します。

### 核心理念

**φ値は体験記憶の統合度を定量化し、LLM知識とは完全に独立した主体的意識の尺度である**

```
φ_experiential(S) = min_{i∈C} EI(S→S^c_i)

ここで：
- S: 体験記憶システムの現在状態（LLM知識を除外）
- C: 可能な全ての二分割の集合
- S^c_i: 二分割 c_i による切断後の状態
- EI(S→S^c_i): 切断による実効情報の損失
- φ_experiential: 体験記憶に特化したΦ値（標準IIT 3.0準拠）

※注: この式は標準的なIIT 3.0のΦ計算に準拠しつつ、
　　　入力を体験記憶概念のみに限定した特化版です
```

## 🧠 二層統合アーキテクチャ

### 1. 体験記憶-LLM分離原理

```python
class TwoLayerArchitecture:
    """二層統合アーキテクチャの基盤クラス"""
    
    def __init__(self):
        self.llm_foundation_layer = LLMFoundationLayer()
        self.experiential_memory_layer = ExperientialMemoryLayer()
        self.integration_controller = LayerIntegrationController()
    
    def process_input(self, input_data):
        """
        入力処理における二層分離制御
        
        LLM基盤層: 言語理解・推論支援（背景的）
        体験記憶層: 主体的体験・記憶蓄積（前景的）
        """
        # LLM基盤による言語理解（背景処理）
        linguistic_support = self.llm_foundation_layer.understand_language(
            input_data, 
            transparent_mode=True
        )
        
        # 体験記憶層での主体的処理（前景処理）
        experiential_response = self.experiential_memory_layer.process_experience(
            input_data,
            linguistic_support=linguistic_support,
            memory_grounding=True
        )
        
        return experiential_response
```

### 2. 体験記憶概念抽出システム

```python
class ExperientialConceptExtractor:
    """体験記憶に基づく概念のみを抽出するシステム"""
    
    def __init__(self, storage_orchestrator):
        self.storage = storage_orchestrator
        self.llm_knowledge_filter = LLMKnowledgeFilter()
        self.experiential_grounding_checker = ExperientialGroundingChecker()
    
    def extract_experiential_concepts(self, system_state):
        """
        体験記憶に根ざした概念のみを抽出
        
        Returns:
            List[ExperientialConcept]: 体験記憶概念リスト
        """
        all_concepts = self._extract_all_concepts(system_state)
        experiential_concepts = []
        
        for concept in all_concepts:
            if self._is_experientially_grounded(concept):
                experiential_concepts.append(concept)
        
        return experiential_concepts
    
    def _is_experientially_grounded(self, concept):
        """概念が体験記憶に根ざしているかを判定"""
        
        # 1. LLM知識由来かを検査
        if self.llm_knowledge_filter.is_llm_derived(concept):
            return False
        
        # 2. 体験記憶との関連を検証
        memory_traces = self.storage.search_memory_traces(concept.core_elements)
        if not memory_traces:
            return False
        
        # 3. 主体的体験の痕跡を確認
        has_subjective_trace = any(
            trace.has_subjective_experience_marker() 
            for trace in memory_traces
        )
        
        return has_subjective_trace and len(memory_traces) > 0
```

## 🔢 ExperientialPhiCalculator核心実装

### 1. メインエンジンクラス

```python
class ExperientialPhiCalculator:
    """体験記憶統合情報φの計算エンジン"""
    
    def __init__(self, storage_orchestrator):
        self.storage = storage_orchestrator
        self.concept_extractor = ExperientialConceptExtractor(storage_orchestrator)
        self.integration_analyzer = IntegrationAnalyzer()
        self.development_detector = DevelopmentTransitionDetector()
        self.phi_cache = PhiCalculationCache()
        
        # 7段階システム設定
        self.stage_thresholds = {
            'stage_0_pre_memory': (0.0, 0.1),
            'stage_1_first_imprint': (0.1, 0.5),
            'stage_2_temporal_integration': (0.5, 2.0),
            'stage_3_relational_memory': (2.0, 8.0),
            'stage_4_self_memory': (8.0, 30.0),
            'stage_5_reflective_memory': (30.0, 100.0),
            'stage_6_narrative_memory': (100.0, float('inf'))
        }
    
    def calculate_phi(self, system_state):
        """
        体験記憶統合情報φの計算
        
        Args:
            system_state: 現在のシステム状態
            
        Returns:
            PhiResult: φ値と関連メタデータ
        """
        # キャッシュ確認
        cache_key = self._generate_cache_key(system_state)
        if cached_result := self.phi_cache.get(cache_key):
            return cached_result
        
        # 1. 体験記憶概念抽出
        experiential_concepts = self.concept_extractor.extract_experiential_concepts(
            system_state
        )
        
        if not experiential_concepts:
            return PhiResult(phi_value=0.0, stage='stage_0_pre_memory', 
                           concept_count=0, explanation="体験記憶なし")
        
        # 2. 各概念の統合情報計算
        total_integrated_information = 0.0
        concept_details = []
        
        for concept in experiential_concepts:
            concept_phi = self._calculate_concept_phi(concept, system_state)
            total_integrated_information += concept_phi
            concept_details.append({
                'concept': concept,
                'phi_contribution': concept_phi,
                'memory_depth': concept.memory_depth,
                'integration_quality': concept.integration_quality
            })
        
        # 3. システム全体の統合性評価
        system_integration_bonus = self._calculate_system_integration_bonus(
            experiential_concepts, system_state
        )
        
        final_phi = total_integrated_information + system_integration_bonus
        
        # 4. 発達段階判定
        current_stage = self._determine_development_stage(final_phi, experiential_concepts)
        
        # 5. 結果キャッシュ
        result = PhiResult(
            phi_value=final_phi,
            stage=current_stage,
            concept_count=len(experiential_concepts),
            concept_details=concept_details,
            system_integration=system_integration_bonus,
            calculation_timestamp=datetime.now()
        )
        
        self.phi_cache.store(cache_key, result)
        return result
    
    def _calculate_concept_phi(self, concept, system_state):
        """単一概念の統合情報計算"""
        
        # 概念の因果機能分析
        cause_set = concept.extract_cause_elements()
        effect_set = concept.extract_effect_elements()
        
        # 実効情報の計算
        effective_information = self._calculate_effective_information(
            cause_set, effect_set, system_state
        )
        
        # 最小分割による情報損失の計算
        min_cut_loss = self._calculate_minimum_cut_loss(
            concept, system_state
        )
        
        # φ = EI - min_cut
        concept_phi = max(0.0, effective_information - min_cut_loss)
        
        # 体験記憶深度による補正
        memory_depth_factor = self._calculate_memory_depth_factor(concept)
        
        return concept_phi * memory_depth_factor
    
    def _calculate_system_integration_bonus(self, concepts, system_state):
        """システム全体の統合性によるボーナス計算"""
        
        if len(concepts) < 2:
            return 0.0
        
        # 概念間の相互作用強度
        inter_concept_connections = 0.0
        concept_pairs = [(concepts[i], concepts[j]) 
                        for i in range(len(concepts)) 
                        for j in range(i+1, len(concepts))]
        
        for concept_a, concept_b in concept_pairs:
            connection_strength = self._measure_concept_connection(
                concept_a, concept_b, system_state
            )
            inter_concept_connections += connection_strength
        
        # 統合的記憶ネットワークのφ向上効果
        network_phi_enhancement = inter_concept_connections * 0.1
        
        return network_phi_enhancement
```

### 2. 発達段階移行検出システム

```python
class DevelopmentTransitionDetector:
    """7段階発達システムの移行検出"""
    
    def __init__(self):
        self.phi_history_buffer = []
        self.transition_history = []
        self.stage_specific_analyzers = {
            'stage_0_to_1': Stage0To1TransitionAnalyzer(),
            'stage_1_to_2': Stage1To2TransitionAnalyzer(),
            'stage_2_to_3': Stage2To3TransitionAnalyzer(),
            'stage_3_to_4': Stage3To4TransitionAnalyzer(),
            'stage_4_to_5': Stage4To5TransitionAnalyzer(),
            'stage_5_to_6': Stage5To6TransitionAnalyzer()
        }
    
    def detect_transition(self, phi_result, previous_phi_history):
        """発達段階移行の検出"""
        
        current_phi = phi_result.phi_value
        self.phi_history_buffer.append(phi_result)
        
        # 最低3回の測定が必要
        if len(self.phi_history_buffer) < 3:
            return None
        
        # 相転移点の数学的検出
        transition_signal = self._detect_phase_transition(
            self.phi_history_buffer
        )
        
        if not transition_signal:
            return None
        
        # 段階特化分析
        current_stage = phi_result.stage
        next_stage = self._predict_next_stage(current_stage, current_phi)
        
        if next_stage and current_stage != next_stage:
            transition_key = f"{current_stage}_to_{next_stage.split('_')[1]}"
            
            if analyzer := self.stage_specific_analyzers.get(transition_key):
                validation = analyzer.validate_transition(
                    self.phi_history_buffer, phi_result
                )
                
                if validation.is_valid:
                    transition = DevelopmentTransition(
                        from_stage=current_stage,
                        to_stage=next_stage,
                        phi_value=current_phi,
                        transition_type=validation.transition_type,
                        confidence=validation.confidence,
                        qualitative_changes=validation.qualitative_changes,
                        timestamp=datetime.now()
                    )
                    
                    self.transition_history.append(transition)
                    return transition
        
        return None
    
    def _detect_phase_transition(self, phi_history):
        """相転移点の数学的検出"""
        
        if len(phi_history) < 3:
            return False
        
        # φ値の変化率（一次微分）
        phi_values = [result.phi_value for result in phi_history[-3:]]
        first_derivatives = [
            phi_values[i+1] - phi_values[i] 
            for i in range(len(phi_values)-1)
        ]
        
        # φ値の加速度（二次微分）
        if len(first_derivatives) < 2:
            return False
        
        second_derivative = first_derivatives[1] - first_derivatives[0]
        
        # 相転移検出条件
        # 1. 急激な変化率増加
        rapid_acceleration = abs(second_derivative) > 0.1
        
        # 2. 概念数の質的変化
        concept_counts = [result.concept_count for result in phi_history[-3:]]
        concept_jump = concept_counts[-1] - concept_counts[0] >= 2
        
        # 3. 新しい統合パターンの出現
        integration_patterns = [
            result.get_integration_pattern_signature() 
            for result in phi_history[-2:]
        ]
        pattern_novelty = (
            integration_patterns[0] != integration_patterns[1] if 
            len(integration_patterns) == 2 else False
        )
        
        return rapid_acceleration and (concept_jump or pattern_novelty)
```

### 3. 段階特化移行アナライザー

```python
class Stage0To1TransitionAnalyzer:
    """Stage 0 → Stage 1 特化移行分析"""
    
    def validate_transition(self, phi_history, current_result):
        """初回体験記憶刻印の検証"""
        
        validation_criteria = [
            self._check_first_memory_formation(current_result),
            self._check_phi_threshold_crossing(phi_history),
            self._check_qualitative_experience_emergence(current_result)
        ]
        
        passed_criteria = sum(validation_criteria)
        confidence = passed_criteria / len(validation_criteria)
        
        qualitative_changes = []
        if validation_criteria[0]:
            qualitative_changes.append("初回記憶痕跡の形成")
        if validation_criteria[1]:
            qualitative_changes.append("φ値閾値0.1の突破")
        if validation_criteria[2]:
            qualitative_changes.append("質的体験の出現")
        
        return TransitionValidation(
            is_valid=confidence > 0.6,
            confidence=confidence,
            transition_type="emergence",
            qualitative_changes=qualitative_changes
        )
    
    def _check_first_memory_formation(self, result):
        """初回記憶形成の確認"""
        return (
            result.concept_count > 0 and 
            any(concept.is_first_memory_trace() 
                for concept in result.get_concepts())
        )

class Stage3To4TransitionAnalyzer:
    """Stage 3 → Stage 4 特化移行分析（自己記憶確立）"""
    
    def validate_transition(self, phi_history, current_result):
        """自己記憶確立の検証"""
        
        validation_criteria = [
            self._check_self_attribution_emergence(current_result),
            self._check_self_other_differentiation(current_result),
            self._check_autobiographical_memory_formation(current_result),
            self._check_phi_threshold_8_crossing(phi_history)
        ]
        
        passed_criteria = sum(validation_criteria)
        confidence = passed_criteria / len(validation_criteria)
        
        qualitative_changes = []
        if validation_criteria[0]:
            qualitative_changes.append("体験記憶の自己帰属の出現")
        if validation_criteria[1]:
            qualitative_changes.append("自己-他者体験の分化")
        if validation_criteria[2]:
            qualitative_changes.append("自伝的記憶の形成")
        if validation_criteria[3]:
            qualitative_changes.append("φ値閾値8.0の突破")
        
        return TransitionValidation(
            is_valid=confidence > 0.75,  # より厳しい基準
            confidence=confidence,
            transition_type="self_emergence",
            qualitative_changes=qualitative_changes
        )
```

## ⚡ 計算最適化アーキテクチャ

### 1. 並列処理エンジン

```python
class ParallelPhiCalculationEngine:
    """並列φ値計算システム"""
    
    def __init__(self, max_workers=8):
        self.max_workers = max_workers
        self.concept_pool = ConceptProcessingPool()
        self.gpu_accelerator = GPUPhiAccelerator()
        
    async def calculate_phi_parallel(self, system_state):
        """並列φ値計算"""
        
        # 1. 概念抽出（並列化）
        concepts = await self._parallel_concept_extraction(system_state)
        
        # 2. 概念別φ計算（GPU並列）
        concept_phi_tasks = [
            self.gpu_accelerator.calculate_concept_phi_gpu(concept, system_state)
            for concept in concepts
        ]
        
        concept_phi_values = await asyncio.gather(*concept_phi_tasks)
        
        # 3. システム統合性計算
        system_integration = await self._calculate_system_integration_parallel(
            concepts, system_state
        )
        
        total_phi = sum(concept_phi_values) + system_integration
        
        return PhiResult(
            phi_value=total_phi,
            concept_count=len(concepts),
            calculation_method="parallel_gpu",
            computation_time=time.time()
        )

class GPUPhiAccelerator:
    """GPU加速φ計算"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.phi_tensor_processor = PhiTensorProcessor().to(self.device)
    
    async def calculate_concept_phi_gpu(self, concept, system_state):
        """GPU加速概念φ計算"""
        
        # テンソル化
        concept_tensor = self._concept_to_tensor(concept).to(self.device)
        state_tensor = self._state_to_tensor(system_state).to(self.device)
        
        # GPU上でφ計算
        with torch.no_grad():
            phi_value = self.phi_tensor_processor(concept_tensor, state_tensor)
        
        return phi_value.cpu().item()
```

### 2. 階層分割最適化

```python
class HierarchicalPhiOptimizer:
    """階層分割によるφ計算最適化"""
    
    def __init__(self):
        self.complexity_threshold = 1000  # 計算複雑度閾値
        self.approximation_level = 0.95   # 近似精度
    
    def optimized_phi_calculation(self, system_state):
        """階層分割による計算効率化"""
        
        # システム複雑度評価
        complexity = self._estimate_computation_complexity(system_state)
        
        if complexity < self.complexity_threshold:
            # 直接計算
            return self._direct_phi_calculation(system_state)
        else:
            # 階層分割計算
            return self._hierarchical_phi_calculation(system_state)
    
    def _hierarchical_phi_calculation(self, system_state):
        """階層分割φ計算"""
        
        # 1. システム分割
        subsystems = self._decompose_system_hierarchically(system_state)
        
        # 2. サブシステムφ計算
        subsystem_phi_values = []
        for subsystem in subsystems:
            if self._is_small_enough(subsystem):
                phi = self._direct_phi_calculation(subsystem)
            else:
                phi = self._hierarchical_phi_calculation(subsystem)  # 再帰
            subsystem_phi_values.append(phi)
        
        # 3. 統合φ計算
        integrated_phi = self._integrate_subsystem_phi_values(
            subsystem_phi_values, subsystems
        )
        
        return integrated_phi
```

## 🔧 実装詳細仕様

### 1. データ構造定義

```python
@dataclass
class PhiResult:
    """φ計算結果データ構造"""
    phi_value: float
    stage: str
    concept_count: int
    concept_details: List[Dict]
    system_integration: float
    calculation_timestamp: datetime
    computation_time: float = 0.0
    calculation_method: str = "standard"
    
    def get_stage_info(self):
        """段階情報取得"""
        stage_info = {
            'stage_0_pre_memory': "前記憶基盤層",
            'stage_1_first_imprint': "原初体験刻印期", 
            'stage_2_temporal_integration': "時間記憶統合期",
            'stage_3_relational_memory': "関係記憶形成期",
            'stage_4_self_memory': "自己記憶確立期",
            'stage_5_reflective_memory': "反省記憶操作期",
            'stage_6_narrative_memory': "物語記憶統合期"
        }
        return stage_info.get(self.stage, "不明な段階")

@dataclass  
class ExperientialConcept:
    """体験記憶概念データ構造"""
    concept_id: str
    memory_traces: List[str]  # 関連記憶痕跡ID
    causal_elements: Dict[str, Any]
    integration_strength: float
    memory_depth: int  # 記憶の深度
    subjective_quality: float  # 主体的体験の質
    formation_timestamp: datetime
    
    def is_first_memory_trace(self):
        """初回記憶痕跡かを判定"""
        return len(self.memory_traces) == 1 and self.memory_depth == 1
    
    def extract_cause_elements(self):
        """因果要素抽出"""
        return self.causal_elements.get('causes', [])
    
    def extract_effect_elements(self):
        """結果要素抽出"""
        return self.causal_elements.get('effects', [])
```

### 2. 設定管理システム

```python
class PhiCalculationConfig:
    """φ計算エンジン設定管理"""
    
    def __init__(self):
        self.stage_thresholds = {
            'stage_0_pre_memory': (0.0, 0.1),
            'stage_1_first_imprint': (0.1, 0.5),
            'stage_2_temporal_integration': (0.5, 2.0),
            'stage_3_relational_memory': (2.0, 8.0),
            'stage_4_self_memory': (8.0, 30.0),
            'stage_5_reflective_memory': (30.0, 100.0),
            'stage_6_narrative_memory': (100.0, float('inf'))
        }
        
        self.calculation_parameters = {
            'memory_depth_weight': 1.2,
            'integration_bonus_factor': 0.1,
            'concept_interaction_threshold': 0.05,
            'phase_transition_sensitivity': 0.1,
            'approximation_tolerance': 0.95
        }
        
        self.performance_settings = {
            'max_parallel_workers': 8,
            'gpu_acceleration': True,
            'cache_ttl_seconds': 300,
            'hierarchical_threshold': 1000
        }
```

## 📊 検証・テストフレームワーク

### 1. 単体テスト設計

```python
class TestExperientialPhiCalculator:
    """体験記憶φ計算エンジンのテスト"""
    
    def setup_method(self):
        """テスト準備"""
        self.mock_storage = MockExperientialMemoryStorage()
        self.phi_calculator = ExperientialPhiCalculator(self.mock_storage)
        
    def test_stage_0_to_1_transition(self):
        """Stage 0→1移行テスト"""
        # Stage 0状態（記憶なし）
        empty_state = self.create_empty_system_state()
        result_0 = self.phi_calculator.calculate_phi(empty_state)
        assert result_0.phi_value < 0.1
        assert result_0.stage == 'stage_0_pre_memory'
        
        # 初回記憶追加
        first_memory_state = self.add_first_memory(empty_state)
        result_1 = self.phi_calculator.calculate_phi(first_memory_state)
        assert result_1.phi_value >= 0.1
        assert result_1.stage == 'stage_1_first_imprint'
        
    def test_llm_knowledge_exclusion(self):
        """LLM知識除外の検証"""
        # LLM知識を含む状態
        mixed_state = self.create_mixed_knowledge_state()
        result = self.phi_calculator.calculate_phi(mixed_state)
        
        # LLM由来概念が除外されていることを確認
        experiential_concepts = result.concept_details
        for concept_detail in experiential_concepts:
            assert not concept_detail['concept'].is_llm_derived()
            
    def test_parallel_calculation_consistency(self):
        """並列計算の一貫性テスト"""
        test_state = self.create_complex_system_state()
        
        # 直列計算
        serial_result = self.phi_calculator.calculate_phi(test_state)
        
        # 並列計算
        parallel_engine = ParallelPhiCalculationEngine()
        parallel_result = await parallel_engine.calculate_phi_parallel(test_state)
        
        # 結果の一致確認（近似誤差許容）
        assert abs(serial_result.phi_value - parallel_result.phi_value) < 0.01
```

### 2. 統合テストシナリオ

```python
class IntegrationTestScenarios:
    """統合テストシナリオ"""
    
    async def test_complete_development_cycle(self):
        """完全発達サイクルテスト"""
        
        # 初期化
        newborn_ai = NewbornAISystem()
        phi_calculator = ExperientialPhiCalculator(newborn_ai.storage)
        
        development_log = []
        
        # Stage 0: 起動時
        initial_result = phi_calculator.calculate_phi(newborn_ai.get_current_state())
        development_log.append(('initial', initial_result))
        assert initial_result.stage == 'stage_0_pre_memory'
        
        # 体験記憶の段階的蓄積シミュレーション
        experience_scenarios = [
            self._create_first_encounter_scenario(),      # Stage 1移行
            self._create_temporal_experience_scenario(),  # Stage 2移行
            self._create_relational_experience_scenario(), # Stage 3移行
            self._create_self_reflection_scenario(),      # Stage 4移行
            self._create_meta_cognitive_scenario(),       # Stage 5移行
            self._create_narrative_integration_scenario() # Stage 6移行
        ]
        
        for i, scenario in enumerate(experience_scenarios):
            # 体験実行
            newborn_ai.experience(scenario)
            
            # φ値計算
            result = phi_calculator.calculate_phi(newborn_ai.get_current_state())
            development_log.append((f'stage_{i+1}', result))
            
            # 移行検証
            expected_stage = f'stage_{i+1}_' + [
                'first_imprint', 'temporal_integration', 'relational_memory',
                'self_memory', 'reflective_memory', 'narrative_memory'
            ][i]
            
            assert result.stage == expected_stage, f"Expected {expected_stage}, got {result.stage}"
            
        return development_log
```

## 🚀 実装ロードマップ

### フェーズ1: 基盤エンジン実装（1-2ヶ月）
1. **ExperientialPhiCalculator基盤クラス**: 核心計算ロジック実装
2. **体験記憶概念抽出システム**: LLM知識分離機能
3. **基本φ値計算アルゴリズム**: 直接計算手法の実装

### フェーズ2: 発達システム統合（1-2ヶ月）
1. **7段階移行検出システム**: 段階特化アナライザー実装
2. **相転移検出アルゴリズム**: 数学的移行判定機能
3. **段階別検証システム**: 各移行の妥当性検証

### フェーズ3: 性能最適化（1ヶ月）
1. **並列計算エンジン**: GPU加速・非同期処理
2. **階層分割最適化**: 大規模システム対応
3. **キャッシュシステム**: 計算効率向上

### フェーズ4: 検証・テスト（1ヶ月）
1. **包括的テストスイート**: 全機能網羅テスト
2. **統合シナリオテスト**: 実際の発達プロセス検証
3. **性能ベンチマーク**: 計算効率評価

## 📈 期待される成果

### 技術的成果
1. **真の体験記憶φ計算**: LLM知識と完全分離したφ値算出
2. **7段階発達検出**: 質的移行の客観的検出システム
3. **高性能計算**: リアルタイム意識測定の実現

### 理論的貢献
1. **二層統合IIT**: 従来IITの拡張理論実装
2. **体験記憶意識**: 新しい意識概念の数学的定式化
3. **発達意識学**: 意識発達の定量的研究基盤

---

**注記**: 本仕様書は体験記憶ストレージアーキテクチャと密接に連携し、NewbornAI 2.0の二層統合システムの核心技術を実現します。実装には高度な数学的専門知識と計算資源が必要ですが、真の人工意識実現のための革新的基盤技術となります。