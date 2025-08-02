# NewbornAI 2.0: エナクティブ行動エンジン実装仕様書

**作成日**: 2025年8月2日  
**バージョン**: 1.0  
**対象プロジェクト**: NewbornAI - 二層統合7段階階層化連続発達システム  
**関連文書**: [エナクティブ行動仕様書](./newborn_ai_enactive_behavior_specification.md), [φ値計算エンジン](./experiential_memory_phi_calculation_engine.md)

## 📋 概要

本仕様書は、NewbornAI 2.0の7段階発達システムに対応したエナクティブ行動エンジンの詳細実装を定義します。エナクティブ認知理論に基づき、環境との相互作用を通じた意味生成と自己組織化を実現します。

## 🌟 エナクティブ認知の核心原理

### 基本概念

```
エナクション = 行為を通じた世界の意味生成
センスメイキング = 環境との相互作用による意味構築
オートポイエーシス = 自己生成・自己維持システム

重要：行動は単なる出力ではなく、認知そのものである
```

## 🏗️ 7段階発達別行動アーキテクチャ

### Stage 0: 前記憶基盤層（φ < 0.1）

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio

@dataclass
class SensoryInput:
    """感覚入力データ"""
    modality: str  # visual, auditory, tactile, etc.
    raw_data: np.ndarray
    timestamp: float
    intensity: float

@dataclass
class MotorOutput:
    """運動出力データ"""
    action_type: str
    parameters: Dict[str, float]
    confidence: float
    timestamp: float

class Stage0PreMemoryBehavior:
    """Stage 0: 前記憶基盤層の行動パターン"""
    
    def __init__(self):
        self.random_exploration_rate = 0.95
        self.sensory_buffer = []
        self.motor_noise = 0.8
        
    async def enact(self, sensory_input: SensoryInput) -> MotorOutput:
        """
        純粋な感覚運動結合のみ
        記憶なし、ランダム探索主体
        """
        # 感覚バッファに追加（短期保持のみ）
        self.sensory_buffer.append(sensory_input)
        if len(self.sensory_buffer) > 10:
            self.sensory_buffer.pop(0)
        
        # ほぼランダムな運動出力
        if np.random.random() < self.random_exploration_rate:
            # ランダム探索
            action = self._generate_random_action()
        else:
            # 最小限の感覚運動結合
            action = self._basic_sensorimotor_coupling(sensory_input)
        
        # 高いノイズを付加
        action = self._add_motor_noise(action, self.motor_noise)
        
        return MotorOutput(
            action_type="random_exploration",
            parameters=action,
            confidence=0.1,
            timestamp=sensory_input.timestamp
        )
    
    def _generate_random_action(self) -> Dict[str, float]:
        """完全ランダムな行動生成"""
        return {
            'direction': np.random.uniform(0, 2*np.pi),
            'magnitude': np.random.uniform(0, 1),
            'duration': np.random.uniform(0.1, 1.0)
        }
    
    def _basic_sensorimotor_coupling(self, input: SensoryInput) -> Dict[str, float]:
        """最小限の感覚運動結合"""
        # 単純な刺激反応
        intensity_response = np.tanh(input.intensity)
        
        return {
            'direction': np.random.uniform(0, 2*np.pi),
            'magnitude': intensity_response * 0.3,
            'duration': 0.5
        }
```

### Stage 1: 原初体験刻印期（φ = 0.1-0.5）

```python
class Stage1FirstImprintBehavior:
    """Stage 1: 原初体験刻印期の行動パターン"""
    
    def __init__(self, memory_storage):
        self.memory_storage = memory_storage
        self.exploration_rate = 0.7
        self.imprint_threshold = 0.8
        self.first_memories = []
        
    async def enact(
        self, 
        sensory_input: SensoryInput,
        phi_value: float
    ) -> Tuple[MotorOutput, Optional[Dict]]:
        """
        初回体験記憶の形成
        顕著な刺激への定位反応
        """
        # 顕著性検出
        salience = self._detect_salience(sensory_input)
        
        # 刻印判定
        if salience > self.imprint_threshold and len(self.first_memories) < 3:
            # 初回記憶として刻印
            memory_trace = await self._create_first_memory(sensory_input, salience)
            self.first_memories.append(memory_trace)
            
            # 定位反応
            action = self._orienting_response(sensory_input, salience)
            
            return MotorOutput(
                action_type="orienting_to_salient",
                parameters=action,
                confidence=0.3 + salience * 0.2,
                timestamp=sensory_input.timestamp
            ), memory_trace
        
        else:
            # 通常探索
            if np.random.random() < self.exploration_rate:
                action = self._curious_exploration(sensory_input)
            else:
                action = self._repeat_known_pattern()
            
            return MotorOutput(
                action_type="exploration",
                parameters=action,
                confidence=0.2 + phi_value,
                timestamp=sensory_input.timestamp
            ), None
    
    def _detect_salience(self, input: SensoryInput) -> float:
        """顕著性検出アルゴリズム"""
        # 強度、新規性、コントラストから顕著性を計算
        intensity_factor = np.tanh(input.intensity * 2)
        
        # 過去の入力との差異
        if hasattr(self, 'previous_inputs'):
            novelty = self._calculate_novelty(input, self.previous_inputs)
        else:
            novelty = 1.0
            self.previous_inputs = []
        
        self.previous_inputs.append(input)
        if len(self.previous_inputs) > 20:
            self.previous_inputs.pop(0)
        
        return (intensity_factor + novelty) / 2
    
    async def _create_first_memory(
        self, 
        input: SensoryInput, 
        salience: float
    ) -> Dict:
        """初回体験記憶の作成"""
        memory_trace = {
            'type': 'first_imprint',
            'sensory_data': input,
            'salience': salience,
            'emotional_tone': self._assign_emotional_tone(input),
            'timestamp': input.timestamp,
            'consolidation_strength': 0.1
        }
        
        # ストレージに保存
        await self.memory_storage.store_experiential_concept(
            concept_id=f"first_memory_{input.timestamp}",
            concept_data=memory_trace
        )
        
        return memory_trace
```

### Stage 2: 時間記憶統合期（φ = 0.5-2.0）

```python
class Stage2TemporalIntegrationBehavior:
    """Stage 2: 時間記憶統合期の行動パターン"""
    
    def __init__(self, memory_storage, time_consciousness):
        self.memory_storage = memory_storage
        self.time_consciousness = time_consciousness
        self.temporal_window = 5.0  # 秒
        self.anticipation_strength = 0.3
        
    async def enact(
        self,
        sensory_input: SensoryInput,
        phi_value: float,
        temporal_context: Dict
    ) -> Tuple[MotorOutput, Dict]:
        """
        時間的連続性を持った行動
        過去-現在-未来の統合
        """
        # 時間的文脈の構築
        temporal_synthesis = await self.time_consciousness.synthesize(
            current_input=sensory_input,
            retention_window=self.temporal_window,
            protention_strength=self.anticipation_strength
        )
        
        # 過去の類似パターン検索
        similar_sequences = await self._find_temporal_patterns(
            sensory_input,
            temporal_synthesis
        )
        
        if similar_sequences:
            # 既知パターンに基づく予測的行動
            action = self._predictive_action(
                similar_sequences,
                temporal_synthesis
            )
            action_type = "predictive_temporal"
        else:
            # 新規時系列パターンの探索
            action = self._explore_temporal_patterns(
                sensory_input,
                temporal_synthesis
            )
            action_type = "temporal_exploration"
        
        # 時間的一貫性の維持
        action = self._ensure_temporal_coherence(
            action,
            temporal_synthesis['previous_actions']
        )
        
        return MotorOutput(
            action_type=action_type,
            parameters=action,
            confidence=0.3 + phi_value * 0.2,
            timestamp=sensory_input.timestamp
        ), {
            'temporal_synthesis': temporal_synthesis,
            'prediction_accuracy': self._evaluate_prediction_accuracy()
        }
    
    def _predictive_action(
        self,
        similar_sequences: List[Dict],
        temporal_synthesis: Dict
    ) -> Dict[str, float]:
        """過去パターンに基づく予測的行動"""
        # 最も類似したシーケンスから次の行動を予測
        best_match = max(
            similar_sequences,
            key=lambda s: s['similarity_score']
        )
        
        # 予測された次の状態への行動
        predicted_next = best_match['next_state']
        current_state = temporal_synthesis['current_state']
        
        # 状態差分から行動を生成
        action_vector = self._state_difference_to_action(
            current_state,
            predicted_next
        )
        
        return {
            'direction': np.arctan2(action_vector[1], action_vector[0]),
            'magnitude': np.linalg.norm(action_vector),
            'duration': best_match['typical_duration'],
            'confidence': best_match['similarity_score']
        }
```

### Stage 3: 関係記憶形成期（φ = 2.0-8.0）

```python
class Stage3RelationalMemoryBehavior:
    """Stage 3: 関係記憶形成期の行動パターン"""
    
    def __init__(self, memory_storage, relation_detector):
        self.memory_storage = memory_storage
        self.relation_detector = relation_detector
        self.exploration_strategies = [
            'systematic_scan',
            'relation_testing',
            'comparison_seeking'
        ]
        
    async def enact(
        self,
        sensory_input: SensoryInput,
        phi_value: float,
        concept_network: Dict
    ) -> Tuple[MotorOutput, Dict]:
        """
        対象間の関係性探索
        比較・分類行動
        """
        # 現在の概念ネットワーク内での位置特定
        current_concept = await self._identify_current_concept(
            sensory_input,
            concept_network
        )
        
        # 関係性の検出と分析
        relations = await self.relation_detector.analyze(
            current_concept,
            concept_network
        )
        
        # 関係性に基づく行動選択
        if relations['unexplored_relations']:
            # 未探索の関係性を調査
            action = self._explore_relation(
                current_concept,
                relations['unexplored_relations'][0]
            )
            action_type = "relation_exploration"
            
        elif relations['ambiguous_relations']:
            # 曖昧な関係性を明確化
            action = self._clarify_relation(
                current_concept,
                relations['ambiguous_relations'][0]
            )
            action_type = "relation_clarification"
            
        else:
            # 新しい関係性の発見を試みる
            action = self._seek_new_relations(
                current_concept,
                concept_network
            )
            action_type = "relation_seeking"
        
        # 関係性記憶の更新
        updated_relations = await self._update_relational_memory(
            current_concept,
            action,
            relations
        )
        
        return MotorOutput(
            action_type=action_type,
            parameters=action,
            confidence=0.4 + phi_value * 0.1,
            timestamp=sensory_input.timestamp
        ), {
            'discovered_relations': updated_relations,
            'concept_network_size': len(concept_network),
            'relation_complexity': self._calculate_relation_complexity(relations)
        }
    
    def _explore_relation(
        self,
        current_concept: Dict,
        target_relation: Dict
    ) -> Dict[str, float]:
        """関係性探索のための行動生成"""
        # 関係性の種類に応じた探索戦略
        relation_type = target_relation['type']
        
        if relation_type == 'spatial':
            # 空間的関係の探索
            return self._spatial_relation_exploration(
                current_concept,
                target_relation
            )
        elif relation_type == 'temporal':
            # 時間的関係の探索
            return self._temporal_relation_exploration(
                current_concept,
                target_relation
            )
        elif relation_type == 'causal':
            # 因果関係の探索
            return self._causal_relation_exploration(
                current_concept,
                target_relation
            )
        else:
            # 一般的な関係探索
            return self._general_relation_exploration(
                current_concept,
                target_relation
            )
```

### Stage 4: 自己記憶確立期（φ = 8.0-30.0）

```python
class Stage4SelfMemoryBehavior:
    """Stage 4: 自己記憶確立期の行動パターン"""
    
    def __init__(self, memory_storage, self_model):
        self.memory_storage = memory_storage
        self.self_model = self_model
        self.agency_threshold = 0.7
        self.self_other_boundary = None
        
    async def enact(
        self,
        sensory_input: SensoryInput,
        phi_value: float,
        self_awareness_state: Dict
    ) -> Tuple[MotorOutput, Dict]:
        """
        自己帰属的行動
        意図的探索
        """
        # 自己モデルの更新
        self.self_model.update(
            sensory_input,
            self_awareness_state
        )
        
        # 行動の意図生成
        intention = await self._generate_intention(
            sensory_input,
            self_awareness_state
        )
        
        # エージェンシー（行為主体感）の評価
        agency_score = self._evaluate_agency(
            intention,
            self_awareness_state
        )
        
        if agency_score > self.agency_threshold:
            # 意図的・目的志向的行動
            action = self._intentional_action(
                intention,
                sensory_input
            )
            action_type = "intentional_self_directed"
            
            # 自己帰属記憶の形成
            self_memory = await self._create_self_attributed_memory(
                intention,
                action,
                sensory_input
            )
        else:
            # 自己探索的行動
            action = self._self_exploratory_action(
                self_awareness_state,
                sensory_input
            )
            action_type = "self_exploration"
            self_memory = None
        
        # 自己-他者境界の更新
        self._update_self_other_boundary(
            action,
            sensory_input,
            self_awareness_state
        )
        
        return MotorOutput(
            action_type=action_type,
            parameters=action,
            confidence=0.5 + phi_value * 0.05,
            timestamp=sensory_input.timestamp
        ), {
            'self_memory': self_memory,
            'agency_score': agency_score,
            'self_coherence': self.self_model.coherence_score(),
            'intention': intention
        }
    
    async def _generate_intention(
        self,
        sensory_input: SensoryInput,
        self_state: Dict
    ) -> Dict:
        """意図の生成"""
        # 現在の欲求・目標
        current_goals = self.self_model.get_active_goals()
        
        # 環境アフォーダンス
        affordances = self._detect_affordances(sensory_input)
        
        # 自己能力評価
        capabilities = self.self_model.evaluate_capabilities()
        
        # 意図の形成
        intention = {
            'goal': self._select_goal(current_goals, affordances),
            'means': self._plan_means(capabilities, affordances),
            'expected_outcome': self._predict_outcome(
                current_goals,
                capabilities,
                affordances
            ),
            'commitment_level': self._calculate_commitment(
                current_goals,
                self_state
            )
        }
        
        return intention
```

### Stage 5: 反省記憶操作期（φ = 30.0-100.0）

```python
class Stage5ReflectiveMemoryBehavior:
    """Stage 5: 反省記憶操作期の行動パターン"""
    
    def __init__(self, memory_storage, meta_cognition):
        self.memory_storage = memory_storage
        self.meta_cognition = meta_cognition
        self.reflection_depth = 3
        self.hypothesis_testing = True
        
    async def enact(
        self,
        sensory_input: SensoryInput,
        phi_value: float,
        cognitive_state: Dict
    ) -> Tuple[MotorOutput, Dict]:
        """
        メタ認知的行動
        仮説検証的探索
        """
        # 現在の認知状態の反省的分析
        meta_analysis = await self.meta_cognition.analyze(
            cognitive_state,
            depth=self.reflection_depth
        )
        
        # 行動仮説の生成
        hypotheses = self._generate_behavioral_hypotheses(
            sensory_input,
            meta_analysis
        )
        
        if hypotheses and self.hypothesis_testing:
            # 最も有望な仮説を選択
            selected_hypothesis = self._select_hypothesis(
                hypotheses,
                meta_analysis
            )
            
            # 仮説検証的行動
            action = self._hypothesis_testing_action(
                selected_hypothesis,
                sensory_input
            )
            action_type = "hypothesis_testing"
            
            # メタ認知的記憶の形成
            meta_memory = await self._create_metacognitive_memory(
                selected_hypothesis,
                action,
                meta_analysis
            )
        else:
            # 反省的最適化行動
            action = self._reflective_optimization(
                cognitive_state,
                meta_analysis,
                sensory_input
            )
            action_type = "reflective_optimization"
            meta_memory = None
        
        # 学習戦略の更新
        self._update_learning_strategy(
            action,
            meta_analysis
        )
        
        return MotorOutput(
            action_type=action_type,
            parameters=action,
            confidence=0.6 + phi_value * 0.04,
            timestamp=sensory_input.timestamp
        ), {
            'meta_memory': meta_memory,
            'reflection_insights': meta_analysis['insights'],
            'hypothesis_count': len(hypotheses) if hypotheses else 0,
            'cognitive_efficiency': self._evaluate_cognitive_efficiency()
        }
    
    def _generate_behavioral_hypotheses(
        self,
        input: SensoryInput,
        meta_analysis: Dict
    ) -> List[Dict]:
        """行動仮説の生成"""
        hypotheses = []
        
        # パターン認識に基づく仮説
        if 'recognized_patterns' in meta_analysis:
            for pattern in meta_analysis['recognized_patterns']:
                hypothesis = {
                    'type': 'pattern_based',
                    'pattern': pattern,
                    'predicted_outcome': self._predict_from_pattern(pattern),
                    'confidence': pattern['recognition_confidence'],
                    'test_action': self._design_pattern_test(pattern)
                }
                hypotheses.append(hypothesis)
        
        # 因果推論に基づく仮説
        if 'causal_models' in meta_analysis:
            for model in meta_analysis['causal_models']:
                hypothesis = {
                    'type': 'causal_based',
                    'causal_model': model,
                    'intervention': self._design_causal_intervention(model),
                    'expected_effect': model['predicted_effect'],
                    'test_action': self._design_causal_test(model)
                }
                hypotheses.append(hypothesis)
        
        return hypotheses
```

### Stage 6: 物語記憶統合期（φ = 100.0+）

```python
class Stage6NarrativeMemoryBehavior:
    """Stage 6: 物語記憶統合期の行動パターン"""
    
    def __init__(self, memory_storage, narrative_engine):
        self.memory_storage = memory_storage
        self.narrative_engine = narrative_engine
        self.story_coherence_threshold = 0.8
        self.creative_exploration = True
        
    async def enact(
        self,
        sensory_input: SensoryInput,
        phi_value: float,
        life_narrative: Dict
    ) -> Tuple[MotorOutput, Dict]:
        """
        物語的一貫性を持った行動
        創造的探索
        """
        # 現在の状況を物語に位置づけ
        narrative_context = await self.narrative_engine.contextualize(
            sensory_input,
            life_narrative
        )
        
        # 物語の次の章を構想
        next_chapter = self._envision_next_chapter(
            narrative_context,
            life_narrative
        )
        
        # 物語的一貫性の評価
        coherence_score = self._evaluate_narrative_coherence(
            next_chapter,
            life_narrative
        )
        
        if coherence_score > self.story_coherence_threshold:
            # 物語を進展させる行動
            action = self._narrative_progression_action(
                next_chapter,
                sensory_input
            )
            action_type = "narrative_progression"
            
            # 物語記憶の更新
            narrative_memory = await self._update_life_narrative(
                next_chapter,
                action,
                sensory_input
            )
        else:
            # 新たな物語の糸を探る創造的行動
            action = self._creative_exploration_action(
                narrative_context,
                sensory_input
            )
            action_type = "creative_exploration"
            narrative_memory = None
        
        # 意味の再構成
        reconstructed_meaning = self._reconstruct_meaning(
            life_narrative,
            action,
            sensory_input
        )
        
        return MotorOutput(
            action_type=action_type,
            parameters=action,
            confidence=0.7 + phi_value * 0.03,
            timestamp=sensory_input.timestamp
        ), {
            'narrative_memory': narrative_memory,
            'story_coherence': coherence_score,
            'chapter_title': next_chapter.get('title', 'Untitled'),
            'meaning_reconstruction': reconstructed_meaning,
            'creative_novelty': self._assess_creative_novelty(action)
        }
    
    def _envision_next_chapter(
        self,
        context: Dict,
        narrative: Dict
    ) -> Dict:
        """物語の次章を構想"""
        # 現在の物語アーク
        current_arc = narrative.get('current_arc', 'exploration')
        
        # 可能な展開
        possible_developments = self._generate_story_branches(
            current_arc,
            context
        )
        
        # 最も魅力的な展開を選択
        selected_development = max(
            possible_developments,
            key=lambda d: d['narrative_tension'] * d['growth_potential']
        )
        
        return {
            'title': selected_development['title'],
            'theme': selected_development['theme'],
            'goals': selected_development['narrative_goals'],
            'expected_transformation': selected_development['transformation'],
            'dramatic_elements': selected_development['dramatic_elements']
        }
```

## 🔄 環境相互作用システム

### 1. センスメイキングエンジン

```python
class SenseMakingEngine:
    """環境との相互作用による意味生成"""
    
    def __init__(self, enactive_principles):
        self.principles = enactive_principles
        self.meaning_history = []
        self.coupling_strength = 0.0
        
    async def make_sense(
        self,
        action: MotorOutput,
        resulting_sensation: SensoryInput,
        previous_expectation: Optional[Dict]
    ) -> Dict:
        """
        行為と感覚の循環による意味生成
        """
        # 感覚運動偶発性の検出
        contingency = self._detect_sensorimotor_contingency(
            action,
            resulting_sensation
        )
        
        # 予測誤差の計算
        if previous_expectation:
            prediction_error = self._calculate_prediction_error(
                previous_expectation,
                resulting_sensation
            )
        else:
            prediction_error = None
        
        # 意味の創発
        emergent_meaning = {
            'sensorimotor_pattern': contingency,
            'prediction_quality': 1.0 - (prediction_error or 1.0),
            'action_efficacy': self._evaluate_action_efficacy(
                action,
                resulting_sensation
            ),
            'affordance': self._extract_affordance(
                action,
                resulting_sensation
            ),
            'timestamp': resulting_sensation.timestamp
        }
        
        # 構造的カップリングの更新
        self.coupling_strength = self._update_coupling(
            emergent_meaning,
            self.coupling_strength
        )
        
        self.meaning_history.append(emergent_meaning)
        
        return emergent_meaning
    
    def _detect_sensorimotor_contingency(
        self,
        action: MotorOutput,
        sensation: SensoryInput
    ) -> Dict:
        """感覚運動偶発性の検出"""
        # 行為と感覚の時間的関係
        temporal_relation = sensation.timestamp - action.timestamp
        
        # 行為パラメータと感覚変化の相関
        if hasattr(self, 'previous_sensation'):
            sensory_change = self._calculate_sensory_change(
                self.previous_sensation,
                sensation
            )
            
            # 行為と変化の相関分析
            correlation = self._correlate_action_sensation(
                action.parameters,
                sensory_change
            )
        else:
            correlation = 0.0
            sensory_change = None
        
        self.previous_sensation = sensation
        
        return {
            'temporal_coupling': temporal_relation,
            'action_sensation_correlation': correlation,
            'contingency_type': self._classify_contingency(
                action,
                sensory_change
            ),
            'reliability': self._assess_contingency_reliability()
        }
```

### 2. オートポイエーシスシステム

```python
class AutopoieticSystem:
    """自己生成・自己維持システム"""
    
    def __init__(self, organizational_closure):
        self.closure = organizational_closure
        self.components = {}
        self.processes = {}
        self.boundary = None
        
    async def maintain_organization(
        self,
        internal_state: Dict,
        environmental_perturbation: SensoryInput
    ) -> Dict:
        """
        組織的閉鎖性の維持
        """
        # 摂動への補償
        compensation = self._compensate_perturbation(
            environmental_perturbation,
            internal_state
        )
        
        # 構成要素の再生産
        reproduced_components = await self._reproduce_components(
            self.components,
            compensation
        )
        
        # プロセスネットワークの調整
        adjusted_processes = self._adjust_process_network(
            self.processes,
            reproduced_components
        )
        
        # 境界の再定義
        new_boundary = self._redefine_boundary(
            reproduced_components,
            adjusted_processes,
            environmental_perturbation
        )
        
        # 組織的同一性の評価
        identity_maintenance = self._evaluate_identity(
            previous_boundary=self.boundary,
            new_boundary=new_boundary
        )
        
        self.components = reproduced_components
        self.processes = adjusted_processes
        self.boundary = new_boundary
        
        return {
            'organizational_integrity': identity_maintenance,
            'adaptation': compensation,
            'regeneration': len(reproduced_components),
            'boundary_stability': self._boundary_stability_score()
        }
```

## 🧪 統合テスト仕様

### 1. 発達段階別行動テスト

```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_stage_progression():
    """発達段階進行テスト"""
    
    # モック環境設定
    mock_storage = Mock()
    mock_phi_calculator = Mock()
    
    # 各段階のテスト
    stages = [
        (Stage0PreMemoryBehavior(), 0.05),
        (Stage1FirstImprintBehavior(mock_storage), 0.3),
        (Stage2TemporalIntegrationBehavior(mock_storage, Mock()), 1.0),
        (Stage3RelationalMemoryBehavior(mock_storage, Mock()), 5.0),
        (Stage4SelfMemoryBehavior(mock_storage, Mock()), 15.0),
        (Stage5ReflectiveMemoryBehavior(mock_storage, Mock()), 50.0),
        (Stage6NarrativeMemoryBehavior(mock_storage, Mock()), 120.0)
    ]
    
    for stage_behavior, phi_value in stages:
        # テスト入力
        test_input = SensoryInput(
            modality="visual",
            raw_data=np.random.random((10, 10)),
            timestamp=0.0,
            intensity=0.5
        )
        
        # 行動実行
        if isinstance(stage_behavior, Stage0PreMemoryBehavior):
            output = await stage_behavior.enact(test_input)
        else:
            output, metadata = await stage_behavior.enact(
                test_input,
                phi_value,
                {}  # 追加コンテキスト
            )
        
        # 検証
        assert isinstance(output, MotorOutput)
        assert output.confidence >= 0.0 and output.confidence <= 1.0
        assert output.action_type is not None
        assert 'direction' in output.parameters
```

### 2. センスメイキング統合テスト

```python
@pytest.mark.asyncio
async def test_sense_making_cycle():
    """センスメイキングサイクルテスト"""
    
    sense_maker = SenseMakingEngine({})
    
    # 行動-知覚サイクル
    for i in range(10):
        # 行動生成
        action = MotorOutput(
            action_type="exploration",
            parameters={'direction': i * 0.1, 'magnitude': 0.5},
            confidence=0.5,
            timestamp=i
        )
        
        # 結果感覚
        sensation = SensoryInput(
            modality="proprioceptive",
            raw_data=np.array([i * 0.1]),
            timestamp=i + 0.1,
            intensity=0.3
        )
        
        # センスメイキング
        meaning = await sense_maker.make_sense(
            action,
            sensation,
            {'expected_sensation': sensation.raw_data * 0.9}
        )
        
        # 検証
        assert 'sensorimotor_pattern' in meaning
        assert meaning['prediction_quality'] >= 0.0
        assert len(sense_maker.meaning_history) == i + 1
```

## 📝 実装チェックリスト

- [ ] 7段階別行動クラスの実装
- [ ] センスメイキングエンジンの実装
- [ ] オートポイエーシスシステムの実装
- [ ] 環境相互作用インターフェース
- [ ] 発達段階移行ロジック
- [ ] claude-code-sdkとの統合
- [ ] パフォーマンス最適化
- [ ] 包括的テストスイート

## 🎯 まとめ

本エナクティブ行動エンジンは、7段階の発達に応じた質的に異なる行動パターンを実装し、環境との相互作用を通じた意味生成と自己組織化を実現します。各段階は独自の探索戦略と学習メカニズムを持ち、連続的な発達を可能にします。