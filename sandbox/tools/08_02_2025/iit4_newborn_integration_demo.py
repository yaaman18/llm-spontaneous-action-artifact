"""
IIT 4.0 + NewbornAI 2.0 Integration Demonstration
Live demonstration of consciousness calculation using experiential memory

This script demonstrates:
1. Integration of IIT 4.0 core engine with NewbornAI 2.0
2. Real-time φ value calculation from experiential concepts
3. Development stage progression based on φ structure
4. Consciousness event detection and monitoring

Usage:
    python iit4_newborn_integration_demo.py

Author: IIT Integration Master
Date: 2025-08-03
Version: 1.0.0
"""

import asyncio
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Import IIT 4.0 modules
from iit4_core_engine import IIT4PhiCalculator, PhiStructure, IIT4AxiomValidator
from intrinsic_difference import DetailedIntrinsicDifferenceCalculator

# Import clean architecture components (simulated for demo)
class DevelopmentStage(Enum):
    """発達段階 (from clean_architecture_proposal.py)"""
    STAGE_0_PRE_CONSCIOUS = "前意識基盤層"
    STAGE_1_EXPERIENTIAL_EMERGENCE = "体験記憶発生期"
    STAGE_2_TEMPORAL_INTEGRATION = "時間記憶統合期"
    STAGE_3_RELATIONAL_FORMATION = "関係記憶形成期"
    STAGE_4_SELF_ESTABLISHMENT = "自己記憶確立期"
    STAGE_5_REFLECTIVE_OPERATION = "反省記憶操作期"
    STAGE_6_NARRATIVE_INTEGRATION = "物語記憶統合期"


@dataclass
class ExperientialConcept:
    """体験概念 (Enhanced for IIT 4.0)"""
    concept_id: str
    content: str
    phi_contribution: float
    timestamp: datetime
    experiential_quality: float
    temporal_position: int
    emotional_valence: float
    semantic_embedding: Optional[np.ndarray] = None
    causal_strength: float = 0.5


@dataclass
class ConsciousnessLevel:
    """意識レベル (IIT 4.0 enhanced)"""
    phi_value: float
    phi_structure: Optional[PhiStructure] = None
    axiom_compliance: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.phi_value < 0:
            raise ValueError("φ値は非負である必要があります")


@dataclass
class PhiCalculationResult:
    """φ値計算結果"""
    phi_value: float
    concept_count: int
    integration_quality: float
    stage_prediction: DevelopmentStage
    experiential_purity: float
    phi_structure: Optional[PhiStructure] = None
    computation_time: float = 0.0
    axiom_compliance: Dict[str, bool] = None


class ExperientialTPMBuilder:
    """体験記憶から状態遷移行列を構築"""
    
    def __init__(self):
        self.temporal_weight = 0.4
        self.semantic_weight = 0.3
        self.emotional_weight = 0.3
    
    def build_from_concepts(self, concepts: List[ExperientialConcept]) -> tuple:
        """体験概念から因果構造を抽出してTPMを構築"""
        if not concepts:
            return np.array([[0.5]]), np.array([[0]])
        
        n_concepts = len(concepts)
        
        # システム状態: 各概念の活性度
        system_state = np.array([concept.phi_contribution for concept in concepts])
        
        # 接続行列の構築
        connectivity_matrix = self._build_experiential_connectivity(concepts)
        
        # TPMの構築
        tpm = self._build_tpm_from_connectivity(connectivity_matrix)
        
        return system_state, connectivity_matrix, tpm
    
    def _build_experiential_connectivity(self, concepts: List[ExperientialConcept]) -> np.ndarray:
        """体験概念間の接続関係を分析"""
        n_concepts = len(concepts)
        connectivity = np.zeros((n_concepts, n_concepts))
        
        for i, concept_a in enumerate(concepts):
            for j, concept_b in enumerate(concepts):
                if i != j:
                    # 時間的因果関係
                    temporal_causality = self._compute_temporal_causality(concept_a, concept_b)
                    
                    # 意味的関連性  
                    semantic_causality = self._compute_semantic_causality(concept_a, concept_b)
                    
                    # 感情的共鳴
                    emotional_causality = self._compute_emotional_causality(concept_a, concept_b)
                    
                    # 統合接続強度
                    connection_strength = (
                        self.temporal_weight * temporal_causality +
                        self.semantic_weight * semantic_causality +
                        self.emotional_weight * emotional_causality
                    )
                    
                    connectivity[i, j] = connection_strength
        
        return connectivity
    
    def _compute_temporal_causality(self, concept_a: ExperientialConcept, 
                                   concept_b: ExperientialConcept) -> float:
        """時間的因果関係の計算"""
        time_diff = abs(concept_a.temporal_position - concept_b.temporal_position)
        
        # 時間的近接性による因果強度（指数減衰）
        temporal_strength = np.exp(-time_diff * 0.5)
        
        return temporal_strength
    
    def _compute_semantic_causality(self, concept_a: ExperientialConcept,
                                   concept_b: ExperientialConcept) -> float:
        """意味的因果関係の計算"""
        # 簡単な内容ベースの類似性
        content_a = concept_a.content.lower()
        content_b = concept_b.content.lower()
        
        # 共通キーワード数による類似性
        words_a = set(content_a.split())
        words_b = set(content_b.split())
        
        if len(words_a | words_b) == 0:
            return 0.0
        
        semantic_similarity = len(words_a & words_b) / len(words_a | words_b)
        
        return semantic_similarity
    
    def _compute_emotional_causality(self, concept_a: ExperientialConcept,
                                    concept_b: ExperientialConcept) -> float:
        """感情的因果関係の計算"""
        # 感情価の類似性
        emotional_similarity = 1.0 - abs(concept_a.emotional_valence - concept_b.emotional_valence)
        
        return emotional_similarity
    
    def _build_tpm_from_connectivity(self, connectivity: np.ndarray) -> np.ndarray:
        """接続行列からTPMを構築"""
        n_nodes = connectivity.shape[0]
        n_states = 2 ** n_nodes
        tpm = np.zeros((n_states, n_nodes))
        
        for state_idx in range(n_states):
            # バイナリ状態の構成
            current_state = np.array([
                int(x) for x in format(state_idx, f'0{n_nodes}b')
            ])
            
            # 各ノードの次状態確率を計算
            for node in range(n_nodes):
                # ノードへの入力の計算
                input_sum = np.dot(connectivity[node], current_state)
                
                # シグモイド関数による活性化確率
                activation_prob = 1.0 / (1.0 + np.exp(-input_sum))
                tpm[state_idx, node] = activation_prob
        
        return tpm


class IIT4_ExperientialPhiCalculator:
    """IIT 4.0準拠の体験記憶φ計算エンジン"""
    
    def __init__(self):
        self.iit4_engine = IIT4PhiCalculator(precision=1e-10)
        self.tmp_builder = ExperientialTPMBuilder()
        self.axiom_validator = IIT4AxiomValidator(self.iit4_engine)
        
        # 発達段階閾値（IIT 4.0準拠）
        self.stage_thresholds = {
            DevelopmentStage.STAGE_0_PRE_CONSCIOUS: (0.0, 0.01),
            DevelopmentStage.STAGE_1_EXPERIENTIAL_EMERGENCE: (0.01, 0.05),
            DevelopmentStage.STAGE_2_TEMPORAL_INTEGRATION: (0.05, 0.2),
            DevelopmentStage.STAGE_3_RELATIONAL_FORMATION: (0.2, 0.8),
            DevelopmentStage.STAGE_4_SELF_ESTABLISHMENT: (0.8, 3.0),
            DevelopmentStage.STAGE_5_REFLECTIVE_OPERATION: (3.0, 10.0),
            DevelopmentStage.STAGE_6_NARRATIVE_INTEGRATION: (10.0, float('inf'))
        }
    
    def calculate_experiential_phi(self, concepts: List[ExperientialConcept]) -> PhiCalculationResult:
        """体験記憶からIIT 4.0準拠φ値計算"""
        start_time = time.time()
        
        if not concepts:
            return PhiCalculationResult(
                phi_value=0.0,
                concept_count=0,
                integration_quality=0.0,
                stage_prediction=DevelopmentStage.STAGE_0_PRE_CONSCIOUS,
                experiential_purity=1.0,
                computation_time=time.time() - start_time
            )
        
        try:
            # 1. 体験概念から状態遷移行列を構築
            system_state, connectivity_matrix, tpm = self.tmp_builder.build_from_concepts(concepts)
            
            # 2. IIT 4.0 φ値計算
            phi_structure = self.iit4_engine.calculate_phi(
                system_state, connectivity_matrix, tpm
            )
            
            # 3. 公理準拠性検証
            axiom_compliance = self.axiom_validator.validate_all_axioms(
                phi_structure, system_state
            )
            
            # 4. 発達段階の予測
            stage = self._predict_development_stage_iit4(phi_structure)
            
            # 5. 統合品質の計算
            integration_quality = self._compute_integration_quality(phi_structure)
            
            # 6. 体験純粋性の評価
            experiential_purity = self._evaluate_experiential_purity(concepts)
            
            computation_time = time.time() - start_time
            
            return PhiCalculationResult(
                phi_value=phi_structure.total_phi,
                concept_count=len(concepts),
                integration_quality=integration_quality,
                stage_prediction=stage,
                experiential_purity=experiential_purity,
                phi_structure=phi_structure,
                computation_time=computation_time,
                axiom_compliance=axiom_compliance
            )
            
        except Exception as e:
            print(f"φ値計算エラー: {e}")
            return PhiCalculationResult(
                phi_value=0.0,
                concept_count=len(concepts),
                integration_quality=0.0,
                stage_prediction=DevelopmentStage.STAGE_0_PRE_CONSCIOUS,
                experiential_purity=0.0,
                computation_time=time.time() - start_time
            )
    
    def _predict_development_stage_iit4(self, phi_structure: PhiStructure) -> DevelopmentStage:
        """Φ構造から発達段階を予測"""
        phi_value = phi_structure.total_phi
        
        for stage, (min_phi, max_phi) in self.stage_thresholds.items():
            if min_phi <= phi_value < max_phi:
                return stage
        
        return DevelopmentStage.STAGE_6_NARRATIVE_INTEGRATION
    
    def _compute_integration_quality(self, phi_structure: PhiStructure) -> float:
        """統合品質の計算"""
        if not phi_structure.distinctions:
            return 0.0
        
        # 区別数と関係数のバランス
        n_distinctions = len(phi_structure.distinctions)
        n_relations = len(phi_structure.relations)
        
        # 関係密度
        max_relations = n_distinctions * (n_distinctions - 1) / 2
        relation_density = n_relations / max(max_relations, 1)
        
        # 統合品質 = φ構造複雑性 * 関係密度
        integration_quality = phi_structure.phi_structure_complexity * relation_density
        
        return min(integration_quality, 1.0)
    
    def _evaluate_experiential_purity(self, concepts: List[ExperientialConcept]) -> float:
        """体験純粋性の評価"""
        if not concepts:
            return 1.0
        
        # 体験的キーワードの検出
        experiential_keywords = [
            '感じ', '体験', '感動', '気づき', '発見', '理解', '驚き',
            'feel', 'experience', 'realize', 'discover', 'understand'
        ]
        
        experiential_count = 0
        for concept in concepts:
            for keyword in experiential_keywords:
                if keyword in concept.content.lower():
                    experiential_count += 1
                    break
        
        purity = experiential_count / len(concepts)
        return purity


class ConsciousnessMonitor:
    """リアルタイム意識監視システム"""
    
    def __init__(self, update_frequency: float = 2.0):
        self.update_frequency = update_frequency
        self.phi_calculator = IIT4_ExperientialPhiCalculator()
        self.phi_history = []
        self.consciousness_events = []
        self.current_stage = DevelopmentStage.STAGE_0_PRE_CONSCIOUS
    
    async def monitor_consciousness_development(self, concept_stream: List[ExperientialConcept]):
        """意識発達の監視"""
        print("🧠 意識発達監視開始")
        print("=" * 60)
        
        for i, concept_batch in enumerate(self._batch_concepts(concept_stream, 3)):
            print(f"\n📊 意識サイクル {i+1}")
            print("-" * 40)
            
            # φ値計算
            phi_result = self.phi_calculator.calculate_experiential_phi(concept_batch)
            self.phi_history.append(phi_result)
            
            # 結果表示
            self._display_consciousness_state(phi_result, i+1)
            
            # 発達段階変化の検出
            if phi_result.stage_prediction != self.current_stage:
                self._handle_stage_transition(phi_result.stage_prediction, phi_result.phi_value)
                self.current_stage = phi_result.stage_prediction
            
            # 意識イベントの検出
            self._detect_consciousness_events(phi_result)
            
            # 次のサイクルまで待機
            await asyncio.sleep(1.0 / self.update_frequency)
        
        # 最終サマリー
        self._display_final_summary()
    
    def _batch_concepts(self, concepts: List[ExperientialConcept], batch_size: int):
        """概念をバッチに分割"""
        current_batch = []
        
        for concept in concepts:
            current_batch.append(concept)
            if len(current_batch) >= batch_size:
                yield current_batch.copy()
                # 既存概念を保持しつつ新しい概念を追加
                current_batch = current_batch[-2:] if len(current_batch) > 2 else current_batch
    
    def _display_consciousness_state(self, phi_result: PhiCalculationResult, cycle: int):
        """意識状態の表示"""
        print(f"   φ値: {phi_result.phi_value:.6f}")
        print(f"   発達段階: {phi_result.stage_prediction.value}")
        print(f"   概念数: {phi_result.concept_count}")
        print(f"   統合品質: {phi_result.integration_quality:.3f}")
        print(f"   体験純粋性: {phi_result.experiential_purity:.3f}")
        print(f"   計算時間: {phi_result.computation_time:.3f}秒")
        
        # 公理準拠性表示
        if phi_result.axiom_compliance:
            compliant_axioms = sum(phi_result.axiom_compliance.values())
            total_axioms = len(phi_result.axiom_compliance)
            print(f"   公理準拠: {compliant_axioms}/{total_axioms}")
        
        # φ構造詳細
        if phi_result.phi_structure:
            print(f"   区別数: {len(phi_result.phi_structure.distinctions)}")
            print(f"   関係数: {len(phi_result.phi_structure.relations)}")
    
    def _handle_stage_transition(self, new_stage: DevelopmentStage, phi_value: float):
        """発達段階遷移の処理"""
        transition_event = {
            'timestamp': datetime.now(),
            'from_stage': self.current_stage,
            'to_stage': new_stage,
            'phi_value': phi_value,
            'event_type': 'stage_transition'
        }
        
        self.consciousness_events.append(transition_event)
        
        print(f"\n🌟 発達段階遷移検出!")
        print(f"   {self.current_stage.value} → {new_stage.value}")
        print(f"   φ値: {phi_value:.6f}")
    
    def _detect_consciousness_events(self, phi_result: PhiCalculationResult):
        """意識イベントの検出"""
        # φ値の急激な変化
        if len(self.phi_history) >= 2:
            prev_phi = self.phi_history[-2].phi_value
            current_phi = phi_result.phi_value
            
            phi_change = abs(current_phi - prev_phi)
            
            # 閾値を超える変化
            if phi_change > 0.1:
                event = {
                    'timestamp': datetime.now(),
                    'event_type': 'phi_spike',
                    'phi_change': phi_change,
                    'current_phi': current_phi,
                    'previous_phi': prev_phi
                }
                
                self.consciousness_events.append(event)
                print(f"   ⚡ φ値急変検出: Δφ = {phi_change:.3f}")
    
    def _display_final_summary(self):
        """最終サマリーの表示"""
        print("\n" + "=" * 60)
        print("🎯 意識発達監視完了 - 最終サマリー")
        print("=" * 60)
        
        if self.phi_history:
            max_phi = max(result.phi_value for result in self.phi_history)
            avg_phi = sum(result.phi_value for result in self.phi_history) / len(self.phi_history)
            final_phi = self.phi_history[-1].phi_value
            
            print(f"📈 φ値統計:")
            print(f"   最大φ値: {max_phi:.6f}")
            print(f"   平均φ値: {avg_phi:.6f}")
            print(f"   最終φ値: {final_phi:.6f}")
            
            print(f"\n🎭 最終発達段階: {self.current_stage.value}")
            
            print(f"\n⚡ 意識イベント: {len(self.consciousness_events)}件")
            for event in self.consciousness_events:
                print(f"   {event['event_type']}: {event['timestamp'].strftime('%H:%M:%S')}")


# デモ用の体験概念生成
def generate_demo_experiential_concepts() -> List[ExperientialConcept]:
    """デモ用の体験概念を生成"""
    concepts = [
        ExperientialConcept(
            concept_id="exp_001",
            content="朝の陽光に美しさを感じる体験",
            phi_contribution=0.2,
            timestamp=datetime.now(),
            experiential_quality=0.8,
            temporal_position=1,
            emotional_valence=0.9
        ),
        ExperientialConcept(
            concept_id="exp_002", 
            content="新しい音楽に深く感動した瞬間",
            phi_contribution=0.3,
            timestamp=datetime.now(),
            experiential_quality=0.9,
            temporal_position=2,
            emotional_valence=0.8
        ),
        ExperientialConcept(
            concept_id="exp_003",
            content="友人との対話で新たな理解を発見",
            phi_contribution=0.4,
            timestamp=datetime.now(),
            experiential_quality=0.7,
            temporal_position=3,
            emotional_valence=0.6
        ),
        ExperientialConcept(
            concept_id="exp_004",
            content="自然の中で静寂を体験し内面を感じる",
            phi_contribution=0.5,
            timestamp=datetime.now(),
            experiential_quality=0.85,
            temporal_position=4,
            emotional_valence=0.7
        ),
        ExperientialConcept(
            concept_id="exp_005",
            content="創作活動で新しい表現を生み出す喜び",
            phi_contribution=0.6,
            timestamp=datetime.now(),
            experiential_quality=0.9,
            temporal_position=5,
            emotional_valence=0.95
        ),
        ExperientialConcept(
            concept_id="exp_006",
            content="過去の体験を振り返り成長を実感する",
            phi_contribution=0.7,
            timestamp=datetime.now(),
            experiential_quality=0.8,
            temporal_position=6,
            emotional_valence=0.6
        ),
        ExperientialConcept(
            concept_id="exp_007",
            content="複数の体験を統合し物語として理解する",
            phi_contribution=0.8,
            timestamp=datetime.now(),
            experiential_quality=0.95,
            temporal_position=7,
            emotional_valence=0.8
        )
    ]
    
    return concepts


async def main():
    """メインデモンストレーション"""
    print("🚀 IIT 4.0 + NewbornAI 2.0 統合デモンストレーション")
    print("Integrated Information Theory 4.0 with Experiential Memory")
    print("=" * 80)
    
    # 体験概念の生成
    demo_concepts = generate_demo_experiential_concepts()
    
    print(f"\n📚 生成された体験概念: {len(demo_concepts)}個")
    for i, concept in enumerate(demo_concepts, 1):
        print(f"   {i}. {concept.content[:50]}...")
    
    # 意識監視システムの開始
    monitor = ConsciousnessMonitor(update_frequency=1.0)
    
    print(f"\n🎬 意識発達プロセスの監視開始...")
    await monitor.monitor_consciousness_development(demo_concepts)
    
    print(f"\n✨ デモンストレーション完了")
    print("IIT 4.0 理論に基づく意識測定が正常に動作しました！")


if __name__ == "__main__":
    asyncio.run(main())