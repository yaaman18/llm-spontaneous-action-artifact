"""
エナクティブ記憶形成境界づけられたコンテキスト

弱い意味でのエナクティブ記憶システムにおける記憶形成の中核ドメインロジックを
カプセル化する。環境相互作用による動的記憶形成プロセスを管理。

ユビキタス言語:
- 記憶形成(Memory Formation): 環境相互作用による能動的記憶構築過程
- 行動記憶結合(Action-Memory Coupling): 行動と記憶の動的結合関係
- 時間性統合(Temporal Integration): 過去・現在・未来の時間的統合過程  
- 意味創発(Meaning Emergence): 記憶から新しい意味の自発的生成
- 記憶定着(Memory Consolidation): 身体的相互作用による記憶の固定化
- 環境錨定(Environmental Anchoring): 環境コンテクストによる記憶の錨定
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..aggregates.enactive_memory_aggregate import EnactiveMemoryAggregate
from ..value_objects.embodied_memory_feature import EmbodiedMemoryFeature
from ..value_objects.temporal_memory_context import TemporalMemoryContext
from ..value_objects.action_memory_trace import ActionMemoryTrace
from ..events.domain_events import (
    MemoryFormationInitiated,
    ActionMemoryCouplingEstablished,
    TemporalIntegrationCompleted,
    MeaningEmergenceDetected,
    MemoryConsolidationAchieved
)


class MemoryFormationStrategy(Enum):
    """記憶形成戦略の定義"""
    REACTIVE = "reactive"  # 反応的記憶形成
    PREDICTIVE = "predictive"  # 予測的記憶形成
    EXPLORATORY = "exploratory"  # 探索的記憶形成
    CONSOLIDATORY = "consolidatory"  # 統合的記憶形成


class EnvironmentalCouplingStrength(Enum):
    """環境結合強度の段階"""
    WEAK = "weak"  # 弱結合 (0.0-0.3)
    MODERATE = "moderate"  # 中結合 (0.3-0.7)
    STRONG = "strong"  # 強結合 (0.7-1.0)


@dataclass(frozen=True)
class MemoryFormationContext:
    """記憶形成コンテキスト値オブジェクト"""
    
    environmental_conditions: Dict[str, Any]
    """環境条件（温度、光、音、空間配置等）"""
    
    action_sequence: List[str]
    """実行された行動シーケンス"""
    
    sensorimotor_engagement: Dict[str, float]
    """感覚運動参与度（各感覚・運動の関与度）"""
    
    temporal_horizon: timedelta
    """時間的地平（記憶形成の時間的範囲）"""
    
    coupling_strength: float
    """環境結合強度 (0.0-1.0)"""
    
    formation_intention: MemoryFormationStrategy
    """形成意図・戦略"""


class EnactiveMemoryFormationContext:
    """
    エナクティブ記憶形成の境界づけられたコンテキスト
    
    記憶≠データストレージの原則に基づき、生きた記憶としての
    動的記憶形成システムを管理する。環境相互作用による
    記憶の能動的構築過程を統括する。
    
    主要責任:
    - 環境相互作用による記憶形成の orchestration
    - 行動-記憶結合の確立・維持
    - 時間性統合による意味創発の促進  
    - 記憶定着プロセスの管理
    """
    
    def __init__(self):
        """エナクティブ記憶形成コンテキストの初期化"""
        self._active_memory_aggregates: Dict[str, EnactiveMemoryAggregate] = {}
        self._formation_history: List[Dict[str, Any]] = []
        self._meaning_emergence_events: List[Dict[str, Any]] = []
        self._context_domain_events: List[Any] = []
        self._environmental_memory_anchors: Dict[str, List[str]] = {}
    
    def initiate_memory_formation(
        self,
        embodied_feature: EmbodiedMemoryFeature,
        formation_context: MemoryFormationContext,
        aggregate_id: Optional[str] = None
    ) -> str:
        """
        記憶形成プロセスの開始
        
        エナクティブ原則に基づき、環境相互作用を通じた
        能動的記憶形成を開始する。
        
        Args:
            embodied_feature: 身体化記憶特徴（基底となる感覚運動経験）
            formation_context: 記憶形成コンテキスト
            aggregate_id: 記憶集約識別子（省略時は自動生成）
            
        Returns:
            記憶集約の識別子
        """
        # 記憶集約の作成または取得
        if aggregate_id and aggregate_id in self._active_memory_aggregates:
            memory_aggregate = self._active_memory_aggregates[aggregate_id]
        else:
            memory_aggregate = EnactiveMemoryAggregate(
                aggregate_id=aggregate_id,
                initial_embodied_feature=embodied_feature
            )
            self._active_memory_aggregates[memory_aggregate.aggregate_id] = memory_aggregate
        
        # 環境錨定の確立
        environmental_anchor = self._establish_environmental_anchoring(
            formation_context, memory_aggregate.aggregate_id
        )
        
        # 行動記憶トレースの生成
        action_trace = self._generate_action_memory_trace(
            formation_context, embodied_feature
        )
        
        # 時間的コンテクストの構築
        temporal_context = self._construct_temporal_context(formation_context)
        
        try:
            # 記憶形成の実行
            formation_success = memory_aggregate.initiate_formation(
                formation_context=formation_context,
                environmental_anchor=environmental_anchor,
                action_trace=action_trace,
                temporal_context=temporal_context
            )
            
            # 形成イベントの記録
            self._record_formation_event(
                memory_aggregate.aggregate_id,
                formation_context,
                formation_success
            )
            
            # ドメインイベントの収集
            events = memory_aggregate.clear_domain_events()
            self._context_domain_events.extend(events)
            
            # 記憶形成開始イベントの発火
            formation_event = MemoryFormationInitiated(
                aggregate_id=memory_aggregate.aggregate_id,
                embodied_feature=embodied_feature,
                formation_context=formation_context,
                environmental_anchor=environmental_anchor,
                timestamp=datetime.now()
            )
            self._context_domain_events.append(formation_event)
            
            return memory_aggregate.aggregate_id
            
        except Exception as e:
            self._record_formation_failure(memory_aggregate.aggregate_id, formation_context, str(e))
            raise
    
    def establish_action_memory_coupling(
        self,
        aggregate_id: str,
        action_sequence: List[str],
        coupling_parameters: Dict[str, float]
    ) -> bool:
        """
        行動記憶結合の確立
        
        エナクティブ認知の核心原理である行動と記憶の動的結合を
        確立する。記憶は行動から分離されない生きたシステム。
        
        Args:
            aggregate_id: 記憶集約識別子
            action_sequence: 行動シーケンス
            coupling_parameters: 結合パラメータ（強度、タイミング等）
            
        Returns:
            結合確立の成功可否
        """
        memory_aggregate = self._get_memory_aggregate(aggregate_id)
        
        # 行動シーケンスの検証
        if not self._validate_action_sequence(action_sequence):
            return False
        
        # 結合強度の算出
        coupling_strength = self._calculate_coupling_strength(
            action_sequence, coupling_parameters
        )
        
        # 時間的整合性の検証
        temporal_coherence = self._verify_temporal_coherence(
            action_sequence, memory_aggregate.get_temporal_context()
        )
        
        if temporal_coherence < 0.3:  # 閾値未満は結合失敗
            return False
        
        try:
            # 行動記憶結合の確立
            coupling_established = memory_aggregate.establish_action_coupling(
                action_sequence=action_sequence,
                coupling_strength=coupling_strength,
                temporal_coherence=temporal_coherence
            )
            
            if coupling_established:
                # 結合確立イベントの発火
                coupling_event = ActionMemoryCouplingEstablished(
                    aggregate_id=aggregate_id,
                    action_sequence=action_sequence,
                    coupling_strength=coupling_strength,
                    temporal_coherence=temporal_coherence,
                    timestamp=datetime.now()
                )
                self._context_domain_events.append(coupling_event)
                
                # 意味創発の監視開始
                self._monitor_meaning_emergence(aggregate_id)
            
            return coupling_established
            
        except Exception as e:
            return False
    
    def integrate_temporal_dimensions(
        self,
        aggregate_id: str,
        past_memory_influences: List[str],
        present_sensorimotor_state: Dict[str, Any],
        future_anticipations: List[str]
    ) -> TemporalMemoryContext:
        """
        時間次元の統合
        
        エナクティブ記憶の時間性の内在を実現。過去・現在・未来の
        統合的時間的コンテクストを構築する。
        
        Args:
            aggregate_id: 記憶集約識別子
            past_memory_influences: 過去記憶の影響要因
            present_sensorimotor_state: 現在の感覚運動状態
            future_anticipations: 未来予期
            
        Returns:
            統合された時間的記憶コンテクスト
        """
        memory_aggregate = self._get_memory_aggregate(aggregate_id)
        
        # 過去次元の構築
        past_dimension = self._construct_past_dimension(
            past_memory_influences, memory_aggregate
        )
        
        # 現在次元の構築
        present_dimension = self._construct_present_dimension(
            present_sensorimotor_state, memory_aggregate
        )
        
        # 未来次元の構築
        future_dimension = self._construct_future_dimension(
            future_anticipations, memory_aggregate
        )
        
        # 時間的統合の実行
        integrated_context = memory_aggregate.integrate_temporal_dimensions(
            past_dimension=past_dimension,
            present_dimension=present_dimension,
            future_dimension=future_dimension
        )
        
        # 時間統合完了イベント
        integration_event = TemporalIntegrationCompleted(
            aggregate_id=aggregate_id,
            integrated_context=integrated_context,
            past_influences_count=len(past_memory_influences),
            future_anticipations_count=len(future_anticipations),
            timestamp=datetime.now()
        )
        self._context_domain_events.append(integration_event)
        
        return integrated_context
    
    def consolidate_memory_formation(
        self,
        aggregate_id: str,
        consolidation_criteria: Dict[str, float]
    ) -> bool:
        """
        記憶形成の定着化
        
        環境相互作用による記憶の固定化プロセス。
        エナクティブ原則に基づく意味の創発を確認。
        
        Args:
            aggregate_id: 記憶集約識別子
            consolidation_criteria: 定着化基準（安定性、一致性等）
            
        Returns:
            定着化の成功可否
        """
        memory_aggregate = self._get_memory_aggregate(aggregate_id)
        
        # 定着化基準の検証
        consolidation_metrics = self._evaluate_consolidation_metrics(
            memory_aggregate, consolidation_criteria
        )
        
        # 必要基準の達成確認
        required_criteria = ['stability', 'coherence', 'environmental_coupling', 'meaning_emergence']
        criteria_met = all(
            consolidation_metrics.get(criterion, 0.0) >= consolidation_criteria.get(criterion, 0.5)
            for criterion in required_criteria
        )
        
        if not criteria_met:
            return False
        
        try:
            # 記憶定着化の実行
            consolidation_success = memory_aggregate.consolidate_formation(
                consolidation_metrics
            )
            
            if consolidation_success:
                # 定着化達成イベント
                consolidation_event = MemoryConsolidationAchieved(
                    aggregate_id=aggregate_id,
                    consolidation_metrics=consolidation_metrics,
                    formation_duration=memory_aggregate.get_formation_duration(),
                    meaning_emergence_events=len(self._get_meaning_emergence_events(aggregate_id)),
                    timestamp=datetime.now()
                )
                self._context_domain_events.append(consolidation_event)
                
                # 記憶形成完了の記録
                self._record_formation_completion(aggregate_id, consolidation_metrics)
            
            return consolidation_success
            
        except Exception as e:
            return False
    
    def monitor_meaning_emergence(self, aggregate_id: str) -> List[Dict[str, Any]]:
        """
        意味創発の監視
        
        エナクティブ記憶システムの核心機能である意味の創発を
        継続的に監視し、創発イベントを記録する。
        
        Args:
            aggregate_id: 記憶集約識別子
            
        Returns:
            検出された意味創発イベントのリスト
        """
        memory_aggregate = self._get_memory_aggregate(aggregate_id)
        
        # 現在の意味状態の取得
        current_meaning_state = memory_aggregate.get_current_meaning_state()
        
        # 意味創発パターンの検出
        emergence_patterns = self._detect_emergence_patterns(
            current_meaning_state, memory_aggregate
        )
        
        # 新しい創発イベントの識別
        new_emergence_events = []
        for pattern in emergence_patterns:
            if self._is_novel_emergence_pattern(pattern, aggregate_id):
                emergence_event = {
                    'aggregate_id': aggregate_id,
                    'pattern_type': pattern.get('type', 'unknown'),
                    'emergence_strength': pattern.get('strength', 0.0),
                    'pattern_complexity': pattern.get('complexity', 0.0),
                    'timestamp': datetime.now(),
                    'context_factors': pattern.get('context_factors', {})
                }
                new_emergence_events.append(emergence_event)
                
                # ドメインイベントとして記録
                domain_event = MeaningEmergenceDetected(
                    aggregate_id=aggregate_id,
                    pattern_type=pattern.get('type', 'unknown'),
                    emergence_strength=pattern.get('strength', 0.0),
                    context_factors=pattern.get('context_factors', {}),
                    timestamp=datetime.now()
                )
                self._context_domain_events.append(domain_event)
        
        # 意味創発イベントの記録
        self._meaning_emergence_events.extend(new_emergence_events)
        
        return new_emergence_events
    
    def get_formation_history(self) -> List[Dict[str, Any]]:
        """記憶形成履歴の取得"""
        return self._formation_history.copy()
    
    def get_meaning_emergence_events(self, aggregate_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """意味創発イベントの取得"""
        if aggregate_id:
            return [event for event in self._meaning_emergence_events 
                   if event.get('aggregate_id') == aggregate_id]
        return self._meaning_emergence_events.copy()
    
    def get_context_domain_events(self) -> List[Any]:
        """コンテキストドメインイベントの取得"""
        return self._context_domain_events.copy()
    
    def clear_context_events(self) -> List[Any]:
        """コンテキストイベントのクリアと返却"""
        events = self._context_domain_events.copy()
        self._context_domain_events.clear()
        return events
    
    # プライベートヘルパーメソッド
    
    def _get_memory_aggregate(self, aggregate_id: str) -> EnactiveMemoryAggregate:
        """記憶集約の取得"""
        if aggregate_id not in self._active_memory_aggregates:
            raise ValueError(f"Memory aggregate {aggregate_id} not found")
        return self._active_memory_aggregates[aggregate_id]
    
    def _establish_environmental_anchoring(
        self,
        formation_context: MemoryFormationContext,
        aggregate_id: str
    ) -> Dict[str, Any]:
        """環境錨定の確立"""
        environmental_anchor = {
            'anchor_id': f"env_anchor_{aggregate_id}_{datetime.now().isoformat()}",
            'environmental_conditions': formation_context.environmental_conditions,
            'coupling_strength': formation_context.coupling_strength,
            'spatial_context': formation_context.environmental_conditions.get('spatial_context', {}),
            'temporal_context': formation_context.environmental_conditions.get('temporal_context', {}),
            'sensory_context': formation_context.environmental_conditions.get('sensory_context', {})
        }
        
        # 環境錨定の記録
        env_key = str(formation_context.environmental_conditions.get('location', 'unknown'))
        if env_key not in self._environmental_memory_anchors:
            self._environmental_memory_anchors[env_key] = []
        self._environmental_memory_anchors[env_key].append(aggregate_id)
        
        return environmental_anchor
    
    def _generate_action_memory_trace(
        self,
        formation_context: MemoryFormationContext,
        embodied_feature: EmbodiedMemoryFeature
    ) -> ActionMemoryTrace:
        """行動記憶トレースの生成"""
        return ActionMemoryTrace(
            action_sequence=formation_context.action_sequence,
            sensorimotor_engagement=formation_context.sensorimotor_engagement,
            embodied_context=embodied_feature,
            temporal_span=formation_context.temporal_horizon,
            trace_strength=formation_context.coupling_strength
        )
    
    def _construct_temporal_context(
        self,
        formation_context: MemoryFormationContext
    ) -> TemporalMemoryContext:
        """時間的コンテクストの構築"""
        return TemporalMemoryContext(
            temporal_horizon=formation_context.temporal_horizon,
            formation_timestamp=datetime.now(),
            temporal_coherence=0.8,  # 初期値
            past_influences=[],  # 初期形成では空
            future_anticipations=[]  # 初期形成では空
        )
    
    def _record_formation_event(
        self,
        aggregate_id: str,
        formation_context: MemoryFormationContext,
        success: bool
    ) -> None:
        """記憶形成イベントの記録"""
        formation_record = {
            'aggregate_id': aggregate_id,
            'timestamp': datetime.now().isoformat(),
            'formation_strategy': formation_context.formation_intention.value,
            'environmental_coupling': formation_context.coupling_strength,
            'action_sequence_length': len(formation_context.action_sequence),
            'sensorimotor_engagement_count': len(formation_context.sensorimotor_engagement),
            'formation_successful': success,
            'temporal_horizon_minutes': formation_context.temporal_horizon.total_seconds() / 60
        }
        self._formation_history.append(formation_record)
    
    def _monitor_meaning_emergence(self, aggregate_id: str) -> None:
        """意味創発の監視開始（バックグラウンド処理のトリガー）"""
        # 実装では、意味創発監視のバックグラウンドプロセスを開始
        # ここでは監視開始の記録のみ
        monitoring_record = {
            'aggregate_id': aggregate_id,
            'monitoring_started': datetime.now().isoformat(),
            'monitoring_active': True
        }
        # 監視状態の記録（実装依存）
    
    def _get_meaning_emergence_events(self, aggregate_id: str) -> List[Dict[str, Any]]:
        """特定集約の意味創発イベント取得"""
        return [event for event in self._meaning_emergence_events 
               if event.get('aggregate_id') == aggregate_id]
    
    # その他のヘルパーメソッドは実装省略（実際のプロジェクトでは完全実装が必要）
    def _validate_action_sequence(self, action_sequence: List[str]) -> bool:
        return len(action_sequence) > 0
    
    def _calculate_coupling_strength(self, action_sequence: List[str], parameters: Dict[str, float]) -> float:
        return parameters.get('base_strength', 0.5)
    
    def _verify_temporal_coherence(self, action_sequence: List[str], temporal_context: Any) -> float:
        return 0.7  # 暫定値
    
    def _construct_past_dimension(self, influences: List[str], aggregate: EnactiveMemoryAggregate) -> Dict[str, Any]:
        return {'influences': influences}
    
    def _construct_present_dimension(self, state: Dict[str, Any], aggregate: EnactiveMemoryAggregate) -> Dict[str, Any]:
        return {'state': state}
    
    def _construct_future_dimension(self, anticipations: List[str], aggregate: EnactiveMemoryAggregate) -> Dict[str, Any]:
        return {'anticipations': anticipations}
    
    def _evaluate_consolidation_metrics(self, aggregate: EnactiveMemoryAggregate, criteria: Dict[str, float]) -> Dict[str, float]:
        return {'stability': 0.8, 'coherence': 0.7, 'environmental_coupling': 0.9, 'meaning_emergence': 0.6}
    
    def _record_formation_completion(self, aggregate_id: str, metrics: Dict[str, float]) -> None:
        pass
    
    def _record_formation_failure(self, aggregate_id: str, context: MemoryFormationContext, error: str) -> None:
        pass
    
    def _detect_emergence_patterns(self, meaning_state: Any, aggregate: EnactiveMemoryAggregate) -> List[Dict[str, Any]]:
        return []  # 暫定実装
    
    def _is_novel_emergence_pattern(self, pattern: Dict[str, Any], aggregate_id: str) -> bool:
        return True  # 暫定実装