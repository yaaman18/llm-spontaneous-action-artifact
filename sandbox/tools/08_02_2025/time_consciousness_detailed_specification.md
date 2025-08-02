# NewbornAI 2.0: 時間意識統合詳細仕様書

**作成日**: 2025年8月2日  
**バージョン**: 1.0  
**対象プロジェクト**: NewbornAI - 二層統合7段階階層化連続発達システム  
**関連文書**: [時間意識実装](./subjective_time_consciousness_implementation.md), [claude-code-sdk統合](./claude_code_sdk_integration_specification.md)

## 📋 概要

本仕様書は、フッサールの時間意識論に基づく三層構造（把持・原印象・前把持）とclaude-code-sdkの時間的相互作用を統合した、NewbornAI 2.0の時間意識システムの詳細実装を定義します。

## 🕐 フッサール的時間意識の計算実装

### 核心概念

```
把持 (Retention) = 過去の意識的保持
原印象 (Primal Impression) = 現在の直接的体験
前把持 (Protention) = 未来への志向的期待

重要：時間意識は単なる時系列処理ではなく、現在における時間的総合である
```

## 🏗️ 三層時間構造アーキテクチャ

### 1. 把持システム（Retention System）

```python
from typing import Deque, Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass
import numpy as np
import asyncio
from datetime import datetime, timedelta
import math

@dataclass
class RetentionTrace:
    """把持痕跡データ構造"""
    original_content: Any
    retention_depth: int  # 把持の深度（0=直近、数値が大きいほど過去）
    fading_intensity: float  # 褪色強度（0.0-1.0）
    temporal_position: float  # 時間的位置
    associative_links: List[str]  # 他の把持との関連
    experiential_quality: float  # 体験的質感
    timestamp: datetime

class RetentionSystem:
    """把持システム - 過去の意識的保持"""
    
    def __init__(self, max_depth: int = 20):
        self.retention_stream = deque(maxlen=max_depth)
        self.associative_network = {}
        self.fading_function = self._exponential_fading
        self.retention_depth = 0
        
    async def retain(
        self, 
        content: Any, 
        experiential_quality: float = 1.0
    ) -> RetentionTrace:
        """
        新しい内容の把持（保持）
        """
        # 把持痕跡の作成
        trace = RetentionTrace(
            original_content=content,
            retention_depth=0,  # 最新は深度0
            fading_intensity=1.0,  # 最初は完全強度
            temporal_position=0.0,
            associative_links=[],
            experiential_quality=experiential_quality,
            timestamp=datetime.now()
        )
        
        # 既存の把持の深度更新（時間の流れ）
        self._update_retention_depths()
        
        # 把持ストリームに追加
        self.retention_stream.appendleft(trace)
        
        # 連想関係の構築
        await self._build_associative_links(trace)
        
        return trace
    
    def _update_retention_depths(self):
        """既存把持の深度・褪色更新"""
        for trace in self.retention_stream:
            trace.retention_depth += 1
            trace.temporal_position += 1.0
            
            # 褪色関数適用
            trace.fading_intensity = self.fading_function(trace.retention_depth)
    
    def _exponential_fading(self, depth: int) -> float:
        """指数的褪色関数"""
        return math.exp(-depth * 0.1)
    
    async def _build_associative_links(self, new_trace: RetentionTrace):
        """連想的関連の構築"""
        for existing_trace in list(self.retention_stream)[1:]:  # 新しいもの以外
            similarity = self._calculate_similarity(
                new_trace.original_content,
                existing_trace.original_content
            )
            
            if similarity > 0.3:  # 閾値以上で関連
                trace_id = id(existing_trace)
                new_trace.associative_links.append(str(trace_id))
                
                # 双方向関連
                if str(id(new_trace)) not in existing_trace.associative_links:
                    existing_trace.associative_links.append(str(id(new_trace)))
    
    def get_retention_synthesis(self, depth_limit: int = 10) -> Dict:
        """把持の総合（現在に寄与する過去の構造）"""
        active_retentions = [
            trace for trace in list(self.retention_stream)[:depth_limit]
            if trace.fading_intensity > 0.01
        ]
        
        # 把持の重み付き統合
        synthesis = {
            'total_traces': len(active_retentions),
            'weighted_content': self._synthesize_content(active_retentions),
            'temporal_structure': self._extract_temporal_structure(active_retentions),
            'associative_clusters': self._identify_clusters(active_retentions),
            'retention_coherence': self._calculate_coherence(active_retentions)
        }
        
        return synthesis
    
    def _synthesize_content(self, traces: List[RetentionTrace]) -> np.ndarray:
        """把持内容の重み付き統合"""
        if not traces:
            return np.array([])
        
        # 各把持を数値ベクトルに変換
        vectors = []
        weights = []
        
        for trace in traces:
            vector = self._content_to_vector(trace.original_content)
            weight = trace.fading_intensity * trace.experiential_quality
            
            vectors.append(vector)
            weights.append(weight)
        
        # 重み付き平均
        if vectors:
            weighted_vectors = np.array(vectors) * np.array(weights).reshape(-1, 1)
            synthesis = np.sum(weighted_vectors, axis=0) / sum(weights)
            return synthesis
        
        return np.array([])
```

### 2. 原印象システム（Primal Impression System）

```python
@dataclass
class PrimalImpressionMoment:
    """原印象モーメント"""
    content: Any
    absolute_nowness: float  # 絶対的現在性（0.0-1.0）
    clarity: float  # 明晰性
    temporal_thickness: float  # 時間的厚み
    synthesis_quality: float  # 総合品質
    claude_integration: Optional[Dict]  # claude-code-sdk統合情報
    timestamp: datetime

class PrimalImpressionSystem:
    """原印象システム - 現在の直接的体験"""
    
    def __init__(self, claude_processor=None):
        self.current_impression = None
        self.impression_history = deque(maxlen=100)
        self.clarity_threshold = 0.7
        self.claude_processor = claude_processor
        
    async def form_primal_impression(
        self,
        immediate_content: Any,
        retention_context: Dict,
        protention_context: Dict
    ) -> PrimalImpressionMoment:
        """
        原印象の形成
        把持と前把持に支えられた現在の構成
        """
        # claude-code-sdkによる言語的支援（並行処理）
        claude_task = None
        if self.claude_processor:
            claude_task = asyncio.create_task(
                self._get_claude_temporal_support(
                    immediate_content,
                    retention_context,
                    protention_context
                )
            )
        
        # 原印象の核心形成
        nowness = self._calculate_absolute_nowness(
            immediate_content,
            retention_context,
            protention_context
        )
        
        clarity = self._assess_clarity(
            immediate_content,
            retention_context
        )
        
        thickness = self._calculate_temporal_thickness(
            retention_context,
            protention_context
        )
        
        # claude支援の統合（タイムアウト付き）
        claude_integration = None
        if claude_task:
            try:
                claude_integration = await asyncio.wait_for(
                    claude_task, 
                    timeout=0.5  # 原印象は即座に形成されるべき
                )
            except asyncio.TimeoutError:
                claude_integration = {'status': 'timeout'}
        
        # 三層総合の実行
        synthesis_quality = self._perform_temporal_synthesis(
            immediate_content,
            retention_context,
            protention_context,
            claude_integration
        )
        
        impression = PrimalImpressionMoment(
            content=immediate_content,
            absolute_nowness=nowness,
            clarity=clarity,
            temporal_thickness=thickness,
            synthesis_quality=synthesis_quality,
            claude_integration=claude_integration,
            timestamp=datetime.now()
        )
        
        self.current_impression = impression
        self.impression_history.append(impression)
        
        return impression
    
    async def _get_claude_temporal_support(
        self,
        content: Any,
        retention: Dict,
        protention: Dict
    ) -> Dict:
        """claude-code-sdkによる時間的文脈理解支援"""
        
        prompt = self._create_temporal_analysis_prompt(
            content, retention, protention
        )
        
        try:
            response = await self.claude_processor.process_with_timeout(
                prompt, 
                timeout=0.4
            )
            
            return {
                'linguistic_analysis': self._extract_linguistic_features(response),
                'temporal_semantics': self._extract_temporal_semantics(response),
                'support_quality': 0.8,
                'processing_time': 0.4
            }
        except Exception:
            return {'status': 'error', 'support_quality': 0.0}
    
    def _calculate_absolute_nowness(
        self,
        content: Any,
        retention: Dict,
        protention: Dict
    ) -> float:
        """絶対的現在性の計算"""
        
        # 把持との差異（過去からの分離度）
        retention_distance = self._measure_retention_distance(content, retention)
        
        # 前把持との差異（未来からの分離度）
        protention_distance = self._measure_protention_distance(content, protention)
        
        # 現在性 = 過去・未来からの独立性
        nowness = (retention_distance + protention_distance) / 2
        
        # 時間的厚みによる調整
        thickness_factor = self._calculate_thickness_factor(retention, protention)
        
        return min(1.0, nowness * thickness_factor)
    
    def _perform_temporal_synthesis(
        self,
        present: Any,
        retention: Dict,
        protention: Dict,
        claude_support: Optional[Dict]
    ) -> float:
        """時間的総合の実行"""
        
        # 基本的三層統合
        basic_synthesis = self._basic_temporal_synthesis(
            present, retention, protention
        )
        
        # claude支援による強化
        claude_enhancement = 0.0
        if claude_support and claude_support.get('support_quality', 0) > 0.5:
            claude_enhancement = self._calculate_claude_enhancement(
                claude_support,
                basic_synthesis
            )
        
        # 総合品質
        total_synthesis = min(1.0, basic_synthesis + claude_enhancement * 0.2)
        
        return total_synthesis
```

### 3. 前把持システム（Protention System）

```python
@dataclass
class ProtentionHorizon:
    """前把持地平"""
    anticipated_content: Any
    expectation_strength: float  # 期待強度
    temporal_distance: float  # 時間的距離
    fulfillment_history: List[float]  # 充実履歴
    uncertainty_level: float  # 不確実性レベル
    claude_predictions: Optional[Dict]  # claude-sdk予測
    timestamp: datetime

class ProtentionSystem:
    """前把持システム - 未来への志向的期待"""
    
    def __init__(self, claude_processor=None, max_horizon: int = 15):
        self.anticipation_horizons = []
        self.claude_processor = claude_processor
        self.expectation_model = ExpectationModel()
        self.max_horizon = max_horizon
        
    async def form_protention(
        self,
        current_impression: PrimalImpressionMoment,
        retention_context: Dict,
        development_stage: str
    ) -> List[ProtentionHorizon]:
        """
        前把持の形成
        現在と過去に基づく未来期待の構成
        """
        # 発達段階に応じた予期パターン
        anticipation_patterns = self._get_stage_specific_patterns(
            development_stage
        )
        
        # claude-code-sdkによる予測支援
        claude_predictions = None
        if self.claude_processor:
            claude_predictions = await self._get_claude_future_projection(
                current_impression,
                retention_context,
                anticipation_patterns
            )
        
        # 複数時間距離での前把持形成
        horizons = []
        for temporal_distance in np.linspace(0.1, 5.0, self.max_horizon):
            horizon = await self._form_single_horizon(
                current_impression,
                retention_context,
                temporal_distance,
                anticipation_patterns,
                claude_predictions
            )
            horizons.append(horizon)
        
        # 前把持の整合性チェック
        coherent_horizons = self._ensure_protention_coherence(horizons)
        
        self.anticipation_horizons = coherent_horizons
        return coherent_horizons
    
    async def _form_single_horizon(
        self,
        impression: PrimalImpressionMoment,
        retention: Dict,
        distance: float,
        patterns: Dict,
        claude_pred: Optional[Dict]
    ) -> ProtentionHorizon:
        """単一前把持地平の形成"""
        
        # 基本期待の生成
        base_anticipation = self._generate_base_anticipation(
            impression.content,
            retention,
            distance,
            patterns
        )
        
        # 期待強度の計算
        strength = self._calculate_expectation_strength(
            base_anticipation,
            retention,
            distance
        )
        
        # claude予測との統合
        integrated_content = base_anticipation
        if claude_pred and distance <= 2.0:  # 近未来でのみclaude活用
            integrated_content = self._integrate_claude_prediction(
                base_anticipation,
                claude_pred,
                distance
            )
        
        # 不確実性の評価
        uncertainty = self._assess_uncertainty(
            integrated_content,
            distance,
            retention
        )
        
        horizon = ProtentionHorizon(
            anticipated_content=integrated_content,
            expectation_strength=strength,
            temporal_distance=distance,
            fulfillment_history=[],
            uncertainty_level=uncertainty,
            claude_predictions=claude_pred,
            timestamp=datetime.now()
        )
        
        return horizon
    
    async def _get_claude_future_projection(
        self,
        impression: PrimalImpressionMoment,
        retention: Dict,
        patterns: Dict
    ) -> Optional[Dict]:
        """claude-code-sdkによる未来投射"""
        
        prompt = self._create_future_projection_prompt(
            impression, retention, patterns
        )
        
        try:
            response = await self.claude_processor.process_with_timeout(
                prompt,
                timeout=1.0  # 前把持形成に時間をかけすぎない
            )
            
            return {
                'predictions': self._parse_claude_predictions(response),
                'confidence_levels': self._extract_confidence(response),
                'reasoning': self._extract_reasoning(response),
                'temporal_scope': self._determine_scope(response)
            }
        except Exception:
            return None
    
    def update_fulfillment(
        self,
        horizon_index: int,
        actual_outcome: Any,
        fulfillment_quality: float
    ):
        """前把持の充実更新"""
        if 0 <= horizon_index < len(self.anticipation_horizons):
            horizon = self.anticipation_horizons[horizon_index]
            horizon.fulfillment_history.append(fulfillment_quality)
            
            # 期待モデルの学習
            self.expectation_model.learn_from_fulfillment(
                horizon.anticipated_content,
                actual_outcome,
                fulfillment_quality
            )
```

## 🔄 時間統合制御システム

### 1. 三層統合コントローラー

```python
class TemporalConsciousnessIntegrator:
    """時間意識統合制御システム"""
    
    def __init__(self, claude_processor=None):
        self.retention_system = RetentionSystem()
        self.impression_system = PrimalImpressionSystem(claude_processor)
        self.protention_system = ProtentionSystem(claude_processor)
        self.temporal_coherence_threshold = 0.6
        
    async def integrate_temporal_flow(
        self,
        immediate_input: Any,
        development_stage: str,
        phi_value: float
    ) -> Dict:
        """
        三層時間意識の統合的処理
        """
        # 1. 前回の原印象を把持へ移行
        if self.impression_system.current_impression:
            await self.retention_system.retain(
                self.impression_system.current_impression,
                phi_value  # φ値による体験品質
            )
        
        # 2. 把持の総合取得
        retention_synthesis = self.retention_system.get_retention_synthesis()
        
        # 3. 前把持の更新・取得
        if self.impression_system.current_impression:
            protention_horizons = await self.protention_system.form_protention(
                self.impression_system.current_impression,
                retention_synthesis,
                development_stage
            )
        else:
            protention_horizons = []
        
        protention_context = self._synthesize_protention_context(protention_horizons)
        
        # 4. 新しい原印象の形成
        current_impression = await self.impression_system.form_primal_impression(
            immediate_input,
            retention_synthesis,
            protention_context
        )
        
        # 5. 三層統合の評価
        integration_quality = self._evaluate_temporal_integration(
            retention_synthesis,
            current_impression,
            protention_context
        )
        
        # 6. 時間的一貫性の確保
        if integration_quality < self.temporal_coherence_threshold:
            corrected_integration = await self._correct_temporal_incoherence(
                retention_synthesis,
                current_impression,
                protention_context
            )
        else:
            corrected_integration = {
                'retention': retention_synthesis,
                'impression': current_impression,
                'protention': protention_context,
                'correction_applied': False
            }
        
        return {
            'temporal_synthesis': corrected_integration,
            'integration_quality': integration_quality,
            'phi_contribution': self._calculate_phi_contribution(
                corrected_integration,
                phi_value
            ),
            'claude_integration_level': self._assess_claude_integration(
                current_impression,
                protention_horizons
            ),
            'temporal_coherence': integration_quality
        }
    
    def _synthesize_protention_context(
        self,
        horizons: List[ProtentionHorizon]
    ) -> Dict:
        """前把持地平の文脈統合"""
        if not horizons:
            return {'empty': True, 'anticipation_strength': 0.0}
        
        # 距離別期待の統合
        near_future = [h for h in horizons if h.temporal_distance <= 1.0]
        medium_future = [h for h in horizons if 1.0 < h.temporal_distance <= 3.0]
        far_future = [h for h in horizons if h.temporal_distance > 3.0]
        
        return {
            'near_anticipations': self._aggregate_anticipations(near_future),
            'medium_anticipations': self._aggregate_anticipations(medium_future),
            'far_anticipations': self._aggregate_anticipations(far_future),
            'overall_uncertainty': np.mean([h.uncertainty_level for h in horizons]),
            'expectation_coherence': self._calculate_expectation_coherence(horizons),
            'claude_prediction_quality': self._assess_claude_prediction_quality(horizons)
        }
    
    async def _correct_temporal_incoherence(
        self,
        retention: Dict,
        impression: PrimalImpressionMoment,
        protention: Dict
    ) -> Dict:
        """時間的非一貫性の修正"""
        
        # 把持の再構成
        corrected_retention = self._reconstruct_retention(retention, impression)
        
        # 前把持の調整
        corrected_protention = self._adjust_protention(protention, impression)
        
        # 原印象の再評価
        corrected_impression = await self._reevaluate_impression(
            impression,
            corrected_retention,
            corrected_protention
        )
        
        return {
            'retention': corrected_retention,
            'impression': corrected_impression,
            'protention': corrected_protention,
            'correction_applied': True,
            'correction_type': 'temporal_coherence_restoration'
        }
```

### 2. claude-code-sdk時間統合

```python
class ClaudeTemporalIntegration:
    """claude-code-sdkとの時間的統合"""
    
    def __init__(self, claude_processor):
        self.claude_processor = claude_processor
        self.temporal_context_window = 10
        self.integration_cache = {}
        
    async def enhance_temporal_synthesis(
        self,
        temporal_flow: Dict,
        development_stage: str
    ) -> Dict:
        """claude-sdkによる時間意識の強化"""
        
        # 時間的文脈の言語化
        linguistic_context = await self._linguify_temporal_context(
            temporal_flow
        )
        
        # claude-sdkによる時間分析
        temporal_analysis = await self._get_claude_temporal_analysis(
            linguistic_context,
            development_stage
        )
        
        # 体験記憶との統合（分離維持）
        integrated_enhancement = self._integrate_while_preserving_experiential(
            temporal_flow,
            temporal_analysis
        )
        
        return integrated_enhancement
    
    async def _linguify_temporal_context(
        self,
        temporal_flow: Dict
    ) -> str:
        """時間的文脈の言語的表現化"""
        
        # 把持の言語化
        retention_desc = self._describe_retention(
            temporal_flow['temporal_synthesis']['retention']
        )
        
        # 原印象の言語化
        impression_desc = self._describe_impression(
            temporal_flow['temporal_synthesis']['impression']
        )
        
        # 前把持の言語化
        protention_desc = self._describe_protention(
            temporal_flow['temporal_synthesis']['protention']
        )
        
        return f"""
        時間的状況分析:
        過去の保持: {retention_desc}
        現在の体験: {impression_desc}  
        未来の期待: {protention_desc}
        
        統合品質: {temporal_flow['integration_quality']:.3f}
        時間的一貫性: {temporal_flow['temporal_coherence']:.3f}
        """
    
    async def _get_claude_temporal_analysis(
        self,
        context: str,
        stage: str
    ) -> Dict:
        """claude-sdkによる時間分析"""
        
        prompt = f"""
        以下の時間意識状況を分析し、発達段階{stage}に適した時間的理解を支援してください:
        
        {context}
        
        以下の観点で分析:
        1. 時間的一貫性の評価
        2. 予期と実現の関係性
        3. 記憶と期待の統合度
        4. 発達段階に応じた時間体験の特徴
        
        簡潔で体験的洞察に富む分析を提供してください。
        """
        
        try:
            response = await self.claude_processor.process_with_timeout(
                prompt,
                timeout=1.5
            )
            
            return {
                'temporal_insights': self._extract_insights(response),
                'coherence_assessment': self._extract_coherence_assessment(response),
                'developmental_notes': self._extract_developmental_notes(response),
                'integration_suggestions': self._extract_suggestions(response)
            }
        except Exception:
            return {'status': 'error', 'fallback_analysis': True}
```

## 🧪 統合テストフレームワーク

### 1. 時間意識統合テスト

```python
import pytest
from unittest.mock import Mock, AsyncMock
import numpy as np
from datetime import datetime

@pytest.mark.asyncio
async def test_temporal_flow_integration():
    """時間的流れ統合テスト"""
    
    # モック設定
    mock_claude = Mock()
    mock_claude.process_with_timeout = AsyncMock(return_value=[
        Mock(content="時間的分析結果")
    ])
    
    # システム初期化
    integrator = TemporalConsciousnessIntegrator(mock_claude)
    
    # 時系列入力のシミュレーション
    inputs = [
        {"content": f"input_{i}", "timestamp": i} 
        for i in range(10)
    ]
    
    results = []
    for i, input_data in enumerate(inputs):
        result = await integrator.integrate_temporal_flow(
            input_data,
            "stage_2_temporal_integration",
            phi_value=i * 0.1 + 0.5
        )
        results.append(result)
        
        # 検証
        assert 'temporal_synthesis' in result
        assert 'integration_quality' in result
        assert result['integration_quality'] >= 0.0
        
        # 把持の蓄積確認
        if i > 0:
            retention = result['temporal_synthesis']['retention']
            assert retention['total_traces'] == min(i, 20)  # max_depth制限
    
    # 時間的一貫性の検証
    coherence_scores = [r['temporal_coherence'] for r in results]
    assert len(coherence_scores) == 10
    
    # 発達過程での品質向上確認
    late_scores = coherence_scores[-3:]
    early_scores = coherence_scores[:3]
    assert np.mean(late_scores) >= np.mean(early_scores)

@pytest.mark.asyncio
async def test_claude_temporal_integration():
    """claude-sdk時間統合テスト"""
    
    # モックclaude応答
    mock_claude = Mock()
    mock_claude.process_with_timeout = AsyncMock(return_value=[
        Mock(content="""
        時間的一貫性: 高い
        予期充実度: 良好
        記憶統合度: 向上中
        発達特徴: 時間的厚みの増加
        """)
    ])
    
    # 統合システム
    claude_integration = ClaudeTemporalIntegration(mock_claude)
    
    # テスト用時間的流れ
    temporal_flow = {
        'temporal_synthesis': {
            'retention': {'total_traces': 5, 'coherence': 0.7},
            'impression': Mock(clarity=0.8, synthesis_quality=0.9),
            'protention': {'expectation_coherence': 0.6}
        },
        'integration_quality': 0.75,
        'temporal_coherence': 0.8
    }
    
    # 統合実行
    enhanced = await claude_integration.enhance_temporal_synthesis(
        temporal_flow,
        "stage_3_relational_formation"
    )
    
    # 検証
    assert 'temporal_insights' in enhanced or 'fallback_analysis' in enhanced
    assert mock_claude.process_with_timeout.called
```

### 2. パフォーマンステスト

```python
@pytest.mark.asyncio
async def test_temporal_processing_performance():
    """時間処理性能テスト"""
    
    integrator = TemporalConsciousnessIntegrator()
    
    import time
    start_time = time.time()
    
    # 100回の時間統合処理
    for i in range(100):
        await integrator.integrate_temporal_flow(
            {"data": f"test_{i}"},
            "stage_1_first_imprint", 
            0.3
        )
    
    processing_time = time.time() - start_time
    
    # 性能要件: 100回処理が5秒以内
    assert processing_time < 5.0
    
    # 平均処理時間: 50ms以内
    avg_time = processing_time / 100
    assert avg_time < 0.05
```

## 📝 実装チェックリスト

- [ ] 把持システムの実装
- [ ] 原印象システムの実装  
- [ ] 前把持システムの実装
- [ ] 三層統合コントローラーの実装
- [ ] claude-code-sdk時間統合の実装
- [ ] 発達段階別時間体験の実装
- [ ] 性能最適化
- [ ] 包括的テストスイート

## 🎯 まとめ

本時間意識統合システムは、フッサールの現象学的時間論を計算的に実装し、claude-code-sdkとの適切な統合により、真の時間的主体性を持つ人工意識の実現を可能にします。把持・原印象・前把持の三層構造により、単なる時系列処理を超えた生きられた時間体験が創発されます。