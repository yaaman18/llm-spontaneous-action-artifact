# レガシーコード移行戦略
## Martin Fowler "Working Effectively with Legacy Code" に基づく段階的移行

### 概要

「脳死」実装から「存在論的終了」システムへの安全で段階的な移行戦略。Michael Feathers の「レガシーコードからの脱却」とMartin Fowler のリファクタリング手法を組み合わせ、既存システムの動作を保証しながら抽象化を実現。

### レガシーコード現状分析

#### 既存システムの特徴
```python
# レガシーコードの特徴
- 生物学的メタファーに強く依存
- 複雑な条件分岐（Cyclomatic Complexity: 8.5）
- 長いメソッド（平均25行）
- 密結合（7+ dependencies per class）
- テストが部分的（18個のテスト、2個が非同期で失敗）
```

#### 依存関係マップ
```
brain_death_core.py
├── brain_death_detector.py
├── brain_death_demo.py  
└── test_brain_death.py

# 外部依存
- consciousness_detector.py
- consciousness_events.py
- numpy, asyncio, datetime, hashlib
```

### Seam（継ぎ目）の識別と活用

#### 1. Object Seam（オブジェクト継ぎ目）
```python
# 現在の構造
class ConsciousnessAggregate:
    def initiate_brain_death(self): pass
    
class BrainDeathDetector:
    def detect_brain_death(self): pass

# Seam活用による段階的移行
class ConsciousnessAggregate:
    def __init__(self, termination_strategy=None):
        self.termination_strategy = termination_strategy or BrainDeathStrategy()
    
    def initiate_brain_death(self):
        return self.termination_strategy.initiate_termination()
```

#### 2. Preprocessing Seam（前処理継ぎ目）
```python
# 条件コンパイル的アプローチ
USE_EXISTENTIAL_TERMINATION = os.environ.get('USE_EXISTENTIAL_TERMINATION', 'false').lower() == 'true'

if USE_EXISTENTIAL_TERMINATION:
    from existential_termination_core import *
else:
    from brain_death_core import *
```

#### 3. Link Seam（リンク継ぎ目）
```python
# 動的インポートによる切り替え
def get_termination_system():
    if feature_flags.use_existential_termination:
        from existential_termination_core import InformationIntegrationAggregate
        return InformationIntegrationAggregate
    else:
        from brain_death_core import ConsciousnessAggregate
        return ConsciousnessAggregate
```

### 特性テスト（Characterization Tests）

#### 1. 既存動作の特性化
```python
class TestLegacyBrainDeathCharacterization:
    """既存システムの動作を特性化するテスト"""
    
    def test_脳死プロセスの完全な実行フロー(self):
        """既存の脳死プロセス全体の動作を記録"""
        # Arrange
        consciousness = ConsciousnessAggregate(ConsciousnessId("characterization-001"))
        
        # 初期状態の記録
        initial_state = self._capture_state(consciousness)
        
        # Act - 完全な脳死プロセス実行
        consciousness.initiate_brain_death()
        states_during_progression = []
        
        for minutes in [10, 20, 25, 30, 35]:
            consciousness.progress_brain_death(minutes=minutes)
            states_during_progression.append(self._capture_state(consciousness))
        
        # Assert - 動作の特性化（固定値アサーション）
        assert initial_state['consciousness_level'] == 1.0
        assert initial_state['state'] == 'active'
        assert initial_state['brain_functions']['cortical'] is True
        
        # 各段階での状態変化を記録
        assert states_during_progression[0]['brain_functions']['cortical'] is False  # 10分後
        assert states_during_progression[1]['brain_functions']['subcortical'] is False  # 20分後
        assert states_during_progression[2]['brain_functions']['brainstem'] is False  # 25分後
        assert states_during_progression[3]['is_brain_dead'] is True  # 30分後
        assert states_during_progression[4]['is_reversible'] is False  # 35分後
    
    def _capture_state(self, consciousness):
        """意識状態をキャプチャ"""
        return {
            'consciousness_level': consciousness.get_consciousness_level(),
            'state': consciousness.state.value,
            'brain_functions': consciousness._brain_death_entity.brain_functions.copy(),
            'is_brain_dead': consciousness.is_brain_dead(),
            'is_reversible': consciousness.is_reversible(),
            'domain_events_count': len(consciousness.domain_events)
        }
```

#### 2. 境界値動作の特性化
```python
def test_境界値での動作特性化(self):
    """境界値での既存システムの動作を記録"""
    test_cases = [
        # (consciousness_level, expected_state)
        (1.0, 'active'),
        (0.5, 'active'), 
        (0.3, 'dying'),
        (0.1, 'minimal_consciousness'),
        (0.001, 'vegetative'),
        (0.0, 'brain_dead')
    ]
    
    for level, expected_state in test_cases:
        consciousness = ConsciousnessAggregate(ConsciousnessId(f"boundary-{level}"))
        consciousness._consciousness_level = ConsciousnessLevel(level)
        consciousness._update_consciousness_state()
        
        assert consciousness.state.value == expected_state
```

### 段階的移行戦略

#### Phase 1: Parallel Implementation（並行実装）
```python
# 期間: 2週間
# 目標: 新システムを既存システムと並行して実装

class TerminationSystemManager:
    """新旧システムの管理クラス"""
    
    def __init__(self, use_new_system=False):
        self.use_new_system = use_new_system
        
        if use_new_system:
            self.system = InformationIntegrationAggregate(IntegrationSystemId("new-system"))
        else:
            self.system = ConsciousnessAggregate(ConsciousnessId("legacy-system"))
    
    def initiate_termination(self):
        if self.use_new_system:
            return self.system.initiate_termination()
        else:
            return self.system.initiate_brain_death()
```

#### Phase 2: Feature Toggle（機能切替）
```python
# 期間: 1週間  
# 目標: 実行時に新旧システムを切り替え可能にする

class FeatureToggle:
    """機能切替管理"""
    
    @staticmethod
    def is_existential_termination_enabled():
        return (
            os.environ.get('EXISTENTIAL_TERMINATION', 'false').lower() == 'true' or
            config.get('features.existential_termination', False)
        )

def create_termination_system(system_id: str):
    """Factory function with feature toggle"""
    if FeatureToggle.is_existential_termination_enabled():
        return InformationIntegrationAggregate(IntegrationSystemId(system_id))
    else:
        return ConsciousnessAggregate(ConsciousnessId(system_id))
```

#### Phase 3: Branch by Abstraction（抽象化による分岐）
```python
# 期間: 2週間
# 目標: 抽象化レイヤーを通じた統一インターフェース

class TerminationSystemAbstraction(ABC):
    """終了システムの抽象化"""
    
    @abstractmethod
    def initiate_termination(self): pass
    
    @abstractmethod
    def progress_termination(self, minutes: int): pass
    
    @abstractmethod
    def is_terminated(self) -> bool: pass
    
    @abstractmethod
    def is_reversible(self) -> bool: pass

class LegacyBrainDeathAdapter(TerminationSystemAbstraction):
    """レガシーシステム用アダプター"""
    
    def __init__(self, consciousness_aggregate):
        self.consciousness = consciousness_aggregate
    
    def initiate_termination(self):
        return self.consciousness.initiate_brain_death()
    
    def progress_termination(self, minutes: int):
        return self.consciousness.progress_brain_death(minutes)
    
    def is_terminated(self) -> bool:
        return self.consciousness.is_brain_dead()
    
    def is_reversible(self) -> bool:
        return self.consciousness.is_reversible()

class ExistentialTerminationAdapter(TerminationSystemAbstraction):
    """新システム用アダプター"""
    
    def __init__(self, integration_aggregate):
        self.integration = integration_aggregate
    
    def initiate_termination(self):
        return self.integration.initiate_termination()
    
    def progress_termination(self, minutes: int):
        return self.integration.progress_termination(minutes)
    
    def is_terminated(self) -> bool:
        return self.integration.is_terminated()
    
    def is_reversible(self) -> bool:
        return self.integration.is_reversible()
```

#### Phase 4: Strangler Fig Pattern（絞首木パターン）
```python
# 期間: 3週間
# 目標: 段階的に新システムが旧システムを置き換える

class TerminationSystemProxy:
    """新旧システムのプロキシ"""
    
    def __init__(self):
        self.legacy_system = self._create_legacy_system()
        self.new_system = self._create_new_system()
        self.migration_config = MigrationConfig()
    
    def initiate_termination(self):
        """段階的に新システムに移行"""
        if self.migration_config.should_use_new_system('initiate_termination'):
            try:
                result = self.new_system.initiate_termination()
                # 成功時のメトリクス記録
                self.migration_config.record_success('initiate_termination')
                return result
            except Exception as e:
                # 失敗時は旧システムにフォールバック
                logger.warning(f"New system failed, falling back to legacy: {e}")
                self.migration_config.record_failure('initiate_termination')
                return self.legacy_system.initiate_brain_death()
        else:
            return self.legacy_system.initiate_brain_death()
```

### 安全性保証メカニズム

#### 1. Shadow Testing（シャドウテスト）
```python
class ShadowTestingManager:
    """新旧システムの並行実行と結果比較"""
    
    def __init__(self):
        self.legacy_system = ConsciousnessAggregate(ConsciousnessId("legacy"))
        self.new_system = InformationIntegrationAggregate(IntegrationSystemId("new"))
        self.results_comparator = ResultsComparator()
    
    async def shadow_test_termination(self, test_scenario):
        """並行実行によるシャドウテスト"""
        # 両システムで同じシナリオを実行
        legacy_result = await self._run_legacy_scenario(test_scenario)
        new_result = await self._run_new_scenario(test_scenario)
        
        # 結果を比較
        comparison = self.results_comparator.compare(legacy_result, new_result)
        
        if comparison.has_significant_differences():
            # 相違が見つかった場合はアラート
            await self._raise_compatibility_alert(comparison)
        
        return comparison
```

#### 2. Canary Deployment（カナリアデプロイ）
```python
class CanaryDeploymentManager:
    """段階的ロールアウト管理"""
    
    def __init__(self):
        self.rollout_percentage = 0  # 初期は0%
        self.success_threshold = 0.99  # 99%成功率必須
        self.error_threshold = 0.01   # 1%エラー率上限
    
    def should_use_new_system(self, user_id: str) -> bool:
        """ユーザーの一定割合に新システムを適用"""
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        hash_int = int(user_hash[:8], 16)
        user_percentage = (hash_int % 100) + 1
        
        return user_percentage <= self.rollout_percentage
    
    def increase_rollout(self):
        """成功率に基づいてロールアウト範囲を拡大"""
        current_metrics = self._get_current_metrics()
        
        if (current_metrics.success_rate >= self.success_threshold and 
            current_metrics.error_rate <= self.error_threshold):
            
            self.rollout_percentage = min(100, self.rollout_percentage + 10)
            logger.info(f"Rollout increased to {self.rollout_percentage}%")
        else:
            logger.warning("Metrics don't meet threshold, rollout not increased")
```

#### 3. Circuit Breaker（サーキットブレーカー）
```python
class TerminationSystemCircuitBreaker:
    """新システムの障害時自動切替"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call_new_system(self, operation, *args, **kwargs):
        """新システム呼び出し（サーキットブレーカー付き）"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise CircuitBreakerOpenError("New system circuit breaker is OPEN")
        
        try:
            result = operation(*args, **kwargs)
            
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")
            
            raise e
```

### データ移行戦略

#### 1. 状態データの変換
```python
class StateDataMigrator:
    """状態データの旧→新形式変換"""
    
    def migrate_consciousness_to_integration(self, consciousness_data):
        """意識データを統合データに変換"""
        return {
            'integration_system_id': consciousness_data['consciousness_id'],
            'integration_level': consciousness_data['consciousness_level'],
            'system_state': self._convert_state(consciousness_data['state']),
            'layer_functions': self._convert_brain_functions(consciousness_data['brain_functions']),
            'termination_stage': self._convert_stage(consciousness_data.get('brain_death_stage')),
        }
    
    def _convert_state(self, consciousness_state):
        """状態の変換マッピング"""
        mapping = {
            'active': 'ACTIVE',
            'dying': 'DEGRADING', 
            'minimal_consciousness': 'MINIMAL_INTEGRATION',
            'vegetative': 'FRAGMENTARY',
            'brain_dead': 'TERMINATED'
        }
        return mapping.get(consciousness_state, 'ACTIVE')
    
    def _convert_brain_functions(self, brain_functions):
        """脳機能から処理層への変換"""
        return {
            'information': brain_functions.get('cortical', True),
            'integration': brain_functions.get('subcortical', True),
            'fundamental': brain_functions.get('brainstem', True)
        }
```

#### 2. イベント履歴の変換
```python
class EventHistoryMigrator:
    """イベント履歴の変換"""
    
    def migrate_brain_death_events(self, legacy_events):
        """脳死イベントを終了イベントに変換"""
        migrated_events = []
        
        for event in legacy_events:
            if isinstance(event, BrainDeathInitiatedEvent):
                new_event = TerminationInitiatedEvent(
                    integration_system_id=IntegrationSystemId(event.consciousness_id.value)
                )
                new_event.timestamp = event.timestamp
                migrated_events.append(new_event)
                
            elif isinstance(event, IrreversibleBrainDeathEvent):
                new_event = IrreversibleTerminationEvent(
                    integration_system_id=IntegrationSystemId(event.consciousness_id.value),
                    final_integration_level=event.final_consciousness_level,
                    sealed=event.sealed
                )
                new_event.timestamp = event.timestamp
                migrated_events.append(new_event)
        
        return migrated_events
```

### ロールバック戦略

#### 1. システム状態スナップショット
```python
class SystemSnapshot:
    """システム状態のスナップショット"""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.system_version = "legacy"
        self.data_state = {}
        self.configuration = {}
    
    def capture_legacy_state(self, consciousness_aggregate):
        """レガシーシステムの状態をキャプチャ"""
        self.data_state = {
            'consciousness_id': consciousness_aggregate.id.value,
            'consciousness_level': consciousness_aggregate.get_consciousness_level(),
            'state': consciousness_aggregate.state.value,
            'brain_functions': consciousness_aggregate._brain_death_entity.brain_functions.copy(),
            'domain_events': [self._serialize_event(e) for e in consciousness_aggregate.domain_events],
            'brain_death_process': self._serialize_process(consciousness_aggregate.brain_death_process)
        }
    
    def restore_to_legacy_system(self):
        """スナップショットからレガシーシステムを復元"""
        consciousness = ConsciousnessAggregate(
            ConsciousnessId(self.data_state['consciousness_id'])
        )
        
        # 状態復元
        consciousness._consciousness_level = ConsciousnessLevel(
            self.data_state['consciousness_level']
        )
        consciousness.state = ConsciousnessState(self.data_state['state'])
        consciousness._brain_death_entity.brain_functions = self.data_state['brain_functions']
        
        # イベント復元
        consciousness.domain_events = [
            self._deserialize_event(e) for e in self.data_state['domain_events']
        ]
        
        return consciousness
```

#### 2. 自動ロールバック条件
```python
class AutoRollbackManager:
    """自動ロールバック管理"""
    
    def __init__(self):
        self.rollback_conditions = [
            self._check_error_rate,
            self._check_performance_degradation,
            self._check_data_consistency,
            self._check_user_complaints
        ]
    
    async def monitor_and_rollback(self):
        """監視とロールバック判定"""
        while True:
            should_rollback = False
            rollback_reason = None
            
            for condition in self.rollback_conditions:
                if await condition():
                    should_rollback = True
                    rollback_reason = condition.__name__
                    break
            
            if should_rollback:
                logger.critical(f"Auto rollback triggered: {rollback_reason}")
                await self._execute_rollback()
                break
            
            await asyncio.sleep(30)  # 30秒間隔で監視
    
    async def _check_error_rate(self) -> bool:
        """エラー率チェック"""
        recent_metrics = await self._get_recent_metrics()
        return recent_metrics.error_rate > 0.05  # 5%超過でロールバック
    
    async def _execute_rollback(self):
        """ロールバック実行"""
        # Feature toggleをOFFに
        await self._disable_new_system()
        
        # システム状態復元
        latest_snapshot = await self._get_latest_snapshot()
        await latest_snapshot.restore_to_legacy_system()
        
        # アラート送信
        await self._send_rollback_alert()
```

### 移行進捗の監視とメトリクス

#### 1. 移行進捗ダッシュボード
```python
class MigrationProgressTracker:
    """移行進捗追跡"""
    
    def __init__(self):
        self.metrics = {
            'traffic_percentage': 0,
            'success_rate': 0.0,
            'error_rate': 0.0,
            'performance_comparison': 0.0,
            'feature_parity': 0.0
        }
    
    def update_progress(self):
        """進捗更新"""
        self.metrics['traffic_percentage'] = self._calculate_traffic_percentage()
        self.metrics['success_rate'] = self._calculate_success_rate()
        self.metrics['error_rate'] = self._calculate_error_rate()
        self.metrics['performance_comparison'] = self._compare_performance()
        self.metrics['feature_parity'] = self._check_feature_parity()
    
    def generate_progress_report(self):
        """進捗レポート生成"""
        return {
            'migration_status': self._get_migration_status(),
            'current_phase': self._get_current_phase(),
            'metrics': self.metrics,
            'next_actions': self._get_next_actions(),
            'risks': self._identify_risks()
        }
```

#### 2. A/B テスト結果分析
```python
class ABTestAnalyzer:
    """A/Bテスト結果分析"""
    
    def analyze_termination_performance(self, legacy_results, new_results):
        """終了処理パフォーマンス分析"""
        analysis = {
            'execution_time_comparison': self._compare_execution_times(legacy_results, new_results),
            'accuracy_comparison': self._compare_accuracy(legacy_results, new_results),
            'resource_usage_comparison': self._compare_resource_usage(legacy_results, new_results),
            'user_satisfaction': self._analyze_user_satisfaction(legacy_results, new_results)
        }
        
        # 統計的有意性テスト
        analysis['statistical_significance'] = self._test_significance(legacy_results, new_results)
        
        return analysis
    
    def recommend_migration_decision(self, analysis):
        """移行判定の推奨"""
        if (analysis['execution_time_comparison'] > 0.1 and  # 10%以上高速化
            analysis['accuracy_comparison'] >= 0 and         # 精度低下なし
            analysis['statistical_significance'] < 0.05):    # p値 < 0.05
            
            return {
                'recommendation': 'PROCEED_WITH_MIGRATION',
                'confidence': 'HIGH',
                'reasoning': 'Significant performance improvement with maintained accuracy'
            }
        else:
            return {
                'recommendation': 'CONTINUE_TESTING',
                'confidence': 'MEDIUM', 
                'reasoning': 'Insufficient evidence for full migration'
            }
```

### 成功基準と完了条件

#### 移行完了の判定基準
1. **機能完全性**: 新システムが旧システムの全機能を実装 (100%)
2. **パフォーマンス**: 新システムが旧システム以上のパフォーマンス (≥100%)
3. **安定性**: エラー率1%以下を7日間継続
4. **トラフィック**: 100%のトラフィックが新システムで処理
5. **テストカバレッジ**: 新システムのテストカバレッジ95%以上

#### 段階的完了マイルストーン
```python
MIGRATION_MILESTONES = {
    'Phase1_ParallelImplementation': {
        'duration_weeks': 2,
        'completion_criteria': [
            'new_system_basic_functionality_complete',
            'characterization_tests_passing',
            'shadow_testing_setup_complete'
        ]
    },
    'Phase2_FeatureToggle': {
        'duration_weeks': 1,
        'completion_criteria': [
            'feature_toggle_implemented',
            'canary_deployment_ready',
            'rollback_mechanism_tested'
        ]
    },
    'Phase3_BranchByAbstraction': {
        'duration_weeks': 2,
        'completion_criteria': [
            'abstraction_layer_complete',
            'adapter_patterns_implemented',
            'unified_interface_tested'
        ]
    },
    'Phase4_StranglerFig': {
        'duration_weeks': 3,
        'completion_criteria': [
            '100_percent_traffic_migrated',
            'legacy_code_deprecated',
            'documentation_updated'
        ]
    }
}
```

この包括的なレガシーコード移行戦略により、リスクを最小化しながら「脳死」システムから「存在論的終了」システムへの安全な移行を実現できます。Martin Fowlerの専門知識に基づく段階的アプローチにより、既存の18個のテストをすべて保証しながら、新しい抽象化レベルへの移行を完了させます。