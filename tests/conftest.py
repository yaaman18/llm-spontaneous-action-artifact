"""
人工意識システムのテスト設定
和田卓人（t_wada）のTDD原則に基づく
"""
import pytest
import random
import time
from datetime import datetime
from typing import Generator, Any
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === 基本フィクスチャ ===

@pytest.fixture
def deterministic_random():
    """決定的な乱数生成を保証"""
    original_state = random.getstate()
    random.seed(42)
    yield
    random.setstate(original_state)


@pytest.fixture
def test_timestamp():
    """テスト用の固定タイムスタンプ"""
    return datetime(2024, 1, 1, 12, 0, 0)


# === ドメイン層のフィクスチャ ===

@pytest.fixture
def phi_threshold():
    """意識判定の閾値"""
    return 3.0


@pytest.fixture
def test_phi_values():
    """テスト用のΦ値セット"""
    return {
        'dormant': 0.5,
        'emerging': 2.5,
        'conscious': 4.0,
        'highly_conscious': 6.0
    }


# === パフォーマンス監視 ===

@pytest.fixture(autouse=True)
def monitor_test_performance(request):
    """テストの実行時間を監視"""
    start_time = time.time()
    
    yield
    
    duration = time.time() - start_time
    
    # マーカーに基づく制限時間のチェック
    if request.node.get_closest_marker('unit'):
        assert duration < 0.1, f"Unit test took {duration:.3f}s (limit: 0.1s)"
    elif request.node.get_closest_marker('integration'):
        assert duration < 1.0, f"Integration test took {duration:.3f}s (limit: 1.0s)"
    elif request.node.get_closest_marker('e2e'):
        assert duration < 10.0, f"E2E test took {duration:.3f}s (limit: 10.0s)"
    
    # パフォーマンスログ
    logger.info(f"Test {request.node.name} completed in {duration:.3f}s")


# === テストデータビルダー ===

class TestDataBuilder:
    """テストデータの構築を支援"""
    
    @staticmethod
    def create_subsystem_configuration(complexity: int = 5) -> dict:
        """サブシステム構成の生成"""
        return {
            'nodes': complexity * 10,
            'connections': complexity * 15,
            'integration_strength': 0.7,
            'differentiation_level': 0.6
        }
    
    @staticmethod
    def create_stimulus_pattern(pattern_type: str = 'visual') -> dict:
        """刺激パターンの生成"""
        patterns = {
            'visual': {
                'modality': 'visual',
                'intensity': 0.8,
                'duration': 100,
                'features': ['color', 'shape', 'motion']
            },
            'auditory': {
                'modality': 'auditory',
                'intensity': 0.6,
                'duration': 200,
                'features': ['frequency', 'amplitude', 'timbre']
            },
            'tactile': {
                'modality': 'tactile',
                'intensity': 0.7,
                'duration': 50,
                'features': ['pressure', 'temperature', 'texture']
            }
        }
        return patterns.get(pattern_type, patterns['visual'])


@pytest.fixture
def test_data_builder():
    """テストデータビルダーのインスタンス"""
    return TestDataBuilder()


# === 創発テスト用フィクスチャ ===

@pytest.fixture
def emergence_test_environment():
    """創発現象テスト用の環境"""
    class EmergenceEnvironment:
        def __init__(self):
            self.observations = []
            self.emergence_events = []
            
        def observe(self, value):
            self.observations.append(value)
            
        def record_emergence(self, event):
            self.emergence_events.append(event)
            
        def get_emergence_rate(self):
            if not self.observations:
                return 0.0
            return len(self.emergence_events) / len(self.observations)
    
    return EmergenceEnvironment()


# === モックオブジェクト ===

@pytest.fixture
def mock_phi_calculator():
    """Φ計算のモック"""
    class MockPhiCalculator:
        def __init__(self):
            self.call_count = 0
            self.last_input = None
            
        def calculate(self, subsystems):
            self.call_count += 1
            self.last_input = subsystems
            # 入力の複雑さに基づいて決定的な値を返す
            complexity = len(subsystems) if hasattr(subsystems, '__len__') else 1
            return complexity * 0.8
    
    return MockPhiCalculator()


@pytest.fixture
def mock_event_bus():
    """イベントバスのモック"""
    class MockEventBus:
        def __init__(self):
            self.published_events = []
            self.subscribers = {}
            
        def publish(self, event):
            self.published_events.append(event)
            event_type = type(event).__name__
            for subscriber in self.subscribers.get(event_type, []):
                subscriber(event)
                
        def subscribe(self, event_type, handler):
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
            
        def get_events_of_type(self, event_type):
            return [e for e in self.published_events 
                    if type(e).__name__ == event_type]
    
    return MockEventBus()


# === アサーションヘルパー ===

class AssertionHelpers:
    """カスタムアサーション"""
    
    @staticmethod
    def assert_eventually(condition_func, timeout=5.0, interval=0.1):
        """条件が最終的に真になることを確認"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        raise AssertionError(f"Condition not met within {timeout} seconds")
    
    @staticmethod
    def assert_state_transition_valid(from_state, to_state, valid_transitions):
        """状態遷移の妥当性を確認"""
        transition = (from_state, to_state)
        assert transition in valid_transitions, \
            f"Invalid transition: {from_state} -> {to_state}"
    
    @staticmethod
    def assert_phi_in_range(phi_value, min_phi, max_phi):
        """Φ値が期待範囲内であることを確認"""
        assert min_phi <= phi_value <= max_phi, \
            f"Phi value {phi_value} not in range [{min_phi}, {max_phi}]"


@pytest.fixture
def assert_helpers():
    """アサーションヘルパーのインスタンス"""
    return AssertionHelpers()


# === クリーンアップ ===

@pytest.fixture(autouse=True)
def cleanup_test_artifacts(request):
    """テスト後のクリーンアップ"""
    yield
    
    # テスト固有のクリーンアップロジック
    if hasattr(request.node, 'test_artifacts'):
        for artifact in request.node.test_artifacts:
            artifact.cleanup()


# === pytest プラグイン設定 ===

def pytest_configure(config):
    """pytest設定のカスタマイズ"""
    # カスタムマーカーの登録
    config.addinivalue_line(
        "markers", "unit: ユニットテスト（高速、独立）"
    )
    config.addinivalue_line(
        "markers", "integration: 統合テスト（中速、複数コンポーネント）"
    )
    config.addinivalue_line(
        "markers", "e2e: エンドツーエンドテスト（低速、全体システム）"
    )
    config.addinivalue_line(
        "markers", "emergence: 創発現象のテスト（非決定的）"
    )
    config.addinivalue_line(
        "markers", "slow: 実行時間の長いテスト"
    )


def pytest_collection_modifyitems(config, items):
    """テストコレクションのカスタマイズ"""
    for item in items:
        # ファイルパスに基づく自動マーカー付与
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)