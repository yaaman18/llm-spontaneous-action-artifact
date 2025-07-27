"""
動的Φ境界検出システムのユニットテスト
適応的な意識判定閾値の実装
"""
import pytest
from typing import List, Callable
from collections import deque


class TestDynamicPhiBoundaryDetector:
    """動的境界検出のTDD実装"""
    
    @pytest.mark.unit
    def test_boundary_detector_initialization(self):
        """境界検出器の初期化テスト"""
        # When: 境界検出器を初期化
        from domain.services import DynamicPhiBoundaryDetector
        detector = DynamicPhiBoundaryDetector()
        
        # Then: デフォルト値が設定される
        assert detector.current_threshold == 3.0
        assert detector.observation_window_size == 100
        assert len(detector.observations) == 0
        assert detector.adaptation_rate == 0.1
    
    @pytest.mark.unit
    def test_custom_initialization_parameters(self):
        """カスタムパラメータでの初期化"""
        from domain.services import DynamicPhiBoundaryDetector
        
        # Given: カスタム設定
        custom_threshold = 4.0
        custom_window = 50
        custom_rate = 0.2
        
        # When: カスタムパラメータで初期化
        detector = DynamicPhiBoundaryDetector(
            initial_threshold=custom_threshold,
            window_size=custom_window,
            adaptation_rate=custom_rate
        )
        
        # Then: カスタム値が適用される
        assert detector.current_threshold == custom_threshold
        assert detector.observation_window_size == custom_window
        assert detector.adaptation_rate == custom_rate
    
    @pytest.mark.unit
    def test_phi_observation_recording(self):
        """Φ値観測の記録テスト"""
        from domain.services import DynamicPhiBoundaryDetector
        from domain.value_objects import PhiValue
        
        # Given: 検出器と観測値
        detector = DynamicPhiBoundaryDetector(window_size=5)
        phi_values = [PhiValue(i) for i in [2.0, 3.0, 4.0, 5.0, 6.0]]
        
        # When: 値を観測
        for phi in phi_values:
            detector.observe(phi)
        
        # Then: すべての値が記録される
        assert len(detector.observations) == 5
        assert all(obs in detector.observations for obs in phi_values)
    
    @pytest.mark.unit
    def test_observation_window_sliding(self):
        """観測ウィンドウのスライディングテスト"""
        from domain.services import DynamicPhiBoundaryDetector
        from domain.value_objects import PhiValue
        
        # Given: ウィンドウサイズ3の検出器
        detector = DynamicPhiBoundaryDetector(window_size=3)
        
        # When: 5つの値を観測
        for i in range(5):
            detector.observe(PhiValue(i))
        
        # Then: 最新の3つのみ保持
        assert len(detector.observations) == 3
        assert detector.observations[0].value == 2
        assert detector.observations[-1].value == 4
    
    @pytest.mark.unit
    def test_adaptive_threshold_increase(self):
        """閾値の上方適応テスト"""
        from domain.services import DynamicPhiBoundaryDetector
        from domain.value_objects import PhiValue
        
        # Given: 初期閾値3.0の検出器
        detector = DynamicPhiBoundaryDetector(
            initial_threshold=3.0,
            adaptation_rate=0.1
        )
        
        # When: 高いΦ値を連続観測
        high_values = [PhiValue(5.0), PhiValue(5.2), PhiValue(5.1)]
        for phi in high_values:
            detector.observe(phi)
        
        # Then: 閾値が上昇
        assert detector.current_threshold > 3.0
        assert detector.current_threshold < 5.0  # 急激な変化は避ける
    
    @pytest.mark.unit
    def test_adaptive_threshold_decrease(self):
        """閾値の下方適応テスト"""
        from domain.services import DynamicPhiBoundaryDetector
        from domain.value_objects import PhiValue
        
        # Given: 高い初期閾値の検出器
        detector = DynamicPhiBoundaryDetector(
            initial_threshold=5.0,
            adaptation_rate=0.1
        )
        
        # When: 低いΦ値を連続観測
        low_values = [PhiValue(2.0), PhiValue(2.2), PhiValue(2.1)]
        for phi in low_values:
            detector.observe(phi)
        
        # Then: 閾値が下降
        assert detector.current_threshold < 5.0
        assert detector.current_threshold > 2.0  # 最小閾値は維持
    
    @pytest.mark.unit
    def test_threshold_stability_with_mixed_values(self):
        """混在値での閾値安定性テスト"""
        from domain.services import DynamicPhiBoundaryDetector
        from domain.value_objects import PhiValue
        
        # Given: 検出器
        detector = DynamicPhiBoundaryDetector(
            initial_threshold=3.0,
            adaptation_rate=0.05  # 低い適応率
        )
        initial = detector.current_threshold
        
        # When: 高低混在の値を観測
        mixed_values = [
            PhiValue(2.0), PhiValue(6.0), PhiValue(2.5),
            PhiValue(5.5), PhiValue(3.0), PhiValue(4.0)
        ]
        for phi in mixed_values:
            detector.observe(phi)
        
        # Then: 閾値は大きく変動しない
        assert abs(detector.current_threshold - initial) < 0.5
    
    @pytest.mark.unit
    def test_boundary_change_event_emission(self, mock_event_bus):
        """境界変更イベントの発火テスト"""
        from domain.services import DynamicPhiBoundaryDetector
        from domain.value_objects import PhiValue
        from domain.events import PhiBoundaryChanged
        
        # Given: イベントバスに接続された検出器
        detector = DynamicPhiBoundaryDetector(event_bus=mock_event_bus)
        
        # イベントリスナーを設定
        captured_events = []
        mock_event_bus.subscribe('PhiBoundaryChanged', captured_events.append)
        
        # When: 大幅な変化を引き起こす観測
        for i in range(10):
            detector.observe(PhiValue(6.0))  # 高い値を連続観測
        
        # Then: 境界変更イベントが発火
        assert len(captured_events) > 0
        assert all(isinstance(e, PhiBoundaryChanged) for e in captured_events)
        assert captured_events[-1].new_threshold > 3.0
    
    @pytest.mark.unit
    def test_statistical_threshold_calculation(self):
        """統計的閾値計算のテスト"""
        from domain.services import DynamicPhiBoundaryDetector
        from domain.value_objects import PhiValue
        import numpy as np
        
        # Given: 正規分布に従うΦ値
        np.random.seed(42)
        values = np.random.normal(loc=4.0, scale=0.5, size=100)
        
        detector = DynamicPhiBoundaryDetector(
            window_size=100,
            use_statistical_method=True
        )
        
        # When: すべての値を観測
        for v in values:
            detector.observe(PhiValue(v))
        
        # Then: 閾値は平均値付近に収束
        assert 3.8 < detector.current_threshold < 4.2
        
        # 統計情報も利用可能
        stats = detector.get_statistics()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'percentile_75' in stats
    
    @pytest.mark.unit
    def test_minimum_threshold_enforcement(self):
        """最小閾値の強制テスト"""
        from domain.services import DynamicPhiBoundaryDetector
        from domain.value_objects import PhiValue
        
        # Given: 最小閾値を設定した検出器
        detector = DynamicPhiBoundaryDetector(
            initial_threshold=3.0,
            minimum_threshold=2.0
        )
        
        # When: 非常に低い値を観測
        for _ in range(20):
            detector.observe(PhiValue(0.1))
        
        # Then: 閾値は最小値を下回らない
        assert detector.current_threshold >= 2.0
    
    @pytest.mark.unit
    def test_rapid_adaptation_mode(self):
        """急速適応モードのテスト"""
        from domain.services import DynamicPhiBoundaryDetector
        from domain.value_objects import PhiValue
        
        # Given: 急速適応モードの検出器
        detector = DynamicPhiBoundaryDetector(
            adaptation_rate=0.3,
            rapid_adaptation_enabled=True
        )
        
        # When: 突然の大きな変化
        detector.observe(PhiValue(3.0))
        detector.observe(PhiValue(8.0))  # 突然の跳躍
        
        # Then: 閾値が素早く追従
        assert detector.current_threshold > 4.0
        
        # 急速適応の履歴も記録
        assert detector.rapid_adaptation_triggered
        assert len(detector.adaptation_history) > 0
    
    @pytest.mark.unit
    def test_consciousness_state_classification(self):
        """意識状態分類との統合テスト"""
        from domain.services import DynamicPhiBoundaryDetector
        from domain.value_objects import PhiValue
        
        # Given: 検出器
        detector = DynamicPhiBoundaryDetector()
        
        # When: 様々な値を観測
        test_cases = [
            (PhiValue(1.0), 'dormant'),
            (PhiValue(2.5), 'emerging'),
            (PhiValue(3.5), 'conscious'),
            (PhiValue(6.0), 'highly_conscious')
        ]
        
        # Then: 動的閾値に基づいて分類
        for phi, expected_base_state in test_cases:
            classification = detector.classify_consciousness(phi)
            assert classification.base_state == expected_base_state
            assert classification.threshold_used == detector.current_threshold
            assert classification.confidence > 0