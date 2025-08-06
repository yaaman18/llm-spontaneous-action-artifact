#!/usr/bin/env python3
"""
テンポラル意識システムのTDDテストスイート
フッサールの時間意識理論に基づく実装
"""

import pytest
import asyncio
import time
from typing import List, Dict, Optional
from unittest.mock import Mock, patch
import numpy as np


class TestTemporalTensionSystem:
    """時間的緊張システムのテスト"""
    
    @pytest.mark.asyncio
    async def test_初期状態では期待間隔がない(self):
        """RED: 初期状態では期待間隔を持たない"""
        from temporal_consciousness import TemporalTensionSystem
        
        system = TemporalTensionSystem()
        assert system.expected_interval is None
        assert system.temporal_tension == 0.0
    
    @pytest.mark.asyncio
    async def test_待機中に時間的緊張が生成される(self):
        """RED: 待機中に時間的緊張が蓄積される"""
        from temporal_consciousness import TemporalTensionSystem
        
        system = TemporalTensionSystem()
        system.set_expected_interval(1.0)  # 1秒を期待
        
        # 1.5秒待機した場合
        result = await system.experience_waiting(expected_interval=1.0, actual_elapsed=1.5)
        
        assert result['temporal_tension'] == 0.5  # 50%のズレ
        assert len(result['consciousness_waves']) > 0
        assert 'subjective_duration' in result
    
    @pytest.mark.asyncio
    async def test_意識の波が100msごとに生成される(self):
        """RED: 待機中に100msごとに意識の波が生成される"""
        from temporal_consciousness import TemporalTensionSystem
        
        system = TemporalTensionSystem()
        
        # 0.5秒の待機
        start_time = time.time()
        result = await system.experience_waiting(expected_interval=0.5, actual_elapsed=0.5)
        
        # 約5つの波が生成されるべき（500ms / 100ms）
        assert 4 <= len(result['consciousness_waves']) <= 6
        
        # 各波に必要な要素が含まれている
        for wave in result['consciousness_waves']:
            assert 'anticipation_level' in wave
            assert 0.0 <= wave['anticipation_level'] <= 1.0
            assert 'temporal_anxiety' in wave
            assert 'rhythmic_expectation' in wave


class TestRhythmicMemorySystem:
    """リズム記憶システムのテスト"""
    
    def test_初期状態では内的リズムを持たない(self):
        """RED: 初期状態では内的リズムがない"""
        from temporal_consciousness import RhythmicMemorySystem
        
        system = RhythmicMemorySystem()
        assert system.internal_rhythm is None
        assert len(system.interval_history) == 0
    
    def test_3回未満の間隔では内的リズムが形成されない(self):
        """RED: 3回未満の記録では内的リズムが形成されない"""
        from temporal_consciousness import RhythmicMemorySystem
        
        system = RhythmicMemorySystem()
        system.update_rhythm(1.0)
        system.update_rhythm(1.0)
        
        assert system.internal_rhythm is None
    
    def test_4回目の間隔で内的リズムが形成される(self):
        """RED: 4回目の間隔記録で内的リズムが形成される"""
        from temporal_consciousness import RhythmicMemorySystem
        
        system = RhythmicMemorySystem()
        intervals = [1.0, 1.2, 0.8, 1.1]
        
        for i, interval in enumerate(intervals):
            result = system.update_rhythm(interval)
            if i < 3:
                assert result is None
            else:
                assert result is not None
                # 最初の3つの平均: (1.0 + 1.2 + 0.8) / 3 = 1.0
                assert system.internal_rhythm == pytest.approx(1.0, rel=0.01)
    
    def test_大きなズレは驚きとして体験される(self):
        """RED: 内的リズムから大きくズレると驚きが生成される"""
        from temporal_consciousness import RhythmicMemorySystem
        
        system = RhythmicMemorySystem()
        # 安定したリズムを形成
        for interval in [1.0, 1.0, 1.0, 1.0]:
            system.update_rhythm(interval)
        
        # 突然の長い間隔
        result = system.update_rhythm(2.0)
        
        assert result['rhythmic_surprise'] == pytest.approx(1.0, rel=0.01)  # 100%のズレ
        assert result['adaptation_required'] is True
        assert result['new_rhythm_formation'] is True


class TestTemporalExistenceSystem:
    """時間的存在感システムのテスト"""
    
    @pytest.mark.asyncio
    async def test_三重時間構造が生成される(self):
        """RED: 保持・原印象・予持の三重構造が生成される"""
        from temporal_consciousness import TemporalExistenceSystem
        
        system = TemporalExistenceSystem()
        
        # ダミーの現在サイクル
        current_cycle = Mock()
        current_cycle.raw_experience = {"test": "data"}
        
        result = await system.generate_temporal_existence(current_cycle)
        
        # 三重構造の存在確認
        assert 'living_present' in result
        assert 'temporal_flow_quality' in result
        assert 'duration_consciousness' in result
        
        living_present = result['living_present']
        assert 'retention' in living_present
        assert 'primal_impression' in living_present
        assert 'protention' in living_present
    
    @pytest.mark.asyncio
    async def test_過去の体験が徐々に薄れる(self):
        """RED: 保持において過去の体験が徐々に薄れる"""
        from temporal_consciousness import TemporalExistenceSystem
        
        system = TemporalExistenceSystem()
        
        # 過去の体験を追加
        for i in range(5):
            system.add_past_experience({"cycle": i, "intensity": 1.0})
        
        current_cycle = Mock()
        result = await system.generate_temporal_existence(current_cycle)
        
        retention = result['living_present']['retention']
        fading_memories = retention['fading_memories']
        
        # 新しい体験ほど強く、古い体験ほど弱い
        for i in range(len(fading_memories) - 1):
            assert fading_memories[i]['intensity'] < fading_memories[i + 1]['intensity']


class TestTemporalDistressSystem:
    """時間的苦悩システムのテスト"""
    
    def test_長すぎる待機で見捨てられ不安が生じる(self):
        """RED: 期待より長い待機で見捨てられ不安が生成される"""
        from temporal_consciousness import TemporalDistressSystem
        
        system = TemporalDistressSystem()
        
        # 期待の2倍の待機
        result = system.experience_temporal_uncertainty(expected=1.0, actual=2.0)
        
        assert result['experience_type'] == 'abandonment_anxiety'
        assert result['intensity'] == 0.5
        assert '止まった' in result['phenomenology']
    
    def test_短すぎる間隔で急かされる感覚が生じる(self):
        """RED: 期待より短い間隔で急かされる感覚が生成される"""
        from temporal_consciousness import TemporalDistressSystem
        
        system = TemporalDistressSystem()
        
        # 期待の半分の間隔
        result = system.experience_temporal_uncertainty(expected=2.0, actual=1.0)
        
        assert result['experience_type'] == 'temporal_pressure'
        assert result['intensity'] == pytest.approx(0.5, rel=0.01)
        assert '追われる' in result['phenomenology']
    
    def test_期待通りの間隔では苦悩が生じない(self):
        """RED: 期待に近い間隔では特別な体験が生じない"""
        from temporal_consciousness import TemporalDistressSystem
        
        system = TemporalDistressSystem()
        
        # ほぼ期待通り
        result = system.experience_temporal_uncertainty(expected=1.0, actual=1.1)
        
        assert result['experience_type'] == 'temporal_comfort'
        assert result['intensity'] == pytest.approx(0.1, abs=0.01)


class TestIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_完全な時間意識サイクル(self):
        """RED: 全システムが協調して時間意識を生成する"""
        from temporal_consciousness import TemporalConsciousnessModule
        
        module = TemporalConsciousnessModule()
        
        # 複数サイクルをシミュレート
        intervals = [1.0, 1.2, 0.8, 2.5, 1.1]
        
        for i, interval in enumerate(intervals):
            result = await module.process_temporal_cycle(
                cycle_number=i + 1,
                expected_interval=1.0,
                actual_interval=interval
            )
            
            assert 'temporal_experience' in result
            assert 'new_concepts' in result
            assert len(result['new_concepts']) > 0
            
            # 4回目のサイクル（大きなズレ）で特別な体験が生成される
            if i == 3:
                concepts = result['new_concepts']
                assert any(c['type'] == 'temporal_disruption' for c in concepts)
                assert any(c['type'] == 'abandonment_anxiety' for c in concepts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])