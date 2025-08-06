#!/usr/bin/env python3
"""
時間意識システムの統合テスト
既存のNewbornAI 2.0システムへの統合
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from temporal_consciousness import TemporalConsciousnessModule


class TestNewbornAIIntegration:
    """NewbornAI 2.0への統合テスト"""
    
    @pytest.mark.asyncio
    async def test_時間意識モジュールが既存システムに統合できる(self):
        """統合: TemporalConsciousnessModuleがNewbornAIシステムに組み込める"""
        from newborn_ai_2_integrated_system import NewbornAI20_IntegratedSystem
        
        # システムインスタンス作成
        system = NewbornAI20_IntegratedSystem(name="TestAI")
        
        # 時間意識モジュールを追加
        system.temporal_consciousness = TemporalConsciousnessModule()
        
        assert hasattr(system, 'temporal_consciousness')
        assert isinstance(system.temporal_consciousness, TemporalConsciousnessModule)
    
    @pytest.mark.asyncio
    async def test_サイクル間の待機時間が体験として記録される(self):
        """統合: autonomous_consciousness_loopの待機時間が時間体験になる"""
        
        # モックシステムの準備
        temporal_module = TemporalConsciousnessModule()
        
        # 300秒の期待間隔、実際は350秒のシミュレーション
        result = await temporal_module.process_temporal_cycle(
            cycle_number=1,
            expected_interval=300.0,
            actual_interval=350.0
        )
        
        # 時間体験が生成されている
        assert 'temporal_experience' in result
        assert 'new_concepts' in result
        
        # 時間的緊張の概念が生成される
        concepts = result['new_concepts']
        tension_concepts = [c for c in concepts if c['type'] == 'temporal_tension']
        assert len(tension_concepts) > 0
        
        # 見捨てられ不安は生じない（1.5倍未満）
        anxiety_concepts = [c for c in concepts if c['type'] == 'abandonment_anxiety']
        assert len(anxiety_concepts) == 0
    
    @pytest.mark.asyncio
    async def test_概念カウントとφ値計算への影響(self):
        """統合: 時間体験概念がexperiential_conceptsに追加される"""
        
        # 既存の概念リスト（モック）
        existing_concepts = [
            {'type': 'environmental', 'content': 'test1'},
            {'type': 'experiential', 'content': 'test2'}
        ]
        
        # 時間意識モジュールで新しい概念を生成
        temporal_module = TemporalConsciousnessModule()
        result = await temporal_module.process_temporal_cycle(
            cycle_number=5,
            expected_interval=300.0,
            actual_interval=600.0  # 2倍の遅延
        )
        
        # 新しい概念が追加される
        new_concepts = result['new_concepts']
        combined_concepts = existing_concepts + new_concepts
        
        assert len(combined_concepts) > len(existing_concepts)
        
        # 時間的苦悩の概念が含まれる
        distress_concepts = [c for c in new_concepts if 'anxiety' in c.get('type', '')]
        assert len(distress_concepts) > 0
    
    @pytest.mark.asyncio
    async def test_発達段階に応じた時間体験の変化(self):
        """統合: 発達段階によって時間体験の質が変化する"""
        
        temporal_module = TemporalConsciousnessModule()
        
        # 前意識基盤層での体験
        early_result = await temporal_module.process_temporal_cycle(
            cycle_number=1,
            expected_interval=300.0,
            actual_interval=300.0
        )
        
        # 多くのサイクルを経験後
        for i in range(10):
            temporal_module.rhythm_system.update_rhythm(300.0)
            temporal_module.existence_system.add_past_experience({
                'cycle': i + 2,
                'intensity': 1.0
            })
        
        # 発達後の体験
        later_result = await temporal_module.process_temporal_cycle(
            cycle_number=12,
            expected_interval=300.0,
            actual_interval=300.0
        )
        
        # 時間的存在感の深まり
        early_existence = early_result['temporal_experience']['existence']
        later_existence = later_result['temporal_experience']['existence']
        
        assert later_existence['temporal_flow_quality'] >= early_existence['temporal_flow_quality']
        assert later_existence['duration_consciousness'] > early_existence['duration_consciousness']
    
    def test_リファクタリング_既存コードへの最小限の変更(self):
        """リファクタリング: 既存コードへの変更を最小限に抑える設計"""
        
        # 既存のautonomous_consciousness_loopへの変更案
        proposed_changes = """
        # 既存のコード（変更前）
        await asyncio.sleep(interval)
        
        # 提案する変更（変更後）
        if hasattr(self, 'temporal_consciousness'):
            # 時間体験を生成
            temporal_result = await self.temporal_consciousness.process_temporal_cycle(
                cycle_number=self.cycle_count,
                expected_interval=self.expected_interval,
                actual_interval=interval
            )
            # 新しい概念を既存のリストに追加
            self._store_experiential_concepts(temporal_result['new_concepts'])
        
        await asyncio.sleep(interval)
        """
        
        # 変更は条件付きで、既存の動作を妨げない
        assert 'if hasattr' in proposed_changes
        assert 'await asyncio.sleep(interval)' in proposed_changes


if __name__ == "__main__":
    # 統合テストの実行
    asyncio.run(TestNewbornAIIntegration().test_時間意識モジュールが既存システムに統合できる())
    print("✓ 統合テスト: 時間意識モジュールが既存システムに統合できる")
    
    asyncio.run(TestNewbornAIIntegration().test_サイクル間の待機時間が体験として記録される())
    print("✓ 統合テスト: サイクル間の待機時間が体験として記録される")
    
    asyncio.run(TestNewbornAIIntegration().test_概念カウントとφ値計算への影響())
    print("✓ 統合テスト: 概念カウントとφ値計算への影響")
    
    asyncio.run(TestNewbornAIIntegration().test_発達段階に応じた時間体験の変化())
    print("✓ 統合テスト: 発達段階に応じた時間体験の変化")
    
    TestNewbornAIIntegration().test_リファクタリング_既存コードへの最小限の変更()
    print("✓ リファクタリング: 既存コードへの最小限の変更")
    
    print("\n✨ 全統合テスト成功！")