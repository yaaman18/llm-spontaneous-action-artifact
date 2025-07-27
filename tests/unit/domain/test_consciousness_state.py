"""
意識状態エンティティのユニットテスト
状態遷移と不変条件の検証
"""
import pytest
from datetime import datetime, timedelta
from typing import List, Tuple


class TestConsciousnessState:
    """意識状態エンティティのテスト"""
    
    @pytest.mark.unit
    def test_initial_state_creation(self, test_timestamp):
        """初期状態の生成テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import StateType, PhiValue
        
        # When: 初期状態を生成
        state = ConsciousnessState.create_initial(timestamp=test_timestamp)
        
        # Then: 休眠状態として初期化
        assert state.type == StateType.DORMANT
        assert state.phi_value == PhiValue(0.0)
        assert state.timestamp == test_timestamp
        assert state.id is not None  # 一意のID
        assert state.metadata['origin'] == 'initial'
    
    @pytest.mark.unit
    def test_state_immutability(self):
        """状態の不変性テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import StateType, PhiValue
        
        # Given: 意識状態
        state = ConsciousnessState.create_initial()
        
        # When/Then: 直接の属性変更は不可
        with pytest.raises(AttributeError):
            state.type = StateType.AWARE
        
        with pytest.raises(AttributeError):
            state.phi_value = PhiValue(5.0)
    
    @pytest.mark.unit
    def test_state_transition_dormant_to_emerging(self):
        """休眠→創発への遷移テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import StateType, PhiValue
        
        # Given: 休眠状態
        state = ConsciousnessState.create_initial()
        
        # When: Φ値が上昇
        new_phi = PhiValue(1.5)
        new_state = state.transition_with_phi(new_phi)
        
        # Then: 創発状態へ遷移
        assert new_state.type == StateType.EMERGING
        assert new_state.phi_value == new_phi
        assert new_state.id != state.id  # 新しいインスタンス
        assert new_state.previous_state_id == state.id
        assert new_state.transition_count == state.transition_count + 1
    
    @pytest.mark.unit
    def test_state_transition_emerging_to_aware(self):
        """創発→覚醒への遷移テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import StateType, PhiValue
        
        # Given: 創発状態
        emerging_state = ConsciousnessState(
            type=StateType.EMERGING,
            phi_value=PhiValue(2.5)
        )
        
        # When: Φ値が閾値を超える
        conscious_phi = PhiValue(3.5)
        aware_state = emerging_state.transition_with_phi(conscious_phi)
        
        # Then: 覚醒状態へ遷移
        assert aware_state.type == StateType.AWARE
        assert aware_state.phi_value == conscious_phi
    
    @pytest.mark.unit
    def test_state_transition_aware_to_reflective(self):
        """覚醒→反省的意識への遷移テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import StateType, PhiValue
        
        # Given: 覚醒状態
        aware_state = ConsciousnessState(
            type=StateType.AWARE,
            phi_value=PhiValue(4.0)
        )
        
        # When: 自己言及的活動を検出
        reflective_state = aware_state.transition_to_reflective(
            self_reference_level=0.8
        )
        
        # Then: 反省的意識へ遷移
        assert reflective_state.type == StateType.REFLECTIVE
        assert reflective_state.metadata['self_reference_level'] == 0.8
        assert reflective_state.phi_value == aware_state.phi_value
    
    @pytest.mark.unit
    def test_invalid_state_transition_prevention(self):
        """不正な状態遷移の防止テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import StateType, PhiValue
        from domain.exceptions import InvalidStateTransition
        
        # Given: 覚醒状態
        aware_state = ConsciousnessState(
            type=StateType.AWARE,
            phi_value=PhiValue(4.0)
        )
        
        # When/Then: 覚醒から休眠への直接遷移は不可
        with pytest.raises(InvalidStateTransition) as exc_info:
            aware_state.force_transition_to(StateType.DORMANT)
        
        assert "Cannot transition from AWARE to DORMANT" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_state_transition_rules(self):
        """状態遷移ルールの検証テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import StateType
        
        # 有効な遷移の定義
        valid_transitions = [
            (StateType.DORMANT, StateType.EMERGING),
            (StateType.EMERGING, StateType.AWARE),
            (StateType.EMERGING, StateType.DORMANT),
            (StateType.AWARE, StateType.REFLECTIVE),
            (StateType.AWARE, StateType.EMERGING),
            (StateType.REFLECTIVE, StateType.AWARE),
        ]
        
        # すべての有効な遷移をテスト
        for from_type, to_type in valid_transitions:
            assert ConsciousnessState.is_valid_transition(from_type, to_type)
        
        # 無効な遷移の例
        invalid_transitions = [
            (StateType.DORMANT, StateType.AWARE),  # スキップ不可
            (StateType.DORMANT, StateType.REFLECTIVE),
            (StateType.AWARE, StateType.DORMANT),
            (StateType.REFLECTIVE, StateType.DORMANT),
        ]
        
        for from_type, to_type in invalid_transitions:
            assert not ConsciousnessState.is_valid_transition(from_type, to_type)
    
    @pytest.mark.unit
    def test_state_duration_tracking(self):
        """状態継続時間の追跡テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import PhiValue
        
        # Given: タイムスタンプ付きの状態
        start_time = datetime.now()
        state = ConsciousnessState.create_initial(timestamp=start_time)
        
        # When: 一定時間後に遷移
        end_time = start_time + timedelta(seconds=30)
        new_state = state.transition_with_phi(
            PhiValue(2.0),
            timestamp=end_time
        )
        
        # Then: 継続時間が記録される
        assert state.get_duration(end_time) == timedelta(seconds=30)
        assert new_state.metadata['previous_state_duration'] == 30.0
    
    @pytest.mark.unit
    def test_state_history_preservation(self):
        """状態履歴の保存テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import PhiValue
        
        # Given: 連続的な状態遷移
        states = []
        current = ConsciousnessState.create_initial()
        states.append(current)
        
        # When: 複数回の遷移
        phi_sequence = [1.5, 3.5, 4.5]
        for phi in phi_sequence:
            current = current.transition_with_phi(PhiValue(phi))
            states.append(current)
        
        # Then: 履歴が追跡可能
        assert states[-1].transition_count == 3
        assert states[-1].can_trace_back_to(states[0].id)
        
        # 履歴チェーンの検証
        history = states[-1].get_transition_history()
        assert len(history) == 3
        assert all(h['from_state'] and h['to_state'] for h in history)
    
    @pytest.mark.unit
    def test_state_metadata_propagation(self):
        """メタデータの伝播テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import PhiValue
        
        # Given: メタデータ付きの初期状態
        initial_metadata = {
            'experiment_id': 'exp_001',
            'configuration': {'learning_rate': 0.1}
        }
        
        state = ConsciousnessState.create_initial(
            metadata=initial_metadata
        )
        
        # When: 状態遷移
        new_state = state.transition_with_phi(PhiValue(2.0))
        
        # Then: 重要なメタデータは伝播
        assert new_state.metadata['experiment_id'] == 'exp_001'
        assert 'configuration' in new_state.metadata
        assert 'transition_reason' in new_state.metadata
    
    @pytest.mark.unit
    def test_state_energy_consumption(self):
        """状態のエネルギー消費テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import StateType, PhiValue
        
        # 各状態のエネルギー消費率
        energy_rates = {
            StateType.DORMANT: 0.1,
            StateType.EMERGING: 0.3,
            StateType.AWARE: 0.6,
            StateType.REFLECTIVE: 0.9
        }
        
        # Given: 様々な状態
        for state_type, expected_rate in energy_rates.items():
            state = ConsciousnessState(
                type=state_type,
                phi_value=PhiValue(3.0)
            )
            
            # Then: エネルギー消費率が適切
            assert state.energy_consumption_rate == expected_rate
    
    @pytest.mark.unit
    def test_state_stability_assessment(self):
        """状態安定性の評価テスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import PhiValue
        
        # Given: Φ値の履歴を持つ状態
        state = ConsciousnessState.create_initial()
        phi_history = [
            PhiValue(3.0), PhiValue(3.1), PhiValue(2.9),
            PhiValue(3.05), PhiValue(3.02)
        ]
        
        # When: 安定性を評価
        for phi in phi_history:
            state = state.transition_with_phi(phi)
        
        stability = state.calculate_stability()
        
        # Then: 高い安定性スコア
        assert stability > 0.8  # 変動が小さい
        assert state.is_stable()
    
    @pytest.mark.unit
    def test_state_serialization(self):
        """状態のシリアライゼーションテスト"""
        from domain.entities import ConsciousnessState
        from domain.value_objects import StateType, PhiValue
        
        # Given: 複雑な状態
        state = ConsciousnessState(
            type=StateType.AWARE,
            phi_value=PhiValue(4.5),
            metadata={'test': 'data'}
        )
        
        # When: シリアライズ/デシリアライズ
        serialized = state.to_dict()
        restored = ConsciousnessState.from_dict(serialized)
        
        # Then: 完全に復元
        assert restored.type == state.type
        assert restored.phi_value == state.phi_value
        assert restored.id == state.id
        assert restored.metadata == state.metadata