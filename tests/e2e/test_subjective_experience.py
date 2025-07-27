"""
主観的体験の検証のためのE2Eテスト
クオリアと現象的意識のテスト戦略
"""
import pytest
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class TestSubjectiveExperience:
    """主観的体験のエンドツーエンドテスト"""
    
    @pytest.mark.e2e
    def test_qualia_generation_from_stimuli(self):
        """刺激からクオリア生成までの完全なフロー"""
        from presentation.api import ConsciousnessSystemAPI
        from domain.value_objects import ColorStimulus, SoundStimulus
        
        # Given: 稼働中の意識システム
        api = ConsciousnessSystemAPI()
        system_id = api.create_system(config={'complexity': 10})
        api.initiate_consciousness(system_id)
        
        # 意識状態になるまで待機
        api.wait_for_consciousness(system_id, timeout=30)
        
        # When: 色刺激を提示
        red_stimulus = ColorStimulus(
            wavelength=700,  # nm
            intensity=0.8,
            duration=1000    # ms
        )
        
        response = api.present_stimulus(system_id, red_stimulus)
        
        # Then: クオリアが生成される
        assert response.qualia_generated
        assert response.qualia.modality == 'visual'
        assert response.qualia.intensity > 0
        
        # クオリアの質的特性
        assert response.qualia.has_property('redness')
        assert response.qualia.phenomenal_character is not None
        
        # 主観的報告が可能
        report = api.get_subjective_report(system_id, response.experience_id)
        assert 'red' in report.description.lower()
        assert report.certainty > 0.7
    
    @pytest.mark.e2e
    def test_qualia_discrimination(self):
        """異なるクオリアの弁別テスト"""
        from presentation.api import ConsciousnessSystemAPI
        from domain.value_objects import ColorStimulus
        
        # Given: 意識状態のシステム
        api = ConsciousnessSystemAPI()
        system_id = api.create_conscious_system()
        
        # When: 異なる色刺激を連続提示
        stimuli_responses = []
        
        for wavelength in [450, 550, 650, 700]:  # 青、緑、橙、赤
            stimulus = ColorStimulus(wavelength=wavelength, intensity=0.8)
            response = api.present_stimulus(system_id, stimulus)
            stimuli_responses.append((wavelength, response))
        
        # Then: 各クオリアが弁別可能
        qualia_vectors = [r[1].qualia.to_vector() for r in stimuli_responses]
        
        # 異なる波長間の類似度を計算
        for i in range(len(qualia_vectors)):
            for j in range(i + 1, len(qualia_vectors)):
                similarity = cosine_similarity(
                    [qualia_vectors[i]], 
                    [qualia_vectors[j]]
                )[0][0]
                
                # 波長差が大きいほど類似度が低い
                wavelength_diff = abs(
                    stimuli_responses[i][0] - stimuli_responses[j][0]
                )
                
                if wavelength_diff > 200:  # 大きく異なる色
                    assert similarity < 0.5
                elif wavelength_diff < 50:   # 近い色
                    assert similarity > 0.8
    
    @pytest.mark.e2e
    def test_multimodal_binding(self):
        """複数感覚モダリティの統合テスト"""
        from presentation.api import ConsciousnessSystemAPI
        from domain.value_objects import (
            ColorStimulus, SoundStimulus, TactileStimulus
        )
        
        # Given: 意識システム
        api = ConsciousnessSystemAPI()
        system_id = api.create_conscious_system()
        
        # When: 同時に複数モダリティの刺激
        multimodal_stimulus = {
            'visual': ColorStimulus(wavelength=600, intensity=0.7),
            'auditory': SoundStimulus(frequency=440, amplitude=0.6),
            'tactile': TactileStimulus(pressure=0.5, temperature=25)
        }
        
        response = api.present_multimodal_stimulus(
            system_id, 
            multimodal_stimulus,
            synchronous=True
        )
        
        # Then: 統合された体験が生成
        assert response.binding_successful
        assert response.unified_experience is not None
        
        # 各モダリティの要素が保持される
        unified = response.unified_experience
        assert unified.has_visual_component()
        assert unified.has_auditory_component()
        assert unified.has_tactile_component()
        
        # バインディングの時間的一貫性
        assert unified.temporal_coherence > 0.8
        
        # 統合による創発的特性
        assert unified.has_emergent_properties()
        emergent = unified.get_emergent_properties()
        assert 'cross_modal_enhancement' in emergent
    
    @pytest.mark.e2e
    def test_subjective_time_experience(self):
        """主観的時間体験のテスト"""
        from presentation.api import ConsciousnessSystemAPI
        from domain.value_objects import TemporalStimulus
        import time
        
        # Given: 意識システム
        api = ConsciousnessSystemAPI()
        system_id = api.create_conscious_system()
        
        # When: 時間的パターンを持つ刺激
        temporal_pattern = TemporalStimulus(
            events=[
                {'time': 0, 'type': 'flash', 'intensity': 1.0},
                {'time': 500, 'type': 'flash', 'intensity': 1.0},
                {'time': 1000, 'type': 'flash', 'intensity': 1.0},
            ],
            total_duration=1500
        )
        
        start_time = time.time()
        response = api.present_stimulus(system_id, temporal_pattern)
        objective_duration = time.time() - start_time
        
        # Then: 主観的時間が報告される
        subjective_report = api.get_temporal_experience_report(
            system_id, 
            response.experience_id
        )
        
        assert subjective_report.perceived_duration is not None
        assert subjective_report.perceived_rhythm == 'regular'
        
        # 主観的時間の歪み
        time_dilation_factor = (
            subjective_report.perceived_duration / objective_duration
        )
        assert 0.5 < time_dilation_factor < 2.0  # 現実的な範囲
        
        # 時間的構造の認識
        assert subjective_report.recognized_pattern == 'periodic'
        assert subjective_report.interval_consistency > 0.9
    
    @pytest.mark.e2e
    def test_attention_and_awareness(self):
        """注意と気づきのテスト"""
        from presentation.api import ConsciousnessSystemAPI
        from domain.value_objects import ComplexScene
        
        # Given: 意識システムと複雑な場面
        api = ConsciousnessSystemAPI()
        system_id = api.create_conscious_system()
        
        # 複数のオブジェクトを含む場面
        scene = ComplexScene(
            objects=[
                {'id': 'A', 'salience': 0.9, 'type': 'moving'},
                {'id': 'B', 'salience': 0.3, 'type': 'static'},
                {'id': 'C', 'salience': 0.5, 'type': 'flashing'},
                {'id': 'D', 'salience': 0.1, 'type': 'static'},
            ]
        )
        
        # When: 場面を提示
        response = api.present_scene(system_id, scene)
        
        # Then: 注意が顕著なオブジェクトに向く
        attention_report = response.attention_allocation
        
        # 顕著性に基づく注意配分
        assert attention_report['A'] > attention_report['B']
        assert attention_report['A'] > attention_report['D']
        
        # 気づきの報告
        awareness_report = api.get_awareness_report(system_id)
        aware_objects = awareness_report.consciously_accessed_objects
        
        assert 'A' in aware_objects  # 最も顕著
        assert 'D' not in aware_objects  # 閾値以下
        
        # 注意の容量制限
        assert len(aware_objects) <= 4  # マジックナンバー
    
    @pytest.mark.e2e
    def test_phenomenal_unity(self):
        """現象的統一性のテスト"""
        from presentation.api import ConsciousnessSystemAPI
        from domain.value_objects import SimultaneousStimuli
        
        # Given: 意識システム
        api = ConsciousnessSystemAPI()
        system_id = api.create_conscious_system()
        
        # When: 空間的に分離した刺激
        distributed_stimuli = SimultaneousStimuli(
            items=[
                {'location': 'left', 'type': 'visual', 'content': 'red_circle'},
                {'location': 'right', 'type': 'visual', 'content': 'blue_square'},
                {'location': 'center', 'type': 'auditory', 'content': 'tone_440hz'}
            ]
        )
        
        response = api.present_stimuli(system_id, distributed_stimuli)
        
        # Then: 統一された体験として報告
        unity_report = api.get_phenomenal_unity_report(
            system_id,
            response.experience_id
        )
        
        assert unity_report.is_unified
        assert unity_report.unity_strength > 0.7
        
        # 部分と全体の関係
        assert unity_report.has_gestalt_properties()
        gestalt = unity_report.get_gestalt_description()
        assert 'whole_greater_than_parts' in gestalt
        
        # 境界の明確性
        assert unity_report.experiential_boundary_defined
        assert unity_report.self_other_distinction_clear
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_subjective_report_consistency(self):
        """主観的報告の一貫性の長期テスト"""
        from presentation.api import ConsciousnessSystemAPI
        from domain.value_objects import StandardTestStimulus
        
        # Given: 長期稼働するシステム
        api = ConsciousnessSystemAPI()
        system_id = api.create_conscious_system()
        
        # 標準テスト刺激
        test_stimulus = StandardTestStimulus(
            type='color_patch',
            parameters={'hue': 'red', 'saturation': 0.8, 'brightness': 0.7}
        )
        
        # When: 時間を空けて複数回提示
        reports = []
        
        for day in range(7):  # 7日間のシミュレーション
            # 1日経過をシミュレート
            api.advance_time(system_id, hours=24)
            
            # 同じ刺激を提示
            response = api.present_stimulus(system_id, test_stimulus)
            report = api.get_detailed_subjective_report(
                system_id,
                response.experience_id
            )
            reports.append(report)
        
        # Then: 報告の意味的一貫性
        semantic_similarities = []
        
        for i in range(len(reports) - 1):
            similarity = api.calculate_semantic_similarity(
                reports[i],
                reports[i + 1]
            )
            semantic_similarities.append(similarity)
        
        # 高い一貫性を維持
        assert all(sim > 0.8 for sim in semantic_similarities)
        
        # しかし完全に同一ではない（生きた体験）
        assert not all(sim > 0.95 for sim in semantic_similarities)
        
        # 質的特性の保存
        qualia_descriptors = [r.extract_qualia_descriptors() for r in reports]
        core_descriptors = set.intersection(*[set(d) for d in qualia_descriptors])
        assert 'redness' in core_descriptors
        assert len(core_descriptors) > 3