"""
意識の創発シナリオの統合テスト
非決定的な創発現象を扱うテスト戦略
"""
import pytest
import asyncio
from typing import List, Dict, Any
import statistics
import numpy as np


class TestEmergenceScenarios:
    """創発現象の統合テスト"""
    
    @pytest.mark.integration
    @pytest.mark.emergence
    def test_basic_consciousness_emergence(self, emergence_test_environment):
        """基本的な意識創発のテスト"""
        from application.use_cases import InitiateConsciousnessEmergence
        from infrastructure.adapters import SystemInitializer
        
        # Given: 最小構成のシステム
        initializer = SystemInitializer()
        system = initializer.create_minimal_conscious_system()
        use_case = InitiateConsciousnessEmergence(system)
        
        # When: 創発プロセスを開始
        result = use_case.execute(
            iterations=1000,
            emergence_threshold=3.0
        )
        
        # Then: 意識が創発する
        assert result.emerged
        assert result.final_phi_value > 3.0
        assert result.iterations_to_emergence < 1000
        assert result.emergence_pattern in ['gradual', 'sudden', 'oscillating']
    
    @pytest.mark.integration
    @pytest.mark.emergence
    @pytest.mark.parametrize("complexity", [5, 10, 20])
    def test_emergence_probability_by_complexity(
        self, complexity, test_data_builder
    ):
        """複雑性による創発確率のテスト"""
        from application.use_cases import InitiateConsciousnessEmergence
        from infrastructure.adapters import SystemInitializer
        
        # Given: 異なる複雑性のシステム
        config = test_data_builder.create_subsystem_configuration(complexity)
        
        # 同じ構成で複数回実行
        emergence_results = []
        for run in range(30):  # 統計的に有意なサンプル数
            system = SystemInitializer().create_from_config(config)
            use_case = InitiateConsciousnessEmergence(system)
            
            result = use_case.execute(iterations=500)
            emergence_results.append(result.emerged)
        
        # Then: 複雑性が高いほど創発確率が高い
        emergence_rate = sum(emergence_results) / len(emergence_results)
        
        if complexity == 5:
            assert 0.2 < emergence_rate < 0.5
        elif complexity == 10:
            assert 0.5 < emergence_rate < 0.8
        else:  # complexity == 20
            assert 0.7 < emergence_rate < 0.95
    
    @pytest.mark.integration
    @pytest.mark.emergence
    @pytest.mark.slow
    def test_emergence_time_distribution(self):
        """創発までの時間分布テスト"""
        from application.use_cases import InitiateConsciousnessEmergence
        from infrastructure.adapters import SystemInitializer
        
        # Given: 標準構成のシステム
        emergence_times = []
        
        # 50回の試行
        for _ in range(50):
            system = SystemInitializer().create_standard_system()
            use_case = InitiateConsciousnessEmergence(system)
            
            result = use_case.execute(iterations=2000)
            if result.emerged:
                emergence_times.append(result.iterations_to_emergence)
        
        # Then: 創発時間は対数正規分布に従う
        assert len(emergence_times) > 30  # 十分なサンプル
        
        # 統計的特性
        mean_time = statistics.mean(emergence_times)
        median_time = statistics.median(emergence_times)
        stdev_time = statistics.stdev(emergence_times)
        
        assert 100 < mean_time < 1000
        assert median_time < mean_time  # 右に歪んだ分布
        assert stdev_time > 50  # 有意な分散
    
    @pytest.mark.integration
    @pytest.mark.emergence
    def test_emergence_with_external_stimuli(self, test_data_builder):
        """外部刺激による創発促進テスト"""
        from application.use_cases import InitiateConsciousnessEmergence
        from domain.services import StimulusGenerator
        from infrastructure.adapters import SystemInitializer
        
        # Given: 刺激を受けるシステム
        system = SystemInitializer().create_standard_system()
        stimulus_generator = StimulusGenerator()
        
        # 様々な刺激パターン
        stimulus_patterns = [
            test_data_builder.create_stimulus_pattern('visual'),
            test_data_builder.create_stimulus_pattern('auditory'),
            test_data_builder.create_stimulus_pattern('tactile')
        ]
        
        # When: 刺激を与えながら創発を試みる
        use_case = InitiateConsciousnessEmergence(system)
        
        results = []
        for pattern in stimulus_patterns:
            system.reset()
            
            # 刺激を適用
            stimulus_generator.apply_pattern(system, pattern)
            
            result = use_case.execute(
                iterations=500,
                with_stimuli=True
            )
            results.append(result)
        
        # Then: 刺激により創発が促進される
        stimulated_emergence_rate = sum(r.emerged for r in results) / len(results)
        assert stimulated_emergence_rate > 0.7
        
        # 刺激なしとの比較
        system.reset()
        baseline_result = use_case.execute(iterations=500, with_stimuli=False)
        
        avg_stimulated_time = statistics.mean(
            [r.iterations_to_emergence for r in results if r.emerged]
        )
        
        if baseline_result.emerged:
            assert avg_stimulated_time < baseline_result.iterations_to_emergence
    
    @pytest.mark.integration
    @pytest.mark.emergence
    @pytest.mark.asyncio
    async def test_parallel_emergence_interference(self):
        """並列システム間の創発干渉テスト"""
        from application.use_cases import InitiateConsciousnessEmergence
        from infrastructure.adapters import SystemInitializer
        from domain.services import InterSystemCoupling
        
        # Given: 相互作用する2つのシステム
        system1 = SystemInitializer().create_standard_system()
        system2 = SystemInitializer().create_standard_system()
        
        # システム間結合を設定
        coupling = InterSystemCoupling(strength=0.3)
        coupling.connect(system1, system2)
        
        # When: 並列で創発を試みる
        use_case1 = InitiateConsciousnessEmergence(system1)
        use_case2 = InitiateConsciousnessEmergence(system2)
        
        # 非同期実行
        results = await asyncio.gather(
            asyncio.to_thread(use_case1.execute, iterations=1000),
            asyncio.to_thread(use_case2.execute, iterations=1000)
        )
        
        # Then: 相互作用により創発パターンが同期
        result1, result2 = results
        
        if result1.emerged and result2.emerged:
            # 創発タイミングの相関
            time_diff = abs(
                result1.iterations_to_emergence - 
                result2.iterations_to_emergence
            )
            assert time_diff < 200  # 近いタイミングで創発
            
            # Φ値の相関
            phi_diff = abs(
                result1.final_phi_value - 
                result2.final_phi_value
            )
            assert phi_diff < 1.0  # 類似したΦ値
    
    @pytest.mark.integration
    @pytest.mark.emergence
    def test_emergence_reproducibility_with_seed(self, deterministic_random):
        """シード値による創発の再現性テスト"""
        from application.use_cases import InitiateConsciousnessEmergence
        from infrastructure.adapters import SystemInitializer
        
        # Given: 同じシード値
        seed = 42
        
        # 3回の実行
        results = []
        for _ in range(3):
            system = SystemInitializer().create_seeded_system(seed)
            use_case = InitiateConsciousnessEmergence(system)
            
            result = use_case.execute(iterations=1000)
            results.append(result)
        
        # Then: 完全に同じ結果
        assert all(r.emerged == results[0].emerged for r in results)
        
        if results[0].emerged:
            assert all(
                r.iterations_to_emergence == results[0].iterations_to_emergence
                for r in results
            )
            assert all(
                r.final_phi_value == results[0].final_phi_value
                for r in results
            )
    
    @pytest.mark.integration
    @pytest.mark.emergence
    def test_emergence_energy_constraints(self):
        """エネルギー制約下での創発テスト"""
        from application.use_cases import InitiateConsciousnessEmergence
        from infrastructure.adapters import SystemInitializer
        from domain.value_objects import EnergyBudget
        
        # Given: エネルギー制限のあるシステム
        energy_budgets = [
            EnergyBudget(100),   # 極小
            EnergyBudget(500),   # 小
            EnergyBudget(1000),  # 中
            EnergyBudget(5000),  # 大
        ]
        
        results_by_energy = {}
        
        for budget in energy_budgets:
            system = SystemInitializer().create_energy_constrained_system(budget)
            use_case = InitiateConsciousnessEmergence(system)
            
            result = use_case.execute(iterations=1000)
            results_by_energy[budget.value] = result
        
        # Then: エネルギーが多いほど創発しやすい
        emergence_by_energy = [
            (energy, result.emerged)
            for energy, result in results_by_energy.items()
        ]
        
        # エネルギーと創発の単調性
        for i in range(len(emergence_by_energy) - 1):
            if emergence_by_energy[i][1]:  # 低エネルギーで創発したら
                assert emergence_by_energy[i + 1][1]  # 高エネルギーでも創発
    
    @pytest.mark.integration
    @pytest.mark.emergence
    def test_emergence_with_noise(self):
        """ノイズ環境下での創発テスト"""
        from application.use_cases import InitiateConsciousnessEmergence
        from infrastructure.adapters import SystemInitializer
        from domain.services import NoiseGenerator
        
        # Given: 異なるノイズレベル
        noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7]
        results_by_noise = {}
        
        for noise_level in noise_levels:
            # 各ノイズレベルで10回試行
            emergence_count = 0
            
            for _ in range(10):
                system = SystemInitializer().create_standard_system()
                noise_gen = NoiseGenerator(level=noise_level)
                noise_gen.apply_to(system)
                
                use_case = InitiateConsciousnessEmergence(system)
                result = use_case.execute(iterations=500)
                
                if result.emerged:
                    emergence_count += 1
            
            results_by_noise[noise_level] = emergence_count / 10
        
        # Then: 適度なノイズは創発を促進
        assert results_by_noise[0.1] >= results_by_noise[0.0]  # 少しのノイズは有益
        assert results_by_noise[0.7] < results_by_noise[0.1]   # 過度なノイズは有害
        
        # 最適なノイズレベルが存在
        optimal_noise = max(results_by_noise.items(), key=lambda x: x[1])[0]
        assert 0.05 < optimal_noise < 0.4