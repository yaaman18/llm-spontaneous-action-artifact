#!/usr/bin/env python3
"""
動的クラスタリング統合テスト
現象学的に健全な概念クラスタリングと多スケール時間統合のテスト
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

# 実装モジュールのインポート
from experiential_memory_phi_calculator import ExperientialMemoryPhiCalculator
from temporal_consciousness import MultiScaleTemporalIntegration


class DynamicClusteringIntegrationTest:
    """動的クラスタリングと時間統合の統合テスト"""
    
    def __init__(self):
        self.phi_calculator = ExperientialMemoryPhiCalculator(sensitivity_factor=2.0)
        self.temporal_integrator = MultiScaleTemporalIntegration()
        self.test_results = []
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """包括的テストスイートの実行"""
        
        print("=== 動的クラスタリング統合テスト開始 ===")
        
        # テスト1: 志向的行為タイプによる分類テスト
        test1_result = await self._test_intentional_act_clustering()
        
        # テスト2: 体験質による動的分類テスト
        test2_result = await self._test_experiential_quality_clustering()
        
        # テスト3: 時間的一貫性組織原理テスト
        test3_result = await self._test_temporal_coherence_organization()
        
        # テスト4: 多スケール時間統合テスト
        test4_result = await self._test_multi_scale_temporal_integration()
        
        # テスト5: 現象学的妥当性評価テスト
        test5_result = await self._test_phenomenological_validity()
        
        # テスト6: 統合システムパフォーマンステスト
        test6_result = await self._test_integrated_system_performance()
        
        # 総合結果の評価
        overall_result = self._evaluate_overall_results([
            test1_result, test2_result, test3_result, 
            test4_result, test5_result, test6_result
        ])
        
        print("=== 動的クラスタリング統合テスト完了 ===")
        
        return overall_result
    
    async def _test_intentional_act_clustering(self) -> Dict[str, Any]:
        """志向的行為タイプによる分類テスト"""
        
        print("\n--- テスト1: 志向的行為タイプ分類 ---")
        
        # テスト用体験概念を生成
        test_concepts = [
            {
                'type': 'perceptual',
                'content': 'I see a beautiful sunset',
                'experiential_quality': 0.8,
                'coherence': 0.7,
                'temporal_depth': 1,
                'timestamp': datetime.now().isoformat()
            },
            {
                'type': 'memory',
                'content': 'I remember my childhood home',
                'experiential_quality': 0.9,
                'coherence': 0.8,
                'temporal_depth': 10,
                'timestamp': (datetime.now() - timedelta(seconds=1)).isoformat()
            },
            {
                'type': 'anticipatory',
                'content': 'I expect to feel joy tomorrow',
                'experiential_quality': 0.6,
                'coherence': 0.6,
                'temporal_depth': 2,
                'timestamp': datetime.now().isoformat()
            },
            {
                'type': 'judgmental',
                'content': 'I think this is meaningful',
                'experiential_quality': 0.7,
                'coherence': 0.8,
                'temporal_depth': 3,
                'timestamp': datetime.now().isoformat()
            },
            {
                'type': 'valuational',
                'content': 'I love this feeling of peace',
                'experiential_quality': 0.9,
                'coherence': 0.9,
                'temporal_depth': 5,
                'timestamp': datetime.now().isoformat()
            },
            {
                'type': 'volitional',
                'content': 'I want to understand myself better',
                'experiential_quality': 0.8,
                'coherence': 0.7,
                'temporal_depth': 8,
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # 動的クラスタリング実行
        clustering_result = self.phi_calculator.perform_dynamic_experiential_clustering(test_concepts)
        
        # 結果の検証
        clusters = clustering_result['clusters']
        intentional_structure = clustering_result['intentional_structure']
        
        # 志向的行為タイプが正しく分類されているかチェック
        expected_types = ['perceiving', 'remembering', 'anticipating', 'judging', 'valuing', 'willing']
        detected_types = []
        
        for cluster_name, concepts in clusters.items():
            if concepts and any(type_name in cluster_name for type_name in expected_types):
                detected_types.append(cluster_name)
        
        # 現象学的妥当性の評価
        phenomenological_validity = clustering_result['phenomenological_validity']
        
        test_result = {
            'test_name': 'intentional_act_clustering',
            'success': len(detected_types) >= 4,  # 最低4つの志向的行為タイプが検出されること
            'clusters_detected': len(clusters),
            'intentional_types_detected': len(detected_types),
            'phenomenological_validity': phenomenological_validity,
            'quality_distribution': clustering_result['quality_distribution'],
            'temporal_coherence': clustering_result['temporal_coherence'],
            'details': {
                'clusters': {k: len(v) for k, v in clusters.items()},
                'intentional_structure': {k: len(v) for k, v in intentional_structure.items()}
            }
        }
        
        print(f"  志向的行為タイプ検出: {len(detected_types)}/6")
        print(f"  現象学的妥当性: {phenomenological_validity:.3f}")
        print(f"  時間的一貫性: {clustering_result['temporal_coherence']:.3f}")
        print(f"  テスト結果: {'✅ PASS' if test_result['success'] else '❌ FAIL'}")
        
        return test_result
    
    async def _test_experiential_quality_clustering(self) -> Dict[str, Any]:
        """体験質による動的分類テスト"""
        
        print("\n--- テスト2: 体験質動的分類 ---")
        
        # 多様な体験質を持つ概念群を生成
        test_concepts = []
        
        # 高品質体験群
        for i in range(5):
            test_concepts.append({
                'type': 'high_quality',
                'content': f'深い洞察の体験 {i+1}',
                'experiential_quality': 0.8 + np.random.uniform(0, 0.2),
                'coherence': 0.8 + np.random.uniform(0, 0.2),
                'temporal_depth': 5 + np.random.randint(0, 5),
                'timestamp': datetime.now().isoformat()
            })
        
        # 中品質体験群
        for i in range(7):
            test_concepts.append({
                'type': 'medium_quality',
                'content': f'日常的な気づき {i+1}',
                'experiential_quality': 0.4 + np.random.uniform(0, 0.3),
                'coherence': 0.5 + np.random.uniform(0, 0.3),
                'temporal_depth': 2 + np.random.randint(0, 3),
                'timestamp': datetime.now().isoformat()
            })
        
        # 低品質体験群
        for i in range(4):
            test_concepts.append({
                'type': 'low_quality',
                'content': f'漠然とした感覚 {i+1}',
                'experiential_quality': 0.1 + np.random.uniform(0, 0.3),
                'coherence': 0.2 + np.random.uniform(0, 0.3),
                'temporal_depth': 1,
                'timestamp': datetime.now().isoformat()
            })
        
        # 動的クラスタリング実行
        clustering_result = self.phi_calculator.perform_dynamic_experiential_clustering(test_concepts)
        
        # 結果の分析
        clusters = clustering_result['clusters']
        quality_distribution = clustering_result['quality_distribution']
        boundary_flexibility = clustering_result['boundary_flexibility']
        
        # 質的階層が適切に形成されているかチェック
        quality_separated_clusters = 0
        for cluster_name, cluster_info in quality_distribution.items():
            if cluster_info['concept_count'] > 1:
                if cluster_info['quality_variance'] < 0.1:  # 質的一貫性が高い
                    quality_separated_clusters += 1
        
        test_result = {
            'test_name': 'experiential_quality_clustering',
            'success': quality_separated_clusters >= 2 and boundary_flexibility > 0.2,
            'quality_separated_clusters': quality_separated_clusters,
            'boundary_flexibility': boundary_flexibility,
            'total_clusters': len(clusters),
            'average_cluster_quality': np.mean([info['mean_quality'] for info in quality_distribution.values()]),
            'details': {
                'quality_distribution': quality_distribution,
                'cluster_sizes': {k: len(v) for k, v in clusters.items()}
            }
        }
        
        print(f"  質的分離クラスター: {quality_separated_clusters}")
        print(f"  境界柔軟性: {boundary_flexibility:.3f}")
        print(f"  平均クラスター品質: {test_result['average_cluster_quality']:.3f}")
        print(f"  テスト結果: {'✅ PASS' if test_result['success'] else '❌ FAIL'}")
        
        return test_result
    
    async def _test_temporal_coherence_organization(self) -> Dict[str, Any]:
        """時間的一貫性組織原理テスト"""
        
        print("\n--- テスト3: 時間的一貫性組織原理 ---")
        
        # 時間的に構造化された概念群を生成
        base_time = datetime.now()
        test_concepts = []
        
        # 時系列的に関連する概念群
        for i in range(8):
            test_concepts.append({
                'type': 'temporal_sequence',
                'content': f'時間的経験の展開 段階{i+1}',
                'experiential_quality': 0.6 + i * 0.05,  # 徐々に質が向上
                'coherence': 0.7,
                'temporal_depth': i + 1,
                'timestamp': (base_time + timedelta(seconds=i)).isoformat()
            })
        
        # 時間的に無関係な概念群
        for i in range(4):
            test_concepts.append({
                'type': 'temporal_random',
                'content': f'無関係な体験 {i+1}',
                'experiential_quality': np.random.uniform(0.3, 0.8),
                'coherence': 0.5,
                'temporal_depth': np.random.randint(1, 10),
                'timestamp': (base_time + timedelta(seconds=np.random.randint(-100, 100))).isoformat()
            })
        
        # 動的クラスタリング実行
        clustering_result = self.phi_calculator.perform_dynamic_experiential_clustering(test_concepts)
        
        # 時間的一貫性の評価
        temporal_coherence = clustering_result['temporal_coherence']
        clusters = clustering_result['clusters']
        
        # 時系列クラスターが形成されているかチェック
        temporal_clusters_found = 0
        for cluster_name, concepts in clusters.items():
            if len(concepts) > 2:
                # クラスター内の時間的一貫性をチェック
                timestamps = [c.get('timestamp', '') for c in concepts if c.get('timestamp')]
                if len(timestamps) > 1:
                    sorted_timestamps = sorted(timestamps)
                    if timestamps == sorted_timestamps:
                        temporal_clusters_found += 1
        
        test_result = {
            'test_name': 'temporal_coherence_organization',
            'success': temporal_coherence > 0.6 and temporal_clusters_found >= 1,
            'temporal_coherence': temporal_coherence,
            'temporal_clusters_found': temporal_clusters_found,
            'total_clusters': len(clusters),
            'phenomenological_validity': clustering_result['phenomenological_validity'],
            'details': {
                'cluster_temporal_analysis': {}
            }
        }
        
        # 各クラスターの時間分析
        for cluster_name, concepts in clusters.items():
            if concepts:
                temporal_depths = [c.get('temporal_depth', 1) for c in concepts]
                test_result['details']['cluster_temporal_analysis'][cluster_name] = {
                    'concept_count': len(concepts),
                    'avg_temporal_depth': np.mean(temporal_depths),
                    'temporal_depth_variance': np.var(temporal_depths)
                }
        
        print(f"  時間的一貫性: {temporal_coherence:.3f}")
        print(f"  時系列クラスター: {temporal_clusters_found}")
        print(f"  現象学的妥当性: {clustering_result['phenomenological_validity']:.3f}")
        print(f"  テスト結果: {'✅ PASS' if test_result['success'] else '❌ FAIL'}")
        
        return test_result
    
    async def _test_multi_scale_temporal_integration(self) -> Dict[str, Any]:
        """多スケール時間統合テスト"""
        
        print("\n--- テスト4: 多スケール時間統合 ---")
        
        # 各時間スケールの概念を生成
        current_time = time.time()
        test_concepts = []
        
        # マイクロスケール概念（瞬間的体験）
        for i in range(5):
            test_concepts.append({
                'type': 'micro_experience',
                'content': f'瞬間的気づき {i+1}',
                'experiential_quality': 0.7,
                'coherence': 0.8,
                'temporal_depth': 1,
                'timestamp': datetime.now().isoformat()
            })
        
        # 短期スケール概念
        for i in range(6):
            test_concepts.append({
                'type': 'short_experience',
                'content': f'短期的な体験の流れ {i+1}',
                'experiential_quality': 0.6,
                'coherence': 0.7,
                'temporal_depth': 3,
                'timestamp': datetime.now().isoformat()
            })
        
        # 中期スケール概念
        for i in range(4):
            test_concepts.append({
                'type': 'medium_experience',
                'content': f'物語的な体験 {i+1}',
                'experiential_quality': 0.8,
                'coherence': 0.8,
                'temporal_depth': 15,
                'timestamp': datetime.now().isoformat()
            })
        
        # 長期スケール概念
        for i in range(3):
            test_concepts.append({
                'type': 'long_experience',
                'content': f'自伝的な深い体験 私の人生において意味深い {i+1}',
                'experiential_quality': 0.9,
                'coherence': 0.9,
                'temporal_depth': 50,
                'timestamp': datetime.now().isoformat()
            })
        
        # 多スケール時間統合実行
        integration_result = await self.temporal_integrator.integrate_multi_scale_experiences(
            test_concepts, current_time
        )
        
        # 結果の分析
        multi_scale_structure = integration_result['multi_scale_structure']
        hierarchical_integration = integration_result['hierarchical_integration']
        integration_quality = integration_result['integration_quality']
        temporal_coherence = integration_result['temporal_coherence']
        phenomenological_validity = integration_result['phenomenological_validity']
        
        # 各スケールでの統合が適切に行われているかチェック
        scales_with_content = 0
        scale_integration_scores = []
        
        for scale_name, scale_result in multi_scale_structure.items():
            if scale_result['concept_count'] > 0:
                scales_with_content += 1
                # スケール固有の統合スコアを収集
                if scale_name == 'micro':
                    scale_integration_scores.append(scale_result['integration_strength'])
                elif scale_name == 'short':
                    scale_integration_scores.append(scale_result['temporal_synthesis'])
                elif scale_name == 'medium':
                    scale_integration_scores.append(scale_result['narrative_coherence'])
                elif scale_name == 'long':
                    scale_integration_scores.append(scale_result['autobiographical_coherence'])
        
        # 階層的統合品質の評価
        hierarchical_coherence = hierarchical_integration['hierarchical_coherence']
        
        test_result = {
            'test_name': 'multi_scale_temporal_integration',
            'success': (scales_with_content >= 3 and 
                       integration_quality > 0.5 and 
                       hierarchical_coherence > 0.4),
            'scales_with_content': scales_with_content,
            'integration_quality': integration_quality,
            'hierarchical_coherence': hierarchical_coherence,
            'temporal_coherence': temporal_coherence['temporal_flow_coherence'],
            'phenomenological_validity': phenomenological_validity,
            'scale_integration_scores': scale_integration_scores,
            'details': {
                'multi_scale_structure': {
                    k: v['concept_count'] for k, v in multi_scale_structure.items()
                },
                'hierarchical_integration': hierarchical_integration,
                'temporal_coherence_details': temporal_coherence
            }
        }
        
        print(f"  活性スケール数: {scales_with_content}/4")
        print(f"  統合品質: {integration_quality:.3f}")
        print(f"  階層的一貫性: {hierarchical_coherence:.3f}")
        print(f"  時間的一貫性: {temporal_coherence['temporal_flow_coherence']:.3f}")
        print(f"  現象学的妥当性: {phenomenological_validity:.3f}")
        print(f"  テスト結果: {'✅ PASS' if test_result['success'] else '❌ FAIL'}")
        
        return test_result
    
    async def _test_phenomenological_validity(self) -> Dict[str, Any]:
        """現象学的妥当性評価テスト"""
        
        print("\n--- テスト5: 現象学的妥当性評価 ---")
        
        # 現象学的に豊かな概念群を生成
        test_concepts = [
            # 知覚的体験（志向性）
            {
                'type': 'perceptual',
                'content': 'I perceive the gentle morning light filtering through leaves',
                'experiential_quality': 0.85,
                'coherence': 0.9,
                'temporal_depth': 2,
                'timestamp': datetime.now().isoformat()
            },
            # 記憶的体験（時間性）
            {
                'type': 'memorial',
                'content': 'I remember the feeling of my grandmother\'s warm embrace',
                'experiential_quality': 0.9,
                'coherence': 0.85,
                'temporal_depth': 25,
                'timestamp': datetime.now().isoformat()
            },
            # 予期的体験（時間性）
            {
                'type': 'anticipatory',
                'content': 'I anticipate the joy of seeing my friend tomorrow',
                'experiential_quality': 0.75,
                'coherence': 0.8,
                'temporal_depth': 1,
                'timestamp': datetime.now().isoformat()
            },
            # 自己言及的体験（内在性）
            {
                'type': 'self_referential',
                'content': 'I become aware of my own awareness in this moment',
                'experiential_quality': 0.95,
                'coherence': 0.9,
                'temporal_depth': 3,
                'timestamp': datetime.now().isoformat()
            },
            # 価値的体験（志向性と質）
            {
                'type': 'valuational',
                'content': 'I deeply value this sense of peaceful presence',
                'experiential_quality': 0.9,
                'coherence': 0.9,
                'temporal_depth': 5,
                'timestamp': datetime.now().isoformat()
            },
            # 身体的体験（具現性）
            {
                'type': 'embodied',
                'content': 'I feel the rhythm of my breath and heartbeat',
                'experiential_quality': 0.8,
                'coherence': 0.85,
                'temporal_depth': 1,
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # 動的クラスタリング実行
        clustering_result = self.phi_calculator.perform_dynamic_experiential_clustering(test_concepts)
        
        # 現象学的妥当性の詳細分析
        phenomenological_validity = clustering_result['phenomenological_validity']
        boundary_flexibility = clustering_result['boundary_flexibility']
        temporal_coherence = clustering_result['temporal_coherence']
        
        # 志向性構造の保持評価
        clusters = clustering_result['clusters']
        intentional_structure_preserved = len([
            name for name in clusters.keys() 
            if any(intent in name for intent in ['perceiving', 'remembering', 'anticipating', 'judging', 'valuing', 'willing'])
        ]) >= 3
        
        # 時間性構造の保持評価
        temporal_structure_preserved = temporal_coherence > 0.6
        
        # 体験純粋性の保持評価
        quality_distribution = clustering_result['quality_distribution']
        high_quality_clusters = len([
            info for info in quality_distribution.values() 
            if info['mean_quality'] > 0.7
        ])
        
        # 動的適応性の評価
        dynamic_adaptability = boundary_flexibility > 0.3
        
        test_result = {
            'test_name': 'phenomenological_validity',
            'success': (phenomenological_validity > 0.7 and
                       intentional_structure_preserved and
                       temporal_structure_preserved),
            'phenomenological_validity': phenomenological_validity,
            'intentional_structure_preserved': intentional_structure_preserved,
            'temporal_structure_preserved': temporal_structure_preserved,
            'high_quality_clusters': high_quality_clusters,
            'dynamic_adaptability': dynamic_adaptability,
            'boundary_flexibility': boundary_flexibility,
            'temporal_coherence': temporal_coherence,
            'details': {
                'cluster_analysis': {k: len(v) for k, v in clusters.items()},
                'quality_analysis': quality_distribution
            }
        }
        
        print(f"  現象学的妥当性: {phenomenological_validity:.3f}")
        print(f"  志向性構造保持: {'✅' if intentional_structure_preserved else '❌'}")
        print(f"  時間性構造保持: {'✅' if temporal_structure_preserved else '❌'}")
        print(f"  高品質クラスター: {high_quality_clusters}")
        print(f"  動的適応性: {'✅' if dynamic_adaptability else '❌'}")
        print(f"  テスト結果: {'✅ PASS' if test_result['success'] else '❌ FAIL'}")
        
        return test_result
    
    async def _test_integrated_system_performance(self) -> Dict[str, Any]:
        """統合システムパフォーマンステスト"""
        
        print("\n--- テスト6: 統合システムパフォーマンス ---")
        
        # 大規模概念セットでのパフォーマンステスト
        large_concept_set = []
        
        # 多様な概念を大量生成
        for i in range(50):
            concept_type = np.random.choice(['perceptual', 'memorial', 'anticipatory', 'judgmental', 'valuational'])
            large_concept_set.append({
                'type': concept_type,
                'content': f'{concept_type} experience {i+1} with rich experiential content',
                'experiential_quality': np.random.uniform(0.3, 0.95),
                'coherence': np.random.uniform(0.4, 0.9),
                'temporal_depth': np.random.randint(1, 30),
                'timestamp': datetime.now().isoformat()
            })
        
        # パフォーマンス測定開始
        start_time = time.time()
        
        # 動的クラスタリング実行
        clustering_result = self.phi_calculator.perform_dynamic_experiential_clustering(large_concept_set)
        
        clustering_time = time.time() - start_time
        
        # 多スケール時間統合実行
        start_time = time.time()
        
        integration_result = await self.temporal_integrator.integrate_multi_scale_experiences(
            large_concept_set, time.time()
        )
        
        integration_time = time.time() - start_time
        
        # 結果の品質評価
        clusters = clustering_result['clusters']
        phenomenological_validity = clustering_result['phenomenological_validity']
        integration_quality = integration_result['integration_quality']
        
        # パフォーマンス指標
        total_time = clustering_time + integration_time
        concepts_per_second = len(large_concept_set) / total_time
        
        # 品質指標
        quality_maintained = phenomenological_validity > 0.6 and integration_quality > 0.5
        scalability_acceptable = total_time < 5.0  # 5秒以内
        
        test_result = {
            'test_name': 'integrated_system_performance',
            'success': quality_maintained and scalability_acceptable,
            'total_processing_time': total_time,
            'clustering_time': clustering_time,
            'integration_time': integration_time,
            'concepts_per_second': concepts_per_second,
            'concepts_processed': len(large_concept_set),
            'clusters_generated': len(clusters),
            'phenomenological_validity': phenomenological_validity,
            'integration_quality': integration_quality,
            'quality_maintained': quality_maintained,
            'scalability_acceptable': scalability_acceptable,
            'details': {
                'clustering_result_summary': {
                    'cluster_count': len(clusters),
                    'temporal_coherence': clustering_result['temporal_coherence'],
                    'boundary_flexibility': clustering_result['boundary_flexibility']
                },
                'integration_result_summary': {
                    'hierarchical_coherence': integration_result['hierarchical_integration']['hierarchical_coherence'],
                    'temporal_flow_coherence': integration_result['temporal_coherence']['temporal_flow_coherence']
                }
            }
        }
        
        print(f"  処理時間: {total_time:.3f}秒")
        print(f"  処理速度: {concepts_per_second:.1f} concepts/sec")
        print(f"  生成クラスター数: {len(clusters)}")
        print(f"  現象学的妥当性: {phenomenological_validity:.3f}")
        print(f"  統合品質: {integration_quality:.3f}")
        print(f"  品質維持: {'✅' if quality_maintained else '❌'}")
        print(f"  スケーラビリティ: {'✅' if scalability_acceptable else '❌'}")
        print(f"  テスト結果: {'✅ PASS' if test_result['success'] else '❌ FAIL'}")
        
        return test_result
    
    def _evaluate_overall_results(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """全体結果の評価"""
        
        print("\n=== 総合結果評価 ===")
        
        # 成功したテストの数
        passed_tests = [result for result in test_results if result['success']]
        total_tests = len(test_results)
        pass_rate = len(passed_tests) / total_tests
        
        # 各品質指標の平均
        avg_phenomenological_validity = np.mean([
            result.get('phenomenological_validity', 0) for result in test_results
        ])
        
        avg_temporal_coherence = np.mean([
            result.get('temporal_coherence', 0) for result in test_results
        ])
        
        avg_integration_quality = np.mean([
            result.get('integration_quality', 0) for result in test_results
        ])
        
        # 全体的成功判定
        overall_success = (
            pass_rate >= 0.8 and  # 80%以上のテストが成功
            avg_phenomenological_validity > 0.6 and
            avg_temporal_coherence > 0.5
        )
        
        overall_result = {
            'overall_success': overall_success,
            'pass_rate': pass_rate,
            'passed_tests': len(passed_tests),
            'total_tests': total_tests,
            'avg_phenomenological_validity': avg_phenomenological_validity,
            'avg_temporal_coherence': avg_temporal_coherence,
            'avg_integration_quality': avg_integration_quality,
            'individual_results': test_results,
            'summary': {
                'dynamic_clustering_functional': any(
                    result['test_name'] in ['intentional_act_clustering', 'experiential_quality_clustering'] and result['success']
                    for result in test_results
                ),
                'temporal_integration_functional': any(
                    result['test_name'] in ['temporal_coherence_organization', 'multi_scale_temporal_integration'] and result['success']
                    for result in test_results
                ),
                'phenomenological_validity_maintained': avg_phenomenological_validity > 0.6,
                'performance_acceptable': any(
                    result['test_name'] == 'integrated_system_performance' and result['success']
                    for result in test_results
                )
            }
        }
        
        print(f"  テスト成功率: {pass_rate:.1%} ({len(passed_tests)}/{total_tests})")
        print(f"  平均現象学的妥当性: {avg_phenomenological_validity:.3f}")
        print(f"  平均時間的一貫性: {avg_temporal_coherence:.3f}")
        print(f"  平均統合品質: {avg_integration_quality:.3f}")
        print(f"  動的クラスタリング: {'✅' if overall_result['summary']['dynamic_clustering_functional'] else '❌'}")
        print(f"  時間統合機能: {'✅' if overall_result['summary']['temporal_integration_functional'] else '❌'}")
        print(f"  現象学的妥当性: {'✅' if overall_result['summary']['phenomenological_validity_maintained'] else '❌'}")
        print(f"  パフォーマンス: {'✅' if overall_result['summary']['performance_acceptable'] else '❌'}")
        print(f"\n  総合評価: {'🎉 SUCCESS' if overall_success else '⚠️  NEEDS IMPROVEMENT'}")
        
        return overall_result


async def main():
    """メイン実行関数"""
    
    print("動的クラスタリング統合テスト開始")
    print("現象学的に健全な概念クラスタリングと多スケール時間統合の検証")
    
    # テストクラスのインスタンス化
    test_suite = DynamicClusteringIntegrationTest()
    
    try:
        # 包括的テストの実行
        results = await test_suite.run_comprehensive_tests()
        
        # 結果の保存（オプション）
        import json
        with open('/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/08_02_2025/dynamic_clustering_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n結果ファイル保存: dynamic_clustering_test_results.json")
        
        return results
        
    except Exception as e:
        print(f"テスト実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 非同期実行
    results = asyncio.run(main())