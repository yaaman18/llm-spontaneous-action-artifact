"""
IIT 4.0 Implementation Test Suite
Comprehensive testing of the IIT 4.0 core engine and intrinsic difference calculations

This test suite validates:
1. IIT 4.0 axiom compliance
2. Mathematical correctness of φ calculation
3. Intrinsic difference computation accuracy
4. Integration with NewbornAI 2.0 architecture

Author: IIT Integration Master
Date: 2025-08-03
Version: 1.0.0
"""

import pytest
import numpy as np
import logging
from typing import Dict, List, Any
import asyncio
import time
from dataclasses import dataclass

# Import our IIT 4.0 modules
from iit4_core_engine import (
    IIT4PhiCalculator, IntrinsicDifferenceCalculator, 
    PhiStructure, CauseEffectState, IIT4AxiomValidator
)
from intrinsic_difference import (
    DetailedIntrinsicDifferenceCalculator, IntrinsicDifferenceValidator,
    OptimalPurviewFinder, StateSpaceAnalyzer
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """テスト結果の構造化データ"""
    test_name: str
    passed: bool
    phi_value: float
    execution_time: float
    error_message: str = ""
    additional_metrics: Dict[str, float] = None


class IIT4BasicTests:
    """基本的なIIT 4.0機能テスト"""
    
    def __init__(self):
        self.phi_calculator = IIT4PhiCalculator(precision=1e-10)
        self.id_calculator = IntrinsicDifferenceCalculator(precision=1e-10)
        self.validator = IIT4AxiomValidator(self.phi_calculator)
    
    def test_simple_two_node_system(self) -> TestResult:
        """2ノードシステムの基本テスト"""
        start_time = time.time()
        
        try:
            # 2ノードシステムの構築
            system_state = np.array([1, 1])
            connectivity_matrix = np.array([
                [0, 1],
                [1, 0]
            ])
            
            # φ値計算
            phi_structure = self.phi_calculator.calculate_phi(
                system_state, connectivity_matrix
            )
            
            execution_time = time.time() - start_time
            
            # 基本検証
            passed = (
                phi_structure.total_phi > 0 and
                len(phi_structure.maximal_substrate) > 0 and
                len(phi_structure.distinctions) > 0
            )
            
            return TestResult(
                test_name="simple_two_node_system",
                passed=passed,
                phi_value=phi_structure.total_phi,
                execution_time=execution_time,
                additional_metrics={
                    'num_distinctions': len(phi_structure.distinctions),
                    'num_relations': len(phi_structure.relations),
                    'substrate_size': len(phi_structure.maximal_substrate)
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="simple_two_node_system",
                passed=False,
                phi_value=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def test_three_node_network(self) -> TestResult:
        """3ノードネットワークの複雑性テスト"""
        start_time = time.time()
        
        try:
            # 3ノードの相互接続ネットワーク
            system_state = np.array([1, 0, 1])
            connectivity_matrix = np.array([
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0]
            ])
            
            phi_structure = self.phi_calculator.calculate_phi(
                system_state, connectivity_matrix
            )
            
            execution_time = time.time() - start_time
            
            # 複雑性検証
            passed = (
                phi_structure.total_phi > 0 and
                phi_structure.phi_structure_complexity > 0 and
                len(phi_structure.distinctions) >= 2
            )
            
            return TestResult(
                test_name="three_node_network",
                passed=passed,
                phi_value=phi_structure.total_phi,
                execution_time=execution_time,
                additional_metrics={
                    'complexity': phi_structure.phi_structure_complexity,
                    'exclusion_definiteness': phi_structure.exclusion_definiteness,
                    'composition_richness': phi_structure.composition_richness
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="three_node_network",
                passed=False,
                phi_value=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def test_disconnected_system(self) -> TestResult:
        """非接続システムのφ値テスト（φ=0であるべき）"""
        start_time = time.time()
        
        try:
            # 非接続システム
            system_state = np.array([1, 1, 1])
            connectivity_matrix = np.zeros((3, 3))  # 接続なし
            
            phi_structure = self.phi_calculator.calculate_phi(
                system_state, connectivity_matrix
            )
            
            execution_time = time.time() - start_time
            
            # 非接続システムでは統合情報量が小さいべき
            passed = phi_structure.total_phi < 0.1
            
            return TestResult(
                test_name="disconnected_system",
                passed=passed,
                phi_value=phi_structure.total_phi,
                execution_time=execution_time
            )
            
        except Exception as e:
            return TestResult(
                test_name="disconnected_system",
                passed=False,
                phi_value=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class IIT4AxiomTests:
    """IIT 4.0公理準拠性テスト"""
    
    def __init__(self):
        self.phi_calculator = IIT4PhiCalculator()
        self.axiom_validator = IIT4AxiomValidator(self.phi_calculator)
    
    def test_all_axioms_compliance(self) -> TestResult:
        """全公理の準拠性テスト"""
        start_time = time.time()
        
        try:
            # テスト用システム
            system_state = np.array([1, 1, 0, 1])
            connectivity_matrix = np.array([
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0]
            ])
            
            phi_structure = self.phi_calculator.calculate_phi(
                system_state, connectivity_matrix
            )
            
            # 全公理の検証
            axiom_results = self.axiom_validator.validate_all_axioms(
                phi_structure, system_state
            )
            
            execution_time = time.time() - start_time
            
            # 全公理が満たされているかチェック
            all_passed = all(axiom_results.values())
            
            return TestResult(
                test_name="all_axioms_compliance",
                passed=all_passed,
                phi_value=phi_structure.total_phi,
                execution_time=execution_time,
                additional_metrics=axiom_results
            )
            
        except Exception as e:
            return TestResult(
                test_name="all_axioms_compliance",
                passed=False,
                phi_value=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def test_existence_axiom(self) -> TestResult:
        """公理0: 存在の個別テスト"""
        start_time = time.time()
        
        try:
            # 活動的システム
            active_system = np.array([1, 1, 1])
            connectivity = np.array([
                [0, 1, 1],
                [1, 0, 1], 
                [1, 1, 0]
            ])
            
            phi_structure = self.phi_calculator.calculate_phi(
                active_system, connectivity
            )
            
            # 存在検証
            exists = self.axiom_validator.validate_existence(phi_structure, active_system)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="existence_axiom",
                passed=exists,
                phi_value=phi_structure.total_phi,
                execution_time=execution_time
            )
            
        except Exception as e:
            return TestResult(
                test_name="existence_axiom",
                passed=False,
                phi_value=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class IntrinsicDifferenceTests:
    """内在的差異計算テスト"""
    
    def __init__(self):
        self.detailed_calculator = DetailedIntrinsicDifferenceCalculator()
        self.validator = IntrinsicDifferenceValidator()
    
    def test_id_calculation_accuracy(self) -> TestResult:
        """ID計算精度テスト"""
        start_time = time.time()
        
        try:
            # テストシステム
            mechanism = frozenset([0, 1])
            candidate_purviews = frozenset([0, 1, 2])
            
            # 簡単なTPM
            tpm = np.array([
                [0.1, 0.2, 0.3],  # 状態000
                [0.2, 0.3, 0.4],  # 状態001
                [0.3, 0.4, 0.5],  # 状態010
                [0.4, 0.5, 0.6],  # 状態011
                [0.5, 0.6, 0.7],  # 状態100
                [0.6, 0.7, 0.8],  # 状態101
                [0.7, 0.8, 0.9],  # 状態110
                [0.8, 0.9, 0.1],  # 状態111
            ])
            
            system_state = np.array([1, 0, 1])
            
            # 完全ID計算
            id_result = self.detailed_calculator.compute_full_intrinsic_difference(
                mechanism, candidate_purviews, tpm, system_state
            )
            
            # 結果検証
            validation_results = self.validator.validate_id_computation(id_result)
            
            execution_time = time.time() - start_time
            
            # 全検証項目が通過しているかチェック
            all_valid = all(validation_results.values())
            
            return TestResult(
                test_name="id_calculation_accuracy",
                passed=all_valid,
                phi_value=id_result['phi_value'],
                execution_time=execution_time,
                additional_metrics={
                    'total_id': id_result['total_id'],
                    'cause_id': id_result['cause_id'],
                    'effect_id': id_result['effect_id'],
                    **validation_results
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="id_calculation_accuracy", 
                passed=False,
                phi_value=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def test_optimal_purview_finding(self) -> TestResult:
        """最適範囲発見テスト"""
        start_time = time.time()
        
        try:
            purview_finder = OptimalPurviewFinder(max_purview_size=3)
            
            mechanism = frozenset([0])
            candidate_nodes = frozenset([0, 1, 2])
            
            # 単純なTPM
            tpm = np.array([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.1],
                [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7],
                [0.8, 0.1, 0.2],
                [0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8],
            ])
            
            system_state = np.array([1, 1, 0])
            
            # 最適範囲発見
            optimal_purview, max_id = purview_finder.find_optimal_purview(
                mechanism, candidate_nodes, tpm, system_state, 'cause'
            )
            
            execution_time = time.time() - start_time
            
            # 最適範囲が見つかり、ID値が正であることを確認
            passed = len(optimal_purview) > 0 and max_id > 0
            
            return TestResult(
                test_name="optimal_purview_finding",
                passed=passed,
                phi_value=max_id,
                execution_time=execution_time,
                additional_metrics={
                    'purview_size': len(optimal_purview),
                    'max_id_value': max_id
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="optimal_purview_finding",
                passed=False,
                phi_value=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class NewbornAIIntegrationTests:
    """NewbornAI 2.0統合テスト"""
    
    def __init__(self):
        self.phi_calculator = IIT4PhiCalculator()
    
    def test_experiential_concept_integration(self) -> TestResult:
        """体験概念との統合テスト"""
        start_time = time.time()
        
        try:
            # 体験概念をシミュレート
            experiential_concepts = [
                {
                    'content': '美しい朝日を体験した',
                    'phi_contribution': 0.3,
                    'temporal_position': 1,
                    'emotional_valence': 0.8
                },
                {
                    'content': '新しい音楽に感動した',
                    'phi_contribution': 0.4,
                    'temporal_position': 2,
                    'emotional_valence': 0.9
                },
                {
                    'content': '友人との深い対話を体験',
                    'phi_contribution': 0.5,
                    'temporal_position': 3,
                    'emotional_valence': 0.7
                }
            ]
            
            # 体験概念からシステム状態を構築
            system_state, connectivity_matrix = self._build_system_from_concepts(
                experiential_concepts
            )
            
            # φ値計算
            phi_structure = self.phi_calculator.calculate_phi(
                system_state, connectivity_matrix
            )
            
            execution_time = time.time() - start_time
            
            # 統合検証: 体験概念数とφ構造の一貫性
            passed = (
                phi_structure.total_phi > 0 and
                len(phi_structure.distinctions) >= len(experiential_concepts) // 2
            )
            
            return TestResult(
                test_name="experiential_concept_integration",
                passed=passed,
                phi_value=phi_structure.total_phi,
                execution_time=execution_time,
                additional_metrics={
                    'concept_count': len(experiential_concepts),
                    'distinction_count': len(phi_structure.distinctions),
                    'integration_ratio': len(phi_structure.distinctions) / len(experiential_concepts)
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="experiential_concept_integration",
                passed=False,
                phi_value=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _build_system_from_concepts(self, concepts: List[Dict]) -> tuple:
        """体験概念からシステム状態と接続行列を構築"""
        n_concepts = len(concepts)
        
        # システム状態: 各概念の活性度
        system_state = np.array([
            concept['phi_contribution'] for concept in concepts
        ])
        
        # 接続行列: 時間的・感情的関連性
        connectivity_matrix = np.zeros((n_concepts, n_concepts))
        
        for i, concept_a in enumerate(concepts):
            for j, concept_b in enumerate(concepts):
                if i != j:
                    # 時間的近接性
                    temporal_diff = abs(
                        concept_a['temporal_position'] - concept_b['temporal_position']
                    )
                    temporal_strength = max(0, 1.0 - temporal_diff * 0.3)
                    
                    # 感情的類似性
                    emotional_similarity = 1.0 - abs(
                        concept_a['emotional_valence'] - concept_b['emotional_valence']
                    )
                    
                    # 統合強度
                    connection_strength = (temporal_strength + emotional_similarity) / 2.0
                    connectivity_matrix[i, j] = connection_strength
        
        return system_state, connectivity_matrix


class PerformanceTests:
    """パフォーマンステスト"""
    
    def __init__(self):
        self.phi_calculator = IIT4PhiCalculator()
    
    def test_scalability_performance(self) -> TestResult:
        """スケーラビリティパフォーマンステスト"""
        start_time = time.time()
        
        try:
            # 中規模システム（6ノード）
            n_nodes = 6
            system_state = np.random.choice([0, 1], size=n_nodes)
            connectivity_matrix = np.random.rand(n_nodes, n_nodes)
            connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2  # 対称化
            
            # φ値計算
            phi_structure = self.phi_calculator.calculate_phi(
                system_state, connectivity_matrix
            )
            
            execution_time = time.time() - start_time
            
            # パフォーマンス基準: 6ノードで3秒以内
            performance_acceptable = execution_time < 3.0
            
            # 結果の妥当性
            result_valid = phi_structure.total_phi >= 0
            
            passed = performance_acceptable and result_valid
            
            return TestResult(
                test_name="scalability_performance",
                passed=passed,
                phi_value=phi_structure.total_phi,
                execution_time=execution_time,
                additional_metrics={
                    'nodes_count': n_nodes,
                    'performance_acceptable': performance_acceptable,
                    'result_valid': result_valid
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="scalability_performance",
                passed=False,
                phi_value=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class ComprehensiveTestSuite:
    """包括的テストスイート"""
    
    def __init__(self):
        self.basic_tests = IIT4BasicTests()
        self.axiom_tests = IIT4AxiomTests()
        self.id_tests = IntrinsicDifferenceTests()
        self.integration_tests = NewbornAIIntegrationTests()
        self.performance_tests = PerformanceTests()
    
    def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """全テストの実行"""
        results = {
            'basic_tests': [],
            'axiom_tests': [],
            'id_tests': [],
            'integration_tests': [],
            'performance_tests': []
        }
        
        print("🧠 IIT 4.0 Implementation Test Suite 開始")
        print("=" * 60)
        
        # 基本テスト
        print("\n📊 基本機能テスト")
        results['basic_tests'].append(self.basic_tests.test_simple_two_node_system())
        results['basic_tests'].append(self.basic_tests.test_three_node_network())
        results['basic_tests'].append(self.basic_tests.test_disconnected_system())
        
        # 公理テスト
        print("\n📜 公理準拠性テスト")
        results['axiom_tests'].append(self.axiom_tests.test_all_axioms_compliance())
        results['axiom_tests'].append(self.axiom_tests.test_existence_axiom())
        
        # ID計算テスト
        print("\n🔢 内在的差異計算テスト")
        results['id_tests'].append(self.id_tests.test_id_calculation_accuracy())
        results['id_tests'].append(self.id_tests.test_optimal_purview_finding())
        
        # 統合テスト
        print("\n🔗 NewbornAI 2.0統合テスト")
        results['integration_tests'].append(
            self.integration_tests.test_experiential_concept_integration()
        )
        
        # パフォーマンステスト
        print("\n⚡ パフォーマンステスト")
        results['performance_tests'].append(
            self.performance_tests.test_scalability_performance()
        )
        
        return results
    
    def print_test_summary(self, results: Dict[str, List[TestResult]]):
        """テスト結果サマリーの出力"""
        print("\n" + "=" * 60)
        print("🎯 テスト結果サマリー")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        total_execution_time = 0.0
        
        for category, test_list in results.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            
            for test_result in test_list:
                total_tests += 1
                if test_result.passed:
                    passed_tests += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                
                total_execution_time += test_result.execution_time
                
                print(f"  {status} {test_result.test_name}")
                print(f"      φ値: {test_result.phi_value:.6f}")
                print(f"      実行時間: {test_result.execution_time:.3f}秒")
                
                if test_result.error_message:
                    print(f"      エラー: {test_result.error_message}")
                
                if test_result.additional_metrics:
                    for key, value in test_result.additional_metrics.items():
                        print(f"      {key}: {value}")
        
        print(f"\n📈 総合結果:")
        print(f"   成功率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"   総実行時間: {total_execution_time:.3f}秒")
        
        return passed_tests / total_tests


# メイン実行部分
async def main():
    """テストスイートのメイン実行"""
    import time
    
    test_suite = ComprehensiveTestSuite()
    
    print("🔬 IIT 4.0 for NewbornAI 2.0 - 実装検証テスト")
    print("Tononi et al. (2023) 理論準拠性検証")
    
    # 全テスト実行
    results = test_suite.run_all_tests()
    
    # 結果サマリー
    success_rate = test_suite.print_test_summary(results)
    
    if success_rate >= 0.8:
        print("\n🎉 IIT 4.0実装は理論的に健全です！")
    else:
        print(f"\n⚠️  改善が必要です（成功率: {success_rate*100:.1f}%）")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())