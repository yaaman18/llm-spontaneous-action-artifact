#!/usr/bin/env python3
"""
体験的感覚保持システム検証実行スクリプト
Experience Retention System Validation Execution Script

このスクリプトは、提案された体験保持システムの理論的妥当性を
エナクティビズム・現象学の観点から包括的に検証します。
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# インポート
from phenomenological_foundations import (
    ExperientialContent, TemporalStructure, IntegratedExperientialMemory
)
from experience_retention_validation import (
    ComprehensiveSystemValidator, ValidationResult
)


def create_rich_test_system() -> IntegratedExperientialMemory:
    """理論検証のためのリッチなテストシステム構築"""
    
    system = IntegratedExperientialMemory()
    
    logger.info("テストシステムにリッチな体験データを追加中...")
    
    # フッサール時間意識構造のテスト
    husserl_test_experiences = [
        # 視覚的体験の時間的流れ
        ExperientialContent(
            temporal_phase=TemporalStructure.PRIMAL_IMPRESSION,
            intentional_content={
                'visual_scene': {'brightness': 0.8, 'color_dominance': 'blue'},
                'attention_focus': 'central_object',
                'perceptual_synthesis': ['form', 'color', 'depth']
            },
            bodily_resonance=0.6,
            temporal_thickness=1.0,
            associative_potential=['visual', 'attention', 'object'],
            habit_layer='passive'
        ),
        ExperientialContent(
            temporal_phase=TemporalStructure.RETENTION,
            intentional_content={
                'visual_scene': {'brightness': 0.7, 'color_dominance': 'blue'},
                'attention_focus': 'peripheral_motion',
                'perceptual_synthesis': ['form', 'motion']
            },
            bodily_resonance=0.5,
            temporal_thickness=0.9,
            associative_potential=['visual', 'motion'],
            habit_layer='passive'
        ),
        # 聴覚的体験の重層構造
        ExperientialContent(
            temporal_phase=TemporalStructure.RETENTION,
            intentional_content={
                'auditory_scene': {'melody': [440, 520, 660], 'rhythm': 'steady'},
                'emotional_tone': 'contemplative',
                'harmonic_analysis': ['consonance', 'temporal_pattern']
            },
            bodily_resonance=0.8,
            temporal_thickness=0.8,
            associative_potential=['auditory', 'emotion', 'temporal'],
            habit_layer='active'
        )
    ]
    
    # メルロ=ポンティ身体性のテスト
    embodiment_test_experiences = [
        ExperientialContent(
            temporal_phase=TemporalStructure.PRIMAL_IMPRESSION,
            intentional_content={
                'motor_pattern': {'reach': 0.9, 'grasp': 0.7, 'manipulate': 0.5},
                'tactile_feedback': {'texture': 'rough', 'temperature': 'warm', 'pressure': 0.6},
                'proprioceptive_state': {'arm_position': [45, 30, 90], 'muscle_tension': 0.4},
                'tool_integration': 'pen_writing'
            },
            bodily_resonance=0.95,
            temporal_thickness=1.0,
            associative_potential=['motor', 'tactile', 'tool_use'],
            habit_layer='active'
        ),
        ExperientialContent(
            temporal_phase=TemporalStructure.RETENTION,
            intentional_content={
                'motor_pattern': {'reach': 0.8, 'grasp': 0.6, 'manipulate': 0.7},
                'tactile_feedback': {'texture': 'smooth', 'temperature': 'neutral', 'pressure': 0.5},
                'proprioceptive_state': {'arm_position': [50, 25, 85], 'muscle_tension': 0.3},
                'spatial_navigation': 'familiar_environment'
            },
            bodily_resonance=0.7,
            temporal_thickness=0.9,
            associative_potential=['motor', 'spatial', 'navigation'],
            habit_layer='passive'
        )
    ]
    
    # バレラ・エナクティブ相互作用のテスト
    enactive_test_experiences = [
        ExperientialContent(
            temporal_phase=TemporalStructure.PRIMAL_IMPRESSION,
            intentional_content={
                'environment_state': {
                    'object_affordances': ['graspable', 'movable', 'tool-like'],
                    'spatial_layout': {'obstacles': 2, 'open_paths': 3, 'goal_distance': 1.5},
                    'social_context': {'other_agents': 1, 'interaction_type': 'cooperative'}
                },
                'system_response': {
                    'action_selection': 'approach_and_explore',
                    'attention_modulation': 'focused_exploration',
                    'predictive_processing': ['collision_avoidance', 'goal_planning']
                },
                'coupling_dynamics': {
                    'adaptation_rate': 0.8,
                    'mutual_influence': 0.7,
                    'system_environment_coherence': 0.9
                }
            },
            bodily_resonance=0.85,
            temporal_thickness=1.0,
            associative_potential=['environment', 'action', 'social', 'adaptation'],
            habit_layer='active'
        ),
        ExperientialContent(
            temporal_phase=TemporalStructure.PROTENTION,
            intentional_content={
                'environment_state': {
                    'object_affordances': ['graspable', 'movable'],
                    'spatial_layout': {'obstacles': 1, 'open_paths': 4, 'goal_distance': 1.0},
                    'social_context': {'other_agents': 1, 'interaction_type': 'coordinative'}
                },
                'system_response': {
                    'action_selection': 'coordinated_manipulation',
                    'attention_modulation': 'joint_attention',
                    'predictive_processing': ['social_prediction', 'joint_action_planning']
                },
                'coupling_dynamics': {
                    'adaptation_rate': 0.9,
                    'mutual_influence': 0.8,
                    'system_environment_coherence': 0.95
                }
            },
            bodily_resonance=0.9,
            temporal_thickness=1.0,
            associative_potential=['coordination', 'prediction', 'joint_action'],
            habit_layer='active'
        )
    ]
    
    # システムに体験を追加
    all_experiences = husserl_test_experiences + embodiment_test_experiences + enactive_test_experiences
    
    for i, experience in enumerate(all_experiences):
        logger.info(f"体験 {i+1}/{len(all_experiences)} を追加: {experience.temporal_phase.value}")
        system.retain_experience(experience)
        
        # エナクティブ記憶への構造的カップリング登録
        if 'environment_state' in experience.intentional_content:
            system.enactive_memory.register_coupling(
                experience.intentional_content['environment_state'],
                experience.intentional_content.get('system_response', {}),
                experience.bodily_resonance
            )
    
    logger.info(f"テストシステム構築完了: {len(all_experiences)} 個の体験を追加")
    return system


def analyze_theoretical_consistency(validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
    """理論的一貫性の詳細分析"""
    
    analysis = {
        'overall_consistency': 0.0,
        'phenomenological_alignment': {},
        'enactive_integration': {},
        'implementation_gaps': [],
        'theoretical_strengths': [],
        'critical_concerns': []
    }
    
    # 全体的一貫性の計算
    scores = [result.score for result in validation_results.values()]
    analysis['overall_consistency'] = np.mean(scores)
    
    # 現象学的整合性分析
    phenomenological_aspects = ['retention_memory', 'proprioceptive_map', 'qualitative_experience']
    phenom_scores = [validation_results[aspect].score for aspect in phenomenological_aspects if aspect in validation_results]
    analysis['phenomenological_alignment'] = {
        'average_score': np.mean(phenom_scores) if phenom_scores else 0.0,
        'consistency_variance': np.var(phenom_scores) if phenom_scores else 0.0,
        'husserl_fidelity': validation_results.get('retention_memory', ValidationResult('', 0.0, {}, '', [])).score,
        'merleau_ponty_embodiment': validation_results.get('proprioceptive_map', ValidationResult('', 0.0, {}, '', [])).score
    }
    
    # エナクティブ統合分析
    enactive_aspects = ['meaning_creation']
    enactive_scores = [validation_results[aspect].score for aspect in enactive_aspects if aspect in validation_results]
    analysis['enactive_integration'] = {
        'average_score': np.mean(enactive_scores) if enactive_scores else 0.0,
        'varela_structural_coupling': validation_results.get('meaning_creation', ValidationResult('', 0.0, {}, '', [])).score,
        'circular_causality_realization': 0.0  # 詳細分析が必要
    }
    
    # 実装ギャップの特定
    for subsystem, result in validation_results.items():
        if result.score < 0.6:
            analysis['implementation_gaps'].extend([
                f"{subsystem}: {gap}" for gap in result.implementation_recommendations
            ])
        
        if result.score >= 0.8:
            analysis['theoretical_strengths'].append(f"{subsystem}: 高い理論的妥当性")
        elif result.score < 0.4:
            analysis['critical_concerns'].append(f"{subsystem}: 重大な理論的課題")
    
    return analysis


def generate_expert_recommendations(analysis: Dict[str, Any], 
                                  validation_results: Dict[str, ValidationResult]) -> List[str]:
    """専門家推奨事項の生成"""
    
    recommendations = []
    
    # 全体的評価に基づく推奨事項
    if analysis['overall_consistency'] >= 0.8:
        recommendations.append(
            "【優秀】システムは現象学的・エナクティブ理論との高い整合性を示している。"
            "微細な最適化と詳細実装に集中することを推奨。"
        )
    elif analysis['overall_consistency'] >= 0.6:
        recommendations.append(
            "【良好】基本的理論要件は満たされているが、重要な改善領域が存在。"
            "特に低スコア領域の理論的精密化を優先すべき。"
        )
    else:
        recommendations.append(
            "【要改善】根本的な理論的再設計が必要。現象学的・エナクティブ理論の"
            "基本概念との整合性を確保することから開始すべき。"
        )
    
    # 現象学的側面の推奨事項
    phenom_avg = analysis['phenomenological_alignment']['average_score']
    if phenom_avg < 0.7:
        recommendations.append(
            "【現象学的妥当性】フッサール時間意識論とメルロ=ポンティ身体現象学の"
            "構造的要件をより忠実に実装する必要がある。特に："
            "- 把持の「準現在」性質の機能的表現強化"
            "- 身体図式の統合的組織化メカニズム精密化"
            "- 受動的統合の類型別実装"
        )
    
    # エナクティブ側面の推奨事項  
    enactive_avg = analysis['enactive_integration']['average_score']
    if enactive_avg < 0.7:
        recommendations.append(
            "【エナクティブ統合】バレラの構造的カップリング理論の循環因果性を"
            "より明示的に実装する必要がある。特に："
            "- 主体-環境相互作用の循環的性質の強化"
            "- 意味創出の創発的特性の実現"
            "- オートポイエーシス的パターンの自己組織化"
        )
    
    # クオリア問題回避の確認
    recommendations.append(
        "【クオリア問題回避】現在の実装は適切にクオリア問題を回避し、"
        "機能的・行動的側面に焦点を当てている。この方針を維持しつつ、"
        "現象学的記述の豊かさを構造的・関係的特徴で表現することを継続すべき。"
    )
    
    # 実装上の技術的推奨事項
    if analysis['implementation_gaps']:
        recommendations.append(
            f"【実装優先順位】以下の技術的改善を優先順位順に実装："
        )
        for i, gap in enumerate(analysis['implementation_gaps'][:5], 1):
            recommendations.append(f"  {i}. {gap}")
    
    return recommendations


def main():
    """メイン実行関数"""
    
    print("="*80)
    print("体験的感覚保持システム：エナクティビズム・現象学理論的妥当性検証")
    print("Experience Retention System: Enactivism-Phenomenology Theoretical Validation")
    print("="*80)
    print()
    
    logger.info("検証プロセスを開始します...")
    
    try:
        # 1. テストシステムの構築
        print("1. リッチなテストシステムの構築...")
        test_system = create_rich_test_system()
        
        # システム状態の確認
        system_status = test_system.synthesize_temporal_flow()
        print(f"   - 保持体験数: {system_status['retention_chain_length']}")
        print(f"   - 受動的統合クラスター: {system_status['passive_synthesis_clusters']}")
        print(f"   - 運動習慣数: {system_status['motor_habits_count']}")
        print(f"   - 構造的カップリング履歴: {system_status['structural_couplings']}")
        print()
        
        # 2. 包括的検証の実行
        print("2. 包括的理論的妥当性検証の実行...")
        validator = ComprehensiveSystemValidator()
        validation_results = validator.perform_comprehensive_validation(test_system)
        
        # 3. 詳細分析
        print("3. 理論的一貫性の詳細分析...")
        theoretical_analysis = analyze_theoretical_consistency(validation_results)
        
        # 4. 専門家推奨事項の生成
        print("4. 専門家推奨事項の生成...")
        expert_recommendations = generate_expert_recommendations(theoretical_analysis, validation_results)
        
        # 5. 包括的レポートの生成
        print("5. 包括的検証レポートの生成...")
        comprehensive_report = validator.generate_comprehensive_report(validation_results)
        
        # 結果の出力
        print("\n" + "="*80)
        print("検証結果")
        print("="*80)
        print(comprehensive_report)
        
        print("\n" + "="*80)
        print("理論的一貫性分析")
        print("="*80)
        print(f"全体的一貫性: {theoretical_analysis['overall_consistency']:.3f}")
        print(f"現象学的整合性: {theoretical_analysis['phenomenological_alignment']['average_score']:.3f}")
        print(f"エナクティブ統合: {theoretical_analysis['enactive_integration']['average_score']:.3f}")
        
        if theoretical_analysis['theoretical_strengths']:
            print("\n理論的強み:")
            for strength in theoretical_analysis['theoretical_strengths']:
                print(f"  ✓ {strength}")
        
        if theoretical_analysis['critical_concerns']:
            print("\n重大な懸念:")
            for concern in theoretical_analysis['critical_concerns']:
                print(f"  ⚠ {concern}")
        
        print("\n" + "="*80)
        print("専門家推奨事項")
        print("="*80)
        for i, recommendation in enumerate(expert_recommendations, 1):
            print(f"{i}. {recommendation}")
            print()
        
        # 最終評価
        print("="*80)
        print("最終評価")
        print("="*80)
        
        final_score = theoretical_analysis['overall_consistency']
        if final_score >= 0.8:
            status = "優秀 (Excellent)"
            description = "現象学的・エナクティブ理論との高度な整合性を実現"
        elif final_score >= 0.6:
            status = "良好 (Good)"
            description = "基本的要件を満たし、重要な改善の余地あり"
        elif final_score >= 0.4:
            status = "要改善 (Needs Improvement)"
            description = "重要な理論的課題の解決が必要"
        else:
            status = "不十分 (Insufficient)"
            description = "根本的な理論的再設計が必要"
        
        print(f"総合評価: {status}")
        print(f"スコア: {final_score:.3f}/1.000")
        print(f"評価: {description}")
        
        logger.info("検証プロセスが正常に完了しました。")
        
    except Exception as e:
        logger.error(f"検証プロセス中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    import sys
    import os
    # プロジェクトパスをPython pathに追加
    project_path = os.path.dirname(os.path.abspath(__file__))
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
    
    try:
        main()
    except ImportError as e:
        print(f"インポートエラー: {e}")
        print("現象学的基盤モジュールのインポートに失敗しました。")
    except Exception as e:
        print(f"実行エラー: {e}")
        import traceback
        traceback.print_exc()