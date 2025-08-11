#!/usr/bin/env python3
"""
エナクティブ意識フレームワーク V3.0 - システムチェック

全コンポーネントの動作確認と依存関係チェック
"""

import sys
import importlib
import numpy as np
from typing import Dict, List, Tuple


def check_dependencies() -> Dict[str, bool]:
    """依存パッケージの確認"""
    dependencies = {
        'numpy': False,
        'jax': False,
        'jaxlib': False,
        'optax': False,
        'equinox': False,
        'scipy': False,
        'pytest': False,
    }
    
    print("【依存パッケージチェック】")
    print("-" * 40)
    
    for package in dependencies:
        try:
            importlib.import_module(package)
            dependencies[package] = True
            print(f"✅ {package}: インストール済み")
        except ImportError:
            print(f"❌ {package}: 未インストール")
    
    return dependencies


def check_core_components() -> Dict[str, Tuple[bool, str]]:
    """コアコンポーネントの確認"""
    components = {}
    
    print("\n【コアコンポーネントチェック】")
    print("-" * 40)
    
    # 1. Domain層
    try:
        from domain.entities.predictive_coding_core import PredictiveCodingCore
        components['PredictiveCodingCore'] = (True, "正常")
        print("✅ PredictiveCodingCore: 正常")
    except Exception as e:
        components['PredictiveCodingCore'] = (False, str(e))
        print(f"❌ PredictiveCodingCore: {e}")
    
    try:
        from domain.entities.self_organizing_map import SelfOrganizingMap
        components['SelfOrganizingMap'] = (True, "正常")
        print("✅ SelfOrganizingMap: 正常")
    except Exception as e:
        components['SelfOrganizingMap'] = (False, str(e))
        print(f"❌ SelfOrganizingMap: {e}")
    
    try:
        from domain.value_objects.consciousness_state import ConsciousnessState
        components['ConsciousnessState'] = (True, "正常")
        print("✅ ConsciousnessState: 正常")
    except Exception as e:
        components['ConsciousnessState'] = (False, str(e))
        print(f"❌ ConsciousnessState: {e}")
    
    try:
        from domain.value_objects.phi_value import PhiValue
        components['PhiValue'] = (True, "正常")
        print("✅ PhiValue: 正常")
    except Exception as e:
        components['PhiValue'] = (False, str(e))
        print(f"❌ PhiValue: {e}")
    
    # 2. Application層（Factory パターン）
    try:
        from domain.factories.consciousness_factory import ConsciousnessFactory
        components['ConsciousnessFactory'] = (True, "正常")
        print("✅ ConsciousnessFactory: 正常")
    except Exception as e:
        components['ConsciousnessFactory'] = (False, str(e))
        print(f"❌ ConsciousnessFactory: {e}")
    
    try:
        from domain.aggregates.consciousness_aggregate import ConsciousnessAggregate
        components['ConsciousnessAggregate'] = (True, "正常")
        print("✅ ConsciousnessAggregate: 正常")
    except Exception as e:
        components['ConsciousnessAggregate'] = (False, str(e))
        print(f"❌ ConsciousnessAggregate: {e}")
    
    # 3. Infrastructure層
    try:
        from infrastructure.jax_predictive_coding_core import JaxPredictiveCodingCore
        components['JaxPredictiveCodingCore'] = (True, "正常")
        print("✅ JaxPredictiveCodingCore: 正常")
    except Exception as e:
        components['JaxPredictiveCodingCore'] = (False, str(e))
        print(f"❌ JaxPredictiveCodingCore: {e}")
    
    try:
        from infrastructure.basic_som import BasicSOM
        components['BasicSOM'] = (True, "正常")
        print("✅ BasicSOM: 正常")
    except Exception as e:
        components['BasicSOM'] = (False, str(e))
        print(f"❌ BasicSOM: {e}")
    
    try:
        from domain.policies.consciousness_policies import ConsciousnessEmergencePolicy
        components['ConsciousnessEmergencePolicy'] = (True, "正常")
        print("✅ ConsciousnessEmergencePolicy: 正常")
    except Exception as e:
        components['ConsciousnessEmergencePolicy'] = (False, str(e))
        print(f"❌ ConsciousnessEmergencePolicy: {e}")
    
    # 4. Adapters層
    try:
        from ngc_learn_adapter import HybridPredictiveCodingAdapter
        components['HybridPredictiveCodingAdapter'] = (True, "正常")
        print("✅ HybridPredictiveCodingAdapter: 正常")
    except Exception as e:
        components['HybridPredictiveCodingAdapter'] = (False, str(e))
        print(f"❌ HybridPredictiveCodingAdapter: {e}")
    
    return components


def check_ngc_learn_availability() -> Tuple[bool, str]:
    """NGC-Learn利用可能性チェック"""
    print("\n【NGC-Learn統合チェック】")
    print("-" * 40)
    
    try:
        import ngclearn
        print("✅ NGC-Learn: インストール済み")
        return True, "NGC-Learn利用可能"
    except ImportError:
        print("⚠️ NGC-Learn: 未インストール（JAXフォールバック使用）")
        return False, "JAXフォールバックモード"


def test_basic_functionality() -> bool:
    """基本機能テスト"""
    print("\n【基本機能テスト】")
    print("-" * 40)
    
    try:
        from ngc_learn_adapter import HybridPredictiveCodingAdapter
        from domain.value_objects.precision_weights import PrecisionWeights
        
        # システム初期化
        adapter = HybridPredictiveCodingAdapter(3, 10)
        print(f"✅ アダプター初期化: {adapter.engine_type}エンジン")
        
        # 精度重み（numpy配列として正しく初期化）
        weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
        print("✅ 精度重み初期化: 成功")
        
        # テスト入力
        input_data = np.random.rand(10)
        
        # 処理実行
        state = adapter.process_input(input_data, weights)
        print(f"✅ 処理実行: エラー={state.total_error:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本機能テスト失敗: {e}")
        return False


def check_memory_usage():
    """メモリ使用状況確認（簡易）"""
    print("\n【メモリ使用状況】")
    print("-" * 40)
    
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"RSS (Resident Set Size): {memory_info.rss / 1024 / 1024:.2f} MB")
        print(f"VMS (Virtual Memory Size): {memory_info.vms / 1024 / 1024:.2f} MB")
    except ImportError:
        print("⚠️ psutilが未インストールのため、メモリ情報取得不可")
    except Exception as e:
        print(f"⚠️ メモリ情報取得エラー: {e}")


def display_system_info():
    """システム情報表示"""
    print("\n【システム情報】")
    print("-" * 40)
    
    import platform
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    try:
        import jax
        print(f"JAX: {jax.__version__}")
        print(f"JAX Backend: {jax.default_backend()}")
    except:
        print("JAX: 情報取得不可")


def main():
    """メインチェック実行"""
    print("=" * 60)
    print("  エナクティブ意識フレームワーク V3.0")
    print("  システム診断ツール")
    print("=" * 60)
    
    all_checks_passed = True
    
    # 1. 依存関係チェック
    deps = check_dependencies()
    if not all(deps.values()):
        all_checks_passed = False
    
    # 2. コアコンポーネントチェック
    components = check_core_components()
    component_status = all(status for status, _ in components.values())
    if not component_status:
        all_checks_passed = False
    
    # 3. NGC-Learn確認
    ngc_available, ngc_status = check_ngc_learn_availability()
    
    # 4. 基本機能テスト
    basic_test_passed = test_basic_functionality()
    if not basic_test_passed:
        all_checks_passed = False
    
    # 5. メモリ使用状況
    check_memory_usage()
    
    # 6. システム情報
    display_system_info()
    
    # 最終サマリー
    print("\n" + "=" * 60)
    print("【診断結果サマリー】")
    print("-" * 40)
    
    if all_checks_passed:
        print("✅ システム状態: 正常")
        print("✅ 全コンポーネント動作確認済み")
        print(f"✅ 動作モード: {ngc_status}")
        print("\n🧠 システムは研究開発での使用準備が整っています")
    else:
        print("⚠️ システム状態: 一部問題あり")
        print("   上記のエラーメッセージを確認してください")
        print(f"⚠️ 動作モード: {ngc_status}")
    
    print("=" * 60)
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    exit(main())