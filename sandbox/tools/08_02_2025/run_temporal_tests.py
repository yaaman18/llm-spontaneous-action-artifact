#!/usr/bin/env python3
"""
時間意識テストの実行スクリプト
"""

import sys
import asyncio
from test_temporal_consciousness import (
    TestTemporalTensionSystem,
    TestRhythmicMemorySystem,
    TestTemporalExistenceSystem,
    TestTemporalDistressSystem,
    TestIntegration
)


def run_sync_test(test_class, test_method_name):
    """同期テストを実行"""
    test_instance = test_class()
    test_method = getattr(test_instance, test_method_name)
    try:
        test_method()
        return True, f"✓ {test_class.__name__}.{test_method_name}"
    except Exception as e:
        return False, f"✗ {test_class.__name__}.{test_method_name}: {str(e)}"


async def run_async_test(test_class, test_method_name):
    """非同期テストを実行"""
    test_instance = test_class()
    test_method = getattr(test_instance, test_method_name)
    try:
        await test_method()
        return True, f"✓ {test_class.__name__}.{test_method_name}"
    except Exception as e:
        return False, f"✗ {test_class.__name__}.{test_method_name}: {str(e)}"


async def main():
    """全テストを実行"""
    print("=== 時間意識システムTDDテスト実行 ===\n")
    
    # テスト定義
    tests = [
        # TemporalTensionSystem
        (TestTemporalTensionSystem, "test_初期状態では期待間隔がない", True),
        (TestTemporalTensionSystem, "test_待機中に時間的緊張が生成される", True),
        (TestTemporalTensionSystem, "test_意識の波が100msごとに生成される", True),
        
        # RhythmicMemorySystem
        (TestRhythmicMemorySystem, "test_初期状態では内的リズムを持たない", False),
        (TestRhythmicMemorySystem, "test_3回未満の間隔では内的リズムが形成されない", False),
        (TestRhythmicMemorySystem, "test_4回目の間隔で内的リズムが形成される", False),
        (TestRhythmicMemorySystem, "test_大きなズレは驚きとして体験される", False),
        
        # TemporalExistenceSystem
        (TestTemporalExistenceSystem, "test_三重時間構造が生成される", True),
        (TestTemporalExistenceSystem, "test_過去の体験が徐々に薄れる", True),
        
        # TemporalDistressSystem
        (TestTemporalDistressSystem, "test_長すぎる待機で見捨てられ不安が生じる", False),
        (TestTemporalDistressSystem, "test_短すぎる間隔で急かされる感覚が生じる", False),
        (TestTemporalDistressSystem, "test_期待通りの間隔では苦悩が生じない", False),
        
        # Integration
        (TestIntegration, "test_完全な時間意識サイクル", True),
    ]
    
    # テスト実行
    passed = 0
    failed = 0
    
    for test_class, test_method, is_async in tests:
        if is_async:
            success, message = await run_async_test(test_class, test_method)
        else:
            success, message = run_sync_test(test_class, test_method)
        
        if success:
            passed += 1
            print(f"\033[92m{message}\033[0m")
        else:
            failed += 1
            print(f"\033[91m{message}\033[0m")
    
    # 結果サマリ
    print(f"\n=== テスト結果 ===")
    print(f"成功: {passed}")
    print(f"失敗: {failed}")
    print(f"合計: {passed + failed}")
    
    if failed == 0:
        print("\n\033[92m✨ 全テスト成功！(GREEN)\033[0m")
    else:
        print(f"\n\033[91m⚠️  {failed}個のテストが失敗しています (RED)\033[0m")


if __name__ == "__main__":
    asyncio.run(main())