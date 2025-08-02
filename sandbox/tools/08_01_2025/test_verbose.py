#!/usr/bin/env python3
"""
新生AI自律システムのverbose機能テスト

短時間でverbose出力をテストします。
"""

import asyncio
from newborn_ai import NewbornAI

async def test_verbose_ai():
    """Verbose機能のテスト"""
    print("🧪 Verbose機能テスト開始")
    print("=" * 60)
    
    # Verboseモードで新生AIを作成
    ai = NewbornAI("test_ai", verbose=True)
    
    print("\n🔍 思考・探索テスト:")
    print("-" * 40)
    
    # 1回だけ思考・探索を実行
    try:
        messages = await ai.think_and_explore()
        
        print(f"\n🔍 取得したメッセージ数: {len(messages) if messages else 0}")
        if messages:
            for i, msg in enumerate(messages):
                print(f"  メッセージ {i}: {type(msg)}")
                if hasattr(msg, 'result'):
                    print(f"    結果: {msg.result[:100] if msg.result else 'None'}...")
                elif hasattr(msg, 'content'):
                    print(f"    内容: {msg.content[:100] if msg.content else 'None'}...")
                else:
                    print(f"    属性: {dir(msg)}")
        
        print("\n📊 結果処理テスト:")
        print("-" * 40)
        ai._process_exploration_results(messages)
        
        print("\n✅ テスト完了!")
        print(f"📈 発達段階: {ai._get_current_curiosity_stage()}")
        print(f"📂 探索したファイル数: {len(ai.files_explored)}")
        print(f"💡 洞察数: {len(ai.insights)}")
        print(f"👤 他者認識レベル: {ai.other_awareness_level}")
        
    except Exception as e:
        import traceback
        print(f"❌ エラー: {e}")
        print("📋 詳細なトレースバック:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_verbose_ai())