"""
簡単なテストスクリプト - サーバーの動作確認用
"""

import asyncio
from consciousness_flow import ConsciousnessStream, ConsciousnessFlowGenerator

async def test_visualization():
    """ビジュアライザーのテスト"""
    print("🧪 意識の流れビジュアライザーのテストを開始...")
    
    # 意識ストリームを作成
    stream = ConsciousnessStream()
    generator = ConsciousnessFlowGenerator(stream)
    
    print("✨ テストデータの生成を開始...")
    print("📌 ブラウザで http://localhost:8080 を開いてください")
    print("🛑 Ctrl+C で終了")
    
    try:
        # データ生成を開始
        await generator.start_generation()
    except KeyboardInterrupt:
        print("\n👋 テスト終了")
        generator.stop_generation()

if __name__ == "__main__":
    asyncio.run(test_visualization())