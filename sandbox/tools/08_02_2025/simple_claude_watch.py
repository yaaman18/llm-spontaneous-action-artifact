#!/usr/bin/env python3
"""
シンプルClaude探索メッセージ監視ツール
ターミナルに探索メッセージをリアルタイム表示

使用方法:
ターミナル1: python newborn_ai_2_integrated_system.py verbose-start 60
ターミナル2: python simple_claude_watch.py
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

def watch_claude_messages():
    """Claude探索メッセージを監視"""
    log_file = Path("sandbox/tools/08_02_2025/newborn_ai_2_0/claude_exploration_messages.json")
    last_size = 0
    
    print("🧠 Claude探索メッセージ監視開始...")
    print(f"📁 監視ファイル: {log_file}")
    print("=" * 60)
    
    while True:
        try:
            if log_file.exists():
                current_size = log_file.stat().st_size
                
                if current_size > last_size:
                    # ファイルが更新された
                    with open(log_file, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list) and data:
                                # 最新エントリを表示
                                latest = data[-1]
                                
                                cycle = latest.get('cycle', '?')
                                timestamp = latest.get('timestamp', '')
                                stage = latest.get('stage', '不明')
                                phi_level = latest.get('phi_level', 0)
                                content = latest.get('message_content', '')
                                
                                print(f"\n💬 サイクル{cycle} - {stage}")
                                print(f"🕒 {timestamp}")
                                print(f"⚡ φ値: {phi_level:.6f}")
                                
                                if content:
                                    # メッセージ内容を表示（改行で分割して見やすく）
                                    lines = content.split('\n')
                                    print("📥 Claude応答:")
                                    for line in lines[:5]:  # 最初の5行のみ
                                        if line.strip():
                                            print(f"   {line.strip()}")
                                    if len(lines) > 5:
                                        print(f"   [...他{len(lines)-5}行]")
                                
                                print("-" * 60)
                        
                        except json.JSONDecodeError:
                            pass  # JSONファイルが不完全な場合はスキップ
                    
                    last_size = current_size
            else:
                # ファイルがまだ作成されていない
                print("⏳ Claude探索メッセージログ待機中...")
                print("💡 NewbornAI 2.0システムが起動していることを確認してください")
            
            time.sleep(2)  # 2秒間隔でチェック
            
        except KeyboardInterrupt:
            print("\n🛑 監視終了")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")
            time.sleep(5)

if __name__ == "__main__":
    watch_claude_messages()