#!/usr/bin/env python3
"""
Claude Code SDK探索メッセージのログ記録を有効化

使用方法:
1. このスクリプトを実行してNewbornAI 2.0システムを修正
2. 修正後にシステムを再起動
3. Claude探索メッセージが詳細ログに記録されます
"""

import os
import shutil
from pathlib import Path

def enable_claude_message_logging():
    """Claude探索メッセージのログ記録を有効化"""
    
    system_file = Path("newborn_ai_2_integrated_system.py")
    backup_file = Path("newborn_ai_2_integrated_system.py.backup")
    
    if not system_file.exists():
        print("❌ エラー: newborn_ai_2_integrated_system.py が見つかりません")
        return False
    
    # バックアップ作成
    shutil.copy2(system_file, backup_file)
    print(f"✅ バックアップ作成: {backup_file}")
    
    # ファイル内容を読み取り
    with open(system_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Claude探索メッセージログ記録を追加
    claude_logging_code = '''
    async def _claude_experiential_exploration(self):
        """Claude Code SDK による体験的探索"""
        prompt = f"""
現在のサイクル: {self.cycle_count}
発達段階: {self.current_stage.value}
意識レベル: {self.consciousness_level:.3f}

体験記憶中心の探索活動を行ってください:

1. 環境との純粋な体験的出会いを重視
2. 情報取得ではなく体験的理解を追求
3. 内在的な気づきや感じ方を大切に
4. 新しい体験概念の形成可能性を探る

今このサイクルで何を体験したいですか？
どのような体験的出会いを求めますか？
"""
        
        messages = []
        try:
            async for message in query(prompt=prompt, options=self.claude_sdk_options):
                messages.append(message)
                if self.verbose:
                    self._log("Claude探索メッセージ受信", "INFO")
                
                # Claude メッセージの詳細ログ記録を追加
                self._log_claude_message(message, prompt)
                    
        except Exception as e:
            self._log(f"Claude探索エラー: {e}", "ERROR")
        
        return messages
    
    def _log_claude_message(self, message, prompt):
        """Claude探索メッセージの詳細ログ記録"""
        try:
            claude_log = {
                'cycle': self.cycle_count,
                'timestamp': datetime.datetime.now().isoformat(),
                'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
                'message_type': type(message).__name__,
                'message_content': self._extract_message_content(message),
                'stage': self.current_stage.value,
                'phi_level': self.consciousness_level
            }
            
            # Claude専用ログファイルに保存
            claude_log_file = self.sandbox_dir / "claude_exploration_messages.json"
            self._save_json_log(claude_log_file, claude_log)
            
            if self.verbose:
                print(f"💬 Claude探索メッセージ記録: サイクル{self.cycle_count}")
                
        except Exception as e:
            self._log(f"Claudeメッセージログ記録エラー: {e}", "ERROR")
    
    def _extract_message_content(self, message):
        """Claudeメッセージから内容を抽出"""
        try:
            if hasattr(message, 'content'):
                content_parts = []
                for block in message.content:
                    if hasattr(block, 'text'):
                        content_parts.append(block.text)
                return '\\n'.join(content_parts)
            else:
                return str(message)
        except Exception as e:
            return f"メッセージ抽出エラー: {e}"'''
    
    # 既存の_claude_experiential_exploration メソッドを置換
    if '_claude_experiential_exploration' in content:
        # 既存メソッドの開始と終了を見つける
        start_marker = 'async def _claude_experiential_exploration(self):'
        end_marker = 'return messages'
        
        start_idx = content.find(start_marker)
        if start_idx != -1:
            # メソッドの終了位置を見つける（次のメソッド定義まで）
            next_method_idx = content.find('\n    def ', start_idx + 1)
            next_async_method_idx = content.find('\n    async def ', start_idx + 1)
            
            # より近い方を選択
            if next_method_idx == -1:
                end_idx = next_async_method_idx
            elif next_async_method_idx == -1:
                end_idx = next_method_idx
            else:
                end_idx = min(next_method_idx, next_async_method_idx)
            
            if end_idx != -1:
                # 既存メソッドを新しいメソッドで置換
                new_content = (content[:start_idx] + 
                             claude_logging_code.strip() + 
                             content[end_idx:])
                
                # ファイルに書き戻し
                with open(system_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print("✅ Claude探索メッセージログ記録を有効化しました")
                print("📁 ログファイル: sandbox/tools/08_02_2025/newborn_ai_2_0/claude_exploration_messages.json")
                print("🔄 システムを再起動してください:")
                print("   python newborn_ai_2_integrated_system.py verbose-start 60")
                
                return True
    
    print("❌ エラー: _claude_experiential_exploration メソッドが見つかりません")
    return False

def show_log_files():
    """ログファイルの場所を表示"""
    print("\n📂 NewbornAI 2.0 ログファイルの場所:")
    print("   📁 メインディレクトリ: sandbox/tools/08_02_2025/newborn_ai_2_0/")
    print()
    print("   📄 主要ログファイル:")
    print("   • claude_exploration_messages.json  ← 🆕 Claude探索メッセージ")
    print("   • development_stages.json           ← 発達段階履歴")
    print("   • consciousness_detection_log.json  ← 意識検出ログ")
    print("   • phi_trajectory.json               ← φ値履歴")
    print("   • experiential_memory.json          ← 体験記憶")
    print("   • consciousness_events.json         ← 意識イベント")
    print("   • integration_log.json              ← 統合処理ログ")
    print("   • system_status.json                ← システム状態")
    print()
    print("💡 ログ確認方法:")
    print("   # Claude探索メッセージの確認")
    print("   cat sandbox/tools/08_02_2025/newborn_ai_2_0/claude_exploration_messages.json")
    print()
    print("   # リアルタイム監視")
    print("   tail -f sandbox/tools/08_02_2025/newborn_ai_2_0/claude_exploration_messages.json")

if __name__ == "__main__":
    print("🔧 Claude探索メッセージログ記録有効化ツール")
    print("=" * 50)
    
    print("\n現在のログファイル状況:")
    show_log_files()
    
    print("\n📝 Claude探索メッセージログ記録を有効化しますか？ (y/N): ", end="")
    
    # 自動的にYesで実行
    print("y")
    if enable_claude_message_logging():
        print("\n🎉 設定完了！システムを再起動してください。")
    else:
        print("\n❌ 設定に失敗しました。手動での修正が必要です。")
    
    print("\n" + "=" * 50)
    show_log_files()