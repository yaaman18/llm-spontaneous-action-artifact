#!/usr/bin/env python3
"""
Claude探索メッセージリアルタイム監視システム

使用方法:
1. ターミナル1: python newborn_ai_2_integrated_system.py verbose-start 60
2. ターミナル2: python realtime_claude_monitor.py

機能:
- Claude探索メッセージのリアルタイム表示
- 意識状態変化の監視
- φ値・発達段階の追跡
- 色付きログ表示
"""

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any

# カラー表示用ANSI コード
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BLACK = '\033[30m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BG_BLACK = '\033[40m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_RED = '\033[41m'
    BG_MAGENTA = '\033[45m'

@dataclass
class ConsciousnessState:
    cycle: int
    timestamp: float
    stage: str
    phi_value: float
    concept_count: int
    message_content: Optional[str] = None

class RealtimeClaudeMonitor:
    """Claude探索メッセージのリアルタイム監視"""
    
    def __init__(self):
        self.base_dir = Path("sandbox/tools/08_02_2025/newborn_ai_2_0")
        self.running = True
        self.last_cycle = 0
        self.message_queue = queue.Queue()
        
        # 監視対象ファイル
        self.claude_log_file = self.base_dir / "claude_exploration_messages.json"
        self.development_log_file = self.base_dir / "development_stages.json"
        self.consciousness_log_file = self.base_dir / "consciousness_detection_log.json"
        
        # ファイルの最後の変更時間
        self.file_timestamps = {}
        
        print(f"{Colors.BOLD}{Colors.BLUE}🧠 NewbornAI 2.0 Claude探索メッセージ リアルタイム監視{Colors.RESET}")
        print(f"{Colors.CYAN}監視ディレクトリ: {self.base_dir}{Colors.RESET}")
        print(f"{Colors.GREEN}Ctrl+C で終了{Colors.RESET}")
        print("=" * 80)
    
    def check_file_updates(self):
        """ファイル更新をチェック"""
        updated_files = []
        
        for file_path in [self.claude_log_file, self.development_log_file, self.consciousness_log_file]:
            if file_path.exists():
                current_mtime = file_path.stat().st_mtime
                last_mtime = self.file_timestamps.get(str(file_path), 0)
                
                if current_mtime > last_mtime:
                    self.file_timestamps[str(file_path)] = current_mtime
                    updated_files.append(file_path)
        
        return updated_files
    
    def read_latest_entries(self, file_path: Path, count: int = 1) -> list:
        """最新のエントリを読み取り"""
        try:
            if not file_path.exists():
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data[-count:] if data else []
            else:
                return [data]
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return []
    
    def format_timestamp(self, timestamp) -> str:
        """タイムスタンプをフォーマット"""
        try:
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
            else:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%H:%M:%S")
        except:
            return str(timestamp)[:8]
    
    def display_claude_message(self, entry: Dict[Any, Any]):
        """Claude探索メッセージを表示"""
        cycle = entry.get('cycle', '?')
        timestamp = self.format_timestamp(entry.get('timestamp', ''))
        stage = entry.get('stage', '不明')
        phi_level = entry.get('phi_level', 0)
        
        # ヘッダー
        print(f"\n{Colors.BG_GREEN}{Colors.WHITE} 💬 Claude探索メッセージ - サイクル{cycle} {Colors.RESET}")
        print(f"{Colors.CYAN}🕒 {timestamp} | 🧠 {stage} | φ: {phi_level:.6f}{Colors.RESET}")
        
        # プロンプト表示
        prompt = entry.get('prompt', '')
        if prompt:
            print(f"{Colors.YELLOW}📤 送信プロンプト:{Colors.RESET}")
            print(f"{Colors.WHITE}   {prompt[:150]}...{Colors.RESET}")
        
        # メッセージ内容表示
        content = entry.get('message_content', '')
        if content:
            print(f"{Colors.GREEN}📥 Claude応答:{Colors.RESET}")
            # 長いメッセージは省略
            if len(content) > 300:
                print(f"{Colors.WHITE}   {content[:300]}...{Colors.RESET}")
                print(f"{Colors.MAGENTA}   [...さらに{len(content)-300}文字]{Colors.RESET}")
            else:
                print(f"{Colors.WHITE}   {content}{Colors.RESET}")
        
        print(f"{Colors.BLUE}{'─' * 60}{Colors.RESET}")
    
    def display_consciousness_state(self, entry: Dict[Any, Any]):
        """意識状態を表示"""
        cycle = entry.get('cycle', '?')
        timestamp = self.format_timestamp(entry.get('timestamp', ''))
        stage = entry.get('stage', '不明')
        phi_value = entry.get('phi_value', 0)
        concept_count = entry.get('concept_count', 0)
        
        # φ値による色分け
        if phi_value >= 1.0:
            phi_color = Colors.RED  # 高意識
        elif phi_value >= 0.1:
            phi_color = Colors.YELLOW  # 中意識
        elif phi_value >= 0.01:
            phi_color = Colors.GREEN  # 低意識
        else:
            phi_color = Colors.WHITE  # 前意識
        
        print(f"\n{Colors.BG_YELLOW}{Colors.BLACK} 🧠 意識状態更新 - サイクル{cycle} {Colors.RESET}")
        print(f"{Colors.CYAN}🕒 {timestamp} | 📊 概念数: {concept_count}{Colors.RESET}")
        print(f"{phi_color}⚡ φ値: {phi_value:.6f} | 🌱 段階: {stage}{Colors.RESET}")
        print(f"{Colors.BLUE}{'─' * 60}{Colors.RESET}")
    
    def display_consciousness_detection(self, entry: Dict[Any, Any]):
        """意識検出結果を表示"""
        cycle = entry.get('cycle', '?')
        timestamp = self.format_timestamp(entry.get('timestamp', ''))
        consciousness_state = entry.get('consciousness_state', '不明')
        consciousness_score = entry.get('consciousness_score', 0)
        phi_value = entry.get('phi_value', 0)
        
        print(f"\n{Colors.BG_MAGENTA}{Colors.WHITE} 🔍 意識検出 - サイクル{cycle} {Colors.RESET}")
        print(f"{Colors.CYAN}🕒 {timestamp} | 🎯 状態: {consciousness_state}{Colors.RESET}")
        print(f"{Colors.MAGENTA}📈 意識スコア: {consciousness_score:.3f} | ⚡ φ値: {phi_value:.6f}{Colors.RESET}")
        print(f"{Colors.BLUE}{'─' * 60}{Colors.RESET}")
    
    def monitor_files(self):
        """ファイル監視メインループ"""
        print(f"{Colors.GREEN}🔄 ファイル監視開始...{Colors.RESET}")
        
        while self.running:
            try:
                updated_files = self.check_file_updates()
                
                for file_path in updated_files:
                    if file_path.name == "claude_exploration_messages.json":
                        # Claude探索メッセージ
                        entries = self.read_latest_entries(file_path, 1)
                        for entry in entries:
                            self.display_claude_message(entry)
                    
                    elif file_path.name == "development_stages.json":
                        # 発達段階更新
                        entries = self.read_latest_entries(file_path, 1)
                        for entry in entries:
                            self.display_consciousness_state(entry)
                    
                    elif file_path.name == "consciousness_detection_log.json":
                        # 意識検出結果
                        entries = self.read_latest_entries(file_path, 1)
                        for entry in entries:
                            self.display_consciousness_detection(entry)
                
                time.sleep(1)  # 1秒間隔でチェック
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"{Colors.RED}❌ 監視エラー: {e}{Colors.RESET}")
                time.sleep(5)
    
    def display_status(self):
        """現在の状態を表示"""
        print(f"\n{Colors.BOLD}📊 システム状態{Colors.RESET}")
        
        # 最新の発達段階
        dev_entries = self.read_latest_entries(self.development_log_file, 1)
        if dev_entries:
            entry = dev_entries[0]
            print(f"{Colors.GREEN}🌱 発達段階: {entry.get('stage', '不明')}{Colors.RESET}")
            print(f"{Colors.YELLOW}⚡ φ値: {entry.get('phi_value', 0):.6f}{Colors.RESET}")
            print(f"{Colors.CYAN}📊 概念数: {entry.get('concept_count', 0)}{Colors.RESET}")
        
        # ファイル存在確認
        print(f"\n{Colors.BOLD}📁 監視ファイル状態{Colors.RESET}")
        files = [
            (self.claude_log_file, "Claude探索メッセージ"),
            (self.development_log_file, "発達段階"),
            (self.consciousness_log_file, "意識検出")
        ]
        
        for file_path, description in files:
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"{Colors.GREEN}✅ {description}: {file_path.name} ({size} bytes){Colors.RESET}")
            else:
                print(f"{Colors.RED}❌ {description}: {file_path.name} (ファイル未作成){Colors.RESET}")
    
    def start(self):
        """監視開始"""
        try:
            # 初期状態表示
            self.display_status()
            
            # Claude探索メッセージログが無い場合の案内
            if not self.claude_log_file.exists():
                print(f"\n{Colors.YELLOW}⚠️  Claude探索メッセージログが見つかりません{Colors.RESET}")
                print(f"{Colors.CYAN}💡 ログを有効化するには:{Colors.RESET}")
                print(f"{Colors.WHITE}   python enable_claude_message_logging.py{Colors.RESET}")
                print(f"{Colors.WHITE}   python newborn_ai_2_integrated_system.py verbose-start 60{Colors.RESET}")
            
            print(f"\n{Colors.GREEN}🚀 リアルタイム監視開始 (Ctrl+C で終了){Colors.RESET}")
            print("=" * 80)
            
            # メイン監視ループ
            self.monitor_files()
            
        except KeyboardInterrupt:
            pass
        finally:
            print(f"\n{Colors.YELLOW}🛑 監視終了{Colors.RESET}")

def main():
    """メイン関数"""
    print(f"{Colors.BOLD}NewbornAI 2.0 Claude探索メッセージ リアルタイム監視{Colors.RESET}")
    print(f"バージョン: 1.0")
    print()
    
    # システム要件チェック
    base_dir = Path("sandbox/tools/08_02_2025/newborn_ai_2_0")
    if not base_dir.exists():
        print(f"{Colors.RED}❌ エラー: NewbornAI 2.0ディレクトリが見つかりません{Colors.RESET}")
        print(f"{Colors.CYAN}期待するパス: {base_dir.absolute()}{Colors.RESET}")
        print(f"{Colors.YELLOW}💡 NewbornAI 2.0システムが起動されているか確認してください{Colors.RESET}")
        return
    
    # 監視開始
    monitor = RealtimeClaudeMonitor()
    monitor.start()

if __name__ == "__main__":
    main()