#!/usr/bin/env python3
"""
独り言ログの\\nエスケープ文字を完全に実改行に変換
"""

import re

def fix_all_newlines(file_path):
    """すべての\\nを実際の改行に変換"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # まずバックアップ
    backup_path = file_path + '.backup2'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # \\n\\nを\\n\\nに変換（ダブル改行）
    content = content.replace('\\n\\n', '\n\n')
    
    # 残りの\\nを実改行に変換
    content = content.replace('\\n', '\n')
    
    # AssistantMessageやResultMessageなどの技術情報を削除
    content = re.sub(r'AssistantMessage\(content=\[TextBlock\(text=[\'\"](.*?)[\'\"]?\)\]\)', r'\\1', content, flags=re.DOTALL)
    content = re.sub(r'ResultMessage\([^)]*\)', '', content, flags=re.DOTALL)
    content = re.sub(r'AssistantMessage\(content=\[ToolUseBlock\([^]]*\]\)', '', content, flags=re.DOTALL)
    content = re.sub(r'UserMessage\([^)]*\)', '', content, flags=re.DOTALL)
    
    # 独り言マーカーの前に改行を追加
    content = re.sub(r'(💭\s*\*\*思考\*\*)', r'\n\1', content)
    content = re.sub(r'(✨\s*\*\*発見\*\*)', r'\n\1', content)  
    content = re.sub(r'(💡\s*\*\*アイデア\*\*)', r'\n\1', content)
    content = re.sub(r'(🤔\s*\*\*悩み\*\*)', r'\n\1', content)
    
    # 見出しの前に改行
    content = re.sub(r'([^\n])(##\s+)', r'\\1\n\n\\2', content)
    content = re.sub(r'([^\n])(###\s+)', r'\\1\n\n\\2', content)
    
    # コードブロックの整形
    content = re.sub(r'```([a-z]*)', r'\n```\\1', content)
    content = re.sub(r'```\n*([^`]+)```', r'```\n\\1\n```\n', content, flags=re.DOTALL)
    
    # 連続する空行を2つまでに制限
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    # ファイルの先頭と末尾の空行を整理
    content = content.strip() + '\n'
    
    return content

def main():
    file_path = '/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/hitorigoto/claude_thoughts_20250731_214225.md'
    
    print("🔧 \\nエスケープ文字を完全に修正中...")
    
    fixed_content = fix_all_newlines(file_path)
    
    # 修正済みコンテンツを保存
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("✅ 修正完了!")
    print("📄 元ファイルが更新されました")
    print("💾 バックアップ: claude_thoughts_20250731_214225.md.backup2")

if __name__ == "__main__":
    main()