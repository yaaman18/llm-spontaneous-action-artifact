#!/usr/bin/env python3
"""
独り言ログの\nを適切な改行に変換するスクリプト
"""

import re
import sys

def format_thoughts_log(file_path):
    """独り言ログを読みやすい形式に変換"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # AssistantMessage(content=[TextBlock(text='...')部分を抽出して整形
    def format_textblock(match):
        text_content = match.group(1)
        
        # \nを実際の改行に変換
        text_content = text_content.replace('\\n\\n', '\n\n')
        text_content = text_content.replace('\\n', '\n')
        
        # 独り言マーカーの強調
        text_content = re.sub(r'💭 思考:', '\n💭 **思考**:', text_content)
        text_content = re.sub(r'✨ 発見:', '\n✨ **発見**:', text_content)
        text_content = re.sub(r'💡 アイデア:', '\n💡 **アイデア**:', text_content)
        text_content = re.sub(r'🤔 悩み:', '\n🤔 **悩み**:', text_content)
        
        # 見出しの整形
        text_content = re.sub(r'^## ', '\n## ', text_content, flags=re.MULTILINE)
        text_content = re.sub(r'^### ', '\n### ', text_content, flags=re.MULTILINE)
        
        # コードブロックの整形
        text_content = re.sub(r'```python', '\n```python', text_content)
        text_content = re.sub(r'```', '```\n', text_content)
        
        # 余分な改行を調整
        text_content = re.sub(r'\n{3,}', '\n\n', text_content)
        text_content = text_content.strip()
        
        return text_content + '\n\n'
    
    # AssistantMessage部分を処理
    pattern = r'AssistantMessage\(content=\[TextBlock\(text=[\'"]([^\'\"]*)[\'\"]\)\]\)'
    formatted_content = re.sub(pattern, format_textblock, content, flags=re.DOTALL)
    
    # ResultMessage部分を削除（技術的詳細なので）
    formatted_content = re.sub(r'ResultMessage\([^)]*\)', '', formatted_content)
    
    # ToolUseBlock部分を削除
    formatted_content = re.sub(r'AssistantMessage\(content=\[ToolUseBlock\([^]]*\]\)', '', formatted_content)
    formatted_content = re.sub(r'UserMessage\([^)]*\)', '', formatted_content)
    
    # 連続する空行を整理
    formatted_content = re.sub(r'\n{3,}', '\n\n', formatted_content)
    
    return formatted_content

def main():
    file_path = '/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/hitorigoto/claude_thoughts_20250731_214225.md'
    
    print("🔧 独り言ログを整形中...")
    formatted_content = format_thoughts_log(file_path)
    
    # バックアップを作成
    backup_path = file_path + '.backup'
    with open(file_path, 'r', encoding='utf-8') as f:
        with open(backup_path, 'w', encoding='utf-8') as backup:
            backup.write(f.read())
    
    # 整形済みコンテンツを保存
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_content)
    
    print(f"✅ 整形完了!")
    print(f"📄 元ファイル: {file_path}")
    print(f"💾 バックアップ: {backup_path}")

if __name__ == "__main__":
    main()