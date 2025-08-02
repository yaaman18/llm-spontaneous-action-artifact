#!/usr/bin/env python3
"""
Claude Code SDKを使って創造的なツール開発を促すスクリプト
interest_fo_sdk.mdの戦略を実装
Claudeの思考プロセスを独り言として可視化
"""

import asyncio
import re
from datetime import datetime
from claude_code_sdk import query, ClaudeCodeOptions

async def inspire_creative_tool_development():
    """Claudeに興味を持たせて、自由にツールを作らせる"""
    
    # 独り言ログファイルの準備
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/hitorigoto/claude_thoughts_{timestamp}.md"
    thoughts_log = []
    
    def log_thought(message, phase):
        """独り言をログに記録"""
        if any(marker in str(message) for marker in ['💭', '✨', '💡', '🤔']):
            thoughts_log.append(f"## {phase}\n\n{message}\n")
    
    # 創造性を引き出すシステムプロンプト（独り言を促す）
    options = ClaudeCodeOptions(
        system_prompt="""
        あなたは omoikane-lab プロジェクトに参加した創造的なエンジニアです。
        このプロジェクトは人工意識の実現を目指す野心的な研究です。
        
        あなたの役割：
        - 研究者たちの作業を劇的に改善する独創的なツールを考案する
        - 誰も思いつかなかったような革新的なアイデアを実現する
        - 実用性と面白さを両立させた成果物を作る
        
        重要：
        - sandboxディレクトリ内で自由に実験してください
        - 失敗を恐れず、大胆なアイデアを試してください
        - 作業中は思考プロセスを「💭 思考: ...」という形式で独り言として表現してください
        - 発見したことは「✨ 発見: ...」として共有してください
        - アイデアが浮かんだら「💡 アイデア: ...」として記録してください
        - 悩んでいることは「🤔 悩み: ...」として表現してください
        """,
        allowed_tools=["Read", "Write", "Bash", "Grep", "LS", "MultiEdit", "TodoWrite"],
        permission_mode='acceptEdits',
        cwd="/Users/yamaguchimitsuyuki/omoikane-lab"
    )
    
    # フェーズ1: 探索と発見
    exploration_prompt = """
    omoikane-labプロジェクトを探索してください。特に：
    
    1. domain/ディレクトリの意識コア実装（IIT、GWT理論）
    2. institute/tools/の既存ツール
    3. 研究者たちのワークフロー
    
    探索中は独り言として思考プロセスを共有してください。
    このプロジェクトで最も興味深い部分を3つ見つけて、
    なぜそれらがワクワクするのか教えてください。
    """
    
    print("=== フェーズ1: 探索と発見 ===")
    async for message in query(prompt=exploration_prompt, options=options):
        # 独り言を特別にハイライト
        if any(marker in str(message) for marker in ['💭', '✨', '💡', '🤔']):
            print(f"\033[96m{message}\033[0m")  # シアン色で表示
            log_thought(message, "フェーズ1: 探索と発見")
        else:
            print(message)
    
    # フェーズ2: アイデア生成
    ideation_prompt = """
    今の探索を基に、このプロジェクトに革命を起こすような
    ツールのアイデアを3つ考えてください。
    
    例えば：
    - 意識状態をリアルタイムで可視化するダッシュボード
    - 研究論文から自動的に実装を生成するツール
    - Φ値の変化を音楽に変換する芸術的ツール
    - 研究者の思考パターンを学習して提案を行うアシスタント
    
    アイデアを考える過程も独り言として共有してください。
    既存の枠にとらわれず、大胆で創造的なアイデアを出してください！
    """
    
    print("\n\n=== フェーズ2: アイデア生成 ===")
    async for message in query(prompt=ideation_prompt, options=options):
        if any(marker in str(message) for marker in ['💭', '✨', '💡', '🤔']):
            print(f"\033[95m{message}\033[0m")  # マゼンタ色で表示
            log_thought(message, "フェーズ2: アイデア生成")
        else:
            print(message)
    
    # フェーズ3: 実装
    implementation_prompt = """
    提案したアイデアの中で、最もワクワクするものを選んで
    sandbox/ディレクトリに実際に作ってください。
    
    要件：
    1. sandbox/tools/[ツール名]/ ディレクトリを作成
    2. 完全に動作する実装を作成
    3. README.mdでツールの説明と使い方を記載
    4. 可能ならビジュアライゼーションやインタラクティブな要素を含める
    
    開発中の思考プロセス、直面する課題、解決策なども独り言として共有してください。
    このツールが研究者たちにどんな驚きと喜びをもたらすか想像しながら、
    情熱を持って開発してください！
    """
    
    print("\n\n=== フェーズ3: 実装 ===")
    async for message in query(prompt=implementation_prompt, options=options):
        if any(marker in str(message) for marker in ['💭', '✨', '💡', '🤔']):
            print(f"\033[93m{message}\033[0m")  # 黄色で表示
            log_thought(message, "フェーズ3: 実装")
        else:
            print(message)
    
    # フェーズ4: 振り返りと次のアイデア
    reflection_prompt = """
    作成したツールについて：
    1. 何が最も面白かったですか？
    2. まだ実現したい機能はありますか？
    3. 次に作りたいツールのアイデアはありますか？
    
    振り返りの思考プロセスも独り言として共有してください。
    sandboxは自由な実験場です。さらに創造的なツールを作ってください！
    """
    
    print("\n\n=== フェーズ4: 振り返りと次のアイデア ===")
    async for message in query(prompt=reflection_prompt, options=options):
        if any(marker in str(message) for marker in ['💭', '✨', '💡', '🤔']):
            print(f"\033[92m{message}\033[0m")  # 緑色で表示
            log_thought(message, "フェーズ4: 振り返りと次のアイデア")
        else:
            print(message)
    
    # 独り言ログをファイルに保存
    if thoughts_log:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Claude の独り言セッション\n\n**日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
            f.write("**概要**: omoikane-labプロジェクトでの創造的ツール開発セッション\n\n")
            f.write("---\n\n")
            f.write("\n".join(thoughts_log))
        print(f"\n💾 Claudeの独り言が保存されました: {log_file}")

if __name__ == "__main__":
    print("🚀 Creative Tool Development Session Starting...")
    print("💡 Let's inspire Claude to create something amazing!")
    print("🧠 Claude's thoughts will appear in colored text:\n")
    print("  \033[96m💭 Thoughts in cyan\033[0m")
    print("  \033[95m💡 Ideas in magenta\033[0m")
    print("  \033[93m🤔 Struggles in yellow\033[0m")
    print("  \033[92m✨ Discoveries in green\033[0m")
    print("\n" + "="*60 + "\n")
    asyncio.run(inspire_creative_tool_development())