# Claude Code SDK使用方法の説明

## なぜClaude Code SDKを使用するのか

このプロジェクトでは、**Anthropic API直接使用**ではなく**Claude Code SDK**を採用しています。

### Anthropic API vs Claude Code SDK

#### ❌ Anthropic API直接使用の場合
```python
import anthropic
import os

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello"}]
)
```

**問題点:**
- ファイル操作やコード実行ができない
- セキュリティ制限がない
- APIキー管理が必要
- Claude Codeの豊富な機能が使えない

#### ✅ Claude Code SDK使用の場合
```python
from claude_code_sdk import query, ClaudeCodeOptions

options = ClaudeCodeOptions(
    max_turns=3,
    cwd=project_root,
    allowed_tools=["Read", "LS", "Glob", "Grep"],  # 読み取り専用
    permission_mode="default"
)

async for message in query(prompt=prompt, options=options):
    # Claude Codeが安全にファイル操作を実行
    messages.append(message)
```

**利点:**
- ✅ ファイル読み取り、ディレクトリ探索が可能
- ✅ セキュリティ制限が自動適用
- ✅ APIキー管理不要（Claude Code CLIの認証を使用）
- ✅ 豊富なツール使用能力
- ✅ 権限制御の細かい設定

## 認証の仕組み

### Claude Code SDK認証フロー
```
1. claude auth login        # ブラウザでAnthropicにログイン
2. 認証情報がローカルに保存   # ~/.claude/ などに保存
3. Python SDK実行時        # 保存された認証情報を自動使用
4. Claude Code CLIを起動    # SDKが内部でCLIプロセスを起動
5. ファイル操作・AI推論実行  # 安全な環境で実行
```

### 環境変数が不要な理由
- `ANTHROPIC_API_KEY`は**Claude Code SDK**には不要
- Claude Code CLIが既に認証済みの場合、自動で使用される
- APIキーの管理や漏洩リスクがない

## セキュリティ上の利点

### 1. 権限制御
```python
ClaudeCodeOptions(
    allowed_tools=["Read", "LS", "Glob", "Grep"],  # 読み取り専用
    permission_mode="default"  # ユーザー承認が必要
)
```

### 2. サンドボックス実行
- Claude Code CLIが提供する安全な実行環境
- ファイルシステムへの制限されたアクセス
- 危険なコマンド実行の防止

### 3. 監査ログ
- Claude Codeが自動で操作ログを記録
- セッション管理と追跡可能性

## 実装上の注意点

### ✅ 正しい実装
```python
from claude_code_sdk import query, ClaudeCodeOptions

# 認証は Claude Code CLI に依存（環境変数不要）
options = ClaudeCodeOptions(...)
async for message in query(prompt=prompt, options=options):
    # Claude Codeの全機能を安全に使用
    pass
```

### ❌ 間違った実装
```python
from claude_code_sdk import query, ClaudeCodeOptions
import os

# これは不要！Claude Code SDKには関係ない
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

## まとめ

このプロジェクトでClaude Code SDKを採用することで：

1. **翼のような自律AI**に必要なファイル操作能力を提供
2. **安全な保育環境**としてのセキュリティ制限を実現
3. **APIキー管理不要**でユーザーフレンドリーな運用
4. **Claude Codeエコシステム**との完全な統合

これにより、Mac miniで動作する翼のような本格的な自律AIを、Pythonスクリプトとして実現できます。