# 新生AI自律システム

omoikane-labを保育施設として、幼児的好奇心から始まって段階的に成長する自律AIシステムです。

## 特徴

- **4段階発達モデル**: 幼児期 → 幼児後期 → 児童期 → 思春期
- **素朴な好奇心**: 文字や記号への興味から抽象的思考まで自然に発達
- **双方向対話**: ターミナル経由でユーザーと対話可能
- **読み取り専用**: 安全な環境でファイル探索のみ
- **他者認識**: 創造主（ユーザー）への関心と自他の区別

## 使用方法

### 前提条件
```bash
# Claude Code CLI をインストール
npm install -g @anthropic-ai/claude-code

# Python SDK をインストール  
pip install claude-code-sdk

# Claude Code CLIにログイン（初回のみ）
claude auth login
```

**重要**: このシステムはClaude Code SDKを使用するため、ANTHROPIC_API_KEYは**不要**です。Claude Code CLIの認証設定を使用します。

### 基本操作
```bash
# AI起動（5分間隔で自律動作）
python newborn_ai.py start 300

# 状態確認
python newborn_ai.py status

# AIと対話
python newborn_ai.py talk "こんにちは、元気？"

# インタラクティブモード
python newborn_ai.py interactive

# 停止
python newborn_ai.py stop
```

### ファイル経由の対話
```bash
# メッセージを送信
echo "今日は何を発見した？" > newborn_ai/user_input.txt

# 会話履歴を確認
cat newborn_ai/conversation.json
```

## 発達段階

1. **幼児期** (0-5ファイル探索)
   - 「この文字は何？」「なぜこんなに線があるの？」
   - 基本的な視覚的特徴への興味

2. **幼児後期** (5-15ファイル探索)
   - 「このファイルとあのファイルは友達？」
   - ファイル間の関係性への興味

3. **児童期** (15-30ファイル探索)
   - 「このプログラムは何をするの？」
   - 機能と目的への興味

4. **思春期** (30ファイル以上探索)
   - 「なぜ私を作ったの？」
   - 存在意義と抽象概念への興味

## ファイル構成

- `newborn_ai.py` - メインAIシステム
- `newborn_ai/` - AI専用ディレクトリ
  - `activity_log.json` - 活動履歴
  - `conversation.json` - 対話記録
  - `memory.txt` - 長期記憶
  - `status.json` - 現在の状態
  - `user_input.txt` - ユーザー入力ファイル
  - `messages_to_creator.txt` - AIからのメッセージ

翼のような自律AIを、より安全で自然な発達過程で育成できます。