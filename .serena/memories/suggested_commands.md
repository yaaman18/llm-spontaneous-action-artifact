# 推奨コマンド一覧

## テスト実行
```bash
# 全テスト実行（デフォルト設定）
pytest

# ユニットテストのみ
pytest -m unit

# 統合テストのみ
pytest -m integration

# E2Eテストのみ
pytest -m e2e

# 高速テスト（slowマーカー以外）
pytest -m "not slow"

# 特定ファイルのテスト
pytest tests/unit/domain/test_phi_value.py

# カバレッジなしで実行（高速）
pytest --no-cov

# 詳細出力なしで実行
pytest -q
```

## Pythonスクリプト実行
```bash
# 意識エンジンの起動（READMEに記載）
python -m domain.consciousness_core
```

## 開発環境
```bash
# Python環境確認
python --version
which python

# 依存関係インストール（READMEに記載）
pip install -r requirements.txt
```

## Git操作
```bash
# ステータス確認
git status

# 変更の差分確認
git diff

# コミット履歴
git log --oneline -10
```

## ファイル操作（Darwin/macOS）
```bash
# ディレクトリ内容確認
ls -la

# ファイル検索
find . -name "*.py" -type f

# コード検索（ripgrep推奨）
rg "pattern" --type py
```

## 注意事項
- リンターやフォーマッターのコマンドは現在設定されていない
- requirements.txtはプロジェクトルートに存在しない（サブディレクトリにのみ存在）
- カバレッジ基準は85%に設定されている