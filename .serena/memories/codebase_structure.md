# コードベース構造

## ドメイン層 (domain/)
意識の核心概念を実装
- `entities.py` - ConsciousnessState（意識状態エンティティ）
- `value_objects.py` - PhiValue, StateType（値オブジェクト）
- `services.py` - DynamicPhiBoundaryDetector（動的Φ境界検出）
- `strategies.py` - Φ計算ストラテジパターン
- `events.py` - ドメインイベント定義
- `exceptions.py` - ドメイン例外
- `observers.py` - オブザーバーパターン実装
- `caching.py` - Φ計算キャッシュ
- `parallel.py` - 並列実行最適化
- `fluent_api.py` - Fluent APIインターフェース
- `consciousness_core.py` - 意識コアコンポーネント

## インフラストラクチャ層 (infrastructure/)
技術的実装詳細
- `azure_openai_client.py` - Azure OpenAI統合
- `config.py` - 設定管理
- `monitoring.py` - モニタリング実装
- `error_handling.py` - エラーハンドリング
- `integration_example.py` - 統合サンプル

## その他の重要ディレクトリ
- `institute/` - AI研究所システム（エージェント・ツール）
- `sandbox/` - 実験的実装
- `memory/` - （用途不明、調査必要）
- `archive/` - アーカイブ

## 設定ファイル
- `pytest.ini` - pytest設定（カバレッジ85%基準）
- `.gitignore` - PDF, pyc, __pycache__を除外