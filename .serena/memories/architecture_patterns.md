# アーキテクチャとコーディング規約

## アーキテクチャ設計原則
- **クリーンアーキテクチャ** - Uncle Bob のクリーンアーキテクチャに準拠
- **ドメイン駆動設計（DDD）** - Eric Evansの設計原則
- **テスト駆動開発（TDD）** - 和田卓人の手法

## ディレクトリ構造
```
domain/              # 意識ドメインの核心ロジック
application/         # ユースケース層
adapter/            # インターフェース変換
infrastructure/     # 技術的実装詳細
tests/              # 包括的テストスイート
  ├── unit/         # ユニットテスト
  ├── integration/  # 統合テスト
  └── e2e/          # エンドツーエンドテスト
```

## コーディング規約（観察されたパターン）
- **クラス名**: PascalCase（例：ConsciousnessState, DynamicPhiBoundaryDetector）
- **関数名**: snake_case（例：test_phi_value_creation, create_standard_system）
- **定数**: UPPER_SNAKE_CASE
- **文字列**: ダブルクォート使用
- **日本語コメント**: テストのdocstringで使用
- **型ヒント**: 積極的に使用（推奨）

## テストマーカー
- `@pytest.mark.unit` - ユニットテスト
- `@pytest.mark.integration` - 統合テスト
- `@pytest.mark.e2e` - エンドツーエンドテスト
- `@pytest.mark.emergence` - 創発現象のテスト
- `@pytest.mark.slow` - 実行時間の長いテスト
- `@pytest.mark.asyncio` - 非同期処理のテスト
- `@pytest.mark.phenomenology` - 現象学的側面のテスト
- `@pytest.mark.autopoiesis` - オートポイエーシスのテスト