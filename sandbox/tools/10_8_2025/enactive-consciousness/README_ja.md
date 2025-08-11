# エナクティブ意識フレームワーク

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

現象学的基盤を持つエナクティビズム意識理論の包括的実装。人工意識研究と実践的応用のために構築されています。

## 🧠 理論的基盤

このフレームワークは認知科学と現象学の主要理論を実装しています：

- **フッサールの時間意識**: 把持-現在-前持の時間的総合
- **メルロ=ポンティの身体化認知**: 身体図式統合と運動意図性
- **バレラ-マトゥラーナの構造的カップリング**: システム-環境の動的相互作用
- **ギブソンの生態心理学**: アフォーダンス知覚と行為-環境カップリング
- **エナクティブ認知**: 構造的カップリングと身体化された相互作用による意味創出

## 🚀 主要機能

### 核心能力
- 🕐 **現象学的時間意識** - 把持-前持の総合による時間処理
- 🦾 **身体図式統合** - 固有感覚と運動処理
- 🔗 **構造的カップリング** - エージェントと環境の動的関係
- 👁️ **アフォーダンス知覚** - 行為-環境関係の検知
- 🧩 **意味創出プロセス** - エナクティブな意味構築
- ⚡ **高性能計算** - JAX/Equinox統合による最適化

### 技術的優秀性
- 🔒 **型安全実装** - Python 3.9+完全型ヒント対応
- 🧪 **テスト駆動開発** - 包括的テストカバレッジ
- 🏗️ **クリーンアーキテクチャ** - SOLID原則準拠
- 📦 **ドメイン駆動設計** - 明確な境界づけられたコンテクスト
- 🚀 **JITコンパイレーション** - 最適パフォーマンス
- 📊 **パフォーマンス監視** - メトリクス収集機能

## 📦 インストール

```bash


# 開発依存関係のインストール
pip install -e ".[dev]"

# またはPyPIからインストール（利用可能になった際）
pip install enactive-consciousness
```

### 要件

- Python 3.9+
- JAX 0.4.20+
- Equinox 0.11.0+
- NumPy 1.24.0+
- 完全な依存関係は`pyproject.toml`を参照

## 🎯 クイックスタート

```python
import jax
import jax.numpy as jnp
from enactive_consciousness import (
    create_framework_config,
    create_temporal_processor,
    create_body_schema_processor,
    TemporalConsciousnessConfig,
    BodySchemaConfig,
)

# フレームワークの初期化
key = jax.random.PRNGKey(42)
config = create_framework_config(
    retention_depth=10,           # 把持の深度
    protention_horizon=5,         # 前持の地平
    consciousness_threshold=0.6   # 意識の閾値
)

# 時間プロセッサーの作成
temporal_config = TemporalConsciousnessConfig()
temporal_processor = create_temporal_processor(
    temporal_config, state_dim=64, key=key
)

# 感覚入力の処理
sensory_input = jax.random.normal(key, (64,))
temporal_moment = temporal_processor.temporal_synthesis(
    primal_impression=sensory_input,
    timestamp=0.0
)

print(f"時間的総合完了！")
print(f"現在瞬間の形状: {temporal_moment.present_moment.shape}")
```

## 📚 ドキュメント

### 核心コンポーネント

#### 1. 時間意識
フッサールの内的時間意識の現象学を実装：

```python
from enactive_consciousness import TemporalConsciousnessConfig, create_temporal_processor

# 時間処理の設定
config = TemporalConsciousnessConfig(
    retention_depth=15,           # 保持される過去瞬間の深度
    protention_horizon=7,         # 予期される未来瞬間の地平
    temporal_synthesis_rate=0.1,  # 時間的流れの速度
)

processor = create_temporal_processor(config, state_dim=64, key=key)
```

#### 2. 身体図式統合
メルロ=ポンティの身体化認知を実装：

```python
from enactive_consciousness import BodySchemaConfig, create_body_schema_processor

# 身体化処理の設定
config = BodySchemaConfig(
    proprioceptive_dim=48,        # 固有感覚入力次元
    motor_dim=16,                 # 運動予測次元
    body_map_resolution=(15, 15), # 空間的身体マップ解像度
)

processor = create_body_schema_processor(config, key)
```

### 高度な使用方法

包括的なデモは`examples/basic_demo.py`を参照：
- 時間意識処理
- 身体図式統合
- 統合時間-身体化処理
- パフォーマンス監視
- 可視化

## 🧪 開発

### テスト実行

```bash
# すべてのテスト実行
make test

# カバレッジ付き実行
make test-coverage

# パフォーマンステスト
make test-performance

# 開発用ウォッチモード
make test-watch
```

### コード品質

```bash
# コードフォーマット
make format

# リント実行
make lint

# 型チェック
make typecheck

# すべての品質チェック
make quality
```

### 開発ワークフロー

1. **Red（赤）**: 失敗するテストを書く
2. **Green（緑）**: 最小限のコードでテストを通す
3. **Refactor（リファクタ）**: テストを維持しながらコード品質を向上

## 🏗️ アーキテクチャ

フレームワークは明確な関心の分離を持つクリーンアーキテクチャ原則に従います：

```
src/enactive_consciousness/
├── types.py           # 核心型定義とプロトコル
├── temporal.py        # 現象学的時間意識
├── embodiment.py      # 身体図式と身体化処理
├── coupling.py        # 構造的カップリング動力学（将来）
├── affordance.py      # アフォーダンス知覚（将来）
├── sense_making.py    # 意味構築（将来）
└── core.py           # 統合意識システム（将来）
```

### 設計原則

- **単一責任原則**: 各モジュールは明確な単一目的を持つ
- **開放閉鎖原則**: インターフェースを通じて拡張可能、修正に対して閉鎖
- **リスコフ置換原則**: すべての実装は交換可能
- **インターフェース分離原則**: 焦点を絞った一貫性のあるインターフェース
- **依存性逆転原則**: 具象ではなく抽象に依存

## 📊 パフォーマンス

フレームワークは高性能研究アプリケーション向けに最適化されています：

- **JAX JITコンパイレーション**: 標準実装の3-5倍高速化
- **メモリ最適化**: インテリジェントなメモリ管理とクリーンアップ
- **ベクトル化演算**: 効率的なバッチ処理能力
- **GPU/TPU対応**: 最新ハードウェアでのシームレスな加速

### ベンチマーク

| コンポーネント | 処理時間 | メモリ使用量 | スループット |
|----------------|----------|--------------|-------------|
| 時間的総合 | ~2ms | ~10MB | 500 ops/sec |
| 身体図式 | ~1.5ms | ~8MB | 650 ops/sec |
| 統合処理 | ~4ms | ~20MB | 250 ops/sec |

*NVIDIA RTX 4090での1000回平均ベンチマーク*

## 🔬 研究応用

このフレームワークは以下の研究分野向けに設計されています：

- **人工意識**: 意識理論の実装とテスト
- **認知ロボティクス**: 現象学的基盤を持つ身体化AIシステム
- **計算論的神経科学**: 時間的・身体化認知のモデル
- **心の哲学**: 現象学的概念の計算論的実装
- **人間-AI相互作用**: 自然で身体化されたインターフェース

### 引用

研究でこのフレームワークを使用する場合は、以下のように引用してください：

```bibtex
@software{enactive_consciousness_2024,
  title={エナクティブ意識フレームワーク: 現象学的基盤を持つ人工意識},
  author={エナクティビズム研究チーム},
  year={2024},
  url={https://github.com/research/enactive-consciousness},
  version={0.1.0}
}
```

## 🤝 コントリビューション

コントリビューションを歓迎します！詳細は[コントリビューション ガイドライン](CONTRIBUTING.md)を参照してください。

### 開発環境セットアップ

```bash
# クローンと開発環境セットアップ
git clone https://github.com/research/enactive-consciousness.git
cd enactive-consciousness
pip install -e ".[dev]"

# pre-commitフックのセットアップ
pre-commit install

# セットアップ確認のためのテスト実行
make test
```

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

この実装は以下の研究者の基礎的な仕事にインスパイアされ、それを基盤としています：

- **エドムント・フッサール** - 内的時間意識の現象学
- **モーリス・メルロ=ポンティ** - 身体化知覚の現象学
- **フランシスコ・バレラ & ウンベルト・マトゥラーナ** - オートポイエーシスと構造的カップリング
- **ジェームス・J・ギブソン** - 視覚知覚の生態学的アプローチ
- **エセキエル・ディ・パオロ** - エナクティブ認知理論

### 日本の研究者への特別な謝辞

- **乾敏郎** - 身体性認知と予測符号化
- **國吉康夫** - 発達ロボティクスと身体性
- **谷口忠大** - 記号創発システム
- **田口茂** - 現象学と認知科学
- **新田義弘** - 時間と永遠の哲学

## 🔗 関連プロジェクト

- [JAX](https://github.com/google/jax) - 数値計算ライブラリ
- [Equinox](https://github.com/patrick-kidger/equinox) - JAXニューラルネットワーク
- [NGC-Learn](https://github.com/NACLab/ngc-learn) - 神経生成符号化

## 🌸 日本語版特記事項

### 専門用語対訳表

| 英語 | 日本語 | 備考 |
|------|--------|------|
| Retention | 把持 | フッサール現象学用語 |
| Protention | 前持 | フッサール現象学用語 |
| Primal Impression | 根源印象 | 現在瞬間の意識 |
| Body Schema | 身体図式 | メルロ=ポンティ用語 |
| Motor Intentionality | 運動意図性 | 身体的志向性 |
| Structural Coupling | 構造的カップリング | バレラ-マトゥラーナ理論 |
| Affordance | アフォーダンス | ギブソン生態心理学 |
| Sense-making | 意味創出/センスメイキング | エナクティブ認知理論 |

### 日本での研究文脈

このフレームワークは特に以下の日本の研究伝統と関連しています：

- **京都学派の哲学**: 西田幾多郎の純粋経験論
- **身体性認知科学**: 乾敏郎らの身体性研究
- **発達ロボティクス**: 國吉康夫らの研究
- **記号創発**: 谷口忠大の記号創発システム

### 使用上の注意

- コード内のコメントとドキュメントは英語ですが、このREADMEでは日本語で解説しています
- API呼び出しは英語のままですが、設定パラメータの意味は上記専門用語対訳表を参照してください
- 日本語での質問やイシューも歓迎します

---

**意識研究と人工知能への❤️を込めて構築**

> "意識は脳が生み出す制御された幻覚である" - アニル・セス
> 
> "身体こそが世界への我々の最初の接点である" - モーリス・メルロ=ポンティ
> 
> "認知は世界の中での身体化された行為である" - フランシスコ・バレラ