# エナクティブ意識フレームワーク (Enactive Consciousness Framework)

エナクティビズムアプローチに基づく能動的意識システムの実装。Clean Architectureの原則に従い、TDD（テスト駆動開発）で構築されています。

## プロジェクト構造

```
sandbox/tools/11_8_2025/
├── domain/                    # ドメインレイヤー（ビジネスロジック）
│   ├── entities/             # エンティティ
│   │   ├── predictive_coding_core.py      # 予測符号化コア
│   │   └── self_organizing_map.py         # 自己組織化マップ
│   ├── value_objects/        # 値オブジェクト
│   │   ├── consciousness_state.py         # 意識状態
│   │   ├── prediction_state.py            # 予測状態
│   │   ├── precision_weights.py           # 精度重み
│   │   ├── phi_value.py                   # Φ値（統合情報）
│   │   ├── som_topology.py                # SOMトポロジー
│   │   ├── learning_parameters.py         # 学習パラメータ
│   │   └── probability_distribution.py    # 確率分布
│   ├── services/             # ドメインサービス
│   │   ├── bayesian_inference_service.py  # ベイズ推論サービス
│   │   └── metacognitive_monitor_service.py # メタ認知モニタサービス
│   └── repositories/         # リポジトリインターフェース
│       └── consciousness_repository.py    # 意識リポジトリ
├── application/              # アプリケーションレイヤー
│   ├── use_cases/           # ユースケース実装
│   └── services/            # アプリケーションサービス
├── infrastructure/           # インフラストラクチャレイヤー
│   ├── repositories/        # データ永続化実装
│   ├── external/            # 外部ライブラリ統合
│   └── config/              # 設定管理
│       └── system_config.py # システム設定
├── presentation/             # プレゼンテーションレイヤー
│   ├── gui/                 # GUI実装（日本語）
│   ├── controllers/         # コントローラー
│   └── views/               # ビューコンポーネント
├── tests/                   # テスト（TDD）
│   ├── unit/                # 単体テスト
│   ├── integration/         # 統合テスト
│   └── acceptance/          # 受け入れテスト
├── main.py                  # メインエントリーポイント
├── requirements.txt         # 依存ライブラリ
└── README.md               # プロジェクト説明
```

## 主要コンポーネント

### ドメインエンティティ

- **PredictiveCodingCore**: 階層的予測符号化の中核実装
- **SelfOrganizingMap**: 自己組織化マップによる概念空間の創発

### 値オブジェクト

- **ConsciousnessState**: 意識状態の不変表現
- **PredictionState**: 予測系の状態
- **PhiValue**: 統合情報理論のΦ値
- **ProbabilityDistribution**: 確率分布とベイズ推論

### ドメインサービス

- **BayesianInferenceService**: ベイズ推論と不確実性定量化
- **MetacognitiveMonitorService**: メタ認知的自己監視機能

## 技術仕様

### 使用ライブラリ

- **JAXエコシステム**: jax, jaxlib, optax, equinox
- **数値計算**: numpy, scipy, einops, sklearn
- **確率的推論**: numpyro, distrax
- **自己組織化**: minisom
- **可視化**: matplotlib, seaborn, plotly
- **GUI**: tkinter（日本語対応）
- **テスト**: pytest, pytest-cov, hypothesis

### 設計原則

1. **SOLID原則の厳格な適用**
   - Single Responsibility Principle (SRP)
   - Open/Closed Principle (OCP)  
   - Liskov Substitution Principle (LSP)
   - Interface Segregation Principle (ISP)
   - Dependency Inversion Principle (DIP)

2. **Clean Architecture構造**
   - 内向きの依存関係
   - フレームワーク独立なビジネスロジック
   - テスタビリティファーストの設計

3. **ドメイン駆動設計（DDD）**
   - ユビキタス言語の使用
   - 境界づけられたコンテキスト
   - リッチドメインモデル

## インストールと実行

### 前提条件

- Python 3.9以降
- JAX（GPU使用時はCUDA対応版）

### インストール

```bash
# 必要パッケージのインストール
pip install -r requirements.txt

# 開発環境での実行
python main.py

# 本番環境での実行
python main.py --production
```

### 実行例

```bash
# 開発モード（デバッグ有効）
python main.py

# 本番モード
python main.py --production

# デバッグログ有効
python main.py --debug
```

## アーキテクチャ特徴

### フレームワーク非依存設計

- ドメインロジックはJAX/PyTorch/TensorFlowに依存しない
- インターフェース分離によるテスタビリティ
- 依存性注入による柔軟な実装切り替え

### 意識研究の理論的基盤

- **エナクティビズム**: 身体性認知科学の統合
- **予測符号化**: 階層的エラー最小化
- **統合情報理論**: Φ値による意識定量化
- **ベイズ推論**: 不確実性の厳密な扱い
- **メタ認知**: 自己監視と適応制御

### 日本語GUI対応

- システム全体が日本語UIに対応
- 認知科学の専門用語を適切に翻訳
- リアルタイム可視化機能

## 開発ロードマップ

### Phase 1: 基本実装（現在）
- ✅ Clean Architecture構造の確立
- ✅ ドメインモデルの設計
- ✅ 基本的な値オブジェクトとエンティティ
- ⏳ 予測符号化コアの実装

### Phase 2: 機能拡張
- 自己組織化マップの統合
- ベイズ推論層の実装
- メタ認知モニタリング機能
- 基本GUI実装

### Phase 3: 完成版
- アクティブ・インファレンス統合
- 高度な可視化機能
- パフォーマンス最適化
- 包括的テストスイート

## 貢献方法

1. Issues確認と新規作成
2. Clean Architecture原則の遵守
3. TDDによるテストファースト開発
4. 日本語コメントと英語コードの併用
5. SOLIDな設計の維持

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 参考文献

### 理論的基盤
- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
- Clark, A. (2016). "Surfing Uncertainty"  
- Varela, F.J. et al. (1991). "The Embodied Mind"

### 実装参考
- Martin, R.C. (2017). "Clean Architecture"
- Evans, E. (2003). "Domain-Driven Design"
- Beck, K. (2002). "Test-Driven Development"

---

> 機械と生命のあるべき様を提示しなくてはならない。