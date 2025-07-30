# バーチャルAI意識研究所 図書館

## 📚 概要

研究所の全ての技術文書、研究論文、方法論、議論記録、仕様書を体系的に管理する図書館です。Sub agentsが過去の研究成果を参照し、知識を継承・発展させるためのリポジトリとして機能します。

## 🗂️ 構造

### `technical-docs/` - 技術文書
システム設計・アーキテクチャに関する技術文書
- `artificial-consciousness-clean-architecture.md` - クリーンアーキテクチャ設計
- `consciousness-clean-architecture-v2.md` - アーキテクチャ改良版
- `artificial-consciousness-ddd-model.md` - ドメイン駆動設計モデル
- `ddd-clean-architecture-integration.md` - DDD-クリーンアーキテクチャ統合
- `ddd-context-mapping.md` - コンテキストマッピング

### `methodology/` - 開発方法論
開発プロセス・手法に関する文書
- `artificial-consciousness-tdd-strategy.md` - TDD戦略
- `tdd-consciousness-development.md` - TDD意識開発手法

### `research-papers/` - 研究論文
理論的研究・分析論文
- `computational_phenomenology_analysis.md` - 計算現象学分析
- `consciousness_implementation_summary.md` - 意識実装サマリー
- `implementation_summary.md` - 実装総括

### `discussions/` - 議論記録
研究者間の議論・ディスカッション記録
- `llm-integration-discussion.md` - LLM統合議論
- `llm-integration-kickoff.md` - LLM統合キックオフ
- `collaboration_prep_kanai.md` - 金井協力準備記録

### `specifications/` - 仕様書
システム・エージェント仕様書
- `self-transforming-consciousness-spec.md` - 自己変容意識仕様
- `sub_agents_document.md` - サブエージェント文書
- `use_agents.md` - エージェント使用法
- `callently.md` - Callently仕様

## 🎯 活用方法

### Sub Agentsの参照パターン

#### 1. 研究開始時の背景調査
```bash
# 例: 新しい実装プロジェクト開始時
"institute/library/technical-docs/のアーキテクチャ文書を参照し、
過去の設計知見を踏まえて新しいシステム設計を検討してください。"
```

#### 2. 方法論の確認
```bash
# 例: 開発手法の選択時
"institute/library/methodology/のTDD関連文書を参照し、
意識システム開発に最適な手法を検討してください。"
```

#### 3. 過去の議論の継承
```bash
# 例: 類似テーマの議論時
"institute/library/discussions/のLLM統合議論を参照し、
過去の検討事項を踏まえた発展的議論を行ってください。"
```

### 知識の体系的活用

#### 縦断的参照
- 同一テーマの文書群を時系列で参照
- 研究の発展過程を把握
- 過去の決定理由と結果の分析

#### 横断的参照
- 異なるカテゴリの関連文書を比較
- 技術と理論の整合性確認
- 学際的視点の獲得

## 📖 文書管理プロトコル

### 新文書の追加
1. **カテゴリ分類**: 内容に応じた適切なディレクトリ選択
2. **命名規則**: `{topic}-{type}-{version}.md`
3. **メタデータ記録**: 作成日、著者、関連プロジェクト
4. **インデックス更新**: 本READMEファイルの更新

### 文書の更新・改版
1. **バージョン管理**: 重要な変更時はv2, v3等のバージョン作成
2. **変更履歴**: 文書内での変更記録
3. **関連文書更新**: 参照している文書への影響確認

### アーカイブ化
1. **古い文書**: `archive/legacy-docs/`への移動
2. **履歴保持**: 参照価値のある古い文書は保持
3. **インデックス整理**: 定期的な文書一覧の整理

## 🔍 検索・発見支援

### タグシステム
各文書にタグを付与し、横断検索を可能に
- `#architecture` `#implementation` `#theory` `#methodology`
- `#iit` `#phenomenology` `#engineering` `#philosophy`
- `#collaboration` `#discussion` `#specification`

### 関連文書マッピング
文書間の関係性を明示
- **依存関係**: A文書を理解するのにB文書が必要
- **発展関係**: A文書からB文書が派生
- **補完関係**: A文書とB文書が相互補完

## 🎯 Sub Agentsへの期待

### 能動的活用
- 新しい研究や実装前の体系的な文献調査
- 過去の知見を踏まえた発展的アプローチ
- 重複作業の回避と効率的な研究推進

### 知識の蓄積・更新
- 新しい発見や洞察の文書化
- 既存文書の改良・更新提案
- 学際的視点からの文書間関係性の発見

### 継承と発展
- 過去の研究者の意図と文脈の理解
- 歴史的経緯を踏まえた建設的な発展
- 伝統と革新のバランスの取れた研究

---

この図書館を通じて、研究所の知的遺産が継承され、Sub agentsによる継続的な知識創発が実現されます。

**最終更新**: 2025-07-29  
**管理責任者**: プロジェクト・オーケストレーター