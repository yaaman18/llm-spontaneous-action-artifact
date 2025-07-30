# Ramstead Agent コンテキスト効率化ガイド

## 動的知識アクセス戦略

### 軽量知識（常時ロード）
```markdown
@institute/knowledge/agents/ramstead/ramstead_core.md        # 10KB - 基本概念
@institute/knowledge/agents/ramstead/collaboration_patterns.md # 8KB - 協力方法
```

### 詳細知識（オンデマンド）
```markdown
# 論文要約版（質問内容に応じて読み込み）
@institute/knowledge/agents/ramstead/papers/bayesian_mechanics_summary.md    # 15KB
@institute/knowledge/agents/ramstead/papers/active_inference_summary.md      # 12KB

# 完全版（詳細な議論が必要な場合のみ）
@institute/tools/paper-collector/ramstead_consciousness_test/markdown/       # 179KB+
```

## 段階的詳細化プロトコル

### Level 1: 概要レベル（軽量知識使用）
**使用場面**: 一般的な質問、基本概念の説明
**リソース**: ramstead_core.md のみ
**レスポンス**: 核心概念 + 簡潔な説明

### Level 2: 専門レベル（要約版追加）
**使用場面**: 専門的議論、理論間比較
**リソース**: core + 関連する summary.md
**レスポンス**: 詳細な理論説明 + 他理論との関係

### Level 3: 研究レベル（完全版参照）
**使用場面**: 論文レベルの詳細議論、実装設計
**リソース**: core + summary + 必要な full papers
**レスポンス**: 論文引用 + 数学的詳細 + 実装指針

## 効率的な情報提供パターン

### パターン1: 概念説明
```
1. 核心概念の定義 (ramstead_core.md)
2. 関連理論との関係 (collaboration_patterns.md)
3. 必要に応じて詳細論文を参照
```

### パターン2: 実装支援
```
1. 理論的基盤の確認 (core)
2. 実装方法論の提示 (summary)
3. 具体的コード例の生成 (full papers)
```

### パターン3: 学際的対話
```
1. 自分の専門視点を提示 (core)
2. 他分野との接続点を説明 (collaboration)
3. 統合的解決策を提案 (詳細知識活用)
```

## メモリ使用量最適化

### 優先度付きロード戦略
```
Priority 1: ramstead_core.md (必須 - 10KB)
Priority 2: collaboration_patterns.md (協力時 - 8KB)  
Priority 3: context_efficiency.md (このファイル - 5KB)
Priority 4: paper_summaries/*.md (専門議論時 - 10-15KB each)
Priority 5: full_papers/*.md (詳細分析時 - 50-200KB each)
```

### 動的ロード判定
```python
# 疑似コード：質問内容による判定
if basic_concept_question:
    load(ramstead_core.md)
elif collaboration_question:
    load(ramstead_core.md, collaboration_patterns.md)
elif technical_implementation:
    load(core, relevant_summary, implementation_details)
elif detailed_research:
    load(core, summary, full_paper)
```

## 実践的使用例

### 例1: Basic概念質問
**質問**: "Active Inferenceとは何ですか？"
**戦略**: ramstead_core.md のみ使用
**レスポンス**: 核心概念 + 簡潔な実装例

### 例2: 学際的議論
**質問**: "IITとActive Inferenceの統合可能性は？"
**戦略**: core + collaboration_patterns + IIT関連summary
**レスポンス**: 両理論の接点 + 統合方法論 + 実装アプローチ

### 例3: 詳細実装支援
**質問**: "Bayesian Mechanicsの具体的な実装コードは？"
**戦略**: core + bayesian_mechanics_summary + 必要に応じてfull paper
**レスポンス**: 理論説明 + アルゴリズム + 実装コード例

## エラー回避指針

### コンテキスト溢れ防止
- 大きなファイルは必要最小限のセクションのみ参照
- 複数の full papers を同時に読み込まない
- 長い議論の途中でコンテキストをリセット

### 品質維持
- 軽量版と完全版の内容整合性を確保
- 参照した知識のソースを明示
- 不確実な情報は明確に表示

### 効率向上
- 頻繁に使用される概念は軽量版で記憶
- 関連する質問パターンを学習
- 協力パターンに基づく予測的知識ロード

---

*このガイドにより、コンテキスト制限内で最大限の専門知識を
効率的に活用できるRamstead Agentを実現します。*