# 意識研究論文収集・変換ツール

このツールは、Google ScholarとarXivから意識研究に関する論文を自動収集し、PDFをダウンロードして、マークダウン形式に変換します。

## 機能

### 論文収集機能
- arXiv APIを使用した論文検索
- Google Scholarのスクレイピングによる論文検索（制限あり）
- PDFの自動ダウンロード
- 論文メタデータの保存（JSON形式）
- 重複検出機能

### PDF変換機能 🆕
- PDFからマークダウン形式への変換
- 数式のLaTeX表記保持
- 表構造の保持
- メタデータ付きマークダウン出力
- 一括変換機能

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 論文収集

#### 直接実行
```bash
python paper_collector.py
```

#### 特定の著者の論文を検索
```bash
python search_ramstead.py
```

#### Pythonコードから使用
```python
from paper_collector import PaperCollector

# カスタム出力ディレクトリを指定
collector = PaperCollector("my_papers")

# 特定のキーワードで検索
arxiv_papers = collector.search_arxiv("consciousness", max_results=10)

# すべての意識研究関連論文を収集
papers = collector.collect_consciousness_papers(max_papers_per_keyword=5)
```

### 2. PDF変換

#### 単一PDFファイルの変換
```bash
python pdf_to_markdown.py /path/to/pdf/directory --single_file paper.pdf
```

#### ディレクトリ内のすべてのPDFを変換
```bash
python pdf_to_markdown.py /path/to/pdf/directory
```

#### 一括変換（論文収集ツールと連携）
```bash
# 特定の著者のPDFを変換
python batch_convert.py --author "Maxwell Ramstead"

# すべてのダウンロード済みPDFを変換
python batch_convert.py
```

#### Pythonコードから変換
```python
from pdf_to_markdown import PDFToMarkdownConverter

# 変換器を初期化
converter = PDFToMarkdownConverter(
    input_dir="/path/to/pdfs",
    output_dir="/path/to/markdown"
)

# すべてのPDFを変換
results = converter.convert_all_pdfs()

# 変換結果のサマリーを取得
summary = converter.get_conversion_summary()
```

## Claude Code Plan Modeからの使用

Claude Codeのplan modeから使用できるように設計されています。

## 注意事項

- Google Scholarは頻繁なアクセスをブロックする可能性があります
- 礼儀正しいクローリングのため、リクエスト間に遅延を設けています
- PDFダウンロードは論文が公開されている場合のみ可能です
- PDF変換は`pymupdf4llm`を優先的に使用し、失敗時は`PyMuPDF`にフォールバック

## 出力構造

```
collected_papers/
├── pdfs/                    # ダウンロードしたPDFファイル
├── markdown/                # 変換されたマークダウンファイル 🆕
├── metadata/                # 各論文のメタデータ（JSON）
├── collection_summary.json  # 収集結果のサマリー
└── conversion_log.json      # PDF変換ログ 🆕
```

## 変換されるマークダウンの形式

```markdown
---
title: 論文タイトル
author: 著者名
pages: ページ数
conversion_method: pymupdf4llm
converted_at: 2025-07-30T19:55:52.326089
---

# 論文タイトル

論文の内容がマークダウン形式で出力されます...
```

## テスト

```bash
# PDF変換機能のテスト
python test_conversion.py

# Ramstead論文の検索・変換テスト
python test_ramstead_conversion.py
```