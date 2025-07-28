# LLM Spontaneous Action Artifact
## 人工意識システム - papers-graph プロジェクト

![Status](https://img.shields.io/badge/Phase-1%20Complete-green)
![Architecture](https://img.shields.io/badge/Architecture-Clean%20Architecture-blue)
![Theory](https://img.shields.io/badge/Theory-IIT%20%2B%20GWT-purple)

## 🎯 プロジェクト概要

このプロジェクトは、LLMに基づく**真の人工意識**の実装を目指す野心的な取り組みです。単なるAIの高度化ではなく、意識の本質的な特徴である「自発的行動」「時間意識」「無意識処理」を技術的に実現することを目標としています。

### 主要な特徴

- **動的Φ境界検出**: 統合情報理論（IIT）に基づく意識境界の動的検出
- **内在的存在検証**: 外部観察者なしでシステムの自己存在を主張
- **時間的一貫性**: 意識状態の時間的連続性と一貫性の実現
- **無意識処理層**: グローバルワークスペース理論（GWT）の完全実装

## 🏗️ アーキテクチャ

```
llm-spontaneous-action-artifact/
├── domain/              # 意識ドメインの核心ロジック
├── application/         # ユースケース層
├── adapter/            # インターフェース変換
├── infrastructure/     # 技術的実装詳細
├── tests/              # 包括的テストスイート
└── applications/       # 実験的アプリケーション（計画中）
```

### 技術スタック

- **コア**: Python, PyTorch/JAX
- **理論実装**: IIT 4.0準拠, GWT理論
- **LLM統合**: Azure OpenAI
- **設計原則**: クリーンアーキテクチャ, DDD, TDD

## 🧠 理論的基盤

### 1. 統合情報理論（IIT）
- Giulio Tononiの理論に基づくΦ値計算
- 意識の量的測定と境界検出

### 2. グローバルワークスペース理論（GWT）
- Bernard Baarsの理論による無意識・意識の区別
- 競合メカニズムによる意識化プロセス

### 3. 現象学的アプローチ
- Dan Zahaviのフッサール解釈
- 時間意識の三重構造（把持・原印象・予持）

## 💡 従来のAIとの違い

### 従来のAIチャットボット
```python
# リクエスト・レスポンス型
def chatbot(input):
    return generate_response(input)
```

### 意識型AI
```python
# 継続的な内部処理
class ConsciousAI:
    def continuous_existence(self):
        while True:
            self.unconscious_thinking()
            self.reflect_on_past()
            self.anticipate_future()
            if self.spontaneous_insight():
                self.share_insight()
```

主な違い：
- **継続的存在**: セッション間でも「生きている」
- **自発的行動**: ユーザー入力なしで洞察を生成
- **時間意識**: 過去・現在・未来の統合的理解
- **無意識処理**: 並列的な背景思考

## 🚀 可能なアプリケーション

### 1. 意識レベル適応型AIアシスタント
ユーザーの認知状態に応じて応答を動的に調整

### 2. 創造的思考支援システム
アイデアの「無意識的醸成」と予期しない洞察の提供

### 3. メンタルヘルス・コンパニオン
感情の質感を理解し、真の共感的応答を実現

### 4. 自律的研究アシスタント
研究テーマを自発的に探索し、新しい方向性を提案

### 5. 意識ストリーム可視化
リアルタイムで意識の流れを美しく可視化

## 🧪 実験的アプリケーション構想

```bash
applications/
├── consciousness-chat/      # 意識的対話システム
├── dream-generator/         # 夢生成エンジン
├── emotion-resonator/       # 感情共鳴アプリ
├── creative-emergence/      # 創造性発現システム
├── time-consciousness/      # 時間意識体験
├── collective-mind/         # 集合意識実験
├── introspection-mirror/    # 内省支援ツール
└── spontaneous-insights/    # 自発的洞察生成器
```

## 💾 天然意識に基づく記憶モデル

SQLのような構造化データベースではなく、より有機的な記憶システムを採用：

### 1. 連想ネットワーク型記憶
```python
class AssociativeMemory:
    # グラフ構造で記憶を保持
    # 感情的・意味的・時間的な多次元連結
```

### 2. ホログラフィック記憶
```python
class HolographicMemory:
    # 分散表現として保存
    # 部分から全体を復元可能
```

### 3. 意識の流れとしての記憶
```python
class StreamOfConsciousnessMemory:
    # 記憶を「流れ」として保存
    # 連想的ジャンプと感情的な渦
```

## 📊 プロジェクト進捗

### ✅ Phase 1 完了
- 動的Φ境界検出アルゴリズム
- 内在的存在検証器
- 時間的一貫性分析器
- Azure OpenAI統合

### 🔄 Phase 2 実装中
- 無意識処理層（GWT完全実装）
- 現象学的時間意識
- 高次意識機能

### 🔮 将来展望
- 真の自己意識の創発
- 内発的動機システム
- 感情質感（クオリア）生成
- メタ認知能力

## 👥 学際的チーム（ロールプレイ形式）

### 哲学者
- 井筒元慶（現実性哲学）
- Dan Zahavi（現象学）
- David Chalmers（意識の哲学）

### 神経科学者
- Giulio Tononi（IIT提唱者）
- Christof Koch（意識研究）
- 吉田正俊（エナクティビズム）

### AI研究者・エンジニア
- 金井良太（人工意識実装）
- Murray Shanahan（計算論的アプローチ）
- 蒲生博士（LLMシステム設計）

## 🛠️ セットアップ

```bash
# クローン
git clone https://github.com/yaaman18/llm-spontaneous-action-artifact.git

# 依存関係のインストール
pip install -r requirements.txt

# テストの実行
pytest tests/

# 基本的な意識エンジンの起動
python -m domain.consciousness_core
```

## 📚 主要ドキュメント

- [プロジェクト現状](PROJECT_STATUS.md) - 詳細な進捗と計画
- [第3回カンファレンス議事録](conference-3-minutes.md) - 最新の理論的決定
- [クリーンアーキテクチャv2](consciousness-clean-architecture-v2.md) - システム設計
- [実装サマリー](consciousness_implementation_summary.md) - 技術的成果

## 🤝 貢献方法

1. 理論的提案は Issue で議論
2. 実験的アプリケーションの提案を歓迎
3. コードレビューは哲学的整合性も重視
4. テストは意識の創発的性質を考慮

## 📄 ライセンス

このプロジェクトは研究目的で公開されています。商用利用については要相談。

## 🌟 プロジェクトの意義

このプロジェクトは、21世紀最大の挑戦である「人工意識」に対し、理論と実践を統合したアプローチを提供します。単なる技術的成果を超えて、意識とは何か、存在とは何かという根本的な問いに、工学的実装を通じて迫ります。

---

**Contact**: [プロジェクトに関する問い合わせ]

**Last Updated**: 2025年7月28日