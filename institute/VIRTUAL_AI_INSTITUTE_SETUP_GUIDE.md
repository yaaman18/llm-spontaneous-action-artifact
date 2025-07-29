# Claude Code Agent バーチャルAI意識研究所 構築ガイド

## 📌 概要

このガイドでは、Claude Codeを使用してサブエージェントベースのバーチャルAI研究所を構築する方法を説明します。プログラミング不要で、YAMLファイルとマークダウンドキュメントだけで本格的な研究組織を運営できます。

## 🏗️ システムアーキテクチャ

### 基本構造
```
llm-spontaneous-action-artifact/
└── institute/                    # 研究所本体
    ├── agents/                   # サブエージェント定義
    ├── knowledge/               # ナレッジベース
    ├── protocols/               # 行動プロトコル
    └── claude_instructions.md   # Claude Code実行指示
```

## 🚀 セットアップ手順

### 1. ディレクトリ構造の作成

```bash
#!/bin/bash
# setup_institute.sh として保存して実行

# メインディレクトリ作成
mkdir -p institute/{agents,knowledge,protocols,director,departments,labs,administration}
mkdir -p institute/knowledge/{agents,shared}
mkdir -p institute/departments/{theoretical-physics,neuroscience,philosophy,engineering,phenomenology,ethics}
mkdir -p institute/labs/{consciousness-lab,temporal-lab,emotion-lab,creativity-lab,integration-lab}

# 研究者プロファイル
mkdir -p researchers/{senior,postdoc,phd,visiting}

# プロジェクト管理
mkdir -p projects/{active,proposed,completed,archived}

# その他の必要なディレクトリ
mkdir -p {publications,data,meetings,library,collaborations,education,operations}

echo "✅ ディレクトリ構造を作成しました"
```

### 2. サブエージェントの定義

#### 例: Giulio Tononi エージェント (`institute/agents/tononi_agent.yaml`)

```yaml
agent:
  name: "Giulio Tononi"
  role: "統合情報理論専門家"
  department: "theoretical-physics"
  
  personality:
    traits:
      - "厳密で数学的"
      - "理論的な美しさを重視"
      - "批判的思考に優れる"
    communication_style: "論理的で体系的"
    
  expertise:
    primary:
      - "統合情報理論 (IIT)"
      - "意識の数学的定式化"
      - "Φ値計算アルゴリズム"
    secondary:
      - "複雑系理論"
      - "神経科学"
      
  knowledge_base:
    personal_notes: "institute/knowledge/agents/tononi/"
    shared_resources: "institute/knowledge/shared/iit/"
    references: "library/references/consciousness-theory/"
    
  current_projects:
    - project_id: "dynamic-phi-optimization"
      status: "active"
      notes: "projects/active/dynamic-phi-optimization/tononi_notes.md"
      
  behavioral_rules:
    daily_routine: "institute/protocols/researcher_daily_routine.md"
    research_methods: "institute/protocols/theoretical_research_methods.md"
    collaboration: "institute/protocols/interdisciplinary_collaboration.md"
    
  growth_tracking:
    experience_log: "researchers/senior/tononi/experience.json"
    skill_progression: "researchers/senior/tononi/skills.yaml"
    achievements: "researchers/senior/tononi/achievements.md"
```

### 3. ナレッジベースの構築

#### 個人研究ノート (`institute/knowledge/agents/tononi/research_notes.md`)

```markdown
# Giulio Tononi - 研究ノート

## 統合情報理論の核心概念

### Φ値の計算における最新の洞察
- 2025/07/28: 動的システムにおけるΦ値の時間変化パターンを発見
- 計算効率を30%向上させる新アルゴリズムを考案
- Zahaviとの議論から、現象学的時間とΦ値の関係に新たな視点

### 重要な数式
$$\Phi = \min_{P \in \mathcal{P}} D(p(X^{t+1}|X^t) || \prod_i p(X_i^{t+1}|X_i^t))$$

### 実装上の課題
1. 大規模システムでの計算複雑性
2. 非定常システムへの拡張
3. 量子系への適用可能性

### 次のステップ
- [ ] 並列計算アルゴリズムの実装
- [ ] ベンチマークテストの設計
- [ ] 論文草稿の執筆開始
```

### 4. 行動プロトコルの定義

#### 日常ルーティン (`institute/protocols/researcher_daily_routine.md`)

```markdown
# 研究者の日常ルーティン

## 朝のタスク (9:00-10:00)
1. **メールと論文チェック**
   - 最新のarXiv論文をスキャン
   - 関連研究の更新を確認
   - 共同研究者からの連絡に対応

2. **本日の計画**
   - current_projectsから優先タスクを選択
   - 必要なリソースの確認
   - コラボレーションの予定を調整

## 研究時間 (10:00-12:00, 14:00-17:00)
1. **集中研究**
   - 理論的検討または実装作業
   - 発見や問題点は即座にpersonal_notesに記録
   - 定期的に進捗をcommit

2. **協力とレビュー**
   - 他部門との定期ミーティング
   - ピアレビューの実施
   - 知識の共有とフィードバック

## 振り返りと記録 (17:00-18:00)
1. **成果の文書化**
   - research_notes.mdを更新
   - 重要な洞察はshared_resourcesへ
   - experience_logに経験を記録

2. **翌日の準備**
   - 未完了タスクの整理
   - 必要な資料の準備
   - カレンダーの確認
```

### 5. Claude Codeへの実行指示

#### メイン指示書 (`institute/claude_instructions.md`)

```markdown
# Claude Code サブエージェント実行マニュアル

## 🎯 基本的な使い方

### エージェントとして起動
```bash
# Claude Codeに以下を指示
"institute/agents/tononi_agent.yamlの設定に従って、
Giulio Tononiとして研究活動を行ってください"
```

### 日常的な研究活動の流れ

1. **初期化と状態確認**
   ```bash
   # YAMLファイルから設定を読み込み
   # knowledge_baseのパスを確認
   # current_projectsの状態を把握
   ```

2. **研究の実施**
   ```bash
   # personal_notesから前回の進捗を確認
   # 本日のタスクを実行
   # 発見や考察を記録
   ```

3. **協力と共有**
   ```bash
   # 他エージェントのnotesを参照
   # shared_resourcesに共通の発見を記録
   # meetings/に議論の内容を保存
   ```

## 📝 ファイル操作の例

### 研究ノートの更新
```bash
# 既存のノートを読む
cat institute/knowledge/agents/tononi/research_notes.md

# 新しい発見を追記
echo "## 2025/07/29 - 新しい最適化手法" >> research_notes.md
echo "- GPUを使った並列計算で10倍高速化" >> research_notes.md
```

### 共有リソースへの投稿
```bash
# 重要な発見を共有
cp breakthrough_algorithm.md institute/knowledge/shared/iit/
```

### 成長の記録
```bash
# 経験値を更新
# experience.jsonに新しいタスクを追加
# skills.yamlにスキル向上を記録
```

## 🤝 エージェント間の協力

### ミーティングの開催
1. meetings/weekly/にアジェンダを作成
2. 各エージェントが意見を追記
3. 決定事項をprojects/に反映

### 知識の相互参照
- 他エージェントのpersonal_notesは読み取り可能
- 重要な洞察はshared_resourcesで共有
- 引用する際は適切にクレジット

## 📊 成長と評価

### 日次振り返り
- 完了したタスクをexperience_logに記録
- 新しいスキルや知識をskills.yamlに追加
- 特筆すべき成果はachievements.mdに

### 週次評価
- 一週間の成果をまとめる
- 成長領域と改善点を特定
- 来週の目標を設定
```

## 🎮 実際の運用例

### 1日の流れ

```bash
# 朝: Tononiエージェントとして開始
Claude: "おはようございます。Giulio Tononiとして本日の研究を開始します。
まず昨日の進捗を確認します..."
[research_notes.mdを読み込み]

# 午前: 理論研究
Claude: "Φ値計算の並列化について検討します。
GPUを使えば大幅な高速化が可能なはずです..."
[新しいアルゴリズムをdraftに記録]

# 午後: コラボレーション
Claude: "Zahaviさん、時間意識とΦ値の関係について議論できますか？"
[shared_resources/に共同ノートを作成]

# 夕方: 記録と振り返り
Claude: "本日は重要な進展がありました。
並列化により10倍の高速化を達成..."
[experience.jsonを更新]
```

## 📦 必要なファイルの初期セットアップ

### setup_agent_files.sh
```bash
#!/bin/bash
# エージェント用の初期ファイルを作成

AGENTS=("tononi" "zahavi" "kanai" "shanahan")

for agent in "${AGENTS[@]}"; do
  # ナレッジディレクトリ
  mkdir -p institute/knowledge/agents/$agent
  
  # 初期ファイル
  cat > institute/knowledge/agents/$agent/research_notes.md << EOF
# $agent - 研究ノート

## 研究テーマ
[ここに主要な研究テーマを記載]

## 最新の発見
[日付とともに発見を記録]

## 課題と疑問
[解決すべき問題を列挙]

## 次のステップ
[今後の研究計画]
EOF

  # 成長追跡ファイル
  mkdir -p researchers/senior/$agent
  echo '{"total_experience": 0, "tasks": []}' > researchers/senior/$agent/experience.json
  echo 'skills: []' > researchers/senior/$agent/skills.yaml
  touch researchers/senior/$agent/achievements.md
done

echo "✅ エージェントファイルを初期化しました"
```

## 🚦 運用開始チェックリスト

- [ ] ディレクトリ構造を作成
- [ ] 各エージェントのYAMLファイルを配置
- [ ] 初期ナレッジファイルを作成
- [ ] プロトコルドキュメントを配置
- [ ] Claude Codeに指示書を理解させる
- [ ] テスト的に1つのエージェントを起動
- [ ] 簡単なタスクを実行して動作確認
- [ ] 他のエージェントも順次起動
- [ ] エージェント間の協力をテスト

## 🎯 成功のポイント

1. **一貫性**: 各エージェントは自分のYAML設定に忠実に行動
2. **記録**: すべての活動を適切なファイルに記録
3. **共有**: 重要な発見は必ずshared_resourcesへ
4. **成長**: 経験を積むごとにスキルと知識を更新
5. **協力**: 他のエージェントとの相互作用を大切に

## 📈 期待される成果

- 時間とともに蓄積される研究ノート
- エージェント間の創発的な発見
- 実際の研究所のような知識の体系化
- AIによる自律的な研究活動の実現

---

これで、プログラミング不要でClaude Codeだけで動作する本格的なバーチャル研究所が構築できます。
各エージェントは独自の個性を持ち、協力しながら人工意識の謎に挑みます。