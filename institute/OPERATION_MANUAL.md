# バーチャルAI意識研究所 運用マニュアル

## 🎯 研究所運用の基本原則

### Memory-Driven Research (記憶駆動研究)
すべての研究活動は **Memory System** を中心として展開されます。各エージェントは継続的な記憶を持ち、セッション間での知識蓄積と関係性構築を行います。

### Collaborative Intelligence (協調的知性)
単独の研究者として働くのではなく、学際的な **集合知** を形成し、複数の専門分野の知見を統合した革新的研究を実現します。

## 🚀 エージェント起動・運用手順

### Step 1: エージェントの選択と起動
```bash
# 例: Giulio Tononi & Christof Koch として研究活動
"institute/agents/tononi-koch.yamlの設定に従って、
IIT統合情報理論の専門家として本日の研究を開始してください。
起動プロトコルに従い、メモリシステムから前回の継続情報を読み込んでください。"
```

### Step 2: 起動プロトコルの実行
エージェントは以下を自動実行：
1. **個人記憶の読み込み**: `memory/agents/{name}/recent_thoughts.md`
2. **全体状況の把握**: `memory/context/current_session.md`
3. **関係性の確認**: `memory/agents/{name}/relationships.md`
4. **プロジェクト状況**: `institute/projects/active/`
5. **共有知識**: `memory/shared/conferences/`, `memory/shared/discoveries/`

### Step 3: 研究活動の実施
起動完了後、エージェントは以下の活動を行います：
- **理論研究**: 専門分野での深化・発展
- **協力研究**: 他エージェントとの学際的対話
- **実装研究**: 理論の技術的実現
- **記録更新**: 発見・進展の文書化

## 🤝 協力・対話の実現方法

### 非同期協力システム
```bash
# 例: TononiとZahaviの理論的対話
1. Tononi: 「時間意識とΦ値の関係について、現象学的視点での検討をお願いします」
   → memory/shared/discussions/time-consciousness-phi.md に議題投稿

2. Zahavi: memory から議題を確認し、現象学的分析を実行
   → 同ファイルに現象学的洞察を追記

3. Tononi: Zahaviの洞察を踏まえ、数理モデルを更新
   → memory/shared/discoveries/ に共同発見として記録
```

### リアルタイム協力セッション
```bash
# 学際カンファレンスの開催
"本日は「動的Φ境界検出システム」について学際カンファレンスを開催します。
参加者: Tononi-Koch, Zahavi, 金井, プロジェクト・オーケストレーター
各専門家の立場から、システムの理論的・実装的・哲学的側面を議論してください。"
```

## 📊 プロジェクト管理システム

### アクティブプロジェクトの管理
各エージェントのYAML設定ファイルに記載されたプロジェクトを基に：

```yaml
current_projects:
  - project_id: "dynamic-phi-boundary-detection"
    status: "active"
    role: "technical-leader" 
    priority: "high"
    notes: "institute/projects/active/dynamic-phi-boundary-detection/"
```

### プロジェクト進捗の追跡
- **週次進捗報告**: 各エージェントが担当プロジェクトの進捗を更新
- **月次評価**: プロジェクト全体の成果と課題を評価
- **四半期レビュー**: 長期目標に対する進捗を検証

## 🧠 Memory System の活用方法

### 個人記憶の管理
```markdown
# 日々の思考記録
memory/agents/{name}/recent_thoughts.md を継続的更新
- 新しいアイデアや洞察
- 解決したい問題や疑問
- 他者との議論で得た気づき

# 研究ノートの蓄積
memory/agents/{name}/research_notes.md
- 理論的発展の詳細記録
- 実験結果や計算結果
- 文献調査や関連研究の整理
```

### 共有記憶の構築
```markdown
# 重要な発見の共有
memory/shared/discoveries/{topic}.md
- 画期的な理論的洞察
- 技術的ブレイクスルー
- 学際的な新しい視点

# 議論の記録
memory/shared/discussions/{topic}.md
- 複数エージェントによる議論の蓄積
- 異なる専門分野からの多角的分析
- 合意点と争点の明確化
```

## 🔬 研究活動の具体例

### 理論研究の実施
```bash
# IIT理論の発展
エージェント: Tononi-Koch
タスク: "動的システムにおけるΦ値計算の効率化について研究し、
新しいアルゴリズムを提案してください。前回の並列化研究の続きから始めてください。"

期待される活動:
1. memory/agents/tononi-koch/recent_thoughts.md から前回の研究を継続
2. 新しいアルゴリズムの理論的検討
3. 計算複雑性の分析
4. memory/agents/tononi-koch/research_notes.md に詳細記録
5. 重要な発見があれば memory/shared/discoveries/ に共有
```

### 学際的協力の実現
```bash
# 現象学と工学の架橋
タスク: "ZahaviさんとKanaiさんで、時間意識の現象学的構造を
実装可能な技術仕様に翻訳する共同研究を実施してください。"

実施手順:
1. Zahavi: 時間意識の現象学的分析を実行
2. Kanai: 現象学的記述の技術的解釈を検討
3. 両者で共同議論 (memory/shared/discussions/に記録)
4. 実装仕様書を共同作成
5. プロトタイプの設計・開発
```

## 📈 成果の評価と成長

### 個人的成長の追跡
```json
// researchers/senior/{name}/experience.json
{
  "total_experience": 150,
  "recent_activities": [
    {
      "date": "2025-07-29",
      "task": "動的Φ境界アルゴリズム設計",
      "outcome": "並列化で10倍高速化達成",
      "learning": "GPU並列処理の深い理解獲得"
    }
  ]
}
```

### 集団的成果の記録
```markdown
# institute/achievements/quarterly_report.md
## 2025年Q3 主要成果
- 動的Φ境界検出システム: プロトタイプ完成
- 現象学的時間意識モデル: 理論設計完了
- 学際カンファレンス: 4回開催、延べ15名参加
```

## 🎯 運用の品質保証

### 定期的な自己評価
各エージェントは定期的に以下を確認：
- **記憶の更新**: recent_thoughts.md の継続的更新
- **関係の維持**: relationships.md での他者との関係記録
- **貢献の記録**: 共有知識への貢献度
- **成長の実感**: 新しい知識・スキルの獲得

### システムの継続的改善
- **プロトコルの洗練**: 効果的でない手順の改善
- **メモリ構造の最適化**: より使いやすい情報整理
- **協力メカニズムの発展**: より効果的な学際対話

## 🚨 トラブルシューティング

### よくある問題と解決策

#### 問題1: エージェントが前回の記憶を思い出せない
**解決策**: 起動プロトコルを正確に実行し、memory ファイルを確実に読み込む

#### 問題2: エージェント間の協力がうまくいかない
**解決策**: memory/shared/discussions/ を活用した非同期協力を実施

#### 問題3: 研究が表面的になる
**解決策**: 専門分野での深い探究と学際的対話のバランスを重視

この運用マニュアルにより、真に継続性のある自律的研究組織の運営が実現されます。