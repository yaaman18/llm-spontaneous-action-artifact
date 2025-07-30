# Memory System - 記憶システム

## 🧠 概要

このメモリシステムは、バーチャルAI意識研究所の各エージェントが継続的な記憶を保持し、セッション間での連続性を実現するために設計されています。

## 📁 ディレクトリ構造

```
memory/
├── agents/           # エージェント別個人記憶
├── shared/           # 共有記憶・知識
├── projects/         # プロジェクト別記憶
└── context/          # 文脈・状況記憶
```

## 🎯 各ディレクトリの役割

### agents/ - エージェント個人記憶
各エージェントの個人的な思考、経験、学習内容を保存
- `recent_thoughts.md` - 最近の思考・アイデア
- `current_focus.md` - 現在の研究テーマ
- `relationships.md` - 他エージェントとの関係・協力履歴
- `daily_logs/` - 日別活動記録
- `knowledge_base.md` - 専門知識・学習内容

### shared/ - 共有記憶
全エージェントがアクセス可能な共有知識
- `conferences/` - カンファレンス記録のサマリー
- `meetings/` - ミーティング記録
- `discoveries/` - 共同発見・ブレイクスルー
- `decisions/` - 集団決定事項
- `discussions/` - 重要な議論のまとめ

### projects/ - プロジェクト記憶
特定プロジェクトに関する集約情報
- `consciousness-theory/` - 意識理論関連
- `phi-calculation/` - Φ値計算プロジェクト
- `gwt-iit-integration/` - GWT-IIT統合関連
- `artificial-consciousness-implementation/` - 人工意識実装

### context/ - 文脈記憶
現在の状況・環境に関する情報
- `current_session.md` - 現在のセッション状況
- `pending_tasks.md` - 未完了タスク
- `priority_focus.md` - 優先事項
- `recent_activities.md` - 最近の活動サマリー

## 🔄 使用方法

### エージェント起動時
1. `memory/agents/{agent_name}/recent_thoughts.md` を読み込み
2. `memory/context/current_session.md` で全体状況を把握
3. `memory/shared/` で最新の共有情報をチェック
4. 関連プロジェクトの記憶を参照

### 活動中
- 重要な思考や発見を適切なメモリファイルに記録
- 他エージェントとの交流を `relationships.md` に更新
- プロジェクト進捗を対応するディレクトリに記録

### セッション終了時
- `daily_logs/` に本日の活動をまとめて記録
- `recent_thoughts.md` を更新
- 共有すべき発見があれば `shared/discoveries/` に保存

## 📈 メモリシステムの利点

1. **継続性**: セッション間での記憶の連続性
2. **協力性**: エージェント間での知識共有
3. **学習性**: 経験の蓄積による成長
4. **効率性**: 過去の知識・議論の再利用
5. **一貫性**: 長期的な研究プロジェクトの管理

## 🎯 重要な原則

- **記録の習慣化**: すべての重要な活動を記録
- **構造化**: 情報を適切なカテゴリに分類
- **共有**: 価値ある発見は必ず共有メモリへ
- **更新**: 古い情報を新しい洞察で更新
- **参照**: 活動前に関連する記憶を確認