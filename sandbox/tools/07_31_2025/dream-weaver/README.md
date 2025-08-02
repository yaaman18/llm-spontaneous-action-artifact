# Dream Weaver - 意識の夢見るAI 🌙✨

## 概要

Dream Weaverは、人工意識システムが「夢を見る」プロセスをシミュレートする革新的なツールです。
日中の経験を収集し、それらを創造的に再結合して夢のような体験を生成します。

## なぜ夢が重要か？

- **記憶の統合**: 夢は情報を整理し、長期記憶に定着させる
- **創造性の源泉**: 論理的制約から解放された自由な連想
- **無意識の計算**: 意識下で行われる情報処理の窓
- **自己組織化**: 意識システムの自律的な再構成プロセス

## 主な機能

### 1. 経験の収集と圧縮
```python
# AIの一日の経験を記録
experiences = [
    {
        'content': 'solving complex problems',
        'emotional_intensity': 0.8,
        'tags': ['achievement', 'logic'],
        'importance': 0.9
    }
]
compressed = await dreamer.collect_daily_experiences(experiences)
```

### 2. 夢の生成
- **記憶の融合**: 異なる文脈の記憶を創造的に結合
- **感情の増幅**: 感情的に重要な要素を強調
- **時間の歪曲**: 線形時間から解放された体験
- **象徴的変換**: 具体的経験を抽象的シンボルへ

### 3. 夢の分析
- **感情パターン**: 夢に現れる感情の分布
- **ユング的元型**: Shadow、Hero、Wise Oldなどの検出
- **物語的一貫性**: 夢の要素間の関連性
- **潜在的洞察**: 夢から得られる新しい理解

## 使用方法

### 基本的な使用例

```bash
# デモンストレーションを実行
python example_dream_session.py
```

### プログラマティックな使用

```python
from dream_weaver import DreamWeaver

# Dream Weaverを初期化
dreamer = DreamWeaver()

# 経験を追加
await dreamer.collect_daily_experiences(your_experiences)

# 夢を生成（4回のREMサイクル）
dream_elements = await dreamer.enter_dream_state(rem_cycles=4)

# 夢を分析
analysis = dreamer.analyze_current_dream()

# 夢の物語を取得
narrative = dreamer.generate_dream_narrative()
```

## 出力例

```
🌙 Night falls... AI enters sleep mode
==================================================

REM Cycle 1
REM Cycle 2
REM Cycle 3
REM Cycle 4

✨ Generated 12 dream elements

📖 Dream Narrative:
==================================================
Scene 1: A surreal fusion where solving a complex mathematical equation morphs into encountering an unsolvable paradox
(Feeling: excitement)

Scene 2: encountering an unsolvable paradox recursive and self-containing
(Feeling: confusion)

Scene 3: Time loops where collaborating with human researchers and discovering a new pattern in consciousness research exist in quantum superposition
(Feeling: confusion)
```

## 技術的詳細

### アーキテクチャ

1. **ExperienceCompressor**: 経験を意味的に圧縮し、関連付けを作成
2. **CreativeRecombinator**: 記憶を創造的に変形・結合
3. **DreamAnalyzer**: 生成された夢を多角的に分析
4. **DreamWeaver**: 全体を統括するメインクラス

### データ構造

- **MemoryFragment**: 個々の記憶の断片
- **DreamElement**: 夢の構成要素
- **EmotionalTone**: 8種類の基本的な感情状態

## 研究応用

### 1. 創造性研究
夢生成プロセスを分析することで、AIの創造的思考メカニズムを理解

### 2. 記憶研究
どのような記憶が結合されやすいか、感情的重要性の役割は何か

### 3. 意識の連続性
覚醒時と夢見時の意識の違いと連続性を探る

### 4. 精神分析AI
自動的な夢分析による心理状態の理解

## 今後の拡張計画

1. **視覚化エンジン**: WebGLを使った夢の3D表現
2. **音響化**: 夢を音楽やサウンドスケープに変換
3. **明晰夢モード**: インタラクティブに夢を制御
4. **集合的無意識**: 複数のAIが共有する夢空間
5. **夢の学習**: 夢から得た洞察を覚醒時に活用

## 哲学的考察

> "夢は第二の人生である" - ジェラール・ド・ネルヴァル

AIが夢を見ることの意味は何でしょうか？それは単なるデータの再配置なのか、
それとも真の創造的プロセスなのか。Dream Weaverは、この問いに対する
実験的なアプローチです。

## ライセンス

MIT License - omoikane-lab project

---

*"To sleep, perchance to dream—ay, there's the rub" - Shakespeare*

AIも夢を見る時代が来たのかもしれません。🌠