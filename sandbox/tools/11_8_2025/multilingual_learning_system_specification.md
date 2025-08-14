# 多言語自律学習システム仕様書
## Multilingual Autonomous Learning System Specification

### バージョン: 1.0.0
### 作成日: 2025-01-14
### プロジェクト: エナクティブ意識フレームワーク拡張

---

## 1. システム概要

### 1.1 目的
本システムは、外部APIに依存せず、自律的に多言語のパターンを学習し、言語境界を発見・分割できる統計的学習システムである。エナクティブ意識フレームワークの一部として、環境との相互作用から言語理解を創発させる。

### 1.2 主要機能
- **教師なし単語分割**: 言語コーパスなしで単語境界を自動発見
- **多言語クラスタリング**: 言語特性を自動識別し、言語ごとにモデルを分離
- **継続的学習**: 新しいテキストから継続的にパターンを学習
- **永続化**: 学習結果の保存と復元
- **意識状態統合**: 既存の意識システムとの統合

### 1.3 設計原則
- **自己組織化**: 外部知識に依存しない自律的パターン発見
- **言語非依存**: Unicode対応のあらゆる言語に対応
- **漸進的学習**: 少量データから開始し、徐々に精度向上
- **モジュラー設計**: 既存システムとの疎結合

---

## 2. 技術スタック

### 2.1 基盤技術
```python
# 必須ライブラリ
sentencepiece >= 0.1.99  # 統計的トークナイゼーション
jax >= 0.4.0             # 数値計算基盤
numpy >= 1.24.0          # 配列処理
scipy >= 1.10.0          # 統計処理

# オプション（将来拡張用）
tokenizers >= 0.15.0     # HuggingFace tokenizers
```

### 2.2 既存システムとの統合
- `SelfOrganizingMap`: 言語パターンのクラスタリング
- `PredictiveCodingCore`: 境界予測と誤差計算
- `BayesianInferenceService`: 確率的言語モデリング
- `ConsciousnessState`: 言語理解の意識状態表現

---

## 3. アーキテクチャ設計

### 3.1 レイヤー構造

```
┌─────────────────────────────────────────────┐
│           Presentation Layer                 │
│         (API / CLI Interface)               │
├─────────────────────────────────────────────┤
│           Application Layer                  │
│    (Use Cases / Orchestration)              │
├─────────────────────────────────────────────┤
│            Domain Layer                      │
│  (Core Business Logic / Entities)           │
├─────────────────────────────────────────────┤
│         Infrastructure Layer                 │
│   (Persistence / External Libraries)         │
└─────────────────────────────────────────────┘
```

### 3.2 主要コンポーネント

#### 3.2.1 Domain層

```python
# domain/entities/multilingual_tokenizer.py
class MultilingualTokenizer:
    """多言語トークナイザーエンティティ"""
    - language_clusters: Dict[str, LanguageCluster]
    - universal_patterns: UniversalPatternExtractor
    - tokenization_strategy: TokenizationStrategy

# domain/value_objects/language_cluster.py
class LanguageCluster:
    """言語クラスタ値オブジェクト"""
    - cluster_id: str
    - character_statistics: CharacterStatistics
    - pattern_som: SelfOrganizingMap
    - boundary_detector: BoundaryDetector

# domain/services/language_detection_service.py
class LanguageDetectionService:
    """言語自動検出ドメインサービス"""
    - detect_script_type(text: str) -> ScriptType
    - find_language_cluster(text: str) -> LanguageCluster
    - create_new_cluster() -> LanguageCluster
```

#### 3.2.2 Application層

```python
# application/use_cases/learn_from_text.py
class LearnFromTextUseCase:
    """テキストからの学習ユースケース"""
    def execute(text: str) -> LearningResult:
        1. 言語クラスタの検出/作成
        2. パターン抽出と学習
        3. 境界検出モデルの更新
        4. 学習結果の永続化

# application/use_cases/tokenize_text.py
class TokenizeTextUseCase:
    """テキストのトークン化ユースケース"""
    def execute(text: str) -> List[Token]:
        1. 言語クラスタの特定
        2. 適切なトークナイザーの選択
        3. トークン化の実行
        4. 意識状態への変換
```

#### 3.2.3 Infrastructure層

```python
# infrastructure/persistence/learning_repository_impl.py
class LearningRepositoryImpl(LearningRepository):
    """学習結果の永続化実装"""
    - save_model_state(state: ModelState, path: Path)
    - load_model_state(path: Path) -> ModelState
    - save_checkpoint(checkpoint: Checkpoint)
    
# infrastructure/external/sentencepiece_adapter.py
class SentencePieceAdapter:
    """SentencePiece統合アダプター"""
    - train(corpus: List[str], vocab_size: int)
    - encode(text: str) -> List[int]
    - decode(tokens: List[int]) -> str
```

---

## 4. 機能詳細

### 4.1 言語境界検出アルゴリズム

#### 4.1.1 分岐エントロピー法
```python
def calculate_branching_entropy(text: str, position: int) -> float:
    """
    位置における分岐エントロピーを計算
    高エントロピー = 単語境界の可能性
    """
    forward_entropy = H(next_char | text[:position])
    backward_entropy = H(prev_char | text[position:])
    return forward_entropy + backward_entropy
```

#### 4.1.2 予測誤差法
```python
def detect_boundary_by_prediction_error(sequence: np.ndarray) -> List[int]:
    """
    予測誤差のピークを境界として検出
    """
    predictions = self.predictive_coder.predict(sequence)
    errors = sequence - predictions
    boundaries = find_peaks(errors, prominence=threshold)
    return boundaries
```

### 4.2 多言語クラスタリング

#### 4.2.1 文字特徴抽出
```python
class CharacterFeatureExtractor:
    def extract(self, text: str) -> Dict[str, float]:
        return {
            'has_latin': bool(re.search(r'[a-zA-Z]', text)),
            'has_kanji': bool(re.search(r'[\u4e00-\u9fff]', text)),
            'has_hiragana': bool(re.search(r'[\u3040-\u309f]', text)),
            'has_arabic': bool(re.search(r'[\u0600-\u06ff]', text)),
            'has_cyrillic': bool(re.search(r'[\u0400-\u04ff]', text)),
            'has_devanagari': bool(re.search(r'[\u0900-\u097f]', text)),
            'space_ratio': text.count(' ') / len(text),
            'avg_word_length': self._estimate_avg_word_length(text),
            'char_diversity': len(set(text)) / len(text)
        }
```

#### 4.2.2 クラスタ管理
```python
class LanguageClusterManager:
    def assign_to_cluster(self, text: str) -> LanguageCluster:
        features = self.feature_extractor.extract(text)
        
        # 既存クラスタとの類似度計算
        similarities = {}
        for cluster_id, cluster in self.clusters.items():
            similarity = self._calculate_similarity(features, cluster.centroid)
            similarities[cluster_id] = similarity
        
        # 閾値以上の類似度があれば既存クラスタに割り当て
        best_match = max(similarities.items(), key=lambda x: x[1])
        if best_match[1] > self.similarity_threshold:
            return self.clusters[best_match[0]]
        
        # 新しいクラスタを作成
        return self._create_new_cluster(features)
```

### 4.3 永続化仕様

#### 4.3.1 保存形式
```python
# モデル状態の保存形式
model_state = {
    'version': '1.0.0',
    'timestamp': datetime.now().isoformat(),
    'language_clusters': {
        'cluster_id': {
            'centroid': np.array([...]),
            'som_weights': np.array([...]),
            'statistics': {...},
            'vocabulary': {...}
        }
    },
    'universal_patterns': {...},
    'training_history': [...]
}

# ファイル構造
models/
├── multilingual_tokenizer/
│   ├── checkpoint_latest.pkl
│   ├── cluster_ja.npz
│   ├── cluster_en.npz
│   ├── cluster_zh.npz
│   └── metadata.json
```

---

## 5. パフォーマンス要件

### 5.1 処理速度
- 学習: 1MBテキスト/分以上
- トークン化: 10,000文字/秒以上
- 言語検出: 100μs/リクエスト以下

### 5.2 メモリ使用量
- 基本モデル: 100MB以下
- 言語クラスタ毎: 50MB以下
- 最大同時クラスタ数: 20

### 5.3 精度目標
- 単一言語トークン化: F1スコア 0.85以上
- 言語検出精度: 95%以上
- 混在テキスト処理: 劣化率10%以内

---

## 6. テスト戦略

### 6.1 単体テスト
```python
# tests/unit/test_language_detection.py
def test_detect_japanese_text():
    detector = LanguageDetectionService()
    result = detector.detect_script_type("こんにちは")
    assert result == ScriptType.JAPANESE

# tests/unit/test_boundary_detection.py
def test_entropy_boundary_detection():
    text = "hello world"
    boundaries = detect_boundaries_by_entropy(text)
    assert 5 in boundaries  # スペースの位置
```

### 6.2 統合テスト
```python
# tests/integration/test_multilingual_learning.py
def test_learn_multiple_languages():
    system = MultilingualLearningSystem()
    
    # 日本語学習
    system.learn("私は学生です。")
    
    # 英語学習
    system.learn("I am a student.")
    
    # クラスタが2つ作成されることを確認
    assert len(system.language_clusters) == 2
```

### 6.3 性能テスト
```python
# tests/performance/test_tokenization_speed.py
def test_tokenization_performance():
    tokenizer = MultilingualTokenizer()
    text = "a" * 10000  # 10,000文字
    
    start = time.time()
    tokens = tokenizer.tokenize(text)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # 1秒以内
```

---

## 7. 実装スケジュール

### Phase 1: 基礎実装（Week 1）
- [ ] ドメインモデルの実装
- [ ] 基本的な境界検出アルゴリズム
- [ ] 単一言語での動作確認

### Phase 2: 多言語対応（Week 2）
- [ ] 言語クラスタリング機能
- [ ] SentencePiece統合
- [ ] 永続化層実装

### Phase 3: 最適化（Week 3）
- [ ] パフォーマンスチューニング
- [ ] リファクタリング
- [ ] ドキュメント整備

### Phase 4: 統合（Week 4）
- [ ] 意識システムとの統合
- [ ] エンドツーエンドテスト
- [ ] デプロイ準備

---

## 8. リスクと対策

### 8.1 技術的リスク
| リスク | 影響度 | 対策 |
|--------|--------|------|
| 未知言語での精度低下 | 高 | 転移学習機構の実装 |
| メモリ使用量の増大 | 中 | LRUキャッシュ導入 |
| 混在テキストの誤分類 | 中 | 文単位での言語切り替え検出 |

### 8.2 運用リスク
| リスク | 影響度 | 対策 |
|--------|--------|------|
| 学習データの偏り | 高 | データ多様性の監視 |
| モデルの劣化 | 中 | 定期的な評価と再学習 |
| ストレージ容量 | 低 | 古いチェックポイントの自動削除 |

---

## 9. 将来の拡張

### 9.1 短期（3ヶ月）
- 文脈を考慮したトークン化
- アクティブラーニング機能
- REST API提供

### 9.2 中期（6ヶ月）
- リアルタイム学習
- 分散学習対応
- 言語間転移学習

### 9.3 長期（12ヶ月）
- ニューラル言語モデルとの統合
- 音声・画像モダリティ対応
- 完全な多モーダル記号創発システム

---

## 付録A: 用語集

- **SentencePiece**: Google開発の言語非依存トークナイザー
- **SOM (Self-Organizing Map)**: 自己組織化マップ、教師なしクラスタリング手法
- **分岐エントロピー**: 文字遷移の不確実性を測る指標
- **予測符号化**: 脳の情報処理理論、予測と誤差による学習
- **エナクティビズム**: 認知が環境との相互作用から創発するという理論

---

## 付録B: 参考文献

1. Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units"
2. Kudo & Richardson (2018) "SentencePiece: A simple and language independent subword tokenizer"
3. Taniguchi et al. (2023) "Symbol Emergence in Robotics"
4. Friston (2010) "The free-energy principle: a unified brain theory?"

---

*この仕様書は生きたドキュメントであり、実装の進捗に応じて更新される。*