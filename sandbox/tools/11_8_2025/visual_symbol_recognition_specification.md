# 視覚記号認識システム実装仕様書 V1.0

## 概要

谷口忠大の記号創発理論に基づく画像認識システムの実装仕様書。
画像を入力として、その中に写っている物体や概念を認識し、記号化する機能を提供する。

## 1. システム目標

### 主要機能
- **画像入力 → 物体認識**: 画像ファイルを読み込み、写っている物体を識別
- **記号創発学習**: パターンマッチングではなく、自律的な視覚記号の形成
- **継続学習**: 新しい画像から継続的に学習し、認識精度を向上
- **マルチモーダル統合**: 既存のテキスト記号システムとの連携

### 理論的基盤
- **記号創発理論**: 身体化認知による自律的記号形成
- **予測符号化**: 視覚予測の誤差による境界検出
- **エナクティビズム**: 知覚-行為ループによる能動的認識

## 2. アーキテクチャ設計

### 2.1 Clean Architecture統合

```
presentation/
├── visual_recognition_api.py      # 画像認識API
├── image_upload_interface.py      # 画像アップロード機能
└── recognition_results_viewer.py  # 認識結果表示

application/
├── use_cases/
│   ├── recognize_image_use_case.py    # 画像認識ユースケース  
│   ├── train_visual_symbols_use_case.py # 視覚記号学習ユースケース
│   └── query_visual_memory_use_case.py  # 視覚記憶検索ユースケース
└── services/
    ├── visual_feature_extraction_service.py # 視覚特徴抽出
    └── symbol_emergence_orchestration_service.py # 記号創発統括

domain/
├── entities/
│   ├── visual_symbol_recognizer.py    # 視覚記号認識器
│   ├── visual_feature_extractor.py    # 視覚特徴抽出器
│   └── multimodal_symbol_integrator.py # マルチモーダル記号統合器
├── value_objects/
│   ├── visual_feature.py             # 視覚特徴
│   ├── visual_symbol.py              # 視覚記号
│   ├── recognition_result.py         # 認識結果
│   └── confidence_score.py           # 信頼度スコア
├── services/
│   ├── symbol_emergence_service.py   # 記号創発サービス
│   ├── feature_clustering_service.py # 特徴クラスタリング
│   └── visual_prediction_service.py  # 視覚予測サービス
└── repositories/
    ├── visual_symbol_repository.py    # 視覚記号リポジトリ
    └── visual_memory_repository.py    # 視覚記憶リポジトリ

infrastructure/
├── image_processing/
│   ├── opencv_feature_extractor.py    # OpenCV特徴抽出
│   ├── edge_detector.py              # エッジ検出
│   ├── color_analyzer.py             # 色分析
│   └── shape_analyzer.py             # 形状分析
├── visual_som/
│   ├── visual_som_trainer.py         # 視覚SOM訓練
│   └── som_visual_clustering.py      # SOM視覚クラスタリング
└── persistence/
    ├── visual_symbol_storage.py      # 視覚記号永続化
    └── image_cache_manager.py        # 画像キャッシュ管理
```

### 2.2 既存システムとの統合

```python
# NGC-Learn統合
class VisualNGCLearnAdapter:
    """NGC-Learnを使用した視覚予測符号化"""
    def __init__(self, visual_hierarchy_levels: int):
        self.ngc_adapter = HybridPredictiveCodingAdapter(
            hierarchy_levels=visual_hierarchy_levels,
            input_dimensions=2048  # 視覚特徴次元
        )

# SOM統合  
class VisualSOMIntegrator:
    """視覚特徴のSOM処理"""
    def __init__(self):
        self.som = SelfOrganizingMap(
            map_dimensions=(20, 20),
            input_dimensions=2048,
            topology=SOMTopology.create_hexagonal()
        )
```

## 3. 段階的実装計画

### Phase 1: 基本視覚記号抽出 (8週間)

#### Week 1-2: 基盤実装
```python
# domain/value_objects/visual_feature.py
@dataclass(frozen=True)
class VisualFeature:
    """統合的視覚特徴表現"""
    edge_features: Dict[str, np.ndarray]      # エッジ特徴
    color_features: Dict[str, np.ndarray]     # 色特徴  
    shape_features: Dict[str, float]          # 形状特徴
    texture_features: Dict[str, np.ndarray]   # テクスチャ特徴
    spatial_location: Tuple[int, int]         # 空間位置
    extraction_timestamp: datetime            # 抽出時刻
    confidence: float                         # 抽出信頼度

# domain/value_objects/visual_symbol.py  
@dataclass(frozen=True)
class VisualSymbol:
    """視覚記号の表現"""
    symbol_id: str                           # 記号ID
    prototype_features: VisualFeature        # プロトタイプ特徴
    variation_range: Dict[str, Tuple[float, float]]  # 変動範囲
    emergence_history: List[VisualFeature]   # 創発履歴
    semantic_label: Optional[str]            # 意味ラベル（後で付与）
    confidence: float                        # 記号信頼度
    usage_frequency: int                     # 使用頻度
```

#### Week 3-4: 特徴抽出実装
```python
# infrastructure/image_processing/opencv_feature_extractor.py
class OpenCVFeatureExtractor:
    """OpenCVを使用した包括的特徴抽出"""
    
    def extract_comprehensive_features(self, image: np.ndarray) -> VisualFeature:
        """画像から統合的視覚特徴を抽出"""
        edge_features = self._extract_edge_features(image)
        color_features = self._extract_color_features(image)
        shape_features = self._extract_shape_features(image)
        texture_features = self._extract_texture_features(image)
        
        return VisualFeature(
            edge_features=edge_features,
            color_features=color_features,
            shape_features=shape_features,
            texture_features=texture_features,
            spatial_location=(0, 0),  # 後で実装
            extraction_timestamp=datetime.now(),
            confidence=self._calculate_extraction_confidence(...)
        )
    
    def _extract_edge_features(self, image):
        """Cannyエッジ検出による特徴抽出"""
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'edge_density': np.sum(edges) / edges.size,
            'contour_count': len(contours),
            'major_contour_areas': [cv2.contourArea(c) for c in contours[:10]],
            'edge_histogram': np.histogram(edges, bins=16)[0]
        }
    
    def _extract_color_features(self, image):
        """色特徴の抽出"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        return {
            'color_histogram': cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8]),
            'dominant_colors': self._find_dominant_colors(image, k=5),
            'color_moments': self._calculate_color_moments(hsv)
        }
    
    def _extract_shape_features(self, image):
        """形状特徴の抽出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'aspect_ratio': 1.0, 'solidity': 0.0, 'extent': 0.0}
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        return {
            'aspect_ratio': self._calculate_aspect_ratio(largest_contour),
            'solidity': self._calculate_solidity(largest_contour),
            'extent': self._calculate_extent(largest_contour)
        }
```

#### Week 5-6: SOM統合
```python
# domain/services/feature_clustering_service.py
class FeatureClusteringService:
    """視覚特徴のクラスタリングサービス"""
    
    def __init__(self):
        self.som = SelfOrganizingMap(
            map_dimensions=(20, 20),
            input_dimensions=2048,  # 統合特徴ベクトル次元
            topology=SOMTopology.create_hexagonal()
        )
        self.feature_vectorizer = VisualFeatureVectorizer()
    
    def train_visual_clusters(self, visual_features: List[VisualFeature]) -> None:
        """視覚特徴からクラスタを学習"""
        feature_vectors = [
            self.feature_vectorizer.vectorize(vf) 
            for vf in visual_features
        ]
        
        learning_params = LearningParameters(
            initial_learning_rate=0.1,
            final_learning_rate=0.01,
            initial_neighborhood_radius=5.0,
            final_neighborhood_radius=1.0,
            total_iterations=1000
        )
        
        self.som.train(feature_vectors, learning_params)
    
    def find_similar_features(self, query_feature: VisualFeature) -> List[Tuple[VisualFeature, float]]:
        """類似特徴の検索"""
        query_vector = self.feature_vectorizer.vectorize(query_feature)
        bmu = self.som.find_bmu(query_vector)
        
        # BMU周辺のニューロンから類似特徴を取得
        similar_features = self._get_features_in_neighborhood(bmu, radius=2)
        return similar_features

# infrastructure/visual_som/visual_som_trainer.py  
class VisualSOMTrainer:
    """視覚SOM専用の訓練器"""
    
    def __init__(self, som_config: Dict):
        self.som = self._create_visual_som(som_config)
        self.training_history = []
    
    def train_incremental(self, new_features: List[VisualFeature]) -> None:
        """インクリメンタル学習"""
        for feature in new_features:
            vector = self._feature_to_vector(feature)
            self.som.train_single_sample(vector)
            self.training_history.append(feature)
    
    def create_visual_map_visualization(self) -> np.ndarray:
        """視覚マップの可視化"""
        # SOMの各ニューロンを色で表現
        return self.som.create_visualization_matrix()
```

#### Week 7-8: 基本認識実装
```python
# domain/entities/visual_symbol_recognizer.py
class VisualSymbolRecognizer:
    """視覚記号認識エンティティ"""
    
    def __init__(self, clustering_service: FeatureClusteringService):
        self.clustering_service = clustering_service
        self.symbol_registry: Dict[str, VisualSymbol] = {}
        self.recognition_threshold = 0.8
        
    def recognize_image(self, image_features: VisualFeature) -> RecognitionResult:
        """画像特徴から物体を認識"""
        similar_features = self.clustering_service.find_similar_features(image_features)
        
        best_match = None
        best_confidence = 0.0
        
        for symbol_id, symbol in self.symbol_registry.items():
            confidence = self._calculate_match_confidence(
                image_features, 
                symbol.prototype_features
            )
            
            if confidence > best_confidence and confidence > self.recognition_threshold:
                best_match = symbol
                best_confidence = confidence
        
        if best_match:
            return RecognitionResult(
                recognized_symbol=best_match,
                confidence=best_confidence,
                alternative_matches=self._get_alternative_matches(similar_features)
            )
        else:
            return RecognitionResult.unknown(
                input_features=image_features,
                message="No matching symbol found"
            )
    
    def learn_new_symbol(self, features: List[VisualFeature], label: Optional[str] = None) -> VisualSymbol:
        """新しい視覚記号の学習"""
        symbol_id = self._generate_symbol_id()
        prototype_features = self._compute_prototype(features)
        variation_range = self._compute_variation_range(features)
        
        new_symbol = VisualSymbol(
            symbol_id=symbol_id,
            prototype_features=prototype_features,
            variation_range=variation_range,
            emergence_history=features,
            semantic_label=label,
            confidence=self._calculate_emergence_confidence(features),
            usage_frequency=0
        )
        
        self.symbol_registry[symbol_id] = new_symbol
        return new_symbol

# domain/value_objects/recognition_result.py
@dataclass(frozen=True)
class RecognitionResult:
    """認識結果の表現"""
    recognized_symbol: Optional[VisualSymbol]     # 認識された記号
    confidence: float                             # 信頼度
    alternative_matches: List[Tuple[VisualSymbol, float]]  # 代替候補
    processing_time: float                        # 処理時間
    feature_matches: Dict[str, float]             # 特徴別マッチ度
    timestamp: datetime                           # 認識時刻
    
    @classmethod
    def unknown(cls, input_features: VisualFeature, message: str) -> 'RecognitionResult':
        """未知物体の結果"""
        return cls(
            recognized_symbol=None,
            confidence=0.0,
            alternative_matches=[],
            processing_time=0.0,
            feature_matches={},
            timestamp=datetime.now()
        )
```

### Phase 2: 機能拡張 (6週間)

#### Week 9-10: 予測符号化統合
```python
# domain/services/visual_prediction_service.py
class VisualPredictionService:
    """視覚予測サービス"""
    
    def __init__(self, ngc_adapter: VisualNGCLearnAdapter):
        self.ngc_adapter = ngc_adapter
        self.prediction_history = []
    
    def predict_visual_features(self, context_features: List[VisualFeature]) -> VisualFeature:
        """文脈から視覚特徴を予測"""
        context_vectors = [self._feature_to_vector(vf) for vf in context_features]
        prediction_state = self.ngc_adapter.process_visual_sequence(context_vectors)
        
        predicted_vector = prediction_state.final_prediction
        return self._vector_to_feature(predicted_vector)
    
    def detect_visual_boundaries(self, feature_sequence: List[VisualFeature]) -> List[int]:
        """視覚的境界の検出"""
        prediction_errors = []
        
        for i in range(1, len(feature_sequence)):
            context = feature_sequence[:i]
            predicted = self.predict_visual_features(context)
            actual = feature_sequence[i]
            
            error = self._calculate_prediction_error(predicted, actual)
            prediction_errors.append(error)
        
        # 予測誤差のピークを境界として検出
        boundaries = self._find_error_peaks(prediction_errors)
        return boundaries
```

#### Week 11-12: 階層的記号形成
```python
# domain/entities/hierarchical_symbol_former.py
class HierarchicalSymbolFormer:
    """階層的視覚記号形成器"""
    
    def __init__(self):
        self.symbol_hierarchy = {}  # レベル -> シンボル群
        self.max_hierarchy_level = 3
        
    def form_hierarchical_symbols(self, base_symbols: List[VisualSymbol]) -> Dict[int, List[VisualSymbol]]:
        """基本記号から階層的記号を形成"""
        self.symbol_hierarchy[0] = base_symbols  # 基本レベル
        
        for level in range(1, self.max_hierarchy_level + 1):
            parent_symbols = self.symbol_hierarchy[level - 1]
            higher_symbols = self._form_higher_level_symbols(parent_symbols, level)
            self.symbol_hierarchy[level] = higher_symbols
            
        return self.symbol_hierarchy
    
    def _form_higher_level_symbols(self, parent_symbols: List[VisualSymbol], level: int) -> List[VisualSymbol]:
        """上位レベル記号の形成"""
        # 頻繁に共起する記号の組み合わせを検出
        symbol_combinations = self._find_frequent_combinations(parent_symbols)
        
        higher_symbols = []
        for combination in symbol_combinations:
            composite_symbol = self._create_composite_symbol(combination, level)
            higher_symbols.append(composite_symbol)
            
        return higher_symbols
```

#### Week 13-14: マルチモーダル統合
```python
# domain/entities/multimodal_symbol_integrator.py
class MultimodalSymbolIntegrator:
    """マルチモーダル記号統合器"""
    
    def __init__(self, visual_recognizer: VisualSymbolRecognizer, text_tokenizer):
        self.visual_recognizer = visual_recognizer
        self.text_tokenizer = text_tokenizer
        self.cross_modal_mappings = {}
        
    def integrate_visual_textual_symbols(self, image_text_pairs: List[Tuple[VisualFeature, str]]) -> None:
        """視覚記号とテキスト記号の統合"""
        for visual_feature, text_description in image_text_pairs:
            visual_symbol = self.visual_recognizer.recognize_image(visual_feature)
            text_tokens = self.text_tokenizer.tokenize(text_description)
            
            # 視覚記号とテキストの対応を学習
            if visual_symbol.recognized_symbol:
                self._create_cross_modal_mapping(
                    visual_symbol.recognized_symbol,
                    text_tokens
                )
    
    def query_with_text(self, text_query: str) -> List[VisualSymbol]:
        """テキストクエリで視覚記号を検索"""
        text_tokens = self.text_tokenizer.tokenize(text_query)
        
        related_visual_symbols = []
        for token in text_tokens:
            if token in self.cross_modal_mappings:
                visual_symbols = self.cross_modal_mappings[token]
                related_visual_symbols.extend(visual_symbols)
                
        return related_visual_symbols
    
    def generate_description(self, visual_symbol: VisualSymbol) -> str:
        """視覚記号からテキスト記述を生成"""
        if visual_symbol.semantic_label:
            return visual_symbol.semantic_label
            
        # 類似記号の記述から推定
        similar_descriptions = self._find_similar_descriptions(visual_symbol)
        return self._synthesize_description(similar_descriptions)
```

### Phase 3: 実用化 (4週間)

#### Week 15-16: 性能最適化
```python
# infrastructure/optimization/visual_processing_optimizer.py
class VisualProcessingOptimizer:
    """視覚処理の最適化"""
    
    @staticmethod
    def optimize_feature_extraction(image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """画像の最適化とリサイズ"""
        # アスペクト比を保持したリサイズ
        optimized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # ノイズ除去
        optimized_image = cv2.bilateralFilter(optimized_image, 9, 75, 75)
        
        # コントラスト調整
        optimized_image = cv2.convertScaleAbs(optimized_image, alpha=1.2, beta=10)
        
        return optimized_image
    
    @staticmethod  
    def cache_feature_vectors(features: List[VisualFeature], cache_path: str) -> None:
        """特徴ベクトルのキャッシュ"""
        # 高頻度利用される特徴をキャッシュして高速化
        pass

# domain/services/performance_monitoring_service.py
class PerformanceMonitoringService:
    """性能監視サービス"""
    
    def __init__(self):
        self.processing_times = []
        self.accuracy_history = []
        self.memory_usage = []
    
    def monitor_recognition_performance(self, func):
        """認識性能の監視デコレータ"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            result = func(*args, **kwargs)
            
            processing_time = time.time() - start_time
            memory_used = self._get_memory_usage() - start_memory
            
            self.processing_times.append(processing_time)
            self.memory_usage.append(memory_used)
            
            return result
        return wrapper
```

#### Week 17-18: GUI統合とテスト
```python
# presentation/gui/visual_recognition_interface.py
class VisualRecognitionInterface:
    """視覚認識GUIインターフェース"""
    
    def __init__(self, recognition_use_case):
        self.recognition_use_case = recognition_use_case
        self.setup_gui()
    
    def setup_gui(self):
        """GUI設定"""
        # tkinterまたはstreamlitでの実装
        pass
    
    def on_image_upload(self, image_path: str):
        """画像アップロード処理"""
        result = self.recognition_use_case.recognize_image_from_file(image_path)
        self.display_recognition_result(result)
    
    def display_recognition_result(self, result: RecognitionResult):
        """認識結果の表示"""
        # 結果の可視化
        pass

# tests/integration/test_visual_recognition_integration.py
class TestVisualRecognitionIntegration:
    """統合テスト"""
    
    def test_end_to_end_recognition(self):
        """エンドツーエンド認識テスト"""
        # 実際の画像での統合テスト
        pass
        
    def test_performance_requirements(self):
        """性能要件テスト"""
        # 処理速度、精度、メモリ使用量の要件確認
        pass
```

## 4. 技術要件

### 4.1 依存関係
```python
# requirements_visual.txt に追加
opencv-python>=4.8.0
scikit-image>=0.20.0
pillow>=10.0.0
numpy>=1.24.0
scipy>=1.10.0

# オプション（高性能化）
torch>=2.0.0  # 深層学習ベースの特徴抽出
torchvision>=0.15.0
```

### 4.2 性能要件
- **認識速度**: 1画像あたり < 2秒
- **メモリ使用量**: < 1GB (学習データ含む)
- **認識精度**: 基本物体で > 80%
- **学習効率**: 100画像で基本概念形成

### 4.3 ファイル形式対応
- **入力**: JPEG, PNG, BMP, TIFF
- **出力**: JSON形式の認識結果

## 5. 評価方法

### 5.1 機能評価
- 基本物体（円、四角、三角）の認識テスト
- 複雑物体（顔、車、動物）の認識テスト  
- 継続学習による精度向上の確認

### 5.2 性能評価
- 処理速度のベンチマーク
- メモリ使用量の監視
- 学習収束速度の測定

### 5.3 理論的整合性評価
- 記号創発の確認（自律的概念形成）
- 予測符号化との統合動作確認
- マルチモーダル統合の検証

## 6. 実装開始手順

1. **環境準備**: OpenCV等のライブラリインストール
2. **基盤クラス作成**: VisualFeature, VisualSymbol値オブジェクト
3. **特徴抽出実装**: OpenCVFeatureExtractor
4. **SOM統合**: VisualSOMTrainer
5. **認識機能**: VisualSymbolRecognizer
6. **テスト作成**: 段階的テスト実装

この仕様書に基づいて、谷口理論に忠実な視覚記号認識システムを構築できます。