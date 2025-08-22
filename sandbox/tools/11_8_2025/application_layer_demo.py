"""
アプリケーション層使用例

Clean ArchitectureとDDD原則に従った視覚記号認識システムの
アプリケーション層使用例。谷口忠大の記号創発理論の実装を示す。
"""

import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# アプリケーション層のインポート
from application import (
    RecognizeImageUseCase,
    TrainVisualSymbolsUseCase,
    QueryVisualMemoryUseCase,
    VisualFeatureExtractionService,
    SymbolEmergenceOrchestrationService,
    ImageRecognitionRequest,
    SymbolLearningRequest,
    VisualMemoryQueryRequest,
    QueryType,
    SortOrder
)

# ドメイン層のインポート
from domain.entities.visual_symbol_recognizer import VisualSymbolRecognizer
from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.visual_symbol import VisualSymbol

# インフラ層の代替実装（Mock）
from application.services.visual_feature_extraction_service import IFeatureExtractor
from application.services.symbol_emergence_orchestration_service import ISymbolRepository


class MockFeatureExtractor(IFeatureExtractor):
    """特徴抽出器のモック実装"""
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """模擬特徴抽出"""
        height, width = image.shape[:2]
        
        # エッジ特徴のモック
        edge_density = np.random.uniform(0.1, 0.8)
        edge_histogram = np.random.random(32)
        contour_count = np.random.randint(1, 15)
        
        # 色特徴のモック
        color_histogram = np.random.random(64)
        dominant_colors = np.random.randint(0, 255, (3, 3))
        
        # 形状特徴のモック
        aspect_ratio = width / height
        solidity = np.random.uniform(0.7, 1.0)
        extent = np.random.uniform(0.5, 0.9)
        circularity = np.random.uniform(0.1, 1.0)
        
        return {
            'edge_features': {
                'edge_density': edge_density,
                'edge_histogram': edge_histogram,
                'contour_count': contour_count
            },
            'color_features': {
                'color_histogram': color_histogram,
                'dominant_colors': dominant_colors
            },
            'shape_features': {
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'extent': extent,
                'circularity': circularity,
                'rectangularity': np.random.uniform(0.3, 0.8),
                'compactness': np.random.uniform(0.4, 0.9)
            },
            'texture_features': {}
        }


class MockSymbolRepository(ISymbolRepository):
    """記号リポジトリのモック実装"""
    
    def __init__(self):
        self.symbols: Dict[str, VisualSymbol] = {}
        self.id_counter = 1
    
    def save_symbol(self, symbol: VisualSymbol) -> str:
        """記号の保存"""
        if symbol.symbol_id not in self.symbols:
            symbol_id = symbol.symbol_id or f"mock_symbol_{self.id_counter}"
            self.id_counter += 1
        else:
            symbol_id = symbol.symbol_id
        
        self.symbols[symbol_id] = symbol
        return symbol_id
    
    def find_symbol_by_id(self, symbol_id: str) -> VisualSymbol:
        """IDによる記号検索"""
        return self.symbols.get(symbol_id)
    
    def find_similar_symbols(self, feature: VisualFeature, threshold: float = 0.8) -> List[tuple]:
        """類似記号の検索"""
        similar_symbols = []
        
        for symbol in self.symbols.values():
            similarity = symbol.prototype_features.calculate_similarity(feature)
            if similarity >= threshold:
                similar_symbols.append((symbol, similarity))
        
        # 類似度の降順でソート
        similar_symbols.sort(key=lambda x: x[1], reverse=True)
        return similar_symbols
    
    def get_all_symbols(self) -> List[VisualSymbol]:
        """全記号の取得"""
        return list(self.symbols.values())


def create_sample_image() -> np.ndarray:
    """サンプル画像の作成"""
    # 100x100のランダム画像
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 簡単な図形を描画（円）
    center = (50, 50)
    radius = 20
    y, x = np.ogrid[:100, :100]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    image[mask] = [255, 0, 0]  # 赤い円
    
    return image


def create_sample_visual_features(complexity: float = 0.6) -> List[VisualFeature]:
    """サンプル視覚特徴の作成"""
    features = []
    
    for i in range(3):
        # 特徴パラメータの生成
        edge_density = np.random.uniform(0.3, 0.7) * complexity
        edge_histogram = np.random.random(32) * complexity
        contour_count = int(np.random.uniform(3, 10) * complexity)
        
        color_histogram = np.random.random(64) * complexity
        dominant_colors = np.random.randint(0, 255, (2, 3))
        
        aspect_ratio = np.random.uniform(0.8, 1.2)
        solidity = np.random.uniform(0.7, 1.0) * complexity
        extent = np.random.uniform(0.6, 0.9) * complexity
        
        feature = VisualFeature(
            edge_features={
                'edge_density': edge_density,
                'edge_histogram': edge_histogram,
                'contour_count': contour_count
            },
            color_features={
                'color_histogram': color_histogram,
                'dominant_colors': dominant_colors
            },
            shape_features={
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'extent': extent,
                'circularity': np.random.uniform(0.4, 0.8),
                'rectangularity': np.random.uniform(0.3, 0.7),
                'compactness': np.random.uniform(0.4, 0.8)
            },
            texture_features={},
            spatial_location=(np.random.randint(10, 90), np.random.randint(10, 90)),
            extraction_timestamp=datetime.now(),
            confidence=np.random.uniform(0.6, 0.9)
        )
        
        features.append(feature)
    
    return features


def demo_image_recognition():
    """画像認識ユースケースのデモ"""
    print("=" * 60)
    print("画像認識ユースケースのデモ")
    print("=" * 60)
    
    # 依存関係の構築
    mock_extractor = MockFeatureExtractor()
    mock_repository = MockSymbolRepository()
    
    # 事前にサンプル記号を追加
    sample_features = create_sample_visual_features(complexity=0.7)
    sample_symbol = VisualSymbol.create_from_features(
        features=sample_features,
        semantic_label="sample_object"
    )
    mock_repository.save_symbol(sample_symbol)
    
    # サービスの構築
    feature_extraction_service = VisualFeatureExtractionService(
        feature_extractor=mock_extractor,
        quality_threshold=0.5
    )
    
    recognizer = VisualSymbolRecognizer(recognition_threshold=0.6)
    recognizer.symbol_registry[sample_symbol.symbol_id] = sample_symbol
    
    orchestration_service = SymbolEmergenceOrchestrationService(
        recognizer=recognizer,
        symbol_repository=mock_repository,
        auto_learning_enabled=True
    )
    
    # ユースケースの構築
    use_case = RecognizeImageUseCase(
        feature_extraction_service=feature_extraction_service,
        emergence_orchestration_service=orchestration_service,
        enable_performance_monitoring=True
    )
    
    # 認識リクエストの作成
    sample_image = create_sample_image()
    
    request = ImageRecognitionRequest(
        image_array=sample_image,
        recognition_threshold=0.6,
        enable_learning=True,
        return_alternatives=True,
        max_alternatives=3,
        include_debug_info=True,
        session_id="demo_session_001"
    )
    
    # 認識の実行
    print("画像認識を実行中...")
    response = use_case.execute(request)
    
    # 結果の表示
    print(f"認識結果: {response.success}")
    print(f"ステータス: {response.recognition_status.value}")
    print(f"信頼度: {response.confidence:.3f}")
    print(f"処理時間: {response.processing_time:.3f}秒")
    print(f"認識された記号: {response.recognized_symbol_id}")
    print(f"意味ラベル: {response.recognized_label}")
    print(f"代替候補数: {len(response.alternative_matches)}")
    print(f"記号更新: {len(response.symbol_updates)}")
    print(f"メッセージ: {response.message}")
    
    # 統計情報の表示
    print("\n実行統計:")
    stats = use_case.get_execution_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return response


def demo_symbol_learning():
    """記号学習ユースケースのデモ"""
    print("\n" + "=" * 60)
    print("記号学習ユースケースのデモ")
    print("=" * 60)
    
    # 依存関係の構築
    mock_repository = MockSymbolRepository()
    recognizer = VisualSymbolRecognizer()
    
    # ユースケースの構築
    use_case = TrainVisualSymbolsUseCase(
        visual_symbol_recognizer=recognizer,
        symbol_repository=mock_repository,
        enable_validation=True,
        enable_merge_detection=True
    )
    
    # 学習リクエストの作成
    training_features = create_sample_visual_features(complexity=0.8)
    
    request = SymbolLearningRequest(
        training_features=training_features,
        semantic_label="learned_object",
        min_instances=2,
        confidence_threshold=0.6,
        enable_validation=True,
        merge_similar_symbols=True,
        session_id="demo_session_002"
    )
    
    # 学習の実行
    print("記号学習を実行中...")
    response = use_case.execute(request)
    
    # 結果の表示
    print(f"学習結果: {response.success}")
    print(f"学習された記号ID: {response.learned_symbol_id}")
    print(f"記号信頼度: {response.symbol_confidence:.3f}")
    print(f"処理時間: {response.processing_time:.3f}秒")
    print(f"学習インスタンス数: {response.training_instances}")
    print(f"妥当性検証インスタンス数: {response.validated_instances}")
    print(f"プロトタイプ品質: {response.prototype_quality:.3f}")
    print(f"変動カバレッジ: {response.variation_coverage:.3f}")
    print(f"学習戦略: {response.learning_strategy}")
    print(f"統合操作数: {len(response.merge_operations)}")
    print(f"警告数: {len(response.learning_warnings)}")
    print(f"推奨アクション数: {len(response.recommended_actions)}")
    print(f"メッセージ: {response.message}")
    
    # 詳細統計の表示
    if response.symbol_statistics:
        print("\n記号統計:")
        for key, value in response.symbol_statistics.items():
            print(f"  {key}: {value}")
    
    # 学習統計の表示
    print("\n学習統計:")
    stats = use_case.get_learning_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return response


def demo_memory_query():
    """記憶検索ユースケースのデモ"""
    print("\n" + "=" * 60)
    print("記憶検索ユースケースのデモ")
    print("=" * 60)
    
    # 依存関係の構築
    mock_repository = MockSymbolRepository()
    recognizer = VisualSymbolRecognizer()
    
    # サンプル記号を事前に追加
    for i in range(5):
        features = create_sample_visual_features(complexity=0.5 + i * 0.1)
        symbol = VisualSymbol.create_from_features(
            features=features,
            semantic_label=f"test_object_{i}"
        )
        symbol_id = mock_repository.save_symbol(symbol)
        recognizer.symbol_registry[symbol_id] = symbol
    
    # ユースケースの構築
    use_case = QueryVisualMemoryUseCase(
        visual_symbol_recognizer=recognizer,
        symbol_repository=mock_repository,
        enable_advanced_analysis=True,
        enable_caching=True
    )
    
    # 検索リクエストの作成（統計クエリ）
    stats_request = VisualMemoryQueryRequest(
        query_type=QueryType.STATISTICS_QUERY,
        include_statistics=True,
        calculate_memory_metrics=True,
        session_id="demo_session_003"
    )
    
    # 統計クエリの実行
    print("統計クエリを実行中...")
    stats_response = use_case.execute(stats_request)
    
    print(f"統計クエリ結果: {stats_response.success}")
    print(f"総記号数: {stats_response.total_matches}")
    print(f"処理時間: {stats_response.processing_time:.3f}秒")
    
    if stats_response.symbol_statistics:
        print("\n記憶統計:")
        for key, value in stats_response.symbol_statistics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
    
    # ラベル検索クエリの作成
    label_request = VisualMemoryQueryRequest(
        query_type=QueryType.SYMBOL_BY_LABEL,
        target_label="test_object_2",
        max_results=10,
        sort_order=SortOrder.CONFIDENCE_DESC,
        include_statistics=True,
        session_id="demo_session_004"
    )
    
    # ラベル検索の実行
    print("\nラベル検索を実行中...")
    label_response = use_case.execute(label_request)
    
    print(f"ラベル検索結果: {label_response.success}")
    print(f"結果数: {label_response.results_count}")
    print(f"処理時間: {label_response.processing_time:.3f}秒")
    
    if label_response.symbols:
        print("\n検索された記号:")
        for symbol in label_response.symbols:
            print(f"  ID: {symbol['symbol_id']}")
            print(f"  ラベル: {symbol['semantic_label']}")
            print(f"  信頼度: {symbol['confidence']:.3f}")
            print(f"  使用頻度: {symbol['usage_frequency']}")
            print()
    
    # 記憶分析クエリの作成
    analysis_request = VisualMemoryQueryRequest(
        query_type=QueryType.MEMORY_ANALYSIS,
        analyze_relationships=True,
        calculate_memory_metrics=True,
        session_id="demo_session_005"
    )
    
    # 記憶分析の実行
    print("記憶分析を実行中...")
    analysis_response = use_case.execute(analysis_request)
    
    print(f"記憶分析結果: {analysis_response.success}")
    print(f"処理時間: {analysis_response.processing_time:.3f}秒")
    
    if analysis_response.memory_analysis:
        print("\n記憶分析:")
        analysis = analysis_response.memory_analysis
        print(f"  健全性スコア: {analysis.get('health_score', 0.0):.3f}")
        print(f"  健全性レベル: {analysis.get('health_level', 'unknown')}")
        
        if 'recommendations' in analysis:
            print(f"  推奨事項: {len(analysis['recommendations'])}")
            for rec in analysis['recommendations']:
                print(f"    - {rec}")
    
    # クエリ統計の表示
    print("\nクエリ統計:")
    stats = use_case.get_query_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return stats_response, label_response, analysis_response


def main():
    """メインデモ関数"""
    print("視覚記号認識システム - アプリケーション層デモ")
    print("Clean Architecture & DDD 実装")
    print("谷口忠大の記号創発理論に基づく実装")
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 各ユースケースのデモ実行
        recognition_response = demo_image_recognition()
        learning_response = demo_symbol_learning()
        memory_responses = demo_memory_query()
        
        print("\n" + "=" * 60)
        print("デモ完了")
        print("=" * 60)
        print("すべてのアプリケーション層コンポーネントが正常に動作しました。")
        
        # 成功サマリー
        print(f"画像認識: {'成功' if recognition_response.success else '失敗'}")
        print(f"記号学習: {'成功' if learning_response.success else '失敗'}")
        print(f"記憶検索: {'成功' if all(r.success for r in memory_responses) else '失敗'}")
        
    except Exception as e:
        print(f"デモ実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()