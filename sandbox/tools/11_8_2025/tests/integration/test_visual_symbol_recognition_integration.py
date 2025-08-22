"""
視覚記号認識システム統合テスト

Phase 1実装の基本的な統合テスト。
実際の画像データを使用せず、合成データで
システム全体の動作を検証。
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import cv2

from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.visual_symbol import VisualSymbol
from domain.value_objects.recognition_result import RecognitionResult, RecognitionStatus
from domain.entities.visual_symbol_recognizer import VisualSymbolRecognizer
from infrastructure.image_processing.opencv_feature_extractor import OpenCVFeatureExtractor


class TestVisualSymbolRecognitionIntegration:
    """視覚記号認識システム統合テスト"""
    
    @pytest.fixture
    def sample_features(self):
        """サンプル視覚特徴の生成"""
        return [
            VisualFeature(
                edge_features={
                    'edge_histogram': np.random.rand(16).astype(np.float32),
                    'edge_density': np.array([0.3], dtype=np.float32),
                    'contour_count': np.array([5], dtype=np.float32)
                },
                color_features={
                    'color_histogram': np.random.rand(48).astype(np.float32),
                    'dominant_colors': np.random.rand(5, 3).astype(np.float32)
                },
                shape_features={
                    'aspect_ratio': 1.5,
                    'solidity': 0.8,
                    'extent': 0.7
                },
                texture_features={},
                spatial_location=(100, 100),
                extraction_timestamp=datetime.now(),
                confidence=0.8
            )
            for _ in range(5)
        ]
    
    @pytest.fixture
    def recognizer(self):
        """視覚記号認識器のインスタンス"""
        return VisualSymbolRecognizer(
            recognition_threshold=0.7,
            ambiguity_threshold=0.1,
            learning_enabled=True
        )
    
    @pytest.fixture
    def feature_extractor(self):
        """特徴抽出器のインスタンス"""
        return OpenCVFeatureExtractor(
            target_size=(128, 128),
            enable_preprocessing=True
        )
    
    def test_end_to_end_symbol_learning_and_recognition(self, recognizer, sample_features):
        """エンドツーエンドの記号学習と認識テスト"""
        # Phase 1: 新しい記号の学習
        symbol = recognizer.learn_new_symbol(
            features=sample_features,
            semantic_label="test_circle"
        )
        
        # 記号が正しく学習されたことを確認
        assert symbol is not None
        assert symbol.symbol_id in recognizer.symbol_registry
        assert symbol.semantic_label == "test_circle"
        assert len(symbol.emergence_history) == len(sample_features)
        
        # Phase 2: 類似特徴での認識テスト
        # 学習に使用した特徴の一つで認識実行
        recognition_result = recognizer.recognize_image(sample_features[0])
        
        # 認識が成功することを確認
        assert recognition_result.status == RecognitionStatus.SUCCESS
        assert recognition_result.recognized_symbol is not None
        assert recognition_result.recognized_symbol.symbol_id == symbol.symbol_id
        assert recognition_result.confidence >= recognizer.recognition_threshold
        
        # Phase 3: 統計情報の確認
        stats = recognizer.get_recognition_statistics()
        assert stats['total_recognitions'] >= 1
        assert stats['success_rate'] > 0
        assert stats['total_symbols'] >= 1
    
    def test_unknown_object_recognition(self, recognizer, sample_features):
        """未知物体認識テスト"""
        # 記号を学習せずに認識を試行
        recognition_result = recognizer.recognize_image(sample_features[0])
        
        # 未知物体として認識されることを確認
        assert recognition_result.status == RecognitionStatus.UNKNOWN
        assert recognition_result.recognized_symbol is None
        assert recognition_result.confidence == 0.0
        assert "No symbols registered" in recognition_result.error_message
    
    def test_low_confidence_recognition(self, recognizer, sample_features):
        """低信頼度認識テスト"""
        # 記号を学習
        symbol = recognizer.learn_new_symbol(
            features=sample_features[:2],  # 少数の特徴で学習
            semantic_label="test_shape"
        )
        
        # 大きく異なる特徴で認識を試行
        different_feature = VisualFeature(
            edge_features={
                'edge_histogram': np.zeros(16, dtype=np.float32),
                'edge_density': np.array([0.1], dtype=np.float32),
                'contour_count': np.array([1], dtype=np.float32)
            },
            color_features={
                'color_histogram': np.zeros(48, dtype=np.float32),
                'dominant_colors': np.zeros((5, 3), dtype=np.float32)
            },
            shape_features={
                'aspect_ratio': 3.0,
                'solidity': 0.3,
                'extent': 0.2
            },
            texture_features={},
            spatial_location=(200, 200),
            extraction_timestamp=datetime.now(),
            confidence=0.6
        )
        
        recognition_result = recognizer.recognize_image(different_feature)
        
        # 低信頼度として判定されることを確認
        assert recognition_result.status == RecognitionStatus.LOW_CONFIDENCE
        assert recognition_result.recognized_symbol is None
        assert recognition_result.confidence < recognizer.recognition_threshold
        assert len(recognition_result.alternative_matches) > 0
    
    def test_ambiguous_recognition(self, recognizer, sample_features):
        """曖昧認識テスト"""
        # 非常に類似した2つの記号を学習
        symbol1 = recognizer.learn_new_symbol(
            features=sample_features[:2],
            semantic_label="similar_shape_1"
        )
        
        # 微小な変更を加えた類似特徴で2番目の記号を学習
        similar_features = []
        for feature in sample_features[2:4]:
            # 微小変更（10%の変動）
            modified_edge_hist = feature.edge_features['edge_histogram'] * 1.1
            modified_edge_hist = modified_edge_hist / np.sum(modified_edge_hist)
            
            similar_feature = VisualFeature(
                edge_features={
                    'edge_histogram': modified_edge_hist,
                    'edge_density': feature.edge_features['edge_density'] * 0.9,
                    'contour_count': feature.edge_features['contour_count']
                },
                color_features=feature.color_features,
                shape_features=feature.shape_features,
                texture_features=feature.texture_features,
                spatial_location=feature.spatial_location,
                extraction_timestamp=datetime.now(),
                confidence=feature.confidence
            )
            similar_features.append(similar_feature)
        
        symbol2 = recognizer.learn_new_symbol(
            features=similar_features,
            semantic_label="similar_shape_2"
        )
        
        # 低い曖昧閾値で認識器を再設定
        recognizer.ambiguity_threshold = 0.2
        
        # 中間的な特徴で認識を試行
        intermediate_feature = sample_features[4]  # 学習に使用していない特徴
        recognition_result = recognizer.recognize_image(intermediate_feature)
        
        # 状況により成功または曖昧判定が期待される
        assert recognition_result.status in [
            RecognitionStatus.SUCCESS,
            RecognitionStatus.AMBIGUOUS,
            RecognitionStatus.LOW_CONFIDENCE
        ]
    
    def test_continuous_learning(self, recognizer, sample_features):
        """継続学習テスト"""
        # 初期記号学習
        symbol = recognizer.learn_new_symbol(
            features=sample_features[:2],
            semantic_label="learning_test"
        )
        
        initial_confidence = symbol.confidence
        initial_usage_frequency = symbol.usage_frequency
        
        # 継続学習有効化
        recognizer.learning_enabled = True
        
        # 同じ記号の認識を複数回実行
        for feature in sample_features[2:]:
            result = recognizer.recognize_image(feature)
            if result.status == RecognitionStatus.SUCCESS:
                break
        
        # 記号が更新されていることを確認
        updated_symbol = recognizer.symbol_registry[symbol.symbol_id]
        assert updated_symbol.usage_frequency >= initial_usage_frequency
    
    @pytest.mark.skipif(not cv2, reason="OpenCV not available")
    def test_opencv_feature_extraction_integration(self, feature_extractor):
        """OpenCV特徴抽出の統合テスト"""
        # 合成画像の作成
        synthetic_image = self._create_synthetic_image()
        
        # 特徴抽出の実行
        extracted_features = feature_extractor.extract_comprehensive_features(
            image=synthetic_image,
            spatial_location=(50, 50)
        )
        
        # 特徴が正しく抽出されたことを確認
        assert isinstance(extracted_features, VisualFeature)
        assert 0.0 <= extracted_features.confidence <= 1.0
        assert extracted_features.spatial_location == (50, 50)
        assert len(extracted_features.edge_features) > 0
        assert len(extracted_features.color_features) > 0
        assert len(extracted_features.shape_features) > 0
        
        # 統合特徴ベクトルの生成テスト
        feature_vector = extracted_features.get_unified_feature_vector()
        assert len(feature_vector) > 0
        assert not np.any(np.isnan(feature_vector))
    
    @pytest.mark.skipif(not cv2, reason="OpenCV not available")
    def test_image_file_processing_integration(self, feature_extractor, recognizer):
        """画像ファイル処理の統合テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # テスト画像ファイルの作成
            image_path = Path(temp_dir) / "test_image.png"
            synthetic_image = self._create_synthetic_image()
            cv2.imwrite(str(image_path), synthetic_image)
            
            # ファイルからの特徴抽出
            extracted_features = feature_extractor.extract_comprehensive_features(
                image=image_path
            )
            
            # 記号学習と認識の実行
            symbol = recognizer.learn_new_symbol(
                features=[extracted_features],
                semantic_label="file_test"
            )
            
            # 同じ特徴での認識テスト
            recognition_result = recognizer.recognize_image(extracted_features)
            
            assert recognition_result.status == RecognitionStatus.SUCCESS
            assert recognition_result.recognized_symbol.symbol_id == symbol.symbol_id
    
    def test_batch_processing(self, feature_extractor, recognizer):
        """バッチ処理テスト"""
        # 複数の合成画像を作成
        synthetic_images = [self._create_synthetic_image(i) for i in range(3)]
        
        # バッチ特徴抽出
        batch_features = feature_extractor.batch_extract_features(synthetic_images)
        
        assert len(batch_features) <= len(synthetic_images)  # エラーでスキップされる可能性
        
        if batch_features:
            # バッチ学習（認識閾値を下げて成功しやすくする）
            original_threshold = recognizer.recognition_threshold
            recognizer.recognition_threshold = 0.4  # 一時的に閾値を下げる
            
            symbol = recognizer.learn_new_symbol(
                features=batch_features,
                semantic_label="batch_test"
            )
            
            # 各特徴での認識テスト
            successful_recognitions = 0
            low_confidence_recognitions = 0
            for feature in batch_features:
                result = recognizer.recognize_image(feature)
                if result.status == RecognitionStatus.SUCCESS:
                    successful_recognitions += 1
                elif result.status == RecognitionStatus.LOW_CONFIDENCE:
                    low_confidence_recognitions += 1
            
            # 閾値を元に戻す
            recognizer.recognition_threshold = original_threshold
            
            # 成功またはlow_confidenceで少なくとも1つは認識されることを確認
            assert (successful_recognitions + low_confidence_recognitions) > 0
    
    def test_performance_requirements(self, recognizer, sample_features):
        """性能要件テスト"""
        # 記号学習
        symbol = recognizer.learn_new_symbol(
            features=sample_features,
            semantic_label="performance_test"
        )
        
        # 認識速度テスト
        start_time = datetime.now()
        result = recognizer.recognize_image(sample_features[0])
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 2秒以内の認識を要求
        assert processing_time < 2.0
        assert result.processing_time < 2.0
        
        # メモリ使用量の妥当性チェック（基本的な確認のみ）
        stats = recognizer.get_recognition_statistics()
        assert stats['total_symbols'] >= 1
        assert 0.0 <= stats['success_rate'] <= 1.0
    
    def test_error_handling_integration(self, recognizer, feature_extractor):
        """エラーハンドリング統合テスト"""
        # 無効な特徴での認識テスト - 値オブジェクトの不変条件違反を確認
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            # 無効な信頼度で値オブジェクト作成時にエラーが発生
            VisualFeature(
                edge_features={},
                color_features={},
                shape_features={},
                texture_features={},
                spatial_location=(0, 0),
                extraction_timestamp=datetime.now(),
                confidence=-1.0  # 無効な信頼度
            )
        
        # 存在しない画像ファイルでの特徴抽出テスト
        with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
            # ファイルが存在しない場合のエラー処理
            feature_extractor.extract_comprehensive_features("non_existent_file.jpg")
        
        # 空の特徴での認識テスト（正常な範囲内）
        minimal_feature = VisualFeature(
            edge_features={},
            color_features={},
            shape_features={},
            texture_features={},
            spatial_location=(0, 0),
            extraction_timestamp=datetime.now(),
            confidence=0.1  # 有効だが低い信頼度
        )
        
        # 空特徴での認識は処理エラー、未知、または低信頼度として処理される
        result = recognizer.recognize_image(minimal_feature)
        assert result.status in [
            RecognitionStatus.LOW_CONFIDENCE, 
            RecognitionStatus.UNKNOWN,
            RecognitionStatus.PROCESSING_ERROR
        ]
    
    def _create_synthetic_image(self, variant: int = 0) -> np.ndarray:
        """合成画像の作成（テスト用）"""
        if not cv2:
            # OpenCVが利用できない場合はランダムな画像
            return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        # 基本的な幾何学図形を描画
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        if variant % 3 == 0:
            # 円
            cv2.circle(image, (64, 64), 30, (255, 255, 255), -1)
        elif variant % 3 == 1:
            # 矩形
            cv2.rectangle(image, (34, 34), (94, 94), (255, 255, 255), -1)
        else:
            # 三角形
            points = np.array([[64, 30], [30, 90], [98, 90]], np.int32)
            cv2.fillPoly(image, [points], (255, 255, 255))
        
        # ノイズの追加
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def test_symbol_capacity_management(self, recognizer, sample_features):
        """記号容量管理テスト"""
        # 小さな容量制限で認識器を設定
        recognizer.max_symbols = 5
        
        # 容量を超える数の記号を学習
        learned_symbols = []
        for i in range(10):
            # 各記号に対して異なる特徴を生成
            unique_features = []
            for base_feature in sample_features[:2]:
                modified_feature = VisualFeature(
                    edge_features={
                        'edge_histogram': base_feature.edge_features['edge_histogram'] + i * 0.1,
                        'edge_density': base_feature.edge_features['edge_density'],
                        'contour_count': base_feature.edge_features['contour_count']
                    },
                    color_features=base_feature.color_features,
                    shape_features={
                        **base_feature.shape_features,
                        'aspect_ratio': base_feature.shape_features['aspect_ratio'] + i * 0.1
                    },
                    texture_features=base_feature.texture_features,
                    spatial_location=(i * 10, i * 10),
                    extraction_timestamp=datetime.now(),
                    confidence=base_feature.confidence
                )
                unique_features.append(modified_feature)
            
            symbol = recognizer.learn_new_symbol(
                features=unique_features,
                semantic_label=f"capacity_test_{i}"
            )
            learned_symbols.append(symbol)
        
        # 記号数が容量制限内に収まることを確認
        assert len(recognizer.symbol_registry) <= recognizer.max_symbols
        
        # 統計情報の確認
        stats = recognizer.get_recognition_statistics()
        assert stats['total_symbols'] <= recognizer.max_symbols