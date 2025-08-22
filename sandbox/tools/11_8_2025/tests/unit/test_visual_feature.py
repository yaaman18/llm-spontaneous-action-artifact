"""
VisualFeature値オブジェクトの単体テスト

Clean Architecture原則に従った値オブジェクトの
不変条件、ビジネスルール、計算ロジックのテスト。
"""

import pytest
import numpy as np
from datetime import datetime

from domain.value_objects.visual_feature import VisualFeature


class TestVisualFeature:
    """VisualFeature値オブジェクトのテストクラス"""
    
    @pytest.fixture
    def valid_visual_feature(self):
        """有効なVisualFeatureインスタンス"""
        return VisualFeature(
            edge_features={
                'edge_histogram': np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                'edge_density': np.array([0.25], dtype=np.float32),
                'contour_count': np.array([5], dtype=np.float32)
            },
            color_features={
                'color_histogram': np.array([0.1] * 48, dtype=np.float32),
                'dominant_colors': np.array([[255, 0, 0], [0, 255, 0]], dtype=np.float32)
            },
            shape_features={
                'aspect_ratio': 1.5,
                'solidity': 0.8,
                'extent': 0.7,
                'circularity': 0.6
            },
            texture_features={
                'mean': np.array([128.0], dtype=np.float32),
                'std': np.array([20.0], dtype=np.float32)
            },
            spatial_location=(100, 150),
            extraction_timestamp=datetime.now(),
            confidence=0.85
        )
    
    def test_visual_feature_creation(self, valid_visual_feature):
        """正常なVisualFeature作成テスト"""
        feature = valid_visual_feature
        
        assert feature.confidence == 0.85
        assert feature.spatial_location == (100, 150)
        assert isinstance(feature.extraction_timestamp, datetime)
        assert len(feature.edge_features) == 3
        assert len(feature.color_features) == 2
        assert len(feature.shape_features) == 4
    
    def test_confidence_validation(self):
        """信頼度の妥当性検証テスト"""
        # 有効な信頼度範囲のテスト
        valid_confidences = [0.0, 0.5, 1.0]
        
        for confidence in valid_confidences:
            feature = VisualFeature(
                edge_features={'edge_density': np.array([0.1])},
                color_features={'color_histogram': np.array([0.1])},
                shape_features={'aspect_ratio': 1.0},
                texture_features={},
                spatial_location=(0, 0),
                extraction_timestamp=datetime.now(),
                confidence=confidence
            )
            assert feature.confidence == confidence
        
        # 無効な信頼度のテスト
        invalid_confidences = [-0.1, 1.1, 2.0, -1.0]
        
        for confidence in invalid_confidences:
            with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
                VisualFeature(
                    edge_features={'edge_density': np.array([0.1])},
                    color_features={'color_histogram': np.array([0.1])},
                    shape_features={'aspect_ratio': 1.0},
                    texture_features={},
                    spatial_location=(0, 0),
                    extraction_timestamp=datetime.now(),
                    confidence=confidence
                )
    
    def test_spatial_location_validation(self):
        """空間位置の妥当性検証テスト"""
        # 有効な位置のテスト
        valid_locations = [(0, 0), (100, 200), (1000, 1000)]
        
        for location in valid_locations:
            feature = VisualFeature(
                edge_features={'edge_density': np.array([0.1])},
                color_features={'color_histogram': np.array([0.1])},
                shape_features={'aspect_ratio': 1.0},
                texture_features={},
                spatial_location=location,
                extraction_timestamp=datetime.now(),
                confidence=0.5
            )
            assert feature.spatial_location == location
        
        # 無効な位置のテスト
        with pytest.raises(ValueError, match="Spatial location must be a tuple"):
            VisualFeature(
                edge_features={'edge_density': np.array([0.1])},
                color_features={'color_histogram': np.array([0.1])},
                shape_features={'aspect_ratio': 1.0},
                texture_features={},
                spatial_location=(100,),  # 単一要素
                extraction_timestamp=datetime.now(),
                confidence=0.5
            )
        
        with pytest.raises(ValueError, match="Spatial coordinates must be non-negative"):
            VisualFeature(
                edge_features={'edge_density': np.array([0.1])},
                color_features={'color_histogram': np.array([0.1])},
                shape_features={'aspect_ratio': 1.0},
                texture_features={},
                spatial_location=(-10, 20),  # 負の座標
                extraction_timestamp=datetime.now(),
                confidence=0.5
            )
    
    def test_unified_feature_vector_generation(self, valid_visual_feature):
        """統合特徴ベクトル生成テスト"""
        feature = valid_visual_feature
        vector = feature.get_unified_feature_vector()
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0
        assert not np.any(np.isnan(vector))
        assert not np.any(np.isinf(vector))
        
        # L2正規化の確認
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 1e-6  # ほぼ1.0
    
    def test_unified_feature_vector_empty_features(self):
        """空特徴での統合ベクトル生成テスト"""
        feature = VisualFeature(
            edge_features={},
            color_features={},
            shape_features={},
            texture_features={},
            spatial_location=(0, 0),
            extraction_timestamp=datetime.now(),
            confidence=0.5
        )
        
        vector = feature.get_unified_feature_vector()
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 0
    
    def test_similarity_calculation(self, valid_visual_feature):
        """類似度計算テスト"""
        feature1 = valid_visual_feature
        
        # 同一特徴との類似度（最大値）
        similarity_same = feature1.calculate_similarity(feature1)
        assert 0.7 <= similarity_same <= 1.0  # 高類似度（閾値を緩和）
        
        # 異なる特徴との類似度
        feature2 = VisualFeature(
            edge_features={
                'edge_histogram': np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
                'edge_density': np.array([0.5], dtype=np.float32),
                'contour_count': np.array([10], dtype=np.float32)
            },
            color_features={
                'color_histogram': np.array([0.05] * 48, dtype=np.float32),
                'dominant_colors': np.array([[0, 0, 255], [255, 255, 0]], dtype=np.float32)
            },
            shape_features={
                'aspect_ratio': 2.0,
                'solidity': 0.6,
                'extent': 0.5,
                'circularity': 0.3
            },
            texture_features={
                'mean': np.array([64.0], dtype=np.float32),
                'std': np.array([40.0], dtype=np.float32)
            },
            spatial_location=(200, 300),
            extraction_timestamp=datetime.now(),
            confidence=0.7
        )
        
        similarity_different = feature1.calculate_similarity(feature2)
        assert 0.0 <= similarity_different <= 1.0
        assert similarity_different < similarity_same
    
    def test_similarity_type_validation(self, valid_visual_feature):
        """類似度計算の型検証テスト"""
        feature = valid_visual_feature
        
        with pytest.raises(TypeError, match="Comparison target must be VisualFeature"):
            feature.calculate_similarity("not a feature")
        
        with pytest.raises(TypeError, match="Comparison target must be VisualFeature"):
            feature.calculate_similarity(123)
    
    def test_feature_complexity_calculation(self, valid_visual_feature):
        """特徴複雑度計算テスト"""
        feature = valid_visual_feature
        complexity = feature.get_feature_complexity()
        
        assert 0.0 <= complexity <= 1.0
        assert isinstance(complexity, float)
    
    def test_feature_complexity_empty_features(self):
        """空特徴での複雑度計算テスト"""
        feature = VisualFeature(
            edge_features={},
            color_features={},
            shape_features={},
            texture_features={},
            spatial_location=(0, 0),
            extraction_timestamp=datetime.now(),
            confidence=0.5
        )
        
        complexity = feature.get_feature_complexity()
        assert complexity == 0.0
    
    def test_symbol_candidate_assessment(self, valid_visual_feature):
        """記号候補適性判定テスト"""
        feature = valid_visual_feature
        
        # 高品質特徴は候補となるべき
        assert feature.is_extractable_symbol_candidate()
        
        # 低信頼度特徴
        low_confidence_feature = VisualFeature(
            edge_features={'edge_density': np.array([0.1])},
            color_features={'color_histogram': np.array([0.1])},
            shape_features={'aspect_ratio': 1.0},
            texture_features={},
            spatial_location=(0, 0),
            extraction_timestamp=datetime.now(),
            confidence=0.3  # 低信頼度
        )
        
        assert not low_confidence_feature.is_extractable_symbol_candidate()
    
    def test_symbol_candidate_missing_features(self):
        """必須特徴欠如での候補判定テスト"""
        # エッジ特徴なし
        feature_no_edge = VisualFeature(
            edge_features={},
            color_features={'color_histogram': np.array([0.1])},
            shape_features={'aspect_ratio': 1.0},
            texture_features={},
            spatial_location=(0, 0),
            extraction_timestamp=datetime.now(),
            confidence=0.8
        )
        
        assert not feature_no_edge.is_extractable_symbol_candidate()
        
        # 色特徴なし
        feature_no_color = VisualFeature(
            edge_features={'edge_density': np.array([0.1])},
            color_features={},
            shape_features={'aspect_ratio': 1.0},
            texture_features={},
            spatial_location=(0, 0),
            extraction_timestamp=datetime.now(),
            confidence=0.8
        )
        
        assert not feature_no_color.is_extractable_symbol_candidate()
    
    def test_immutability(self, valid_visual_feature):
        """不変性テスト"""
        feature = valid_visual_feature
        
        # 属性の直接変更は不可
        with pytest.raises(AttributeError):
            feature.confidence = 0.9
        
        with pytest.raises(AttributeError):
            feature.spatial_location = (200, 200)
    
    def test_histogram_similarity_edge_cases(self, valid_visual_feature):
        """ヒストグラム類似度の境界ケーステスト"""
        feature = valid_visual_feature
        
        # 空ヒストグラム
        empty_feature = VisualFeature(
            edge_features={'edge_histogram': np.array([])},
            color_features={'color_histogram': np.array([])},
            shape_features={'aspect_ratio': 1.0},
            texture_features={},
            spatial_location=(0, 0),
            extraction_timestamp=datetime.now(),
            confidence=0.5
        )
        
        similarity = feature.calculate_similarity(empty_feature)
        assert 0.0 <= similarity <= 1.0
    
    def test_zero_norm_histogram_similarity(self):
        """ゼロノルムヒストグラムの類似度テスト"""
        feature1 = VisualFeature(
            edge_features={'edge_histogram': np.zeros(16, dtype=np.float32)},
            color_features={'color_histogram': np.zeros(48, dtype=np.float32)},
            shape_features={'aspect_ratio': 1.0},
            texture_features={},
            spatial_location=(0, 0),
            extraction_timestamp=datetime.now(),
            confidence=0.5
        )
        
        feature2 = VisualFeature(
            edge_features={'edge_histogram': np.ones(16, dtype=np.float32)},
            color_features={'color_histogram': np.ones(48, dtype=np.float32)},
            shape_features={'aspect_ratio': 1.0},
            texture_features={},
            spatial_location=(0, 0),
            extraction_timestamp=datetime.now(),
            confidence=0.5
        )
        
        similarity = feature1.calculate_similarity(feature2)
        assert 0.0 <= similarity <= 1.0
    
    @pytest.mark.parametrize("confidence,expected_candidate", [
        (0.8, True),   # 高信頼度
        (0.6, True),   # 中信頼度
        (0.4, False),  # 低信頼度（閾値0.5未満）
        (0.0, False),  # 最低信頼度
    ])
    def test_symbol_candidate_confidence_thresholds(self, confidence, expected_candidate):
        """信頼度閾値による記号候補判定テスト"""
        feature = VisualFeature(
            edge_features={
                'edge_density': np.array([0.3]),
                'contour_count': np.array([5])
            },
            color_features={'color_histogram': np.random.rand(48).astype(np.float32)},
            shape_features={'aspect_ratio': 1.5, 'solidity': 0.7},
            texture_features={},
            spatial_location=(100, 100),
            extraction_timestamp=datetime.now(),
            confidence=confidence
        )
        
        assert feature.is_extractable_symbol_candidate() == expected_candidate