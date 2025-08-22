"""
視覚特徴抽出サービス

Clean Architecture原則に従った視覚特徴抽出の統合サービス。
インフラ層の特徴抽出器を統合し、ドメイン層のVisualFeatureを生成。
"""

import logging
import time
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import cv2

from domain.value_objects.visual_feature import VisualFeature


class IFeatureExtractor(ABC):
    """特徴抽出器の抽象インターフェース"""
    
    @abstractmethod
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像から特徴を抽出
        
        Args:
            image: 入力画像配列
            
        Returns:
            抽出された特徴辞書
        """
        pass


class VisualFeatureExtractionService:
    """
    視覚特徴抽出統合サービス
    
    Clean Architecture原則:
    - アプリケーション層のサービス
    - ドメインロジックとインフラ層の協調
    - 外部依存関係の抽象化（依存性逆転）
    
    谷口忠大の記号創発理論の実装:
    - マルチモーダル特徴統合
    - 適応的特徴選択
    - 文脈依存特徴抽出
    """
    
    def __init__(self, 
                 feature_extractor: IFeatureExtractor,
                 quality_threshold: float = 0.5,
                 spatial_context_enabled: bool = True,
                 adaptive_extraction: bool = True):
        """
        視覚特徴抽出サービスの初期化
        
        Args:
            feature_extractor: 特徴抽出器実装
            quality_threshold: 品質閾値
            spatial_context_enabled: 空間文脈考慮の有効化
            adaptive_extraction: 適応的抽出の有効化
        """
        self.feature_extractor = feature_extractor
        self.quality_threshold = quality_threshold
        self.spatial_context_enabled = spatial_context_enabled
        self.adaptive_extraction = adaptive_extraction
        
        # 抽出統計
        self.extraction_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'quality_improvements': 0,
            'adaptive_adjustments': 0
        }
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        self.logger.info("VisualFeatureExtractionService initialized")
    
    def extract_from_image_array(self, 
                                image_array: np.ndarray,
                                spatial_location: Optional[tuple] = None,
                                extraction_context: Optional[Dict[str, Any]] = None) -> VisualFeature:
        """
        画像配列から視覚特徴を抽出
        
        Args:
            image_array: 入力画像配列
            spatial_location: 空間位置（x, y）
            extraction_context: 抽出コンテキスト
            
        Returns:
            抽出された視覚特徴
            
        Raises:
            ValueError: 画像が無効な場合
            RuntimeError: 抽出処理エラー
        """
        start_time = time.time()
        
        try:
            # 画像の妥当性チェック
            validated_image = self._validate_and_preprocess_image(image_array)
            
            # 空間位置の決定
            if spatial_location is None:
                spatial_location = self._determine_default_location(validated_image)
            
            # 基本特徴抽出
            raw_features = self.feature_extractor.extract_features(validated_image)
            
            # 適応的品質改善
            if self.adaptive_extraction:
                raw_features = self._apply_adaptive_improvements(
                    validated_image, raw_features, extraction_context
                )
            
            # 空間文脈の統合
            if self.spatial_context_enabled and extraction_context:
                raw_features = self._integrate_spatial_context(
                    raw_features, spatial_location, extraction_context
                )
            
            # ドメイン値オブジェクトの構築
            visual_feature = self._build_visual_feature(
                raw_features, spatial_location, start_time
            )
            
            # 品質検証
            if not self._validate_feature_quality(visual_feature):
                raise RuntimeError("Extracted feature quality below threshold")
            
            # 統計更新
            self._update_extraction_stats(True, time.time() - start_time)
            
            return visual_feature
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            self._update_extraction_stats(False, time.time() - start_time)
            raise RuntimeError(f"Feature extraction error: {e}")
    
    def extract_from_image_path(self,
                               image_path: str,
                               spatial_location: Optional[tuple] = None,
                               extraction_context: Optional[Dict[str, Any]] = None) -> VisualFeature:
        """
        画像ファイルから視覚特徴を抽出
        
        Args:
            image_path: 画像ファイルパス
            spatial_location: 空間位置
            extraction_context: 抽出コンテキスト
            
        Returns:
            抽出された視覚特徴
        """
        try:
            # 画像読み込み
            image_array = cv2.imread(image_path)
            if image_array is None:
                raise ValueError(f"Cannot load image from path: {image_path}")
            
            # RGB変換
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            return self.extract_from_image_array(
                image_array, spatial_location, extraction_context
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract features from image path {image_path}: {e}")
            raise
    
    def extract_batch_features(self,
                              images: List[Union[np.ndarray, str]],
                              spatial_locations: Optional[List[tuple]] = None,
                              extraction_context: Optional[Dict[str, Any]] = None) -> List[VisualFeature]:
        """
        バッチ特徴抽出
        
        Args:
            images: 画像リスト（配列またはパス）
            spatial_locations: 空間位置リスト
            extraction_context: 抽出コンテキスト
            
        Returns:
            抽出された視覚特徴リスト
        """
        features = []
        errors = []
        
        for i, image in enumerate(images):
            try:
                # 空間位置の取得
                location = None
                if spatial_locations and i < len(spatial_locations):
                    location = spatial_locations[i]
                
                # 特徴抽出
                if isinstance(image, str):
                    feature = self.extract_from_image_path(image, location, extraction_context)
                else:
                    feature = self.extract_from_image_array(image, location, extraction_context)
                
                features.append(feature)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract features from image {i}: {e}")
                errors.append((i, str(e)))
        
        if errors:
            self.logger.info(f"Batch extraction completed with {len(errors)} errors out of {len(images)} images")
        
        return features
    
    def _validate_and_preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """画像の妥当性チェックと前処理"""
        if image_array is None or image_array.size == 0:
            raise ValueError("Empty or None image array")
        
        # 次元チェック
        if len(image_array.shape) not in [2, 3]:
            raise ValueError(f"Invalid image dimensions: {image_array.shape}")
        
        # サイズチェック
        height, width = image_array.shape[:2]
        if height < 10 or width < 10:
            raise ValueError(f"Image too small: {width}x{height}")
        
        # データ型の正規化
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
        
        # グレースケール変換（必要に応じて）
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        return image_array
    
    def _determine_default_location(self, image: np.ndarray) -> tuple:
        """デフォルト空間位置の決定"""
        height, width = image.shape[:2]
        return (width // 2, height // 2)  # 画像中央
    
    def _apply_adaptive_improvements(self,
                                    image: np.ndarray,
                                    features: Dict[str, Any],
                                    context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """適応的品質改善の適用"""
        improved_features = features.copy()
        
        try:
            # エッジ特徴の改善
            if 'edge_features' in features:
                improved_features['edge_features'] = self._enhance_edge_features(
                    image, features['edge_features'], context
                )
                self.extraction_stats['adaptive_adjustments'] += 1
            
            # 色特徴の改善
            if 'color_features' in features:
                improved_features['color_features'] = self._enhance_color_features(
                    image, features['color_features'], context
                )
            
            # 形状特徴の改善
            if 'shape_features' in features:
                improved_features['shape_features'] = self._enhance_shape_features(
                    image, features['shape_features'], context
                )
            
        except Exception as e:
            self.logger.warning(f"Adaptive improvement failed: {e}")
            return features  # 元の特徴を返す
        
        return improved_features
    
    def _enhance_edge_features(self,
                              image: np.ndarray,
                              edge_features: Dict[str, Any],
                              context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """エッジ特徴の強化"""
        enhanced = edge_features.copy()
        
        # エッジ強調パラメータの適応調整
        if context and 'brightness' in context:
            brightness = context['brightness']
            if brightness < 0.3:  # 暗い画像
                # Cannyのしきい値を下げる
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 30, 80)  # 低しきい値
                enhanced['enhanced_edges'] = edges
        
        return enhanced
    
    def _enhance_color_features(self,
                               image: np.ndarray,
                               color_features: Dict[str, Any],
                               context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """色特徴の強化"""
        enhanced = color_features.copy()
        
        # 色空間変換による追加特徴
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # HSV特徴の追加
            enhanced['hsv_histogram'] = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            
            # L*a*b*特徴の追加
            enhanced['lab_histogram'] = cv2.calcHist([lab], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            
        except Exception as e:
            self.logger.warning(f"Color enhancement failed: {e}")
        
        return enhanced
    
    def _enhance_shape_features(self,
                               image: np.ndarray,
                               shape_features: Dict[str, Any],
                               context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """形状特徴の強化"""
        enhanced = shape_features.copy()
        
        try:
            # 輪郭検出による詳細形状分析
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 最大輪郭の詳細分析
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Hu moments
                moments = cv2.moments(largest_contour)
                if moments['m00'] != 0:
                    hu_moments = cv2.HuMoments(moments).flatten()
                    enhanced['hu_moments'] = hu_moments
                
                # 輪郭の複雑度
                perimeter = cv2.arcLength(largest_contour, True)
                area = cv2.contourArea(largest_contour)
                if perimeter > 0:
                    enhanced['complexity'] = area / (perimeter ** 2)
            
        except Exception as e:
            self.logger.warning(f"Shape enhancement failed: {e}")
        
        return enhanced
    
    def _integrate_spatial_context(self,
                                  features: Dict[str, Any],
                                  spatial_location: tuple,
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """空間文脈の統合"""
        contextual_features = features.copy()
        
        try:
            # 空間位置による重み付け
            x, y = spatial_location
            
            # 中心からの距離による重み
            if 'image_center' in context:
                center_x, center_y = context['image_center']
                distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                contextual_features['spatial_centrality'] = 1.0 / (1.0 + distance_from_center / 100.0)
            
            # 近隣コンテキストの考慮
            if 'neighboring_features' in context:
                neighboring = context['neighboring_features']
                contextual_features['spatial_coherence'] = self._calculate_spatial_coherence(
                    features, neighboring
                )
            
        except Exception as e:
            self.logger.warning(f"Spatial context integration failed: {e}")
        
        return contextual_features
    
    def _calculate_spatial_coherence(self, 
                                    current_features: Dict[str, Any],
                                    neighboring_features: List[Dict[str, Any]]) -> float:
        """空間的一貫性の計算"""
        if not neighboring_features:
            return 0.5  # 中立値
        
        # 簡易的な一貫性計算（実装は特徴に依存）
        coherence_scores = []
        
        for neighbor in neighboring_features:
            # 色の一貫性
            if ('color_features' in current_features and 
                'color_features' in neighbor):
                color_coherence = self._color_similarity(
                    current_features['color_features'],
                    neighbor['color_features']
                )
                coherence_scores.append(color_coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _color_similarity(self, color1: Dict[str, Any], color2: Dict[str, Any]) -> float:
        """色類似度の計算"""
        # 簡易的な実装
        if ('color_histogram' in color1 and 'color_histogram' in color2):
            hist1 = np.array(color1['color_histogram']).flatten()
            hist2 = np.array(color2['color_histogram']).flatten()
            
            if hist1.size > 0 and hist2.size > 0:
                min_size = min(hist1.size, hist2.size)
                hist1_norm = hist1[:min_size]
                hist2_norm = hist2[:min_size]
                
                # コサイン類似度
                if np.linalg.norm(hist1_norm) > 0 and np.linalg.norm(hist2_norm) > 0:
                    similarity = np.dot(hist1_norm, hist2_norm) / (
                        np.linalg.norm(hist1_norm) * np.linalg.norm(hist2_norm)
                    )
                    return max(0.0, similarity)
        
        return 0.5  # デフォルト値
    
    def _build_visual_feature(self,
                             raw_features: Dict[str, Any],
                             spatial_location: tuple,
                             start_time: float) -> VisualFeature:
        """ドメイン値オブジェクトの構築"""
        try:
            # 特徴の整理と変換
            edge_features = raw_features.get('edge_features', {})
            color_features = raw_features.get('color_features', {})
            shape_features = raw_features.get('shape_features', {})
            texture_features = raw_features.get('texture_features', {})
            
            # 信頼度の計算
            confidence = self._calculate_extraction_confidence(raw_features)
            
            return VisualFeature(
                edge_features=edge_features,
                color_features=color_features,
                shape_features=shape_features,
                texture_features=texture_features,
                spatial_location=spatial_location,
                extraction_timestamp=datetime.now(),
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Failed to build VisualFeature: {e}")
            raise RuntimeError(f"VisualFeature construction failed: {e}")
    
    def _calculate_extraction_confidence(self, features: Dict[str, Any]) -> float:
        """抽出信頼度の計算"""
        confidence_factors = []
        
        # 特徴の完全性
        expected_features = ['edge_features', 'color_features', 'shape_features']
        completeness = sum(1 for f in expected_features if f in features) / len(expected_features)
        confidence_factors.append(completeness)
        
        # エッジ特徴の品質
        if 'edge_features' in features:
            edge_quality = self._assess_edge_quality(features['edge_features'])
            confidence_factors.append(edge_quality)
        
        # 色特徴の品質
        if 'color_features' in features:
            color_quality = self._assess_color_quality(features['color_features'])
            confidence_factors.append(color_quality)
        
        # 形状特徴の品質
        if 'shape_features' in features:
            shape_quality = self._assess_shape_quality(features['shape_features'])
            confidence_factors.append(shape_quality)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _assess_edge_quality(self, edge_features: Dict[str, Any]) -> float:
        """エッジ特徴品質の評価"""
        quality_scores = []
        
        # エッジ密度の妥当性
        if 'edge_density' in edge_features:
            density = edge_features['edge_density']
            if isinstance(density, np.ndarray):
                density = float(density.flat[0])
            else:
                density = float(density)
            
            # 適切な密度範囲（0.1-0.8）
            if 0.1 <= density <= 0.8:
                quality_scores.append(1.0)
            else:
                quality_scores.append(0.5)
        
        # エッジヒストグラムの妥当性
        if 'edge_histogram' in edge_features:
            hist = edge_features['edge_histogram']
            if isinstance(hist, np.ndarray) and hist.size > 0:
                # ヒストグラムの分散（多様性）
                hist_variance = np.var(hist)
                quality_scores.append(min(hist_variance / 1000.0, 1.0))
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _assess_color_quality(self, color_features: Dict[str, Any]) -> float:
        """色特徴品質の評価"""
        quality_scores = []
        
        # 色ヒストグラムの妥当性
        if 'color_histogram' in color_features:
            hist = color_features['color_histogram']
            if isinstance(hist, np.ndarray) and hist.size > 0:
                # ヒストグラムの非零要素率
                non_zero_ratio = np.count_nonzero(hist) / hist.size
                quality_scores.append(min(non_zero_ratio * 2.0, 1.0))
        
        # 主要色の妥当性
        if 'dominant_colors' in color_features:
            colors = color_features['dominant_colors']
            if isinstance(colors, np.ndarray) and colors.size > 0:
                quality_scores.append(1.0)
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _assess_shape_quality(self, shape_features: Dict[str, Any]) -> float:
        """形状特徴品質の評価"""
        quality_scores = []
        
        # 基本形状特徴の存在チェック
        required_shape_keys = ['aspect_ratio', 'solidity', 'extent']
        present_keys = sum(1 for key in required_shape_keys if key in shape_features)
        completeness = present_keys / len(required_shape_keys)
        quality_scores.append(completeness)
        
        # 値の妥当性チェック
        for key in required_shape_keys:
            if key in shape_features:
                value = shape_features[key]
                if isinstance(value, (int, float)) and 0.0 <= value <= 10.0:
                    quality_scores.append(1.0)
                else:
                    quality_scores.append(0.0)
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _validate_feature_quality(self, feature: VisualFeature) -> bool:
        """特徴品質の妥当性検証"""
        try:
            # 基本的な整合性チェック
            if feature.confidence < self.quality_threshold:
                return False
            
            # 特徴候補としての適性チェック
            if not feature.is_extractable_symbol_candidate():
                return False
            
            # 統合特徴ベクトルの妥当性チェック
            unified_vector = feature.get_unified_feature_vector()
            if unified_vector.size == 0 or np.any(np.isnan(unified_vector)):
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Feature quality validation failed: {e}")
            return False
    
    def _update_extraction_stats(self, success: bool, processing_time: float):
        """抽出統計の更新"""
        self.extraction_stats['total_extractions'] += 1
        
        if success:
            self.extraction_stats['successful_extractions'] += 1
        else:
            self.extraction_stats['failed_extractions'] += 1
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """抽出統計の取得"""
        total = max(self.extraction_stats['total_extractions'], 1)
        
        return {
            'total_extractions': self.extraction_stats['total_extractions'],
            'success_rate': self.extraction_stats['successful_extractions'] / total,
            'failure_rate': self.extraction_stats['failed_extractions'] / total,
            'quality_improvements': self.extraction_stats['quality_improvements'],
            'adaptive_adjustments': self.extraction_stats['adaptive_adjustments'],
            'quality_threshold': self.quality_threshold,
            'spatial_context_enabled': self.spatial_context_enabled,
            'adaptive_extraction': self.adaptive_extraction
        }
    
    def update_extraction_parameters(self,
                                   quality_threshold: Optional[float] = None,
                                   spatial_context_enabled: Optional[bool] = None,
                                   adaptive_extraction: Optional[bool] = None):
        """抽出パラメータの更新"""
        if quality_threshold is not None:
            if not (0.0 <= quality_threshold <= 1.0):
                raise ValueError("Quality threshold must be between 0.0 and 1.0")
            self.quality_threshold = quality_threshold
            
        if spatial_context_enabled is not None:
            self.spatial_context_enabled = spatial_context_enabled
            
        if adaptive_extraction is not None:
            self.adaptive_extraction = adaptive_extraction
            
        self.logger.info(f"Extraction parameters updated: threshold={self.quality_threshold}, "
                        f"spatial_context={self.spatial_context_enabled}, "
                        f"adaptive={self.adaptive_extraction}")