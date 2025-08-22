"""
OpenCV視覚特徴抽出器

OpenCVライブラリを使用した包括的な視覚特徴抽出の実装。
記号創発理論に基づく多次元特徴の統合抽出を提供。
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging
from pathlib import Path

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Some color analysis features will be limited.")

from domain.value_objects.visual_feature import VisualFeature


class OpenCVFeatureExtractor:
    """
    OpenCVベースの包括的視覚特徴抽出器
    
    Clean Architecture原則:
    - インフラ層の具体実装
    - ドメイン層のVisualFeatureに依存
    - 外部ライブラリ（OpenCV）の抽象化
    
    記号創発理論の実装:
    - 多次元特徴の統合抽出
    - 身体化認知に基づく特徴選択
    - 時間的・空間的文脈の保持
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 256),
                 edge_threshold_low: int = 50,
                 edge_threshold_high: int = 150,
                 color_clusters: int = 5,
                 enable_preprocessing: bool = True):
        """
        特徴抽出器の初期化
        
        Args:
            target_size: リサイズ目標サイズ
            edge_threshold_low: Cannyエッジ検出の下位閾値
            edge_threshold_high: Cannyエッジ検出の上位閾値
            color_clusters: 主要色クラスタ数
            enable_preprocessing: 前処理の有効化
        """
        self.target_size = target_size
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high
        self.color_clusters = color_clusters
        self.enable_preprocessing = enable_preprocessing
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        # OpenCVの利用可能性チェック
        self._verify_opencv_availability()
    
    def _verify_opencv_availability(self):
        """OpenCV機能の利用可能性チェック"""
        try:
            # OpenCVの基本機能テスト
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            self.logger.info("OpenCV successfully initialized")
        except Exception as e:
            self.logger.error(f"OpenCV initialization failed: {e}")
            raise RuntimeError(f"OpenCV is not properly installed or configured: {e}")
    
    def extract_comprehensive_features(self, 
                                     image: Union[np.ndarray, str, Path],
                                     spatial_location: Tuple[int, int] = (0, 0)) -> VisualFeature:
        """
        画像から統合的視覚特徴を抽出
        
        記号創発理論に基づく多次元特徴抽出。
        エッジ、色、形状、テクスチャの統合的分析。
        
        Args:
            image: 入力画像（numpy配列またはファイルパス）
            spatial_location: 空間位置（デフォルト: (0, 0)）
            
        Returns:
            統合視覚特徴オブジェクト
            
        Raises:
            ValueError: 画像読み込みエラー
            RuntimeError: 特徴抽出処理エラー
        """
        try:
            # 画像の読み込みと前処理
            processed_image = self._load_and_preprocess_image(image)
            
            # 各特徴次元の抽出
            edge_features = self._extract_edge_features(processed_image)
            color_features = self._extract_color_features(processed_image)
            shape_features = self._extract_shape_features(processed_image)
            texture_features = self._extract_texture_features(processed_image)
            
            # 抽出信頼度の計算
            extraction_confidence = self._calculate_extraction_confidence(
                processed_image, edge_features, color_features, shape_features
            )
            
            # VisualFeature値オブジェクトの生成
            visual_feature = VisualFeature(
                edge_features=edge_features,
                color_features=color_features,
                shape_features=shape_features,
                texture_features=texture_features,
                spatial_location=spatial_location,
                extraction_timestamp=datetime.now(),
                confidence=extraction_confidence
            )
            
            self.logger.info(f"Feature extraction completed with confidence: {extraction_confidence:.3f}")
            return visual_feature
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise RuntimeError(f"Feature extraction error: {e}")
    
    def _load_and_preprocess_image(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        画像の読み込みと前処理
        
        Args:
            image: 入力画像
            
        Returns:
            前処理済み画像
        """
        # 画像の読み込み
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image from: {image}")
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # 前処理の適用
        if self.enable_preprocessing:
            img = self._apply_preprocessing(img)
        
        return img
    
    def _apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        画像前処理の適用
        
        記号創発に適した画像前処理：
        - リサイズ
        - ノイズ除去
        - コントラスト調整
        """
        # アスペクト比を保持したリサイズ
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # アスペクト比計算
        aspect_ratio = w / h
        if aspect_ratio > target_w / target_h:
            # 幅基準
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:
            # 高さ基準
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        
        # リサイズ
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # パディングで目標サイズに調整
        if new_w != target_w or new_h != target_h:
            delta_w = target_w - new_w
            delta_h = target_h - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            resized = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # ノイズ除去（バイラテラルフィルタ）
        denoised = cv2.bilateralFilter(resized, 9, 75, 75)
        
        # コントラスト調整（CLAHE）
        if len(denoised.shape) == 3:
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def _extract_edge_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        エッジ特徴の抽出
        
        Cannyエッジ検出とコントア分析による
        形状的特徴の抽出。
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Cannyエッジ検出
        edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
        
        # コントア検出
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # エッジ密度
        edge_density = np.sum(edges > 0) / edges.size
        
        # コントア統計
        contour_areas = [cv2.contourArea(c) for c in contours]
        contour_perimeters = [cv2.arcLength(c, True) for c in contours]
        
        # 主要コントアの選択（面積上位）
        major_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        major_contour_areas = [cv2.contourArea(c) for c in major_contours]
        
        # エッジ方向ヒストグラム
        edge_histogram = self._calculate_edge_orientation_histogram(gray, edges)
        
        return {
            'edges': edges.astype(np.float32),
            'edge_density': np.array([edge_density], dtype=np.float32),
            'contour_count': np.array([len(contours)], dtype=np.float32),
            'major_contour_areas': np.array(major_contour_areas + [0.0] * (10 - len(major_contour_areas)), dtype=np.float32),
            'edge_histogram': edge_histogram,
            'contour_areas_stats': np.array([
                np.mean(contour_areas) if contour_areas else 0.0,
                np.std(contour_areas) if contour_areas else 0.0,
                max(contour_areas) if contour_areas else 0.0
            ], dtype=np.float32),
            'contour_perimeter_stats': np.array([
                np.mean(contour_perimeters) if contour_perimeters else 0.0,
                np.std(contour_perimeters) if contour_perimeters else 0.0,
                max(contour_perimeters) if contour_perimeters else 0.0
            ], dtype=np.float32)
        }
    
    def _calculate_edge_orientation_histogram(self, gray: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """エッジ方向ヒストグラムの計算"""
        # Sobelフィルタでエッジ方向計算
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 方向角度計算
        angles = np.arctan2(sobel_y, sobel_x)
        
        # エッジピクセルのみの角度を抽出
        edge_angles = angles[edges > 0]
        
        # ヒストグラム計算（16方向）
        hist, _ = np.histogram(edge_angles, bins=16, range=(-np.pi, np.pi))
        
        # 正規化
        hist_normalized = hist.astype(np.float32)
        if np.sum(hist_normalized) > 0:
            hist_normalized = hist_normalized / np.sum(hist_normalized)
        
        return hist_normalized
    
    def _extract_color_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        色特徴の抽出
        
        色ヒストグラム、主要色、色モーメントによる
        色彩的特徴の抽出。
        """
        # HSV色空間への変換
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            bgr = image
        else:
            # グレースケール画像の場合
            return {
                'color_histogram': np.zeros(512, dtype=np.float32),
                'dominant_colors': np.zeros((self.color_clusters, 3), dtype=np.float32),
                'color_moments': np.zeros(9, dtype=np.float32)
            }
        
        # 色ヒストグラム（HSV各チャネル）
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # 統合ヒストグラム
        color_histogram = np.concatenate([
            h_hist.flatten(), s_hist.flatten(), v_hist.flatten()
        ]).astype(np.float32)
        
        # 正規化
        if np.sum(color_histogram) > 0:
            color_histogram = color_histogram / np.sum(color_histogram)
        
        # 主要色の抽出（K-means）
        dominant_colors = self._extract_dominant_colors(bgr)
        
        # 色モーメント（平均、標準偏差、歪度）
        color_moments = self._calculate_color_moments(hsv)
        
        return {
            'color_histogram': color_histogram,
            'dominant_colors': dominant_colors,
            'color_moments': color_moments,
            'hsv_mean': np.mean(hsv.reshape(-1, 3), axis=0).astype(np.float32),
            'hsv_std': np.std(hsv.reshape(-1, 3), axis=0).astype(np.float32)
        }
    
    def _extract_dominant_colors(self, image: np.ndarray) -> np.ndarray:
        """主要色の抽出（K-means使用）"""
        if not SKLEARN_AVAILABLE:
            # scikit-learn未使用の場合は平均色を返す
            mean_color = np.mean(image.reshape(-1, 3), axis=0)
            return np.tile(mean_color, (self.color_clusters, 1)).astype(np.float32)
        
        try:
            # 画像を1次元配列に変換
            pixels = image.reshape(-1, 3)
            
            # K-meansクラスタリング
            kmeans = KMeans(n_clusters=self.color_clusters, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # クラスタ中心（主要色）
            dominant_colors = kmeans.cluster_centers_.astype(np.float32)
            
            # 色の頻度でソート
            labels = kmeans.labels_
            label_counts = np.bincount(labels)
            sorted_indices = np.argsort(label_counts)[::-1]
            
            return dominant_colors[sorted_indices]
            
        except Exception as e:
            self.logger.warning(f"Dominant color extraction failed: {e}")
            # フォールバック：平均色
            mean_color = np.mean(image.reshape(-1, 3), axis=0)
            return np.tile(mean_color, (self.color_clusters, 1)).astype(np.float32)
    
    def _calculate_color_moments(self, hsv_image: np.ndarray) -> np.ndarray:
        """色モーメント（平均、標準偏差、歪度）の計算"""
        moments = []
        
        for channel in range(3):  # H, S, V
            channel_data = hsv_image[:, :, channel].flatten()
            
            # 1次モーメント（平均）
            mean = np.mean(channel_data)
            moments.append(mean)
            
            # 2次モーメント（標準偏差）
            std = np.std(channel_data)
            moments.append(std)
            
            # 3次モーメント（歪度）
            if std > 0:
                skewness = np.mean(((channel_data - mean) / std) ** 3)
            else:
                skewness = 0.0
            moments.append(skewness)
        
        return np.array(moments, dtype=np.float32)
    
    def _extract_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        形状特徴の抽出
        
        アスペクト比、凸包性、拡張度などの
        幾何学的特徴の抽出。
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 2値化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # コントア検出
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'aspect_ratio': 1.0,
                'solidity': 0.0,
                'extent': 0.0,
                'circularity': 0.0,
                'rectangularity': 0.0,
                'compactness': 0.0
            }
        
        # 最大コントアを使用
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 各種形状特徴の計算
        aspect_ratio = self._calculate_aspect_ratio(largest_contour)
        solidity = self._calculate_solidity(largest_contour)
        extent = self._calculate_extent(largest_contour)
        circularity = self._calculate_circularity(largest_contour)
        rectangularity = self._calculate_rectangularity(largest_contour)
        compactness = self._calculate_compactness(largest_contour)
        
        return {
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'extent': extent,
            'circularity': circularity,
            'rectangularity': rectangularity,
            'compactness': compactness
        }
    
    def _calculate_aspect_ratio(self, contour: np.ndarray) -> float:
        """アスペクト比の計算"""
        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        
        if h > 0:
            return max(w, h) / min(w, h)
        return 1.0
    
    def _calculate_solidity(self, contour: np.ndarray) -> float:
        """凸包性（Solidity）の計算"""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area > 0:
            return area / hull_area
        return 0.0
    
    def _calculate_extent(self, contour: np.ndarray) -> float:
        """拡張度（Extent）の計算"""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        
        if rect_area > 0:
            return area / rect_area
        return 0.0
    
    def _calculate_circularity(self, contour: np.ndarray) -> float:
        """円形度の計算"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            return 4 * np.pi * area / (perimeter ** 2)
        return 0.0
    
    def _calculate_rectangularity(self, contour: np.ndarray) -> float:
        """矩形度の計算"""
        area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        rect_area = w * h
        
        if rect_area > 0:
            return area / rect_area
        return 0.0
    
    def _calculate_compactness(self, contour: np.ndarray) -> float:
        """コンパクト性の計算"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area > 0:
            return (perimeter ** 2) / area
        return 0.0
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        テクスチャ特徴の抽出（基本実装）
        
        将来的にLBP、Gabor、GLCMなどを実装予定。
        現在は基本的な統計量のみ。
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 基本的なテクスチャ統計
        texture_stats = {
            'mean': np.array([np.mean(gray)], dtype=np.float32),
            'std': np.array([np.std(gray)], dtype=np.float32),
            'variance': np.array([np.var(gray)], dtype=np.float32),
            'entropy': np.array([self._calculate_entropy(gray)], dtype=np.float32),
            'energy': np.array([self._calculate_energy(gray)], dtype=np.float32)
        }
        
        return texture_stats
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """画像のエントロピー計算"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # 正規化
        
        # エントロピー計算
        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_energy(self, image: np.ndarray) -> float:
        """画像のエネルギー計算"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # 正規化
        
        # エネルギー（二乗和）
        energy = np.sum(hist ** 2)
        return energy
    
    def _calculate_extraction_confidence(self,
                                       image: np.ndarray,
                                       edge_features: Dict,
                                       color_features: Dict,
                                       shape_features: Dict) -> float:
        """
        特徴抽出の信頼度計算
        
        画像品質、特徴の豊富さ、処理の安定性から
        総合的な抽出信頼度を算出。
        """
        confidences = []
        
        # 画像品質の評価
        image_quality = self._assess_image_quality(image)
        confidences.append(image_quality)
        
        # エッジ特徴の信頼度
        edge_density = edge_features.get('edge_density', np.array([0]))[0]
        edge_confidence = min(edge_density * 5.0, 1.0)  # エッジ密度の正規化
        confidences.append(edge_confidence)
        
        # 色特徴の信頼度
        color_hist_sum = np.sum(color_features.get('color_histogram', np.array([0])))
        color_confidence = min(color_hist_sum * 2.0, 1.0) if color_hist_sum > 0 else 0.5
        confidences.append(color_confidence)
        
        # 形状特徴の信頼度
        shape_values = list(shape_features.values())
        if shape_values:
            # 形状特徴の有効性（0でない値の割合）
            valid_shape_ratio = sum(1 for v in shape_values if v > 0.01) / len(shape_values)
            confidences.append(valid_shape_ratio)
        
        # 総合信頼度（平均）
        overall_confidence = np.mean(confidences) if confidences else 0.5
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """画像品質の評価"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # コントラスト評価
        contrast = np.std(gray) / 255.0
        
        # 鮮鋭度評価（Laplacianの分散）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 1000.0, 1.0)  # 正規化
        
        # 明度の適切性
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # 0.5が最適とする
        
        # 統合品質スコア
        quality_score = (0.4 * contrast + 0.4 * sharpness + 0.2 * brightness_score)
        
        return min(1.0, max(0.0, quality_score))
    
    def batch_extract_features(self, 
                              images: List[Union[np.ndarray, str, Path]],
                              spatial_locations: Optional[List[Tuple[int, int]]] = None) -> List[VisualFeature]:
        """
        複数画像からの一括特徴抽出
        
        Args:
            images: 画像リスト
            spatial_locations: 空間位置リスト（オプション）
            
        Returns:
            視覚特徴のリスト
        """
        features = []
        
        if spatial_locations is None:
            spatial_locations = [(0, 0)] * len(images)
        
        for i, (image, location) in enumerate(zip(images, spatial_locations)):
            try:
                feature = self.extract_comprehensive_features(image, location)
                features.append(feature)
                self.logger.debug(f"Extracted features for image {i+1}/{len(images)}")
            except Exception as e:
                self.logger.error(f"Failed to extract features for image {i}: {e}")
                # エラーの場合はスキップ
                continue
        
        return features