"""
視覚特徴値オブジェクト

谷口忠大の記号創発理論に基づく視覚特徴の統合表現。
エッジ、色、形状、テクスチャなどの多様な特徴を統合し、
記号創発プロセスの基盤となる値オブジェクト。
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from datetime import datetime
import numpy as np


@dataclass(frozen=True)
class VisualFeature:
    """
    統合的視覚特徴表現
    
    記号創発理論において、視覚的記号の基盤となる特徴を表現。
    複数の特徴次元を統合し、空間的・時間的文脈を保持する。
    
    Clean Architecture原則:
    - 不変な値オブジェクト（frozen=True）
    - ビジネスロジックを含まない純粋なデータ構造
    - 外部依存関係を持たない
    """
    
    edge_features: Dict[str, np.ndarray]
    """エッジ特徴: Cannyエッジ検出やコントア情報"""
    
    color_features: Dict[str, np.ndarray] 
    """色特徴: ヒストグラム、主要色、色モーメント"""
    
    shape_features: Dict[str, float]
    """形状特徴: アスペクト比、凸包性、拡張度"""
    
    texture_features: Dict[str, np.ndarray]
    """テクスチャ特徴: LBP、Gabor、GLCM（将来実装）"""
    
    spatial_location: Tuple[int, int]
    """空間位置: 画像内での位置（x, y）"""
    
    extraction_timestamp: datetime
    """抽出時刻: 時間的文脈の保持"""
    
    confidence: float
    """抽出信頼度: 0.0-1.0の範囲"""
    
    def __post_init__(self):
        """
        値オブジェクト不変条件の検証
        
        Clean Architecture原則:
        - ドメインルールの自己検証
        - 不正状態の防止
        """
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if len(self.spatial_location) != 2:
            raise ValueError("Spatial location must be a tuple of (x, y)")
        
        if self.spatial_location[0] < 0 or self.spatial_location[1] < 0:
            raise ValueError("Spatial coordinates must be non-negative")
    
    def get_unified_feature_vector(self) -> np.ndarray:
        """
        統合特徴ベクトルの生成
        
        SOMや予測符号化で使用される統一的な特徴表現を生成。
        各特徴次元を正規化して結合する。
        
        Returns:
            統合された特徴ベクトル（次元数は特徴に依存）
        """
        feature_vectors = []
        
        # エッジ特徴のベクトル化
        if 'edge_histogram' in self.edge_features:
            feature_vectors.append(self.edge_features['edge_histogram'].flatten())
        
        # 色特徴のベクトル化
        if 'color_histogram' in self.color_features:
            feature_vectors.append(self.color_features['color_histogram'].flatten())
        
        # 形状特徴のベクトル化
        if self.shape_features:
            shape_vector = np.array([
                self.shape_features.get('aspect_ratio', 0.0),
                self.shape_features.get('solidity', 0.0),
                self.shape_features.get('extent', 0.0)
            ])
            feature_vectors.append(shape_vector)
        
        # テクスチャ特徴のベクトル化（将来実装）
        # if 'lbp_histogram' in self.texture_features:
        #     feature_vectors.append(self.texture_features['lbp_histogram'].flatten())
        
        if not feature_vectors:
            return np.array([])
        
        # 特徴ベクトルの結合と正規化
        unified_vector = np.concatenate(feature_vectors)
        
        # L2正規化
        norm = np.linalg.norm(unified_vector)
        if norm > 0:
            unified_vector = unified_vector / norm
            
        return unified_vector
    
    def calculate_similarity(self, other: 'VisualFeature') -> float:
        """
        他の視覚特徴との類似度計算
        
        記号創発において重要な特徴間距離を計算。
        複数の特徴次元を考慮した総合的類似度を返す。
        
        Args:
            other: 比較対象の視覚特徴
            
        Returns:
            類似度スコア（0.0-1.0、1.0が最も類似）
        """
        if not isinstance(other, VisualFeature):
            raise TypeError("Comparison target must be VisualFeature")
        
        similarities = []
        
        # エッジ特徴の類似度
        if ('edge_histogram' in self.edge_features and 
            'edge_histogram' in other.edge_features):
            edge_sim = self._calculate_histogram_similarity(
                self.edge_features['edge_histogram'],
                other.edge_features['edge_histogram']
            )
            similarities.append(edge_sim)
        
        # 色特徴の類似度
        if ('color_histogram' in self.color_features and 
            'color_histogram' in other.color_features):
            color_sim = self._calculate_histogram_similarity(
                self.color_features['color_histogram'],
                other.color_features['color_histogram']
            )
            similarities.append(color_sim)
        
        # 形状特徴の類似度
        shape_sim = self._calculate_shape_similarity(other)
        similarities.append(shape_sim)
        
        # 空間位置の近接性（重みは低く）
        spatial_sim = self._calculate_spatial_similarity(other)
        similarities.append(spatial_sim * 0.1)  # 重み付き
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_histogram_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """ヒストグラム間の類似度計算（コサイン類似度）"""
        if hist1.size == 0 or hist2.size == 0:
            return 0.0
        
        # 正規化とサイズ統一
        hist1_norm = hist1.flatten()
        hist2_norm = hist2.flatten()
        
        # サイズを統一（最小サイズに合わせる）
        min_size = min(hist1_norm.size, hist2_norm.size)
        if min_size == 0:
            return 0.0
        
        hist1_unified = hist1_norm[:min_size]
        hist2_unified = hist2_norm[:min_size]
        
        if np.linalg.norm(hist1_unified) == 0 or np.linalg.norm(hist2_unified) == 0:
            return 0.0
        
        # コサイン類似度
        similarity = np.dot(hist1_unified, hist2_unified) / (
            np.linalg.norm(hist1_unified) * np.linalg.norm(hist2_unified)
        )
        
        return max(0.0, similarity)  # 負の値を避ける
    
    def _calculate_shape_similarity(self, other: 'VisualFeature') -> float:
        """形状特徴の類似度計算"""
        shape_keys = ['aspect_ratio', 'solidity', 'extent']
        similarities = []
        
        for key in shape_keys:
            if key in self.shape_features and key in other.shape_features:
                val1 = self.shape_features[key]
                val2 = other.shape_features[key]
                
                # 正規化された差分から類似度計算
                max_val = max(abs(val1), abs(val2), 1e-6)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(0.0, similarity))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_spatial_similarity(self, other: 'VisualFeature') -> float:
        """空間位置の類似度計算"""
        x1, y1 = self.spatial_location
        x2, y2 = other.spatial_location
        
        # ユークリッド距離の逆数を類似度とする
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # 最大距離を1000ピクセルと仮定した正規化
        normalized_distance = min(distance / 1000.0, 1.0)
        
        return 1.0 - normalized_distance
    
    def get_feature_complexity(self) -> float:
        """
        特徴の複雑度計算
        
        記号創発における特徴の情報量を推定。
        複雑度が高い特徴ほど記号化の候補となりやすい。
        
        Returns:
            特徴複雑度スコア（0.0-1.0）
        """
        complexity_scores = []
        
        # エッジ複雑度（コントア数とエッジ密度から）
        if 'contour_count' in self.edge_features and 'edge_density' in self.edge_features:
            contour_count = self.edge_features['contour_count']
            edge_density = self.edge_features['edge_density']
            
            # numpy配列から値を抽出（型安全性向上）
            if isinstance(contour_count, np.ndarray):
                contour_count = float(contour_count.flat[0]) if contour_count.size > 0 else 0.0
            else:
                contour_count = float(contour_count)
                
            if isinstance(edge_density, np.ndarray):
                edge_density = float(edge_density.flat[0]) if edge_density.size > 0 else 0.0
            else:
                edge_density = float(edge_density)
            
            # 正規化されたコントア複雑度
            contour_complexity = min(float(contour_count) / 20.0, 1.0)  # 20コントアを最大と仮定
            edge_complexity = min(float(edge_density) * 2.0, 1.0)  # 密度の2倍を複雑度とする
            
            complexity_scores.append((contour_complexity + edge_complexity) / 2.0)
        
        # 色複雑度（色の多様性から）
        if 'color_histogram' in self.color_features:
            hist = self.color_features['color_histogram']
            if hist.size > 0:
                # ヒストグラムのエントロピーから複雑度計算
                hist_sum = np.sum(hist)
                if hist_sum > 0:
                    hist_normalized = hist / hist_sum
                    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-6))
                    
                    # エントロピーの正規化（最大エントロピーで除算）
                    max_entropy = np.log2(hist.size) if hist.size > 1 else 1.0
                    color_complexity = entropy / max_entropy if max_entropy > 0 else 0.0
                    complexity_scores.append(color_complexity)
        
        # 形状複雑度（形状特徴の変動から）
        shape_values = list(self.shape_features.values())
        if shape_values:
            # 値を確実にfloatに変換
            shape_values = [float(v) for v in shape_values]
            shape_variance = np.var(shape_values)
            shape_complexity = min(float(shape_variance) * 4.0, 1.0)  # 分散の4倍を複雑度とする
            complexity_scores.append(shape_complexity)
        
        return np.mean(complexity_scores) if complexity_scores else 0.0
    
    def is_extractable_symbol_candidate(self) -> bool:
        """
        記号抽出候補としての適性判定
        
        記号創発理論に基づく候補選定。
        十分な複雑度と信頼度を持つ特徴のみを候補とする。
        
        Returns:
            記号候補としての適性（True/False）
        """
        # 最小信頼度チェック
        if self.confidence < 0.5:
            return False
        
        # 最小複雑度チェック
        if self.get_feature_complexity() < 0.3:
            return False
        
        # 必須特徴の存在チェック
        has_essential_features = (
            bool(self.edge_features) and
            bool(self.color_features) and
            bool(self.shape_features)
        )
        
        return has_essential_features