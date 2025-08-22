"""
視覚記号値オブジェクト

谷口忠大の記号創発理論における視覚記号の表現。
プロトタイプ学習による記号形成と、使用履歴による
動的な記号進化を支援する値オブジェクト。
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
import uuid

from .visual_feature import VisualFeature


@dataclass(frozen=True)
class VisualSymbol:
    """
    視覚記号の統合表現
    
    記号創発理論における視覚記号を表現。プロトタイプ特徴、
    変動範囲、創発履歴を保持し、継続学習による記号進化を支援。
    
    Clean Architecture原則:
    - 不変な値オブジェクト（frozen=True）
    - ドメイン概念の純粋な表現
    - 外部依存関係なし
    """
    
    symbol_id: str
    """記号の一意識別子"""
    
    prototype_features: VisualFeature
    """プロトタイプ特徴（記号の中心的特徴表現）"""
    
    variation_range: Dict[str, Tuple[float, float]]
    """特徴変動範囲（各特徴次元の許容範囲）"""
    
    emergence_history: List[VisualFeature]
    """創発履歴（記号形成に寄与した特徴リスト）"""
    
    semantic_label: Optional[str]
    """意味ラベル（人間が付与する概念名、オプション）"""
    
    confidence: float
    """記号信頼度（0.0-1.0）"""
    
    usage_frequency: int
    """使用頻度（認識回数）"""
    
    creation_timestamp: datetime = None
    """作成日時"""
    
    last_updated: datetime = None  
    """最終更新日時"""
    
    def __post_init__(self):
        """
        値オブジェクト不変条件の検証
        
        Clean Architecture原則:
        - ドメインルールの自己検証
        - 不正状態の防止
        """
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.usage_frequency < 0:
            raise ValueError(f"Usage frequency must be non-negative, got {self.usage_frequency}")
        
        if not self.emergence_history:
            raise ValueError("Emergence history cannot be empty")
        
        if not self.symbol_id:
            # 自動生成
            object.__setattr__(self, 'symbol_id', self._generate_symbol_id())
        
        # タイムスタンプの自動設定
        now = datetime.now()
        if self.creation_timestamp is None:
            object.__setattr__(self, 'creation_timestamp', now)
        if self.last_updated is None:
            object.__setattr__(self, 'last_updated', now)
    
    @staticmethod
    def _generate_symbol_id() -> str:
        """一意な記号IDの生成"""
        return f"vs_{uuid.uuid4().hex[:8]}"
    
    @classmethod
    def create_from_features(
        cls,
        features: List[VisualFeature],
        semantic_label: Optional[str] = None,
        symbol_id: Optional[str] = None
    ) -> 'VisualSymbol':
        """
        特徴群から新しい視覚記号を創発
        
        記号創発理論に基づく記号形成プロセス。
        複数の特徴からプロトタイプを計算し、変動範囲を学習。
        
        Args:
            features: 記号形成に使用する特徴群
            semantic_label: 意味ラベル（オプション）
            symbol_id: 記号ID（指定されない場合は自動生成）
            
        Returns:
            新しく創発された視覚記号
        """
        if not features:
            raise ValueError("Cannot create symbol from empty feature list")
        
        # プロトタイプ特徴の計算
        prototype = cls._compute_prototype_features(features)
        
        # 変動範囲の計算
        variation_range = cls._compute_variation_range(features)
        
        # 初期信頼度の計算
        initial_confidence = cls._calculate_emergence_confidence(features)
        
        return cls(
            symbol_id=symbol_id or cls._generate_symbol_id(),
            prototype_features=prototype,
            variation_range=variation_range,
            emergence_history=features.copy(),
            semantic_label=semantic_label,
            confidence=initial_confidence,
            usage_frequency=0,
            creation_timestamp=datetime.now(),
            last_updated=datetime.now()
        )
    
    @staticmethod
    def _compute_prototype_features(features: List[VisualFeature]) -> VisualFeature:
        """
        特徴群からプロトタイプ特徴を計算
        
        各特徴次元の中央値または平均値を取り、
        最も代表的な特徴を構成する。
        
        Args:
            features: 特徴群
            
        Returns:
            プロトタイプ特徴
        """
        if not features:
            raise ValueError("Cannot compute prototype from empty feature list")
        
        # エッジ特徴の統合
        edge_features = {}
        if all('edge_histogram' in f.edge_features for f in features):
            histograms = [f.edge_features['edge_histogram'] for f in features]
            # 型安全性：全ての配列が同じ形状であることを確認
            histograms = [np.asarray(h).flatten() for h in histograms]
            # 最小サイズに揃える
            min_size = min(h.size for h in histograms)
            if min_size > 0:
                histograms_normalized = [h[:min_size] for h in histograms]
                edge_features['edge_histogram'] = np.mean(histograms_normalized, axis=0)
        
        if all('edge_density' in f.edge_features for f in features):
            densities = [f.edge_features['edge_density'] for f in features]
            # 型安全性：numpy配列とスカラー値を統一的に処理
            density_values = []
            for d in densities:
                if isinstance(d, np.ndarray):
                    # 配列の場合は最初の要素を取得
                    density_values.append(float(d.flat[0]))
                else:
                    # スカラー値の場合はそのまま使用
                    density_values.append(float(d))
            edge_features['edge_density'] = np.mean(density_values)
        
        if all('contour_count' in f.edge_features for f in features):
            counts = [f.edge_features['contour_count'] for f in features]
            # 型安全性：numpy配列とスカラー値を統一的に処理
            count_values = []
            for c in counts:
                if isinstance(c, np.ndarray):
                    count_values.append(int(c.flat[0]))
                else:
                    count_values.append(int(c))
            edge_features['contour_count'] = int(np.median(count_values))
        
        # 色特徴の統合
        color_features = {}
        if all('color_histogram' in f.color_features for f in features):
            histograms = [f.color_features['color_histogram'] for f in features]
            # 型安全性：全ての配列が同じ形状であることを確認
            histograms = [np.asarray(h).flatten() for h in histograms]
            # 最小サイズに揃える
            min_size = min(h.size for h in histograms)
            if min_size > 0:
                histograms_normalized = [h[:min_size] for h in histograms]
                color_features['color_histogram'] = np.mean(histograms_normalized, axis=0)
        
        # 形状特徴の統合
        shape_features = {}
        shape_keys = ['aspect_ratio', 'solidity', 'extent']
        for key in shape_keys:
            if all(key in f.shape_features for f in features):
                values = [f.shape_features[key] for f in features]
                # 型安全性：値をfloatに統一
                float_values = [float(v) for v in values]
                shape_features[key] = float(np.mean(float_values))
        
        # テクスチャ特徴の統合（将来実装）
        texture_features = {}
        
        # 空間位置の中央値
        locations = [f.spatial_location for f in features]
        x_coords = [loc[0] for loc in locations]
        y_coords = [loc[1] for loc in locations]
        median_location = (int(np.median(x_coords)), int(np.median(y_coords)))
        
        # 信頼度の平均
        confidences = [f.confidence for f in features]
        avg_confidence = np.mean(confidences)
        
        return VisualFeature(
            edge_features=edge_features,
            color_features=color_features,
            shape_features=shape_features,
            texture_features=texture_features,
            spatial_location=median_location,
            extraction_timestamp=datetime.now(),
            confidence=avg_confidence
        )
    
    @staticmethod
    def _compute_variation_range(features: List[VisualFeature]) -> Dict[str, Tuple[float, float]]:
        """
        特徴群から変動範囲を計算
        
        各特徴次元の最小値と最大値を計算し、
        記号の許容変動範囲を定義する。
        
        Args:
            features: 特徴群
            
        Returns:
            特徴変動範囲の辞書
        """
        variation_range = {}
        
        # 形状特徴の変動範囲
        shape_keys = ['aspect_ratio', 'solidity', 'extent']
        for key in shape_keys:
            values = [f.shape_features.get(key, 0.0) for f in features if key in f.shape_features]
            if values:
                variation_range[key] = (min(values), max(values))
        
        # エッジ密度の変動範囲
        edge_densities = []
        for f in features:
            if 'edge_density' in f.edge_features:
                density = f.edge_features['edge_density']
                # 型安全性：numpy配列とスカラー値を統一的に処理
                if isinstance(density, np.ndarray):
                    edge_densities.append(float(density.flat[0]))
                else:
                    edge_densities.append(float(density))
        
        if edge_densities:
            variation_range['edge_density'] = (min(edge_densities), max(edge_densities))
        
        # 信頼度の変動範囲
        confidences = [f.confidence for f in features]
        variation_range['confidence'] = (min(confidences), max(confidences))
        
        return variation_range
    
    @staticmethod
    def _calculate_emergence_confidence(features: List[VisualFeature]) -> float:
        """
        創発信頼度の計算
        
        特徴群の一貫性と品質から記号の信頼度を算出。
        特徴間の類似度が高いほど高信頼度となる。
        
        Args:
            features: 特徴群
            
        Returns:
            創発信頼度（0.0-1.0）
        """
        if len(features) <= 1:
            return features[0].confidence if features else 0.0
        
        # 特徴間の類似度計算
        similarities = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                similarity = features[i].calculate_similarity(features[j])
                similarities.append(similarity)
        
        # 平均類似度
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # 個別特徴の信頼度平均
        avg_feature_confidence = np.mean([f.confidence for f in features])
        
        # 複雑度の考慮
        complexities = [f.get_feature_complexity() for f in features]
        avg_complexity = np.mean(complexities)
        
        # 統合信頼度の計算（重み付き平均）
        emergence_confidence = (
            0.4 * avg_similarity +      # 一貫性の重要度
            0.4 * avg_feature_confidence +  # 特徴品質の重要度
            0.2 * avg_complexity          # 複雑度の重要度
        )
        
        return min(1.0, max(0.0, emergence_confidence))
    
    def matches_feature(self, feature: VisualFeature, threshold: float = 0.7) -> bool:
        """
        入力特徴が記号にマッチするかを判定
        
        プロトタイプ特徴との類似度と変動範囲を考慮した
        記号マッチング判定を行う。
        
        Args:
            feature: 判定対象の特徴
            threshold: マッチング閾値（0.0-1.0）
            
        Returns:
            マッチング結果（True/False）
        """
        # プロトタイプとの類似度チェック
        similarity = self.prototype_features.calculate_similarity(feature)
        if similarity < threshold:
            return False
        
        # 変動範囲内チェック
        return self._is_within_variation_range(feature)
    
    def _is_within_variation_range(self, feature: VisualFeature) -> bool:
        """特徴が変動範囲内にあるかチェック"""
        # 形状特徴の範囲チェック
        for key, (min_val, max_val) in self.variation_range.items():
            if key in ['aspect_ratio', 'solidity', 'extent']:
                if key in feature.shape_features:
                    value = feature.shape_features[key]
                    if not (min_val <= value <= max_val):
                        return False
            elif key == 'edge_density':
                if 'edge_density' in feature.edge_features:
                    value = feature.edge_features['edge_density']
                    if not (min_val <= value <= max_val):
                        return False
        
        return True
    
    def calculate_match_confidence(self, feature: VisualFeature) -> float:
        """
        特徴とのマッチング信頼度を計算
        
        Args:
            feature: 対象特徴
            
        Returns:
            マッチング信頼度（0.0-1.0）
        """
        # プロトタイプ類似度
        prototype_similarity = self.prototype_features.calculate_similarity(feature)
        
        # 変動範囲内の度合い
        range_score = self._calculate_range_score(feature)
        
        # 記号自体の信頼度を考慮
        symbol_confidence = self.confidence
        
        # 使用頻度による信頼性（正規化）
        frequency_score = min(self.usage_frequency / 100.0, 1.0)
        
        # 統合マッチング信頼度
        match_confidence = (
            0.4 * prototype_similarity +
            0.2 * range_score +
            0.3 * symbol_confidence +
            0.1 * frequency_score
        )
        
        return min(1.0, max(0.0, match_confidence))
    
    def _calculate_range_score(self, feature: VisualFeature) -> float:
        """変動範囲との適合度スコア計算"""
        scores = []
        
        for key, (min_val, max_val) in self.variation_range.items():
            if key in feature.shape_features or key == 'edge_density':
                # 特徴値の取得
                if key in ['aspect_ratio', 'solidity', 'extent']:
                    value = feature.shape_features.get(key, 0.0)
                elif key == 'edge_density':
                    density = feature.edge_features.get('edge_density', 0.0)
                    # 型安全性：numpy配列とスカラー値を統一的に処理
                    if isinstance(density, np.ndarray):
                        value = float(density.flat[0])
                    else:
                        value = float(density)
                else:
                    continue
                
                # 範囲内スコア計算
                if min_val <= value <= max_val:
                    scores.append(1.0)
                else:
                    # 範囲外の場合、距離に基づくスコア
                    range_width = max_val - min_val
                    if value < min_val:
                        distance = min_val - value
                    else:
                        distance = value - max_val
                    
                    # 正規化距離からスコア計算
                    normalized_distance = distance / (range_width + 1e-6)
                    score = max(0.0, 1.0 - normalized_distance)
                    scores.append(score)
        
        return np.mean(scores) if scores else 1.0
    
    def update_with_new_instance(self, feature: VisualFeature) -> 'VisualSymbol':
        """
        新しい特徴インスタンスで記号を更新
        
        継続学習による記号の進化を実現。
        プロトタイプと変動範囲を更新し、使用頻度を増加。
        
        Args:
            feature: 新しい特徴インスタンス
            
        Returns:
            更新された視覚記号（新しいインスタンス）
        """
        # 創発履歴の更新
        updated_history = self.emergence_history + [feature]
        
        # 最新の特徴を含めたプロトタイプ再計算
        updated_prototype = self._compute_prototype_features(updated_history)
        
        # 変動範囲の更新
        updated_variation = self._compute_variation_range(updated_history)
        
        # 信頼度の更新（新しい特徴の影響を考慮）
        updated_confidence = self._calculate_emergence_confidence(updated_history)
        
        # 使用頻度の増加
        updated_frequency = self.usage_frequency + 1
        
        return VisualSymbol(
            symbol_id=self.symbol_id,
            prototype_features=updated_prototype,
            variation_range=updated_variation,
            emergence_history=updated_history,
            semantic_label=self.semantic_label,
            confidence=updated_confidence,
            usage_frequency=updated_frequency,
            creation_timestamp=self.creation_timestamp,
            last_updated=datetime.now()
        )
    
    def get_symbol_statistics(self) -> Dict[str, float]:
        """
        記号の統計情報を取得
        
        Returns:
            記号の各種統計値
        """
        return {
            'confidence': self.confidence,
            'usage_frequency': self.usage_frequency,
            'emergence_instances': len(self.emergence_history),
            'prototype_complexity': self.prototype_features.get_feature_complexity(),
            'variation_width': self._calculate_average_variation_width(),
            'age_days': (datetime.now() - self.creation_timestamp).days,
            'days_since_update': (datetime.now() - self.last_updated).days
        }
    
    def _calculate_average_variation_width(self) -> float:
        """変動範囲の平均幅を計算"""
        widths = []
        for min_val, max_val in self.variation_range.values():
            width = max_val - min_val
            widths.append(width)
        
        return np.mean(widths) if widths else 0.0
    
    def is_stable_symbol(self, min_instances: int = 5, min_confidence: float = 0.6) -> bool:
        """
        安定した記号かどうかを判定
        
        記号創発において、十分な学習を経た安定記号かを判定。
        
        Args:
            min_instances: 最小インスタンス数
            min_confidence: 最小信頼度
            
        Returns:
            安定性判定結果
        """
        return (
            len(self.emergence_history) >= min_instances and
            self.confidence >= min_confidence and
            self.usage_frequency > 0
        )