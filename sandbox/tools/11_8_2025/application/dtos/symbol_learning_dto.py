"""
記号学習DTOモジュール

視覚記号学習システムの入出力データ転送オブジェクト。
記号創発理論に基づく学習プロセスのデータ交換を担う。
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import numpy as np

from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.visual_symbol import VisualSymbol


@dataclass(frozen=True)
class SymbolLearningRequest:
    """
    記号学習リクエストDTO
    
    新しい視覚記号の学習要求を表現。
    記号創発理論に基づく学習パラメータを提供。
    """
    
    # 学習データ
    training_features: List[VisualFeature]
    """学習に使用する特徴リスト"""
    
    semantic_label: Optional[str] = None
    """意味ラベル（概念名）"""
    
    symbol_id: Optional[str] = None
    """指定記号ID（Noneの場合自動生成）"""
    
    # 学習パラメータ
    min_instances: int = 3
    """学習に必要な最小インスタンス数"""
    
    confidence_threshold: float = 0.5
    """学習成功の最小信頼度"""
    
    enable_validation: bool = True
    """特徴妥当性検証の有効化"""
    
    merge_similar_symbols: bool = True
    """類似記号の統合有効化"""
    
    similarity_threshold: float = 0.8
    """記号統合の類似度閾値"""
    
    # メタデータ
    learning_context: Optional[Dict[str, Any]] = None
    """学習コンテキスト情報"""
    
    session_id: Optional[str] = None
    """セッション識別子"""
    
    request_timestamp: datetime = None
    """リクエスト時刻"""
    
    # 高度なオプション
    prototype_method: str = "mean"
    """プロトタイプ計算方法（mean, median, weighted）"""
    
    variation_tolerance: float = 0.2
    """変動許容度（0.0-1.0）"""
    
    incremental_update: bool = False
    """既存記号のインクリメンタル更新フラグ"""
    
    def __post_init__(self):
        """リクエストの妥当性チェック"""
        # 時刻の自動設定
        if self.request_timestamp is None:
            object.__setattr__(self, 'request_timestamp', datetime.now())
        
        # 学習データの存在チェック
        if not self.training_features:
            raise ValueError("Training features cannot be empty")
        
        # パラメータの範囲チェック
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.variation_tolerance <= 1.0):
            raise ValueError("Variation tolerance must be between 0.0 and 1.0")
        
        if self.min_instances < 1:
            raise ValueError("Min instances must be at least 1")
        
        # プロトタイプ計算方法の妥当性チェック
        valid_methods = {"mean", "median", "weighted"}
        if self.prototype_method not in valid_methods:
            raise ValueError(f"Prototype method must be one of {valid_methods}")
    
    def get_learning_strategy(self) -> str:
        """学習戦略の取得"""
        if self.incremental_update:
            return "incremental"
        elif self.merge_similar_symbols:
            return "merge_based"
        else:
            return "independent"
    
    def validate_training_data(self) -> List[str]:
        """学習データの妥当性チェック"""
        issues = []
        
        # 十分なインスタンス数チェック
        if len(self.training_features) < self.min_instances:
            issues.append(f"Insufficient training instances: {len(self.training_features)} < {self.min_instances}")
        
        # 特徴の品質チェック
        low_quality_count = 0
        for i, feature in enumerate(self.training_features):
            if not feature.is_extractable_symbol_candidate():
                low_quality_count += 1
        
        if low_quality_count > len(self.training_features) * 0.5:
            issues.append(f"Too many low-quality features: {low_quality_count}/{len(self.training_features)}")
        
        # 特徴の一貫性チェック
        if len(self.training_features) > 1:
            similarities = []
            for i in range(len(self.training_features)):
                for j in range(i + 1, len(self.training_features)):
                    sim = self.training_features[i].calculate_similarity(self.training_features[j])
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity < 0.3:
                    issues.append(f"Low feature consistency: average similarity = {avg_similarity:.3f}")
        
        return issues


@dataclass(frozen=True)
class SymbolLearningResponse:
    """
    記号学習レスポンスDTO
    
    視覚記号学習の結果を外部システムに返却するデータ構造。
    学習成功・失敗の詳細情報と記号分析結果を提供。
    """
    
    # 学習結果の基本情報
    success: bool
    """学習成功フラグ"""
    
    learned_symbol_id: Optional[str] = None
    """学習された記号のID"""
    
    symbol_confidence: Optional[float] = None
    """学習された記号の信頼度"""
    
    processing_time: float = 0.0
    """学習処理時間（秒）"""
    
    # 学習統計
    training_instances: int = 0
    """使用された学習インスタンス数"""
    
    validated_instances: int = 0
    """妥当性検証を通過したインスタンス数"""
    
    prototype_quality: Optional[float] = None
    """プロトタイプ品質スコア"""
    
    variation_coverage: Optional[float] = None
    """変動範囲カバレッジ"""
    
    # 記号分析結果
    symbol_statistics: Optional[Dict[str, float]] = None
    """記号統計情報"""
    
    feature_analysis: Optional[Dict[str, Any]] = None
    """特徴分析結果"""
    
    similarity_analysis: Optional[Dict[str, float]] = None
    """類似性分析結果"""
    
    # 学習プロセス情報
    merge_operations: List[str] = None
    """実行された統合操作"""
    
    learning_warnings: List[str] = None
    """学習中の警告メッセージ"""
    
    validation_results: Optional[Dict[str, Any]] = None
    """妥当性検証結果"""
    
    # メタデータ
    semantic_label: Optional[str] = None
    """学習された記号の意味ラベル"""
    
    learning_strategy: Optional[str] = None
    """使用された学習戦略"""
    
    message: Optional[str] = None
    """ユーザー向けメッセージ"""
    
    error_details: Optional[str] = None
    """エラー詳細（失敗時）"""
    
    session_id: Optional[str] = None
    """セッション識別子"""
    
    response_timestamp: datetime = None
    """レスポンス時刻"""
    
    # 継続学習関連
    recommended_actions: List[str] = None
    """推奨される追加アクション"""
    
    learning_feedback: Optional[Dict[str, Any]] = None
    """学習フィードバック"""
    
    def __post_init__(self):
        """レスポンスの初期化処理"""
        # 時刻の自動設定
        if self.response_timestamp is None:
            object.__setattr__(self, 'response_timestamp', datetime.now())
        
        # リストの初期化
        if self.merge_operations is None:
            object.__setattr__(self, 'merge_operations', [])
        
        if self.learning_warnings is None:
            object.__setattr__(self, 'learning_warnings', [])
        
        if self.recommended_actions is None:
            object.__setattr__(self, 'recommended_actions', [])
        
        # 妥当性チェック
        if self.symbol_confidence is not None:
            if not (0.0 <= self.symbol_confidence <= 1.0):
                raise ValueError("Symbol confidence must be between 0.0 and 1.0")
        
        if self.processing_time < 0.0:
            raise ValueError("Processing time must be non-negative")
        
        if self.training_instances < 0:
            raise ValueError("Training instances count must be non-negative")
    
    @classmethod
    def success_response(
        cls,
        learned_symbol: VisualSymbol,
        training_instances: int,
        processing_time: float,
        session_id: Optional[str] = None,
        learning_strategy: Optional[str] = None,
        merge_operations: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ) -> 'SymbolLearningResponse':
        """
        成功レスポンスの作成
        
        Args:
            learned_symbol: 学習された記号
            training_instances: 学習インスタンス数
            processing_time: 処理時間
            session_id: セッション識別子
            learning_strategy: 学習戦略
            merge_operations: 統合操作リスト
            warnings: 警告メッセージリスト
        
        Returns:
            成功学習レスポンス
        """
        # 記号統計の計算
        symbol_stats = learned_symbol.get_symbol_statistics()
        
        # 特徴分析の実行
        feature_analysis = {
            'prototype_complexity': learned_symbol.prototype_features.get_feature_complexity(),
            'emergence_instances': len(learned_symbol.emergence_history),
            'variation_dimensions': len(learned_symbol.variation_range),
            'spatial_coverage': cls._calculate_spatial_coverage(learned_symbol.emergence_history)
        }
        
        # プロトタイプ品質の計算
        prototype_quality = learned_symbol.prototype_features.confidence
        
        # 変動カバレッジの計算
        variation_coverage = cls._calculate_variation_coverage(learned_symbol)
        
        # 推奨アクションの生成
        recommended_actions = cls._generate_recommendations(learned_symbol, symbol_stats)
        
        return cls(
            success=True,
            learned_symbol_id=learned_symbol.symbol_id,
            symbol_confidence=learned_symbol.confidence,
            processing_time=processing_time,
            training_instances=training_instances,
            validated_instances=len(learned_symbol.emergence_history),
            prototype_quality=prototype_quality,
            variation_coverage=variation_coverage,
            symbol_statistics=symbol_stats,
            feature_analysis=feature_analysis,
            merge_operations=merge_operations or [],
            learning_warnings=warnings or [],
            semantic_label=learned_symbol.semantic_label,
            learning_strategy=learning_strategy,
            message=f"Successfully learned symbol '{learned_symbol.semantic_label or learned_symbol.symbol_id}' with {learned_symbol.confidence:.1%} confidence",
            session_id=session_id,
            recommended_actions=recommended_actions
        )
    
    @classmethod
    def failure_response(
        cls,
        error_message: str,
        training_instances: int = 0,
        processing_time: float = 0.0,
        session_id: Optional[str] = None,
        validation_results: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None
    ) -> 'SymbolLearningResponse':
        """
        失敗レスポンスの作成
        
        Args:
            error_message: エラーメッセージ
            training_instances: 試行した学習インスタンス数
            processing_time: 処理時間
            session_id: セッション識別子
            validation_results: 妥当性検証結果
            warnings: 警告メッセージリスト
        
        Returns:
            失敗学習レスポンス
        """
        return cls(
            success=False,
            processing_time=processing_time,
            training_instances=training_instances,
            validation_results=validation_results,
            learning_warnings=warnings or [],
            message=f"Symbol learning failed: {error_message}",
            error_details=error_message,
            session_id=session_id
        )
    
    @staticmethod
    def _calculate_spatial_coverage(features: List[VisualFeature]) -> float:
        """空間カバレッジの計算"""
        if len(features) < 2:
            return 0.0
        
        positions = [f.spatial_location for f in features]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # 正規化（1000ピクセルを最大範囲と仮定）
        coverage = min((x_range + y_range) / 2000.0, 1.0)
        return coverage
    
    @staticmethod
    def _calculate_variation_coverage(symbol: VisualSymbol) -> float:
        """変動カバレッジの計算"""
        if not symbol.variation_range:
            return 0.0
        
        # 各次元の変動幅を正規化して平均
        normalized_widths = []
        for key, (min_val, max_val) in symbol.variation_range.items():
            width = max_val - min_val
            # 次元に応じた正規化（簡易版）
            if key in ['aspect_ratio', 'solidity', 'extent']:
                normalized_width = min(width / 2.0, 1.0)  # 最大変動を2.0と仮定
            else:
                normalized_width = min(width / 100.0, 1.0)  # 一般的な正規化
            normalized_widths.append(normalized_width)
        
        return np.mean(normalized_widths) if normalized_widths else 0.0
    
    @staticmethod
    def _generate_recommendations(symbol: VisualSymbol, stats: Dict[str, float]) -> List[str]:
        """推奨アクションの生成"""
        recommendations = []
        
        # 信頼度に基づく推奨
        if symbol.confidence < 0.7:
            recommendations.append("collect_more_training_examples")
        
        # インスタンス数に基づく推奨  
        if len(symbol.emergence_history) < 5:
            recommendations.append("increase_training_diversity")
        
        # 複雑度に基づく推奨
        if symbol.prototype_features.get_feature_complexity() < 0.4:
            recommendations.append("add_more_distinctive_features")
        
        # 変動範囲に基づく推奨
        if stats.get('variation_width', 0.0) > 0.5:
            recommendations.append("refine_feature_discrimination")
        
        # 使用頻度に基づく推奨（新規記号の場合）
        if symbol.usage_frequency == 0:
            recommendations.append("validate_with_recognition_tests")
        
        return recommendations
    
    def get_learning_quality_score(self) -> float:
        """学習品質スコアの計算"""
        if not self.success:
            return 0.0
        
        quality_factors = []
        
        # 記号信頼度
        if self.symbol_confidence is not None:
            quality_factors.append(self.symbol_confidence)
        
        # プロトタイプ品質
        if self.prototype_quality is not None:
            quality_factors.append(self.prototype_quality)
        
        # 変動カバレッジ
        if self.variation_coverage is not None:
            quality_factors.append(self.variation_coverage)
        
        # 妥当性検証率
        if self.training_instances > 0:
            validation_rate = self.validated_instances / self.training_instances
            quality_factors.append(validation_rate)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """サマリー辞書への変換"""
        return {
            'success': self.success,
            'learned_symbol_id': self.learned_symbol_id,
            'symbol_confidence': self.symbol_confidence,
            'learning_quality': self.get_learning_quality_score(),
            'training_instances': self.training_instances,
            'validated_instances': self.validated_instances,
            'processing_time': self.processing_time,
            'semantic_label': self.semantic_label,
            'learning_strategy': self.learning_strategy,
            'merge_operations_count': len(self.merge_operations) if self.merge_operations else 0,
            'warnings_count': len(self.learning_warnings) if self.learning_warnings else 0,
            'recommendations_count': len(self.recommended_actions) if self.recommended_actions else 0,
            'message': self.message,
            'timestamp': self.response_timestamp.isoformat()
        }