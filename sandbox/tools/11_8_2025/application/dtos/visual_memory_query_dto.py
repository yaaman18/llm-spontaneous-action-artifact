"""
視覚記憶検索DTOモジュール

視覚記憶システムの検索・クエリ機能のデータ転送オブジェクト。
記号創発理論に基づく記憶検索とメタ認知機能を支援。
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.visual_symbol import VisualSymbol


class QueryType(Enum):
    """検索タイプ"""
    SYMBOL_BY_FEATURE = "symbol_by_feature"        # 特徴による記号検索
    SYMBOL_BY_LABEL = "symbol_by_label"            # ラベルによる記号検索
    SYMBOL_BY_ID = "symbol_by_id"                  # IDによる記号検索
    SIMILAR_SYMBOLS = "similar_symbols"            # 類似記号検索
    STATISTICS_QUERY = "statistics_query"          # 統計情報クエリ
    USAGE_HISTORY = "usage_history"                # 使用履歴クエリ
    MEMORY_ANALYSIS = "memory_analysis"            # 記憶分析クエリ


class SortOrder(Enum):
    """ソート順"""
    CONFIDENCE_DESC = "confidence_desc"
    CONFIDENCE_ASC = "confidence_asc" 
    USAGE_FREQUENCY_DESC = "usage_frequency_desc"
    USAGE_FREQUENCY_ASC = "usage_frequency_asc"
    CREATION_TIME_DESC = "creation_time_desc"
    CREATION_TIME_ASC = "creation_time_asc"
    SIMILARITY_DESC = "similarity_desc"
    ALPHABETICAL = "alphabetical"


@dataclass(frozen=True)
class VisualMemoryQueryRequest:
    """
    視覚記憶クエリリクエストDTO
    
    記憶システムに対する様々な検索・分析要求を表現。
    記号創発理論に基づく高度な記憶検索機能を提供。
    """
    
    # 基本クエリ情報
    query_type: QueryType
    """クエリタイプ"""
    
    # 検索条件（クエリタイプに応じて使用）
    target_feature: Optional[VisualFeature] = None
    """検索対象の特徴（特徴検索時）"""
    
    target_label: Optional[str] = None
    """検索対象ラベル（ラベル検索時）"""
    
    target_symbol_id: Optional[str] = None
    """検索対象記号ID（ID検索時）"""
    
    reference_symbol: Optional[VisualSymbol] = None
    """参照記号（類似検索時）"""
    
    # 検索パラメータ
    similarity_threshold: float = 0.5
    """類似度閾値"""
    
    confidence_threshold: float = 0.0
    """最小信頼度閾値"""
    
    max_results: int = 10
    """最大結果数"""
    
    include_inactive: bool = False
    """非アクティブ記号の含有"""
    
    # フィルタリング条件
    created_after: Optional[datetime] = None
    """作成日時フィルタ（以降）"""
    
    created_before: Optional[datetime] = None
    """作成日時フィルタ（以前）"""
    
    used_after: Optional[datetime] = None
    """使用日時フィルタ（以降）"""
    
    min_usage_frequency: int = 0
    """最小使用頻度"""
    
    label_pattern: Optional[str] = None
    """ラベルパターン（部分マッチ）"""
    
    # ソート・集約
    sort_order: SortOrder = SortOrder.CONFIDENCE_DESC
    """ソート順"""
    
    group_by_label: bool = False
    """ラベル別グループ化"""
    
    include_statistics: bool = False
    """統計情報の含有"""
    
    # メタデータ
    session_id: Optional[str] = None
    """セッション識別子"""
    
    request_timestamp: datetime = None
    """リクエスト時刻"""
    
    # 高度な分析オプション
    analyze_relationships: bool = False
    """記号間関係の分析"""
    
    include_emergence_history: bool = False
    """創発履歴の含有"""
    
    calculate_memory_metrics: bool = False
    """記憶メトリクスの計算"""
    
    context_info: Optional[Dict[str, Any]] = None
    """コンテキスト情報"""
    
    def __post_init__(self):
        """リクエストの妥当性チェック"""
        # 時刻の自動設定
        if self.request_timestamp is None:
            object.__setattr__(self, 'request_timestamp', datetime.now())
        
        # パラメータの範囲チェック
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        if self.max_results <= 0:
            raise ValueError("Max results must be positive")
        
        if self.min_usage_frequency < 0:
            raise ValueError("Min usage frequency must be non-negative")
        
        # 日時フィルタの整合性チェック
        if (self.created_after and self.created_before and 
            self.created_after >= self.created_before):
            raise ValueError("Created after must be before created before")
        
        # クエリタイプに応じた必須パラメータチェック
        self._validate_query_parameters()
    
    def _validate_query_parameters(self):
        """クエリタイプ別パラメータ妥当性チェック"""
        if self.query_type == QueryType.SYMBOL_BY_FEATURE:
            if self.target_feature is None:
                raise ValueError("Target feature is required for feature-based query")
        
        elif self.query_type == QueryType.SYMBOL_BY_LABEL:
            if not self.target_label:
                raise ValueError("Target label is required for label-based query")
        
        elif self.query_type == QueryType.SYMBOL_BY_ID:
            if not self.target_symbol_id:
                raise ValueError("Target symbol ID is required for ID-based query")
        
        elif self.query_type == QueryType.SIMILAR_SYMBOLS:
            if self.reference_symbol is None and self.target_feature is None:
                raise ValueError("Reference symbol or target feature is required for similarity query")
    
    def get_time_range_filter(self) -> Optional[tuple]:
        """時刻範囲フィルタの取得"""
        if self.created_after or self.created_before:
            return (self.created_after, self.created_before)
        return None
    
    def is_complex_query(self) -> bool:
        """複雑クエリかどうかの判定"""
        complex_indicators = [
            self.analyze_relationships,
            self.include_emergence_history,
            self.calculate_memory_metrics,
            self.group_by_label and self.max_results > 100,
            len([f for f in [self.created_after, self.created_before, 
                           self.used_after, self.label_pattern] if f]) > 2
        ]
        return any(complex_indicators)


@dataclass(frozen=True)
class VisualMemoryQueryResponse:
    """
    視覚記憶クエリレスポンスDTO
    
    記憶システムの検索・分析結果を外部システムに返却。
    記号創発理論に基づく豊富な記憶分析情報を提供。
    """
    
    # 検索結果の基本情報
    success: bool
    """検索成功フラグ"""
    
    query_type: QueryType
    """実行されたクエリタイプ"""
    
    results_count: int
    """結果件数"""
    
    total_matches: int
    """総マッチ件数（ページネーション対応）"""
    
    processing_time: float
    """処理時間（秒）"""
    
    # 検索結果データ
    symbols: List[Dict[str, Any]] = None
    """記号検索結果リスト"""
    
    symbol_statistics: Optional[Dict[str, Any]] = None
    """記号統計情報"""
    
    similarity_scores: Optional[Dict[str, float]] = None
    """類似度スコア（記号ID -> スコア）"""
    
    # 分析結果
    memory_analysis: Optional[Dict[str, Any]] = None
    """記憶分析結果"""
    
    relationship_graph: Optional[Dict[str, List[str]]] = None
    """記号関係グラフ"""
    
    usage_patterns: Optional[Dict[str, Any]] = None
    """使用パターン分析"""
    
    temporal_distribution: Optional[Dict[str, int]] = None
    """時間分布情報"""
    
    # グループ化結果
    grouped_results: Optional[Dict[str, List[Dict[str, Any]]]] = None
    """グループ化結果（ラベル別等）"""
    
    # メタデータ
    message: Optional[str] = None
    """ユーザー向けメッセージ"""
    
    error_details: Optional[str] = None
    """エラー詳細（失敗時）"""
    
    session_id: Optional[str] = None
    """セッション識別子"""
    
    response_timestamp: datetime = None
    """レスポンス時刻"""
    
    # 検索品質情報
    search_quality: Optional[Dict[str, float]] = None
    """検索品質メトリクス"""
    
    recommendations: List[str] = None
    """検索改善推奨事項"""
    
    def __post_init__(self):
        """レスポンスの初期化処理"""
        # 時刻の自動設定
        if self.response_timestamp is None:
            object.__setattr__(self, 'response_timestamp', datetime.now())
        
        # リストの初期化
        if self.symbols is None:
            object.__setattr__(self, 'symbols', [])
        
        if self.recommendations is None:
            object.__setattr__(self, 'recommendations', [])
        
        # 妥当性チェック
        if self.results_count < 0:
            raise ValueError("Results count must be non-negative")
        
        if self.total_matches < self.results_count:
            raise ValueError("Total matches must be >= results count")
        
        if self.processing_time < 0.0:
            raise ValueError("Processing time must be non-negative")
    
    @classmethod
    def success_response(
        cls,
        query_type: QueryType,
        symbols: List[Dict[str, Any]],
        total_matches: int,
        processing_time: float,
        session_id: Optional[str] = None,
        similarity_scores: Optional[Dict[str, float]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None
    ) -> 'VisualMemoryQueryResponse':
        """
        成功レスポンスの作成
        
        Args:
            query_type: クエリタイプ
            symbols: 記号結果リスト
            total_matches: 総マッチ件数
            processing_time: 処理時間
            session_id: セッション識別子
            similarity_scores: 類似度スコア
            statistics: 統計情報
            analysis: 分析結果
        
        Returns:
            成功クエリレスポンス
        """
        results_count = len(symbols)
        
        # 検索品質の計算
        search_quality = cls._calculate_search_quality(
            symbols, similarity_scores, processing_time
        )
        
        # 推奨事項の生成
        recommendations = cls._generate_search_recommendations(
            query_type, results_count, total_matches, search_quality
        )
        
        # メッセージの生成
        message = cls._generate_success_message(query_type, results_count, total_matches)
        
        return cls(
            success=True,
            query_type=query_type,
            results_count=results_count,
            total_matches=total_matches,
            processing_time=processing_time,
            symbols=symbols,
            symbol_statistics=statistics,
            similarity_scores=similarity_scores,
            memory_analysis=analysis,
            message=message,
            session_id=session_id,
            search_quality=search_quality,
            recommendations=recommendations
        )
    
    @classmethod
    def failure_response(
        cls,
        query_type: QueryType,
        error_message: str,
        processing_time: float = 0.0,
        session_id: Optional[str] = None
    ) -> 'VisualMemoryQueryResponse':
        """
        失敗レスポンスの作成
        
        Args:
            query_type: クエリタイプ
            error_message: エラーメッセージ
            processing_time: 処理時間
            session_id: セッション識別子
        
        Returns:
            失敗クエリレスポンス
        """
        return cls(
            success=False,
            query_type=query_type,
            results_count=0,
            total_matches=0,
            processing_time=processing_time,
            message=f"Query failed: {error_message}",
            error_details=error_message,
            session_id=session_id
        )
    
    @staticmethod
    def _calculate_search_quality(
        symbols: List[Dict[str, Any]],
        similarity_scores: Optional[Dict[str, float]],
        processing_time: float
    ) -> Dict[str, float]:
        """検索品質の計算"""
        quality_metrics = {}
        
        if symbols:
            # 結果の信頼度分布
            confidences = [s.get('confidence', 0.0) for s in symbols]
            quality_metrics['avg_confidence'] = np.mean(confidences)
            quality_metrics['min_confidence'] = min(confidences)
            quality_metrics['confidence_std'] = np.std(confidences)
            
            # 使用頻度分布
            frequencies = [s.get('usage_frequency', 0) for s in symbols]
            if frequencies:
                quality_metrics['avg_usage_frequency'] = np.mean(frequencies)
                quality_metrics['popular_results_ratio'] = sum(1 for f in frequencies if f > 5) / len(frequencies)
        
        # 類似度分析
        if similarity_scores:
            similarities = list(similarity_scores.values())
            quality_metrics['avg_similarity'] = np.mean(similarities)
            quality_metrics['similarity_std'] = np.std(similarities)
        
        # 検索効率
        if processing_time > 0:
            quality_metrics['search_efficiency'] = min(1.0 / processing_time, 1.0)
        
        # 結果の多様性（簡易版）
        if len(symbols) > 1:
            labels = [s.get('semantic_label', '') for s in symbols]
            unique_labels = len(set(labels))
            quality_metrics['result_diversity'] = unique_labels / len(symbols)
        
        return quality_metrics
    
    @staticmethod
    def _generate_search_recommendations(
        query_type: QueryType,
        results_count: int,
        total_matches: int,
        search_quality: Dict[str, float]
    ) -> List[str]:
        """検索推奨事項の生成"""
        recommendations = []
        
        # 結果数に基づく推奨
        if results_count == 0:
            recommendations.append("try_lowering_thresholds")
            recommendations.append("expand_search_criteria")
        elif results_count > 50:
            recommendations.append("add_more_specific_filters")
            recommendations.append("increase_similarity_threshold")
        
        # 検索品質に基づく推奨
        if search_quality.get('avg_confidence', 0.0) < 0.6:
            recommendations.append("consider_retraining_symbols")
        
        if search_quality.get('result_diversity', 1.0) < 0.3:
            recommendations.append("diversify_search_parameters")
        
        # クエリタイプ別推奨
        if query_type == QueryType.SYMBOL_BY_FEATURE:
            if search_quality.get('avg_similarity', 0.0) < 0.7:
                recommendations.append("refine_feature_extraction")
        
        return recommendations
    
    @staticmethod
    def _generate_success_message(
        query_type: QueryType,
        results_count: int,
        total_matches: int
    ) -> str:
        """成功メッセージの生成"""
        if results_count == 0:
            return f"No symbols found matching {query_type.value} criteria"
        elif results_count == total_matches:
            return f"Found {results_count} symbols matching {query_type.value} criteria"
        else:
            return f"Found {total_matches} matching symbols, returning top {results_count}"
    
    def get_top_symbols(self, n: int = 5) -> List[Dict[str, Any]]:
        """上位N件の記号を取得"""
        return self.symbols[:n] if self.symbols else []
    
    def get_symbols_by_confidence(self, min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """指定信頼度以上の記号を取得"""
        if not self.symbols:
            return []
        
        return [s for s in self.symbols if s.get('confidence', 0.0) >= min_confidence]
    
    def get_memory_insight_summary(self) -> Dict[str, Any]:
        """記憶洞察サマリーの生成"""
        summary = {
            'total_symbols_searched': self.total_matches,
            'results_returned': self.results_count,
            'search_efficiency': self.search_quality.get('search_efficiency', 0.0) if self.search_quality else 0.0,
            'average_confidence': self.search_quality.get('avg_confidence', 0.0) if self.search_quality else 0.0,
            'query_complexity': 'high' if self.processing_time > 1.0 else 'low',
            'recommendations_count': len(self.recommendations) if self.recommendations else 0
        }
        
        if self.memory_analysis:
            summary['memory_health'] = self.memory_analysis.get('health_score', 0.0)
            summary['symbol_distribution'] = self.memory_analysis.get('distribution_stats', {})
        
        return summary
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """サマリー辞書への変換"""
        return {
            'success': self.success,
            'query_type': self.query_type.value,
            'results_count': self.results_count,
            'total_matches': self.total_matches,
            'processing_time': self.processing_time,
            'has_similarity_scores': bool(self.similarity_scores),
            'has_statistics': bool(self.symbol_statistics),
            'has_analysis': bool(self.memory_analysis),
            'recommendations_count': len(self.recommendations) if self.recommendations else 0,
            'message': self.message,
            'timestamp': self.response_timestamp.isoformat()
        }