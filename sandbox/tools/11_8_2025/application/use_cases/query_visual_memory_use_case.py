"""
視覚記憶検索ユースケース

Clean Architecture原則に従った記憶検索の中核ユースケース。
谷口忠大の記号創発理論に基づく高度な記憶検索とメタ認知機能を提供。
"""

import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

from application.dtos.visual_memory_query_dto import (
    VisualMemoryQueryRequest, 
    VisualMemoryQueryResponse,
    QueryType,
    SortOrder
)
from application.services.symbol_emergence_orchestration_service import ISymbolRepository
from domain.entities.visual_symbol_recognizer import VisualSymbolRecognizer
from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.visual_symbol import VisualSymbol


class QueryVisualMemoryUseCase:
    """
    視覚記憶検索ユースケース
    
    Clean Architecture原則:
    - アプリケーション層のユースケース
    - 記憶検索ワークフローの統合制御
    - ドメインロジックとの協調
    
    谷口忠大の記号創発理論の実装:
    - 連想記憶検索
    - 階層的記憶構造
    - メタ認知的記憶分析
    - 記憶ネットワーク探索
    """
    
    def __init__(self,
                 visual_symbol_recognizer: VisualSymbolRecognizer,
                 symbol_repository: ISymbolRepository,
                 enable_advanced_analysis: bool = True,
                 enable_caching: bool = True,
                 cache_ttl_minutes: int = 30):
        """
        記憶検索ユースケースの初期化
        
        Args:
            visual_symbol_recognizer: 視覚記号認識器
            symbol_repository: 記号リポジトリ
            enable_advanced_analysis: 高度分析の有効化
            enable_caching: キャッシュの有効化
            cache_ttl_minutes: キャッシュ有効期限（分）
        """
        self.visual_symbol_recognizer = visual_symbol_recognizer
        self.symbol_repository = symbol_repository
        self.enable_advanced_analysis = enable_advanced_analysis
        self.enable_caching = enable_caching
        self.cache_ttl_minutes = cache_ttl_minutes
        
        # クエリ統計
        self.query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'cached_responses': 0,
            'complex_queries': 0,
            'avg_query_time': 0.0,
            'total_query_time': 0.0,
            'query_type_distribution': {}
        }
        
        # 簡易キャッシュ
        self.query_cache = {} if enable_caching else None
        self.cache_timestamps = {} if enable_caching else None
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        self.logger.info("QueryVisualMemoryUseCase initialized")
    
    def execute(self, request: VisualMemoryQueryRequest) -> VisualMemoryQueryResponse:
        """
        記憶検索ユースケースの実行
        
        Args:
            request: 記憶検索リクエスト
            
        Returns:
            記憶検索レスポンス
        """
        start_time = time.time()
        self.query_stats['total_queries'] += 1
        
        # クエリタイプ統計の更新
        query_type_key = request.query_type.value
        self.query_stats['query_type_distribution'][query_type_key] = (
            self.query_stats['query_type_distribution'].get(query_type_key, 0) + 1
        )
        
        self.logger.info(f"Processing memory query: {request.query_type.value}")
        
        try:
            # 1. キャッシュチェック
            if self.enable_caching:
                cached_response = self._check_cache(request)
                if cached_response:
                    self.query_stats['cached_responses'] += 1
                    self.logger.debug("Returning cached response")
                    return cached_response
            
            # 2. 複雑クエリの検出
            if request.is_complex_query():
                self.query_stats['complex_queries'] += 1
            
            # 3. クエリタイプ別処理
            if request.query_type == QueryType.SYMBOL_BY_FEATURE:
                response = self._process_feature_query(request)
                
            elif request.query_type == QueryType.SYMBOL_BY_LABEL:
                response = self._process_label_query(request)
                
            elif request.query_type == QueryType.SYMBOL_BY_ID:
                response = self._process_id_query(request)
                
            elif request.query_type == QueryType.SIMILAR_SYMBOLS:
                response = self._process_similarity_query(request)
                
            elif request.query_type == QueryType.STATISTICS_QUERY:
                response = self._process_statistics_query(request)
                
            elif request.query_type == QueryType.USAGE_HISTORY:
                response = self._process_usage_history_query(request)
                
            elif request.query_type == QueryType.MEMORY_ANALYSIS:
                response = self._process_memory_analysis_query(request)
                
            else:
                raise ValueError(f"Unsupported query type: {request.query_type}")
            
            # 4. 後処理
            response = self._apply_post_processing(response, request)
            
            # 5. キャッシュ更新
            if self.enable_caching and response.success:
                self._update_cache(request, response)
            
            # 統計更新
            self._update_query_stats(response.success, time.time() - start_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Memory query failed: {e}")
            self._update_query_stats(False, time.time() - start_time)
            
            return VisualMemoryQueryResponse.failure_response(
                query_type=request.query_type,
                error_message=str(e),
                processing_time=time.time() - start_time,
                session_id=request.session_id
            )
    
    def _check_cache(self, request: VisualMemoryQueryRequest) -> Optional[VisualMemoryQueryResponse]:
        """キャッシュチェック"""
        if not self.query_cache:
            return None
        
        try:
            cache_key = self._generate_cache_key(request)
            
            if cache_key in self.query_cache:
                # TTLチェック
                cached_time = self.cache_timestamps.get(cache_key)
                if cached_time:
                    age_minutes = (datetime.now() - cached_time).total_seconds() / 60
                    if age_minutes <= self.cache_ttl_minutes:
                        return self.query_cache[cache_key]
                    else:
                        # 期限切れキャッシュの削除
                        del self.query_cache[cache_key]
                        del self.cache_timestamps[cache_key]
        
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")
        
        return None
    
    def _generate_cache_key(self, request: VisualMemoryQueryRequest) -> str:
        """キャッシュキーの生成"""
        key_components = [
            request.query_type.value,
            str(request.similarity_threshold),
            str(request.confidence_threshold),
            str(request.max_results),
            request.sort_order.value
        ]
        
        # クエリ固有のキー要素を追加
        if request.target_label:
            key_components.append(f"label_{request.target_label}")
        if request.target_symbol_id:
            key_components.append(f"id_{request.target_symbol_id}")
        if request.label_pattern:
            key_components.append(f"pattern_{request.label_pattern}")
        
        return "_".join(key_components)
    
    def _update_cache(self, request: VisualMemoryQueryRequest, response: VisualMemoryQueryResponse):
        """キャッシュ更新"""
        try:
            cache_key = self._generate_cache_key(request)
            self.query_cache[cache_key] = response
            self.cache_timestamps[cache_key] = datetime.now()
            
            # キャッシュサイズ制限（簡易LRU）
            if len(self.query_cache) > 100:
                oldest_key = min(self.cache_timestamps.keys(), key=self.cache_timestamps.get)
                del self.query_cache[oldest_key]
                del self.cache_timestamps[oldest_key]
        
        except Exception as e:
            self.logger.warning(f"Cache update failed: {e}")
    
    def _process_feature_query(self, request: VisualMemoryQueryRequest) -> VisualMemoryQueryResponse:
        """特徴ベース検索の処理"""
        if not request.target_feature:
            raise ValueError("Target feature is required for feature-based query")
        
        # 類似記号の検索
        similar_symbols = self.symbol_repository.find_similar_symbols(
            request.target_feature,
            threshold=request.similarity_threshold
        )
        
        # フィルタリングと変換
        filtered_symbols = self._apply_filters(similar_symbols, request)
        sorted_symbols = self._apply_sorting(filtered_symbols, request)
        limited_symbols = sorted_symbols[:request.max_results]
        
        # 類似度スコアの構築
        similarity_scores = {symbol.symbol_id: score for symbol, score in limited_symbols}
        
        # シンボル情報の構築
        symbol_data = []
        for symbol, score in limited_symbols:
            symbol_info = self._build_symbol_info(symbol, request)
            symbol_info['similarity_score'] = score
            symbol_data.append(symbol_info)
        
        # 統計情報の構築
        statistics = self._build_statistics(symbol_data) if request.include_statistics else None
        
        # 分析情報の構築
        analysis = self._build_analysis(symbol_data, request) if request.analyze_relationships else None
        
        return VisualMemoryQueryResponse.success_response(
            query_type=request.query_type,
            symbols=symbol_data,
            total_matches=len(similar_symbols),
            processing_time=0.0,  # 実際の処理時間は呼び出し元で設定
            session_id=request.session_id,
            similarity_scores=similarity_scores,
            statistics=statistics,
            analysis=analysis
        )
    
    def _process_label_query(self, request: VisualMemoryQueryRequest) -> VisualMemoryQueryResponse:
        """ラベルベース検索の処理"""
        if not request.target_label:
            raise ValueError("Target label is required for label-based query")
        
        all_symbols = self.symbol_repository.get_all_symbols()
        
        # ラベルマッチング
        if request.label_pattern:
            # パターンマッチング（部分一致）
            matched_symbols = [
                (symbol, 1.0) for symbol in all_symbols
                if symbol.semantic_label and request.label_pattern.lower() in symbol.semantic_label.lower()
            ]
        else:
            # 完全一致
            matched_symbols = [
                (symbol, 1.0) for symbol in all_symbols
                if symbol.semantic_label == request.target_label
            ]
        
        # フィルタリングと変換
        filtered_symbols = self._apply_filters(matched_symbols, request)
        sorted_symbols = self._apply_sorting(filtered_symbols, request)
        limited_symbols = sorted_symbols[:request.max_results]
        
        # シンボル情報の構築
        symbol_data = [
            self._build_symbol_info(symbol, request) 
            for symbol, _ in limited_symbols
        ]
        
        # グループ化（必要に応じて）
        grouped_results = None
        if request.group_by_label:
            grouped_results = self._group_symbols_by_label(symbol_data)
        
        return VisualMemoryQueryResponse.success_response(
            query_type=request.query_type,
            symbols=symbol_data,
            total_matches=len(matched_symbols),
            processing_time=0.0,
            session_id=request.session_id
        )
    
    def _process_id_query(self, request: VisualMemoryQueryRequest) -> VisualMemoryQueryResponse:
        """IDベース検索の処理"""
        if not request.target_symbol_id:
            raise ValueError("Target symbol ID is required for ID-based query")
        
        symbol = self.symbol_repository.find_symbol_by_id(request.target_symbol_id)
        
        if symbol:
            symbol_info = self._build_symbol_info(symbol, request)
            symbol_data = [symbol_info]
            total_matches = 1
        else:
            symbol_data = []
            total_matches = 0
        
        return VisualMemoryQueryResponse.success_response(
            query_type=request.query_type,
            symbols=symbol_data,
            total_matches=total_matches,
            processing_time=0.0,
            session_id=request.session_id
        )
    
    def _process_similarity_query(self, request: VisualMemoryQueryRequest) -> VisualMemoryQueryResponse:
        """類似性検索の処理"""
        reference_feature = None
        
        if request.reference_symbol:
            reference_feature = request.reference_symbol.prototype_features
        elif request.target_feature:
            reference_feature = request.target_feature
        else:
            raise ValueError("Reference symbol or target feature is required for similarity query")
        
        # 類似記号の検索
        similar_symbols = self.symbol_repository.find_similar_symbols(
            reference_feature,
            threshold=request.similarity_threshold
        )
        
        # 参照記号自体を結果から除外
        if request.reference_symbol:
            similar_symbols = [
                (symbol, score) for symbol, score in similar_symbols
                if symbol.symbol_id != request.reference_symbol.symbol_id
            ]
        
        # フィルタリングと変換
        filtered_symbols = self._apply_filters(similar_symbols, request)
        sorted_symbols = self._apply_sorting(filtered_symbols, request)
        limited_symbols = sorted_symbols[:request.max_results]
        
        # 類似度スコアとシンボル情報の構築
        similarity_scores = {symbol.symbol_id: score for symbol, score in limited_symbols}
        symbol_data = []
        
        for symbol, score in limited_symbols:
            symbol_info = self._build_symbol_info(symbol, request)
            symbol_info['similarity_score'] = score
            symbol_data.append(symbol_info)
        
        # 関係性分析
        analysis = None
        if request.analyze_relationships and len(symbol_data) > 1:
            analysis = self._analyze_symbol_relationships(symbol_data, reference_feature)
        
        return VisualMemoryQueryResponse.success_response(
            query_type=request.query_type,
            symbols=symbol_data,
            total_matches=len(similar_symbols),
            processing_time=0.0,
            session_id=request.session_id,
            similarity_scores=similarity_scores,
            analysis=analysis
        )
    
    def _process_statistics_query(self, request: VisualMemoryQueryRequest) -> VisualMemoryQueryResponse:
        """統計クエリの処理"""
        all_symbols = self.symbol_repository.get_all_symbols()
        recognizer_stats = self.visual_symbol_recognizer.get_recognition_statistics()
        
        # 記号統計の計算
        symbol_statistics = self._calculate_comprehensive_statistics(all_symbols)
        
        # 時間分布の計算
        temporal_distribution = self._calculate_temporal_distribution(all_symbols)
        
        # 使用パターンの分析
        usage_patterns = self._analyze_usage_patterns(all_symbols)
        
        # メモリ分析
        memory_analysis = None
        if request.calculate_memory_metrics:
            memory_analysis = self._perform_comprehensive_memory_analysis(all_symbols)
        
        return VisualMemoryQueryResponse.success_response(
            query_type=request.query_type,
            symbols=[],  # 統計クエリでは個別記号は返さない
            total_matches=len(all_symbols),
            processing_time=0.0,
            session_id=request.session_id,
            statistics=symbol_statistics,
            analysis=memory_analysis
        )
    
    def _process_usage_history_query(self, request: VisualMemoryQueryRequest) -> VisualMemoryQueryResponse:
        """使用履歴クエリの処理"""
        all_symbols = self.symbol_repository.get_all_symbols()
        
        # 使用頻度による絞り込み
        filtered_symbols = [
            (symbol, float(symbol.usage_frequency)) for symbol in all_symbols
            if symbol.usage_frequency >= request.min_usage_frequency
        ]
        
        # 日時フィルタの適用
        if request.used_after:
            filtered_symbols = [
                (symbol, score) for symbol, score in filtered_symbols
                if symbol.last_updated >= request.used_after
            ]
        
        # 信頼度フィルタの適用
        if request.confidence_threshold > 0:
            filtered_symbols = [
                (symbol, score) for symbol, score in filtered_symbols
                if symbol.confidence >= request.confidence_threshold
            ]
        
        # ソートと制限
        sorted_symbols = sorted(filtered_symbols, key=lambda x: x[1], reverse=True)
        limited_symbols = sorted_symbols[:request.max_results]
        
        # シンボル情報の構築
        symbol_data = []
        for symbol, usage_score in limited_symbols:
            symbol_info = self._build_symbol_info(symbol, request)
            symbol_info['usage_score'] = usage_score
            symbol_data.append(symbol_info)
        
        # 使用パターン分析
        usage_patterns = self._analyze_usage_patterns([symbol for symbol, _ in limited_symbols])
        
        return VisualMemoryQueryResponse.success_response(
            query_type=request.query_type,
            symbols=symbol_data,
            total_matches=len(filtered_symbols),
            processing_time=0.0,
            session_id=request.session_id,
            analysis={'usage_patterns': usage_patterns}
        )
    
    def _process_memory_analysis_query(self, request: VisualMemoryQueryRequest) -> VisualMemoryQueryResponse:
        """記憶分析クエリの処理"""
        all_symbols = self.symbol_repository.get_all_symbols()
        
        # 包括的記憶分析
        memory_analysis = self._perform_comprehensive_memory_analysis(all_symbols)
        
        # 記号関係グラフの構築
        relationship_graph = None
        if request.analyze_relationships:
            relationship_graph = self._build_symbol_relationship_graph(all_symbols)
        
        # 時間分布の計算
        temporal_distribution = self._calculate_temporal_distribution(all_symbols)
        
        return VisualMemoryQueryResponse.success_response(
            query_type=request.query_type,
            symbols=[],  # 分析クエリでは個別記号は返さない
            total_matches=len(all_symbols),
            processing_time=0.0,
            session_id=request.session_id,
            analysis=memory_analysis
        )
    
    def _apply_filters(self, 
                      symbols_with_scores: List[Tuple[VisualSymbol, float]], 
                      request: VisualMemoryQueryRequest) -> List[Tuple[VisualSymbol, float]]:
        """フィルタリングの適用"""
        filtered = symbols_with_scores
        
        # 信頼度フィルタ
        if request.confidence_threshold > 0:
            filtered = [
                (symbol, score) for symbol, score in filtered
                if symbol.confidence >= request.confidence_threshold
            ]
        
        # 非アクティブ記号の除外
        if not request.include_inactive:
            filtered = [
                (symbol, score) for symbol, score in filtered
                if symbol.usage_frequency > 0
            ]
        
        # 作成日時フィルタ
        if request.created_after:
            filtered = [
                (symbol, score) for symbol, score in filtered
                if symbol.creation_timestamp >= request.created_after
            ]
        
        if request.created_before:
            filtered = [
                (symbol, score) for symbol, score in filtered
                if symbol.creation_timestamp <= request.created_before
            ]
        
        # 使用頻度フィルタ
        if request.min_usage_frequency > 0:
            filtered = [
                (symbol, score) for symbol, score in filtered
                if symbol.usage_frequency >= request.min_usage_frequency
            ]
        
        return filtered
    
    def _apply_sorting(self, 
                      symbols_with_scores: List[Tuple[VisualSymbol, float]], 
                      request: VisualMemoryQueryRequest) -> List[Tuple[VisualSymbol, float]]:
        """ソートの適用"""
        if request.sort_order == SortOrder.CONFIDENCE_DESC:
            return sorted(symbols_with_scores, key=lambda x: x[0].confidence, reverse=True)
        elif request.sort_order == SortOrder.CONFIDENCE_ASC:
            return sorted(symbols_with_scores, key=lambda x: x[0].confidence, reverse=False)
        elif request.sort_order == SortOrder.USAGE_FREQUENCY_DESC:
            return sorted(symbols_with_scores, key=lambda x: x[0].usage_frequency, reverse=True)
        elif request.sort_order == SortOrder.USAGE_FREQUENCY_ASC:
            return sorted(symbols_with_scores, key=lambda x: x[0].usage_frequency, reverse=False)
        elif request.sort_order == SortOrder.CREATION_TIME_DESC:
            return sorted(symbols_with_scores, key=lambda x: x[0].creation_timestamp, reverse=True)
        elif request.sort_order == SortOrder.CREATION_TIME_ASC:
            return sorted(symbols_with_scores, key=lambda x: x[0].creation_timestamp, reverse=False)
        elif request.sort_order == SortOrder.SIMILARITY_DESC:
            return sorted(symbols_with_scores, key=lambda x: x[1], reverse=True)
        elif request.sort_order == SortOrder.ALPHABETICAL:
            return sorted(symbols_with_scores, key=lambda x: x[0].semantic_label or x[0].symbol_id)
        else:
            return symbols_with_scores
    
    def _build_symbol_info(self, symbol: VisualSymbol, request: VisualMemoryQueryRequest) -> Dict[str, Any]:
        """シンボル情報の構築"""
        symbol_info = {
            'symbol_id': symbol.symbol_id,
            'semantic_label': symbol.semantic_label,
            'confidence': symbol.confidence,
            'usage_frequency': symbol.usage_frequency,
            'creation_timestamp': symbol.creation_timestamp.isoformat(),
            'last_updated': symbol.last_updated.isoformat(),
            'is_stable': symbol.is_stable_symbol()
        }
        
        # 詳細情報の追加（必要に応じて）
        if request.include_emergence_history:
            symbol_info['emergence_instances'] = len(symbol.emergence_history)
            symbol_info['prototype_complexity'] = symbol.prototype_features.get_feature_complexity()
        
        if request.calculate_memory_metrics:
            symbol_stats = symbol.get_symbol_statistics()
            symbol_info.update(symbol_stats)
        
        return symbol_info
    
    def _build_statistics(self, symbol_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """統計情報の構築"""
        if not symbol_data:
            return {}
        
        confidences = [s['confidence'] for s in symbol_data]
        usage_frequencies = [s['usage_frequency'] for s in symbol_data]
        
        return {
            'total_symbols': len(symbol_data),
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'avg_usage_frequency': np.mean(usage_frequencies),
            'total_usage': sum(usage_frequencies),
            'stable_symbols': sum(1 for s in symbol_data if s.get('is_stable', False)),
            'active_symbols': sum(1 for s in symbol_data if s['usage_frequency'] > 0)
        }
    
    def _build_analysis(self, symbol_data: List[Dict[str, Any]], request: VisualMemoryQueryRequest) -> Dict[str, Any]:
        """分析情報の構築"""
        analysis = {}
        
        if symbol_data:
            # 品質分析
            analysis['quality_metrics'] = self._analyze_symbol_quality(symbol_data)
            
            # 分布分析
            analysis['distribution_analysis'] = self._analyze_symbol_distribution(symbol_data)
            
            # トレンド分析（時系列）
            if request.analyze_relationships:
                analysis['trend_analysis'] = self._analyze_temporal_trends(symbol_data)
        
        return analysis
    
    def _calculate_comprehensive_statistics(self, symbols: List[VisualSymbol]) -> Dict[str, Any]:
        """包括的統計の計算"""
        if not symbols:
            return {}
        
        # 基本統計
        confidences = [s.confidence for s in symbols]
        usage_frequencies = [s.usage_frequency for s in symbols]
        complexities = [s.prototype_features.get_feature_complexity() for s in symbols]
        
        # 記号年齢の計算
        now = datetime.now()
        ages_days = [(now - s.creation_timestamp).days for s in symbols]
        
        return {
            'total_symbols': len(symbols),
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(min(confidences)),
                'max': float(max(confidences)),
                'median': float(np.median(confidences))
            },
            'usage_stats': {
                'mean': float(np.mean(usage_frequencies)),
                'total': sum(usage_frequencies),
                'max': max(usage_frequencies),
                'active_ratio': sum(1 for f in usage_frequencies if f > 0) / len(usage_frequencies)
            },
            'complexity_stats': {
                'mean': float(np.mean(complexities)),
                'std': float(np.std(complexities)),
                'distribution': self._calculate_complexity_distribution(complexities)
            },
            'age_stats': {
                'mean_days': float(np.mean(ages_days)),
                'oldest_days': max(ages_days),
                'newest_days': min(ages_days)
            },
            'stability_stats': {
                'stable_count': sum(1 for s in symbols if s.is_stable_symbol()),
                'stable_ratio': sum(1 for s in symbols if s.is_stable_symbol()) / len(symbols)
            },
            'semantic_stats': {
                'labeled_count': sum(1 for s in symbols if s.semantic_label),
                'labeled_ratio': sum(1 for s in symbols if s.semantic_label) / len(symbols),
                'unique_labels': len(set(s.semantic_label for s in symbols if s.semantic_label))
            }
        }
    
    def _calculate_temporal_distribution(self, symbols: List[VisualSymbol]) -> Dict[str, int]:
        """時間分布の計算"""
        if not symbols:
            return {}
        
        # 月別分布
        monthly_counts = {}
        for symbol in symbols:
            month_key = symbol.creation_timestamp.strftime('%Y-%m')
            monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
        
        return monthly_counts
    
    def _analyze_usage_patterns(self, symbols: List[VisualSymbol]) -> Dict[str, Any]:
        """使用パターンの分析"""
        if not symbols:
            return {}
        
        # 使用頻度分布
        usage_frequencies = [s.usage_frequency for s in symbols]
        
        # 使用レベルの分類
        high_usage = sum(1 for f in usage_frequencies if f > 20)
        medium_usage = sum(1 for f in usage_frequencies if 5 <= f <= 20)
        low_usage = sum(1 for f in usage_frequencies if 1 <= f < 5)
        unused = sum(1 for f in usage_frequencies if f == 0)
        
        return {
            'usage_distribution': {
                'high_usage': high_usage,
                'medium_usage': medium_usage,
                'low_usage': low_usage,
                'unused': unused
            },
            'usage_statistics': {
                'total_usage': sum(usage_frequencies),
                'average_usage': np.mean(usage_frequencies),
                'usage_concentration': np.var(usage_frequencies)  # 使用の集中度
            },
            'popular_symbols': [
                {'symbol_id': s.symbol_id, 'usage_frequency': s.usage_frequency, 'label': s.semantic_label}
                for s in sorted(symbols, key=lambda x: x.usage_frequency, reverse=True)[:5]
            ]
        }
    
    def _perform_comprehensive_memory_analysis(self, symbols: List[VisualSymbol]) -> Dict[str, Any]:
        """包括的記憶分析"""
        if not symbols:
            return {'health_score': 0.0, 'analysis': 'No symbols in memory'}
        
        # 記憶健全性スコアの計算
        health_factors = []
        
        # 品質要因
        avg_confidence = np.mean([s.confidence for s in symbols])
        health_factors.append(avg_confidence)
        
        # 安定性要因
        stable_ratio = sum(1 for s in symbols if s.is_stable_symbol()) / len(symbols)
        health_factors.append(stable_ratio)
        
        # 活用度要因
        active_ratio = sum(1 for s in symbols if s.usage_frequency > 0) / len(symbols)
        health_factors.append(active_ratio)
        
        # 多様性要因
        complexities = [s.prototype_features.get_feature_complexity() for s in symbols]
        diversity_score = min(np.std(complexities) * 2, 1.0)  # 標準偏差による多様性
        health_factors.append(diversity_score)
        
        health_score = np.mean(health_factors)
        
        # 詳細分析
        analysis_details = {
            'quality_assessment': {
                'average_confidence': avg_confidence,
                'high_quality_ratio': sum(1 for s in symbols if s.confidence > 0.8) / len(symbols)
            },
            'stability_assessment': {
                'stable_symbols_ratio': stable_ratio,
                'mature_symbols': sum(1 for s in symbols if (datetime.now() - s.creation_timestamp).days > 7)
            },
            'utilization_assessment': {
                'active_symbols_ratio': active_ratio,
                'underutilized_symbols': sum(1 for s in symbols if s.confidence > 0.7 and s.usage_frequency < 2)
            },
            'diversity_assessment': {
                'complexity_diversity': diversity_score,
                'semantic_diversity': len(set(s.semantic_label for s in symbols if s.semantic_label))
            }
        }
        
        return {
            'health_score': health_score,
            'health_level': self._categorize_health_score(health_score),
            'detailed_analysis': analysis_details,
            'recommendations': self._generate_memory_recommendations(symbols, health_score)
        }
    
    def _categorize_health_score(self, score: float) -> str:
        """健全性スコアの分類"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _generate_memory_recommendations(self, symbols: List[VisualSymbol], health_score: float) -> List[str]:
        """記憶改善推奨事項の生成"""
        recommendations = []
        
        if health_score < 0.6:
            recommendations.append("consider_symbol_quality_improvement")
            
        unused_symbols = sum(1 for s in symbols if s.usage_frequency == 0)
        if unused_symbols > len(symbols) * 0.3:
            recommendations.append("cleanup_unused_symbols")
            
        low_confidence_symbols = sum(1 for s in symbols if s.confidence < 0.5)
        if low_confidence_symbols > len(symbols) * 0.2:
            recommendations.append("retrain_low_confidence_symbols")
            
        if len(symbols) > 1000:
            recommendations.append("consider_memory_consolidation")
            
        return recommendations
    
    def _apply_post_processing(self, 
                              response: VisualMemoryQueryResponse, 
                              request: VisualMemoryQueryRequest) -> VisualMemoryQueryResponse:
        """後処理の適用"""
        # グループ化処理
        if request.group_by_label and response.symbols:
            grouped_results = self._group_symbols_by_label(response.symbols)
            
            # 新しいレスポンスの作成（frozen dataclassのため）
            return VisualMemoryQueryResponse(
                success=response.success,
                query_type=response.query_type,
                results_count=response.results_count,
                total_matches=response.total_matches,
                processing_time=response.processing_time,
                symbols=response.symbols,
                symbol_statistics=response.symbol_statistics,
                similarity_scores=response.similarity_scores,
                memory_analysis=response.memory_analysis,
                relationship_graph=response.relationship_graph,
                usage_patterns=response.usage_patterns,
                temporal_distribution=response.temporal_distribution,
                grouped_results=grouped_results,
                message=response.message,
                error_details=response.error_details,
                session_id=response.session_id,
                response_timestamp=response.response_timestamp,
                search_quality=response.search_quality,
                recommendations=response.recommendations
            )
        
        return response
    
    def _group_symbols_by_label(self, symbols: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """ラベル別グループ化"""
        grouped = {}
        
        for symbol in symbols:
            label = symbol.get('semantic_label') or 'unlabeled'
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(symbol)
        
        return grouped
    
    def _update_query_stats(self, success: bool, processing_time: float):
        """クエリ統計の更新"""
        if success:
            self.query_stats['successful_queries'] += 1
        
        self.query_stats['total_query_time'] += processing_time
        total_queries = max(self.query_stats['total_queries'], 1)
        self.query_stats['avg_query_time'] = (
            self.query_stats['total_query_time'] / total_queries
        )
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """クエリ統計の取得"""
        total_queries = max(self.query_stats['total_queries'], 1)
        
        return {
            'total_queries': self.query_stats['total_queries'],
            'successful_queries': self.query_stats['successful_queries'],
            'cached_responses': self.query_stats['cached_responses'],
            'complex_queries': self.query_stats['complex_queries'],
            'success_rate': self.query_stats['successful_queries'] / total_queries,
            'cache_hit_rate': self.query_stats['cached_responses'] / total_queries,
            'complex_query_rate': self.query_stats['complex_queries'] / total_queries,
            'avg_query_time': self.query_stats['avg_query_time'],
            'total_query_time': self.query_stats['total_query_time'],
            'query_type_distribution': self.query_stats['query_type_distribution'],
            'cache_size': len(self.query_cache) if self.query_cache else 0,
            'advanced_analysis_enabled': self.enable_advanced_analysis,
            'caching_enabled': self.enable_caching
        }
    
    def clear_cache(self):
        """キャッシュのクリア"""
        if self.query_cache:
            self.query_cache.clear()
            self.cache_timestamps.clear()
            self.logger.info("Query cache cleared")
    
    def configure_caching(self, enable_caching: bool, cache_ttl_minutes: Optional[int] = None):
        """キャッシュ設定の変更"""
        if not enable_caching and self.enable_caching:
            self.clear_cache()
            
        self.enable_caching = enable_caching
        
        if cache_ttl_minutes is not None:
            self.cache_ttl_minutes = cache_ttl_minutes
        
        if enable_caching and not self.query_cache:
            self.query_cache = {}
            self.cache_timestamps = {}
        
        self.logger.info(f"Caching configured: enabled={enable_caching}, ttl={self.cache_ttl_minutes}min")
    
    # 高度分析メソッドのプレースホルダー（簡易実装）
    def _analyze_symbol_quality(self, symbol_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """記号品質分析"""
        return {'average_quality': np.mean([s['confidence'] for s in symbol_data])}
    
    def _analyze_symbol_distribution(self, symbol_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """記号分布分析"""
        return {'distribution_entropy': len(set(s['semantic_label'] for s in symbol_data if s['semantic_label']))}
    
    def _analyze_temporal_trends(self, symbol_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """時系列トレンド分析"""
        return {'trend': 'stable'}  # 簡易実装
    
    def _analyze_symbol_relationships(self, symbol_data: List[Dict[str, Any]], reference_feature: VisualFeature) -> Dict[str, Any]:
        """記号関係性分析"""
        return {'relationship_strength': np.mean([s.get('similarity_score', 0.0) for s in symbol_data])}
    
    def _build_symbol_relationship_graph(self, symbols: List[VisualSymbol]) -> Dict[str, List[str]]:
        """記号関係グラフの構築"""
        # 簡易実装：類似度に基づく関係
        graph = {}
        similarity_threshold = 0.7
        
        for i, symbol1 in enumerate(symbols):
            related_symbols = []
            for j, symbol2 in enumerate(symbols):
                if i != j:
                    similarity = symbol1.prototype_features.calculate_similarity(symbol2.prototype_features)
                    if similarity >= similarity_threshold:
                        related_symbols.append(symbol2.symbol_id)
            
            graph[symbol1.symbol_id] = related_symbols
        
        return graph
    
    def _calculate_complexity_distribution(self, complexities: List[float]) -> Dict[str, int]:
        """複雑度分布の計算"""
        distribution = {'low': 0, 'medium': 0, 'high': 0}
        
        for complexity in complexities:
            if complexity < 0.3:
                distribution['low'] += 1
            elif complexity < 0.7:
                distribution['medium'] += 1
            else:
                distribution['high'] += 1
        
        return distribution