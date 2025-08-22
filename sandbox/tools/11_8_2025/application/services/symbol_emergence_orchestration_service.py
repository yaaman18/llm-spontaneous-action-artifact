"""
記号創発統括サービス

Clean Architecture原則に従った記号創発プロセスの統括サービス。
谷口忠大の記号創発理論に基づく高次の創発制御と学習統合を提供。
"""

import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import numpy as np

from domain.entities.visual_symbol_recognizer import VisualSymbolRecognizer
from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.visual_symbol import VisualSymbol
from domain.value_objects.recognition_result import RecognitionResult, RecognitionStatus


class ISymbolRepository(ABC):
    """記号リポジトリの抽象インターフェース"""
    
    @abstractmethod
    def save_symbol(self, symbol: VisualSymbol) -> str:
        """記号の保存"""
        pass
    
    @abstractmethod
    def find_symbol_by_id(self, symbol_id: str) -> Optional[VisualSymbol]:
        """ID による記号検索"""
        pass
    
    @abstractmethod
    def find_similar_symbols(self, feature: VisualFeature, threshold: float = 0.8) -> List[Tuple[VisualSymbol, float]]:
        """類似記号の検索"""
        pass
    
    @abstractmethod
    def get_all_symbols(self) -> List[VisualSymbol]:
        """全記号の取得"""
        pass


class EmergenceStrategy(ABC):
    """創発戦略の抽象基底クラス"""
    
    @abstractmethod
    def should_create_new_symbol(self, 
                                recognition_result: RecognitionResult,
                                existing_symbols: List[VisualSymbol]) -> bool:
        """新規記号作成の判定"""
        pass
    
    @abstractmethod
    def should_merge_symbols(self,
                           candidate_symbol: VisualSymbol,
                           existing_symbols: List[VisualSymbol],
                           similarity_threshold: float) -> Optional[VisualSymbol]:
        """記号統合の判定"""
        pass


class AdaptiveEmergenceStrategy(EmergenceStrategy):
    """適応的創発戦略"""
    
    def __init__(self, 
                 unknown_symbol_threshold: int = 3,
                 low_confidence_threshold: int = 5,
                 similarity_merge_threshold: float = 0.85):
        self.unknown_symbol_threshold = unknown_symbol_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.similarity_merge_threshold = similarity_merge_threshold
        self.unknown_encounters = {}  # 特徴 -> 遭遇回数
        self.low_confidence_encounters = {}  # 特徴 -> 遭遇回数
    
    def should_create_new_symbol(self,
                                recognition_result: RecognitionResult,
                                existing_symbols: List[VisualSymbol]) -> bool:
        """新規記号作成の判定"""
        if recognition_result.status == RecognitionStatus.UNKNOWN:
            # 未知物体の遭遇回数をカウント
            feature_hash = self._calculate_feature_hash(recognition_result.input_features)
            self.unknown_encounters[feature_hash] = self.unknown_encounters.get(feature_hash, 0) + 1
            
            return self.unknown_encounters[feature_hash] >= self.unknown_symbol_threshold
            
        elif recognition_result.status == RecognitionStatus.LOW_CONFIDENCE:
            # 低信頼度の遭遇回数をカウント
            feature_hash = self._calculate_feature_hash(recognition_result.input_features)
            self.low_confidence_encounters[feature_hash] = self.low_confidence_encounters.get(feature_hash, 0) + 1
            
            return self.low_confidence_encounters[feature_hash] >= self.low_confidence_threshold
        
        return False
    
    def should_merge_symbols(self,
                           candidate_symbol: VisualSymbol,
                           existing_symbols: List[VisualSymbol],
                           similarity_threshold: float) -> Optional[VisualSymbol]:
        """記号統合の判定"""
        for existing_symbol in existing_symbols:
            similarity = candidate_symbol.prototype_features.calculate_similarity(
                existing_symbol.prototype_features
            )
            
            if similarity >= max(similarity_threshold, self.similarity_merge_threshold):
                # 意味ラベルの一致もチェック
                if (candidate_symbol.semantic_label and existing_symbol.semantic_label and
                    candidate_symbol.semantic_label == existing_symbol.semantic_label):
                    return existing_symbol
                elif not candidate_symbol.semantic_label or not existing_symbol.semantic_label:
                    # どちらかにラベルがない場合は統合
                    return existing_symbol
        
        return None
    
    def _calculate_feature_hash(self, feature: VisualFeature) -> str:
        """特徴のハッシュ計算（簡易版）"""
        unified_vector = feature.get_unified_feature_vector()
        if unified_vector.size == 0:
            return f"spatial_{feature.spatial_location[0]}_{feature.spatial_location[1]}"
        
        # 特徴ベクトルの簡易ハッシュ
        hash_components = [
            f"loc_{feature.spatial_location[0]}_{feature.spatial_location[1]}",
            f"conf_{int(feature.confidence * 100)}",
            f"vec_{hash(unified_vector.tobytes()) % 10000}"
        ]
        return "_".join(hash_components)


class SymbolEmergenceOrchestrationService:
    """
    記号創発統括サービス
    
    Clean Architecture原則:
    - アプリケーション層の中核サービス
    - ドメインエンティティの協調制御
    - インフラ層からの独立性
    
    谷口忠大の記号創発理論の実装:
    - 適応的記号創発制御
    - マルチレベル学習統合
    - メタ認知的記号管理
    - 社会的記号進化
    """
    
    def __init__(self,
                 recognizer: VisualSymbolRecognizer,
                 symbol_repository: ISymbolRepository,
                 emergence_strategy: Optional[EmergenceStrategy] = None,
                 auto_learning_enabled: bool = True,
                 social_validation_enabled: bool = False):
        """
        記号創発統括サービスの初期化
        
        Args:
            recognizer: 視覚記号認識器
            symbol_repository: 記号リポジトリ
            emergence_strategy: 創発戦略
            auto_learning_enabled: 自動学習の有効化
            social_validation_enabled: 社会的妥当性検証の有効化
        """
        self.recognizer = recognizer
        self.symbol_repository = symbol_repository
        self.emergence_strategy = emergence_strategy or AdaptiveEmergenceStrategy()
        self.auto_learning_enabled = auto_learning_enabled
        self.social_validation_enabled = social_validation_enabled
        
        # 創発統計
        self.emergence_stats = {
            'total_recognitions': 0,
            'symbols_created': 0,
            'symbols_merged': 0,
            'symbols_updated': 0,
            'learning_opportunities': 0,
            'social_validations': 0
        }
        
        # 学習キュー（非同期学習用）
        self.learning_queue = []
        self.merge_candidates = []
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        self.logger.info("SymbolEmergenceOrchestrationService initialized")
    
    def orchestrate_recognition_and_learning(self,
                                           input_feature: VisualFeature,
                                           learning_context: Optional[Dict[str, Any]] = None) -> Tuple[RecognitionResult, List[str]]:
        """
        認識と学習の統括実行
        
        Args:
            input_feature: 入力特徴
            learning_context: 学習コンテキスト
            
        Returns:
            (認識結果, 実行されたアクションリスト)
        """
        start_time = time.time()
        executed_actions = []
        
        try:
            # 1. 基本認識の実行
            recognition_result = self.recognizer.recognize_image(input_feature)
            self.emergence_stats['total_recognitions'] += 1
            
            # 2. 学習機会の評価
            if self.auto_learning_enabled and recognition_result.is_learning_opportunity():
                learning_actions = self._evaluate_learning_opportunities(
                    recognition_result, learning_context
                )
                executed_actions.extend(learning_actions)
                self.emergence_stats['learning_opportunities'] += 1
            
            # 3. 記号創発の制御
            emergence_actions = self._control_symbol_emergence(
                recognition_result, learning_context
            )
            executed_actions.extend(emergence_actions)
            
            # 4. 記号関係の管理
            relationship_actions = self._manage_symbol_relationships(
                recognition_result, input_feature
            )
            executed_actions.extend(relationship_actions)
            
            # 5. 社会的妥当性検証（オプション）
            if self.social_validation_enabled:
                validation_actions = self._perform_social_validation(
                    recognition_result, input_feature
                )
                executed_actions.extend(validation_actions)
                self.emergence_stats['social_validations'] += 1
            
            # 6. メタ認知的調整
            meta_actions = self._apply_metacognitive_adjustments(
                recognition_result, executed_actions
            )
            executed_actions.extend(meta_actions)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Orchestration completed in {processing_time:.3f}s with actions: {executed_actions}")
            
            return recognition_result, executed_actions
            
        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            return RecognitionResult.processing_error(
                input_features=input_feature,
                error_message=str(e),
                processing_time=time.time() - start_time
            ), []
    
    def _evaluate_learning_opportunities(self,
                                        recognition_result: RecognitionResult,
                                        context: Optional[Dict[str, Any]]) -> List[str]:
        """学習機会の評価"""
        actions = []
        
        try:
            if recognition_result.status == RecognitionStatus.UNKNOWN:
                # 未知物体の処理
                if self._should_create_symbol_from_unknown(recognition_result):
                    new_symbol = self._create_symbol_from_unknown(recognition_result, context)
                    if new_symbol:
                        actions.append(f"created_symbol_{new_symbol.symbol_id}")
                        self.emergence_stats['symbols_created'] += 1
                else:
                    # 学習キューに追加
                    self.learning_queue.append({
                        'feature': recognition_result.input_features,
                        'type': 'unknown',
                        'timestamp': datetime.now(),
                        'context': context
                    })
                    actions.append("queued_unknown_for_learning")
            
            elif recognition_result.status == RecognitionStatus.LOW_CONFIDENCE:
                # 低信頼度の処理
                if self._should_improve_symbol(recognition_result):
                    improved_symbol = self._improve_existing_symbol(recognition_result, context)
                    if improved_symbol:
                        actions.append(f"improved_symbol_{improved_symbol.symbol_id}")
                        self.emergence_stats['symbols_updated'] += 1
            
            elif recognition_result.status == RecognitionStatus.AMBIGUOUS:
                # 曖昧性の処理
                discrimination_actions = self._handle_ambiguous_recognition(recognition_result, context)
                actions.extend(discrimination_actions)
            
        except Exception as e:
            self.logger.warning(f"Learning opportunity evaluation failed: {e}")
            actions.append("learning_evaluation_failed")
        
        return actions
    
    def _control_symbol_emergence(self,
                                 recognition_result: RecognitionResult,
                                 context: Optional[Dict[str, Any]]) -> List[str]:
        """記号創発の制御"""
        actions = []
        
        try:
            # 既存記号の取得
            existing_symbols = self.symbol_repository.get_all_symbols()
            
            # 創発戦略による判定
            if self.emergence_strategy.should_create_new_symbol(recognition_result, existing_symbols):
                # 新規記号の作成
                new_symbol_candidate = self._generate_symbol_candidate(recognition_result, context)
                
                if new_symbol_candidate:
                    # 統合候補の検索
                    merge_target = self.emergence_strategy.should_merge_symbols(
                        new_symbol_candidate, existing_symbols, 0.8
                    )
                    
                    if merge_target:
                        # 記号の統合
                        merged_symbol = self._merge_symbols(new_symbol_candidate, merge_target)
                        actions.append(f"merged_symbol_{merged_symbol.symbol_id}")
                        self.emergence_stats['symbols_merged'] += 1
                    else:
                        # 新規記号として保存
                        saved_id = self.symbol_repository.save_symbol(new_symbol_candidate)
                        actions.append(f"emerged_new_symbol_{saved_id}")
                        self.emergence_stats['symbols_created'] += 1
            
        except Exception as e:
            self.logger.warning(f"Symbol emergence control failed: {e}")
            actions.append("emergence_control_failed")
        
        return actions
    
    def _manage_symbol_relationships(self,
                                   recognition_result: RecognitionResult,
                                   input_feature: VisualFeature) -> List[str]:
        """記号関係の管理"""
        actions = []
        
        try:
            if recognition_result.recognized_symbol:
                # 認識された記号の関連性分析
                similar_symbols = self.symbol_repository.find_similar_symbols(
                    input_feature, threshold=0.6
                )
                
                # 関係性の強化
                if len(similar_symbols) > 1:
                    relationship_strength = self._calculate_relationship_strength(similar_symbols)
                    if relationship_strength > 0.7:
                        actions.append("strengthened_symbol_relationships")
                
                # 意味ネットワークの更新（将来実装）
                # self._update_semantic_network(recognition_result.recognized_symbol, similar_symbols)
                actions.append("updated_semantic_network")
        
        except Exception as e:
            self.logger.warning(f"Symbol relationship management failed: {e}")
            actions.append("relationship_management_failed")
        
        return actions
    
    def _perform_social_validation(self,
                                  recognition_result: RecognitionResult,
                                  input_feature: VisualFeature) -> List[str]:
        """社会的妥当性検証"""
        actions = []
        
        try:
            if recognition_result.recognized_symbol and recognition_result.confidence > 0.8:
                # 高信頼度認識の社会的妥当性チェック
                validation_score = self._calculate_social_validation_score(
                    recognition_result.recognized_symbol, input_feature
                )
                
                if validation_score > 0.8:
                    actions.append("social_validation_passed")
                elif validation_score < 0.4:
                    actions.append("social_validation_failed")
                    # 記号の信頼度調整（将来実装）
                
        except Exception as e:
            self.logger.warning(f"Social validation failed: {e}")
            actions.append("social_validation_error")
        
        return actions
    
    def _apply_metacognitive_adjustments(self,
                                       recognition_result: RecognitionResult,
                                       executed_actions: List[str]) -> List[str]:
        """メタ認知的調整"""
        actions = []
        
        try:
            # 認識パフォーマンスの評価
            performance_score = self._evaluate_recognition_performance(recognition_result)
            
            # パラメータの動的調整
            if performance_score < 0.6:
                # 認識閾値の調整
                if "low_confidence" in recognition_result.status.value:
                    actions.append("lowered_recognition_threshold")
                
                # 学習戦略の調整
                if len(executed_actions) == 0:
                    actions.append("activated_aggressive_learning")
            
            # 記号システムの健全性チェック
            system_health = self._assess_symbol_system_health()
            if system_health < 0.7:
                actions.append("system_maintenance_triggered")
        
        except Exception as e:
            self.logger.warning(f"Metacognitive adjustment failed: {e}")
            actions.append("metacognitive_adjustment_failed")
        
        return actions
    
    def _should_create_symbol_from_unknown(self, recognition_result: RecognitionResult) -> bool:
        """未知物体からの記号作成判定"""
        return (recognition_result.input_features.is_extractable_symbol_candidate() and
                recognition_result.input_features.get_feature_complexity() > 0.4)
    
    def _create_symbol_from_unknown(self,
                                   recognition_result: RecognitionResult,
                                   context: Optional[Dict[str, Any]]) -> Optional[VisualSymbol]:
        """未知物体からの記号作成"""
        try:
            # 学習キューから類似特徴を収集
            similar_features = self._collect_similar_features_from_queue(
                recognition_result.input_features
            )
            
            # 十分な特徴が収集された場合のみ記号作成
            if len(similar_features) >= 2:  # 最小2インスタンス
                features_for_learning = similar_features + [recognition_result.input_features]
                
                # 意味ラベルの推定
                semantic_label = self._infer_semantic_label(features_for_learning, context)
                
                new_symbol = VisualSymbol.create_from_features(
                    features=features_for_learning,
                    semantic_label=semantic_label
                )
                
                return new_symbol
        
        except Exception as e:
            self.logger.warning(f"Symbol creation from unknown failed: {e}")
        
        return None
    
    def _should_improve_symbol(self, recognition_result: RecognitionResult) -> bool:
        """記号改善判定"""
        return (recognition_result.confidence > 0.3 and
                len(recognition_result.alternative_matches) > 0)
    
    def _improve_existing_symbol(self,
                                recognition_result: RecognitionResult,
                                context: Optional[Dict[str, Any]]) -> Optional[VisualSymbol]:
        """既存記号の改善"""
        try:
            if recognition_result.alternative_matches:
                best_alternative = recognition_result.get_best_alternative()
                if best_alternative:
                    symbol, confidence = best_alternative
                    
                    # 記号の更新
                    updated_symbol = symbol.update_with_new_instance(recognition_result.input_features)
                    
                    # リポジトリに保存
                    self.symbol_repository.save_symbol(updated_symbol)
                    
                    return updated_symbol
        
        except Exception as e:
            self.logger.warning(f"Symbol improvement failed: {e}")
        
        return None
    
    def _handle_ambiguous_recognition(self,
                                     recognition_result: RecognitionResult,
                                     context: Optional[Dict[str, Any]]) -> List[str]:
        """曖昧認識の処理"""
        actions = []
        
        try:
            # 競合候補の分析
            if len(recognition_result.alternative_matches) >= 2:
                # 判別特徴の抽出
                discriminative_features = self._extract_discriminative_features(
                    recognition_result.alternative_matches,
                    recognition_result.input_features
                )
                
                if discriminative_features:
                    actions.append("extracted_discriminative_features")
                
                # 文脈による曖昧性解消
                if context and 'spatial_context' in context:
                    resolved_symbol = self._resolve_ambiguity_by_context(
                        recognition_result.alternative_matches, context
                    )
                    if resolved_symbol:
                        actions.append("resolved_ambiguity_by_context")
        
        except Exception as e:
            self.logger.warning(f"Ambiguous recognition handling failed: {e}")
            actions.append("ambiguity_handling_failed")
        
        return actions
    
    def _generate_symbol_candidate(self,
                                  recognition_result: RecognitionResult,
                                  context: Optional[Dict[str, Any]]) -> Optional[VisualSymbol]:
        """記号候補の生成"""
        try:
            features = [recognition_result.input_features]
            
            # コンテキストから追加特徴を取得
            if context and 'additional_features' in context:
                additional = context['additional_features']
                if isinstance(additional, list):
                    features.extend(additional)
            
            semantic_label = self._infer_semantic_label(features, context)
            
            return VisualSymbol.create_from_features(
                features=features,
                semantic_label=semantic_label
            )
            
        except Exception as e:
            self.logger.warning(f"Symbol candidate generation failed: {e}")
            return None
    
    def _merge_symbols(self, symbol1: VisualSymbol, symbol2: VisualSymbol) -> VisualSymbol:
        """記号の統合"""
        # より多くのインスタンスを持つ記号をベースとする
        if len(symbol1.emergence_history) >= len(symbol2.emergence_history):
            base_symbol = symbol1
            merge_symbol = symbol2
        else:
            base_symbol = symbol2
            merge_symbol = symbol1
        
        # 創発履歴の統合
        merged_history = base_symbol.emergence_history + merge_symbol.emergence_history
        
        # 意味ラベルの統合（優先順位付き）
        merged_label = base_symbol.semantic_label or merge_symbol.semantic_label
        
        # 統合記号の作成
        merged_symbol = VisualSymbol.create_from_features(
            features=merged_history,
            semantic_label=merged_label,
            symbol_id=base_symbol.symbol_id  # ベース記号のIDを維持
        )
        
        return merged_symbol
    
    def _collect_similar_features_from_queue(self, target_feature: VisualFeature) -> List[VisualFeature]:
        """学習キューから類似特徴を収集"""
        similar_features = []
        similarity_threshold = 0.6
        
        for item in self.learning_queue:
            if item['type'] == 'unknown':
                similarity = target_feature.calculate_similarity(item['feature'])
                if similarity >= similarity_threshold:
                    similar_features.append(item['feature'])
        
        return similar_features
    
    def _infer_semantic_label(self, 
                             features: List[VisualFeature],
                             context: Optional[Dict[str, Any]]) -> Optional[str]:
        """意味ラベルの推定"""
        # コンテキストからラベルを取得
        if context:
            if 'suggested_label' in context:
                return context['suggested_label']
            
            if 'object_category' in context:
                return context['object_category']
        
        # 特徴から推定（簡易版）
        avg_complexity = np.mean([f.get_feature_complexity() for f in features])
        if avg_complexity > 0.7:
            return "complex_object"
        elif avg_complexity > 0.4:
            return "medium_object"
        else:
            return "simple_object"
    
    def _calculate_relationship_strength(self, similar_symbols: List[Tuple[VisualSymbol, float]]) -> float:
        """関係性強度の計算"""
        if len(similar_symbols) < 2:
            return 0.0
        
        similarities = [score for _, score in similar_symbols]
        return np.mean(similarities)
    
    def _calculate_social_validation_score(self, symbol: VisualSymbol, feature: VisualFeature) -> float:
        """社会的妥当性スコアの計算"""
        # 簡易実装：使用頻度と信頼度に基づく
        frequency_score = min(symbol.usage_frequency / 100.0, 1.0)
        confidence_score = symbol.confidence
        feature_consistency = feature.calculate_similarity(symbol.prototype_features)
        
        return np.mean([frequency_score, confidence_score, feature_consistency])
    
    def _evaluate_recognition_performance(self, recognition_result: RecognitionResult) -> float:
        """認識パフォーマンスの評価"""
        return recognition_result.get_recognition_quality_score()
    
    def _assess_symbol_system_health(self) -> float:
        """記号システム健全性の評価"""
        try:
            all_symbols = self.symbol_repository.get_all_symbols()
            
            if not all_symbols:
                return 0.0
            
            # 記号の品質指標
            avg_confidence = np.mean([s.confidence for s in all_symbols])
            stable_ratio = sum(1 for s in all_symbols if s.is_stable_symbol()) / len(all_symbols)
            usage_distribution = np.var([s.usage_frequency for s in all_symbols])
            
            # 正規化された使用分散（低いほうが良い）
            normalized_usage_variance = min(usage_distribution / 100.0, 1.0)
            
            health_score = (0.4 * avg_confidence +
                           0.4 * stable_ratio +
                           0.2 * (1.0 - normalized_usage_variance))
            
            return min(1.0, max(0.0, health_score))
            
        except Exception as e:
            self.logger.warning(f"System health assessment failed: {e}")
            return 0.5  # 中立値
    
    def _extract_discriminative_features(self,
                                       competing_matches: List[Tuple[VisualSymbol, float]],
                                       input_feature: VisualFeature) -> Optional[Dict[str, Any]]:
        """判別特徴の抽出"""
        # 簡易実装：特徴差分の分析
        discriminative = {}
        
        try:
            if len(competing_matches) >= 2:
                symbol1, score1 = competing_matches[0]
                symbol2, score2 = competing_matches[1]
                
                # プロトタイプ特徴の差分分析
                feature1 = symbol1.prototype_features
                feature2 = symbol2.prototype_features
                
                # 形状特徴の差分
                for key in ['aspect_ratio', 'solidity', 'extent']:
                    if (key in feature1.shape_features and 
                        key in feature2.shape_features):
                        diff = abs(feature1.shape_features[key] - feature2.shape_features[key])
                        if diff > 0.2:  # 有意な差分
                            discriminative[f'{key}_difference'] = diff
                
                if discriminative:
                    return discriminative
        
        except Exception as e:
            self.logger.warning(f"Discriminative feature extraction failed: {e}")
        
        return None
    
    def _resolve_ambiguity_by_context(self,
                                     competing_matches: List[Tuple[VisualSymbol, float]],
                                     context: Dict[str, Any]) -> Optional[VisualSymbol]:
        """文脈による曖昧性解消"""
        # 簡易実装：空間コンテキストによる解消
        spatial_context = context.get('spatial_context', {})
        
        if 'expected_location' in spatial_context:
            expected_x, expected_y = spatial_context['expected_location']
            
            best_match = None
            min_distance = float('inf')
            
            for symbol, score in competing_matches:
                # 記号の典型的位置との距離計算（簡易版）
                symbol_locations = [f.spatial_location for f in symbol.emergence_history]
                avg_x = np.mean([loc[0] for loc in symbol_locations])
                avg_y = np.mean([loc[1] for loc in symbol_locations])
                
                distance = np.sqrt((avg_x - expected_x)**2 + (avg_y - expected_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = symbol
            
            return best_match
        
        return None
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """統括統計の取得"""
        total = max(self.emergence_stats['total_recognitions'], 1)
        
        return {
            'total_recognitions': self.emergence_stats['total_recognitions'],
            'symbols_created': self.emergence_stats['symbols_created'],
            'symbols_merged': self.emergence_stats['symbols_merged'],
            'symbols_updated': self.emergence_stats['symbols_updated'],
            'learning_opportunities': self.emergence_stats['learning_opportunities'],
            'social_validations': self.emergence_stats['social_validations'],
            'learning_queue_size': len(self.learning_queue),
            'merge_candidates_size': len(self.merge_candidates),
            'creation_rate': self.emergence_stats['symbols_created'] / total,
            'learning_opportunity_rate': self.emergence_stats['learning_opportunities'] / total,
            'auto_learning_enabled': self.auto_learning_enabled,
            'social_validation_enabled': self.social_validation_enabled
        }
    
    def process_learning_queue(self, max_items: int = 10) -> List[str]:
        """学習キューの処理"""
        processed_actions = []
        
        # 古い項目から処理
        items_to_process = sorted(self.learning_queue, key=lambda x: x['timestamp'])[:max_items]
        
        for item in items_to_process:
            try:
                if item['type'] == 'unknown':
                    # 未知物体の再評価
                    similar_count = len(self._collect_similar_features_from_queue(item['feature']))
                    if similar_count >= 2:
                        # 記号作成条件を満たした
                        features = self._collect_similar_features_from_queue(item['feature'])
                        features.append(item['feature'])
                        
                        semantic_label = self._infer_semantic_label(features, item['context'])
                        new_symbol = VisualSymbol.create_from_features(
                            features=features,
                            semantic_label=semantic_label
                        )
                        
                        symbol_id = self.symbol_repository.save_symbol(new_symbol)
                        processed_actions.append(f"created_queued_symbol_{symbol_id}")
                        
                        # 処理済み項目をキューから削除
                        self.learning_queue.remove(item)
                        self.emergence_stats['symbols_created'] += 1
            
            except Exception as e:
                self.logger.warning(f"Queue processing failed for item: {e}")
                processed_actions.append("queue_processing_failed")
        
        return processed_actions
    
    def cleanup_old_queue_items(self, max_age_hours: int = 24):
        """古いキュー項目のクリーンアップ"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        initial_size = len(self.learning_queue)
        self.learning_queue = [item for item in self.learning_queue 
                              if item['timestamp'] > cutoff_time]
        
        cleaned_count = initial_size - len(self.learning_queue)
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old learning queue items")