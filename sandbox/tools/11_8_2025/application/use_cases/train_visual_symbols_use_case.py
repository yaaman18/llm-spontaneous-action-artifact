"""
視覚記号学習ユースケース

Clean Architecture原則に従った記号学習の中核ユースケース。
谷口忠大の記号創発理論に基づく体系的な記号学習プロセスを実現。
"""

import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

from application.dtos.symbol_learning_dto import SymbolLearningRequest, SymbolLearningResponse
from application.services.visual_feature_extraction_service import VisualFeatureExtractionService
from application.services.symbol_emergence_orchestration_service import (
    SymbolEmergenceOrchestrationService, 
    ISymbolRepository
)
from domain.entities.visual_symbol_recognizer import VisualSymbolRecognizer
from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.visual_symbol import VisualSymbol


class TrainVisualSymbolsUseCase:
    """
    視覚記号学習ユースケース
    
    Clean Architecture原則:
    - アプリケーション層のユースケース
    - 記号学習ワークフローの統合制御
    - ドメインロジックとの協調
    
    谷口忠大の記号創発理論の実装:
    - プロトタイプベース学習
    - 継続的記号改良
    - 社会的記号妥当性検証
    - 適応的学習戦略
    """
    
    def __init__(self,
                 visual_symbol_recognizer: VisualSymbolRecognizer,
                 symbol_repository: ISymbolRepository,
                 feature_extraction_service: Optional[VisualFeatureExtractionService] = None,
                 enable_validation: bool = True,
                 enable_merge_detection: bool = True):
        """
        記号学習ユースケースの初期化
        
        Args:
            visual_symbol_recognizer: 視覚記号認識器
            symbol_repository: 記号リポジトリ
            feature_extraction_service: 特徴抽出サービス（オプション）
            enable_validation: 妥当性検証の有効化
            enable_merge_detection: 記号統合検出の有効化
        """
        self.visual_symbol_recognizer = visual_symbol_recognizer
        self.symbol_repository = symbol_repository
        self.feature_extraction_service = feature_extraction_service
        self.enable_validation = enable_validation
        self.enable_merge_detection = enable_merge_detection
        
        # 学習統計
        self.learning_stats = {
            'total_training_sessions': 0,
            'symbols_created': 0,
            'symbols_merged': 0,
            'symbols_updated': 0,
            'validation_failures': 0,
            'successful_trainings': 0,
            'avg_training_time': 0.0,
            'total_training_time': 0.0
        }
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        self.logger.info("TrainVisualSymbolsUseCase initialized")
    
    def execute(self, request: SymbolLearningRequest) -> SymbolLearningResponse:
        """
        記号学習ユースケースの実行
        
        Args:
            request: 記号学習リクエスト
            
        Returns:
            記号学習レスポンス
        """
        start_time = time.time()
        self.learning_stats['total_training_sessions'] += 1
        
        self.logger.info(f"Starting symbol learning with {len(request.training_features)} features")
        
        try:
            # 1. 入力検証
            validation_results = self._validate_training_request(request)
            if validation_results['has_errors']:
                return SymbolLearningResponse.failure_response(
                    error_message=f"Validation failed: {', '.join(validation_results['errors'])}",
                    training_instances=len(request.training_features),
                    processing_time=time.time() - start_time,
                    session_id=request.session_id,
                    validation_results=validation_results
                )
            
            # 2. 特徴前処理
            processed_features = self._preprocess_training_features(request)
            
            # 3. 記号統合検出
            merge_candidate = None
            merge_operations = []
            
            if self.enable_merge_detection and request.merge_similar_symbols:
                merge_candidate, merge_operations = self._detect_merge_candidate(
                    processed_features, request
                )
            
            # 4. 記号学習の実行
            if merge_candidate:
                # 既存記号の更新
                learned_symbol = self._update_existing_symbol(
                    merge_candidate, processed_features, request
                )
                learning_type = "merge_based"
                self.learning_stats['symbols_merged'] += 1
                
            elif request.incremental_update and request.symbol_id:
                # インクリメンタル更新
                learned_symbol = self._perform_incremental_update(
                    processed_features, request
                )
                learning_type = "incremental"
                self.learning_stats['symbols_updated'] += 1
                
            else:
                # 新規記号作成
                learned_symbol = self._create_new_symbol(processed_features, request)
                learning_type = "independent"
                self.learning_stats['symbols_created'] += 1
            
            # 5. 記号の保存
            saved_symbol_id = self.symbol_repository.save_symbol(learned_symbol)
            
            # 6. 認識器への統合
            self._integrate_symbol_to_recognizer(learned_symbol)
            
            # 7. 学習後検証
            post_validation = self._perform_post_learning_validation(
                learned_symbol, processed_features
            )
            
            # 8. レスポンスの構築
            response = SymbolLearningResponse.success_response(
                learned_symbol=learned_symbol,
                training_instances=len(request.training_features),
                processing_time=time.time() - start_time,
                session_id=request.session_id,
                learning_strategy=learning_type,
                merge_operations=merge_operations,
                warnings=validation_results.get('warnings', [])
            )
            
            # 統計更新
            self._update_learning_stats(True, time.time() - start_time)
            
            self.logger.info(f"Symbol learning completed: {learned_symbol.symbol_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Symbol learning failed: {e}")
            self._update_learning_stats(False, time.time() - start_time)
            
            return SymbolLearningResponse.failure_response(
                error_message=str(e),
                training_instances=len(request.training_features) if request.training_features else 0,
                processing_time=time.time() - start_time,
                session_id=request.session_id
            )
    
    def _validate_training_request(self, request: SymbolLearningRequest) -> Dict[str, Any]:
        """学習リクエストの妥当性検証"""
        validation_result = {
            'has_errors': False,
            'errors': [],
            'warnings': [],
            'quality_metrics': {}
        }
        
        try:
            # リクエスト自体の妥当性チェックは __post_init__ で実行済み
            
            # 学習データの品質チェック
            if self.enable_validation:
                data_issues = request.validate_training_data()
                if data_issues:
                    validation_result['warnings'].extend(data_issues)
            
            # 特徴の品質評価
            quality_scores = []
            valid_features = []
            
            for i, feature in enumerate(request.training_features):
                try:
                    if feature.is_extractable_symbol_candidate():
                        quality_scores.append(feature.get_feature_complexity())
                        valid_features.append(feature)
                    else:
                        validation_result['warnings'].append(
                            f"Feature {i} is not a suitable symbol candidate"
                        )
                except Exception as e:
                    validation_result['warnings'].append(
                        f"Feature {i} quality assessment failed: {e}"
                    )
            
            # 最小品質要件
            if not valid_features:
                validation_result['has_errors'] = True
                validation_result['errors'].append("No valid features for symbol learning")
            
            elif len(valid_features) < request.min_instances:
                validation_result['has_errors'] = True
                validation_result['errors'].append(
                    f"Insufficient valid features: {len(valid_features)} < {request.min_instances}"
                )
            
            # 品質メトリクスの計算
            if quality_scores:
                validation_result['quality_metrics'] = {
                    'avg_complexity': sum(quality_scores) / len(quality_scores),
                    'min_complexity': min(quality_scores),
                    'max_complexity': max(quality_scores),
                    'valid_feature_ratio': len(valid_features) / len(request.training_features)
                }
        
        except Exception as e:
            validation_result['has_errors'] = True
            validation_result['errors'].append(f"Validation process failed: {e}")
        
        return validation_result
    
    def _preprocess_training_features(self, request: SymbolLearningRequest) -> List[VisualFeature]:
        """学習特徴の前処理"""
        processed_features = []
        
        for feature in request.training_features:
            try:
                # 基本的な品質チェック
                if feature.is_extractable_symbol_candidate():
                    processed_features.append(feature)
                else:
                    self.logger.debug(f"Filtered out low-quality feature with confidence {feature.confidence}")
            
            except Exception as e:
                self.logger.warning(f"Feature preprocessing failed: {e}")
        
        # 特徴の順序を安定させる（再現性のため）
        processed_features.sort(key=lambda f: (f.extraction_timestamp, f.confidence), reverse=True)
        
        self.logger.info(f"Preprocessed {len(processed_features)} valid features from {len(request.training_features)}")
        
        return processed_features
    
    def _detect_merge_candidate(self, 
                               features: List[VisualFeature],
                               request: SymbolLearningRequest) -> tuple[Optional[VisualSymbol], List[str]]:
        """記号統合候補の検出"""
        merge_operations = []
        
        try:
            # 代表特徴を用いた類似記号検索
            representative_feature = self._select_representative_feature(features)
            
            similar_symbols = self.symbol_repository.find_similar_symbols(
                representative_feature, 
                threshold=request.similarity_threshold
            )
            
            if similar_symbols:
                # 最も類似度の高い記号を統合候補とする
                best_match, similarity = similar_symbols[0]
                
                merge_operations.append(f"found_similar_symbol_{best_match.symbol_id}_similarity_{similarity:.3f}")
                
                # 意味ラベルの整合性チェック
                if request.semantic_label and best_match.semantic_label:
                    if request.semantic_label != best_match.semantic_label:
                        merge_operations.append(f"semantic_label_conflict_{request.semantic_label}_vs_{best_match.semantic_label}")
                        
                        # 意味ラベルが異なる場合は統合しない
                        return None, merge_operations
                
                merge_operations.append(f"selected_merge_candidate_{best_match.symbol_id}")
                return best_match, merge_operations
        
        except Exception as e:
            self.logger.warning(f"Merge candidate detection failed: {e}")
            merge_operations.append(f"merge_detection_failed_{str(e)}")
        
        return None, merge_operations
    
    def _select_representative_feature(self, features: List[VisualFeature]) -> VisualFeature:
        """代表特徴の選択"""
        if len(features) == 1:
            return features[0]
        
        # 最も複雑度が高く、信頼度も高い特徴を選択
        scored_features = [
            (f, f.get_feature_complexity() * f.confidence) 
            for f in features
        ]
        
        return max(scored_features, key=lambda x: x[1])[0]
    
    def _update_existing_symbol(self,
                               existing_symbol: VisualSymbol,
                               new_features: List[VisualFeature],
                               request: SymbolLearningRequest) -> VisualSymbol:
        """既存記号の更新（統合）"""
        try:
            # 新しい特徴を既存記号に統合
            updated_symbol = existing_symbol
            
            for feature in new_features:
                updated_symbol = updated_symbol.update_with_new_instance(feature)
            
            # 意味ラベルの更新（必要に応じて）
            if request.semantic_label and not updated_symbol.semantic_label:
                # 既存記号にラベルがない場合は新しいラベルを適用
                updated_symbol = VisualSymbol(
                    symbol_id=updated_symbol.symbol_id,
                    prototype_features=updated_symbol.prototype_features,
                    variation_range=updated_symbol.variation_range,
                    emergence_history=updated_symbol.emergence_history,
                    semantic_label=request.semantic_label,
                    confidence=updated_symbol.confidence,
                    usage_frequency=updated_symbol.usage_frequency,
                    creation_timestamp=updated_symbol.creation_timestamp,
                    last_updated=datetime.now()
                )
            
            self.logger.info(f"Updated existing symbol {updated_symbol.symbol_id} with {len(new_features)} new features")
            
            return updated_symbol
            
        except Exception as e:
            self.logger.error(f"Symbol update failed: {e}")
            raise RuntimeError(f"Failed to update existing symbol: {e}")
    
    def _perform_incremental_update(self,
                                   features: List[VisualFeature],
                                   request: SymbolLearningRequest) -> VisualSymbol:
        """インクリメンタル更新の実行"""
        try:
            # 指定されたIDの記号を取得
            existing_symbol = self.symbol_repository.find_symbol_by_id(request.symbol_id)
            
            if not existing_symbol:
                raise ValueError(f"Symbol with ID {request.symbol_id} not found for incremental update")
            
            return self._update_existing_symbol(existing_symbol, features, request)
            
        except Exception as e:
            self.logger.error(f"Incremental update failed: {e}")
            raise RuntimeError(f"Incremental update error: {e}")
    
    def _create_new_symbol(self,
                          features: List[VisualFeature],
                          request: SymbolLearningRequest) -> VisualSymbol:
        """新規記号の作成"""
        try:
            # プロトタイプ計算方法の選択
            if request.prototype_method == "weighted":
                # 重み付きプロトタイプ計算（信頼度による重み付け）
                weighted_features = self._apply_feature_weights(features)
                new_symbol = VisualSymbol.create_from_features(
                    features=weighted_features,
                    semantic_label=request.semantic_label,
                    symbol_id=request.symbol_id
                )
            else:
                # 標準プロトタイプ計算
                new_symbol = VisualSymbol.create_from_features(
                    features=features,
                    semantic_label=request.semantic_label,
                    symbol_id=request.symbol_id
                )
            
            self.logger.info(f"Created new symbol {new_symbol.symbol_id} with {len(features)} features")
            
            return new_symbol
            
        except Exception as e:
            self.logger.error(f"New symbol creation failed: {e}")
            raise RuntimeError(f"Failed to create new symbol: {e}")
    
    def _apply_feature_weights(self, features: List[VisualFeature]) -> List[VisualFeature]:
        """特徴重み付けの適用"""
        # 信頼度による重み付け（簡易版）
        # 実際の実装では特徴の重要度に基づいた重み付けを行う
        weighted_features = []
        
        for feature in features:
            # 高信頼度の特徴を複数回含める（重み付け効果）
            repeat_count = max(1, int(feature.confidence * 3))  # 最大3回
            weighted_features.extend([feature] * repeat_count)
        
        return weighted_features
    
    def _integrate_symbol_to_recognizer(self, symbol: VisualSymbol):
        """認識器への記号統合"""
        try:
            # 認識器の記号レジストリに追加
            self.visual_symbol_recognizer.symbol_registry[symbol.symbol_id] = symbol
            
            self.logger.debug(f"Integrated symbol {symbol.symbol_id} to recognizer")
            
        except Exception as e:
            self.logger.warning(f"Symbol integration to recognizer failed: {e}")
            # 非致命的エラーとして処理継続
    
    def _perform_post_learning_validation(self,
                                        learned_symbol: VisualSymbol,
                                        training_features: List[VisualFeature]) -> Dict[str, Any]:
        """学習後検証"""
        validation_results = {
            'symbol_quality': 0.0,
            'recognition_accuracy': 0.0,
            'consistency_score': 0.0,
            'validation_passed': False
        }
        
        try:
            # 記号品質の評価
            symbol_stats = learned_symbol.get_symbol_statistics()
            validation_results['symbol_quality'] = symbol_stats.get('confidence', 0.0)
            
            # 認識精度の評価
            recognition_scores = []
            for feature in training_features:
                try:
                    match_confidence = learned_symbol.calculate_match_confidence(feature)
                    recognition_scores.append(match_confidence)
                except Exception:
                    continue
            
            if recognition_scores:
                validation_results['recognition_accuracy'] = sum(recognition_scores) / len(recognition_scores)
            
            # 一貫性スコアの計算
            if len(training_features) > 1:
                similarities = []
                for i in range(len(training_features)):
                    for j in range(i + 1, len(training_features)):
                        sim = training_features[i].calculate_similarity(training_features[j])
                        similarities.append(sim)
                
                if similarities:
                    validation_results['consistency_score'] = sum(similarities) / len(similarities)
            
            # 総合判定
            overall_score = (
                0.4 * validation_results['symbol_quality'] +
                0.4 * validation_results['recognition_accuracy'] +
                0.2 * validation_results['consistency_score']
            )
            
            validation_results['validation_passed'] = overall_score >= 0.6
            
        except Exception as e:
            self.logger.warning(f"Post-learning validation failed: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def _update_learning_stats(self, success: bool, processing_time: float):
        """学習統計の更新"""
        if success:
            self.learning_stats['successful_trainings'] += 1
        
        self.learning_stats['total_training_time'] += processing_time
        total_sessions = max(self.learning_stats['total_training_sessions'], 1)
        self.learning_stats['avg_training_time'] = (
            self.learning_stats['total_training_time'] / total_sessions
        )
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """学習統計の取得"""
        total_sessions = max(self.learning_stats['total_training_sessions'], 1)
        
        return {
            'total_training_sessions': self.learning_stats['total_training_sessions'],
            'symbols_created': self.learning_stats['symbols_created'],
            'symbols_merged': self.learning_stats['symbols_merged'],
            'symbols_updated': self.learning_stats['symbols_updated'],
            'successful_trainings': self.learning_stats['successful_trainings'],
            'validation_failures': self.learning_stats['validation_failures'],
            'success_rate': self.learning_stats['successful_trainings'] / total_sessions,
            'avg_training_time': self.learning_stats['avg_training_time'],
            'total_training_time': self.learning_stats['total_training_time'],
            'merge_rate': self.learning_stats['symbols_merged'] / total_sessions,
            'creation_rate': self.learning_stats['symbols_created'] / total_sessions,
            'validation_enabled': self.enable_validation,
            'merge_detection_enabled': self.enable_merge_detection
        }
    
    def batch_train_symbols(self, 
                           requests: List[SymbolLearningRequest]) -> List[SymbolLearningResponse]:
        """バッチ記号学習"""
        responses = []
        
        self.logger.info(f"Starting batch training with {len(requests)} requests")
        
        for i, request in enumerate(requests):
            try:
                response = self.execute(request)
                responses.append(response)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Batch progress: {i + 1}/{len(requests)} completed")
            
            except Exception as e:
                self.logger.error(f"Batch training item {i} failed: {e}")
                error_response = SymbolLearningResponse.failure_response(
                    error_message=f"Batch item {i} failed: {e}",
                    session_id=request.session_id if hasattr(request, 'session_id') else None
                )
                responses.append(error_response)
        
        successful_count = sum(1 for r in responses if r.success)
        self.logger.info(f"Batch training completed: {successful_count}/{len(requests)} successful")
        
        return responses
    
    def configure_learning_parameters(self,
                                    enable_validation: Optional[bool] = None,
                                    enable_merge_detection: Optional[bool] = None):
        """学習パラメータの設定"""
        if enable_validation is not None:
            self.enable_validation = enable_validation
        
        if enable_merge_detection is not None:
            self.enable_merge_detection = enable_merge_detection
        
        self.logger.info(f"Learning parameters updated: "
                        f"validation={self.enable_validation}, "
                        f"merge_detection={self.enable_merge_detection}")