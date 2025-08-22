"""
画像認識ユースケース

Clean Architecture原則に従った画像認識の中核ユースケース。
視覚記号認識システムの主要ワークフローを統合制御。
"""

import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

from application.dtos.image_recognition_dto import ImageRecognitionRequest, ImageRecognitionResponse
from application.services.visual_feature_extraction_service import VisualFeatureExtractionService
from application.services.symbol_emergence_orchestration_service import SymbolEmergenceOrchestrationService
from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.recognition_result import RecognitionResult


class RecognizeImageUseCase:
    """
    画像認識ユースケース
    
    Clean Architecture原則:
    - アプリケーション層のユースケース
    - ビジネスワークフローの統合制御
    - 外部依存関係からの独立性
    
    谷口忠大の記号創発理論の実装:
    - 統合的視覚認識プロセス
    - 適応学習との統合
    - 文脈依存認識
    """
    
    def __init__(self,
                 feature_extraction_service: VisualFeatureExtractionService,
                 emergence_orchestration_service: SymbolEmergenceOrchestrationService,
                 enable_performance_monitoring: bool = True,
                 enable_detailed_logging: bool = False):
        """
        画像認識ユースケースの初期化
        
        Args:
            feature_extraction_service: 特徴抽出サービス
            emergence_orchestration_service: 記号創発統括サービス
            enable_performance_monitoring: パフォーマンス監視の有効化
            enable_detailed_logging: 詳細ログの有効化
        """
        self.feature_extraction_service = feature_extraction_service
        self.emergence_orchestration_service = emergence_orchestration_service
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_detailed_logging = enable_detailed_logging
        
        # 実行統計
        self.execution_stats = {
            'total_requests': 0,
            'successful_recognitions': 0,
            'failed_extractions': 0,
            'learning_triggered': 0,
            'avg_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        log_level = logging.DEBUG if enable_detailed_logging else logging.INFO
        self.logger.setLevel(log_level)
        self.logger.info("RecognizeImageUseCase initialized")
    
    def execute(self, request: ImageRecognitionRequest) -> ImageRecognitionResponse:
        """
        画像認識ユースケースの実行
        
        Args:
            request: 画像認識リクエスト
            
        Returns:
            画像認識レスポンス
        """
        start_time = time.time()
        self.execution_stats['total_requests'] += 1
        
        if self.enable_detailed_logging:
            self.logger.debug(f"Processing recognition request: {request.get_primary_input_type()}")
        
        try:
            # 1. 入力検証
            self._validate_request(request)
            
            # 2. 視覚特徴の抽出
            visual_feature = self._extract_visual_features(request)
            
            # 3. 学習コンテキストの構築
            learning_context = self._build_learning_context(request)
            
            # 4. 認識と学習の統括実行
            recognition_result, learning_actions = self.emergence_orchestration_service.orchestrate_recognition_and_learning(
                visual_feature, learning_context
            )
            
            # 5. レスポンスの構築
            response = self._build_response(
                recognition_result, 
                learning_actions,
                request,
                time.time() - start_time
            )
            
            # 6. 統計の更新
            self._update_execution_stats(response, time.time() - start_time)
            
            if self.enable_detailed_logging:
                self.logger.debug(f"Recognition completed: {response.recognition_status.value}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Recognition use case execution failed: {e}")
            return self._build_error_response(request, str(e), time.time() - start_time)
    
    def _validate_request(self, request: ImageRecognitionRequest):
        """リクエストの妥当性検証"""
        # 基本的な妥当性チェックは ImageRecognitionRequest の __post_init__ で実行済み
        
        # 追加の業務ルール検証
        if request.recognition_threshold is not None:
            if not (0.1 <= request.recognition_threshold <= 0.95):
                raise ValueError("Recognition threshold should be between 0.1 and 0.95 for practical use")
        
        if request.max_alternatives > 20:
            raise ValueError("Max alternatives should not exceed 20 for performance reasons")
        
        # 画像サイズの制限チェック（image_arrayが提供されている場合）
        if request.image_array is not None:
            height, width = request.image_array.shape[:2]
            if height * width > 10000000:  # 10MP制限
                raise ValueError(f"Image size too large: {width}x{height} pixels")
    
    def _extract_visual_features(self, request: ImageRecognitionRequest) -> VisualFeature:
        """視覚特徴の抽出"""
        try:
            # 既に特徴が抽出されている場合はそれを使用
            if request.visual_features:
                if self.enable_detailed_logging:
                    self.logger.debug("Using pre-extracted visual features")
                return request.visual_features
            
            # 抽出コンテキストの構築
            extraction_context = {}
            
            if request.spatial_context:
                extraction_context.update(request.spatial_context)
            
            # 入力タイプに応じた特徴抽出
            if request.image_array is not None:
                spatial_location = self._determine_spatial_location(request)
                visual_feature = self.feature_extraction_service.extract_from_image_array(
                    request.image_array,
                    spatial_location,
                    extraction_context
                )
                
            elif request.image_path:
                spatial_location = self._determine_spatial_location(request)
                visual_feature = self.feature_extraction_service.extract_from_image_path(
                    request.image_path,
                    spatial_location,
                    extraction_context
                )
                
            elif request.image_data:
                # バイナリデータからの抽出（OpenCVでデコード）
                import cv2
                import numpy as np
                
                nparr = np.frombuffer(request.image_data, np.uint8)
                image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image_array is None:
                    raise ValueError("Cannot decode image data")
                
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                
                spatial_location = self._determine_spatial_location(request)
                visual_feature = self.feature_extraction_service.extract_from_image_array(
                    image_array,
                    spatial_location,
                    extraction_context
                )
            else:
                raise ValueError("No valid input source provided")
            
            if self.enable_detailed_logging:
                complexity = visual_feature.get_feature_complexity()
                self.logger.debug(f"Extracted visual features with complexity: {complexity:.3f}")
            
            return visual_feature
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            self.execution_stats['failed_extractions'] += 1
            raise RuntimeError(f"Feature extraction error: {e}")
    
    def _determine_spatial_location(self, request: ImageRecognitionRequest) -> Optional[tuple]:
        """空間位置の決定"""
        # 空間コンテキストから位置情報を取得
        if request.spatial_context and 'target_location' in request.spatial_context:
            return tuple(request.spatial_context['target_location'])
        
        # デフォルト位置（画像中央）は特徴抽出サービスで決定
        return None
    
    def _build_learning_context(self, request: ImageRecognitionRequest) -> Dict[str, Any]:
        """学習コンテキストの構築"""
        context = {}
        
        # リクエストからコンテキスト情報を収集
        if request.spatial_context:
            context['spatial_context'] = request.spatial_context
        
        # 学習パラメータの設定
        context['learning_enabled'] = request.enable_learning
        context['request_timestamp'] = request.request_timestamp
        context['session_id'] = request.session_id
        
        # 認識閾値の上書き情報
        if request.recognition_threshold:
            context['threshold_override'] = request.recognition_threshold
        
        return context
    
    def _build_response(self,
                       recognition_result: RecognitionResult,
                       learning_actions: List[str],
                       request: ImageRecognitionRequest,
                       processing_time: float) -> ImageRecognitionResponse:
        """レスポンスの構築"""
        # 基本レスポンスの構築
        response = ImageRecognitionResponse.from_recognition_result(
            recognition_result,
            session_id=request.session_id,
            include_debug_info=request.include_debug_info,
            max_alternatives=request.max_alternatives
        )
        
        # 学習アクションの追加
        symbol_updates = []
        for action in learning_actions:
            if "symbol_" in action and ("created" in action or "improved" in action or "merged" in action):
                # アクションから記号IDを抽出
                parts = action.split("_")
                if len(parts) >= 2:
                    symbol_updates.append(parts[-1])  # 最後の部分が記号ID
        
        # レスポンスの更新（frozen dataclassのため新しいインスタンス作成）
        updated_response = ImageRecognitionResponse(
            success=response.success,
            recognition_status=response.recognition_status,
            confidence=response.confidence,
            processing_time=processing_time,  # 全体の処理時間で上書き
            recognized_symbol_id=response.recognized_symbol_id,
            recognized_label=response.recognized_label,
            symbol_confidence=response.symbol_confidence,
            alternative_matches=response.alternative_matches,
            feature_analysis=response.feature_analysis,
            spatial_location=response.spatial_location,
            message=response.message,
            error_details=response.error_details,
            debug_info=response.debug_info,
            session_id=response.session_id,
            response_timestamp=response.response_timestamp,
            learning_feedback=response.learning_feedback,
            symbol_updates=symbol_updates
        )
        
        return updated_response
    
    def _build_error_response(self,
                             request: ImageRecognitionRequest,
                             error_message: str,
                             processing_time: float) -> ImageRecognitionResponse:
        """エラーレスポンスの構築"""
        return ImageRecognitionResponse(
            success=False,
            recognition_status=RecognitionStatus.PROCESSING_ERROR,
            confidence=0.0,
            processing_time=processing_time,
            error_details=error_message,
            session_id=request.session_id,
            message=f"Recognition failed: {error_message}"
        )
    
    def _update_execution_stats(self, response: ImageRecognitionResponse, processing_time: float):
        """実行統計の更新"""
        if response.success:
            self.execution_stats['successful_recognitions'] += 1
        
        if response.symbol_updates:
            self.execution_stats['learning_triggered'] += 1
        
        # 処理時間統計の更新
        self.execution_stats['total_processing_time'] += processing_time
        total_requests = max(self.execution_stats['total_requests'], 1)
        self.execution_stats['avg_processing_time'] = (
            self.execution_stats['total_processing_time'] / total_requests
        )
        
        # パフォーマンス監視
        if self.enable_performance_monitoring:
            self._monitor_performance(processing_time, response)
    
    def _monitor_performance(self, processing_time: float, response: ImageRecognitionResponse):
        """パフォーマンス監視"""
        # 処理時間の監視
        if processing_time > 5.0:  # 5秒以上の場合は警告
            self.logger.warning(f"Slow recognition processing: {processing_time:.3f}s")
        
        # 成功率の監視
        total_requests = self.execution_stats['total_requests']
        if total_requests >= 100:  # 100回以上の実行後に監視開始
            success_rate = self.execution_stats['successful_recognitions'] / total_requests
            if success_rate < 0.8:  # 成功率80%未満で警告
                self.logger.warning(f"Low recognition success rate: {success_rate:.1%}")
        
        # メモリ使用量の監視（簡易版）
        if hasattr(response, 'debug_info') and response.debug_info:
            feature_complexity = response.debug_info.get('input_complexity', 0.0)
            if feature_complexity > 0.9:
                self.logger.info(f"Processing high complexity features: {feature_complexity:.3f}")
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """実行統計の取得"""
        total_requests = max(self.execution_stats['total_requests'], 1)
        
        return {
            'total_requests': self.execution_stats['total_requests'],
            'successful_recognitions': self.execution_stats['successful_recognitions'],
            'failed_extractions': self.execution_stats['failed_extractions'],
            'learning_triggered': self.execution_stats['learning_triggered'],
            'success_rate': self.execution_stats['successful_recognitions'] / total_requests,
            'extraction_failure_rate': self.execution_stats['failed_extractions'] / total_requests,
            'learning_trigger_rate': self.execution_stats['learning_triggered'] / total_requests,
            'avg_processing_time': self.execution_stats['avg_processing_time'],
            'total_processing_time': self.execution_stats['total_processing_time'],
            'performance_monitoring_enabled': self.enable_performance_monitoring,
            'detailed_logging_enabled': self.enable_detailed_logging
        }
    
    def reset_statistics(self):
        """統計のリセット"""
        self.execution_stats = {
            'total_requests': 0,
            'successful_recognitions': 0,
            'failed_extractions': 0,
            'learning_triggered': 0,
            'avg_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        self.logger.info("Execution statistics reset")
    
    def configure_performance_monitoring(self, 
                                       enable_monitoring: Optional[bool] = None,
                                       enable_detailed_logging: Optional[bool] = None):
        """パフォーマンス監視の設定"""
        if enable_monitoring is not None:
            self.enable_performance_monitoring = enable_monitoring
        
        if enable_detailed_logging is not None:
            self.enable_detailed_logging = enable_detailed_logging
            log_level = logging.DEBUG if enable_detailed_logging else logging.INFO
            self.logger.setLevel(log_level)
        
        self.logger.info(f"Performance monitoring configured: "
                        f"monitoring={self.enable_performance_monitoring}, "
                        f"detailed_logging={self.enable_detailed_logging}")


# RecognitionStatus の import が不足していたため追加
from domain.value_objects.recognition_result import RecognitionStatus