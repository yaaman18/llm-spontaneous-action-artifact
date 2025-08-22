"""
画像認識DTOモジュール

視覚記号認識システムの入出力データ転送オブジェクト。
Clean Architecture原則に従い、外部レイヤーとドメインロジックを分離。
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import numpy as np

from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.visual_symbol import VisualSymbol
from domain.value_objects.recognition_result import RecognitionResult, RecognitionStatus


@dataclass(frozen=True)
class ImageRecognitionRequest:
    """
    画像認識リクエストDTO
    
    外部システムからの画像認識要求を表現。
    様々な入力形式に対応し、処理オプションを提供。
    """
    
    # 入力データ（いずれか必須）
    image_path: Optional[str] = None
    """画像ファイルパス"""
    
    image_data: Optional[bytes] = None
    """画像バイナリデータ"""
    
    image_array: Optional[np.ndarray] = None
    """画像配列データ"""
    
    visual_features: Optional[VisualFeature] = None
    """事前抽出された視覚特徴"""
    
    # 処理オプション
    recognition_threshold: Optional[float] = None
    """認識閾値の上書き（None の場合デフォルト値使用）"""
    
    enable_learning: bool = True
    """継続学習の有効化"""
    
    return_alternatives: bool = True
    """代替候補の返却有効化"""
    
    max_alternatives: int = 5
    """最大代替候補数"""
    
    include_debug_info: bool = False
    """デバッグ情報の含有フラグ"""
    
    spatial_context: Optional[Dict[str, Any]] = None
    """空間コンテキスト情報"""
    
    session_id: Optional[str] = None
    """セッション識別子（ログ・追跡用）"""
    
    request_timestamp: datetime = None
    """リクエスト時刻"""
    
    def __post_init__(self):
        """リクエストの妥当性チェック"""
        # 時刻の自動設定
        if self.request_timestamp is None:
            object.__setattr__(self, 'request_timestamp', datetime.now())
        
        # 入力データの存在チェック
        input_sources = [
            self.image_path,
            self.image_data,
            self.image_array is not None,
            self.visual_features
        ]
        
        if not any(input_sources):
            raise ValueError("At least one input source must be provided")
        
        # 閾値の範囲チェック
        if self.recognition_threshold is not None:
            if not (0.0 <= self.recognition_threshold <= 1.0):
                raise ValueError("Recognition threshold must be between 0.0 and 1.0")
        
        # 代替候補数の妥当性チェック
        if self.max_alternatives < 0:
            raise ValueError("Max alternatives must be non-negative")
    
    def get_primary_input_type(self) -> str:
        """主要な入力タイプを取得"""
        if self.visual_features:
            return "visual_features"
        elif self.image_array is not None:
            return "image_array"
        elif self.image_data:
            return "image_data"
        elif self.image_path:
            return "image_path"
        return "unknown"
    
    def has_spatial_context(self) -> bool:
        """空間コンテキスト情報の有無を確認"""
        return self.spatial_context is not None and bool(self.spatial_context)


@dataclass(frozen=True) 
class ImageRecognitionResponse:
    """
    画像認識レスポンスDTO
    
    視覚記号認識の結果を外部システムに返却するデータ構造。
    成功・失敗の両ケースに対応し、詳細な結果情報を提供。
    """
    
    # 認識結果の基本情報
    success: bool
    """認識成功フラグ"""
    
    recognition_status: RecognitionStatus
    """詳細な認識ステータス"""
    
    confidence: float
    """認識信頼度（0.0-1.0）"""
    
    processing_time: float
    """処理時間（秒）"""
    
    # 認識された記号情報
    recognized_symbol_id: Optional[str] = None
    """認識された記号のID"""
    
    recognized_label: Optional[str] = None
    """認識されたラベル（意味名）"""
    
    symbol_confidence: Optional[float] = None
    """記号自体の信頼度"""
    
    # 代替候補情報
    alternative_matches: List[Dict[str, Any]] = None
    """代替候補リスト"""
    
    # 詳細情報
    feature_analysis: Optional[Dict[str, float]] = None
    """特徴別分析結果"""
    
    spatial_location: Optional[tuple] = None
    """空間位置情報"""
    
    # メタデータ
    message: Optional[str] = None
    """ユーザー向けメッセージ"""
    
    error_details: Optional[str] = None
    """エラー詳細（失敗時）"""
    
    debug_info: Optional[Dict[str, Any]] = None
    """デバッグ情報"""
    
    session_id: Optional[str] = None
    """セッション識別子"""
    
    response_timestamp: datetime = None
    """レスポンス時刻"""
    
    # 学習関連情報
    learning_feedback: Optional[Dict[str, Any]] = None
    """学習フィードバック"""
    
    symbol_updates: List[str] = None
    """更新された記号IDリスト"""
    
    def __post_init__(self):
        """レスポンスの初期化処理"""
        # 時刻の自動設定
        if self.response_timestamp is None:
            object.__setattr__(self, 'response_timestamp', datetime.now())
        
        # リストの初期化
        if self.alternative_matches is None:
            object.__setattr__(self, 'alternative_matches', [])
        
        if self.symbol_updates is None:
            object.__setattr__(self, 'symbol_updates', [])
        
        # 妥当性チェック
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if self.processing_time < 0.0:
            raise ValueError("Processing time must be non-negative")
    
    @classmethod
    def from_recognition_result(
        cls,
        result: RecognitionResult,
        session_id: Optional[str] = None,
        include_debug_info: bool = False,
        max_alternatives: int = 5
    ) -> 'ImageRecognitionResponse':
        """
        RecognitionResultからレスポンスDTOを作成
        
        Args:
            result: ドメイン層の認識結果
            session_id: セッション識別子
            include_debug_info: デバッグ情報の含有フラグ
            max_alternatives: 最大代替候補数
        
        Returns:
            画像認識レスポンス
        """
        # 基本情報の変換
        success = result.is_successful()
        
        # 認識された記号情報の取得
        recognized_symbol_id = None
        recognized_label = None  
        symbol_confidence = None
        
        if result.recognized_symbol:
            recognized_symbol_id = result.recognized_symbol.symbol_id
            recognized_label = result.recognized_symbol.semantic_label
            symbol_confidence = result.recognized_symbol.confidence
        
        # 代替候補の変換
        alternative_matches = []
        for symbol, conf in result.alternative_matches[:max_alternatives]:
            alt_match = {
                'symbol_id': symbol.symbol_id,
                'label': symbol.semantic_label,
                'confidence': conf,
                'symbol_confidence': symbol.confidence,
                'usage_frequency': symbol.usage_frequency
            }
            alternative_matches.append(alt_match)
        
        # 空間位置の取得
        spatial_location = result.input_features.spatial_location
        
        # デバッグ情報の構築
        debug_info = None
        if include_debug_info:
            debug_info = result.get_debug_summary()
        
        # 学習フィードバックの取得
        learning_feedback = result.create_learning_feedback()
        
        return cls(
            success=success,
            recognition_status=result.status,
            confidence=result.confidence,
            processing_time=result.processing_time,
            recognized_symbol_id=recognized_symbol_id,
            recognized_label=recognized_label,
            symbol_confidence=symbol_confidence,
            alternative_matches=alternative_matches,
            feature_analysis=result.feature_matches,
            spatial_location=spatial_location,
            message=result.format_user_message(),
            error_details=result.error_message,
            debug_info=debug_info,
            session_id=session_id,
            learning_feedback=learning_feedback,
            symbol_updates=[]  # Use caseで更新される
        )
    
    def get_best_alternative(self) -> Optional[Dict[str, Any]]:
        """最良の代替候補を取得"""
        if not self.alternative_matches:
            return None
        
        return max(self.alternative_matches, key=lambda x: x['confidence'])
    
    def get_confidence_level(self) -> str:
        """信頼度レベルの分類"""
        if self.confidence >= 0.8:
            return "high"
        elif self.confidence >= 0.6:
            return "medium"
        elif self.confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """サマリー辞書への変換"""
        return {
            'success': self.success,
            'status': self.recognition_status.value,
            'confidence': self.confidence,
            'confidence_level': self.get_confidence_level(),
            'recognized_symbol': self.recognized_symbol_id,
            'recognized_label': self.recognized_label,
            'processing_time': self.processing_time,
            'alternatives_count': len(self.alternative_matches),
            'message': self.message,
            'timestamp': self.response_timestamp.isoformat()
        }