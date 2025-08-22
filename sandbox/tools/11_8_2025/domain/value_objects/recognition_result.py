"""
認識結果値オブジェクト

視覚記号認識の結果を表現する値オブジェクト。
成功・失敗の両ケースを統一的に扱い、
デバッグと学習改善に必要な詳細情報を提供。
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from enum import Enum

from .visual_feature import VisualFeature
from .visual_symbol import VisualSymbol


class RecognitionStatus(Enum):
    """認識ステータス"""
    SUCCESS = "success"          # 成功認識
    UNKNOWN = "unknown"          # 未知物体
    LOW_CONFIDENCE = "low_confidence"  # 低信頼度
    AMBIGUOUS = "ambiguous"      # 曖昧（複数候補）
    PROCESSING_ERROR = "processing_error"  # 処理エラー


@dataclass(frozen=True)
class RecognitionResult:
    """
    視覚記号認識結果の統合表現
    
    記号創発理論における認識プロセスの完全な結果を表現。
    成功・失敗の詳細情報と、継続学習に必要なメタ情報を保持。
    
    Clean Architecture原則:
    - 不変な値オブジェクト（frozen=True）
    - 認識プロセスの完全な結果表現
    - 外部依存関係なし
    """
    
    input_features: VisualFeature
    """入力された視覚特徴"""
    
    recognized_symbol: Optional[VisualSymbol]
    """認識された記号（成功時のみ）"""
    
    confidence: float
    """認識信頼度（0.0-1.0）"""
    
    status: RecognitionStatus
    """認識ステータス"""
    
    alternative_matches: List[Tuple[VisualSymbol, float]]
    """代替候補リスト（記号, 信頼度）のペア"""
    
    processing_time: float
    """処理時間（秒）"""
    
    feature_matches: Dict[str, float]
    """特徴別マッチ度（デバッグ用）"""
    
    timestamp: datetime
    """認識実行時刻"""
    
    error_message: Optional[str] = None
    """エラーメッセージ（失敗時）"""
    
    detailed_metrics: Optional[Dict[str, float]] = None
    """詳細メトリクス（性能分析用）"""
    
    def __post_init__(self):
        """
        値オブジェクト不変条件の検証
        
        Clean Architecture原則:
        - ドメインルールの自己検証  
        - 不正状態の防止
        """
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.processing_time < 0.0:
            raise ValueError(f"Processing time must be non-negative, got {self.processing_time}")
        
        # ステータスと認識結果の整合性チェック
        if self.status == RecognitionStatus.SUCCESS and self.recognized_symbol is None:
            raise ValueError("SUCCESS status requires recognized_symbol")
        
        if self.status == RecognitionStatus.UNKNOWN and self.recognized_symbol is not None:
            raise ValueError("UNKNOWN status should not have recognized_symbol")
        
        if self.status == RecognitionStatus.PROCESSING_ERROR and not self.error_message:
            raise ValueError("PROCESSING_ERROR status requires error_message")
    
    @classmethod
    def success(
        cls,
        input_features: VisualFeature,
        recognized_symbol: VisualSymbol,
        confidence: float,
        alternative_matches: List[Tuple[VisualSymbol, float]] = None,
        processing_time: float = 0.0,
        feature_matches: Dict[str, float] = None
    ) -> 'RecognitionResult':
        """
        成功認識結果の作成
        
        Args:
            input_features: 入力特徴
            recognized_symbol: 認識された記号
            confidence: 信頼度
            alternative_matches: 代替候補
            processing_time: 処理時間
            feature_matches: 特徴別マッチ度
            
        Returns:
            成功認識結果
        """
        return cls(
            input_features=input_features,
            recognized_symbol=recognized_symbol,
            confidence=confidence,
            status=RecognitionStatus.SUCCESS,
            alternative_matches=alternative_matches or [],
            processing_time=processing_time,
            feature_matches=feature_matches or {},
            timestamp=datetime.now()
        )
    
    @classmethod
    def unknown(
        cls,
        input_features: VisualFeature,
        processing_time: float = 0.0,
        message: str = "No matching symbol found"
    ) -> 'RecognitionResult':
        """
        未知物体認識結果の作成
        
        Args:
            input_features: 入力特徴
            processing_time: 処理時間
            message: メッセージ
            
        Returns:
            未知物体認識結果
        """
        return cls(
            input_features=input_features,
            recognized_symbol=None,
            confidence=0.0,
            status=RecognitionStatus.UNKNOWN,
            alternative_matches=[],
            processing_time=processing_time,
            feature_matches={},
            timestamp=datetime.now(),
            error_message=message
        )
    
    @classmethod
    def low_confidence(
        cls,
        input_features: VisualFeature,
        best_match: VisualSymbol,
        confidence: float,
        threshold: float,
        alternative_matches: List[Tuple[VisualSymbol, float]] = None,
        processing_time: float = 0.0
    ) -> 'RecognitionResult':
        """
        低信頼度認識結果の作成
        
        Args:
            input_features: 入力特徴
            best_match: 最良マッチ（但し閾値未満）
            confidence: 信頼度
            threshold: 認識閾値
            alternative_matches: 代替候補
            processing_time: 処理時間
            
        Returns:
            低信頼度認識結果
        """
        return cls(
            input_features=input_features,
            recognized_symbol=None,  # 閾値未満のため認識失敗
            confidence=confidence,
            status=RecognitionStatus.LOW_CONFIDENCE,
            alternative_matches=alternative_matches or [(best_match, confidence)],
            processing_time=processing_time,
            feature_matches={},
            timestamp=datetime.now(),
            error_message=f"Best match confidence {confidence:.3f} below threshold {threshold:.3f}"
        )
    
    @classmethod
    def ambiguous(
        cls,
        input_features: VisualFeature,
        competing_matches: List[Tuple[VisualSymbol, float]],
        processing_time: float = 0.0
    ) -> 'RecognitionResult':
        """
        曖昧認識結果の作成（複数の高信頼度候補）
        
        Args:
            input_features: 入力特徴
            competing_matches: 競合候補リスト
            processing_time: 処理時間
            
        Returns:
            曖昧認識結果
        """
        # 最も高い信頼度を取得
        max_confidence = max(confidence for _, confidence in competing_matches) if competing_matches else 0.0
        
        return cls(
            input_features=input_features,
            recognized_symbol=None,  # 曖昧のため特定できず
            confidence=max_confidence,
            status=RecognitionStatus.AMBIGUOUS,
            alternative_matches=competing_matches,
            processing_time=processing_time,
            feature_matches={},
            timestamp=datetime.now(),
            error_message=f"Multiple high-confidence matches found ({len(competing_matches)} candidates)"
        )
    
    @classmethod
    def processing_error(
        cls,
        input_features: VisualFeature,
        error_message: str,
        processing_time: float = 0.0
    ) -> 'RecognitionResult':
        """
        処理エラー結果の作成
        
        Args:
            input_features: 入力特徴
            error_message: エラーメッセージ
            processing_time: 処理時間
            
        Returns:
            処理エラー結果
        """
        return cls(
            input_features=input_features,
            recognized_symbol=None,
            confidence=0.0,
            status=RecognitionStatus.PROCESSING_ERROR,
            alternative_matches=[],
            processing_time=processing_time,
            feature_matches={},
            timestamp=datetime.now(),
            error_message=error_message
        )
    
    def is_successful(self) -> bool:
        """認識が成功したかの判定"""
        return self.status == RecognitionStatus.SUCCESS
    
    def is_learning_opportunity(self) -> bool:
        """
        学習機会としての適性判定
        
        未知物体や低信頼度の結果は新しい記号学習の
        機会として活用可能かを判定する。
        
        Returns:
            学習機会としての適性
        """
        learning_statuses = {
            RecognitionStatus.UNKNOWN,
            RecognitionStatus.LOW_CONFIDENCE,
            RecognitionStatus.AMBIGUOUS
        }
        
        return (
            self.status in learning_statuses and
            self.input_features.is_extractable_symbol_candidate() and
            self.processing_time < 10.0  # 異常な処理時間でない
        )
    
    def get_best_alternative(self) -> Optional[Tuple[VisualSymbol, float]]:
        """
        最良の代替候補を取得
        
        Returns:
            最良代替候補（記号, 信頼度）またはNone
        """
        if not self.alternative_matches:
            return None
        
        return max(self.alternative_matches, key=lambda x: x[1])
    
    def get_recognition_quality_score(self) -> float:
        """
        認識品質スコアの計算
        
        認識結果の総合的な品質を0.0-1.0で評価。
        成功度、信頼度、処理効率を統合した指標。
        
        Returns:
            認識品質スコア（0.0-1.0）
        """
        # ステータス別の基本スコア
        status_scores = {
            RecognitionStatus.SUCCESS: 1.0,
            RecognitionStatus.LOW_CONFIDENCE: 0.6,
            RecognitionStatus.AMBIGUOUS: 0.5,
            RecognitionStatus.UNKNOWN: 0.3,
            RecognitionStatus.PROCESSING_ERROR: 0.0
        }
        
        base_score = status_scores[self.status]
        
        # 信頼度による重み付け
        confidence_weight = 0.7 * self.confidence
        
        # 処理効率による重み付け（2秒以下を理想とする）
        efficiency_score = min(2.0 / max(self.processing_time, 0.1), 1.0)
        efficiency_weight = 0.2 * efficiency_score
        
        # 代替候補の豊富さ（情報量）
        alternative_richness = min(len(self.alternative_matches) / 5.0, 1.0)  # 5候補を最大とする
        richness_weight = 0.1 * alternative_richness
        
        quality_score = base_score * (confidence_weight + efficiency_weight + richness_weight)
        
        return min(1.0, max(0.0, quality_score))
    
    def get_debug_summary(self) -> Dict[str, any]:
        """
        デバッグ用サマリー情報の取得
        
        Returns:
            デバッグ情報辞書
        """
        return {
            'status': self.status.value,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'recognized_symbol_id': self.recognized_symbol.symbol_id if self.recognized_symbol else None,
            'alternative_count': len(self.alternative_matches),
            'input_complexity': self.input_features.get_feature_complexity(),
            'input_confidence': self.input_features.confidence,
            'quality_score': self.get_recognition_quality_score(),
            'is_learning_opportunity': self.is_learning_opportunity(),
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat()
        }
    
    def format_user_message(self) -> str:
        """
        ユーザー向けメッセージの生成
        
        Returns:
            ユーザーフレンドリーな認識結果メッセージ
        """
        if self.status == RecognitionStatus.SUCCESS:
            symbol_name = self.recognized_symbol.semantic_label or self.recognized_symbol.symbol_id
            return f"認識成功: {symbol_name} (信頼度: {self.confidence:.1%})"
        
        elif self.status == RecognitionStatus.LOW_CONFIDENCE:
            best_alt = self.get_best_alternative()
            if best_alt:
                symbol_name = best_alt[0].semantic_label or best_alt[0].symbol_id
                return f"低信頼度: {symbol_name}の可能性 (信頼度: {best_alt[1]:.1%})"
            return "認識信頼度が不足しています"
        
        elif self.status == RecognitionStatus.AMBIGUOUS:
            count = len(self.alternative_matches)
            return f"曖昧な結果: {count}個の候補が見つかりました"
        
        elif self.status == RecognitionStatus.UNKNOWN:
            return "未知の物体です。新しい記号として学習可能です。"
        
        elif self.status == RecognitionStatus.PROCESSING_ERROR:
            return f"処理エラー: {self.error_message}"
        
        return "予期しない認識結果です"
    
    def create_learning_feedback(self) -> Optional[Dict[str, any]]:
        """
        学習フィードバック情報の作成
        
        継続学習システムが使用する学習信号を生成。
        
        Returns:
            学習フィードバック情報またはNone
        """
        if not self.is_learning_opportunity():
            return None
        
        feedback = {
            'input_features': self.input_features,
            'learning_type': self.status.value,
            'confidence_gap': 1.0 - self.confidence,  # 改善必要度
            'feature_quality': self.input_features.get_feature_complexity(),
            'timestamp': self.timestamp,
            'suggested_actions': []
        }
        
        # 学習タイプ別の提案
        if self.status == RecognitionStatus.UNKNOWN:
            feedback['suggested_actions'].append('create_new_symbol')
        elif self.status == RecognitionStatus.LOW_CONFIDENCE:
            feedback['suggested_actions'].extend(['improve_existing_symbol', 'collect_more_examples'])
        elif self.status == RecognitionStatus.AMBIGUOUS:
            feedback['suggested_actions'].extend(['refine_discriminative_features', 'increase_training_data'])
        
        return feedback