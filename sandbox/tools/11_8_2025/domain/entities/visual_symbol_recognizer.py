"""
視覚記号認識器エンティティ

谷口忠大の記号創発理論に基づく視覚記号認識の中核エンティティ。
プロトタイプベースの記号マッチングと継続学習による
記号進化を管理する。
"""

import time
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import uuid
import logging
import numpy as np
from collections import defaultdict

from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.visual_symbol import VisualSymbol
from domain.value_objects.recognition_result import RecognitionResult, RecognitionStatus


class VisualSymbolRecognizer:
    """
    視覚記号認識エンティティ
    
    Clean Architecture原則:
    - ドメインロジックの中核エンティティ
    - 外部依存関係を持たない純粋なビジネスロジック
    - 状態変更の責任を持つ
    
    記号創発理論の実装:
    - プロトタイプベースの記号表現
    - 類似度計算による記号マッチング
    - 継続学習による記号進化
    - 使用頻度による記号重要度管理
    """
    
    def __init__(self,
                 recognition_threshold: float = 0.55,  # 現実的な認識性能に最適化
                 ambiguity_threshold: float = 0.1,
                 max_symbols: int = 1000,
                 learning_enabled: bool = True):
        """
        視覚記号認識器の初期化
        
        Args:
            recognition_threshold: 認識成功の最小信頼度（0.55に最適化）
            ambiguity_threshold: 曖昧判定の信頼度差分閾値
            max_symbols: 最大記号数
            learning_enabled: 継続学習の有効化
        """
        self.recognition_threshold = recognition_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.max_symbols = max_symbols
        self.learning_enabled = learning_enabled
        
        # 記号レジストリ
        self.symbol_registry: Dict[str, VisualSymbol] = {}
        
        # 認識統計
        self.recognition_stats = {
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'unknown_objects': 0,
            'low_confidence_cases': 0,
            'ambiguous_cases': 0,
            'processing_errors': 0
        }
        
        # 記号使用履歴（性能最適化用）
        self.symbol_usage_history: Dict[str, List[datetime]] = defaultdict(list)
        self.symbol_last_access: Dict[str, datetime] = {}
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"VisualSymbolRecognizer initialized with threshold: {recognition_threshold} (optimized for realistic performance)")
    
    def recognize_image(self, image_features: VisualFeature) -> RecognitionResult:
        """
        画像特徴から視覚記号を認識
        
        記号創発理論に基づく認識プロセス：
        1. 候補記号の検索
        2. 類似度計算とランキング
        3. 閾値判定と結果決定
        4. 統計更新
        
        Args:
            image_features: 入力画像の視覚特徴
            
        Returns:
            認識結果オブジェクト
        """
        start_time = time.time()
        
        try:
            # 入力特徴の妥当性チェック
            if not self._validate_input_features(image_features):
                return RecognitionResult.processing_error(
                    input_features=image_features,
                    error_message="Invalid input features",
                    processing_time=time.time() - start_time
                )
            
            # 記号が登録されていない場合
            if not self.symbol_registry:
                result = RecognitionResult.unknown(
                    input_features=image_features,
                    processing_time=time.time() - start_time,
                    message="No symbols registered in the system"
                )
                self._update_recognition_stats(result)
                return result
            
            # 候補記号の検索とスコア計算
            candidate_matches = self._find_candidate_matches(image_features)
            
            # 結果の決定
            result = self._determine_recognition_result(
                image_features,
                candidate_matches,
                time.time() - start_time
            )
            
            # 統計と履歴の更新
            self._update_recognition_stats(result)
            if result.recognized_symbol:
                self._update_symbol_usage(result.recognized_symbol.symbol_id)
            
            # 継続学習の実行
            if self.learning_enabled:
                self._apply_continuous_learning(image_features, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Recognition error: {e}")
            return RecognitionResult.processing_error(
                input_features=image_features,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _validate_input_features(self, features: VisualFeature) -> bool:
        """入力特徴の妥当性チェック"""
        try:
            # 基本的な整合性チェック
            if not (0.0 <= features.confidence <= 1.0):
                return False
            
            # 必要な特徴の存在チェック
            if not features.edge_features or not features.color_features:
                return False
            
            # 数値の妥当性チェック
            unified_vector = features.get_unified_feature_vector()
            if unified_vector.size == 0 or np.any(np.isnan(unified_vector)):
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Feature validation error: {e}")
            return False
    
    def _find_candidate_matches(self, image_features: VisualFeature) -> List[Tuple[VisualSymbol, float]]:
        """
        候補記号の検索とスコア計算
        
        全登録記号に対して類似度を計算し、
        有望な候補をランキング順で返す。
        """
        candidate_matches = []
        
        for symbol_id, symbol in self.symbol_registry.items():
            try:
                # 記号とのマッチング信頼度計算
                match_confidence = symbol.calculate_match_confidence(image_features)
                
                # 候補として追加
                candidate_matches.append((symbol, match_confidence))
                
            except Exception as e:
                self.logger.warning(f"Error calculating match for symbol {symbol_id}: {e}")
                continue
        
        # 信頼度の降順でソート
        candidate_matches.sort(key=lambda x: x[1], reverse=True)
        
        return candidate_matches
    
    def _determine_recognition_result(self,
                                    image_features: VisualFeature,
                                    candidate_matches: List[Tuple[VisualSymbol, float]],
                                    processing_time: float) -> RecognitionResult:
        """
        候補マッチから最終的な認識結果を決定
        
        記号創発理論に基づく決定ルール：
        - 閾値以上の単一候補 → 成功認識
        - 閾値以上の複数候補（類似信頼度） → 曖昧判定
        - 閾値未満の最良候補 → 低信頼度判定
        - 候補なし → 未知物体判定
        """
        if not candidate_matches:
            return RecognitionResult.unknown(
                input_features=image_features,
                processing_time=processing_time,
                message="No candidate symbols found"
            )
        
        best_match, best_confidence = candidate_matches[0]
        
        # 特徴別マッチ度の計算
        feature_matches = self._calculate_feature_matches(image_features, best_match)
        
        # 閾値以上の候補を取得
        above_threshold = [(symbol, conf) for symbol, conf in candidate_matches 
                          if conf >= self.recognition_threshold]
        
        if not above_threshold:
            # 全候補が閾値未満 → 低信頼度判定
            return RecognitionResult.low_confidence(
                input_features=image_features,
                best_match=best_match,
                confidence=best_confidence,
                threshold=self.recognition_threshold,
                alternative_matches=candidate_matches[:5],  # 上位5候補
                processing_time=processing_time
            )
        
        elif len(above_threshold) == 1:
            # 単一の高信頼度候補 → 成功認識
            return RecognitionResult.success(
                input_features=image_features,
                recognized_symbol=best_match,
                confidence=best_confidence,
                alternative_matches=candidate_matches[1:6],  # 2-6位の候補
                processing_time=processing_time,
                feature_matches=feature_matches
            )
        
        else:
            # 複数の高信頼度候補 → 曖昧判定の検討
            second_best_confidence = above_threshold[1][1]
            confidence_gap = best_confidence - second_best_confidence
            
            if confidence_gap < self.ambiguity_threshold:
                # 信頼度が拮抗 → 曖昧判定
                return RecognitionResult.ambiguous(
                    input_features=image_features,
                    competing_matches=above_threshold,
                    processing_time=processing_time
                )
            else:
                # 明確な最良候補 → 成功認識
                return RecognitionResult.success(
                    input_features=image_features,
                    recognized_symbol=best_match,
                    confidence=best_confidence,
                    alternative_matches=candidate_matches[1:6],
                    processing_time=processing_time,
                    feature_matches=feature_matches
                )
    
    def _calculate_feature_matches(self, image_features: VisualFeature, symbol: VisualSymbol) -> Dict[str, float]:
        """特徴別マッチ度の詳細計算"""
        feature_matches = {}
        
        try:
            # エッジ特徴マッチ度
            edge_similarity = self._calculate_edge_similarity(
                image_features.edge_features,
                symbol.prototype_features.edge_features
            )
            feature_matches['edge_similarity'] = edge_similarity
            
            # 色特徴マッチ度
            color_similarity = self._calculate_color_similarity(
                image_features.color_features,
                symbol.prototype_features.color_features
            )
            feature_matches['color_similarity'] = color_similarity
            
            # 形状特徴マッチ度
            shape_similarity = self._calculate_shape_similarity(
                image_features.shape_features,
                symbol.prototype_features.shape_features
            )
            feature_matches['shape_similarity'] = shape_similarity
            
            # 統合類似度
            overall_similarity = image_features.calculate_similarity(symbol.prototype_features)
            feature_matches['overall_similarity'] = overall_similarity
            
        except Exception as e:
            self.logger.warning(f"Error calculating feature matches: {e}")
            feature_matches = {'error': 0.0}
        
        return feature_matches
    
    def _calculate_edge_similarity(self, features1: Dict, features2: Dict) -> float:
        """エッジ特徴間の類似度計算"""
        if not features1 or not features2:
            return 0.0
        
        similarities = []
        
        # エッジ密度の比較
        if 'edge_density' in features1 and 'edge_density' in features2:
            # 型安全性：numpy配列とスカラー値を統一的に処理
            d1 = features1['edge_density']
            density1 = float(d1.flat[0]) if isinstance(d1, np.ndarray) else float(d1)
            
            d2 = features2['edge_density']
            density2 = float(d2.flat[0]) if isinstance(d2, np.ndarray) else float(d2)
            
            density_sim = 1.0 - abs(density1 - density2) / max(density1 + density2, 1e-6)
            similarities.append(density_sim)
        
        # エッジヒストグラムの比較
        if 'edge_histogram' in features1 and 'edge_histogram' in features2:
            hist1 = np.asarray(features1['edge_histogram']).flatten()
            hist2 = np.asarray(features2['edge_histogram']).flatten()
            
            if hist1.size > 0 and hist2.size > 0:
                # サイズを統一（最小サイズに合わせる）
                min_size = min(hist1.size, hist2.size)
                hist1_norm = hist1[:min_size]
                hist2_norm = hist2[:min_size]
                
                # コサイン類似度
                norm1, norm2 = np.linalg.norm(hist1_norm), np.linalg.norm(hist2_norm)
                if norm1 > 0 and norm2 > 0:
                    cosine_sim = np.dot(hist1_norm, hist2_norm) / (norm1 * norm2)
                    similarities.append(max(0.0, cosine_sim))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_color_similarity(self, features1: Dict, features2: Dict) -> float:
        """色特徴間の類似度計算"""
        if not features1 or not features2:
            return 0.0
        
        similarities = []
        
        # 色ヒストグラムの比較
        if 'color_histogram' in features1 and 'color_histogram' in features2:
            hist1 = np.asarray(features1['color_histogram']).flatten()
            hist2 = np.asarray(features2['color_histogram']).flatten()
            
            if hist1.size > 0 and hist2.size > 0:
                # サイズを統一（最小サイズに合わせる）
                min_size = min(hist1.size, hist2.size)
                hist1_norm = hist1[:min_size]
                hist2_norm = hist2[:min_size]
                
                # ヒストグラム交差（intersection）
                intersection = np.sum(np.minimum(hist1_norm, hist2_norm))
                union = np.sum(np.maximum(hist1_norm, hist2_norm))
                if union > 0:
                    similarities.append(intersection / union)
        
        # 主要色の比較
        if 'dominant_colors' in features1 and 'dominant_colors' in features2:
            colors1 = np.asarray(features1['dominant_colors'])
            colors2 = np.asarray(features2['dominant_colors'])
            
            # 二次元配列に再形成（色データの場合）
            if colors1.ndim == 1:
                colors1 = colors1.reshape(-1, 3) if colors1.size >= 3 else colors1.reshape(-1, 1)
            if colors2.ndim == 1:
                colors2 = colors2.reshape(-1, 3) if colors2.size >= 3 else colors2.reshape(-1, 1)
            
            if colors1.size > 0 and colors2.size > 0 and colors1.shape[1] == colors2.shape[1]:
                # 最近傍主要色間の平均距離
                color_distances = []
                for c1 in colors1:
                    if colors2.shape[0] > 0:
                        min_dist = min(np.linalg.norm(c1 - c2) for c2 in colors2)
                        color_distances.append(min_dist)
                
                if color_distances:
                    avg_distance = np.mean(color_distances)
                    # 距離を類似度に変換（正規化）
                    color_sim = max(0.0, 1.0 - avg_distance / 255.0)
                    similarities.append(color_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_shape_similarity(self, features1: Dict, features2: Dict) -> float:
        """形状特徴間の類似度計算"""
        if not features1 or not features2:
            return 0.0
        
        similarities = []
        
        shape_keys = ['aspect_ratio', 'solidity', 'extent', 'circularity', 'rectangularity', 'compactness']
        
        for key in shape_keys:
            if key in features1 and key in features2:
                val1, val2 = features1[key], features2[key]
                
                # 正規化された差分から類似度計算
                max_val = max(abs(val1), abs(val2), 1e-6)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(0.0, similarity))
        
        return np.mean(similarities) if similarities else 0.0
    
    def learn_new_symbol(self, 
                        features: List[VisualFeature], 
                        semantic_label: Optional[str] = None,
                        symbol_id: Optional[str] = None) -> VisualSymbol:
        """
        新しい視覚記号の学習
        
        記号創発理論に基づく記号形成プロセス：
        1. 特徴群からプロトタイプ計算
        2. 変動範囲の学習
        3. 記号レジストリへの追加
        4. 記号数上限の管理
        
        Args:
            features: 記号形成に使用する特徴群
            semantic_label: 意味ラベル（オプション）
            symbol_id: 記号ID（指定されない場合は自動生成）
            
        Returns:
            新しく学習された視覚記号
            
        Raises:
            ValueError: 特徴群が無効な場合
            RuntimeError: 記号作成エラー
        """
        try:
            # 入力特徴の妥当性チェック
            if not features:
                raise ValueError("Cannot learn symbol from empty feature list")
            
            # 特徴品質のチェック
            valid_features = [f for f in features if self._validate_input_features(f)]
            if not valid_features:
                raise ValueError("No valid features provided for symbol learning")
            
            # 記号の作成
            new_symbol = VisualSymbol.create_from_features(
                features=valid_features,
                semantic_label=semantic_label,
                symbol_id=symbol_id
            )
            
            # 記号数上限の管理
            if len(self.symbol_registry) >= self.max_symbols:
                self._manage_symbol_capacity()
            
            # 記号レジストリへの追加
            self.symbol_registry[new_symbol.symbol_id] = new_symbol
            
            self.logger.info(f"New symbol learned: {new_symbol.symbol_id} with {len(valid_features)} features")
            
            return new_symbol
            
        except Exception as e:
            self.logger.error(f"Symbol learning error: {e}")
            raise RuntimeError(f"Failed to learn new symbol: {e}")
    
    def _manage_symbol_capacity(self):
        """記号数上限管理（古い/使用頻度の低い記号を削除）"""
        if len(self.symbol_registry) < self.max_symbols:
            return
        
        # 削除候補の選定（使用頻度と最終アクセス時刻を考慮）
        removal_candidates = []
        
        for symbol_id, symbol in self.symbol_registry.items():
            last_access = self.symbol_last_access.get(symbol_id, symbol.creation_timestamp)
            days_since_access = (datetime.now() - last_access).days
            
            # 削除候補スコア（低いほど削除されやすい）
            candidate_score = (
                symbol.usage_frequency * 0.4 +  # 使用頻度
                symbol.confidence * 0.3 +       # 記号信頼度
                max(0, 30 - days_since_access) * 0.3  # 最近性（30日を最大とする）
            )
            
            removal_candidates.append((symbol_id, candidate_score))
        
        # スコアの昇順でソート（削除候補順）
        removal_candidates.sort(key=lambda x: x[1])
        
        # 容量の10%を削除
        removal_count = max(1, len(self.symbol_registry) // 10)
        
        for symbol_id, _ in removal_candidates[:removal_count]:
            del self.symbol_registry[symbol_id]
            if symbol_id in self.symbol_usage_history:
                del self.symbol_usage_history[symbol_id]
            if symbol_id in self.symbol_last_access:
                del self.symbol_last_access[symbol_id]
        
        self.logger.info(f"Removed {removal_count} symbols for capacity management")
    
    def _update_recognition_stats(self, result: RecognitionResult):
        """認識統計の更新"""
        self.recognition_stats['total_recognitions'] += 1
        
        if result.status == RecognitionStatus.SUCCESS:
            self.recognition_stats['successful_recognitions'] += 1
        elif result.status == RecognitionStatus.UNKNOWN:
            self.recognition_stats['unknown_objects'] += 1
        elif result.status == RecognitionStatus.LOW_CONFIDENCE:
            self.recognition_stats['low_confidence_cases'] += 1
        elif result.status == RecognitionStatus.AMBIGUOUS:
            self.recognition_stats['ambiguous_cases'] += 1
        elif result.status == RecognitionStatus.PROCESSING_ERROR:
            self.recognition_stats['processing_errors'] += 1
    
    def _update_symbol_usage(self, symbol_id: str):
        """記号使用履歴の更新"""
        now = datetime.now()
        self.symbol_usage_history[symbol_id].append(now)
        self.symbol_last_access[symbol_id] = now
        
        # 履歴の上限管理（直近100回まで保持）
        if len(self.symbol_usage_history[symbol_id]) > 100:
            self.symbol_usage_history[symbol_id] = self.symbol_usage_history[symbol_id][-100:]
    
    def _apply_continuous_learning(self, image_features: VisualFeature, result: RecognitionResult):
        """
        継続学習の適用
        
        認識結果に基づいて記号の更新や新規学習を実行。
        """
        if not self.learning_enabled:
            return
        
        try:
            if result.status == RecognitionStatus.SUCCESS and result.recognized_symbol:
                # 成功認識の場合：記号のインクリメンタル更新
                self._update_symbol_with_instance(result.recognized_symbol, image_features)
            
            elif result.is_learning_opportunity():
                # 学習機会の場合：将来的な自動記号生成（現在はログのみ）
                self.logger.info(f"Learning opportunity detected: {result.status.value}")
                
        except Exception as e:
            self.logger.warning(f"Continuous learning error: {e}")
    
    def _update_symbol_with_instance(self, symbol: VisualSymbol, new_feature: VisualFeature):
        """記号インスタンスでの更新"""
        try:
            updated_symbol = symbol.update_with_new_instance(new_feature)
            self.symbol_registry[symbol.symbol_id] = updated_symbol
            
        except Exception as e:
            self.logger.warning(f"Symbol update error for {symbol.symbol_id}: {e}")
    
    def get_recognition_statistics(self) -> Dict[str, float]:
        """認識統計の取得"""
        total = max(self.recognition_stats['total_recognitions'], 1)
        
        return {
            'total_recognitions': self.recognition_stats['total_recognitions'],
            'success_rate': self.recognition_stats['successful_recognitions'] / total,
            'unknown_rate': self.recognition_stats['unknown_objects'] / total,
            'low_confidence_rate': self.recognition_stats['low_confidence_cases'] / total,
            'ambiguous_rate': self.recognition_stats['ambiguous_cases'] / total,
            'error_rate': self.recognition_stats['processing_errors'] / total,
            'total_symbols': len(self.symbol_registry),
            'average_symbol_confidence': np.mean([s.confidence for s in self.symbol_registry.values()]) if self.symbol_registry else 0.0
        }
    
    def get_symbol_summary(self) -> List[Dict[str, any]]:
        """登録記号のサマリー取得"""
        summaries = []
        
        for symbol in self.symbol_registry.values():
            usage_count = len(self.symbol_usage_history.get(symbol.symbol_id, []))
            last_used = self.symbol_last_access.get(symbol.symbol_id, symbol.creation_timestamp)
            
            summary = {
                'symbol_id': symbol.symbol_id,
                'semantic_label': symbol.semantic_label,
                'confidence': symbol.confidence,
                'usage_frequency': symbol.usage_frequency,
                'recent_usage_count': usage_count,
                'creation_date': symbol.creation_timestamp.date(),
                'last_used': last_used.date(),
                'days_since_creation': (datetime.now() - symbol.creation_timestamp).days,
                'is_stable': symbol.is_stable_symbol()
            }
            summaries.append(summary)
        
        # 使用頻度の降順でソート
        summaries.sort(key=lambda x: x['usage_frequency'], reverse=True)
        
        return summaries
    
    def cleanup_unused_symbols(self, days_threshold: int = 30, min_usage_threshold: int = 2):
        """未使用記号のクリーンアップ"""
        symbols_to_remove = []
        
        for symbol_id, symbol in self.symbol_registry.items():
            last_used = self.symbol_last_access.get(symbol_id, symbol.creation_timestamp)
            days_unused = (datetime.now() - last_used).days
            
            if (days_unused > days_threshold and 
                symbol.usage_frequency < min_usage_threshold):
                symbols_to_remove.append(symbol_id)
        
        # 記号の削除
        for symbol_id in symbols_to_remove:
            del self.symbol_registry[symbol_id]
            if symbol_id in self.symbol_usage_history:
                del self.symbol_usage_history[symbol_id]
            if symbol_id in self.symbol_last_access:
                del self.symbol_last_access[symbol_id]
        
        self.logger.info(f"Cleaned up {len(symbols_to_remove)} unused symbols")
        return len(symbols_to_remove)