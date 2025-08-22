"""
視覚記号認識システム基本デモ

Phase 1実装の基本機能を実演するデモスクリプト。
合成画像を使用して記号学習と認識のプロセスを示す。
"""

import sys
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import tempfile

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not available. Using synthetic feature data only.")

from domain.value_objects.visual_feature import VisualFeature
from domain.value_objects.visual_symbol import VisualSymbol
from domain.value_objects.recognition_result import RecognitionStatus
from domain.entities.visual_symbol_recognizer import VisualSymbolRecognizer

if OPENCV_AVAILABLE:
    from infrastructure.image_processing.opencv_feature_extractor import OpenCVFeatureExtractor

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisualSymbolRecognitionDemo:
    """視覚記号認識システムデモクラス"""
    
    def __init__(self):
        """デモの初期化"""
        self.recognizer = VisualSymbolRecognizer(
            recognition_threshold=0.55,  # 現実的な閾値に最適化
            ambiguity_threshold=0.1,
            learning_enabled=True
        )
        
        if OPENCV_AVAILABLE:
            self.feature_extractor = OpenCVFeatureExtractor(
                target_size=(128, 128),
                enable_preprocessing=True
            )
        else:
            self.feature_extractor = None
    
    def run_demo(self):
        """デモの実行"""
        print("\n" + "="*60)
        print("視覚記号認識システム Phase 1 デモ")
        print(f"認識閾値: {self.recognizer.recognition_threshold:.3f} (最適化済み)")
        print("="*60)
        
        try:
            # Phase 1: 合成特徴での基本デモ
            print("\n--- Phase 1: 合成特徴での基本機能デモ ---")
            self._demo_synthetic_features()
            
            # Phase 2: OpenCV特徴抽出デモ（利用可能な場合）
            if OPENCV_AVAILABLE:
                print("\n--- Phase 2: OpenCV特徴抽出デモ ---")
                self._demo_opencv_feature_extraction()
                
                print("\n--- Phase 3: 画像ベース記号学習・認識デモ ---")
                self._demo_image_based_recognition()
            else:
                print("\n--- OpenCVが利用できないため、画像処理デモをスキップ ---")
            
            # Phase 4: 統計情報表示
            print("\n--- Phase 4: システム統計情報 ---")
            self._show_system_statistics()
            
            print("\n" + "="*60)
            print("デモ完了")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Demo execution error: {e}")
            print(f"\nデモ実行エラー: {e}")
    
    def _demo_synthetic_features(self):
        """合成特徴での基本機能デモ"""
        print("1. 合成視覚特徴の生成...")
        
        # 円形物体の特徴群を生成
        circle_features = self._create_circle_features(5)
        print(f"   円形特徴群: {len(circle_features)}個")
        
        # 四角形物体の特徴群を生成
        square_features = self._create_square_features(5)
        print(f"   四角形特徴群: {len(square_features)}個")
        
        print("\n2. 視覚記号の学習...")
        
        # 円記号の学習
        circle_symbol = self.recognizer.learn_new_symbol(
            features=circle_features,
            semantic_label="circle"
        )
        print(f"   円記号学習完了: {circle_symbol.symbol_id} (信頼度: {circle_symbol.confidence:.3f})")
        
        # 四角記号の学習
        square_symbol = self.recognizer.learn_new_symbol(
            features=square_features,
            semantic_label="square"
        )
        print(f"   四角記号学習完了: {square_symbol.symbol_id} (信頼度: {square_symbol.confidence:.3f})")
        
        print("\n3. 記号認識テスト...")
        
        # 円特徴での認識テスト
        test_circle_feature = circle_features[0]
        circle_result = self.recognizer.recognize_image(test_circle_feature)
        self._print_recognition_result("円特徴", circle_result)
        
        # 四角特徴での認識テスト
        test_square_feature = square_features[0]
        square_result = self.recognizer.recognize_image(test_square_feature)
        self._print_recognition_result("四角特徴", square_result)
        
        # 未知特徴での認識テスト
        triangle_feature = self._create_triangle_feature()
        triangle_result = self.recognizer.recognize_image(triangle_feature)
        self._print_recognition_result("三角特徴（未知）", triangle_result)
    
    def _demo_opencv_feature_extraction(self):
        """OpenCV特徴抽出デモ"""
        print("1. 合成画像の生成と特徴抽出...")
        
        # 合成画像の作成
        circle_image = self._create_synthetic_circle_image()
        square_image = self._create_synthetic_square_image()
        
        # 特徴抽出
        circle_features = self.feature_extractor.extract_comprehensive_features(
            circle_image, spatial_location=(64, 64)
        )
        square_features = self.feature_extractor.extract_comprehensive_features(
            square_image, spatial_location=(64, 64)
        )
        
        print(f"   円画像特徴抽出完了 (信頼度: {circle_features.confidence:.3f})")
        print(f"   四角画像特徴抽出完了 (信頼度: {square_features.confidence:.3f})")
        
        # 特徴詳細の表示
        self._show_feature_details("円画像", circle_features)
        self._show_feature_details("四角画像", square_features)
    
    def _demo_image_based_recognition(self):
        """画像ベース記号学習・認識デモ"""
        print("1. 複数画像からの記号学習...")
        
        # 複数の円画像を生成
        circle_images = [self._create_synthetic_circle_image(i) for i in range(3)]
        circle_features_list = []
        
        for i, image in enumerate(circle_images):
            features = self.feature_extractor.extract_comprehensive_features(image)
            circle_features_list.append(features)
            print(f"   円画像{i+1}: 特徴抽出完了")
        
        # 円記号の学習
        image_circle_symbol = self.recognizer.learn_new_symbol(
            features=circle_features_list,
            semantic_label="image_circle"
        )
        print(f"   画像円記号学習完了: {image_circle_symbol.symbol_id}")
        
        print("\n2. 新しい画像での認識テスト...")
        
        # テスト画像での認識
        test_circle_image = self._create_synthetic_circle_image(99)
        test_features = self.feature_extractor.extract_comprehensive_features(test_circle_image)
        
        recognition_result = self.recognizer.recognize_image(test_features)
        self._print_recognition_result("テスト円画像", recognition_result)
        
        # 継続学習のテスト
        if recognition_result.status == RecognitionStatus.SUCCESS:
            print("   継続学習により記号が更新されました")
            updated_symbol = self.recognizer.symbol_registry[recognition_result.recognized_symbol.symbol_id]
            print(f"   更新後使用頻度: {updated_symbol.usage_frequency}")
    
    def _show_system_statistics(self):
        """システム統計情報の表示"""
        stats = self.recognizer.get_recognition_statistics()
        
        print("システム統計:")
        print(f"  総認識回数: {stats['total_recognitions']}")
        print(f"  認識成功率: {stats['success_rate']:.1%}")
        print(f"  未知物体率: {stats['unknown_rate']:.1%}")
        print(f"  低信頼度率: {stats['low_confidence_rate']:.1%}")
        print(f"  曖昧判定率: {stats['ambiguous_rate']:.1%}")
        print(f"  エラー率: {stats['error_rate']:.1%}")
        print(f"  登録記号数: {stats['total_symbols']}")
        print(f"  平均記号信頼度: {stats['average_symbol_confidence']:.3f}")
        
        print("\n登録記号詳細:")
        symbol_summaries = self.recognizer.get_symbol_summary()
        for i, summary in enumerate(symbol_summaries[:5]):  # 上位5記号
            print(f"  {i+1}. {summary['semantic_label']} ({summary['symbol_id'][:8]}...)")
            print(f"     信頼度: {summary['confidence']:.3f}, 使用頻度: {summary['usage_frequency']}")
            print(f"     安定性: {'安定' if summary['is_stable'] else '不安定'}")
    
    def _create_circle_features(self, count: int) -> list:
        """円形特徴群の生成"""
        features = []
        for i in range(count):
            # 円形に特化した特徴を生成
            feature = VisualFeature(
                edge_features={
                    'edge_histogram': self._generate_circular_edge_histogram() + np.random.normal(0, 0.03, 16),  # ノイズ削減
                    'edge_density': np.array([0.4 + np.random.normal(0, 0.03)], dtype=np.float32),
                    'contour_count': np.array([1 + np.random.randint(0, 3)], dtype=np.float32)
                },
                color_features={
                    'color_histogram': np.random.rand(48).astype(np.float32),
                    'dominant_colors': np.array([[200, 200, 200]], dtype=np.float32)
                },
                shape_features={
                    'aspect_ratio': 1.0 + np.random.normal(0, 0.05),  # 円に近い（安定性向上）
                    'solidity': 0.9 + np.random.normal(0, 0.03),     # 高い充実度
                    'extent': 0.78 + np.random.normal(0, 0.03),      # 円の理論値π/4≈0.785
                    'circularity': 0.9 + np.random.normal(0, 0.03)   # 高い円形度
                },
                texture_features={},
                spatial_location=(100 + i*10, 100 + i*10),
                extraction_timestamp=datetime.now(),
                confidence=0.85 + np.random.uniform(0, 0.1)  # より高品質な特徴
            )
            features.append(feature)
        return features
    
    def _create_square_features(self, count: int) -> list:
        """四角形特徴群の生成"""
        features = []
        for i in range(count):
            # 四角形に特化した特徴を生成
            feature = VisualFeature(
                edge_features={
                    'edge_histogram': self._generate_rectangular_edge_histogram() + np.random.normal(0, 0.03, 16),  # ノイズ削減
                    'edge_density': np.array([0.3 + np.random.normal(0, 0.03)], dtype=np.float32),
                    'contour_count': np.array([1 + np.random.randint(0, 2)], dtype=np.float32)
                },
                color_features={
                    'color_histogram': np.random.rand(48).astype(np.float32),
                    'dominant_colors': np.array([[150, 150, 150]], dtype=np.float32)
                },
                shape_features={
                    'aspect_ratio': 1.0 + np.random.normal(0, 0.03),  # 正方形に近い（安定性向上）
                    'solidity': 1.0,                                   # 完全な充実度
                    'extent': 1.0,                                     # 完全な拡張度
                    'circularity': 0.6 + np.random.normal(0, 0.03)    # 低い円形度
                },
                texture_features={},
                spatial_location=(200 + i*10, 200 + i*10),
                extraction_timestamp=datetime.now(),
                confidence=0.85 + np.random.uniform(0, 0.1)  # より高品質な特徴
            )
            features.append(feature)
        return features
    
    def _create_triangle_feature(self) -> VisualFeature:
        """三角形特徴の生成（未知物体として）"""
        return VisualFeature(
            edge_features={
                'edge_histogram': self._generate_triangular_edge_histogram(),
                'edge_density': np.array([0.35], dtype=np.float32),
                'contour_count': np.array([1], dtype=np.float32)
            },
            color_features={
                'color_histogram': np.random.rand(48).astype(np.float32),
                'dominant_colors': np.array([[100, 100, 100]], dtype=np.float32)
            },
            shape_features={
                'aspect_ratio': 1.2,
                'solidity': 0.85,
                'extent': 0.65,
                'circularity': 0.4
            },
            texture_features={},
            spatial_location=(300, 300),
            extraction_timestamp=datetime.now(),
            confidence=0.75
        )
    
    def _generate_circular_edge_histogram(self) -> np.ndarray:
        """円形のエッジ方向ヒストグラム生成"""
        # 円形は全方向のエッジを持つ
        histogram = np.ones(16, dtype=np.float32)
        histogram = histogram / np.sum(histogram)
        return histogram
    
    def _generate_rectangular_edge_histogram(self) -> np.ndarray:
        """四角形のエッジ方向ヒストグラム生成"""
        # 四角形は主に4方向（0°, 45°, 90°, 135°）のエッジを持つ
        histogram = np.zeros(16, dtype=np.float32)
        histogram[[0, 4, 8, 12]] = [0.25, 0.25, 0.25, 0.25]  # 4方向に集中
        return histogram
    
    def _generate_triangular_edge_histogram(self) -> np.ndarray:
        """三角形のエッジ方向ヒストグラム生成"""
        # 三角形は主に3方向のエッジを持つ
        histogram = np.zeros(16, dtype=np.float32)
        histogram[[0, 5, 11]] = [0.33, 0.33, 0.34]  # 3方向に集中
        return histogram
    
    def _create_synthetic_circle_image(self, variant: int = 0) -> np.ndarray:
        """合成円画像の生成"""
        if not OPENCV_AVAILABLE:
            return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # 円の描画
        center = (64, 64)
        radius = 25 + (variant % 10)  # バリエーション
        color = (200 + (variant % 56), 200, 200)
        cv2.circle(image, center, radius, color, -1)
        
        # ノイズの追加
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def _create_synthetic_square_image(self, variant: int = 0) -> np.ndarray:
        """合成四角画像の生成"""
        if not OPENCV_AVAILABLE:
            return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # 四角の描画
        size = 50 + (variant % 20)
        top_left = (64 - size//2, 64 - size//2)
        bottom_right = (64 + size//2, 64 + size//2)
        color = (150, 150 + (variant % 106), 150)
        cv2.rectangle(image, top_left, bottom_right, color, -1)
        
        # ノイズの追加
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def _print_recognition_result(self, test_name: str, result):
        """認識結果の表示"""
        print(f"\n   {test_name}の認識結果:")
        print(f"     ステータス: {result.status.value}")
        print(f"     信頼度: {result.confidence:.3f}")
        print(f"     処理時間: {result.processing_time:.3f}秒")
        
        if result.recognized_symbol:
            print(f"     認識記号: {result.recognized_symbol.semantic_label} ({result.recognized_symbol.symbol_id[:8]}...)")
        
        if result.alternative_matches:
            print(f"     代替候補数: {len(result.alternative_matches)}")
        
        if result.error_message:
            print(f"     メッセージ: {result.error_message}")
        
        print(f"     ユーザーメッセージ: {result.format_user_message()}")
    
    def _show_feature_details(self, name: str, features: VisualFeature):
        """特徴詳細の表示"""
        print(f"\n   {name}の特徴詳細:")
        print(f"     エッジ密度: {features.edge_features.get('edge_density', [0])[0]:.3f}")
        print(f"     輪郭数: {features.edge_features.get('contour_count', [0])[0]:.0f}")
        print(f"     アスペクト比: {features.shape_features.get('aspect_ratio', 0):.3f}")
        print(f"     充実度: {features.shape_features.get('solidity', 0):.3f}")
        print(f"     拡張度: {features.shape_features.get('extent', 0):.3f}")
        print(f"     円形度: {features.shape_features.get('circularity', 0):.3f}")
        print(f"     特徴複雑度: {features.get_feature_complexity():.3f}")
        print(f"     記号候補適性: {'適性あり' if features.is_extractable_symbol_candidate() else '適性なし'}")


def main():
    """メイン実行関数"""
    demo = VisualSymbolRecognitionDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()