#!/usr/bin/env python3
"""
Simple script to run only the visualization example from basic_usage.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def create_mock_data():
    """Create mock consciousness data for visualization."""
    moments_data = []
    
    for i in range(20):
        moment_data = {
            'moment': i,
            'integration_confidence': 0.3 + 0.4 * np.sin(i * 0.3) + 0.1 * np.random.random(),
            'body_confidence': 0.4 + 0.3 * np.cos(i * 0.2) + 0.1 * np.random.random(),
            'consciousness_level': np.random.choice(['Low', 'Medium', 'High', 'Peak'])
        }
        moments_data.append(moment_data)
    
    return moments_data

def visualization_example(moments_data):
    """意識処理の可視化を作成する。"""
    print("\n=== 可視化例 ===")
    
    if not moments_data:
        print("可視化用のデータがありません")
        return
    
    try:
        import matplotlib.pyplot as plt
        # 日本語フォント設定（シンプル版）
        try:
            plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
        except:
            pass
        
        # プロット用データの抽出
        moments = [data['moment'] for data in moments_data]
        integration_confidence = [data['integration_confidence'] for data in moments_data]
        body_confidence = [data['body_confidence'] for data in moments_data]
        
        # プロット作成
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle('エナクティブ意識フレームワーク - 意識状態の可視化\n（現象学的時間意識と身体化認知の統合分析）', 
                    fontsize=14, fontweight='bold')
        
        # 信頼度指標のプロット
        plt.subplot(1, 2, 1)
        line1 = plt.plot(moments, integration_confidence, 'b-o', label='統合信頼度', linewidth=2, markersize=6)
        line2 = plt.plot(moments, body_confidence, 'r-s', label='身体スキーマ信頼度', linewidth=2, markersize=6)
        plt.xlabel('時間モーメント', fontsize=12)
        plt.ylabel('信頼度スコア', fontsize=12)
        plt.title('時間経過における意識統合の信頼度変化\n（フッサール現象学的時間構造に基づく）', 
                 fontsize=11, pad=20)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 説明テキストの追加
        plt.text(0.02, 0.98, 
                '【統合信頼度】\n意識の統合的一貫性\n（保持-現在-予持の統合）\n\n【身体スキーマ信頼度】\nメルロ=ポンティの\n身体化認知の確実性', 
                transform=plt.gca().transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # 意識レベル分布のプロット
        plt.subplot(1, 2, 2)
        levels = [data['consciousness_level'] for data in moments_data]
        level_counts = {level: levels.count(level) for level in set(levels)}
        
        # レベルを日本語に変換
        japanese_levels = {'Low': '低位', 'Medium': '中位', 'High': '高位', 'Peak': '最高位'}
        jp_level_counts = {japanese_levels.get(level, level): count for level, count in level_counts.items()}
        
        colors = ['lightcoral', 'gold', 'lightgreen', 'royalblue']
        bars = plt.bar(jp_level_counts.keys(), jp_level_counts.values(), 
                      alpha=0.8, color=colors[:len(jp_level_counts)])
        plt.xlabel('意識レベル', fontsize=12)
        plt.ylabel('出現頻度', fontsize=12)
        plt.title('意識レベルの分布パターン\n（統合情報理論ΦとエナクティブカップリングΣの統合）', 
                 fontsize=11, pad=20)
        
        # バーに数値ラベルを追加
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # 説明テキストの追加
        plt.text(0.02, 0.98, 
                '【意識レベル分類】\n低位: 基礎的知覚統合\n中位: 身体-環境カップリング\n高位: 反省的意識\n最高位: 超越論的統覚', 
                transform=plt.gca().transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout()
        
        # プロット保存
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "consciousness_analysis_jp.png", dpi=150, bbox_inches='tight')
        
        print(f"可視化結果を保存しました: {output_dir / 'consciousness_analysis_jp.png'}")
        plt.show()
        
    except ImportError:
        print("Matplotlibが利用できません")
    except Exception as e:
        print(f"可視化エラー: {e}")

def main():
    """Main execution function."""
    print("Enactive Consciousness - Visualization Demo")
    print("=" * 50)
    
    # Create mock data
    moments_data = create_mock_data()
    
    # Run visualization
    visualization_example(moments_data)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()