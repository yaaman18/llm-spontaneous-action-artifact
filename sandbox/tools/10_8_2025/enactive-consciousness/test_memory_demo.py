#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    def test_memory_visualization():
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        # モックデータ
        timestamps = np.arange(15)
        significances = 0.3 + 0.4 * np.sin(timestamps * 0.3)
        recalls = np.cumsum(np.random.poisson(1, 15))
        coherence = 0.5 + 0.3 * np.cos(timestamps * 0.2)
        meaning_strength = 0.4 + 0.4 * np.tanh(timestamps * 0.1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('体験記憶システム - テスト結果', fontsize=12)
        
        ax1.plot(timestamps, significances, 'b-o', label='重要度重み')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(timestamps, recalls, 'r-s', label='想起回数')
        ax1.set_title('体験記憶の発達')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(timestamps, coherence, 'g-o', label='円環的一貫性')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(timestamps, meaning_strength, 'm-^', label='意味強度')
        ax2.set_title('円環的因果性動力学')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 意識レベル
        consciousness = 0.4 + 0.3 * np.tanh(timestamps * 0.15)
        integration = 0.5 + 0.2 * np.sin(timestamps * 0.25)
        ax3.plot(timestamps, consciousness, 'purple', marker='o', label='意識レベル')
        ax3.plot(timestamps, integration, 'orange', marker='s', label='統合一貫性')
        ax3.axhline(y=0.55, color='red', linestyle='--', label='意識閾値')
        ax3.set_title('統合意識の発達')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 性能概観
        categories = ['意識\nレベル', '統合\n一貫性', '円環的\n因果性']
        values = [np.mean(consciousness), np.mean(integration), np.mean(coherence)]
        bars = ax4.bar(categories, values, color=['purple', 'orange', 'green'], alpha=0.7)
        ax4.set_title('システム統合性能')
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "memory_demo_test.png", dpi=100)
        print(f"✅ Experiential Memory テスト完了: {output_dir / 'memory_demo_test.png'}")
        plt.close()
    
    test_memory_visualization()
    
except Exception as e:
    print(f"❌ Experiential Memory テスト失敗: {e}")
    sys.exit(1)
