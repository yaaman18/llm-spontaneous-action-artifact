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
    
    def test_predictive_visualization():
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        # モックデータ
        timestamps = np.arange(20)
        errors = 1.0 * np.exp(-timestamps * 0.1) + 0.1 * np.random.random(20)
        confidence = 1.0 - 0.8 * np.exp(-timestamps * 0.1)
        convergence = np.tanh(timestamps * 0.2)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('階層予測コーディングシステム - テスト結果', fontsize=12)
        
        axes[0, 0].plot(timestamps, errors, 'b-', linewidth=2)
        axes[0, 0].set_title('予測誤差の進化')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(timestamps, confidence, 'g-', linewidth=2)
        axes[0, 1].set_title('予測信頼度')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].fill_between(timestamps, convergence, alpha=0.6, color='orange')
        axes[1, 0].set_title('予測収束状況')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(errors, confidence, alpha=0.6, color='purple')
        axes[1, 1].set_title('誤差-信頼度相関')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "predictive_demo_test.png", dpi=100)
        print(f"✅ Predictive Coding テスト完了: {output_dir / 'predictive_demo_test.png'}")
        plt.close()
    
    test_predictive_visualization()
    
except Exception as e:
    print(f"❌ Predictive Coding テスト失敗: {e}")
    sys.exit(1)
