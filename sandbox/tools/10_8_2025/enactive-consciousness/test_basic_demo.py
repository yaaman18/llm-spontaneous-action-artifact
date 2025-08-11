#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import matplotlib
    matplotlib.use('Agg')  # ノンGUIバックエンド
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    import numpy as np
    from pathlib import Path
    
    # モックデータ作成
    class MockMoment:
        def __init__(self, timestamp, weights):
            self.timestamp = timestamp
            self.synthesis_weights = weights
    
    class MockState:
        def __init__(self, confidence):
            self.schema_confidence = confidence
    
    # テスト用可視化関数
    def test_basic_visualization():
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        temporal_moments = [MockMoment(i, jnp.array([0.3+0.1*i, 0.4, 0.3-0.1*i])) for i in range(10)]
        body_states = [MockState(0.5 + 0.3*np.sin(i*0.1)) for i in range(10)]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle('エナクティブ意識システム - 基本デモテスト', fontsize=12)
        
        timestamps = [m.timestamp for m in temporal_moments]
        weights = jnp.stack([m.synthesis_weights for m in temporal_moments])
        
        ax1.plot(timestamps, weights[:, 0], label='保持', marker='o')
        ax1.plot(timestamps, weights[:, 1], label='現在', marker='s') 
        ax1.plot(timestamps, weights[:, 2], label='予持', marker='^')
        ax1.set_title('時間統合重み')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        confidences = [s.schema_confidence for s in body_states]
        ax2.plot(range(len(confidences)), confidences, 'g-o')
        ax2.set_title('身体スキーマ信頼度')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "basic_demo_test.png", dpi=100)
        print(f"✅ Basic Demo テスト完了: {output_dir / 'basic_demo_test.png'}")
        plt.close()
    
    test_basic_visualization()
    
except Exception as e:
    print(f"❌ Basic Demo テスト失敗: {e}")
    sys.exit(1)
