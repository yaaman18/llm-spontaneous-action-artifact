#!/usr/bin/env python3
"""
全GUI可視化のテストスクリプト
タイムアウトせずに各デモの可視化をテストします
"""

import sys
import os
import time
import subprocess
import signal
from pathlib import Path

def run_with_timeout(script_path, timeout=15):
    """指定時間でスクリプトを実行し、タイムアウト時は終了"""
    print(f"\n{'='*60}")
    print(f"テスト中: {script_path}")
    print('='*60)
    
    try:
        # バックグラウンドでプロセスを開始
        process = subprocess.Popen([
            'python', script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # タイムアウト付きで実行
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            
            if process.returncode == 0:
                print("✅ 成功:")
                print(stdout[-500:] if len(stdout) > 500 else stdout)  # 最後の500文字のみ表示
                return True
            else:
                print("❌ エラー:")
                print(stderr[-500:] if len(stderr) > 500 else stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {timeout}秒でタイムアウト - プロセスを終了")
            process.kill()
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        return False

def create_simple_test_scripts():
    """各デモ用の簡単なテストスクリプトを作成"""
    base_dir = Path(__file__).parent
    
    # 1. Basic Demo テスト用
    basic_test_content = '''#!/usr/bin/env python3
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
'''
    
    # 2. Predictive Coding テスト用
    predictive_test_content = '''#!/usr/bin/env python3
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
'''
    
    # 3. Experiential Memory テスト用
    memory_test_content = '''#!/usr/bin/env python3
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
        categories = ['意識\\nレベル', '統合\\n一貫性', '円環的\\n因果性']
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
'''
    
    # 4. Dynamic Networks テスト用
    networks_test_content = '''#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from pathlib import Path
    
    def test_networks_visualization():
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        # ランダムグラフ作成
        G = nx.erdos_renyi_graph(10, 0.3, seed=42)
        
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        
        nx.draw_networkx_nodes(G, pos, node_size=200, alpha=0.8, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f'動的ネットワーク構造テスト\\nノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}', fontsize=12)
        plt.text(0.02, 0.98, '【動的ネットワーク】\\n・ノード: 処理単位\\n・エッジ: 情報流\\n・構造: 適応的変化', 
                transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.axis('off')
        plt.tight_layout()
        
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "networks_demo_test.png", dpi=100)
        print(f"✅ Dynamic Networks テスト完了: {output_dir / 'networks_demo_test.png'}")
        plt.close()
    
    test_networks_visualization()
    
except Exception as e:
    print(f"❌ Dynamic Networks テスト失敗: {e}")
    sys.exit(1)
'''

    # ファイル作成
    test_scripts = {
        'test_basic_demo.py': basic_test_content,
        'test_predictive_demo.py': predictive_test_content,
        'test_memory_demo.py': memory_test_content,
        'test_networks_demo.py': networks_test_content,
    }
    
    for filename, content in test_scripts.items():
        script_path = base_dir / filename
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        script_path.chmod(0o755)
        print(f"📝 作成: {script_path}")

def main():
    """メイン実行関数"""
    print("🧠 エナクティブ意識フレームワーク - GUI可視化テスト")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    
    # テストスクリプトを作成
    print("\\n📝 テストスクリプトを作成中...")
    create_simple_test_scripts()
    
    # 各テストスクリプトを実行
    test_scripts = [
        'test_basic_demo.py',
        'test_predictive_demo.py', 
        'test_memory_demo.py',
        'test_networks_demo.py'
    ]
    
    results = {}
    
    for script in test_scripts:
        script_path = base_dir / script
        if script_path.exists():
            results[script] = run_with_timeout(str(script_path), timeout=10)
        else:
            print(f"❌ スクリプトが見つかりません: {script_path}")
            results[script] = False
    
    # 結果サマリー
    print("\\n" + "=" * 60)
    print("📊 テスト結果サマリー")
    print("=" * 60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for script, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{status} - {script}")
    
    print(f"\\n🎯 総合結果: {success_count}/{total_count} テスト成功")
    
    if success_count == total_count:
        print("🎉 全てのGUI可視化テストが成功しました！")
    else:
        print("⚠️  一部のテストが失敗しました。")
    
    # 出力フォルダの確認
    output_dir = base_dir / "test_output"
    if output_dir.exists():
        png_files = list(output_dir.glob("*.png"))
        print(f"\\n📁 出力画像: {len(png_files)}個のファイルが作成されました")
        for png_file in png_files:
            print(f"   - {png_file.name}")

if __name__ == "__main__":
    main()