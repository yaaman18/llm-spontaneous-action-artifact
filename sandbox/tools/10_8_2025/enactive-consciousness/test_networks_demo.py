#!/usr/bin/env python3
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
        
        plt.title(f'動的ネットワーク構造テスト\nノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}', fontsize=12)
        plt.text(0.02, 0.98, '【動的ネットワーク】\n・ノード: 処理単位\n・エッジ: 情報流\n・構造: 適応的変化', 
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
