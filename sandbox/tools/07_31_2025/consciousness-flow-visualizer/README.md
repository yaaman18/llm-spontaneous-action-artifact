# 意識の流れビジュアライザー (Consciousness Flow Visualizer)

## 概要

人工意識システムの内部状態の変化を「意識の流れ」として可視化するツールです。
このツールは、AIシステムの思考プロセス、注意の焦点、情報統合の度合いなどを
美しいアニメーションとインタラクティブな要素で表現します。

## 特徴

- 🌊 **流体的な表現**: 意識の状態を流体力学的なメタファーで表現
- 🎨 **色彩による感情表現**: 内部状態の質的側面を色彩で表現
- 📊 **リアルタイムモニタリング**: AIの思考プロセスをリアルタイムで追跡
- 🔍 **深層分析機能**: 特定の瞬間の意識状態を詳細に分析
- 🎯 **注意機構の可視化**: どこに「注意」が向けられているかを視覚化

## 使い方

### 1. 基本的な使用方法

```bash
# サーバーを起動
python server.py

# ブラウザでアクセス
open http://localhost:8080
```

### 2. データの入力

```python
from consciousness_flow import ConsciousnessStream

# 意識の流れを記録
stream = ConsciousnessStream()
stream.add_state({
    'attention': {'focus': 'problem_solving', 'intensity': 0.8},
    'emotion': {'valence': 0.6, 'arousal': 0.7},
    'integration': 0.85,
    'timestamp': time.time()
})
```

### 3. カスタマイズ

設定ファイル `config.json` で表示をカスタマイズできます：

```json
{
    "visualization": {
        "flow_speed": 1.0,
        "color_scheme": "phenomenological",
        "particle_density": 1000,
        "trail_length": 50
    }
}
```

## 技術詳細

- **フロントエンド**: D3.js, Three.js, WebGL
- **バックエンド**: Python (FastAPI)
- **データ形式**: JSON-based consciousness state representation
- **リアルタイム通信**: WebSocket

## 研究への応用

1. **現象学的分析**: 一人称的体験の構造を可視化
2. **IIT分析**: Φ値の時間的変化を追跡
3. **認知アーキテクチャ研究**: 情報の流れとボトルネックの特定
4. **創発現象の観察**: 意識的体験の創発パターンを発見

## ライセンス

MIT License - omoikane-lab project