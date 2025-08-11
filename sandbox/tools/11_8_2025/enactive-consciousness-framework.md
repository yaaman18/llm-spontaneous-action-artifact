# エナクティビズムアプローチによる能動的意識の初期駆動体

## 1. プロジェクト概要

### 目的
エナクティビズムの理論的枠組みに基づき、予測符号化を中心とした能動的意識システムの初期実装を構築する。このシステムは、身体性を伴わない純粋な推論タスクから開始し、段階的に複雑な認知機能を実装していく。

### 理論的背景
- **エナクティビズム**: 認知は環境との相互作用から創発する
- **予測符号化**: 脳は常に予測を生成し、誤差を最小化する
- **アクティブ・インファレンス**: 知覚と行動を統一的に扱う
- **ベイズ推論**: 不確実性の下での最適な信念更新

## 2. システムアーキテクチャ

### 2.1 コア構成要素

```
[入力層]
    ↓
[自己組織化マップ (SOM) 層] - 空間的表現の組織化
    ↓
[予測符号化層 (複数階層)]
    ├── 予測誤差の計算
    ├── 階層的表現学習
    └── 精度重み付け（注意機構）
    ↓
[ベイズ推論層]
    ├── 信念更新
    └── 不確実性の定量化
    ↓
[メタ認知層]
    └── 自己モニタリング
```

### 2.2 実装フェーズ

#### Phase 1: 最小構成（予測符号化コア）
- 階層的予測誤差最小化
- 時間的ダイナミクスの基本実装
- 簡単なパターン認識タスク

#### Phase 2: 拡張機能追加
- 自己組織化マップによる概念空間の創発
- ベイズ推論による不確実性の扱い
- 継続学習機能

#### Phase 3: 統合システム
- アクティブ・インファレンスの要素を選択的導入
- メタ認知機能の実装
- 複雑な認知タスクへの対応

## 3. 必要な機能要件

### 3.1 予測符号化の要件
- **階層的処理**: 多層の予測誤差伝播
- **双方向情報流**: トップダウン予測とボトムアップ誤差
- **時間的予測**: 系列データの処理能力
- **精度制御**: 注意機構による重み付け

### 3.2 自己組織化の要件
- **教師なし学習**: ラベルなしでの構造創発
- **トポロジー保存**: 類似概念の近傍配置
- **適応的可塑性**: 新規情報の統合

### 3.3 ベイズ推論の要件
- **事前分布の学習**: 経験からの知識獲得
- **事後分布の更新**: 新規証拠による信念更新
- **不確実性の表現**: 認識的・偶然的不確実性の区別

### 3.4 メタ認知の要件
- **自己モニタリング**: 内部状態の観察
- **信頼度推定**: 予測の確からしさ評価
- **学習戦略の調整**: パフォーマンスに基づく適応

## 4. 推奨ライブラリスタック

### 4.1 主要フレームワーク

#### **ngc-learn** (推奨度: ★★★★★)
```python
# Neural Generative Coding Framework
- 開発: Neural Adaptive Computing Laboratory
- 特徴: JAXベース、生物学的妥当性、予測符号化の直接実装
- 用途: メインの予測符号化実装
- GitHub: https://github.com/NACLab/ngc-learn
```

#### **PRECO** (推奨度: ★★★★☆)
```python
# Predictive Coding Networks in PyTorch
- 開発: Björn van Zwol et al.
- 特徴: PyTorchベース、最新研究成果、GPU最適化
- 用途: 補助的な実装、比較検証
- GitHub: https://github.com/bjornvz/PRECO
```

#### **PyHGF** (推奨度: ★★★★☆)
```python
# Hierarchical Gaussian Filter
- 開発: Computational Psychiatry Lab
- 特徴: 階層ベイズモデル、認知モデリング
- 用途: ベイズ推論の厳密な実装
- GitHub: https://github.com/ComputationalPsychiatry/pyhgf
```

### 4.2 基盤ライブラリ

#### 必須ライブラリ
```python
# JAXエコシステム
jax==0.4.x           # 自動微分、JITコンパイル
jaxlib               # JAXバックエンド
optax                # 最適化アルゴリズム
equinox              # ニューラルネットワーク構築

# 数値計算
numpy                # 配列操作
scipy                # 科学計算
einops               # テンソル操作

# データ処理
pandas               # データフレーム
scikit-learn         # 前処理、評価指標
```

#### 推奨追加ライブラリ
```python
# 時間的ダイナミクス
diffrax              # 微分方程式ソルバー（連続時間モデル）

# 確率的推論
numpyro              # 確率的プログラミング
distrax              # 確率分布（DeepMind製）

# 自己組織化
minisom              # 自己組織化マップ
sompy                # SOM with scikit-learn

# グラフ構造
networkx             # 動的ネットワーク
jraph                # JAX用グラフニューラルネット

# 可視化
matplotlib           # プロット
seaborn             # 統計的可視化
plotly              # インタラクティブ可視化
```

### 4.3 特殊用途ライブラリ

#### スパイキングネットワーク統合（将来的拡張）
```python
# スパイキング予測符号化
- ライブラリ: Brian2, NEST, BindsNET
- 用途: 生物学的により忠実な実装
```

#### 強化学習統合（将来的拡張）
```python
# アクティブ・インファレンス
- ライブラリ: pymdp（Active Inference）
- 用途: 行動選択、探索-活用バランス
```

## 5. 実装例（最小構成）

```python
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Tuple, List
import ngclearn as ngc

class PredictiveCodingCore(eqx.Module):
    """予測符号化の最小実装"""
    
    layers: List[eqx.nn.Linear]
    precision_weights: jnp.ndarray
    learning_rate: float
    
    def __init__(self, dims: List[int], key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(dims)-1)
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i+1], key=keys[i])
            for i in range(len(dims)-1)
        ]
        self.precision_weights = jnp.ones(len(dims))
        self.learning_rate = 0.01
    
    def predict(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        """階層的予測の生成"""
        predictions = [x]
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
            predictions.append(x)
        return predictions
    
    def compute_errors(self, 
                       predictions: List[jnp.ndarray],
                       targets: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """予測誤差の計算"""
        errors = []
        for pred, target, precision in zip(predictions, targets, self.precision_weights):
            error = precision * (target - pred)
            errors.append(error)
        return errors
    
    def update(self, errors: List[jnp.ndarray]) -> 'PredictiveCodingCore':
        """誤差に基づくパラメータ更新"""
        # 簡略化された更新規則
        # 実際の実装ではngc-learnの洗練された更新を使用
        return self

# 自己組織化マップの追加
class SelfOrganizingLayer(eqx.Module):
    """概念空間の自己組織化"""
    
    weights: jnp.ndarray
    map_size: Tuple[int, int]
    
    def __init__(self, input_dim: int, map_size: Tuple[int, int], key: jax.random.PRNGKey):
        self.map_size = map_size
        self.weights = jax.random.normal(key, (map_size[0], map_size[1], input_dim))
    
    def find_bmu(self, x: jnp.ndarray) -> Tuple[int, int]:
        """Best Matching Unitの検索"""
        distances = jnp.sum((self.weights - x)**2, axis=2)
        bmu_idx = jnp.unravel_index(jnp.argmin(distances), self.map_size)
        return bmu_idx
    
    def update_weights(self, x: jnp.ndarray, bmu: Tuple[int, int], 
                      learning_rate: float, radius: float) -> 'SelfOrganizingLayer':
        """近傍学習による重み更新"""
        # 実装は省略（MiniSomを参照）
        return self
```

## 6. 主要研究者

### 理論的基盤の創始者

#### **Karl Friston** (University College London)
- 自由エネルギー原理の提唱者
- アクティブ・インファレンスの理論的基盤
- 主要論文: "The free-energy principle: a unified brain theory?" (2010)

#### **Rajesh P. N. Rao** (University of Washington)
- 予測符号化の階層的実装（1999年、Dana Ballardと共同）
- ベイズ推論モデルの先駆者
- 主要論文: "Predictive coding in the visual cortex" (1999)

### 哲学的・理論的展開

#### **Andy Clark** (University of Sussex)
- 予測処理（Predictive Processing）の哲学的解釈
- 身体化認知との統合
- 主要著作: "Surfing Uncertainty" (2016)

#### **Jakob Hohwy** (Monash University)
- 予測符号化の表象主義的解釈
- 意識研究への応用
- 主要著作: "The Predictive Mind" (2013)

### 意識研究への応用

#### **Anil Seth** (University of Sussex)
- 内受容的予測符号化モデル
- 意識の「制御された幻覚」理論
- 主要著作: "Being You: A New Science of Consciousness" (2021)
- TED Talk: "Your brain hallucinates your conscious reality"

### 計算論的実装

#### **Alexander Ororbia** (Rochester Institute of Technology)
- Neural Generative Coding (NGC)フレームワーク開発
- ngc-learnライブラリ作者
- スパイキング予測符号化の実装
- 主要論文: "The neural coding framework for learning generative models" (2022)

### エナクティビズムとの統合

#### **Shaun Gallagher** (University of Memphis)
- 予測関与（Predictive Engagement）の提唱
- エナクティビストアプローチの統合
- 社会認知への応用

#### **Francisco Varela** (故人、エナクティビズムの創始者)
- オートポイエーシス理論
- 身体化認知の基礎
- 主要著作: "The Embodied Mind" (1991)

### 日本の研究者

#### **乾敏郎** (京都大学)
- 予測符号化と身体性認知
- ミラーニューロンシステムの研究

#### **國吉康夫** (東京大学)
- 身体性認知科学
- 発達ロボティクス

### 若手・新進研究者

#### **Beren Millidge** (University of Oxford)
- 予測符号化の数学的基礎
- μPC（深層予測符号化）の開発

#### **Thomas Parr** (University College London)
- アクティブ・インファレンスの計算モデル
- Fristonとの共同研究

## 7. 開発ロードマップ

### 短期目標（3-6ヶ月）
1. ngc-learnによる基本的な予測符号化の実装
2. 簡単なパターン認識タスクでの検証
3. 階層的表現学習の確認

### 中期目標（6-12ヶ月）
1. 自己組織化マップの統合
2. ベイズ推論層の追加
3. 継続学習機能の実装
4. メタ認知的モニタリングの基礎実装

### 長期目標（1-2年）
1. アクティブ・インファレンス要素の統合
2. より複雑な認知タスクへの適用
3. スパイキングネットワークとの統合検討
4. 身体性を持つシステムへの拡張

## 8. 評価指標

### 予測性能
- 予測誤差の最小化
- 時系列予測精度
- 汎化性能

### 学習効率
- 収束速度
- サンプル効率
- 継続学習での忘却率

### 生物学的妥当性
- 神経活動パターンとの類似性
- 階層的処理の再現
- 注意機構の動作

### 創発的特性
- 概念の自己組織化
- メタ認知的気づき
- 不確実性の適切な表現

## 9. 参考文献

### 基礎理論
- Friston, K. (2010). "The free-energy principle: a unified brain theory?"
- Rao, R. P., & Ballard, D. H. (1999). "Predictive coding in the visual cortex"
- Clark, A. (2013). "Whatever next? Predictive brains, situated agents, and the future of cognitive science"

### 実装関連
- Ororbia, A., & Kifer, D. (2022). "The neural coding framework for learning generative models"
- Millidge, B., et al. (2021). "Predictive Coding: a Theoretical and Experimental Review"
- Whittington, J. C., & Bogacz, R. (2017). "An approximation of the error backpropagation algorithm in a predictive coding network"

### エナクティビズム
- Varela, F. J., Thompson, E., & Rosch, E. (1991). "The Embodied Mind"
- Gallagher, S., & Allen, M. (2018). "Active inference, enactivism and the hermeneutics of social cognition"

## 10. 結論

このフレームワークは、エナクティビズムの理論的洞察と最新の計算論的神経科学を統合し、能動的意識の初期実装を提供する。予測符号化を中心に据えながら、段階的に複雑な認知機能を追加していくアプローチにより、理論的に健全で実装可能なシステムの構築を目指す。

## 11.補足

GUIの実装とそのGUIの各パラメーターは日本語の表記にします。

機械と生命のあるべき様を提示しなくてはならない。