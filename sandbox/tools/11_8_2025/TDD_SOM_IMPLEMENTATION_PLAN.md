# SOM実装TDD戦略マスタープラン

**プロジェクト**: エナクティブ意識フレームワーク用SOM実装  
**TDD指導原理**: 和田卓人（t_wada）のテスト駆動開発手法  
**技術スタック**: JAX + pytest + hypothesis  

---

## 📋 TDD実装フェーズとスケジュール

### **Phase 1: BMU計算のTDD (優先度: 最高)**

**期間**: 2-3日  
**Red-Green-Refactor サイクル**: 3-4回転予定

#### **Red Stage (失敗テスト作成)**
- [ ] ✅ 基本BMU検索失敗テスト (`test_som_tdd_phase1.py`)
- [ ] ✅ 距離計算精度失敗テスト
- [ ] ✅ 境界条件処理失敗テスト  
- [ ] ✅ JAX互換性失敗テスト

#### **Green Stage (最小実装)**
```python
# 実装予定: sandbox/tools/11_8_2025/som_bmu_jax.py
@jax.jit
def find_bmu_basic(weights: jnp.ndarray, input_vector: jnp.ndarray) -> Tuple[int, int]:
    """最小限のBMU実装（JIT最適化前）"""
    distances = jnp.linalg.norm(weights - input_vector, axis=2)
    return jnp.unravel_index(jnp.argmin(distances), weights.shape[:2])
```

#### **Refactor Stage (最適化)**
- [ ] JAX JIT最適化適用
- [ ] vmap による並列化
- [ ] メモリ効率改善
- [ ] 数値安定性向上

**完了基準**:
- [ ] BMU検索精度100% (テストケース100件)
- [ ] 実行時間 < 1ms (64x64マップ, 100次元入力)
- [ ] メモリ使用量 < 10MB
- [ ] 既存テスト破綻なし

---

### **Phase 2: 重み更新アルゴリズムTDD (優先度: 高)**

**期間**: 3-4日  
**依存**: Phase 1完了

#### **Red Stage**
- [ ] 学習率パラメータ失敗テスト
- [ ] 近傍関数計算失敗テスト
- [ ] 時間減衰機能失敗テスト
- [ ] 予測符号化統合失敗テスト

#### **Green Stage** 
```python
# 実装予定: sandbox/tools/11_8_2025/som_weight_update.py
@jax.jit
def update_weights_basic(weights, input_vector, bmu_coords, learning_rate, sigma):
    """基本的な重み更新実装"""
    # ガウシアン近傍関数
    # 重み更新ルール
    # 返り値: 更新された重み
```

#### **Refactor Stage**
- [ ] バッチ更新対応
- [ ] 適応的学習率
- [ ] 近傍形状最適化

**完了基準**:
- [ ] 収束性能テスト合格
- [ ] 数値安定性確保
- [ ] 大規模データ対応 (10万サンプル)

---

### **Phase 3: エナクティブ統合TDD (優先度: 高)**

**期間**: 4-5日  
**依存**: Phase 1-2完了 + 既存予測符号化システム連携

#### **Red Stage**
- [ ] ✅ 予測符号化統合失敗テスト (`test_som_predictive_integration.py`)
- [ ] ✅ 現象学的品質計算失敗テスト
- [ ] ✅ 時間的一貫性失敗テスト
- [ ] ✅ Φ値寄与度計算失敗テスト

#### **Green Stage**
```python
# 実装予定: sandbox/tools/11_8_2025/som_enactive_integration.py
class EnactiveConsciousnessIntegration:
    def process_predictive_state(self, predictive_state, som_state):
        """予測状態とSOM状態の統合処理"""
        
    def compute_consciousness_contribution(self):
        """意識レベルへの寄与度計算"""
        
    def update_phenomenological_quality(self):
        """現象学的品質の更新"""
```

#### **Refactor Stage**
- [ ] 既存システムとの最適化統合
- [ ] リアルタイム処理対応
- [ ] スケーラビリティ向上

**完了基準**:
- [ ] 既存予測符号化テスト 100% pass
- [ ] 意識レベル計算精度向上 (現行比+10%)
- [ ] 統合処理遅延 < 5ms

---

## 🔧 JAX JIT最適化戦略

### **最適化レベル段階**

1. **Level 1: 基本JIT適用**
   ```python
   @jax.jit
   def basic_function(x): return jnp.sum(x**2)
   ```

2. **Level 2: vmap並列化**
   ```python
   batch_function = jax.vmap(basic_function)
   ```

3. **Level 3: デバイス最適化**
   ```python
   @jax.jit
   @partial(jax.pmap, axis_name='batch')
   def distributed_function(x): return basic_function(x)
   ```

### **性能ベンチマーク目標**

| 機能 | 目標実行時間 | 目標メモリ | JIT効果 |
|------|-------------|-----------|---------|
| BMU検索 (64x64) | < 1ms | < 10MB | 10x高速化 |
| 重み更新 (1000サンプル) | < 50ms | < 100MB | 5x高速化 |
| 統合処理 | < 5ms | < 20MB | 3x高速化 |

---

## 🧪 既存テストシステム統合方針

### **非破壊統合原則**
1. **既存テスト維持**: 全ての既存テストが引き続き成功
2. **インターフェース保持**: 公開APIの破壊的変更禁止
3. **段階的統合**: 機能毎の個別統合で影響範囲限定

### **統合チェックポイント**
```bash
# Phase完了時に実行する統合確認コマンド
pytest tests/unit/domain/ -v --tb=short
pytest tests/integration/ -v --maxfail=3 
pytest sandbox/tools/11_8_2025/tests/ -v -m "not slow"
```

### **統合テストマトリクス**

| 既存システム | 統合レベル | テスト数 | 成功基準 |
|-------------|-----------|---------|----------|
| 予測符号化 | 深度統合 | 50+ | 95%+ pass |
| Φ値計算 | インターフェース統合 | 20+ | 100% pass |
| 意識状態 | データ統合 | 30+ | 98%+ pass |
| 時間的一貫性 | 部分統合 | 15+ | 90%+ pass |

---

## 📊 TDD進捗管理

### **マイルストーン管理**
```python
# 進捗確認テスト: tests/test_tdd_progress.py
def test_phase1_milestone():
    assert all([
        bmu_implementation_complete(),
        performance_targets_met(), 
        integration_tests_pass()
    ])

def test_phase2_milestone():
    # Phase2完了基準
    
def test_phase3_milestone():
    # Phase3完了基準
```

### **品質ゲート**
各Phase完了前に以下を満たす必要がある：

1. **コードカバレッジ**: 90%以上
2. **テスト成功率**: Phase内95%以上, 既存90%以上
3. **性能要件**: ベンチマーク基準クリア
4. **コードレビュー**: TDD原則準拠確認

---

## 🚀 実行手順

### **Day 1: 環境整備 + Phase1 Red**
```bash
# 1. テスト環境確認
cd /Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/11_8_2025
python -m pytest tests/test_som_tdd_phase1.py -v -x

# 2. 依存関係インストール
pip install jax jaxlib hypothesis

# 3. Red Stage実行（すべて失敗することを確認）
python -m pytest tests/test_som_tdd_phase1.py -m "unit" -x
```

### **Day 2-3: Phase1 Green + Refactor**
```bash
# 1. 最小実装でテストを通す
# - som_bmu_jax.py の基本実装
# - テスト1つずつ通しながら機能追加

# 2. Refactor で最適化
# - JAX JIT適用
# - 性能要件確認
```

### **Day 4-7: Phase2実行**
```bash
# Phase1完了後にPhase2開始
python -m pytest tests/test_som_weight_update.py
```

### **Day 8-12: Phase3実行**
```bash 
# Phase2完了後にPhase3開始
python -m pytest tests/test_som_predictive_integration.py
```

---

## ⚠️ リスク管理

### **技術リスク**
1. **JAX互換性問題**: 既存NumPyコードとの互換性
   - **軽減策**: 段階的移行とfallback実装
2. **性能要件未達**: JIT最適化効果不足
   - **軽減策**: プロファイリング駆動最適化
3. **既存システム破綻**: 統合時の予期しない副作用
   - **軽減策**: 包括的回帰テスト

### **スケジュールリスク**
1. **Phase間依存**: 前段階の遅れが後続に影響
   - **軽減策**: 並行作業可能な部分の特定
2. **複雑性過小評価**: 統合の技術的複雑さ
   - **軽減策**: 段階的統合とMVP志向

---

## 📈 成功基準

### **最終目標**
- [ ] **機能**: SOM + 予測符号化 + Φ値計算の完全統合
- [ ] **性能**: 既存システム比で2-10x高速化
- [ ] **品質**: テストカバレッジ90%以上, 既存テスト破綻なし
- [ ] **保守性**: TDD原則に基づく高品質コード

### **検証方法**
```bash
# 最終統合テスト実行
python -m pytest tests/ sandbox/tools/11_8_2025/tests/ -v --cov=./ --cov-report=html
python -c "
import sandbox.tools.11_8_2025.enactive_som as som
framework = som.EnactiveConsciousnessFramework()
# エンドツーエンドテスト実行
"
```

---

**実装開始**: Ready to GO! 🎯  
**TDD原則**: Red → Green → Refactor  
**品質保証**: テストファーストアプローチで確実な実装を目指す