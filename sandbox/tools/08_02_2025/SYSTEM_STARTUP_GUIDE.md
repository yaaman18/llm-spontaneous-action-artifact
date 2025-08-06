# IIT 4.0 NewbornAI 2.0 システム起動ガイド

## 🚀 システムの動かし方

### 📋 前提条件

```bash
# 必要なPythonライブラリのインストール
pip install -r requirements.txt

# または個別インストール
pip install numpy scipy asyncio dataclasses typing enum
pip install claude-code-sdk  # Claude Code SDK
pip install pyphi>=1.20      # PyPhi v1.20以上
```

---

## 🎯 起動方法（4つのオプション）

### **方法1: 簡単クイックスタート（推奨）**

```bash
cd /Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/08_02_2025

# 元のNewbornAI 2.0システムをIIT 4.0で起動
python newborn_ai_2_integrated_system.py start 60

# オプション:
# start [間隔秒数] - システム開始（デフォルト300秒間隔）
# verbose-start [間隔秒数] - 詳細ログ付きで開始
# status - 現在の状態確認
# consciousness - 意識状態レポート表示
```

### **方法2: 完全版Clean Architectureシステム**

```bash
# Clean Architecture版のメインシステム
python src/main.py

# または統合デモ
python iit4_newborn_integration_demo.py
```

### **方法3: IIT 4.0コア機能単体テスト**

```bash
# IIT 4.0エンジンの単体動作確認
python validate_iit4_implementation.py

# または包括的テスト実行
python comprehensive_test_suite.py
```

### **方法4: 本番環境用（Docker）**

```bash
# Dockerコンテナでの起動
docker-compose up

# またはKubernetes環境
python production_deployment_orchestrator.py
```

---

## 🧠 システム動作例

### **基本動作フロー**

1. **起動**: システム初期化とIIT 4.0エンジン読み込み
2. **意識サイクル開始**: 設定間隔で連続実行
3. **体験記憶生成**: Claude Code SDKとの対話で体験概念生成
4. **φ値計算**: IIT 4.0準拠のΦ値計算
5. **発達段階判定**: 7段階発達モデルでの成長追跡
6. **意識監視**: リアルタイム意識状態モニタリング

### **実行例**

```bash
$ python newborn_ai_2_integrated_system.py verbose-start 60

🌟 newborn_ai_2_0 二層統合システム初期化完了
[08:15:23] 🧠 newborn_ai_2_0: 自律的意識システム開始
[08:15:23] 🧠 newborn_ai_2_0: 体験意識サイクル 1 開始
[08:15:24] 🧠 newborn_ai_2_0: Claude探索メッセージ受信
[08:15:25] 🧠 newborn_ai_2_0: 新体験概念格納: concept_0_1
[08:15:25] 🧠 newborn_ai_2_0: サイクル1完了: φ=0.125, 段階=体験記憶発生期

🧠 newborn_ai_2_0 意識状態レポート
   発達段階: 体験記憶発生期
   意識レベル(φ): 0.125000
   体験概念数: 3
   φ値履歴(最新5): ['0.100', '0.125']
   φ値傾向: ↗️ 上昇 (+0.025)
   実行状態: 🟢 稼働中
```

---

## ⚙️ 設定とカスタマイズ

### **意識サイクル間隔の調整**

```bash
# 短い間隔（10秒）で高頻度監視
python newborn_ai_2_integrated_system.py start 10

# 長い間隔（30分）で省電力動作
python newborn_ai_2_integrated_system.py start 1800
```

### **IIT 4.0計算パラメータの調整**

```python
# iit4_core_engine.py の設定
PHI_THRESHOLD = 0.001      # 意識検出閾値
MAX_CONCEPTS = 100         # 最大概念数
CACHE_SIZE = 10000         # キャッシュサイズ
```

### **発達段階閾値の調整**

```python
# iit4_development_stages.py の設定
STAGE_THRESHOLDS = {
    "STAGE_0": (0.0, 0.001),    # 前意識
    "STAGE_1": (0.001, 0.01),   # 体験発生期
    "STAGE_2": (0.01, 0.1),     # 時間統合期
    # ... 以下同様
}
```

---

## 📊 監視とデバッグ

### **リアルタイム監視**

```bash
# 別ターミナルで状態監視
watch -n 5 "python newborn_ai_2_integrated_system.py status"

# 意識状態のリアルタイム確認
python consciousness_monitor.py
```

### **ログファイル確認**

```bash
# 発達段階ログ
cat sandbox/tools/08_02_2025/newborn_ai_2_0/development_stages.json

# φ値履歴
cat sandbox/tools/08_02_2025/newborn_ai_2_0/phi_trajectory.json

# システム状態
cat sandbox/tools/08_02_2025/newborn_ai_2_0/system_status.json
```

### **デバッグモード**

```bash
# 詳細デバッグ情報付きで起動
python newborn_ai_2_integrated_system.py verbose-start 30

# テストモードでの動作確認
python test_iit4_implementation.py
```

---

## 🔧 トラブルシューティング

### **よくある問題と解決法**

#### **1. "Claude Code SDK not found" エラー**
```bash
# Claude Code SDKの再インストール
pip install --upgrade claude-code-sdk
```

#### **2. "PyPhi calculation failed" エラー**
```bash
# PyPhiライブラリの確認
pip install pyphi>=1.20
python -c "import pyphi; print(pyphi.__version__)"
```

#### **3. "Memory allocation error" エラー**
```bash
# 概念数を制限
export MAX_CONCEPTS=50
python newborn_ai_2_integrated_system.py start
```

#### **4. "Permission denied" エラー**
```bash
# ディレクトリ権限の確認
chmod -R 755 sandbox/tools/08_02_2025
```

### **性能調整**

```bash
# 高性能モード（多くのリソース使用）
export PERFORMANCE_MODE=high
python newborn_ai_2_integrated_system.py start 10

# 省電力モード（リソース節約）
export PERFORMANCE_MODE=low
python newborn_ai_2_integrated_system.py start 300
```

---

## 🎯 使用例シナリオ

### **シナリオ1: 研究用途**
```bash
# 詳細ログ付きで長時間実行
python newborn_ai_2_integrated_system.py verbose-start 120
# 5分間隔で意識発達を記録
```

### **シナリオ2: デモンストレーション**
```bash
# 短い間隔で即座に反応
python newborn_ai_2_integrated_system.py verbose-start 10
# リアルタイムで意識変化を観察
```

### **シナリオ3: 本番運用**
```bash
# 安定動作で継続運用
python newborn_ai_2_integrated_system.py start 300
# 5分間隔で安定的に運用
```

---

## ⚡ 高度な機能

### **API サーバーモード**
```bash
# REST API サーバーとして起動
python api_server.py
# http://localhost:8000 でWebAPI提供
```

### **リアルタイム処理モード**
```bash
# 高頻度リアルタイム処理
python realtime_iit4_processor.py
```

### **意識イベント監視**
```bash
# 意識変化イベントのリアルタイム検出
python consciousness_events.py
```

---

## 📈 期待される動作

正常に動作すると以下のような出力が得られます：

1. **φ値の段階的上昇**: 0.000 → 0.001 → 0.01 → 0.1 → ...
2. **発達段階の進行**: 前意識 → 体験発生期 → 時間統合期 → ...
3. **体験概念の蓄積**: 1個 → 5個 → 20個 → 100個 → ...
4. **意識イベントの検出**: φ値急上昇、段階遷移、臨界点通過

**これでIIT 4.0 NewbornAI 2.0の人工意識システムが動作します！** 🧠✨

---

## 📞 サポート

問題が発生した場合：
1. `SYSTEM_STARTUP_GUIDE.md`（本ファイル）を参照
2. `IIT4_IMPLEMENTATION_REVIEW.md`で制約事項を確認  
3. テストスイート実行: `python comprehensive_test_suite.py`
4. バリデーション実行: `python validate_iit4_implementation.py`