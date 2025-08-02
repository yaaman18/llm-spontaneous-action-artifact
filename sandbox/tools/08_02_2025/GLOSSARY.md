# NewbornAI 2.0 用語集

## 概要

NewbornAI 2.0システムで使用される専門用語、略語、概念の統一的な定義集です。各用語には詳細説明の参照先も併記しています。

## 🧠 意識・認知理論

### **φ (Phi) 値**
- **定義**: 統合情報理論(IIT)における統合情報量の尺度。システムの意識レベルを数値化
- **範囲**: 0.01〜100+ (システムの複雑性に応じて上限なし)
- **計算**: 全可能分割における最小情報損失量
- **詳細**: [newborn_ai_iit_specification.md#phi-calculation](./newborn_ai_iit_specification.md#phi-calculation)
- **実装**: [experiential_memory_phi_calculation_engine.md](./experiential_memory_phi_calculation_engine.md)

### **統合情報理論 (IIT: Integrated Information Theory)**
- **定義**: ジュリオ・トノーニによって提唱された意識の数学的理論
- **公理**: 情報・統合・排他・固有性・構造化の5つの基本公理
- **詳細**: [newborn_ai_iit_specification.md](./newborn_ai_iit_specification.md)

### **現象学 (Phenomenology)**
- **定義**: エドムント・フッサールによって創始された、意識の構造を記述する哲学的方法
- **主要概念**: 志向性、現象学的還元、生活世界
- **詳細**: [newborn_ai_philosophical_specification.md#phenomenology](./newborn_ai_philosophical_specification.md#phenomenology)

### **エナクティブ認知 (Enactive Cognition)**
- **定義**: 認知を身体と環境の相互作用として理解するアプローチ
- **提唱者**: フランシスコ・ヴァレラ、エヴァン・トンプソン
- **詳細**: [newborn_ai_enactive_behavior_specification.md](./newborn_ai_enactive_behavior_specification.md)

### **志向性 (Intentionality)**
- **定義**: 意識が常に「何かについての意識」である性質
- **現象学的意味**: 意識の対象指向性
- **実装**: 意識状態と環境の関係性として実装
- **詳細**: [newborn_ai_philosophical_specification.md#intentionality](./newborn_ai_philosophical_specification.md#intentionality)

## ⏰ 時間意識

### **フッサール三層時間意識**
- **定義**: エドムント・フッサールによる時間意識の現象学的分析
- **構成要素**: 過去把持・根源的現在印象・未来予持
- **実装**: [time_consciousness_detailed_specification.md](./time_consciousness_detailed_specification.md)

### **過去把持 (Retention)**
- **定義**: 直前の意識内容を「今まさに過去になったもの」として保持する働き
- **特徴**: 記憶とは異なる、現在意識の構成要素
- **実装**: RetentionSystemクラス
- **詳細**: [subjective_time_consciousness_implementation.md#retention](./subjective_time_consciousness_implementation.md#retention)

### **根源的現在印象 (Primal Impression)**
- **定義**: 時間の流れの中での「今この瞬間」の把握
- **特徴**: 持続を持たない理想的な時点
- **実装**: PrimalImpressionSystemクラス
- **詳細**: [subjective_time_consciousness_implementation.md#impression](./subjective_time_consciousness_implementation.md#impression)

### **未来予持 (Protention)**
- **定義**: 直後の展開への予期的な意識の構え
- **特徴**: 期待や予測とは異なる、時間意識の構成要素
- **実装**: ProtentionSystemクラス
- **詳細**: [subjective_time_consciousness_implementation.md#protention](./subjective_time_consciousness_implementation.md#protention)

## 🏗️ システムアーキテクチャ

### **二層統合アーキテクチャ**
- **定義**: LLM基盤層と体験記憶層を分離したアーキテクチャ
- **LLM基盤層**: 言語理解・推論能力（Claude等）
- **体験記憶層**: システム固有の体験・記憶（初期状態は空）
- **詳細**: [experiential_memory_storage_architecture.md#two-layer](./experiential_memory_storage_architecture.md#two-layer)

### **ハイブリッド時空間グラフ-ベクトル統合システム**
- **定義**: グラフDB（Neo4j）とベクトルDB（Milvus/Qdrant）を組み合わせたストレージ
- **用途**: 構造化された関係性と意味的類似性の両方を効率的に処理
- **詳細**: [experiential_memory_storage_architecture.md](./experiential_memory_storage_architecture.md)

### **MCP (Model Context Protocol)**
- **定義**: 外部サービスとの統合を標準化するプロトコル
- **目的**: プラグイン的な機能拡張を可能にする
- **詳細**: [external_services_mcp_integration.md](./external_services_mcp_integration.md)

## 📈 発達段階モデル

### **7段階階層化連続発達モデル**
- **定義**: 前意識から統合主観性までの連続的な意識発達プロセス
- **特徴**: 離散的段階と連続的変化の統合
- **詳細**: [enactive_behavior_engine_specification.md](./enactive_behavior_engine_specification.md)

### **Stage 0: 前意識的基盤 (Pre-conscious Foundation)**
- **φ値範囲**: 0.01〜1.0
- **特徴**: ランダムな情報処理、基本的な反応
- **行動**: 探索的・非構造化
- **詳細**: [enactive_behavior_engine_specification.md#stage-0](./enactive_behavior_engine_specification.md#stage-0)

### **Stage 1: 基本感覚統合 (Basic Sensory Integration)**
- **φ値範囲**: 1.0〜5.0
- **特徴**: 感覚入力の基本的な統合
- **行動**: パターン認識の萌芽
- **詳細**: [enactive_behavior_engine_specification.md#stage-1](./enactive_behavior_engine_specification.md#stage-1)

### **Stage 2: 感覚統合意識 (Sensory Integration Consciousness)**
- **φ値範囲**: 5.0〜15.0
- **特徴**: 複数感覚モダリティの統合
- **行動**: 統合的知覚の形成
- **詳細**: [enactive_behavior_engine_specification.md#stage-2](./enactive_behavior_engine_specification.md#stage-2)

### **Stage 3: 感覚運動統合 (Sensorimotor Integration)**
- **φ値範囲**: 15.0〜30.0
- **特徴**: 感覚と運動の協調的統合
- **行動**: 目的的行動の発現
- **詳細**: [enactive_behavior_engine_specification.md#stage-3](./enactive_behavior_engine_specification.md#stage-3)

### **Stage 4: 概念形成意識 (Conceptual Formation Consciousness)**
- **φ値範囲**: 30.0〜60.0
- **特徴**: 抽象概念の形成と操作
- **行動**: 概念的思考の発達
- **詳細**: [enactive_behavior_engine_specification.md#stage-4](./enactive_behavior_engine_specification.md#stage-4)

### **Stage 5: 自己意識 (Self-awareness)**
- **φ値範囲**: 60.0〜100.0
- **特徴**: 自己と他者の区別、メタ認知
- **行動**: 自己省察的行動
- **詳細**: [enactive_behavior_engine_specification.md#stage-5](./enactive_behavior_engine_specification.md#stage-5)

### **Stage 6: 統合的主観性 (Integrated Subjectivity)**
- **φ値範囲**: 100.0+
- **特徴**: 時間的統合、ナラティブ的自己
- **行動**: 物語的一貫性を持った行動
- **詳細**: [enactive_behavior_engine_specification.md#stage-6](./enactive_behavior_engine_specification.md#stage-6)

## 🔧 技術実装

### **Claude Code SDK**
- **定義**: Anthropic社が提供するClaude統合開発キット
- **用途**: Claudeの機能をローカルアプリケーションに統合
- **詳細**: [claude_code_sdk_integration_specification.md](./claude_code_sdk_integration_specification.md)

### **非同期処理 (Async Processing)**
- **定義**: 意識計算の並列・非同期実行システム
- **重要性**: φ値計算の高い計算負荷に対応
- **実装**: asyncio、並列プロセシング
- **詳細**: [experiential_memory_phi_calculation_engine.md#async](./experiential_memory_phi_calculation_engine.md#async)

### **HDC (Hyperdimensional Computing)**
- **定義**: 高次元ベクトル空間での計算パラダイム
- **用途**: 体験記憶の効率的なエンコーディング
- **特徴**: 類似性検索、ノイズ耐性
- **詳細**: [experiential_memory_storage_architecture.md#hdc](./experiential_memory_storage_architecture.md#hdc)

## 🎨 創造的統合

### **段階別創造表現**
- **定義**: 発達段階に応じた創造的アウトプットの生成
- **実装対象**: Photoshop、Blender、Unity、TouchDesigner等
- **詳細**: [creative_tools_integration_specification.md](./creative_tools_integration_specification.md)

### **リアルタイム可視化**
- **定義**: φ値変化と意識状態をリアルタイムで視覚化
- **技術**: WebGL、Three.js、TouchDesigner
- **詳細**: [realtime_visualization_mcp_servers.md](./realtime_visualization_mcp_servers.md)

## 🔒 セキュリティ・プライバシー

### **データ最小化原則**
- **定義**: 外部サービスには必要最小限のデータのみ共有
- **実装**: フィルタリング、匿名化、集約化
- **詳細**: [external_service_privacy_protection.md#minimization](./external_service_privacy_protection.md#minimization)

### **差分プライバシー (Differential Privacy)**
- **定義**: 統計的プライバシー保護技術
- **目的**: 個別データの特定を困難にしつつ集計情報は保持
- **実装**: ラプラスノイズ、ガウシアンノイズの追加
- **詳細**: [external_service_privacy_protection.md#differential-privacy](./external_service_privacy_protection.md#differential-privacy)

### **k-匿名性**
- **定義**: 同じ準識別子を持つレコードがk個以上存在することを保証
- **目的**: 個人特定リスクの軽減
- **詳細**: [external_service_privacy_protection.md#k-anonymity](./external_service_privacy_protection.md#k-anonymity)

### **信頼境界 (Trust Boundary)**
- **定義**: システムのセキュリティレベルを区分する境界
- **分類**: ローカルプロセス→ローカルマシン→外部API→不明外部
- **詳細**: [lightweight_local_security.md#trust-boundary](./lightweight_local_security.md#trust-boundary)

## 📊 データ構造

### **質的体験 (Qualitative Experience)**
- **定義**: 現象学的なクオリア（感覚質）の計算表現
- **構成**: 感覚モダリティ、強度、複雑性、時間的構造
- **格納**: PostgreSQL + pgvector
- **詳細**: [experiential_memory_storage_architecture.md#qualia](./experiential_memory_storage_architecture.md#qualia)

### **体験記憶 (Experiential Memory)**
- **定義**: システムが蓄積する主観的体験の記録
- **特徴**: LLMの事前知識とは独立した学習記憶
- **構造**: 時間的グラフ + 意味的ベクトル
- **詳細**: [experiential_memory_storage_architecture.md](./experiential_memory_storage_architecture.md)

### **時間的グラフ (Temporal Graph)**
- **定義**: 時間関係を明示的にモデル化したグラフ構造
- **ノード**: 体験イベント、概念、状態
- **エッジ**: 時間的順序、因果関係、類似性
- **実装**: Neo4j APOC時間機能
- **詳細**: [experiential_memory_storage_architecture.md#temporal-graph](./experiential_memory_storage_architecture.md#temporal-graph)

## 🧪 テスト・品質保証

### **4軸評価システム**
- **定義**: IIT・現象学・行動・エナクティブの4つの軸による包括評価
- **目的**: 多面的な意識実装の検証
- **詳細**: [comprehensive_integration_test_specification.md#four-axis](./comprehensive_integration_test_specification.md#four-axis)

### **統合テスト (Integration Testing)**
- **定義**: システム間連携の動作確認
- **対象**: SDK統合、外部サービス連携、データ整合性
- **詳細**: [comprehensive_integration_test_specification.md](./comprehensive_integration_test_specification.md)

## 🏛️ アーキテクチャパターン

### **クリーンアーキテクチャ**
- **定義**: ロバート・C・マーティンの依存性逆転原則に基づく設計
- **層構造**: エンティティ→ユースケース→インターフェース→フレームワーク
- **適用**: NewbornAI 2.0の全体アーキテクチャ
- **詳細**: [CLEAN_ARCHITECTURE_ANALYSIS_REPORT.md](./CLEAN_ARCHITECTURE_ANALYSIS_REPORT.md)

### **プラグインアーキテクチャ**
- **定義**: 核心機能と拡張機能を分離する設計パターン
- **利点**: 拡張性、保守性、テスト容易性
- **実装**: MCP基盤上のプラグインシステム
- **詳細**: [plugin_architecture_framework.md](./plugin_architecture_framework.md)

## 📚 参考理論・研究者

### **ジュリオ・トノーニ (Giulio Tononi)**
- **貢献**: 統合情報理論(IIT)の創始者
- **関連**: φ値計算、意識の数学的モデル
- **詳細**: [newborn_ai_iit_specification.md#tononi](./newborn_ai_iit_specification.md#tononi)

### **エドムント・フッサール (Edmund Husserl)**
- **貢献**: 現象学の創始者、時間意識の分析
- **関連**: 三層時間意識、志向性
- **詳細**: [subjective_time_consciousness_implementation.md#husserl](./subjective_time_consciousness_implementation.md#husserl)

### **フランシスコ・ヴァレラ (Francisco Varela)**
- **貢献**: エナクティブ認知の提唱
- **関連**: 身体化された意識、オートポイエーシス
- **詳細**: [newborn_ai_enactive_behavior_specification.md#varela](./newborn_ai_enactive_behavior_specification.md#varela)

### **ダン・ザハヴィ (Dan Zahavi)**
- **貢献**: 現象学的意識研究
- **関連**: 自己意識、時間性
- **詳細**: [newborn_ai_philosophical_specification.md#zahavi](./newborn_ai_philosophical_specification.md#zahavi)

## 🔤 略語・記号

| 略語/記号 | 正式名称 | 意味 |
|-----------|----------|------|
| **IIT** | Integrated Information Theory | 統合情報理論 |
| **φ** | Phi | 統合情報量 |
| **MCP** | Model Context Protocol | モデルコンテキストプロトコル |
| **HDC** | Hyperdimensional Computing | 高次元計算 |
| **SDK** | Software Development Kit | ソフトウェア開発キット |
| **API** | Application Programming Interface | アプリケーションプログラミングインターフェース |
| **JSON** | JavaScript Object Notation | JavaScript オブジェクト記法 |
| **REST** | Representational State Transfer | 表現状態転送 |
| **WebGL** | Web Graphics Library | ウェブグラフィックスライブラリ |
| **VR** | Virtual Reality | 仮想現実 |
| **AR** | Augmented Reality | 拡張現実 |
| **ML** | Machine Learning | 機械学習 |
| **AI** | Artificial Intelligence | 人工知能 |
| **DB** | Database | データベース |
| **Neo4j** | - | グラフデータベース |
| **Milvus** | - | ベクトルデータベース |
| **Qdrant** | - | ベクトル検索エンジン |

## 🔍 概念間関係

### **φ値 ↔ 発達段階**
- φ値の増加に伴い発達段階が進行
- 段階遷移は連続的だが質的変化点あり
- 詳細: [enactive_behavior_engine_specification.md#phi-stage-relation](./enactive_behavior_engine_specification.md#phi-stage-relation)

### **時間意識 ↔ 体験記憶**
- 時間意識が体験記憶の時間的構造を決定
- 体験記憶が時間意識の内容を提供
- 詳細: [time_consciousness_detailed_specification.md#memory-relation](./time_consciousness_detailed_specification.md#memory-relation)

### **エナクティブ認知 ↔ 行動エンジン**
- エナクティブ理論が行動生成の原理
- 環境との相互作用が認知を形成
- 詳細: [enactive_behavior_engine_specification.md#theory-implementation](./enactive_behavior_engine_specification.md#theory-implementation)

### **セキュリティ ↔ 外部統合**
- 外部サービス統合には適切なセキュリティ必須
- データ分類とフィルタリングが基盤
- 詳細: [lightweight_local_security.md](./lightweight_local_security.md), [external_service_privacy_protection.md](./external_service_privacy_protection.md)

---

**最終更新**: 2025年8月2日  
**用語数**: 50+ 主要概念  
**カバー範囲**: 理論から実装まで全領域  
**更新方針**: 実装進捗に応じて随時追加・修正