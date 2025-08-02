# Omoikane Lab - 新機能使用ガイド

## 📖 目次

1. [システム概要](#システム概要)
2. [クイックスタート](#クイックスタート)
3. [ハルシネーション検出の使い方](#ハルシネーション検出の使い方)
4. [専門家検証システムの使い方](#専門家検証システムの使い方)
5. [リアルタイム検証ダッシュボード](#リアルタイム検証ダッシュボード)
6. [知識グラフシステム](#知識グラフシステム)
7. [新エージェントとの協働](#新エージェントとの協働)
8. [高度な使用例](#高度な使用例)
9. [トラブルシューティング](#トラブルシューティング)

---

## システム概要

新しく構築された深層知識検証・ハルシネーション検出システムは、以下の革新的機能を提供します：

### 🔍 主要機能
- **セマンティックエントロピー法**による高精度ハルシネーション検出
- **分野専門家システム**による深層知識検証
- **専門家コンセンサス**による多角的評価
- **RAG統合**による外部知識ベース活用
- **Neo4j知識グラフ**による関係性分析
- **リアルタイム検証**によるインタラクティブな使用体験

---

## クイックスタート

### 1. システム起動

```bash
# 1. 依存関係インストール
cd /Users/yamaguchimitsuyuki/omoikane-lab
pip install -r institute/systems/hallucination_detection/requirements.txt

# 2. Neo4j起動（知識グラフ用）
cd institute/systems/knowledge_graph/
docker-compose up -d neo4j

# 3. リアルタイム検証サーバー起動
cd ../realtime_verification/
python api_server.py
```

### 2. ダッシュボードアクセス

ブラウザで以下のURLにアクセス：
```
http://localhost:8000/static/dashboard.html
```

### 3. 基本的な検証

ダッシュボードで以下を試してください：

**例1: 正確な情報**
```
検証対象: 統合情報理論は意識研究の重要な理論フレームワークである
分野: 意識研究
検証レベル: 中程度
```

**例2: ハルシネーション検出**
```
検証対象: IITによると、Φ値が42以上なら確実に意識がある
分野: 意識研究  
検証レベル: 専門家レベル
```

---

## ハルシネーション検出の使い方

### Python APIでの使用

```python
import asyncio
from institute.systems.hallucination_detection.core import HallucinationDetectionEngine

async def detect_hallucination():
    # エージェント設定読み込み
    agents_config = {
        "hallucination-detector": {...},  # YAML設定から読み込み
        "fact-checker": {...}
    }
    
    # 検出エンジン初期化
    detector = HallucinationDetectionEngine(agents_config)
    
    # 検証実行
    result = await detector.detect_hallucination(
        statement="量子もつれが直接意識を生み出すことが証明されている",
        context="意識理論の議論中",
        domain_hint="consciousness"
    )
    
    # 結果分析
    print(f"ハルシネーション検出: {result.is_hallucination}")
    print(f"信頼度: {result.confidence_score:.2%}")
    print(f"セマンティックエントロピー: {result.semantic_entropy:.3f}")
    print(f"検出根拠: {result.evidence}")
    
    if result.is_hallucination:
        print("⚠️ この文には問題がある可能性があります")
        print("推奨修正:")
        for correction in result.corrected_text or []:
            print(f"  - {correction}")

# 実行
asyncio.run(detect_hallucination())
```

### 検出結果の解釈

| セマンティックエントロピー | 判定 | 対応 |
|-------------------------|-----|-----|
| 0.0 - 0.3 | 低リスク | 問題なし |
| 0.3 - 0.5 | 中リスク | 要注意 |
| 0.5 - 0.7 | 高リスク | 検証推奨 |
| 0.7 - 1.0 | 極高リスク | 修正必要 |

---

## 専門家検証システムの使い方

### 1. 分野専門家による検証

```python
from institute.systems.knowledge_verification.domain_specialists import (
    DomainSpecialistFactory, VerificationLevel
)

async def expert_verification():
    # 意識研究専門家
    consciousness_expert = DomainSpecialistFactory.create_specialist('consciousness')
    
    result = await consciousness_expert.verify_statement(
        statement="現象学における時間意識は、把持・原印象・予持の三重構造を持つ",
        context="フッサール現象学の議論",
        verification_level=VerificationLevel.EXPERT
    )
    
    print(f"検証結果: {'有効' if result.is_valid else '無効'}")
    print(f"信頼度: {result.confidence_score:.2%}")
    print(f"専門家所見:")
    print(result.specialist_notes)
    
    if result.corrections:
        print("推奨修正:")
        for correction in result.corrections:
            print(f"  - {correction}")

asyncio.run(expert_verification())
```

### 2. 利用可能な専門家

| 専門家 | 専門分野 | 得意領域 |
|-------|---------|---------|
| `consciousness` | 意識研究 | IIT、GWT、現象学、クオリア |
| `philosophy` | 哲学 | 存在論、認識論、心身問題 | 
| `mathematics` | 数学 | 証明、論理、計算理論 |

### 3. 検証レベルの選択

```python
# 検証レベル別の使い分け
levels = {
    VerificationLevel.SURFACE: "基本的なキーワードチェック",
    VerificationLevel.SHALLOW: "概念的整合性の確認", 
    VerificationLevel.MODERATE: "専門知識による検証",
    VerificationLevel.DEEP: "詳細な理論的分析",
    VerificationLevel.EXPERT: "最高レベルの専門検証"
}
```

---

## リアルタイム検証ダッシュボード

### 1. ダッシュボードの機能

#### 基本画面構成
- **システム状態表示**: 接続状況とシステム稼働状況
- **統計情報**: 検証実績、ハルシネーション検出率、処理性能
- **検証フォーム**: 文の入力と検証パラメータ設定
- **結果表示**: リアルタイム結果とビジュアル表示

#### 検証フォームの使い方

**必須項目:**
- **検証対象の文**: 検証したい文章を入力

**オプション項目:**
- **文脈情報**: 文の背景や状況を説明
- **分野ヒント**: 自動判定 / 意識研究 / 哲学 / 数学 / 神経科学
- **検証レベル**: 表面的 〜 専門家レベル（5段階）
- **専門家コンセンサス**: 複数専門家による合意形成の要否

### 2. 結果の見方

#### 検証結果カード
```
✅ 有効 / ❌ 無効                    信頼度: 85%
検証文: 統合情報理論は意識研究の重要な理論である

⚠️ ハルシネーションが検出されました（該当する場合）

💡 推奨事項:
  - より具体的な文脈を提供してください
  - 専門用語の定義を明確にしてください
```

#### 統計情報
- **総検証数**: これまでの検証実行回数
- **ハルシネーション検出率**: 検出された割合
- **平均処理時間**: システムの応答性能
- **コンセンサス形成率**: 専門家が合意に達した割合

### 3. WebSocket接続での使用

```javascript
// JavaScript での WebSocket 使用例
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function() {
    console.log('WebSocket接続確立');
};

// 検証リクエスト送信
const request = {
    type: 'verify_request',
    data: {
        statement: '人工意識は2030年までに実現可能である',
        context: '技術予測の議論',
        domain_hint: 'consciousness',
        verification_level: 'expert',
        require_consensus: true
    }
};

ws.send(JSON.stringify(request));

// 結果受信
ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    if (message.type === 'verification_result') {
        const result = message.data;
        console.log(`検証結果: ${result.is_valid ? '有効' : '無効'}`);
        console.log(`信頼度: ${result.confidence_score}`);
        console.log(`処理時間: ${result.processing_time}秒`);
    }
};
```

---

## 知識グラフシステム

### 1. Neo4j知識グラフの活用

```python
from institute.systems.knowledge_graph.neo4j_manager import Neo4jKnowledgeGraph

async def use_knowledge_graph():
    # 知識グラフ初期化
    kg = Neo4jKnowledgeGraph()
    await kg.initialize()
    
    # 関連概念検索
    related = await kg.find_related_concepts(
        "consciousness", 
        max_depth=2,
        relation_types=[RelationType.SUPPORTS, RelationType.RELATED_TO]
    )
    
    print("意識に関連する概念:")
    for concept in related['related_concepts']:
        print(f"  - {concept['name']} (距離: {concept['distance']})")
    
    # 矛盾検出
    contradictions = await kg.detect_contradictions("consciousness_studies")
    
    if contradictions:
        print("検出された矛盾:")
        for contradiction in contradictions:
            print(f"  {contradiction['concept1']} ⚡ {contradiction['concept2']}")
    
    # 知識ギャップ特定
    gaps = await kg.find_knowledge_gaps("consciousness_studies")
    
    if gaps:
        print("知識ギャップ:")
        for gap in gaps:
            print(f"  {gap['gap_type']}: {gap.get('suggestion', '')}")

asyncio.run(use_knowledge_graph())
```

### 2. 知識グラフの構築

```python
from institute.systems.knowledge_graph.neo4j_manager import KnowledgeGraphBuilder

async def build_knowledge_base():
    # 知識グラフビルダー
    builder = KnowledgeGraphBuilder(kg)
    
    # 論文から知識グラフを構築
    await builder.build_from_papers(
        "institute/tools/paper-collector/ramstead_consciousness_test/markdown"
    )
    
    # 研究者ネットワーク構築
    await builder.build_researcher_network("institute/agents")
    
    print("知識グラフ構築完了")

asyncio.run(build_knowledge_base())
```

---

## 新エージェントとの協働

### 1. ハルシネーション検出専門家 (Dr. Sarah Chen)

```bash
# エージェント起動例（Claude Codeでの使用）
"Dr. Sarah Chen（ハルシネーション検出専門家）として、
institute/agents/hallucination-detector.yamlの設定に従って活動してください。

以下の文に対してセマンティックエントロピー法による詳細検証を実行してください：
'統合情報理論では、Φ値が100を超えると超意識状態になる'

検証プロセス:
1. セマンティック一貫性分析
2. 統計的異常検知
3. IIT理論との整合性確認
4. 信頼度スコア算出
5. 修正提案の作成"
```

**期待される応答例:**
```
【ハルシネーション検出分析】

🔍 セマンティックエントロピー分析:
- エントロピー値: 0.85 (高リスク)
- 一貫性スコア: 0.23 (低)

⚠️ 検出された問題:
1. 「Φ値が100を超える」- IITではΦ値に上限値の概念なし
2. 「超意識状態」- 理論的に未定義の概念
3. 具体的数値の誤用

🎯 修正提案:
「統合情報理論では、Φ値が高いほど意識の統合度が高いとされる」

信頼度: 92% (ハルシネーション確実)
```

### 2. ファクトチェック専門家 (Dr. Michael Thompson)

```bash
"Dr. Michael Thompson（ファクトチェック専門家）として、
以下の主張について多角的ファクトチェックを実行してください：

'2024年の研究で、人工意識の実現に必要な計算量が確定した'

検証項目:
1. 2024年の関連研究調査
2. 情報源の信頼性評価  
3. 計算量に関する主張の検証
4. 専門家コンセンサスの確認
5. 最終判定と根拠"
```

### 3. メタ検証統括者 (Dr. Elena Rodriguez)

```bash
"Dr. Elena Rodriguez（メタ検証統括者）として、
以下の検証プロセス全体を監督し、品質保証を実行してください：

対象: '現象学的還元により、意識の本質が数学的に記述可能になる'

統括業務:
1. 各専門家の検証結果統合
2. 品質メトリクス評価
3. 専門家間の意見調整
4. 最終品質判定
5. システム改善提案"
```

---

## 高度な使用例

### 1. 複数文の一括検証

```python
async def batch_verification():
    system = RealtimeVerificationSystem()
    await system.initialize()
    
    statements = [
        "意識は脳の神経活動から創発する",
        "量子もつれが意識の統一性を説明する",
        "人工知能は2030年に人間レベルに達する",
        "現象学は科学的方法論である",
        "自由意志は決定論と両立可能である"
    ]
    
    results = []
    for statement in statements:
        request = VerificationRequest(
            statement=statement,
            verification_level="deep",
            require_consensus=True
        )
        
        result = await system.verify_statement(request)
        results.append({
            'statement': statement,
            'valid': result.is_valid,
            'confidence': result.confidence_score,
            'hallucination': result.hallucination_detected
        })
    
    # 統計分析
    valid_rate = sum(1 for r in results if r['valid']) / len(results)
    hallucination_rate = sum(1 for r in results if r['hallucination']) / len(results)
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    print(f"検証結果統計:")
    print(f"有効率: {valid_rate:.1%}")
    print(f"ハルシネーション検出率: {hallucination_rate:.1%}")
    print(f"平均信頼度: {avg_confidence:.1%}")

asyncio.run(batch_verification())
```

### 2. 学術論文の自動検証

```python
async def verify_academic_paper():
    # 論文の主要主張を抽出（実装例）
    paper_claims = [
        "本研究では、新しいIITアルゴリズムを提案する",
        "提案手法により、Φ値計算が10倍高速化された", 
        "実験結果は統計的に有意である（p<0.001）",
        "この成果は意識研究に革命をもたらす"
    ]
    
    paper_verification = []
    
    for claim in paper_claims:
        # 専門家による詳細検証
        consciousness_expert = DomainSpecialistFactory.create_specialist('consciousness')
        math_expert = DomainSpecialistFactory.create_specialist('mathematics')
        
        # 並列検証
        results = await asyncio.gather(
            consciousness_expert.verify_statement(claim, verification_level=VerificationLevel.EXPERT),
            math_expert.verify_statement(claim, verification_level=VerificationLevel.EXPERT)
        )
        
        # コンセンサス形成
        consensus_engine = ConsensusEngine()
        # consensus = await consensus_engine.form_consensus(claim, expert_opinions)
        
        paper_verification.append({
            'claim': claim,
            'consciousness_valid': results[0].is_valid,
            'math_valid': results[1].is_valid,
            'overall_confidence': (results[0].confidence_score + results[1].confidence_score) / 2
        })
    
    # 論文全体の信頼度評価
    overall_validity = all(v['consciousness_valid'] and v['math_valid'] for v in paper_verification)
    print(f"論文全体の妥当性: {'有効' if overall_validity else '要検討'}")

asyncio.run(verify_academic_paper())
```

### 3. リアルタイム議論支援

```python
class DebateSupport:
    def __init__(self):
        self.verification_system = None
        self.conversation_history = []
    
    async def initialize(self):
        self.verification_system = RealtimeVerificationSystem()
        await self.verification_system.initialize()
    
    async def analyze_statement(self, speaker: str, statement: str, context: str = ""):
        # リアルタイム検証
        request = VerificationRequest(
            statement=statement,
            context=f"{context}\n発言者: {speaker}",
            verification_level="moderate",
            require_consensus=True
        )
        
        result = await self.verification_system.verify_statement(request)
        
        # 議論履歴に追加
        self.conversation_history.append({
            'speaker': speaker,
            'statement': statement,
            'verification': result,
            'timestamp': datetime.now()
        })
        
        # 即座フィードバック
        if result.hallucination_detected:
            print(f"⚠️ {speaker}の発言に不正確な情報が含まれている可能性があります")
            
        if result.recommendations:
            print(f"💡 推奨事項:")
            for rec in result.recommendations:
                print(f"  - {rec}")
        
        return result
    
    def get_discussion_summary(self):
        if not self.conversation_history:
            return "議論履歴がありません"
        
        total_statements = len(self.conversation_history)
        problematic_statements = sum(1 for h in self.conversation_history 
                                   if h['verification'].hallucination_detected)
        
        return f"""
議論品質サマリー:
- 総発言数: {total_statements}
- 問題のある発言: {problematic_statements}
- 信頼性: {((total_statements - problematic_statements) / total_statements * 100):.1f}%
        """

# 使用例
async def support_debate():
    support = DebateSupport()
    await support.initialize()
    
    # 議論をシミュレート
    await support.analyze_statement(
        "研究者A", 
        "IITによると、意識レベルはΦ値で完全に決定される",
        "意識理論に関する学術議論"
    )
    
    await support.analyze_statement(
        "研究者B",
        "それは過度な単純化です。IITはΦ値を重要視しますが、他の要因も考慮すべきです",
        "意識理論に関する学術議論"
    )
    
    print(support.get_discussion_summary())

asyncio.run(support_debate())
```

---

## トラブルシューティング

### 1. よくある問題と解決策

#### 問題: "Could not connect to Neo4j"
```bash
# 解決策1: Neo4jコンテナの状態確認
docker ps | grep neo4j

# 解決策2: Neo4j再起動
cd institute/systems/knowledge_graph/
docker-compose restart neo4j

# 解決策3: ポート確認
netstat -an | grep 7687
```

#### 問題: "Module not found" エラー
```bash
# 解決策: パス設定の確認
export PYTHONPATH="${PYTHONPATH}:/Users/yamaguchimitsuyuki/omoikane-lab/institute/systems"

# または、__init__.py ファイル作成
touch institute/systems/__init__.py
touch institute/systems/hallucination_detection/__init__.py
```

#### 問題: WebSocket接続エラー
```bash
# 解決策1: サーバー起動確認
curl http://localhost:8000/status

# 解決策2: ファイアウォール確認
sudo lsof -i :8000

# 解決策3: ブラウザキャッシュクリア
```

### 2. デバッグモードでの実行

```python
import logging

# ログレベル設定
logging.basicConfig(level=logging.DEBUG)

# 詳細ログ出力での実行
async def debug_verification():
    system = RealtimeVerificationSystem()
    await system.initialize()
    
    # デバッグ情報付きで検証
    result = await system.verify_statement(VerificationRequest(
        statement="テスト文",
        verification_level="deep"
    ))
    
    # システム状態確認
    status = system.get_system_status()
    print(f"システム状態: {status}")

asyncio.run(debug_verification())
```

### 3. パフォーマンス最適化

```python
# 並列処理数の調整
import asyncio

# セマフォによる同期制御
semaphore = asyncio.Semaphore(5)  # 最大5並列

async def optimized_batch_processing(statements):
    async def process_with_limit(statement):
        async with semaphore:
            return await verify_statement(statement)
    
    # 並列実行
    tasks = [process_with_limit(stmt) for stmt in statements]
    results = await asyncio.gather(*tasks)
    return results
```

### 4. カスタム設定

```python
# カスタム検証設定
custom_config = {
    "hallucination_threshold": 0.6,  # デフォルト: 0.5
    "consensus_requirement": 0.75,   # デフォルト: 0.7
    "timeout_seconds": 30,           # デフォルト: 120
    "max_parallel_requests": 10      # デフォルト: 5
}

# 設定適用
system = RealtimeVerificationSystem()
system.update_config(custom_config)
```

---

## 📞 サポート・問い合わせ

### ヘルプ・リソース
- **システムドキュメント**: `institute/systems/README.md`
- **API仕様**: `http://localhost:8000/docs` (FastAPI自動生成)
- **統合テスト**: `python institute/systems/integration_test.py`

### 機能リクエスト・バグレポート
新機能の追加やバグの報告は、以下の手順で行ってください：

1. **統合テスト実行**: 問題の再現確認
2. **ログ収集**: デバッグ情報の保存
3. **詳細報告**: 期待動作と実際の動作の明記

---

**Last Updated**: 2025年7月30日  
**Version**: 1.0.0

このガイドを活用して、omoikane-labの新システムを最大限にご活用ください！🚀