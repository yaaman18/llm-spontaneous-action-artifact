# Omoikane Lab - API リファレンス

## 📚 目次

1. [API概要](#api概要)
2. [認証・セットアップ](#認証セットアップ)
3. [REST API](#rest-api)
4. [WebSocket API](#websocket-api)
5. [Python SDK](#python-sdk)
6. [レスポンス形式](#レスポンス形式)
7. [エラーハンドリング](#エラーハンドリング)
8. [使用例](#使用例)

---

## API概要

Omoikane Lab の深層知識検証・ハルシネーション検出システムは、以下のAPIインターフェースを提供します：

### エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|---------|---------------|------|
| `GET` | `/` | API状態確認 |
| `GET` | `/status` | システム状態取得 |
| `POST` | `/verify` | 文の検証実行 |
| `GET` | `/history` | 検証履歴取得 |
| `GET` | `/stats` | 統計情報取得 |
| `WebSocket` | `/ws` | リアルタイム検証 |

### ベースURL
```
http://localhost:8000
```

---

## 認証・セットアップ

### 1. サーバー起動

```bash
cd /Users/yamaguchimitsuyuki/omoikane-lab/institute/systems/realtime_verification
python api_server.py
```

### 2. ヘルスチェック

```bash
curl http://localhost:8000/
```

**レスポンス例:**
```json
{
  "message": "Omoikane Lab Realtime Verification API",
  "status": "running"
}
```

---

## REST API

### 1. システム状態取得

**エンドポイント:** `GET /status`

```bash
curl http://localhost:8000/status
```

**レスポンス:**
```json
{
  "system_initialized": true,
  "verification_stats": {
    "total_verifications": 42,
    "hallucinations_detected": 8,
    "average_processing_time": 1.23,
    "consensus_achieved": 35
  },
  "active_modules": {
    "hallucination_detector": true,
    "consensus_engine": true,
    "rag_integration": true
  }
}
```

### 2. 文の検証実行

**エンドポイント:** `POST /verify`

**リクエスト形式:**
```json
{
  "statement": "統合情報理論では、意識はΦ値で定量化される",
  "context": "意識理論の議論中",
  "domain_hint": "consciousness",
  "verification_level": "moderate",
  "require_consensus": true
}
```

**パラメータ仕様:**

| パラメータ | 型 | 必須 | 説明 | 取りうる値 |
|----------|---|-----|------|-----------|
| `statement` | string | ✅ | 検証対象の文 | 任意の文字列 |
| `context` | string | ❌ | 文脈情報 | 任意の文字列 |
| `domain_hint` | string | ❌ | 分野ヒント | `consciousness`, `philosophy`, `mathematics` |
| `verification_level` | string | ❌ | 検証レベル | `surface`, `shallow`, `moderate`, `deep`, `expert` |
| `require_consensus` | boolean | ❌ | コンセンサス要求 | `true`, `false` |

**cURL例:**
```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "statement": "意識は脳の電気活動によって完全に説明される",
    "verification_level": "deep",
    "require_consensus": true
  }'
```

**レスポンス例:**
```json
{
  "request_id": "req_1690737600123",
  "statement": "意識は脳の電気活動によって完全に説明される",
  "is_valid": false,
  "confidence_score": 0.73,
  "hallucination_detected": false,
  "expert_consensus": {
    "consensus_type": "simple_majority",
    "overall_validity": false,
    "confidence_score": 0.68,
    "participating_experts": ["consciousness_specialist", "philosophy_specialist"],
    "synthesized_conclusion": "この主張は過度に還元主義的であり、意識の主観的側面を無視している"
  },
  "domain_analysis": {
    "consciousness": {
      "is_valid": false,
      "confidence_score": 0.65,
      "findings": ["意識のハード問題を考慮していない", "現象学的側面が欠如"]
    },
    "philosophy": {
      "is_valid": false,
      "confidence_score": 0.71,
      "findings": ["心身問題の単純化", "質的体験の軽視"]
    }
  },
  "recommendations": [
    "意識の主観的側面について言及を追加してください",
    "物理主義の限界について考慮してください"
  ],
  "processing_time": 2.34,
  "timestamp": "2025-07-30T12:34:56Z"
}
```

### 3. 検証履歴取得

**エンドポイント:** `GET /history`

**パラメータ:**
- `limit` (optional): 取得件数の上限（デフォルト: 50）

```bash
curl "http://localhost:8000/history?limit=10"
```

**レスポンス:**
```json
{
  "history": [
    {
      "request_id": "req_1690737600123",
      "statement": "意識は脳の電気活動である",
      "is_valid": false,
      "confidence_score": 0.73,
      "timestamp": "2025-07-30T12:34:56Z"
    }
  ],
  "total_count": 42
}
```

### 4. 統計情報取得

**エンドポイント:** `GET /stats`

```bash
curl http://localhost:8000/stats
```

**レスポンス:**
```json
{
  "total_verifications": 42,
  "hallucinations_detected": 8,
  "average_processing_time": 1.23,
  "consensus_achieved": 35
}
```

---

## WebSocket API

### 1. 接続確立

**エンドポイント:** `ws://localhost:8000/ws`

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function() {
    console.log('WebSocket接続確立');
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('受信:', message);
};
```

### 2. メッセージ形式

#### 検証リクエスト送信
```javascript
const request = {
    type: 'verify_request',
    data: {
        statement: '人工意識は2030年までに実現される',
        context: '技術予測',
        domain_hint: 'consciousness',
        verification_level: 'expert',
        require_consensus: true
    }
};

ws.send(JSON.stringify(request));
```

#### Ping-Pong (接続維持)
```javascript
// Ping送信
ws.send(JSON.stringify({type: 'ping'}));

// Pong受信
ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    if (message.type === 'pong') {
        console.log('Pong受信:', message.timestamp);
    }
};
```

### 3. 受信メッセージタイプ

| タイプ | 説明 | データ |
|-------|------|-------|
| `connection_established` | 接続確立通知 | `{message: string}` |
| `verification_started` | 検証開始通知 | `{request_id: string}` |
| `verification_result` | 検証結果 | `VerificationResponse` オブジェクト |
| `pong` | Pingへの応答 | `{timestamp: string}` |
| `error` | エラー通知 | `{message: string}` |

---

## Python SDK

### 1. 基本的な使用方法

```python
import asyncio
import aiohttp
import json

class OmoikaneClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def verify_statement(self, statement, **kwargs):
        """文を検証"""
        data = {"statement": statement, **kwargs}
        
        async with self.session.post(f"{self.base_url}/verify", json=data) as response:
            return await response.json()
    
    async def get_status(self):
        """システム状態を取得"""
        async with self.session.get(f"{self.base_url}/status") as response:
            return await response.json()
    
    async def get_history(self, limit=50):
        """検証履歴を取得"""
        async with self.session.get(f"{self.base_url}/history?limit={limit}") as response:
            return await response.json()

# 使用例
async def main():
    async with OmoikaneClient() as client:
        # システム状態確認
        status = await client.get_status()
        print(f"システム状態: {status['system_initialized']}")
        
        # 文の検証
        result = await client.verify_statement(
            statement="統合情報理論は意識研究の基礎理論である",
            verification_level="expert",
            require_consensus=True
        )
        
        print(f"検証結果: {'有効' if result['is_valid'] else '無効'}")
        print(f"信頼度: {result['confidence_score']:.2%}")

asyncio.run(main())
```

### 2. WebSocket クライアント

```python
import asyncio
import websockets
import json

class OmoikaneWebSocketClient:
    def __init__(self, uri="ws://localhost:8000/ws"):
        self.uri = uri
        self.websocket = None
    
    async def connect(self):
        """WebSocket接続"""
        self.websocket = await websockets.connect(self.uri)
        
        # 接続確立メッセージを待機
        message = await self.websocket.recv()
        return json.loads(message)
    
    async def verify_statement(self, statement, **kwargs):
        """WebSocket経由で文を検証"""
        request = {
            'type': 'verify_request',
            'data': {'statement': statement, **kwargs}
        }
        
        await self.websocket.send(json.dumps(request))
        
        # 結果を待機
        while True:
            message = json.loads(await self.websocket.recv())
            
            if message['type'] == 'verification_result':
                return message['data']
            elif message['type'] == 'error':
                raise Exception(message['message'])
    
    async def close(self):
        """接続を閉じる"""
        if self.websocket:
            await self.websocket.close()

# 使用例
async def websocket_example():
    client = OmoikaneWebSocketClient()
    
    try:
        await client.connect()
        
        result = await client.verify_statement(
            statement="現象学は科学的手法である",
            verification_level="deep"
        )
        
        print(f"WebSocket検証結果: {result['is_valid']}")
        
    finally:
        await client.close()

asyncio.run(websocket_example())
```

### 3. バッチ処理クラス

```python
class OmoikaneBatchProcessor:
    def __init__(self, client: OmoikaneClient):
        self.client = client
    
    async def verify_batch(self, statements, **common_params):
        """複数文の一括検証"""
        tasks = []
        
        for statement in statements:
            task = self.client.verify_statement(statement, **common_params)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果整理
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'statement': statements[i],
                    'error': str(result),
                    'success': False
                })
            else:
                processed_results.append({
                    'statement': statements[i],
                    'result': result,
                    'success': True
                })
        
        return processed_results
    
    def analyze_batch_results(self, results):
        """バッチ結果の統計分析"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {'error': 'No successful verifications'}
        
        valid_count = sum(1 for r in successful_results if r['result']['is_valid'])
        hallucination_count = sum(1 for r in successful_results if r['result']['hallucination_detected'])
        avg_confidence = sum(r['result']['confidence_score'] for r in successful_results) / len(successful_results)
        avg_processing_time = sum(r['result']['processing_time'] for r in successful_results) / len(successful_results)
        
        return {
            'total_statements': len(results),
            'successful_verifications': len(successful_results),
            'valid_statements': valid_count,
            'hallucinations_detected': hallucination_count,
            'success_rate': len(successful_results) / len(results),
            'validity_rate': valid_count / len(successful_results),
            'hallucination_rate': hallucination_count / len(successful_results),
            'average_confidence': avg_confidence,
            'average_processing_time': avg_processing_time
        }

# 使用例
async def batch_example():
    async with OmoikaneClient() as client:
        processor = OmoikaneBatchProcessor(client)
        
        statements = [
            "意識は脳の電気活動である",
            "量子もつれが意識を説明する", 
            "人工知能は感情を持てない",
            "現象学は客観科学である",
            "自由意志は幻想である"
        ]
        
        results = await processor.verify_batch(
            statements,
            verification_level="moderate",
            require_consensus=True
        )
        
        analysis = processor.analyze_batch_results(results)
        print(f"バッチ処理統計: {analysis}")

asyncio.run(batch_example())
```

---

## レスポンス形式

### 1. VerificationResponse オブジェクト

```typescript
interface VerificationResponse {
  request_id: string;                    // リクエストID
  statement: string;                     // 検証対象文
  is_valid: boolean;                     // 妥当性判定
  confidence_score: number;              // 信頼度 (0.0-1.0)
  hallucination_detected: boolean;       // ハルシネーション検出
  expert_consensus?: ExpertConsensus;    // 専門家コンセンサス
  domain_analysis: DomainAnalysis;       // 分野別分析
  recommendations: string[];             // 推奨事項
  processing_time: number;               // 処理時間(秒)
  timestamp: string;                     // タイムスタンプ
}
```

### 2. ExpertConsensus オブジェクト

```typescript
interface ExpertConsensus {
  consensus_type: string;                // コンセンサスタイプ
  overall_validity: boolean;             // 全体妥当性
  confidence_score: number;              // コンセンサス信頼度
  participating_experts: string[];       // 参加専門家
  agreeing_experts: string[];            // 賛成専門家
  dissenting_experts: string[];          // 反対専門家
  synthesized_conclusion: string;        // 統合結論
  recommendations: string[];             // 推奨事項
  minority_opinions: string[];           // 少数意見
}
```

### 3. DomainAnalysis オブジェクト

```typescript
interface DomainAnalysis {
  [domain: string]: {
    is_valid: boolean;                   // 分野での妥当性
    confidence_score: number;            // 分野別信頼度
    verification_level: string;          // 検証レベル
    findings: string[];                  // 発見事項
    corrections: string[];               // 修正提案
    red_flags: string[];                 // 警告事項
    specialist_notes: string;            // 専門家ノート
  }
}
```

---

## エラーハンドリング

### 1. HTTPエラーコード

| コード | 説明 | 対応 |
|-------|------|------|
| `200` | 成功 | - |
| `400` | 不正なリクエスト | リクエスト形式を確認 |
| `422` | バリデーションエラー | パラメータを確認 |
| `500` | サーバーエラー | システム管理者に連絡 |
| `503` | サービス利用不可 | しばらく待ってから再試行 |

### 2. エラーレスポンス形式

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "検証対象の文が空です",
    "details": {
      "field": "statement",
      "constraint": "必須項目"
    }
  },
  "request_id": "req_1690737600123",
  "timestamp": "2025-07-30T12:34:56Z"
}
```

### 3. Python エラーハンドリング例

```python
import aiohttp
import asyncio

async def robust_verification(statement):
    """エラー処理付きの検証"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "http://localhost:8000/verify",
                json={"statement": statement},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                elif response.status == 400:
                    error_data = await response.json()
                    raise ValueError(f"不正なリクエスト: {error_data['error']['message']}")
                elif response.status == 422:
                    error_data = await response.json()
                    raise ValueError(f"バリデーションエラー: {error_data['error']['message']}")
                elif response.status == 500:
                    raise RuntimeError("サーバーエラーが発生しました")
                else:
                    raise RuntimeError(f"予期しないエラー: {response.status}")
                    
        except asyncio.TimeoutError:
            raise TimeoutError("リクエストがタイムアウトしました")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"接続エラー: {e}")

# 使用例
async def safe_verification_example():
    try:
        result = await robust_verification("意識は計算プロセスである")
        print(f"検証成功: {result['is_valid']}")
        
    except ValueError as e:
        print(f"入力エラー: {e}")
    except TimeoutError as e:
        print(f"タイムアウト: {e}")
    except ConnectionError as e:
        print(f"接続問題: {e}")
    except Exception as e:
        print(f"予期しないエラー: {e}")

asyncio.run(safe_verification_example())
```

---

## 使用例

### 1. 学術論文検証システム

```python
class AcademicPaperVerifier:
    def __init__(self):
        self.client = None
    
    async def __aenter__(self):
        self.client = OmoikaneClient()
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    def extract_claims(self, paper_text):
        """論文から主要主張を抽出（簡易版）"""
        # 実際の実装では高度なNLP処理
        claims = []
        sentences = paper_text.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in 
                   ['propose', 'demonstrate', 'prove', 'show', 'conclude']):
                claims.append(sentence.strip())
        
        return claims[:10]  # 最大10件
    
    async def verify_paper(self, paper_text, title=""):
        """論文全体を検証"""
        claims = self.extract_claims(paper_text)
        
        verification_results = []
        for claim in claims:
            try:
                result = await self.client.verify_statement(
                    statement=claim,
                    context=f"学術論文: {title}",
                    verification_level="expert",
                    require_consensus=True
                )
                verification_results.append(result)
                
            except Exception as e:
                verification_results.append({
                    'error': str(e),
                    'statement': claim
                })
        
        # 論文全体の評価
        valid_claims = sum(1 for r in verification_results 
                          if not r.get('error') and r['is_valid'])
        total_claims = len(verification_results)
        
        paper_score = valid_claims / total_claims if total_claims > 0 else 0
        
        return {
            'paper_title': title,
            'total_claims': total_claims,
            'valid_claims': valid_claims,
            'paper_score': paper_score,
            'detailed_results': verification_results,
            'recommendation': self._get_paper_recommendation(paper_score)
        }
    
    def _get_paper_recommendation(self, score):
        """論文推奨度を判定"""
        if score >= 0.9:
            return "優秀 - 高い信頼性"
        elif score >= 0.7:
            return "良好 - 概ね信頼できる"
        elif score >= 0.5:
            return "要検討 - 部分的な問題あり"
        else:
            return "要改善 - 重大な問題あり"

# 使用例
async def verify_academic_paper():
    paper_text = """
    We propose a new approach to consciousness measurement.
    Our method demonstrates significant improvements over existing techniques.
    The results show that artificial consciousness is achievable by 2025.
    We conclude that this breakthrough will revolutionize AI research.
    """
    
    async with AcademicPaperVerifier() as verifier:
        result = await verifier.verify_paper(
            paper_text, 
            title="Advances in Consciousness Measurement"
        )
        
        print(f"論文評価: {result['recommendation']}")
        print(f"妥当な主張: {result['valid_claims']}/{result['total_claims']}")

asyncio.run(verify_academic_paper())
```

### 2. リアルタイム議論監視システム

```python
import asyncio
import websockets
import json
from datetime import datetime

class DebateMonitor:
    def __init__(self):
        self.websocket = None
        self.debate_log = []
    
    async def start_monitoring(self):
        """議論監視開始"""
        self.websocket = await websockets.connect("ws://localhost:8000/ws")
        
        # 接続確立を待機
        await self.websocket.recv()
        print("議論監視システム開始")
    
    async def analyze_statement(self, speaker, statement):
        """発言をリアルタイム分析"""
        request = {
            'type': 'verify_request',
            'data': {
                'statement': statement,
                'context': f'議論中の発言 - 発言者: {speaker}',
                'verification_level': 'moderate',
                'require_consensus': True
            }
        }
        
        await self.websocket.send(json.dumps(request))
        
        # 結果待機
        while True:
            message = json.loads(await self.websocket.recv())
            
            if message['type'] == 'verification_result':
                result = message['data']
                
                # 議論ログに記録
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'speaker': speaker,
                    'statement': statement,
                    'is_valid': result['is_valid'],
                    'confidence': result['confidence_score'],
                    'hallucination': result['hallucination_detected'],
                    'recommendations': result['recommendations']
                }
                self.debate_log.append(log_entry)
                
                # リアルタイムフィードバック
                await self._provide_feedback(log_entry)
                
                return result
    
    async def _provide_feedback(self, log_entry):
        """リアルタイムフィードバック提供"""
        if log_entry['hallucination']:
            print(f"⚠️  {log_entry['speaker']}の発言に不正確な情報が含まれています")
        
        if not log_entry['is_valid']:
            print(f"🤔 {log_entry['speaker']}の発言について要検討")
            
        if log_entry['recommendations']:
            print(f"💡 {log_entry['speaker']}への推奨:")
            for rec in log_entry['recommendations'][:2]:  # 最大2つ
                print(f"   - {rec}")
    
    def get_debate_summary(self):
        """議論サマリー生成"""
        if not self.debate_log:
            return "議論データがありません"
        
        total_statements = len(self.debate_log)
        valid_statements = sum(1 for entry in self.debate_log if entry['is_valid'])
        hallucinations = sum(1 for entry in self.debate_log if entry['hallucination'])
        
        speakers = set(entry['speaker'] for entry in self.debate_log)
        
        speaker_stats = {}
        for speaker in speakers:
            speaker_entries = [e for e in self.debate_log if e['speaker'] == speaker]
            speaker_stats[speaker] = {
                'total_statements': len(speaker_entries),
                'valid_statements': sum(1 for e in speaker_entries if e['is_valid']),
                'avg_confidence': sum(e['confidence'] for e in speaker_entries) / len(speaker_entries)
            }
        
        return {
            'total_statements': total_statements,
            'valid_statements': valid_statements,
            'hallucinations_detected': hallucinations, 
            'validity_rate': valid_statements / total_statements,
            'hallucination_rate': hallucinations / total_statements,
            'speaker_statistics': speaker_stats,
            'debate_quality': 'excellent' if valid_statements / total_statements > 0.8 else 
                             'good' if valid_statements / total_statements > 0.6 else 'poor'
        }
    
    async def close(self):
        """監視終了"""
        if self.websocket:
            await self.websocket.close()

# 使用例
async def monitor_debate():
    monitor = DebateMonitor()
    
    try:
        await monitor.start_monitoring()
        
        # 議論をシミュレート
        debate_statements = [
            ("研究者A", "IITによると、意識はΦ値で完全に定量化できる"),
            ("研究者B", "それは過度な単純化です。意識には質的側面もあります"),
            ("研究者A", "しかし、科学的には測定可能な指標が必要です"),
            ("研究者C", "現象学的アプローチも重要な洞察を提供します")
        ]
        
        for speaker, statement in debate_statements:
            await monitor.analyze_statement(speaker, statement)
            await asyncio.sleep(1)  # 1秒間隔
        
        # 議論サマリー出力
        summary = monitor.get_debate_summary()
        print("\n=== 議論サマリー ===")
        print(f"総発言数: {summary['total_statements']}")
        print(f"妥当性率: {summary['validity_rate']:.1%}")
        print(f"議論品質: {summary['debate_quality']}")
        
    finally:
        await monitor.close()

asyncio.run(monitor_debate())
```

---

**Last Updated**: 2025年7月30日  
**API Version**: 1.0.0

このAPIリファレンスを活用して、Omoikane Labの高度な検証システムを最大限にご利用ください！🚀