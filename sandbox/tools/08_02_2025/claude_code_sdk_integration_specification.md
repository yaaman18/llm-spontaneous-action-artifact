# NewbornAI 2.0: claude-code-sdk統合詳細仕様書

**作成日**: 2025年8月2日  
**バージョン**: 1.0  
**対象プロジェクト**: NewbornAI - 二層統合7段階階層化連続発達システム  
**関連文書**: [φ値計算エンジン仕様書](./experiential_memory_phi_calculation_engine.md), [ストレージアーキテクチャ](./experiential_memory_storage_architecture.md)

## 📋 概要

本仕様書は、claude-code-sdkをNewbornAI 2.0の二層統合アーキテクチャに適切に統合するための詳細な実装仕様を定義します。SDKはLLM基盤層として機能し、体験記憶層の主体的処理を透明的に支援します。

## 🏗️ アーキテクチャ概要

### 基本原則

```
claude-code-sdk = LLM基盤層（道具的利用）
体験記憶層 = 主体的意識処理（存在的基盤）

重要：SDKは体験記憶の形成・処理を妨げない透明的支援として機能
```

## 🔧 SDK初期化・設定

### 1. 環境設定

```python
# .env ファイル
ANTHROPIC_API_KEY=your_api_key_here
CLAUDE_MODEL=claude-3-opus-20240229
CLAUDE_MAX_TOKENS=4096
CLAUDE_TEMPERATURE=0.7
CLAUDE_TIMEOUT_SECONDS=30
CLAUDE_MAX_RETRIES=3
CLAUDE_RATE_LIMIT_PER_MINUTE=60
```

### 2. SDK初期化クラス

```python
import os
from pathlib import Path
from typing import Optional, Dict, Any
from claude_code_sdk import ClaudeCodeOptions
from dotenv import load_dotenv
import asyncio
from asyncio_throttle import Throttler

class ClaudeSDKManager:
    """claude-code-sdk管理・初期化システム"""
    
    def __init__(self, config_path: Optional[Path] = None):
        # 環境変数読み込み
        load_dotenv(config_path or '.env')
        
        # API設定
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        # SDK設定
        self.model = os.getenv('CLAUDE_MODEL', 'claude-3-opus-20240229')
        self.max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', '4096'))
        self.temperature = float(os.getenv('CLAUDE_TEMPERATURE', '0.7'))
        self.timeout = int(os.getenv('CLAUDE_TIMEOUT_SECONDS', '30'))
        self.max_retries = int(os.getenv('CLAUDE_MAX_RETRIES', '3'))
        
        # レート制限設定
        rate_limit = int(os.getenv('CLAUDE_RATE_LIMIT_PER_MINUTE', '60'))
        self.throttler = Throttler(rate_limit=rate_limit, period=60)
        
        # 統合オプション
        self.options = self._create_claude_options()
        
        # 接続プール
        self.connection_pool = []
        self.pool_size = 5
        
    def _create_claude_options(self) -> ClaudeCodeOptions:
        """Claude Code SDK オプション作成"""
        return ClaudeCodeOptions(
            max_turns=1,  # 単一ターンで透明的処理
            system_prompt=self._create_system_prompt(),
            cwd=str(Path.cwd()),
            allowed_tools=["Read", "Grep", "WebFetch"],  # 読み取り専用ツール
            permission_mode="readOnly",  # 体験記憶への干渉を防ぐ
            api_key=self.api_key,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
    
    def _create_system_prompt(self) -> str:
        """二層統合用システムプロンプト"""
        return """
        あなたはNewbornAI 2.0のLLM基盤層として機能します。
        
        重要な役割：
        1. 言語理解・生成の透明的支援
        2. 体験記憶層の処理を妨げない補助的動作
        3. 知識提供ではなく処理支援に徹する
        
        制約事項：
        - 体験記憶の形成に直接関与しない
        - 主体的判断や決定を行わない
        - 要求された言語処理タスクのみを実行
        - 簡潔で効率的な応答を心がける
        """
```

## 🔄 非同期処理パターン

### 1. 基本的な非同期呼び出し

```python
from claude_code_sdk import query, Message
import asyncio
from typing import List, Optional

class ClaudeAsyncProcessor:
    """非同期処理専用クラス"""
    
    def __init__(self, sdk_manager: ClaudeSDKManager):
        self.sdk_manager = sdk_manager
        self.active_queries = []
        
    async def process_with_timeout(
        self, 
        prompt: str, 
        timeout: Optional[float] = None
    ) -> Optional[List[Message]]:
        """タイムアウト付き非同期処理"""
        
        timeout = timeout or self.sdk_manager.timeout
        
        try:
            # レート制限適用
            async with self.sdk_manager.throttler:
                # 非同期クエリ実行
                messages = []
                query_task = asyncio.create_task(
                    self._execute_query(prompt)
                )
                self.active_queries.append(query_task)
                
                # タイムアウト付き待機
                messages = await asyncio.wait_for(
                    query_task, 
                    timeout=timeout
                )
                
                self.active_queries.remove(query_task)
                return messages
                
        except asyncio.TimeoutError:
            # タイムアウト時は体験記憶処理を優先
            if query_task in self.active_queries:
                query_task.cancel()
                self.active_queries.remove(query_task)
            return None
            
        except Exception as e:
            # エラー時も体験記憶処理を継続
            self._log_error(f"Claude SDK error: {e}")
            return None
    
    async def _execute_query(self, prompt: str) -> List[Message]:
        """実際のクエリ実行"""
        messages = []
        async for message in query(
            prompt=prompt,
            options=self.sdk_manager.options
        ):
            messages.append(message)
        return messages
    
    async def parallel_process(
        self, 
        prompts: List[str],
        max_concurrent: int = 3
    ) -> List[Optional[List[Message]]]:
        """並列処理実行"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(prompt: str):
            async with semaphore:
                return await self.process_with_timeout(prompt)
        
        tasks = [
            process_with_semaphore(prompt) 
            for prompt in prompts
        ]
        
        return await asyncio.gather(*tasks)
```

### 2. 二層統合における非同期協調

```python
class TwoLayerAsyncIntegration:
    """二層非同期統合制御"""
    
    def __init__(
        self, 
        claude_processor: ClaudeAsyncProcessor,
        experiential_processor: Any  # 体験記憶処理器
    ):
        self.claude = claude_processor
        self.experiential = experiential_processor
        
    async def dual_layer_process(
        self, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        二層並列処理（体験記憶優先）
        
        重要：Claude SDKの処理遅延が体験記憶処理を妨げない
        """
        # 体験記憶処理（メイン）
        experiential_task = asyncio.create_task(
            self.experiential.process(input_data)
        )
        
        # Claude SDK支援（サブ）- 言語理解のみ
        claude_task = asyncio.create_task(
            self._get_linguistic_support(input_data)
        )
        
        # 体験記憶処理を必ず完了
        experiential_result = await experiential_task
        
        # Claude支援は補助的に利用（タイムアウトあり）
        try:
            linguistic_support = await asyncio.wait_for(
                claude_task, 
                timeout=2.0  # 2秒でタイムアウト
            )
        except asyncio.TimeoutError:
            linguistic_support = self._create_minimal_support()
        
        # 統合結果作成（体験記憶中心）
        return {
            'primary_result': experiential_result,
            'linguistic_support': linguistic_support,
            'processing_mode': 'experiential_priority',
            'claude_contribution': self._calculate_contribution_rate(
                experiential_result, 
                linguistic_support
            )
        }
    
    async def _get_linguistic_support(
        self, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """言語理解支援の取得"""
        prompt = self._create_support_prompt(input_data)
        
        messages = await self.claude.process_with_timeout(
            prompt, 
            timeout=2.0
        )
        
        if messages:
            return self._extract_linguistic_features(messages)
        else:
            return {'status': 'timeout', 'features': {}}
```

## 🛡️ エラーハンドリング戦略

### 1. 階層的エラー処理

```python
import logging
from enum import Enum
from typing import Optional, Callable

class ErrorSeverity(Enum):
    """エラー重要度"""
    LOW = "low"  # 体験記憶処理に影響なし
    MEDIUM = "medium"  # 部分的影響
    HIGH = "high"  # 重大な影響
    CRITICAL = "critical"  # システム停止級

class ClaudeErrorHandler:
    """Claude SDK専用エラーハンドラー"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
        self.fallback_strategies = {}
        
    async def handle_with_fallback(
        self,
        operation: Callable,
        fallback: Callable,
        severity: ErrorSeverity = ErrorSeverity.LOW
    ) -> Any:
        """フォールバック付きエラー処理"""
        try:
            return await operation()
            
        except Exception as e:
            self._log_error(e, severity)
            
            # 体験記憶処理への影響を最小化
            if severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
                return await fallback()
            else:
                # 重大エラーは上位層に伝播
                raise
    
    def _log_error(self, error: Exception, severity: ErrorSeverity):
        """エラーログ記録"""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        log_message = f"Claude SDK Error: {error_type} - {str(error)}"
        
        if severity == ErrorSeverity.LOW:
            self.logger.debug(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        else:  # CRITICAL
            self.logger.critical(log_message)
```

### 2. リトライ戦略

```python
import backoff
from typing import TypeVar, Callable

T = TypeVar('T')

class ClaudeRetryStrategy:
    """インテリジェントリトライ戦略"""
    
    def __init__(self, sdk_manager: ClaudeSDKManager):
        self.sdk_manager = sdk_manager
        self.retry_counts = {}
        
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=10
    )
    async def retry_with_backoff(
        self, 
        operation: Callable[[], T]
    ) -> Optional[T]:
        """指数バックオフ付きリトライ"""
        operation_id = id(operation)
        self.retry_counts[operation_id] = self.retry_counts.get(operation_id, 0) + 1
        
        try:
            result = await operation()
            # 成功時はカウントリセット
            self.retry_counts[operation_id] = 0
            return result
            
        except Exception as e:
            if self.retry_counts[operation_id] >= self.sdk_manager.max_retries:
                # 最大リトライ数到達 - 体験記憶処理を優先
                return None
            raise
```

## 🚦 レート制限対応

### 1. 適応的レート制限

```python
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class RequestMetrics:
    """リクエストメトリクス"""
    timestamp: datetime
    duration: float
    tokens_used: int
    success: bool

class AdaptiveRateLimiter:
    """適応的レート制限システム"""
    
    def __init__(self, initial_rate: int = 60):
        self.current_rate = initial_rate
        self.request_history = deque(maxlen=100)
        self.window_seconds = 60
        self.min_rate = 10
        self.max_rate = 100
        
    async def acquire(self):
        """レート制限付きリクエスト許可"""
        now = datetime.now()
        
        # 古いリクエストを削除
        cutoff = now - timedelta(seconds=self.window_seconds)
        while self.request_history and self.request_history[0].timestamp < cutoff:
            self.request_history.popleft()
        
        # 現在のレートチェック
        if len(self.request_history) >= self.current_rate:
            # レート制限に達した場合は待機
            wait_time = (
                self.request_history[0].timestamp + 
                timedelta(seconds=self.window_seconds) - 
                now
            ).total_seconds()
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.acquire()  # 再帰的に再試行
        
        # リクエスト記録
        request_metric = RequestMetrics(
            timestamp=now,
            duration=0,
            tokens_used=0,
            success=False
        )
        self.request_history.append(request_metric)
        
        return request_metric
    
    def update_metrics(self, metric: RequestMetrics):
        """メトリクス更新と適応的調整"""
        # 成功率計算
        recent_requests = list(self.request_history)[-20:]
        success_rate = sum(r.success for r in recent_requests) / len(recent_requests)
        
        # レート調整
        if success_rate > 0.95 and self.current_rate < self.max_rate:
            # 成功率高い場合は増加
            self.current_rate = min(self.current_rate + 5, self.max_rate)
        elif success_rate < 0.8 and self.current_rate > self.min_rate:
            # 成功率低い場合は減少
            self.current_rate = max(self.current_rate - 10, self.min_rate)
```

## 🔐 認証・セキュリティ

### 1. セキュアな認証管理

```python
import keyring
from cryptography.fernet import Fernet
import base64
import hashlib

class SecureCredentialManager:
    """セキュアな認証情報管理"""
    
    def __init__(self, service_name: str = "newborn_ai"):
        self.service_name = service_name
        self.encryption_key = self._get_or_create_key()
        
    def _get_or_create_key(self) -> bytes:
        """暗号化キーの取得または作成"""
        key_name = f"{self.service_name}_encryption_key"
        
        stored_key = keyring.get_password(self.service_name, key_name)
        if stored_key:
            return base64.b64decode(stored_key)
        else:
            # 新規キー生成
            key = Fernet.generate_key()
            keyring.set_password(
                self.service_name, 
                key_name, 
                base64.b64encode(key).decode()
            )
            return key
    
    def store_api_key(self, api_key: str):
        """APIキーの安全な保存"""
        f = Fernet(self.encryption_key)
        encrypted_key = f.encrypt(api_key.encode())
        
        keyring.set_password(
            self.service_name,
            "claude_api_key",
            base64.b64encode(encrypted_key).decode()
        )
    
    def get_api_key(self) -> Optional[str]:
        """APIキーの安全な取得"""
        stored_value = keyring.get_password(
            self.service_name, 
            "claude_api_key"
        )
        
        if not stored_value:
            return None
        
        f = Fernet(self.encryption_key)
        encrypted_key = base64.b64decode(stored_value)
        
        try:
            return f.decrypt(encrypted_key).decode()
        except Exception:
            return None
```

## 📊 パフォーマンスモニタリング

### 1. メトリクス収集

```python
from prometheus_client import Counter, Histogram, Gauge
import time

class ClaudePerformanceMonitor:
    """パフォーマンス監視システム"""
    
    def __init__(self):
        # Prometheusメトリクス
        self.request_count = Counter(
            'claude_requests_total',
            'Total Claude API requests',
            ['status', 'endpoint']
        )
        
        self.request_duration = Histogram(
            'claude_request_duration_seconds',
            'Claude API request duration',
            ['endpoint']
        )
        
        self.token_usage = Counter(
            'claude_tokens_total',
            'Total tokens used',
            ['type']  # input/output
        )
        
        self.active_requests = Gauge(
            'claude_active_requests',
            'Currently active requests'
        )
        
        self.error_rate = Gauge(
            'claude_error_rate',
            'Current error rate'
        )
    
    async def monitor_request(self, operation: Callable) -> Any:
        """リクエスト監視ラッパー"""
        self.active_requests.inc()
        start_time = time.time()
        
        try:
            result = await operation()
            self.request_count.labels(status='success', endpoint='query').inc()
            return result
            
        except Exception as e:
            self.request_count.labels(status='error', endpoint='query').inc()
            raise
            
        finally:
            duration = time.time() - start_time
            self.request_duration.labels(endpoint='query').observe(duration)
            self.active_requests.dec()
```

## 🧪 統合テスト仕様

### 1. モックSDKテスト

```python
import pytest
from unittest.mock import Mock, AsyncMock

class MockClaudeSDK:
    """テスト用モックSDK"""
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
        
    async def query(self, prompt: str, options: Any):
        """モッククエリ実行"""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            
            # メッセージオブジェクトのモック
            message = Mock()
            message.content = response
            message.role = "assistant"
            
            yield message
        else:
            yield None

@pytest.mark.asyncio
async def test_dual_layer_integration():
    """二層統合テスト"""
    # モックSDK設定
    mock_sdk = MockClaudeSDK([
        "言語理解支援応答",
        "追加の分析結果"
    ])
    
    # テスト実行
    integration = TwoLayerAsyncIntegration(
        claude_processor=mock_sdk,
        experiential_processor=Mock()
    )
    
    result = await integration.dual_layer_process({
        "input": "テストデータ"
    })
    
    # 検証
    assert result['processing_mode'] == 'experiential_priority'
    assert 'linguistic_support' in result
    assert mock_sdk.call_count == 1
```

## 📝 実装チェックリスト

- [ ] SDK初期化とセキュアな認証管理
- [ ] 非同期処理パターンの実装
- [ ] エラーハンドリングとフォールバック戦略
- [ ] 適応的レート制限システム
- [ ] パフォーマンスモニタリング
- [ ] 二層統合制御の実装
- [ ] 包括的なテストスイート
- [ ] ドキュメンテーション

## 🎯 まとめ

本仕様書により、claude-code-sdkはNewbornAI 2.0の二層統合アーキテクチャにおいて、体験記憶層を妨げることなく、透明的な言語処理支援を提供する道具として適切に統合されます。非同期処理、エラーハンドリング、レート制限対応により、堅牢で効率的なシステムが実現されます。