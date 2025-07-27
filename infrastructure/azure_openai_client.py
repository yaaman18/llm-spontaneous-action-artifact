"""
Azure OpenAI Service統合
廣里敏明（Hirosato Gamo）による実装

プロダクション環境でのスケーラブルなLLM統合を実現。
"""
import os
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import backoff
from collections import deque
import json
import logging
from enum import Enum

# Azure OpenAI SDK（仮想的なインポート）
# from azure.ai.openai.aio import AsyncAzureOpenAI


class ModelType(Enum):
    """使用可能なモデルタイプ"""
    GPT_35_TURBO = "gpt-35-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"


@dataclass
class RateLimitConfig:
    """レート制限設定"""
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    max_retries: int = 3
    backoff_factor: float = 2.0


@dataclass
class UsageMetrics:
    """使用状況メトリクス"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    error_count: int = 0
    average_latency_ms: float = 0.0


class AzureOpenAIClient:
    """
    Azure OpenAI Serviceクライアント
    
    プロダクション環境での要件：
    - レート制限の遵守
    - 自動リトライとバックオフ
    - コスト追跡と最適化
    - 非同期処理によるスループット向上
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 api_version: str = "2024-02-01",
                 default_model: ModelType = ModelType.GPT_4_TURBO,
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 enable_logging: bool = True):
        """
        Args:
            api_key: Azure OpenAI APIキー
            endpoint: エンドポイントURL
            api_version: API バージョン
            default_model: デフォルトモデル
            rate_limit_config: レート制限設定
            enable_logging: ロギングの有効化
        """
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        self.default_model = default_model
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        
        # ロギング設定
        if enable_logging:
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)
        else:
            self.logger = None
        
        # レート制限トラッキング
        self._request_times = deque(maxlen=self.rate_limit_config.requests_per_minute)
        self._token_usage = deque(maxlen=100)  # 直近100リクエストのトークン使用量
        
        # メトリクス
        self.metrics = UsageMetrics()
        
        # セッション管理
        self._session: Optional[aiohttp.ClientSession] = None
        
        # モデル別の価格設定（1000トークンあたりのドル）
        self.pricing = {
            ModelType.GPT_35_TURBO: {"input": 0.0005, "output": 0.0015},
            ModelType.GPT_4: {"input": 0.03, "output": 0.06},
            ModelType.GPT_4_TURBO: {"input": 0.01, "output": 0.03},
            ModelType.GPT_4O: {"input": 0.005, "output": 0.015}
        }
    
    async def __aenter__(self):
        """非同期コンテキストマネージャー入口"""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャー出口"""
        if self._session:
            await self._session.close()
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def create_chat_completion(self,
                                   messages: List[Dict[str, str]],
                                   model: Optional[ModelType] = None,
                                   temperature: float = 0.7,
                                   max_tokens: int = 1000,
                                   top_p: float = 1.0,
                                   frequency_penalty: float = 0.0,
                                   presence_penalty: float = 0.0,
                                   stop: Optional[List[str]] = None,
                                   stream: bool = False,
                                   **kwargs) -> Dict[str, Any]:
        """
        チャット補完を作成
        
        Args:
            messages: メッセージリスト
            model: 使用するモデル
            temperature: 生成温度
            max_tokens: 最大トークン数
            top_p: nucleus sampling パラメータ
            frequency_penalty: 頻度ペナルティ
            presence_penalty: 存在ペナルティ
            stop: 停止シーケンス
            stream: ストリーミングの有効化
            **kwargs: その他のパラメータ
            
        Returns:
            API応答
        """
        # レート制限チェック
        await self._check_rate_limit()
        
        # リクエストの準備
        model = model or self.default_model
        request_data = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream
        }
        
        if stop:
            request_data["stop"] = stop
        
        request_data.update(kwargs)
        
        # タイミング開始
        start_time = datetime.now()
        
        try:
            # API呼び出し
            response = await self._make_request(
                f"/openai/deployments/{model.value}/chat/completions",
                request_data
            )
            
            # メトリクス更新
            self._update_metrics(response, model, start_time)
            
            return response
            
        except Exception as e:
            self.metrics.error_count += 1
            if self.logger:
                self.logger.error(f"API call failed: {str(e)}")
            raise
    
    async def create_embedding(self,
                             input_text: Union[str, List[str]],
                             model: str = "text-embedding-ada-002") -> Dict[str, Any]:
        """
        テキスト埋め込みを作成
        
        Args:
            input_text: 入力テキスト
            model: 埋め込みモデル
            
        Returns:
            埋め込みベクトル
        """
        await self._check_rate_limit()
        
        request_data = {
            "input": input_text,
            "model": model
        }
        
        response = await self._make_request(
            f"/openai/deployments/{model}/embeddings",
            request_data
        )
        
        return response
    
    async def _make_request(self,
                          path: str,
                          data: Dict[str, Any]) -> Dict[str, Any]:
        """
        APIリクエストを実行
        
        Args:
            path: APIパス
            data: リクエストデータ
            
        Returns:
            API応答
        """
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        url = f"{self.endpoint}{path}?api-version={self.api_version}"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        async with self._session.post(
            url,
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            response_data = await response.json()
            
            if response.status != 200:
                error_msg = response_data.get("error", {}).get("message", "Unknown error")
                raise Exception(f"API error: {error_msg}")
            
            return response_data
    
    async def _check_rate_limit(self) -> None:
        """レート制限をチェックし、必要に応じて待機"""
        now = datetime.now()
        
        # 古いリクエストタイムスタンプを削除
        cutoff_time = now - timedelta(minutes=1)
        while self._request_times and self._request_times[0] < cutoff_time:
            self._request_times.popleft()
        
        # レート制限に達している場合は待機
        if len(self._request_times) >= self.rate_limit_config.requests_per_minute:
            wait_time = (self._request_times[0] + timedelta(minutes=1) - now).total_seconds()
            if wait_time > 0:
                if self.logger:
                    self.logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # リクエストタイムスタンプを記録
        self._request_times.append(now)
    
    def _update_metrics(self,
                       response: Dict[str, Any],
                       model: ModelType,
                       start_time: datetime) -> None:
        """メトリクスを更新"""
        # レイテンシ計算
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # トークン使用量
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        
        # コスト計算
        pricing = self.pricing.get(model, self.pricing[ModelType.GPT_4_TURBO])
        cost = (
            (prompt_tokens / 1000) * pricing["input"] +
            (completion_tokens / 1000) * pricing["output"]
        )
        
        # メトリクス更新
        self.metrics.total_requests += 1
        self.metrics.total_tokens += total_tokens
        self.metrics.total_cost += cost
        
        # 平均レイテンシの更新（移動平均）
        alpha = 0.1  # 平滑化係数
        self.metrics.average_latency_ms = (
            alpha * latency_ms + 
            (1 - alpha) * self.metrics.average_latency_ms
        )
        
        # トークン使用量を記録
        self._token_usage.append({
            "timestamp": datetime.now(),
            "tokens": total_tokens,
            "cost": cost
        })
        
        if self.logger:
            self.logger.info(
                f"Request completed: {total_tokens} tokens, "
                f"${cost:.4f}, {latency_ms:.0f}ms"
            )
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """使用状況のサマリーを取得"""
        recent_usage = list(self._token_usage)
        
        # 直近1時間のコスト
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_cost = sum(
            u["cost"] for u in recent_usage 
            if u["timestamp"] > hour_ago
        )
        
        return {
            "total_requests": self.metrics.total_requests,
            "total_tokens": self.metrics.total_tokens,
            "total_cost": f"${self.metrics.total_cost:.2f}",
            "recent_cost_per_hour": f"${recent_cost:.2f}",
            "error_rate": (
                self.metrics.error_count / self.metrics.total_requests 
                if self.metrics.total_requests > 0 else 0
            ),
            "average_latency_ms": self.metrics.average_latency_ms,
            "current_rpm": len(self._request_times)
        }
    
    async def optimize_for_cost(self,
                              messages: List[Dict[str, str]],
                              quality_threshold: float = 0.8) -> Dict[str, Any]:
        """
        コスト最適化されたリクエスト
        
        品質閾値に基づいて適切なモデルを選択
        """
        # メッセージの複雑さを評価
        total_length = sum(len(m["content"]) for m in messages)
        
        # 簡単なタスクには安価なモデルを使用
        if total_length < 500 and quality_threshold < 0.7:
            model = ModelType.GPT_35_TURBO
            temperature = 0.5
        elif quality_threshold >= 0.9:
            model = ModelType.GPT_4
            temperature = 0.7
        else:
            model = ModelType.GPT_4_TURBO
            temperature = 0.6
        
        if self.logger:
            self.logger.info(f"Cost optimization: selected {model.value}")
        
        return await self.create_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=min(1000, total_length * 2)  # 適応的なトークン制限
        )
    
    async def batch_process(self,
                          requests: List[Dict[str, Any]],
                          max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """
        バッチ処理で複数のリクエストを効率的に処理
        
        Args:
            requests: リクエストのリスト
            max_concurrent: 最大同時実行数
            
        Returns:
            応答のリスト
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(request):
            async with semaphore:
                return await self.create_chat_completion(**request)
        
        tasks = [process_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)