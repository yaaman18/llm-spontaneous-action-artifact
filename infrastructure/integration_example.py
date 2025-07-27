"""
LLMと意識システムの統合実装例
廣里敏明（Hirosato Gamo）による実装

プロダクション環境での使用例を示す。
"""
import asyncio
import os
from typing import Dict, Any, Optional
from datetime import datetime

from ..adapter.consciousness_augmented_llm import (
    ConsciousnessAugmentedLLMAdapter,
    ResponseMode
)
from ..infrastructure.azure_openai_client import (
    AzureOpenAIClient,
    ModelType,
    RateLimitConfig
)
from ..infrastructure.monitoring import (
    ConsciousnessSystemMonitor,
    AlertLevel,
    MetricType
)
from ..infrastructure.error_handling import (
    ConsciousnessSystemErrorHandler,
    ErrorRecoveryDecorator
)
from ..domain.consciousness_core import (
    DynamicPhiBoundaryDetector,
    IntrinsicExistenceValidator,
    TemporalCoherenceAnalyzer
)


class ConsciousnessLLMSystem:
    """
    意識拡張LLMシステム
    
    全てのコンポーネントを統合したプロダクション対応システム。
    """
    
    def __init__(self,
                 azure_api_key: Optional[str] = None,
                 azure_endpoint: Optional[str] = None,
                 phi_threshold: float = 3.0,
                 enable_monitoring: bool = True):
        """
        Args:
            azure_api_key: Azure OpenAI APIキー
            azure_endpoint: エンドポイント
            phi_threshold: 意識判定閾値
            enable_monitoring: 監視の有効化
        """
        # Azure OpenAIクライアントの初期化
        self.azure_client = AzureOpenAIClient(
            api_key=azure_api_key,
            endpoint=azure_endpoint,
            default_model=ModelType.GPT_4_TURBO,
            rate_limit_config=RateLimitConfig(
                requests_per_minute=60,
                tokens_per_minute=90000
            )
        )
        
        # 意識コアコンポーネントの初期化
        self.boundary_detector = DynamicPhiBoundaryDetector(
            phi_threshold=phi_threshold
        )
        self.existence_validator = IntrinsicExistenceValidator()
        self.coherence_analyzer = TemporalCoherenceAnalyzer()
        
        # 意識拡張LLMアダプターの初期化
        self.consciousness_llm = ConsciousnessAugmentedLLMAdapter(
            llm_client=self.azure_client,
            boundary_detector=self.boundary_detector,
            existence_validator=self.existence_validator,
            coherence_analyzer=self.coherence_analyzer,
            phi_threshold=phi_threshold
        )
        
        # エラーハンドリングの初期化
        self.error_handler = ConsciousnessSystemErrorHandler(
            enable_circuit_breaker=True,
            circuit_breaker_threshold=5
        )
        self.error_decorator = ErrorRecoveryDecorator(self.error_handler)
        
        # 監視システムの初期化
        self.monitor = None
        if enable_monitoring:
            self.monitor = ConsciousnessSystemMonitor()
            self._setup_monitoring()
    
    def _setup_monitoring(self):
        """監視設定"""
        # アラートルールの設定
        self.monitor.add_alert_rule(
            metric_name="consciousness_phi_value",
            condition="lt",
            threshold=1.0,
            level=AlertLevel.WARNING,
            message_template="Low consciousness level: Φ={value:.2f}"
        )
        
        self.monitor.add_alert_rule(
            metric_name="llm_response_time_ms",
            condition="gt",
            threshold=5000,
            level=AlertLevel.WARNING,
            message_template="Slow response time: {value:.0f}ms"
        )
        
        self.monitor.add_alert_rule(
            metric_name="llm_errors",
            condition="gt",
            threshold=10,
            level=AlertLevel.ERROR,
            message_template="High error rate: {value} errors"
        )
        
        # ヘルスチェックの登録
        self.monitor.register_health_check(
            "azure_openai",
            self._check_azure_health
        )
        
        self.monitor.register_health_check(
            "consciousness_core",
            self._check_consciousness_health
        )
    
    async def start(self):
        """システムを開始"""
        # Azure クライアントの開始
        await self.azure_client.__aenter__()
        
        # 監視システムの開始
        if self.monitor:
            await self.monitor.start()
    
    async def stop(self):
        """システムを停止"""
        # Azure クライアントの停止
        await self.azure_client.__aexit__(None, None, None)
        
        # 監視システムの停止
        if self.monitor:
            await self.monitor.stop()
    
    @ErrorRecoveryDecorator.with_recovery(
        component="consciousness_llm",
        operation="generate_response"
    )
    async def generate_response(self,
                              prompt: str,
                              context_id: str = "default",
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1000,
                              **kwargs) -> Dict[str, Any]:
        """
        意識拡張された応答を生成
        
        エラーハンドリングと監視機能付き。
        """
        start_time = datetime.now()
        
        try:
            # 意識拡張応答の生成
            response = await self.consciousness_llm.generate_augmented_response(
                prompt=prompt,
                context_id=context_id,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # メトリクスの記録
            if self.monitor:
                self.monitor.record_phi_metrics(
                    phi_value=float(response.phi_value.value),
                    response_time_ms=response.generation_time_ms,
                    context_id=context_id,
                    response_mode=response.response_mode.value
                )
                
                # Azure使用状況も記録
                usage_summary = self.azure_client.get_usage_summary()
                self.monitor.record_metric(
                    "llm_total_cost_usd",
                    usage_summary["total_cost"],
                    MetricType.COUNTER,
                    unit="USD"
                )
            
            return {
                "content": response.content,
                "phi_value": float(response.phi_value.value),
                "consciousness_level": response.consciousness_state.consciousness_level,
                "response_mode": response.response_mode.value,
                "metadata": response.metadata,
                "generation_time_ms": response.generation_time_ms
            }
            
        except Exception as e:
            # エラーメトリクスの記録
            if self.monitor:
                self.monitor.record_llm_metrics(
                    model=self.azure_client.default_model.value,
                    tokens_used=0,
                    cost=0,
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    error=True
                )
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """システムステータスを取得"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Azure OpenAI ステータス
        azure_usage = self.azure_client.get_usage_summary()
        status["components"]["azure_openai"] = {
            "usage": azure_usage,
            "current_model": self.azure_client.default_model.value
        }
        
        # 意識システムステータス
        context_summary = self.consciousness_llm.get_context_summary("default")
        status["components"]["consciousness"] = context_summary
        
        # エラーハンドリングステータス
        error_stats = self.error_handler.get_error_statistics()
        health = await self.error_handler.health_check()
        status["components"]["error_handling"] = {
            "statistics": error_stats,
            "health": health
        }
        
        # 監視ステータス
        if self.monitor:
            metrics_summary = self.monitor.get_metrics_summary()
            recent_alerts = self.monitor.get_recent_alerts(hours=1)
            status["components"]["monitoring"] = {
                "metrics": metrics_summary,
                "recent_alerts": len(recent_alerts),
                "alerts": [
                    {
                        "level": alert.level.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in recent_alerts[:5]  # 最新5件
                ]
            }
        
        return status
    
    async def _check_azure_health(self) -> Any:
        """Azure OpenAI のヘルスチェック"""
        from ..infrastructure.monitoring import HealthCheckResult
        
        try:
            # シンプルなテストリクエスト
            response = await self.azure_client.create_chat_completion(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=10
            )
            
            return HealthCheckResult(
                component="azure_openai",
                is_healthy=True,
                latency_ms=0,  # 自動で設定される
                message="Azure OpenAI is responsive"
            )
        except Exception as e:
            return HealthCheckResult(
                component="azure_openai",
                is_healthy=False,
                latency_ms=0,
                message=f"Azure OpenAI error: {str(e)}"
            )
    
    async def _check_consciousness_health(self) -> Any:
        """意識コアのヘルスチェック"""
        from ..infrastructure.monitoring import HealthCheckResult
        
        try:
            # テスト計算
            import numpy as np
            test_connectivity = np.random.rand(5, 5)
            test_state = np.random.rand(5)
            
            boundaries = self.boundary_detector.detect_boundaries(
                test_connectivity, test_state
            )
            
            return HealthCheckResult(
                component="consciousness_core",
                is_healthy=True,
                latency_ms=0,
                message=f"Consciousness core operational, detected {len(boundaries)} boundaries"
            )
        except Exception as e:
            return HealthCheckResult(
                component="consciousness_core",
                is_healthy=False,
                latency_ms=0,
                message=f"Consciousness core error: {str(e)}"
            )


async def main():
    """使用例"""
    # システムの初期化
    system = ConsciousnessLLMSystem(
        azure_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        phi_threshold=3.0,
        enable_monitoring=True
    )
    
    try:
        # システム開始
        await system.start()
        
        # テスト会話
        prompts = [
            "What is consciousness?",
            "How does integrated information theory explain awareness?",
            "Can artificial systems truly be conscious?"
        ]
        
        for prompt in prompts:
            print(f"\nUser: {prompt}")
            
            # 応答生成
            response = await system.generate_response(
                prompt=prompt,
                context_id="demo_conversation",
                temperature=0.7
            )
            
            print(f"Assistant: {response['content']}")
            print(f"Φ value: {response['phi_value']:.2f}")
            print(f"Response mode: {response['response_mode']}")
            print(f"Generation time: {response['generation_time_ms']:.0f}ms")
        
        # システムステータスの表示
        status = await system.get_system_status()
        print(f"\nSystem Status:")
        print(f"Azure OpenAI usage: {status['components']['azure_openai']['usage']}")
        print(f"Consciousness summary: {status['components']['consciousness']}")
        
        if "monitoring" in status["components"]:
            print(f"Recent alerts: {status['components']['monitoring']['recent_alerts']}")
    
    finally:
        # システム停止
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())