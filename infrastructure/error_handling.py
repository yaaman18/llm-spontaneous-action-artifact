"""
エラーハンドリングとリカバリシステム
廣里敏明（Hirosato Gamo）による実装

プロダクション環境での堅牢性を確保。
"""
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import traceback
import logging
from collections import defaultdict, deque
import json


class ErrorType(Enum):
    """エラータイプの分類"""
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    CONSCIOUSNESS_COMPUTE = "consciousness_compute"
    LLM_GENERATION = "llm_generation"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """リカバリ戦略"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    DEGRADE = "degrade"
    FAIL_FAST = "fail_fast"


@dataclass
class ErrorContext:
    """エラーコンテキスト"""
    error_type: ErrorType
    exception: Exception
    timestamp: datetime
    component: str
    operation: str
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None


@dataclass
class RecoveryAction:
    """リカバリアクション"""
    strategy: RecoveryStrategy
    delay_seconds: float = 0
    max_retries: int = 3
    fallback_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """サーキットブレーカーの状態"""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    consecutive_successes: int = 0
    last_success_time: Optional[datetime] = None


class ConsciousnessSystemErrorHandler:
    """
    意識システムエラーハンドラー
    
    LLMと意識コアシステムの統合環境でのエラーを
    適切に処理し、システムの回復力を提供する。
    """
    
    def __init__(self,
                 enable_circuit_breaker: bool = True,
                 circuit_breaker_threshold: int = 5,
                 circuit_breaker_timeout_seconds: int = 60,
                 max_error_history: int = 1000):
        """
        Args:
            enable_circuit_breaker: サーキットブレーカーの有効化
            circuit_breaker_threshold: エラー閾値
            circuit_breaker_timeout_seconds: タイムアウト時間
            max_error_history: エラー履歴の最大保持数
        """
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout_seconds = circuit_breaker_timeout_seconds
        
        # エラー履歴
        self._error_history: deque = deque(maxlen=max_error_history)
        
        # コンポーネント別のサーキットブレーカー
        self._circuit_breakers: Dict[str, CircuitBreakerState] = defaultdict(
            CircuitBreakerState
        )
        
        # エラータイプ別のハンドラー
        self._error_handlers: Dict[ErrorType, Callable] = {
            ErrorType.NETWORK: self._handle_network_error,
            ErrorType.RATE_LIMIT: self._handle_rate_limit_error,
            ErrorType.AUTHENTICATION: self._handle_auth_error,
            ErrorType.VALIDATION: self._handle_validation_error,
            ErrorType.CONSCIOUSNESS_COMPUTE: self._handle_consciousness_error,
            ErrorType.LLM_GENERATION: self._handle_llm_error,
            ErrorType.SYSTEM: self._handle_system_error,
            ErrorType.UNKNOWN: self._handle_unknown_error
        }
        
        # カスタムリカバリストラテジー
        self._recovery_strategies: Dict[str, Callable] = {}
        
        # ロガー
        self.logger = logging.getLogger(__name__)
    
    async def handle_error(self,
                         error: Exception,
                         component: str,
                         operation: str,
                         context: Optional[Dict[str, Any]] = None) -> RecoveryAction:
        """
        エラーを処理してリカバリアクションを決定
        
        Args:
            error: 発生したエラー
            component: エラーが発生したコンポーネント
            operation: 実行中だった操作
            context: 追加のコンテキスト情報
            
        Returns:
            リカバリアクション
        """
        # エラータイプを分類
        error_type = self._classify_error(error)
        
        # エラーコンテキストを作成
        error_context = ErrorContext(
            error_type=error_type,
            exception=error,
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            metadata=context or {},
            traceback=traceback.format_exc()
        )
        
        # エラー履歴に追加
        self._error_history.append(error_context)
        
        # ログ記録
        self.logger.error(
            f"Error in {component}.{operation}: {error_type.value} - {str(error)}",
            exc_info=True
        )
        
        # サーキットブレーカーチェック
        if self.enable_circuit_breaker:
            if self._check_circuit_breaker(component):
                return RecoveryAction(
                    strategy=RecoveryStrategy.CIRCUIT_BREAK,
                    metadata={"reason": "Circuit breaker is open"}
                )
        
        # エラータイプ別のハンドラーを実行
        handler = self._error_handlers.get(error_type, self._handle_unknown_error)
        recovery_action = await handler(error_context)
        
        # サーキットブレーカーの更新
        if self.enable_circuit_breaker:
            self._update_circuit_breaker(component, error_context, recovery_action)
        
        return recovery_action
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """エラーを分類"""
        error_message = str(error).lower()
        error_type_name = type(error).__name__
        
        # ネットワークエラー
        if any(keyword in error_message for keyword in 
               ["connection", "timeout", "network", "socket"]):
            return ErrorType.NETWORK
        
        # レート制限エラー
        if any(keyword in error_message for keyword in 
               ["rate limit", "too many requests", "429"]):
            return ErrorType.RATE_LIMIT
        
        # 認証エラー
        if any(keyword in error_message for keyword in 
               ["unauthorized", "authentication", "401", "403"]):
            return ErrorType.AUTHENTICATION
        
        # バリデーションエラー
        if any(keyword in error_message for keyword in 
               ["validation", "invalid", "bad request", "400"]):
            return ErrorType.VALIDATION
        
        # 意識計算エラー
        if "phi" in error_message or "consciousness" in error_message:
            return ErrorType.CONSCIOUSNESS_COMPUTE
        
        # LLM生成エラー
        if any(keyword in error_message for keyword in 
               ["llm", "openai", "generation", "completion"]):
            return ErrorType.LLM_GENERATION
        
        # システムエラー
        if error_type_name in ["OSError", "SystemError", "MemoryError"]:
            return ErrorType.SYSTEM
        
        return ErrorType.UNKNOWN
    
    async def _handle_network_error(self,
                                  context: ErrorContext) -> RecoveryAction:
        """ネットワークエラーの処理"""
        # 指数バックオフでリトライ
        retry_delay = min(60, 2 ** context.retry_count)
        
        return RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            delay_seconds=retry_delay,
            max_retries=5,
            metadata={"backoff": "exponential"}
        )
    
    async def _handle_rate_limit_error(self,
                                     context: ErrorContext) -> RecoveryAction:
        """レート制限エラーの処理"""
        # Retry-Afterヘッダーがあれば使用
        retry_after = context.metadata.get("retry_after", 60)
        
        return RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            delay_seconds=retry_after,
            max_retries=3,
            metadata={"reason": "rate_limit"}
        )
    
    async def _handle_auth_error(self,
                               context: ErrorContext) -> RecoveryAction:
        """認証エラーの処理"""
        # 認証エラーは即座に失敗
        return RecoveryAction(
            strategy=RecoveryStrategy.FAIL_FAST,
            metadata={"reason": "authentication_failed"}
        )
    
    async def _handle_validation_error(self,
                                     context: ErrorContext) -> RecoveryAction:
        """バリデーションエラーの処理"""
        # バリデーションエラーは修正が必要なので即座に失敗
        return RecoveryAction(
            strategy=RecoveryStrategy.FAIL_FAST,
            metadata={"reason": "validation_failed"}
        )
    
    async def _handle_consciousness_error(self,
                                        context: ErrorContext) -> RecoveryAction:
        """意識計算エラーの処理"""
        # デフォルトのΦ値でフォールバック
        return RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            fallback_value={"phi_value": 0.0, "mode": "dormant"},
            metadata={"reason": "consciousness_computation_failed"}
        )
    
    async def _handle_llm_error(self,
                              context: ErrorContext) -> RecoveryAction:
        """LLM生成エラーの処理"""
        # より簡単なモデルにデグレード
        return RecoveryAction(
            strategy=RecoveryStrategy.DEGRADE,
            metadata={
                "fallback_model": "gpt-35-turbo",
                "reduced_features": True
            }
        )
    
    async def _handle_system_error(self,
                                 context: ErrorContext) -> RecoveryAction:
        """システムエラーの処理"""
        # システムエラーは慎重にリトライ
        return RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            delay_seconds=30,
            max_retries=2,
            metadata={"reason": "system_error"}
        )
    
    async def _handle_unknown_error(self,
                                  context: ErrorContext) -> RecoveryAction:
        """未知のエラーの処理"""
        # 安全のため即座に失敗
        return RecoveryAction(
            strategy=RecoveryStrategy.FAIL_FAST,
            metadata={"reason": "unknown_error"}
        )
    
    def _check_circuit_breaker(self, component: str) -> bool:
        """サーキットブレーカーの状態をチェック"""
        breaker = self._circuit_breakers[component]
        
        if not breaker.is_open:
            return False
        
        # タイムアウトチェック
        if breaker.last_failure_time:
            time_since_failure = (
                datetime.now() - breaker.last_failure_time
            ).total_seconds()
            
            if time_since_failure > self.circuit_breaker_timeout_seconds:
                # ハーフオープン状態に移行
                breaker.is_open = False
                breaker.failure_count = 0
                self.logger.info(f"Circuit breaker for {component} is half-open")
                return False
        
        return True
    
    def _update_circuit_breaker(self,
                              component: str,
                              error_context: ErrorContext,
                              recovery_action: RecoveryAction):
        """サーキットブレーカーの状態を更新"""
        breaker = self._circuit_breakers[component]
        
        if recovery_action.strategy == RecoveryStrategy.FAIL_FAST:
            # 失敗をカウント
            breaker.failure_count += 1
            breaker.last_failure_time = datetime.now()
            breaker.consecutive_successes = 0
            
            # 閾値を超えたらオープン
            if breaker.failure_count >= self.circuit_breaker_threshold:
                breaker.is_open = True
                self.logger.warning(
                    f"Circuit breaker for {component} is now OPEN "
                    f"(failures: {breaker.failure_count})"
                )
        else:
            # 成功をカウント
            breaker.consecutive_successes += 1
            breaker.last_success_time = datetime.now()
            
            # 連続成功でリセット
            if breaker.consecutive_successes >= 3:
                breaker.failure_count = 0
                breaker.is_open = False
    
    def register_recovery_strategy(self,
                                 name: str,
                                 strategy: Callable[[ErrorContext], RecoveryAction]):
        """カスタムリカバリストラテジーを登録"""
        self._recovery_strategies[name] = strategy
    
    def get_error_statistics(self,
                           time_window_minutes: int = 60) -> Dict[str, Any]:
        """エラー統計を取得"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_errors = [
            err for err in self._error_history
            if err.timestamp > cutoff_time
        ]
        
        # エラータイプ別の集計
        error_by_type = defaultdict(int)
        error_by_component = defaultdict(int)
        
        for err in recent_errors:
            error_by_type[err.error_type.value] += 1
            error_by_component[err.component] += 1
        
        # サーキットブレーカーの状態
        circuit_breaker_status = {
            component: {
                "is_open": breaker.is_open,
                "failure_count": breaker.failure_count,
                "consecutive_successes": breaker.consecutive_successes
            }
            for component, breaker in self._circuit_breakers.items()
        }
        
        return {
            "total_errors": len(recent_errors),
            "errors_by_type": dict(error_by_type),
            "errors_by_component": dict(error_by_component),
            "circuit_breakers": circuit_breaker_status,
            "time_window_minutes": time_window_minutes
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """システムのヘルスチェック"""
        # エラー率の計算
        recent_errors = self.get_error_statistics(time_window_minutes=5)
        error_rate = recent_errors["total_errors"] / 5.0  # エラー/分
        
        # 開いているサーキットブレーカーの数
        open_breakers = sum(
            1 for breaker in self._circuit_breakers.values()
            if breaker.is_open
        )
        
        # ヘルス判定
        is_healthy = error_rate < 10 and open_breakers == 0
        
        return {
            "is_healthy": is_healthy,
            "error_rate_per_minute": error_rate,
            "open_circuit_breakers": open_breakers,
            "total_components": len(self._circuit_breakers)
        }


class ErrorRecoveryDecorator:
    """
    エラーリカバリデコレーター
    
    関数やメソッドにエラーハンドリングとリカバリを追加。
    """
    
    def __init__(self, error_handler: ConsciousnessSystemErrorHandler):
        self.error_handler = error_handler
    
    def with_recovery(self,
                     component: str,
                     operation: str,
                     fallback_value: Optional[Any] = None):
        """リカバリ機能付きデコレーター"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                retry_count = 0
                last_error = None
                
                while True:
                    try:
                        # 関数を実行
                        result = await func(*args, **kwargs)
                        return result
                        
                    except Exception as e:
                        last_error = e
                        
                        # エラーハンドリング
                        recovery_action = await self.error_handler.handle_error(
                            error=e,
                            component=component,
                            operation=operation,
                            context={
                                "args": str(args)[:100],
                                "kwargs": str(kwargs)[:100],
                                "retry_count": retry_count
                            }
                        )
                        
                        # リカバリアクションに基づいて処理
                        if recovery_action.strategy == RecoveryStrategy.RETRY:
                            if retry_count < recovery_action.max_retries:
                                retry_count += 1
                                await asyncio.sleep(recovery_action.delay_seconds)
                                continue
                            else:
                                if fallback_value is not None:
                                    return fallback_value
                                raise
                        
                        elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
                            return recovery_action.fallback_value or fallback_value
                        
                        elif recovery_action.strategy == RecoveryStrategy.FAIL_FAST:
                            raise
                        
                        elif recovery_action.strategy == RecoveryStrategy.CIRCUIT_BREAK:
                            if fallback_value is not None:
                                return fallback_value
                            raise Exception(f"Circuit breaker is open for {component}")
                        
                        elif recovery_action.strategy == RecoveryStrategy.DEGRADE:
                            # デグレードモードで再実行
                            kwargs["degrade_mode"] = True
                            kwargs.update(recovery_action.metadata)
                            continue
                
                if last_error:
                    raise last_error
            
            return wrapper
        return decorator