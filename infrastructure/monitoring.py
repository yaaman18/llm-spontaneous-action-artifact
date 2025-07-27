"""
監視とロギングシステム
廣里敏明（Hirosato Gamo）による実装

プロダクション環境での可観測性とデバッグを支援。
"""
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import logging
from enum import Enum
import numpy as np
from pathlib import Path


class MetricType(Enum):
    """メトリクスタイプ"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """アラートレベル"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """メトリクス定義"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class Alert:
    """アラート定義"""
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    metric_name: Optional[str] = None
    threshold: Optional[float] = None
    actual_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """ヘルスチェック結果"""
    component: str
    is_healthy: bool
    latency_ms: float
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsciousnessSystemMonitor:
    """
    意識システム監視
    
    LLMと意識コアシステムの統合環境を監視し、
    パフォーマンスと健全性を追跡する。
    """
    
    def __init__(self,
                 log_dir: Optional[Path] = None,
                 metrics_retention_hours: int = 24,
                 alert_retention_hours: int = 168,  # 1週間
                 enable_prometheus: bool = True):
        """
        Args:
            log_dir: ログディレクトリ
            metrics_retention_hours: メトリクス保持時間
            alert_retention_hours: アラート保持時間
            enable_prometheus: Prometheus形式の出力を有効化
        """
        self.log_dir = log_dir or Path("./logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics_retention_hours = metrics_retention_hours
        self.alert_retention_hours = alert_retention_hours
        self.enable_prometheus = enable_prometheus
        
        # メトリクスストレージ
        self._metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=metrics_retention_hours * 60)  # 分単位
        )
        
        # アラートストレージ
        self._alerts: deque = deque(maxlen=alert_retention_hours * 60)
        
        # アラートルール
        self._alert_rules: List[Dict[str, Any]] = []
        
        # ヘルスチェック関数
        self._health_checks: Dict[str, Callable] = {}
        
        # ロガー設定
        self._setup_logging()
        
        # 定期的なクリーンアップタスク
        self._cleanup_task = None
    
    def _setup_logging(self):
        """ロギング設定"""
        # メトリクス用ロガー
        self.metrics_logger = logging.getLogger(f"{__name__}.metrics")
        metrics_handler = logging.FileHandler(
            self.log_dir / f"metrics_{datetime.now():%Y%m%d}.log"
        )
        metrics_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.metrics_logger.addHandler(metrics_handler)
        self.metrics_logger.setLevel(logging.INFO)
        
        # アラート用ロガー
        self.alert_logger = logging.getLogger(f"{__name__}.alerts")
        alert_handler = logging.FileHandler(
            self.log_dir / f"alerts_{datetime.now():%Y%m%d}.log"
        )
        alert_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.alert_logger.addHandler(alert_handler)
        self.alert_logger.setLevel(logging.WARNING)
    
    async def start(self):
        """監視システムを開始"""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self.metrics_logger.info("Monitoring system started")
    
    async def stop(self):
        """監視システムを停止"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.metrics_logger.info("Monitoring system stopped")
    
    def record_metric(self,
                     name: str,
                     value: float,
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None,
                     unit: Optional[str] = None):
        """
        メトリクスを記録
        
        Args:
            name: メトリクス名
            value: 値
            metric_type: メトリクスタイプ
            labels: ラベル
            unit: 単位
        """
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            unit=unit
        )
        
        # メトリクスストレージに追加
        key = self._metric_key(name, labels)
        self._metrics[key].append(metric)
        
        # ログに記録
        self.metrics_logger.info(
            f"Metric recorded: {name}={value}{unit or ''} {labels or {}}"
        )
        
        # アラートルールをチェック
        asyncio.create_task(self._check_alert_rules(metric))
    
    def record_phi_metrics(self,
                          phi_value: float,
                          response_time_ms: float,
                          context_id: str,
                          response_mode: str):
        """意識システム固有のメトリクスを記録"""
        # Φ値
        self.record_metric(
            "consciousness_phi_value",
            phi_value,
            MetricType.GAUGE,
            {"context_id": context_id, "mode": response_mode}
        )
        
        # 応答時間
        self.record_metric(
            "llm_response_time_ms",
            response_time_ms,
            MetricType.HISTOGRAM,
            {"context_id": context_id, "mode": response_mode},
            "ms"
        )
        
        # 意識レベルの分布
        consciousness_level = self._categorize_phi(phi_value)
        self.record_metric(
            "consciousness_level_distribution",
            1,
            MetricType.COUNTER,
            {"level": consciousness_level}
        )
    
    def record_llm_metrics(self,
                          model: str,
                          tokens_used: int,
                          cost: float,
                          latency_ms: float,
                          error: bool = False):
        """LLM固有のメトリクスを記録"""
        # トークン使用量
        self.record_metric(
            "llm_tokens_used",
            tokens_used,
            MetricType.COUNTER,
            {"model": model}
        )
        
        # コスト
        self.record_metric(
            "llm_cost_usd",
            cost,
            MetricType.COUNTER,
            {"model": model},
            "USD"
        )
        
        # レイテンシ
        self.record_metric(
            "llm_latency_ms",
            latency_ms,
            MetricType.HISTOGRAM,
            {"model": model},
            "ms"
        )
        
        # エラー率
        if error:
            self.record_metric(
                "llm_errors",
                1,
                MetricType.COUNTER,
                {"model": model}
            )
    
    def add_alert_rule(self,
                      metric_name: str,
                      condition: str,  # "gt", "lt", "eq"
                      threshold: float,
                      level: AlertLevel,
                      message_template: str):
        """
        アラートルールを追加
        
        Args:
            metric_name: 監視するメトリクス名
            condition: 条件（gt: より大きい, lt: より小さい, eq: 等しい）
            threshold: 閾値
            level: アラートレベル
            message_template: メッセージテンプレート
        """
        self._alert_rules.append({
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "level": level,
            "message_template": message_template
        })
    
    async def _check_alert_rules(self, metric: Metric):
        """アラートルールをチェック"""
        for rule in self._alert_rules:
            if metric.name != rule["metric_name"]:
                continue
            
            # 条件をチェック
            triggered = False
            if rule["condition"] == "gt" and metric.value > rule["threshold"]:
                triggered = True
            elif rule["condition"] == "lt" and metric.value < rule["threshold"]:
                triggered = True
            elif rule["condition"] == "eq" and metric.value == rule["threshold"]:
                triggered = True
            
            if triggered:
                alert = Alert(
                    id=f"{metric.name}_{datetime.now().timestamp()}",
                    level=rule["level"],
                    message=rule["message_template"].format(
                        value=metric.value,
                        threshold=rule["threshold"],
                        metric=metric.name
                    ),
                    timestamp=datetime.now(),
                    metric_name=metric.name,
                    threshold=rule["threshold"],
                    actual_value=metric.value,
                    metadata=metric.labels
                )
                
                await self._fire_alert(alert)
    
    async def _fire_alert(self, alert: Alert):
        """アラートを発火"""
        self._alerts.append(alert)
        
        # ログレベルに応じて記録
        log_method = {
            AlertLevel.INFO: self.alert_logger.info,
            AlertLevel.WARNING: self.alert_logger.warning,
            AlertLevel.ERROR: self.alert_logger.error,
            AlertLevel.CRITICAL: self.alert_logger.critical
        }
        
        log_method[alert.level](
            f"Alert: {alert.message} (metric={alert.metric_name}, "
            f"value={alert.actual_value}, threshold={alert.threshold})"
        )
    
    def register_health_check(self,
                            component: str,
                            check_func: Callable[[], HealthCheckResult]):
        """ヘルスチェック関数を登録"""
        self._health_checks[component] = check_func
    
    async def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """全てのヘルスチェックを実行"""
        results = {}
        
        for component, check_func in self._health_checks.items():
            try:
                start_time = datetime.now()
                
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                result.latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                results[component] = result
                
                # ヘルスチェック結果をメトリクスとして記録
                self.record_metric(
                    f"health_check_{component}",
                    1 if result.is_healthy else 0,
                    MetricType.GAUGE,
                    {"component": component}
                )
                
            except Exception as e:
                results[component] = HealthCheckResult(
                    component=component,
                    is_healthy=False,
                    latency_ms=0,
                    message=str(e)
                )
        
        return results
    
    def get_metrics_summary(self,
                          metric_name: Optional[str] = None,
                          time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        メトリクスのサマリーを取得
        
        Args:
            metric_name: 特定のメトリクス名（Noneの場合は全て）
            time_window_minutes: 時間窓（分）
            
        Returns:
            メトリクスサマリー
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        summary = {}
        
        for key, metrics in self._metrics.items():
            if metric_name and not key.startswith(metric_name):
                continue
            
            recent_metrics = [
                m for m in metrics 
                if m.timestamp > cutoff_time
            ]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary[key] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "latest": values[-1]
                }
        
        return summary
    
    def get_recent_alerts(self,
                         level: Optional[AlertLevel] = None,
                         hours: int = 24) -> List[Alert]:
        """最近のアラートを取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        alerts = [
            alert for alert in self._alerts
            if alert.timestamp > cutoff_time
        ]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def export_prometheus_metrics(self) -> str:
        """Prometheus形式でメトリクスをエクスポート"""
        if not self.enable_prometheus:
            return ""
        
        lines = []
        
        for key, metrics in self._metrics.items():
            if not metrics:
                continue
            
            latest_metric = metrics[-1]
            metric_name = latest_metric.name.replace(".", "_")
            
            # メトリクスのヘルプとタイプ
            lines.append(f"# HELP {metric_name} {latest_metric.name}")
            lines.append(f"# TYPE {metric_name} {latest_metric.type.value}")
            
            # ラベル文字列の構築
            if latest_metric.labels:
                label_str = ",".join(
                    f'{k}="{v}"' for k, v in latest_metric.labels.items()
                )
                label_str = f"{{{label_str}}}"
            else:
                label_str = ""
            
            # メトリクス値
            lines.append(f"{metric_name}{label_str} {latest_metric.value}")
        
        return "\n".join(lines)
    
    def _metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """メトリクスのキーを生成"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _categorize_phi(self, phi_value: float) -> str:
        """Φ値をカテゴリ分類"""
        if phi_value < 1.0:
            return "dormant"
        elif phi_value < 3.0:
            return "emerging"
        elif phi_value < 6.0:
            return "conscious"
        else:
            return "highly_conscious"
    
    async def _periodic_cleanup(self):
        """定期的なクリーンアップ"""
        while True:
            try:
                await asyncio.sleep(3600)  # 1時間ごと
                
                # 古いメトリクスを削除
                cutoff_time = datetime.now() - timedelta(
                    hours=self.metrics_retention_hours
                )
                
                for metrics in self._metrics.values():
                    while metrics and metrics[0].timestamp < cutoff_time:
                        metrics.popleft()
                
                # ログファイルのローテーション
                self._rotate_logs()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.alert_logger.error(f"Cleanup error: {str(e)}")
    
    def _rotate_logs(self):
        """ログファイルのローテーション"""
        # 新しい日付のログファイルを作成
        for logger, name in [(self.metrics_logger, "metrics"), 
                            (self.alert_logger, "alerts")]:
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
            
            new_handler = logging.FileHandler(
                self.log_dir / f"{name}_{datetime.now():%Y%m%d}.log"
            )
            new_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(new_handler)