"""
プロダクション設定
廣里敏明（Hirosato Gamo）による実装

環境に応じた設定管理を提供。
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class Environment(Enum):
    """環境タイプ"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI設定"""
    api_key: str
    endpoint: str
    api_version: str = "2024-02-01"
    default_model: str = "gpt-4-turbo"
    max_retries: int = 3
    timeout_seconds: int = 30
    
    # レート制限設定
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    
    # コスト制限
    daily_cost_limit_usd: float = 100.0
    alert_cost_threshold_usd: float = 50.0


@dataclass
class ConsciousnessConfig:
    """意識システム設定"""
    phi_threshold: float = 3.0
    min_subsystem_size: int = 3
    max_subsystem_size: Optional[int] = 20
    
    # 履歴設定
    state_history_size: int = 100
    phi_history_size: int = 100
    
    # パフォーマンス設定
    enable_caching: bool = True
    cache_ttl_seconds: int = 300


@dataclass
class MonitoringConfig:
    """監視設定"""
    enable_monitoring: bool = True
    enable_prometheus: bool = True
    
    # メトリクス設定
    metrics_retention_hours: int = 24
    alert_retention_hours: int = 168
    
    # ログ設定
    log_dir: str = "./logs"
    log_rotation_days: int = 7
    log_level: str = "INFO"
    
    # アラート設定
    alert_email_enabled: bool = False
    alert_email_recipients: list = field(default_factory=list)
    alert_webhook_url: Optional[str] = None


@dataclass
class ErrorHandlingConfig:
    """エラーハンドリング設定"""
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    
    # リトライ設定
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    retry_max_delay_seconds: float = 60.0
    
    # エラー履歴
    max_error_history: int = 1000


@dataclass
class SystemConfig:
    """システム全体の設定"""
    environment: Environment
    azure_openai: AzureOpenAIConfig
    consciousness: ConsciousnessConfig
    monitoring: MonitoringConfig
    error_handling: ErrorHandlingConfig
    
    # パフォーマンス設定
    enable_async: bool = True
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    
    # セキュリティ設定
    enable_request_validation: bool = True
    max_prompt_length: int = 4000
    allowed_origins: list = field(default_factory=list)


class ConfigurationManager:
    """
    設定管理マネージャー
    
    環境に応じた設定の読み込みと管理を行う。
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Args:
            config_dir: 設定ファイルディレクトリ
        """
        self.config_dir = config_dir or Path("./config")
        self._config_cache: Dict[Environment, SystemConfig] = {}
    
    def load_config(self, 
                   environment: Optional[Environment] = None) -> SystemConfig:
        """
        設定を読み込み
        
        Args:
            environment: 環境（Noneの場合は環境変数から判定）
            
        Returns:
            システム設定
        """
        # 環境の判定
        if environment is None:
            env_str = os.environ.get("ENVIRONMENT", "development")
            environment = Environment(env_str.lower())
        
        # キャッシュチェック
        if environment in self._config_cache:
            return self._config_cache[environment]
        
        # 基本設定の読み込み
        base_config = self._load_base_config()
        
        # 環境固有の設定の読み込み
        env_config = self._load_environment_config(environment)
        
        # 環境変数からの上書き
        final_config = self._override_from_env(base_config, env_config)
        
        # キャッシュに保存
        self._config_cache[environment] = final_config
        
        return final_config
    
    def _load_base_config(self) -> Dict[str, Any]:
        """基本設定を読み込み"""
        base_path = self.config_dir / "base.json"
        
        if base_path.exists():
            with open(base_path, 'r') as f:
                return json.load(f)
        
        # デフォルト設定
        return {
            "azure_openai": {
                "api_version": "2024-02-01",
                "default_model": "gpt-4-turbo",
                "max_retries": 3,
                "timeout_seconds": 30,
                "requests_per_minute": 60,
                "tokens_per_minute": 90000,
                "daily_cost_limit_usd": 100.0,
                "alert_cost_threshold_usd": 50.0
            },
            "consciousness": {
                "phi_threshold": 3.0,
                "min_subsystem_size": 3,
                "max_subsystem_size": 20,
                "state_history_size": 100,
                "phi_history_size": 100,
                "enable_caching": True,
                "cache_ttl_seconds": 300
            },
            "monitoring": {
                "enable_monitoring": True,
                "enable_prometheus": True,
                "metrics_retention_hours": 24,
                "alert_retention_hours": 168,
                "log_dir": "./logs",
                "log_rotation_days": 7,
                "log_level": "INFO"
            },
            "error_handling": {
                "enable_circuit_breaker": True,
                "circuit_breaker_threshold": 5,
                "circuit_breaker_timeout_seconds": 60,
                "max_retries": 3,
                "retry_backoff_base": 2.0,
                "retry_max_delay_seconds": 60.0,
                "max_error_history": 1000
            }
        }
    
    def _load_environment_config(self, 
                               environment: Environment) -> Dict[str, Any]:
        """環境固有の設定を読み込み"""
        env_path = self.config_dir / f"{environment.value}.json"
        
        if env_path.exists():
            with open(env_path, 'r') as f:
                return json.load(f)
        
        # 環境別のデフォルト設定
        env_defaults = {
            Environment.DEVELOPMENT: {
                "enable_async": True,
                "max_concurrent_requests": 5,
                "request_timeout_seconds": 30,
                "enable_request_validation": False,
                "monitoring": {
                    "log_level": "DEBUG"
                }
            },
            Environment.STAGING: {
                "enable_async": True,
                "max_concurrent_requests": 10,
                "request_timeout_seconds": 30,
                "enable_request_validation": True,
                "monitoring": {
                    "log_level": "INFO"
                }
            },
            Environment.PRODUCTION: {
                "enable_async": True,
                "max_concurrent_requests": 20,
                "request_timeout_seconds": 15,
                "enable_request_validation": True,
                "monitoring": {
                    "log_level": "WARNING",
                    "alert_email_enabled": True
                }
            }
        }
        
        return env_defaults.get(environment, {})
    
    def _override_from_env(self,
                         base_config: Dict[str, Any],
                         env_config: Dict[str, Any]) -> SystemConfig:
        """環境変数から設定を上書き"""
        # 設定をマージ
        config_dict = self._deep_merge(base_config, env_config)
        
        # 環境変数から上書き
        # Azure OpenAI
        if "AZURE_OPENAI_API_KEY" in os.environ:
            config_dict.setdefault("azure_openai", {})["api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
        
        if "AZURE_OPENAI_ENDPOINT" in os.environ:
            config_dict.setdefault("azure_openai", {})["endpoint"] = os.environ["AZURE_OPENAI_ENDPOINT"]
        
        # 意識システム
        if "PHI_THRESHOLD" in os.environ:
            config_dict.setdefault("consciousness", {})["phi_threshold"] = float(os.environ["PHI_THRESHOLD"])
        
        # 監視
        if "ENABLE_MONITORING" in os.environ:
            config_dict.setdefault("monitoring", {})["enable_monitoring"] = os.environ["ENABLE_MONITORING"].lower() == "true"
        
        # SystemConfigオブジェクトを構築
        environment = Environment(os.environ.get("ENVIRONMENT", "development").lower())
        
        return SystemConfig(
            environment=environment,
            azure_openai=AzureOpenAIConfig(**config_dict.get("azure_openai", {})),
            consciousness=ConsciousnessConfig(**config_dict.get("consciousness", {})),
            monitoring=MonitoringConfig(**config_dict.get("monitoring", {})),
            error_handling=ErrorHandlingConfig(**config_dict.get("error_handling", {})),
            enable_async=config_dict.get("enable_async", True),
            max_concurrent_requests=config_dict.get("max_concurrent_requests", 10),
            request_timeout_seconds=config_dict.get("request_timeout_seconds", 30),
            enable_request_validation=config_dict.get("enable_request_validation", True),
            max_prompt_length=config_dict.get("max_prompt_length", 4000),
            allowed_origins=config_dict.get("allowed_origins", [])
        )
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """辞書を深くマージ"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self,
                   config: SystemConfig,
                   environment: Environment,
                   override: bool = False):
        """設定を保存"""
        file_path = self.config_dir / f"{environment.value}.json"
        
        if file_path.exists() and not override:
            raise FileExistsError(f"Config file already exists: {file_path}")
        
        # SystemConfigを辞書に変換
        config_dict = {
            "environment": environment.value,
            "azure_openai": {
                k: v for k, v in config.azure_openai.__dict__.items()
                if k != "api_key"  # APIキーは保存しない
            },
            "consciousness": config.consciousness.__dict__,
            "monitoring": config.monitoring.__dict__,
            "error_handling": config.error_handling.__dict__,
            "enable_async": config.enable_async,
            "max_concurrent_requests": config.max_concurrent_requests,
            "request_timeout_seconds": config.request_timeout_seconds,
            "enable_request_validation": config.enable_request_validation,
            "max_prompt_length": config.max_prompt_length,
            "allowed_origins": config.allowed_origins
        }
        
        # ディレクトリを作成
        self.config_dir.mkdir(exist_ok=True)
        
        # ファイルに保存
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate_config(self, config: SystemConfig) -> List[str]:
        """設定を検証"""
        errors = []
        
        # Azure OpenAI設定の検証
        if not hasattr(config.azure_openai, 'api_key') or not config.azure_openai.api_key:
            errors.append("Azure OpenAI API key is required")
        
        if not hasattr(config.azure_openai, 'endpoint') or not config.azure_openai.endpoint:
            errors.append("Azure OpenAI endpoint is required")
        
        # 意識システム設定の検証
        if config.consciousness.phi_threshold < 0:
            errors.append("Phi threshold must be non-negative")
        
        if config.consciousness.min_subsystem_size < 2:
            errors.append("Minimum subsystem size must be at least 2")
        
        # 監視設定の検証
        if config.monitoring.metrics_retention_hours < 1:
            errors.append("Metrics retention must be at least 1 hour")
        
        # エラーハンドリング設定の検証
        if config.error_handling.circuit_breaker_threshold < 1:
            errors.append("Circuit breaker threshold must be at least 1")
        
        return errors