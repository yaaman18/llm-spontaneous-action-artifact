"""
System Configuration.

Framework-agnostic configuration management for the enactive consciousness
system. Handles loading, validation, and access to system parameters
following the Single Responsibility Principle.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
from enum import Enum


class FrameworkType(Enum):
    """Supported computation frameworks."""
    JAX = "jax"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemConfig:
    """
    System-wide configuration settings.
    
    Immutable configuration object that centralizes all system
    parameters and settings. Follows the Configuration pattern
    to provide framework-agnostic parameter management.
    """
    
    # Framework settings
    framework: FrameworkType = field(default=FrameworkType.JAX)
    device: str = field(default="cpu")  # "cpu", "cuda", "tpu"
    precision: str = field(default="float32")  # "float16", "float32", "float64"
    
    # System paths
    data_directory: Path = field(default_factory=lambda: Path("./data"))
    model_directory: Path = field(default_factory=lambda: Path("./models"))
    log_directory: Path = field(default_factory=lambda: Path("./logs"))
    cache_directory: Path = field(default_factory=lambda: Path("./cache"))
    
    # Logging configuration
    log_level: LogLevel = field(default=LogLevel.INFO)
    log_to_file: bool = field(default=True)
    log_to_console: bool = field(default=True)
    max_log_file_size: int = field(default=10_000_000)  # 10MB
    log_backup_count: int = field(default=5)
    
    # Performance settings
    max_memory_usage: Optional[int] = field(default=None)  # bytes
    num_threads: int = field(default=4)
    enable_jit_compilation: bool = field(default=True)
    enable_memory_mapping: bool = field(default=True)
    
    # Development settings
    debug_mode: bool = field(default=False)
    profiling_enabled: bool = field(default=False)
    seed: Optional[int] = field(default=42)
    
    # System monitoring
    enable_health_checks: bool = field(default=True)
    health_check_interval: int = field(default=30)  # seconds
    performance_monitoring: bool = field(default=True)
    
    # Persistence settings
    database_url: Optional[str] = field(default=None)
    enable_data_persistence: bool = field(default=True)
    backup_interval: int = field(default=3600)  # seconds
    
    # GUI settings
    gui_language: str = field(default="ja")  # Japanese as specified
    gui_theme: str = field(default="dark")
    gui_update_frequency: int = field(default=10)  # Hz
    enable_real_time_visualization: bool = field(default=True)
    
    # Experimental features
    experimental_features: List[str] = field(default_factory=list)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    # Custom configuration
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_paths()
        self._validate_device()
        self._validate_threading()
        self._ensure_directories_exist()

    def _validate_paths(self) -> None:
        """Validate all path configurations."""
        paths_to_check = [
            self.data_directory,
            self.model_directory, 
            self.log_directory,
            self.cache_directory
        ]
        
        for path in paths_to_check:
            if not isinstance(path, Path):
                raise ValueError(f"Path must be Path object, got {type(path)}")

    def _validate_device(self) -> None:
        """Validate device configuration."""
        valid_devices = {"cpu", "cuda", "gpu", "tpu"}
        if not any(self.device.startswith(device) for device in valid_devices):
            raise ValueError(f"Invalid device: {self.device}")

    def _validate_threading(self) -> None:
        """Validate threading configuration."""
        if self.num_threads < 1:
            raise ValueError("Number of threads must be positive")
        if self.num_threads > os.cpu_count() * 2:
            raise ValueError(f"Too many threads: {self.num_threads} > {os.cpu_count() * 2}")

    def _ensure_directories_exist(self) -> None:
        """Create directories if they don't exist."""
        directories = [
            self.data_directory,
            self.model_directory,
            self.log_directory,
            self.cache_directory
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU device is configured."""
        return self.device.startswith(("cuda", "gpu"))

    @property
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug_mode or self.log_level == LogLevel.DEBUG

    @property
    def effective_num_threads(self) -> int:
        """Get effective number of threads based on device."""
        if self.is_gpu_available:
            return 1  # GPU typically uses single thread
        return self.num_threads

    def get_log_file_path(self) -> Path:
        """Get the path for the main log file."""
        return self.log_directory / "consciousness_system.log"

    def get_cache_size_limit(self) -> Optional[int]:
        """Get cache size limit in bytes."""
        if self.max_memory_usage:
            return self.max_memory_usage // 4  # Use 1/4 of memory for cache
        return None

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a specific feature is enabled."""
        return (
            feature_name in self.experimental_features or
            self.feature_flags.get(feature_name, False)
        )

    def get_framework_config(self) -> Dict[str, Any]:
        """Get framework-specific configuration."""
        base_config = {
            "precision": self.precision,
            "device": self.device,
            "enable_jit": self.enable_jit_compilation,
            "num_threads": self.effective_num_threads
        }
        
        if self.framework == FrameworkType.JAX:
            return {
                **base_config,
                "platform": self.device,
                "memory_fraction": 0.8 if self.is_gpu_available else None
            }
        elif self.framework == FrameworkType.PYTORCH:
            return {
                **base_config,
                "device": f"cuda:{self.device.split(':')[-1]}" if self.is_gpu_available else "cpu"
            }
        else:
            return base_config

    def update_custom_param(self, key: str, value: Any) -> 'SystemConfig':
        """
        Create new config with updated custom parameter.
        
        Args:
            key: Parameter key
            value: Parameter value
            
        Returns:
            New SystemConfig with updated parameter
        """
        new_custom_params = self.custom_params.copy()
        new_custom_params[key] = value
        
        # Create new instance with updated parameters
        return SystemConfig(
            framework=self.framework,
            device=self.device,
            precision=self.precision,
            data_directory=self.data_directory,
            model_directory=self.model_directory,
            log_directory=self.log_directory,
            cache_directory=self.cache_directory,
            log_level=self.log_level,
            log_to_file=self.log_to_file,
            log_to_console=self.log_to_console,
            max_log_file_size=self.max_log_file_size,
            log_backup_count=self.log_backup_count,
            max_memory_usage=self.max_memory_usage,
            num_threads=self.num_threads,
            enable_jit_compilation=self.enable_jit_compilation,
            enable_memory_mapping=self.enable_memory_mapping,
            debug_mode=self.debug_mode,
            profiling_enabled=self.profiling_enabled,
            seed=self.seed,
            enable_health_checks=self.enable_health_checks,
            health_check_interval=self.health_check_interval,
            performance_monitoring=self.performance_monitoring,
            database_url=self.database_url,
            enable_data_persistence=self.enable_data_persistence,
            backup_interval=self.backup_interval,
            gui_language=self.gui_language,
            gui_theme=self.gui_theme,
            gui_update_frequency=self.gui_update_frequency,
            enable_real_time_visualization=self.enable_real_time_visualization,
            experimental_features=self.experimental_features.copy(),
            feature_flags=self.feature_flags.copy(),
            custom_params=new_custom_params
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "framework": self.framework.value,
            "device": self.device,
            "precision": self.precision,
            "data_directory": str(self.data_directory),
            "model_directory": str(self.model_directory),
            "log_directory": str(self.log_directory),
            "cache_directory": str(self.cache_directory),
            "log_level": self.log_level.value,
            "log_to_file": self.log_to_file,
            "log_to_console": self.log_to_console,
            "max_log_file_size": self.max_log_file_size,
            "log_backup_count": self.log_backup_count,
            "max_memory_usage": self.max_memory_usage,
            "num_threads": self.num_threads,
            "enable_jit_compilation": self.enable_jit_compilation,
            "enable_memory_mapping": self.enable_memory_mapping,
            "debug_mode": self.debug_mode,
            "profiling_enabled": self.profiling_enabled,
            "seed": self.seed,
            "enable_health_checks": self.enable_health_checks,
            "health_check_interval": self.health_check_interval,
            "performance_monitoring": self.performance_monitoring,
            "database_url": self.database_url,
            "enable_data_persistence": self.enable_data_persistence,
            "backup_interval": self.backup_interval,
            "gui_language": self.gui_language,
            "gui_theme": self.gui_theme,
            "gui_update_frequency": self.gui_update_frequency,
            "enable_real_time_visualization": self.enable_real_time_visualization,
            "experimental_features": self.experimental_features,
            "feature_flags": self.feature_flags,
            "custom_params": self.custom_params
        }

    @classmethod
    def create_development_config(cls) -> 'SystemConfig':
        """Create configuration optimized for development."""
        return cls(
            debug_mode=True,
            log_level=LogLevel.DEBUG,
            profiling_enabled=True,
            enable_jit_compilation=False,  # Faster iteration
            gui_update_frequency=1,  # Slower for debugging
            experimental_features=["debug_introspection", "detailed_logging"]
        )

    @classmethod
    def create_production_config(cls) -> 'SystemConfig':
        """Create configuration optimized for production."""
        return cls(
            debug_mode=False,
            log_level=LogLevel.INFO,
            profiling_enabled=False,
            enable_jit_compilation=True,
            performance_monitoring=True,
            enable_data_persistence=True,
            gui_update_frequency=30  # Higher performance
        )