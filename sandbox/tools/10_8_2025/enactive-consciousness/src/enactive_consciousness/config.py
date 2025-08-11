"""Unified configuration management for the enactive consciousness framework.

This module implements Martin Fowler's Extract Class refactoring to centralize
configuration management and provide type-safe configuration handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

import jax
from pydantic import BaseModel, ConfigDict, Field, validator

# Optional imports
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from omegaconf import DictConfig, OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

from .types import ConsciousnessLevel, CouplingStrength

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception for configuration-related errors."""
    pass


class ConfigSource(Enum):
    """Sources for configuration data."""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    OVERRIDE = "override"


@dataclass(frozen=True)
class ConfigMetadata:
    """Metadata for configuration tracking."""
    source: ConfigSource
    timestamp: float
    version: str = "1.0"
    validated: bool = False


class SystemConfig(BaseModel):
    """System-wide configuration parameters."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='forbid',
    )
    
    # Logging configuration
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file: Optional[Path] = Field(default=None)
    
    # Performance configuration
    enable_jit: bool = Field(default=True)
    enable_memory_optimization: bool = Field(default=True)
    max_memory_mb: float = Field(default=1024.0, gt=0.0)
    jit_static_args: tuple = Field(default=())
    
    # Device configuration
    device: str = Field(default="auto", pattern=r"^(auto|cpu|gpu|tpu)$")
    precision: str = Field(default="float32", pattern=r"^(float16|float32|float64)$")
    
    # Debugging configuration
    debug_mode: bool = Field(default=False)
    profile_execution: bool = Field(default=False)
    save_intermediates: bool = Field(default=False)


class TemporalConfig(BaseModel):
    """Configuration for temporal consciousness processing."""
    
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    # Phenomenological parameters
    retention_depth: int = Field(default=10, ge=1, le=100)
    protention_horizon: int = Field(default=5, ge=1, le=50)
    primal_impression_width: float = Field(default=0.1, gt=0.0, le=1.0)
    temporal_synthesis_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    temporal_decay_factor: float = Field(default=0.95, gt=0.0, le=1.0)
    
    # Neural architecture parameters
    state_dim: int = Field(default=128, ge=32, le=1024)
    attention_heads: int = Field(default=4, ge=1, le=16)
    mlp_width_multiplier: float = Field(default=2.0, gt=1.0, le=4.0)
    mlp_depth: int = Field(default=2, ge=1, le=8)
    
    # Performance parameters
    memory_efficient: bool = Field(default=True)
    gradient_checkpointing: bool = Field(default=False)


class EmbodimentConfig(BaseModel):
    """Configuration for embodied processing."""
    
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    # Body schema parameters
    proprioceptive_dim: int = Field(default=64, ge=16, le=512)
    motor_dim: int = Field(default=32, ge=8, le=256)
    body_map_resolution: tuple[int, int] = Field(default=(20, 20))
    
    # Processing parameters
    boundary_sensitivity: float = Field(default=0.7, ge=0.0, le=1.0)
    schema_adaptation_rate: float = Field(default=0.01, gt=0.0, le=0.1)
    motor_intention_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Architecture parameters
    hidden_dim_multiplier: float = Field(default=2.0, gt=1.0, le=4.0)
    attention_dim: int = Field(default=128, ge=32, le=512)
    
    @validator('body_map_resolution')
    def validate_map_resolution(cls, v):
        if len(v) != 2 or any(dim <= 0 for dim in v):
            raise ValueError("body_map_resolution must be tuple of two positive integers")
        return v


class CouplingConfig(BaseModel):
    """Configuration for structural coupling."""
    
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    # Coupling parameters
    coupling_strength: CouplingStrength = Field(default=CouplingStrength.MODERATE)
    environmental_dim: int = Field(default=128, ge=32, le=1024)
    agent_dim: int = Field(default=128, ge=32, le=1024)
    
    # Dynamics parameters
    stability_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    perturbation_sensitivity: float = Field(default=0.1, gt=0.0, le=1.0)
    adaptation_rate: float = Field(default=0.02, gt=0.0, le=0.1)
    
    # History parameters
    history_length: int = Field(default=50, ge=10, le=200)


class ConsciousnessConfig(BaseModel):
    """Configuration for consciousness integration."""
    
    model_config = ConfigDict(validate_assignment=True, extra='forbid')
    
    # Consciousness parameters
    consciousness_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    integration_method: str = Field(default="weighted_sum", pattern=r"^(weighted_sum|attention|gating)$")
    levels_enabled: list[ConsciousnessLevel] = Field(
        default=[ConsciousnessLevel.BASIC, ConsciousnessLevel.REFLECTIVE]
    )
    
    # Integration architecture
    integration_dim: int = Field(default=256, ge=64, le=1024)
    meta_cognitive_depth: int = Field(default=3, ge=1, le=8)
    
    # Performance parameters
    parallel_processing: bool = Field(default=True)
    async_integration: bool = Field(default=False)


class UnifiedConfig(BaseModel):
    """Unified configuration for the entire framework."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='forbid',
    )
    
    # System configuration
    system: SystemConfig = Field(default_factory=SystemConfig)
    
    # Component configurations
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    embodiment: EmbodimentConfig = Field(default_factory=EmbodimentConfig)
    coupling: CouplingConfig = Field(default_factory=CouplingConfig)
    consciousness: ConsciousnessConfig = Field(default_factory=ConsciousnessConfig)
    
    # Metadata
    metadata: ConfigMetadata = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set default metadata if not provided
        if self.metadata is None:
            self.metadata = ConfigMetadata(
                source=ConfigSource.DEFAULT,
                timestamp=jax.random.PRNGKey(0)[0].item(),  # Simple timestamp
                validated=True,
            )
    
    @validator('temporal')
    def validate_temporal_consistency(cls, v, values):
        """Cross-validate temporal configuration."""
        # Ensure state dimensions are consistent
        if 'embodiment' in values:
            embodiment = values['embodiment']
            if v.state_dim != embodiment.proprioceptive_dim:
                logger.warning(
                    f"Temporal state_dim ({v.state_dim}) != embodiment proprioceptive_dim "
                    f"({embodiment.proprioceptive_dim}). This may cause integration issues."
                )
        return v
    
    def get_jax_config(self) -> Dict[str, Any]:
        """Get JAX-specific configuration."""
        return {
            'jax_enable_x64': self.system.precision == 'float64',
            'jax_platform_name': None if self.system.device == 'auto' else self.system.device,
            'jax_debug_nans': self.system.debug_mode,
        }
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        level = getattr(logging, self.system.log_level.upper())
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format=self.system.log_format,
            filename=self.system.log_file,
        )
        
        # Set JAX logging level
        logging.getLogger('jax').setLevel(level)
        
        logger.info(f"Logging configured: level={self.system.log_level}")


class ConfigManager:
    """Manager for configuration loading, validation, and updates."""
    
    def __init__(self):
        self._config: Optional[UnifiedConfig] = None
        self._config_history: list[UnifiedConfig] = []
        self._watchers: list[callable] = []
    
    def load_from_file(self, config_path: Union[str, Path]) -> UnifiedConfig:
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            if config_path.suffix in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ConfigurationError("PyYAML not installed for YAML support")
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                import json
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
            else:
                raise ConfigurationError(f"Unsupported config file format: {config_path.suffix}")
            
            config = UnifiedConfig(**config_dict)
            config.metadata = ConfigMetadata(
                source=ConfigSource.FILE,
                timestamp=config_path.stat().st_mtime,
                validated=True,
            )
            
            self._set_config(config)
            logger.info(f"Configuration loaded from: {config_path}")
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}")
    
    def load_from_dict(self, config_dict: Dict[str, Any], source: ConfigSource = ConfigSource.OVERRIDE) -> UnifiedConfig:
        """Load configuration from dictionary."""
        try:
            config = UnifiedConfig(**config_dict)
            config.metadata = ConfigMetadata(
                source=source,
                timestamp=jax.random.PRNGKey(0)[0].item(),
                validated=True,
            )
            
            self._set_config(config)
            logger.info(f"Configuration loaded from dictionary (source: {source.value})")
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from dictionary: {e}")
    
    def get_config(self) -> UnifiedConfig:
        """Get current configuration."""
        if self._config is None:
            logger.info("No configuration loaded, using defaults")
            self._config = UnifiedConfig()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> UnifiedConfig:
        """Update configuration with new values."""
        current_config = self.get_config()
        
        # Create deep copy of current config
        config_dict = current_config.model_dump()
        
        # Apply updates
        self._deep_update(config_dict, updates)
        
        # Create new config
        new_config = UnifiedConfig(**config_dict)
        new_config.metadata = ConfigMetadata(
            source=ConfigSource.OVERRIDE,
            timestamp=jax.random.PRNGKey(0)[0].item(),
            validated=True,
        )
        
        self._set_config(new_config)
        logger.info(f"Configuration updated with {len(updates)} changes")
        return new_config
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save current configuration to file."""
        if self._config is None:
            raise ConfigurationError("No configuration to save")
        
        config_path = Path(config_path)
        config_dict = self._config.model_dump(exclude={'metadata'})
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ConfigurationError("PyYAML not installed for YAML support")
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif config_path.suffix == '.json':
            import json
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ConfigurationError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Configuration saved to: {config_path}")
    
    def add_watcher(self, callback: callable) -> None:
        """Add configuration change watcher."""
        self._watchers.append(callback)
    
    def _set_config(self, config: UnifiedConfig) -> None:
        """Set configuration and notify watchers."""
        if self._config is not None:
            self._config_history.append(self._config)
        
        self._config = config
        
        # Setup system configuration
        config.setup_logging()
        
        # Notify watchers
        for watcher in self._watchers:
            try:
                watcher(config)
            except Exception as e:
                logger.error(f"Configuration watcher error: {e}")
    
    def _deep_update(self, base_dict: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep update dictionary with new values."""
        for key, value in updates.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_history(self) -> list[UnifiedConfig]:
        """Get configuration history."""
        return self._config_history.copy()
    
    def rollback(self, steps: int = 1) -> UnifiedConfig:
        """Rollback configuration to previous version."""
        if len(self._config_history) < steps:
            raise ConfigurationError(f"Cannot rollback {steps} steps, only {len(self._config_history)} available")
        
        # Get previous config
        previous_config = self._config_history[-steps]
        
        # Remove rolled back configs from history
        self._config_history = self._config_history[:-steps]
        
        # Set as current
        self._set_config(previous_config)
        
        logger.info(f"Configuration rolled back {steps} steps")
        return previous_config


# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions
def get_config() -> UnifiedConfig:
    """Get current configuration."""
    return config_manager.get_config()

def load_config(config_path: Union[str, Path]) -> UnifiedConfig:
    """Load configuration from file."""
    return config_manager.load_from_file(config_path)

def update_config(**kwargs) -> UnifiedConfig:
    """Update configuration with keyword arguments."""
    return config_manager.update_config(kwargs)


# Export public API
__all__ = [
    'SystemConfig',
    'TemporalConfig', 
    'EmbodimentConfig',
    'CouplingConfig',
    'ConsciousnessConfig',
    'UnifiedConfig',
    'ConfigManager',
    'ConfigSource',
    'ConfigMetadata',
    'ConfigurationError',
    'config_manager',
    'get_config',
    'load_config',
    'update_config',
]