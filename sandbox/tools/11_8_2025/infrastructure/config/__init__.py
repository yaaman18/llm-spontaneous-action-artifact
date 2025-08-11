"""
Configuration management for the enactive consciousness system.

Centralized configuration handling following the Configuration pattern
to manage system parameters, framework settings, and environment-specific
configurations.
"""

from .system_config import SystemConfig

__all__ = [
    "SystemConfig",
]