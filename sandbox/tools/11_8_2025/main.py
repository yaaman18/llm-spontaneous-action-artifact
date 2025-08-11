"""
Main entry point for the Enactive Consciousness Framework.

This module serves as the main entry point for the enactive consciousness
system, following Clean Architecture principles with proper dependency
injection and framework-agnostic design.
"""

import asyncio
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from infrastructure.config.system_config import SystemConfig, LogLevel, FrameworkType
from domain.value_objects.consciousness_state import ConsciousnessState
from domain.value_objects.phi_value import PhiValue
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.probability_distribution import ProbabilityDistribution


def setup_logging(config: SystemConfig) -> None:
    """
    Configure logging based on system configuration.
    
    Args:
        config: System configuration containing logging settings
    """
    log_level = getattr(logging, config.log_level.value.upper())
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()  # Remove default handlers
    
    # Console handler
    if config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.log_to_file:
        file_handler = logging.handlers.RotatingFileHandler(
            config.get_log_file_path(),
            maxBytes=config.max_log_file_size,
            backupCount=config.log_backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def initialize_framework(config: SystemConfig) -> None:
    """
    Initialize the computational framework based on configuration.
    
    Args:
        config: System configuration specifying framework
    """
    logger = logging.getLogger(__name__)
    
    try:
        if config.framework == FrameworkType.JAX:
            import jax
            import jax.numpy as jnp
            
            # Configure JAX
            if config.seed is not None:
                key = jax.random.PRNGKey(config.seed)
                logger.info(f"Initialized JAX with seed: {config.seed}")
            
            # Set device
            if config.device != "cpu":
                logger.info(f"JAX will attempt to use device: {config.device}")
            
            logger.info("JAX framework initialized successfully")
            
        elif config.framework == FrameworkType.PYTORCH:
            import torch
            
            # Set device
            if config.is_gpu_available and torch.cuda.is_available():
                device = torch.device(config.device)
                torch.cuda.set_device(device)
                logger.info(f"PyTorch initialized with CUDA device: {device}")
            else:
                logger.info("PyTorch initialized with CPU")
            
            # Set seed
            if config.seed is not None:
                torch.manual_seed(config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(config.seed)
                logger.info(f"PyTorch seed set to: {config.seed}")
        
        else:
            logger.warning(f"Framework {config.framework} not yet implemented")
            
    except ImportError as e:
        logger.error(f"Failed to import required framework: {e}")
        raise
    except Exception as e:
        logger.error(f"Framework initialization failed: {e}")
        raise


async def create_minimal_consciousness_demo() -> ConsciousnessState:
    """
    Create a minimal consciousness state for demonstration.
    
    This function demonstrates the basic usage of the domain objects
    and value objects in the enactive consciousness system.
    
    Returns:
        A minimal consciousness state
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating minimal consciousness demonstration...")
    
    # Create minimal Φ value indicating low-level consciousness
    phi_value = PhiValue(
        value=0.3,
        complexity=1.2,
        integration=0.25,
        system_size=5,
        computation_method="approximate",
        confidence=0.8
    )
    
    # Create basic prediction state
    prediction_state = PredictionState.create_empty(hierarchy_levels=3)
    
    # Create uncertainty distribution
    uncertainty_distribution = ProbabilityDistribution.uniform(10)
    
    # Create minimal consciousness state
    consciousness_state = ConsciousnessState(
        phi_value=phi_value,
        prediction_state=prediction_state,
        uncertainty_distribution=uncertainty_distribution,
        metacognitive_confidence=0.15,
        phenomenological_markers={"demo": True, "system": "minimal"}
    )
    
    logger.info(f"Created consciousness state:")
    logger.info(f"  - Φ value: {consciousness_state.phi_value.value:.3f}")
    logger.info(f"  - Is conscious: {consciousness_state.is_conscious}")
    logger.info(f"  - Consciousness level: {consciousness_state.consciousness_level:.3f}")
    logger.info(f"  - Metacognitive confidence: {consciousness_state.metacognitive_confidence:.3f}")
    
    return consciousness_state


async def run_system_health_check(config: SystemConfig) -> bool:
    """
    Perform system health check to verify all components are working.
    
    Args:
        config: System configuration
        
    Returns:
        True if all health checks pass
    """
    logger = logging.getLogger(__name__)
    logger.info("Performing system health check...")
    
    try:
        # Check directory structure
        required_dirs = [
            config.data_directory,
            config.model_directory,
            config.log_directory,
            config.cache_directory
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                logger.error(f"Required directory missing: {directory}")
                return False
            logger.debug(f"Directory exists: {directory}")
        
        # Test domain object creation
        consciousness_state = await create_minimal_consciousness_demo()
        if not isinstance(consciousness_state, ConsciousnessState):
            logger.error("Failed to create consciousness state")
            return False
        
        # Verify framework initialization
        framework_config = config.get_framework_config()
        logger.debug(f"Framework config: {framework_config}")
        
        logger.info("✓ All system health checks passed")
        return True
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return False


async def main(config_type: str = "development") -> None:
    """
    Main entry point for the enactive consciousness system.
    
    Args:
        config_type: Type of configuration ("development" or "production")
    """
    # Create system configuration
    if config_type == "production":
        config = SystemConfig.create_production_config()
    else:
        config = SystemConfig.create_development_config()
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("エナクティブ意識フレームワーク (Enactive Consciousness Framework)")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config_type}")
    logger.info(f"Framework: {config.framework.value}")
    logger.info(f"Device: {config.device}")
    logger.info(f"GUI Language: {config.gui_language}")
    logger.info(f"Debug Mode: {config.debug_mode}")
    
    try:
        # Initialize computational framework
        initialize_framework(config)
        
        # Perform system health check
        health_check_passed = await run_system_health_check(config)
        if not health_check_passed:
            logger.error("System health check failed. Exiting.")
            sys.exit(1)
        
        # Create and demonstrate minimal consciousness
        consciousness_state = await create_minimal_consciousness_demo()
        
        logger.info("System initialization completed successfully!")
        logger.info("Ready for consciousness simulation and monitoring.")
        
        # In future versions, this would start the main consciousness loop
        # and GUI interface. For now, we just demonstrate the architecture.
        
        logger.info("System ready. Press Ctrl+C to exit.")
        
        # Keep the system running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
            
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        if config.debug_mode:
            logger.exception("Full exception trace:")
        sys.exit(1)
    
    finally:
        logger.info("System shutdown completed")


def cli_entry_point() -> None:
    """
    Command-line interface entry point.
    
    Handles command-line arguments and starts the main system.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enactive Consciousness Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with development configuration
  python main.py --production      # Run with production configuration
  python main.py --debug           # Run with debug logging
        """
    )
    
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production configuration"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true", 
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="起動時にGUIモニターを開く"
    )
    
    args = parser.parse_args()
    
    # Determine configuration type
    if args.production:
        config_type = "production"
    else:
        config_type = "development"
    
    # Run the main system
    try:
        if args.gui:
            # GUIモード
            from gui.consciousness_monitor import ConsciousnessMonitor
            print("GUIモニターを起動しています...")
            app = ConsciousnessMonitor()
            app.run()
        else:
            # CLIモード
            asyncio.run(main(config_type))
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_entry_point()