"""
Consciousness System Configuration and Wiring
Complete dependency injection setup for IIT 4.0 NewbornAI system

This module configures all dependencies and wires up the consciousness system
using the dependency injection container and interface abstractions.

Author: Martin Fowler's Refactoring Agent
Date: 2025-08-03
Version: 1.0.0
"""

import logging
from typing import Dict, Any, Optional

from dependency_injection_container import (
    ConsciousnessDependencyContainer, ServiceLifetime, 
    configure_consciousness_dependencies, get_global_container
)
from consciousness_interfaces import (
    IPhiCalculator, IExperientialPhiCalculator, IConsciousnessDetector,
    IDevelopmentStageManager, IExperientialMemoryRepository, IStreamingPhiProcessor,
    IConsciousnessDevelopmentAnalyzer, IAxiomValidator, IIntrinsicDifferenceCalculator,
    IPhiPredictor, IPhiCache, IPerformanceMonitor, IConfigurationManager,
    ILoggingService, IConsciousnessSystemFactory
)
from consciousness_adapters import (
    PhiCalculatorAdapter, ExperientialPhiCalculatorAdapter,
    DevelopmentStageManagerAdapter, StreamingPhiProcessorAdapter,
    ConsciousnessDevelopmentAnalyzerAdapter, AxiomValidatorAdapter,
    IntrinsicDifferenceCalculatorAdapter, PhiPredictorAdapter, PhiCacheAdapter,
    SimpleConsciousnessDetector, SimpleExperientialMemoryRepository,
    ConsciousnessSystemFactoryImpl
)

logger = logging.getLogger(__name__)


class ConsciousnessSystemConfigurator:
    """
    Centralized configurator for the consciousness system
    Implements the Configuration pattern for dependency management
    """
    
    def __init__(self, 
                 container: Optional[ConsciousnessDependencyContainer] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize system configurator
        
        Args:
            container: Optional custom container
            config: Optional configuration overrides
        """
        self.container = container or get_global_container()
        self.config = config or self._get_default_config()
        self._is_configured = False
    
    def configure_system(self) -> ConsciousnessDependencyContainer:
        """
        Configure the complete consciousness system with proper dependency injection
        
        Returns:
            Configured dependency container
        """
        if self._is_configured:
            logger.info("System already configured, skipping reconfiguration")
            return self.container
        
        logger.info("Configuring consciousness system with dependency injection...")
        
        # Configure core calculation components
        self._configure_core_calculators()
        
        # Configure high-level consciousness services
        self._configure_consciousness_services()
        
        # Configure analysis and development components
        self._configure_analysis_components()
        
        # Configure infrastructure services
        self._configure_infrastructure_services()
        
        # Configure system factory
        self._configure_system_factory()
        
        # Perform health check
        self._perform_system_health_check()
        
        self._is_configured = True
        logger.info("Consciousness system configuration completed successfully")
        
        return self.container
    
    def _configure_core_calculators(self) -> None:
        """Configure core calculation components"""
        
        logger.debug("Configuring core calculation components...")
        
        # Intrinsic Difference Calculator (foundation)
        self.container.register_singleton(
            IIntrinsicDifferenceCalculator,
            implementation_type=None,
            factory=lambda: IntrinsicDifferenceCalculatorAdapter(
                precision=self.config['calculation']['precision']
            )
        )
        
        # Core Phi Calculator  
        self.container.register_singleton(
            IPhiCalculator,
            implementation_type=None,
            factory=lambda: PhiCalculatorAdapter(
                precision=self.config['calculation']['precision'],
                max_mechanism_size=self.config['calculation']['max_mechanism_size']
            )
        )
        
        # Experiential Phi Calculator
        self.container.register_singleton(
            IExperientialPhiCalculator,
            implementation_type=None,
            factory=lambda: ExperientialPhiCalculatorAdapter(
                precision=self.config['calculation']['precision'],
                max_concept_size=self.config['experiential']['max_concept_size']
            )
        )
        
        logger.debug("Core calculation components configured")
    
    def _configure_consciousness_services(self) -> None:
        """Configure high-level consciousness services"""
        
        logger.debug("Configuring consciousness services...")
        
        # Consciousness Detector
        self.container.register_singleton(
            IConsciousnessDetector,
            implementation_type=None,
            factory=self._create_consciousness_detector
        )
        
        # Development Stage Manager
        self.container.register_singleton(
            IDevelopmentStageManager,
            implementation_type=None,
            factory=lambda: DevelopmentStageManagerAdapter()
        )
        
        # Experiential Memory Repository
        self.container.register_singleton(
            IExperientialMemoryRepository,
            implementation_type=None,
            factory=lambda: SimpleExperientialMemoryRepository(
                max_memories=self.config['memory']['max_memories']
            )
        )
        
        logger.debug("Consciousness services configured")
    
    def _configure_analysis_components(self) -> None:
        """Configure analysis and development components"""
        
        logger.debug("Configuring analysis components...")
        
        # Consciousness Development Analyzer
        self.container.register_singleton(
            IConsciousnessDevelopmentAnalyzer,
            implementation_type=None,
            factory=lambda: ConsciousnessDevelopmentAnalyzerAdapter()
        )
        
        # Axiom Validator
        self.container.register_singleton(
            IAxiomValidator,
            implementation_type=None,
            factory=self._create_axiom_validator
        )
        
        # Streaming Phi Processor
        self.container.register_singleton(
            IStreamingPhiProcessor,
            implementation_type=None,
            factory=self._create_streaming_processor
        )
        
        logger.debug("Analysis components configured")
    
    def _configure_infrastructure_services(self) -> None:
        """Configure infrastructure services (cache, prediction, monitoring)"""
        
        logger.debug("Configuring infrastructure services...")
        
        # Phi Cache
        self.container.register_singleton(
            IPhiCache,
            implementation_type=None,
            factory=lambda: PhiCacheAdapter(
                max_memory_mb=self.config['cache']['max_memory_mb'],
                compression_threshold=self.config['cache']['compression_threshold']
            )
        )
        
        # Phi Predictor
        self.container.register_singleton(
            IPhiPredictor,
            implementation_type=None,
            factory=lambda: PhiPredictorAdapter(
                model_type=self.config['prediction']['model_type']
            )
        )
        
        logger.debug("Infrastructure services configured")
    
    def _configure_system_factory(self) -> None:
        """Configure system factory for creating new instances"""
        
        logger.debug("Configuring system factory...")
        
        self.container.register_singleton(
            IConsciousnessSystemFactory,
            implementation_type=None,
            factory=lambda: ConsciousnessSystemFactoryImpl()
        )
        
        logger.debug("System factory configured")
    
    def _create_consciousness_detector(self) -> IConsciousnessDetector:
        """Factory method for consciousness detector with injected dependencies"""
        
        phi_calculator = self.container.resolve(IPhiCalculator)
        experiential_calculator = self.container.resolve(IExperientialPhiCalculator)
        
        return SimpleConsciousnessDetector(phi_calculator, experiential_calculator)
    
    def _create_axiom_validator(self) -> IAxiomValidator:
        """Factory method for axiom validator with injected dependencies"""
        
        phi_calculator = self.container.resolve(IPhiCalculator)
        return AxiomValidatorAdapter(phi_calculator)
    
    def _create_streaming_processor(self) -> IStreamingPhiProcessor:
        """Factory method for streaming processor with injected dependencies"""
        
        return StreamingPhiProcessorAdapter(
            default_window_size=self.config['streaming']['window_size'],
            max_concurrent_windows=self.config['streaming']['max_windows'],
            target_throughput_rps=self.config['streaming']['target_throughput']
        )
    
    def _perform_system_health_check(self) -> None:
        """Perform comprehensive system health check"""
        
        logger.debug("Performing system health check...")
        
        try:
            # Test core services resolution
            phi_calculator = self.container.resolve(IPhiCalculator)
            experiential_calculator = self.container.resolve(IExperientialPhiCalculator)
            consciousness_detector = self.container.resolve(IConsciousnessDetector)
            development_manager = self.container.resolve(IDevelopmentStageManager)
            
            # Verify interface compliance
            assert hasattr(phi_calculator, 'calculate_phi')
            assert hasattr(experiential_calculator, 'calculate_experiential_phi')
            assert hasattr(consciousness_detector, 'detect_consciousness_level')
            assert hasattr(development_manager, 'map_phi_to_development_stage')
            
            # Get container health stats
            stats = self.container.get_container_stats()
            success_rate = stats['resolution_stats']['success_rate_percent']
            
            if success_rate < 95.0:
                logger.warning(f"Container success rate below threshold: {success_rate:.1f}%")
            
            logger.info(f"System health check passed (success rate: {success_rate:.1f}%)")
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        
        return {
            'calculation': {
                'precision': 1e-10,
                'max_mechanism_size': 8
            },
            'experiential': {
                'max_concept_size': 8
            },
            'memory': {
                'max_memories': 10000
            },
            'cache': {
                'max_memory_mb': 500,
                'compression_threshold': 1000
            },
            'prediction': {
                'model_type': 'random_forest'
            },
            'streaming': {
                'window_size': 60.0,
                'max_windows': 10,
                'target_throughput': 1000
            },
            'monitoring': {
                'enable_performance_tracking': True,
                'log_level': 'INFO'
            }
        }
    
    def get_service_summary(self) -> Dict[str, Any]:
        """Get summary of configured services"""
        
        if not self._is_configured:
            return {'status': 'not_configured'}
        
        stats = self.container.get_container_stats()
        
        return {
            'status': 'configured',
            'total_services': stats['total_registered_services'],
            'singleton_instances': stats['singleton_instances'],
            'resolution_stats': stats['resolution_stats'],
            'is_healthy': stats['is_healthy'],
            'last_health_check': stats['last_health_check'],
            'configuration': self.config
        }
    
    def reconfigure_service(self, 
                           service_type: type, 
                           new_implementation: Any = None,
                           new_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Reconfigure a specific service
        
        Args:
            service_type: Service interface type to reconfigure
            new_implementation: New implementation instance
            new_config: New configuration for the service
        """
        logger.info(f"Reconfiguring service: {service_type.__name__}")
        
        # Update configuration if provided
        if new_config:
            self.config.update(new_config)
        
        # Re-register with new implementation
        if new_implementation:
            self.container.register_instance(service_type, new_implementation)
        
        logger.info(f"Service {service_type.__name__} reconfigured successfully")


# Global configurator instance
_global_configurator: Optional[ConsciousnessSystemConfigurator] = None


def get_global_configurator() -> ConsciousnessSystemConfigurator:
    """Get or create global system configurator"""
    
    global _global_configurator
    
    if _global_configurator is None:
        _global_configurator = ConsciousnessSystemConfigurator()
    
    return _global_configurator


def configure_consciousness_system(config: Optional[Dict[str, Any]] = None) -> ConsciousnessDependencyContainer:
    """
    Convenience function to configure the consciousness system
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured dependency container
    """
    configurator = get_global_configurator()
    
    if config:
        configurator.config.update(config)
    
    return configurator.configure_system()


def get_consciousness_service(service_type: type) -> Any:
    """
    Convenience function to get configured consciousness service
    
    Args:
        service_type: Service interface type
        
    Returns:
        Service instance
    """
    container = get_global_container()
    return container.resolve(service_type)


# High-level consciousness system interface

class ConsciousnessSystem:
    """
    High-level consciousness system facade
    Provides simplified interface to the complete consciousness system
    """
    
    def __init__(self, container: Optional[ConsciousnessDependencyContainer] = None):
        """
        Initialize consciousness system
        
        Args:
            container: Optional pre-configured container
        """
        if container is None:
            # Auto-configure if not provided
            container = configure_consciousness_system()
        
        self.container = container
        
        # Resolve core services
        self.phi_calculator = container.resolve(IPhiCalculator)
        self.experiential_calculator = container.resolve(IExperientialPhiCalculator)
        self.consciousness_detector = container.resolve(IConsciousnessDetector)
        self.development_manager = container.resolve(IDevelopmentStageManager)
        self.memory_repository = container.resolve(IExperientialMemoryRepository)
        self.development_analyzer = container.resolve(IConsciousnessDevelopmentAnalyzer)
        
        logger.info("Consciousness system initialized with dependency injection")
    
    async def analyze_consciousness(self, input_data: Any) -> Dict[str, Any]:
        """
        Comprehensive consciousness analysis
        
        Args:
            input_data: Input for consciousness analysis
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Detect consciousness level
            consciousness_level = await self.consciousness_detector.detect_consciousness_level(input_data)
            
            # Analyze consciousness quality
            quality_metrics = await self.consciousness_detector.analyze_consciousness_quality(input_data)
            
            # Calculate experiential phi if applicable
            experiential_result = None
            if isinstance(input_data, dict) and 'experiential_concepts' in input_data:
                experiential_result = await self.experiential_calculator.calculate_experiential_phi(
                    input_data['experiential_concepts'],
                    input_data.get('temporal_context'),
                    input_data.get('narrative_context')
                )
            
            return {
                'consciousness_level': consciousness_level,
                'quality_metrics': quality_metrics,
                'experiential_result': experiential_result,
                'analysis_timestamp': logger.info.__self__.name,
                'system_status': 'operational'
            }
            
        except Exception as e:
            logger.error(f"Consciousness analysis failed: {e}")
            return {
                'consciousness_level': 0.0,
                'quality_metrics': {},
                'experiential_result': None,
                'error': str(e),
                'system_status': 'error'
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        container_stats = self.container.get_container_stats()
        
        return {
            'system_name': 'IIT 4.0 NewbornAI Consciousness System',
            'version': '2.0.0',
            'dependency_injection': 'enabled',
            'container_stats': container_stats,
            'services': {
                'phi_calculator': type(self.phi_calculator).__name__,
                'experiential_calculator': type(self.experiential_calculator).__name__,
                'consciousness_detector': type(self.consciousness_detector).__name__,
                'development_manager': type(self.development_manager).__name__,
                'memory_repository': type(self.memory_repository).__name__,
                'development_analyzer': type(self.development_analyzer).__name__
            }
        }


# Quick setup function for immediate use
def setup_consciousness_system(custom_config: Optional[Dict[str, Any]] = None) -> ConsciousnessSystem:
    """
    Quick setup function for consciousness system
    
    Args:
        custom_config: Optional custom configuration
        
    Returns:
        Ready-to-use consciousness system
    """
    logger.info("Setting up consciousness system with dependency injection...")
    
    # Configure system
    container = configure_consciousness_system(custom_config)
    
    # Create system facade
    system = ConsciousnessSystem(container)
    
    logger.info("Consciousness system setup completed")
    
    return system