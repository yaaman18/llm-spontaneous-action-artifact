"""
Dependency Injection Container for Consciousness System
IoC (Inversion of Control) implementation following Martin Fowler's patterns

This module implements a comprehensive dependency injection container
specifically designed for the IIT 4.0 NewbornAI consciousness system.

Key Features:
- Constructor injection
- Interface-based dependency resolution
- Singleton and transient lifetimes
- Circular dependency detection
- Configuration-driven service registration
- Performance monitoring and health checks

Author: Martin Fowler's Refactoring Agent
Date: 2025-08-03
Version: 1.0.0
"""

import inspect
import logging
from typing import Dict, Any, Type, Optional, Callable, List, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import weakref
import time
from functools import wraps

from consciousness_interfaces import (
    IPhiCalculator, IExperientialPhiCalculator, IConsciousnessDetector,
    IDevelopmentStageManager, IExperientialMemoryRepository, IStreamingPhiProcessor,
    IConsciousnessDevelopmentAnalyzer, IAxiomValidator, IIntrinsicDifferenceCalculator,
    IPhiPredictor, IPhiCache, IPerformanceMonitor, IConfigurationManager,
    ILoggingService, IConsciousnessSystemFactory, IServiceLocator
)

logger = logging.getLogger(__name__)


class ServiceLifetime(Enum):
    """Service lifetime management options"""
    SINGLETON = "singleton"      # Single instance throughout application lifetime
    TRANSIENT = "transient"      # New instance every time requested
    SCOPED = "scoped"           # Single instance per scope (e.g., per request)


class DependencyScope(Enum):
    """Dependency resolution scopes"""
    APPLICATION = "application"  # Application-wide scope
    REQUEST = "request"         # Per-request scope
    THREAD = "thread"           # Per-thread scope


@dataclass
class ServiceDescriptor:
    """Describes how a service should be created and managed"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory_function: Optional[Callable] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    instance: Optional[Any] = None
    dependencies: List[Type] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    registration_time: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    # Health monitoring
    creation_time_ms: float = 0.0
    failure_count: int = 0
    last_error: Optional[str] = None


@dataclass
class DependencyResolutionContext:
    """Context for dependency resolution operations"""
    requesting_type: Optional[Type] = None
    resolution_path: List[Type] = field(default_factory=list)
    scope: DependencyScope = DependencyScope.APPLICATION
    parameters: Dict[str, Any] = field(default_factory=dict)


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected"""
    def __init__(self, resolution_path: List[Type]):
        self.resolution_path = resolution_path
        path_names = " -> ".join(t.__name__ for t in resolution_path)
        super().__init__(f"Circular dependency detected: {path_names}")


class ServiceRegistrationError(Exception):
    """Raised when service registration fails"""
    pass


class DependencyResolutionError(Exception):
    """Raised when dependency resolution fails"""
    pass


def injectable(lifetime: ServiceLifetime = ServiceLifetime.SINGLETON):
    """
    Decorator to mark classes as injectable services
    
    Args:
        lifetime: Service lifetime management
    """
    def decorator(cls):
        cls._injectable_lifetime = lifetime
        return cls
    return decorator


def inject(*dependencies: Type):
    """
    Decorator to specify dependencies for a class constructor
    
    Args:
        dependencies: Types to inject as constructor parameters
    """
    def decorator(cls):
        cls._injectable_dependencies = dependencies
        return cls
    return decorator


class ConsciousnessDependencyContainer:
    """
    Advanced dependency injection container for consciousness system
    
    Implements Martin Fowler's Dependency Injection patterns:
    - Constructor Injection
    - Setter Injection
    - Interface Injection
    """
    
    def __init__(self, name: str = "ConsciousnessContainer"):
        """
        Initialize dependency container
        
        Args:
            name: Container name for identification
        """
        self.name = name
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[DependencyScope, Dict[Type, Any]] = {
            scope: {} for scope in DependencyScope
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._thread_local = threading.local()
        
        # Performance monitoring
        self._resolution_stats = {
            'total_resolutions': 0,
            'successful_resolutions': 0,
            'failed_resolutions': 0,
            'average_resolution_time_ms': 0.0,
            'circular_dependency_detections': 0
        }
        
        # Health monitoring
        self._health_checks: List[Callable[[], bool]] = []
        self._is_healthy = True
        self._last_health_check = datetime.now()
        
        logger.info(f"Dependency container '{name}' initialized")
    
    def register_singleton(self,
                          service_type: Type,
                          implementation_type: Optional[Type] = None,
                          factory: Optional[Callable] = None,
                          **config) -> 'ConsciousnessDependencyContainer':
        """
        Register service as singleton
        
        Args:
            service_type: Interface or abstract type
            implementation_type: Concrete implementation
            factory: Optional factory function
            **config: Configuration parameters
            
        Returns:
            Self for method chaining
        """
        return self.register_service(
            service_type, implementation_type, factory,
            ServiceLifetime.SINGLETON, **config
        )
    
    def register_transient(self,
                          service_type: Type,
                          implementation_type: Optional[Type] = None,
                          factory: Optional[Callable] = None,
                          **config) -> 'ConsciousnessDependencyContainer':
        """
        Register service as transient
        
        Args:
            service_type: Interface or abstract type
            implementation_type: Concrete implementation
            factory: Optional factory function
            **config: Configuration parameters
            
        Returns:
            Self for method chaining
        """
        return self.register_service(
            service_type, implementation_type, factory,
            ServiceLifetime.TRANSIENT, **config
        )
    
    def register_service(self,
                        service_type: Type,
                        implementation_type: Optional[Type] = None,
                        factory: Optional[Callable] = None,
                        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
                        **config) -> 'ConsciousnessDependencyContainer':
        """
        Register service with container
        
        Args:
            service_type: Interface or abstract type to register
            implementation_type: Concrete implementation class
            factory: Optional factory function for custom creation
            lifetime: Service lifetime management
            **config: Configuration parameters
            
        Returns:
            Self for method chaining
            
        Raises:
            ServiceRegistrationError: If registration fails
        """
        with self._lock:
            try:
                # Validate registration
                if service_type in self._services:
                    logger.warning(f"Service {service_type.__name__} already registered, overriding")
                
                if not implementation_type and not factory:
                    # Try to use the service type itself as implementation
                    if not inspect.isabstract(service_type):
                        implementation_type = service_type
                    else:
                        raise ServiceRegistrationError(
                            f"Abstract service {service_type.__name__} requires implementation_type or factory"
                        )
                
                # Analyze dependencies
                dependencies = []
                if implementation_type:
                    dependencies = self._analyze_dependencies(implementation_type)
                
                # Create service descriptor
                descriptor = ServiceDescriptor(
                    service_type=service_type,
                    implementation_type=implementation_type,
                    factory_function=factory,
                    lifetime=lifetime,
                    dependencies=dependencies,
                    configuration=config
                )
                
                self._services[service_type] = descriptor
                
                logger.info(f"Registered service: {service_type.__name__} -> "
                           f"{implementation_type.__name__ if implementation_type else 'factory'} "
                           f"({lifetime.value})")
                
                return self
                
            except Exception as e:
                error_msg = f"Failed to register service {service_type.__name__}: {e}"
                logger.error(error_msg)
                raise ServiceRegistrationError(error_msg) from e
    
    def register_instance(self,
                         service_type: Type,
                         instance: Any) -> 'ConsciousnessDependencyContainer':
        """
        Register pre-created instance
        
        Args:
            service_type: Interface type
            instance: Pre-created instance
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                lifetime=ServiceLifetime.SINGLETON,
                instance=instance
            )
            
            self._services[service_type] = descriptor
            self._instances[service_type] = instance
            
            logger.info(f"Registered instance: {service_type.__name__}")
            return self
    
    def resolve(self, service_type: Type, context: Optional[DependencyResolutionContext] = None) -> Any:
        """
        Resolve service instance
        
        Args:
            service_type: Type to resolve
            context: Optional resolution context
            
        Returns:
            Service instance
            
        Raises:
            DependencyResolutionError: If resolution fails
            CircularDependencyError: If circular dependency detected
        """
        start_time = time.time()
        
        try:
            with self._lock:
                self._resolution_stats['total_resolutions'] += 1
                
                if context is None:
                    context = DependencyResolutionContext()
                
                # Check for circular dependencies
                if service_type in context.resolution_path:
                    self._resolution_stats['circular_dependency_detections'] += 1
                    raise CircularDependencyError(context.resolution_path + [service_type])
                
                # Add to resolution path
                context.resolution_path.append(service_type)
                
                try:
                    instance = self._resolve_internal(service_type, context)
                    self._resolution_stats['successful_resolutions'] += 1
                    return instance
                    
                finally:
                    # Remove from resolution path
                    context.resolution_path.pop()
        
        except Exception as e:
            self._resolution_stats['failed_resolutions'] += 1
            logger.error(f"Failed to resolve {service_type.__name__}: {e}")
            raise DependencyResolutionError(f"Failed to resolve {service_type.__name__}") from e
        
        finally:
            # Update performance stats
            resolution_time = (time.time() - start_time) * 1000
            self._update_resolution_stats(resolution_time)
    
    def _resolve_internal(self, service_type: Type, context: DependencyResolutionContext) -> Any:
        """Internal service resolution logic"""
        
        # Check if service is registered
        if service_type not in self._services:
            raise DependencyResolutionError(f"Service {service_type.__name__} is not registered")
        
        descriptor = self._services[service_type]
        
        # Update access statistics
        descriptor.access_count += 1
        descriptor.last_accessed = datetime.now()
        
        # Handle singleton lifetime
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._instances:
                return self._instances[service_type]
            
            # Create singleton instance
            instance = self._create_instance(descriptor, context)
            self._instances[service_type] = instance
            return instance
        
        # Handle scoped lifetime
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            scope_instances = self._scoped_instances[context.scope]
            if service_type in scope_instances:
                return scope_instances[service_type]
            
            instance = self._create_instance(descriptor, context)
            scope_instances[service_type] = instance
            return instance
        
        # Handle transient lifetime
        else:  # TRANSIENT
            return self._create_instance(descriptor, context)
    
    def _create_instance(self, descriptor: ServiceDescriptor, context: DependencyResolutionContext) -> Any:
        """Create service instance using descriptor"""
        
        creation_start = time.time()
        
        try:
            # Use pre-created instance if available
            if descriptor.instance is not None:
                return descriptor.instance
            
            # Use factory function if available
            if descriptor.factory_function:
                # Resolve factory dependencies
                factory_deps = self._resolve_factory_dependencies(descriptor.factory_function, context)
                instance = descriptor.factory_function(**factory_deps, **descriptor.configuration)
            
            # Use implementation type
            elif descriptor.implementation_type:
                # Resolve constructor dependencies
                constructor_args = self._resolve_constructor_dependencies(descriptor.implementation_type, context)
                instance = descriptor.implementation_type(*constructor_args, **descriptor.configuration)
            
            else:
                raise DependencyResolutionError(f"No creation method available for {descriptor.service_type.__name__}")
            
            # Record creation time
            descriptor.creation_time_ms = (time.time() - creation_start) * 1000
            
            # Perform setter injection if needed
            self._perform_setter_injection(instance, context)
            
            logger.debug(f"Created instance: {descriptor.service_type.__name__} "
                        f"(creation time: {descriptor.creation_time_ms:.2f}ms)")
            
            return instance
            
        except Exception as e:
            descriptor.failure_count += 1
            descriptor.last_error = str(e)
            logger.error(f"Failed to create instance of {descriptor.service_type.__name__}: {e}")
            raise
    
    def _analyze_dependencies(self, implementation_type: Type) -> List[Type]:
        """Analyze constructor dependencies for implementation type"""
        
        try:
            # Check for explicit dependency decoration
            if hasattr(implementation_type, '_injectable_dependencies'):
                return list(implementation_type._injectable_dependencies)
            
            # Analyze constructor signature
            signature = inspect.signature(implementation_type.__init__)
            dependencies = []
            
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                # Use type annotation if available
                if param.annotation != inspect.Parameter.empty:
                    if hasattr(param.annotation, '__origin__'):
                        # Skip generic types for now
                        continue
                    dependencies.append(param.annotation)
            
            return dependencies
            
        except Exception as e:
            logger.warning(f"Could not analyze dependencies for {implementation_type.__name__}: {e}")
            return []
    
    def _resolve_constructor_dependencies(self, implementation_type: Type, context: DependencyResolutionContext) -> List[Any]:
        """Resolve constructor dependencies"""
        
        dependencies = []
        
        try:
            signature = inspect.signature(implementation_type.__init__)
            
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                # Use type annotation for dependency resolution
                if param.annotation != inspect.Parameter.empty:
                    if hasattr(param.annotation, '__origin__'):
                        # Skip generic types - would need more sophisticated handling
                        continue
                    
                    try:
                        dependency = self._resolve_internal(param.annotation, context)
                        dependencies.append(dependency)
                    except DependencyResolutionError:
                        # Check if parameter has default value
                        if param.default != inspect.Parameter.empty:
                            dependencies.append(param.default)
                        else:
                            raise
        
        except Exception as e:
            logger.error(f"Failed to resolve constructor dependencies for {implementation_type.__name__}: {e}")
            raise DependencyResolutionError(f"Constructor dependency resolution failed") from e
        
        return dependencies
    
    def _resolve_factory_dependencies(self, factory_function: Callable, context: DependencyResolutionContext) -> Dict[str, Any]:
        """Resolve factory function dependencies"""
        
        dependencies = {}
        
        try:
            signature = inspect.signature(factory_function)
            
            for param_name, param in signature.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    if hasattr(param.annotation, '__origin__'):
                        continue
                    
                    try:
                        dependency = self._resolve_internal(param.annotation, context)
                        dependencies[param_name] = dependency
                    except DependencyResolutionError:
                        if param.default != inspect.Parameter.empty:
                            dependencies[param_name] = param.default
                        else:
                            raise
        
        except Exception as e:
            logger.error(f"Failed to resolve factory dependencies: {e}")
            raise DependencyResolutionError(f"Factory dependency resolution failed") from e
        
        return dependencies
    
    def _perform_setter_injection(self, instance: Any, context: DependencyResolutionContext) -> None:
        """Perform setter-based dependency injection"""
        
        try:
            # Look for methods marked with @inject decorator or properties
            for attr_name in dir(instance):
                if attr_name.startswith('_'):
                    continue
                
                attr = getattr(instance, attr_name)
                
                # Check for injectable methods
                if hasattr(attr, '_injectable_dependencies'):
                    dependencies = attr._injectable_dependencies
                    resolved_deps = [self._resolve_internal(dep, context) for dep in dependencies]
                    attr(*resolved_deps)
        
        except Exception as e:
            logger.warning(f"Setter injection failed for {type(instance).__name__}: {e}")
    
    def _update_resolution_stats(self, resolution_time_ms: float) -> None:
        """Update resolution performance statistics"""
        
        current_avg = self._resolution_stats['average_resolution_time_ms']
        total_resolutions = self._resolution_stats['total_resolutions']
        
        # Update running average
        if total_resolutions > 0:
            self._resolution_stats['average_resolution_time_ms'] = (
                (current_avg * (total_resolutions - 1) + resolution_time_ms) / total_resolutions
            )
    
    def clear_scope(self, scope: DependencyScope) -> None:
        """Clear all instances in specified scope"""
        
        with self._lock:
            if scope in self._scoped_instances:
                cleared_count = len(self._scoped_instances[scope])
                self._scoped_instances[scope].clear()
                logger.info(f"Cleared {cleared_count} instances from {scope.value} scope")
    
    def get_service_info(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """Get information about registered service"""
        
        if service_type not in self._services:
            return None
        
        descriptor = self._services[service_type]
        
        return {
            'service_type': service_type.__name__,
            'implementation_type': descriptor.implementation_type.__name__ if descriptor.implementation_type else None,
            'lifetime': descriptor.lifetime.value,
            'access_count': descriptor.access_count,
            'last_accessed': descriptor.last_accessed,
            'creation_time_ms': descriptor.creation_time_ms,
            'failure_count': descriptor.failure_count,
            'last_error': descriptor.last_error,
            'has_instance': service_type in self._instances,
            'dependencies': [dep.__name__ for dep in descriptor.dependencies]
        }
    
    def get_container_stats(self) -> Dict[str, Any]:
        """Get container performance and health statistics"""
        
        with self._lock:
            total_services = len(self._services)
            singleton_instances = len(self._instances)
            scoped_instances = sum(len(instances) for instances in self._scoped_instances.values())
            
            success_rate = 0.0
            if self._resolution_stats['total_resolutions'] > 0:
                success_rate = (self._resolution_stats['successful_resolutions'] / 
                               self._resolution_stats['total_resolutions']) * 100
            
            return {
                'container_name': self.name,
                'total_registered_services': total_services,
                'singleton_instances': singleton_instances,
                'scoped_instances': scoped_instances,
                'is_healthy': self._is_healthy,
                'last_health_check': self._last_health_check,
                'resolution_stats': {
                    **self._resolution_stats,
                    'success_rate_percent': success_rate
                },
                'service_types': [service_type.__name__ for service_type in self._services.keys()]
            }
    
    def add_health_check(self, health_check: Callable[[], bool]) -> None:
        """Add health check function"""
        self._health_checks.append(health_check)
    
    def perform_health_check(self) -> bool:
        """Perform health checks on container and services"""
        
        try:
            # Run all registered health checks
            for health_check in self._health_checks:
                if not health_check():
                    self._is_healthy = False
                    return False
            
            # Check container statistics
            stats = self.get_container_stats()
            resolution_stats = stats['resolution_stats']
            
            # Health criteria
            if resolution_stats['success_rate_percent'] < 95.0:  # Less than 95% success rate
                self._is_healthy = False
                return False
            
            if resolution_stats['circular_dependency_detections'] > 10:  # Too many circular deps
                self._is_healthy = False
                return False
            
            self._is_healthy = True
            self._last_health_check = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._is_healthy = False
            return False
    
    def dispose(self) -> None:
        """Dispose of container and clean up resources"""
        
        with self._lock:
            logger.info(f"Disposing container '{self.name}'")
            
            # Dispose singleton instances that implement IDisposable
            for instance in self._instances.values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing instance: {e}")
            
            # Clear all collections
            self._services.clear()
            self._instances.clear()
            for scope_instances in self._scoped_instances.values():
                scope_instances.clear()
            
            logger.info(f"Container '{self.name}' disposed successfully")


# Global container instance for application-wide dependency injection
_global_container: Optional[ConsciousnessDependencyContainer] = None


def get_global_container() -> ConsciousnessDependencyContainer:
    """Get or create global dependency container"""
    
    global _global_container
    
    if _global_container is None:
        _global_container = ConsciousnessDependencyContainer("GlobalConsciousnessContainer")
    
    return _global_container


def configure_consciousness_dependencies() -> ConsciousnessDependencyContainer:
    """
    Configure standard consciousness system dependencies
    
    Returns:
        Configured dependency container
    """
    
    container = get_global_container()
    
    # This will be populated with actual implementations in the next step
    logger.info("Consciousness dependency configuration completed")
    
    return container


# Utility functions for dependency injection

def resolve_service(service_type: Type) -> Any:
    """
    Convenience function to resolve service from global container
    
    Args:
        service_type: Type to resolve
        
    Returns:
        Service instance
    """
    return get_global_container().resolve(service_type)


def inject_dependencies(target_class: Type) -> Type:
    """
    Class decorator to enable automatic dependency injection
    
    Args:
        target_class: Class to enable injection for
        
    Returns:
        Modified class with injection capability
    """
    original_init = target_class.__init__
    
    @wraps(original_init)
    def injected_init(self, *args, **kwargs):
        # Resolve dependencies and inject them
        container = get_global_container()
        
        # Analyze constructor dependencies
        dependencies = container._analyze_dependencies(target_class)
        
        # Resolve dependencies
        resolved_deps = []
        for dep_type in dependencies:
            try:
                resolved_deps.append(container.resolve(dep_type))
            except DependencyResolutionError:
                # Skip if can't resolve - might be provided in args/kwargs
                pass
        
        # Call original constructor with resolved dependencies + provided args
        original_init(self, *resolved_deps, *args, **kwargs)
    
    target_class.__init__ = injected_init
    return target_class