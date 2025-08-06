"""
Consciousness System Interface Abstractions
Dependency Inversion Principle (DIP) Compliance Layer

This module provides abstract interfaces for all major consciousness components,
enabling dependency injection and testability while maintaining performance.

Author: Martin Fowler's Refactoring Agent
Date: 2025-08-03
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, FrozenSet, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

# Core data structures that need to be shared across interfaces
@dataclass(frozen=True)
class PhiCalculationInput:
    """Input data for phi calculations"""
    system_state: np.ndarray
    connectivity_matrix: np.ndarray
    tpm: Optional[np.ndarray] = None
    precision: float = 1e-10

@dataclass
class PhiCalculationResult:
    """Result from phi calculations"""
    phi_value: float
    calculation_confidence: float
    processing_time_ms: float
    cache_hit: bool = False


class IPhiCalculator(ABC):
    """
    Abstract interface for φ value calculation engines
    Supports both basic and experiential phi calculations
    """
    
    @abstractmethod
    def calculate_phi(self, 
                     system_state: np.ndarray,
                     connectivity_matrix: np.ndarray,
                     tpm: Optional[np.ndarray] = None) -> Any:
        """
        Calculate φ value for given system state
        
        Args:
            system_state: Current system state vector
            connectivity_matrix: System connectivity matrix
            tpm: Optional transition probability matrix
            
        Returns:
            PhiStructure or similar calculation result
        """
        pass
    
    @abstractmethod
    def get_calculation_stats(self) -> Dict[str, Any]:
        """Get calculator performance statistics"""
        pass


class IExperientialPhiCalculator(ABC):
    """
    Abstract interface for experiential φ calculations
    Handles pure experiential consciousness processing
    """
    
    @abstractmethod
    async def calculate_experiential_phi(self,
                                       experiential_concepts: List[Dict],
                                       temporal_context: Optional[Dict] = None,
                                       narrative_context: Optional[Dict] = None) -> Any:
        """
        Calculate experiential φ value
        
        Args:
            experiential_concepts: Pure experiential concepts
            temporal_context: Temporal integration context
            narrative_context: Narrative coherence context
            
        Returns:
            ExperientialPhiResult with comprehensive analysis
        """
        pass
    
    @abstractmethod
    def get_phi_history_analysis(self) -> Dict[str, Any]:
        """Get historical phi analysis and trends"""
        pass


class IConsciousnessDetector(ABC):
    """
    Abstract interface for consciousness detection and analysis
    Provides high-level consciousness assessment capabilities
    """
    
    @abstractmethod
    async def detect_consciousness_level(self,
                                       input_data: Any) -> float:
        """
        Detect current consciousness level
        
        Args:
            input_data: Input for consciousness analysis
            
        Returns:
            float: Consciousness level (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    async def analyze_consciousness_quality(self,
                                          input_data: Any) -> Dict[str, float]:
        """
        Analyze consciousness quality metrics
        
        Returns:
            Dict with quality metrics like integration, temporal_depth, etc.
        """
        pass


class IDevelopmentStageManager(ABC):
    """
    Abstract interface for development stage management
    Handles consciousness development progression and analysis
    """
    
    @abstractmethod
    def map_phi_to_development_stage(self,
                                   phi_structure: Any,
                                   experiential_result: Optional[Any] = None,
                                   axiom_compliance: Optional[Dict[str, bool]] = None) -> Any:
        """
        Map φ structure to development stage
        
        Args:
            phi_structure: IIT φ structure
            experiential_result: Optional experiential phi result
            axiom_compliance: Optional axiom compliance data
            
        Returns:
            DevelopmentMetrics with comprehensive stage analysis
        """
        pass
    
    @abstractmethod
    def detect_stage_transitions(self, current_metrics: Any) -> Optional[Any]:
        """Detect and classify stage transitions"""
        pass
    
    @abstractmethod
    def predict_development_trajectory(self,
                                     target_stage: Any,
                                     time_horizon_days: int = 30) -> Any:
        """Predict development trajectory to target stage"""
        pass


class IExperientialMemoryRepository(ABC):
    """
    Abstract interface for experiential memory storage and retrieval
    Handles persistent storage of consciousness experiences
    """
    
    @abstractmethod
    async def store_experience(self,
                             experience_id: str,
                             experiential_data: Dict[str, Any],
                             phi_result: Any) -> bool:
        """
        Store experiential memory entry
        
        Args:
            experience_id: Unique identifier for the experience
            experiential_data: Raw experiential data
            phi_result: Associated phi calculation result
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    async def retrieve_experiences(self,
                                 query_params: Dict[str, Any],
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve experiences matching query parameters
        
        Args:
            query_params: Search criteria
            limit: Maximum number of results
            
        Returns:
            List of matching experiences
        """
        pass
    
    @abstractmethod
    async def get_temporal_sequence(self,
                                  start_time: datetime,
                                  end_time: datetime) -> List[Dict[str, Any]]:
        """Get experiences in temporal sequence"""
        pass


class IStreamingPhiProcessor(ABC):
    """
    Abstract interface for high-throughput streaming phi processing
    Handles real-time consciousness stream analysis
    """
    
    @abstractmethod
    async def start_streaming(self) -> None:
        """Start streaming phi calculation"""
        pass
    
    @abstractmethod
    async def stop_streaming(self) -> None:
        """Stop streaming phi calculation"""
        pass
    
    @abstractmethod
    async def submit_event(self, event: Any) -> bool:
        """
        Submit event for streaming processing
        
        Args:
            event: Streaming phi event
            
        Returns:
            bool: True if event was queued successfully
        """
        pass
    
    @abstractmethod
    async def process_event_stream(self,
                                 event_stream: AsyncGenerator[Any, None],
                                 stream_id: str = "default") -> AsyncGenerator[Any, None]:
        """
        Process stream of phi events
        
        Args:
            event_stream: Async generator of events
            stream_id: Stream identifier
            
        Yields:
            Processed results
        """
        pass
    
    @abstractmethod
    async def get_streaming_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics"""
        pass


class IConsciousnessDevelopmentAnalyzer(ABC):
    """
    Abstract interface for consciousness development analysis
    Provides long-term development pattern analysis and recommendations
    """
    
    @abstractmethod
    async def analyze_development_pattern(self,
                                        development_history: Optional[List[Any]] = None) -> Any:
        """
        Analyze long-term development pattern
        
        Args:
            development_history: Optional history override
            
        Returns:
            DevelopmentPattern: Identified pattern type
        """
        pass
    
    @abstractmethod
    async def compare_with_norms(self, current_metrics: Any) -> Dict[str, Any]:
        """
        Compare current development with established norms
        
        Args:
            current_metrics: Current development metrics
            
        Returns:
            Dict: Comprehensive comparison analysis
        """
        pass
    
    @abstractmethod
    async def generate_personalized_recommendations(self,
                                                  current_metrics: Any,
                                                  goal_preferences: Optional[Dict] = None) -> List[Any]:
        """
        Generate personalized development recommendations
        
        Args:
            current_metrics: Current development metrics
            goal_preferences: Optional goal preferences
            
        Returns:
            List[PersonalizedRecommendation]: Personalized recommendations
        """
        pass
    
    @abstractmethod
    def get_development_insights(self) -> List[Any]:
        """Generate development insights from analysis"""
        pass


class IAxiomValidator(ABC):
    """
    Abstract interface for IIT axiom validation
    Ensures theoretical compliance with IIT 4.0 principles
    """
    
    @abstractmethod
    def validate_all_axioms(self,
                          phi_structure: Any,
                          system_state: np.ndarray) -> Dict[str, bool]:
        """
        Validate all IIT axioms
        
        Args:
            phi_structure: Phi structure to validate
            system_state: System state vector
            
        Returns:
            Dict mapping axiom names to compliance status
        """
        pass
    
    @abstractmethod
    def validate_existence(self, phi_structure: Any, system_state: np.ndarray) -> bool:
        """Validate axiom of existence"""
        pass
    
    @abstractmethod
    def validate_intrinsicality(self, phi_structure: Any) -> bool:
        """Validate axiom of intrinsicality"""
        pass
    
    @abstractmethod
    def validate_information(self, phi_structure: Any) -> bool:
        """Validate axiom of information"""
        pass
    
    @abstractmethod
    def validate_integration(self, phi_structure: Any) -> bool:
        """Validate axiom of integration"""
        pass
    
    @abstractmethod
    def validate_exclusion(self, phi_structure: Any) -> bool:
        """Validate axiom of exclusion"""
        pass
    
    @abstractmethod
    def validate_composition(self, phi_structure: Any) -> bool:
        """Validate axiom of composition"""
        pass


class IIntrinsicDifferenceCalculator(ABC):
    """
    Abstract interface for intrinsic difference calculations
    Core component of IIT phi calculations
    """
    
    @abstractmethod
    def compute_id(self,
                  mechanism: FrozenSet[int],
                  purview: FrozenSet[int],
                  tpm: np.ndarray,
                  current_state: np.ndarray,
                  direction: str = 'cause') -> float:
        """
        Compute intrinsic difference
        
        Args:
            mechanism: Mechanism node set
            purview: Purview node set  
            tpm: Transition probability matrix
            current_state: Current system state
            direction: 'cause' or 'effect'
            
        Returns:
            float: Intrinsic difference value
        """
        pass


class IPhiPredictor(ABC):
    """
    Abstract interface for phi value prediction
    Machine learning-based future phi estimation
    """
    
    @abstractmethod
    def add_training_sample(self, features: Dict[str, float], actual_phi: float) -> None:
        """Add training sample for model improvement"""
        pass
    
    @abstractmethod
    async def predict_phi(self,
                        current_features: Dict[str, float],
                        prediction_horizon_seconds: float = 60.0) -> Optional[float]:
        """
        Predict future phi value
        
        Args:
            current_features: Current feature state
            prediction_horizon_seconds: How far ahead to predict
            
        Returns:
            Predicted phi value or None if insufficient data
        """
        pass
    
    @abstractmethod
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        pass


class IPhiCache(ABC):
    """
    Abstract interface for phi calculation caching
    High-performance cache with memory management
    """
    
    @abstractmethod
    async def get(self, cache_key: str) -> Optional[Any]:
        """Get cached result"""
        pass
    
    @abstractmethod
    async def put(self, cache_key: str, result: Any) -> None:
        """Cache result"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class IPerformanceMonitor(ABC):
    """
    Abstract interface for system performance monitoring
    Tracks metrics, throughput, and system health
    """
    
    @abstractmethod
    async def record_metric(self,
                          metric_name: str,
                          value: float,
                          timestamp: Optional[datetime] = None) -> None:
        """Record performance metric"""
        pass
    
    @abstractmethod
    async def get_metrics_summary(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """Get performance metrics summary"""
        pass
    
    @abstractmethod
    async def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """Detect performance issues and anomalies"""
        pass


class IConfigurationManager(ABC):
    """
    Abstract interface for system configuration management
    Handles dynamic configuration and parameter tuning
    """
    
    @abstractmethod
    def get_config(self, config_key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    def set_config(self, config_key: str, value: Any) -> None:
        """Set configuration value"""
        pass
    
    @abstractmethod
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration values"""
        pass
    
    @abstractmethod
    async def reload_config(self) -> None:
        """Reload configuration from source"""
        pass


class ILoggingService(ABC):
    """
    Abstract interface for consciousness-aware logging
    Specialized logging for consciousness system events
    """
    
    @abstractmethod
    def log_phi_calculation(self,
                          phi_value: float,
                          calculation_time: float,
                          context: Dict[str, Any]) -> None:
        """Log phi calculation event"""
        pass
    
    @abstractmethod
    def log_stage_transition(self,
                           from_stage: Any,
                           to_stage: Any,
                           transition_context: Dict[str, Any]) -> None:
        """Log development stage transition"""
        pass
    
    @abstractmethod
    def log_consciousness_event(self,
                              event_type: str,
                              consciousness_level: float,
                              context: Dict[str, Any]) -> None:
        """Log consciousness-related event"""
        pass


# Factory interfaces for dependency injection

class IConsciousnessSystemFactory(ABC):
    """
    Abstract factory for creating consciousness system components
    Enables flexible component instantiation and testing
    """
    
    @abstractmethod
    def create_phi_calculator(self, config: Dict[str, Any]) -> IPhiCalculator:
        """Create phi calculator instance"""
        pass
    
    @abstractmethod
    def create_experiential_phi_calculator(self, config: Dict[str, Any]) -> IExperientialPhiCalculator:
        """Create experiential phi calculator instance"""
        pass
    
    @abstractmethod
    def create_consciousness_detector(self, config: Dict[str, Any]) -> IConsciousnessDetector:
        """Create consciousness detector instance"""
        pass
    
    @abstractmethod
    def create_development_stage_manager(self, config: Dict[str, Any]) -> IDevelopmentStageManager:
        """Create development stage manager instance"""
        pass
    
    @abstractmethod
    def create_memory_repository(self, config: Dict[str, Any]) -> IExperientialMemoryRepository:
        """Create memory repository instance"""
        pass
    
    @abstractmethod
    def create_streaming_processor(self, config: Dict[str, Any]) -> IStreamingPhiProcessor:
        """Create streaming processor instance"""
        pass
    
    @abstractmethod
    def create_development_analyzer(self, config: Dict[str, Any]) -> IConsciousnessDevelopmentAnalyzer:
        """Create development analyzer instance"""
        pass


# Service locator interface for dependency management

class IServiceLocator(ABC):
    """
    Abstract service locator for dependency resolution
    Central registry for consciousness system services
    """
    
    @abstractmethod
    def register_service(self,
                        service_type: type,
                        service_instance: Any,
                        singleton: bool = True) -> None:
        """Register service instance"""
        pass
    
    @abstractmethod
    def get_service(self, service_type: type) -> Any:
        """Get service instance"""
        pass
    
    @abstractmethod
    def resolve_dependencies(self, target_class: type) -> Any:
        """Resolve and inject dependencies for target class"""
        pass
    
    @abstractmethod
    def clear_services(self) -> None:
        """Clear all registered services"""
        pass