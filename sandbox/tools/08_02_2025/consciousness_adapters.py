"""
Consciousness System Adapters
Bridge existing concrete implementations to new interfaces

This module provides adapter classes that wrap existing implementations
to make them comply with the new DIP-compliant interfaces while maintaining
backward compatibility and existing functionality.

Author: Martin Fowler's Refactoring Agent  
Date: 2025-08-03
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Any, FrozenSet, AsyncGenerator
import numpy as np
from datetime import datetime
import logging

from consciousness_interfaces import (
    IPhiCalculator, IExperientialPhiCalculator, IConsciousnessDetector,
    IDevelopmentStageManager, IExperientialMemoryRepository, IStreamingPhiProcessor,
    IConsciousnessDevelopmentAnalyzer, IAxiomValidator, IIntrinsicDifferenceCalculator,
    IPhiPredictor, IPhiCache, IPerformanceMonitor, IConfigurationManager,
    ILoggingService, IConsciousnessSystemFactory
)

# Import existing concrete implementations
from iit4_core_engine import IIT4PhiCalculator, IIT4AxiomValidator, IntrinsicDifferenceCalculator
from iit4_experiential_phi_calculator import IIT4_ExperientialPhiCalculator
from iit4_development_stages import IIT4DevelopmentStageMapper
from streaming_phi_calculator import StreamingPhiCalculator, PhiCache, PhiPredictor
from consciousness_development_analyzer import ConsciousnessDevelopmentAnalyzer

logger = logging.getLogger(__name__)


class PhiCalculatorAdapter(IPhiCalculator):
    """
    Adapter for IIT4PhiCalculator to implement IPhiCalculator interface
    """
    
    def __init__(self, precision: float = 1e-10, max_mechanism_size: int = 8):
        """
        Initialize phi calculator adapter
        
        Args:
            precision: Numerical precision for calculations  
            max_mechanism_size: Maximum mechanism size for calculations
        """
        self._calculator = IIT4PhiCalculator(precision, max_mechanism_size)
        self._stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'average_calculation_time_ms': 0.0
        }
    
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
            PhiStructure: IIT 4.0 phi structure result
        """
        import time
        start_time = time.time()
        
        try:
            self._stats['total_calculations'] += 1
            
            result = self._calculator.calculate_phi(system_state, connectivity_matrix, tpm)
            
            self._stats['successful_calculations'] += 1
            
            # Update average calculation time
            calc_time = (time.time() - start_time) * 1000
            self._update_average_time(calc_time)
            
            return result
            
        except Exception as e:
            self._stats['failed_calculations'] += 1
            logger.error(f"Phi calculation failed: {e}")
            raise
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """Get calculator performance statistics"""
        
        success_rate = 0.0
        if self._stats['total_calculations'] > 0:
            success_rate = (self._stats['successful_calculations'] / 
                           self._stats['total_calculations']) * 100
        
        return {
            **self._stats,
            'success_rate_percent': success_rate
        }
    
    def _update_average_time(self, calc_time_ms: float) -> None:
        """Update average calculation time"""
        
        current_avg = self._stats['average_calculation_time_ms']
        total_calcs = self._stats['successful_calculations']
        
        if total_calcs > 0:
            self._stats['average_calculation_time_ms'] = (
                (current_avg * (total_calcs - 1) + calc_time_ms) / total_calcs
            )


class ExperientialPhiCalculatorAdapter(IExperientialPhiCalculator):
    """
    Adapter for IIT4_ExperientialPhiCalculator to implement IExperientialPhiCalculator interface
    """
    
    def __init__(self, precision: float = 1e-10, max_concept_size: int = 8):
        """
        Initialize experiential phi calculator adapter
        
        Args:
            precision: Numerical precision for calculations
            max_concept_size: Maximum size for experiential concepts
        """
        self._calculator = IIT4_ExperientialPhiCalculator(precision, max_concept_size)
    
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
            ExperientialPhiResult: Comprehensive experiential analysis
        """
        return await self._calculator.calculate_experiential_phi(
            experiential_concepts, temporal_context, narrative_context
        )
    
    def get_phi_history_analysis(self) -> Dict[str, Any]:
        """Get historical phi analysis and trends"""
        return self._calculator.get_phi_history_analysis()


class DevelopmentStageManagerAdapter(IDevelopmentStageManager):
    """
    Adapter for IIT4DevelopmentStageMapper to implement IDevelopmentStageManager interface
    """
    
    def __init__(self):
        """Initialize development stage manager adapter"""
        self._mapper = IIT4DevelopmentStageMapper()
    
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
            DevelopmentMetrics: Comprehensive stage analysis
        """
        return self._mapper.map_phi_to_development_stage(
            phi_structure, experiential_result, axiom_compliance
        )
    
    def detect_stage_transitions(self, current_metrics: Any) -> Optional[Any]:
        """Detect and classify stage transitions"""
        return self._mapper.detect_stage_transitions(current_metrics)
    
    def predict_development_trajectory(self,
                                     target_stage: Any,
                                     time_horizon_days: int = 30) -> Any:
        """Predict development trajectory to target stage"""
        return self._mapper.predict_development_trajectory(target_stage, time_horizon_days)


class StreamingPhiProcessorAdapter(IStreamingPhiProcessor):
    """
    Adapter for StreamingPhiCalculator to implement IStreamingPhiProcessor interface
    """
    
    def __init__(self,
                 streaming_mode: Any = None,
                 default_window_size: float = 60.0,
                 max_concurrent_windows: int = 10,
                 target_throughput_rps: int = 1000):
        """
        Initialize streaming phi processor adapter
        
        Args:
            streaming_mode: Processing mode
            default_window_size: Default window size in seconds
            max_concurrent_windows: Maximum concurrent processing windows
            target_throughput_rps: Target throughput in requests per second
        """
        # Import here to avoid circular dependencies
        from streaming_phi_calculator import StreamingMode
        
        if streaming_mode is None:
            streaming_mode = StreamingMode.ADAPTIVE
            
        self._processor = StreamingPhiCalculator(
            streaming_mode=streaming_mode,
            default_window_size=default_window_size,
            max_concurrent_windows=max_concurrent_windows,
            target_throughput_rps=target_throughput_rps
        )
    
    async def start_streaming(self) -> None:
        """Start streaming phi calculation"""
        await self._processor.start_streaming()
    
    async def stop_streaming(self) -> None:
        """Stop streaming phi calculation"""
        await self._processor.stop_streaming()
    
    async def submit_event(self, event: Any) -> bool:
        """
        Submit event for streaming processing
        
        Args:
            event: Streaming phi event
            
        Returns:
            bool: True if event was queued successfully
        """
        return await self._processor.submit_event(event)
    
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
        async for result in self._processor.process_event_stream(event_stream, stream_id):
            yield result
    
    async def get_streaming_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics"""
        return await self._processor.get_streaming_stats()


class ConsciousnessDevelopmentAnalyzerAdapter(IConsciousnessDevelopmentAnalyzer):
    """
    Adapter for ConsciousnessDevelopmentAnalyzer to implement IConsciousnessDevelopmentAnalyzer interface
    """
    
    def __init__(self):
        """Initialize consciousness development analyzer adapter"""
        self._analyzer = ConsciousnessDevelopmentAnalyzer()
    
    async def analyze_development_pattern(self,
                                        development_history: Optional[List[Any]] = None) -> Any:
        """
        Analyze long-term development pattern
        
        Args:
            development_history: Optional history override
            
        Returns:
            DevelopmentPattern: Identified pattern type
        """
        return await self._analyzer.analyze_development_pattern(development_history)
    
    async def compare_with_norms(self, current_metrics: Any) -> Dict[str, Any]:
        """
        Compare current development with established norms
        
        Args:
            current_metrics: Current development metrics
            
        Returns:
            Dict: Comprehensive comparison analysis
        """
        return await self._analyzer.compare_with_norms(current_metrics)
    
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
        return await self._analyzer.generate_personalized_recommendations(
            current_metrics, goal_preferences
        )
    
    def get_development_insights(self) -> List[Any]:
        """Generate development insights from analysis"""
        return self._analyzer.get_development_insights()


class AxiomValidatorAdapter(IAxiomValidator):
    """
    Adapter for IIT4AxiomValidator to implement IAxiomValidator interface
    """
    
    def __init__(self, phi_calculator: IPhiCalculator):
        """
        Initialize axiom validator adapter
        
        Args:
            phi_calculator: Phi calculator for validation
        """
        # Extract the underlying calculator for the validator
        if hasattr(phi_calculator, '_calculator'):
            underlying_calculator = phi_calculator._calculator
        else:
            # Assume it's already the correct type
            underlying_calculator = phi_calculator
            
        self._validator = IIT4AxiomValidator(underlying_calculator)
    
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
        return self._validator.validate_all_axioms(phi_structure, system_state)
    
    def validate_existence(self, phi_structure: Any, system_state: np.ndarray) -> bool:
        """Validate axiom of existence"""
        return self._validator.validate_existence(phi_structure, system_state)
    
    def validate_intrinsicality(self, phi_structure: Any) -> bool:
        """Validate axiom of intrinsicality"""
        return self._validator.validate_intrinsicality(phi_structure)
    
    def validate_information(self, phi_structure: Any) -> bool:
        """Validate axiom of information"""
        return self._validator.validate_information(phi_structure)
    
    def validate_integration(self, phi_structure: Any) -> bool:
        """Validate axiom of integration"""
        return self._validator.validate_integration(phi_structure)
    
    def validate_exclusion(self, phi_structure: Any) -> bool:
        """Validate axiom of exclusion"""
        return self._validator.validate_exclusion(phi_structure)
    
    def validate_composition(self, phi_structure: Any) -> bool:
        """Validate axiom of composition"""
        return self._validator.validate_composition(phi_structure)


class IntrinsicDifferenceCalculatorAdapter(IIntrinsicDifferenceCalculator):
    """
    Adapter for IntrinsicDifferenceCalculator to implement IIntrinsicDifferenceCalculator interface
    """
    
    def __init__(self, precision: float = 1e-10):
        """
        Initialize intrinsic difference calculator adapter
        
        Args:
            precision: Numerical precision for calculations
        """
        self._calculator = IntrinsicDifferenceCalculator(precision)
    
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
        return self._calculator.compute_id(
            mechanism, purview, tpm, current_state, direction
        )


class PhiPredictorAdapter(IPhiPredictor):
    """
    Adapter for PhiPredictor to implement IPhiPredictor interface
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize phi predictor adapter
        
        Args:
            model_type: Type of ML model
        """
        self._predictor = PhiPredictor(model_type)
    
    def add_training_sample(self, features: Dict[str, float], actual_phi: float) -> None:
        """Add training sample for model improvement"""
        self._predictor.add_training_sample(features, actual_phi)
    
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
        return await self._predictor.predict_phi(current_features, prediction_horizon_seconds)
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        return self._predictor.get_prediction_stats()


class PhiCacheAdapter(IPhiCache):
    """
    Adapter for PhiCache to implement IPhiCache interface
    """
    
    def __init__(self, max_memory_mb: int = 500, compression_threshold: int = 1000):
        """
        Initialize phi cache adapter
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            compression_threshold: Number of entries before compression
        """
        self._cache = PhiCache(max_memory_mb, compression_threshold)
    
    async def get(self, cache_key: str) -> Optional[Any]:
        """Get cached result"""
        # PhiCache expects an event object, so we need a simple wrapper
        class CacheKeyEvent:
            def get_content_hash(self):
                return cache_key
        
        event = CacheKeyEvent()
        return await self._cache.get(event)
    
    async def put(self, cache_key: str, result: Any) -> None:
        """Cache result"""
        # Similar wrapper for put operation
        class CacheKeyEvent:
            def get_content_hash(self):
                return cache_key
        
        event = CacheKeyEvent()
        await self._cache.put(event, result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self._cache.get_stats()


class SimpleConsciousnessDetector(IConsciousnessDetector):
    """
    Simple implementation of consciousness detector interface
    """
    
    def __init__(self,
                 phi_calculator: IPhiCalculator,
                 experiential_calculator: IExperientialPhiCalculator):
        """
        Initialize consciousness detector
        
        Args:
            phi_calculator: Phi calculator for basic consciousness detection
            experiential_calculator: Experiential phi calculator for quality analysis
        """
        self._phi_calculator = phi_calculator
        self._experiential_calculator = experiential_calculator
    
    async def detect_consciousness_level(self, input_data: Any) -> float:
        """
        Detect current consciousness level
        
        Args:
            input_data: Input for consciousness analysis
            
        Returns:
            float: Consciousness level (0.0 to 1.0)
        """
        try:
            # Handle different input types
            if isinstance(input_data, dict):
                if 'experiential_concepts' in input_data:
                    # Experiential input
                    result = await self._experiential_calculator.calculate_experiential_phi(
                        input_data['experiential_concepts'],
                        input_data.get('temporal_context'),
                        input_data.get('narrative_context')
                    )
                    return result.consciousness_level
                elif 'system_state' in input_data and 'connectivity_matrix' in input_data:
                    # Phi structure input
                    phi_structure = self._phi_calculator.calculate_phi(
                        input_data['system_state'],
                        input_data['connectivity_matrix'],
                        input_data.get('tpm')
                    )
                    # Convert phi to consciousness level (0-1 scale)
                    return min(1.0, phi_structure.total_phi / 10.0)
            
            # Default low consciousness
            return 0.1
            
        except Exception as e:
            logger.error(f"Consciousness detection failed: {e}")
            return 0.0
    
    async def analyze_consciousness_quality(self, input_data: Any) -> Dict[str, float]:
        """
        Analyze consciousness quality metrics
        
        Returns:
            Dict with quality metrics
        """
        try:
            if isinstance(input_data, dict) and 'experiential_concepts' in input_data:
                result = await self._experiential_calculator.calculate_experiential_phi(
                    input_data['experiential_concepts'],
                    input_data.get('temporal_context'),
                    input_data.get('narrative_context')
                )
                
                return {
                    'integration_quality': result.integration_quality,
                    'temporal_depth': result.temporal_depth,
                    'self_reference_strength': result.self_reference_strength,
                    'narrative_coherence': result.narrative_coherence,
                    'experiential_purity': result.experiential_purity
                }
            
            # Default quality metrics
            return {
                'integration_quality': 0.5,
                'temporal_depth': 0.5,
                'self_reference_strength': 0.5,
                'narrative_coherence': 0.5,
                'experiential_purity': 0.5
            }
            
        except Exception as e:
            logger.error(f"Consciousness quality analysis failed: {e}")
            return {
                'integration_quality': 0.0,
                'temporal_depth': 0.0,
                'self_reference_strength': 0.0,
                'narrative_coherence': 0.0,
                'experiential_purity': 0.0
            }


class SimpleExperientialMemoryRepository(IExperientialMemoryRepository):
    """
    Simple in-memory implementation of experiential memory repository
    """
    
    def __init__(self, max_memories: int = 10000):
        """
        Initialize memory repository
        
        Args:
            max_memories: Maximum number of memories to store
        """
        self._memories: Dict[str, Dict[str, Any]] = {}
        self._temporal_index: List[Tuple[datetime, str]] = []
        self._max_memories = max_memories
    
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
        try:
            timestamp = datetime.now()
            
            memory_entry = {
                'experience_id': experience_id,
                'timestamp': timestamp,
                'experiential_data': experiential_data,
                'phi_result': phi_result
            }
            
            self._memories[experience_id] = memory_entry
            self._temporal_index.append((timestamp, experience_id))
            
            # Maintain size limit
            if len(self._memories) > self._max_memories:
                # Remove oldest memories
                oldest_entries = sorted(self._temporal_index, key=lambda x: x[0])
                for _, old_id in oldest_entries[:len(self._memories) - self._max_memories]:
                    if old_id in self._memories:
                        del self._memories[old_id]
                
                # Update temporal index
                self._temporal_index = [
                    (ts, exp_id) for ts, exp_id in self._temporal_index
                    if exp_id in self._memories
                ]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store experience {experience_id}: {e}")
            return False
    
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
        try:
            results = []
            
            for memory_entry in self._memories.values():
                # Simple matching logic
                matches = True
                
                for key, value in query_params.items():
                    if key in memory_entry:
                        if memory_entry[key] != value:
                            matches = False
                            break
                
                if matches:
                    results.append(memory_entry)
                    if len(results) >= limit:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve experiences: {e}")
            return []
    
    async def get_temporal_sequence(self,
                                  start_time: datetime,
                                  end_time: datetime) -> List[Dict[str, Any]]:
        """Get experiences in temporal sequence"""
        try:
            results = []
            
            for timestamp, experience_id in self._temporal_index:
                if start_time <= timestamp <= end_time:
                    if experience_id in self._memories:
                        results.append(self._memories[experience_id])
            
            # Sort by timestamp
            results.sort(key=lambda x: x['timestamp'])
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get temporal sequence: {e}")
            return []


class ConsciousnessSystemFactoryImpl(IConsciousnessSystemFactory):
    """
    Implementation of consciousness system factory
    """
    
    def create_phi_calculator(self, config: Dict[str, Any]) -> IPhiCalculator:
        """Create phi calculator instance"""
        precision = config.get('precision', 1e-10)
        max_mechanism_size = config.get('max_mechanism_size', 8)
        return PhiCalculatorAdapter(precision, max_mechanism_size)
    
    def create_experiential_phi_calculator(self, config: Dict[str, Any]) -> IExperientialPhiCalculator:
        """Create experiential phi calculator instance"""
        precision = config.get('precision', 1e-10)
        max_concept_size = config.get('max_concept_size', 8)
        return ExperientialPhiCalculatorAdapter(precision, max_concept_size)
    
    def create_consciousness_detector(self, config: Dict[str, Any]) -> IConsciousnessDetector:
        """Create consciousness detector instance"""
        # This would normally be injected, but for now create dependencies
        phi_calc = self.create_phi_calculator(config)
        exp_calc = self.create_experiential_phi_calculator(config)
        return SimpleConsciousnessDetector(phi_calc, exp_calc)
    
    def create_development_stage_manager(self, config: Dict[str, Any]) -> IDevelopmentStageManager:
        """Create development stage manager instance"""
        return DevelopmentStageManagerAdapter()
    
    def create_memory_repository(self, config: Dict[str, Any]) -> IExperientialMemoryRepository:
        """Create memory repository instance"""
        max_memories = config.get('max_memories', 10000)
        return SimpleExperientialMemoryRepository(max_memories)
    
    def create_streaming_processor(self, config: Dict[str, Any]) -> IStreamingPhiProcessor:
        """Create streaming processor instance"""
        return StreamingPhiProcessorAdapter(
            default_window_size=config.get('window_size', 60.0),
            max_concurrent_windows=config.get('max_windows', 10),
            target_throughput_rps=config.get('target_throughput', 1000)
        )
    
    def create_development_analyzer(self, config: Dict[str, Any]) -> IConsciousnessDevelopmentAnalyzer:
        """Create development analyzer instance"""
        return ConsciousnessDevelopmentAnalyzerAdapter()