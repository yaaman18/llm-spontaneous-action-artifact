"""
Metacognitive Monitor Domain Service.

Abstract service interface for metacognitive monitoring and self-awareness
capabilities in the enactive consciousness system. Implements higher-order
cognitive processes that monitor and evaluate system performance.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy.typing as npt
from ..value_objects.consciousness_state import ConsciousnessState
from ..value_objects.prediction_state import PredictionState
from ..value_objects.phi_value import PhiValue
from ..value_objects.probability_distribution import ProbabilityDistribution


class MetacognitiveDimension(Enum):
    """Dimensions of metacognitive monitoring."""
    CONFIDENCE = "confidence"
    AWARENESS = "awareness" 
    CONTROL = "control"
    STRATEGY = "strategy"
    KNOWLEDGE = "knowledge"
    REGULATION = "regulation"


class MonitoringLevel(Enum):
    """Levels of metacognitive monitoring."""
    OBJECT_LEVEL = "object_level"      # First-order cognitive processes
    META_LEVEL = "meta_level"          # Monitoring of object-level
    META_META_LEVEL = "meta_meta_level"  # Monitoring of meta-level


@dataclass
class MetacognitiveAssessment:
    """Immutable assessment of metacognitive state."""
    confidence_level: float
    awareness_depth: float
    control_effectiveness: float
    strategy_appropriateness: float
    knowledge_accuracy: float
    regulation_success: float
    timestamp: datetime
    monitoring_level: MonitoringLevel
    assessment_metadata: Dict[str, Any]


class MetacognitiveMonitorService(ABC):
    """
    Abstract domain service for metacognitive monitoring.
    
    This service implements higher-order cognitive processes that
    monitor, evaluate, and regulate the consciousness system's
    own cognitive processes. It provides self-awareness and
    adaptive control capabilities.
    
    Key responsibilities:
    - Self-monitoring of cognitive processes
    - Confidence estimation and calibration
    - Strategy selection and adaptation
    - Meta-memory and meta-learning
    - Cognitive control and regulation
    """

    @abstractmethod
    def assess_prediction_confidence(
        self,
        prediction_state: PredictionState,
        historical_performance: List[PredictionState],
        context_factors: Dict[str, Any]
    ) -> float:
        """
        Assess confidence in current predictions based on past performance.
        
        Implements metacognitive confidence assessment by evaluating
        prediction quality patterns, uncertainty estimates, and
        contextual factors that affect prediction reliability.
        
        Args:
            prediction_state: Current prediction state to evaluate
            historical_performance: Recent prediction performance history
            context_factors: Contextual information affecting confidence
            
        Returns:
            Confidence level [0, 1] where 1 is maximum confidence
            
        Raises:
            ValueError: If prediction state or history is invalid
        """
        pass

    @abstractmethod
    def monitor_learning_effectiveness(
        self,
        learning_trajectory: List[Dict[str, float]],
        time_window: timedelta,
        performance_metrics: List[str]
    ) -> Dict[str, float]:
        """
        Monitor the effectiveness of ongoing learning processes.
        
        Evaluates learning progress across multiple dimensions,
        identifying patterns of improvement, plateaus, or degradation
        in performance metrics.
        
        Args:
            learning_trajectory: Time series of learning performance
            time_window: Time window for analysis
            performance_metrics: Metrics to evaluate ("accuracy", "convergence", etc.)
            
        Returns:
            Dictionary with effectiveness scores for each metric
            
        Raises:
            ValueError: If trajectory data is insufficient
        """
        pass

    @abstractmethod
    def detect_metacognitive_failures(
        self,
        consciousness_states: List[ConsciousnessState],
        failure_patterns: List[str],
        detection_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Detect metacognitive failures and biases in reasoning.
        
        Identifies patterns indicating metacognitive failures such as:
        - Overconfidence bias
        - Dunning-Kruger effects
        - Illusion of knowledge
        - Confirmation bias
        - Anchoring effects
        
        Args:
            consciousness_states: Recent consciousness state history
            failure_patterns: Types of failures to detect
            detection_threshold: Confidence threshold for detection
            
        Returns:
            List of detected failure instances with details
            
        Raises:
            ValueError: If states or patterns are invalid
        """
        pass

    @abstractmethod
    def recommend_strategy_adaptations(
        self,
        current_performance: Dict[str, float],
        performance_goals: Dict[str, float],
        available_strategies: List[str],
        resource_constraints: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Recommend adaptations to cognitive strategies based on performance.
        
        Analyzes performance gaps and recommends specific strategy
        modifications to improve goal achievement within resource
        constraints.
        
        Args:
            current_performance: Current performance metrics
            performance_goals: Target performance levels
            available_strategies: List of available cognitive strategies
            resource_constraints: Available computational/temporal resources
            
        Returns:
            List of strategy recommendations with expected benefits
            
        Raises:
            ValueError: If performance data or goals are invalid
        """
        pass

    @abstractmethod
    def calibrate_confidence_estimates(
        self,
        confidence_history: List[Tuple[float, bool]],
        calibration_method: str = "isotonic_regression"
    ) -> Callable[[float], float]:
        """
        Calibrate confidence estimates to improve accuracy.
        
        Learns a calibration function that maps raw confidence scores
        to calibrated probabilities that better reflect actual
        performance accuracy.
        
        Args:
            confidence_history: History of (confidence, correctness) pairs
            calibration_method: Calibration algorithm to use
            
        Returns:
            Calibration function that maps raw to calibrated confidence
            
        Raises:
            ValueError: If history data is insufficient
            CalibrationError: If calibration fitting fails
        """
        pass

    @abstractmethod
    def assess_meta_awareness_depth(
        self,
        consciousness_state: ConsciousnessState,
        introspection_queries: List[str]
    ) -> Dict[str, float]:
        """
        Assess the depth of metacognitive awareness.
        
        Evaluates how well the system can introspect and report
        on its own internal states, processes, and capabilities.
        
        Args:
            consciousness_state: Current consciousness state
            introspection_queries: Questions to assess self-awareness
            
        Returns:
            Dictionary with awareness depth scores for different aspects
            
        Raises:
            ValueError: If consciousness state is invalid
        """
        pass

    @abstractmethod
    def monitor_attention_allocation(
        self,
        attention_weights: npt.NDArray,
        task_demands: Dict[str, float],
        performance_feedback: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Monitor and evaluate attention allocation effectiveness.
        
        Assesses whether attention resources are being allocated
        optimally given task demands and performance feedback.
        
        Args:
            attention_weights: Current attention weight distribution
            task_demands: Required attention for different task aspects
            performance_feedback: Performance results from current allocation
            
        Returns:
            Dictionary with attention allocation analysis:
            - efficiency: How well attention matches task demands
            - effectiveness: Performance impact of current allocation
            - recommendations: Suggested attention adjustments
            
        Raises:
            ValueError: If attention weights or task demands are invalid
        """
        pass

    @abstractmethod
    def evaluate_cognitive_flexibility(
        self,
        strategy_switching_history: List[Dict[str, Any]],
        context_changes: List[Dict[str, Any]],
        adaptation_latency: List[float]
    ) -> float:
        """
        Evaluate cognitive flexibility and adaptability.
        
        Measures how effectively the system adapts its cognitive
        strategies in response to changing contexts and requirements.
        
        Args:
            strategy_switching_history: Record of strategy changes
            context_changes: Environmental/task context changes
            adaptation_latency: Time taken to adapt to changes
            
        Returns:
            Cognitive flexibility score [0, 1]
            
        Raises:
            ValueError: If history data is inconsistent
        """
        pass

    @abstractmethod
    def generate_metacognitive_insights(
        self,
        performance_data: Dict[str, List[float]],
        context_data: Dict[str, Any],
        insight_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate insights about cognitive processes and patterns.
        
        Analyzes performance and context data to identify patterns,
        strengths, weaknesses, and opportunities for improvement.
        
        Args:
            performance_data: Historical performance across metrics
            context_data: Contextual information about performance
            insight_types: Types of insights to generate
            
        Returns:
            List of insight dictionaries with analysis and recommendations
            
        Raises:
            ValueError: If data is insufficient for insight generation
        """
        pass

    @abstractmethod
    def predict_performance_trajectory(
        self,
        current_state: ConsciousnessState,
        planned_strategies: List[str],
        time_horizon: timedelta,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Predict future performance trajectory given current state and strategies.
        
        Uses metacognitive knowledge to forecast how performance
        will evolve under different strategic choices.
        
        Args:
            current_state: Current consciousness state
            planned_strategies: Cognitive strategies to evaluate
            time_horizon: Prediction time horizon
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with trajectory predictions:
            - predicted_performance: Expected performance over time
            - confidence_intervals: Uncertainty bounds
            - strategy_effectiveness: Expected impact of each strategy
            
        Raises:
            ValueError: If state or strategies are invalid
            PredictionError: If trajectory prediction fails
        """
        pass

    @abstractmethod
    def perform_comprehensive_assessment(
        self,
        consciousness_state: ConsciousnessState,
        performance_history: List[Dict[str, float]],
        assessment_dimensions: List[MetacognitiveDimension]
    ) -> MetacognitiveAssessment:
        """
        Perform comprehensive metacognitive assessment.
        
        Evaluates the system across multiple dimensions of
        metacognitive functioning to provide holistic insight
        into self-awareness and regulatory capabilities.
        
        Args:
            consciousness_state: Current consciousness state
            performance_history: Recent performance data
            assessment_dimensions: Dimensions to evaluate
            
        Returns:
            Comprehensive metacognitive assessment
            
        Raises:
            ValueError: If inputs are invalid
            AssessmentError: If assessment computation fails
        """
        pass


class MetacognitiveMonitorError(Exception):
    """Base exception for metacognitive monitoring operations."""
    pass


class CalibrationError(MetacognitiveMonitorError):
    """Raised when confidence calibration fails."""
    pass


class PredictionError(MetacognitiveMonitorError):
    """Raised when performance trajectory prediction fails."""
    pass


class AssessmentError(MetacognitiveMonitorError):
    """Raised when metacognitive assessment computation fails."""
    pass