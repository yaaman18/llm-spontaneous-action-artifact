"""
Learning Domain Specifications.

Specifications that encapsulate complex business rules for learning
convergence, environmental coupling, and prediction quality using
the Specification pattern within the enactivist framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy.typing as npt

from ..value_objects.prediction_state import PredictionState
from ..value_objects.som_topology import SOMTopology


class LearningSpecification(ABC):
    """
    Abstract base class for learning-related specifications.
    
    Implements the Specification pattern for encapsulating complex
    business rules about learning processes and convergence criteria.
    """
    
    @abstractmethod
    def is_satisfied_by(self, candidate: Any) -> bool:
        """
        Check if the candidate satisfies this specification.
        
        Args:
            candidate: Object to check against specification
            
        Returns:
            True if candidate satisfies the specification
        """
        pass


class LearningConvergenceSpecification(LearningSpecification):
    """
    Specification for learning convergence criteria.
    
    Defines when learning can be considered to have converged based on
    prediction error stabilization, trajectory analysis, and enactivist
    principles of environmental attunement.
    """
    
    def __init__(
        self,
        error_threshold: float = 0.1,
        stability_window: int = 10,
        min_improvement_rate: float = 0.001,
        max_oscillation_amplitude: float = 0.05,
        require_environmental_consistency: bool = True
    ):
        """
        Initialize learning convergence specification.
        
        Args:
            error_threshold: Maximum error for convergence
            stability_window: Number of epochs to check for stability
            min_improvement_rate: Minimum rate of improvement required
            max_oscillation_amplitude: Maximum allowed error oscillation
            require_environmental_consistency: Whether environmental consistency is required
        """
        self.error_threshold = error_threshold
        self.stability_window = stability_window
        self.min_improvement_rate = min_improvement_rate
        self.max_oscillation_amplitude = max_oscillation_amplitude
        self.require_environmental_consistency = require_environmental_consistency
    
    def is_satisfied_by(
        self,
        prediction_state: PredictionState,
        learning_trajectory: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if learning has converged based on prediction state and trajectory.
        
        Args:
            prediction_state: Current prediction state
            learning_trajectory: History of learning progress
            
        Returns:
            True if learning convergence criteria are met
        """
        # Criterion 1: Current error below threshold
        if prediction_state.total_error > self.error_threshold:
            return False
        
        # Criterion 2: Prediction system stability
        if not prediction_state.is_stable:
            return False
        
        # Criterion 3: Trajectory-based convergence analysis
        if not self._analyze_trajectory_convergence(learning_trajectory):
            return False
        
        # Criterion 4: Environmental consistency (enactivist requirement)
        if self.require_environmental_consistency:
            if not self._check_environmental_consistency(learning_trajectory):
                return False
        
        return True
    
    def _analyze_trajectory_convergence(self, trajectory: List[Dict[str, Any]]) -> bool:
        """
        Analyze learning trajectory for convergence indicators.
        
        Args:
            trajectory: Learning trajectory data
            
        Returns:
            True if trajectory shows convergence
        """
        if len(trajectory) < self.stability_window:
            return False
        
        # Get recent errors
        recent_trajectory = trajectory[-self.stability_window:]
        recent_errors = [entry.get('prediction_error', float('inf')) for entry in recent_trajectory]
        
        # Check error stability (low variance)
        if len(recent_errors) > 1:
            mean_error = sum(recent_errors) / len(recent_errors)
            error_variance = sum((e - mean_error) ** 2 for e in recent_errors) / len(recent_errors)
            error_std = error_variance ** 0.5
            
            # Oscillation check
            if error_std > self.max_oscillation_amplitude:
                return False
        
        # Check improvement trend
        if len(trajectory) >= 20:  # Need sufficient history
            early_errors = [
                entry.get('prediction_error', float('inf')) 
                for entry in trajectory[-20:-10]
            ]
            early_mean = sum(early_errors) / len(early_errors) if early_errors else float('inf')
            recent_mean = sum(recent_errors) / len(recent_errors)
            
            improvement_rate = (early_mean - recent_mean) / 10  # Per epoch
            if improvement_rate < self.min_improvement_rate and recent_mean > self.error_threshold * 0.5:
                return False
        
        return True
    
    def _check_environmental_consistency(self, trajectory: List[Dict[str, Any]]) -> bool:
        """
        Check environmental consistency in learning trajectory.
        
        Args:
            trajectory: Learning trajectory data
            
        Returns:
            True if environmental coupling is consistent
        """
        if len(trajectory) < 5:
            return True  # Not enough data to assess
        
        # Check coupling strength consistency
        recent_couplings = [
            entry.get('coupling_strength', 0) 
            for entry in trajectory[-10:]
        ]
        
        if recent_couplings:
            coupling_variance = (
                sum((c - sum(recent_couplings) / len(recent_couplings)) ** 2 
                    for c in recent_couplings) / len(recent_couplings)
            )
            # High variance in coupling indicates inconsistent environmental interaction
            if coupling_variance > 0.1:
                return False
        
        return True
    
    def calculate_convergence_score(
        self,
        prediction_state: PredictionState,
        learning_trajectory: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate continuous convergence score [0, 1].
        
        Args:
            prediction_state: Current prediction state
            learning_trajectory: Learning trajectory history
            
        Returns:
            Convergence score indicating closeness to convergence
        """
        scores = []
        
        # Error score (closer to threshold is better)
        error_score = max(0.0, 1.0 - prediction_state.total_error / self.error_threshold)
        scores.append(error_score)
        
        # Stability score
        stability_score = 1.0 if prediction_state.is_stable else 0.0
        scores.append(stability_score)
        
        # Trajectory stability score
        if len(learning_trajectory) >= self.stability_window:
            recent_errors = [
                entry.get('prediction_error', float('inf')) 
                for entry in learning_trajectory[-self.stability_window:]
            ]
            if recent_errors:
                mean_error = sum(recent_errors) / len(recent_errors)
                error_variance = sum((e - mean_error) ** 2 for e in recent_errors) / len(recent_errors)
                trajectory_score = max(0.0, 1.0 - error_variance / self.max_oscillation_amplitude)
                scores.append(trajectory_score)
        
        return sum(scores) / len(scores) if scores else 0.0


class EnvironmentalCouplingSpecification(LearningSpecification):
    """
    Specification for environmental coupling quality.
    
    Defines criteria for effective environmental coupling based on
    enactivist principles of structural coupling and sensorimotor
    contingencies.
    """
    
    def __init__(
        self,
        min_coupling_strength: float = 0.3,
        max_coupling_variability: float = 0.2,
        require_sensorimotor_consistency: bool = True,
        min_environmental_complexity: float = 0.1
    ):
        """
        Initialize environmental coupling specification.
        
        Args:
            min_coupling_strength: Minimum coupling strength
            max_coupling_variability: Maximum allowed coupling variability
            require_sensorimotor_consistency: Whether sensorimotor consistency is required
            min_environmental_complexity: Minimum environmental complexity
        """
        self.min_coupling_strength = min_coupling_strength
        self.max_coupling_variability = max_coupling_variability
        self.require_sensorimotor_consistency = require_sensorimotor_consistency
        self.min_environmental_complexity = min_environmental_complexity
    
    def is_satisfied_by(self, coupling_data: Dict[str, Any]) -> bool:
        """
        Check if environmental coupling satisfies specification.
        
        Args:
            coupling_data: Data about environmental coupling
            
        Returns:
            True if coupling criteria are met
        """
        coupling_strength = coupling_data.get('coupling_strength', 0.0)
        
        # Criterion 1: Minimum coupling strength
        if coupling_strength < self.min_coupling_strength:
            return False
        
        # Criterion 2: Coupling variability check
        coupling_history = coupling_data.get('coupling_history', [])
        if len(coupling_history) > 1:
            mean_coupling = sum(coupling_history) / len(coupling_history)
            coupling_variance = sum((c - mean_coupling) ** 2 for c in coupling_history) / len(coupling_history)
            coupling_std = coupling_variance ** 0.5
            
            if coupling_std > self.max_coupling_variability:
                return False
        
        # Criterion 3: Environmental complexity
        env_complexity = coupling_data.get('environmental_complexity', 0.0)
        if env_complexity < self.min_environmental_complexity:
            return False
        
        # Criterion 4: Sensorimotor consistency (enactivist requirement)
        if self.require_sensorimotor_consistency:
            sensorimotor_coherence = coupling_data.get('sensorimotor_coherence', 0.0)
            if sensorimotor_coherence < 0.5:
                return False
        
        return True
    
    def is_som_configuration_valid(
        self,
        map_dimensions: Tuple[int, int],
        input_dimensions: int,
        topology: SOMTopology
    ) -> bool:
        """
        Check if SOM configuration supports environmental coupling.
        
        Args:
            map_dimensions: SOM grid dimensions
            input_dimensions: Input vector dimensionality
            topology: SOM topology configuration
            
        Returns:
            True if SOM configuration supports good environmental coupling
        """
        map_size = map_dimensions[0] * map_dimensions[1]
        
        # Criterion 1: Adequate map size relative to input complexity
        min_map_size = max(input_dimensions // 2, 4)
        if map_size < min_map_size:
            return False
        
        # Criterion 2: Not too large (over-parameterization)
        max_map_size = input_dimensions * 10
        if map_size > max_map_size:
            return False
        
        # Criterion 3: Reasonable aspect ratio
        aspect_ratio = max(map_dimensions) / min(map_dimensions)
        if aspect_ratio > 5.0:  # Too elongated
            return False
        
        return True
    
    def calculate_coupling_quality_score(self, coupling_data: Dict[str, Any]) -> float:
        """
        Calculate continuous coupling quality score [0, 1].
        
        Args:
            coupling_data: Environmental coupling data
            
        Returns:
            Quality score for environmental coupling
        """
        scores = []
        
        # Coupling strength score
        coupling_strength = coupling_data.get('coupling_strength', 0.0)
        strength_score = min(coupling_strength / self.min_coupling_strength, 1.0)
        scores.append(strength_score)
        
        # Stability score
        coupling_history = coupling_data.get('coupling_history', [])
        if len(coupling_history) > 1:
            mean_coupling = sum(coupling_history) / len(coupling_history)
            coupling_variance = sum((c - mean_coupling) ** 2 for c in coupling_history) / len(coupling_history)
            stability_score = max(0.0, 1.0 - coupling_variance / self.max_coupling_variability)
            scores.append(stability_score)
        
        # Environmental complexity score
        env_complexity = coupling_data.get('environmental_complexity', 0.0)
        complexity_score = min(env_complexity / self.min_environmental_complexity, 1.0)
        scores.append(complexity_score)
        
        # Sensorimotor consistency score
        if self.require_sensorimotor_consistency:
            sensorimotor_coherence = coupling_data.get('sensorimotor_coherence', 0.0)
            consistency_score = min(sensorimotor_coherence / 0.5, 1.0)
            scores.append(consistency_score)
        
        return sum(scores) / len(scores) if scores else 0.0


class PredictionQualitySpecification(LearningSpecification):
    """
    Specification for prediction quality standards.
    
    Defines criteria for acceptable prediction quality in the context
    of enactive consciousness where predictions must support effective
    environmental interaction.
    """
    
    def __init__(
        self,
        max_total_error: float = 2.0,
        max_error_variance: float = 0.5,
        min_prediction_consistency: float = 0.7,
        require_hierarchical_coherence: bool = True
    ):
        """
        Initialize prediction quality specification.
        
        Args:
            max_total_error: Maximum allowable total prediction error
            max_error_variance: Maximum variance in hierarchical errors
            min_prediction_consistency: Minimum prediction consistency
            require_hierarchical_coherence: Whether hierarchical coherence is required
        """
        self.max_total_error = max_total_error
        self.max_error_variance = max_error_variance
        self.min_prediction_consistency = min_prediction_consistency
        self.require_hierarchical_coherence = require_hierarchical_coherence
    
    def is_satisfied_by(self, prediction_state: PredictionState) -> bool:
        """
        Check if prediction state satisfies quality criteria.
        
        Args:
            prediction_state: Prediction state to evaluate
            
        Returns:
            True if prediction quality criteria are met
        """
        # Criterion 1: Total error within bounds
        if prediction_state.total_error > self.max_total_error:
            return False
        
        # Criterion 2: Error variance (hierarchical coherence)
        if prediction_state.error_variance > self.max_error_variance:
            return False
        
        # Criterion 3: Prediction quality score
        if prediction_state.prediction_quality < self.min_prediction_consistency:
            return False
        
        # Criterion 4: Hierarchical coherence
        if self.require_hierarchical_coherence:
            if not self._check_hierarchical_coherence(prediction_state):
                return False
        
        return True
    
    def _check_hierarchical_coherence(self, prediction_state: PredictionState) -> bool:
        """
        Check hierarchical coherence in prediction errors.
        
        Args:
            prediction_state: Prediction state to check
            
        Returns:
            True if hierarchical errors show coherent pattern
        """
        errors = prediction_state.hierarchical_errors
        
        if len(errors) < 2:
            return True
        
        # Generally, lower levels should have higher precision (lower error)
        # but this is not a strict requirement - allow some flexibility
        
        # Check that no level has extremely high error compared to others
        mean_error = sum(errors) / len(errors)
        for error in errors:
            if error > mean_error * 3:  # Any level 3x higher than mean
                return False
        
        return True
    
    def calculate_quality_score(self, prediction_state: PredictionState) -> float:
        """
        Calculate continuous quality score [0, 1] for prediction state.
        
        Args:
            prediction_state: Prediction state to score
            
        Returns:
            Quality score
        """
        scores = []
        
        # Error score
        error_score = max(0.0, 1.0 - prediction_state.total_error / self.max_total_error)
        scores.append(error_score)
        
        # Variance score
        variance_score = max(0.0, 1.0 - prediction_state.error_variance / self.max_error_variance)
        scores.append(variance_score)
        
        # Consistency score (using prediction_quality property)
        consistency_score = prediction_state.prediction_quality
        scores.append(consistency_score)
        
        # Hierarchical coherence score
        if self.require_hierarchical_coherence:
            coherence_score = 1.0 if self._check_hierarchical_coherence(prediction_state) else 0.0
            scores.append(coherence_score)
        
        # Stability score
        stability_score = 1.0 if prediction_state.is_stable else 0.5
        scores.append(stability_score)
        
        return sum(scores) / len(scores) if scores else 0.0