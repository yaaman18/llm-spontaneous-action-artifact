"""
Learning Domain Policies.

Policies that encapsulate complex decision-making logic for adaptive
learning rates, environmental coupling adaptation, and prediction error
regulation using the Policy pattern within the enactivist framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from ..value_objects.prediction_state import PredictionState
from ..value_objects.learning_parameters import LearningParameters


class LearningPolicy(ABC):
    """
    Abstract base class for learning-related policies.
    
    Implements the Policy pattern for encapsulating complex business
    logic and decision-making rules about learning processes.
    """
    
    @abstractmethod
    def apply(self, context: Any) -> Any:
        """
        Apply the policy to the given context.
        
        Args:
            context: Context for policy application
            
        Returns:
            Result of policy application
        """
        pass


class AdaptiveLearningRatePolicy(LearningPolicy):
    """
    Policy for adaptive learning rate adjustment.
    
    Implements enactivist learning where learning rates adapt based on
    environmental coupling strength, prediction quality, and temporal
    dynamics to maintain effective environmental interaction.
    """
    
    def __init__(
        self,
        base_learning_rate: float = 0.01,
        min_learning_rate: float = 0.001,
        max_learning_rate: float = 0.1,
        coupling_influence: float = 0.5,
        error_influence: float = 0.3,
        momentum_factor: float = 0.1
    ):
        """
        Initialize adaptive learning rate policy.
        
        Args:
            base_learning_rate: Base learning rate
            min_learning_rate: Minimum allowed learning rate
            max_learning_rate: Maximum allowed learning rate
            coupling_influence: Influence of environmental coupling
            error_influence: Influence of prediction error
            momentum_factor: Momentum factor for smooth transitions
        """
        self.base_learning_rate = base_learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.coupling_influence = coupling_influence
        self.error_influence = error_influence
        self.momentum_factor = momentum_factor
        self._previous_learning_rates: List[float] = []
    
    def calculate_adaptive_rate(
        self,
        epoch: int,
        prediction_state: Optional[PredictionState],
        learning_params: LearningParameters
    ) -> float:
        """
        Calculate adaptive learning rate based on current context.
        
        Args:
            epoch: Current learning epoch
            prediction_state: Current prediction state
            learning_params: Current learning parameters
            
        Returns:
            Adapted learning rate
        """
        # Start with base rate adjusted for epoch
        base_rate = learning_params.current_learning_rate(epoch)
        
        adaptation_factors = []
        
        # Factor 1: Prediction error adaptation
        if prediction_state is not None:
            error_factor = self._calculate_error_adaptation_factor(prediction_state)
            adaptation_factors.append(error_factor * self.error_influence)
        
        # Factor 2: Environmental coupling adaptation (would be passed via context)
        # For now, using a placeholder - in practice, this would come from aggregate
        coupling_factor = self._calculate_coupling_adaptation_factor(epoch)
        adaptation_factors.append(coupling_factor * self.coupling_influence)
        
        # Factor 3: Temporal momentum
        momentum_factor = self._calculate_momentum_factor()
        adaptation_factors.append(momentum_factor * self.momentum_factor)
        
        # Calculate final adaptive rate
        total_adaptation = sum(adaptation_factors) if adaptation_factors else 0
        adaptive_rate = base_rate * (1.0 + total_adaptation)
        
        # Apply bounds
        adaptive_rate = max(self.min_learning_rate, min(adaptive_rate, self.max_learning_rate))
        
        # Store for momentum calculation
        self._previous_learning_rates.append(adaptive_rate)
        if len(self._previous_learning_rates) > 10:  # Keep only recent rates
            self._previous_learning_rates.pop(0)
        
        return adaptive_rate
    
    def _calculate_error_adaptation_factor(self, prediction_state: PredictionState) -> float:
        """
        Calculate adaptation factor based on prediction error.
        
        Higher errors -> higher learning rate (up to a point)
        Very high errors -> lower learning rate (instability protection)
        
        Args:
            prediction_state: Current prediction state
            
        Returns:
            Error adaptation factor [-1, 1]
        """
        total_error = prediction_state.total_error
        
        # Optimal error range for aggressive learning
        optimal_error_range = (0.5, 2.0)
        
        if total_error < optimal_error_range[0]:
            # Low error - can reduce learning rate
            return -0.3
        elif total_error <= optimal_error_range[1]:
            # Medium error - increase learning rate proportionally
            normalized_error = (total_error - optimal_error_range[0]) / (
                optimal_error_range[1] - optimal_error_range[0]
            )
            return 0.5 * normalized_error
        else:
            # High error - reduce learning rate for stability
            excess_error = total_error - optimal_error_range[1]
            reduction_factor = min(excess_error / 5.0, 0.8)  # Cap reduction
            return -reduction_factor
    
    def _calculate_coupling_adaptation_factor(self, epoch: int) -> float:
        """
        Calculate adaptation factor based on environmental coupling.
        
        Args:
            epoch: Current epoch (used for coupling estimation)
            
        Returns:
            Coupling adaptation factor [-1, 1]
        """
        # Simplified coupling estimation based on epoch
        # In practice, this would use actual coupling measurements
        
        # Early epochs: assume building coupling
        if epoch < 50:
            return 0.2  # Slight increase for exploration
        elif epoch < 200:
            return 0.0  # Stable phase
        else:
            return -0.1  # Mature phase, reduce rate
    
    def _calculate_momentum_factor(self) -> float:
        """
        Calculate momentum factor based on learning rate history.
        
        Returns:
            Momentum factor [-1, 1]
        """
        if len(self._previous_learning_rates) < 3:
            return 0.0
        
        recent_rates = self._previous_learning_rates[-3:]
        
        # Check for trends
        if all(recent_rates[i] <= recent_rates[i+1] for i in range(len(recent_rates)-1)):
            # Increasing trend - continue momentum
            return 0.1
        elif all(recent_rates[i] >= recent_rates[i+1] for i in range(len(recent_rates)-1)):
            # Decreasing trend - continue momentum
            return -0.1
        else:
            # Oscillating - dampen changes
            return 0.0


class EnvironmentalCouplingPolicy(LearningPolicy):
    """
    Policy for adapting learning based on environmental coupling strength.
    
    Implements enactivist principles where learning effectiveness depends
    on the quality of structural coupling between system and environment.
    """
    
    def __init__(
        self,
        coupling_sensitivity: float = 0.8,
        adaptation_rate: float = 0.1,
        stability_requirement: float = 0.6,
        coupling_memory_length: int = 20
    ):
        """
        Initialize environmental coupling policy.
        
        Args:
            coupling_sensitivity: Sensitivity to coupling changes
            adaptation_rate: Rate of parameter adaptation
            stability_requirement: Required coupling stability
            coupling_memory_length: Length of coupling history to maintain
        """
        self.coupling_sensitivity = coupling_sensitivity
        self.adaptation_rate = adaptation_rate
        self.stability_requirement = stability_requirement
        self.coupling_memory_length = coupling_memory_length
        self._coupling_history: List[float] = []
    
    def adapt_learning_to_coupling(
        self,
        current_params: LearningParameters,
        coupling_strength: float
    ) -> LearningParameters:
        """
        Adapt learning parameters based on environmental coupling strength.
        
        Args:
            current_params: Current learning parameters
            coupling_strength: Current environmental coupling strength
            
        Returns:
            Adapted learning parameters
        """
        # Store coupling history
        self._coupling_history.append(coupling_strength)
        if len(self._coupling_history) > self.coupling_memory_length:
            self._coupling_history.pop(0)
        
        # Calculate coupling stability
        coupling_stability = self._calculate_coupling_stability()
        
        # Adapt learning rate based on coupling
        adapted_learning_rate = self._adapt_learning_rate_for_coupling(
            current_params.initial_learning_rate, coupling_strength, coupling_stability
        )
        
        # Adapt neighborhood parameters for SOM
        adapted_neighborhood_radius = self._adapt_neighborhood_for_coupling(
            current_params.neighborhood_radius, coupling_strength
        )
        
        # Create adapted parameters
        return LearningParameters(
            initial_learning_rate=adapted_learning_rate,
            learning_rate_decay=current_params.learning_rate_decay,
            neighborhood_radius=adapted_neighborhood_radius,
            neighborhood_decay=current_params.neighborhood_decay
        )
    
    def adapt_to_environmental_change(
        self,
        current_params: LearningParameters,
        change_requirements: Dict[str, float],
        adaptation_strength: float
    ) -> LearningParameters:
        """
        Adapt parameters in response to environmental changes.
        
        Args:
            current_params: Current learning parameters
            change_requirements: Required adaptations
            adaptation_strength: Strength of adaptation response
            
        Returns:
            Adapted learning parameters
        """
        # Extract adaptation requirements
        lr_adjustment = change_requirements.get('learning_rate_adjustment', 0.0)
        neighborhood_adjustment = change_requirements.get('neighborhood_adjustment', 0.0)
        precision_adjustment = change_requirements.get('precision_reweighting', 0.0)
        
        # Apply adaptations with strength scaling
        new_learning_rate = current_params.initial_learning_rate * (
            1.0 + lr_adjustment * adaptation_strength
        )
        
        new_neighborhood_radius = current_params.neighborhood_radius * (
            1.0 + neighborhood_adjustment * adaptation_strength
        )
        
        # Ensure bounds
        new_learning_rate = max(0.001, min(new_learning_rate, 0.1))
        new_neighborhood_radius = max(1.0, min(new_neighborhood_radius, 10.0))
        
        return LearningParameters(
            initial_learning_rate=new_learning_rate,
            learning_rate_decay=current_params.learning_rate_decay,
            neighborhood_radius=new_neighborhood_radius,
            neighborhood_decay=current_params.neighborhood_decay
        )
    
    def _calculate_coupling_stability(self) -> float:
        """
        Calculate stability of environmental coupling.
        
        Returns:
            Coupling stability score [0, 1]
        """
        if len(self._coupling_history) < 5:
            return 0.5  # Neutral when insufficient history
        
        # Calculate variance in recent coupling values
        recent_couplings = self._coupling_history[-10:]
        mean_coupling = sum(recent_couplings) / len(recent_couplings)
        variance = sum((c - mean_coupling) ** 2 for c in recent_couplings) / len(recent_couplings)
        
        # Convert variance to stability score
        stability = max(0.0, 1.0 - variance * 10)  # Scale variance
        return min(1.0, stability)
    
    def _adapt_learning_rate_for_coupling(
        self,
        current_rate: float,
        coupling_strength: float,
        coupling_stability: float
    ) -> float:
        """
        Adapt learning rate based on coupling characteristics.
        
        Args:
            current_rate: Current learning rate
            coupling_strength: Environmental coupling strength
            coupling_stability: Coupling stability score
            
        Returns:
            Adapted learning rate
        """
        # Strong coupling + stable -> can use higher learning rate
        if coupling_strength > 0.7 and coupling_stability > self.stability_requirement:
            adaptation_factor = 1.0 + (coupling_strength - 0.7) * self.coupling_sensitivity
        # Weak coupling -> use lower learning rate for safety
        elif coupling_strength < 0.3:
            adaptation_factor = 0.5 + coupling_strength * 0.5
        # Unstable coupling -> reduce learning rate
        elif coupling_stability < self.stability_requirement:
            adaptation_factor = coupling_stability
        else:
            adaptation_factor = 1.0
        
        adapted_rate = current_rate * adaptation_factor
        return max(0.001, min(adapted_rate, 0.1))
    
    def _adapt_neighborhood_for_coupling(
        self,
        current_radius: float,
        coupling_strength: float
    ) -> float:
        """
        Adapt neighborhood radius based on coupling strength.
        
        Args:
            current_radius: Current neighborhood radius
            coupling_strength: Environmental coupling strength
            
        Returns:
            Adapted neighborhood radius
        """
        # Strong coupling -> can use smaller neighborhood (more precise)
        if coupling_strength > 0.7:
            adaptation_factor = 0.8 + (1.0 - coupling_strength) * 0.2
        # Weak coupling -> use larger neighborhood (more exploration)
        elif coupling_strength < 0.3:
            adaptation_factor = 1.2 + (0.3 - coupling_strength) * 2.0
        else:
            adaptation_factor = 1.0
        
        adapted_radius = current_radius * adaptation_factor
        return max(1.0, min(adapted_radius, 10.0))


class PredictionErrorRegulationPolicy(LearningPolicy):
    """
    Policy for regulating prediction error dynamics.
    
    Implements error regulation that maintains effective learning while
    preventing instability, based on enactivist principles of error
    minimization through environmental interaction.
    """
    
    def __init__(
        self,
        target_error_range: tuple = (0.1, 1.0),
        error_regulation_strength: float = 0.3,
        instability_threshold: float = 5.0,
        convergence_tolerance: float = 0.01
    ):
        """
        Initialize prediction error regulation policy.
        
        Args:
            target_error_range: Target range for prediction errors
            error_regulation_strength: Strength of error regulation
            instability_threshold: Threshold for detecting instability
            convergence_tolerance: Tolerance for convergence detection
        """
        self.target_error_range = target_error_range
        self.error_regulation_strength = error_regulation_strength
        self.instability_threshold = instability_threshold
        self.convergence_tolerance = convergence_tolerance
        self._error_history: List[float] = []
    
    def regulate_prediction_errors(
        self,
        current_errors: List[float],
        learning_context: Dict[str, Any]
    ) -> List[float]:
        """
        Regulate prediction errors to maintain learning effectiveness.
        
        Args:
            current_errors: Current hierarchical prediction errors
            learning_context: Context about learning state
            
        Returns:
            Regulated prediction errors
        """
        regulated_errors = current_errors.copy()
        total_error = sum(abs(e) for e in current_errors)
        
        # Store error history
        self._error_history.append(total_error)
        if len(self._error_history) > 50:
            self._error_history.pop(0)
        
        # Check for instability
        if total_error > self.instability_threshold:
            regulated_errors = self._apply_stability_regulation(regulated_errors)
        
        # Check for convergence stagnation
        elif self._detect_convergence_stagnation():
            regulated_errors = self._apply_exploration_boost(regulated_errors)
        
        # Apply target range regulation
        regulated_errors = self._apply_target_range_regulation(regulated_errors)
        
        return regulated_errors
    
    def _apply_stability_regulation(self, errors: List[float]) -> List[float]:
        """
        Apply regulation to prevent instability.
        
        Args:
            errors: Current prediction errors
            
        Returns:
            Stability-regulated errors
        """
        # Reduce error magnitudes to prevent instability
        regulation_factor = self.instability_threshold / (sum(abs(e) for e in errors) + 1e-10)
        regulation_factor = min(regulation_factor, 1.0) * self.error_regulation_strength
        
        regulated_errors = []
        for error in errors:
            regulated_error = error * (1.0 - regulation_factor + regulation_factor * 0.5)
            regulated_errors.append(regulated_error)
        
        return regulated_errors
    
    def _apply_exploration_boost(self, errors: List[float]) -> List[float]:
        """
        Apply exploration boost to overcome convergence stagnation.
        
        Args:
            errors: Current prediction errors
            
        Returns:
            Exploration-boosted errors
        """
        # Add small random perturbations to encourage exploration
        boosted_errors = []
        for error in errors:
            exploration_noise = np.random.normal(0, 0.1) * self.error_regulation_strength
            boosted_error = error + exploration_noise
            boosted_errors.append(boosted_error)
        
        return boosted_errors
    
    def _apply_target_range_regulation(self, errors: List[float]) -> List[float]:
        """
        Apply regulation to maintain errors within target range.
        
        Args:
            errors: Current prediction errors
            
        Returns:
            Range-regulated errors
        """
        total_error = sum(abs(e) for e in errors)
        target_min, target_max = self.target_error_range
        
        regulated_errors = errors.copy()
        
        if total_error < target_min:
            # Errors too low - might need more learning challenge
            boost_factor = target_min / (total_error + 1e-10)
            boost_factor = min(boost_factor, 2.0) * self.error_regulation_strength
            regulated_errors = [e * (1.0 + boost_factor) for e in errors]
        
        elif total_error > target_max:
            # Errors too high - reduce for stability
            reduction_factor = target_max / total_error
            reduction_factor = max(reduction_factor, 0.5) * self.error_regulation_strength
            regulated_errors = [e * reduction_factor for e in errors]
        
        return regulated_errors
    
    def _detect_convergence_stagnation(self) -> bool:
        """
        Detect if learning has stagnated without proper convergence.
        
        Returns:
            True if convergence stagnation is detected
        """
        if len(self._error_history) < 20:
            return False
        
        recent_errors = self._error_history[-20:]
        
        # Check if errors are stable but not converged
        error_variance = np.var(recent_errors)
        mean_error = np.mean(recent_errors)
        
        # Low variance but high mean error indicates stagnation
        is_stable = error_variance < 0.1
        is_not_converged = mean_error > self.target_error_range[1]
        
        return is_stable and is_not_converged