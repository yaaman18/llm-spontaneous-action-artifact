"""
Learning Aggregate Root.

Aggregate managing predictive coding, self-organizing maps, and Bayesian learning
processes. Ensures learning consistency and enforces enactivism-based learning
invariants where learning emerges from environmental interaction.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid
import numpy.typing as npt

from ..entities.predictive_coding_core import PredictiveCodingCore
from ..entities.self_organizing_map import SelfOrganizingMap
from ..value_objects.prediction_state import PredictionState
from ..value_objects.learning_parameters import LearningParameters
from ..value_objects.precision_weights import PrecisionWeights
from ..value_objects.som_topology import SOMTopology
from ..events.domain_events import (
    LearningEpochCompleted,
    PredictionErrorThresholdCrossed,
    SelfOrganizationConverged,
    EnvironmentalCouplingStrengthened,
    AdaptiveLearningRateChanged
)
from ..specifications.learning_specifications import (
    LearningConvergenceSpecification,
    EnvironmentalCouplingSpecification,
    PredictionQualitySpecification
)
from ..policies.learning_policies import (
    AdaptiveLearningRatePolicy,
    EnvironmentalCouplingPolicy,
    PredictionErrorRegulationPolicy
)


class LearningAggregate:
    """
    Aggregate root for learning processes in enactive consciousness.
    
    Manages the interaction between predictive coding, self-organization,
    and Bayesian inference to ensure coherent learning dynamics that
    reflect enactivist principles of environmental coupling.
    
    Key responsibilities:
    - Coordinate predictive coding and SOM learning
    - Enforce learning invariants and consistency
    - Manage environmental coupling strength
    - Generate learning-related domain events
    - Apply adaptive learning policies
    """
    
    def __init__(
        self,
        aggregate_id: str = None,
        predictive_coding_core: Optional[PredictiveCodingCore] = None,
        self_organizing_map: Optional[SelfOrganizingMap] = None,
        learning_params: Optional[LearningParameters] = None
    ):
        """
        Initialize learning aggregate.
        
        Args:
            aggregate_id: Unique identifier for this aggregate
            predictive_coding_core: Predictive coding entity
            self_organizing_map: SOM entity
            learning_params: Learning parameters configuration
        """
        self._aggregate_id = aggregate_id or str(uuid.uuid4())
        self._predictive_coding_core = predictive_coding_core
        self._self_organizing_map = self_organizing_map
        self._learning_params = learning_params or LearningParameters.create_default()
        
        # Learning state tracking
        self._learning_epochs_completed = 0
        self._current_prediction_state: Optional[PredictionState] = None
        self._environmental_coupling_strength = 0.0
        self._learning_trajectory: List[Dict[str, Any]] = []
        self._domain_events: List[Any] = []
        self._version = 0
        self._created_at = datetime.now()
        
        # Domain specifications for learning rules
        self._convergence_spec = LearningConvergenceSpecification()
        self._coupling_spec = EnvironmentalCouplingSpecification()
        self._quality_spec = PredictionQualitySpecification()
        
        # Domain policies for adaptive learning
        self._learning_rate_policy = AdaptiveLearningRatePolicy()
        self._coupling_policy = EnvironmentalCouplingPolicy()
        self._error_regulation_policy = PredictionErrorRegulationPolicy()
    
    @property
    def aggregate_id(self) -> str:
        """Unique identifier for this aggregate."""
        return self._aggregate_id
    
    @property
    def version(self) -> int:
        """Version number for optimistic locking."""
        return self._version
    
    @property
    def domain_events(self) -> List[Any]:
        """Accumulated domain events."""
        return self._domain_events.copy()
    
    @property
    def learning_epochs_completed(self) -> int:
        """Number of learning epochs completed."""
        return self._learning_epochs_completed
    
    @property
    def environmental_coupling_strength(self) -> float:
        """Strength of environmental coupling [0, 1]."""
        return self._environmental_coupling_strength
    
    @property
    def current_prediction_state(self) -> Optional[PredictionState]:
        """Current prediction state of the learning system."""
        return self._current_prediction_state
    
    @property
    def is_learning_converged(self) -> bool:
        """Check if learning has converged."""
        return (
            self._current_prediction_state is not None and
            self._convergence_spec.is_satisfied_by(
                self._current_prediction_state, self._learning_trajectory
            )
        )
    
    def perform_learning_epoch(
        self,
        input_data: npt.NDArray,
        target_data: Optional[npt.NDArray] = None,
        environmental_context: Optional[Dict[str, Any]] = None
    ) -> PredictionState:
        """
        Perform a complete learning epoch with predictive coding and SOM.
        
        Implements enactivist learning where the system learns through
        environmental interaction, not just pattern matching.
        
        Args:
            input_data: Input sensory data
            target_data: Optional target data for supervised aspects
            environmental_context: Context about environmental interaction
            
        Returns:
            Updated prediction state after learning epoch
            
        Raises:
            LearningInvariantViolation: If learning violates aggregate invariants
        """
        if self._predictive_coding_core is None:
            raise ValueError("Predictive coding core not initialized")
        
        # Calculate environmental coupling strength
        coupling_strength = self._calculate_environmental_coupling(
            input_data, environmental_context
        )
        
        # Apply environmental coupling policy
        adapted_learning_params = self._coupling_policy.adapt_learning_to_coupling(
            self._learning_params, coupling_strength
        )
        
        # Apply adaptive learning rate policy
        current_learning_rate = self._learning_rate_policy.calculate_adaptive_rate(
            self._learning_epochs_completed,
            self._current_prediction_state,
            adapted_learning_params
        )
        
        # Create precision weights based on environmental coupling
        precision_weights = self._create_precision_weights_from_coupling(coupling_strength)
        
        # Perform predictive coding step
        new_prediction_state = self._predictive_coding_core.process_input(
            input_data, precision_weights, current_learning_rate
        )
        
        # Perform SOM learning if available
        if self._self_organizing_map is not None and self._self_organizing_map.is_trained:
            som_learning_params = LearningParameters(
                initial_learning_rate=current_learning_rate,
                learning_rate_decay=adapted_learning_params.learning_rate_decay,
                neighborhood_radius=adapted_learning_params.neighborhood_radius,
                neighborhood_decay=adapted_learning_params.neighborhood_decay
            )
            
            bmu_position = self._self_organizing_map.train_single_iteration(
                input_data.flatten() if input_data.ndim > 1 else input_data,
                som_learning_params
            )
        
        # Validate learning step
        self._validate_learning_step(new_prediction_state, coupling_strength)
        
        # Update aggregate state
        previous_prediction_state = self._current_prediction_state
        self._current_prediction_state = new_prediction_state
        self._environmental_coupling_strength = coupling_strength
        self._learning_epochs_completed += 1
        self._version += 1
        
        # Record learning trajectory
        self._learning_trajectory.append({
            'epoch': self._learning_epochs_completed,
            'prediction_error': new_prediction_state.total_error,
            'coupling_strength': coupling_strength,
            'learning_rate': current_learning_rate,
            'timestamp': datetime.now()
        })
        
        # Generate domain events
        self._generate_learning_events(
            previous_prediction_state, new_prediction_state, coupling_strength
        )
        
        return new_prediction_state
    
    def initialize_som(
        self,
        map_dimensions: Tuple[int, int],
        input_dimensions: int,
        topology: SOMTopology,
        initialization_method: str = "random"
    ) -> None:
        """
        Initialize self-organizing map with environmental awareness.
        
        Args:
            map_dimensions: SOM grid dimensions
            input_dimensions: Input vector dimensionality
            topology: SOM topology configuration
            initialization_method: Weight initialization method
        """
        # Create SOM (this would use a concrete implementation)
        # For now, we'll assume an implementation exists
        
        # Validate SOM configuration against environmental requirements
        if not self._coupling_spec.is_som_configuration_valid(
            map_dimensions, input_dimensions, topology
        ):
            raise LearningInvariantViolation(
                "SOM configuration incompatible with environmental coupling requirements"
            )
        
        # Initialize weights with environmental bias if coupling context exists
        # This would be implemented in concrete SOM class
        
        self._version += 1
    
    def adapt_to_environmental_change(
        self,
        environmental_change_signal: Dict[str, Any],
        adaptation_strength: float = 1.0
    ) -> None:
        """
        Adapt learning parameters in response to environmental change.
        
        Implements enactivist principle of structural coupling where
        the system adapts its internal dynamics to environmental changes.
        
        Args:
            environmental_change_signal: Signal indicating environmental change
            adaptation_strength: Strength of adaptation response [0, 1]
            
        Raises:
            LearningInvariantViolation: If adaptation violates learning invariants
        """
        # Calculate required adaptation
        adaptation_requirements = self._analyze_environmental_change(
            environmental_change_signal
        )
        
        # Apply coupling policy for adaptation
        adapted_params = self._coupling_policy.adapt_to_environmental_change(
            self._learning_params,
            adaptation_requirements,
            adaptation_strength
        )
        
        # Validate adaptation maintains learning invariants
        if not self._validate_parameter_adaptation(adapted_params):
            raise LearningInvariantViolation(
                "Parameter adaptation violates learning invariants"
            )
        
        # Apply adaptation
        previous_params = self._learning_params
        self._learning_params = adapted_params
        self._version += 1
        
        # Generate environmental coupling event
        self._domain_events.append(
            EnvironmentalCouplingStrengthened(
                aggregate_id=self._aggregate_id,
                coupling_strength=self._environmental_coupling_strength,
                adaptation_type=environmental_change_signal.get('change_type', 'unknown'),
                timestamp=datetime.now()
            )
        )
    
    def reset_learning_state(self) -> None:
        """Reset learning state while preserving structure."""
        if self._predictive_coding_core:
            self._predictive_coding_core.reset_state()
        
        if self._self_organizing_map:
            self._self_organizing_map.reset_training()
        
        self._learning_epochs_completed = 0
        self._current_prediction_state = None
        self._environmental_coupling_strength = 0.0
        self._learning_trajectory.clear()
        self._version += 1
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive learning metrics.
        
        Returns:
            Dictionary with learning performance metrics
        """
        if not self._learning_trajectory:
            return {}
        
        recent_errors = [
            entry['prediction_error'] 
            for entry in self._learning_trajectory[-10:]
        ]
        
        return {
            'epochs_completed': self._learning_epochs_completed,
            'current_prediction_error': (
                self._current_prediction_state.total_error 
                if self._current_prediction_state else None
            ),
            'average_recent_error': sum(recent_errors) / len(recent_errors) if recent_errors else 0,
            'environmental_coupling_strength': self._environmental_coupling_strength,
            'is_converged': self.is_learning_converged,
            'learning_trajectory_length': len(self._learning_trajectory),
            'current_learning_rate': self._learning_params.current_learning_rate(
                self._learning_epochs_completed
            )
        }
    
    def clear_domain_events(self) -> List[Any]:
        """Clear and return accumulated domain events."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events
    
    def _calculate_environmental_coupling(
        self,
        input_data: npt.NDArray,
        environmental_context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate environmental coupling strength based on input and context.
        
        Implements enactivist measure of how coupled the system is with
        its environment based on sensorimotor patterns and context.
        
        Args:
            input_data: Current sensory input
            environmental_context: Environmental interaction context
            
        Returns:
            Coupling strength [0, 1]
        """
        coupling_factors = []
        
        # Factor 1: Input variability (higher variability = stronger coupling)
        if input_data.size > 1:
            input_variance = float(input_data.var())
            coupling_factors.append(min(input_variance, 1.0))
        
        # Factor 2: Environmental context richness
        if environmental_context:
            context_richness = len(environmental_context) / 10.0  # Normalize
            coupling_factors.append(min(context_richness, 1.0))
        
        # Factor 3: Historical coupling consistency
        if len(self._learning_trajectory) > 1:
            recent_couplings = [
                entry.get('coupling_strength', 0) 
                for entry in self._learning_trajectory[-5:]
            ]
            coupling_consistency = 1.0 - (max(recent_couplings) - min(recent_couplings))
            coupling_factors.append(max(coupling_consistency, 0.0))
        
        # Default coupling if no factors available
        if not coupling_factors:
            return 0.5
        
        return sum(coupling_factors) / len(coupling_factors)
    
    def _create_precision_weights_from_coupling(self, coupling_strength: float) -> PrecisionWeights:
        """
        Create precision weights based on environmental coupling strength.
        
        Args:
            coupling_strength: Environmental coupling strength
            
        Returns:
            Precision weights reflecting coupling
        """
        # Higher coupling = higher precision at lower levels (more environmental sensitivity)
        if self._predictive_coding_core:
            levels = self._predictive_coding_core.hierarchy_levels
            weights = []
            for level in range(levels):
                # Lower levels get higher precision with stronger coupling
                level_weight = coupling_strength * (1.0 - level / levels) + 0.1
                weights.append(level_weight)
            
            return PrecisionWeights(weights)
        
        # Default precision weights
        return PrecisionWeights([1.0])
    
    def _validate_learning_step(
        self,
        new_prediction_state: PredictionState,
        coupling_strength: float
    ) -> None:
        """
        Validate learning step maintains aggregate invariants.
        
        Args:
            new_prediction_state: New prediction state
            coupling_strength: Environmental coupling strength
            
        Raises:
            LearningInvariantViolation: If invariants are violated
        """
        # Invariant 1: Prediction quality must meet minimum standards
        if not self._quality_spec.is_satisfied_by(new_prediction_state):
            raise LearningInvariantViolation(
                "Prediction quality below minimum standards"
            )
        
        # Invariant 2: Environmental coupling must be within valid range
        if not (0.0 <= coupling_strength <= 1.0):
            raise LearningInvariantViolation(
                f"Environmental coupling strength out of range: {coupling_strength}"
            )
        
        # Invariant 3: Learning must show progress over time
        if len(self._learning_trajectory) > 10:
            recent_errors = [
                entry['prediction_error'] 
                for entry in self._learning_trajectory[-10:]
            ]
            if all(error >= recent_errors[0] for error in recent_errors[1:]):
                raise LearningInvariantViolation(
                    "Learning showing no progress over recent epochs"
                )
    
    def _generate_learning_events(
        self,
        previous_state: Optional[PredictionState],
        new_state: PredictionState,
        coupling_strength: float
    ) -> None:
        """Generate appropriate domain events for learning progress."""
        # Always generate epoch completion event
        self._domain_events.append(
            LearningEpochCompleted(
                aggregate_id=self._aggregate_id,
                epoch_number=self._learning_epochs_completed,
                prediction_error=new_state.total_error,
                coupling_strength=coupling_strength,
                timestamp=datetime.now()
            )
        )
        
        # Check for error threshold crossing
        if previous_state and previous_state.total_error > 1.0 and new_state.total_error <= 1.0:
            self._domain_events.append(
                PredictionErrorThresholdCrossed(
                    aggregate_id=self._aggregate_id,
                    threshold_value=1.0,
                    previous_error=previous_state.total_error,
                    new_error=new_state.total_error,
                    timestamp=datetime.now()
                )
            )
        
        # Check for convergence
        if self.is_learning_converged:
            self._domain_events.append(
                SelfOrganizationConverged(
                    aggregate_id=self._aggregate_id,
                    final_error=new_state.total_error,
                    epochs_to_convergence=self._learning_epochs_completed,
                    timestamp=datetime.now()
                )
            )
    
    def _analyze_environmental_change(
        self,
        change_signal: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze environmental change signal to determine adaptation requirements."""
        adaptation_requirements = {}
        
        change_type = change_signal.get('change_type', 'unknown')
        change_magnitude = change_signal.get('magnitude', 0.5)
        
        if change_type == 'sensory_shift':
            adaptation_requirements['learning_rate_adjustment'] = change_magnitude
            adaptation_requirements['precision_reweighting'] = change_magnitude * 0.8
        elif change_type == 'context_change':
            adaptation_requirements['neighborhood_adjustment'] = change_magnitude
            adaptation_requirements['coupling_strength_adjustment'] = change_magnitude * 1.2
        
        return adaptation_requirements
    
    def _validate_parameter_adaptation(self, adapted_params: LearningParameters) -> bool:
        """Validate that parameter adaptation maintains learning invariants."""
        # Check learning rate bounds
        if not (0.0001 <= adapted_params.initial_learning_rate <= 1.0):
            return False
        
        # Check decay parameters
        if not (0.9 <= adapted_params.learning_rate_decay <= 1.0):
            return False
        
        return True


class LearningInvariantViolation(Exception):
    """Raised when learning aggregate invariants are violated."""
    pass