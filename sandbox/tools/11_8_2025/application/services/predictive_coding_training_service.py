"""
Predictive Coding Training Service.

Application service for training the hierarchical predictive coding system.
Follows the Application Service pattern from DDD, orchestrating domain
entities and value objects to accomplish training use cases.

Implements the Free Energy Principle training protocol:
1. Generate predictions from current model
2. Compute prediction errors against targets
3. Propagate errors through hierarchy
4. Update parameters to minimize free energy
5. Adapt precision weights based on error patterns
"""

from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import numpy.typing as npt

from domain.entities.predictive_coding_core import PredictiveCodingCore
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.precision_weights import PrecisionWeights
from domain.value_objects.learning_parameters import LearningParameters
from domain.events.domain_events import (
    LearningEpochCompleted, 
    PredictionErrorThresholdCrossed,
    AdaptiveLearningRateChanged,
    DomainEvent
)
from infrastructure.jax_predictive_coding_core import JaxPredictiveCodingCore


@dataclass
class TrainingConfiguration:
    """Configuration for predictive coding training."""
    max_epochs: int = 1000
    convergence_threshold: float = 0.001
    early_stopping_patience: int = 50
    validation_frequency: int = 10
    checkpoint_frequency: int = 100
    learning_rate_schedule: str = "adaptive"  # "fixed", "adaptive", "exponential"
    precision_adaptation_rate: float = 0.01
    enable_monitoring: bool = True
    monitoring_metrics: List[str] = None
    
    def __post_init__(self):
        if self.monitoring_metrics is None:
            self.monitoring_metrics = [
                "total_error", "free_energy", "precision_entropy", 
                "convergence_rate", "stability_measure"
            ]


@dataclass  
class TrainingMetrics:
    """Training metrics and monitoring data."""
    epoch: int
    total_error: float
    free_energy_estimate: float
    precision_entropy: float
    convergence_rate: float
    stability_measure: float
    learning_rate: float
    validation_error: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PredictiveCodingTrainingService:
    """
    Application service for training predictive coding systems.
    
    Orchestrates the training process by coordinating domain entities,
    managing training state, and publishing domain events. Implements
    advanced training strategies based on the Free Energy Principle.
    """
    
    def __init__(
        self,
        predictive_coding_core: PredictiveCodingCore,
        learning_parameters: LearningParameters,
        config: Optional[TrainingConfiguration] = None
    ):
        """
        Initialize training service.
        
        Args:
            predictive_coding_core: The predictive coding entity to train
            learning_parameters: Learning configuration parameters
            config: Training configuration (defaults to reasonable values)
        """
        self._core = predictive_coding_core
        self._learning_params = learning_parameters
        self._config = config or TrainingConfiguration()
        
        # Training state
        self._current_epoch = 0
        self._training_history: List[TrainingMetrics] = []
        self._best_validation_error = float('inf')
        self._patience_counter = 0
        self._is_training = False
        
        # Event storage for domain events
        self._domain_events: List[DomainEvent] = []
        
        # Adaptive learning rate state
        self._current_learning_rate = learning_parameters.base_learning_rate
        self._learning_rate_history: List[float] = []
        
    def train_online(
        self,
        data_stream: Callable[[], Tuple[npt.NDArray, PrecisionWeights]],
        max_steps: int = 1000,
        validation_data: Optional[Callable[[], Tuple[npt.NDArray, PrecisionWeights]]] = None
    ) -> List[TrainingMetrics]:
        """
        Train the predictive coding system online with streaming data.
        
        Implements online learning where the system continuously adapts
        to new sensory input, following enactivist principles of
        continuous environmental coupling.
        
        Args:
            data_stream: Generator function yielding (input_data, precision_weights)
            max_steps: Maximum number of training steps
            validation_data: Optional validation data generator
            
        Returns:
            List of training metrics for each step
        """
        self._is_training = True
        online_metrics = []
        
        try:
            for step in range(max_steps):
                # Get next data sample
                input_data, precision_weights = data_stream()
                
                # Perform one training step
                prediction_state = self._core.process_input(
                    input_data, precision_weights, self._current_learning_rate
                )
                
                # Compute metrics
                metrics = self._compute_training_metrics(
                    step, prediction_state, validation_data
                )
                online_metrics.append(metrics)
                
                # Check for convergence or early stopping
                if self._should_stop_training(metrics):
                    break
                
                # Adapt learning rate and precision weights
                if step % self._config.validation_frequency == 0:
                    precision_weights = self._adapt_precision_weights(
                        prediction_state, precision_weights
                    )
                    self._adapt_learning_rate(metrics)
                
                # Publish domain events
                self._publish_training_events(step, prediction_state, metrics)
                
            return online_metrics
            
        finally:
            self._is_training = False
    
    def train_batch(
        self,
        training_data: List[Tuple[npt.NDArray, PrecisionWeights]],
        validation_data: Optional[List[Tuple[npt.NDArray, PrecisionWeights]]] = None
    ) -> List[TrainingMetrics]:
        """
        Train the predictive coding system on batch data.
        
        Implements batch training with epochs, validation, and early stopping.
        Useful for offline training on collected datasets.
        
        Args:
            training_data: List of (input_data, precision_weights) pairs
            validation_data: Optional validation dataset
            
        Returns:
            List of training metrics for each epoch
        """
        self._is_training = True
        batch_metrics = []
        
        try:
            for epoch in range(self._config.max_epochs):
                self._current_epoch = epoch
                
                # Training phase
                epoch_predictions = []
                total_epoch_error = 0.0
                
                for input_data, precision_weights in training_data:
                    prediction_state = self._core.process_input(
                        input_data, precision_weights, self._current_learning_rate
                    )
                    epoch_predictions.append(prediction_state)
                    total_epoch_error += prediction_state.total_error
                
                # Compute epoch metrics
                avg_prediction_state = self._aggregate_prediction_states(epoch_predictions)
                metrics = self._compute_training_metrics(
                    epoch, avg_prediction_state, validation_data
                )
                batch_metrics.append(metrics)
                self._training_history.append(metrics)
                
                # Check convergence and early stopping
                if self._should_stop_training(metrics):
                    print(f"Training converged at epoch {epoch}")
                    break
                
                # Adaptive adjustments
                if epoch % self._config.validation_frequency == 0:
                    self._adapt_learning_rate(metrics)
                    
                # Checkpointing
                if epoch % self._config.checkpoint_frequency == 0:
                    self._save_checkpoint(epoch, metrics)
                
                # Publish domain events
                self._publish_training_events(epoch, avg_prediction_state, metrics)
            
            return batch_metrics
            
        finally:
            self._is_training = False
    
    def _compute_training_metrics(
        self,
        step_or_epoch: int,
        prediction_state: PredictionState,
        validation_data: Optional[Callable] = None
    ) -> TrainingMetrics:
        """Compute comprehensive training metrics."""
        
        # Basic error metrics
        total_error = prediction_state.total_error
        
        # Free energy estimate (if available)
        free_energy_estimate = 0.0
        if hasattr(self._core, 'get_free_energy_estimate'):
            free_energy_estimate = self._core.get_free_energy_estimate()
        
        # Precision entropy
        precision_entropy = self._compute_precision_entropy()
        
        # Convergence rate (based on error trajectory)
        convergence_rate = self._compute_convergence_rate()
        
        # Stability measure (error variance over recent steps)  
        stability_measure = self._compute_stability_measure()
        
        # Validation error if validation data provided
        validation_error = None
        if validation_data is not None:
            validation_error = self._compute_validation_error(validation_data)
        
        return TrainingMetrics(
            epoch=step_or_epoch,
            total_error=total_error,
            free_energy_estimate=free_energy_estimate,
            precision_entropy=precision_entropy,
            convergence_rate=convergence_rate,
            stability_measure=stability_measure,
            learning_rate=self._current_learning_rate,
            validation_error=validation_error
        )
    
    def _compute_precision_entropy(self) -> float:
        """Compute entropy of precision weights distribution."""
        if hasattr(self._core, 'get_precision_estimates'):
            precision_estimates = self._core.get_precision_estimates()
            if precision_estimates:
                precisions = np.array(list(precision_estimates.values()))
                # Normalize to get probability distribution
                normalized = precisions / np.sum(precisions)
                # Compute entropy
                entropy = -np.sum(normalized * np.log(normalized + 1e-10))
                return float(entropy)
        return 0.0
    
    def _compute_convergence_rate(self) -> float:
        """Compute convergence rate based on error trajectory."""
        if len(self._training_history) < 2:
            return 0.0
        
        recent_errors = [m.total_error for m in self._training_history[-10:]]
        if len(recent_errors) < 2:
            return 0.0
            
        # Linear regression on log errors to estimate convergence rate
        x = np.arange(len(recent_errors))
        log_errors = np.log(np.array(recent_errors) + 1e-10)
        
        # Simple slope calculation
        if len(x) > 1:
            slope = (log_errors[-1] - log_errors[0]) / (x[-1] - x[0])
            return float(-slope)  # Negative slope indicates convergence
        
        return 0.0
    
    def _compute_stability_measure(self) -> float:
        """Compute stability measure based on error variance."""
        if len(self._training_history) < 5:
            return 0.0
        
        recent_errors = [m.total_error for m in self._training_history[-10:]]
        error_variance = float(np.var(recent_errors))
        
        # Stability is inverse of variance (more stable = lower variance)
        stability = 1.0 / (1.0 + error_variance)
        return stability
    
    def _compute_validation_error(
        self, 
        validation_data: Callable
    ) -> float:
        """Compute validation error."""
        try:
            val_input, val_precision = validation_data()
            val_predictions = self._core.generate_predictions(val_input, val_precision)
            val_targets = self._core._create_targets_from_input(val_input, val_predictions)
            val_errors = self._core.compute_prediction_errors(val_predictions, val_targets)
            
            # Compute total validation error
            total_val_error = sum(np.mean(np.abs(err)) for err in val_errors)
            return float(total_val_error)
            
        except Exception:
            return float('nan')
    
    def _should_stop_training(self, metrics: TrainingMetrics) -> bool:
        """Determine if training should stop based on convergence criteria."""
        
        # Convergence check
        if metrics.total_error < self._config.convergence_threshold:
            return True
        
        # Early stopping based on validation error
        if metrics.validation_error is not None:
            if metrics.validation_error < self._best_validation_error:
                self._best_validation_error = metrics.validation_error
                self._patience_counter = 0
            else:
                self._patience_counter += 1
                
            if self._patience_counter >= self._config.early_stopping_patience:
                return True
        
        # Divergence check
        if metrics.total_error > 100.0:  # Arbitrary large threshold
            print("Training diverged - stopping")
            return True
        
        return False
    
    def _adapt_precision_weights(
        self,
        prediction_state: PredictionState,
        current_precision_weights: PrecisionWeights
    ) -> PrecisionWeights:
        """Adapt precision weights based on prediction performance."""
        
        # Use prediction errors to adapt precision
        errors = prediction_state.hierarchical_errors
        
        # Adapt weights using the domain object method
        adapted_weights = current_precision_weights.adapt_weights(
            errors, self._config.precision_adaptation_rate
        )
        
        return adapted_weights
    
    def _adapt_learning_rate(self, metrics: TrainingMetrics) -> None:
        """Adapt learning rate based on training progress."""
        
        if self._config.learning_rate_schedule == "fixed":
            return
        
        previous_lr = self._current_learning_rate
        
        if self._config.learning_rate_schedule == "adaptive":
            # Increase LR if converging too slowly, decrease if unstable
            if metrics.convergence_rate < 0.01 and metrics.stability_measure > 0.8:
                self._current_learning_rate *= 1.1  # Increase
            elif metrics.stability_measure < 0.5:
                self._current_learning_rate *= 0.9  # Decrease
            
            # Clamp learning rate
            self._current_learning_rate = np.clip(
                self._current_learning_rate, 
                self._learning_params.min_learning_rate,
                self._learning_params.max_learning_rate
            )
        
        elif self._config.learning_rate_schedule == "exponential":
            # Exponential decay
            decay_rate = 0.95
            self._current_learning_rate = (
                self._learning_params.base_learning_rate * 
                (decay_rate ** self._current_epoch)
            )
        
        # Record learning rate change
        self._learning_rate_history.append(self._current_learning_rate)
        
        # Publish event if learning rate changed significantly
        if abs(self._current_learning_rate - previous_lr) > 0.001:
            event = AdaptiveLearningRateChanged(
                aggregate_id=f"training_service_{id(self)}",
                new_learning_rate=self._current_learning_rate,
                previous_learning_rate=previous_lr,
                adaptation_reason=f"schedule_{self._config.learning_rate_schedule}"
            )
            self._domain_events.append(event)
    
    def _aggregate_prediction_states(
        self, 
        states: List[PredictionState]
    ) -> PredictionState:
        """Aggregate multiple prediction states into summary state."""
        
        if not states:
            return PredictionState.create_empty(self._core.hierarchy_levels)
        
        # Average errors across states
        all_errors = [state.hierarchical_errors for state in states]
        avg_errors = [
            sum(errors[i] for errors in all_errors) / len(all_errors)
            for i in range(len(all_errors[0]))
        ]
        
        # Use most recent state as template and update errors
        latest_state = states[-1]
        return latest_state.with_updated_errors(avg_errors)
    
    def _publish_training_events(
        self,
        step_or_epoch: int,
        prediction_state: PredictionState,
        metrics: TrainingMetrics
    ) -> None:
        """Publish domain events for training progress."""
        
        # Learning epoch completed event
        epoch_event = LearningEpochCompleted(
            aggregate_id=f"training_service_{id(self)}",
            epoch_number=step_or_epoch,
            prediction_error=metrics.total_error,
            coupling_strength=metrics.stability_measure,
            learning_rate=self._current_learning_rate
        )
        self._domain_events.append(epoch_event)
        
        # Check for threshold crossings
        self._check_error_thresholds(metrics)
    
    def _check_error_thresholds(self, metrics: TrainingMetrics) -> None:
        """Check if prediction error has crossed significant thresholds."""
        
        significant_thresholds = [0.1, 0.01, 0.001]
        
        if len(self._training_history) > 0:
            previous_error = self._training_history[-1].total_error
            current_error = metrics.total_error
            
            for threshold in significant_thresholds:
                # Check if we crossed below threshold
                if previous_error > threshold >= current_error:
                    event = PredictionErrorThresholdCrossed(
                        aggregate_id=f"training_service_{id(self)}",
                        threshold_value=threshold,
                        previous_error=previous_error,
                        new_error=current_error
                    )
                    self._domain_events.append(event)
    
    def _save_checkpoint(self, epoch: int, metrics: TrainingMetrics) -> None:
        """Save training checkpoint (placeholder for actual implementation)."""
        # In a real implementation, this would save model parameters,
        # training state, and metrics to persistent storage
        checkpoint_data = {
            'epoch': epoch,
            'metrics': metrics.to_dict() if hasattr(metrics, 'to_dict') else vars(metrics),
            'learning_rate': self._current_learning_rate,
            'model_state': 'placeholder_for_model_parameters'
        }
        
        # Log checkpoint creation
        print(f"Checkpoint saved at epoch {epoch}")
    
    def get_training_history(self) -> List[TrainingMetrics]:
        """Get complete training history."""
        return self._training_history.copy()
    
    def get_domain_events(self) -> List[DomainEvent]:
        """Get all domain events generated during training."""
        return self._domain_events.copy()
    
    def clear_domain_events(self) -> None:
        """Clear accumulated domain events."""
        self._domain_events.clear()
    
    def is_training(self) -> bool:
        """Check if training is currently in progress."""
        return self._is_training
    
    def get_current_learning_rate(self) -> float:
        """Get current adaptive learning rate."""
        return self._current_learning_rate
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training process."""
        if not self._training_history:
            return {"status": "not_trained"}
        
        latest_metrics = self._training_history[-1]
        
        return {
            "status": "completed" if not self._is_training else "in_progress",
            "total_epochs": len(self._training_history),
            "final_error": latest_metrics.total_error,
            "final_free_energy": latest_metrics.free_energy_estimate,
            "final_learning_rate": latest_metrics.learning_rate,
            "convergence_achieved": latest_metrics.total_error < self._config.convergence_threshold,
            "stability_measure": latest_metrics.stability_measure,
            "best_validation_error": self._best_validation_error if self._best_validation_error != float('inf') else None,
            "training_duration": (
                self._training_history[-1].timestamp - self._training_history[0].timestamp
            ).total_seconds() if len(self._training_history) > 1 else 0
        }