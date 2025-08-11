"""
JAX-based Predictive Coding Core Implementation.

Concrete implementation of hierarchical predictive coding using JAX for
high-performance computation. Implements the Free Energy Principle through
variational Bayesian inference with precision-weighted prediction error
minimization.

Mathematical Foundation:
- Variational Free Energy: F = ∫ q(x) ln[q(x)/p(x,s)] dx
- Prediction Error: ε = x - μ  
- Precision-weighted Error: ε̃ = Π ε (where Π is precision matrix)
- Learning Rule: Δθ ∝ -∇_θ F
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import optax
from typing import List, Tuple, Optional, NamedTuple, Callable, Dict
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from domain.entities.predictive_coding_core import PredictiveCodingCore
from domain.entities.self_organizing_map import SelfOrganizingMap
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.precision_weights import PrecisionWeights
from domain.value_objects.som_topology import SOMTopology
from domain.value_objects.learning_parameters import LearningParameters


class HierarchicalState(NamedTuple):
    """Internal state of hierarchical predictive coding system."""
    predictions: List[jnp.ndarray]  # μ_i for each level i
    prediction_errors: List[jnp.ndarray]  # ε_i for each level i
    precision_matrices: List[jnp.ndarray]  # Π_i for each level i
    learning_rates: List[float]  # η_i for each level i
    free_energy: float  # Variational free energy F


@dataclass
class FreeEnergyTerms:
    """Components of variational free energy calculation."""
    accuracy_term: float  # -ln p(s|μ) - accuracy of predictions
    complexity_term: float  # KL[q(μ)|p(μ)] - complexity cost
    precision_term: float  # Terms involving precision adaptation
    total_free_energy: float  # Sum of all terms


class JaxPredictiveCodingCore(PredictiveCodingCore):
    """
    JAX-based hierarchical predictive coding implementation.
    
    Implements the Free Energy Principle through hierarchical message passing
    where higher levels predict lower levels, and prediction errors drive
    learning through gradient descent on variational free energy.
    
    Key Features:
    - Hierarchical predictive processing
    - Precision-weighted error minimization  
    - Active inference through attention modulation
    - Temporal dynamics with prediction over time
    - Efficient JAX transformations (jit, vmap, grad)
    """
    
    def __init__(
        self, 
        hierarchy_levels: int, 
        input_dimensions: int,
        hidden_dimensions: Optional[List[int]] = None,
        learning_rate: float = 0.01,
        precision_init: float = 1.0,
        temporal_window: int = 10,
        enable_active_inference: bool = True,
        som_integration: Optional[SelfOrganizingMap] = None
    ):
        """
        Initialize JAX-based predictive coding core.
        
        Args:
            hierarchy_levels: Number of hierarchical levels
            input_dimensions: Dimensionality of input data
            hidden_dimensions: Dimensions for each level (defaults to decreasing)
            learning_rate: Base learning rate for parameter updates
            precision_init: Initial precision value
            temporal_window: Number of time steps for temporal predictions
            enable_active_inference: Enable attention-based active inference
            som_integration: Optional SOM for spatial organization
        """
        super().__init__(hierarchy_levels, input_dimensions)
        
        # Architecture configuration
        self._hidden_dimensions = hidden_dimensions or self._default_hidden_dims()
        self._learning_rate = learning_rate
        self._precision_init = precision_init
        self._temporal_window = temporal_window
        self._enable_active_inference = enable_active_inference
        
        # SOM integration (Dependency Injection)
        self._som_integration = som_integration
        self._som_enhanced_predictions = enable_active_inference and som_integration is not None
        
        # Initialize parameters and optimizers
        self._initialize_parameters()
        self._initialize_optimizers()
        
        # Internal state
        self._current_hierarchical_state: Optional[HierarchicalState] = None
        self._temporal_buffer: List[jnp.ndarray] = []
        self._iteration_count = 0
    
    def _default_hidden_dims(self) -> List[int]:
        """Create default hidden dimensions with decreasing size."""
        dims = []
        current_dim = self.input_dimensions
        
        for level in range(self.hierarchy_levels):
            # Decrease dimension by factor based on level
            dim = max(1, int(current_dim * (0.7 ** level)))
            dims.append(dim)
            current_dim = dim
        
        return dims
    
    def _initialize_parameters(self) -> None:
        """Initialize learnable parameters for each hierarchical level."""
        key = jax.random.PRNGKey(42)  # Should be configurable
        
        self._params = {}
        self._precision_params = {}
        
        # Initialize prediction weights for each level
        for level in range(self.hierarchy_levels):
            level_key, key = jax.random.split(key)
            
            input_dim = (self.input_dimensions if level == 0 
                        else self._hidden_dimensions[level - 1])
            output_dim = self._hidden_dimensions[level]
            
            # Prediction weights (for generating μ_i from μ_{i+1})
            self._params[f'W_pred_{level}'] = jax.random.normal(
                level_key, (input_dim, output_dim)
            ) * 0.1
            
            # Bias terms
            self._params[f'b_pred_{level}'] = jnp.zeros(output_dim)
            
            # Error propagation weights (for ε_i → ε_{i+1})  
            if level < self.hierarchy_levels - 1:
                self._params[f'W_error_{level}'] = jax.random.normal(
                    level_key, (output_dim, self._hidden_dimensions[level + 1])
                ) * 0.1
            
            # Precision parameters (learnable precision matrices)
            self._precision_params[f'log_precision_{level}'] = (
                jnp.log(self._precision_init) * jnp.ones(output_dim)
            )
    
    def _initialize_optimizers(self) -> None:
        """Initialize Adam optimizers for parameters and precision."""
        # Prediction parameter optimizer
        self._pred_optimizer = optax.adam(learning_rate=self._learning_rate)
        self._pred_opt_state = self._pred_optimizer.init(self._params)
        
        # Precision optimizer (typically slower adaptation)
        self._prec_optimizer = optax.adam(learning_rate=self._learning_rate * 0.1)
        self._prec_opt_state = self._prec_optimizer.init(self._precision_params)
    
    def _forward_pass(
        self, 
        input_data: jnp.ndarray, 
        params: dict
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Forward pass through hierarchical levels.
        
        Args:
            input_data: Input sensory data
            params: Network parameters
            
        Returns:
            Tuple of (predictions, representations) for each level
        """
        predictions = []
        representations = []
        
        current_input = input_data
        
        for level in range(self.hierarchy_levels):
            # Generate prediction at this level
            W_pred = params[f'W_pred_{level}']
            b_pred = params[f'b_pred_{level}']
            
            # Ensure proper dimensions for matrix multiplication
            if current_input.ndim == 1:
                current_input = current_input[None, :]  # Add batch dimension
            
            # Check dimension compatibility
            input_dim = current_input.shape[-1]
            weight_input_dim = W_pred.shape[0]
            
            if input_dim != weight_input_dim:
                # Project input to match weight dimensions
                if input_dim > weight_input_dim:
                    # Reduce dimensions via averaging
                    projection = current_input.reshape(current_input.shape[0], -1, weight_input_dim)
                    current_input = jnp.mean(projection, axis=1)
                else:
                    # Expand dimensions via padding
                    padding_size = weight_input_dim - input_dim
                    padding = jnp.zeros((current_input.shape[0], padding_size))
                    current_input = jnp.concatenate([current_input, padding], axis=-1)
            
            # Linear prediction followed by nonlinearity
            prediction = jnp.tanh(current_input @ W_pred + b_pred)
            predictions.append(prediction)
            
            # Create representation for next level
            # Higher levels receive aggregated information
            if prediction.ndim > 1:
                representation = jnp.mean(prediction, axis=0, keepdims=True)
            else:
                representation = prediction[None, :]
            
            representations.append(representation)
            current_input = representation
        
        return predictions, representations
    
    def _compute_free_energy(
        self,
        predictions: List[jnp.ndarray],
        targets: List[jnp.ndarray], 
        precision_params: dict
    ) -> FreeEnergyTerms:
        """
        Compute variational free energy following Friston's formulation.
        
        F = D_KL[q(θ|s) || p(θ)] - E_q[ln p(s|θ)]
        
        Decomposed as:
        - Accuracy: -½ Σᵢ εᵢᵀ Πᵢ εᵢ (negative log-likelihood)  
        - Complexity: ½ Σᵢ (μᵢ - μᵢ₀)ᵀ Πᵢ₀ (μᵢ - μᵢ₀) (KL divergence from prior)
        - Precision: ½ Σᵢ (ln|Πᵢ| - tr(Πᵢ Σᵢ)) (precision terms)
        
        Args:
            predictions: Hierarchical predictions μ_i
            targets: Target values for each level  
            precision_params: Precision parameters Π_i
            
        Returns:
            FreeEnergyTerms containing all energy components
        """
        accuracy_term = 0.0
        complexity_term = 0.0
        precision_term = 0.0
        
        for level, (pred, target) in enumerate(zip(predictions, targets)):
            # Get precision for this level
            log_precision = precision_params[f'log_precision_{level}']
            precision = jnp.exp(log_precision)
            
            # Prediction error: ε_i = s_i - μ_i
            error = target - pred
            
            # Accuracy term: -½ εᵢᵀ Πᵢ εᵢ 
            # This represents the negative log-likelihood of sensory data
            precision_weighted_error = error * precision
            accuracy_term += 0.5 * jnp.sum(error * precision_weighted_error)
            
            # Complexity term: KL divergence from prior beliefs
            # Regularizes predictions toward hierarchical consistency
            if level < len(predictions) - 1:
                # Use higher level prediction as prior mean
                prior_mean = jnp.mean(predictions[level + 1]) * jnp.ones_like(pred)
            else:
                # Top level uses zero prior
                prior_mean = jnp.zeros_like(pred)
            
            prediction_deviation = pred - prior_mean
            complexity_term += 0.5 * jnp.sum(prediction_deviation * precision * prediction_deviation)
            
            # Precision term: ln|Π| - trace terms for precision estimation
            # Encourages optimal precision (not too high, not too low)
            log_det_precision = jnp.sum(log_precision)  # For diagonal precision
            trace_term = jnp.sum(precision * jnp.var(error))  # Empirical variance
            
            precision_term += 0.5 * (log_det_precision - trace_term)
        
        total_free_energy = accuracy_term + complexity_term - precision_term
        
        return FreeEnergyTerms(
            accuracy_term=accuracy_term,
            complexity_term=complexity_term,
            precision_term=precision_term,
            total_free_energy=total_free_energy
        )
    
    def generate_predictions(
        self, 
        input_data: npt.NDArray,
        precision_weights: PrecisionWeights
    ) -> List[npt.NDArray]:
        """
        Generate hierarchical predictions from input data.
        
        Implements top-down predictive processing where higher levels
        generate predictions for lower levels based on current beliefs.
        
        Args:
            input_data: Input sensory data
            precision_weights: Precision weights for attention modulation
            
        Returns:
            List of predictions for each hierarchical level
            
        Raises:
            ValueError: If input shape doesn't match expected dimensions
        """
        if input_data.shape[-1] != self.input_dimensions:
            raise ValueError(
                f"Input dimensions {input_data.shape[-1]} != {self.input_dimensions}"
            )
        
        # Convert to JAX array
        jax_input = jnp.array(input_data)
        
        # Ensure we have a batch dimension
        if jax_input.ndim == 1:
            jax_input = jax_input[None, :]
        
        # Forward pass through hierarchy
        predictions, representations = self._forward_pass(jax_input, self._params)
        
        # Apply attention modulation if enabled
        if self._enable_active_inference:
            predictions = self._apply_attention_modulation(predictions, precision_weights)
        
        # Apply SOM-based spatial organization if integrated
        if self._som_enhanced_predictions:
            predictions = self._apply_som_spatial_organization(predictions, jax_input)
        
        # Convert back to numpy for domain layer compatibility
        np_predictions = [np.array(pred) for pred in predictions]
        
        return np_predictions
    
    def _apply_attention_modulation(
        self, 
        predictions: List[jnp.ndarray],
        precision_weights: PrecisionWeights
    ) -> List[jnp.ndarray]:
        """
        Apply attention-based modulation to predictions.
        
        Implements active inference by modulating prediction strength
        based on precision weights (attention).
        """
        modulated_predictions = []
        
        for level, prediction in enumerate(predictions):
            if level < precision_weights.hierarchy_levels:
                # Get normalized attention weight for this level
                attention_weight = precision_weights.get_normalized_weight_at_level(level)
                
                # Modulate prediction strength  
                modulated = prediction * (1.0 + attention_weight)
                modulated_predictions.append(modulated)
            else:
                modulated_predictions.append(prediction)
        
        return modulated_predictions
    
    def compute_prediction_errors(
        self,
        predictions: List[npt.NDArray],
        targets: List[npt.NDArray]
    ) -> List[npt.NDArray]:
        """
        Compute prediction errors across hierarchical levels.
        
        Args:
            predictions: Hierarchical predictions μ_i
            targets: Target values for each level
            
        Returns:
            List of prediction errors ε_i = targets_i - predictions_i
            
        Raises:
            ValueError: If predictions and targets don't match in structure
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"Prediction count {len(predictions)} != target count {len(targets)}"
            )
        
        errors = []
        for level, (pred, target) in enumerate(zip(predictions, targets)):
            pred_jax = jnp.array(pred)
            target_jax = jnp.array(target)
            
            if pred_jax.shape != target_jax.shape:
                raise ValueError(
                    f"Level {level}: prediction shape {pred_jax.shape} != "
                    f"target shape {target_jax.shape}"
                )
            
            # Compute prediction error
            error = target_jax - pred_jax
            errors.append(np.array(error))
        
        return errors
    
    def propagate_errors(
        self,
        prediction_errors: List[npt.NDArray],
        precision_weights: PrecisionWeights
    ) -> Tuple[List[npt.NDArray], PredictionState]:
        """
        Propagate errors through the hierarchy with precision weighting.
        
        Implements hierarchical message passing where prediction errors
        are propagated upward, modulated by precision weights.
        
        Args:
            prediction_errors: Errors from each hierarchical level
            precision_weights: Precision weights for error scaling
            
        Returns:
            Tuple of (propagated_errors, new_prediction_state)
            
        Raises:
            ValueError: If error propagation fails
        """
        try:
            # Convert to JAX arrays
            jax_errors = [jnp.array(err) for err in prediction_errors]
            
            # Apply precision weighting
            weighted_errors = self._apply_precision_weighting(jax_errors, precision_weights)
            
            # Propagate errors upward through hierarchy
            propagated_errors = self._hierarchical_error_propagation(weighted_errors)
            
            # Compute new prediction state
            new_state = self._create_prediction_state(
                propagated_errors, precision_weights
            )
            
            # Update internal state
            self._current_hierarchical_state = HierarchicalState(
                predictions=[],  # Will be filled during next forward pass
                prediction_errors=propagated_errors,
                precision_matrices=[],  # Will be updated
                learning_rates=[self._learning_rate] * self.hierarchy_levels,
                free_energy=new_state.total_error  # Approximation
            )
            
            # Convert back to numpy
            np_propagated_errors = [np.array(err) for err in propagated_errors]
            
            return np_propagated_errors, new_state
            
        except Exception as e:
            raise ValueError(f"Error propagation failed: {str(e)}")
    
    def _apply_precision_weighting(
        self, 
        errors: List[jnp.ndarray],
        precision_weights: PrecisionWeights
    ) -> List[jnp.ndarray]:
        """Apply precision-based weighting to prediction errors."""
        weighted_errors = []
        
        for level, error in enumerate(errors):
            if level < precision_weights.hierarchy_levels:
                weight = precision_weights.get_normalized_weight_at_level(level)
                weighted_error = error * weight
            else:
                weighted_error = error
            
            weighted_errors.append(weighted_error)
        
        return weighted_errors
    
    def _hierarchical_error_propagation(
        self, 
        weighted_errors: List[jnp.ndarray]
    ) -> List[jnp.ndarray]:
        """
        Propagate errors upward through hierarchical levels.
        
        Higher levels receive aggregated error information from lower levels
        to update their beliefs about the causes of sensory input.
        """
        propagated_errors = []
        
        for level in range(len(weighted_errors)):
            current_error = weighted_errors[level]
            
            # Add error from lower level if available
            if level > 0:
                # Project lower level error to current level
                if f'W_error_{level-1}' in self._params:
                    lower_error_projected = weighted_errors[level-1] @ self._params[f'W_error_{level-1}']
                    current_error = current_error + lower_error_projected * 0.1  # Scaling factor
            
            propagated_errors.append(current_error)
        
        return propagated_errors
    
    def _create_prediction_state(
        self,
        propagated_errors: List[jnp.ndarray],
        precision_weights: PrecisionWeights
    ) -> PredictionState:
        """Create PredictionState from propagated errors."""
        # Compute error magnitudes
        error_magnitudes = [float(jnp.mean(jnp.abs(error))) for error in propagated_errors]
        
        # Compute precision-weighted errors
        precision_weighted_errors = []
        for level, error_mag in enumerate(error_magnitudes):
            if level < precision_weights.hierarchy_levels:
                weight = precision_weights.get_normalized_weight_at_level(level)
                weighted = error_mag * weight
            else:
                weighted = error_mag
            precision_weighted_errors.append(weighted)
        
        # Determine convergence status
        total_error = sum(error_magnitudes)
        if total_error < 0.001:
            convergence_status = "converged"
        elif total_error > 10.0:
            convergence_status = "diverged" 
        elif hasattr(self, '_previous_total_error') and total_error < self._previous_total_error * 0.95:
            convergence_status = "converging"
        else:
            convergence_status = "not_converged"
        
        self._previous_total_error = total_error
        self._iteration_count += 1
        
        return PredictionState(
            hierarchical_errors=error_magnitudes,
            hierarchical_predictions=[],  # Will be filled by caller if needed
            precision_weighted_errors=precision_weighted_errors,
            convergence_status=convergence_status,
            learning_iteration=self._iteration_count,
            metadata={
                'free_energy_estimate': total_error,
                'temporal_window': self._temporal_window,
                'active_inference_enabled': self._enable_active_inference
            }
        )
    
    def update_predictions(
        self,
        learning_rate: float,
        propagated_errors: List[npt.NDArray]
    ) -> None:
        """
        Update internal prediction models based on propagated errors.
        
        Implements gradient descent on variational free energy to update
        both prediction parameters and precision estimates.
        
        Args:
            learning_rate: Learning rate for parameter updates
            propagated_errors: Errors propagated through hierarchy
            
        Raises:
            ValueError: If learning_rate is not in valid range (0, 1]
        """
        if not (0 < learning_rate <= 1.0):
            raise ValueError(f"Learning rate {learning_rate} not in (0, 1]")
        
        # Convert errors to JAX arrays
        jax_errors = [jnp.array(err) for err in propagated_errors]
        
        # Compute gradients and update parameters
        self._update_prediction_parameters(jax_errors, learning_rate)
        
        # Update precision parameters
        self._update_precision_parameters(jax_errors, learning_rate)
        
        # Update temporal buffer for temporal dynamics
        self._update_temporal_buffer(jax_errors)
    
    def _update_prediction_parameters(
        self, 
        propagated_errors: List[jnp.ndarray],
        learning_rate: float
    ) -> None:
        """Update prediction parameters using gradient descent."""
        
        def loss_fn(params):
            # Create dummy predictions for loss computation
            dummy_input = jnp.ones((1, self.input_dimensions))
            predictions, _ = self._forward_pass(dummy_input, params)
            
            # Use propagated errors as loss signal
            total_loss = 0.0
            for level, (pred, error) in enumerate(zip(predictions, propagated_errors)):
                if error.shape[0] > 0:  # Ensure error has content
                    # Mean squared error weighted by hierarchy level
                    level_weight = 1.0 / (level + 1)  # Higher levels get lower weight
                    loss = level_weight * jnp.mean(error ** 2)
                    total_loss += loss
            
            return total_loss
        
        # Compute gradients
        grads = grad(loss_fn)(self._params)
        
        # Apply optimizer update
        updates, self._pred_opt_state = self._pred_optimizer.update(
            grads, self._pred_opt_state
        )
        self._params = optax.apply_updates(self._params, updates)
    
    def _update_precision_parameters(
        self,
        propagated_errors: List[jnp.ndarray], 
        learning_rate: float
    ) -> None:
        """Update precision parameters based on error magnitudes."""
        
        def precision_loss_fn(precision_params):
            total_loss = 0.0
            
            for level, error in enumerate(propagated_errors):
                if f'log_precision_{level}' in precision_params:
                    log_precision = precision_params[f'log_precision_{level}']
                    precision = jnp.exp(log_precision)
                    
                    # Precision should increase when errors are consistent (low variance)
                    # and decrease when errors are inconsistent (high variance)
                    error_variance = jnp.var(error) + 1e-6
                    target_log_precision = -0.5 * jnp.log(error_variance)
                    
                    # Loss encourages precision to match error statistics
                    loss = jnp.mean((log_precision - target_log_precision) ** 2)
                    total_loss += loss
            
            return total_loss
        
        # Compute precision gradients
        precision_grads = grad(precision_loss_fn)(self._precision_params)
        
        # Apply precision optimizer update  
        precision_updates, self._prec_opt_state = self._prec_optimizer.update(
            precision_grads, self._prec_opt_state
        )
        self._precision_params = optax.apply_updates(
            self._precision_params, precision_updates
        )
    
    def _update_temporal_buffer(self, errors: List[jnp.ndarray]) -> None:
        """Update temporal buffer for temporal dynamics."""
        # Concatenate all errors into single vector for temporal tracking
        flattened_errors = [err.flatten() for err in errors]  # Use .flatten() method instead
        concatenated_error = jnp.concatenate(flattened_errors)
        
        # Add to buffer
        self._temporal_buffer.append(concatenated_error)
        
        # Maintain buffer size
        if len(self._temporal_buffer) > self._temporal_window:
            self._temporal_buffer.pop(0)
    
    def _create_targets_from_input(
        self,
        input_data: npt.NDArray,
        predictions: List[npt.NDArray]
    ) -> List[npt.NDArray]:
        """
        Create hierarchical targets from input data.
        
        In predictive coding, targets are created by considering:
        1. Bottom-up sensory information
        2. Top-down predictions from higher levels
        3. Temporal consistency constraints
        """
        targets = []
        
        for level, prediction in enumerate(predictions):
            if level == 0:
                # Level 0: targets come from sensory input
                # Ensure input matches prediction shape exactly
                pred_shape = prediction.shape
                input_shape = input_data.shape
                
                if len(pred_shape) != len(input_shape):
                    # Match number of dimensions
                    if len(pred_shape) > len(input_shape):
                        # Expand input dimensions
                        target = input_data.reshape((1,) + input_data.shape)
                        # Pad to match exact shape
                        for i in range(len(pred_shape)):
                            if target.shape[i] != pred_shape[i]:
                                if i < len(target.shape):
                                    if target.shape[i] < pred_shape[i]:
                                        padding = np.zeros(pred_shape)
                                        padding[:target.shape[0], :min(target.shape[1], pred_shape[1])] = target[:, :min(target.shape[1], pred_shape[1])]
                                        target = padding
                                        break
                    else:
                        # Reduce input dimensions  
                        target = input_data.reshape(pred_shape)
                else:
                    # Same number of dimensions, just reshape
                    target = input_data.reshape(pred_shape)
            else:
                # Higher levels: targets come from lower level predictions
                prev_prediction = predictions[level-1]
                
                # Ensure shape compatibility
                if prev_prediction.shape != prediction.shape:
                    # Aggregate or project to match target shape
                    if np.prod(prev_prediction.shape) >= np.prod(prediction.shape):
                        # Downsample
                        target = np.mean(prev_prediction.reshape(-1, np.prod(prediction.shape)), axis=0).reshape(prediction.shape)
                    else:
                        # Upsample via repetition
                        repeat_factor = np.prod(prediction.shape) // np.prod(prev_prediction.shape)
                        target = np.repeat(prev_prediction.flatten(), repeat_factor)[:np.prod(prediction.shape)].reshape(prediction.shape)
                else:
                    target = prev_prediction.copy()
                
                # Add small noise for learning dynamics
                target = target + np.random.randn(*prediction.shape) * 0.01
            
            targets.append(target)
        
        return targets
    
    def get_free_energy_estimate(self) -> float:
        """
        Get current estimate of variational free energy.
        
        Returns approximate free energy based on current prediction errors
        and precision estimates.
        """
        if self._current_hierarchical_state is None:
            return float('inf')
        
        return self._current_hierarchical_state.free_energy
    
    def get_precision_estimates(self) -> Dict[str, float]:
        """Get current precision estimates for each hierarchical level."""
        precision_estimates = {}
        
        for level in range(self.hierarchy_levels):
            if f'log_precision_{level}' in self._precision_params:
                log_precision = self._precision_params[f'log_precision_{level}']
                precision = float(jnp.exp(jnp.mean(log_precision)))
                precision_estimates[f'level_{level}'] = precision
        
        return precision_estimates
    
    def enable_temporal_dynamics(self, enable: bool = True) -> None:
        """Enable or disable temporal dynamics processing."""
        if not enable:
            self._temporal_buffer.clear()
    
    def get_hierarchical_representations(self) -> Optional[List[npt.NDArray]]:
        """
        Get current hierarchical representations.
        
        Returns learned representations at each hierarchical level
        that can be used for analysis or visualization.
        """
        if self._current_hierarchical_state is None:
            return None
        
        # Extract representations from current parameters
        representations = []
        for level in range(self.hierarchy_levels):
            W_pred = self._params[f'W_pred_{level}']
            # Use weight matrix as representation
            representations.append(np.array(W_pred))
        
        return representations
    
    def compute_free_energy(
        self,
        predictions: List[npt.NDArray],
        targets: List[npt.NDArray],
        precision_weights: PrecisionWeights
    ) -> float:
        """
        Compute variational free energy for domain layer interface.
        
        Args:
            predictions: Hierarchical predictions μ_i
            targets: Target values for each level
            precision_weights: Precision weights
            
        Returns:
            Scalar free energy value
        """
        # Convert to JAX arrays
        jax_predictions = [jnp.array(pred) for pred in predictions]
        jax_targets = [jnp.array(target) for target in targets]
        
        # Compute free energy using internal method
        free_energy_terms = self._compute_free_energy(
            jax_predictions, jax_targets, self._precision_params
        )
        
        return float(free_energy_terms.total_free_energy)
    
    def update_precisions(
        self,
        prediction_errors: List[npt.NDArray],
        learning_rate: float = 0.01
    ) -> PrecisionWeights:
        """
        Update precision weights based on error statistics.
        
        Implements: Π̇_i = γ(⟨ε_i²⟩ - Π_i^{-1})
        
        Args:
            prediction_errors: Current prediction errors ε_i
            learning_rate: Precision adaptation rate γ
            
        Returns:
            Updated precision weights
        """
        # Convert prediction errors to variance estimates
        error_variances = []
        
        for level, error in enumerate(prediction_errors):
            if len(error.flatten()) > 1:
                error_var = float(np.var(error))
            else:
                error_var = float(np.abs(error).item()) ** 2
            
            # Add small epsilon to avoid numerical issues
            error_variances.append(max(error_var, 1e-6))
        
        # Update precision parameters using the learning rule
        for level, error_var in enumerate(error_variances):
            if f'log_precision_{level}' in self._precision_params:
                current_log_precision = self._precision_params[f'log_precision_{level}']
                current_precision = jnp.exp(current_log_precision)
                
                # Target precision is inverse of error variance
                target_precision = 1.0 / error_var
                
                # Exponential moving average update
                new_precision = (
                    (1.0 - learning_rate) * current_precision + 
                    learning_rate * target_precision
                )
                
                # Update log precision parameter with clipping for stability
                clipped_precision = jnp.clip(new_precision, 1e-6, 1e6)  # Prevent extreme values
                self._precision_params[f'log_precision_{level}'] = jnp.log(clipped_precision)
        
        # Create new precision weights for return
        precision_values = []
        for level in range(self.hierarchy_levels):
            if f'log_precision_{level}' in self._precision_params:
                log_precision = self._precision_params[f'log_precision_{level}']
                # Clip log values to prevent overflow in exp
                clipped_log = jnp.clip(log_precision, -10, 10)  
                precision = float(jnp.exp(jnp.mean(clipped_log)))
                # Final check for NaN/inf
                if np.isnan(precision) or np.isinf(precision):
                    precision = 1.0
            else:
                precision = 1.0
            precision_values.append(precision)
        
        return PrecisionWeights(
            weights=np.array(precision_values),
            normalization_method="softmax",
            temperature=1.0,
            adaptation_rate=learning_rate,
            metadata={"updated_by": "jax_predictive_coding_core"}
        )
    
    def _apply_som_spatial_organization(
        self,
        predictions: List[jnp.ndarray],
        input_data: jnp.ndarray
    ) -> List[jnp.ndarray]:
        """
        Apply SOM-based spatial organization to predictions.
        
        Integrates SOM topological organization with predictive coding
        to enhance spatial representation learning.
        
        Args:
            predictions: Current hierarchical predictions
            input_data: Original input data
            
        Returns:
            Spatially organized predictions
        """
        if self._som_integration is None:
            return predictions
        
        organized_predictions = []
        
        for level, prediction in enumerate(predictions):
            if level == 0:  # Apply SOM organization to first level only
                # Convert JAX array to numpy for SOM interface
                np_input = np.array(input_data)
                if np_input.ndim > 1:
                    np_input = np_input.reshape(-1)  # Flatten for SOM
                
                # Ensure SOM is initialized
                if not self._som_integration.is_trained:
                    # Initialize with current data dimensions
                    if np_input.shape[0] == self._som_integration.input_dimensions:
                        self._som_integration.initialize_weights("random")
                
                if self._som_integration.is_trained:
                    try:
                        # Find best matching unit
                        bmu_pos = self._som_integration.find_best_matching_unit(np_input)
                        
                        # Get SOM weight at BMU position
                        som_weights = self._som_integration.weight_matrix
                        if som_weights is not None:
                            bmu_weight = som_weights[bmu_pos[1], bmu_pos[0]]  # y, x indexing
                            
                            # Modulate prediction with SOM topological structure
                            # Scale factor based on SOM activation
                            activation_strength = float(np.exp(-np.linalg.norm(np_input - bmu_weight)))
                            
                            # Apply spatial modulation
                            spatial_modulation = 1.0 + 0.1 * activation_strength
                            organized_prediction = prediction * spatial_modulation
                            organized_predictions.append(organized_prediction)
                        else:
                            organized_predictions.append(prediction)
                    except (ValueError, RuntimeError):
                        # Fallback if SOM operations fail
                        organized_predictions.append(prediction)
                else:
                    organized_predictions.append(prediction)
            else:
                # Higher levels remain unchanged
                organized_predictions.append(prediction)
        
        return organized_predictions
    
    def get_som_integration_status(self) -> dict:
        """
        Get status of SOM integration.
        
        Returns:
            Dictionary with SOM integration information
        """
        if self._som_integration is None:
            return {"enabled": False, "status": "not_integrated"}
        
        return {
            "enabled": True,
            "som_trained": self._som_integration.is_trained,
            "som_dimensions": self._som_integration.map_dimensions,
            "som_input_dimensions": self._som_integration.input_dimensions,
            "topology": self._som_integration.topology.to_dict(),
            "enhancement_active": self._som_enhanced_predictions
        }
    
    def update_som_from_predictions(
        self,
        predictions: List[npt.NDArray],
        learning_params: Optional[LearningParameters] = None
    ) -> None:
        """
        Update SOM using current predictions (optional feedback loop).
        
        Args:
            predictions: Current hierarchical predictions  
            learning_params: SOM learning parameters
        """
        if (self._som_integration is None or 
            not self._som_integration.is_trained or 
            len(predictions) == 0):
            return
        
        try:
            # Use first level predictions for SOM training
            first_level_pred = predictions[0]
            if first_level_pred.size == self._som_integration.input_dimensions:
                input_vec = first_level_pred.flatten()
                
                # Create default learning parameters if not provided
                if learning_params is None:
                    learning_params = LearningParameters(
                        initial_learning_rate=0.01,
                        final_learning_rate=0.001,
                        initial_neighborhood_radius=2.0,
                        final_neighborhood_radius=0.5,
                        total_iterations=1000
                    )
                
                # Single iteration SOM update
                self._som_integration.train_single_iteration(input_vec, learning_params)
                
        except (ValueError, RuntimeError) as e:
            # Log error but don't break prediction flow
            pass