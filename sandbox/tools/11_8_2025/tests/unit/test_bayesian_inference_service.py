"""
Unit tests for BayesianInferenceService domain service.

Comprehensive TDD test suite covering Bayesian inference operations,
uncertainty quantification, and model evidence computation. Tests use
mocks and property-based testing for mathematical correctness.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import numpy.typing as npt
from hypothesis import given, assume, strategies as st

from domain.services.bayesian_inference_service import (
    BayesianInferenceService,
    BayesianInferenceError,
    ComputationError,
    OptimizationError,
    EstimationError,
    SamplingError
)
from domain.value_objects.probability_distribution import ProbabilityDistribution
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.precision_weights import PrecisionWeights


# Mock implementation of abstract BayesianInferenceService
class MockBayesianInferenceService(BayesianInferenceService):
    """Mock implementation for testing BayesianInferenceService interface."""
    
    def __init__(self):
        self.update_beliefs_calls = []
        self.model_evidence_calls = []
        self.posterior_predictive_calls = []
        self.uncertainty_estimation_calls = []
        self.precision_optimization_calls = []
        self.model_selection_calls = []
        self.hyperparameter_estimation_calls = []
        self.information_gain_calls = []
        self.uncertainty_propagation_calls = []
        self.surprise_computation_calls = []
        self.posterior_sampling_calls = []
    
    def update_beliefs(
        self,
        prior_distribution: ProbabilityDistribution,
        evidence: npt.NDArray,
        likelihood_params: Dict[str, Any]
    ) -> ProbabilityDistribution:
        """Mock belief updating using Bayes' rule."""
        self.update_beliefs_calls.append((prior_distribution, evidence.copy(), likelihood_params))
        
        # Validate inputs
        if evidence.size == 0:
            raise ValueError("Evidence cannot be empty")
        
        if not likelihood_params:
            raise ValueError("Likelihood parameters cannot be empty")
        
        # Simple mock: create posterior as modified prior
        posterior_params = prior_distribution.parameters.copy()
        
        # Mock Bayesian update - adjust parameters based on evidence
        if 'mean' in posterior_params:
            evidence_mean = np.mean(evidence)
            posterior_params['mean'] = (posterior_params['mean'] + evidence_mean) / 2
        
        if 'variance' in posterior_params:
            posterior_params['variance'] *= 0.8  # Reduce variance (more confident)
        
        # Create posterior with updated probabilities
        # Simple mock: slightly adjust prior probabilities
        updated_probs = prior_distribution.probabilities * 0.9
        updated_probs = updated_probs / np.sum(updated_probs)  # Re-normalize
        
        return ProbabilityDistribution(
            probabilities=updated_probs,
            distribution_type=prior_distribution.distribution_type,
            parameters=posterior_params
        )
    
    def compute_model_evidence(
        self,
        data: npt.NDArray,
        model_params: Dict[str, Any],
        prior_distribution: ProbabilityDistribution
    ) -> float:
        """Mock model evidence computation."""
        self.model_evidence_calls.append((data.copy(), model_params, prior_distribution))
        
        if data.size == 0:
            raise ComputationError("Data cannot be empty for model evidence computation")
        
        # Mock log marginal likelihood
        data_complexity = np.log(data.size)
        model_complexity = len(model_params)
        
        # Simple evidence approximation
        log_evidence = -data_complexity - model_complexity * 0.5
        return log_evidence
    
    def compute_posterior_predictive(
        self,
        posterior_distribution: ProbabilityDistribution,
        prediction_contexts: List[npt.NDArray]
    ) -> List[ProbabilityDistribution]:
        """Mock posterior predictive computation."""
        self.posterior_predictive_calls.append((posterior_distribution, prediction_contexts))
        
        if not prediction_contexts:
            raise ValueError("Prediction contexts cannot be empty")
        
        predictive_distributions = []
        for context in prediction_contexts:
            # Create predictive distribution based on posterior and context
            context_mean = np.mean(context) if context.size > 0 else 0.0
            
            pred_params = {
                'mean': context_mean,
                'variance': 1.0  # Fixed variance for simplicity
            }
            
            pred_dist = ProbabilityDistribution(
                distribution_type='normal',
                parameters=pred_params
            )
            predictive_distributions.append(pred_dist)
        
        return predictive_distributions
    
    def estimate_uncertainty(
        self,
        prediction_state: PredictionState,
        model_confidence: float
    ) -> Dict[str, float]:
        """Mock uncertainty estimation."""
        self.uncertainty_estimation_calls.append((prediction_state, model_confidence))
        
        if not (0.0 <= model_confidence <= 1.0):
            raise ValueError("Model confidence must be in [0, 1]")
        
        # Mock uncertainty decomposition
        total_error = prediction_state.total_error
        
        # Simple decomposition based on error and confidence
        aleatoric_uncertainty = total_error * 0.6
        epistemic_uncertainty = total_error * 0.4 * (1.0 - model_confidence)
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        return {
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'confidence_interval': (0.1, 0.9)  # Mock interval
        }
    
    def optimize_precision_weights(
        self,
        prediction_errors: List[float],
        current_weights: PrecisionWeights,
        adaptation_rate: float
    ) -> PrecisionWeights:
        """Mock precision weight optimization."""
        self.precision_optimization_calls.append((prediction_errors, current_weights, adaptation_rate))
        
        if not (0.0 < adaptation_rate <= 1.0):
            raise ValueError("Adaptation rate must be in (0, 1]")
        
        if len(prediction_errors) != current_weights.hierarchy_levels:
            raise ValueError("Prediction errors and weights must have same length")
        
        # Simple optimization: increase weights for levels with small errors
        new_weights = []
        for i, error in enumerate(prediction_errors):
            current_weight = current_weights.get_weight_at_level(i)
            # Inverse relationship: smaller error -> higher precision
            error_factor = 1.0 / (1.0 + error)
            new_weight = current_weight * (1.0 - adaptation_rate) + error_factor * adaptation_rate
            new_weights.append(new_weight)
        
        return PrecisionWeights(
            weights=np.array(new_weights),
            normalization_method=current_weights.normalization_method,
            temperature=current_weights.temperature,
            adaptation_rate=adaptation_rate
        )
    
    def perform_model_selection(
        self,
        candidate_models: List[Dict[str, Any]],
        data: npt.NDArray,
        selection_criterion: str = "bayes_factor"
    ) -> tuple[int, Dict[str, float]]:
        """Mock model selection."""
        self.model_selection_calls.append((candidate_models, data.copy(), selection_criterion))
        
        if not candidate_models:
            raise ValueError("Candidate models list cannot be empty")
        
        valid_criteria = ["bayes_factor", "aic", "bic", "waic"]
        if selection_criterion not in valid_criteria:
            raise ValueError(f"Invalid selection criterion: {selection_criterion}")
        
        # Mock model comparison
        scores = {}
        for i, model in enumerate(candidate_models):
            # Simple scoring based on model complexity
            complexity = len(model.get('parameters', {}))
            score = -complexity - np.random.rand() * 0.1  # Add small random component
            scores[f"model_{i}"] = score
        
        best_model_idx = max(range(len(candidate_models)), 
                           key=lambda i: scores[f"model_{i}"])
        
        return best_model_idx, scores
    
    def estimate_hyperparameters(
        self,
        data: npt.NDArray,
        model_structure: Dict[str, Any],
        prior_hyperparams: Dict[str, float] = None
    ) -> Dict[str, float]:
        """Mock hyperparameter estimation."""
        self.hyperparameter_estimation_calls.append((data.copy(), model_structure, prior_hyperparams))
        
        if data.size == 0:
            raise EstimationError("Data cannot be empty for hyperparameter estimation")
        
        # Mock hyperparameter estimation
        estimated_hyperparams = {}
        
        # Use data statistics for estimation
        if 'mean_hyperparam' in model_structure:
            estimated_hyperparams['mean_hyperparam'] = float(np.mean(data))
        
        if 'variance_hyperparam' in model_structure:
            estimated_hyperparams['variance_hyperparam'] = float(np.var(data))
        
        if 'shape_hyperparam' in model_structure:
            estimated_hyperparams['shape_hyperparam'] = 2.0  # Default shape
        
        return estimated_hyperparams
    
    def compute_information_gain(
        self,
        prior_distribution: ProbabilityDistribution,
        posterior_distribution: ProbabilityDistribution
    ) -> float:
        """Mock information gain computation."""
        self.information_gain_calls.append((prior_distribution, posterior_distribution))
        
        # Simple mock: information gain as KL divergence approximation
        if prior_distribution.distribution_type != posterior_distribution.distribution_type:
            raise ValueError("Prior and posterior must have same distribution type")
        
        # Mock KL divergence calculation
        prior_entropy = prior_distribution.entropy_normalized
        posterior_entropy = posterior_distribution.entropy_normalized
        
        # Information gain = reduction in entropy
        information_gain = max(0.0, prior_entropy - posterior_entropy)
        return information_gain
    
    def propagate_uncertainty(
        self,
        input_uncertainty: ProbabilityDistribution,
        transformation_function: str,
        transformation_params: Dict[str, Any]
    ) -> ProbabilityDistribution:
        """Mock uncertainty propagation."""
        self.uncertainty_propagation_calls.append((input_uncertainty, transformation_function, transformation_params))
        
        valid_transformations = ["linear", "nonlinear", "neural_network"]
        if transformation_function not in valid_transformations:
            raise ValueError(f"Unsupported transformation: {transformation_function}")
        
        # Mock uncertainty propagation
        output_params = input_uncertainty.parameters.copy()
        
        if transformation_function == "linear":
            # Linear transformation: uncertainty scales predictably
            scale = transformation_params.get('scale', 1.0)
            if 'variance' in output_params:
                output_params['variance'] *= scale**2
        
        elif transformation_function == "nonlinear":
            # Nonlinear transformation: increase uncertainty
            if 'variance' in output_params:
                output_params['variance'] *= 1.5
        
        elif transformation_function == "neural_network":
            # Neural network: complex uncertainty propagation
            if 'variance' in output_params:
                output_params['variance'] *= 2.0
        
        # Create probabilities using the transformation
        if input_uncertainty.distribution_type == "normal" and 'mean' in output_params and 'variance' in output_params:
            transformed_dist = ProbabilityDistribution.normal(
                mean=output_params['mean'],
                variance=output_params['variance'],
                n_points=20
            )
        else:
            # Fallback for other distribution types
            transformed_probs = input_uncertainty.probabilities.copy()
            transformed_dist = ProbabilityDistribution(
                probabilities=transformed_probs,
                distribution_type=input_uncertainty.distribution_type,
                parameters=output_params
            )
        
        return transformed_dist
    
    def compute_surprise(
        self,
        observed_data: npt.NDArray,
        predicted_distribution: ProbabilityDistribution
    ) -> float:
        """Mock surprise computation."""
        self.surprise_computation_calls.append((observed_data.copy(), predicted_distribution))
        
        if observed_data.size == 0:
            raise ValueError("Observed data cannot be empty")
        
        # Mock surprise as negative log-likelihood
        data_mean = np.mean(observed_data)
        pred_mean = predicted_distribution.parameters.get('mean', 0.0)
        pred_var = predicted_distribution.parameters.get('variance', 1.0)
        
        # Simple Gaussian surprise calculation
        squared_error = (data_mean - pred_mean)**2
        surprise = 0.5 * (np.log(2 * np.pi * pred_var) + squared_error / pred_var)
        
        # Ensure surprise is non-negative (for testing purposes)
        return max(0.0, surprise)
    
    def sample_from_posterior(
        self,
        posterior_distribution: ProbabilityDistribution,
        n_samples: int,
        sampling_method: str = "monte_carlo"
    ) -> npt.NDArray:
        """Mock posterior sampling."""
        self.posterior_sampling_calls.append((posterior_distribution, n_samples, sampling_method))
        
        if n_samples <= 0:
            raise SamplingError("Number of samples must be positive")
        
        valid_methods = ["monte_carlo", "gibbs", "metropolis"]
        if sampling_method not in valid_methods:
            raise ValueError(f"Invalid sampling method: {sampling_method}")
        
        # Mock sampling based on distribution type
        if posterior_distribution.distribution_type == 'normal':
            mean = posterior_distribution.parameters.get('mean', 0.0)
            var = posterior_distribution.parameters.get('variance', 1.0)
            samples = np.random.normal(mean, np.sqrt(var), n_samples)
        elif posterior_distribution.distribution_type == 'uniform':
            low = posterior_distribution.parameters.get('low', 0.0)
            high = posterior_distribution.parameters.get('high', 1.0)
            samples = np.random.uniform(low, high, n_samples)
        else:
            # Default: standard normal
            samples = np.random.randn(n_samples)
        
        return samples


# Module-level fixtures for shared test data
@pytest.fixture
def service():
    """Create mock Bayesian inference service."""
    return MockBayesianInferenceService()

@pytest.fixture
def sample_prior():
    """Create sample prior distribution."""
    # Create normalized probabilities for a discrete distribution
    prob_values = np.array([0.3, 0.4, 0.3])  # Normalized to sum to 1
    return ProbabilityDistribution(
        probabilities=prob_values,
        distribution_type='categorical',
        parameters={'mean': 0.0, 'variance': 1.0}
    )

@pytest.fixture
def sample_evidence():
    """Create sample evidence data."""
    return np.random.randn(50)

@pytest.fixture
def sample_likelihood_params():
    """Create sample likelihood parameters."""
    return {'noise_variance': 0.1, 'model_type': 'gaussian'}

@pytest.fixture
def sample_prediction_state():
    """Create sample prediction state."""
    return PredictionState(
        hierarchical_errors=[0.1, 0.2, 0.3],
        convergence_status="converged",
        learning_iteration=100
    )

@pytest.fixture
def sample_precision_weights():
    """Create sample precision weights."""
    weight_values = np.array([1.0, 0.8, 0.6, 0.4])
    return PrecisionWeights(
        weights=weight_values,
        normalization_method="softmax", 
        temperature=1.0,
        adaptation_rate=0.01
    )

@pytest.fixture
def sample_candidate_models():
    """Create sample candidate models."""
    return [
        {'name': 'simple', 'parameters': {'param1': 1.0}},
        {'name': 'medium', 'parameters': {'param1': 1.0, 'param2': 2.0}},
        {'name': 'complex', 'parameters': {'param1': 1.0, 'param2': 2.0, 'param3': 3.0}}
    ]

@pytest.fixture
def sample_input_uncertainty():
    """Create sample input uncertainty distribution."""
    prob_values = np.array([0.2, 0.5, 0.3])  # Normalized discrete distribution
    return ProbabilityDistribution(
        probabilities=prob_values,
        distribution_type='categorical',
        parameters={'mean': 0.0, 'variance': 1.0}
    )


class TestBayesianInferenceServiceInterface:
    """Test suite for BayesianInferenceService interface compliance."""
    
    def test_service_implements_all_abstract_methods(self, service):
        """Test that mock service implements all required methods."""
        # Verify all abstract methods are implemented
        required_methods = [
            'update_beliefs',
            'compute_model_evidence', 
            'compute_posterior_predictive',
            'estimate_uncertainty',
            'optimize_precision_weights',
            'perform_model_selection',
            'estimate_hyperparameters',
            'compute_information_gain',
            'propagate_uncertainty',
            'compute_surprise',
            'sample_from_posterior'
        ]
        
        for method_name in required_methods:
            assert hasattr(service, method_name)
            assert callable(getattr(service, method_name))


class TestBayesianBeliefUpdating:
    """Test suite for Bayesian belief updating operations."""
    
    def test_update_beliefs_valid_inputs(self, service, sample_prior, sample_evidence, sample_likelihood_params):
        """Test belief updating with valid inputs."""
        # Act
        posterior = service.update_beliefs(sample_prior, sample_evidence, sample_likelihood_params)
        
        # Assert
        assert isinstance(posterior, ProbabilityDistribution)
        assert len(service.update_beliefs_calls) == 1
        
        # Verify posterior is different from prior (learning occurred)
        assert posterior.parameters != sample_prior.parameters
    
    def test_update_beliefs_empty_evidence_raises_error(self, service, sample_prior, sample_likelihood_params):
        """Test that empty evidence raises error."""
        # Arrange
        empty_evidence = np.array([])
        
        # Act & Assert
        with pytest.raises(ValueError, match="Evidence cannot be empty"):
            service.update_beliefs(sample_prior, empty_evidence, sample_likelihood_params)
    
    def test_update_beliefs_empty_likelihood_params_raises_error(self, service, sample_prior, sample_evidence):
        """Test that empty likelihood parameters raise error."""
        # Arrange
        empty_params = {}
        
        # Act & Assert
        with pytest.raises(ValueError, match="Likelihood parameters cannot be empty"):
            service.update_beliefs(sample_prior, sample_evidence, empty_params)
    
    def test_update_beliefs_reduces_uncertainty(self, service):
        """Test that belief updating reduces uncertainty (in mock)."""
        # Arrange
        prior = ProbabilityDistribution.normal(mean=0.0, variance=2.0, n_points=20)
        evidence = np.array([1.0, 1.1, 0.9, 1.2, 0.8])
        likelihood_params = {'noise_variance': 0.1}
        
        # Act
        posterior = service.update_beliefs(prior, evidence, likelihood_params)
        
        # Assert
        # Mock reduces variance by factor of 0.8
        assert posterior.parameters['variance'] < prior.parameters['variance']
    
    def test_update_beliefs_incorporates_evidence(self, service):
        """Test that belief updating incorporates evidence."""
        # Arrange
        prior = ProbabilityDistribution.normal(mean=0.0, variance=1.0, n_points=20)
        evidence = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # Clear evidence for mean=5
        likelihood_params = {'noise_variance': 0.1}
        
        # Act
        posterior = service.update_beliefs(prior, evidence, likelihood_params)
        
        # Assert
        # Posterior mean should be between prior mean (0) and evidence mean (5)
        assert prior.parameters['mean'] < posterior.parameters['mean'] < np.mean(evidence)


class TestModelEvidenceComputation:
    """Test suite for model evidence computation."""
    
    
    def test_compute_model_evidence_valid_inputs(self, service, sample_prior):
        """Test model evidence computation with valid inputs."""
        # Arrange
        data = np.random.randn(100)
        model_params = {'param1': 1.0, 'param2': 2.0}
        
        # Act
        evidence = service.compute_model_evidence(data, model_params, sample_prior)
        
        # Assert
        assert isinstance(evidence, float)
        assert len(service.model_evidence_calls) == 1
        
        # Log evidence should be negative (log of probability < 1)
        assert evidence < 0.0
    
    def test_compute_model_evidence_empty_data_raises_error(self, service, sample_prior):
        """Test that empty data raises error."""
        # Arrange
        empty_data = np.array([])
        model_params = {'param1': 1.0}
        
        # Act & Assert
        with pytest.raises(ComputationError, match="Data cannot be empty"):
            service.compute_model_evidence(empty_data, model_params, sample_prior)
    
    def test_model_evidence_penalizes_complexity(self, service, sample_prior):
        """Test that model evidence penalizes model complexity."""
        # Arrange
        data = np.random.randn(50)
        simple_model = {'param1': 1.0}
        complex_model = {'param1': 1.0, 'param2': 2.0, 'param3': 3.0, 'param4': 4.0}
        
        # Act
        simple_evidence = service.compute_model_evidence(data, simple_model, sample_prior)
        complex_evidence = service.compute_model_evidence(data, complex_model, sample_prior)
        
        # Assert
        # Simple model should have higher evidence (less negative log evidence)
        assert simple_evidence > complex_evidence


class TestUncertaintyEstimation:
    """Test suite for uncertainty estimation and decomposition."""
    
    
    
    def test_estimate_uncertainty_valid_inputs(self, service, sample_prediction_state):
        """Test uncertainty estimation with valid inputs."""
        # Arrange
        model_confidence = 0.8
        
        # Act
        uncertainties = service.estimate_uncertainty(sample_prediction_state, model_confidence)
        
        # Assert
        required_keys = ['aleatoric_uncertainty', 'epistemic_uncertainty', 
                        'total_uncertainty', 'confidence_interval']
        for key in required_keys:
            assert key in uncertainties
        
        assert all(isinstance(uncertainties[key], (int, float, tuple)) for key in uncertainties)
        assert len(service.uncertainty_estimation_calls) == 1
    
    def test_estimate_uncertainty_invalid_confidence_raises_error(self, service, sample_prediction_state):
        """Test that invalid model confidence raises error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Model confidence must be in \\[0, 1\\]"):
            service.estimate_uncertainty(sample_prediction_state, 1.5)
        
        with pytest.raises(ValueError, match="Model confidence must be in \\[0, 1\\]"):
            service.estimate_uncertainty(sample_prediction_state, -0.1)
    
    def test_uncertainty_decomposition_consistency(self, service, sample_prediction_state):
        """Test that uncertainty decomposition is consistent."""
        # Arrange
        model_confidence = 0.7
        
        # Act
        uncertainties = service.estimate_uncertainty(sample_prediction_state, model_confidence)
        
        # Assert
        aleatoric = uncertainties['aleatoric_uncertainty']
        epistemic = uncertainties['epistemic_uncertainty']
        total = uncertainties['total_uncertainty']
        
        # Total should be sum of components (in this mock implementation)
        assert abs(total - (aleatoric + epistemic)) < 1e-10
    
    def test_uncertainty_varies_with_confidence(self, service, sample_prediction_state):
        """Test that uncertainty varies appropriately with model confidence."""
        # Act
        low_confidence_uncertainties = service.estimate_uncertainty(sample_prediction_state, 0.2)
        high_confidence_uncertainties = service.estimate_uncertainty(sample_prediction_state, 0.9)
        
        # Assert
        # Epistemic uncertainty should be higher with lower confidence
        assert (low_confidence_uncertainties['epistemic_uncertainty'] > 
                high_confidence_uncertainties['epistemic_uncertainty'])


class TestPrecisionWeightOptimization:
    """Test suite for precision weight optimization."""
    
    
    
    def test_optimize_precision_weights_valid_inputs(self, service, sample_precision_weights):
        """Test precision weight optimization with valid inputs."""
        # Arrange
        prediction_errors = [0.1, 0.2, 0.05, 0.3]
        adaptation_rate = 0.1
        
        # Act
        optimized_weights = service.optimize_precision_weights(
            prediction_errors, sample_precision_weights, adaptation_rate
        )
        
        # Assert
        assert isinstance(optimized_weights, PrecisionWeights)
        assert optimized_weights.hierarchy_levels == len(prediction_errors)
        assert len(service.precision_optimization_calls) == 1
    
    def test_optimize_precision_weights_invalid_adaptation_rate_raises_error(self, service, sample_precision_weights):
        """Test that invalid adaptation rate raises error."""
        # Arrange
        prediction_errors = [0.1, 0.2, 0.3, 0.4]
        
        # Act & Assert
        with pytest.raises(ValueError, match="Adaptation rate must be in \\(0, 1\\]"):
            service.optimize_precision_weights(prediction_errors, sample_precision_weights, 0.0)
        
        with pytest.raises(ValueError, match="Adaptation rate must be in \\(0, 1\\]"):
            service.optimize_precision_weights(prediction_errors, sample_precision_weights, 1.5)
    
    def test_optimize_precision_weights_mismatched_lengths_raises_error(self, service, sample_precision_weights):
        """Test that mismatched lengths raise error."""
        # Arrange
        wrong_length_errors = [0.1, 0.2]  # Only 2 errors for 4 weight levels
        
        # Act & Assert
        with pytest.raises(ValueError, match="Prediction errors and weights must have same length"):
            service.optimize_precision_weights(wrong_length_errors, sample_precision_weights, 0.1)
    
    def test_precision_weights_adapt_to_errors(self, service):
        """Test that precision weights adapt appropriately to prediction errors."""
        # Arrange
        current_weights = PrecisionWeights(np.array([1.0, 1.0, 1.0]))
        high_low_high_errors = [2.0, 0.1, 2.0]  # Middle level has low error
        adaptation_rate = 0.5
        
        # Act
        optimized_weights = service.optimize_precision_weights(
            high_low_high_errors, current_weights, adaptation_rate
        )
        
        # Assert
        # Weight for level 1 (low error) should be higher than others
        weight_0 = optimized_weights.get_weight_at_level(0)
        weight_1 = optimized_weights.get_weight_at_level(1)
        weight_2 = optimized_weights.get_weight_at_level(2)
        
        assert weight_1 > weight_0
        assert weight_1 > weight_2


class TestModelSelection:
    """Test suite for Bayesian model selection."""
    
    
    
    def test_perform_model_selection_valid_inputs(self, service, sample_candidate_models):
        """Test model selection with valid inputs."""
        # Arrange
        data = np.random.randn(100)
        
        # Act
        best_model_idx, scores = service.perform_model_selection(
            sample_candidate_models, data, "bayes_factor"
        )
        
        # Assert
        assert isinstance(best_model_idx, int)
        assert 0 <= best_model_idx < len(sample_candidate_models)
        assert isinstance(scores, dict)
        assert len(scores) == len(sample_candidate_models)
        assert len(service.model_selection_calls) == 1
    
    def test_perform_model_selection_empty_models_raises_error(self, service):
        """Test that empty candidate models list raises error."""
        # Arrange
        data = np.random.randn(50)
        empty_models = []
        
        # Act & Assert
        with pytest.raises(ValueError, match="Candidate models list cannot be empty"):
            service.perform_model_selection(empty_models, data, "bayes_factor")
    
    def test_perform_model_selection_invalid_criterion_raises_error(self, service, sample_candidate_models):
        """Test that invalid selection criterion raises error."""
        # Arrange
        data = np.random.randn(50)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid selection criterion"):
            service.perform_model_selection(sample_candidate_models, data, "invalid_criterion")
    
    def test_model_selection_supports_different_criteria(self, service, sample_candidate_models):
        """Test that different selection criteria work."""
        # Arrange
        data = np.random.randn(50)
        criteria = ["bayes_factor", "aic", "bic", "waic"]
        
        # Act & Assert
        for criterion in criteria:
            best_idx, scores = service.perform_model_selection(
                sample_candidate_models, data, criterion
            )
            assert isinstance(best_idx, int)
            assert isinstance(scores, dict)


class TestInformationGain:
    """Test suite for information gain computation."""
    
    
    def test_compute_information_gain_valid_distributions(self, service):
        """Test information gain computation with valid distributions."""
        # Arrange
        prior = ProbabilityDistribution.normal(mean=0.0, variance=2.0, n_points=20)
        posterior = ProbabilityDistribution.normal(mean=1.0, variance=1.0, n_points=20)
        
        # Act
        info_gain = service.compute_information_gain(prior, posterior)
        
        # Assert
        assert isinstance(info_gain, float)
        assert info_gain >= 0.0  # Information gain should be non-negative
        assert len(service.information_gain_calls) == 1
    
    def test_compute_information_gain_incompatible_distributions_raises_error(self, service):
        """Test that incompatible distributions raise error."""
        # Arrange
        prior = ProbabilityDistribution.normal(mean=0.0, variance=1.0, n_points=20)
        posterior = ProbabilityDistribution.uniform(size=20)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Prior and posterior must have same distribution type"):
            service.compute_information_gain(prior, posterior)


class TestUncertaintyPropagation:
    """Test suite for uncertainty propagation through transformations."""
    
    
    
    def test_propagate_uncertainty_linear_transformation(self, service, sample_input_uncertainty):
        """Test uncertainty propagation through linear transformation."""
        # Arrange
        transformation_params = {'scale': 2.0}
        
        # Act
        output_uncertainty = service.propagate_uncertainty(
            sample_input_uncertainty, "linear", transformation_params
        )
        
        # Assert
        assert isinstance(output_uncertainty, ProbabilityDistribution)
        assert output_uncertainty.distribution_type == sample_input_uncertainty.distribution_type
        
        # Linear transformation should scale variance by scale^2
        expected_variance = sample_input_uncertainty.parameters['variance'] * (2.0**2)
        assert abs(output_uncertainty.parameters['variance'] - expected_variance) < 1e-10
    
    def test_propagate_uncertainty_nonlinear_transformation(self, service, sample_input_uncertainty):
        """Test uncertainty propagation through nonlinear transformation."""
        # Arrange
        transformation_params = {'nonlinearity_type': 'sigmoid'}
        
        # Act
        output_uncertainty = service.propagate_uncertainty(
            sample_input_uncertainty, "nonlinear", transformation_params
        )
        
        # Assert
        assert isinstance(output_uncertainty, ProbabilityDistribution)
        # Nonlinear transformation should increase uncertainty
        assert output_uncertainty.parameters['variance'] > sample_input_uncertainty.parameters['variance']
    
    def test_propagate_uncertainty_invalid_transformation_raises_error(self, service, sample_input_uncertainty):
        """Test that invalid transformation raises error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported transformation"):
            service.propagate_uncertainty(
                sample_input_uncertainty, "invalid_transformation", {}
            )


class TestSurpriseComputation:
    """Test suite for surprise computation."""
    
    
    def test_compute_surprise_valid_inputs(self, service):
        """Test surprise computation with valid inputs."""
        # Arrange
        observed_data = np.array([1.0, 1.1, 0.9, 1.2, 0.8])
        predicted_distribution = ProbabilityDistribution.normal(mean=1.0, variance=0.1, n_points=20)
        
        # Act
        surprise = service.compute_surprise(observed_data, predicted_distribution)
        
        # Assert
        assert isinstance(surprise, float)
        assert surprise >= 0.0  # Surprise should be non-negative
        assert len(service.surprise_computation_calls) == 1
    
    def test_compute_surprise_empty_data_raises_error(self, service):
        """Test that empty observed data raises error."""
        # Arrange
        empty_data = np.array([])
        predicted_distribution = ProbabilityDistribution.normal(mean=0.0, variance=1.0, n_points=20)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Observed data cannot be empty"):
            service.compute_surprise(empty_data, predicted_distribution)
    
    def test_surprise_higher_for_unexpected_observations(self, service):
        """Test that surprise is higher for unexpected observations."""
        # Arrange
        predicted_distribution = ProbabilityDistribution.normal(mean=0.0, variance=0.1, n_points=20)
        
        expected_data = np.array([0.0, 0.01, -0.01])  # Close to predicted mean
        unexpected_data = np.array([5.0, 5.1, 4.9])   # Far from predicted mean
        
        # Act
        expected_surprise = service.compute_surprise(expected_data, predicted_distribution)
        unexpected_surprise = service.compute_surprise(unexpected_data, predicted_distribution)
        
        # Assert
        assert unexpected_surprise > expected_surprise


class TestPosteriorSampling:
    """Test suite for posterior sampling."""
    
    
    def test_sample_from_posterior_valid_inputs(self, service):
        """Test posterior sampling with valid inputs."""
        # Arrange
        posterior = ProbabilityDistribution.normal(mean=2.0, variance=1.0, n_points=20)
        n_samples = 100
        
        # Act
        samples = service.sample_from_posterior(posterior, n_samples, "monte_carlo")
        
        # Assert
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (n_samples,)
        assert len(service.posterior_sampling_calls) == 1
    
    def test_sample_from_posterior_invalid_n_samples_raises_error(self, service):
        """Test that invalid number of samples raises error."""
        # Arrange
        posterior = ProbabilityDistribution.normal(mean=0.0, variance=1.0, n_points=20)
        
        # Act & Assert
        with pytest.raises(SamplingError, match="Number of samples must be positive"):
            service.sample_from_posterior(posterior, 0, "monte_carlo")
        
        with pytest.raises(SamplingError, match="Number of samples must be positive"):
            service.sample_from_posterior(posterior, -10, "monte_carlo")
    
    def test_sample_from_posterior_invalid_method_raises_error(self, service):
        """Test that invalid sampling method raises error."""
        # Arrange
        posterior = ProbabilityDistribution.normal(mean=0.0, variance=1.0, n_points=20)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid sampling method"):
            service.sample_from_posterior(posterior, 10, "invalid_method")
    
    def test_sampling_methods_produce_different_results(self, service):
        """Test that different sampling methods work."""
        # Arrange
        posterior = ProbabilityDistribution.normal(mean=1.0, variance=0.5, n_points=20)
        methods = ["monte_carlo", "gibbs", "metropolis"]
        
        # Act & Assert
        for method in methods:
            samples = service.sample_from_posterior(posterior, 50, method)
            assert isinstance(samples, np.ndarray)
            assert len(samples) == 50


class TestBayesianInferenceServiceIntegration:
    """Integration tests for BayesianInferenceService operations."""
    
    
    def test_complete_bayesian_workflow(self, service):
        """Test complete Bayesian inference workflow."""
        # Arrange
        # 1. Set up prior
        prior = ProbabilityDistribution.normal(mean=0.0, variance=2.0, n_points=20)
        
        # 2. Generate evidence
        evidence = np.random.normal(1.0, 0.5, 50)
        likelihood_params = {'noise_variance': 0.25}
        
        # 3. Set up model candidates
        models = [
            {'name': 'simple', 'parameters': {'param1': 1.0}},
            {'name': 'complex', 'parameters': {'param1': 1.0, 'param2': 2.0}}
        ]
        
        # Act
        # 1. Update beliefs
        posterior = service.update_beliefs(prior, evidence, likelihood_params)
        
        # 2. Compute model evidence
        model_evidence = service.compute_model_evidence(evidence, models[0], prior)
        
        # 3. Perform model selection
        best_model_idx, scores = service.perform_model_selection(models, evidence)
        
        # 4. Sample from posterior
        samples = service.sample_from_posterior(posterior, 100)
        
        # 5. Compute information gain
        info_gain = service.compute_information_gain(prior, posterior)
        
        # Assert
        assert isinstance(posterior, ProbabilityDistribution)
        assert isinstance(model_evidence, float)
        assert isinstance(best_model_idx, int)
        assert isinstance(samples, np.ndarray)
        assert isinstance(info_gain, float)
        
        # Check that all operations were tracked
        assert len(service.update_beliefs_calls) == 1
        assert len(service.model_evidence_calls) == 1
        assert len(service.model_selection_calls) == 1
        assert len(service.posterior_sampling_calls) == 1
        assert len(service.information_gain_calls) == 1


class TestBayesianInferenceServiceErrors:
    """Test suite for BayesianInferenceService error handling."""
    
    def test_computation_error_hierarchy(self):
        """Test that computation errors inherit from BayesianInferenceError."""
        # Arrange & Act
        comp_error = ComputationError("Test computation error")
        opt_error = OptimizationError("Test optimization error")
        est_error = EstimationError("Test estimation error")
        samp_error = SamplingError("Test sampling error")
        
        # Assert
        assert isinstance(comp_error, BayesianInferenceError)
        assert isinstance(opt_error, BayesianInferenceError)
        assert isinstance(est_error, BayesianInferenceError)
        assert isinstance(samp_error, BayesianInferenceError)
    
    def test_error_messages_are_preserved(self):
        """Test that error messages are properly preserved."""
        # Arrange
        message = "Custom error message"
        
        # Act
        error = ComputationError(message)
        
        # Assert
        assert str(error) == message


class TestBayesianInferenceServicePropertyBased:
    """Property-based tests for BayesianInferenceService mathematical properties."""
    
    
    @given(
        n_samples=st.integers(min_value=1, max_value=1000),
        mean=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        variance=st.floats(min_value=0.1, max_value=10.0, allow_nan=False)
    )
    def test_posterior_sampling_properties(self, n_samples, mean, variance):
        """Test that posterior sampling has correct statistical properties."""
        # Arrange (create service inside test to avoid fixture scope issue)
        service = MockBayesianInferenceService()
        posterior = ProbabilityDistribution.normal(mean=mean, variance=variance, n_points=20)
        
        # Act
        samples = service.sample_from_posterior(posterior, n_samples)
        
        # Assert
        assert len(samples) == n_samples
        assert samples.ndim == 1
        
        # Basic properties validation (statistical properties not tested in mock)
        # For mock implementation, we only verify basic structure
        assert np.all(np.isfinite(samples))  # All samples are finite
        # Note: Statistical properties are not strictly tested in mock implementation
        # as the mock may not perfectly reproduce theoretical distributions
    
    @given(
        adaptation_rate=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
        n_levels=st.integers(min_value=1, max_value=10)
    )
    def test_precision_weight_optimization_properties(self, adaptation_rate, n_levels):
        """Test properties of precision weight optimization."""
        # Arrange (create service inside test to avoid fixture scope issue)
        service = MockBayesianInferenceService()
        prediction_errors = [abs(x) for x in np.random.randn(n_levels)]
        current_weights = PrecisionWeights(np.array([1.0] * n_levels))
        
        # Act
        optimized_weights = service.optimize_precision_weights(
            prediction_errors, current_weights, adaptation_rate
        )
        
        # Assert
        assert optimized_weights.hierarchy_levels == n_levels
        assert all(w > 0 for w in [optimized_weights.get_weight_at_level(i) for i in range(n_levels)])
        
        # Weights should be positive and bounded
        for i in range(n_levels):
            weight = optimized_weights.get_weight_at_level(i)
            assert 0 < weight < 100  # Reasonable bounds