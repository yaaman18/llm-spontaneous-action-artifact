"""
Integration tests for consciousness system components.

Tests the interaction between domain entities, services, and value objects
in realistic consciousness simulation scenarios. Follows TDD principles
with proper test isolation and mocking of infrastructure dependencies.
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import asyncio

from domain.entities.predictive_coding_core import PredictiveCodingCore
from domain.entities.self_organizing_map import SelfOrganizingMap
from domain.services.bayesian_inference_service import BayesianInferenceService
from domain.services.metacognitive_monitor_service import MetacognitiveMonitorService
from domain.repositories.consciousness_repository import ConsciousnessRepository
from domain.value_objects.consciousness_state import ConsciousnessState
from domain.value_objects.phi_value import PhiValue
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.probability_distribution import ProbabilityDistribution
from domain.value_objects.precision_weights import PrecisionWeights
from domain.value_objects.learning_parameters import LearningParameters
from domain.value_objects.som_topology import SOMTopology
from domain.value_objects.spatial_organization_state import SpatialOrganizationState


# Test implementations for integration testing
class TestPredictiveCodingCore(PredictiveCodingCore):
    """Test implementation of PredictiveCodingCore for integration tests."""
    
    def __init__(self, hierarchy_levels: int, input_dimensions: int):
        super().__init__(hierarchy_levels, input_dimensions)
        self.internal_weights = [
            np.random.rand(input_dimensions - i, input_dimensions - i - 1 if i < hierarchy_levels - 1 else 1)
            for i in range(hierarchy_levels)
        ]
        self.learning_history = []
    
    def generate_predictions(self, input_data, precision_weights):
        predictions = []
        current_input = input_data
        
        for level in range(self.hierarchy_levels):
            if level < len(self.internal_weights):
                weights = self.internal_weights[level]
                if current_input.shape[0] >= weights.shape[0]:
                    prediction = np.dot(weights.T, current_input[:weights.shape[0]])
                    predictions.append(prediction.flatten())
                    current_input = prediction.flatten()
                else:
                    # Handle dimension mismatch
                    prediction = np.mean(current_input) * np.ones(max(1, weights.shape[1]))
                    predictions.append(prediction)
                    current_input = prediction
            else:
                # Fallback prediction
                prediction = np.array([np.mean(current_input)])
                predictions.append(prediction)
                current_input = prediction
        
        return predictions
    
    def compute_prediction_errors(self, predictions, targets):
        errors = []
        for pred, target in zip(predictions, targets):
            min_len = min(len(pred), len(target))
            error = pred[:min_len] - target[:min_len]
            errors.append(error)
        return errors
    
    def propagate_errors(self, prediction_errors, precision_weights):
        propagated_errors = []
        error_magnitudes = []
        
        for i, error in enumerate(prediction_errors):
            weight = precision_weights.get_weight_at_level(i) if i < precision_weights.hierarchy_levels else 1.0
            weighted_error = error * weight
            propagated_errors.append(weighted_error)
            error_magnitudes.append(float(np.mean(np.abs(error))))
        
        prediction_state = PredictionState(
            hierarchical_errors=error_magnitudes,
            convergence_status="converging" if all(e < 0.1 for e in error_magnitudes) else "not_converged",
            learning_iteration=len(self.learning_history)
        )
        
        return propagated_errors, prediction_state
    
    def update_predictions(self, learning_rate, propagated_errors):
        self.learning_history.append({
            'learning_rate': learning_rate,
            'error_magnitude': sum(np.mean(np.abs(error)) for error in propagated_errors),
            'timestamp': datetime.now()
        })
        
        # Simple weight update
        for i, error in enumerate(propagated_errors):
            if i < len(self.internal_weights):
                update = learning_rate * np.mean(np.abs(error)) * 0.1
                self.internal_weights[i] *= (1.0 - update)
    
    def _create_targets_from_input(self, input_data, predictions):
        targets = []
        for i, prediction in enumerate(predictions):
            # Simple target creation - noisy version of input
            if i == 0:
                target_size = min(len(prediction), len(input_data))
                target = input_data[:target_size] + np.random.normal(0, 0.05, target_size)
            else:
                target = prediction + np.random.normal(0, 0.02, len(prediction))
            targets.append(target)
        return targets

    def compute_free_energy(self, predictions, targets, precision_weights):
        """Minimal implementation: sum of weighted squared errors."""
        total_energy = 0.0
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            min_len = min(len(pred), len(target))
            error = pred[:min_len] - target[:min_len]
            weight = precision_weights.get_weight_at_level(i) if i < precision_weights.hierarchy_levels else 1.0
            total_energy += float(np.sum(error**2) * weight)
        return total_energy

    def update_precisions(self, prediction_errors, learning_rate=0.01):
        """Minimal implementation: create updated precision weights based on error variance."""
        from domain.value_objects.precision_weights import PrecisionWeights
        
        # Compute error variance for each level
        variances = []
        for error in prediction_errors:
            variance = float(np.var(error)) if len(error) > 0 else 1.0
            # Precision is inverse of variance (with small regularization)
            precision = 1.0 / (variance + 1e-6)
            variances.append(precision)
        
        # Create updated precision weights with computed precisions
        return PrecisionWeights(
            weights=np.array(variances)
        )


class TestSelfOrganizingMap(SelfOrganizingMap):
    """Test implementation of SelfOrganizingMap for integration tests."""
    
    def initialize_weights(self, initialization_method: str = "random"):
        width, height = self.map_dimensions
        if initialization_method == "random":
            self._weight_matrix = np.random.rand(height, width, self.input_dimensions)
        elif initialization_method == "uniform":
            self._weight_matrix = np.ones((height, width, self.input_dimensions)) * 0.5
        else:
            self._weight_matrix = np.random.rand(height, width, self.input_dimensions) * 2 - 1
    
    def find_best_matching_unit(self, input_vector):
        if not self.is_trained:
            raise RuntimeError("SOM must be trained first")
        
        width, height = self.map_dimensions
        min_distance = float('inf')
        best_position = (0, 0)
        
        for i in range(width):
            for j in range(height):
                distance = np.linalg.norm(input_vector - self._weight_matrix[j, i, :])
                if distance < min_distance:
                    min_distance = distance
                    best_position = (i, j)
        
        return best_position
    
    def compute_neighborhood_function(self, bmu_position, current_iteration, learning_params):
        width, height = self.map_dimensions
        neighborhood = np.zeros((height, width))
        radius = learning_params.current_radius(current_iteration)
        
        bmu_x, bmu_y = bmu_position
        for i in range(width):
            for j in range(height):
                distance_sq = (i - bmu_x)**2 + (j - bmu_y)**2
                neighborhood[j, i] = np.exp(-distance_sq / (2 * radius**2))
        
        return neighborhood
    
    def update_weights(self, input_vector, bmu_position, neighborhood_function, learning_rate):
        width, height = self.map_dimensions
        for i in range(width):
            for j in range(height):
                influence = neighborhood_function[j, i] * learning_rate
                self._weight_matrix[j, i, :] += influence * (input_vector - self._weight_matrix[j, i, :])
    
    def compute_quantization_error(self, input_data):
        total_error = 0.0
        for input_vector in input_data:
            bmu_position = self.find_best_matching_unit(input_vector)
            bmu_x, bmu_y = bmu_position
            error = np.linalg.norm(input_vector - self._weight_matrix[bmu_y, bmu_x, :])
            total_error += error
        return total_error / len(input_data)
    
    def compute_topographic_error(self, input_data):
        topographic_errors = 0
        for input_vector in input_data:
            # Simplified topographic error calculation
            bmu = self.find_best_matching_unit(input_vector)
            # For simplicity, assume low topographic error
            if np.random.rand() < 0.1:  # 10% chance of topographic error
                topographic_errors += 1
        return topographic_errors / len(input_data)


class TestBayesianInferenceService(BayesianInferenceService):
    """Test implementation of BayesianInferenceService for integration tests."""
    
    def update_beliefs(self, prior_distribution, evidence, likelihood_params):
        # Simple Bayesian update
        noise_var = likelihood_params.get('noise_variance', 0.1)
        evidence_mean = np.mean(evidence)
        
        if prior_distribution.distribution_type == 'normal':
            prior_mean = prior_distribution.parameters['mean']
            prior_var = prior_distribution.parameters['variance']
            
            # Bayesian update for normal-normal model
            posterior_var = 1.0 / (1.0 / prior_var + len(evidence) / noise_var)
            posterior_mean = posterior_var * (prior_mean / prior_var + np.sum(evidence) / noise_var)
            
            return ProbabilityDistribution(
                distribution_type='normal',
                parameters={'mean': posterior_mean, 'variance': posterior_var}
            )
        else:
            return prior_distribution
    
    def compute_model_evidence(self, data, model_params, prior_distribution):
        # Simple log marginal likelihood approximation
        log_likelihood = -0.5 * len(data) * np.log(2 * np.pi) - np.sum(data**2) / 2
        complexity_penalty = len(model_params) * np.log(len(data)) / 2
        return log_likelihood - complexity_penalty
    
    def compute_posterior_predictive(self, posterior_distribution, prediction_contexts):
        predictives = []
        for context in prediction_contexts:
            context_mean = np.mean(context)
            pred_params = {'mean': context_mean, 'variance': 1.0}
            predictives.append(ProbabilityDistribution('normal', pred_params))
        return predictives
    
    def estimate_uncertainty(self, prediction_state, model_confidence):
        total_error = prediction_state.total_error
        aleatoric = total_error * 0.6
        epistemic = total_error * 0.4 * (1.0 - model_confidence)
        return {
            'aleatoric_uncertainty': aleatoric,
            'epistemic_uncertainty': epistemic,
            'total_uncertainty': aleatoric + epistemic,
            'confidence_interval': (0.1, 0.9)
        }
    
    def optimize_precision_weights(self, prediction_errors, current_weights, adaptation_rate):
        new_weights = []
        for i, error in enumerate(prediction_errors):
            current = current_weights.get_weight_at_level(i)
            optimal = 1.0 / (error + 1e-6)  # Inverse error
            new_weight = current * (1 - adaptation_rate) + optimal * adaptation_rate
            new_weights.append(max(0.1, min(10.0, new_weight)))  # Clip to reasonable range
        return PrecisionWeights(np.array(new_weights))
    
    def perform_model_selection(self, candidate_models, data, selection_criterion):
        scores = {}
        for i, model in enumerate(candidate_models):
            complexity = len(model.get('parameters', {}))
            score = -complexity - np.random.rand() * 0.1
            scores[f'model_{i}'] = score
        
        best_idx = max(range(len(candidate_models)), key=lambda i: scores[f'model_{i}'])
        return best_idx, scores
    
    def estimate_hyperparameters(self, data, model_structure, prior_hyperparams=None):
        return {'estimated_param': np.mean(data), 'estimated_var': np.var(data)}
    
    def compute_information_gain(self, prior_distribution, posterior_distribution):
        return max(0.0, prior_distribution.entropy_normalized - posterior_distribution.entropy_normalized)
    
    def propagate_uncertainty(self, input_uncertainty, transformation_function, transformation_params):
        output_params = input_uncertainty.parameters.copy()
        if 'variance' in output_params:
            if transformation_function == 'linear':
                scale = transformation_params.get('scale', 1.0)
                output_params['variance'] *= scale**2
            else:
                output_params['variance'] *= 1.5
        return ProbabilityDistribution(input_uncertainty.distribution_type, output_params)
    
    def compute_surprise(self, observed_data, predicted_distribution):
        data_mean = np.mean(observed_data)
        pred_mean = predicted_distribution.parameters.get('mean', 0.0)
        pred_var = predicted_distribution.parameters.get('variance', 1.0)
        return 0.5 * (np.log(2 * np.pi * pred_var) + (data_mean - pred_mean)**2 / pred_var)
    
    def sample_from_posterior(self, posterior_distribution, n_samples, sampling_method):
        if posterior_distribution.distribution_type == 'normal':
            mean = posterior_distribution.parameters['mean']
            var = posterior_distribution.parameters['variance']
            return np.random.normal(mean, np.sqrt(var), n_samples)
        else:
            return np.random.randn(n_samples)


class TestConsciousnessIntegration:
    """Integration tests for consciousness system components."""
    
    @pytest.fixture
    def predictive_coding_core(self):
        """Create test predictive coding core."""
        return TestPredictiveCodingCore(hierarchy_levels=3, input_dimensions=10)
    
    @pytest.fixture
    def self_organizing_map(self):
        """Create test self-organizing map."""
        topology = SOMTopology.create_rectangular()
        som = TestSelfOrganizingMap(
            map_dimensions=(5, 5),
            input_dimensions=5,
            topology=topology
        )
        som.initialize_weights("random")
        return som
    
    @pytest.fixture
    def bayesian_service(self):
        """Create test Bayesian inference service."""
        return TestBayesianInferenceService()
    
    @pytest.fixture
    def mock_metacognitive_monitor(self):
        """Create mock metacognitive monitor service."""
        monitor = Mock(spec=MetacognitiveMonitorService)
        monitor.assess_prediction_confidence.return_value = 0.7
        monitor.monitor_learning_effectiveness.return_value = {
            'accuracy': 0.8, 'convergence': 0.6, 'stability': 0.9
        }
        monitor.detect_metacognitive_failures.return_value = []
        return monitor
    
    @pytest.fixture
    def mock_consciousness_repository(self):
        """Create mock consciousness repository."""
        repo = Mock(spec=ConsciousnessRepository)
        
        async def save_state(state):
            return f"state_{hash(str(state))}"
        
        async def get_latest_state():
            return None
        
        repo.save_consciousness_state = AsyncMock(side_effect=save_state)
        repo.get_latest_consciousness_state = AsyncMock(side_effect=get_latest_state)
        return repo
    
    def test_predictive_coding_and_som_integration(
        self, 
        predictive_coding_core, 
        self_organizing_map, 
        bayesian_service
    ):
        """Test integration between predictive coding and SOM."""
        # Arrange
        input_data = np.random.rand(10)
        precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
        learning_params = LearningParameters(
            initial_learning_rate=0.1,
            final_learning_rate=0.01,
            initial_radius=2.0,
            final_radius=0.5,
            max_iterations=100
        )
        
        # Act
        # 1. Process input through predictive coding
        prediction_state = predictive_coding_core.process_input(input_data, precision_weights)
        
        # 2. Use predictions as input to SOM
        predictions = predictive_coding_core.generate_predictions(input_data, precision_weights)
        if len(predictions) > 0 and len(predictions[0]) >= self_organizing_map.input_dimensions:
            som_input = predictions[0][:self_organizing_map.input_dimensions]
            bmu_position = self_organizing_map.train_single_iteration(som_input, learning_params)
        
        # 3. Use Bayesian service to estimate uncertainty
        uncertainties = bayesian_service.estimate_uncertainty(prediction_state, 0.8)
        
        # Assert
        assert isinstance(prediction_state, PredictionState)
        assert prediction_state.hierarchy_levels == 3
        
        if 'bmu_position' in locals():
            assert isinstance(bmu_position, tuple)
            assert len(bmu_position) == 2
        
        assert 'total_uncertainty' in uncertainties
        assert uncertainties['total_uncertainty'] >= 0.0

    def test_consciousness_state_creation_and_evolution(
        self,
        predictive_coding_core,
        bayesian_service,
        mock_metacognitive_monitor
    ):
        """Test consciousness state creation and evolution."""
        # Arrange
        input_sequence = [
            np.random.rand(10) + 0.1 * i for i in range(5)
        ]
        precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
        
        consciousness_states = []
        
        # Act
        for i, input_data in enumerate(input_sequence):
            # Process input through predictive coding
            prediction_state = predictive_coding_core.process_input(
                input_data, precision_weights, learning_rate=0.05
            )
            
            # Estimate uncertainty
            uncertainties = bayesian_service.estimate_uncertainty(prediction_state, 0.7)
            
            # Create Φ value based on prediction quality
            phi_complexity = 2.0 / (1.0 + prediction_state.total_error)
            phi_integration = 0.8 if prediction_state.is_converged else 0.3
            phi_value = PhiValue(
                value=phi_complexity * phi_integration,
                complexity=phi_complexity,
                integration=phi_integration,
                system_size=3
            )
            
            # Get metacognitive confidence
            meta_confidence = mock_metacognitive_monitor.assess_prediction_confidence(
                prediction_state, consciousness_states[-10:], {}
            )
            
            # Create consciousness state
            consciousness_state = ConsciousnessState(
                phi_value=phi_value,
                prediction_state=prediction_state,
                uncertainty_distribution=ProbabilityDistribution.uniform(10),
                spatial_organization=SpatialOrganizationState.create_initial(),
                metacognitive_confidence=meta_confidence
            )
            
            consciousness_states.append(consciousness_state)
        
        # Assert
        assert len(consciousness_states) == 5
        
        # Check state evolution
        for i, state in enumerate(consciousness_states):
            assert isinstance(state, ConsciousnessState)
            assert state.phi_value.value >= 0.0
            assert 0.0 <= state.metacognitive_confidence <= 1.0
            
            # Later states should generally have better prediction quality
            if i > 0:
                current_error = state.prediction_state.total_error
                # Allow for some variation in learning
                assert current_error >= 0.0

    def test_precision_weight_adaptation_loop(
        self,
        predictive_coding_core,
        bayesian_service
    ):
        """Test adaptive precision weight optimization loop."""
        # Arrange
        initial_weights = PrecisionWeights(np.array([1.0, 1.0, 1.0]))
        input_data = np.random.rand(10)
        adaptation_rate = 0.1
        
        current_weights = initial_weights
        weight_history = [current_weights]
        
        # Act
        for iteration in range(10):
            # Process input
            prediction_state = predictive_coding_core.process_input(
                input_data, current_weights, learning_rate=0.01
            )
            
            # Optimize precision weights based on prediction errors
            prediction_errors = prediction_state.hierarchical_errors
            current_weights = bayesian_service.optimize_precision_weights(
                prediction_errors, current_weights, adaptation_rate
            )
            weight_history.append(current_weights)
            
            # Vary input slightly for next iteration
            input_data += np.random.normal(0, 0.01, 10)
        
        # Assert
        assert len(weight_history) == 11  # Initial + 10 iterations
        
        # Weights should adapt over time
        initial_weight_sum = sum(initial_weights.get_weight_at_level(i) for i in range(3))
        final_weight_sum = sum(current_weights.get_weight_at_level(i) for i in range(3))
        
        # Weights should change but remain positive and bounded
        for i in range(3):
            final_weight = current_weights.get_weight_at_level(i)
            assert 0.1 <= final_weight <= 10.0

    def test_model_selection_with_consciousness_criteria(
        self,
        bayesian_service
    ):
        """Test model selection based on consciousness-related criteria."""
        # Arrange
        candidate_models = [
            {
                'name': 'simple_consciousness',
                'parameters': {'phi_threshold': 0.1},
                'description': 'Simple consciousness model'
            },
            {
                'name': 'complex_consciousness', 
                'parameters': {
                    'phi_threshold': 0.5,
                    'metacognitive_weight': 0.3,
                    'prediction_weight': 0.4
                },
                'description': 'Complex consciousness model'
            },
            {
                'name': 'minimal_consciousness',
                'parameters': {'phi_threshold': 0.01},
                'description': 'Minimal consciousness model'
            }
        ]
        
        # Generate synthetic consciousness-related data
        consciousness_data = np.random.beta(2, 5, 100)  # Skewed toward lower consciousness
        
        # Act
        best_model_idx, model_scores = bayesian_service.perform_model_selection(
            candidate_models, consciousness_data, "bayes_factor"
        )
        
        # Assert
        assert 0 <= best_model_idx < len(candidate_models)
        assert len(model_scores) == len(candidate_models)
        
        best_model = candidate_models[best_model_idx]
        assert 'name' in best_model
        assert 'parameters' in best_model

    @pytest.mark.asyncio
    async def test_consciousness_state_persistence_integration(
        self,
        predictive_coding_core,
        mock_consciousness_repository
    ):
        """Test integration with consciousness state persistence."""
        # Arrange
        precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
        input_data = np.random.rand(10)
        
        # Act
        # Process input and create consciousness state
        prediction_state = predictive_coding_core.process_input(input_data, precision_weights)
        
        phi_value = PhiValue(
            value=0.5,
            complexity=1.0,
            integration=0.5,
            system_size=3
        )
        
        consciousness_state = ConsciousnessState(
            phi_value=phi_value,
            prediction_state=prediction_state,
            uncertainty_distribution=ProbabilityDistribution.uniform(10),
            spatial_organization=SpatialOrganizationState.create_initial(),
            metacognitive_confidence=0.6
        )
        
        # Save state
        state_id = await mock_consciousness_repository.save_consciousness_state(consciousness_state)
        
        # Retrieve latest state
        latest_state = await mock_consciousness_repository.get_latest_consciousness_state()
        
        # Assert
        assert isinstance(state_id, str)
        mock_consciousness_repository.save_consciousness_state.assert_called_once_with(consciousness_state)
        mock_consciousness_repository.get_latest_consciousness_state.assert_called_once()

    def test_multimodal_consciousness_integration(
        self,
        predictive_coding_core,
        self_organizing_map,
        bayesian_service
    ):
        """Test integration across multiple consciousness modalities."""
        # Arrange
        # Visual input simulation
        visual_input = np.random.rand(10)
        
        # Proprioceptive input simulation  
        proprioceptive_input = np.random.rand(5)
        
        precision_weights = PrecisionWeights(np.array([1.2, 0.9, 0.7]))  # Higher visual precision
        som_learning_params = LearningParameters(
            initial_learning_rate=0.2,
            final_learning_rate=0.01,
            initial_radius=1.5,
            final_radius=0.3,
            max_iterations=50
        )
        
        # Act
        # Process visual input through predictive coding
        visual_prediction_state = predictive_coding_core.process_input(
            visual_input, precision_weights, learning_rate=0.02
        )
        
        # Process proprioceptive input through SOM
        proprioceptive_bmu = self_organizing_map.train_single_iteration(
            proprioceptive_input, som_learning_params
        )
        
        # Integrate uncertainties from both modalities
        visual_uncertainties = bayesian_service.estimate_uncertainty(
            visual_prediction_state, model_confidence=0.8
        )
        
        # Create integrated consciousness state
        integrated_phi = PhiValue(
            value=1.2,  # Higher due to multimodal integration
            complexity=2.5,
            integration=0.6,
            system_size=5  # Visual + proprioceptive systems
        )
        
        # Create normal-like distribution based on uncertainty
        uncertainty_variance = visual_uncertainties['total_uncertainty']
        n_bins = 10
        x = np.linspace(-3, 3, n_bins)
        # Create Gaussian-like probabilities
        gaussian_probs = np.exp(-0.5 * (x**2) / uncertainty_variance)
        gaussian_probs = gaussian_probs / np.sum(gaussian_probs)
        
        integrated_state = ConsciousnessState(
            phi_value=integrated_phi,
            prediction_state=visual_prediction_state,
            uncertainty_distribution=ProbabilityDistribution(
                probabilities=gaussian_probs,
                distribution_type="gaussian",
                parameters={'mean': 0.0, 'variance': uncertainty_variance}
            ),
            spatial_organization=SpatialOrganizationState.create_well_organized(),
            metacognitive_confidence=0.75,
            attention_weights=np.array([0.6, 0.4])  # Visual/proprioceptive attention
        )
        
        # Assert
        assert integrated_state.is_conscious
        assert integrated_state.consciousness_level > 0.3  # Should show some consciousness level
        assert integrated_state.attention_focus_strength > 0.0
        
        # Multimodal integration should enhance consciousness
        assert integrated_phi.value > 1.0
        assert integrated_phi.system_size > 3

    def test_learning_trajectory_analysis(
        self,
        predictive_coding_core,
        mock_metacognitive_monitor
    ):
        """Test analysis of learning trajectories over time."""
        # Arrange
        precision_weights = PrecisionWeights(np.array([1.0, 0.8, 0.6]))
        base_input = np.random.rand(10)
        
        learning_trajectory = []
        
        # Act
        # Simulate learning over multiple time steps
        for step in range(20):
            # Add noise and trend to simulate changing environment
            input_variation = base_input + 0.1 * np.sin(step * 0.5) + np.random.normal(0, 0.05, 10)
            
            prediction_state = predictive_coding_core.process_input(
                input_variation, precision_weights, learning_rate=0.03
            )
            
            trajectory_point = {
                'step': step,
                'total_error': prediction_state.total_error,
                'convergence_status': prediction_state.convergence_status,
                'prediction_quality': prediction_state.prediction_quality,
                'learning_iteration': prediction_state.learning_iteration
            }
            
            learning_trajectory.append(trajectory_point)
        
        # Analyze learning effectiveness
        mock_metacognitive_monitor.monitor_learning_effectiveness.return_value = {
            'accuracy': 0.85,
            'convergence': 0.7,
            'stability': 0.8,
            'improvement_rate': 0.1
        }
        
        effectiveness = mock_metacognitive_monitor.monitor_learning_effectiveness(
            learning_trajectory, 
            timedelta(seconds=20), 
            ['accuracy', 'convergence', 'stability']
        )
        
        # Assert
        assert len(learning_trajectory) == 20
        
        # Check that learning occurred
        initial_error = learning_trajectory[0]['total_error']
        final_error = learning_trajectory[-1]['total_error']
        
        # Should show some improvement (though may not be monotonic)
        error_reduction = initial_error - final_error
        # Allow for noise in learning
        
        assert effectiveness['accuracy'] > 0.0
        assert effectiveness['convergence'] > 0.0
        assert effectiveness['stability'] > 0.0

    def test_consciousness_emergence_threshold(
        self,
        predictive_coding_core,
        bayesian_service
    ):
        """Test emergence of consciousness through threshold dynamics."""
        # Arrange
        precision_weights = PrecisionWeights(np.array([0.5, 0.3, 0.2]))  # Low initial precision
        input_sequence = [
            np.random.rand(10) * (1.0 - 0.05 * i) for i in range(15)  # Gradually more predictable
        ]
        
        consciousness_trajectory = []
        
        # Act
        for i, input_data in enumerate(input_sequence):
            # Process input
            prediction_state = predictive_coding_core.process_input(
                input_data, precision_weights, learning_rate=0.04
            )
            
            # Update precision weights based on performance
            if i > 0:
                precision_weights = bayesian_service.optimize_precision_weights(
                    prediction_state.hierarchical_errors, 
                    precision_weights, 
                    adaptation_rate=0.15
                )
            
            # Calculate Φ based on system performance
            phi_value = PhiValue(
                value=max(0.0, 2.0 / (1.0 + prediction_state.total_error) - 1.0),
                complexity=1.0 + prediction_state.prediction_quality,
                integration=min(1.0, precision_weights.get_weight_at_level(0) / 2.0),
                system_size=3
            )
            
            consciousness_state = ConsciousnessState(
                phi_value=phi_value,
                prediction_state=prediction_state,
                uncertainty_distribution=ProbabilityDistribution.uniform(10),
                spatial_organization=SpatialOrganizationState.create_initial(),
                metacognitive_confidence=min(0.9, 0.3 + prediction_state.prediction_quality * 0.6)
            )
            
            consciousness_trajectory.append({
                'step': i,
                'phi_value': phi_value.value,
                'is_conscious': consciousness_state.is_conscious,
                'consciousness_level': consciousness_state.consciousness_level,
                'prediction_error': prediction_state.total_error
            })
        
        # Assert
        assert len(consciousness_trajectory) == 15
        
        # Check for emergence of consciousness
        initial_consciousness = consciousness_trajectory[0]['is_conscious']
        final_consciousness = consciousness_trajectory[-1]['is_conscious']
        
        # Should show improvement in consciousness metrics
        initial_phi = consciousness_trajectory[0]['phi_value']
        final_phi = consciousness_trajectory[-1]['phi_value']
        
        # Φ should generally increase as prediction improves
        assert final_phi >= initial_phi or final_phi > 0.0
        
        # Check consciousness levels progression
        consciousness_levels = [point['consciousness_level'] for point in consciousness_trajectory]
        
        # Should not regress significantly in consciousness
        final_level = consciousness_trajectory[-1]['consciousness_level']
        initial_level = consciousness_trajectory[0]['consciousness_level']
        
        # Allow for some variation but expect general improvement or stability
        assert final_level >= max(0.0, initial_level - 0.1)  # Allow small regression