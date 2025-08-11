"""
Global pytest configuration and fixtures for the enactive consciousness project.

This module provides shared test fixtures, test data factories, and configuration
following TDD best practices. Fixtures support the Arrange phase of AAA pattern
and enable test isolation and repeatability.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock

# Import domain objects
from domain.value_objects.phi_value import PhiValue
from domain.value_objects.consciousness_state import ConsciousnessState
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.probability_distribution import ProbabilityDistribution
from domain.value_objects.precision_weights import PrecisionWeights
from domain.value_objects.learning_parameters import LearningParameters
from domain.value_objects.som_topology import SOMTopology


# ========================
# Value Object Factories
# ========================

@pytest.fixture
def phi_value_factory():
    """Factory for creating PhiValue instances with various configurations."""
    def _create_phi_value(
        value: float = 0.5,
        complexity: float = 1.0,
        integration: float = 0.5,
        system_size: int = 3,
        computation_method: str = "approximate",
        confidence: float = 0.8
    ) -> PhiValue:
        return PhiValue(
            value=value,
            complexity=complexity,
            integration=integration,
            system_size=system_size,
            computation_method=computation_method,
            confidence=confidence
        )
    return _create_phi_value


@pytest.fixture
def prediction_state_factory():
    """Factory for creating PredictionState instances."""
    def _create_prediction_state(
        hierarchy_levels: int = 3,
        errors: Optional[List[float]] = None,
        convergence_status: str = "not_converged",
        learning_iteration: int = 0
    ) -> PredictionState:
        if errors is None:
            errors = [0.1 * (i + 1) for i in range(hierarchy_levels)]
        
        return PredictionState(
            hierarchical_errors=errors,
            convergence_status=convergence_status,
            learning_iteration=learning_iteration,
            timestamp=datetime.now()
        )
    return _create_prediction_state


@pytest.fixture
def consciousness_state_factory(phi_value_factory, prediction_state_factory):
    """Factory for creating ConsciousnessState instances."""
    def _create_consciousness_state(
        phi_value: Optional[PhiValue] = None,
        prediction_state: Optional[PredictionState] = None,
        metacognitive_confidence: float = 0.3,
        attention_weights: Optional[np.ndarray] = None
    ) -> ConsciousnessState:
        if phi_value is None:
            phi_value = phi_value_factory()
        if prediction_state is None:
            prediction_state = prediction_state_factory()
        
        return ConsciousnessState(
            phi_value=phi_value,
            prediction_state=prediction_state,
            uncertainty_distribution=ProbabilityDistribution.uniform(5),
            metacognitive_confidence=metacognitive_confidence,
            attention_weights=attention_weights
        )
    return _create_consciousness_state


# ========================
# Property-Based Testing Strategies
# ========================

@pytest.fixture
def phi_value_property_generators():
    """Generators for property-based testing of PhiValue."""
    def generate_valid_phi_values(count: int = 10):
        """Generate valid PhiValue instances for property testing."""
        import random
        
        for _ in range(count):
            yield PhiValue(
                value=random.uniform(0.0, 10.0),
                complexity=random.uniform(0.1, 5.0),
                integration=random.uniform(0.1, 1.0),
                system_size=random.randint(1, 20),
                computation_method=random.choice(["exact", "approximate", "heuristic"]),
                confidence=random.uniform(0.1, 1.0)
            )
    
    def generate_edge_case_phi_values():
        """Generate edge case PhiValue instances."""
        edge_cases = [
            # Zero phi
            PhiValue(0.0, 0.0, 0.0, 1, "exact", 1.0),
            # Minimal phi
            PhiValue(0.001, 0.01, 0.1, 1, "approximate", 0.1),
            # High phi
            PhiValue(50.0, 10.0, 5.0, 10, "heuristic", 0.9),
            # Single element system
            PhiValue(0.1, 0.2, 0.5, 1, "exact", 1.0),
            # Large system
            PhiValue(5.0, 20.0, 0.25, 100, "approximate", 0.5)
        ]
        return edge_cases
    
    return {
        'valid_generator': generate_valid_phi_values,
        'edge_cases': generate_edge_case_phi_values
    }


@pytest.fixture
def prediction_error_patterns():
    """Common prediction error patterns for testing."""
    return {
        'decreasing': [1.0, 0.5, 0.25, 0.125],  # Converging
        'increasing': [0.1, 0.2, 0.4, 0.8],     # Diverging
        'oscillating': [0.5, 0.1, 0.6, 0.05],   # Unstable
        'stable_low': [0.1, 0.11, 0.09, 0.1],   # Converged
        'stable_high': [2.0, 2.1, 1.9, 2.0],    # High error plateau
        'noisy': [0.3, 0.7, 0.2, 0.9, 0.1],     # Noisy convergence
    }


# ========================
# Mock Objects and Test Doubles
# ========================

@pytest.fixture
def mock_bayesian_inference_service():
    """Mock BayesianInferenceService for testing."""
    mock_service = Mock()
    
    # Configure default behaviors
    mock_service.update_beliefs.return_value = ProbabilityDistribution.uniform(5)
    mock_service.compute_model_evidence.return_value = -2.5
    mock_service.estimate_uncertainty.return_value = {
        'aleatoric_uncertainty': 0.1,
        'epistemic_uncertainty': 0.2,
        'total_uncertainty': 0.3,
        'confidence_interval': (0.2, 0.8)
    }
    mock_service.compute_surprise.return_value = 1.5
    
    return mock_service


@pytest.fixture
def mock_metacognitive_monitor():
    """Mock MetacognitiveMonitorService for testing."""
    mock_monitor = Mock()
    
    # Configure default behaviors
    mock_monitor.assess_prediction_confidence.return_value = 0.7
    mock_monitor.monitor_learning_effectiveness.return_value = {
        'accuracy': 0.8,
        'convergence': 0.6,
        'stability': 0.9
    }
    mock_monitor.detect_metacognitive_failures.return_value = []
    
    return mock_monitor


@pytest.fixture
def mock_consciousness_repository():
    """Mock ConsciousnessRepository for testing."""
    mock_repo = Mock()
    
    # Configure async methods
    async def mock_save_state(state):
        return f"state_{hash(str(state))}"
    
    async def mock_get_state(state_id):
        return None  # Customizable in tests
    
    async def mock_health_check():
        return {
            'status': 'healthy',
            'response_time': 50,
            'storage_usage': {'used': 100, 'total': 1000},
            'error_count': 0
        }
    
    mock_repo.save_consciousness_state.side_effect = mock_save_state
    mock_repo.get_consciousness_state.side_effect = mock_get_state
    mock_repo.health_check.side_effect = mock_health_check
    
    return mock_repo


@pytest.fixture
def mock_predictive_coding_core():
    """Mock PredictiveCodingCore for testing."""
    from unittest.mock import create_autospec
    from domain.entities.predictive_coding_core import PredictiveCodingCore
    
    mock_core = create_autospec(PredictiveCodingCore, spec_set=True)
    mock_core.hierarchy_levels = 3
    mock_core.input_dimensions = 10
    
    # Configure method behaviors
    mock_core.generate_predictions.return_value = [
        np.random.rand(10),
        np.random.rand(8),
        np.random.rand(5)
    ]
    mock_core.compute_prediction_errors.return_value = [0.1, 0.2, 0.3]
    
    return mock_core


@pytest.fixture
def mock_self_organizing_map():
    """Mock SelfOrganizingMap for testing."""
    from unittest.mock import create_autospec
    from domain.entities.self_organizing_map import SelfOrganizingMap
    
    mock_som = create_autospec(SelfOrganizingMap, spec_set=True)
    mock_som.map_dimensions = (10, 10)
    mock_som.input_dimensions = 5
    mock_som.is_trained = True
    mock_som.training_iterations = 100
    
    # Configure method behaviors
    mock_som.find_best_matching_unit.return_value = (5, 5)
    mock_som.compute_quantization_error.return_value = 0.15
    mock_som.compute_topographic_error.return_value = 0.05
    
    return mock_som


# ========================
# Test Data Builders
# ========================

class TestDataBuilder:
    """Base class for test data builders using the Builder pattern."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset builder to default state."""
        raise NotImplementedError
    
    def build(self):
        """Build the final object."""
        raise NotImplementedError


class PhiValueBuilder(TestDataBuilder):
    """Builder for creating PhiValue test data."""
    
    def reset(self):
        self._value = 0.5
        self._complexity = 1.0
        self._integration = 0.5
        self._system_size = 3
        self._computation_method = "approximate"
        self._confidence = 0.8
        return self
    
    def with_value(self, value: float):
        self._value = value
        return self
    
    def with_complexity(self, complexity: float):
        self._complexity = complexity
        return self
    
    def with_integration(self, integration: float):
        self._integration = integration
        return self
    
    def unconscious(self):
        self._value = 0.0
        self._complexity = 0.0
        self._integration = 0.0
        return self
    
    def minimal_consciousness(self):
        self._value = 0.1
        self._complexity = 0.2
        self._integration = 0.5
        return self
    
    def high_consciousness(self):
        self._value = 5.0
        self._complexity = 10.0
        self._integration = 0.5
        return self
    
    def build(self) -> PhiValue:
        return PhiValue(
            value=self._value,
            complexity=self._complexity,
            integration=self._integration,
            system_size=self._system_size,
            computation_method=self._computation_method,
            confidence=self._confidence
        )


@pytest.fixture
def phi_value_builder():
    """Fixture providing PhiValue builder."""
    return PhiValueBuilder()


# ========================
# Test Environment Configuration
# ========================

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        'random_seed': 42,
        'tolerance': 1e-6,
        'max_iterations': 1000,
        'timeout_seconds': 30,
        'property_test_count': 100,
        'benchmark_iterations': 10
    }


@pytest.fixture(autouse=True)
def setup_numpy_seed(test_config):
    """Set numpy random seed for reproducible tests."""
    np.random.seed(test_config['random_seed'])


# ========================
# Performance Testing Utilities
# ========================

@pytest.fixture
def performance_timer():
    """Timer utility for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


# ========================
# Integration Test Utilities
# ========================

@pytest.fixture
def integration_test_environment():
    """Setup for integration testing."""
    return {
        'temp_directory': '/tmp/consciousness_test',
        'test_database_url': 'sqlite:///:memory:',
        'mock_external_services': True,
        'enable_logging': False
    }


# ========================
# Japanese Text Testing Support
# ========================

@pytest.fixture
def japanese_text_samples():
    """Japanese text samples for GUI and internationalization testing."""
    return {
        'consciousness_levels': {
            'unconscious': '無意識',
            'minimal': '最小意識',
            'moderate': '中等意識', 
            'high': '高度意識',
            'very_high': '最高意識'
        },
        'prediction_states': {
            'converged': '収束済み',
            'converging': '収束中',
            'not_converged': '未収束',
            'diverged': '発散'
        },
        'error_messages': {
            'invalid_phi': 'Φ値が無効です',
            'computation_failed': '計算に失敗しました',
            'state_inconsistent': '状態が矛盾しています'
        }
    }


# ========================
# Hypothesis Strategy Fixtures
# ========================

@pytest.fixture
def hypothesis_strategies():
    """Hypothesis strategies for property-based testing."""
    try:
        from hypothesis import strategies as st
        
        return {
            'phi_values': st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            'complexities': st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            'integrations': st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            'system_sizes': st.integers(min_value=1, max_value=1000),
            'confidence_levels': st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            'hierarchy_levels': st.integers(min_value=1, max_value=10),
            'error_lists': st.lists(
                st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
                min_size=1, max_size=10
            )
        }
    except ImportError:
        # Hypothesis not available, return None
        return None