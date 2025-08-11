"""
Unit tests for ConsciousnessState value object.

Tests the immutable consciousness state representation following
TDD principles with comprehensive coverage of value object behavior.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

from domain.value_objects.consciousness_state import ConsciousnessState
from domain.value_objects.phi_value import PhiValue
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.probability_distribution import ProbabilityDistribution
from domain.value_objects.spatial_organization_state import SpatialOrganizationState


class TestConsciousnessState:
    """Test suite for ConsciousnessState value object."""

    @pytest.fixture
    def minimal_phi_value(self):
        """Create a minimal Φ value for testing."""
        return PhiValue(
            value=0.5,
            complexity=1.0,
            integration=0.5,
            system_size=3,
            computation_method="approximate",
            confidence=0.8
        )

    @pytest.fixture
    def basic_prediction_state(self):
        """Create a basic prediction state for testing."""
        return PredictionState.create_empty(hierarchy_levels=3)

    @pytest.fixture
    def uniform_distribution(self):
        """Create a uniform probability distribution for testing."""
        return ProbabilityDistribution.uniform(5)

    @pytest.fixture
    def basic_spatial_organization(self):
        """Create a basic spatial organization state for testing."""
        return SpatialOrganizationState.create_initial()

    @pytest.fixture
    def basic_consciousness_state(self, minimal_phi_value, basic_prediction_state, uniform_distribution, basic_spatial_organization):
        """Create a basic consciousness state for testing."""
        return ConsciousnessState(
            phi_value=minimal_phi_value,
            prediction_state=basic_prediction_state,
            uncertainty_distribution=uniform_distribution,
            spatial_organization=basic_spatial_organization,
            metacognitive_confidence=0.3
        )

    def test_consciousness_state_creation(self, basic_consciousness_state):
        """Test basic consciousness state creation."""
        assert basic_consciousness_state.phi_value.value == 0.5
        assert basic_consciousness_state.metacognitive_confidence == 0.3
        assert isinstance(basic_consciousness_state.timestamp, datetime)

    def test_consciousness_state_immutability(self, basic_consciousness_state):
        """Test that consciousness state is immutable."""
        with pytest.raises(AttributeError):
            basic_consciousness_state.phi_value = PhiValue.create_zero()

    def test_metacognitive_confidence_validation(self, minimal_phi_value, basic_prediction_state, uniform_distribution, basic_spatial_organization):
        """Test validation of metacognitive confidence range."""
        # Test invalid confidence values
        with pytest.raises(ValueError, match="Metacognitive confidence must be in \\[0, 1\\]"):
            ConsciousnessState(
                phi_value=minimal_phi_value,
                prediction_state=basic_prediction_state,
                uncertainty_distribution=uniform_distribution,
                spatial_organization=basic_spatial_organization,
                metacognitive_confidence=-0.1
            )

        with pytest.raises(ValueError, match="Metacognitive confidence must be in \\[0, 1\\]"):
            ConsciousnessState(
                phi_value=minimal_phi_value,
                prediction_state=basic_prediction_state,
                uncertainty_distribution=uniform_distribution,
                spatial_organization=basic_spatial_organization,
                metacognitive_confidence=1.1
            )

    def test_attention_weights_validation(self, minimal_phi_value, basic_prediction_state, uniform_distribution, basic_spatial_organization):
        """Test validation of attention weights."""
        # Test invalid attention weights (don't sum to 1)
        invalid_weights = np.array([0.3, 0.4, 0.1])  # Sum = 0.8
        with pytest.raises(ValueError, match="Attention weights must sum to 1.0"):
            ConsciousnessState(
                phi_value=minimal_phi_value,
                prediction_state=basic_prediction_state,
                uncertainty_distribution=uniform_distribution,
                spatial_organization=basic_spatial_organization,
                attention_weights=invalid_weights
            )

        # Test valid attention weights
        valid_weights = np.array([0.3, 0.4, 0.3])  # Sum = 1.0
        state = ConsciousnessState(
            phi_value=minimal_phi_value,
            prediction_state=basic_prediction_state,
            uncertainty_distribution=uniform_distribution,
            spatial_organization=basic_spatial_organization,
            attention_weights=valid_weights
        )
        assert np.allclose(state.attention_weights, valid_weights)

    def test_is_conscious_property(self, minimal_phi_value, basic_prediction_state, uniform_distribution, basic_spatial_organization):
        """Test consciousness detection logic."""
        # Test conscious state
        conscious_state = ConsciousnessState(
            phi_value=minimal_phi_value,  # Φ > 0
            prediction_state=basic_prediction_state,  # Low error
            uncertainty_distribution=uniform_distribution,
            spatial_organization=basic_spatial_organization,
            metacognitive_confidence=0.2  # > 0.1
        )
        assert conscious_state.is_conscious

        # Test unconscious state (zero Φ)
        unconscious_phi = PhiValue.create_zero()
        unconscious_state = ConsciousnessState(
            phi_value=unconscious_phi,
            prediction_state=basic_prediction_state,
            uncertainty_distribution=uniform_distribution,
            spatial_organization=basic_spatial_organization,
            metacognitive_confidence=0.2
        )
        assert not unconscious_state.is_conscious

    def test_consciousness_level_calculation(self, basic_consciousness_state):
        """Test consciousness level calculation."""
        level = basic_consciousness_state.consciousness_level
        assert 0.0 <= level <= 1.0
        assert level > 0  # Should be conscious

    def test_attention_focus_strength(self, minimal_phi_value, basic_prediction_state, uniform_distribution, basic_spatial_organization):
        """Test attention focus strength calculation."""
        # Test with no attention weights
        state_no_attention = ConsciousnessState(
            phi_value=minimal_phi_value,
            prediction_state=basic_prediction_state,
            uncertainty_distribution=uniform_distribution,
            spatial_organization=basic_spatial_organization
        )
        assert state_no_attention.attention_focus_strength == 0.0

        # Test with focused attention (one dominant weight)
        focused_weights = np.array([0.8, 0.1, 0.1])  # Focused on first element
        focused_state = ConsciousnessState(
            phi_value=minimal_phi_value,
            prediction_state=basic_prediction_state,
            uncertainty_distribution=uniform_distribution,
            spatial_organization=basic_spatial_organization,
            attention_weights=focused_weights
        )
        focus_strength = focused_state.attention_focus_strength
        assert focus_strength > 0.3  # Should be fairly focused

        # Test with uniform attention
        uniform_weights = np.array([1/3, 1/3, 1/3])  # Uniform distribution
        uniform_state = ConsciousnessState(
            phi_value=minimal_phi_value,
            prediction_state=basic_prediction_state,
            uncertainty_distribution=uniform_distribution,
            spatial_organization=basic_spatial_organization,
            attention_weights=uniform_weights
        )
        uniform_focus = uniform_state.attention_focus_strength
        assert uniform_focus < focus_strength  # Less focused than the focused case

    def test_with_updated_phi(self, basic_consciousness_state):
        """Test creating new state with updated Φ value."""
        new_phi = PhiValue(
            value=1.0,
            complexity=2.0,
            integration=0.5,
            system_size=3
        )
        
        updated_state = basic_consciousness_state.with_updated_phi(new_phi)
        
        # Check that new state has updated Φ
        assert updated_state.phi_value.value == 1.0
        assert updated_state.phi_value.complexity == 2.0
        
        # Check that original state is unchanged (immutability)
        assert basic_consciousness_state.phi_value.value == 0.5
        
        # Check that other properties are preserved
        assert updated_state.metacognitive_confidence == basic_consciousness_state.metacognitive_confidence
        assert updated_state.prediction_state == basic_consciousness_state.prediction_state

    def test_with_updated_prediction(self, basic_consciousness_state):
        """Test creating new state with updated prediction state."""
        new_prediction_state = PredictionState(
            hierarchical_errors=[0.1, 0.2, 0.3],
            convergence_status="converged",
            learning_iteration=100
        )
        
        updated_state = basic_consciousness_state.with_updated_prediction(new_prediction_state)
        
        # Check that prediction state is updated
        assert updated_state.prediction_state.learning_iteration == 100
        assert updated_state.prediction_state.convergence_status == "converged"
        
        # Check immutability of original state
        assert basic_consciousness_state.prediction_state.learning_iteration == 0

    def test_add_phenomenological_marker(self, basic_consciousness_state):
        """Test adding phenomenological markers."""
        updated_state = basic_consciousness_state.add_phenomenological_marker("test_key", "test_value")
        
        # Check that marker was added
        assert updated_state.phenomenological_markers["test_key"] == "test_value"
        
        # Check immutability
        assert "test_key" not in basic_consciousness_state.phenomenological_markers

    def test_to_dict_conversion(self, basic_consciousness_state):
        """Test conversion to dictionary format."""
        state_dict = basic_consciousness_state.to_dict()
        
        # Check required keys are present
        required_keys = [
            "phi_value", "prediction_state", "uncertainty_distribution",
            "spatial_organization", "timestamp", "metacognitive_confidence", 
            "is_conscious", "consciousness_level", "attention_focus_strength"
        ]
        
        for key in required_keys:
            assert key in state_dict
        
        # Check data types
        assert isinstance(state_dict["phi_value"], dict)
        assert isinstance(state_dict["is_conscious"], bool)
        assert isinstance(state_dict["consciousness_level"], float)
        assert isinstance(state_dict["timestamp"], str)  # ISO format

    def test_create_minimal_consciousness(self):
        """Test factory method for minimal consciousness."""
        minimal_state = ConsciousnessState.create_minimal_consciousness()
        
        assert minimal_state.is_conscious
        assert minimal_state.phi_value.value > 0
        assert minimal_state.metacognitive_confidence > 0
        assert "minimal" in minimal_state.phenomenological_markers

    def test_timestamp_consistency(self, basic_consciousness_state):
        """Test timestamp consistency validation."""
        # Create a prediction state with inconsistent timestamp
        old_prediction_state = PredictionState(
            hierarchical_errors=[0.1, 0.2, 0.3],
            timestamp=datetime.now() - timedelta(hours=1)  # 1 hour ago
        )
        
        # This should raise a validation error due to timestamp inconsistency
        with pytest.raises(ValueError, match="Prediction state timestamp inconsistent"):
            ConsciousnessState(
                phi_value=basic_consciousness_state.phi_value,
                prediction_state=old_prediction_state,
                uncertainty_distribution=basic_consciousness_state.uncertainty_distribution,
                spatial_organization=basic_consciousness_state.spatial_organization,
                timestamp=datetime.now()  # Current time
            )

    def test_consciousness_state_equality(self, basic_consciousness_state):
        """Test value-based equality of consciousness states."""
        # Create identical state
        identical_state = ConsciousnessState(
            phi_value=basic_consciousness_state.phi_value,
            prediction_state=basic_consciousness_state.prediction_state,
            uncertainty_distribution=basic_consciousness_state.uncertainty_distribution,
            spatial_organization=basic_consciousness_state.spatial_organization,
            timestamp=basic_consciousness_state.timestamp,
            metacognitive_confidence=basic_consciousness_state.metacognitive_confidence,
            attention_weights=basic_consciousness_state.attention_weights,
            phenomenological_markers=basic_consciousness_state.phenomenological_markers
        )
        
        # States should be equal based on values (frozen dataclass behavior)
        assert basic_consciousness_state == identical_state

    def test_with_updated_spatial_organization(self, basic_consciousness_state):
        """Test creating new state with updated spatial organization."""
        new_spatial_org = SpatialOrganizationState.create_well_organized()
        
        updated_state = basic_consciousness_state.with_updated_spatial_organization(new_spatial_org)
        
        # Check that spatial organization is updated
        assert updated_state.spatial_organization.organization_quality == 0.8
        assert updated_state.spatial_organization.is_well_organized
        
        # Check immutability of original state
        assert basic_consciousness_state.spatial_organization.organization_quality == 0.1
        
        # Check that other properties are preserved
        assert updated_state.phi_value == basic_consciousness_state.phi_value
        assert updated_state.prediction_state == basic_consciousness_state.prediction_state

    def test_consciousness_level_includes_spatial_contribution(self, basic_consciousness_state):
        """Test that consciousness level calculation includes spatial organization contribution."""
        # Create state with well-organized spatial state
        well_organized_spatial = SpatialOrganizationState.create_well_organized()
        state_with_good_spatial = basic_consciousness_state.with_updated_spatial_organization(well_organized_spatial)
        
        # Consciousness level should be higher with better spatial organization
        base_level = basic_consciousness_state.consciousness_level
        enhanced_level = state_with_good_spatial.consciousness_level
        
        assert enhanced_level > base_level, "Better spatial organization should increase consciousness level"

    def test_spatial_organization_properties(self, basic_consciousness_state):
        """Test spatial organization state integration."""
        # Test accessing spatial organization properties
        assert hasattr(basic_consciousness_state.spatial_organization, 'consciousness_contribution')
        assert 0.0 <= basic_consciousness_state.spatial_organization.consciousness_contribution <= 1.0
        
        # Test spatial organization in dictionary conversion
        state_dict = basic_consciousness_state.to_dict()
        assert "spatial_organization" in state_dict
        assert isinstance(state_dict["spatial_organization"], dict)
        assert "organization_quality" in state_dict["spatial_organization"]