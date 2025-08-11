"""
Unit tests for SpatialOrganizationState value object.

Tests the immutable spatial organization state representation following
TDD principles with comprehensive coverage of SOM-based domain concepts.
"""

import pytest
from datetime import datetime
import numpy as np

from domain.value_objects.spatial_organization_state import SpatialOrganizationState


class TestSpatialOrganizationState:
    """Test suite for SpatialOrganizationState value object."""

    @pytest.fixture
    def basic_spatial_state(self):
        """Create a basic spatial organization state for testing."""
        return SpatialOrganizationState(
            optimal_representation=(3, 4),
            structural_coherence=0.6,
            organization_quality=0.5,
            phenomenological_quality=0.4,
            temporal_consistency=0.3,
            map_dimensions=(10, 10),
            quantization_error=0.5,
            activation_history=[(3, 4), (3, 3), (4, 4)]
        )

    def test_spatial_organization_creation(self, basic_spatial_state):
        """Test basic spatial organization state creation."""
        assert basic_spatial_state.optimal_representation == (3, 4)
        assert basic_spatial_state.structural_coherence == 0.6
        assert basic_spatial_state.organization_quality == 0.5
        assert basic_spatial_state.map_dimensions == (10, 10)
        assert isinstance(basic_spatial_state.timestamp, datetime)

    def test_spatial_organization_immutability(self, basic_spatial_state):
        """Test that spatial organization state is immutable."""
        with pytest.raises(AttributeError):
            basic_spatial_state.optimal_representation = (5, 5)
        
        with pytest.raises(AttributeError):
            basic_spatial_state.organization_quality = 0.8

    def test_optimal_representation_validation(self):
        """Test validation of optimal representation coordinates."""
        # Test invalid row coordinate
        with pytest.raises(ValueError, match="Invalid optimal representation row"):
            SpatialOrganizationState(
                optimal_representation=(10, 5),  # Row 10 >= map_dimensions[0] (10)
                structural_coherence=0.5,
                organization_quality=0.5,
                phenomenological_quality=0.5,
                temporal_consistency=0.5,
                map_dimensions=(10, 10)
            )

        # Test invalid column coordinate
        with pytest.raises(ValueError, match="Invalid optimal representation col"):
            SpatialOrganizationState(
                optimal_representation=(5, 10),  # Col 10 >= map_dimensions[1] (10)
                structural_coherence=0.5,
                organization_quality=0.5,
                phenomenological_quality=0.5,
                temporal_consistency=0.5,
                map_dimensions=(10, 10)
            )

    def test_quality_metrics_validation(self):
        """Test validation of quality metrics range [0, 1]."""
        metrics_to_test = [
            ("structural_coherence", -0.1),
            ("structural_coherence", 1.1),
            ("organization_quality", -0.1),
            ("organization_quality", 1.1),
            ("phenomenological_quality", -0.1),
            ("phenomenological_quality", 1.1),
            ("temporal_consistency", -0.1),
            ("temporal_consistency", 1.1)
        ]
        
        for metric_name, invalid_value in metrics_to_test:
            kwargs = {
                "optimal_representation": (5, 5),
                "structural_coherence": 0.5,
                "organization_quality": 0.5,
                "phenomenological_quality": 0.5,
                "temporal_consistency": 0.5,
                "map_dimensions": (10, 10),
                metric_name: invalid_value
            }
            
            with pytest.raises(ValueError, match=f"{metric_name} must be in \\[0, 1\\]"):
                SpatialOrganizationState(**kwargs)

    def test_activation_history_validation(self):
        """Test validation of activation history coordinates."""
        # Test invalid activation history entry
        with pytest.raises(ValueError, match="Invalid activation history entry"):
            SpatialOrganizationState(
                optimal_representation=(5, 5),
                structural_coherence=0.5,
                organization_quality=0.5,
                phenomenological_quality=0.5,
                temporal_consistency=0.5,
                map_dimensions=(10, 10),
                activation_history=[(5, 5), (10, 5)]  # (10, 5) is out of bounds
            )

    def test_is_well_organized_property(self, basic_spatial_state):
        """Test is_well_organized property."""
        # Basic state has organization_quality = 0.5, should not be well organized
        assert not basic_spatial_state.is_well_organized
        
        # Create well organized state
        well_organized_state = basic_spatial_state.with_updated_quality_metrics(
            organization_quality=0.8
        )
        assert well_organized_state.is_well_organized

    def test_spatial_stability_calculation(self, basic_spatial_state):
        """Test spatial stability calculation from activation history."""
        stability = basic_spatial_state.spatial_stability
        assert 0.0 <= stability <= 1.0
        
        # Test with stable history (same position repeated)
        stable_state = basic_spatial_state.with_updated_position((5, 5))
        stable_state = stable_state.with_updated_position((5, 5))
        stable_state = stable_state.with_updated_position((5, 5))
        
        # Should have high stability due to repeated same position
        assert stable_state.spatial_stability > basic_spatial_state.spatial_stability

    def test_consciousness_contribution_calculation(self, basic_spatial_state):
        """Test consciousness contribution calculation."""
        contribution = basic_spatial_state.consciousness_contribution
        assert 0.0 <= contribution <= 1.0
        
        # Test with better quality metrics
        enhanced_state = basic_spatial_state.with_updated_quality_metrics(
            structural_coherence=0.9,
            organization_quality=0.8,
            phenomenological_quality=0.7
        )
        enhanced_contribution = enhanced_state.consciousness_contribution
        
        assert enhanced_contribution > contribution

    def test_representational_precision_calculation(self, basic_spatial_state):
        """Test representational precision calculation."""
        precision = basic_spatial_state.representational_precision
        assert 0.0 <= precision <= 1.0
        
        # Test with lower quantization error (should increase precision)
        precise_state = basic_spatial_state.with_updated_quality_metrics(
            quantization_error=0.1
        )
        precise_precision = precise_state.representational_precision
        
        assert precise_precision > precision

    def test_with_updated_position(self, basic_spatial_state):
        """Test creating new state with updated position."""
        new_position = (7, 8)
        updated_state = basic_spatial_state.with_updated_position(new_position)
        
        # Check that position is updated
        assert updated_state.optimal_representation == new_position
        
        # Check that activation history is updated
        assert new_position in updated_state.activation_history
        assert len(updated_state.activation_history) == len(basic_spatial_state.activation_history) + 1
        
        # Check immutability of original state
        assert basic_spatial_state.optimal_representation == (3, 4)
        assert len(basic_spatial_state.activation_history) == 3

    def test_with_updated_quality_metrics(self, basic_spatial_state):
        """Test creating new state with updated quality metrics."""
        updated_state = basic_spatial_state.with_updated_quality_metrics(
            structural_coherence=0.8,
            organization_quality=0.7,
            phenomenological_quality=0.6,
            quantization_error=0.2
        )
        
        # Check that metrics are updated
        assert updated_state.structural_coherence == 0.8
        assert updated_state.organization_quality == 0.7
        assert updated_state.phenomenological_quality == 0.6
        assert updated_state.quantization_error == 0.2
        
        # Check that other properties are preserved
        assert updated_state.optimal_representation == basic_spatial_state.optimal_representation
        assert updated_state.map_dimensions == basic_spatial_state.map_dimensions
        
        # Check immutability of original state
        assert basic_spatial_state.structural_coherence == 0.6
        assert basic_spatial_state.organization_quality == 0.5

    def test_activation_history_truncation(self, basic_spatial_state):
        """Test that activation history is properly truncated."""
        state = basic_spatial_state
        
        # Add many positions to exceed the 20-position limit
        for i in range(25):
            new_pos = (min(i % 10, 9), min((i + 1) % 10, 9))
            state = state.with_updated_position(new_pos)
        
        # Should not exceed 20 positions in history
        assert len(state.activation_history) <= 20

    def test_temporal_consistency_update(self, basic_spatial_state):
        """Test temporal consistency calculation with position updates."""
        # Add positions that are close together (should increase consistency)
        state = basic_spatial_state.with_updated_position((3, 5))  # Close to (3, 4)
        state = state.with_updated_position((3, 4))  # Back to nearby position
        
        # Should have reasonable temporal consistency
        assert 0.0 <= state.temporal_consistency <= 1.0

    def test_add_metadata(self, basic_spatial_state):
        """Test adding metadata to spatial organization state."""
        updated_state = basic_spatial_state.add_metadata("test_key", "test_value")
        
        # Check that metadata was added
        assert updated_state.metadata["test_key"] == "test_value"
        
        # Check immutability
        assert "test_key" not in basic_spatial_state.metadata

    def test_to_dict_conversion(self, basic_spatial_state):
        """Test conversion to dictionary format."""
        state_dict = basic_spatial_state.to_dict()
        
        # Check required keys are present
        required_keys = [
            "optimal_representation", "structural_coherence", "organization_quality",
            "phenomenological_quality", "temporal_consistency", "map_dimensions",
            "quantization_error", "activation_history", "timestamp", "metadata",
            "is_well_organized", "spatial_stability", "consciousness_contribution",
            "representational_precision"
        ]
        
        for key in required_keys:
            assert key in state_dict, f"Missing key: {key}"
        
        # Check data types
        assert isinstance(state_dict["optimal_representation"], tuple)
        assert isinstance(state_dict["is_well_organized"], bool)
        assert isinstance(state_dict["consciousness_contribution"], float)
        assert isinstance(state_dict["timestamp"], str)  # ISO format

    def test_create_initial_factory(self):
        """Test factory method for initial spatial organization state."""
        initial_state = SpatialOrganizationState.create_initial((8, 8))
        
        assert initial_state.map_dimensions == (8, 8)
        assert initial_state.optimal_representation == (4, 4)  # Center position
        assert initial_state.organization_quality == 0.1  # Low initial organization
        assert initial_state.structural_coherence == 0.5  # Neutral coherence
        assert len(initial_state.activation_history) == 1
        assert "initialized" in initial_state.metadata

    def test_create_well_organized_factory(self):
        """Test factory method for well-organized spatial state."""
        well_organized_state = SpatialOrganizationState.create_well_organized((12, 12))
        
        assert well_organized_state.map_dimensions == (12, 12)
        assert well_organized_state.is_well_organized
        assert well_organized_state.structural_coherence >= 0.8
        assert well_organized_state.organization_quality >= 0.7
        assert well_organized_state.phenomenological_quality >= 0.7
        assert "well_organized" in well_organized_state.metadata

    def test_spatial_organization_equality(self, basic_spatial_state):
        """Test value-based equality of spatial organization states."""
        # Create identical state
        identical_state = SpatialOrganizationState(
            optimal_representation=basic_spatial_state.optimal_representation,
            structural_coherence=basic_spatial_state.structural_coherence,
            organization_quality=basic_spatial_state.organization_quality,
            phenomenological_quality=basic_spatial_state.phenomenological_quality,
            temporal_consistency=basic_spatial_state.temporal_consistency,
            map_dimensions=basic_spatial_state.map_dimensions,
            quantization_error=basic_spatial_state.quantization_error,
            activation_history=basic_spatial_state.activation_history,
            timestamp=basic_spatial_state.timestamp,
            metadata=basic_spatial_state.metadata
        )
        
        # States should be equal based on values (frozen dataclass behavior)
        assert basic_spatial_state == identical_state

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test minimum map size
        min_state = SpatialOrganizationState.create_initial((1, 1))
        assert min_state.optimal_representation == (0, 0)
        assert min_state.spatial_stability >= 0.0
        
        # Test with empty activation history (should handle gracefully)
        empty_history_state = SpatialOrganizationState(
            optimal_representation=(0, 0),
            structural_coherence=0.5,
            organization_quality=0.5,
            phenomenological_quality=0.5,
            temporal_consistency=0.0,
            map_dimensions=(1, 1),
            activation_history=[]
        )
        assert empty_history_state.spatial_stability == 0.0