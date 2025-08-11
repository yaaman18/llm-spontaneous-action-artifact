"""
Unit tests for PhiValue value object.

Comprehensive TDD test suite covering integrated information (Φ) value
calculations, validations, and edge cases. Tests follow the AAA pattern
and include property-based testing for mathematical properties.
"""

import pytest
import math
from hypothesis import given, assume, strategies as st

from domain.value_objects.phi_value import PhiValue


class TestPhiValueCreation:
    """Test suite for PhiValue creation and validation."""
    
    def test_valid_phi_value_creation(self):
        """Test creating valid PhiValue instance."""
        # Arrange
        value, complexity, integration = 0.5, 1.0, 0.5
        system_size = 3
        
        # Act
        phi = PhiValue(
            value=value,
            complexity=complexity,
            integration=integration,
            system_size=system_size
        )
        
        # Assert
        assert phi.value == value
        assert phi.complexity == complexity
        assert phi.integration == integration
        assert phi.system_size == system_size
        assert phi.computation_method == "approximate"  # default
        assert phi.confidence == 1.0  # default

    def test_phi_value_with_all_parameters(self):
        """Test PhiValue creation with all parameters specified."""
        # Arrange
        params = {
            'value': 2.5,
            'complexity': 3.0,
            'integration': 0.8,
            'system_size': 5,
            'computation_method': 'exact',
            'confidence': 0.9,
            'metadata': {'test': 'data'}
        }
        
        # Act
        phi = PhiValue(**params)
        
        # Assert
        assert phi.value == params['value']
        assert phi.complexity == params['complexity']
        assert phi.integration == params['integration']
        assert phi.system_size == params['system_size']
        assert phi.computation_method == params['computation_method']
        assert phi.confidence == params['confidence']
        assert phi.metadata == params['metadata']

    def test_negative_phi_value_raises_error(self):
        """Test that negative Φ values are rejected."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Φ value must be non-negative"):
            PhiValue(value=-0.1, complexity=1.0, integration=0.5)

    def test_nan_phi_value_raises_error(self):
        """Test that NaN Φ values are rejected."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Φ value must be finite"):
            PhiValue(value=float('nan'), complexity=1.0, integration=0.5)

    def test_infinite_phi_value_raises_error(self):
        """Test that infinite Φ values are rejected."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Φ value must be finite"):
            PhiValue(value=float('inf'), complexity=1.0, integration=0.5)

    def test_negative_complexity_raises_error(self):
        """Test that negative complexity values are rejected."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Complexity must be non-negative"):
            PhiValue(value=0.5, complexity=-0.1, integration=0.5)

    def test_negative_integration_raises_error(self):
        """Test that negative integration values are rejected."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Integration must be non-negative"):
            PhiValue(value=0.5, complexity=1.0, integration=-0.1)

    def test_zero_system_size_raises_error(self):
        """Test that zero system size is rejected."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="System size must be positive"):
            PhiValue(value=0.5, complexity=1.0, integration=0.5, system_size=0)

    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence values are rejected."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Confidence must be in \\[0, 1\\]"):
            PhiValue(value=0.5, complexity=1.0, integration=0.5, confidence=1.5)
        
        with pytest.raises(ValueError, match="Confidence must be in \\[0, 1\\]"):
            PhiValue(value=0.5, complexity=1.0, integration=0.5, confidence=-0.1)

    def test_invalid_computation_method_raises_error(self):
        """Test that invalid computation methods are rejected."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Invalid computation method"):
            PhiValue(
                value=0.5, complexity=1.0, integration=0.5, 
                computation_method="invalid_method"
            )


class TestPhiValueProperties:
    """Test suite for PhiValue computed properties."""
    
    def test_is_conscious_with_positive_phi(self):
        """Test consciousness detection with positive Φ."""
        # Arrange
        phi = PhiValue(value=0.1, complexity=1.0, integration=0.1)
        
        # Act & Assert
        assert phi.is_conscious is True

    def test_is_conscious_with_zero_phi(self):
        """Test consciousness detection with zero Φ."""
        # Arrange
        phi = PhiValue(value=0.0, complexity=0.0, integration=0.0)
        
        # Act & Assert
        assert phi.is_conscious is False

    def test_normalized_value_calculation(self):
        """Test normalized Φ value calculation."""
        # Arrange
        phi = PhiValue(value=3.0, complexity=1.0, integration=1.0, system_size=3)
        
        # Act
        normalized = phi.normalized_value
        
        # Assert
        assert normalized == 1.0  # 3.0 / 3

    def test_consciousness_level_categories(self):
        """Test consciousness level categorization."""
        # Arrange & Act & Assert
        test_cases = [
            (0.0, "unconscious"),
            (0.05, "minimal"),
            (0.5, "moderate"),
            (2.0, "high"),
            (10.0, "very_high")
        ]
        
        for value, expected_level in test_cases:
            phi = PhiValue(value=value, complexity=1.0, integration=1.0)
            assert phi.consciousness_level == expected_level

    def test_integration_complexity_ratio(self):
        """Test integration to complexity ratio calculation."""
        # Arrange
        phi = PhiValue(value=1.0, complexity=2.0, integration=1.0)
        
        # Act
        ratio = phi.integration_complexity_ratio
        
        # Assert
        assert ratio == 0.5  # 1.0 / 2.0

    def test_integration_complexity_ratio_zero_complexity(self):
        """Test ratio calculation with zero complexity."""
        # Arrange
        phi = PhiValue(value=0.0, complexity=0.0, integration=0.0)
        
        # Act
        ratio = phi.integration_complexity_ratio
        
        # Assert
        assert ratio == 0.0

    def test_theoretical_maximum_calculation(self):
        """Test theoretical maximum Φ calculation."""
        # Arrange
        phi = PhiValue(value=1.0, complexity=1.0, integration=1.0, system_size=4)
        
        # Act
        theoretical_max = phi.theoretical_maximum
        
        # Assert
        expected = 4 * math.log2(4)  # system_size * log2(system_size)
        assert theoretical_max == expected

    def test_theoretical_maximum_single_element(self):
        """Test theoretical maximum for single element system."""
        # Arrange
        phi = PhiValue(value=0.5, complexity=1.0, integration=1.0, system_size=1)
        
        # Act
        theoretical_max = phi.theoretical_maximum
        
        # Assert
        assert theoretical_max == 1.0

    def test_efficiency_calculation(self):
        """Test integration efficiency calculation."""
        # Arrange
        system_size = 4
        theoretical_max = system_size * math.log2(system_size)
        phi_value = theoretical_max * 0.5  # 50% efficient
        
        phi = PhiValue(
            value=phi_value, complexity=1.0, integration=1.0, 
            system_size=system_size
        )
        
        # Act
        efficiency = phi.efficiency
        
        # Assert
        assert abs(efficiency - 0.5) < 1e-10  # Allow for floating point precision


class TestPhiValueComparison:
    """Test suite for PhiValue comparison operations."""
    
    def test_compare_with_higher_phi(self):
        """Test comparing with higher Φ value."""
        # Arrange
        phi1 = PhiValue(value=1.0, complexity=1.0, integration=1.0)
        phi2 = PhiValue(value=2.0, complexity=1.5, integration=1.2)
        
        # Act
        comparison = phi1.compare_with(phi2)
        
        # Assert
        assert comparison['phi_difference'] == -1.0  # 1.0 - 2.0
        assert comparison['phi_ratio'] == 0.5  # 1.0 / 2.0
        assert comparison['complexity_difference'] == -0.5
        assert comparison['integration_difference'] == -0.2

    def test_compare_with_zero_phi(self):
        """Test comparing with zero Φ value (edge case)."""
        # Arrange
        phi1 = PhiValue(value=1.0, complexity=1.0, integration=1.0)
        phi2 = PhiValue(value=0.0, complexity=0.0, integration=0.0)
        
        # Act
        comparison = phi1.compare_with(phi2)
        
        # Assert
        assert comparison['phi_difference'] == 1.0
        assert comparison['phi_ratio'] == 1.0 / 1e-10  # Uses small epsilon to avoid division by zero


class TestPhiValueImmutability:
    """Test suite for PhiValue immutability and value object behavior."""
    
    def test_phi_value_is_immutable(self):
        """Test that PhiValue instances are immutable."""
        # Arrange
        phi = PhiValue(value=1.0, complexity=1.0, integration=1.0)
        
        # Act & Assert
        with pytest.raises(AttributeError):
            phi.value = 2.0

    def test_with_updated_value_creates_new_instance(self):
        """Test that updating value creates new instance."""
        # Arrange
        original_phi = PhiValue(value=1.0, complexity=1.0, integration=1.0)
        
        # Act
        updated_phi = original_phi.with_updated_value(2.0)
        
        # Assert
        assert updated_phi.value == 2.0
        assert original_phi.value == 1.0  # Original unchanged
        assert updated_phi is not original_phi  # Different instances
        assert updated_phi.complexity == original_phi.complexity  # Other properties preserved

    def test_with_updated_components_recalculates_phi(self):
        """Test that updating components recalculates Φ."""
        # Arrange
        original_phi = PhiValue(value=1.0, complexity=2.0, integration=0.5)
        
        # Act
        updated_phi = original_phi.with_updated_components(
            new_complexity=4.0, new_integration=0.8
        )
        
        # Assert
        assert updated_phi.complexity == 4.0
        assert updated_phi.integration == 0.8
        assert updated_phi.value == 4.0 * 0.8  # complexity * integration (simplified)
        assert original_phi.complexity == 2.0  # Original unchanged

    def test_add_metadata_creates_new_instance(self):
        """Test that adding metadata creates new instance."""
        # Arrange
        original_phi = PhiValue(value=1.0, complexity=1.0, integration=1.0)
        
        # Act
        updated_phi = original_phi.add_metadata("test_key", "test_value")
        
        # Assert
        assert updated_phi.metadata["test_key"] == "test_value"
        assert "test_key" not in original_phi.metadata
        assert updated_phi is not original_phi


class TestPhiValueSerialization:
    """Test suite for PhiValue serialization and factory methods."""
    
    def test_to_dict_includes_all_properties(self):
        """Test dictionary conversion includes all properties."""
        # Arrange
        phi = PhiValue(
            value=1.5, complexity=2.0, integration=0.75, 
            system_size=4, computation_method="exact", confidence=0.9
        )
        
        # Act
        phi_dict = phi.to_dict()
        
        # Assert
        required_keys = [
            'value', 'complexity', 'integration', 'system_size',
            'computation_method', 'confidence', 'metadata',
            'is_conscious', 'normalized_value', 'consciousness_level',
            'integration_complexity_ratio', 'theoretical_maximum', 'efficiency'
        ]
        
        for key in required_keys:
            assert key in phi_dict

    def test_create_zero_factory(self):
        """Test zero Φ factory method."""
        # Act
        zero_phi = PhiValue.create_zero()
        
        # Assert
        assert zero_phi.value == 0.0
        assert zero_phi.complexity == 0.0
        assert zero_phi.integration == 0.0
        assert not zero_phi.is_conscious
        assert zero_phi.consciousness_level == "unconscious"

    def test_create_minimal_factory(self):
        """Test minimal consciousness factory method."""
        # Act
        minimal_phi = PhiValue.create_minimal(system_size=5)
        
        # Assert
        assert minimal_phi.value == 0.1
        assert minimal_phi.complexity == 0.2
        assert minimal_phi.integration == 0.5
        assert minimal_phi.system_size == 5
        assert minimal_phi.is_conscious
        assert minimal_phi.consciousness_level == "minimal"


class TestPhiValuePropertyBased:
    """Property-based tests for PhiValue mathematical properties."""
    
    @given(
        value=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        complexity=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        integration=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        system_size=st.integers(min_value=1, max_value=100)
    )
    def test_phi_value_properties_invariants(self, value, complexity, integration, system_size):
        """Test PhiValue invariants hold for all valid inputs."""
        # Assume valid inputs (hypothesis will generate edge cases)
        assume(not math.isnan(value) and not math.isinf(value))
        assume(not math.isnan(complexity) and not math.isinf(complexity))
        assume(not math.isnan(integration) and not math.isinf(integration))
        
        # Arrange & Act
        phi = PhiValue(
            value=value, complexity=complexity, integration=integration,
            system_size=system_size
        )
        
        # Assert invariants
        assert phi.value >= 0.0  # Φ is always non-negative
        assert phi.complexity >= 0.0  # Complexity is always non-negative
        assert phi.integration >= 0.0  # Integration is always non-negative
        assert phi.system_size >= 1  # System size is always positive
        assert 0.0 <= phi.efficiency <= 1.0  # Efficiency is in [0, 1]
        assert phi.normalized_value >= 0.0  # Normalized value is non-negative
        
        # Consciousness properties
        if phi.value > 0.0:
            assert phi.is_conscious
        else:
            assert not phi.is_conscious

    @given(
        phi_value=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    def test_consciousness_level_consistency(self, phi_value):
        """Test consciousness level categorization consistency."""
        # Arrange & Act
        phi = PhiValue(value=phi_value, complexity=1.0, integration=1.0)
        level = phi.consciousness_level
        
        # Assert consistent categorization
        if phi_value <= 0.0:
            assert level == "unconscious"
        elif phi_value <= 0.1:
            assert level == "minimal"
        elif phi_value <= 1.0:
            assert level == "moderate"
        elif phi_value <= 5.0:
            assert level == "high"
        else:
            assert level == "very_high"

    @given(
        complexity=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        integration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    def test_integration_complexity_ratio_bounds(self, complexity, integration):
        """Test integration/complexity ratio properties."""
        # Arrange & Act
        phi = PhiValue(
            value=complexity * integration, 
            complexity=complexity, 
            integration=integration
        )
        
        # Assert
        ratio = phi.integration_complexity_ratio
        assert ratio >= 0.0  # Ratio should be non-negative
        assert abs(ratio - (integration / complexity)) < 1e-10  # Should equal integration/complexity


class TestPhiValueEdgeCases:
    """Test suite for PhiValue edge cases and boundary conditions."""
    
    def test_very_large_phi_value(self):
        """Test handling of very large Φ values."""
        # Arrange
        large_value = 1e6
        phi = PhiValue(value=large_value, complexity=1e3, integration=1e3)
        
        # Act & Assert
        assert phi.is_conscious
        assert phi.consciousness_level == "very_high"
        assert not math.isinf(phi.efficiency)  # Should not overflow

    def test_very_small_nonzero_phi_value(self):
        """Test handling of very small non-zero Φ values."""
        # Arrange
        tiny_value = 1e-10
        phi = PhiValue(value=tiny_value, complexity=1e-5, integration=1e-5)
        
        # Act & Assert
        assert phi.is_conscious  # Should still be conscious
        assert phi.consciousness_level == "minimal"
        assert phi.efficiency >= 0.0

    def test_maximum_system_size(self):
        """Test handling of large system sizes."""
        # Arrange
        large_system = 1000
        phi = PhiValue(
            value=1.0, complexity=1.0, integration=1.0, 
            system_size=large_system
        )
        
        # Act & Assert
        assert not math.isinf(phi.theoretical_maximum)
        assert not math.isnan(phi.theoretical_maximum)
        assert phi.efficiency <= 1.0

    def test_phi_equality_with_floating_point_precision(self):
        """Test PhiValue equality considering floating point precision."""
        # Arrange
        phi1 = PhiValue(value=0.1, complexity=0.3, integration=1.0/3.0)
        phi2 = PhiValue(value=0.1, complexity=0.3, integration=1.0/3.0)
        
        # Act & Assert
        # Should be equal due to frozen dataclass equality
        assert phi1 == phi2
        assert hash(phi1) == hash(phi2)