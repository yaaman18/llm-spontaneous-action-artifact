"""
General domain object factory for the enactive consciousness framework.

This factory provides a unified interface for creating various domain objects
with proper validation and initialization.
"""

from typing import Dict, Any, Optional, Type, TypeVar
from ..aggregates.consciousness_aggregate import ConsciousnessAggregate
from ..aggregates.learning_aggregate import LearningAggregate
from ..value_objects.consciousness_state import ConsciousnessState
from ..value_objects.phi_value import PhiValue
from .consciousness_factory import ConsciousnessFactory
from .learning_factory import LearningFactory

T = TypeVar('T')


class DomainObjectFactory:
    """
    Unified factory for creating domain objects.
    
    This factory provides a single entry point for creating various
    domain objects with consistent initialization and validation.
    """
    
    @staticmethod
    def create_minimal_system() -> Dict[str, Any]:
        """
        Create a minimal working consciousness system.
        
        Returns:
            Dictionary containing initialized system components
        """
        consciousness_factory = ConsciousnessFactory()
        learning_factory = LearningFactory()
        
        # Create basic components
        phi_value = PhiValue(value=0.3, complexity=0.8, integration=0.6)
        consciousness_state = ConsciousnessState(
            phi_value=phi_value,
            attention_focus=0.7,
            metacognitive_confidence=0.5,
            environmental_coupling_strength=0.6
        )
        
        # Create aggregates
        consciousness_aggregate = consciousness_factory.create_consciousness_system(
            initial_phi=0.3,
            environmental_coupling=0.6
        )
        
        learning_aggregate = learning_factory.create_learning_aggregate(
            environmental_coupling_strength=0.6
        )
        
        return {
            'consciousness_state': consciousness_state,
            'consciousness_aggregate': consciousness_aggregate,
            'learning_aggregate': learning_aggregate,
            'phi_value': phi_value
        }
    
    @staticmethod
    def create_demo_system() -> Dict[str, Any]:
        """
        Create a demonstration system with realistic parameters.
        
        Returns:
            Dictionary containing demo system components
        """
        consciousness_factory = ConsciousnessFactory()
        learning_factory = LearningFactory()
        
        # Create demo components with realistic values
        phi_value = PhiValue(value=0.75, complexity=0.9, integration=0.8)
        consciousness_state = ConsciousnessState(
            phi_value=phi_value,
            attention_focus=0.8,
            metacognitive_confidence=0.7,
            environmental_coupling_strength=0.8
        )
        
        # Create aggregates with demo parameters
        consciousness_aggregate = consciousness_factory.create_consciousness_system(
            initial_phi=0.75,
            environmental_coupling=0.8
        )
        
        learning_params = learning_factory.create_learning_parameters(
            learning_rate=0.001,
            momentum=0.95,
            adaptation_rate=0.0001
        )
        
        learning_aggregate = learning_factory.create_learning_aggregate(
            learning_params=learning_params,
            environmental_coupling_strength=0.8
        )
        
        return {
            'consciousness_state': consciousness_state,
            'consciousness_aggregate': consciousness_aggregate,
            'learning_aggregate': learning_aggregate,
            'phi_value': phi_value,
            'learning_parameters': learning_params
        }
    
    @staticmethod
    def validate_system_consistency(system_components: Dict[str, Any]) -> bool:
        """
        Validate that system components are consistent with each other.
        
        Args:
            system_components: Dictionary of system components
            
        Returns:
            True if components are consistent, False otherwise
        """
        try:
            # Check that consciousness state exists and is valid
            if 'consciousness_state' not in system_components:
                return False
                
            consciousness_state = system_components['consciousness_state']
            if not isinstance(consciousness_state, ConsciousnessState):
                return False
                
            # Check that phi value is consistent
            if 'phi_value' in system_components:
                phi_value = system_components['phi_value']
                if not isinstance(phi_value, PhiValue):
                    return False
                    
                # Phi values should match between state and standalone
                if abs(consciousness_state.phi_value.value - phi_value.value) > 1e-6:
                    return False
            
            # Additional consistency checks can be added here
            return True
            
        except Exception:
            return False