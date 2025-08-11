"""
Consciousness Domain Factory.

Factory for creating consciousness-related domain objects with proper
initialization, validation, and enactivist principles applied.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from ..aggregates.consciousness_aggregate import ConsciousnessAggregate
from ..value_objects.consciousness_state import ConsciousnessState
from ..value_objects.phi_value import PhiValue
from ..value_objects.prediction_state import PredictionState
from ..value_objects.probability_distribution import ProbabilityDistribution
from ..value_objects.spatial_organization_state import SpatialOrganizationState


class ConsciousnessFactory:
    """
    Factory for creating consciousness domain objects.
    
    Encapsulates the complex logic for creating properly initialized
    consciousness states, aggregates, and related objects based on
    enactivist principles and domain requirements.
    """
    
    def __init__(self):
        """Initialize the consciousness factory."""
        self._creation_history: List[Dict[str, Any]] = []
    
    def create_consciousness_aggregate(
        self,
        system_complexity: int = 10,
        environmental_richness: float = 0.5,
        initial_coupling_strength: float = 0.3,
        consciousness_potential: float = 0.2
    ) -> ConsciousnessAggregate:
        """
        Create consciousness aggregate with enactivist initialization.
        
        Args:
            system_complexity: Complexity of the cognitive system
            environmental_richness: Richness of environmental context
            initial_coupling_strength: Initial environmental coupling
            consciousness_potential: Potential for consciousness emergence
            
        Returns:
            Initialized consciousness aggregate
        """
        # Create initial Φ value based on system characteristics
        initial_phi = self._create_initial_phi_value(
            system_complexity, consciousness_potential
        )
        
        # Create initial prediction state
        initial_prediction = self._create_initial_prediction_state(system_complexity)
        
        # Create initial uncertainty distribution
        initial_uncertainty = self._create_initial_uncertainty_distribution(
            environmental_richness
        )
        
        # Calculate initial metacognitive confidence
        initial_metacognitive = self._calculate_initial_metacognitive_confidence(
            consciousness_potential, environmental_richness
        )
        
        # Create phenomenological markers for enactivist context
        initial_markers = self._create_enactivist_markers(
            initial_coupling_strength, environmental_richness
        )
        
        # Create initial spatial organization
        initial_spatial_organization = SpatialOrganizationState.create_initial()
        
        # Create consciousness state
        initial_state = ConsciousnessState(
            phi_value=initial_phi,
            prediction_state=initial_prediction,
            uncertainty_distribution=initial_uncertainty,
            spatial_organization=initial_spatial_organization,
            metacognitive_confidence=initial_metacognitive,
            phenomenological_markers=initial_markers
        )
        
        # Create aggregate
        aggregate = ConsciousnessAggregate(initial_state=initial_state)
        
        # Record creation
        self._record_creation('consciousness_aggregate', {
            'aggregate_id': aggregate.aggregate_id,
            'system_complexity': system_complexity,
            'environmental_richness': environmental_richness,
            'consciousness_potential': consciousness_potential,
            'initial_phi_value': initial_phi.value
        })
        
        return aggregate
    
    def create_emergent_consciousness_state(
        self,
        environmental_input: np.ndarray,
        prediction_errors: List[float],
        coupling_strength: float,
        attention_context: Optional[Dict[str, Any]] = None
    ) -> ConsciousnessState:
        """
        Create consciousness state for emergence scenario.
        
        This factory method creates consciousness states that emerge from
        environmental interaction patterns, following enactivist principles.
        
        Args:
            environmental_input: Sensory/environmental input data
            prediction_errors: Hierarchical prediction errors
            coupling_strength: Strength of environmental coupling
            attention_context: Context for attention allocation
            
        Returns:
            Consciousness state configured for emergence
        """
        # Calculate Φ value from environmental interaction patterns
        emergence_phi = self._calculate_emergence_phi_value(
            environmental_input, prediction_errors, coupling_strength
        )
        
        # Create prediction state from errors
        prediction_state = self._create_prediction_state_from_errors(prediction_errors)
        
        # Create uncertainty distribution from input characteristics
        uncertainty_dist = self._create_uncertainty_from_input(environmental_input)
        
        # Calculate metacognitive confidence from coupling quality
        metacognitive_confidence = self._calculate_emergence_metacognitive_confidence(
            coupling_strength, prediction_errors
        )
        
        # Generate attention weights from context
        attention_weights = None
        if attention_context:
            attention_weights = self._generate_attention_weights_from_context(
                attention_context, environmental_input.shape
            )
        
        # Create emergence-specific phenomenological markers
        emergence_markers = self._create_emergence_markers(
            environmental_input, coupling_strength, prediction_errors
        )
        
        # Create spatial organization for emergence
        emergence_spatial_organization = SpatialOrganizationState.create_initial()
        
        consciousness_state = ConsciousnessState(
            phi_value=emergence_phi,
            prediction_state=prediction_state,
            uncertainty_distribution=uncertainty_dist,
            spatial_organization=emergence_spatial_organization,
            metacognitive_confidence=metacognitive_confidence,
            attention_weights=attention_weights,
            phenomenological_markers=emergence_markers
        )
        
        # Record creation
        self._record_creation('emergent_consciousness_state', {
            'phi_value': emergence_phi.value,
            'coupling_strength': coupling_strength,
            'prediction_error_total': sum(abs(e) for e in prediction_errors),
            'environmental_complexity': float(np.var(environmental_input))
        })
        
        return consciousness_state
    
    def create_stable_consciousness_state(
        self,
        target_consciousness_level: float,
        prediction_quality: float,
        environmental_coherence: float,
        attention_focus_areas: List[str]
    ) -> ConsciousnessState:
        """
        Create stable consciousness state for sustained awareness.
        
        Args:
            target_consciousness_level: Target level of consciousness [0, 1]
            prediction_quality: Quality of predictive processing [0, 1]
            environmental_coherence: Coherence with environment [0, 1]
            attention_focus_areas: Areas of attentional focus
            
        Returns:
            Stable consciousness state
        """
        # Create Φ value for target consciousness level
        stable_phi = self._create_phi_for_consciousness_level(
            target_consciousness_level, environmental_coherence
        )
        
        # Create high-quality prediction state
        stable_prediction = self._create_stable_prediction_state(prediction_quality)
        
        # Create low-uncertainty distribution for stability
        stable_uncertainty = self._create_stable_uncertainty_distribution()
        
        # High metacognitive confidence for stable consciousness
        stable_metacognitive = min(target_consciousness_level + 0.2, 1.0)
        
        # Create focused attention weights
        stable_attention = self._create_focused_attention_weights(
            len(attention_focus_areas)
        )
        
        # Create stability markers
        stability_markers = self._create_stability_markers(
            target_consciousness_level, attention_focus_areas
        )
        
        # Create well-organized spatial organization for stability
        stable_spatial_organization = SpatialOrganizationState.create_well_organized()
        
        consciousness_state = ConsciousnessState(
            phi_value=stable_phi,
            prediction_state=stable_prediction,
            uncertainty_distribution=stable_uncertainty,
            spatial_organization=stable_spatial_organization,
            metacognitive_confidence=stable_metacognitive,
            attention_weights=stable_attention,
            phenomenological_markers=stability_markers
        )
        
        # Record creation
        self._record_creation('stable_consciousness_state', {
            'target_level': target_consciousness_level,
            'prediction_quality': prediction_quality,
            'environmental_coherence': environmental_coherence,
            'attention_areas': len(attention_focus_areas)
        })
        
        return consciousness_state
    
    def create_minimal_consciousness_state(self) -> ConsciousnessState:
        """
        Create minimal consciousness state for initialization or testing.
        
        Returns:
            Minimal consciousness state
        """
        return ConsciousnessState.create_minimal_consciousness()
    
    def _create_initial_phi_value(
        self,
        system_complexity: int,
        consciousness_potential: float
    ) -> PhiValue:
        """Create initial Φ value based on system characteristics."""
        # Base Φ on consciousness potential
        base_phi = consciousness_potential * 2.0  # Scale to reasonable range
        
        # Complexity and integration components
        complexity = system_complexity * 0.1
        integration = consciousness_potential * complexity * 0.8
        
        return PhiValue(
            value=base_phi,
            complexity=complexity,
            integration=integration,
            system_size=system_complexity,
            computation_method="enactivist_initialization",
            confidence=0.7
        )
    
    def _create_initial_prediction_state(self, system_complexity: int) -> PredictionState:
        """Create initial prediction state."""
        # Determine hierarchy levels based on system complexity
        hierarchy_levels = min(max(system_complexity // 3, 2), 5)
        
        # Initial errors decrease with hierarchy level
        initial_errors = [1.0 / (level + 1) for level in range(hierarchy_levels)]
        
        return PredictionState(
            hierarchical_errors=initial_errors,
            timestamp=datetime.now(),
            convergence_status="not_converged",
            learning_iteration=0,
            metadata={'factory_created': True, 'system_complexity': system_complexity}
        )
    
    def _create_initial_uncertainty_distribution(
        self,
        environmental_richness: float
    ) -> ProbabilityDistribution:
        """Create initial uncertainty distribution."""
        # More environmental richness -> higher initial uncertainty
        distribution_size = int(10 + environmental_richness * 20)
        
        if environmental_richness < 0.3:
            # Low richness -> uniform distribution
            return ProbabilityDistribution.uniform(distribution_size)
        else:
            # Higher richness -> more complex distribution
            # Create a peaked distribution with some spread
            probabilities = np.exp(-np.arange(distribution_size) * 0.2)
            probabilities = probabilities / probabilities.sum()
            return ProbabilityDistribution(probabilities)
    
    def _calculate_initial_metacognitive_confidence(
        self,
        consciousness_potential: float,
        environmental_richness: float
    ) -> float:
        """Calculate initial metacognitive confidence."""
        # Higher consciousness potential and environmental richness
        # lead to higher initial metacognitive confidence
        base_confidence = consciousness_potential * 0.5
        environmental_boost = environmental_richness * 0.3
        
        return min(base_confidence + environmental_boost, 0.8)
    
    def _create_enactivist_markers(
        self,
        coupling_strength: float,
        environmental_richness: float
    ) -> Dict[str, Any]:
        """Create phenomenological markers for enactivist context."""
        return {
            'enactivist_initialization': True,
            'environmental_coupling': coupling_strength,
            'environmental_richness': environmental_richness,
            'structural_coupling_active': coupling_strength > 0.2,
            'sensorimotor_contingencies': environmental_richness > 0.4,
            'initialization_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_emergence_phi_value(
        self,
        environmental_input: np.ndarray,
        prediction_errors: List[float],
        coupling_strength: float
    ) -> PhiValue:
        """Calculate Φ value for consciousness emergence."""
        # Φ emerges from interaction between system and environment
        input_complexity = float(np.var(environmental_input))
        error_coherence = 1.0 / (1.0 + sum(abs(e) for e in prediction_errors))
        
        # Emergence Φ based on enactivist principles
        emergence_phi = coupling_strength * error_coherence * min(input_complexity, 2.0)
        complexity = input_complexity * 0.5
        integration = coupling_strength * error_coherence
        
        return PhiValue(
            value=emergence_phi,
            complexity=complexity,
            integration=integration,
            system_size=len(prediction_errors),
            computation_method="enactivist_initialization",
            confidence=coupling_strength
        )
    
    def _create_prediction_state_from_errors(
        self,
        prediction_errors: List[float]
    ) -> PredictionState:
        """Create prediction state from error values."""
        return PredictionState(
            hierarchical_errors=prediction_errors,
            timestamp=datetime.now(),
            convergence_status="converging" if sum(abs(e) for e in prediction_errors) < 2.0 else "not_converged",
            learning_iteration=0,
            metadata={'emergence_created': True}
        )
    
    def _create_uncertainty_from_input(
        self,
        environmental_input: np.ndarray
    ) -> ProbabilityDistribution:
        """Create uncertainty distribution from environmental input."""
        input_variance = float(np.var(environmental_input))
        input_entropy = float(-np.sum(np.histogram(environmental_input, bins=10, density=True)[0] * 
                                    np.log(np.histogram(environmental_input, bins=10, density=True)[0] + 1e-10)))
        
        # Create distribution reflecting input characteristics
        distribution_size = min(max(int(input_entropy * 5), 5), 20)
        
        # Use input variance to shape distribution
        if input_variance > 1.0:
            # High variance -> more uniform distribution
            return ProbabilityDistribution.uniform(distribution_size)
        else:
            # Low variance -> more peaked distribution
            probabilities = np.exp(-np.arange(distribution_size) * (2.0 - input_variance))
            probabilities = probabilities / probabilities.sum()
            return ProbabilityDistribution(probabilities)
    
    def _calculate_emergence_metacognitive_confidence(
        self,
        coupling_strength: float,
        prediction_errors: List[float]
    ) -> float:
        """Calculate metacognitive confidence for emergence."""
        error_quality = 1.0 / (1.0 + sum(abs(e) for e in prediction_errors))
        return min(coupling_strength * error_quality * 0.8, 0.9)
    
    def _generate_attention_weights_from_context(
        self,
        attention_context: Dict[str, Any],
        input_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Generate attention weights from context."""
        # Determine attention dimensions
        attention_dims = attention_context.get('attention_dimensions', input_shape[0] if input_shape else 5)
        
        # Get attention biases from context
        focus_areas = attention_context.get('focus_areas', [])
        environmental_salience = attention_context.get('environmental_salience', 0.5)
        
        # Create base weights
        weights = np.ones(attention_dims) / attention_dims
        
        # Apply focus biases
        if focus_areas:
            for area_idx in focus_areas:
                if 0 <= area_idx < attention_dims:
                    weights[area_idx] *= 2.0
        
        # Apply environmental salience (bias toward early indices)
        environmental_bias = np.exp(-np.arange(attention_dims) * 0.2)
        weights = weights * (1.0 - environmental_salience) + environmental_bias * environmental_salience
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    def _create_emergence_markers(
        self,
        environmental_input: np.ndarray,
        coupling_strength: float,
        prediction_errors: List[float]
    ) -> Dict[str, Any]:
        """Create markers for consciousness emergence."""
        return {
            'emergence_event': True,
            'environmental_coupling': coupling_strength,
            'input_complexity': float(np.var(environmental_input)),
            'prediction_error_total': sum(abs(e) for e in prediction_errors),
            'emergence_quality': coupling_strength * (1.0 / (1.0 + sum(abs(e) for e in prediction_errors))),
            'emergence_timestamp': datetime.now().isoformat(),
            'enactivist_initialization': True
        }
    
    def _create_phi_for_consciousness_level(
        self,
        target_level: float,
        environmental_coherence: float
    ) -> PhiValue:
        """Create Φ value for target consciousness level."""
        # Map consciousness level to Φ value
        phi_value = target_level * 5.0  # Scale to reasonable Φ range
        
        complexity = target_level * 3.0
        integration = target_level * environmental_coherence * 2.0
        
        return PhiValue(
            value=phi_value,
            complexity=complexity,
            integration=integration,
            system_size=int(target_level * 10 + 5),
            computation_method="target_level_creation",
            confidence=min(target_level + 0.3, 1.0)
        )
    
    def _create_stable_prediction_state(self, prediction_quality: float) -> PredictionState:
        """Create stable prediction state."""
        # High quality -> low errors
        hierarchy_levels = 4
        base_error = (1.0 - prediction_quality) * 0.5
        errors = [base_error * (1.0 + level * 0.1) for level in range(hierarchy_levels)]
        
        return PredictionState(
            hierarchical_errors=errors,
            timestamp=datetime.now(),
            convergence_status="converged" if prediction_quality > 0.8 else "converging",
            learning_iteration=100,  # Mature state
            metadata={'stable_creation': True, 'prediction_quality': prediction_quality}
        )
    
    def _create_stable_uncertainty_distribution(self) -> ProbabilityDistribution:
        """Create stable (low entropy) uncertainty distribution."""
        # Peaked distribution for stability
        probabilities = np.exp(-np.arange(10) * 0.5)
        probabilities = probabilities / probabilities.sum()
        return ProbabilityDistribution(probabilities.tolist())
    
    def _create_focused_attention_weights(self, num_areas: int) -> np.ndarray:
        """Create focused attention weight distribution."""
        weights = np.zeros(max(num_areas, 3))
        
        # Focus on primary area
        weights[0] = 0.6
        
        # Distribute rest
        if len(weights) > 1:
            remaining = 0.4 / (len(weights) - 1)
            weights[1:] = remaining
        
        return weights
    
    def _create_stability_markers(
        self,
        target_level: float,
        attention_areas: List[str]
    ) -> Dict[str, Any]:
        """Create markers for stable consciousness."""
        return {
            'stable_consciousness': True,
            'target_consciousness_level': target_level,
            'attention_focus_areas': attention_areas,
            'stability_created': True,
            'creation_timestamp': datetime.now().isoformat()
        }
    
    def _record_creation(self, object_type: str, parameters: Dict[str, Any]) -> None:
        """Record object creation in factory history."""
        creation_record = {
            'object_type': object_type,
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters
        }
        self._creation_history.append(creation_record)
    
    def get_creation_history(self) -> List[Dict[str, Any]]:
        """Get factory creation history."""
        return self._creation_history.copy()