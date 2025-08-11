"""
Consciousness Domain Specifications.

Specifications that encapsulate complex business rules for consciousness
emergence, stability, and attentional coherence using the Specification pattern.
These specifications embody the ubiquitous language of consciousness research
and enactivism.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy.typing as npt

from ..value_objects.consciousness_state import ConsciousnessState
from ..value_objects.phi_value import PhiValue


class ConsciousnessSpecification(ABC):
    """
    Abstract base class for consciousness-related specifications.
    
    Implements the Specification pattern for encapsulating complex
    business rules about consciousness states and transitions.
    """
    
    @abstractmethod
    def is_satisfied_by(self, candidate: Any) -> bool:
        """
        Check if the candidate satisfies this specification.
        
        Args:
            candidate: Object to check against specification
            
        Returns:
            True if candidate satisfies the specification
        """
        pass
    
    def and_specification(self, other: 'ConsciousnessSpecification') -> 'AndConsciousnessSpecification':
        """Combine specifications with logical AND."""
        return AndConsciousnessSpecification(self, other)
    
    def or_specification(self, other: 'ConsciousnessSpecification') -> 'OrConsciousnessSpecification':
        """Combine specifications with logical OR."""
        return OrConsciousnessSpecification(self, other)
    
    def not_specification(self) -> 'NotConsciousnessSpecification':
        """Negate this specification."""
        return NotConsciousnessSpecification(self)


class ConsciousnessEmergenceSpecification(ConsciousnessSpecification):
    """
    Specification for consciousness emergence criteria.
    
    Defines the complex conditions under which consciousness can be
    considered to have emerged in the enactive system, based on
    Integrated Information Theory and enactivist principles.
    """
    
    def __init__(
        self,
        min_phi_value: float = 0.1,
        min_metacognitive_confidence: float = 0.15,
        max_prediction_error: float = 2.0,
        min_integration_complexity_ratio: float = 0.3,
        require_environmental_coupling: bool = True
    ):
        """
        Initialize consciousness emergence specification.
        
        Args:
            min_phi_value: Minimum Φ value for consciousness
            min_metacognitive_confidence: Minimum metacognitive awareness
            max_prediction_error: Maximum allowable prediction error
            min_integration_complexity_ratio: Minimum integration/complexity ratio
            require_environmental_coupling: Whether environmental coupling is required
        """
        self.min_phi_value = min_phi_value
        self.min_metacognitive_confidence = min_metacognitive_confidence
        self.max_prediction_error = max_prediction_error
        self.min_integration_complexity_ratio = min_integration_complexity_ratio
        self.require_environmental_coupling = require_environmental_coupling
    
    def is_satisfied_by(self, consciousness_state: ConsciousnessState) -> bool:
        """
        Check if consciousness state satisfies emergence criteria.
        
        Based on multiple convergent indicators from IIT, enactivism,
        and predictive processing theories.
        
        Args:
            consciousness_state: Consciousness state to evaluate
            
        Returns:
            True if consciousness emergence criteria are met
        """
        # Criterion 1: Sufficient integrated information (Φ > threshold)
        if consciousness_state.phi_value.value < self.min_phi_value:
            return False
        
        # Criterion 2: Metacognitive awareness above threshold
        if consciousness_state.metacognitive_confidence < self.min_metacognitive_confidence:
            return False
        
        # Criterion 3: Prediction quality indicates coherent internal model
        if consciousness_state.prediction_state.total_error > self.max_prediction_error:
            return False
        
        # Criterion 4: Integration-complexity balance for genuine consciousness
        integration_ratio = consciousness_state.phi_value.integration_complexity_ratio
        if integration_ratio < self.min_integration_complexity_ratio:
            return False
        
        # Criterion 5: Environmental coupling (enactivist requirement)
        if self.require_environmental_coupling:
            # Check for phenomenological markers indicating environmental interaction
            coupling_markers = consciousness_state.phenomenological_markers.get('environmental_coupling', 0)
            if coupling_markers < 0.1:
                return False
        
        # Criterion 6: Temporal coherence - predictions must be stable
        if not consciousness_state.prediction_state.is_stable:
            return False
        
        # All criteria satisfied
        return True
    
    def get_emergence_score(self, consciousness_state: ConsciousnessState) -> float:
        """
        Calculate a continuous emergence score [0, 1] for the consciousness state.
        
        Args:
            consciousness_state: Consciousness state to score
            
        Returns:
            Emergence score indicating how close to satisfying emergence criteria
        """
        scores = []
        
        # Phi value score
        phi_score = min(consciousness_state.phi_value.value / self.min_phi_value, 1.0)
        scores.append(phi_score)
        
        # Metacognitive score
        meta_score = min(
            consciousness_state.metacognitive_confidence / self.min_metacognitive_confidence,
            1.0
        )
        scores.append(meta_score)
        
        # Prediction quality score
        error_score = max(0.0, 1.0 - consciousness_state.prediction_state.total_error / self.max_prediction_error)
        scores.append(error_score)
        
        # Integration score
        integration_score = min(
            consciousness_state.phi_value.integration_complexity_ratio / self.min_integration_complexity_ratio,
            1.0
        )
        scores.append(integration_score)
        
        # Environmental coupling score
        if self.require_environmental_coupling:
            coupling_markers = consciousness_state.phenomenological_markers.get('environmental_coupling', 0)
            coupling_score = min(coupling_markers / 0.1, 1.0)
            scores.append(coupling_score)
        
        # Stability score
        stability_score = 1.0 if consciousness_state.prediction_state.is_stable else 0.0
        scores.append(stability_score)
        
        return sum(scores) / len(scores)


class ConsciousnessStabilitySpecification(ConsciousnessSpecification):
    """
    Specification for consciousness state stability.
    
    Defines criteria for stable, sustained consciousness rather than
    fleeting conscious moments. Based on temporal coherence and
    consistency requirements.
    """
    
    def __init__(
        self,
        min_duration_seconds: float = 1.0,
        max_phi_variance: float = 0.2,
        max_attention_drift: float = 0.3,
        require_prediction_convergence: bool = True
    ):
        """
        Initialize consciousness stability specification.
        
        Args:
            min_duration_seconds: Minimum duration for stable consciousness
            max_phi_variance: Maximum allowed variance in Φ values
            max_attention_drift: Maximum allowed attention drift
            require_prediction_convergence: Whether prediction convergence is required
        """
        self.min_duration_seconds = min_duration_seconds
        self.max_phi_variance = max_phi_variance
        self.max_attention_drift = max_attention_drift
        self.require_prediction_convergence = require_prediction_convergence
    
    def is_satisfied_by(self, consciousness_state: ConsciousnessState) -> bool:
        """
        Check if consciousness state satisfies stability criteria.
        
        Args:
            consciousness_state: Consciousness state to evaluate
            
        Returns:
            True if consciousness stability criteria are met
        """
        # For single state, check intrinsic stability markers
        
        # Criterion 1: Prediction system stability
        if self.require_prediction_convergence and not consciousness_state.prediction_state.is_stable:
            return False
        
        # Criterion 2: Φ value should indicate stable integration
        if consciousness_state.phi_value.confidence < 0.7:  # Low confidence indicates instability
            return False
        
        # Criterion 3: Attention focus should be coherent (not scattered)
        if consciousness_state.attention_focus_strength < 0.3:  # Very unfocused
            return False
        
        # Criterion 4: Reasonable consciousness level (not extreme)
        consciousness_level = consciousness_state.consciousness_level
        if consciousness_level < 0.1 or consciousness_level > 0.95:  # Avoid extremes
            return False
        
        return True
    
    def evaluate_stability_over_time(
        self,
        consciousness_states: List[ConsciousnessState]
    ) -> bool:
        """
        Evaluate stability over a sequence of consciousness states.
        
        Args:
            consciousness_states: Sequence of consciousness states
            
        Returns:
            True if sequence shows stable consciousness
        """
        if len(consciousness_states) < 2:
            return len(consciousness_states) == 1 and self.is_satisfied_by(consciousness_states[0])
        
        # Check duration criterion
        duration = (consciousness_states[-1].timestamp - consciousness_states[0].timestamp).total_seconds()
        if duration < self.min_duration_seconds:
            return False
        
        # Check Φ variance
        phi_values = [state.phi_value.value for state in consciousness_states]
        phi_variance = sum((phi - sum(phi_values) / len(phi_values)) ** 2 for phi in phi_values) / len(phi_values)
        if phi_variance > self.max_phi_variance:
            return False
        
        # Check attention drift
        attention_strengths = [
            state.attention_focus_strength 
            for state in consciousness_states 
            if state.attention_weights is not None
        ]
        if attention_strengths:
            max_strength = max(attention_strengths)
            min_strength = min(attention_strengths)
            if (max_strength - min_strength) > self.max_attention_drift:
                return False
        
        # All individual states must satisfy basic stability
        return all(self.is_satisfied_by(state) for state in consciousness_states)


class AttentionalCoherenceSpecification(ConsciousnessSpecification):
    """
    Specification for attentional coherence and focus quality.
    
    Defines criteria for coherent attention patterns that support
    conscious experience rather than scattered or incoherent attention.
    """
    
    def __init__(
        self,
        min_focus_strength: float = 0.2,
        max_entropy_threshold: float = 0.8,
        require_weight_normalization: bool = True,
        min_dominant_weight_ratio: float = 0.3
    ):
        """
        Initialize attentional coherence specification.
        
        Args:
            min_focus_strength: Minimum attention focus strength
            max_entropy_threshold: Maximum entropy in attention distribution
            require_weight_normalization: Whether attention weights must sum to 1
            min_dominant_weight_ratio: Minimum ratio for dominant attention component
        """
        self.min_focus_strength = min_focus_strength
        self.max_entropy_threshold = max_entropy_threshold
        self.require_weight_normalization = require_weight_normalization
        self.min_dominant_weight_ratio = min_dominant_weight_ratio
    
    def is_satisfied_by(self, attention_weights: List[float]) -> bool:
        """
        Check if attention weights satisfy coherence criteria.
        
        Args:
            attention_weights: Attention weight distribution
            
        Returns:
            True if attention weights show coherent pattern
        """
        if not attention_weights:
            return False
        
        import numpy as np
        weights_array = np.array(attention_weights)
        
        # Criterion 1: Weight normalization
        if self.require_weight_normalization:
            weight_sum = weights_array.sum()
            if not (0.99 <= weight_sum <= 1.01):  # Allow small numerical errors
                return False
        
        # Criterion 2: Focus strength (low entropy)
        if len(attention_weights) > 1:
            # Calculate normalized entropy
            non_zero_weights = weights_array[weights_array > 1e-10]
            if len(non_zero_weights) > 0:
                entropy = -sum(w * np.log(w) for w in non_zero_weights)
                max_entropy = np.log(len(non_zero_weights))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                focus_strength = 1.0 - normalized_entropy
                if focus_strength < self.min_focus_strength:
                    return False
        
        # Criterion 3: Entropy threshold
        if len(attention_weights) > 1:
            # Recalculate entropy for threshold check
            non_zero_weights = weights_array[weights_array > 1e-10]
            if len(non_zero_weights) > 0:
                entropy = -sum(w * np.log(w) for w in non_zero_weights)
                max_entropy = np.log(len(attention_weights))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                if normalized_entropy > self.max_entropy_threshold:
                    return False
        
        # Criterion 4: Dominant component
        if len(attention_weights) > 1:
            max_weight = max(attention_weights)
            total_weight = sum(attention_weights)
            dominant_ratio = max_weight / total_weight if total_weight > 0 else 0
            
            if dominant_ratio < self.min_dominant_weight_ratio:
                return False
        
        return True
    
    def calculate_coherence_score(self, attention_weights: List[float]) -> float:
        """
        Calculate a continuous coherence score [0, 1] for attention weights.
        
        Args:
            attention_weights: Attention weight distribution
            
        Returns:
            Coherence score
        """
        if not attention_weights:
            return 0.0
        
        import numpy as np
        weights_array = np.array(attention_weights)
        scores = []
        
        # Normalization score
        if self.require_weight_normalization:
            weight_sum = weights_array.sum()
            norm_score = 1.0 - abs(weight_sum - 1.0)  # Closer to 1.0 is better
            scores.append(max(0.0, norm_score))
        
        # Focus strength score
        if len(attention_weights) > 1:
            non_zero_weights = weights_array[weights_array > 1e-10]
            if len(non_zero_weights) > 0:
                entropy = -sum(w * np.log(w) for w in non_zero_weights)
                max_entropy = np.log(len(non_zero_weights))
                focus_strength = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)
                scores.append(max(0.0, focus_strength))
        
        # Dominant component score
        if len(attention_weights) > 1:
            max_weight = max(attention_weights)
            total_weight = sum(attention_weights)
            dominant_ratio = max_weight / total_weight if total_weight > 0 else 0
            
            # Score based on how well it meets minimum ratio
            dominant_score = min(dominant_ratio / self.min_dominant_weight_ratio, 1.0)
            scores.append(dominant_score)
        
        return sum(scores) / len(scores) if scores else 0.0


class AndConsciousnessSpecification(ConsciousnessSpecification):
    """Logical AND of two consciousness specifications."""
    
    def __init__(self, spec1: ConsciousnessSpecification, spec2: ConsciousnessSpecification):
        self.spec1 = spec1
        self.spec2 = spec2
    
    def is_satisfied_by(self, candidate: Any) -> bool:
        return self.spec1.is_satisfied_by(candidate) and self.spec2.is_satisfied_by(candidate)


class OrConsciousnessSpecification(ConsciousnessSpecification):
    """Logical OR of two consciousness specifications."""
    
    def __init__(self, spec1: ConsciousnessSpecification, spec2: ConsciousnessSpecification):
        self.spec1 = spec1
        self.spec2 = spec2
    
    def is_satisfied_by(self, candidate: Any) -> bool:
        return self.spec1.is_satisfied_by(candidate) or self.spec2.is_satisfied_by(candidate)


class NotConsciousnessSpecification(ConsciousnessSpecification):
    """Logical NOT of a consciousness specification."""
    
    def __init__(self, spec: ConsciousnessSpecification):
        self.spec = spec
    
    def is_satisfied_by(self, candidate: Any) -> bool:
        return not self.spec.is_satisfied_by(candidate)