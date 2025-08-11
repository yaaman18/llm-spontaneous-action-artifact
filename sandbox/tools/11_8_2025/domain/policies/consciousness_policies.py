"""
Consciousness Domain Policies.

Policies that encapsulate complex decision-making logic for consciousness
emergence, attention regulation, and metacognitive monitoring using the
Policy pattern within the enactivist framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from ..value_objects.consciousness_state import ConsciousnessState
from ..value_objects.phi_value import PhiValue
from ..value_objects.prediction_state import PredictionState


class ConsciousnessPolicy(ABC):
    """
    Abstract base class for consciousness-related policies.
    
    Implements the Policy pattern for encapsulating complex business
    logic and decision-making rules about consciousness processes.
    """
    
    @abstractmethod
    def apply(self, context: Any) -> Any:
        """
        Apply the policy to the given context.
        
        Args:
            context: Context for policy application
            
        Returns:
            Result of policy application
        """
        pass


class ConsciousnessEmergencePolicy(ConsciousnessPolicy):
    """
    Policy for regulating consciousness emergence and maintenance.
    
    Implements enactivist principles where consciousness emerges through
    dynamic interaction patterns and must be actively maintained through
    environmental coupling and internal coherence.
    """
    
    def __init__(
        self,
        emergence_threshold: float = 0.1,
        stability_requirement: float = 0.8,
        environmental_coupling_weight: float = 0.6,
        metacognitive_amplification: float = 1.2
    ):
        """
        Initialize consciousness emergence policy.
        
        Args:
            emergence_threshold: Threshold for consciousness emergence
            stability_requirement: Required stability for sustained consciousness
            environmental_coupling_weight: Weight given to environmental coupling
            metacognitive_amplification: Amplification factor for metacognitive processes
        """
        self.emergence_threshold = emergence_threshold
        self.stability_requirement = stability_requirement
        self.environmental_coupling_weight = environmental_coupling_weight
        self.metacognitive_amplification = metacognitive_amplification
    
    def apply(self, context: ConsciousnessState) -> ConsciousnessState:
        """
        Apply the consciousness emergence policy to the given state.
        
        Args:
            context: Consciousness state to apply emergence regulation to
            
        Returns:
            Regulated consciousness state
        """
        return self.apply_emergence_regulation(context)
    
    def apply_emergence_regulation(self, consciousness_state: ConsciousnessState) -> ConsciousnessState:
        """
        Apply emergence regulation to consciousness state.
        
        Implements dynamic regulation where consciousness emerges and is
        sustained through environmental interaction and internal coherence.
        
        Args:
            consciousness_state: Current consciousness state
            
        Returns:
            Regulated consciousness state
        """
        # Get current environmental coupling strength
        coupling_strength = consciousness_state.phenomenological_markers.get(
            'environmental_coupling', 0.0
        )
        
        # Calculate emergence factor based on multiple criteria
        emergence_factor = self._calculate_emergence_factor(consciousness_state, coupling_strength)
        
        # Apply emergence regulation to Φ value
        regulated_phi = self._regulate_phi_value(consciousness_state.phi_value, emergence_factor)
        
        # Apply metacognitive amplification if consciousness is emerging
        regulated_metacognitive = self._regulate_metacognitive_confidence(
            consciousness_state.metacognitive_confidence, emergence_factor
        )
        
        # Create regulated consciousness state
        return ConsciousnessState(
            phi_value=regulated_phi,
            prediction_state=consciousness_state.prediction_state,
            uncertainty_distribution=consciousness_state.uncertainty_distribution,
            spatial_organization=consciousness_state.spatial_organization,
            metacognitive_confidence=regulated_metacognitive,
            attention_weights=consciousness_state.attention_weights,
            phenomenological_markers=self._update_emergence_markers(
                consciousness_state.phenomenological_markers, emergence_factor
            )
        )
    
    def _calculate_emergence_factor(
        self,
        consciousness_state: ConsciousnessState,
        coupling_strength: float
    ) -> float:
        """
        Calculate emergence factor based on multiple consciousness indicators.
        
        Args:
            consciousness_state: Current consciousness state
            coupling_strength: Environmental coupling strength
            
        Returns:
            Emergence factor [0, 1+] for consciousness regulation
        """
        factors = []
        
        # Factor 1: Φ value relative to threshold
        phi_factor = consciousness_state.phi_value.value / self.emergence_threshold
        factors.append(phi_factor)
        
        # Factor 2: Prediction quality (better predictions support consciousness)
        prediction_factor = consciousness_state.prediction_state.prediction_quality
        factors.append(prediction_factor)
        
        # Factor 3: Environmental coupling (enactivist requirement)
        coupling_factor = coupling_strength * self.environmental_coupling_weight
        factors.append(coupling_factor)
        
        # Factor 4: Metacognitive coherence
        meta_factor = consciousness_state.metacognitive_confidence
        factors.append(meta_factor)
        
        # Factor 5: Attention focus strength
        attention_factor = consciousness_state.attention_focus_strength
        factors.append(attention_factor)
        
        # Calculate weighted emergence factor
        emergence_factor = sum(factors) / len(factors)
        
        # Apply stability requirement
        if emergence_factor > 1.0 and not consciousness_state.prediction_state.is_stable:
            emergence_factor *= self.stability_requirement
        
        return max(0.0, emergence_factor)
    
    def _regulate_phi_value(self, phi_value: PhiValue, emergence_factor: float) -> PhiValue:
        """
        Regulate Φ value based on emergence factor.
        
        Args:
            phi_value: Current Φ value
            emergence_factor: Emergence regulation factor
            
        Returns:
            Regulated Φ value
        """
        # Apply emergence regulation to complexity and integration
        regulated_complexity = phi_value.complexity * min(emergence_factor, 1.2)
        regulated_integration = phi_value.integration * min(emergence_factor, 1.2)
        
        # Update Φ value with regulated components
        return phi_value.with_updated_components(
            new_complexity=regulated_complexity,
            new_integration=regulated_integration
        )
    
    def _regulate_metacognitive_confidence(
        self,
        current_confidence: float,
        emergence_factor: float
    ) -> float:
        """
        Regulate metacognitive confidence based on emergence dynamics.
        
        Args:
            current_confidence: Current metacognitive confidence
            emergence_factor: Emergence regulation factor
            
        Returns:
            Regulated metacognitive confidence
        """
        # Apply amplification when consciousness is emerging
        if emergence_factor > 1.0:
            amplified_confidence = current_confidence * self.metacognitive_amplification
            return min(amplified_confidence, 1.0)
        else:
            # Gradual decay when consciousness is not supported
            return current_confidence * emergence_factor
    
    def _update_emergence_markers(
        self,
        current_markers: Dict[str, Any],
        emergence_factor: float
    ) -> Dict[str, Any]:
        """Update phenomenological markers with emergence information."""
        updated_markers = current_markers.copy()
        updated_markers['emergence_factor'] = emergence_factor
        updated_markers['emergence_regulated'] = True
        
        if emergence_factor > 1.0:
            updated_markers['consciousness_emerging'] = True
        elif emergence_factor < 0.5:
            updated_markers['consciousness_fading'] = True
        
        return updated_markers


class AttentionRegulationPolicy(ConsciousnessPolicy):
    """
    Policy for regulating attention dynamics and focus patterns.
    
    Implements enactivist attention regulation where attention emerges
    from environmental interaction patterns and supports conscious
    experience through selective environmental coupling.
    """
    
    def __init__(
        self,
        min_focus_threshold: float = 0.2,
        max_dispersion_allowed: float = 0.8,
        environmental_bias_strength: float = 0.4,
        coherence_reinforcement: float = 1.1
    ):
        """
        Initialize attention regulation policy.
        
        Args:
            min_focus_threshold: Minimum attention focus required
            max_dispersion_allowed: Maximum attention dispersion allowed
            environmental_bias_strength: Strength of environmental bias
            coherence_reinforcement: Factor for reinforcing coherent attention
        """
        self.min_focus_threshold = min_focus_threshold
        self.max_dispersion_allowed = max_dispersion_allowed
        self.environmental_bias_strength = environmental_bias_strength
        self.coherence_reinforcement = coherence_reinforcement
    
    def apply(self, context: Any) -> List[float]:
        """
        Apply attention regulation policy.
        
        Args:
            context: Dict containing 'weights' and 'consciousness_state'
            
        Returns:
            Regulated attention weights
        """
        proposed_weights = context.get('weights', [])
        consciousness_state = context.get('consciousness_state')
        return self.regulate_attention_weights(proposed_weights, consciousness_state)
    
    def regulate_attention_weights(
        self,
        proposed_weights: List[float],
        consciousness_state: ConsciousnessState
    ) -> List[float]:
        """
        Regulate attention weights to maintain coherent focus patterns.
        
        Args:
            proposed_weights: Proposed attention weight distribution
            consciousness_state: Current consciousness state
            
        Returns:
            Regulated attention weights
        """
        weights = np.array(proposed_weights.copy())
        
        # Step 1: Ensure normalization
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        
        # Step 2: Apply minimum focus requirement
        weights = self._enforce_minimum_focus(weights)
        
        # Step 3: Limit excessive dispersion
        weights = self._limit_attention_dispersion(weights)
        
        # Step 4: Apply environmental bias
        weights = self._apply_environmental_bias(weights, consciousness_state)
        
        # Step 5: Reinforce coherent patterns
        weights = self._reinforce_coherence(weights)
        
        # Final normalization
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        
        return weights.tolist()
    
    def _enforce_minimum_focus(self, weights: np.ndarray) -> np.ndarray:
        """
        Enforce minimum attention focus by boosting dominant components.
        
        Args:
            weights: Current attention weights
            
        Returns:
            Weights with enforced minimum focus
        """
        if len(weights) <= 1:
            return weights
        
        # Calculate current focus strength
        max_weight = np.max(weights)
        focus_strength = max_weight
        
        if focus_strength < self.min_focus_threshold:
            # Boost the dominant component
            dominant_idx = np.argmax(weights)
            boost_amount = self.min_focus_threshold - focus_strength
            
            # Add boost to dominant component
            weights[dominant_idx] += boost_amount
            
            # Redistribute from other components
            other_indices = np.arange(len(weights)) != dominant_idx
            if np.any(other_indices):
                reduction_per_component = boost_amount / np.sum(other_indices)
                weights[other_indices] = np.maximum(
                    0.0, weights[other_indices] - reduction_per_component
                )
        
        return weights
    
    def _limit_attention_dispersion(self, weights: np.ndarray) -> np.ndarray:
        """
        Limit excessive attention dispersion.
        
        Args:
            weights: Current attention weights
            
        Returns:
            Weights with limited dispersion
        """
        if len(weights) <= 1:
            return weights
        
        # Calculate entropy-based dispersion
        non_zero_weights = weights[weights > 1e-10]
        if len(non_zero_weights) > 0:
            entropy = -np.sum(non_zero_weights * np.log(non_zero_weights))
            max_entropy = np.log(len(non_zero_weights))
            dispersion = entropy / max_entropy if max_entropy > 0 else 0
            
            if dispersion > self.max_dispersion_allowed:
                # Concentrate attention by boosting top components
                sorted_indices = np.argsort(weights)[::-1]  # Descending order
                
                # Keep top components, reduce others
                concentration_factor = self.max_dispersion_allowed / dispersion
                for i, idx in enumerate(sorted_indices):
                    if i < len(weights) // 2:  # Top half
                        weights[idx] *= (1.0 + (1.0 - concentration_factor))
                    else:  # Bottom half
                        weights[idx] *= concentration_factor
        
        return weights
    
    def _apply_environmental_bias(
        self,
        weights: np.ndarray,
        consciousness_state: ConsciousnessState
    ) -> np.ndarray:
        """
        Apply environmental bias to attention weights.
        
        Args:
            weights: Current attention weights
            consciousness_state: Current consciousness state
            
        Returns:
            Weights with environmental bias applied
        """
        # Get environmental coupling information
        coupling_markers = consciousness_state.phenomenological_markers.get(
            'environmental_coupling', 0.0
        )
        
        if coupling_markers > 0:
            # Bias attention toward environmental relevant components
            # In practice, this would be based on sensory input salience
            # For now, we apply a general environmental bias
            
            # Bias toward lower indices (assuming they represent more environmental/sensory)
            bias_weights = np.exp(-np.arange(len(weights)) * 0.1)  # Exponential decay
            bias_weights = bias_weights / bias_weights.sum()
            
            # Apply bias
            environmental_component = bias_weights * self.environmental_bias_strength * coupling_markers
            weights = weights * (1.0 - self.environmental_bias_strength * coupling_markers) + environmental_component
        
        return weights
    
    def _reinforce_coherence(self, weights: np.ndarray) -> np.ndarray:
        """
        Reinforce coherent attention patterns.
        
        Args:
            weights: Current attention weights
            
        Returns:
            Weights with reinforced coherence
        """
        if len(weights) <= 1:
            return weights
        
        # Identify coherent patterns (local maxima)
        coherent_indices = []
        for i in range(len(weights)):
            left_neighbor = weights[i-1] if i > 0 else 0
            right_neighbor = weights[i+1] if i < len(weights)-1 else 0
            
            if weights[i] >= left_neighbor and weights[i] >= right_neighbor and weights[i] > 0.1:
                coherent_indices.append(i)
        
        # Reinforce coherent components
        if coherent_indices:
            for idx in coherent_indices:
                weights[idx] *= self.coherence_reinforcement
        
        return weights


class MetacognitiveMonitoringPolicy(ConsciousnessPolicy):
    """
    Policy for metacognitive monitoring and self-awareness regulation.
    
    Implements metacognitive processes that monitor consciousness quality,
    prediction accuracy, and environmental coupling effectiveness based
    on enactivist principles of self-organization and autonomy.
    """
    
    def __init__(
        self,
        confidence_update_rate: float = 0.1,
        prediction_quality_weight: float = 0.4,
        environmental_coherence_weight: float = 0.3,
        temporal_consistency_weight: float = 0.3
    ):
        """
        Initialize metacognitive monitoring policy.
        
        Args:
            confidence_update_rate: Rate of metacognitive confidence updates
            prediction_quality_weight: Weight for prediction quality in monitoring
            environmental_coherence_weight: Weight for environmental coherence
            temporal_consistency_weight: Weight for temporal consistency
        """
        self.confidence_update_rate = confidence_update_rate
        self.prediction_quality_weight = prediction_quality_weight
        self.environmental_coherence_weight = environmental_coherence_weight
        self.temporal_consistency_weight = temporal_consistency_weight
    
    def apply(self, context: Any) -> ConsciousnessState:
        """
        Apply metacognitive monitoring policy.
        
        Args:
            context: Dict containing 'consciousness_state' and 'state_history'
            
        Returns:
            Consciousness state with updated metacognitive confidence
        """
        consciousness_state = context.get('consciousness_state')
        state_history = context.get('state_history', [])
        return self.apply_metacognitive_monitoring(consciousness_state, state_history)
    
    def apply_metacognitive_monitoring(
        self,
        consciousness_state: ConsciousnessState,
        state_history: List[ConsciousnessState]
    ) -> ConsciousnessState:
        """
        Apply metacognitive monitoring to update consciousness state.
        
        Args:
            consciousness_state: Current consciousness state
            state_history: History of consciousness states
            
        Returns:
            Consciousness state with updated metacognitive confidence
        """
        # Calculate new metacognitive confidence
        new_confidence = self._calculate_metacognitive_confidence(
            consciousness_state, state_history
        )
        
        # Update phenomenological markers with metacognitive insights
        updated_markers = self._generate_metacognitive_markers(
            consciousness_state, state_history, new_confidence
        )
        
        # Create updated consciousness state
        return ConsciousnessState(
            phi_value=consciousness_state.phi_value,
            prediction_state=consciousness_state.prediction_state,
            uncertainty_distribution=consciousness_state.uncertainty_distribution,
            spatial_organization=consciousness_state.spatial_organization,
            metacognitive_confidence=new_confidence,
            attention_weights=consciousness_state.attention_weights,
            phenomenological_markers=updated_markers
        )
    
    def _calculate_metacognitive_confidence(
        self,
        current_state: ConsciousnessState,
        state_history: List[ConsciousnessState]
    ) -> float:
        """
        Calculate updated metacognitive confidence.
        
        Args:
            current_state: Current consciousness state
            state_history: History of states for temporal analysis
            
        Returns:
            Updated metacognitive confidence [0, 1]
        """
        confidence_components = []
        
        # Component 1: Prediction quality assessment
        prediction_quality = current_state.prediction_state.prediction_quality
        prediction_confidence = prediction_quality * self.prediction_quality_weight
        confidence_components.append(prediction_confidence)
        
        # Component 2: Environmental coherence assessment
        coupling_strength = current_state.phenomenological_markers.get(
            'environmental_coupling', 0.0
        )
        environmental_confidence = coupling_strength * self.environmental_coherence_weight
        confidence_components.append(environmental_confidence)
        
        # Component 3: Temporal consistency assessment
        if len(state_history) >= 2:
            temporal_confidence = self._assess_temporal_consistency(
                current_state, state_history
            ) * self.temporal_consistency_weight
            confidence_components.append(temporal_confidence)
        
        # Calculate new confidence with gradual updates
        target_confidence = sum(confidence_components)
        current_confidence = current_state.metacognitive_confidence
        
        # Apply update rate for smooth transitions
        new_confidence = (
            current_confidence * (1.0 - self.confidence_update_rate) +
            target_confidence * self.confidence_update_rate
        )
        
        return max(0.0, min(1.0, new_confidence))
    
    def _assess_temporal_consistency(
        self,
        current_state: ConsciousnessState,
        state_history: List[ConsciousnessState]
    ) -> float:
        """
        Assess temporal consistency of consciousness states.
        
        Args:
            current_state: Current consciousness state
            state_history: History of consciousness states
            
        Returns:
            Temporal consistency score [0, 1]
        """
        if len(state_history) < 2:
            return 0.5  # Neutral when insufficient history
        
        recent_states = state_history[-5:] + [current_state]  # Last 5 + current
        
        # Check Φ value stability
        phi_values = [state.phi_value.value for state in recent_states]
        phi_variance = np.var(phi_values) if len(phi_values) > 1 else 0
        phi_consistency = max(0.0, 1.0 - phi_variance)
        
        # Check consciousness level stability
        consciousness_levels = [state.consciousness_level for state in recent_states]
        level_variance = np.var(consciousness_levels) if len(consciousness_levels) > 1 else 0
        level_consistency = max(0.0, 1.0 - level_variance)
        
        # Check attention stability
        attention_scores = []
        for state in recent_states:
            if state.attention_weights is not None:
                attention_scores.append(state.attention_focus_strength)
        
        attention_consistency = 0.5  # Default
        if len(attention_scores) > 1:
            attention_variance = np.var(attention_scores)
            attention_consistency = max(0.0, 1.0 - attention_variance)
        
        # Combined temporal consistency
        return (phi_consistency + level_consistency + attention_consistency) / 3.0
    
    def _generate_metacognitive_markers(
        self,
        current_state: ConsciousnessState,
        state_history: List[ConsciousnessState],
        new_confidence: float
    ) -> Dict[str, Any]:
        """
        Generate metacognitive markers based on monitoring results.
        
        Args:
            current_state: Current consciousness state
            state_history: State history for analysis
            new_confidence: Updated metacognitive confidence
            
        Returns:
            Updated phenomenological markers with metacognitive insights
        """
        markers = current_state.phenomenological_markers.copy()
        
        # Add metacognitive awareness markers
        markers['metacognitive_confidence'] = new_confidence
        markers['self_monitoring_active'] = True
        
        # Add confidence change markers
        confidence_change = new_confidence - current_state.metacognitive_confidence
        if abs(confidence_change) > 0.1:
            markers['confidence_shift'] = confidence_change
            if confidence_change > 0:
                markers['increasing_self_awareness'] = True
            else:
                markers['decreasing_self_awareness'] = True
        
        # Add stability insights
        if len(state_history) >= 3:
            temporal_consistency = self._assess_temporal_consistency(current_state, state_history)
            markers['temporal_consistency'] = temporal_consistency
            
            if temporal_consistency > 0.8:
                markers['stable_consciousness'] = True
            elif temporal_consistency < 0.3:
                markers['unstable_consciousness'] = True
        
        # Add prediction quality insights
        prediction_quality = current_state.prediction_state.prediction_quality
        if prediction_quality > 0.8:
            markers['high_prediction_confidence'] = True
        elif prediction_quality < 0.3:
            markers['poor_prediction_quality'] = True
        
        return markers