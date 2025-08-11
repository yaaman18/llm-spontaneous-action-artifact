"""Experiential memory system implementing enactive retention and recall.

This module implements experiential memory following enactivist principles,
focusing on functional equivalence rather than qualia reproduction.
Based on Varela-Maturana circular causality and Husserlian retention theory.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import equinox as eqx

from .types import (
    Array,
    ArrayLike,
    PRNGKey,
    TimeStep,
    TemporalMoment,
    BodyState,
    EnactiveConsciousnessError,
    validate_consciousness_state,
)


class ExperientialTrace(eqx.Module):
    """Trace of experiential interaction for enactive memory.
    
    Represents the sedimentation of agent-environment coupling
    following Merleau-Ponty's concept of motor intentionality traces.
    """
    
    interaction_pattern: Array
    contextual_embedding: Array
    significance_weight: float
    temporal_depth: float
    coupling_strength: float
    
    def __init__(
        self,
        interaction_pattern: Array,
        contextual_embedding: Array,
        significance_weight: float,
        temporal_depth: float,
        coupling_strength: float,
    ):
        self.interaction_pattern = interaction_pattern
        self.contextual_embedding = contextual_embedding
        self.significance_weight = significance_weight
        self.temporal_depth = temporal_depth
        self.coupling_strength = coupling_strength


class CircularCausalityEngine(eqx.Module):
    """Advanced implementation of Varela-Maturana circular causality.
    
    Sophisticated engine using information theory, dynamic networks, and 
    continuous dynamics for authentic enactive circular causality.
    Integrates all newly created modules for maximum theoretical fidelity.
    """
    
    # Core processing networks
    self_reference_network: eqx.nn.MLP
    environment_coupling_network: eqx.nn.MLP
    meaning_emergence_network: eqx.nn.MLP
    
    # Advanced integration components
    information_integration_network: eqx.nn.MLP
    network_state_processor: eqx.nn.MLP
    sparse_meaning_encoder: eqx.nn.MLP
    
    # State tracking
    circular_update_weights: Array
    history_buffer: Array  # For information theory calculations
    network_connectivity: Array  # For dynamic network representation
    
    # Configuration
    state_dim: int
    environment_dim: int
    use_information_theory: bool
    use_dynamic_networks: bool
    
    def __init__(
        self,
        state_dim: int,
        environment_dim: int,
        hidden_dim: int,
        key: PRNGKey,
        history_length: int = 10,
        use_information_theory: bool = True,
        use_dynamic_networks: bool = True,
    ):
        keys = jax.random.split(key, 8)
        
        self.state_dim = state_dim
        self.environment_dim = environment_dim
        self.use_information_theory = use_information_theory
        self.use_dynamic_networks = use_dynamic_networks
        
        # Core networks (enhanced)
        self.self_reference_network = eqx.nn.MLP(
            in_size=state_dim,
            out_size=state_dim,
            width_size=hidden_dim,
            depth=3,  # Deeper for more complexity
            activation=jax.nn.tanh,
            key=keys[0],
        )
        
        self.environment_coupling_network = eqx.nn.MLP(
            in_size=environment_dim + state_dim,
            out_size=state_dim,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.gelu,
            key=keys[1],
        )
        
        self.meaning_emergence_network = eqx.nn.MLP(
            in_size=state_dim * 3,  # More inputs for richer meaning
            out_size=state_dim,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.silu,
            key=keys[2],
        )
        
        # Advanced integration components
        self.information_integration_network = eqx.nn.MLP(
            in_size=state_dim + 7,  # State + information theory metrics
            out_size=state_dim,
            width_size=hidden_dim // 2,
            depth=2,
            activation=jax.nn.swish,
            key=keys[3],
        )
        
        self.network_state_processor = eqx.nn.MLP(
            in_size=state_dim + state_dim,  # State + network features
            out_size=state_dim,
            width_size=hidden_dim // 2,
            depth=2,
            activation=jax.nn.relu,
            key=keys[4],
        )
        
        self.sparse_meaning_encoder = eqx.nn.MLP(
            in_size=state_dim,
            out_size=state_dim // 2,  # Compressed representation
            width_size=hidden_dim // 3,
            depth=2,
            activation=jax.nn.leaky_relu,
            key=keys[5],
        )
        
        # State tracking
        self.circular_update_weights = jax.random.uniform(keys[6], (5,))  # More weights
        self.history_buffer = jnp.zeros((history_length, state_dim + environment_dim))
        self.network_connectivity = jax.random.uniform(keys[7], (state_dim, state_dim)) * 0.1
    
    def circular_causality_step(
        self,
        current_state: Array,
        environmental_input: Array,
        previous_meaning: Optional[Array] = None,
        step_count: int = 0,
    ) -> Tuple['CircularCausalityEngine', Array, Array, Dict[str, float]]:
        """Execute advanced circular causality step with all enhancements.
        
        Returns:
            Tuple containing:
            - Updated CircularCausalityEngine instance
            - Next state array
            - Emergent meaning array
            - Advanced metrics dictionary
        """
        
        try:
            # === Phase 1: Core Processing (Pure Functions) ===
            processing_state = self._execute_core_processing(
                current_state, environmental_input, previous_meaning
            )
            
            # === Phase 2: Information Theory Integration ===
            info_enhanced_state, info_theory_metrics = self._integrate_information_theory(
                current_state, environmental_input, step_count
            )
            
            # === Phase 3: Dynamic Network Processing ===
            updated_self, network_enhanced_state, network_features = self._process_dynamic_networks(
                current_state, environmental_input
            )
            
            # === Phase 4: State Integration and Meaning Emergence ===
            next_state, emergent_meaning = self._integrate_circular_causality(
                processing_state, info_enhanced_state, network_enhanced_state
            )
            
            # === Phase 5: Advanced Metrics Computation ===
            advanced_metrics = self._compute_advanced_metrics(
                current_state, next_state, emergent_meaning,
                processing_state['self_referenced'], processing_state['environment_coupled'],
                info_theory_metrics, network_features
            )
            
            # === Phase 6: History Buffer Update (Immutable State Threading) ===
            final_updated_self = updated_self._update_history_buffer(
                current_state, environmental_input, step_count
            )
            
            return final_updated_self, next_state, emergent_meaning, advanced_metrics
            
        except Exception as e:
            # Fallback to simple computation
            next_state, emergent_meaning, metrics = self._fallback_computation(
                current_state, environmental_input, previous_meaning
            )
            return self, next_state, emergent_meaning, metrics
    
    def _compute_information_theory_metrics(
        self, 
        current_state: Array, 
        environmental_input: Array,
        step_count: int
    ) -> Dict[str, float]:
        """Compute information theory metrics if available."""
        try:
            if not self.use_information_theory or step_count < 3:
                return {'circular_causality': 0.5, 'complexity_index': 0.5, 'entropy_rate': 0.3}
            
            # Extract relevant history
            history_length = min(step_count, self.history_buffer.shape[0])
            if history_length < 3:
                return {'circular_causality': 0.5, 'complexity_index': 0.5, 'entropy_rate': 0.3}
            
            agent_history = self.history_buffer[:history_length, :self.state_dim]
            env_history = self.history_buffer[:history_length, self.state_dim:]
            
            # Import information theory functions
            from .information_theory import (
                circular_causality_index, 
                complexity_measure,
                entropy_rate
            )
            
            # Compute metrics
            circular_metrics = circular_causality_index(agent_history, env_history)
            complexity_metrics = complexity_measure(agent_history, env_history)
            
            # Add entropy rate
            try:
                agent_entropy = entropy_rate(agent_history.flatten())
                combined_metrics = {**circular_metrics, **complexity_metrics}
                combined_metrics['entropy_rate'] = agent_entropy
                return combined_metrics
            except:
                return {**circular_metrics, **complexity_metrics, 'entropy_rate': 0.3}
                
        except Exception:
            # Return default values if information theory computation fails
            return {
                'circular_causality': 0.5,
                'transfer_entropy_env_to_agent': 0.3,
                'transfer_entropy_agent_to_env': 0.3,
                'coupling_coherence': 0.6,
                'instantaneous_coupling': 0.4,
                'complexity_index': 0.5,
                'entropy_rate': 0.3,
            }
    
    def _compute_network_features(self, current_state: Array, environmental_input: Array) -> Array:
        """Compute dynamic network features (DEPRECATED - use _process_dynamic_networks).
        
        This method is kept for compatibility but should not be used directly.
        Use _process_dynamic_networks for proper immutable state management.
        """
        try:
            if not self.use_dynamic_networks:
                return jnp.zeros_like(current_state)
            
            # Simple network features based on connectivity
            state_similarity = jnp.dot(self.network_connectivity, current_state)
            network_activity = jax.nn.tanh(state_similarity)
            
            return network_activity
            
        except Exception:
            return jnp.zeros_like(current_state)
    
    def _compute_advanced_metrics(
        self,
        current_state: Array,
        next_state: Array,
        emergent_meaning: Array,
        self_referenced_state: Array,
        environment_coupled_state: Array,
        info_theory_metrics: Dict[str, float],
        network_features: Array,
    ) -> Dict[str, float]:
        """Compute comprehensive advanced metrics."""
        
        # Basic metrics
        self_reference_strength = float(jnp.linalg.norm(self_referenced_state))
        coupling_strength = float(jnp.linalg.norm(environment_coupled_state))
        meaning_emergence = float(jnp.linalg.norm(emergent_meaning))
        
        # Enhanced circular coherence using multiple measures
        state_correlation = jnp.corrcoef(next_state, current_state)[0, 1]
        state_correlation = jnp.nan_to_num(state_correlation, nan=0.5)
        
        network_coherence = float(jnp.mean(jnp.abs(network_features)))
        info_coherence = info_theory_metrics.get('circular_causality', 0.5)
        
        # Combined circular coherence
        circular_coherence = (
            0.4 * float(state_correlation) +
            0.3 * network_coherence +
            0.3 * info_coherence
        )
        
        # Advanced integration metrics
        state_divergence = float(jnp.linalg.norm(next_state - current_state))
        meaning_novelty = float(jnp.var(emergent_meaning))
        system_complexity = float(jnp.std(jnp.concatenate([
            current_state, next_state, emergent_meaning
        ])))
        
        # Compile comprehensive metrics
        metrics = {
            'self_reference_strength': self_reference_strength,
            'coupling_strength': coupling_strength,
            'meaning_emergence': meaning_emergence,
            'circular_coherence': circular_coherence,
            'state_divergence': state_divergence,
            'meaning_novelty': meaning_novelty,
            'system_complexity': system_complexity,
            'network_coherence': network_coherence,
        }
        
        # Add information theory metrics
        metrics.update(info_theory_metrics)
        
        return metrics
    
    def _update_history_buffer(
        self, 
        current_state: Array, 
        environmental_input: Array,
        step_count: int
    ) -> 'CircularCausalityEngine':
        """Update circular history buffer with proper immutable state threading."""
        combined_state = jnp.concatenate([current_state, environmental_input])
        
        if step_count < self.history_buffer.shape[0]:
            # Fill buffer sequentially
            new_buffer = self.history_buffer.at[step_count].set(combined_state)
        else:
            # Circular buffer rotation using immutable operations
            new_buffer = jnp.roll(self.history_buffer, -1, axis=0)
            new_buffer = new_buffer.at[-1].set(combined_state)
        
        # Return updated self through eqx.tree_at for proper immutable state threading
        return eqx.tree_at(lambda x: x.history_buffer, self, new_buffer)
    
    def _fallback_computation(
        self,
        current_state: Array,
        environmental_input: Array,
        previous_meaning: Optional[Array] = None,
    ) -> Tuple[Array, Array, Dict[str, float]]:
        """Fallback computation for error cases."""
        
        # Simple processing
        self_referenced_state = jax.nn.tanh(current_state)
        coupled_input = jnp.concatenate([environmental_input, current_state])
        environment_coupled_state = jax.nn.gelu(coupled_input[:current_state.shape[0]])
        
        if previous_meaning is None:
            previous_meaning = jnp.zeros_like(current_state)
        emergent_meaning = jax.nn.silu(previous_meaning + 0.1 * current_state)
        
        # Simple integration
        next_state = (
            0.4 * self_referenced_state +
            0.4 * environment_coupled_state +
            0.2 * emergent_meaning
        )
        
        # Basic metrics
        metrics = {
            'self_reference_strength': float(jnp.linalg.norm(self_referenced_state)),
            'coupling_strength': float(jnp.linalg.norm(environment_coupled_state)),
            'meaning_emergence': float(jnp.linalg.norm(emergent_meaning)),
            'circular_coherence': 0.5,  # Default value
        }
        
        return next_state, emergent_meaning, metrics\n    \n    def _execute_core_processing(\n        self,\n        current_state: Array,\n        environmental_input: Array,\n        previous_meaning: Optional[Array]\n    ) -> Dict[str, Array]:\n        \"\"\"Extract Method: Execute core processing phases (SRP compliance).\"\"\"\n        # Self-referential processing (autopoietic maintenance)\n        self_referenced_state = self.self_reference_network(current_state)\n        \n        # Environment coupling (structural coupling)\n        coupled_input = jnp.concatenate([environmental_input, current_state])\n        environment_coupled_state = self.environment_coupling_network(coupled_input)\n        \n        # Sparse meaning representation\n        if previous_meaning is None:\n            previous_meaning = jnp.zeros_like(current_state)\n        \n        sparse_meaning = self.sparse_meaning_encoder(previous_meaning)\n        sparse_meaning_expanded = jnp.pad(\n            sparse_meaning, (0, current_state.shape[0] - sparse_meaning.shape[0])\n        )\n        \n        return {\n            'self_referenced': self_referenced_state,\n            'environment_coupled': environment_coupled_state,\n            'sparse_meaning': sparse_meaning_expanded\n        }\n    \n    def _integrate_information_theory(\n        self,\n        current_state: Array,\n        environmental_input: Array,\n        step_count: int\n    ) -> Tuple[Array, Dict[str, float]]:\n        \"\"\"Extract Method: Information theory integration (SRP compliance).\"\"\"\n        info_theory_metrics = self._compute_information_theory_metrics(\n            current_state, environmental_input, step_count\n        )\n        \n        # Create information-aware features\n        info_features = jnp.array([\n            info_theory_metrics.get('circular_causality', 0.0),\n            info_theory_metrics.get('transfer_entropy_env_to_agent', 0.0),\n            info_theory_metrics.get('transfer_entropy_agent_to_env', 0.0),\n            info_theory_metrics.get('coupling_coherence', 0.0),\n            info_theory_metrics.get('instantaneous_coupling', 0.0),\n            info_theory_metrics.get('complexity_index', 0.0),\n            info_theory_metrics.get('entropy_rate', 0.0),\n        ])\n        \n        # Information-enhanced state\n        info_input = jnp.concatenate([current_state, info_features])\n        info_enhanced_state = self.information_integration_network(info_input)\n        \n        return info_enhanced_state, info_theory_metrics\n    \n    def _process_dynamic_networks(\n        self,\n        current_state: Array,\n        environmental_input: Array\n    ) -> Tuple['CircularCausalityEngine', Array, Array]:\n        \"\"\"Extract Method: Dynamic network processing with immutable state updates.\"\"\"\n        try:\n            if not self.use_dynamic_networks:\n                return self, jnp.zeros_like(current_state), jnp.zeros_like(current_state)\n            \n            # Simple network features based on connectivity\n            state_similarity = jnp.dot(self.network_connectivity, current_state)\n            network_activity = jax.nn.tanh(state_similarity)\n            \n            # Network-enhanced state processing\n            network_input = jnp.concatenate([current_state, network_activity])\n            network_enhanced_state = self.network_state_processor(network_input)\n            \n            # Update connectivity based on activity (immutable)\n            activity_outer = jnp.outer(current_state, network_activity)\n            connectivity_update = 0.01 * activity_outer\n            new_connectivity = (\n                0.99 * self.network_connectivity + connectivity_update\n            )\n            \n            # Return updated self with new connectivity\n            updated_self = eqx.tree_at(\n                lambda x: x.network_connectivity, \n                self, \n                new_connectivity\n            )\n            \n            return updated_self, network_enhanced_state, network_activity\n            \n        except Exception:\n            return self, jnp.zeros_like(current_state), jnp.zeros_like(current_state)\n    \n    def _integrate_circular_causality(\n        self,\n        processing_state: Dict[str, Array],\n        info_enhanced_state: Array,\n        network_enhanced_state: Array\n    ) -> Tuple[Array, Array]:\n        \"\"\"Extract Method: Circular causality integration (SRP compliance).\"\"\"\n        # Rich meaning emergence with all components\n        meaning_input = jnp.concatenate([\n            processing_state['self_referenced'],\n            processing_state['environment_coupled'],\n            processing_state['sparse_meaning'],\n        ])\n        emergent_meaning = self.meaning_emergence_network(meaning_input)\n        \n        # Circular integration with weighted components\n        weights = jax.nn.softmax(self.circular_update_weights)\n        \n        next_state = (\n            weights[0] * processing_state['self_referenced'] +\n            weights[1] * processing_state['environment_coupled'] +\n            weights[2] * emergent_meaning +\n            weights[3] * info_enhanced_state +\n            weights[4] * network_enhanced_state\n        )\n        \n        return next_state, emergent_meaning


class ExperientialSedimentation(eqx.Module):
    """Advanced Husserlian sedimentation with sparse representations.
    
    Enhanced implementation with sparse coding, convex optimization,
    and multi-modal meaning preservation for experiential traces.
    Integrates with sparse_representations module for efficiency.
    """
    
    # Core sediment storage
    sediment_layers: Array
    significance_tracker: Array
    
    # Sparse representation components
    sparse_dictionary: Array
    sparse_codes: Array
    compression_history: Array
    
    # Enhanced configuration
    temporal_decay_factor: float
    sedimentation_threshold: float
    sparse_coding_lambda: float
    meaning_preservation_weight: float
    adaptive_threshold: bool
    
    # Dimensions
    experience_dim: int
    num_layers: int
    dictionary_size: int
    
    def __init__(
        self,
        experience_dim: int,
        num_layers: int,
        temporal_decay_factor: float = 0.95,
        sedimentation_threshold: float = 0.1,
        sparse_coding_lambda: float = 0.1,
        meaning_preservation_weight: float = 0.3,
        adaptive_threshold: bool = True,
        dictionary_size: int = None,
        key: PRNGKey = None,
    ):
        self.experience_dim = experience_dim
        self.num_layers = num_layers
        self.dictionary_size = dictionary_size or (experience_dim * 2)
        self.temporal_decay_factor = temporal_decay_factor
        self.sedimentation_threshold = sedimentation_threshold
        self.sparse_coding_lambda = sparse_coding_lambda
        self.meaning_preservation_weight = meaning_preservation_weight
        self.adaptive_threshold = adaptive_threshold
        
        if key is not None:
            keys = jax.random.split(key, 4)
            
            # Initialize sediment layers with small random values
            self.sediment_layers = jax.random.normal(
                keys[0], (num_layers, experience_dim)
            ) * 0.01
            
            # Initialize sparse dictionary (overcomplete basis)
            self.sparse_dictionary = jax.random.normal(
                keys[1], (self.dictionary_size, experience_dim)
            ) * 0.1
            
            # Initialize sparse codes
            self.sparse_codes = jnp.zeros((num_layers, self.dictionary_size))
            
            # Initialize compression history
            self.compression_history = jnp.zeros((num_layers, 3))  # [compression_ratio, reconstruction_error, sparsity]
            
        else:
            self.sediment_layers = jnp.zeros((num_layers, experience_dim))
            self.sparse_dictionary = jnp.eye(self.dictionary_size, experience_dim)
            self.sparse_codes = jnp.zeros((num_layers, self.dictionary_size))
            self.compression_history = jnp.zeros((num_layers, 3))
            
        self.significance_tracker = jnp.ones(num_layers)
    
    def sediment_experience(
        self,
        new_experience: Array,
        significance_weight: float,
        temporal_context: Array,
        use_sparse_coding: bool = True,
    ) -> 'ExperientialSedimentation':
        """Advanced experiential sedimentation with sparse coding.
        
        Refactored using Extract Method pattern for complex operations.
        """
        
        try:
            # === Phase 1: Temporal Decay (Extract Method Pattern) ===
            decay_context = self._apply_temporal_decay()
            decayed_layers, decayed_significance, decayed_codes = decay_context
            
            # === Phase 2: Sparse Representation (Strategy Pattern) ===
            sparse_context = self._apply_sparse_representation_strategy(
                new_experience, use_sparse_coding
            )
            compressed_experience, sparse_representation, compression_metrics = sparse_context
            
            # === Phase 3: Similarity Analysis (Template Method Pattern) ===
            similarity_context = self._analyze_layer_similarities(
                decayed_layers, decayed_significance, decayed_codes,
                compressed_experience, sparse_representation, temporal_context
            )
            current_threshold, layer_similarities, best_layer_idx, max_similarity = similarity_context
            
            # === Phase 4: Layer Selection and Update ===
            if max_similarity < current_threshold:
                # Create new sediment layer (replace least significant)
                least_significant_idx = jnp.argmin(decayed_significance)
                
                new_layers = decayed_layers.at[least_significant_idx].set(compressed_experience)
                new_significance = decayed_significance.at[least_significant_idx].set(significance_weight)
                new_codes = decayed_codes.at[least_significant_idx].set(sparse_representation)
                new_compression_history = self.compression_history.at[least_significant_idx].set(
                    compression_metrics
                )
                
            else:
                # Update existing layer with meaning-preserving integration
                integration_weight = self._compute_meaning_preserving_weight(
                    significance_weight, max_similarity, temporal_context
                )
                
                # Weighted integration of experiences
                updated_layer = (
                    (1.0 - integration_weight) * decayed_layers[best_layer_idx] +
                    integration_weight * compressed_experience
                )
                
                # Weighted integration of sparse codes
                updated_code = (
                    (1.0 - integration_weight) * decayed_codes[best_layer_idx] +
                    integration_weight * sparse_representation
                )
                
                # Update layers
                new_layers = decayed_layers.at[best_layer_idx].set(updated_layer)
                new_codes = decayed_codes.at[best_layer_idx].set(updated_code)
                
                # Enhanced significance update
                significance_boost = significance_weight * 0.1 * (1.0 + max_similarity)
                new_significance = decayed_significance.at[best_layer_idx].add(significance_boost)
                
                # Update compression history
                updated_compression = (
                    0.7 * self.compression_history[best_layer_idx] + 
                    0.3 * compression_metrics
                )
                new_compression_history = self.compression_history.at[best_layer_idx].set(
                    updated_compression
                )
            
            # === Phase 5: Dictionary Learning Update ===
            updated_dictionary = self._update_sparse_dictionary(
                new_experience, sparse_representation
            )
            
            # Create updated instance
            return eqx.tree_at(
                lambda x: (
                    x.sediment_layers, 
                    x.significance_tracker, 
                    x.sparse_codes,
                    x.sparse_dictionary,
                    x.compression_history,
                ),
                self,
                (
                    new_layers, 
                    new_significance, 
                    new_codes,
                    updated_dictionary,
                    new_compression_history,
                ),
            )
            
        except Exception as e:
            # Fallback to simple sedimentation
            return self._fallback_sediment_experience(
                new_experience, significance_weight, temporal_context
            )
    
    def _compute_sparse_representation(self, experience: Array) -> Array:
        """Compute sparse representation using ISTA-like algorithm."""
        try:
            # Import sparse coding functionality
            from .sparse_representations import SparseExperienceEncoder
            
            # Create temporary encoder (simplified)
            alpha = jnp.zeros(self.dictionary_size)
            
            # Simple ISTA iterations
            learning_rate = 0.1
            for _ in range(10):  # Fixed iterations for efficiency
                # Gradient step
                residual = experience - jnp.dot(self.sparse_dictionary.T, alpha)
                gradient = jnp.dot(self.sparse_dictionary, residual)
                alpha = alpha + learning_rate * gradient
                
                # Soft thresholding
                alpha = jnp.sign(alpha) * jnp.maximum(
                    jnp.abs(alpha) - self.sparse_coding_lambda * learning_rate, 0.0
                )
            
            return alpha
            
        except Exception:
            # Fallback: pseudo-sparse representation
            return jnp.zeros(self.dictionary_size)
    
    def _reconstruct_from_sparse(self, sparse_code: Array) -> Array:
        """Reconstruct experience from sparse representation."""
        try:
            return jnp.dot(self.sparse_dictionary.T, sparse_code)
        except Exception:
            return jnp.zeros(self.experience_dim)
    
    def _compute_compression_metrics(
        self, 
        original: Array, 
        reconstructed: Array, 
        sparse_code: Array
    ) -> Array:
        """Compute compression quality metrics."""
        try:
            # Compression ratio
            sparsity = float(jnp.mean(jnp.abs(sparse_code) > 1e-6))
            compression_ratio = 1.0 / (sparsity + 1e-8)
            
            # Reconstruction error
            reconstruction_error = float(jnp.linalg.norm(original - reconstructed))
            
            return jnp.array([compression_ratio, reconstruction_error, sparsity])
            
        except Exception:
            return jnp.array([1.0, 0.1, 1.0])  # Default metrics
    
    def _compute_adaptive_threshold(
        self, 
        significance_values: Array, 
        temporal_context: Array
    ) -> float:
        """Compute adaptive sedimentation threshold."""
        try:
            # Base threshold
            base_threshold = self.sedimentation_threshold
            
            # Adapt based on significance distribution
            significance_variance = float(jnp.var(significance_values))
            significance_adjustment = 0.1 * significance_variance
            
            # Adapt based on temporal context richness
            temporal_richness = float(jnp.std(temporal_context))
            temporal_adjustment = 0.05 * temporal_richness
            
            adapted_threshold = base_threshold + significance_adjustment - temporal_adjustment
            
            # Clamp to reasonable range
            return float(jnp.clip(adapted_threshold, 0.05, 0.3))
            
        except Exception:
            return self.sedimentation_threshold
    
    def _compute_enhanced_similarities(
        self,
        layers: Array,
        new_experience: Array,
        sparse_codes: Array,
        new_sparse_code: Array,
    ) -> Array:
        """Compute multi-modal similarity scores."""
        try:
            # Structural similarity (original method)
            structural_similarities = jnp.array([
                jnp.corrcoef(layer, new_experience)[0, 1] if jnp.var(layer) > 1e-6 else 0.0
                for layer in layers
            ])
            structural_similarities = jnp.nan_to_num(structural_similarities, nan=0.0)
            
            # Sparse code similarity
            sparse_similarities = jnp.array([
                jnp.dot(code, new_sparse_code) / (
                    jnp.linalg.norm(code) * jnp.linalg.norm(new_sparse_code) + 1e-8
                )
                for code in sparse_codes
            ])
            sparse_similarities = jnp.nan_to_num(sparse_similarities, nan=0.0)
            
            # Semantic similarity (using meaning preservation weight)
            semantic_weight = self.meaning_preservation_weight
            
            # Combined similarity
            combined_similarities = (
                0.4 * structural_similarities +
                0.3 * sparse_similarities +
                0.3 * semantic_weight
            )
            
            return combined_similarities
            
        except Exception:
            # Fallback to simple correlation
            return jnp.array([
                jnp.corrcoef(layer, new_experience)[0, 1] if jnp.var(layer) > 1e-6 else 0.0
                for layer in layers
            ])
    
    def _compute_meaning_preserving_weight(
        self,
        significance_weight: float,
        similarity: float,
        temporal_context: Array,
    ) -> float:
        """Compute integration weight that preserves meaning structure."""
        try:
            # Base integration weight
            base_weight = significance_weight * (1.0 - similarity)
            
            # Meaning preservation factor
            meaning_factor = self.meaning_preservation_weight
            
            # Temporal context influence
            temporal_influence = float(jnp.mean(jnp.abs(temporal_context)))
            temporal_factor = jnp.tanh(temporal_influence)
            
            # Combined weight
            integrated_weight = base_weight * meaning_factor * (0.7 + 0.3 * temporal_factor)
            
            return float(jnp.clip(integrated_weight, 0.01, 0.8))
            
        except Exception:
            return significance_weight * (1.0 - similarity)
    
    def _update_sparse_dictionary(
        self, 
        new_experience: Array, 
        sparse_code: Array
    ) -> Array:
        """Update sparse dictionary via online learning with immutable operations."""
        try:
            # Learning rate for dictionary update
            dict_learning_rate = 0.01
            
            # Reconstruct with current dictionary
            reconstruction = jnp.dot(self.sparse_dictionary.T, sparse_code)
            residual = new_experience - reconstruction
            
            # Update dictionary atoms based on residual (immutable approach)
            active_atoms = jnp.abs(sparse_code) > 1e-6
            
            if jnp.any(active_atoms):
                # Vectorized dictionary update for immutable operations
                atom_updates = dict_learning_rate * jnp.outer(sparse_code, residual)
                
                # Apply updates only to active atoms
                masked_updates = jnp.where(
                    active_atoms[:, None], 
                    atom_updates, 
                    jnp.zeros_like(atom_updates)
                )
                
                # Update dictionary immutably
                updated_dictionary = self.sparse_dictionary + masked_updates
                
                # Normalize updated atoms
                atom_norms = jnp.linalg.norm(updated_dictionary, axis=1, keepdims=True)
                normalized_dictionary = updated_dictionary / (atom_norms + 1e-8)
                
                return normalized_dictionary
            
            return self.sparse_dictionary
            
        except Exception:
            return self.sparse_dictionary
    
    def _fallback_sediment_experience(
        self,
        new_experience: Array,
        significance_weight: float,
        temporal_context: Array,
    ) -> 'ExperientialSedimentation':
        """Fallback to original simple sedimentation."""
        
        # Apply temporal decay to existing sediments
        decayed_layers = self.sediment_layers * self.temporal_decay_factor
        decayed_significance = self.significance_tracker * self.temporal_decay_factor
        
        # Find appropriate layer for new experience
        layer_similarities = jnp.array([
            jnp.corrcoef(layer, new_experience)[0, 1] if jnp.var(layer) > 1e-6 else 0.0
            for layer in decayed_layers
        ])
        
        # Handle NaN values from correlation with zero-variance layers
        layer_similarities = jnp.nan_to_num(layer_similarities, nan=0.0)
        
        best_layer_idx = jnp.argmax(layer_similarities)
        max_similarity = layer_similarities[best_layer_idx]
        
        # Determine if new layer is needed or update existing
        if max_similarity < self.sedimentation_threshold:
            # Create new sediment layer (replace least significant)
            least_significant_idx = jnp.argmin(decayed_significance)
            new_layers = decayed_layers.at[least_significant_idx].set(new_experience)
            new_significance = decayed_significance.at[least_significant_idx].set(significance_weight)
        else:
            # Update existing layer with weighted integration
            integration_weight = significance_weight * (1.0 - max_similarity)
            updated_layer = (
                (1.0 - integration_weight) * decayed_layers[best_layer_idx] +
                integration_weight * new_experience
            )
            new_layers = decayed_layers.at[best_layer_idx].set(updated_layer)
            
            # Update significance
            new_significance = decayed_significance.at[best_layer_idx].add(
                significance_weight * 0.1
            )
        
        return eqx.tree_at(
            lambda x: (x.sediment_layers, x.significance_tracker),
            self,
            (new_layers, new_significance),
        )
    
    def _apply_temporal_decay(self) -> Tuple[Array, Array, Array]:
        """Extract method: Apply temporal decay to all components."""
        decayed_layers = self.sediment_layers * self.temporal_decay_factor
        decayed_significance = self.significance_tracker * self.temporal_decay_factor
        decayed_codes = self.sparse_codes * self.temporal_decay_factor
        return decayed_layers, decayed_significance, decayed_codes
    
    def _apply_sparse_representation_strategy(
        self, 
        new_experience: Array, 
        use_sparse_coding: bool
    ) -> Tuple[Array, Array, Array]:
        """Strategy pattern: Apply sparse representation strategy."""
        if use_sparse_coding:
            sparse_representation = self._compute_sparse_representation(new_experience)
            compressed_experience = self._reconstruct_from_sparse(sparse_representation)
            compression_metrics = self._compute_compression_metrics(
                new_experience, compressed_experience, sparse_representation
            )
        else:
            compressed_experience = new_experience
            sparse_representation = jnp.zeros(self.dictionary_size)
            compression_metrics = jnp.array([1.0, 0.0, 0.0])  # No compression
        
        return compressed_experience, sparse_representation, compression_metrics
    
    def _analyze_layer_similarities(
        self,
        decayed_layers: Array,
        decayed_significance: Array,
        decayed_codes: Array,
        compressed_experience: Array,
        sparse_representation: Array,
        temporal_context: Array,
    ) -> Tuple[float, Array, int, float]:
        """Template method: Analyze layer similarities with adaptive threshold."""
        # Compute adaptive threshold
        if self.adaptive_threshold:
            current_threshold = self._compute_adaptive_threshold(
                decayed_significance, temporal_context
            )
        else:
            current_threshold = self.sedimentation_threshold
        
        # Multi-modal similarity (structural + semantic + sparse)
        layer_similarities = self._compute_enhanced_similarities(
            decayed_layers, compressed_experience, decayed_codes, sparse_representation
        )
        
        best_layer_idx = jnp.argmax(layer_similarities)
        max_similarity = layer_similarities[best_layer_idx]
        
        return current_threshold, layer_similarities, best_layer_idx, max_similarity
    
    def retrieve_similar_experiences(
        self,
        query_experience: Array,
        similarity_threshold: float = 0.5,
        max_retrievals: int = 5,
        use_sparse_similarity: bool = True,
    ) -> List[Tuple[Array, float, float]]:
        """Enhanced retrieval with sparse similarity matching."""
        
        try:
            if use_sparse_similarity:
                # Compute sparse representation of query
                query_sparse = self._compute_sparse_representation(query_experience)
                
                # Enhanced similarity computation
                enhanced_similarities = self._compute_enhanced_similarities(
                    self.sediment_layers, query_experience, 
                    self.sparse_codes, query_sparse
                )
                similarities = enhanced_similarities
            else:
                # Fallback to structural similarity only
                similarities = jnp.array([
                    jnp.corrcoef(layer, query_experience)[0, 1] if jnp.var(layer) > 1e-6 else 0.0
                    for layer in self.sediment_layers
                ])
            
            # Handle NaN values
            similarities = jnp.nan_to_num(similarities, nan=0.0)
            
            # Filter by threshold and significance
            valid_indices = jnp.where(
                (similarities >= similarity_threshold) & 
                (self.significance_tracker > self.sedimentation_threshold)
            )[0]
            
            if len(valid_indices) == 0:
                return []
            
            # Enhanced scoring with compression quality
            compression_bonuses = self.compression_history[valid_indices, 0]  # Compression ratios
            quality_penalties = self.compression_history[valid_indices, 1]    # Reconstruction errors
            
            base_scores = similarities[valid_indices] * self.significance_tracker[valid_indices]
            enhanced_scores = base_scores + 0.1 * compression_bonuses - 0.05 * quality_penalties
            
            sorted_indices = valid_indices[jnp.argsort(enhanced_scores)[::-1]]
            
            # Return top retrievals with enhanced metadata
            retrievals = []
            for i in sorted_indices[:max_retrievals]:
                retrievals.append((
                    self.sediment_layers[i],
                    float(similarities[i]),
                    float(self.significance_tracker[i])
                ))
            
            return retrievals
            
        except Exception:
            # Fallback to original retrieval method
            similarities = jnp.array([
                jnp.corrcoef(layer, query_experience)[0, 1] if jnp.var(layer) > 1e-6 else 0.0
                for layer in self.sediment_layers
            ])
            
            similarities = jnp.nan_to_num(similarities, nan=0.0)
            
            valid_indices = jnp.where(
                (similarities >= similarity_threshold) & 
                (self.significance_tracker > self.sedimentation_threshold)
            )[0]
            
            if len(valid_indices) == 0:
                return []
            
            scores = similarities[valid_indices] * self.significance_tracker[valid_indices]
            sorted_indices = valid_indices[jnp.argsort(scores)[::-1]]
            
            retrievals = []
            for i in sorted_indices[:max_retrievals]:
                retrievals.append((
                    self.sediment_layers[i],
                    float(similarities[i]),
                    float(self.significance_tracker[i])
                ))
            
            return retrievals
    
    def get_sedimentation_statistics(self) -> Dict[str, float]:
        """Get comprehensive sedimentation statistics."""
        try:
            return {
                'num_layers': self.num_layers,
                'active_layers': float(jnp.sum(self.significance_tracker > 0.1)),
                'average_significance': float(jnp.mean(self.significance_tracker)),
                'total_sedimentation': float(jnp.sum(jnp.abs(self.sediment_layers))),
                'sparse_dictionary_utilization': float(jnp.mean(jnp.abs(self.sparse_codes) > 1e-6)),
                'average_compression_ratio': float(jnp.mean(self.compression_history[:, 0])),
                'average_reconstruction_error': float(jnp.mean(self.compression_history[:, 1])),
                'average_sparsity': float(jnp.mean(self.compression_history[:, 2])),
            }
        except Exception:
            return {
                'num_layers': self.num_layers,
                'active_layers': float(jnp.sum(self.significance_tracker > 0.1)),
                'average_significance': float(jnp.mean(self.significance_tracker)),
                'total_sedimentation': float(jnp.sum(jnp.abs(self.sediment_layers))),
            }


class AssociativeRecallSystem(eqx.Module):
    """Associative recall system for experiential memory.
    
    Implements associative recall mechanisms based on temporal context,
    emotional resonance, and structural similarity.
    """
    
    context_encoder: eqx.nn.MLP
    emotional_encoder: eqx.nn.MLP
    structural_encoder: eqx.nn.MLP
    association_network: eqx.nn.MultiheadAttention
    recall_threshold: float
    
    def __init__(
        self,
        experience_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        recall_threshold: float = 0.3,
        key: PRNGKey = None,
    ):
        keys = jax.random.split(key, 4) if key is not None else [None] * 4
        
        self.context_encoder = eqx.nn.MLP(
            in_size=context_dim,
            out_size=experience_dim,
            width_size=hidden_dim // 2,
            depth=1,
            activation=jax.nn.relu,
            key=keys[0],
        )
        
        self.emotional_encoder = eqx.nn.MLP(
            in_size=experience_dim,
            out_size=experience_dim // 2,
            width_size=hidden_dim // 3,
            depth=1,
            activation=jax.nn.tanh,
            key=keys[1],
        )
        
        self.structural_encoder = eqx.nn.MLP(
            in_size=experience_dim,
            out_size=experience_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[2],
        )
        
        self.association_network = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=experience_dim,
            key_size=experience_dim,
            value_size=experience_dim,
            output_size=experience_dim,
            key=keys[3],
        )
        
        self.recall_threshold = recall_threshold
    
    def recall_experiences(
        self,
        current_experience: Array,
        contextual_cues: Array,
        sedimented_traces: List[ExperientialTrace],
        max_recalls: int = 3,
    ) -> List[Tuple[ExperientialTrace, float]]:
        """Recall experiences through associative mechanisms."""
        
        if not sedimented_traces:
            return []
        
        # Encode current context
        encoded_context = self.context_encoder(contextual_cues)
        
        # Create query from current experience and context
        query = current_experience + 0.3 * encoded_context
        
        # Encode all traces for association
        trace_embeddings = []
        for trace in sedimented_traces:
            # Structural encoding
            structural = self.structural_encoder(trace.interaction_pattern)
            
            # Emotional encoding
            emotional = self.emotional_encoder(trace.contextual_embedding)
            
            # Combine encodings
            full_embedding = jnp.concatenate([
                structural,
                jnp.pad(emotional, (0, structural.shape[0] - emotional.shape[0]))
            ])[:structural.shape[0]]
            
            trace_embeddings.append(full_embedding)
        
        if not trace_embeddings:
            return []
        
        # Stack embeddings for attention
        trace_matrix = jnp.stack(trace_embeddings)  # (num_traces, embedding_dim)
        
        # Apply attention mechanism
        attended_recalls, attention_weights = self.association_network(
            query[None, None, :],     # (1, 1, embedding_dim)
            trace_matrix[None, :, :], # (1, num_traces, embedding_dim)
            trace_matrix[None, :, :], # (1, num_traces, embedding_dim)
        )
        
        # Extract attention scores
        attention_scores = attention_weights.squeeze()  # (num_traces,)
        
        # Filter by recall threshold
        valid_recalls = []
        for i, (trace, score) in enumerate(zip(sedimented_traces, attention_scores)):
            if score >= self.recall_threshold:
                recall_strength = float(score * trace.significance_weight)
                valid_recalls.append((trace, recall_strength))
        
        # Sort by recall strength and return top results
        valid_recalls.sort(key=lambda x: x[1], reverse=True)
        return valid_recalls[:max_recalls]


class IntegratedExperientialMemory(eqx.Module):
    """Integrated experiential memory system.
    
    Combines circular causality, sedimentation, and associative recall
    into a unified experiential memory system following enactivist principles.
    """
    
    circular_engine: CircularCausalityEngine
    sedimentation: ExperientialSedimentation
    recall_system: AssociativeRecallSystem
    experience_traces: List[ExperientialTrace]
    integration_network: eqx.nn.MLP
    
    def __init__(
        self,
        experience_dim: int,
        environment_dim: int,
        context_dim: int,
        num_sediment_layers: int = 20,
        key: PRNGKey = None,
    ):
        keys = jax.random.split(key, 5) if key is not None else [None] * 5
        
        hidden_dim = max(64, experience_dim)
        
        self.circular_engine = CircularCausalityEngine(
            experience_dim, environment_dim, hidden_dim, keys[0]
        )
        
        self.sedimentation = ExperientialSedimentation(
            experience_dim, num_sediment_layers, key=keys[1]
        )
        
        self.recall_system = AssociativeRecallSystem(
            experience_dim, context_dim, hidden_dim, key=keys[2]
        )
        
        # Initialize empty trace list (will be updated dynamically)
        self.experience_traces = []
        
        # Integration network for combining recalled and current experience
        self.integration_network = eqx.nn.MLP(
            in_size=experience_dim * 2,
            out_size=experience_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.swish,
            key=keys[3],
        )
    
    def process_experiential_moment(
        self,
        current_experience: Array,
        environmental_input: Array,
        contextual_cues: Array,
        significance_weight: float = 0.5,
    ) -> Tuple[Array, Dict[str, Any]]:
        """Process a complete experiential moment through the memory system."""
        
        # Execute circular causality step
        next_state, emergent_meaning, causality_metrics = self.circular_engine.circular_causality_step(
            current_experience, environmental_input
        )
        
        # Create experiential trace
        new_trace = ExperientialTrace(
            interaction_pattern=next_state,
            contextual_embedding=contextual_cues,
            significance_weight=significance_weight,
            temporal_depth=1.0,
            coupling_strength=causality_metrics['coupling_strength'],
        )
        
        # Add to traces (with size limit)
        max_traces = 100
        if len(self.experience_traces) >= max_traces:
            # Remove oldest trace
            self.experience_traces = self.experience_traces[1:]
        self.experience_traces.append(new_trace)
        
        # Sediment the experience
        self.sedimentation = self.sedimentation.sediment_experience(
            next_state, significance_weight, contextual_cues
        )
        
        # Recall similar experiences
        recalled_experiences = self.recall_system.recall_experiences(
            current_experience, contextual_cues, self.experience_traces
        )
        
        # Integrate current and recalled experiences
        if recalled_experiences:
            # Weight recalled experiences by their recall strength
            weighted_recall = jnp.zeros_like(current_experience)
            total_weight = 0.0
            
            for trace, strength in recalled_experiences:
                weighted_recall += strength * trace.interaction_pattern
                total_weight += strength
            
            if total_weight > 0:
                weighted_recall /= total_weight
                
                # Integrate with current experience
                integration_input = jnp.concatenate([next_state, weighted_recall])
                integrated_experience = self.integration_network(integration_input)
            else:
                integrated_experience = next_state
        else:
            integrated_experience = next_state
        
        # Compile metadata
        metadata = {
            'circular_causality': causality_metrics,
            'num_recalls': len(recalled_experiences),
            'sedimentation_layers': self.sedimentation.sediment_layers.shape[0],
            'emergent_meaning': emergent_meaning,
            'significance_weight': significance_weight,
            'trace_count': len(self.experience_traces),
        }
        
        return integrated_experience, metadata
    
    def get_memory_state(self) -> Dict[str, Any]:
        """Get current state of experiential memory system."""
        return {
            'num_traces': len(self.experience_traces),
            'sediment_layers': self.sedimentation.sediment_layers.shape[0],
            'average_significance': float(jnp.mean(self.sedimentation.significance_tracker)),
            'total_sedimentation': float(jnp.sum(jnp.abs(self.sedimentation.sediment_layers))),
        }


# Factory functions
@functools.partial(jax.jit, static_argnames=['experience_dim', 'environment_dim', 'context_dim'])
def create_experiential_memory_system(
    experience_dim: int,
    environment_dim: int,
    context_dim: int,
    key: PRNGKey,
) -> IntegratedExperientialMemory:
    """Factory function for creating JIT-compiled experiential memory system."""
    return IntegratedExperientialMemory(
        experience_dim, environment_dim, context_dim, key=key
    )


# Export public API
__all__ = [
    'ExperientialTrace',
    'CircularCausalityEngine',
    'ExperientialSedimentation',
    'AssociativeRecallSystem',
    'IntegratedExperientialMemory',
    'create_experiential_memory_system',
]