"""Enactive coupling and circular causality implementation.

This module implements Varela's enactive theory of circular causality,
featuring agent-environment structural coupling, meaning emergence,
and circular causality dynamics following the validation requirements
from enactivism domain experts.

Key implementation principles:
1. Circular causality between agent and environment
2. Meaning emergence through structural coupling
3. Self-referential system dynamics
4. Avoidance of direct qualia implementation
"""

from __future__ import annotations

import functools
import logging
from typing import Dict, List, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import equinox as eqx

from .core import (
    ProcessorBase,
    StateValidationMixin,
    ConfigurableMixin,
    MemoryManager,
    MetricCollector,
    create_safe_jit_function,
    optimize_for_memory,
    GLOBAL_MEMORY_MANAGER,
    GLOBAL_METRICS,
)

# Configure module logger
logger = logging.getLogger(__name__)

from .types import (
    Array,
    ArrayLike,
    PRNGKey,
    CouplingState,
    MeaningStructure,
    FrameworkConfig,
    CouplingProcessor,
    CouplingError,
    SenseMakingError,
    TimeStep,
)


class EnactiveCouplingConfig(eqx.Module):
    """Configuration for enhanced enactive coupling dynamics.
    
    Based on Varela's enactive approach to cognition featuring:
    - Structural coupling parameters
    - Enhanced circular causality dynamics
    - Meaning emergence thresholds
    - Autopoietic self-organization parameters
    - Experience retention integration
    """
    
    agent_state_dim: int
    environment_state_dim: int
    coupling_strength_initial: float
    perturbation_sensitivity: float
    meaning_emergence_threshold: float
    circular_causality_rate: float
    self_reference_depth: int
    coupling_memory_length: int
    
    # Enhanced circular causality parameters
    feedback_loop_strength: float
    temporal_coupling_horizon: int
    causality_tracking_depth: int
    
    # Emergent meaning generation parameters
    emergence_noise_factor: float
    pattern_novelty_threshold: float
    creative_synthesis_rate: float
    
    # Autopoietic parameters
    autopoietic_closure_strength: float
    self_maintenance_rate: float
    boundary_maintenance_strength: float
    
    def __init__(
        self,
        agent_state_dim: int = 128,
        environment_state_dim: int = 256,
        coupling_strength_initial: float = 0.5,
        perturbation_sensitivity: float = 0.1,
        meaning_emergence_threshold: float = 0.6,
        circular_causality_rate: float = 0.05,
        self_reference_depth: int = 3,
        coupling_memory_length: int = 20,
        # Enhanced parameters
        feedback_loop_strength: float = 0.8,
        temporal_coupling_horizon: int = 15,
        causality_tracking_depth: int = 10,
        emergence_noise_factor: float = 0.1,
        pattern_novelty_threshold: float = 0.7,
        creative_synthesis_rate: float = 0.03,
        autopoietic_closure_strength: float = 0.6,
        self_maintenance_rate: float = 0.02,
        boundary_maintenance_strength: float = 0.5,
    ):
        self.agent_state_dim = agent_state_dim
        self.environment_state_dim = environment_state_dim
        self.coupling_strength_initial = coupling_strength_initial
        self.perturbation_sensitivity = perturbation_sensitivity
        self.meaning_emergence_threshold = meaning_emergence_threshold
        self.circular_causality_rate = circular_causality_rate
        self.self_reference_depth = self_reference_depth
        self.coupling_memory_length = coupling_memory_length
        
        # Enhanced parameters
        self.feedback_loop_strength = feedback_loop_strength
        self.temporal_coupling_horizon = temporal_coupling_horizon
        self.causality_tracking_depth = causality_tracking_depth
        self.emergence_noise_factor = emergence_noise_factor
        self.pattern_novelty_threshold = pattern_novelty_threshold
        self.creative_synthesis_rate = creative_synthesis_rate
        self.autopoietic_closure_strength = autopoietic_closure_strength
        self.self_maintenance_rate = self_maintenance_rate
        self.boundary_maintenance_strength = boundary_maintenance_strength


class StructuralCouplingDynamics(eqx.Module):
    """Structural coupling dynamics following Maturana & Varela.
    
    Implements the fundamental mechanism by which living systems
    maintain their organization through recurrent interactions
    with their environment, creating a history of structural
    changes that are triggered but not specified by perturbations.
    """
    
    coupling_matrix: Array
    perturbation_history: Array
    coupling_strength_trace: Array
    adaptation_weights: Array
    current_position: int
    
    def __init__(
        self, 
        agent_dim: int,
        environment_dim: int,
        memory_length: int,
        key: PRNGKey,
    ):
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Bidirectional coupling matrix (agent <-> environment)
        self.coupling_matrix = jax.random.orthogonal(
            key1, (agent_dim + environment_dim, agent_dim + environment_dim)
        ) * 0.1
        
        # History of environmental perturbations
        self.perturbation_history = jnp.zeros((memory_length, environment_dim))
        
        # Trace of coupling strength evolution
        self.coupling_strength_trace = jnp.ones(memory_length) * 0.5
        
        # Adaptive weights for coupling dynamics
        self.adaptation_weights = jax.random.normal(
            key3, (agent_dim, environment_dim)
        ) * 0.05
        
        self.current_position = 0
    
    def compute_circular_causality(
        self,
        agent_state: Array,
        environmental_perturbation: Array,
        coupling_rate: float = 0.05,
        feedback_strength: float = 0.8,
    ) -> Tuple[Array, Array, float, Dict[str, Array]]:
        """Enhanced circular causality computation with explicit feedback loops.
        
        Implements the core enactive principle with enhanced feedback tracking:
        1. Agent influences environment through action
        2. Environment influences agent through perturbation
        3. Feedback loops create recursive causality chains
        4. Temporal dynamics track causality evolution
        """
        # Combine states for joint dynamics computation
        combined_state = jnp.concatenate([agent_state, environmental_perturbation])
        
        # Apply coupling transformation
        coupled_dynamics = self.coupling_matrix @ combined_state
        
        # Split back to agent and environmental components
        new_agent_state = coupled_dynamics[:len(agent_state)]
        environmental_response = coupled_dynamics[len(agent_state):]
        
        # Enhanced circular causality with explicit feedback loops
        
        # 1. Agent → Environment influence
        agent_to_env_influence = self._compute_agent_to_environment_influence(
            agent_state, environmental_perturbation, coupling_rate
        )
        
        # 2. Environment → Agent influence  
        env_to_agent_influence = self._compute_environment_to_agent_influence(
            environmental_perturbation, agent_state, coupling_rate
        )
        
        # 3. Recursive feedback loops
        feedback_dynamics = self._compute_feedback_loops(
            agent_to_env_influence, env_to_agent_influence, feedback_strength
        )
        
        # 4. Apply circular causality with temporal dynamics
        agent_influenced = agent_state + env_to_agent_influence + feedback_dynamics['agent_feedback']
        environment_influenced = environmental_perturbation + agent_to_env_influence + feedback_dynamics['env_feedback']
        
        # Compute enhanced coupling strength including feedback
        base_resonance = jnp.dot(agent_state, environmental_perturbation[:len(agent_state)])
        feedback_resonance = jnp.dot(feedback_dynamics['agent_feedback'], feedback_dynamics['env_feedback'][:len(agent_state)])
        total_resonance = base_resonance + 0.5 * feedback_resonance
        coupling_strength = jax.nn.sigmoid(total_resonance)
        
        # Return additional feedback information for tracking
        feedback_info = {
            'agent_to_env_influence': agent_to_env_influence,
            'env_to_agent_influence': env_to_agent_influence,
            'feedback_dynamics': feedback_dynamics,
            'total_resonance': total_resonance,
        }
        
        return agent_influenced, environment_influenced, float(coupling_strength), feedback_info
    
    def _compute_agent_to_environment_influence(
        self, agent_state: Array, environmental_state: Array, coupling_rate: float
    ) -> Array:
        """Compute how agent influences environment."""
        # Agent's action potential
        action_potential = jnp.tanh(self.adaptation_weights.T @ agent_state)
        
        # Contextual modulation by environment
        context_modulation = jax.nn.sigmoid(jnp.mean(environmental_state))
        
        # Agent's influence on environment
        influence = coupling_rate * context_modulation * action_potential
        return influence
    
    def _compute_environment_to_agent_influence(
        self, environmental_state: Array, agent_state: Array, coupling_rate: float
    ) -> Array:
        """Compute how environment influences agent."""
        # Environmental affordances
        affordances = jnp.tanh(self.adaptation_weights @ environmental_state)
        
        # Agent's sensitivity to affordances
        sensitivity = jax.nn.sigmoid(jnp.linalg.norm(agent_state) * 0.1)
        
        # Environment's influence on agent
        influence = coupling_rate * sensitivity * affordances
        return influence
    
    def _compute_feedback_loops(
        self, agent_to_env: Array, env_to_agent: Array, feedback_strength: float
    ) -> Dict[str, Array]:
        """Compute recursive feedback loops between influences."""
        # Secondary feedback: agent's influence affects its own next state via environment
        agent_secondary_feedback = feedback_strength * jnp.tanh(
            agent_to_env[:len(env_to_agent)] * 0.5
        )
        
        # Environmental secondary feedback
        env_secondary_feedback = feedback_strength * jnp.tanh(
            env_to_agent[:len(agent_to_env)] * 0.5
        )
        
        return {
            'agent_feedback': agent_secondary_feedback,
            'env_feedback': env_secondary_feedback,
        }
    
    def update_coupling_history(
        self,
        perturbation: Array,
        coupling_strength: float,
    ) -> 'StructuralCouplingDynamics':
        """Update coupling history for temporal coherence."""
        # Update perturbation history
        new_history = jnp.roll(self.perturbation_history, 1, axis=0)
        new_history = new_history.at[0].set(perturbation)
        
        # Update coupling strength trace
        new_strength_trace = jnp.roll(self.coupling_strength_trace, 1)
        new_strength_trace = new_strength_trace.at[0].set(coupling_strength)
        
        # Update position tracker
        new_position = (self.current_position + 1) % self.perturbation_history.shape[0]
        
        return eqx.tree_at(
            lambda x: (x.perturbation_history, x.coupling_strength_trace, x.current_position),
            self,
            (new_history, new_strength_trace, new_position)
        )
    
    def assess_coupling_stability(self) -> float:
        """Assess stability of structural coupling over time."""
        # Compute variance in coupling strength
        strength_variance = jnp.var(self.coupling_strength_trace)
        
        # Compute temporal coherence of perturbations
        perturbation_coherence = jnp.mean([
            jnp.corrcoef(
                self.perturbation_history[i], 
                self.perturbation_history[i+1]
            )[0, 1]
            for i in range(self.perturbation_history.shape[0] - 1)
            if jnp.all(jnp.isfinite(self.perturbation_history[i])) and 
               jnp.all(jnp.isfinite(self.perturbation_history[i+1]))
        ])
        
        # Combine measures (low variance + high coherence = high stability)
        stability = (1.0 / (1.0 + strength_variance)) * jnp.clip(perturbation_coherence, 0.0, 1.0)
        return float(jnp.clip(stability, 0.0, 1.0))


class MeaningEmergenceNetwork(eqx.Module):
    """Enhanced network for emergent meaning generation through enactive coupling.
    
    Implements meaning as truly emergent from the ongoing structural
    coupling between agent and environment. Enhanced with:
    1. Creative synthesis mechanisms for novel meaning generation
    2. Pattern novelty detection for emergence recognition
    3. Noise-driven creativity for unpredictable meaning patterns
    4. Temporal meaning evolution tracking
    """
    
    emergence_encoder: eqx.nn.MLP
    contextual_processor: eqx.nn.GRU
    coherence_evaluator: eqx.nn.MLP
    meaning_decoder: eqx.nn.Linear
    
    # Enhanced components for true emergence
    creative_synthesizer: eqx.nn.MLP
    novelty_detector: eqx.nn.MLP
    pattern_memory: Array
    emergence_tracker: eqx.nn.GRU
    
    def __init__(
        self,
        coupling_dim: int,
        meaning_dim: int,
        hidden_dim: int,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, 8)
        
        # MLP for encoding emergence patterns
        self.emergence_encoder = eqx.nn.MLP(
            in_size=coupling_dim * 2,  # Agent + environment states
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.gelu,
            key=keys[0],
        )
        
        # GRU for temporal context processing
        self.contextual_processor = eqx.nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            key=keys[1],
        )
        
        # MLP for coherence evaluation
        self.coherence_evaluator = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=1,
            width_size=hidden_dim // 2,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[2],
        )
        
        # Linear decoder for meaning structure
        self.meaning_decoder = eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=meaning_dim,
            key=keys[3],
        )
        
        # Enhanced components for creative meaning emergence
        self.creative_synthesizer = eqx.nn.MLP(
            in_size=hidden_dim * 2,  # Context + noise
            out_size=hidden_dim,
            width_size=hidden_dim * 3 // 2,
            depth=3,
            activation=jax.nn.swish,  # Swish for better gradient flow
            key=keys[4],
        )
        
        # Novelty detector for pattern recognition
        self.novelty_detector = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=1,  # Novelty score
            width_size=hidden_dim // 2,
            depth=2,
            activation=jax.nn.sigmoid,
            key=keys[5],
        )
        
        # Memory for pattern tracking
        self.pattern_memory = jnp.zeros((100, hidden_dim))  # Store recent patterns
        
        # GRU for tracking emergence evolution
        self.emergence_tracker = eqx.nn.GRU(
            input_size=hidden_dim + 1,  # Features + novelty score
            hidden_size=hidden_dim // 2,
            key=keys[6],
        )
    
    def generate_emergent_meaning(
        self,
        coupled_agent_state: Array,
        coupled_environment_state: Array,
        coupling_history: Array,
        previous_context: Optional[Array] = None,
        emergence_config: Optional[Dict[str, float]] = None,
        noise_key: Optional[PRNGKey] = None,
    ) -> Tuple[MeaningStructure, Array, Dict[str, float]]:
        """Generate truly emergent meaning with creative synthesis and novelty."""
        if emergence_config is None:
            emergence_config = {
                'noise_factor': 0.1,
                'novelty_threshold': 0.7,
                'creativity_strength': 0.3,
            }
        
        if noise_key is None:
            noise_key = jax.random.PRNGKey(42)
        
        # Encode emergence patterns
        coupling_input = jnp.concatenate([coupled_agent_state, coupled_environment_state])
        emergence_features = self.emergence_encoder(coupling_input)
        
        # Process temporal context
        if previous_context is None:
            previous_context = jnp.zeros(self.contextual_processor.hidden_size)
        
        contextual_features, new_context = self.contextual_processor(
            emergence_features, previous_context
        )
        
        # 1. Creative synthesis with controlled noise for emergent properties
        creative_features, novelty_score = self._apply_creative_synthesis(
            contextual_features, emergence_config, noise_key
        )
        
        # 2. Novelty detection and pattern evaluation
        pattern_novelty = self._evaluate_pattern_novelty(creative_features)
        
        # 3. Update pattern memory with novel patterns
        updated_memory = self._update_pattern_memory(creative_features, pattern_novelty)
        
        # 4. Track emergence evolution
        emergence_state, emergence_context = self._track_emergence_evolution(
            creative_features, novelty_score
        )
        
        # 5. Enhanced coherence evaluation considering novelty
        coherence_raw = self.coherence_evaluator(creative_features)
        base_coherence = jax.nn.sigmoid(coherence_raw.squeeze())
        
        # Adjust coherence for creative emergence (novel patterns may have lower initial coherence)
        novelty_adjustment = 1.0 - 0.3 * novelty_score  # Novel meanings start with uncertainty
        adjusted_coherence = float(base_coherence * novelty_adjustment)
        
        # 6. Decode emergent semantic content
        semantic_content = self.meaning_decoder(creative_features)
        
        # 7. Compute enhanced relevance weights including emergence factors
        coupling_strengths = jnp.mean(coupling_history, axis=0)
        base_relevance = jax.nn.softmax(coupling_strengths)
        
        # Enhance relevance with emergence factors
        emergence_boost = 1.0 + emergence_config['creativity_strength'] * novelty_score
        relevance_weight = base_relevance * emergence_boost
        relevance_weight = relevance_weight / jnp.sum(relevance_weight)  # Renormalize
        
        # Create enhanced meaning structure
        meaning_structure = MeaningStructure(
            semantic_content=semantic_content,
            coherence_measure=adjusted_coherence,
            relevance_weight=relevance_weight,
            emergence_timestamp=0.0,  # Will be set by caller
        )
        
        # Update internal state
        self.pattern_memory = updated_memory
        
        # Return emergence metrics for analysis
        emergence_metrics = {
            'novelty_score': float(novelty_score),
            'pattern_novelty': float(pattern_novelty),
            'creativity_contribution': float(emergence_config['creativity_strength'] * novelty_score),
            'coherence_adjustment': float(novelty_adjustment),
        }
        
        return meaning_structure, new_context, emergence_metrics
    
    def _apply_creative_synthesis(
        self, contextual_features: Array, config: Dict[str, float], noise_key: PRNGKey
    ) -> Tuple[Array, float]:
        """Apply creative synthesis with controlled noise for emergence."""
        # Generate creative noise
        noise = jax.random.normal(noise_key, contextual_features.shape) * config['noise_factor']
        
        # Combine context and noise for creative input
        creative_input = jnp.concatenate([contextual_features, noise])
        
        # Apply creative synthesis network
        creative_features = self.creative_synthesizer(creative_input)
        
        # Measure contribution of creative elements
        creativity_magnitude = jnp.linalg.norm(creative_features - contextual_features)
        novelty_score = jax.nn.sigmoid(creativity_magnitude)
        
        return creative_features, float(novelty_score)
    
    def _evaluate_pattern_novelty(self, features: Array) -> float:
        """Evaluate how novel the current pattern is compared to memory."""
        if jnp.allclose(self.pattern_memory, 0.0):
            return 1.0  # First pattern is maximally novel
        
        # Compute similarities to stored patterns
        similarities = jnp.array([
            jnp.dot(features, pattern) / (jnp.linalg.norm(features) * jnp.linalg.norm(pattern) + 1e-8)
            for pattern in self.pattern_memory
            if jnp.linalg.norm(pattern) > 1e-6
        ])
        
        if len(similarities) == 0:
            return 1.0
        
        # Novelty is inverse of maximum similarity
        max_similarity = jnp.max(similarities)
        novelty = 1.0 - max_similarity
        
        return float(jnp.clip(novelty, 0.0, 1.0))
    
    def _update_pattern_memory(self, features: Array, novelty_score: float) -> Array:
        """Update pattern memory with novel patterns."""
        # Only store significantly novel patterns
        if novelty_score > 0.5:
            # Roll memory and add new pattern
            new_memory = jnp.roll(self.pattern_memory, 1, axis=0)
            new_memory = new_memory.at[0].set(features)
            return new_memory
        
        return self.pattern_memory
    
    def _track_emergence_evolution(
        self, features: Array, novelty_score: float
    ) -> Tuple[Array, Array]:
        """Track how emergence evolves over time."""
        # Combine features with novelty for evolution tracking
        evolution_input = jnp.concatenate([features, jnp.array([novelty_score])])
        
        # Initialize evolution context if needed
        evolution_context = jnp.zeros(self.emergence_tracker.hidden_size)
        
        # Track emergence evolution
        emergence_state, new_evolution_context = self.emergence_tracker(
            evolution_input, evolution_context
        )
        
        return emergence_state, new_evolution_context


class AutopoieticSelfReference(eqx.Module):
    """Autopoietic self-referential dynamics for truly autonomous system behavior.
    
    Implements Maturana & Varela's autopoiesis concept:
    1. Operational closure: System maintains its own organization
    2. Self-production: System produces its own components
    3. Structural determinism: System's structure determines responses
    4. Boundary maintenance: System maintains its own boundaries
    5. Identity preservation through self-referential processes
    """
    
    self_reference_layers: List[eqx.nn.Linear]
    identity_maintainer: eqx.nn.MLP
    autonomy_assessor: eqx.nn.MLP
    
    # Autopoietic components
    self_production_network: eqx.nn.MLP
    boundary_maintainer: eqx.nn.MLP
    organization_tracker: eqx.nn.GRU
    closure_detector: eqx.nn.MLP
    
    # Internal state for autopoiesis
    organizational_memory: Array
    boundary_state: Array
    self_production_history: Array
    
    def __init__(
        self,
        state_dim: int,
        reference_depth: int,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, reference_depth + 6)
        
        # Stack of self-reference layers
        self.self_reference_layers = [
            eqx.nn.Linear(state_dim, state_dim, key=keys[i])
            for i in range(reference_depth)
        ]
        
        # MLP for identity maintenance
        self.identity_maintainer = eqx.nn.MLP(
            in_size=state_dim * reference_depth,
            out_size=state_dim,
            width_size=state_dim,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[reference_depth],
        )
        
        # MLP for autonomy assessment
        self.autonomy_assessor = eqx.nn.MLP(
            in_size=state_dim,
            out_size=1,
            width_size=state_dim // 2,
            depth=2,
            activation=jax.nn.sigmoid,
            key=keys[reference_depth + 1],
        )
        
        # Autopoietic networks
        self.self_production_network = eqx.nn.MLP(
            in_size=state_dim * 2,  # Current state + organizational memory
            out_size=state_dim,
            width_size=state_dim * 3 // 2,
            depth=3,
            activation=jax.nn.swish,
            key=keys[reference_depth + 2],
        )
        
        self.boundary_maintainer = eqx.nn.MLP(
            in_size=state_dim + state_dim // 2,  # State + boundary state
            out_size=state_dim // 2,  # Boundary representation
            width_size=state_dim,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[reference_depth + 3],
        )
        
        self.organization_tracker = eqx.nn.GRU(
            input_size=state_dim,
            hidden_size=state_dim // 2,
            key=keys[reference_depth + 4],
        )
        
        self.closure_detector = eqx.nn.MLP(
            in_size=state_dim + state_dim // 2,  # State + organizational memory
            out_size=1,  # Closure measure
            width_size=state_dim // 2,
            depth=2,
            activation=jax.nn.sigmoid,
            key=keys[reference_depth + 5],
        )
        
        # Initialize autopoietic state
        self.organizational_memory = jnp.zeros(state_dim // 2)
        self.boundary_state = jnp.ones(state_dim // 2) * 0.5  # Neutral boundary
        self.self_production_history = jnp.zeros((10, state_dim))
    
    def process_autopoietic_dynamics(
        self,
        current_state: Array,
        coupling_influence: Array,
        environmental_perturbation: Array,
    ) -> Tuple[Array, float, Dict[str, Any]]:
        """Process full autopoietic dynamics with self-production and boundary maintenance."""
        # 1. Self-referential processing (basic level)
        reference_states = []
        state = current_state
        
        for layer in self.self_reference_layers:
            state = jax.nn.tanh(layer(state) + coupling_influence * 0.1)
            reference_states.append(state)
        
        integrated_references = jnp.concatenate(reference_states)
        maintained_identity = self.identity_maintainer(integrated_references)
        
        # 2. Autopoietic self-production
        self_produced_state, production_metrics = self._perform_self_production(
            maintained_identity, coupling_influence
        )
        
        # 3. Boundary maintenance and detection
        boundary_maintained, boundary_metrics = self._maintain_system_boundary(
            self_produced_state, environmental_perturbation
        )
        
        # 4. Operational closure assessment
        closure_level, closure_metrics = self._assess_operational_closure(
            boundary_maintained
        )
        
        # 5. Update organizational memory
        updated_memory, organization_metrics = self._update_organizational_memory(
            boundary_maintained
        )
        
        # 6. Comprehensive autonomy assessment
        autonomy_components = {
            'identity_preservation': float(jnp.linalg.norm(maintained_identity - current_state)),
            'self_production_strength': production_metrics['production_strength'],
            'boundary_integrity': boundary_metrics['boundary_strength'],
            'operational_closure': closure_level,
        }
        
        # Weighted autonomy score
        autonomy_weights = [0.25, 0.25, 0.25, 0.25]
        autonomy_level = sum(w * v for w, v in zip(autonomy_weights, autonomy_components.values()))
        autonomy_level = float(jnp.clip(autonomy_level, 0.0, 1.0))
        
        # Update internal state
        self.organizational_memory = updated_memory
        self.boundary_state = boundary_metrics['new_boundary_state']
        
        # Update production history
        new_history = jnp.roll(self.self_production_history, 1, axis=0)
        self.self_production_history = new_history.at[0].set(self_produced_state)
        
        # Comprehensive metrics
        autopoietic_metrics = {
            'autonomy_components': autonomy_components,
            'production_metrics': production_metrics,
            'boundary_metrics': boundary_metrics,
            'closure_metrics': closure_metrics,
            'organization_metrics': organization_metrics,
        }
        
        return boundary_maintained, autonomy_level, autopoietic_metrics
    
    def _perform_self_production(
        self, current_state: Array, coupling_influence: Array
    ) -> Tuple[Array, Dict[str, float]]:
        """Perform autopoietic self-production of system components."""
        # Combine current state with organizational memory for self-production
        production_input = jnp.concatenate([current_state, self.organizational_memory])
        
        # Generate self-produced components
        self_produced_components = self.self_production_network(production_input)
        
        # Measure production strength (how much the system changes itself)
        production_strength = float(jnp.linalg.norm(
            self_produced_components - current_state
        ) / (jnp.linalg.norm(current_state) + 1e-8))
        
        # Apply self-production with coupling modulation
        coupling_modulation = jax.nn.sigmoid(jnp.mean(coupling_influence))
        modulated_production = (
            current_state * (1.0 - coupling_modulation * 0.5) +
            self_produced_components * coupling_modulation * 0.5
        )
        
        metrics = {
            'production_strength': production_strength,
            'coupling_modulation': float(coupling_modulation),
        }
        
        return modulated_production, metrics
    
    def _maintain_system_boundary(
        self, current_state: Array, environmental_perturbation: Array
    ) -> Tuple[Array, Dict[str, Any]]:
        """Maintain system boundaries against environmental perturbation."""
        # Combine state with current boundary state
        boundary_input = jnp.concatenate([current_state, self.boundary_state])
        
        # Generate boundary maintenance response
        new_boundary_state = self.boundary_maintainer(boundary_input)
        
        # Compute boundary strength (resistance to perturbation)
        perturbation_magnitude = jnp.linalg.norm(environmental_perturbation)
        boundary_strength = float(jax.nn.sigmoid(
            jnp.linalg.norm(new_boundary_state) - perturbation_magnitude * 0.1
        ))
        
        # Apply boundary filtering to state
        boundary_filter = jax.nn.sigmoid(new_boundary_state * 2.0)  # Strong filtering
        filtered_perturbation = environmental_perturbation[:len(boundary_filter)] * (1.0 - boundary_filter)
        
        # Maintain state with boundary protection
        boundary_maintained_state = current_state + filtered_perturbation * 0.1
        
        metrics = {
            'boundary_strength': boundary_strength,
            'perturbation_filtered': float(jnp.linalg.norm(filtered_perturbation)),
            'new_boundary_state': new_boundary_state,
        }
        
        return boundary_maintained_state, metrics
    
    def _assess_operational_closure(
        self, current_state: Array
    ) -> Tuple[float, Dict[str, float]]:
        """Assess the operational closure of the autopoietic system."""
        # Combine state with organizational memory
        closure_input = jnp.concatenate([current_state, self.organizational_memory])
        
        # Detect operational closure level
        closure_raw = self.closure_detector(closure_input)
        closure_level = float(jax.nn.sigmoid(closure_raw.squeeze()))
        
        # Additional closure metrics
        self_consistency = float(1.0 / (1.0 + jnp.std(current_state)))
        organizational_coherence = float(jnp.dot(
            current_state[:len(self.organizational_memory)], 
            self.organizational_memory
        ) / (jnp.linalg.norm(current_state[:len(self.organizational_memory)]) * 
             jnp.linalg.norm(self.organizational_memory) + 1e-8))
        
        metrics = {
            'closure_level': closure_level,
            'self_consistency': self_consistency,
            'organizational_coherence': organizational_coherence,
        }
        
        return closure_level, metrics
    
    def _update_organizational_memory(
        self, current_state: Array
    ) -> Tuple[Array, Dict[str, float]]:
        """Update organizational memory with current system dynamics."""
        # Track organizational changes
        organization_input = current_state
        
        new_organizational_state, _ = self.organization_tracker(
            organization_input, self.organizational_memory
        )
        
        # Measure organizational stability
        stability = float(1.0 - jnp.linalg.norm(
            new_organizational_state - self.organizational_memory
        ) / (jnp.linalg.norm(self.organizational_memory) + 1e-8))
        
        # Adaptive update rate based on stability
        update_rate = 0.1 * (1.0 - stability)  # Update more when less stable
        updated_memory = (
            (1.0 - update_rate) * self.organizational_memory +
            update_rate * new_organizational_state
        )
        
        metrics = {
            'organizational_stability': stability,
            'update_rate': float(update_rate),
        }
        
        return updated_memory, metrics


class EnactiveCouplingProcessor(ProcessorBase, StateValidationMixin, ConfigurableMixin):
    """Main processor for enactive coupling and circular causality.
    
    Integrates structural coupling dynamics, meaning emergence,
    and self-referential processes to implement Varela's enactive
    approach to cognition and consciousness.
    
    Refactored following Martin Fowler's principles:
    - Extract Method for complex coupling computations
    - Replace Temp with Query for stability metrics
    - Introduce Parameter Object for coupling context
    """
    
    config: EnactiveCouplingConfig
    coupling_dynamics: StructuralCouplingDynamics
    meaning_network: MeaningEmergenceNetwork
    self_reference: AutopoieticSelfReference
    
    def __init__(
        self,
        config: EnactiveCouplingConfig,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, 4)
        
        self.config = config
        
        # Initialize coupling dynamics
        self.coupling_dynamics = StructuralCouplingDynamics(
            config.agent_state_dim,
            config.environment_state_dim,
            config.coupling_memory_length,
            keys[0],
        )
        
        # Initialize meaning emergence network
        hidden_dim = max(64, config.agent_state_dim // 2)
        self.meaning_network = MeaningEmergenceNetwork(
            config.agent_state_dim + config.environment_state_dim,
            config.agent_state_dim,
            hidden_dim,
            keys[1],
        )
        
        # Initialize self-referential dynamics
        self.self_reference = AutopoieticSelfReference(
            config.agent_state_dim,
            config.self_reference_depth,
            keys[2],
        )
        
        # Note: ProcessorBase components initialized when needed
        
        logger.info("EnactiveCouplingProcessor initialized successfully")
    
    @optimize_for_memory
    def compute_coupling_dynamics(
        self,
        agent_state: Array,
        environmental_perturbation: Array,
        timestamp: Optional[TimeStep] = None,
    ) -> CouplingState:
        """Compute complete enactive coupling dynamics.
        
        Implements the full enactive cycle:
        1. Structural coupling computation
        2. Circular causality dynamics
        3. Meaning emergence
        4. Self-referential processing
        
        Refactored into smaller methods for clarity.
        """
        with GLOBAL_MEMORY_MANAGER.track_memory("enactive_coupling"):
            try:
                # Step 1: Validate inputs
                self._validate_coupling_inputs(agent_state, environmental_perturbation)
                
                # Step 2: Compute circular causality
                coupling_results = self._compute_circular_causality(
                    agent_state, environmental_perturbation
                )
                
                # Step 3: Process self-referential dynamics
                identity_maintained, autonomy_level = self._process_self_referential_dynamics(
                    coupling_results['coupled_agent'], coupling_results['environmental_influence']
                )
                
                # Step 4: Update coupling history and assess stability
                coupling_stability = self._update_and_assess_coupling_stability(
                    environmental_perturbation, coupling_results['coupling_strength']
                )
                
                # Step 5: Create and return coupling state
                return self._create_coupling_state(
                    identity_maintained,
                    coupling_results['coupled_environment'],
                    coupling_results['coupling_strength'],
                    environmental_perturbation,
                    coupling_stability,
                )
                
            except Exception as e:
                raise CouplingError(f"Failed to compute coupling dynamics: {e}")
    
    def process(self, *args, **kwargs) -> Any:
        """Implementation of ProcessorBase abstract method."""
        return self.compute_coupling_dynamics(*args, **kwargs)
    
    def _validate_coupling_inputs(self, agent_state: Array, environmental_perturbation: Array) -> None:
        """Extract method: Validate coupling computation inputs."""
        self.validate_input_state(agent_state, "agent_state")
        self.validate_input_state(environmental_perturbation, "environmental_perturbation")
    
    def _compute_circular_causality(
        self, agent_state: Array, environmental_perturbation: Array
    ) -> Dict[str, Any]:
        """Extract method: Compute circular causality dynamics."""
        coupled_agent, coupled_environment, coupling_strength = (
            self.coupling_dynamics.compute_circular_causality(
                agent_state, environmental_perturbation, self.config.circular_causality_rate
            )
        )
        return {
            'coupled_agent': coupled_agent,
            'coupled_environment': coupled_environment,
            'coupling_strength': coupling_strength,
            'environmental_influence': coupled_environment - environmental_perturbation,
        }
    
    def _process_self_referential_dynamics(
        self, coupled_agent_state: Array, environmental_influence: Array
    ) -> Tuple[Array, float]:
        """Extract method: Process self-referential dynamics."""
        identity_maintained, autonomy_level, _ = self.self_reference.process_autopoietic_dynamics(
            coupled_agent_state, environmental_influence, jnp.zeros_like(environmental_influence)
        )
        return identity_maintained, autonomy_level
    
    def _update_and_assess_coupling_stability(
        self, environmental_perturbation: Array, coupling_strength: float
    ) -> float:
        """Extract method: Update coupling history and assess stability."""
        self.coupling_dynamics = self.coupling_dynamics.update_coupling_history(
            environmental_perturbation, coupling_strength
        )
        return self.coupling_dynamics.assess_coupling_stability()
    
    def _create_coupling_state(
        self,
        agent_state: Array,
        environmental_state: Array,
        coupling_strength: float,
        perturbation_history: Array,
        stability_metric: float,
    ) -> CouplingState:
        """Extract method: Create validated coupling state."""
        return CouplingState(
            agent_state=self.validate_output_state(agent_state, "agent_state"),
            environmental_state=environmental_state,
            coupling_strength=coupling_strength,
            perturbation_history=perturbation_history,
            stability_metric=stability_metric,
        )
    
    def generate_emergent_meaning(
        self,
        coupling_state: CouplingState,
        coupling_history: Optional[Array] = None,
        timestamp: Optional[TimeStep] = None,
    ) -> MeaningStructure:
        """Generate emergent meaning from coupling dynamics."""
        try:
            # Use coupling history from dynamics if not provided
            if coupling_history is None:
                coupling_history = self.coupling_dynamics.perturbation_history
            
            # Generate meaning through emergence network
            meaning_structure, _ = self.meaning_network.generate_emergent_meaning(
                coupling_state.agent_state,
                coupling_state.environmental_state,
                coupling_history,
            )
            
            # Set emergence timestamp
            if timestamp is not None:
                meaning_structure = eqx.tree_at(
                    lambda x: x.emergence_timestamp,
                    meaning_structure,
                    timestamp,
                )
            
            return meaning_structure
            
        except Exception as e:
            raise SenseMakingError(f"Failed to generate emergent meaning: {e}")
    
    def assess_enactive_quality(self, coupling_state: CouplingState) -> Dict[str, float]:
        """Assess quality of enactive processing."""
        # Circular causality strength
        circularity = coupling_state.coupling_strength
        
        # Structural coherence
        agent_coherence = 1.0 / (1.0 + jnp.std(coupling_state.agent_state))
        env_coherence = 1.0 / (1.0 + jnp.std(coupling_state.environmental_state))
        structural_coherence = float((agent_coherence + env_coherence) / 2.0)
        
        # Coupling stability
        coupling_stability = coupling_state.stability_metric
        
        # Perturbation responsiveness
        perturbation_magnitude = float(jnp.linalg.norm(coupling_state.perturbation_history))
        responsiveness = perturbation_magnitude / (1.0 + perturbation_magnitude)
        
        # Overall enactive score
        enactive_score = (
            0.3 * circularity +
            0.25 * structural_coherence +
            0.25 * coupling_stability +
            0.2 * responsiveness
        )
        
        return {
            "circular_causality_strength": circularity,
            "structural_coherence": structural_coherence,
            "coupling_stability": coupling_stability,
            "perturbation_responsiveness": responsiveness,
            "overall_enactive_quality": enactive_score,
        }
    
    @property
    def _coupling_strength_threshold(self) -> float:
        """Replace temp with query: Get coupling strength threshold."""
        return self.config.coupling_strength_initial * 0.8
    
    def detect_meaning_emergence_events(
        self,
        coupling_history: List[CouplingState],
        meaning_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Detect events where meaning emerges from coupling dynamics."""
        if meaning_threshold is None:
            meaning_threshold = self.config.meaning_emergence_threshold
        
        emergence_events = []
        
        for i, coupling_state in enumerate(coupling_history):
            # Check if coupling strength exceeds threshold
            if coupling_state.coupling_strength > self._coupling_strength_threshold:
                # Generate meaning for this state
                try:
                    meaning = self.generate_emergent_meaning(coupling_state, timestamp=float(i))
                    
                    if meaning.coherence_measure > meaning_threshold:
                        emergence_events.append({
                            'timestamp': i,
                            'coupling_strength': coupling_state.coupling_strength,
                            'meaning_coherence': meaning.coherence_measure,
                            'stability_metric': coupling_state.stability_metric,
                            'meaning_structure': meaning,
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to generate meaning at timestamp {i}: {e}")
                    continue
        
        return emergence_events


@functools.partial(jax.jit, static_argnames=['config'])
def create_enactive_coupling_processor(
    config: EnactiveCouplingConfig,
    key: PRNGKey,
) -> EnactiveCouplingProcessor:
    """Factory function for JIT-compiled enactive coupling processor."""
    return EnactiveCouplingProcessor(config, key)


# Utility functions for enactive analysis
def analyze_coupling_coherence(
    coupling_sequence: List[CouplingState],
) -> Dict[str, float]:
    """Analyze coherence of coupling dynamics over time."""
    if len(coupling_sequence) < 2:
        return {"temporal_coherence": 0.0, "stability_coherence": 0.0, "overall_coherence": 0.0}
    
    # Temporal coherence: consistency of coupling strength evolution
    coupling_strengths = jnp.array([state.coupling_strength for state in coupling_sequence])
    temporal_coherence = 1.0 - float(jnp.std(coupling_strengths))
    temporal_coherence = jnp.clip(temporal_coherence, 0.0, 1.0)
    
    # Stability coherence: consistency of stability metrics
    stability_metrics = jnp.array([state.stability_metric for state in coupling_sequence])
    stability_coherence = float(jnp.mean(stability_metrics))
    
    # Overall coherence
    overall_coherence = float((temporal_coherence + stability_coherence) / 2.0)
    
    return {
        "temporal_coherence": float(temporal_coherence),
        "stability_coherence": stability_coherence,
        "overall_coherence": overall_coherence,
    }


def create_test_coupling_sequence(
    processor: EnactiveCouplingProcessor,
    agent_sequence: List[Array],
    environment_sequence: List[Array],
    key: PRNGKey,
) -> List[CouplingState]:
    """Create sequence of coupling states for testing."""
    if len(agent_sequence) != len(environment_sequence):
        raise ValueError("Agent and environment sequences must have same length")
    
    coupling_states = []
    
    for i, (agent_state, env_state) in enumerate(zip(agent_sequence, environment_sequence)):
        coupling_state = processor.compute_coupling_dynamics(
            agent_state, env_state, timestamp=float(i)
        )
        coupling_states.append(coupling_state)
    
    return coupling_states


# Export public API
__all__ = [
    'EnactiveCouplingConfig',
    'StructuralCouplingDynamics',
    'MeaningEmergenceNetwork',
    'AutopoieticSelfReference',
    'EnactiveCouplingProcessor',
    'create_enactive_coupling_processor',
    'analyze_coupling_coherence',
    'create_test_coupling_sequence',
]