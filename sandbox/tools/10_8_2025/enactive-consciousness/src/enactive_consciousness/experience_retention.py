"""Experience retention and sedimentation system.

This module implements experience retention mechanisms following
phenomenological and enactive principles, featuring:
1. Experiential sedimentation (building up experience layers)
2. Associative recall (retrieving related experiences)
3. Temporal context preservation (maintaining time-bound experience)

Key implementation principles:
- Experience as temporal flow rather than static representations
- Associative networks for meaning-based recall
- Multi-scale temporal organization
- Functional equivalence avoiding direct qualia implementation
"""

from __future__ import annotations

import functools
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

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
    TimeStep,
    TemporalMoment,
    MeaningStructure,
    ConsciousnessLevel,
    EnactiveConsciousnessError,
)


class ExperienceRetentionConfig(eqx.Module):
    """Configuration for experience retention system.
    
    Parameters for experiential sedimentation, associative recall,
    and temporal context preservation following phenomenological
    and enactive principles.
    """
    
    experience_dim: int
    sedimentation_layers: int
    associative_network_size: int
    temporal_context_depth: int
    retention_decay_rate: float
    association_threshold: float
    sedimentation_rate: float
    recall_sensitivity: float
    context_integration_strength: float
    
    def __init__(
        self,
        experience_dim: int = 256,
        sedimentation_layers: int = 5,
        associative_network_size: int = 1000,
        temporal_context_depth: int = 50,
        retention_decay_rate: float = 0.95,
        association_threshold: float = 0.6,
        sedimentation_rate: float = 0.01,
        recall_sensitivity: float = 0.8,
        context_integration_strength: float = 0.3,
    ):
        self.experience_dim = experience_dim
        self.sedimentation_layers = sedimentation_layers
        self.associative_network_size = associative_network_size
        self.temporal_context_depth = temporal_context_depth
        self.retention_decay_rate = retention_decay_rate
        self.association_threshold = association_threshold
        self.sedimentation_rate = sedimentation_rate
        self.recall_sensitivity = recall_sensitivity
        self.context_integration_strength = context_integration_strength


@eqx.dataclass
class ExperienceTrace:
    """Individual experience trace with temporal and associative data."""
    
    content: Array
    temporal_signature: Array
    associative_links: Array
    sedimentation_level: float
    creation_timestamp: TimeStep
    last_access_timestamp: TimeStep
    access_frequency: int
    
    def update_access(self, current_time: TimeStep) -> 'ExperienceTrace':
        """Update access information for experience trace."""
        return eqx.tree_at(
            lambda x: (x.last_access_timestamp, x.access_frequency),
            self,
            (current_time, self.access_frequency + 1)
        )
    
    def compute_accessibility(self, current_time: TimeStep, decay_rate: float = 0.95) -> float:
        """Compute how accessible this experience is at current time."""
        time_decay = jnp.power(decay_rate, current_time - self.last_access_timestamp)
        frequency_boost = 1.0 + jnp.log(1.0 + self.access_frequency) * 0.1
        sedimentation_weight = self.sedimentation_level
        
        return float(time_decay * frequency_boost * sedimentation_weight)


class ExperienceSedimentation(eqx.Module):
    """Experience sedimentation system implementing phenomenological layering.
    
    Experiences accumulate in layers, with deeper layers representing
    more fundamental, deeply sedimented experiences that influence
    ongoing experience formation.
    """
    
    sedimentation_layers: List[Array]
    layer_weights: Array
    integration_network: eqx.nn.MLP
    sedimentation_tracker: Array
    
    def __init__(
        self,
        experience_dim: int,
        num_layers: int,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, num_layers + 2)
        
        # Initialize sedimentation layers
        self.sedimentation_layers = [
            jnp.zeros((experience_dim,)) for _ in range(num_layers)
        ]
        
        # Weights for layer integration (deeper layers have higher weights)
        layer_indices = jnp.arange(num_layers, dtype=jnp.float32)
        self.layer_weights = jax.nn.softmax(layer_indices * 0.5)
        
        # Network for integrating across layers
        self.integration_network = eqx.nn.MLP(
            in_size=experience_dim * num_layers,
            out_size=experience_dim,
            width_size=experience_dim,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[-2],
        )
        
        # Tracker for sedimentation levels
        self.sedimentation_tracker = jnp.zeros(num_layers)
    
    def sediment_experience(
        self,
        new_experience: Array,
        sedimentation_rate: float = 0.01,
    ) -> 'ExperienceSedimentation':
        """Sediment new experience into appropriate layers."""
        # Compute similarity with existing layers
        layer_similarities = [
            jnp.dot(new_experience, layer) / (jnp.linalg.norm(new_experience) * jnp.linalg.norm(layer) + 1e-8)
            for layer in self.sedimentation_layers
        ]
        
        # Find most similar layer for primary sedimentation
        most_similar_idx = jnp.argmax(jnp.array(layer_similarities))
        
        # Update layers with experience sedimentation
        new_layers = []
        new_tracker = self.sedimentation_tracker.copy()
        
        for i, layer in enumerate(self.sedimentation_layers):
            if i == most_similar_idx:
                # Primary sedimentation in most similar layer
                sedimented_layer = layer + sedimentation_rate * new_experience
                new_tracker = new_tracker.at[i].add(sedimentation_rate)
            else:
                # Secondary sedimentation based on similarity
                similarity = layer_similarities[i]
                secondary_rate = sedimentation_rate * jnp.maximum(similarity, 0.0) * 0.1
                sedimented_layer = layer + secondary_rate * new_experience
                new_tracker = new_tracker.at[i].add(secondary_rate)
            
            new_layers.append(sedimented_layer)
        
        return eqx.tree_at(
            lambda x: (x.sedimentation_layers, x.sedimentation_tracker),
            self,
            (new_layers, new_tracker),
        )
    
    def retrieve_sedimented_context(self, query: Array) -> Array:
        """Retrieve sedimented context relevant to query."""
        # Compute weighted sum of layers based on query relevance
        layer_activations = []
        
        for i, layer in enumerate(self.sedimentation_layers):
            # Compute relevance of layer to query
            relevance = jnp.dot(query, layer) / (jnp.linalg.norm(query) * jnp.linalg.norm(layer) + 1e-8)
            activation = jax.nn.sigmoid(relevance) * self.layer_weights[i]
            layer_activations.append(activation * layer)
        
        # Integrate across layers
        stacked_layers = jnp.concatenate(layer_activations)
        integrated_context = self.integration_network(stacked_layers)
        
        return integrated_context
    
    def get_sedimentation_depth(self) -> float:
        """Get overall sedimentation depth measure."""
        return float(jnp.sum(self.sedimentation_tracker))


class AssociativeRecallNetwork(eqx.Module):
    """Associative network for experience recall.
    
    Implements associative recall based on content similarity,
    temporal proximity, and meaning coherence, following
    principles of phenomenological and enactive cognition.
    """
    
    experience_embeddings: Array
    associative_weights: Array
    temporal_proximity_matrix: Array
    recall_network: eqx.nn.MLP
    attention_mechanism: eqx.nn.MultiheadAttention
    
    def __init__(
        self,
        experience_dim: int,
        network_size: int,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, 4)
        
        # Initialize experience embedding matrix
        self.experience_embeddings = jax.random.normal(
            keys[0], (network_size, experience_dim)
        ) * 0.1
        
        # Initialize associative weight matrix
        self.associative_weights = jax.random.normal(
            keys[1], (network_size, network_size)
        ) * 0.05
        
        # Initialize temporal proximity matrix
        self.temporal_proximity_matrix = jnp.eye(network_size)
        
        # Network for recall processing
        self.recall_network = eqx.nn.MLP(
            in_size=experience_dim * 2,  # Query + retrieved
            out_size=experience_dim,
            width_size=experience_dim,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[2],
        )
        
        # Attention mechanism for selective recall
        self.attention_mechanism = eqx.nn.MultiheadAttention(
            num_heads=4,
            query_size=experience_dim,
            key_size=experience_dim,
            value_size=experience_dim,
            output_size=experience_dim,
            key=keys[3],
        )
    
    def store_experience(
        self,
        experience: Array,
        position: int,
        temporal_context: Array,
    ) -> 'AssociativeRecallNetwork':
        """Store new experience in associative network."""
        # Update experience embedding
        new_embeddings = self.experience_embeddings.at[position].set(experience)
        
        # Update associative weights based on similarity with other experiences
        similarities = jnp.dot(self.experience_embeddings, experience)
        similarities = similarities / (jnp.linalg.norm(self.experience_embeddings, axis=1) + 1e-8)
        
        # Update associative connections
        new_weights = self.associative_weights.at[position].set(similarities)
        new_weights = new_weights.at[:, position].set(similarities)
        
        return eqx.tree_at(
            lambda x: (x.experience_embeddings, x.associative_weights),
            self,
            (new_embeddings, new_weights),
        )
    
    def associative_recall(
        self,
        query: Array,
        recall_threshold: float = 0.6,
        max_recalls: int = 5,
    ) -> Tuple[List[Array], List[float]]:
        """Perform associative recall based on query."""
        # Compute similarities with all stored experiences
        similarities = jnp.dot(self.experience_embeddings, query)
        similarities = similarities / (jnp.linalg.norm(self.experience_embeddings, axis=1) + 1e-8)
        
        # Apply associative weights to enhance related recalls
        associative_boost = jnp.mean(self.associative_weights, axis=1)
        enhanced_similarities = similarities * (1.0 + associative_boost * 0.5)
        
        # Find experiences above threshold
        above_threshold = enhanced_similarities > recall_threshold
        candidate_indices = jnp.where(above_threshold, size=max_recalls, fill_value=-1)[0]
        
        # Sort by similarity
        valid_indices = candidate_indices[candidate_indices >= 0]
        if len(valid_indices) == 0:
            return [], []
        
        sorted_indices = valid_indices[jnp.argsort(enhanced_similarities[valid_indices])[::-1]]
        
        # Retrieve experiences and similarities
        recalled_experiences = [self.experience_embeddings[idx] for idx in sorted_indices]
        recall_strengths = [float(enhanced_similarities[idx]) for idx in sorted_indices]
        
        return recalled_experiences, recall_strengths
    
    def update_temporal_proximity(
        self,
        index1: int,
        index2: int,
        proximity_strength: float,
    ) -> 'AssociativeRecallNetwork':
        """Update temporal proximity between experiences."""
        new_matrix = self.temporal_proximity_matrix.at[index1, index2].set(proximity_strength)
        new_matrix = new_matrix.at[index2, index1].set(proximity_strength)
        
        return eqx.tree_at(
            lambda x: x.temporal_proximity_matrix,
            self,
            new_matrix,
        )


class TemporalContextPreserver(eqx.Module):
    """Temporal context preservation system.
    
    Maintains temporal context of experiences across multiple
    time scales, implementing phenomenological temporal synthesis
    with enactive coupling to environmental temporal flow.
    """
    
    context_buffer: Array
    temporal_weights: Array
    context_encoder: eqx.nn.GRU
    context_decoder: eqx.nn.MLP
    temporal_attention: eqx.nn.MultiheadAttention
    
    def __init__(
        self,
        experience_dim: int,
        context_depth: int,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, 4)
        
        # Initialize context buffer
        self.context_buffer = jnp.zeros((context_depth, experience_dim))
        
        # Temporal weights for context integration
        temporal_indices = jnp.arange(context_depth, dtype=jnp.float32)
        self.temporal_weights = jnp.exp(-temporal_indices * 0.1)  # Exponential decay
        
        # GRU for temporal context encoding
        self.context_encoder = eqx.nn.GRU(
            input_size=experience_dim,
            hidden_size=experience_dim,
            key=keys[0],
        )
        
        # MLP for context decoding
        self.context_decoder = eqx.nn.MLP(
            in_size=experience_dim * 2,  # Current + context
            out_size=experience_dim,
            width_size=experience_dim,
            depth=2,
            activation=jax.nn.tanh,
            key=keys[1],
        )
        
        # Attention for temporal context selection
        self.temporal_attention = eqx.nn.MultiheadAttention(
            num_heads=4,
            query_size=experience_dim,
            key_size=experience_dim,
            value_size=experience_dim,
            output_size=experience_dim,
            key=keys[2],
        )
    
    def update_temporal_context(
        self,
        new_experience: Array,
        previous_state: Optional[Array] = None,
    ) -> Tuple['TemporalContextPreserver', Array]:
        """Update temporal context with new experience."""
        # Update context buffer
        new_buffer = jnp.roll(self.context_buffer, 1, axis=0)
        new_buffer = new_buffer.at[0].set(new_experience)
        
        # Encode temporal context
        if previous_state is None:
            previous_state = jnp.zeros(self.context_encoder.hidden_size)
        
        context_encoding, new_state = self.context_encoder(new_experience, previous_state)
        
        # Apply temporal attention to context buffer
        attended_context, _ = self.temporal_attention(
            context_encoding[None, None, :],  # Query: current encoding
            new_buffer[None, :, :],           # Key: context buffer
            new_buffer[None, :, :],           # Value: context buffer
        )
        attended_context = attended_context.squeeze()
        
        # Update instance
        updated_instance = eqx.tree_at(
            lambda x: x.context_buffer,
            self,
            new_buffer,
        )
        
        return updated_instance, attended_context
    
    def retrieve_temporal_context(
        self,
        query: Array,
        temporal_horizon: int = 10,
    ) -> Array:
        """Retrieve relevant temporal context for query."""
        # Compute relevance of context entries to query
        context_subset = self.context_buffer[:temporal_horizon]
        relevances = jnp.array([
            jnp.dot(query, context) / (jnp.linalg.norm(query) * jnp.linalg.norm(context) + 1e-8)
            for context in context_subset
        ])
        
        # Weight by relevance and temporal decay
        temporal_subset_weights = self.temporal_weights[:temporal_horizon]
        combined_weights = jax.nn.softmax(relevances) * temporal_subset_weights
        
        # Compute weighted context
        weighted_context = jnp.sum(
            context_subset * combined_weights[:, None], axis=0
        )
        
        return weighted_context
    
    def assess_temporal_continuity(self) -> float:
        """Assess temporal continuity of preserved context."""
        # Compute correlations between adjacent context entries
        correlations = []
        for i in range(self.context_buffer.shape[0] - 1):
            if jnp.linalg.norm(self.context_buffer[i]) > 1e-6 and jnp.linalg.norm(self.context_buffer[i+1]) > 1e-6:
                corr = jnp.dot(self.context_buffer[i], self.context_buffer[i+1])
                corr = corr / (jnp.linalg.norm(self.context_buffer[i]) * jnp.linalg.norm(self.context_buffer[i+1]))
                correlations.append(corr)
        
        if len(correlations) == 0:
            return 0.0
        
        # Return mean correlation as continuity measure
        return float(jnp.mean(jnp.array(correlations)))


class ExperienceRetentionSystem(ProcessorBase, StateValidationMixin, ConfigurableMixin):
    """Integrated experience retention system.
    
    Combines experiential sedimentation, associative recall,
    and temporal context preservation into unified system
    following phenomenological and enactive principles.
    
    Refactored following Martin Fowler's principles:
    - Extract Method for complex retention operations
    - Replace Temp with Query for accessibility computations
    - Introduce Parameter Object for retention context
    """
    
    config: ExperienceRetentionConfig
    sedimentation: ExperienceSedimentation
    associative_network: AssociativeRecallNetwork
    temporal_context: TemporalContextPreserver
    experience_traces: List[ExperienceTrace]
    current_position: int
    
    def __init__(
        self,
        config: ExperienceRetentionConfig,
        key: PRNGKey,
    ):
        keys = jax.random.split(key, 4)
        
        self.config = config
        
        # Initialize components
        self.sedimentation = ExperienceSedimentation(
            config.experience_dim,
            config.sedimentation_layers,
            keys[0],
        )
        
        self.associative_network = AssociativeRecallNetwork(
            config.experience_dim,
            config.associative_network_size,
            keys[1],
        )
        
        self.temporal_context = TemporalContextPreserver(
            config.experience_dim,
            config.temporal_context_depth,
            keys[2],
        )
        
        # Initialize experience traces storage
        self.experience_traces = []
        self.current_position = 0
        
        logger.info("ExperienceRetentionSystem initialized successfully")
    
    @optimize_for_memory
    def retain_experience(
        self,
        experience_content: Array,
        temporal_moment: TemporalMoment,
        meaning_structure: Optional[MeaningStructure] = None,
        timestamp: Optional[TimeStep] = None,
    ) -> 'ExperienceRetentionSystem':
        """Retain new experience across all retention mechanisms.
        
        Refactored into smaller methods for better maintainability.
        """
        with GLOBAL_MEMORY_MANAGER.track_memory("experience_retention"):
            try:
                # Step 1: Validate inputs
                self._validate_retention_inputs(experience_content, temporal_moment)
                
                # Step 2: Process experience content
                processed_experience = self._process_experience_content(
                    experience_content, temporal_moment, meaning_structure
                )
                
                # Step 3: Update all retention systems
                updated_components = self._update_retention_systems(
                    processed_experience, timestamp or 0.0
                )
                
                # Step 4: Create and store experience trace
                experience_trace = self._create_experience_trace(
                    processed_experience, temporal_moment, timestamp or 0.0
                )
                
                # Step 5: Return updated system
                return self._create_updated_system(updated_components, experience_trace)
                
            except Exception as e:
                raise EnactiveConsciousnessError(f"Failed to retain experience: {e}")
    
    def process(self, *args, **kwargs) -> Any:
        """Implementation of ProcessorBase abstract method."""
        return self.retain_experience(*args, **kwargs)
    
    def _validate_retention_inputs(self, experience_content: Array, temporal_moment: TemporalMoment) -> None:
        """Extract method: Validate retention inputs."""
        self.validate_input_state(experience_content, "experience_content")
        if experience_content.shape[0] != self.config.experience_dim:
            raise ValueError(f"Experience content dimension {experience_content.shape[0]} does not match config {self.config.experience_dim}")
    
    def _process_experience_content(
        self,
        experience_content: Array,
        temporal_moment: TemporalMoment,
        meaning_structure: Optional[MeaningStructure],
    ) -> Array:
        """Extract method: Process raw experience content."""
        # Combine experience with temporal and meaning information
        temporal_component = jnp.mean(jnp.stack([
            temporal_moment.retention,
            temporal_moment.present_moment,
            temporal_moment.protention,
        ]), axis=0)
        
        # Resize temporal component to match experience dimension if needed
        if temporal_component.shape[0] != experience_content.shape[0]:
            # Simple resize by taking first N elements or padding with zeros
            if temporal_component.shape[0] > experience_content.shape[0]:
                temporal_component = temporal_component[:experience_content.shape[0]]
            else:
                padding = jnp.zeros(experience_content.shape[0] - temporal_component.shape[0])
                temporal_component = jnp.concatenate([temporal_component, padding])
        
        # Weight temporal integration
        temporal_weight = 0.3
        processed = experience_content + temporal_weight * temporal_component
        
        # Add meaning information if available
        if meaning_structure is not None:
            meaning_component = meaning_structure.semantic_content
            if meaning_component.shape[0] == processed.shape[0]:
                meaning_weight = meaning_structure.coherence_measure * 0.2
                processed = processed + meaning_weight * meaning_component
        
        return processed
    
    def _update_retention_systems(self, processed_experience: Array, timestamp: TimeStep) -> Dict[str, Any]:
        """Extract method: Update all retention system components."""
        # Update sedimentation
        updated_sedimentation = self.sedimentation.sediment_experience(
            processed_experience, self.config.sedimentation_rate
        )
        
        # Update associative network
        storage_position = self.current_position % self.config.associative_network_size
        temporal_signature = jnp.array([timestamp, len(self.experience_traces)])
        updated_associative = self.associative_network.store_experience(
            processed_experience, storage_position, temporal_signature
        )
        
        # Update temporal context
        updated_temporal_context, context_state = self.temporal_context.update_temporal_context(
            processed_experience
        )
        
        return {
            'sedimentation': updated_sedimentation,
            'associative_network': updated_associative,
            'temporal_context': updated_temporal_context,
            'context_state': context_state,
        }
    
    def _create_experience_trace(
        self, processed_experience: Array, temporal_moment: TemporalMoment, timestamp: TimeStep
    ) -> ExperienceTrace:
        """Extract method: Create new experience trace."""
        temporal_signature = jnp.stack([
            temporal_moment.retention,
            temporal_moment.present_moment,
            temporal_moment.protention,
        ]).flatten()
        
        # Resize temporal signature if needed
        if temporal_signature.shape[0] > processed_experience.shape[0]:
            temporal_signature = temporal_signature[:processed_experience.shape[0]]
        elif temporal_signature.shape[0] < processed_experience.shape[0]:
            padding = jnp.zeros(processed_experience.shape[0] - temporal_signature.shape[0])
            temporal_signature = jnp.concatenate([temporal_signature, padding])
        
        # Create associative links based on similarity
        associative_links = jnp.zeros(min(len(self.experience_traces), processed_experience.shape[0]))
        
        return ExperienceTrace(
            content=processed_experience,
            temporal_signature=temporal_signature,
            associative_links=associative_links,
            sedimentation_level=1.0,
            creation_timestamp=timestamp,
            last_access_timestamp=timestamp,
            access_frequency=1,
        )
    
    def _create_updated_system(
        self, updated_components: Dict[str, Any], experience_trace: ExperienceTrace
    ) -> 'ExperienceRetentionSystem':
        """Extract method: Create updated system instance."""
        new_traces = self.experience_traces + [experience_trace]
        new_position = self.current_position + 1
        
        return eqx.tree_at(
            lambda x: (
                x.sedimentation,
                x.associative_network,
                x.temporal_context,
                x.experience_traces,
                x.current_position
            ),
            self,
            (
                updated_components['sedimentation'],
                updated_components['associative_network'],
                updated_components['temporal_context'],
                new_traces,
                new_position,
            ),
        )
    
    def recall_related_experiences(
        self,
        query: Array,
        recall_mode: str = "associative",  # "associative", "temporal", "sedimented", "integrated"
        max_recalls: int = 5,
        timestamp: Optional[TimeStep] = None,
    ) -> List[Tuple[ExperienceTrace, float]]:
        """Recall experiences related to query using specified mode."""
        if recall_mode == "associative":
            return self._associative_recall(query, max_recalls)
        elif recall_mode == "temporal":
            return self._temporal_recall(query, max_recalls, timestamp)
        elif recall_mode == "sedimented":
            return self._sedimented_recall(query, max_recalls)
        elif recall_mode == "integrated":
            return self._integrated_recall(query, max_recalls, timestamp)
        else:
            raise ValueError(f"Unknown recall mode: {recall_mode}")
    
    def _associative_recall(self, query: Array, max_recalls: int) -> List[Tuple[ExperienceTrace, float]]:
        """Extract method: Perform associative recall."""
        recalled_experiences, recall_strengths = self.associative_network.associative_recall(
            query, self.config.association_threshold, max_recalls
        )
        
        # Match with experience traces
        results = []
        for i, (exp, strength) in enumerate(zip(recalled_experiences, recall_strengths)):
            if i < len(self.experience_traces):
                results.append((self.experience_traces[i], strength))
        
        return results
    
    def _temporal_recall(
        self, query: Array, max_recalls: int, timestamp: Optional[TimeStep]
    ) -> List[Tuple[ExperienceTrace, float]]:
        """Extract method: Perform temporal recall."""
        if timestamp is None:
            timestamp = 0.0
        
        # Get temporal context
        temporal_context = self.temporal_context.retrieve_temporal_context(query)
        
        # Score experiences by temporal relevance
        scored_experiences = []
        for trace in self.experience_traces:
            temporal_similarity = jnp.dot(temporal_context, trace.temporal_signature)
            temporal_similarity = temporal_similarity / (
                jnp.linalg.norm(temporal_context) * jnp.linalg.norm(trace.temporal_signature) + 1e-8
            )
            accessibility = trace.compute_accessibility(timestamp, self.config.retention_decay_rate)
            
            combined_score = float(temporal_similarity * accessibility)
            scored_experiences.append((trace, combined_score))
        
        # Sort by score and return top results
        scored_experiences.sort(key=lambda x: x[1], reverse=True)
        return scored_experiences[:max_recalls]
    
    def _sedimented_recall(self, query: Array, max_recalls: int) -> List[Tuple[ExperienceTrace, float]]:
        """Extract method: Perform sedimented recall."""
        sedimented_context = self.sedimentation.retrieve_sedimented_context(query)
        
        # Score experiences by sedimentation relevance
        scored_experiences = []
        for trace in self.experience_traces:
            sedimentation_similarity = jnp.dot(sedimented_context, trace.content)
            sedimentation_similarity = sedimentation_similarity / (
                jnp.linalg.norm(sedimented_context) * jnp.linalg.norm(trace.content) + 1e-8
            )
            
            sedimentation_boost = trace.sedimentation_level
            combined_score = float(sedimentation_similarity * sedimentation_boost)
            scored_experiences.append((trace, combined_score))
        
        # Sort and return top results
        scored_experiences.sort(key=lambda x: x[1], reverse=True)
        return scored_experiences[:max_recalls]
    
    def _integrated_recall(
        self, query: Array, max_recalls: int, timestamp: Optional[TimeStep]
    ) -> List[Tuple[ExperienceTrace, float]]:
        """Extract method: Perform integrated recall using all systems."""
        # Get results from all recall modes
        associative_results = self._associative_recall(query, max_recalls * 2)
        temporal_results = self._temporal_recall(query, max_recalls * 2, timestamp)
        sedimented_results = self._sedimented_recall(query, max_recalls * 2)
        
        # Combine and weight results
        combined_scores = {}
        
        # Weight different recall modes
        associative_weight = 0.4
        temporal_weight = 0.3
        sedimented_weight = 0.3
        
        for trace, score in associative_results:
            combined_scores[id(trace)] = combined_scores.get(id(trace), 0) + associative_weight * score
        
        for trace, score in temporal_results:
            combined_scores[id(trace)] = combined_scores.get(id(trace), 0) + temporal_weight * score
        
        for trace, score in sedimented_results:
            combined_scores[id(trace)] = combined_scores.get(id(trace), 0) + sedimented_weight * score
        
        # Create final results
        final_results = []
        trace_map = {id(trace): trace for trace in self.experience_traces}
        
        for trace_id, score in combined_scores.items():
            if trace_id in trace_map:
                final_results.append((trace_map[trace_id], score))
        
        # Sort and return top results
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:max_recalls]
    
    def assess_retention_quality(self) -> Dict[str, float]:
        """Assess quality of experience retention system."""
        # Sedimentation depth
        sedimentation_depth = self.sedimentation.get_sedimentation_depth()
        sedimentation_quality = min(1.0, sedimentation_depth / 10.0)  # Normalize
        
        # Temporal continuity
        temporal_continuity = self.temporal_context.assess_temporal_continuity()
        
        # Experience diversity (based on variance in content)
        if len(self.experience_traces) > 1:
            contents = jnp.stack([trace.content for trace in self.experience_traces[:10]])  # Sample
            content_variance = float(jnp.mean(jnp.var(contents, axis=0)))
            diversity_score = min(1.0, content_variance)
        else:
            diversity_score = 0.0
        
        # Overall retention quality
        retention_quality = (
            0.4 * sedimentation_quality +
            0.3 * temporal_continuity +
            0.3 * diversity_score
        )
        
        return {
            "sedimentation_quality": sedimentation_quality,
            "temporal_continuity": temporal_continuity,
            "experience_diversity": diversity_score,
            "overall_retention_quality": retention_quality,
        }


@functools.partial(jax.jit, static_argnames=['config'])
def create_experience_retention_system(
    config: ExperienceRetentionConfig,
    key: PRNGKey,
) -> ExperienceRetentionSystem:
    """Factory function for JIT-compiled experience retention system."""
    return ExperienceRetentionSystem(config, key)


# Utility functions
def create_test_experience_sequence(
    retention_system: ExperienceRetentionSystem,
    experience_sequence: List[Array],
    temporal_sequence: List[TemporalMoment],
    key: PRNGKey,
) -> ExperienceRetentionSystem:
    """Create sequence of retained experiences for testing."""
    if len(experience_sequence) != len(temporal_sequence):
        raise ValueError("Experience and temporal sequences must have same length")
    
    current_system = retention_system
    
    for i, (experience, temporal_moment) in enumerate(zip(experience_sequence, temporal_sequence)):
        current_system = current_system.retain_experience(
            experience, temporal_moment, timestamp=float(i)
        )
    
    return current_system


def analyze_experience_coherence(
    retention_system: ExperienceRetentionSystem,
) -> Dict[str, float]:
    """Analyze coherence of retained experiences."""
    if len(retention_system.experience_traces) < 2:
        return {"experience_coherence": 0.0, "temporal_coherence": 0.0, "overall_coherence": 0.0}
    
    # Content coherence
    contents = [trace.content for trace in retention_system.experience_traces[:20]]  # Sample
    if len(contents) > 1:
        content_correlations = []
        for i in range(len(contents) - 1):
            corr = jnp.dot(contents[i], contents[i+1])
            corr = corr / (jnp.linalg.norm(contents[i]) * jnp.linalg.norm(contents[i+1]) + 1e-8)
            content_correlations.append(corr)
        
        experience_coherence = float(jnp.mean(jnp.array(content_correlations)))
    else:
        experience_coherence = 0.0
    
    # Temporal coherence from temporal context preserver
    temporal_coherence = retention_system.temporal_context.assess_temporal_continuity()
    
    # Overall coherence
    overall_coherence = (experience_coherence + temporal_coherence) / 2.0
    
    return {
        "experience_coherence": max(0.0, min(1.0, experience_coherence)),
        "temporal_coherence": temporal_coherence,
        "overall_coherence": max(0.0, min(1.0, overall_coherence)),
    }


# Export public API
__all__ = [
    'ExperienceRetentionConfig',
    'ExperienceTrace',
    'ExperienceSedimentation',
    'AssociativeRecallNetwork',
    'TemporalContextPreserver',
    'ExperienceRetentionSystem',
    'create_experience_retention_system',
    'create_test_experience_sequence',
    'analyze_experience_coherence',
]