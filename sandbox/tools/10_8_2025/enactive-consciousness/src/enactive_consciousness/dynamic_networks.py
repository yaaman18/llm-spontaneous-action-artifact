"""Dynamic Networks for Enactive Consciousness.

This module implements adaptive graph structures for cognitive reorganization,
graph neural networks for meaning emergence, and dynamic connectivity patterns
following Varela-Maturana principles of structural coupling and autopoiesis.

Key Features:
1. Adaptive graph structures that reorganize based on experiential interactions
2. Graph neural networks for emergent meaning construction
3. Dynamic connectivity patterns for agent-environment coupling
4. Network metrics for consciousness assessment
5. Integration with circular causality and information theory measures
"""

from __future__ import annotations

import logging
import functools
from typing import Dict, List, Optional, Tuple, Any, Protocol, Union
from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp
import equinox as eqx
import jraph
import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from .types import (
    Array, 
    ArrayLike, 
    PRNGKey, 
    TimeStep,
    CouplingState,
    MeaningStructure,
    TemporalMoment,
    EnactiveConsciousnessError,
)
from .information_theory import (
    mutual_information_kraskov,
    transfer_entropy,
    circular_causality_index,
    complexity_measure,
)

# Configure module logger
logger = logging.getLogger(__name__)


class NetworkError(EnactiveConsciousnessError):
    """Exception for dynamic network processing errors."""
    pass


class NetworkTopology(Enum):
    """Types of network topologies for adaptive restructuring."""
    HIERARCHICAL = "hierarchical"
    SMALL_WORLD = "small_world" 
    SCALE_FREE = "scale_free"
    MODULAR = "modular"
    FULLY_CONNECTED = "fully_connected"


class AdaptationMechanism(Enum):
    """Mechanisms for network adaptation."""
    HEBBIAN = "hebbian"
    HOMEOSTATIC = "homeostatic"  
    STRUCTURAL_PLASTICITY = "structural_plasticity"
    CIRCULAR_CAUSALITY = "circular_causality"


@dataclass(frozen=True, slots=True)
class NetworkState:
    """Current state of dynamic network."""
    
    node_features: Array
    edge_weights: Array
    adjacency_matrix: Array
    global_features: Array
    topology_type: NetworkTopology
    adaptation_strength: float
    consciousness_level: float
    
    def __post_init__(self):
        """Validate network state consistency."""
        if not (0.0 <= self.adaptation_strength <= 1.0):
            raise ValueError("adaptation_strength must be in [0, 1]")
        if not (0.0 <= self.consciousness_level <= 1.0):
            raise ValueError("consciousness_level must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AdaptiveReorganizationResult:
    """Result of adaptive network reorganization."""
    
    new_network_state: NetworkState
    reorganization_strength: float
    structural_changes: Dict[str, float]
    meaning_emergence: float
    consciousness_delta: float


class DynamicNetworkProcessor(eqx.Module):
    """
    Processor for dynamic neural networks in enactive consciousness.
    
    Implements adaptive graph structures following Varela-Maturana principles:
    - Structural coupling through dynamic connectivity
    - Circular causality in network dynamics  
    - Meaning emergence through network reorganization
    - Autopoietic self-organization
    """
    
    # Network architecture parameters
    num_nodes: int
    hidden_dim: int
    num_message_passing_steps: int
    
    # Graph neural network components  
    node_encoder: eqx.nn.MLP
    edge_encoder: eqx.nn.MLP
    message_passing_net: eqx.nn.MLP
    global_features_net: eqx.nn.MLP
    consciousness_estimator: eqx.nn.MLP
    
    # Adaptation parameters
    adaptation_rate: float
    plasticity_threshold: float
    homeostatic_target: float
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 128,
        num_message_passing_steps: int = 3,
        adaptation_rate: float = 0.01,
        plasticity_threshold: float = 0.3,
        homeostatic_target: float = 0.6,
        *,
        key: PRNGKey,
    ):
        """Initialize dynamic network processor.
        
        Args:
            num_nodes: Number of nodes in the network
            hidden_dim: Hidden dimension for neural networks
            num_message_passing_steps: Steps of message passing
            adaptation_rate: Rate of network adaptation
            plasticity_threshold: Threshold for structural plasticity
            homeostatic_target: Target for homeostatic regulation
            key: Random key for initialization
        """
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.adaptation_rate = adaptation_rate
        self.plasticity_threshold = plasticity_threshold
        self.homeostatic_target = homeostatic_target
        
        # Split keys for different components
        keys = jax.random.split(key, 6)
        
        # Initialize neural network components
        self.node_encoder = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=hidden_dim, 
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[0],
        )
        
        self.edge_encoder = eqx.nn.MLP(
            in_size=2 * hidden_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=2, 
            activation=jax.nn.gelu,
            key=keys[1],
        )
        
        self.message_passing_net = eqx.nn.MLP(
            in_size=2 * hidden_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.gelu,
            key=keys[2],
        )
        
        self.global_features_net = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[3],
        )
        
        self.consciousness_estimator = eqx.nn.MLP(
            in_size=hidden_dim * 2,  # Global features + node summary 
            out_size=1,
            width_size=hidden_dim // 2,
            depth=2,
            activation=jax.nn.sigmoid,
            key=keys[4],
        )
    
    def create_initial_network_state(
        self,
        initial_features: Array,
        topology_type: NetworkTopology = NetworkTopology.SMALL_WORLD,
        *,
        key: PRNGKey,
    ) -> NetworkState:
        """Create initial network state with specified topology.
        
        Args:
            initial_features: Initial node features
            topology_type: Type of network topology
            key: Random key for stochastic elements
            
        Returns:
            Initial network state
        """
        try:
            # Validate inputs
            if initial_features.shape[0] != self.num_nodes:
                raise NetworkError(
                    f"Features shape mismatch: expected {self.num_nodes}, "
                    f"got {initial_features.shape[0]}"
                )
            
            # Create adjacency matrix based on topology
            adjacency_matrix = self._create_topology(topology_type, key)
            
            # Initialize edge weights from adjacency
            edge_weights = adjacency_matrix * jax.random.uniform(
                key, adjacency_matrix.shape, minval=0.1, maxval=1.0
            )
            
            # Encode initial node features
            node_features = jax.vmap(self.node_encoder)(initial_features)
            
            # Initialize global features as mean of nodes
            global_features = self.global_features_net(jnp.mean(node_features, axis=0))
            
            # Initial consciousness level estimation
            node_summary = jnp.mean(node_features, axis=0)
            consciousness_input = jnp.concatenate([global_features, node_summary])
            consciousness_raw = self.consciousness_estimator(consciousness_input)[0]
            consciousness_level = float(jnp.clip(consciousness_raw, 0.0, 1.0))
            
            return NetworkState(
                node_features=node_features,
                edge_weights=edge_weights,
                adjacency_matrix=adjacency_matrix,
                global_features=global_features,
                topology_type=topology_type,
                adaptation_strength=0.5,
                consciousness_level=consciousness_level,
            )
            
        except Exception as e:
            raise NetworkError(f"Failed to create initial network state: {e}")
    
    def _create_topology(self, topology_type: NetworkTopology, key: PRNGKey) -> Array:
        """Create adjacency matrix for specified topology."""
        try:
            if topology_type == NetworkTopology.SMALL_WORLD:
                # Watts-Strogatz small world
                G = nx.watts_strogatz_graph(self.num_nodes, k=6, p=0.1)
                return jnp.array(nx.adjacency_matrix(G).todense())
                
            elif topology_type == NetworkTopology.SCALE_FREE:
                # Barabasi-Albert preferential attachment
                G = nx.barabasi_albert_graph(self.num_nodes, m=3)
                return jnp.array(nx.adjacency_matrix(G).todense())
                
            elif topology_type == NetworkTopology.MODULAR:
                # Modular network with community structure
                G = nx.random_partition_graph([self.num_nodes // 4] * 4, 0.7, 0.1)
                return jnp.array(nx.adjacency_matrix(G).todense())
                
            elif topology_type == NetworkTopology.HIERARCHICAL:
                # Hierarchical network structure
                levels = 3
                nodes_per_level = self.num_nodes // levels
                adj = jnp.zeros((self.num_nodes, self.num_nodes))
                
                # Connect within levels
                for level in range(levels):
                    start_idx = level * nodes_per_level
                    end_idx = min((level + 1) * nodes_per_level, self.num_nodes)
                    for i in range(start_idx, end_idx):
                        for j in range(start_idx, end_idx):
                            if i != j:
                                adj = adj.at[i, j].set(1.0)
                
                # Connect between levels (hierarchical)
                for level in range(levels - 1):
                    current_start = level * nodes_per_level
                    next_start = (level + 1) * nodes_per_level
                    next_end = min((level + 2) * nodes_per_level, self.num_nodes)
                    
                    for i in range(current_start, current_start + nodes_per_level):
                        for j in range(next_start, next_end):
                            if jax.random.uniform(key) < 0.3:  # Sparse connections
                                adj = adj.at[i, j].set(1.0)
                                adj = adj.at[j, i].set(1.0)
                
                return adj
                
            elif topology_type == NetworkTopology.FULLY_CONNECTED:
                # Fully connected network
                return jnp.ones((self.num_nodes, self.num_nodes)) - jnp.eye(self.num_nodes)
                
            else:
                raise NetworkError(f"Unknown topology type: {topology_type}")
                
        except Exception as e:
            raise NetworkError(f"Failed to create topology {topology_type}: {e}")
    
    def process_graph_message_passing(
        self, 
        network_state: NetworkState,
        external_input: Optional[Array] = None,
    ) -> NetworkState:
        """Process graph neural network message passing."""
        try:
            node_features = network_state.node_features
            edge_weights = network_state.edge_weights
            adjacency_matrix = network_state.adjacency_matrix
            
            # Add external input if provided
            if external_input is not None:
                if external_input.shape[0] != self.num_nodes:
                    raise NetworkError("External input size mismatch")
                node_features = node_features + external_input
            
            # Message passing iterations
            for step in range(self.num_message_passing_steps):
                # Collect messages from neighbors
                messages = jnp.zeros_like(node_features)
                
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        if adjacency_matrix[i, j] > 0:
                            # Create message from node j to node i
                            edge_features = jnp.concatenate([
                                node_features[i], node_features[j]
                            ])
                            edge_encoded = self.edge_encoder(edge_features)
                            
                            # Weight message by edge strength
                            message = edge_weights[i, j] * edge_encoded
                            messages = messages.at[i].add(message)
                
                # Update node features using messages
                for i in range(self.num_nodes):
                    combined_input = jnp.concatenate([
                        node_features[i], messages[i]
                    ])
                    node_features = node_features.at[i].set(
                        self.message_passing_net(combined_input)
                    )
            
            # Update global features
            global_features = self.global_features_net(jnp.mean(node_features, axis=0))
            
            # Update consciousness level
            node_summary = jnp.mean(node_features, axis=0)
            consciousness_input = jnp.concatenate([global_features, node_summary])
            consciousness_raw = self.consciousness_estimator(consciousness_input)[0]
            consciousness_level = float(jnp.clip(consciousness_raw, 0.0, 1.0))
            
            return NetworkState(
                node_features=node_features,
                edge_weights=edge_weights,
                adjacency_matrix=adjacency_matrix,
                global_features=global_features,
                topology_type=network_state.topology_type,
                adaptation_strength=network_state.adaptation_strength,
                consciousness_level=consciousness_level,
            )
            
        except Exception as e:
            raise NetworkError(f"Graph message passing failed: {e}")
    
    def adaptive_reorganization(
        self,
        network_state: NetworkState,
        coupling_state: CouplingState,
        temporal_moment: Optional[TemporalMoment] = None,
        adaptation_mechanism: AdaptationMechanism = AdaptationMechanism.CIRCULAR_CAUSALITY,
        *,
        key: PRNGKey,
    ) -> AdaptiveReorganizationResult:
        """
        Perform adaptive network reorganization based on experience.
        
        Implements Varela-Maturana principles of structural coupling and
        circular causality for network adaptation.
        """
        try:
            # Compute adaptation signals from coupling dynamics
            adaptation_signals = self._compute_adaptation_signals(
                network_state, coupling_state, temporal_moment
            )
            
            # Apply adaptation mechanism
            if adaptation_mechanism == AdaptationMechanism.CIRCULAR_CAUSALITY:
                new_state = self._circular_causality_adaptation(
                    network_state, adaptation_signals, key
                )
            elif adaptation_mechanism == AdaptationMechanism.HEBBIAN:
                new_state = self._hebbian_adaptation(
                    network_state, adaptation_signals, key
                )
            elif adaptation_mechanism == AdaptationMechanism.HOMEOSTATIC:
                new_state = self._homeostatic_adaptation(
                    network_state, adaptation_signals, key
                )
            elif adaptation_mechanism == AdaptationMechanism.STRUCTURAL_PLASTICITY:
                new_state = self._structural_plasticity_adaptation(
                    network_state, adaptation_signals, key
                )
            else:
                raise NetworkError(f"Unknown adaptation mechanism: {adaptation_mechanism}")
            
            # Compute reorganization metrics
            reorganization_strength = self._compute_reorganization_strength(
                network_state, new_state
            )
            
            structural_changes = self._analyze_structural_changes(
                network_state, new_state
            )
            
            meaning_emergence = self._assess_meaning_emergence(
                network_state, new_state, coupling_state
            )
            
            consciousness_delta = new_state.consciousness_level - network_state.consciousness_level
            
            return AdaptiveReorganizationResult(
                new_network_state=new_state,
                reorganization_strength=reorganization_strength,
                structural_changes=structural_changes,
                meaning_emergence=meaning_emergence,
                consciousness_delta=consciousness_delta,
            )
            
        except Exception as e:
            raise NetworkError(f"Adaptive reorganization failed: {e}")
    
    def _compute_adaptation_signals(
        self,
        network_state: NetworkState,
        coupling_state: CouplingState,
        temporal_moment: Optional[TemporalMoment],
    ) -> Dict[str, Array]:
        """Compute signals that drive network adaptation."""
        try:
            # Coupling-based signals
            coupling_strength = coupling_state.coupling_strength
            stability_metric = coupling_state.stability_metric
            
            # Activity-based signals from network state
            node_activity = jnp.linalg.norm(network_state.node_features, axis=-1)
            edge_activity = jnp.sum(network_state.edge_weights, axis=-1)
            
            # Temporal signals if available
            if temporal_moment is not None:
                temporal_signal = jnp.linalg.norm(temporal_moment.present_moment)
                retention_signal = jnp.linalg.norm(temporal_moment.retention)
            else:
                temporal_signal = 0.0
                retention_signal = 0.0
            
            # Homeostatic signals (deviation from target)
            activity_deviation = jnp.abs(node_activity - self.homeostatic_target)
            
            # Plasticity signals (high activity correlations)
            correlation_matrix = jnp.corrcoef(network_state.node_features)
            plasticity_signal = jnp.mean(jnp.abs(correlation_matrix))
            
            return {
                'coupling_strength': jnp.array(coupling_strength),
                'stability_metric': jnp.array(stability_metric),
                'node_activity': node_activity,
                'edge_activity': edge_activity,
                'temporal_signal': jnp.array(temporal_signal),
                'retention_signal': jnp.array(retention_signal),
                'activity_deviation': activity_deviation,
                'plasticity_signal': jnp.array(plasticity_signal),
            }
            
        except Exception as e:
            raise NetworkError(f"Failed to compute adaptation signals: {e}")
    
    def _circular_causality_adaptation(
        self,
        network_state: NetworkState,
        adaptation_signals: Dict[str, Array],
        key: PRNGKey,
    ) -> NetworkState:
        """Adapt network based on circular causality principles."""
        try:
            # Circular causality: network structure influences activity,
            # activity influences structure
            
            node_features = network_state.node_features
            edge_weights = network_state.edge_weights.copy()
            adjacency_matrix = network_state.adjacency_matrix.copy()
            
            # Adapt edge weights based on activity correlation
            node_activity = adaptation_signals['node_activity']
            coupling_strength = adaptation_signals['coupling_strength']
            
            # Update edge weights based on activity and coupling
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if adjacency_matrix[i, j] > 0:
                        # Activity correlation
                        activity_corr = node_activity[i] * node_activity[j]
                        
                        # Coupling influence
                        coupling_influence = coupling_strength * self.adaptation_rate
                        
                        # Update edge weight
                        new_weight = edge_weights[i, j] + (
                            coupling_influence * activity_corr * self.adaptation_rate
                        )
                        edge_weights = edge_weights.at[i, j].set(
                            jnp.clip(new_weight, 0.01, 2.0)
                        )
            
            # Structural plasticity based on activity
            plasticity_signal = adaptation_signals['plasticity_signal']
            if plasticity_signal > self.plasticity_threshold:
                # Add new connections based on activity similarity
                activity_similarity = jnp.outer(node_activity, node_activity)
                threshold = jnp.percentile(activity_similarity, 90)
                
                new_connections = (activity_similarity > threshold) & (adjacency_matrix == 0)
                adjacency_matrix = adjacency_matrix + new_connections.astype(jnp.float32) * 0.1
                
                # Initialize new edge weights
                edge_weights = edge_weights + new_connections.astype(jnp.float32) * 0.1
            
            # Update adaptation strength based on changes
            adaptation_strength = min(1.0, network_state.adaptation_strength + 
                                    self.adaptation_rate * plasticity_signal)
            
            return NetworkState(
                node_features=node_features,
                edge_weights=edge_weights,
                adjacency_matrix=adjacency_matrix,
                global_features=network_state.global_features,
                topology_type=network_state.topology_type,
                adaptation_strength=adaptation_strength,
                consciousness_level=network_state.consciousness_level,
            )
            
        except Exception as e:
            raise NetworkError(f"Circular causality adaptation failed: {e}")
    
    def _hebbian_adaptation(
        self,
        network_state: NetworkState,
        adaptation_signals: Dict[str, Array],
        key: PRNGKey,
    ) -> NetworkState:
        """Adapt network using Hebbian learning principles."""
        try:
            edge_weights = network_state.edge_weights.copy()
            node_activity = adaptation_signals['node_activity']
            
            # Hebbian rule: "neurons that fire together, wire together"
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if network_state.adjacency_matrix[i, j] > 0:
                        # Hebbian update
                        activity_product = node_activity[i] * node_activity[j]
                        weight_change = self.adaptation_rate * activity_product
                        
                        new_weight = edge_weights[i, j] + weight_change
                        edge_weights = edge_weights.at[i, j].set(
                            jnp.clip(new_weight, 0.01, 2.0)
                        )
            
            adaptation_strength = min(1.0, network_state.adaptation_strength + 
                                    self.adaptation_rate * jnp.mean(node_activity))
            
            return NetworkState(
                node_features=network_state.node_features,
                edge_weights=edge_weights,
                adjacency_matrix=network_state.adjacency_matrix,
                global_features=network_state.global_features,
                topology_type=network_state.topology_type,
                adaptation_strength=adaptation_strength,
                consciousness_level=network_state.consciousness_level,
            )
            
        except Exception as e:
            raise NetworkError(f"Hebbian adaptation failed: {e}")
    
    def _homeostatic_adaptation(
        self,
        network_state: NetworkState,
        adaptation_signals: Dict[str, Array],
        key: PRNGKey,
    ) -> NetworkState:
        """Adapt network using homeostatic regulation."""
        try:
            edge_weights = network_state.edge_weights.copy()
            activity_deviation = adaptation_signals['activity_deviation']
            
            # Homeostatic scaling to maintain target activity
            for i in range(self.num_nodes):
                if activity_deviation[i] > 0.1:  # Significant deviation
                    # Scale all incoming connections
                    scaling_factor = 1.0 - self.adaptation_rate * activity_deviation[i]
                    for j in range(self.num_nodes):
                        if network_state.adjacency_matrix[j, i] > 0:
                            new_weight = edge_weights[j, i] * scaling_factor
                            edge_weights = edge_weights.at[j, i].set(
                                jnp.clip(new_weight, 0.01, 2.0)
                            )
            
            adaptation_strength = network_state.adaptation_strength
            
            return NetworkState(
                node_features=network_state.node_features,
                edge_weights=edge_weights,
                adjacency_matrix=network_state.adjacency_matrix,
                global_features=network_state.global_features,
                topology_type=network_state.topology_type,
                adaptation_strength=adaptation_strength,
                consciousness_level=network_state.consciousness_level,
            )
            
        except Exception as e:
            raise NetworkError(f"Homeostatic adaptation failed: {e}")
    
    def _structural_plasticity_adaptation(
        self,
        network_state: NetworkState,
        adaptation_signals: Dict[str, Array],
        key: PRNGKey,
    ) -> NetworkState:
        """Adapt network structure through structural plasticity."""
        try:
            adjacency_matrix = network_state.adjacency_matrix.copy()
            edge_weights = network_state.edge_weights.copy()
            node_activity = adaptation_signals['node_activity']
            
            # Remove weak connections
            weak_threshold = jnp.percentile(edge_weights[adjacency_matrix > 0], 10)
            weak_connections = (edge_weights < weak_threshold) & (adjacency_matrix > 0)
            adjacency_matrix = adjacency_matrix.at[weak_connections].set(0.0)
            edge_weights = edge_weights.at[weak_connections].set(0.0)
            
            # Add new connections based on activity
            activity_product = jnp.outer(node_activity, node_activity)
            strong_threshold = jnp.percentile(activity_product, 95)
            
            new_connections = (activity_product > strong_threshold) & (adjacency_matrix == 0)
            # Limit number of new connections
            num_new = min(jnp.sum(new_connections), self.num_nodes // 4)
            
            if num_new > 0:
                indices = jnp.where(new_connections)
                selected_indices = jax.random.choice(
                    key, len(indices[0]), (num_new,), replace=False
                )
                
                for idx in selected_indices:
                    i, j = indices[0][idx], indices[1][idx]
                    adjacency_matrix = adjacency_matrix.at[i, j].set(1.0)
                    edge_weights = edge_weights.at[i, j].set(0.5)
            
            adaptation_strength = min(1.0, network_state.adaptation_strength + 
                                    0.1 * self.adaptation_rate)
            
            return NetworkState(
                node_features=network_state.node_features,
                edge_weights=edge_weights,
                adjacency_matrix=adjacency_matrix,
                global_features=network_state.global_features,
                topology_type=network_state.topology_type,
                adaptation_strength=adaptation_strength,
                consciousness_level=network_state.consciousness_level,
            )
            
        except Exception as e:
            raise NetworkError(f"Structural plasticity adaptation failed: {e}")
    
    def _compute_reorganization_strength(
        self, old_state: NetworkState, new_state: NetworkState
    ) -> float:
        """Compute strength of network reorganization."""
        try:
            # Compare network states
            feature_change = jnp.linalg.norm(
                new_state.node_features - old_state.node_features
            )
            weight_change = jnp.linalg.norm(
                new_state.edge_weights - old_state.edge_weights
            )
            structure_change = jnp.sum(jnp.abs(
                new_state.adjacency_matrix - old_state.adjacency_matrix
            ))
            
            # Normalize and combine
            max_feature_change = jnp.linalg.norm(old_state.node_features) + 1e-6
            max_weight_change = jnp.linalg.norm(old_state.edge_weights) + 1e-6
            max_structure_change = jnp.sum(old_state.adjacency_matrix) + 1e-6
            
            normalized_feature_change = feature_change / max_feature_change
            normalized_weight_change = weight_change / max_weight_change
            normalized_structure_change = structure_change / max_structure_change
            
            reorganization_strength = (
                0.4 * normalized_feature_change +
                0.4 * normalized_weight_change +
                0.2 * normalized_structure_change
            )
            
            return float(jnp.clip(reorganization_strength, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Failed to compute reorganization strength: {e}")
            return 0.0
    
    def _analyze_structural_changes(
        self, old_state: NetworkState, new_state: NetworkState
    ) -> Dict[str, float]:
        """Analyze structural changes in the network."""
        try:
            old_adj = old_state.adjacency_matrix
            new_adj = new_state.adjacency_matrix
            
            # Count changes
            added_edges = jnp.sum((new_adj > old_adj).astype(jnp.int32))
            removed_edges = jnp.sum((old_adj > new_adj).astype(jnp.int32))
            total_edges = jnp.sum((old_adj > 0).astype(jnp.int32))
            
            # Compute metrics
            edge_turnover = (added_edges + removed_edges) / (total_edges + 1)
            
            # Network topology metrics
            old_clustering = self._compute_clustering_coefficient(old_adj)
            new_clustering = self._compute_clustering_coefficient(new_adj)
            clustering_change = abs(new_clustering - old_clustering)
            
            # Path length changes (simplified)
            old_mean_degree = jnp.mean(jnp.sum(old_adj, axis=1))
            new_mean_degree = jnp.mean(jnp.sum(new_adj, axis=1))
            degree_change = abs(new_mean_degree - old_mean_degree) / (old_mean_degree + 1e-6)
            
            return {
                'edge_turnover': float(edge_turnover),
                'clustering_change': float(clustering_change),
                'degree_change': float(degree_change),
                'added_edges': float(added_edges),
                'removed_edges': float(removed_edges),
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze structural changes: {e}")
            return {'edge_turnover': 0.0, 'clustering_change': 0.0, 
                   'degree_change': 0.0, 'added_edges': 0.0, 'removed_edges': 0.0}
    
    def _compute_clustering_coefficient(self, adjacency_matrix: Array) -> float:
        """Compute clustering coefficient of the network."""
        try:
            n_nodes = adjacency_matrix.shape[0]
            clustering_coeffs = []
            
            for i in range(n_nodes):
                neighbors = jnp.where(adjacency_matrix[i] > 0)[0]
                k = len(neighbors)
                
                if k < 2:
                    clustering_coeffs.append(0.0)
                    continue
                
                # Count triangles
                triangles = 0
                for j in range(len(neighbors)):
                    for l in range(j + 1, len(neighbors)):
                        if adjacency_matrix[neighbors[j], neighbors[l]] > 0:
                            triangles += 1
                
                clustering_coeff = (2 * triangles) / (k * (k - 1))
                clustering_coeffs.append(clustering_coeff)
            
            return float(jnp.mean(jnp.array(clustering_coeffs)))
            
        except Exception as e:
            logger.warning(f"Failed to compute clustering coefficient: {e}")
            return 0.0
    
    def _assess_meaning_emergence(
        self,
        old_state: NetworkState,
        new_state: NetworkState,
        coupling_state: CouplingState,
    ) -> float:
        """Assess level of meaning emergence from network changes."""
        try:
            # Information-theoretic measures
            old_features_flat = old_state.node_features.flatten()
            new_features_flat = new_state.node_features.flatten()
            
            # Mutual information between old and new states
            if len(old_features_flat) == len(new_features_flat):
                feature_mi = mutual_information_kraskov(
                    old_features_flat, new_features_flat
                )
            else:
                feature_mi = 0.0
            
            # Coupling-based meaning emergence
            coupling_strength = coupling_state.coupling_strength
            stability_metric = coupling_state.stability_metric
            
            # Network complexity change
            old_complexity = jnp.var(old_state.node_features)
            new_complexity = jnp.var(new_state.node_features)
            complexity_change = abs(new_complexity - old_complexity) / (old_complexity + 1e-6)
            
            # Consciousness level change
            consciousness_change = abs(
                new_state.consciousness_level - old_state.consciousness_level
            )
            
            # Combine measures
            meaning_emergence = (
                0.3 * feature_mi +
                0.2 * coupling_strength +
                0.2 * stability_metric +
                0.2 * complexity_change +
                0.1 * consciousness_change
            )
            
            return float(jnp.clip(meaning_emergence, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Failed to assess meaning emergence: {e}")
            return 0.0
    
    def compute_network_metrics(
        self, network_state: NetworkState
    ) -> Dict[str, float]:
        """Compute comprehensive network metrics for consciousness assessment."""
        try:
            adjacency_matrix = network_state.adjacency_matrix
            node_features = network_state.node_features
            edge_weights = network_state.edge_weights
            
            # Basic network metrics
            num_edges = jnp.sum((adjacency_matrix > 0).astype(jnp.int32))
            density = num_edges / (self.num_nodes * (self.num_nodes - 1))
            
            # Clustering coefficient
            clustering_coeff = self._compute_clustering_coefficient(adjacency_matrix)
            
            # Mean degree
            degrees = jnp.sum(adjacency_matrix, axis=1)
            mean_degree = jnp.mean(degrees)
            degree_variance = jnp.var(degrees)
            
            # Feature-based metrics
            feature_variance = jnp.mean(jnp.var(node_features, axis=0))
            feature_entropy = -jnp.sum(
                jnp.log(jnp.abs(node_features) + 1e-8) * jnp.abs(node_features)
            ) / node_features.size
            
            # Edge weight metrics
            weight_mean = jnp.mean(edge_weights[adjacency_matrix > 0])
            weight_variance = jnp.var(edge_weights[adjacency_matrix > 0])
            
            # Integration measures
            global_integration = jnp.linalg.norm(network_state.global_features)
            
            # Small-world metrics (approximation)
            characteristic_path_length = 1.0 / (mean_degree + 1e-6)
            small_worldness = clustering_coeff / (characteristic_path_length + 1e-6)
            
            return {
                'num_edges': float(num_edges),
                'density': float(density),
                'clustering_coefficient': float(clustering_coeff),
                'mean_degree': float(mean_degree),
                'degree_variance': float(degree_variance),
                'feature_variance': float(feature_variance),
                'feature_entropy': float(feature_entropy),
                'weight_mean': float(weight_mean),
                'weight_variance': float(weight_variance),
                'global_integration': float(global_integration),
                'small_worldness': float(small_worldness),
                'consciousness_level': network_state.consciousness_level,
                'adaptation_strength': network_state.adaptation_strength,
            }
            
        except Exception as e:
            raise NetworkError(f"Failed to compute network metrics: {e}")
    
    def assess_consciousness_level(
        self, network_state: NetworkState
    ) -> Dict[str, float]:
        """Assess consciousness level based on network properties."""
        try:
            metrics = self.compute_network_metrics(network_state)
            
            # Integration measures
            integration_score = metrics['global_integration'] * metrics['clustering_coefficient']
            
            # Information measures
            information_score = metrics['feature_entropy'] * metrics['weight_variance']
            
            # Complexity measures
            complexity_score = metrics['small_worldness'] * metrics['feature_variance']
            
            # Differentiation measures  
            differentiation_score = metrics['degree_variance'] * metrics['weight_variance']
            
            # Overall consciousness score
            consciousness_score = (
                0.3 * integration_score +
                0.25 * information_score +
                0.25 * complexity_score +
                0.2 * differentiation_score
            )
            
            return {
                'consciousness_score': float(jnp.clip(consciousness_score, 0.0, 1.0)),
                'integration_score': float(jnp.clip(integration_score, 0.0, 1.0)),
                'information_score': float(jnp.clip(information_score, 0.0, 1.0)),
                'complexity_score': float(jnp.clip(complexity_score, 0.0, 1.0)),
                'differentiation_score': float(jnp.clip(differentiation_score, 0.0, 1.0)),
                'network_consciousness': network_state.consciousness_level,
            }
            
        except Exception as e:
            raise NetworkError(f"Failed to assess consciousness level: {e}")


class NetworkIntegrator(eqx.Module):
    """
    Integrator for dynamic networks with circular causality and information theory.
    
    Combines network dynamics with coupling states and temporal moments
    to create integrated conscious experiences.
    """
    
    network_processor: DynamicNetworkProcessor
    integration_dim: int
    integration_net: eqx.nn.MLP
    
    def __init__(
        self,
        network_processor: DynamicNetworkProcessor,
        integration_dim: int = 256,
        *,
        key: PRNGKey,
    ):
        """Initialize network integrator."""
        self.network_processor = network_processor
        self.integration_dim = integration_dim
        
        self.integration_net = eqx.nn.MLP(
            in_size=integration_dim * 2,  # Network + coupling features
            out_size=integration_dim,
            width_size=integration_dim,
            depth=3,
            activation=jax.nn.gelu,
            key=key,
        )
    
    def integrate_network_dynamics(
        self,
        network_state: NetworkState,
        coupling_state: CouplingState,
        temporal_moment: Optional[TemporalMoment] = None,
        *,
        key: PRNGKey,
    ) -> Tuple[NetworkState, Dict[str, float]]:
        """Integrate network dynamics with coupling and temporal information."""
        try:
            # Process network with coupling influence
            external_input = None
            if coupling_state is not None:
                # Use coupling state as external input
                coupling_features = jnp.concatenate([
                    coupling_state.agent_state.flatten(),
                    coupling_state.environmental_state.flatten(),
                ])
                # Reshape to match network nodes
                external_input = jnp.tile(
                    coupling_features[:self.network_processor.hidden_dim],
                    (self.network_processor.num_nodes, 1)
                )
            
            # Update network state
            updated_network_state = self.network_processor.process_graph_message_passing(
                network_state, external_input
            )
            
            # Compute circular causality between network and coupling
            if coupling_state is not None:
                network_features = updated_network_state.node_features.flatten()
                coupling_features = coupling_state.agent_state.flatten()
                
                # Ensure same length for information theory measures
                min_len = min(len(network_features), len(coupling_features))
                network_truncated = network_features[:min_len]
                coupling_truncated = coupling_features[:min_len]
                
                # Create minimal time series for causality analysis
                # We need at least 3 time steps for transfer entropy
                time_steps = 5
                network_series = jnp.tile(network_truncated, (time_steps, 1))
                coupling_series = jnp.tile(coupling_truncated, (time_steps, 1))
                
                # Add small random perturbations to create variation
                perturbation_key = jax.random.PRNGKey(42)
                keys = jax.random.split(perturbation_key, 2)
                network_series = network_series + 0.01 * jax.random.normal(keys[0], network_series.shape)
                coupling_series = coupling_series + 0.01 * jax.random.normal(keys[1], coupling_series.shape)
                
                try:
                    causality_metrics = circular_causality_index(
                        network_series, coupling_series, history_length=1
                    )
                except Exception as e:
                    # Fallback to basic mutual information
                    mi = mutual_information_kraskov(network_truncated, coupling_truncated)
                    causality_metrics = {
                        'circular_causality': float(mi),
                        'transfer_entropy_env_to_agent': 0.0,
                        'transfer_entropy_agent_to_env': 0.0,
                    }
            else:
                causality_metrics = {'circular_causality': 0.0}
            
            # Assess consciousness with integrated information
            consciousness_metrics = self.network_processor.assess_consciousness_level(
                updated_network_state
            )
            
            # Compute network-coupling strength
            if coupling_state is not None:
                network_features = updated_network_state.node_features.flatten()
                coupling_features = coupling_state.agent_state.flatten()
                min_len = min(len(network_features), len(coupling_features))
                try:
                    network_coupling_strength = float(
                        mutual_information_kraskov(
                            network_features[:min_len],
                            coupling_features[:min_len]
                        )
                    )
                except:
                    network_coupling_strength = 0.0
            else:
                network_coupling_strength = 0.0
            
            # Combine metrics
            integration_metrics = {
                **causality_metrics,
                **consciousness_metrics,
                'network_coupling_strength': network_coupling_strength,
            }
            
            return updated_network_state, integration_metrics
            
        except Exception as e:
            raise NetworkError(f"Network integration failed: {e}")


# Export public API
__all__ = [
    'NetworkError',
    'NetworkTopology', 
    'AdaptationMechanism',
    'NetworkState',
    'AdaptiveReorganizationResult', 
    'DynamicNetworkProcessor',
    'NetworkIntegrator',
]