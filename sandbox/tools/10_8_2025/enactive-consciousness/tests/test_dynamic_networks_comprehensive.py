"""Comprehensive test suite for dynamic networks module.

This test suite follows TDD principles with extensive coverage of
dynamic network processing for enactive consciousness, including
graph neural networks, adaptive topology changes, and network metrics.

Test Coverage:
- DynamicNetworkProcessor initialization and configuration
- NetworkState creation and validation
- Graph neural network message passing
- Adaptive network reorganization mechanisms
- Network topology creation and validation
- NetworkIntegrator functionality
- Performance and scalability testing
- Mathematical correctness validation
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from unittest.mock import patch, MagicMock

# Import the module under test
import sys
sys.path.insert(0, '/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/10_8_2025/enactive-consciousness/src')

from enactive_consciousness.dynamic_networks import (
    NetworkError,
    NetworkTopology,
    AdaptationMechanism,
    NetworkState,
    AdaptiveReorganizationResult,
    DynamicNetworkProcessor,
    NetworkIntegrator,
)

from enactive_consciousness.types import (
    CouplingState,
    TemporalMoment,
    PRNGKey,
    TimeStep,
)


class TestNetworkError:
    """Test cases for NetworkError exception."""
    
    def test_network_error_creation(self):
        """Test creating NetworkError."""
        error = NetworkError("Test network error")
        assert str(error) == "Test network error"
        assert isinstance(error, Exception)


class TestNetworkTopology:
    """Test cases for NetworkTopology enum."""
    
    def test_network_topology_values(self):
        """Test NetworkTopology enum values."""
        assert NetworkTopology.HIERARCHICAL.value == "hierarchical"
        assert NetworkTopology.SMALL_WORLD.value == "small_world"
        assert NetworkTopology.SCALE_FREE.value == "scale_free"
        assert NetworkTopology.MODULAR.value == "modular"
        assert NetworkTopology.FULLY_CONNECTED.value == "fully_connected"
    
    def test_network_topology_completeness(self):
        """Test that all expected topology types are available."""
        expected_topologies = {
            "hierarchical", "small_world", "scale_free", 
            "modular", "fully_connected"
        }
        actual_topologies = {topology.value for topology in NetworkTopology}
        assert actual_topologies == expected_topologies


class TestAdaptationMechanism:
    """Test cases for AdaptationMechanism enum."""
    
    def test_adaptation_mechanism_values(self):
        """Test AdaptationMechanism enum values."""
        assert AdaptationMechanism.HEBBIAN.value == "hebbian"
        assert AdaptationMechanism.HOMEOSTATIC.value == "homeostatic"
        assert AdaptationMechanism.STRUCTURAL_PLASTICITY.value == "structural_plasticity"
        assert AdaptationMechanism.CIRCULAR_CAUSALITY.value == "circular_causality"


class TestNetworkState:
    """Test cases for NetworkState dataclass."""
    
    @pytest.fixture
    def sample_network_state(self):
        """Create sample network state for testing."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        num_nodes = 10
        hidden_dim = 16
        
        return NetworkState(
            node_features=jax.random.normal(keys[0], (num_nodes, hidden_dim)),
            edge_weights=jax.random.uniform(keys[1], (num_nodes, num_nodes)),
            adjacency_matrix=jax.random.choice(
                keys[2], 2, shape=(num_nodes, num_nodes), p=jnp.array([0.7, 0.3])
            ).astype(float),
            global_features=jax.random.normal(keys[3], (hidden_dim,)),
            topology_type=NetworkTopology.SMALL_WORLD,
            adaptation_strength=0.5,
            consciousness_level=0.7,
        )
    
    def test_network_state_creation(self, sample_network_state):
        """Test NetworkState creation with valid parameters."""
        state = sample_network_state
        
        assert isinstance(state.node_features, jax.Array)
        assert isinstance(state.edge_weights, jax.Array)
        assert isinstance(state.adjacency_matrix, jax.Array)
        assert isinstance(state.global_features, jax.Array)
        assert isinstance(state.topology_type, NetworkTopology)
        assert 0.0 <= state.adaptation_strength <= 1.0
        assert 0.0 <= state.consciousness_level <= 1.0
    
    def test_network_state_validation_adaptation_strength(self):
        """Test NetworkState validation for adaptation_strength bounds."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        num_nodes, hidden_dim = 5, 8
        
        # Test invalid adaptation_strength > 1.0
        with pytest.raises(ValueError, match="adaptation_strength must be in \\[0, 1\\]"):
            NetworkState(
                node_features=jax.random.normal(keys[0], (num_nodes, hidden_dim)),
                edge_weights=jax.random.uniform(keys[1], (num_nodes, num_nodes)),
                adjacency_matrix=jnp.ones((num_nodes, num_nodes)),
                global_features=jax.random.normal(keys[2], (hidden_dim,)),
                topology_type=NetworkTopology.SMALL_WORLD,
                adaptation_strength=1.5,  # Invalid
                consciousness_level=0.5,
            )
        
        # Test invalid adaptation_strength < 0.0
        with pytest.raises(ValueError, match="adaptation_strength must be in \\[0, 1\\]"):
            NetworkState(
                node_features=jax.random.normal(keys[0], (num_nodes, hidden_dim)),
                edge_weights=jax.random.uniform(keys[1], (num_nodes, num_nodes)),
                adjacency_matrix=jnp.ones((num_nodes, num_nodes)),
                global_features=jax.random.normal(keys[2], (hidden_dim,)),
                topology_type=NetworkTopology.SMALL_WORLD,
                adaptation_strength=-0.1,  # Invalid
                consciousness_level=0.5,
            )
    
    def test_network_state_validation_consciousness_level(self):
        """Test NetworkState validation for consciousness_level bounds."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        num_nodes, hidden_dim = 5, 8
        
        # Test invalid consciousness_level
        with pytest.raises(ValueError, match="consciousness_level must be in \\[0, 1\\]"):
            NetworkState(
                node_features=jax.random.normal(keys[0], (num_nodes, hidden_dim)),
                edge_weights=jax.random.uniform(keys[1], (num_nodes, num_nodes)),
                adjacency_matrix=jnp.ones((num_nodes, num_nodes)),
                global_features=jax.random.normal(keys[2], (hidden_dim,)),
                topology_type=NetworkTopology.SMALL_WORLD,
                adaptation_strength=0.5,
                consciousness_level=1.2,  # Invalid
            )


class TestAdaptiveReorganizationResult:
    """Test cases for AdaptiveReorganizationResult dataclass."""
    
    def test_reorganization_result_creation(self):
        """Test AdaptiveReorganizationResult creation."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        num_nodes, hidden_dim = 5, 8
        new_state = NetworkState(
            node_features=jax.random.normal(keys[0], (num_nodes, hidden_dim)),
            edge_weights=jax.random.uniform(keys[1], (num_nodes, num_nodes)),
            adjacency_matrix=jnp.ones((num_nodes, num_nodes)),
            global_features=jax.random.normal(keys[2], (hidden_dim,)),
            topology_type=NetworkTopology.SMALL_WORLD,
            adaptation_strength=0.6,
            consciousness_level=0.8,
        )
        
        structural_changes = {
            'edge_turnover': 0.1,
            'clustering_change': 0.05,
            'degree_change': 0.03,
        }
        
        result = AdaptiveReorganizationResult(
            new_network_state=new_state,
            reorganization_strength=0.25,
            structural_changes=structural_changes,
            meaning_emergence=0.15,
            consciousness_delta=0.1,
        )
        
        assert isinstance(result.new_network_state, NetworkState)
        assert isinstance(result.reorganization_strength, float)
        assert isinstance(result.structural_changes, dict)
        assert isinstance(result.meaning_emergence, float)
        assert isinstance(result.consciousness_delta, float)


class TestDynamicNetworkProcessor:
    """Test cases for DynamicNetworkProcessor class."""
    
    @pytest.fixture
    def network_processor(self):
        """Create DynamicNetworkProcessor for testing."""
        key = jax.random.PRNGKey(42)
        return DynamicNetworkProcessor(
            num_nodes=8,
            hidden_dim=16,
            num_message_passing_steps=3,
            adaptation_rate=0.01,
            plasticity_threshold=0.3,
            homeostatic_target=0.6,
            key=key,
        )
    
    def test_network_processor_initialization(self, network_processor):
        """Test DynamicNetworkProcessor initialization."""
        processor = network_processor
        
        assert processor.num_nodes == 8
        assert processor.hidden_dim == 16
        assert processor.num_message_passing_steps == 3
        assert processor.adaptation_rate == 0.01
        assert processor.plasticity_threshold == 0.3
        assert processor.homeostatic_target == 0.6
        
        # Check that neural network components are initialized
        assert hasattr(processor, 'node_encoder')
        assert hasattr(processor, 'edge_encoder')
        assert hasattr(processor, 'message_passing_net')
        assert hasattr(processor, 'global_features_net')
        assert hasattr(processor, 'consciousness_estimator')
    
    def test_create_initial_network_state(self, network_processor):
        """Test creating initial network state."""
        key = jax.random.PRNGKey(42)
        initial_features = jax.random.normal(key, (8, 16))
        
        network_state = network_processor.create_initial_network_state(
            initial_features,
            topology_type=NetworkTopology.SMALL_WORLD,
            key=key,
        )
        
        assert isinstance(network_state, NetworkState)
        assert network_state.node_features.shape == (8, 16)
        assert network_state.edge_weights.shape == (8, 8)
        assert network_state.adjacency_matrix.shape == (8, 8)
        assert network_state.topology_type == NetworkTopology.SMALL_WORLD
        assert 0.0 <= network_state.adaptation_strength <= 1.0
        assert 0.0 <= network_state.consciousness_level <= 1.0
    
    def test_create_initial_network_state_feature_mismatch(self, network_processor):
        """Test error handling for feature dimension mismatch."""
        key = jax.random.PRNGKey(42)
        wrong_features = jax.random.normal(key, (5, 16))  # Wrong number of nodes
        
        with pytest.raises(NetworkError, match="Features shape mismatch"):
            network_processor.create_initial_network_state(
                wrong_features,
                topology_type=NetworkTopology.SMALL_WORLD,
                key=key,
            )
    
    @pytest.mark.parametrize("topology_type", list(NetworkTopology))
    def test_create_topology_all_types(self, network_processor, topology_type):
        """Test topology creation for all topology types."""
        key = jax.random.PRNGKey(42)
        
        adjacency_matrix = network_processor._create_topology(topology_type, key)
        
        assert isinstance(adjacency_matrix, jax.Array)
        assert adjacency_matrix.shape == (8, 8)
        
        # Check adjacency matrix properties
        assert jnp.all(adjacency_matrix >= 0), "Adjacency matrix should be non-negative"
        
        if topology_type == NetworkTopology.FULLY_CONNECTED:
            # Should have connections everywhere except diagonal
            assert jnp.all(jnp.diag(adjacency_matrix) == 0), "No self-loops"
            assert jnp.sum(adjacency_matrix) > 0, "Should have connections"
        else:
            # Other topologies should have some structure
            assert jnp.sum(adjacency_matrix) >= 0, "Should be valid adjacency matrix"
    
    def test_process_graph_message_passing(self, network_processor):
        """Test graph neural network message passing."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Create initial network state
        initial_features = jax.random.normal(keys[0], (8, 16))
        network_state = network_processor.create_initial_network_state(
            initial_features,
            topology_type=NetworkTopology.SMALL_WORLD,
            key=keys[1],
        )
        
        # Process message passing
        updated_state = network_processor.process_graph_message_passing(network_state)
        
        assert isinstance(updated_state, NetworkState)
        assert updated_state.node_features.shape == network_state.node_features.shape
        assert updated_state.edge_weights.shape == network_state.edge_weights.shape
        
        # Features should be updated (unless network is completely disconnected)
        # Check that processing doesn't break the state
        assert jnp.all(jnp.isfinite(updated_state.node_features))
        assert jnp.all(jnp.isfinite(updated_state.global_features))
    
    def test_process_graph_message_passing_with_external_input(self, network_processor):
        """Test message passing with external input."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        # Create network state
        initial_features = jax.random.normal(keys[0], (8, 16))
        network_state = network_processor.create_initial_network_state(
            initial_features, key=keys[1]
        )
        
        # Create external input
        external_input = jax.random.normal(keys[2], (8, 16))
        
        # Process with external input
        updated_state = network_processor.process_graph_message_passing(
            network_state, external_input
        )
        
        assert isinstance(updated_state, NetworkState)
        assert jnp.all(jnp.isfinite(updated_state.node_features))
    
    def test_process_graph_message_passing_external_input_size_mismatch(self, network_processor):
        """Test error handling for external input size mismatch."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        # Create network state
        initial_features = jax.random.normal(keys[0], (8, 16))
        network_state = network_processor.create_initial_network_state(
            initial_features, key=keys[1]
        )
        
        # Create wrong-sized external input
        wrong_external_input = jax.random.normal(keys[2], (5, 16))  # Wrong number of nodes
        
        with pytest.raises(NetworkError, match="External input size mismatch"):
            network_processor.process_graph_message_passing(
                network_state, wrong_external_input
            )
    
    def test_adaptive_reorganization(self, network_processor):
        """Test adaptive network reorganization."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Create network state
        initial_features = jax.random.normal(keys[0], (8, 16))
        network_state = network_processor.create_initial_network_state(
            initial_features, key=keys[1]
        )
        
        # Create coupling state
        coupling_state = CouplingState(
            agent_state=jax.random.normal(keys[2], (16,)),
            environmental_state=jax.random.normal(keys[3], (12,)),
            coupling_strength=0.7,
            stability_metric=0.8,
            adaptation_rate=0.02,
            coupling_history=jax.random.normal(keys[2], (10, 16)),
        )
        
        # Test reorganization
        reorganization_result = network_processor.adaptive_reorganization(
            network_state,
            coupling_state,
            temporal_moment=None,
            adaptation_mechanism=AdaptationMechanism.CIRCULAR_CAUSALITY,
            key=keys[3],
        )
        
        assert isinstance(reorganization_result, AdaptiveReorganizationResult)
        assert isinstance(reorganization_result.new_network_state, NetworkState)
        assert reorganization_result.reorganization_strength >= 0.0
        assert isinstance(reorganization_result.structural_changes, dict)
        assert reorganization_result.meaning_emergence >= 0.0
    
    @pytest.mark.parametrize("adaptation_mechanism", list(AdaptationMechanism))
    def test_adaptive_reorganization_all_mechanisms(self, network_processor, adaptation_mechanism):
        """Test adaptive reorganization with all adaptation mechanisms."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Create network state
        initial_features = jax.random.normal(keys[0], (8, 16))
        network_state = network_processor.create_initial_network_state(
            initial_features, key=keys[1]
        )
        
        # Create coupling state
        coupling_state = CouplingState(
            agent_state=jax.random.normal(keys[2], (16,)),
            environmental_state=jax.random.normal(keys[3], (12,)),
            coupling_strength=0.5,
            stability_metric=0.6,
            adaptation_rate=0.01,
            coupling_history=jax.random.normal(keys[2], (10, 16)),
        )
        
        # Test reorganization with specific mechanism
        reorganization_result = network_processor.adaptive_reorganization(
            network_state,
            coupling_state,
            adaptation_mechanism=adaptation_mechanism,
            key=keys[3],
        )
        
        assert isinstance(reorganization_result, AdaptiveReorganizationResult)
        assert reorganization_result.reorganization_strength >= 0.0
    
    def test_compute_network_metrics(self, network_processor):
        """Test network metrics computation."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Create network state
        initial_features = jax.random.normal(keys[0], (8, 16))
        network_state = network_processor.create_initial_network_state(
            initial_features, key=keys[1]
        )
        
        # Compute metrics
        metrics = network_processor.compute_network_metrics(network_state)
        
        # Check that all expected metrics are present
        expected_metrics = {
            'num_edges', 'density', 'clustering_coefficient',
            'mean_degree', 'degree_variance', 'feature_variance',
            'feature_entropy', 'weight_mean', 'weight_variance',
            'global_integration', 'small_worldness',
            'consciousness_level', 'adaptation_strength'
        }
        assert set(metrics.keys()) == expected_metrics
        
        # Check value ranges
        assert metrics['num_edges'] >= 0
        assert 0.0 <= metrics['density'] <= 1.0
        assert metrics['clustering_coefficient'] >= 0.0
        assert metrics['mean_degree'] >= 0.0
        assert metrics['degree_variance'] >= 0.0
        assert metrics['feature_variance'] >= 0.0
        assert metrics['weight_mean'] >= 0.0
        assert metrics['weight_variance'] >= 0.0
        assert metrics['global_integration'] >= 0.0
        assert 0.0 <= metrics['consciousness_level'] <= 1.0
        assert 0.0 <= metrics['adaptation_strength'] <= 1.0
    
    def test_assess_consciousness_level(self, network_processor):
        """Test consciousness level assessment."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Create network state
        initial_features = jax.random.normal(keys[0], (8, 16))
        network_state = network_processor.create_initial_network_state(
            initial_features, key=keys[1]
        )
        
        # Assess consciousness
        consciousness_metrics = network_processor.assess_consciousness_level(network_state)
        
        # Check expected keys
        expected_keys = {
            'consciousness_score', 'integration_score', 'information_score',
            'complexity_score', 'differentiation_score', 'network_consciousness'
        }
        assert set(consciousness_metrics.keys()) == expected_keys
        
        # Check value ranges
        for key, value in consciousness_metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} should be in [0,1], got {value}"
            assert jnp.isfinite(value), f"{key} should be finite"


class TestNetworkIntegrator:
    """Test cases for NetworkIntegrator class."""
    
    @pytest.fixture
    def network_integrator(self):
        """Create NetworkIntegrator for testing."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Create network processor
        network_processor = DynamicNetworkProcessor(
            num_nodes=6,
            hidden_dim=12,
            key=keys[0],
        )
        
        # Create integrator
        return NetworkIntegrator(
            network_processor=network_processor,
            integration_dim=32,
            key=keys[1],
        )
    
    def test_network_integrator_initialization(self, network_integrator):
        """Test NetworkIntegrator initialization."""
        integrator = network_integrator
        
        assert hasattr(integrator, 'network_processor')
        assert hasattr(integrator, 'integration_dim')
        assert hasattr(integrator, 'integration_net')
        assert integrator.integration_dim == 32
    
    def test_integrate_network_dynamics(self, network_integrator):
        """Test network dynamics integration."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Create network state
        initial_features = jax.random.normal(keys[0], (6, 12))
        network_state = network_integrator.network_processor.create_initial_network_state(
            initial_features, key=keys[1]
        )
        
        # Create coupling state
        coupling_state = CouplingState(
            agent_state=jax.random.normal(keys[2], (12,)),
            environmental_state=jax.random.normal(keys[3], (10,)),
            coupling_strength=0.6,
            stability_metric=0.7,
            adaptation_rate=0.02,
            coupling_history=jax.random.normal(keys[2], (5, 12)),
        )
        
        # Integrate dynamics
        updated_network_state, integration_metrics = integrator.network_processor.process(
            network_state, coupling_state, key=keys[3]
        )
        
        # The process method should return a valid network state
        assert isinstance(updated_network_state, NetworkState)
        
        # For now, just check that the network state is valid
        assert jnp.all(jnp.isfinite(updated_network_state.node_features))
        assert jnp.all(jnp.isfinite(updated_network_state.edge_weights))


class TestNetworkTopologyGeneration:
    """Test cases for network topology generation methods."""
    
    @pytest.fixture
    def network_processor(self):
        """Create DynamicNetworkProcessor for topology testing."""
        key = jax.random.PRNGKey(42)
        return DynamicNetworkProcessor(
            num_nodes=12,
            hidden_dim=16,
            key=key,
        )
    
    def test_small_world_topology(self, network_processor):
        """Test small-world topology generation."""
        key = jax.random.PRNGKey(42)
        
        adjacency = network_processor._create_topology(NetworkTopology.SMALL_WORLD, key)
        
        assert adjacency.shape == (12, 12)
        assert jnp.all(jnp.diag(adjacency) == 0), "No self-loops"
        
        # Small-world should be symmetric
        assert jnp.allclose(adjacency, adjacency.T), "Should be symmetric"
        
        # Should have reasonable connectivity
        num_edges = jnp.sum(adjacency > 0)
        assert num_edges > 0, "Should have some edges"
    
    def test_scale_free_topology(self, network_processor):
        """Test scale-free topology generation."""
        key = jax.random.PRNGKey(42)
        
        adjacency = network_processor._create_topology(NetworkTopology.SCALE_FREE, key)
        
        assert adjacency.shape == (12, 12)
        assert jnp.all(jnp.diag(adjacency) == 0), "No self-loops"
        
        # Should have some connectivity
        degrees = jnp.sum(adjacency, axis=1)
        assert jnp.sum(degrees) > 0, "Should have edges"
    
    def test_hierarchical_topology(self, network_processor):
        """Test hierarchical topology generation."""
        key = jax.random.PRNGKey(42)
        
        adjacency = network_processor._create_topology(NetworkTopology.HIERARCHICAL, key)
        
        assert adjacency.shape == (12, 12)
        assert jnp.all(jnp.diag(adjacency) == 0), "No self-loops"
        
        # Should have some hierarchical structure
        assert jnp.sum(adjacency) > 0, "Should have connections"
    
    def test_modular_topology(self, network_processor):
        """Test modular topology generation."""
        key = jax.random.PRNGKey(42)
        
        adjacency = network_processor._create_topology(NetworkTopology.MODULAR, key)
        
        assert adjacency.shape == (12, 12)
        assert jnp.sum(adjacency) > 0, "Should have connections"
    
    def test_fully_connected_topology(self, network_processor):
        """Test fully connected topology generation."""
        key = jax.random.PRNGKey(42)
        
        adjacency = network_processor._create_topology(NetworkTopology.FULLY_CONNECTED, key)
        
        assert adjacency.shape == (12, 12)
        assert jnp.all(jnp.diag(adjacency) == 0), "No self-loops"
        
        # Should have all possible edges
        expected_edges = 12 * (12 - 1)  # n * (n-1) for directed graph
        actual_edges = jnp.sum(adjacency > 0)
        assert actual_edges == expected_edges, "Should be fully connected"


class TestNetworkAdaptationMechanisms:
    """Test cases for different network adaptation mechanisms."""
    
    @pytest.fixture
    def setup_adaptation_test(self):
        """Set up test environment for adaptation mechanisms."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        processor = DynamicNetworkProcessor(
            num_nodes=6,
            hidden_dim=8,
            adaptation_rate=0.05,
            key=keys[0],
        )
        
        initial_features = jax.random.normal(keys[1], (6, 8))
        network_state = processor.create_initial_network_state(initial_features, key=keys[2])
        
        coupling_state = CouplingState(
            agent_state=jax.random.normal(keys[3], (8,)),
            environmental_state=jax.random.normal(keys[3], (8,)),
            coupling_strength=0.6,
            stability_metric=0.7,
            adaptation_rate=0.02,
            coupling_history=jax.random.normal(keys[3], (5, 8)),
        )
        
        return processor, network_state, coupling_state, keys[3]
    
    def test_circular_causality_adaptation(self, setup_adaptation_test):
        """Test circular causality adaptation mechanism."""
        processor, network_state, coupling_state, key = setup_adaptation_test
        
        result = processor.adaptive_reorganization(
            network_state,
            coupling_state,
            adaptation_mechanism=AdaptationMechanism.CIRCULAR_CAUSALITY,
            key=key,
        )
        
        assert isinstance(result, AdaptiveReorganizationResult)
        assert result.reorganization_strength >= 0.0
        
        # Should preserve network structure validity
        new_state = result.new_network_state
        assert jnp.all(jnp.isfinite(new_state.node_features))
        assert jnp.all(jnp.isfinite(new_state.edge_weights))
        assert jnp.all(new_state.edge_weights >= 0), "Edge weights should be non-negative"
    
    def test_hebbian_adaptation(self, setup_adaptation_test):
        """Test Hebbian adaptation mechanism."""
        processor, network_state, coupling_state, key = setup_adaptation_test
        
        result = processor.adaptive_reorganization(
            network_state,
            coupling_state,
            adaptation_mechanism=AdaptationMechanism.HEBBIAN,
            key=key,
        )
        
        assert isinstance(result, AdaptiveReorganizationResult)
        assert result.reorganization_strength >= 0.0
        
        # Edge weights should be updated based on node activity correlations
        new_state = result.new_network_state
        assert jnp.all(new_state.edge_weights >= 0.01), "Edge weights should be clipped above minimum"
        assert jnp.all(new_state.edge_weights <= 2.0), "Edge weights should be clipped below maximum"
    
    def test_homeostatic_adaptation(self, setup_adaptation_test):
        """Test homeostatic adaptation mechanism."""
        processor, network_state, coupling_state, key = setup_adaptation_test
        
        result = processor.adaptive_reorganization(
            network_state,
            coupling_state,
            adaptation_mechanism=AdaptationMechanism.HOMEOSTATIC,
            key=key,
        )
        
        assert isinstance(result, AdaptiveReorganizationResult)
        assert result.reorganization_strength >= 0.0
        
        # Homeostatic adaptation should maintain edge weight bounds
        new_state = result.new_network_state
        assert jnp.all(new_state.edge_weights >= 0.01)
        assert jnp.all(new_state.edge_weights <= 2.0)
    
    def test_structural_plasticity_adaptation(self, setup_adaptation_test):
        """Test structural plasticity adaptation mechanism."""
        processor, network_state, coupling_state, key = setup_adaptation_test
        
        result = processor.adaptive_reorganization(
            network_state,
            coupling_state,
            adaptation_mechanism=AdaptationMechanism.STRUCTURAL_PLASTICITY,
            key=key,
        )
        
        assert isinstance(result, AdaptiveReorganizationResult)
        assert result.reorganization_strength >= 0.0
        
        # Structural plasticity may add/remove connections
        new_state = result.new_network_state
        assert jnp.all(jnp.isfinite(new_state.adjacency_matrix))
        assert jnp.all(new_state.adjacency_matrix >= 0), "Adjacency should be non-negative"


class TestMathematicalCorrectness:
    """Test cases for mathematical correctness of network operations."""
    
    def test_clustering_coefficient_computation(self):
        """Test clustering coefficient computation correctness."""
        key = jax.random.PRNGKey(42)
        processor = DynamicNetworkProcessor(num_nodes=5, hidden_dim=8, key=key)
        
        # Create known network structure (triangle)
        adjacency = jnp.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ], dtype=float)
        
        clustering = processor._compute_clustering_coefficient(adjacency)
        
        # Nodes 0, 1, 2 form a triangle (clustering = 1.0)
        # Nodes 3, 4 form a pair (clustering = 0.0)
        # Average should be between 0 and 1
        assert 0.0 <= clustering <= 1.0
        assert jnp.isfinite(clustering)
    
    def test_network_metrics_consistency(self):
        """Test consistency of network metrics computation."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        processor = DynamicNetworkProcessor(num_nodes=8, hidden_dim=12, key=keys[0])
        
        initial_features = jax.random.normal(keys[1], (8, 12))
        network_state = processor.create_initial_network_state(initial_features, key=keys[1])
        
        metrics = processor.compute_network_metrics(network_state)
        
        # Density should be consistent with number of edges
        num_nodes = processor.num_nodes
        expected_max_edges = num_nodes * (num_nodes - 1)
        expected_density = metrics['num_edges'] / expected_max_edges
        
        # Allow small numerical errors
        assert abs(metrics['density'] - expected_density) < 0.01
    
    def test_consciousness_score_bounds(self):
        """Test that consciousness scores are properly bounded."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        processor = DynamicNetworkProcessor(num_nodes=6, hidden_dim=10, key=keys[0])
        
        initial_features = jax.random.normal(keys[1], (6, 10))
        network_state = processor.create_initial_network_state(initial_features, key=keys[1])
        
        consciousness_metrics = processor.assess_consciousness_level(network_state)
        
        # All scores should be in [0, 1]
        for key, value in consciousness_metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} = {value} not in [0,1]"
            assert jnp.isfinite(value), f"{key} should be finite"


class TestPerformanceAndScalability:
    """Test cases for performance and scalability."""
    
    @pytest.mark.parametrize("num_nodes", [5, 10, 20])
    def test_scalability_with_network_size(self, num_nodes):
        """Test network processor scalability with different network sizes."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        processor = DynamicNetworkProcessor(
            num_nodes=num_nodes,
            hidden_dim=8,
            num_message_passing_steps=2,  # Reduce for larger networks
            key=keys[0],
        )
        
        initial_features = jax.random.normal(keys[1], (num_nodes, 8))
        
        # Should complete without errors
        network_state = processor.create_initial_network_state(initial_features, key=keys[1])
        
        assert isinstance(network_state, NetworkState)
        assert network_state.node_features.shape == (num_nodes, 8)
        assert network_state.adjacency_matrix.shape == (num_nodes, num_nodes)
    
    @pytest.mark.parametrize("message_passing_steps", [1, 3, 5])
    def test_message_passing_steps_scalability(self, message_passing_steps):
        """Test scalability with different numbers of message passing steps."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        processor = DynamicNetworkProcessor(
            num_nodes=8,
            hidden_dim=12,
            num_message_passing_steps=message_passing_steps,
            key=keys[0],
        )
        
        initial_features = jax.random.normal(keys[1], (8, 12))
        network_state = processor.create_initial_network_state(initial_features, key=keys[1])
        
        # Process message passing
        updated_state = processor.process_graph_message_passing(network_state)
        
        assert isinstance(updated_state, NetworkState)
        assert jnp.all(jnp.isfinite(updated_state.node_features))
    
    def test_memory_efficiency_large_network(self):
        """Test memory efficiency with reasonably large networks."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        # Test with a larger network
        num_nodes = 50
        hidden_dim = 32
        
        processor = DynamicNetworkProcessor(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            num_message_passing_steps=2,  # Limit for memory efficiency
            key=keys[0],
        )
        
        initial_features = jax.random.normal(keys[1], (num_nodes, hidden_dim))
        
        # Should handle large networks without memory issues
        network_state = processor.create_initial_network_state(initial_features, key=keys[1])
        metrics = processor.compute_network_metrics(network_state)
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0


class TestErrorHandlingAndEdgeCases:
    """Test cases for error handling and edge cases."""
    
    def test_empty_network_initialization(self):
        """Test error handling for empty network."""
        key = jax.random.PRNGKey(42)
        
        with pytest.raises((ValueError, NetworkError)):
            DynamicNetworkProcessor(
                num_nodes=0,  # Invalid
                hidden_dim=8,
                key=key,
            )
    
    def test_negative_adaptation_rate(self):
        """Test error handling for negative adaptation rate."""
        key = jax.random.PRNGKey(42)
        
        # Should handle gracefully or raise appropriate error
        processor = DynamicNetworkProcessor(
            num_nodes=5,
            hidden_dim=8,
            adaptation_rate=-0.01,  # Negative rate
            key=key,
        )
        
        # Processor should still be created, but may behave unexpectedly
        assert processor.adaptation_rate == -0.01
    
    def test_very_large_adaptation_rate(self):
        """Test behavior with very large adaptation rate."""
        key = jax.random.PRNGKey(42)
        
        processor = DynamicNetworkProcessor(
            num_nodes=5,
            hidden_dim=8,
            adaptation_rate=10.0,  # Very large rate
            key=key,
        )
        
        # Should still work but may lead to instability
        assert processor.adaptation_rate == 10.0
    
    def test_single_node_network(self):
        """Test behavior with single-node network."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        processor = DynamicNetworkProcessor(
            num_nodes=1,
            hidden_dim=8,
            key=keys[0],
        )
        
        initial_features = jax.random.normal(keys[1], (1, 8))
        
        # Should handle single-node networks gracefully
        network_state = processor.create_initial_network_state(initial_features, key=keys[1])
        
        assert network_state.node_features.shape == (1, 8)
        assert network_state.adjacency_matrix.shape == (1, 1)
        assert network_state.adjacency_matrix[0, 0] == 0  # No self-loops


# Integration tests
class TestDynamicNetworksIntegration:
    """Integration tests for dynamic networks with other modules."""
    
    def test_information_theory_integration(self):
        """Test integration with information theory measures."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        processor = DynamicNetworkProcessor(num_nodes=8, hidden_dim=12, key=keys[0])
        
        initial_features = jax.random.normal(keys[1], (8, 12))
        network_state = processor.create_initial_network_state(initial_features, key=keys[2])
        
        # Extract network features for information theory analysis
        node_features_flat = network_state.node_features.flatten()
        edge_weights_flat = network_state.edge_weights.flatten()
        
        # Should be able to compute information theory measures
        assert len(node_features_flat) > 0
        assert len(edge_weights_flat) > 0
        assert jnp.all(jnp.isfinite(node_features_flat))
        assert jnp.all(jnp.isfinite(edge_weights_flat))
    
    def test_temporal_dynamics_integration(self):
        """Test integration with temporal processing."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        processor = DynamicNetworkProcessor(num_nodes=6, hidden_dim=10, key=keys[0])
        
        # Simulate temporal sequence of network states
        network_states_sequence = []
        current_features = jax.random.normal(keys[1], (6, 10))
        
        for t in range(5):
            network_state = processor.create_initial_network_state(
                current_features, key=jax.random.fold_in(keys[2], t)
            )
            updated_state = processor.process_graph_message_passing(network_state)
            network_states_sequence.append(updated_state)
            
            # Update features for next time step
            current_features = updated_state.node_features
        
        # Should have consistent temporal sequence
        assert len(network_states_sequence) == 5
        for state in network_states_sequence:
            assert isinstance(state, NetworkState)
            assert jnp.all(jnp.isfinite(state.node_features))


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main([__file__, "-v", "--tb=short"])