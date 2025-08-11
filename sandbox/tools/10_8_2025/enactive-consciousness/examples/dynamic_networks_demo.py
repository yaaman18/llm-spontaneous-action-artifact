#!/usr/bin/env python3
"""
Dynamic Networks Demo for Enactive Consciousness.

This demo showcases the dynamic network capabilities including:
1. Adaptive graph structures for cognitive reorganization
2. Graph neural networks for meaning emergence  
3. Dynamic connectivity patterns for agent-environment interaction
4. Network metrics for consciousness assessment
5. Integration with circular causality and information theory

The demo follows Varela-Maturana principles of structural coupling
and demonstrates how networks can dynamically reorganize based on
experiential interactions.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Dict

# Import our enactive consciousness framework
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enactive_consciousness import (
    DynamicNetworkProcessor,
    NetworkIntegrator, 
    NetworkTopology,
    AdaptationMechanism,
    NetworkState,
    create_framework_config,
    CouplingState,
    TemporalMoment,
    circular_causality_index,
    complexity_measure,
)


def create_sample_coupling_state(key: jax.random.PRNGKey, dim: int = 64) -> CouplingState:
    """Create a sample coupling state for testing."""
    keys = jax.random.split(key, 3)
    
    agent_state = jax.random.normal(keys[0], (dim,))
    environmental_state = jax.random.normal(keys[1], (dim,))
    perturbation_history = jax.random.normal(keys[2], (10, dim))
    
    return CouplingState(
        agent_state=agent_state,
        environmental_state=environmental_state,
        coupling_strength=0.7,
        perturbation_history=perturbation_history,
        stability_metric=0.8,
    )


def create_sample_temporal_moment(key: jax.random.PRNGKey, dim: int = 64) -> TemporalMoment:
    """Create a sample temporal moment for testing."""
    keys = jax.random.split(key, 4)
    
    return TemporalMoment(
        timestamp=1.0,
        retention=jax.random.normal(keys[0], (dim,)),
        present_moment=jax.random.normal(keys[1], (dim,)),
        protention=jax.random.normal(keys[2], (dim,)),
        synthesis_weights=jax.random.uniform(keys[3], (dim,)),
    )


def demo_network_topologies(network_processor: DynamicNetworkProcessor, key: jax.random.PRNGKey):
    """Demonstrate different network topologies."""
    print("\n" + "="*50)
    print("DEMONSTRATING NETWORK TOPOLOGIES")
    print("="*50)
    
    keys = jax.random.split(key, 5)
    initial_features = jax.random.normal(keys[0], (network_processor.num_nodes, network_processor.hidden_dim))
    
    topologies = [
        NetworkTopology.SMALL_WORLD,
        NetworkTopology.SCALE_FREE, 
        NetworkTopology.MODULAR,
        NetworkTopology.HIERARCHICAL,
        NetworkTopology.FULLY_CONNECTED,
    ]
    
    network_states = []
    
    for i, topology in enumerate(topologies):
        print(f"\nCreating {topology.value} topology...")
        
        network_state = network_processor.create_initial_network_state(
            initial_features, topology, key=keys[i+1]
        )
        
        # Compute network metrics
        metrics = network_processor.compute_network_metrics(network_state)
        
        print(f"  Nodes: {network_processor.num_nodes}")
        print(f"  Edges: {metrics['num_edges']}")
        print(f"  Density: {metrics['density']:.3f}")
        print(f"  Clustering coefficient: {metrics['clustering_coefficient']:.3f}")
        print(f"  Mean degree: {metrics['mean_degree']:.3f}")
        print(f"  Small-worldness: {metrics['small_worldness']:.3f}")
        print(f"  Consciousness level: {metrics['consciousness_level']:.3f}")
        
        network_states.append(network_state)
    
    return network_states


def demo_adaptive_reorganization(
    network_processor: DynamicNetworkProcessor,
    initial_state: NetworkState,
    key: jax.random.PRNGKey,
):
    """Demonstrate adaptive network reorganization."""
    print("\n" + "="*50) 
    print("DEMONSTRATING ADAPTIVE REORGANIZATION")
    print("="*50)
    
    keys = jax.random.split(key, 10)
    
    # Create coupling state
    coupling_state = create_sample_coupling_state(keys[0])
    temporal_moment = create_sample_temporal_moment(keys[1])
    
    adaptation_mechanisms = [
        AdaptationMechanism.CIRCULAR_CAUSALITY,
        AdaptationMechanism.HEBBIAN,
        AdaptationMechanism.HOMEOSTATIC,
        AdaptationMechanism.STRUCTURAL_PLASTICITY,
    ]
    
    reorganization_results = []
    
    for i, mechanism in enumerate(adaptation_mechanisms):
        print(f"\nTesting {mechanism.value} adaptation...")
        
        result = network_processor.adaptive_reorganization(
            initial_state,
            coupling_state,
            temporal_moment,
            mechanism,
            key=keys[i+2],
        )
        
        print(f"  Reorganization strength: {result.reorganization_strength:.3f}")
        print(f"  Meaning emergence: {result.meaning_emergence:.3f}")
        print(f"  Consciousness delta: {result.consciousness_delta:.3f}")
        print(f"  Edge turnover: {result.structural_changes['edge_turnover']:.3f}")
        print(f"  Clustering change: {result.structural_changes['clustering_change']:.3f}")
        
        reorganization_results.append(result)
    
    return reorganization_results


def demo_information_theory_integration(
    network_processor: DynamicNetworkProcessor,
    network_state: NetworkState,
    key: jax.random.PRNGKey,
):
    """Demonstrate integration with information theory measures."""
    print("\n" + "="*50)
    print("DEMONSTRATING INFORMATION THEORY INTEGRATION")
    print("="*50)
    
    keys = jax.random.split(key, 5)
    
    # Simulate time series of network and coupling states
    time_steps = 50
    network_states = []
    coupling_states = []
    
    current_state = network_state
    
    for t in range(time_steps):
        # Update network state with message passing
        current_state = network_processor.process_graph_message_passing(current_state)
        network_states.append(current_state.node_features.flatten())
        
        # Create corresponding coupling state
        coupling = create_sample_coupling_state(keys[t % 5])
        coupling_states.append(coupling.agent_state)
    
    # Convert to arrays
    network_time_series = jnp.array(network_states)
    coupling_time_series = jnp.array(coupling_states)
    
    print(f"Generated time series: {network_time_series.shape}")
    
    # Compute information theory measures
    print("\nComputing circular causality...")
    causality_metrics = circular_causality_index(
        coupling_time_series, network_time_series
    )
    
    print(f"  Circular causality index: {causality_metrics['circular_causality']:.3f}")
    print(f"  Transfer entropy (coupling -> network): {causality_metrics['transfer_entropy_env_to_agent']:.3f}")
    print(f"  Transfer entropy (network -> coupling): {causality_metrics['transfer_entropy_agent_to_env']:.3f}")
    print(f"  Coupling coherence: {causality_metrics['coupling_coherence']:.3f}")
    
    print("\nComputing complexity measures...")
    complexity_metrics = complexity_measure(
        network_time_series.mean(axis=1),
        coupling_time_series.mean(axis=1),
    )
    
    print(f"  Overall complexity: {complexity_metrics['overall_complexity']:.3f}")
    print(f"  Network complexity: {complexity_metrics['agent_complexity']:.3f}")
    print(f"  Coupling complexity: {complexity_metrics['environment_complexity']:.3f}")
    print(f"  Interaction complexity: {complexity_metrics['interaction_complexity']:.3f}")
    
    return causality_metrics, complexity_metrics


def demo_consciousness_assessment(
    network_processor: DynamicNetworkProcessor,
    network_states: List[NetworkState],
):
    """Demonstrate consciousness level assessment."""
    print("\n" + "="*50)
    print("DEMONSTRATING CONSCIOUSNESS ASSESSMENT")
    print("="*50)
    
    consciousness_levels = []
    
    for i, state in enumerate(network_states[:3]):  # Test first 3 states
        print(f"\nAssessing consciousness for network {i+1}...")
        
        consciousness_metrics = network_processor.assess_consciousness_level(state)
        
        print(f"  Overall consciousness score: {consciousness_metrics['consciousness_score']:.3f}")
        print(f"  Integration score: {consciousness_metrics['integration_score']:.3f}")
        print(f"  Information score: {consciousness_metrics['information_score']:.3f}")
        print(f"  Complexity score: {consciousness_metrics['complexity_score']:.3f}")
        print(f"  Differentiation score: {consciousness_metrics['differentiation_score']:.3f}")
        
        consciousness_levels.append(consciousness_metrics['consciousness_score'])
    
    print(f"\nConsciousness level progression: {consciousness_levels}")
    return consciousness_levels


def demo_network_integration(
    network_processor: DynamicNetworkProcessor,
    network_state: NetworkState,
    key: jax.random.PRNGKey,
):
    """Demonstrate network integration with coupling and temporal dynamics."""
    print("\n" + "="*50)
    print("DEMONSTRATING NETWORK INTEGRATION")
    print("="*50)
    
    keys = jax.random.split(key, 3)
    
    # Create integrator
    integrator = NetworkIntegrator(network_processor, key=keys[0])
    
    # Create test data
    coupling_state = create_sample_coupling_state(keys[1])
    temporal_moment = create_sample_temporal_moment(keys[2])
    
    print("Integrating network with coupling and temporal dynamics...")
    
    updated_state, integration_metrics = integrator.integrate_network_dynamics(
        network_state, coupling_state, temporal_moment, key=keys[0]
    )
    
    print(f"  Network-coupling strength: {integration_metrics['network_coupling_strength']:.3f}")
    print(f"  Circular causality: {integration_metrics['circular_causality']:.3f}")
    print(f"  Consciousness score: {integration_metrics['consciousness_score']:.3f}")
    print(f"  Integration score: {integration_metrics['integration_score']:.3f}")
    
    # Compare before and after
    original_metrics = network_processor.compute_network_metrics(network_state)
    updated_metrics = network_processor.compute_network_metrics(updated_state)
    
    print(f"\nConsciousness level change: {original_metrics['consciousness_level']:.3f} -> {updated_metrics['consciousness_level']:.3f}")
    print(f"Feature variance change: {original_metrics['feature_variance']:.3f} -> {updated_metrics['feature_variance']:.3f}")
    
    return updated_state, integration_metrics


def visualize_network_topology(network_state: NetworkState, title: str):
    """NetworkXを使用してネットワーク位相を可視化する。"""
    try:
        import matplotlib.pyplot as plt
        
        # 簡略化されたフォント設定
        try:
            plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
        except:
            pass
        
        # NetworkXグラフに変換
        G = nx.from_numpy_array(np.array(network_state.adjacency_matrix))
        
        # レイアウト作成
        pos = nx.spring_layout(G, seed=42)
        
        # プロット
        plt.figure(figsize=(10, 8))
        
        # ノードとエッジの描画
        nx.draw_networkx_nodes(G, pos, node_size=200, alpha=0.8, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # 日本語タイトル
        japanese_title_map = {
            "Initial": "初期",
            "Adapted": "適応後", 
            "Network": "ネットワーク",
            "Dynamic": "動的",
            "Static": "静的"
        }
        
        jp_title = title
        for en, jp in japanese_title_map.items():
            jp_title = jp_title.replace(en, jp)
        
        plt.title(f'{jp_title}ネットワーク構造\\nノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}\\n（動的再構成による自己組織化）', fontsize=12)
        
        # 説明テキスト追加
        plt.text(0.02, 0.98, '【動的ネットワーク】\\n・ノード: 処理単位\\n・エッジ: 情報流\\n・構造: 適応的変化', 
                transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.axis('off')
        plt.tight_layout()
        
        # ノンブロッキング表示
        plt.show(block=False)
        plt.pause(0.1)
        
    except ImportError:
        print("Matplotlibが利用できません")


def main():
    """Main demonstration function."""
    print("Dynamic Networks for Enactive Consciousness - Demo")
    print("="*60)
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)
    
    # Create network processor
    print("Initializing Dynamic Network Processor...")
    network_processor = DynamicNetworkProcessor(
        num_nodes=20,
        hidden_dim=64,
        num_message_passing_steps=3,
        adaptation_rate=0.01,
        key=keys[0],
    )
    print(f"Created network processor with {network_processor.num_nodes} nodes")
    
    # Demo 1: Network topologies
    network_states = demo_network_topologies(network_processor, keys[1])
    
    # Demo 2: Adaptive reorganization
    reorganization_results = demo_adaptive_reorganization(
        network_processor, network_states[0], keys[2]
    )
    
    # Demo 3: Information theory integration
    causality_metrics, complexity_metrics = demo_information_theory_integration(
        network_processor, network_states[0], keys[3]
    )
    
    # Demo 4: Consciousness assessment
    consciousness_levels = demo_consciousness_assessment(
        network_processor, network_states
    )
    
    # Demo 5: Network integration
    updated_state, integration_metrics = demo_network_integration(
        network_processor, network_states[0], keys[4]
    )
    
    print("\n" + "="*60)
    print("SUMMARY OF DEMONSTRATION")
    print("="*60)
    print(f"✓ Demonstrated {len(NetworkTopology)} different network topologies")
    print(f"✓ Tested {len(AdaptationMechanism)} adaptation mechanisms")
    print(f"✓ Computed circular causality: {causality_metrics['circular_causality']:.3f}")
    print(f"✓ Measured complexity: {complexity_metrics['overall_complexity']:.3f}")
    print(f"✓ Assessed consciousness levels: {consciousness_levels}")
    print(f"✓ Integrated network with coupling dynamics")
    
    print("\nKey Findings:")
    print(f"- Networks can dynamically reorganize based on coupling states")
    print(f"- Different topologies show distinct consciousness characteristics")
    print(f"- Circular causality emerges from network-environment interactions")
    print(f"- Information theory measures quantify meaning emergence")
    print(f"- Clean architecture enables extensible consciousness modeling")
    
    print("\nDemo completed successfully! 🧠✨")


if __name__ == "__main__":
    main()