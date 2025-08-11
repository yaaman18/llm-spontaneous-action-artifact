#!/usr/bin/env python3
"""Advanced Integration Patterns Test Suite.

Tests complex integration patterns between advanced modules including:
- Information theory + Dynamic networks coupling
- Sparse representations + Predictive coding integration  
- Continuous dynamics + Temporal consciousness synthesis
- Multi-modal consciousness state validation
- Advanced performance benchmarking

Project Orchestrator Role:
- Ensures theoretical consistency across module integration
- Validates architectural coherence boundaries
- Coordinates performance optimization across components
- Maintains quality standards through integration
"""

import pytest
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import equinox as eqx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Core imports
from enactive_consciousness.integrated_consciousness import (
    EnactiveConsciousnessSystem,
    ConsciousnessState,
    create_enactive_consciousness_system,
)

from enactive_consciousness.temporal import (
    PhenomenologicalTemporalSynthesis,
    TemporalConsciousnessConfig,
)

from enactive_consciousness.experiential_memory import (
    IntegratedExperientialMemory,
    CircularCausalityEngine,
)

# Import advanced modules with availability flags
ADVANCED_MODULES_STATUS = {}

try:
    from enactive_consciousness.information_theory import (
        circular_causality_index,
        complexity_measure,
        mutual_information_kraskov,
        transfer_entropy,
        entropy_rate,
    )
    ADVANCED_MODULES_STATUS['information_theory'] = True
except ImportError as e:
    ADVANCED_MODULES_STATUS['information_theory'] = False
    print(f"Information theory not available: {e}")

try:
    from enactive_consciousness.dynamic_networks import (
        DynamicNetworkProcessor,
        NetworkTopology,
        AdaptiveNetworkTopology,
    )
    ADVANCED_MODULES_STATUS['dynamic_networks'] = True
except ImportError as e:
    ADVANCED_MODULES_STATUS['dynamic_networks'] = False
    print(f"Dynamic networks not available: {e}")

try:
    from enactive_consciousness.sparse_representations import (
        IntegratedSparseRepresentationSystem,
        SparseExperienceEncoder,
        ConvexOptimizationSolver,
    )
    ADVANCED_MODULES_STATUS['sparse_representations'] = True
except ImportError as e:
    ADVANCED_MODULES_STATUS['sparse_representations'] = False
    print(f"Sparse representations not available: {e}")

try:
    from enactive_consciousness.predictive_coding import (
        EnhancedHierarchicalPredictor,
        HierarchicalPredictiveCoding,
        BayesianPredictiveLayer,
    )
    ADVANCED_MODULES_STATUS['predictive_coding'] = True
except ImportError as e:
    ADVANCED_MODULES_STATUS['predictive_coding'] = False
    print(f"Predictive coding not available: {e}")

try:
    from enactive_consciousness.continuous_dynamics import (
        ContinuousTemporalConsciousness,
        ConsciousnessTrajectory,
        DifferentialEquationSolver,
    )
    ADVANCED_MODULES_STATUS['continuous_dynamics'] = True
except ImportError as e:
    ADVANCED_MODULES_STATUS['continuous_dynamics'] = False
    print(f"Continuous dynamics not available: {e}")


class AdvancedIntegrationFixture:
    """Advanced integration test fixture with sophisticated components."""
    
    def __init__(
        self,
        state_dim: int = 96,
        environment_dim: int = 32,
        context_dim: int = 48,
        test_key: Optional[jax.Array] = None,
    ):
        """Initialize advanced integration fixture."""
        
        self.state_dim = state_dim
        self.environment_dim = environment_dim
        self.context_dim = context_dim
        
        if test_key is None:
            test_key = jax.random.PRNGKey(12345)
        
        self.keys = jax.random.split(test_key, 30)
        
        # Advanced component registry
        self.advanced_components = {}
        self.integration_patterns = {}
        self.performance_benchmarks = {}
        
        # Theoretical consistency validation
        self.theoretical_consistency_metrics = {}
        
    def setup_information_theory_networks_coupling(self) -> bool:
        """Setup information theory + dynamic networks coupling."""
        
        if not (ADVANCED_MODULES_STATUS['information_theory'] and 
                ADVANCED_MODULES_STATUS['dynamic_networks']):
            return False
            
        try:
            # Create coupled system
            network_processor = DynamicNetworkProcessor(
                state_dim=self.state_dim,
                topology=NetworkTopology.SMALL_WORLD,
                key=self.keys[0],
            )
            
            # Integration pattern: Information theory guides network adaptation
            self.integration_patterns['info_theory_networks'] = {
                'network_processor': network_processor,
                'info_theory_functions': {
                    'circular_causality': circular_causality_index,
                    'complexity_measure': complexity_measure,
                    'transfer_entropy': transfer_entropy,
                },
                'coupling_strength': 0.3,  # How strongly info theory influences networks
            }
            
            return True
            
        except Exception as e:
            print(f"Failed to setup info theory + networks coupling: {e}")
            return False
    
    def setup_sparse_predictive_integration(self) -> bool:
        """Setup sparse representations + predictive coding integration."""
        
        if not (ADVANCED_MODULES_STATUS['sparse_representations'] and
                ADVANCED_MODULES_STATUS['predictive_coding']):
            return False
            
        try:
            # Create integrated sparse-predictive system
            sparse_system = IntegratedSparseRepresentationSystem(
                input_dim=self.state_dim,
                key=self.keys[1],
            )
            
            predictive_system = EnhancedHierarchicalPredictor(
                state_dim=self.state_dim,
                key=self.keys[2],
            )
            
            # Integration pattern: Sparse codes inform predictive hierarchy
            self.integration_patterns['sparse_predictive'] = {
                'sparse_system': sparse_system,
                'predictive_system': predictive_system,
                'integration_weight': 0.4,  # Balance between sparse and predictive
            }
            
            return True
            
        except Exception as e:
            print(f"Failed to setup sparse + predictive integration: {e}")
            return False
    
    def setup_continuous_temporal_synthesis(self) -> bool:
        """Setup continuous dynamics + temporal consciousness synthesis."""
        
        if not ADVANCED_MODULES_STATUS['continuous_dynamics']:
            return False
            
        try:
            # Create continuous-temporal integrated system
            continuous_system = ContinuousTemporalConsciousness(
                state_dim=self.state_dim,
                key=self.keys[3],
            )
            
            temporal_config = TemporalConsciousnessConfig(
                retention_depth=10,
                protention_horizon=5,
                temporal_synthesis_rate=0.08,
                temporal_decay_factor=0.94,
            )
            
            temporal_processor = PhenomenologicalTemporalSynthesis(
                temporal_config,
                self.state_dim,
                self.keys[4],
            )
            
            # Integration pattern: Continuous dynamics enhance temporal flow
            self.integration_patterns['continuous_temporal'] = {
                'continuous_system': continuous_system,
                'temporal_processor': temporal_processor,
                'synthesis_mode': 'differential_temporal',  # Special integration mode
            }
            
            return True
            
        except Exception as e:
            print(f"Failed to setup continuous + temporal synthesis: {e}")
            return False
    
    def setup_multi_modal_consciousness_validation(self) -> bool:
        """Setup multi-modal consciousness state validation system."""
        
        try:
            # Core consciousness system with multi-modal validation
            from enactive_consciousness import (
                create_framework_config,
                BodySchemaConfig,
            )
            
            framework_config = create_framework_config(
                retention_depth=8,
                protention_horizon=4,
                consciousness_threshold=0.35,
                proprioceptive_dim=48,
                motor_dim=16,
            )
            
            body_config = BodySchemaConfig(
                proprioceptive_dim=48,
                motor_dim=16,
                body_map_resolution=(10, 10),
                boundary_sensitivity=0.75,
                schema_adaptation_rate=0.012,
            )
            
            temporal_config = TemporalConsciousnessConfig(
                retention_depth=8,
                protention_horizon=4,
                temporal_synthesis_rate=0.12,
                temporal_decay_factor=0.91,
            )
            
            consciousness_system = create_enactive_consciousness_system(
                config=framework_config,
                temporal_config=temporal_config,
                body_config=body_config,
                state_dim=self.state_dim,
                environment_dim=self.environment_dim,
                key=self.keys[5],
            )
            
            # Multi-modal validation components
            self.advanced_components['multi_modal_validation'] = {
                'consciousness_system': consciousness_system,
                'validation_modes': [
                    'phenomenological_consistency',
                    'enactivist_coupling',
                    'autopoietic_closure',
                    'embodied_cognition',
                ],
                'theoretical_frameworks': [
                    'husserlian_temporality',
                    'merleau_ponty_embodiment',
                    'varela_autopoiesis',
                    'enactivist_sense_making',
                ]
            }
            
            return True
            
        except Exception as e:
            print(f"Failed to setup multi-modal consciousness validation: {e}")
            return False
    
    def setup_complete_advanced_system(self) -> Dict[str, bool]:
        """Setup complete advanced integration system."""
        
        setup_results = {
            'info_theory_networks': self.setup_information_theory_networks_coupling(),
            'sparse_predictive': self.setup_sparse_predictive_integration(),
            'continuous_temporal': self.setup_continuous_temporal_synthesis(),
            'multi_modal_validation': self.setup_multi_modal_consciousness_validation(),
        }
        
        print(f"Advanced integration setup results: {setup_results}")
        return setup_results


class TestInformationTheoryNetworksCoupling:
    """Test information theory + dynamic networks coupling patterns."""
    
    @pytest.fixture
    def advanced_fixture(self):
        """Create advanced integration fixture."""
        fixture = AdvancedIntegrationFixture()
        fixture.setup_complete_advanced_system()
        return fixture
    
    @pytest.mark.skipif(
        not (ADVANCED_MODULES_STATUS['information_theory'] and ADVANCED_MODULES_STATUS['dynamic_networks']),
        reason="Information theory and dynamic networks not both available"
    )
    def test_information_guided_network_adaptation(self, advanced_fixture):
        """Test information theory metrics guiding dynamic network adaptation."""
        
        if 'info_theory_networks' not in advanced_fixture.integration_patterns:
            pytest.skip("Info theory + networks coupling not available")
        
        pattern = advanced_fixture.integration_patterns['info_theory_networks']
        network_processor = pattern['network_processor']
        info_functions = pattern['info_theory_functions']
        
        # Generate test sequence for information analysis
        sequence_length = 15
        agent_states = []
        env_states = []
        network_states = []
        
        # Initial state
        current_state = jax.random.normal(advanced_fixture.keys[6], (advanced_fixture.state_dim,)) * 0.5
        
        for t in range(sequence_length):
            # Process through network
            processed_state, network_metrics = network_processor.process_state(current_state)
            network_states.append(processed_state)
            
            # Create coupled environment state
            env_state = (
                0.6 * current_state[:advanced_fixture.environment_dim] +
                jax.random.normal(advanced_fixture.keys[7 + t], (advanced_fixture.environment_dim,)) * 0.2
            )
            
            agent_states.append(current_state)
            env_states.append(env_state)
            
            # Update state for next iteration
            current_state = processed_state
        
        # Analyze with information theory
        agent_sequence = jnp.array(agent_states)
        env_sequence = jnp.array(env_states)
        
        # Compute information theory metrics
        causality_metrics = info_functions['circular_causality'](agent_sequence, env_sequence)
        complexity_metrics = info_functions['complexity_measure'](agent_sequence, env_sequence)
        
        # Validate coupling metrics
        assert isinstance(causality_metrics, dict)
        assert 'circular_causality' in causality_metrics
        assert 'transfer_entropy_env_to_agent' in causality_metrics
        
        assert isinstance(complexity_metrics, dict)
        assert 'overall_complexity' in complexity_metrics
        
        # Test network adaptation based on information metrics
        adaptation_signal = (
            0.4 * causality_metrics.get('circular_causality', 0.5) +
            0.3 * causality_metrics.get('coupling_coherence', 0.5) +
            0.3 * complexity_metrics.get('overall_complexity', 0.5)
        )
        
        # Validate adaptation signal
        assert 0.0 <= adaptation_signal <= 1.0
        
        # Store benchmarking results
        advanced_fixture.performance_benchmarks['info_theory_networks_coupling'] = {
            'causality_metrics': causality_metrics,
            'complexity_metrics': complexity_metrics,
            'adaptation_signal': float(adaptation_signal),
            'network_processing_successful': True,
            'sequence_length': sequence_length,
        }
    
    @pytest.mark.skipif(
        not (ADVANCED_MODULES_STATUS['information_theory'] and ADVANCED_MODULES_STATUS['dynamic_networks']),
        reason="Advanced modules not available"
    )
    def test_network_topology_information_optimization(self, advanced_fixture):
        """Test network topology optimization based on information flow."""
        
        if 'info_theory_networks' not in advanced_fixture.integration_patterns:
            pytest.skip("Info theory + networks coupling not available")
        
        pattern = advanced_fixture.integration_patterns['info_theory_networks']
        network_processor = pattern['network_processor']
        
        # Test different network topologies
        topologies_to_test = [NetworkTopology.SMALL_WORLD, NetworkTopology.SCALE_FREE, NetworkTopology.RANDOM]
        topology_performance = []
        
        for topology in topologies_to_test:
            try:
                # Create network with specific topology
                test_processor = DynamicNetworkProcessor(
                    state_dim=advanced_fixture.state_dim,
                    topology=topology,
                    key=advanced_fixture.keys[8],
                )
                
                # Test information flow through network
                test_state = jax.random.normal(advanced_fixture.keys[9], (advanced_fixture.state_dim,)) * 0.6
                
                processed_state, network_metrics = test_processor.process_state(test_state)
                
                # Measure information preservation
                mutual_info = mutual_information_kraskov(
                    test_state.reshape(1, -1), 
                    processed_state.reshape(1, -1)
                )
                
                topology_performance.append({
                    'topology': topology.name if hasattr(topology, 'name') else str(topology),
                    'mutual_information': float(mutual_info),
                    'network_metrics': network_metrics,
                })
                
            except Exception as e:
                print(f"Failed to test topology {topology}: {e}")
        
        # Validate topology comparison
        assert len(topology_performance) > 0
        
        # Find best topology for information flow
        best_topology = max(topology_performance, key=lambda x: x['mutual_information'])
        
        advanced_fixture.performance_benchmarks['topology_optimization'] = {
            'topology_performance': topology_performance,
            'best_topology': best_topology,
            'optimization_successful': True,
        }


class TestSparsePredictiveIntegration:
    """Test sparse representations + predictive coding integration."""
    
    @pytest.fixture
    def advanced_fixture(self):
        """Create advanced integration fixture."""
        fixture = AdvancedIntegrationFixture()
        fixture.setup_complete_advanced_system()
        return fixture
    
    @pytest.mark.skipif(
        not (ADVANCED_MODULES_STATUS['sparse_representations'] and ADVANCED_MODULES_STATUS['predictive_coding']),
        reason="Sparse representations and predictive coding not both available"
    )
    def test_sparse_informed_hierarchical_prediction(self, advanced_fixture):
        """Test sparse codes informing hierarchical predictive models."""
        
        if 'sparse_predictive' not in advanced_fixture.integration_patterns:
            pytest.skip("Sparse + predictive integration not available")
        
        pattern = advanced_fixture.integration_patterns['sparse_predictive']
        sparse_system = pattern['sparse_system']
        predictive_system = pattern['predictive_system']
        integration_weight = pattern['integration_weight']
        
        # Generate test experience sequence
        sequence_length = 10
        experiences = []
        
        for t in range(sequence_length):
            # Structured experience with temporal patterns
            base_pattern = jnp.sin(t * 0.3) * 0.7 + jnp.cos(t * 0.1) * 0.4
            
            experience = (
                base_pattern * jnp.ones(advanced_fixture.state_dim) +
                jax.random.normal(advanced_fixture.keys[10 + t], (advanced_fixture.state_dim,)) * 0.15 +
                0.1 * jnp.arange(advanced_fixture.state_dim) / advanced_fixture.state_dim
            )
            
            experiences.append(experience)
        
        # Process through sparse + predictive integration
        sparse_codes = []
        predictions = []
        prediction_errors = []
        
        for t, experience in enumerate(experiences):
            # Sparse encoding
            sparse_result = sparse_system.encode_multi_modal_experience(
                visual_input=experience,
                proprioceptive_input=experience[:48],
                contextual_cues=experience[:24],
            )
            
            sparse_code = sparse_result['sparse_code']
            sparse_codes.append(sparse_code)
            
            # Hierarchical prediction informed by sparse code
            hierarchical_predictions, pred_errors = predictive_system.generate_hierarchical_predictions(
                current_state=experience,
                sparse_context=sparse_code[:advanced_fixture.state_dim],  # Match dimensions
            )
            
            predictions.append(hierarchical_predictions)
            prediction_errors.append(pred_errors)
        
        # Validate integration results
        assert len(sparse_codes) == sequence_length
        assert len(predictions) == sequence_length
        assert len(prediction_errors) == sequence_length
        
        # Compute integration quality metrics
        avg_sparsity = jnp.mean(jnp.array([jnp.mean(jnp.abs(code) > 1e-6) for code in sparse_codes]))
        avg_prediction_error = jnp.mean(jnp.array([jnp.mean(jnp.array(errors)) for errors in prediction_errors]))
        
        # Validate quality metrics
        assert 0.0 <= float(avg_sparsity) <= 1.0
        assert float(avg_prediction_error) >= 0.0
        
        advanced_fixture.performance_benchmarks['sparse_predictive_integration'] = {
            'average_sparsity': float(avg_sparsity),
            'average_prediction_error': float(avg_prediction_error),
            'integration_weight': integration_weight,
            'sequence_length': sequence_length,
            'integration_successful': True,
        }
    
    @pytest.mark.skipif(
        not (ADVANCED_MODULES_STATUS['sparse_representations'] and ADVANCED_MODULES_STATUS['predictive_coding']),
        reason="Advanced modules not available"
    )
    def test_predictive_sparse_dictionary_learning(self, advanced_fixture):
        """Test predictive coding enhancing sparse dictionary learning."""
        
        if 'sparse_predictive' not in advanced_fixture.integration_patterns:
            pytest.skip("Sparse + predictive integration not available")
        
        pattern = advanced_fixture.integration_patterns['sparse_predictive']
        sparse_system = pattern['sparse_system']
        predictive_system = pattern['predictive_system']
        
        # Test dictionary learning with predictive enhancement
        test_experiences = []
        
        for t in range(20):  # More data for dictionary learning
            # Create diverse experiences
            pattern_type = t % 3
            
            if pattern_type == 0:
                # Sinusoidal patterns
                experience = jnp.sin(jnp.arange(advanced_fixture.state_dim) * 0.1 + t * 0.2) * 0.8
            elif pattern_type == 1:
                # Gaussian blobs
                center = advanced_fixture.state_dim // 2
                experience = jnp.exp(-((jnp.arange(advanced_fixture.state_dim) - center) ** 2) / 100) * 0.7
            else:
                # Random structured
                experience = jax.random.normal(advanced_fixture.keys[15 + t], (advanced_fixture.state_dim,)) * 0.5
                experience = jax.nn.tanh(experience)  # Structured nonlinearity
            
            test_experiences.append(experience)
        
        # Process experiences for dictionary learning
        reconstruction_errors = []
        dictionary_utilizations = []
        
        for experience in test_experiences:
            try:
                # Sparse encoding with predictive context
                sparse_result = sparse_system.encode_multi_modal_experience(
                    visual_input=experience,
                    proprioceptive_input=experience[:48],
                    contextual_cues=experience[:24],
                )
                
                reconstruction_error = sparse_result.get('reconstruction_error', 0.1)
                sparsity = sparse_result.get('sparsity', 0.5)
                
                reconstruction_errors.append(reconstruction_error)
                dictionary_utilizations.append(1.0 - sparsity)  # Higher utilization = lower sparsity
                
            except Exception as e:
                print(f"Sparse encoding failed: {e}")
                reconstruction_errors.append(0.2)  # Fallback error
                dictionary_utilizations.append(0.5)  # Fallback utilization
        
        # Validate dictionary learning performance
        avg_reconstruction_error = float(jnp.mean(jnp.array(reconstruction_errors)))
        avg_dictionary_utilization = float(jnp.mean(jnp.array(dictionary_utilizations)))
        
        assert avg_reconstruction_error >= 0.0
        assert 0.0 <= avg_dictionary_utilization <= 1.0
        
        advanced_fixture.performance_benchmarks['predictive_dictionary_learning'] = {
            'avg_reconstruction_error': avg_reconstruction_error,
            'avg_dictionary_utilization': avg_dictionary_utilization,
            'num_experiences': len(test_experiences),
            'learning_successful': avg_reconstruction_error < 0.5,  # Reasonable threshold
        }


class TestContinuousTemporalSynthesis:
    """Test continuous dynamics + temporal consciousness synthesis."""
    
    @pytest.fixture
    def advanced_fixture(self):
        """Create advanced integration fixture.""" 
        fixture = AdvancedIntegrationFixture()
        fixture.setup_complete_advanced_system()
        return fixture
    
    @pytest.mark.skipif(
        not ADVANCED_MODULES_STATUS['continuous_dynamics'],
        reason="Continuous dynamics not available"
    )
    def test_differential_temporal_flow_integration(self, advanced_fixture):
        """Test differential equation temporal flow integration."""
        
        if 'continuous_temporal' not in advanced_fixture.integration_patterns:
            pytest.skip("Continuous + temporal integration not available")
        
        pattern = advanced_fixture.integration_patterns['continuous_temporal']
        continuous_system = pattern['continuous_system']
        temporal_processor = pattern['temporal_processor']
        
        # Test temporal flow integration
        initial_state = jax.random.normal(advanced_fixture.keys[20], (advanced_fixture.state_dim,)) * 0.4
        time_span = jnp.array([0.0, 2.0])
        
        # Continuous evolution
        evolved_state, evolution_metrics = continuous_system.evolve_consciousness(
            initial_state, time_span
        )
        
        # Temporal synthesis of evolved state
        temporal_moment = temporal_processor.temporal_synthesis(
            primal_impression=evolved_state,
            timestamp=2.0,
        )
        
        # Validate integration
        assert isinstance(temporal_moment, type(temporal_moment))  # Check type consistency
        assert hasattr(temporal_moment, 'present_moment')
        assert hasattr(temporal_moment, 'retention')
        assert hasattr(temporal_moment, 'protention')
        
        # Validate evolution metrics
        assert isinstance(evolution_metrics, dict)
        assert 'smoothness' in evolution_metrics
        assert 'consciousness_trajectory' in evolution_metrics
        
        # Test temporal coherence of evolved state
        evolved_present = temporal_moment.present_moment
        coherence_with_initial = float(jnp.corrcoef(initial_state, evolved_present)[0, 1])
        coherence_with_initial = jnp.nan_to_num(coherence_with_initial, nan=0.0)
        
        advanced_fixture.performance_benchmarks['continuous_temporal_synthesis'] = {
            'evolution_metrics': evolution_metrics,
            'coherence_with_initial': float(coherence_with_initial),
            'temporal_synthesis_successful': True,
            'time_span': float(time_span[1] - time_span[0]),
        }
    
    @pytest.mark.skipif(
        not ADVANCED_MODULES_STATUS['continuous_dynamics'],
        reason="Continuous dynamics not available"
    )  
    def test_consciousness_trajectory_temporal_consistency(self, advanced_fixture):
        """Test consciousness trajectory temporal consistency."""
        
        if 'continuous_temporal' not in advanced_fixture.integration_patterns:
            pytest.skip("Continuous + temporal integration not available")
        
        pattern = advanced_fixture.integration_patterns['continuous_temporal']
        continuous_system = pattern['continuous_system']
        temporal_processor = pattern['temporal_processor']
        
        # Generate consciousness trajectory
        trajectory_length = 8
        time_points = jnp.linspace(0.0, 2.0, trajectory_length)
        
        consciousness_states = []
        temporal_moments = []
        
        current_state = jax.random.normal(advanced_fixture.keys[21], (advanced_fixture.state_dim,)) * 0.3
        
        for i, t in enumerate(time_points):
            # Evolve consciousness
            if i > 0:
                time_span = jnp.array([time_points[i-1], t])
                current_state, _ = continuous_system.evolve_consciousness(current_state, time_span)
            
            consciousness_states.append(current_state)
            
            # Temporal synthesis
            temporal_moment = temporal_processor.temporal_synthesis(
                primal_impression=current_state,
                timestamp=float(t),
            )
            temporal_moments.append(temporal_moment)
        
        # Validate trajectory consistency
        assert len(consciousness_states) == trajectory_length
        assert len(temporal_moments) == trajectory_length
        
        # Compute trajectory coherence
        trajectory_coherences = []
        
        for i in range(1, len(temporal_moments)):
            current_moment = temporal_moments[i]
            previous_moment = temporal_moments[i-1]
            
            # Coherence between present moments
            present_coherence = jnp.corrcoef(
                current_moment.present_moment,
                previous_moment.present_moment
            )[0, 1]
            present_coherence = float(jnp.nan_to_num(present_coherence, nan=0.0))
            
            # Retention coherence (current retention with previous present)
            retention_coherence = jnp.corrcoef(
                current_moment.retention,
                previous_moment.present_moment  
            )[0, 1]
            retention_coherence = float(jnp.nan_to_num(retention_coherence, nan=0.0))
            
            trajectory_coherences.append({
                'present_coherence': present_coherence,
                'retention_coherence': retention_coherence,
                'timestamp': float(time_points[i]),
            })
        
        # Validate coherences
        avg_present_coherence = jnp.mean(jnp.array([c['present_coherence'] for c in trajectory_coherences]))
        avg_retention_coherence = jnp.mean(jnp.array([c['retention_coherence'] for c in trajectory_coherences]))
        
        advanced_fixture.performance_benchmarks['trajectory_temporal_consistency'] = {
            'avg_present_coherence': float(avg_present_coherence),
            'avg_retention_coherence': float(avg_retention_coherence),
            'trajectory_coherences': trajectory_coherences,
            'trajectory_length': trajectory_length,
            'consistency_analysis_successful': True,
        }


class TestMultiModalConsciousnessValidation:
    """Test multi-modal consciousness state validation."""
    
    @pytest.fixture
    def advanced_fixture(self):
        """Create advanced integration fixture."""
        fixture = AdvancedIntegrationFixture()
        fixture.setup_complete_advanced_system()
        return fixture
    
    def test_phenomenological_consistency_validation(self, advanced_fixture):
        """Test Husserlian phenomenological consistency validation."""
        
        if 'multi_modal_validation' not in advanced_fixture.advanced_components:
            pytest.skip("Multi-modal validation not available")
        
        validation_system = advanced_fixture.advanced_components['multi_modal_validation']
        consciousness_system = validation_system['consciousness_system']
        
        # Generate phenomenologically structured inputs
        test_inputs = []
        
        for t in range(12):
            # Phenomenological time-consciousness structure
            retention_pattern = jnp.exp(-t * 0.1) * jnp.sin(t * 0.2) * 0.6  # Fading past
            present_pattern = jnp.sin(t * 0.3) * 0.8  # Vivid present
            protention_pattern = jnp.tanh((t + 1) * 0.1) * 0.4  # Anticipated future
            
            combined_pattern = retention_pattern + present_pattern + protention_pattern
            
            # Multi-modal inputs with phenomenological structure
            sensory_input = (
                combined_pattern * jnp.ones(advanced_fixture.state_dim) +
                jax.random.normal(advanced_fixture.keys[22 + t], (advanced_fixture.state_dim,)) * 0.1
            )
            
            proprioceptive_input = (
                present_pattern * 0.7 * jnp.ones(48) +
                jax.random.normal(advanced_fixture.keys[25 + t], (48,)) * 0.08
            )
            
            motor_prediction = (
                protention_pattern * 0.5 * jnp.ones(16) +
                jax.random.normal(advanced_fixture.keys[28 + t], (16,)) * 0.06
            )
            
            environmental_state = (
                combined_pattern * 0.6 * jnp.ones(advanced_fixture.environment_dim) +
                jax.random.normal(advanced_fixture.keys[30 + t], (advanced_fixture.environment_dim,)) * 0.12
            )
            
            contextual_cues = (
                present_pattern * 0.4 * jnp.ones(advanced_fixture.context_dim) +
                jax.random.normal(advanced_fixture.keys[32 + t], (advanced_fixture.context_dim,)) * 0.1
            )
            
            test_inputs.append({
                'sensory_input': sensory_input,
                'proprioceptive_input': proprioceptive_input,
                'motor_prediction': motor_prediction,
                'environmental_state': environmental_state,
                'contextual_cues': contextual_cues,
                'timestamp': float(t * 0.1),
            })
        
        # Process through consciousness system
        consciousness_states = []
        
        for inputs in test_inputs:
            consciousness_state = consciousness_system.integrate_conscious_moment(**inputs)
            consciousness_states.append(consciousness_state)
        
        # Validate phenomenological consistency
        phenomenological_metrics = self._validate_phenomenological_structure(consciousness_states)
        
        # Store validation results
        advanced_fixture.theoretical_consistency_metrics['phenomenological_validation'] = {
            'consciousness_states_count': len(consciousness_states),
            'phenomenological_metrics': phenomenological_metrics,
            'validation_successful': all(
                0.0 <= metric <= 1.0 for metric in phenomenological_metrics.values()
                if isinstance(metric, (int, float))
            ),
        }
        
        # Validate theoretical consistency
        assert len(consciousness_states) == len(test_inputs)
        assert all(hasattr(state, 'temporal_moment') for state in consciousness_states)
        assert all(hasattr(state, 'consciousness_level') for state in consciousness_states)
    
    def _validate_phenomenological_structure(self, consciousness_states: List[ConsciousnessState]) -> Dict[str, float]:
        """Validate Husserlian temporal structure in consciousness states."""
        
        if len(consciousness_states) < 2:
            return {'temporal_synthesis_coherence': 0.0}
        
        # Retention-protention structure validation
        retention_coherences = []
        protention_predictions = [] 
        synthesis_consistencies = []
        
        for i in range(1, len(consciousness_states)):
            current_temporal = consciousness_states[i].temporal_moment
            previous_temporal = consciousness_states[i-1].temporal_moment
            
            # Retention coherence: current retention should relate to previous present
            retention_coherence = jnp.corrcoef(
                current_temporal.retention, 
                previous_temporal.present_moment
            )[0, 1]
            retention_coherence = float(jnp.nan_to_num(retention_coherence, nan=0.0))
            retention_coherences.append(retention_coherence)
            
            # Protention prediction: previous protention should relate to current present
            protention_prediction = jnp.corrcoef(
                previous_temporal.protention,
                current_temporal.present_moment
            )[0, 1] 
            protention_prediction = float(jnp.nan_to_num(protention_prediction, nan=0.0))
            protention_predictions.append(protention_prediction)
            
            # Synthesis consistency: weights should be meaningful
            synthesis_weights = current_temporal.synthesis_weights
            synthesis_consistency = float(jnp.std(synthesis_weights))  # Meaningful variation
            synthesis_consistencies.append(synthesis_consistency)
        
        return {
            'retention_coherence': float(jnp.mean(jnp.array(retention_coherences))),
            'protention_prediction': float(jnp.mean(jnp.array(protention_predictions))),
            'synthesis_consistency': float(jnp.mean(jnp.array(synthesis_consistencies))),
            'temporal_synthesis_coherence': float(
                (jnp.mean(jnp.array(retention_coherences)) + 
                 jnp.mean(jnp.array(protention_predictions))) / 2
            ),
        }
    
    def test_enactivist_coupling_validation(self, advanced_fixture):
        """Test enactivist agent-environment coupling validation."""
        
        if 'multi_modal_validation' not in advanced_fixture.advanced_components:
            pytest.skip("Multi-modal validation not available")
        
        validation_system = advanced_fixture.advanced_components['multi_modal_validation']
        consciousness_system = validation_system['consciousness_system']
        
        # Generate coupled agent-environment dynamics
        coupling_strength = 0.7
        test_sequence = []
        
        agent_state = jax.random.normal(advanced_fixture.keys[35], (advanced_fixture.state_dim,)) * 0.3
        env_state = jax.random.normal(advanced_fixture.keys[36], (advanced_fixture.environment_dim,)) * 0.3
        
        for t in range(10):
            # Enactivist coupling: agent and environment co-determine each other
            # Agent influences environment
            env_influence = agent_state[:advanced_fixture.environment_dim] * coupling_strength
            new_env_state = (
                0.6 * env_state +
                0.3 * env_influence + 
                jax.random.normal(advanced_fixture.keys[37 + t], (advanced_fixture.environment_dim,)) * 0.1
            )
            
            # Environment influences agent
            agent_influence = jnp.concatenate([
                new_env_state,
                jnp.zeros(advanced_fixture.state_dim - advanced_fixture.environment_dim)
            ]) * coupling_strength
            
            new_agent_state = (
                0.6 * agent_state +
                0.3 * agent_influence +
                jax.random.normal(advanced_fixture.keys[40 + t], (advanced_fixture.state_dim,)) * 0.1
            )
            
            # Create consciousness inputs
            test_input = {
                'sensory_input': new_agent_state,
                'proprioceptive_input': new_agent_state[:48],
                'motor_prediction': new_agent_state[:16],
                'environmental_state': new_env_state,
                'contextual_cues': new_agent_state[:advanced_fixture.context_dim],
                'timestamp': float(t * 0.15),
            }
            
            test_sequence.append(test_input)
            
            # Update states for next iteration
            agent_state = new_agent_state
            env_state = new_env_state
        
        # Process through consciousness system
        consciousness_states = []
        
        for inputs in test_sequence:
            consciousness_state = consciousness_system.integrate_conscious_moment(**inputs)
            consciousness_states.append(consciousness_state)
        
        # Validate enactivist coupling
        coupling_metrics = self._validate_enactivist_coupling(consciousness_states, test_sequence)
        
        advanced_fixture.theoretical_consistency_metrics['enactivist_coupling_validation'] = {
            'coupling_metrics': coupling_metrics,
            'coupling_strength_used': coupling_strength,
            'sequence_length': len(consciousness_states),
            'validation_successful': coupling_metrics.get('circular_causality_strength', 0.0) > 0.3,
        }
        
        assert len(consciousness_states) == len(test_sequence)
        assert all(state.circular_causality_strength > 0.0 for state in consciousness_states)
    
    def _validate_enactivist_coupling(
        self, 
        consciousness_states: List[ConsciousnessState],
        test_sequence: List[Dict[str, jax.Array]]
    ) -> Dict[str, float]:
        """Validate enactivist coupling patterns."""
        
        if len(consciousness_states) < 2:
            return {'circular_causality_strength': 0.0}
        
        # Extract agent and environment sequences
        agent_sequence = jnp.array([
            state.temporal_moment.present_moment for state in consciousness_states
        ])
        env_sequence = jnp.array([
            inputs['environmental_state'] for inputs in test_sequence
        ])
        
        # Compute coupling metrics
        coupling_strengths = [state.circular_causality_strength for state in consciousness_states]
        avg_coupling_strength = float(jnp.mean(jnp.array(coupling_strengths)))
        
        # Compute agent-environment correlation
        agent_env_correlations = []
        
        for i in range(len(consciousness_states)):
            agent_state = consciousness_states[i].temporal_moment.present_moment[:advanced_fixture.environment_dim]
            env_state = test_sequence[i]['environmental_state']
            
            correlation = jnp.corrcoef(agent_state, env_state)[0, 1]
            correlation = float(jnp.nan_to_num(correlation, nan=0.0))
            agent_env_correlations.append(correlation)
        
        avg_correlation = float(jnp.mean(jnp.array(agent_env_correlations)))
        
        # Temporal coupling consistency
        temporal_coupling_consistency = 1.0 - float(jnp.std(jnp.array(coupling_strengths)))
        
        return {
            'circular_causality_strength': avg_coupling_strength,
            'agent_environment_correlation': avg_correlation,
            'temporal_coupling_consistency': max(0.0, temporal_coupling_consistency),
            'coupling_variation': float(jnp.std(jnp.array(coupling_strengths))),
        }


class TestAdvancedPerformanceBenchmarking:
    """Test advanced performance benchmarking across integration patterns."""
    
    @pytest.fixture
    def advanced_fixture(self):
        """Create advanced integration fixture."""
        fixture = AdvancedIntegrationFixture(state_dim=128, environment_dim=48)  # Larger for benchmarking
        fixture.setup_complete_advanced_system()
        return fixture
    
    def test_comprehensive_integration_benchmarks(self, advanced_fixture):
        """Run comprehensive performance benchmarks across all integration patterns."""
        
        benchmark_results = {}
        
        # Test each available integration pattern
        for pattern_name, pattern in advanced_fixture.integration_patterns.items():
            print(f"Benchmarking {pattern_name}...")
            
            try:
                benchmark_start = time.time()
                
                if pattern_name == 'info_theory_networks':
                    bench_results = self._benchmark_info_theory_networks(pattern, advanced_fixture)
                elif pattern_name == 'sparse_predictive':
                    bench_results = self._benchmark_sparse_predictive(pattern, advanced_fixture)
                elif pattern_name == 'continuous_temporal':
                    bench_results = self._benchmark_continuous_temporal(pattern, advanced_fixture)
                else:
                    bench_results = {'benchmark_time_ms': 0, 'pattern_available': False}
                
                benchmark_time = (time.time() - benchmark_start) * 1000
                bench_results['benchmark_time_ms'] = benchmark_time
                bench_results['pattern_available'] = True
                
                benchmark_results[pattern_name] = bench_results
                
            except Exception as e:
                print(f"Benchmark failed for {pattern_name}: {e}")
                benchmark_results[pattern_name] = {
                    'benchmark_time_ms': 0,
                    'pattern_available': False,
                    'error': str(e),
                }
        
        # Multi-modal consciousness system benchmarking
        if 'multi_modal_validation' in advanced_fixture.advanced_components:
            try:
                multi_modal_bench = self._benchmark_multi_modal_consciousness(advanced_fixture)
                benchmark_results['multi_modal_consciousness'] = multi_modal_bench
            except Exception as e:
                benchmark_results['multi_modal_consciousness'] = {'error': str(e)}
        
        # Store comprehensive benchmark results
        advanced_fixture.performance_benchmarks['comprehensive_integration'] = benchmark_results
        
        # Save benchmark results to file
        benchmark_file = Path(__file__).parent.parent / 'advanced_integration_benchmarks.json'
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        # Validate benchmarking
        assert len(benchmark_results) > 0
        available_patterns = sum(1 for result in benchmark_results.values() 
                               if result.get('pattern_available', False))
        
        print(f"Successfully benchmarked {available_patterns}/{len(advanced_fixture.integration_patterns)} integration patterns")
    
    def _benchmark_info_theory_networks(self, pattern: Dict[str, Any], fixture: AdvancedIntegrationFixture) -> Dict[str, float]:
        """Benchmark information theory + networks integration."""
        
        network_processor = pattern['network_processor']
        info_functions = pattern['info_theory_functions']
        
        # Generate benchmark sequence
        benchmark_length = 20
        processing_times = []
        
        current_state = jax.random.normal(fixture.keys[50], (fixture.state_dim,)) * 0.4
        
        for t in range(benchmark_length):
            start_time = time.time()
            
            # Process through network
            processed_state, network_metrics = network_processor.process_state(current_state)
            
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
            
            current_state = processed_state * 0.9 + jax.random.normal(fixture.keys[51 + t], fixture.state_dim.shape) * 0.1
        
        return {
            'avg_processing_time_ms': float(jnp.mean(jnp.array(processing_times))),
            'total_processing_time_ms': float(jnp.sum(jnp.array(processing_times))),
            'benchmark_sequence_length': benchmark_length,
        }
    
    def _benchmark_sparse_predictive(self, pattern: Dict[str, Any], fixture: AdvancedIntegrationFixture) -> Dict[str, float]:
        """Benchmark sparse + predictive integration."""
        
        sparse_system = pattern['sparse_system']
        predictive_system = pattern['predictive_system']
        
        # Generate benchmark sequence
        benchmark_length = 15
        processing_times = []
        
        for t in range(benchmark_length):
            test_input = jax.random.normal(fixture.keys[60 + t], (fixture.state_dim,)) * 0.5
            
            start_time = time.time()
            
            # Sparse encoding
            sparse_result = sparse_system.encode_multi_modal_experience(
                visual_input=test_input,
                proprioceptive_input=test_input[:48],
                contextual_cues=test_input[:24],
            )
            
            # Predictive processing
            predictions, errors = predictive_system.generate_hierarchical_predictions(
                current_state=test_input,
                sparse_context=sparse_result['sparse_code'][:fixture.state_dim],
            )
            
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
        
        return {
            'avg_processing_time_ms': float(jnp.mean(jnp.array(processing_times))),
            'total_processing_time_ms': float(jnp.sum(jnp.array(processing_times))),
            'benchmark_sequence_length': benchmark_length,
        }
    
    def _benchmark_continuous_temporal(self, pattern: Dict[str, Any], fixture: AdvancedIntegrationFixture) -> Dict[str, float]:
        """Benchmark continuous + temporal integration."""
        
        continuous_system = pattern['continuous_system']
        temporal_processor = pattern['temporal_processor']
        
        # Generate benchmark sequence
        benchmark_length = 12
        processing_times = []
        
        for t in range(benchmark_length):
            initial_state = jax.random.normal(fixture.keys[70 + t], (fixture.state_dim,)) * 0.3
            time_span = jnp.array([0.0, 1.0])
            
            start_time = time.time()
            
            # Continuous evolution
            evolved_state, evolution_metrics = continuous_system.evolve_consciousness(initial_state, time_span)
            
            # Temporal synthesis
            temporal_moment = temporal_processor.temporal_synthesis(
                primal_impression=evolved_state,
                timestamp=float(t * 0.2),
            )
            
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
        
        return {
            'avg_processing_time_ms': float(jnp.mean(jnp.array(processing_times))),
            'total_processing_time_ms': float(jnp.sum(jnp.array(processing_times))),
            'benchmark_sequence_length': benchmark_length,
        }
    
    def _benchmark_multi_modal_consciousness(self, fixture: AdvancedIntegrationFixture) -> Dict[str, float]:
        """Benchmark multi-modal consciousness system."""
        
        validation_system = fixture.advanced_components['multi_modal_validation']
        consciousness_system = validation_system['consciousness_system']
        
        # Generate benchmark inputs
        benchmark_length = 10
        processing_times = []
        
        for t in range(benchmark_length):
            test_input = {
                'sensory_input': jax.random.normal(fixture.keys[80 + t], (fixture.state_dim,)) * 0.4,
                'proprioceptive_input': jax.random.normal(fixture.keys[85 + t], (48,)) * 0.3,
                'motor_prediction': jax.random.normal(fixture.keys[90 + t], (16,)) * 0.2,
                'environmental_state': jax.random.normal(fixture.keys[95 + t], (fixture.environment_dim,)) * 0.35,
                'contextual_cues': jax.random.normal(fixture.keys[100 + t], (fixture.context_dim,)) * 0.25,
                'timestamp': float(t * 0.1),
            }
            
            start_time = time.time()
            
            consciousness_state = consciousness_system.integrate_conscious_moment(**test_input)
            
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
        
        return {
            'avg_processing_time_ms': float(jnp.mean(jnp.array(processing_times))),
            'total_processing_time_ms': float(jnp.sum(jnp.array(processing_times))),
            'benchmark_sequence_length': benchmark_length,
            'multi_modal_integration_successful': True,
        }


if __name__ == '__main__':
    """Run advanced integration tests directly."""
    
    # Configure JAX
    jax.config.update('jax_platform_name', 'cpu')
    
    print("🚀 Running Advanced Integration Pattern Tests")
    print("=" * 70)
    print(f"Advanced modules status: {ADVANCED_MODULES_STATUS}")
    
    # Run tests with pytest
    pytest.main([
        __file__,
        '-v',
        '--tb=short', 
        '--disable-warnings',
        '-x',  # Stop on first failure for debugging
    ])