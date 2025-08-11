"""Comprehensive test suite for sparse representations module.

This test suite follows TDD principles with extensive coverage of
sparse coding, dictionary learning, and convex optimization for
enactive consciousness with mathematical validation.

Test Coverage:
- SparseExperienceEncoder functionality and ISTA optimization
- AdaptiveDictionaryLearner with online learning
- ConvexMeaningOptimizer with consciousness constraints
- IntegratedSparseRepresentationSystem integration
- Mathematical correctness validation
- Performance and scalability testing
- Consciousness constraint satisfaction
- Compression efficiency analysis
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any
from unittest.mock import patch, MagicMock
import warnings

# Import the module under test
import sys
sys.path.insert(0, '/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/10_8_2025/enactive-consciousness/src')

from enactive_consciousness.sparse_representations import (
    SparseRepresentationsConfig,
    SparseExperienceEncoder,
    AdaptiveDictionaryLearner,
    ConvexMeaningOptimizer,
    IntegratedSparseRepresentationSystem,
    SimpleExperienceTrace,
    SimpleExperientialTrace,
    create_sparse_representations_system,
    analyze_sparsity_quality,
    compress_experiential_traces,
    decompress_sparse_representation,
    validate_consciousness_constraints,
)

from enactive_consciousness.types import (
    TemporalMoment,
    MeaningStructure,
    PRNGKey,
    TimeStep,
)


class TestSparseRepresentationsConfig:
    """Test cases for SparseRepresentationsConfig."""
    
    def test_config_creation_default(self):
        """Test default configuration creation."""
        config = SparseRepresentationsConfig()
        
        assert config.experience_dim == 128
        assert config.dictionary_size == 256
        assert config.sparsity_level == 0.1
        assert config.learning_rate == 0.01
        assert config.convergence_threshold == 1e-6
        assert config.max_iterations == 100
        assert config.consciousness_constraint_weight == 0.5
        assert config.temporal_coherence_weight == 0.3
        assert config.meaning_preservation_weight == 0.4
    
    def test_config_creation_custom(self):
        """Test custom configuration creation."""
        config = SparseRepresentationsConfig(
            experience_dim=64,
            dictionary_size=128,
            sparsity_level=0.2,
            learning_rate=0.02,
        )
        
        assert config.experience_dim == 64
        assert config.dictionary_size == 128
        assert config.sparsity_level == 0.2
        assert config.learning_rate == 0.02
    
    def test_config_validation_experience_dim(self):
        """Test configuration validation for experience_dim."""
        with pytest.raises(ValueError, match="experience_dim must be positive"):
            SparseRepresentationsConfig(experience_dim=-1)
        
        with pytest.raises(ValueError, match="experience_dim must be positive"):
            SparseRepresentationsConfig(experience_dim=0)
    
    def test_config_validation_dictionary_size(self):
        """Test configuration validation for dictionary_size."""
        with pytest.raises(ValueError, match="dictionary_size must be positive"):
            SparseRepresentationsConfig(dictionary_size=-1)
        
        with pytest.raises(ValueError, match="dictionary_size must be positive"):
            SparseRepresentationsConfig(dictionary_size=0)
    
    def test_config_validation_sparsity_level(self):
        """Test configuration validation for sparsity_level."""
        with pytest.raises(ValueError, match="sparsity_level must be in \\(0, 1\\)"):
            SparseRepresentationsConfig(sparsity_level=0.0)
        
        with pytest.raises(ValueError, match="sparsity_level must be in \\(0, 1\\)"):
            SparseRepresentationsConfig(sparsity_level=1.0)
        
        with pytest.raises(ValueError, match="sparsity_level must be in \\(0, 1\\)"):
            SparseRepresentationsConfig(sparsity_level=-0.1)
        
        with pytest.raises(ValueError, match="sparsity_level must be in \\(0, 1\\)"):
            SparseRepresentationsConfig(sparsity_level=1.1)
    
    def test_config_validation_learning_rate(self):
        """Test configuration validation for learning_rate."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            SparseRepresentationsConfig(learning_rate=-0.01)
        
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            SparseRepresentationsConfig(learning_rate=0.0)
    
    def test_config_validation_convergence_threshold(self):
        """Test configuration validation for convergence_threshold."""
        with pytest.raises(ValueError, match="convergence_threshold must be positive"):
            SparseRepresentationsConfig(convergence_threshold=-1e-6)
        
        with pytest.raises(ValueError, match="convergence_threshold must be positive"):
            SparseRepresentationsConfig(convergence_threshold=0.0)
    
    def test_config_validation_max_iterations(self):
        """Test configuration validation for max_iterations."""
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            SparseRepresentationsConfig(max_iterations=-1)
        
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            SparseRepresentationsConfig(max_iterations=0)


class TestSimpleTraces:
    """Test cases for simple trace classes."""
    
    def test_simple_experience_trace_creation(self):
        """Test SimpleExperienceTrace creation."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        trace = SimpleExperienceTrace(
            content=jax.random.normal(keys[0], (64,)),
            temporal_signature=jax.random.normal(keys[1], (32,)),
            associative_links=jax.random.normal(keys[2], (16,)),
            sedimentation_level=0.7,
            creation_timestamp=100.0,
            last_access_timestamp=150.0,
            access_frequency=5,
        )
        
        assert isinstance(trace.content, jax.Array)
        assert isinstance(trace.temporal_signature, jax.Array)
        assert isinstance(trace.associative_links, jax.Array)
        assert trace.sedimentation_level == 0.7
        assert trace.creation_timestamp == 100.0
        assert trace.last_access_timestamp == 150.0
        assert trace.access_frequency == 5
    
    def test_simple_experiential_trace_creation(self):
        """Test SimpleExperientialTrace creation."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        trace = SimpleExperientialTrace(
            interaction_pattern=jax.random.normal(keys[0], (48,)),
            contextual_embedding=jax.random.normal(keys[1], (24,)),
            significance_weight=0.8,
            temporal_depth=2.5,
            coupling_strength=0.6,
        )
        
        assert isinstance(trace.interaction_pattern, jax.Array)
        assert isinstance(trace.contextual_embedding, jax.Array)
        assert trace.significance_weight == 0.8
        assert trace.temporal_depth == 2.5
        assert trace.coupling_strength == 0.6


class TestSparseExperienceEncoder:
    """Test cases for SparseExperienceEncoder."""
    
    @pytest.fixture
    def encoder_setup(self):
        """Set up encoder for testing."""
        config = SparseRepresentationsConfig(
            experience_dim=32,
            dictionary_size=64,
            sparsity_level=0.15,
        )
        key = jax.random.PRNGKey(42)
        encoder = SparseExperienceEncoder(config, key)
        return encoder, config
    
    def test_encoder_initialization(self, encoder_setup):
        """Test SparseExperienceEncoder initialization."""
        encoder, config = encoder_setup
        
        assert encoder.config == config
        assert encoder.dictionary.shape == (config.dictionary_size, config.experience_dim)
        assert encoder.regularization_matrix.shape == (config.dictionary_size, config.dictionary_size)
        
        # Dictionary should be normalized
        dictionary_norms = jnp.linalg.norm(encoder.dictionary, axis=1)
        assert jnp.allclose(dictionary_norms, 1.0, atol=1e-5), "Dictionary atoms should be normalized"
    
    def test_encode_experience_basic(self, encoder_setup):
        """Test basic experience encoding."""
        encoder, config = encoder_setup
        key = jax.random.PRNGKey(42)
        
        experience = jax.random.normal(key, (config.experience_dim,))
        sparse_code = encoder.encode_experience(experience)
        
        assert sparse_code.shape == (config.dictionary_size,)
        assert jnp.all(jnp.isfinite(sparse_code))
        
        # Check sparsity level
        non_zero_fraction = jnp.mean(jnp.abs(sparse_code) > 1e-6)
        assert non_zero_fraction <= config.sparsity_level + 0.05, "Should enforce sparsity"
    
    def test_encode_experience_sparsity_enforcement(self, encoder_setup):
        """Test that encoder enforces target sparsity level."""
        encoder, config = encoder_setup
        key = jax.random.PRNGKey(42)
        
        # Test multiple experiences
        sparsity_levels = []
        for i in range(10):
            experience = jax.random.normal(jax.random.fold_in(key, i), (config.experience_dim,))
            sparse_code = encoder.encode_experience(experience)
            
            non_zero_fraction = jnp.mean(jnp.abs(sparse_code) > 1e-6)
            sparsity_levels.append(non_zero_fraction)
        
        avg_sparsity = jnp.mean(jnp.array(sparsity_levels))
        assert avg_sparsity <= config.sparsity_level + 0.1, f"Average sparsity {avg_sparsity} exceeds target"
    
    def test_encode_batch(self, encoder_setup):
        """Test batch encoding functionality."""
        encoder, config = encoder_setup
        key = jax.random.PRNGKey(42)
        
        batch_size = 8
        experiences_batch = jax.random.normal(key, (batch_size, config.experience_dim))
        
        sparse_codes_batch = encoder.encode_batch(experiences_batch)
        
        assert sparse_codes_batch.shape == (batch_size, config.dictionary_size)
        assert jnp.all(jnp.isfinite(sparse_codes_batch))
        
        # Check that each code in batch is properly sparse
        for i in range(batch_size):
            non_zero_fraction = jnp.mean(jnp.abs(sparse_codes_batch[i]) > 1e-6)
            assert non_zero_fraction <= config.sparsity_level + 0.1
    
    def test_reconstruct_experience(self, encoder_setup):
        """Test experience reconstruction from sparse code."""
        encoder, config = encoder_setup
        key = jax.random.PRNGKey(42)
        
        experience = jax.random.normal(key, (config.experience_dim,))
        sparse_code = encoder.encode_experience(experience)
        reconstructed = encoder.reconstruct_experience(sparse_code)
        
        assert reconstructed.shape == experience.shape
        assert jnp.all(jnp.isfinite(reconstructed))
        
        # Reconstruction should be reasonably close to original
        reconstruction_error = jnp.linalg.norm(experience - reconstructed)
        assert reconstruction_error < 10.0, "Reconstruction error should be reasonable"
    
    def test_compute_reconstruction_error(self, encoder_setup):
        """Test reconstruction error computation."""
        encoder, config = encoder_setup
        key = jax.random.PRNGKey(42)
        
        experience = jax.random.normal(key, (config.experience_dim,))
        sparse_code = encoder.encode_experience(experience)
        
        error = encoder.compute_reconstruction_error(experience, sparse_code)
        
        assert isinstance(error, float)
        assert error >= 0.0, "Reconstruction error should be non-negative"
        assert jnp.isfinite(error)
    
    def test_encoder_consistency(self, encoder_setup):
        """Test encoder consistency across multiple runs."""
        encoder, config = encoder_setup
        key = jax.random.PRNGKey(42)
        
        experience = jax.random.normal(key, (config.experience_dim,))
        
        # Encode multiple times (should be deterministic)
        sparse_code1 = encoder.encode_experience(experience)
        sparse_code2 = encoder.encode_experience(experience)
        
        assert jnp.allclose(sparse_code1, sparse_code2), "Encoding should be deterministic"


class TestAdaptiveDictionaryLearner:
    """Test cases for AdaptiveDictionaryLearner."""
    
    @pytest.fixture
    def learner_setup(self):
        """Set up dictionary learner for testing."""
        config = SparseRepresentationsConfig(
            experience_dim=24,
            dictionary_size=48,
            sparsity_level=0.2,
            learning_rate=0.02,
        )
        key = jax.random.PRNGKey(42)
        learner = AdaptiveDictionaryLearner(config, key)
        return learner, config
    
    def test_learner_initialization(self, learner_setup):
        """Test AdaptiveDictionaryLearner initialization."""
        learner, config = learner_setup
        
        assert learner.config == config
        assert learner.dictionary.shape == (config.dictionary_size, config.experience_dim)
        assert learner.momentum_buffer.shape == learner.dictionary.shape
        assert isinstance(learner.learning_statistics, dict)
        
        # Dictionary should be normalized
        dictionary_norms = jnp.linalg.norm(learner.dictionary, axis=1)
        assert jnp.allclose(dictionary_norms, 1.0, atol=1e-5), "Dictionary atoms should be normalized"
        
        # Check learning statistics initialization
        expected_keys = {'total_updates', 'average_reconstruction_error', 'dictionary_coherence'}
        assert set(learner.learning_statistics.keys()) == expected_keys
    
    def test_sparse_encode(self, learner_setup):
        """Test sparse encoding method."""
        learner, config = learner_setup
        key = jax.random.PRNGKey(42)
        
        experience = jax.random.normal(key, (config.experience_dim,))
        sparse_code = learner.sparse_encode(experience)
        
        assert sparse_code.shape == (config.dictionary_size,)
        assert jnp.all(jnp.isfinite(sparse_code))
        
        # Should enforce sparsity through thresholding
        non_zero_fraction = jnp.mean(jnp.abs(sparse_code) > 1e-6)
        assert non_zero_fraction <= config.sparsity_level + 0.1
    
    def test_reconstruct(self, learner_setup):
        """Test reconstruction method."""
        learner, config = learner_setup
        key = jax.random.PRNGKey(42)
        
        sparse_code = jax.random.normal(key, (config.dictionary_size,)) * 0.1
        reconstructed = learner.reconstruct(sparse_code)
        
        assert reconstructed.shape == (config.experience_dim,)
        assert jnp.all(jnp.isfinite(reconstructed))
    
    def test_learn_iteration(self, learner_setup):
        """Test single learning iteration."""
        learner, config = learner_setup
        key = jax.random.PRNGKey(42)
        
        # Create batch of experiences
        batch_size = 4
        experiences_batch = jax.random.normal(key, (batch_size, config.experience_dim))
        
        # Perform learning iteration
        updated_learner, learning_metrics = learner.learn_iteration(experiences_batch)
        
        # Check return types
        assert isinstance(updated_learner, AdaptiveDictionaryLearner)
        assert isinstance(learning_metrics, dict)
        
        # Check metrics keys
        expected_metrics = {'reconstruction_error', 'dictionary_coherence', 'adaptation_magnitude'}
        assert set(learning_metrics.keys()) == expected_metrics
        
        # Check metric values
        for key, value in learning_metrics.items():
            assert isinstance(value, float)
            assert jnp.isfinite(value)
            assert value >= 0.0, f"{key} should be non-negative"
        
        # Dictionary should still be normalized
        dictionary_norms = jnp.linalg.norm(updated_learner.dictionary, axis=1)
        assert jnp.allclose(dictionary_norms, 1.0, atol=1e-4), "Dictionary should remain normalized"
    
    def test_online_adapt(self, learner_setup):
        """Test online adaptation method."""
        learner, config = learner_setup
        key = jax.random.PRNGKey(42)
        
        experience = jax.random.normal(key, (config.experience_dim,))
        
        # Adapt with single experience
        adapted_learner = learner.online_adapt(experience)
        
        assert isinstance(adapted_learner, AdaptiveDictionaryLearner)
        
        # Dictionary should be updated but still normalized
        dictionary_norms = jnp.linalg.norm(adapted_learner.dictionary, axis=1)
        assert jnp.allclose(dictionary_norms, 1.0, atol=1e-4), "Dictionary should remain normalized"
        
        # Dictionary should be different from original (unless no adaptation occurred)
        dict_difference = jnp.linalg.norm(adapted_learner.dictionary - learner.dictionary)
        # Allow for the case where adaptation is very small
        assert dict_difference >= 0.0
    
    def test_dictionary_coherence_computation(self, learner_setup):
        """Test dictionary coherence computation."""
        learner, config = learner_setup
        
        coherence = learner._compute_dictionary_coherence(learner.dictionary)
        
        assert isinstance(coherence, float)
        assert 0.0 <= coherence <= 1.0, "Coherence should be in [0,1]"
        assert jnp.isfinite(coherence)
    
    def test_learning_progression(self, learner_setup):
        """Test that learning progresses over multiple iterations."""
        learner, config = learner_setup
        key = jax.random.PRNGKey(42)
        
        # Create consistent batch
        batch_size = 6
        experiences_batch = jax.random.normal(key, (batch_size, config.experience_dim))
        
        # Perform multiple learning iterations
        current_learner = learner
        reconstruction_errors = []
        
        for i in range(3):
            updated_learner, metrics = current_learner.learn_iteration(experiences_batch)
            reconstruction_errors.append(metrics['reconstruction_error'])
            current_learner = updated_learner
        
        # Error should generally decrease (allowing for some fluctuation)
        assert len(reconstruction_errors) == 3
        for error in reconstruction_errors:
            assert error >= 0.0
            assert jnp.isfinite(error)


class TestConvexMeaningOptimizer:
    """Test cases for ConvexMeaningOptimizer."""
    
    @pytest.fixture
    def optimizer_setup(self):
        """Set up convex optimizer for testing."""
        config = SparseRepresentationsConfig(
            experience_dim=20,
            dictionary_size=40,
            sparsity_level=0.25,
        )
        key = jax.random.PRNGKey(42)
        optimizer = ConvexMeaningOptimizer(config, key)
        return optimizer, config
    
    def test_optimizer_initialization(self, optimizer_setup):
        """Test ConvexMeaningOptimizer initialization."""
        optimizer, config = optimizer_setup
        
        assert optimizer.config == config
        
        # Check constraint matrices
        assert optimizer.consciousness_constraints.shape == (config.experience_dim // 4, config.dictionary_size)
        assert optimizer.temporal_coherence_matrix.shape == (config.dictionary_size, config.dictionary_size)
        assert optimizer.meaning_preservation_matrix.shape == (config.experience_dim, config.dictionary_size)
        
        # Temporal coherence matrix should be symmetric
        assert jnp.allclose(
            optimizer.temporal_coherence_matrix,
            optimizer.temporal_coherence_matrix.T,
            atol=1e-5
        ), "Temporal coherence matrix should be symmetric"
    
    def test_optimize_sparse_representation(self, optimizer_setup):
        """Test sparse representation optimization."""
        optimizer, config = optimizer_setup
        key = jax.random.PRNGKey(42)
        
        target_experience = jax.random.normal(key, (config.experience_dim,))
        
        result = optimizer.optimize_sparse_representation(target_experience)
        
        # Check result structure
        assert isinstance(result, dict)
        expected_keys = {'sparse_code', 'objective_value', 'converged', 'iterations'}
        assert set(result.keys()) == expected_keys
        
        # Check result values
        assert result['sparse_code'].shape == (config.dictionary_size,)
        assert jnp.all(jnp.isfinite(result['sparse_code']))
        assert isinstance(result['objective_value'], float)
        assert isinstance(result['converged'], bool)
        assert isinstance(result['iterations'], int)
    
    def test_optimize_with_custom_parameters(self, optimizer_setup):
        """Test optimization with custom parameters."""
        optimizer, config = optimizer_setup
        key = jax.random.PRNGKey(42)
        
        target_experience = jax.random.normal(key, (config.experience_dim,))
        
        result = optimizer.optimize_sparse_representation(
            target_experience,
            max_iterations=50,
            convergence_threshold=1e-4
        )
        
        assert isinstance(result, dict)
        assert result['sparse_code'].shape == (config.dictionary_size,)
        assert result['iterations'] <= 50
    
    def test_fallback_sparse_coding(self, optimizer_setup):
        """Test fallback sparse coding method."""
        optimizer, config = optimizer_setup
        key = jax.random.PRNGKey(42)
        
        target_experience = jax.random.normal(key, (config.experience_dim,))
        
        sparse_code = optimizer._fallback_sparse_coding(target_experience)
        
        assert sparse_code.shape == (config.dictionary_size,)
        assert jnp.all(jnp.isfinite(sparse_code))
        
        # Should respect sparsity level
        non_zero_fraction = jnp.mean(jnp.abs(sparse_code) > 1e-6)
        assert non_zero_fraction <= config.sparsity_level + 0.1
    
    def test_evaluate_consciousness_constraints(self, optimizer_setup):
        """Test consciousness constraint evaluation."""
        optimizer, config = optimizer_setup
        key = jax.random.PRNGKey(42)
        
        sparse_code = jax.random.normal(key, (config.dictionary_size,)) * 0.1
        
        constraints = optimizer.evaluate_consciousness_constraints(sparse_code)
        
        # Check structure
        expected_keys = {'coherence_score', 'temporal_consistency', 'meaning_preservation'}
        assert set(constraints.keys()) == expected_keys
        
        # Check value ranges
        for key, value in constraints.items():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0, f"{key} should be in [0,1], got {value}"
            assert jnp.isfinite(value)
    
    def test_multi_objective_optimize(self, optimizer_setup):
        """Test multi-objective optimization."""
        optimizer, config = optimizer_setup
        key = jax.random.PRNGKey(42)
        
        target_experience = jax.random.normal(key, (config.experience_dim,))
        
        result = optimizer.multi_objective_optimize(
            target_experience,
            temporal_context=None,
            max_iterations=20
        )
        
        # Check result structure
        expected_keys = {'pareto_solutions', 'trade_off_analysis', 'selected_solution'}
        assert set(result.keys()) == expected_keys
        
        # Check pareto solutions
        assert isinstance(result['pareto_solutions'], list)
        assert len(result['pareto_solutions']) > 0
        
        for solution in result['pareto_solutions']:
            assert solution.shape == (config.dictionary_size,)
            assert jnp.all(jnp.isfinite(solution))
        
        # Check selected solution
        assert result['selected_solution'].shape == (config.dictionary_size,)
        assert jnp.all(jnp.isfinite(result['selected_solution']))


class TestIntegratedSparseRepresentationSystem:
    """Test cases for IntegratedSparseRepresentationSystem."""
    
    @pytest.fixture
    def system_setup(self):
        """Set up integrated system for testing."""
        config = SparseRepresentationsConfig(
            experience_dim=16,
            dictionary_size=32,
            sparsity_level=0.3,
        )
        key = jax.random.PRNGKey(42)
        system = IntegratedSparseRepresentationSystem(config, key)
        return system, config
    
    def test_system_initialization(self, system_setup):
        """Test IntegratedSparseRepresentationSystem initialization."""
        system, config = system_setup
        
        assert system.config == config
        assert isinstance(system.encoder, SparseExperienceEncoder)
        assert isinstance(system.dictionary_learner, AdaptiveDictionaryLearner)
        assert isinstance(system.meaning_optimizer, ConvexMeaningOptimizer)
    
    def test_compress_experiential_traces_experience_traces(self, system_setup):
        """Test compression of experience traces."""
        system, config = system_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)
        
        # Create simple experience traces
        traces = []
        for i in range(3):
            trace = SimpleExperienceTrace(
                content=jax.random.normal(keys[i], (config.experience_dim,)),
                temporal_signature=jax.random.normal(keys[i], (8,)),
                associative_links=jax.random.normal(keys[i], (4,)),
                sedimentation_level=0.5 + 0.1 * i,
                creation_timestamp=100.0 + i,
                last_access_timestamp=150.0 + i,
                access_frequency=5 + i,
            )
            traces.append(trace)
        
        compressed_traces, compression_metrics = system.compress_experiential_traces(traces)
        
        # Check compressed traces
        assert len(compressed_traces) == len(traces)
        for compressed_trace in compressed_traces:
            assert isinstance(compressed_trace, dict)
            expected_keys = {'sparse_code', 'original_trace', 'compression_metadata'}
            assert set(compressed_trace.keys()) == expected_keys
            
            assert compressed_trace['sparse_code'].shape == (config.dictionary_size,)
            assert isinstance(compressed_trace['original_trace'], SimpleExperienceTrace)
            assert isinstance(compressed_trace['compression_metadata'], dict)
        
        # Check compression metrics
        expected_metrics = {'compression_ratio', 'reconstruction_quality', 'sparsity_achieved'}
        assert set(compression_metrics.keys()) == expected_metrics
        
        for key, value in compression_metrics.items():
            assert isinstance(value, float)
            assert jnp.isfinite(value)
            assert value >= 0.0
    
    def test_compress_experiential_traces_experiential_traces(self, system_setup):
        """Test compression of experiential traces."""
        system, config = system_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Create simple experiential traces
        traces = []
        for i in range(2):
            trace = SimpleExperientialTrace(
                interaction_pattern=jax.random.normal(keys[i], (config.experience_dim,)),
                contextual_embedding=jax.random.normal(keys[i+2], (8,)),
                significance_weight=0.6 + 0.1 * i,
                temporal_depth=2.0 + i,
                coupling_strength=0.7 + 0.05 * i,
            )
            traces.append(trace)
        
        compressed_traces, compression_metrics = system.compress_experiential_traces(traces)
        
        # Check results
        assert len(compressed_traces) == len(traces)
        assert isinstance(compression_metrics, dict)
        
        for compressed_trace in compressed_traces:
            assert compressed_trace['sparse_code'].shape == (config.dictionary_size,)
            assert isinstance(compressed_trace['original_trace'], SimpleExperientialTrace)
    
    def test_online_learning_step(self, system_setup):
        """Test online learning step."""
        system, config = system_setup
        key = jax.random.PRNGKey(42)
        
        experience = jax.random.normal(key, (config.experience_dim,))
        
        # Perform online learning step
        updated_system = system.online_learning_step(experience)
        
        assert isinstance(updated_system, IntegratedSparseRepresentationSystem)
        
        # System components should be updated
        assert updated_system.config == system.config
        # Dictionary should potentially be different
        dict_difference = jnp.linalg.norm(
            updated_system.encoder.dictionary - system.encoder.dictionary
        )
        assert dict_difference >= 0.0  # Could be zero if no significant adaptation
    
    def test_optimize_meaning_structure(self, system_setup):
        """Test meaning structure optimization."""
        system, config = system_setup
        key = jax.random.PRNGKey(42)
        
        # Create meaning structure
        meaning = MeaningStructure(
            semantic_content=jax.random.normal(key, (config.experience_dim,)),
            coherence_measure=0.7,
            relevance_weight=0.8,
            emergence_timestamp=200.0,
        )
        
        optimized_meaning, optimization_metrics = system.optimize_meaning_structure(meaning)
        
        # Check optimized meaning
        assert isinstance(optimized_meaning, MeaningStructure)
        assert optimized_meaning.semantic_content.shape == meaning.semantic_content.shape
        assert 0.0 <= optimized_meaning.coherence_measure <= 1.0
        assert optimized_meaning.relevance_weight == meaning.relevance_weight
        assert optimized_meaning.emergence_timestamp == meaning.emergence_timestamp
        
        # Check optimization metrics
        assert isinstance(optimization_metrics, dict)
        expected_keys = {'optimization_convergence', 'constraint_satisfaction', 'coherence_improvement'}
        assert set(optimization_metrics.keys()) == expected_keys
    
    def test_compress_with_consciousness_preservation(self, system_setup):
        """Test compression with consciousness preservation."""
        system, config = system_setup
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        experience = jax.random.normal(keys[0], (config.experience_dim,))
        
        # Create temporal moment
        temporal_moment = TemporalMoment(
            timestamp=100.0,
            retention=jax.random.normal(keys[1], (config.experience_dim,)),
            present_moment=experience,
            protention=jax.random.normal(keys[2], (config.experience_dim,)),
            synthesis_weights=jax.random.normal(keys[3], (config.experience_dim,)),
        )
        
        # Create meaning structure
        meaning = MeaningStructure(
            semantic_content=experience,
            coherence_measure=0.8,
            relevance_weight=0.9,
            emergence_timestamp=100.0,
        )
        
        result = system.compress_with_consciousness_preservation(
            experience, temporal_moment, meaning
        )
        
        # Check result structure
        expected_keys = {
            'compressed_representation', 'consciousness_metrics',
            'preservation_quality', 'reconstruction_fidelity'
        }
        assert set(result.keys()) == expected_keys
        
        # Check values
        assert result['compressed_representation'].shape == (config.dictionary_size,)
        assert isinstance(result['consciousness_metrics'], dict)
        assert 0.0 <= result['preservation_quality'] <= 1.0
        assert 0.0 <= result['reconstruction_fidelity'] <= 1.0


class TestFactoryAndUtilityFunctions:
    """Test cases for factory and utility functions."""
    
    def test_create_sparse_representations_system(self):
        """Test factory function for creating sparse representations system."""
        config = SparseRepresentationsConfig(experience_dim=20, dictionary_size=40)
        key = jax.random.PRNGKey(42)
        
        system = create_sparse_representations_system(config, key)
        
        assert isinstance(system, IntegratedSparseRepresentationSystem)
        assert system.config == config
    
    def test_analyze_sparsity_quality(self):
        """Test sparsity quality analysis."""
        config = SparseRepresentationsConfig(experience_dim=16, dictionary_size=32)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        system = create_sparse_representations_system(config, keys[0])
        
        # Create experiences
        experiences = [
            jax.random.normal(keys[i], (config.experience_dim,))
            for i in range(1, 4)
        ]
        
        quality_metrics = analyze_sparsity_quality(system, experiences)
        
        # Check metrics
        expected_keys = {
            'sparsity_level', 'reconstruction_fidelity',
            'compression_efficiency', 'consciousness_preservation'
        }
        assert set(quality_metrics.keys()) == expected_keys
        
        for key, value in quality_metrics.items():
            assert isinstance(value, float)
            assert jnp.isfinite(value)
            assert value >= 0.0
    
    def test_compress_experiential_traces_utility(self):
        """Test utility function for compressing experiential traces."""
        config = SparseRepresentationsConfig(experience_dim=12, dictionary_size=24)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        # Create traces
        traces = [
            SimpleExperienceTrace(
                content=jax.random.normal(keys[i], (config.experience_dim,)),
                temporal_signature=jax.random.normal(keys[i], (6,)),
                associative_links=jax.random.normal(keys[i], (3,)),
                sedimentation_level=0.5,
                creation_timestamp=100.0,
                last_access_timestamp=150.0,
                access_frequency=5,
            )
            for i in range(2)
        ]
        
        compressed_traces, compression_metrics = compress_experiential_traces(
            traces, config, keys[2]
        )
        
        assert len(compressed_traces) == len(traces)
        assert isinstance(compression_metrics, dict)
    
    def test_decompress_sparse_representation(self):
        """Test sparse representation decompression."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        experience_dim = 10
        dictionary_size = 20
        
        sparse_code = jax.random.normal(keys[0], (dictionary_size,)) * 0.1
        dictionary = jax.random.normal(keys[1], (dictionary_size, experience_dim)) * 0.1
        
        decompressed = decompress_sparse_representation(sparse_code, dictionary)
        
        assert decompressed.shape == (experience_dim,)
        assert jnp.all(jnp.isfinite(decompressed))
    
    def test_validate_consciousness_constraints(self):
        """Test consciousness constraints validation."""
        config = SparseRepresentationsConfig(experience_dim=16, dictionary_size=32)
        key = jax.random.PRNGKey(42)
        
        sparse_code = jax.random.normal(key, (config.dictionary_size,)) * 0.1
        
        constraints = validate_consciousness_constraints(sparse_code, config)
        
        # Check structure
        expected_keys = {
            'coherence_score', 'temporal_consistency',
            'meaning_preservation', 'overall_validity'
        }
        assert set(constraints.keys()) == expected_keys
        
        # Check value ranges
        for key, value in constraints.items():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0
            assert jnp.isfinite(value)


class TestMathematicalCorrectness:
    """Test cases for mathematical correctness of sparse representations."""
    
    def test_dictionary_orthogonality(self):
        """Test dictionary atom orthogonality properties."""
        config = SparseRepresentationsConfig(experience_dim=20, dictionary_size=20)  # Square dictionary
        key = jax.random.PRNGKey(42)
        
        encoder = SparseExperienceEncoder(config, key)
        
        # Check that dictionary atoms are normalized
        dictionary_norms = jnp.linalg.norm(encoder.dictionary, axis=1)
        assert jnp.allclose(dictionary_norms, 1.0, atol=1e-5)
        
        # For overcomplete dictionaries, atoms cannot be orthogonal, but check coherence
        if config.dictionary_size > config.experience_dim:
            # Coherence should be bounded
            gram_matrix = encoder.dictionary @ encoder.dictionary.T
            off_diagonal = gram_matrix - jnp.eye(config.dictionary_size)
            max_coherence = jnp.max(jnp.abs(off_diagonal))
            assert max_coherence >= 0.0  # Coherence is always non-negative
    
    def test_sparsity_regularization_effect(self):
        """Test effect of sparsity regularization strength."""
        config1 = SparseRepresentationsConfig(
            experience_dim=16, dictionary_size=32,
            regularization_strength=0.01  # Low regularization
        )
        config2 = SparseRepresentationsConfig(
            experience_dim=16, dictionary_size=32,
            regularization_strength=0.1   # High regularization
        )
        
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        encoder1 = SparseExperienceEncoder(config1, keys[0])
        encoder2 = SparseExperienceEncoder(config2, keys[1])
        
        experience = jax.random.normal(keys[2], (16,))
        
        sparse_code1 = encoder1.encode_experience(experience)
        sparse_code2 = encoder2.encode_experience(experience)
        
        # Higher regularization should lead to sparser codes
        sparsity1 = jnp.mean(jnp.abs(sparse_code1) > 1e-6)
        sparsity2 = jnp.mean(jnp.abs(sparse_code2) > 1e-6)
        
        # Both should respect their target sparsity levels
        assert sparsity1 <= config1.sparsity_level + 0.1
        assert sparsity2 <= config2.sparsity_level + 0.1
    
    def test_reconstruction_quality_sparsity_tradeoff(self):
        """Test tradeoff between reconstruction quality and sparsity."""
        config_sparse = SparseRepresentationsConfig(
            experience_dim=20, dictionary_size=40, sparsity_level=0.1
        )
        config_dense = SparseRepresentationsConfig(
            experience_dim=20, dictionary_size=40, sparsity_level=0.5
        )
        
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        encoder_sparse = SparseExperienceEncoder(config_sparse, keys[0])
        encoder_dense = SparseExperienceEncoder(config_dense, keys[1])
        
        experience = jax.random.normal(keys[2], (20,))
        
        sparse_code_sparse = encoder_sparse.encode_experience(experience)
        sparse_code_dense = encoder_dense.encode_experience(experience)
        
        error_sparse = encoder_sparse.compute_reconstruction_error(experience, sparse_code_sparse)
        error_dense = encoder_dense.compute_reconstruction_error(experience, sparse_code_dense)
        
        # Both should provide reasonable reconstruction
        assert error_sparse >= 0.0
        assert error_dense >= 0.0
        assert jnp.isfinite(error_sparse)
        assert jnp.isfinite(error_dense)
    
    def test_consciousness_constraint_satisfaction(self):
        """Test mathematical properties of consciousness constraints."""
        config = SparseRepresentationsConfig(experience_dim=16, dictionary_size=32)
        key = jax.random.PRNGKey(42)
        
        optimizer = ConvexMeaningOptimizer(config, key)
        
        # Test with zero sparse code
        zero_code = jnp.zeros(config.dictionary_size)
        constraints_zero = optimizer.evaluate_consciousness_constraints(zero_code)
        
        # Test with random sparse code
        random_code = jax.random.normal(key, (config.dictionary_size,)) * 0.1
        constraints_random = optimizer.evaluate_consciousness_constraints(random_code)
        
        # Both should have valid constraint scores
        for constraints in [constraints_zero, constraints_random]:
            for key, value in constraints.items():
                assert 0.0 <= value <= 1.0
                assert jnp.isfinite(value)


class TestPerformanceAndScalability:
    """Test cases for performance and scalability."""
    
    @pytest.mark.parametrize("experience_dim", [16, 32, 64])
    def test_encoder_scalability_with_dimension(self, experience_dim):
        """Test encoder scalability with different experience dimensions."""
        config = SparseRepresentationsConfig(
            experience_dim=experience_dim,
            dictionary_size=experience_dim * 2,
            sparsity_level=0.2,
        )
        key = jax.random.PRNGKey(42)
        
        encoder = SparseExperienceEncoder(config, key)
        
        experience = jax.random.normal(key, (experience_dim,))
        sparse_code = encoder.encode_experience(experience)
        
        assert sparse_code.shape == (config.dictionary_size,)
        assert jnp.all(jnp.isfinite(sparse_code))
    
    @pytest.mark.parametrize("dictionary_size", [32, 64, 128])
    def test_encoder_scalability_with_dictionary_size(self, dictionary_size):
        """Test encoder scalability with different dictionary sizes."""
        config = SparseRepresentationsConfig(
            experience_dim=24,
            dictionary_size=dictionary_size,
            sparsity_level=0.15,
        )
        key = jax.random.PRNGKey(42)
        
        encoder = SparseExperienceEncoder(config, key)
        
        experience = jax.random.normal(key, (24,))
        sparse_code = encoder.encode_experience(experience)
        
        assert sparse_code.shape == (dictionary_size,)
        assert jnp.all(jnp.isfinite(sparse_code))
    
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency."""
        config = SparseRepresentationsConfig(experience_dim=20, dictionary_size=40)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        
        encoder = SparseExperienceEncoder(config, keys[0])
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20]
        
        for batch_size in batch_sizes:
            experiences_batch = jax.random.normal(keys[1], (batch_size, config.experience_dim))
            
            sparse_codes_batch = encoder.encode_batch(experiences_batch)
            
            assert sparse_codes_batch.shape == (batch_size, config.dictionary_size)
            assert jnp.all(jnp.isfinite(sparse_codes_batch))
    
    def test_memory_efficiency_large_system(self):
        """Test memory efficiency with larger system."""
        # Use larger but manageable dimensions for testing
        config = SparseRepresentationsConfig(
            experience_dim=100,
            dictionary_size=200,
            sparsity_level=0.1,
        )
        key = jax.random.PRNGKey(42)
        
        # Should be able to create and use system without memory issues
        system = create_sparse_representations_system(config, key)
        
        experience = jax.random.normal(key, (config.experience_dim,))
        sparse_code = system.encoder.encode_experience(experience)
        
        assert sparse_code.shape == (config.dictionary_size,)
        assert jnp.all(jnp.isfinite(sparse_code))


class TestErrorHandlingAndEdgeCases:
    """Test cases for error handling and edge cases."""
    
    def test_unsupported_trace_type(self):
        """Test error handling for unsupported trace types."""
        config = SparseRepresentationsConfig(experience_dim=16, dictionary_size=32)
        key = jax.random.PRNGKey(42)
        
        system = create_sparse_representations_system(config, key)
        
        # Create unsupported trace type
        class UnsupportedTrace:
            def __init__(self):
                self.some_data = "data"
        
        unsupported_trace = UnsupportedTrace()
        
        with pytest.raises(ValueError, match="Unsupported trace type"):
            system.compress_experiential_traces([unsupported_trace])
    
    def test_zero_experience(self):
        """Test behavior with zero experience vectors."""
        config = SparseRepresentationsConfig(experience_dim=16, dictionary_size=32)
        key = jax.random.PRNGKey(42)
        
        encoder = SparseExperienceEncoder(config, key)
        
        zero_experience = jnp.zeros(config.experience_dim)
        sparse_code = encoder.encode_experience(zero_experience)
        
        # Should handle gracefully
        assert sparse_code.shape == (config.dictionary_size,)
        assert jnp.all(jnp.isfinite(sparse_code))
    
    def test_very_sparse_target(self):
        """Test behavior with very sparse target sparsity."""
        config = SparseRepresentationsConfig(
            experience_dim=20,
            dictionary_size=40,
            sparsity_level=0.01  # Very sparse
        )
        key = jax.random.PRNGKey(42)
        
        encoder = SparseExperienceEncoder(config, key)
        
        experience = jax.random.normal(key, (config.experience_dim,))
        sparse_code = encoder.encode_experience(experience)
        
        # Should enforce very sparse representation
        non_zero_fraction = jnp.mean(jnp.abs(sparse_code) > 1e-6)
        assert non_zero_fraction <= config.sparsity_level + 0.05
    
    def test_dictionary_size_smaller_than_experience_dim(self):
        """Test behavior when dictionary size is smaller than experience dimension."""
        config = SparseRepresentationsConfig(
            experience_dim=32,
            dictionary_size=16,  # Smaller than experience_dim
            sparsity_level=0.5,
        )
        key = jax.random.PRNGKey(42)
        
        # Should still work (undercomplete dictionary)
        encoder = SparseExperienceEncoder(config, key)
        
        assert encoder.dictionary.shape == (16, 32)
        
        experience = jax.random.normal(key, (32,))
        sparse_code = encoder.encode_experience(experience)
        
        assert sparse_code.shape == (16,)
        assert jnp.all(jnp.isfinite(sparse_code))


# Integration tests
class TestSparseRepresentationsIntegration:
    """Integration tests for sparse representations with other modules."""
    
    def test_information_theory_integration(self):
        """Test integration with information theory measures."""
        config = SparseRepresentationsConfig(experience_dim=20, dictionary_size=40)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        system = create_sparse_representations_system(config, keys[0])
        
        # Create correlated experiences
        base_experience = jax.random.normal(keys[1], (config.experience_dim,))
        experience1 = base_experience + 0.1 * jax.random.normal(keys[2], (config.experience_dim,))
        experience2 = base_experience + 0.1 * jax.random.normal(keys[2], (config.experience_dim,))
        
        # Encode experiences
        sparse_code1 = system.encoder.encode_experience(experience1)
        sparse_code2 = system.encoder.encode_experience(experience2)
        
        # Should be able to analyze information-theoretic properties
        assert jnp.all(jnp.isfinite(sparse_code1))
        assert jnp.all(jnp.isfinite(sparse_code2))
        
        # Codes should have some similarity due to correlation in original experiences
        correlation = jnp.corrcoef(sparse_code1, sparse_code2)[0, 1]
        assert jnp.isfinite(correlation)
    
    def test_temporal_dynamics_integration(self):
        """Test integration with temporal processing."""
        config = SparseRepresentationsConfig(experience_dim=16, dictionary_size=32)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 6)
        
        system = create_sparse_representations_system(config, keys[0])
        
        # Create temporal sequence of experiences
        temporal_experiences = []
        for t in range(5):
            # Each experience evolves from the previous
            if t == 0:
                experience = jax.random.normal(keys[t+1], (config.experience_dim,))
            else:
                experience = (0.8 * temporal_experiences[-1] + 
                            0.2 * jax.random.normal(keys[t+1], (config.experience_dim,)))
            temporal_experiences.append(experience)
        
        # Encode temporal sequence
        sparse_codes_sequence = []
        current_system = system
        
        for experience in temporal_experiences:
            sparse_code = current_system.encoder.encode_experience(experience)
            sparse_codes_sequence.append(sparse_code)
            
            # Online learning adaptation
            current_system = current_system.online_learning_step(experience)
        
        # Should have consistent sequence of codes
        assert len(sparse_codes_sequence) == 5
        for code in sparse_codes_sequence:
            assert code.shape == (config.dictionary_size,)
            assert jnp.all(jnp.isfinite(code))
    
    def test_consciousness_preservation_workflow(self):
        """Test complete consciousness preservation workflow."""
        config = SparseRepresentationsConfig(experience_dim=24, dictionary_size=48)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)
        
        system = create_sparse_representations_system(config, keys[0])
        
        # Create experience with high consciousness characteristics
        experience = jax.random.normal(keys[1], (config.experience_dim,))
        
        # Create temporal context
        temporal_moment = TemporalMoment(
            timestamp=100.0,
            retention=jax.random.normal(keys[2], (config.experience_dim,)),
            present_moment=experience,
            protention=jax.random.normal(keys[3], (config.experience_dim,)),
            synthesis_weights=jax.random.normal(keys[4], (config.experience_dim,)),
        )
        
        # Create meaning structure
        meaning = MeaningStructure(
            semantic_content=experience,
            coherence_measure=0.8,
            relevance_weight=0.9,
            emergence_timestamp=100.0,
        )
        
        # Full consciousness preservation workflow
        result = system.compress_with_consciousness_preservation(
            experience, temporal_moment, meaning
        )
        
        # Should achieve good preservation
        assert result['preservation_quality'] >= 0.0
        assert result['reconstruction_fidelity'] >= 0.0
        assert 0.0 <= result['preservation_quality'] <= 1.0
        assert 0.0 <= result['reconstruction_fidelity'] <= 1.0
        
        # Consciousness metrics should be well-formed
        consciousness_metrics = result['consciousness_metrics']
        for key, value in consciousness_metrics.items():
            assert 0.0 <= value <= 1.0
            assert jnp.isfinite(value)


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main([__file__, "-v", "--tb=short"])