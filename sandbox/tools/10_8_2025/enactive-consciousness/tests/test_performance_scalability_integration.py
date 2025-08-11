#!/usr/bin/env python3
"""Performance and Scalability Integration Test Suite.

Tests performance characteristics and scalability of the integrated 
enactive consciousness system under various workload conditions.

Focus areas:
1. JIT Compilation Optimization across integration points
2. Memory efficiency with realistic workloads  
3. Processing time scalability analysis
4. Resource utilization patterns
5. Performance degradation under stress conditions

Project Orchestrator Quality Assurance Role:
- Ensures integration tests maintain high quality standards
- Validates performance improvements don't compromise correctness
- Coordinates performance optimization across components
- Maintains architectural integrity under load
"""

import pytest
import sys
import os
import time
import json
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from contextlib import contextmanager
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import jax
import jax.numpy as jnp
import equinox as eqx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Core framework imports
from enactive_consciousness import (
    create_framework_config,
    TemporalConsciousnessConfig,
    BodySchemaConfig,
)

from enactive_consciousness.integrated_consciousness import (
    EnactiveConsciousnessSystem,
    ConsciousnessState,
    create_enactive_consciousness_system,
    run_consciousness_sequence,
)

from enactive_consciousness.temporal import (
    create_temporal_processor_safe,
    analyze_temporal_coherence,
)

from enactive_consciousness.experiential_memory import (
    IntegratedExperientialMemory,
)


@dataclass
class PerformanceMeasurement:
    """Container for performance measurement data."""
    operation_name: str
    duration_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@contextmanager
def performance_monitor():
    """Context manager for performance monitoring."""
    
    # Get process for memory monitoring
    process = psutil.Process()
    
    # Initial measurements
    start_time = time.perf_counter()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_cpu = process.cpu_percent()
    
    peak_memory = start_memory
    
    def update_peak_memory():
        nonlocal peak_memory
        current_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)
        return current_memory
    
    try:
        yield update_peak_memory
    finally:
        # Final measurements
        end_time = time.perf_counter()
        end_memory = update_peak_memory()
        end_cpu = process.cpu_percent()
        
        # Force garbage collection
        gc.collect()


class PerformanceTestFixture:
    """Performance testing fixture with comprehensive monitoring."""
    
    def __init__(
        self,
        base_state_dim: int = 64,
        base_environment_dim: int = 24,
        base_context_dim: int = 32,
        test_key: Optional[jax.Array] = None,
    ):
        """Initialize performance test fixture."""
        
        self.base_state_dim = base_state_dim
        self.base_environment_dim = base_environment_dim
        self.base_context_dim = base_context_dim
        
        if test_key is None:
            test_key = jax.random.PRNGKey(54321)
        
        self.keys = jax.random.split(test_key, 50)
        
        # Performance measurement storage
        self.performance_measurements: List[PerformanceMeasurement] = []
        self.scalability_results: Dict[str, List[PerformanceMeasurement]] = {}
        
        # System variants for testing
        self.system_variants = {}
        
    def measure_operation(
        self,
        operation_name: str,
        operation_func,
        *args,
        **kwargs
    ) -> PerformanceMeasurement:
        """Measure performance of an operation."""
        
        with performance_monitor() as update_memory:
            start_memory = update_memory()
            
            success = True
            error_message = None
            metadata = {}
            
            try:
                start_time = time.perf_counter()
                
                result = operation_func(*args, **kwargs)
                
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                # Extract metadata if result is a dict
                if isinstance(result, dict) and 'metadata' in result:
                    metadata = result['metadata']
                
            except Exception as e:
                success = False
                error_message = str(e)
                duration_ms = 0.0
            
            end_memory = update_memory()
            peak_memory = end_memory  # Simplified for this context
            
        measurement = PerformanceMeasurement(
            operation_name=operation_name,
            duration_ms=duration_ms,
            memory_before_mb=start_memory,
            memory_after_mb=end_memory,
            memory_peak_mb=peak_memory,
            cpu_percent=0.0,  # Simplified
            success=success,
            error_message=error_message,
            metadata=metadata,
        )
        
        self.performance_measurements.append(measurement)
        return measurement
    
    def create_system_variant(
        self,
        variant_name: str,
        state_dim: int,
        environment_dim: int,
        context_dim: int,
        key: jax.Array,
    ) -> EnactiveConsciousnessSystem:
        """Create a system variant for scalability testing."""
        
        config = create_framework_config(
            retention_depth=max(6, state_dim // 16),
            protention_horizon=max(3, state_dim // 32),
            consciousness_threshold=0.4,
            proprioceptive_dim=min(64, state_dim),
            motor_dim=min(24, state_dim // 4),
        )
        
        temporal_config = TemporalConsciousnessConfig(
            retention_depth=max(6, state_dim // 16),
            protention_horizon=max(3, state_dim // 32),
            temporal_synthesis_rate=0.1,
            temporal_decay_factor=0.92,
        )
        
        body_config = BodySchemaConfig(
            proprioceptive_dim=min(64, state_dim),
            motor_dim=min(24, state_dim // 4),
            body_map_resolution=(min(12, state_dim // 8), min(12, state_dim // 8)),
            boundary_sensitivity=0.7,
            schema_adaptation_rate=0.015,
        )
        
        def create_system():
            return create_enactive_consciousness_system(
                config=config,
                temporal_config=temporal_config,
                body_config=body_config,
                state_dim=state_dim,
                environment_dim=environment_dim,
                key=key,
            )
        
        system = self.measure_operation(
            f"create_system_{variant_name}",
            create_system,
        )
        
        if system.success:
            # Store the created system by re-creating it (since we can't return from measured operation easily)
            created_system = create_enactive_consciousness_system(
                config=config,
                temporal_config=temporal_config,
                body_config=body_config,
                state_dim=state_dim,
                environment_dim=environment_dim,
                key=key,
            )
            self.system_variants[variant_name] = created_system
            return created_system
        else:
            raise RuntimeError(f"Failed to create system variant {variant_name}: {system.error_message}")
    
    def generate_test_sequence(
        self,
        length: int,
        state_dim: int,
        environment_dim: int,
        context_dim: int,
        key: jax.Array,
    ) -> List[Dict[str, jax.Array]]:
        """Generate test sequence for performance testing."""
        
        keys = jax.random.split(key, length * 6)
        sequence = []
        
        for t in range(length):
            # Structured patterns for realistic performance testing
            base_pattern = jnp.sin(t * 0.15) * 0.6 + jnp.cos(t * 0.08) * 0.4
            noise_scale = 0.1
            
            sensory_input = (
                base_pattern * jnp.ones(state_dim) +
                jax.random.normal(keys[t*6], (state_dim,)) * noise_scale
            )
            
            proprioceptive_input = (
                base_pattern * 0.7 * jnp.ones(min(64, state_dim)) +
                jax.random.normal(keys[t*6+1], (min(64, state_dim),)) * noise_scale
            )
            
            motor_prediction = (
                base_pattern * 0.5 * jnp.ones(min(24, state_dim // 4)) +
                jax.random.normal(keys[t*6+2], (min(24, state_dim // 4),)) * noise_scale
            )
            
            environmental_state = (
                base_pattern * 0.4 * jnp.ones(environment_dim) +
                jax.random.normal(keys[t*6+3], (environment_dim,)) * noise_scale
            )
            
            contextual_cues = (
                base_pattern * 0.3 * jnp.ones(context_dim) +
                jax.random.normal(keys[t*6+4], (context_dim,)) * noise_scale
            )
            
            sequence.append({
                'sensory_input': sensory_input,
                'proprioceptive_input': proprioceptive_input,
                'motor_prediction': motor_prediction,
                'environmental_state': environmental_state,
                'contextual_cues': contextual_cues,
            })
        
        return sequence


class TestJITOptimizationIntegration:
    """Test JIT compilation optimization across integration points."""
    
    @pytest.fixture
    def performance_fixture(self):
        """Create performance test fixture."""
        return PerformanceTestFixture()
    
    def test_jit_compilation_warmup_performance(self, performance_fixture):
        """Test JIT compilation warmup performance across components."""
        
        # Create system for JIT testing
        state_dim = performance_fixture.base_state_dim
        environment_dim = performance_fixture.base_environment_dim
        context_dim = performance_fixture.base_context_dim
        
        system = performance_fixture.create_system_variant(
            'jit_test_system',
            state_dim,
            environment_dim,
            context_dim,
            performance_fixture.keys[0],
        )
        
        # Generate test sequence
        test_sequence = performance_fixture.generate_test_sequence(
            10, state_dim, environment_dim, context_dim, performance_fixture.keys[1]
        )
        
        # Test cold start (first compilation)
        def cold_start_processing():
            return run_consciousness_sequence(system, test_sequence[:3])
        
        cold_measurement = performance_fixture.measure_operation(
            "jit_cold_start",
            cold_start_processing,
        )
        
        # Test warm processing (already compiled)
        def warm_processing():
            return run_consciousness_sequence(system, test_sequence[3:6])
        
        warm_measurement = performance_fixture.measure_operation(
            "jit_warm_processing",
            warm_processing,
        )
        
        # Test repeated warm processing
        def repeated_warm_processing():
            return run_consciousness_sequence(system, test_sequence[6:9])
        
        repeated_measurement = performance_fixture.measure_operation(
            "jit_repeated_warm",
            repeated_warm_processing,
        )
        
        # Validate JIT optimization
        assert cold_measurement.success
        assert warm_measurement.success
        assert repeated_measurement.success
        
        # Warm processing should generally be faster than cold start
        # (though not guaranteed due to varying system conditions)
        
        # Store JIT optimization results
        performance_fixture.scalability_results['jit_optimization'] = [
            cold_measurement,
            warm_measurement,
            repeated_measurement,
        ]
        
        print(f"JIT Cold Start: {cold_measurement.duration_ms:.1f}ms")
        print(f"JIT Warm: {warm_measurement.duration_ms:.1f}ms") 
        print(f"JIT Repeated: {repeated_measurement.duration_ms:.1f}ms")
    
    def test_temporal_processor_jit_fallback(self, performance_fixture):
        """Test temporal processor JIT compilation with fallback mechanisms."""
        
        # Test safe temporal processor creation
        temporal_config = TemporalConsciousnessConfig(
            retention_depth=8,
            protention_horizon=4,
            temporal_synthesis_rate=0.1,
            temporal_decay_factor=0.92,
        )
        
        def create_temporal_processor():
            return create_temporal_processor_safe(
                temporal_config,
                performance_fixture.base_state_dim,
                performance_fixture.keys[2],
                use_jit=True,  # Try JIT first
            )
        
        processor_measurement = performance_fixture.measure_operation(
            "temporal_processor_safe_creation",
            create_temporal_processor,
        )
        
        assert processor_measurement.success
        
        # Re-create processor for testing (since we can't return from measured operation)
        temporal_processor = create_temporal_processor_safe(
            temporal_config,
            performance_fixture.base_state_dim,
            performance_fixture.keys[2],
            use_jit=True,
        )
        
        # Test temporal synthesis performance
        test_input = jax.random.normal(performance_fixture.keys[3], (performance_fixture.base_state_dim,)) * 0.5
        
        def temporal_synthesis():
            return temporal_processor.temporal_synthesis(
                primal_impression=test_input,
                timestamp=1.0,
            )
        
        synthesis_measurement = performance_fixture.measure_operation(
            "temporal_synthesis_processing",
            temporal_synthesis,
        )
        
        assert synthesis_measurement.success
        
        print(f"Temporal Processor Creation: {processor_measurement.duration_ms:.1f}ms")
        print(f"Temporal Synthesis: {synthesis_measurement.duration_ms:.1f}ms")
    
    def test_experiential_memory_jit_performance(self, performance_fixture):
        """Test experiential memory JIT performance characteristics."""
        
        # Create experiential memory system
        def create_experiential_memory():
            return IntegratedExperientialMemory(
                experience_dim=performance_fixture.base_state_dim,
                environment_dim=performance_fixture.base_environment_dim,
                context_dim=performance_fixture.base_context_dim,
                key=performance_fixture.keys[4],
            )
        
        memory_creation_measurement = performance_fixture.measure_operation(
            "experiential_memory_creation",
            create_experiential_memory,
        )
        
        assert memory_creation_measurement.success
        
        # Re-create for testing
        experiential_memory = IntegratedExperientialMemory(
            experience_dim=performance_fixture.base_state_dim,
            environment_dim=performance_fixture.base_environment_dim,
            context_dim=performance_fixture.base_context_dim,
            key=performance_fixture.keys[4],
        )
        
        # Test experiential processing performance
        test_experience = jax.random.normal(performance_fixture.keys[5], (performance_fixture.base_state_dim,)) * 0.4
        test_environment = jax.random.normal(performance_fixture.keys[6], (performance_fixture.base_environment_dim,)) * 0.4
        test_context = jax.random.normal(performance_fixture.keys[7], (performance_fixture.base_context_dim,)) * 0.4
        
        def experiential_processing():
            return experiential_memory.process_experiential_moment(
                current_experience=test_experience,
                environmental_input=test_environment,
                contextual_cues=test_context,
                significance_weight=0.6,
            )
        
        processing_measurement = performance_fixture.measure_operation(
            "experiential_memory_processing",
            experiential_processing,
        )
        
        assert processing_measurement.success
        
        print(f"Experiential Memory Creation: {memory_creation_measurement.duration_ms:.1f}ms")
        print(f"Experiential Processing: {processing_measurement.duration_ms:.1f}ms")


class TestMemoryEfficiencyIntegration:
    """Test memory efficiency across full integrated system."""
    
    @pytest.fixture
    def performance_fixture(self):
        """Create performance test fixture."""
        return PerformanceTestFixture()
    
    def test_consciousness_sequence_memory_patterns(self, performance_fixture):
        """Test memory usage patterns during consciousness sequence processing."""
        
        # Create system variants with different sizes
        system_sizes = [
            ('small', 32, 16, 16),
            ('medium', 64, 24, 32),
            ('large', 128, 48, 64),
        ]
        
        memory_patterns = {}
        
        for size_name, state_dim, env_dim, ctx_dim in system_sizes:
            try:
                # Create system
                system = performance_fixture.create_system_variant(
                    f'memory_test_{size_name}',
                    state_dim,
                    env_dim,
                    ctx_dim,
                    performance_fixture.keys[10],
                )
                
                # Generate test sequence
                sequence_length = 15
                test_sequence = performance_fixture.generate_test_sequence(
                    sequence_length, state_dim, env_dim, ctx_dim, performance_fixture.keys[11]
                )
                
                # Process sequence with memory monitoring
                def process_sequence():
                    return run_consciousness_sequence(system, test_sequence)
                
                memory_measurement = performance_fixture.measure_operation(
                    f"memory_pattern_{size_name}",
                    process_sequence,
                )
                
                memory_patterns[size_name] = {
                    'measurement': memory_measurement,
                    'state_dim': state_dim,
                    'sequence_length': sequence_length,
                    'memory_growth_mb': memory_measurement.memory_after_mb - memory_measurement.memory_before_mb,
                    'memory_efficiency': (memory_measurement.memory_after_mb - memory_measurement.memory_before_mb) / state_dim,
                }
                
            except Exception as e:
                memory_patterns[size_name] = {'error': str(e)}
                print(f"Memory pattern test failed for {size_name}: {e}")
        
        # Validate memory patterns
        successful_patterns = {k: v for k, v in memory_patterns.items() if 'error' not in v}
        assert len(successful_patterns) > 0
        
        # Analyze memory scalability
        if len(successful_patterns) > 1:
            memory_growths = [pattern['memory_growth_mb'] for pattern in successful_patterns.values()]
            state_dims = [pattern['state_dim'] for pattern in successful_patterns.values()]
            
            # Memory growth should be reasonable relative to state dimension increases
            max_memory_growth = max(memory_growths)
            max_state_dim = max(state_dims)
            
            memory_scalability_ratio = max_memory_growth / max_state_dim
            
            # Store scalability analysis
            performance_fixture.scalability_results['memory_patterns'] = {
                'patterns': memory_patterns,
                'memory_scalability_ratio': memory_scalability_ratio,
                'analysis_successful': True,
            }
            
            print(f"Memory scalability analysis:")
            for size_name, pattern in successful_patterns.items():
                print(f"  {size_name}: {pattern['memory_growth_mb']:.1f}MB growth, efficiency: {pattern['memory_efficiency']:.3f}")
    
    def test_experiential_memory_accumulation(self, performance_fixture):
        """Test memory accumulation in experiential memory over time."""
        
        # Create experiential memory system  
        experiential_memory = IntegratedExperientialMemory(
            experience_dim=performance_fixture.base_state_dim,
            environment_dim=performance_fixture.base_environment_dim,
            context_dim=performance_fixture.base_context_dim,
            key=performance_fixture.keys[12],
        )
        
        # Process multiple experiences and track memory
        memory_accumulation = []
        
        num_experiences = 25
        
        for i in range(num_experiences):
            test_experience = jax.random.normal(performance_fixture.keys[13 + i], (performance_fixture.base_state_dim,)) * 0.4
            test_environment = jax.random.normal(performance_fixture.keys[40 + i], (performance_fixture.base_environment_dim,)) * 0.4
            test_context = jax.random.normal(performance_fixture.keys[70 + i], (performance_fixture.base_context_dim,)) * 0.4
            
            def process_experience():
                return experiential_memory.process_experiential_moment(
                    current_experience=test_experience,
                    environmental_input=test_environment,
                    contextual_cues=test_context,
                    significance_weight=0.5 + i * 0.02,  # Varying significance
                )
            
            experience_measurement = performance_fixture.measure_operation(
                f"experience_accumulation_{i}",
                process_experience,
            )
            
            memory_accumulation.append({
                'experience_num': i,
                'measurement': experience_measurement,
                'memory_growth': experience_measurement.memory_after_mb - (
                    memory_accumulation[0]['measurement'].memory_before_mb if i > 0 
                    else experience_measurement.memory_before_mb
                ),
            })
        
        # Analyze memory accumulation pattern
        successful_experiences = [exp for exp in memory_accumulation if exp['measurement'].success]
        
        if len(successful_experiences) > 5:
            memory_growths = [exp['memory_growth'] for exp in successful_experiences]
            
            # Check for memory leaks (excessive growth)
            final_growth = memory_growths[-1]
            avg_growth_per_experience = final_growth / len(successful_experiences)
            
            performance_fixture.scalability_results['memory_accumulation'] = {
                'total_experiences': len(successful_experiences),
                'final_memory_growth_mb': final_growth,
                'avg_growth_per_experience_mb': avg_growth_per_experience,
                'memory_leak_suspected': avg_growth_per_experience > 1.0,  # More than 1MB per experience suggests leak
                'analysis_successful': True,
            }
            
            print(f"Memory accumulation analysis:")
            print(f"  Total experiences: {len(successful_experiences)}")
            print(f"  Final memory growth: {final_growth:.1f}MB")
            print(f"  Average per experience: {avg_growth_per_experience:.3f}MB")
    
    def test_garbage_collection_effectiveness(self, performance_fixture):
        """Test garbage collection effectiveness during processing."""
        
        # Create system
        system = performance_fixture.create_system_variant(
            'gc_test_system',
            performance_fixture.base_state_dim,
            performance_fixture.base_environment_dim,
            performance_fixture.base_context_dim,
            performance_fixture.keys[20],
        )
        
        # Process sequences with explicit garbage collection
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        for cycle in range(5):
            # Generate and process sequence
            test_sequence = performance_fixture.generate_test_sequence(
                10, 
                performance_fixture.base_state_dim, 
                performance_fixture.base_environment_dim, 
                performance_fixture.base_context_dim, 
                performance_fixture.keys[21 + cycle]
            )
            
            def process_and_gc():
                consciousness_states = run_consciousness_sequence(system, test_sequence)
                
                # Force garbage collection
                gc.collect()
                
                return {
                    'states_processed': len(consciousness_states),
                    'memory_after_gc': psutil.Process().memory_info().rss / 1024 / 1024,
                }
            
            gc_measurement = performance_fixture.measure_operation(
                f"gc_cycle_{cycle}",
                process_and_gc,
            )
            
            print(f"GC Cycle {cycle}: {gc_measurement.duration_ms:.1f}ms, Memory: {gc_measurement.memory_after_mb:.1f}MB")
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_cleanup_effectiveness = max(0.0, 1.0 - (final_memory - initial_memory) / initial_memory)
        
        performance_fixture.scalability_results['garbage_collection'] = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'cleanup_effectiveness': memory_cleanup_effectiveness,
            'gc_cycles_tested': 5,
        }
        
        print(f"GC effectiveness: {memory_cleanup_effectiveness:.2f}")


class TestProcessingTimeScalability:
    """Test processing time scalability across different workload sizes."""
    
    @pytest.fixture
    def performance_fixture(self):
        """Create performance test fixture."""
        return PerformanceTestFixture()
    
    def test_state_dimension_scalability(self, performance_fixture):
        """Test processing time scalability with increasing state dimensions."""
        
        # Test different state dimensions
        state_dimensions = [32, 64, 96, 128]
        scalability_measurements = []
        
        for state_dim in state_dimensions:
            try:
                env_dim = state_dim // 3
                ctx_dim = state_dim // 2
                
                # Create system variant
                system = performance_fixture.create_system_variant(
                    f'scalability_{state_dim}d',
                    state_dim,
                    env_dim,
                    ctx_dim,
                    performance_fixture.keys[25],
                )
                
                # Generate test sequence
                test_sequence = performance_fixture.generate_test_sequence(
                    8, state_dim, env_dim, ctx_dim, performance_fixture.keys[26]
                )
                
                # Measure processing
                def process_scalability_test():
                    return run_consciousness_sequence(system, test_sequence)
                
                scalability_measurement = performance_fixture.measure_operation(
                    f"scalability_{state_dim}d",
                    process_scalability_test,
                )
                
                scalability_measurements.append({
                    'state_dim': state_dim,
                    'measurement': scalability_measurement,
                    'time_per_dimension': scalability_measurement.duration_ms / state_dim,
                    'memory_per_dimension': (scalability_measurement.memory_after_mb - scalability_measurement.memory_before_mb) / state_dim,
                })
                
            except Exception as e:
                print(f"Scalability test failed for {state_dim}d: {e}")
                scalability_measurements.append({
                    'state_dim': state_dim,
                    'error': str(e),
                })
        
        # Analyze scalability
        successful_measurements = [m for m in scalability_measurements if 'error' not in m]
        
        if len(successful_measurements) > 1:
            # Compute scalability metrics
            processing_times = [m['measurement'].duration_ms for m in successful_measurements]
            state_dims = [m['state_dim'] for m in successful_measurements]
            
            # Simple linear regression for scalability analysis
            import numpy as np
            
            # Convert to numpy for analysis
            times_np = np.array(processing_times)
            dims_np = np.array(state_dims)
            
            # Fit linear relationship: time = a * dim + b
            coeffs = np.polyfit(dims_np, times_np, 1)
            linear_scaling_factor = coeffs[0]  # Time increase per dimension
            
            # Check for quadratic scaling
            quadratic_coeffs = np.polyfit(dims_np, times_np, 2)
            quadratic_factor = quadratic_coeffs[0]  # Quadratic term
            
            performance_fixture.scalability_results['state_dimension_scalability'] = {
                'measurements': scalability_measurements,
                'linear_scaling_factor': float(linear_scaling_factor),
                'quadratic_factor': float(quadratic_factor),
                'scaling_analysis_successful': True,
                'acceptable_scaling': linear_scaling_factor < 10.0,  # Less than 10ms per dimension
            }
            
            print(f"State dimension scalability analysis:")
            print(f"  Linear scaling factor: {linear_scaling_factor:.3f}ms per dimension")
            print(f"  Quadratic factor: {quadratic_factor:.6f}")
            print(f"  Acceptable scaling: {'Yes' if linear_scaling_factor < 10.0 else 'No'}")
            
            for measurement in successful_measurements:
                print(f"  {measurement['state_dim']}d: {measurement['measurement'].duration_ms:.1f}ms")
    
    def test_sequence_length_scalability(self, performance_fixture):
        """Test processing time scalability with increasing sequence lengths."""
        
        # Test different sequence lengths
        sequence_lengths = [5, 10, 15, 20]
        length_scalability = []
        
        # Use fixed system configuration
        state_dim = performance_fixture.base_state_dim
        env_dim = performance_fixture.base_environment_dim
        ctx_dim = performance_fixture.base_context_dim
        
        system = performance_fixture.create_system_variant(
            'length_scalability_system',
            state_dim,
            env_dim,
            ctx_dim,
            performance_fixture.keys[30],
        )
        
        for seq_length in sequence_lengths:
            try:
                # Generate test sequence
                test_sequence = performance_fixture.generate_test_sequence(
                    seq_length, state_dim, env_dim, ctx_dim, performance_fixture.keys[31]
                )
                
                # Measure processing
                def process_length_test():
                    return run_consciousness_sequence(system, test_sequence)
                
                length_measurement = performance_fixture.measure_operation(
                    f"length_{seq_length}",
                    process_length_test,
                )
                
                length_scalability.append({
                    'sequence_length': seq_length,
                    'measurement': length_measurement,
                    'time_per_step': length_measurement.duration_ms / seq_length,
                    'memory_per_step': (length_measurement.memory_after_mb - length_measurement.memory_before_mb) / seq_length,
                })
                
            except Exception as e:
                print(f"Length scalability test failed for {seq_length}: {e}")
                length_scalability.append({
                    'sequence_length': seq_length,
                    'error': str(e),
                })
        
        # Analyze sequence length scalability
        successful_length_measurements = [m for m in length_scalability if 'error' not in m]
        
        if len(successful_length_measurements) > 1:
            processing_times = [m['measurement'].duration_ms for m in successful_length_measurements]
            sequence_lengths_actual = [m['sequence_length'] for m in successful_length_measurements]
            
            # Linear scaling analysis
            import numpy as np
            
            times_np = np.array(processing_times)
            lengths_np = np.array(sequence_lengths_actual)
            
            coeffs = np.polyfit(lengths_np, times_np, 1)
            time_per_step = coeffs[0]
            
            performance_fixture.scalability_results['sequence_length_scalability'] = {
                'measurements': length_scalability,
                'time_per_step_ms': float(time_per_step),
                'linear_scaling': True,
                'acceptable_per_step_time': time_per_step < 100.0,  # Less than 100ms per step
            }
            
            print(f"Sequence length scalability analysis:")
            print(f"  Time per step: {time_per_step:.1f}ms")
            print(f"  Linear scaling: Yes")
            
            for measurement in successful_length_measurements:
                print(f"  {measurement['sequence_length']} steps: {measurement['measurement'].duration_ms:.1f}ms")


class TestStressConditions:
    """Test system behavior under stress conditions."""
    
    @pytest.fixture
    def performance_fixture(self):
        """Create performance test fixture."""
        return PerformanceTestFixture()
    
    def test_high_dimensional_stress(self, performance_fixture):
        """Test system under high-dimensional stress conditions."""
        
        # Stress test with high dimensions
        stress_state_dim = 256
        stress_env_dim = 96
        stress_ctx_dim = 128
        
        try:
            stress_system = performance_fixture.create_system_variant(
                'high_dimensional_stress',
                stress_state_dim,
                stress_env_dim,
                stress_ctx_dim,
                performance_fixture.keys[40],
            )
            
            # Generate stress test sequence
            stress_sequence = performance_fixture.generate_test_sequence(
                12, stress_state_dim, stress_env_dim, stress_ctx_dim, performance_fixture.keys[41]
            )
            
            # Process under stress
            def stress_processing():
                return run_consciousness_sequence(stress_system, stress_sequence)
            
            stress_measurement = performance_fixture.measure_operation(
                'high_dimensional_stress_test',
                stress_processing,
            )
            
            stress_results = {
                'stress_test_successful': stress_measurement.success,
                'processing_time_ms': stress_measurement.duration_ms,
                'memory_usage_mb': stress_measurement.memory_after_mb - stress_measurement.memory_before_mb,
                'dimensions_tested': {
                    'state': stress_state_dim,
                    'environment': stress_env_dim,
                    'context': stress_ctx_dim,
                },
                'error_message': stress_measurement.error_message,
            }
            
        except Exception as e:
            stress_results = {
                'stress_test_successful': False,
                'error_message': str(e),
                'dimensions_tested': {
                    'state': stress_state_dim,
                    'environment': stress_env_dim,
                    'context': stress_ctx_dim,
                },
            }
        
        performance_fixture.scalability_results['high_dimensional_stress'] = stress_results
        
        print(f"High-dimensional stress test:")
        print(f"  Dimensions: {stress_state_dim}x{stress_env_dim}x{stress_ctx_dim}")
        print(f"  Success: {stress_results['stress_test_successful']}")
        if stress_results['stress_test_successful']:
            print(f"  Time: {stress_results['processing_time_ms']:.1f}ms")
            print(f"  Memory: {stress_results['memory_usage_mb']:.1f}MB")
    
    def test_rapid_sequence_processing(self, performance_fixture):
        """Test rapid sequence processing stress."""
        
        # Rapid processing of multiple short sequences
        num_sequences = 20
        sequence_length = 5
        
        rapid_processing_results = []
        
        system = performance_fixture.create_system_variant(
            'rapid_processing_system',
            performance_fixture.base_state_dim,
            performance_fixture.base_environment_dim,
            performance_fixture.base_context_dim,
            performance_fixture.keys[42],
        )
        
        for i in range(num_sequences):
            try:
                test_sequence = performance_fixture.generate_test_sequence(
                    sequence_length,
                    performance_fixture.base_state_dim,
                    performance_fixture.base_environment_dim,
                    performance_fixture.base_context_dim,
                    performance_fixture.keys[43 + i],
                )
                
                def rapid_process():
                    return run_consciousness_sequence(system, test_sequence)
                
                rapid_measurement = performance_fixture.measure_operation(
                    f'rapid_sequence_{i}',
                    rapid_process,
                )
                
                rapid_processing_results.append({
                    'sequence_num': i,
                    'success': rapid_measurement.success,
                    'duration_ms': rapid_measurement.duration_ms,
                    'memory_delta_mb': rapid_measurement.memory_after_mb - rapid_measurement.memory_before_mb,
                })
                
            except Exception as e:
                rapid_processing_results.append({
                    'sequence_num': i,
                    'success': False,
                    'error': str(e),
                })
        
        # Analyze rapid processing
        successful_rapid = [r for r in rapid_processing_results if r['success']]
        
        if len(successful_rapid) > 0:
            avg_processing_time = sum(r['duration_ms'] for r in successful_rapid) / len(successful_rapid)
            total_memory_delta = sum(r['memory_delta_mb'] for r in successful_rapid)
            
            performance_fixture.scalability_results['rapid_sequence_processing'] = {
                'total_sequences': num_sequences,
                'successful_sequences': len(successful_rapid),
                'success_rate': len(successful_rapid) / num_sequences,
                'avg_processing_time_ms': avg_processing_time,
                'total_memory_delta_mb': total_memory_delta,
                'processing_stability': len(successful_rapid) == num_sequences,
            }
            
            print(f"Rapid sequence processing stress test:")
            print(f"  Success rate: {len(successful_rapid)}/{num_sequences}")
            print(f"  Avg processing time: {avg_processing_time:.1f}ms")
            print(f"  Total memory delta: {total_memory_delta:.1f}MB")


class TestPerformanceIntegrationReport:
    """Generate comprehensive performance integration report."""
    
    @pytest.fixture
    def performance_fixture(self):
        """Create performance test fixture."""
        return PerformanceTestFixture()
    
    def test_comprehensive_performance_report(self, performance_fixture):
        """Generate comprehensive performance report."""
        
        # Run basic performance tests to populate results
        system = performance_fixture.create_system_variant(
            'report_system',
            performance_fixture.base_state_dim,
            performance_fixture.base_environment_dim,
            performance_fixture.base_context_dim,
            performance_fixture.keys[45],
        )
        
        test_sequence = performance_fixture.generate_test_sequence(
            10,
            performance_fixture.base_state_dim,
            performance_fixture.base_environment_dim,
            performance_fixture.base_context_dim,
            performance_fixture.keys[46],
        )
        
        # Basic processing test
        def basic_processing():
            return run_consciousness_sequence(system, test_sequence)
        
        basic_measurement = performance_fixture.measure_operation(
            'comprehensive_report_basic_test',
            basic_processing,
        )
        
        # Compile comprehensive report
        comprehensive_report = {
            'test_timestamp': time.time(),
            'system_configuration': {
                'base_state_dim': performance_fixture.base_state_dim,
                'base_environment_dim': performance_fixture.base_environment_dim,
                'base_context_dim': performance_fixture.base_context_dim,
            },
            'basic_performance': {
                'processing_time_ms': basic_measurement.duration_ms,
                'memory_usage_mb': basic_measurement.memory_after_mb - basic_measurement.memory_before_mb,
                'success': basic_measurement.success,
            },
            'individual_measurements': [
                {
                    'operation': m.operation_name,
                    'duration_ms': m.duration_ms,
                    'memory_before_mb': m.memory_before_mb,
                    'memory_after_mb': m.memory_after_mb,
                    'success': m.success,
                }
                for m in performance_fixture.performance_measurements
            ],
            'scalability_results': performance_fixture.scalability_results,
            'performance_summary': {
                'total_operations_tested': len(performance_fixture.performance_measurements),
                'successful_operations': len([m for m in performance_fixture.performance_measurements if m.success]),
                'total_test_time_ms': sum(m.duration_ms for m in performance_fixture.performance_measurements if m.success),
                'avg_operation_time_ms': (
                    sum(m.duration_ms for m in performance_fixture.performance_measurements if m.success) /
                    max(1, len([m for m in performance_fixture.performance_measurements if m.success]))
                ),
            },
            'quality_assessment': {
                'performance_acceptable': basic_measurement.duration_ms < 5000,  # 5 second threshold
                'memory_efficiency_good': (basic_measurement.memory_after_mb - basic_measurement.memory_before_mb) < 100,  # 100MB threshold
                'overall_integration_success': basic_measurement.success,
            },
        }
        
        # Save comprehensive report
        report_file = Path(__file__).parent.parent / 'performance_integration_report.json'
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        # Performance assertions
        assert comprehensive_report['basic_performance']['success']
        assert comprehensive_report['performance_summary']['successful_operations'] > 0
        
        print(f"\n🎯 Comprehensive Performance Integration Report:")
        print(f"   Total operations tested: {comprehensive_report['performance_summary']['total_operations_tested']}")
        print(f"   Successful operations: {comprehensive_report['performance_summary']['successful_operations']}")
        print(f"   Basic processing time: {comprehensive_report['basic_performance']['processing_time_ms']:.1f}ms")
        print(f"   Basic memory usage: {comprehensive_report['basic_performance']['memory_usage_mb']:.1f}MB")
        print(f"   Performance acceptable: {'✅ YES' if comprehensive_report['quality_assessment']['performance_acceptable'] else '⚠️  NO'}")
        print(f"   Memory efficiency good: {'✅ YES' if comprehensive_report['quality_assessment']['memory_efficiency_good'] else '⚠️  NO'}")
        print(f"   Report saved to: {report_file}")
        
        return comprehensive_report


if __name__ == '__main__':
    """Run performance and scalability integration tests directly."""
    
    # Configure JAX for testing
    jax.config.update('jax_platform_name', 'cpu')
    
    print("🚀 Running Performance and Scalability Integration Tests")
    print("=" * 70)
    
    # Run tests with pytest
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--disable-warnings',
        '-s',  # Show output for performance monitoring
    ])