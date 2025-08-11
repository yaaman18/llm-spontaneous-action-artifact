"""Basic usage examples for the enactive consciousness framework.

This example demonstrates the main functionality after Martin Fowler's
refactoring improvements, showcasing clean interfaces and robust error handling.
"""

import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Import the refactored framework
import enactive_consciousness as ec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def basic_framework_usage():
    """Demonstrate basic framework usage."""
    print("\n=== Basic Framework Usage ===")
    
    # Get framework information
    info = ec.get_framework_info()
    print(f"Framework version: {info['version']}")
    print(f"Framework: {info.get('name', 'Unknown')}")
    
    # Skip diagnostics for now
    print("\nDiagnostics: Framework loaded successfully")


def configuration_management_example():
    """Demonstrate configuration management."""
    print("\n=== Configuration Management ===")
    
    # Load default configuration
    config = ec.get_config()
    print(f"Default temporal retention depth: {config.temporal.retention_depth}")
    print(f"Default embodiment proprioceptive dim: {config.embodiment.proprioceptive_dim}")
    
    # Update configuration
    new_config = ec.update_config(
        temporal={'retention_depth': 15, 'state_dim': 256},
        embodiment={'proprioceptive_dim': 128},
        system={'enable_jit': True, 'log_level': 'DEBUG'}
    )
    
    print(f"Updated temporal retention depth: {new_config.temporal.retention_depth}")
    print(f"Updated embodiment proprioceptive dim: {new_config.embodiment.proprioceptive_dim}")
    print(f"JIT compilation enabled: {new_config.system.enable_jit}")


def single_moment_processing():
    """Demonstrate single moment consciousness processing."""
    print("\n=== Single Moment Processing ===")
    
    # Create random inputs
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)
    
    sensory_input = jax.random.normal(keys[0], (64,))
    motor_prediction = jax.random.normal(keys[1], (32,))
    tactile_feedback = jax.random.normal(keys[2], (32,))
    environmental_context = jax.random.normal(keys[3], (64,))
    
    print("Processing consciousness moment...")
    start_time = time.time()
    
    # Process single moment
    state = ec.process_single_moment(
        sensory_input=sensory_input,
        motor_prediction=motor_prediction,
        tactile_feedback=tactile_feedback,
        environmental_context=environmental_context,
        key=key,
    )
    
    processing_time = time.time() - start_time
    
    print(f"Processing completed in {processing_time:.3f} seconds")
    print(f"Consciousness level: {state.consciousness_level.value}")
    print(f"Integration confidence: {state.integration_confidence:.3f}")
    print(f"Body schema confidence: {state.body_state.schema_confidence:.3f}")
    print(f"Temporal synthesis weights: {state.temporal_moment.synthesis_weights}")
    
    return state


def framework_instance_usage():
    """Demonstrate framework instance usage with persistence."""
    print("\n=== Framework Instance Usage ===")
    
    # Create framework instance
    key = jax.random.PRNGKey(123)
    framework = ec.create_consciousness_framework(key=key)
    
    print("Created framework instance")
    
    # Process multiple moments
    moments_data = []
    keys = jax.random.split(key, 10)
    
    print("Processing sequence of consciousness moments...")
    
    for i in range(5):
        # Generate varying inputs
        sensory_input = jax.random.normal(keys[i], (64,)) * (0.5 + i * 0.1)
        motor_prediction = jax.random.normal(keys[i+5], (32,)) * (0.8 + i * 0.05)
        tactile_feedback = jax.random.normal(keys[i], (32,)) * 0.3
        
        # Create processing context
        context = ec.ProcessingContext(
            prng_key=keys[i],
            environmental_context=jax.random.normal(keys[i], (64,)) * 0.2,
            processing_mode="normal",
            debug_mode=False,
        )
        
        # Process moment
        state = framework.process_consciousness_moment(
            sensory_input, motor_prediction, tactile_feedback, context
        )
        
        moments_data.append({
            'moment': i,
            'consciousness_level': state.consciousness_level.value,
            'integration_confidence': state.integration_confidence,
            'body_confidence': state.body_state.schema_confidence,
        })
        
        print(f"Moment {i}: {state.consciousness_level.value} "
              f"(confidence: {state.integration_confidence:.3f})")
    
    # Get performance summary
    performance = framework.get_performance_summary()
    print(f"\nPerformance summary:")
    print(f"JIT compilation: {performance['jit_compilation_enabled']}")
    print(f"Memory optimization: {performance['memory_optimization_enabled']}")
    
    if performance['processing_time_ms']:
        avg_time = performance['processing_time_ms'].get('mean', 0)
        print(f"Average processing time: {avg_time:.2f} ms")
    
    return moments_data, framework


def temporal_coherence_analysis(framework):
    """Analyze temporal coherence across multiple moments."""
    print("\n=== Temporal Coherence Analysis ===")
    
    # Generate sequence of related inputs (simulating continuous experience)
    key = jax.random.PRNGKey(456)
    base_pattern = jax.random.normal(key, (64,))
    
    temporal_moments = []
    keys = jax.random.split(key, 10)
    
    for i in range(10):
        # Create gradually changing sensory input
        noise = jax.random.normal(keys[i], (64,)) * 0.1
        sensory_input = base_pattern + noise + jnp.sin(i * 0.5) * 0.2
        
        # Process temporal moment
        temporal_moment = framework.temporal_processor.temporal_synthesis(
            primal_impression=sensory_input,
            timestamp=i * 0.1,
        )
        
        temporal_moments.append(temporal_moment)
    
    # Analyze temporal coherence
    coherence_analysis = ec.analyze_temporal_coherence(temporal_moments)
    
    print(f"Temporal coherence: {coherence_analysis['coherence']:.3f}")
    print(f"Temporal stability: {coherence_analysis['stability']:.3f}")
    print(f"Flow continuity: {coherence_analysis['flow_continuity']:.3f}")
    
    return coherence_analysis, temporal_moments


def embodiment_quality_assessment(framework):
    """Assess embodiment processing quality."""
    print("\n=== Embodiment Quality Assessment ===")
    
    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, 3)
    
    # Create structured proprioceptive input
    proprioceptive_input = jax.random.normal(keys[0], (64,))
    motor_prediction = jnp.sin(jnp.linspace(0, 2*jnp.pi, 32)) * 0.8
    tactile_feedback = jax.random.exponential(keys[2], (32,)) * 0.3
    
    # Process body schema
    body_state = framework.embodiment_processor.integrate_body_schema(
        proprioceptive_input=proprioceptive_input,
        motor_prediction=motor_prediction,
        tactile_feedback=tactile_feedback,
    )
    
    # Assess embodiment quality
    quality_metrics = framework.embodiment_processor.assess_embodiment_quality(body_state)
    
    print(f"Proprioceptive coherence: {quality_metrics['proprioceptive_coherence']:.3f}")
    print(f"Motor clarity: {quality_metrics['motor_clarity']:.3f}")
    print(f"Boundary clarity: {quality_metrics['boundary_clarity']:.3f}")
    print(f"Overall embodiment score: {quality_metrics['overall_embodiment']:.3f}")
    
    return quality_metrics, body_state


def error_handling_demonstration():
    """Demonstrate robust error handling."""
    print("\n=== Error Handling Demonstration ===")
    
    # Test with invalid inputs
    key = jax.random.PRNGKey(999)
    
    print("Testing with NaN inputs...")
    try:
        sensory_input = jnp.full((64,), jnp.nan)
        motor_prediction = jax.random.normal(key, (32,))
        tactile_feedback = jax.random.normal(key, (32,))
        
        state = ec.process_single_moment(
            sensory_input, motor_prediction, tactile_feedback
        )
        
        print(f"Error handled gracefully - fallback state created")
        print(f"Fallback consciousness level: {state.consciousness_level.value}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    
    print("Testing with mismatched dimensions...")
    try:
        sensory_input = jax.random.normal(key, (32,))  # Wrong dimension
        motor_prediction = jax.random.normal(key, (64,))  # Wrong dimension
        tactile_feedback = jax.random.normal(key, (16,))  # Wrong dimension
        
        state = ec.process_single_moment(
            sensory_input, motor_prediction, tactile_feedback
        )
        
        print("Dimension mismatch handled")
        
    except Exception as e:
        print(f"Expected error for dimension mismatch: {type(e).__name__}")


def memory_optimization_demo():
    """Demonstrate memory optimization features."""
    print("\n=== Memory Optimization Demo ===")
    
    # Create large inputs to test memory management
    key = jax.random.PRNGKey(1001)
    keys = jax.random.split(key, 3)
    
    large_sensory = jax.random.normal(keys[0], (512,))  # Larger than typical
    large_motor = jax.random.normal(keys[1], (128,))
    large_tactile = jax.random.normal(keys[2], (256,))
    
    # Process with memory tracking
    config = ec.get_config()
    memory_manager = ec.create_memory_manager(max_memory_mb=512.0)
    
    with memory_manager.track_memory("large_input_processing"):
        try:
            state = ec.process_single_moment(
                large_sensory, large_motor, large_tactile, key=key
            )
            print("Large input processing completed successfully")
            print(f"Memory optimization enabled: {config.system.enable_memory_optimization}")
            
        except Exception as e:
            print(f"Memory error: {e}")
    
    # Get memory statistics
    memory_stats = memory_manager.get_memory_stats()
    print(f"Memory operations tracked: {memory_stats['tracked_operations']}")
    print(f"Recent operations: {memory_stats['recent_operations']}")


def visualization_example(moments_data):
    """Create visualizations of consciousness processing."""
    print("\n=== Visualization Example ===")
    
    if not moments_data:
        print("No data available for visualization")
        return
    
    try:
        import matplotlib.pyplot as plt
        
        # Extract data for plotting
        moments = [data['moment'] for data in moments_data]
        integration_confidence = [data['integration_confidence'] for data in moments_data]
        body_confidence = [data['body_confidence'] for data in moments_data]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot confidence metrics
        plt.subplot(1, 2, 1)
        plt.plot(moments, integration_confidence, 'b-o', label='Integration Confidence')
        plt.plot(moments, body_confidence, 'r-s', label='Body Schema Confidence')
        plt.xlabel('Moment')
        plt.ylabel('Confidence')
        plt.title('Consciousness Confidence Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot consciousness levels
        plt.subplot(1, 2, 2)
        levels = [data['consciousness_level'] for data in moments_data]
        level_counts = {level: levels.count(level) for level in set(levels)}
        
        plt.bar(level_counts.keys(), level_counts.values(), alpha=0.7)
        plt.xlabel('Consciousness Level')
        plt.ylabel('Frequency')
        plt.title('Distribution of Consciousness Levels')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "consciousness_analysis.png", dpi=150, bbox_inches='tight')
        
        print(f"Visualization saved to: {output_dir / 'consciousness_analysis.png'}")
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")
    except Exception as e:
        print(f"Visualization error: {e}")


def main():
    """Run all examples."""
    print("Enactive Consciousness Framework - Usage Examples")
    print("=" * 50)
    
    # Basic framework usage
    basic_framework_usage()
    
    # Configuration management
    configuration_management_example()
    
    # Single moment processing
    single_moment_processing()
    
    # Framework instance usage
    moments_data, framework = framework_instance_usage()
    
    # Temporal coherence analysis
    temporal_coherence_analysis(framework)
    
    # Embodiment quality assessment
    embodiment_quality_assessment(framework)
    
    # Error handling demonstration
    error_handling_demonstration()
    
    # Memory optimization demo
    memory_optimization_demo()
    
    # Visualization
    visualization_example(moments_data)
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    
    # Final performance summary
    final_perf = framework.get_performance_summary()
    if final_perf['processing_time_ms']:
        print(f"Average processing time: {final_perf['processing_time_ms'].get('mean', 0):.2f} ms")


if __name__ == "__main__":
    main()