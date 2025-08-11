"""Enactive Consciousness Framework.

A comprehensive implementation of enactivism-based consciousness theory
with phenomenological foundations, featuring:

- Husserlian temporal consciousness (retention-present-protention)
- Merleau-Ponty body schema integration
- Varela-Maturana structural coupling
- Gibson affordance perception
- Enactive sense-making processes

Built with JAX/Equinox for high-performance computing and modern
Python type safety standards.
"""

__version__ = "0.1.0"
__author__ = "Enactivism Research Team"

# Core framework components
from .types import (
    # Type aliases and core types
    Array, ArrayLike, PRNGKey, PyTree,
    ConsciousnessLevel, CouplingStrength,
    FrameworkConfig,
    
    # Data structures
    TemporalMoment, BodyState, CouplingState,
    AffordanceVector, MeaningStructure,
    
    # Protocols
    TemporalProcessor, EmbodimentProcessor,
    ConsciousnessIntegrator,
    
    # Utilities
    create_framework_config, create_temporal_moment,
)

from .temporal import (
    TemporalConsciousnessConfig,
    PhenomenologicalTemporalSynthesis,
    create_temporal_processor,
    create_temporal_processor_safe,
    create_temporal_processor_no_jit,
    analyze_temporal_coherence,
)

from .embodiment import (
    BodySchemaConfig,
    BodySchemaIntegration,
    create_body_schema_processor,
    create_body_schema_processor_safe,
    create_body_schema_processor_no_jit,
)

from .dynamic_networks import (
    NetworkError,
    NetworkTopology,
    AdaptationMechanism,
    NetworkState,
    AdaptiveReorganizationResult,
    DynamicNetworkProcessor,
    NetworkIntegrator,
)

from .information_theory import (
    InformationTheoryError,
    mutual_information_kraskov,
    transfer_entropy,
    circular_causality_index,
    entropy_rate,
    integrated_information_phi,
    complexity_measure,
)

from .predictive_coding import (
    PredictionScale,
    PredictiveCodingConfig,
    PredictiveState,
    IntegratedPredictiveCoding,
    create_predictive_coding_system,
    optimize_hyperparameters,
)

# Convenience imports for common usage
from .types import EnactiveConsciousnessError

# Public API exports
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    
    # Core types and structures
    "Array", "ArrayLike", "PRNGKey",
    "ConsciousnessLevel", "CouplingStrength", 
    "FrameworkConfig",
    "TemporalMoment", "BodyState",
    "TemporalProcessor", "EmbodimentProcessor",
    
    # Main components
    "TemporalConsciousnessConfig",
    "PhenomenologicalTemporalSynthesis", 
    "BodySchemaConfig",
    "BodySchemaIntegration",
    
    # Dynamic networks
    "NetworkTopology",
    "AdaptationMechanism", 
    "NetworkState",
    "AdaptiveReorganizationResult",
    "DynamicNetworkProcessor",
    "NetworkIntegrator",
    
    # Information theory
    "mutual_information_kraskov",
    "transfer_entropy",
    "circular_causality_index",
    "entropy_rate", 
    "integrated_information_phi",
    "complexity_measure",
    
    # Predictive coding
    "PredictionScale",
    "PredictiveCodingConfig",
    "PredictiveState", 
    "IntegratedPredictiveCoding",
    "create_predictive_coding_system",
    "optimize_hyperparameters",
    
    # Factory functions
    "create_framework_config",
    "create_temporal_processor",
    "create_temporal_processor_safe",
    "create_temporal_processor_no_jit",
    "create_body_schema_processor",
    "create_body_schema_processor_safe",
    "create_body_schema_processor_no_jit",
    
    # Utilities
    "analyze_temporal_coherence",
    "create_temporal_moment",
    
    # Exceptions
    "EnactiveConsciousnessError",
    "NetworkError", 
    "InformationTheoryError",
]

# Framework metadata
FRAMEWORK_INFO = {
    "name": "Enactive Consciousness Framework",
    "version": __version__,
    "description": "Phenomenologically-grounded consciousness implementation",
    "theoretical_foundations": [
        "Husserlian phenomenology of time consciousness",
        "Merleau-Ponty embodied cognition", 
        "Varela-Maturana autopoiesis and structural coupling",
        "Gibson ecological psychology",
        "Enactive cognition theory",
    ],
    "computational_foundations": [
        "JAX automatic differentiation",
        "Equinox neural networks",
        "NGC-Learn predictive coding framework",
        "Multi-scale hierarchical prediction networks",
        "Dynamic error minimization with hyperparameter adaptation",
        "Modern Python type system",
        "Test-driven development",
        "Clean architecture patterns",
    ],
}

def get_framework_info() -> dict:
    """Get comprehensive framework information."""
    return FRAMEWORK_INFO.copy()


def quick_start_example():
    """Print a quick start example for the framework."""
    example = '''
# Enactive Consciousness Framework - Quick Start

import jax
import jax.numpy as jnp
from enactive_consciousness import (
    create_framework_config,
    create_temporal_processor,
    create_body_schema_processor,
    TemporalConsciousnessConfig,
    BodySchemaConfig,
)

# Create configuration
config = create_framework_config(
    retention_depth=10,
    protention_horizon=5,
    consciousness_threshold=0.6
)

# Initialize temporal processor
key = jax.random.PRNGKey(42)
temporal_config = TemporalConsciousnessConfig()
temporal_processor = create_temporal_processor(
    temporal_config, state_dim=64, key=key
)

# Process temporal moment
sensory_input = jax.random.normal(key, (64,))
temporal_moment = temporal_processor.temporal_synthesis(
    primal_impression=sensory_input,
    timestamp=0.0
)

print(f"Temporal synthesis complete!")
print(f"Moment timestamp: {temporal_moment.timestamp}")
print(f"Present moment shape: {temporal_moment.present_moment.shape}")
'''
    print(example)


if __name__ == "__main__":
    print(f"Enactive Consciousness Framework v{__version__}")
    print("=" * 50)
    quick_start_example()