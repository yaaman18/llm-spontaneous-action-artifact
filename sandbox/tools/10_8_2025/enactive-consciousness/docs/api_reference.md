# API Reference

## Core Types and Protocols

### Type Aliases

```python
from enactive_consciousness import Array, ArrayLike, PRNGKey

# JAX array types
Array = jax.Array
ArrayLike = Union[Array, jnp.ndarray, List[float], Tuple[float, ...]]
PRNGKey = jax.random.PRNGKey
```

### Configuration

#### `FrameworkConfig`

Main configuration class for the enactive consciousness framework.

```python
from enactive_consciousness import create_framework_config

config = create_framework_config(
    retention_depth=10,          # int: Depth of temporal retention
    protention_horizon=5,        # int: Horizon of temporal protention
    consciousness_threshold=0.6, # float: Threshold for consciousness detection
    proprioceptive_dim=64,      # int: Proprioceptive input dimension
    motor_dim=32,               # int: Motor prediction dimension
)
```

**Parameters:**
- `retention_depth`: Number of past moments retained in temporal memory (1-100)
- `protention_horizon`: Number of future moments anticipated (1-50)  
- `primal_impression_width`: Width of present moment window (0.0-1.0)
- `consciousness_threshold`: Minimum threshold for conscious processing (0.0-1.0)
- `proprioceptive_dim`: Dimension of proprioceptive input (16-512)
- `motor_dim`: Dimension of motor predictions (8-256)

### Data Structures

#### `TemporalMoment`

Represents a moment in phenomenological time consciousness.

```python
from enactive_consciousness import create_temporal_moment

moment = create_temporal_moment(
    timestamp=0.0,
    retention=past_states,      # Array: Retained past moments
    present_moment=current,     # Array: Current moment synthesis  
    protention=future_states,   # Array: Anticipated future moments
    synthesis_weights=weights,  # Array: Integration weights [3,]
)
```

**Attributes:**
- `timestamp`: Time stamp of the moment
- `retention`: Synthesized representation of retained past
- `present_moment`: Current moment after temporal synthesis
- `protention`: Projected future expectations
- `synthesis_weights`: Relative weights of temporal components

#### `BodyState`

Represents current embodied state.

```python
body_state = processor.integrate_body_schema(
    proprioceptive_input=prop_input,
    motor_prediction=motor_pred,
    tactile_feedback=tactile_input,
)

# Access state components
proprioception = body_state.proprioception    # Array: Integrated proprioceptive state
motor_intention = body_state.motor_intention  # Array: Current motor intention
boundary_signal = body_state.boundary_signal  # Array: Body boundary information  
confidence = body_state.schema_confidence     # float: Schema integration confidence
```

## Temporal Consciousness

### `TemporalConsciousnessConfig`

Configuration for temporal consciousness processing.

```python
from enactive_consciousness import TemporalConsciousnessConfig

config = TemporalConsciousnessConfig(
    retention_depth=15,              # Number of retained moments
    protention_horizon=7,            # Protentional projection horizon
    primal_impression_width=0.1,     # Width of present moment
    temporal_synthesis_rate=0.05,    # Rate of temporal updating
    temporal_decay_factor=0.95,      # Decay factor for retention
)
```

### `PhenomenologicalTemporalSynthesis`

Main temporal consciousness processor implementing Husserl's time theory.

```python
from enactive_consciousness import create_temporal_processor
import jax

key = jax.random.PRNGKey(42)
processor = create_temporal_processor(
    config=temporal_config,
    state_dim=64,  # Dimension of temporal states
    key=key
)

# Process temporal moment
moment = processor.temporal_synthesis(
    primal_impression=sensory_input,     # Array: Current sensory input
    environmental_context=env_context,   # Optional[Array]: Environmental context
    timestamp=current_time,              # Optional[float]: Current timestamp
)
```

**Methods:**

#### `temporal_synthesis()`

Synthesize temporal moment from phenomenological components.

**Parameters:**
- `primal_impression` (Array): Current sensory impression
- `environmental_context` (Optional[Array]): Environmental context for protention
- `timestamp` (Optional[float]): Current time stamp

**Returns:** `TemporalMoment`

#### `update_temporal_flow()`

Update temporal flow with new impression.

**Parameters:**
- `current_moment` (TemporalMoment): Current temporal moment
- `new_impression` (Array): New sensory impression

**Returns:** `TemporalMoment`

### Utility Functions

#### `analyze_temporal_coherence()`

Analyze temporal coherence across sequence of moments.

```python
from enactive_consciousness import analyze_temporal_coherence

metrics = analyze_temporal_coherence(temporal_sequence)
# Returns: Dict[str, float] with coherence, stability, flow_continuity
```

## Embodied Cognition

### `BodySchemaConfig`

Configuration for body schema integration.

```python
from enactive_consciousness import BodySchemaConfig

config = BodySchemaConfig(
    proprioceptive_dim=48,               # Proprioceptive input dimension
    motor_dim=16,                        # Motor prediction dimension
    body_map_resolution=(15, 15),        # Spatial body map resolution
    boundary_sensitivity=0.7,            # Sensitivity to body boundaries
    schema_adaptation_rate=0.01,         # Rate of schema adaptation
    motor_intention_strength=0.5,        # Strength of motor intentions
)
```

### `BodySchemaIntegration`

Main body schema processor implementing Merleau-Ponty's embodied cognition.

```python
from enactive_consciousness import create_body_schema_processor

processor = create_body_schema_processor(config, key)

# Integrate body schema
body_state = processor.integrate_body_schema(
    proprioceptive_input=prop_input,     # Array: Proprioceptive signals
    motor_prediction=motor_pred,         # Array: Motor predictions
    tactile_feedback=tactile_input,      # Array: Tactile feedback
    previous_motor_state=prev_state,     # Optional[Array]: Previous motor state
)
```

**Methods:**

#### `integrate_body_schema()`

Integrate multi-modal body information into unified schema.

**Parameters:**
- `proprioceptive_input` (Array): Current proprioceptive signals
- `motor_prediction` (Array): Predicted motor commands
- `tactile_feedback` (Array): Tactile feedback signals
- `previous_motor_state` (Optional[Array]): Previous motor state for temporal continuity

**Returns:** `BodyState`

#### `generate_motor_intention()`

Generate motor intention from current and goal body states.

**Parameters:**
- `current_state` (BodyState): Current body state
- `goal_state` (Array): Desired goal state
- `intention_strength` (Optional[float]): Strength of intention scaling

**Returns:** `Array` - Generated motor intention

#### `assess_embodiment_quality()`

Assess quality of embodied processing.

**Returns:** `Dict[str, float]` with quality metrics

## Protocols and Interfaces

### `TemporalProcessor`

Protocol for temporal consciousness processing.

```python
from enactive_consciousness import TemporalProcessor
from typing import Protocol, runtime_checkable

@runtime_checkable
class TemporalProcessor(Protocol):
    def synthesize_temporal_moment(
        self,
        retention: Array,
        present_impression: Array,
        protention: Array,
    ) -> TemporalMoment: ...
    
    def update_temporal_flow(
        self,
        current_moment: TemporalMoment,
        new_impression: Array,
    ) -> TemporalMoment: ...
```

### `EmbodimentProcessor`

Protocol for embodied processing.

```python
from enactive_consciousness import EmbodimentProcessor

@runtime_checkable  
class EmbodimentProcessor(Protocol):
    def integrate_body_schema(
        self,
        proprioceptive_input: Array,
        motor_prediction: Array,
        tactile_feedback: Array,
    ) -> BodyState: ...
    
    def generate_motor_intention(
        self,
        current_state: BodyState,
        goal_state: Array,
    ) -> Array: ...
```

## Exception Handling

### Exception Hierarchy

```python
from enactive_consciousness import (
    EnactiveConsciousnessError,    # Base exception
    TemporalSynthesisError,        # Temporal processing errors
    EmbodimentError,               # Body schema errors
)

try:
    moment = processor.temporal_synthesis(invalid_input)
except TemporalSynthesisError as e:
    print(f"Temporal processing failed: {e}")
except EnactiveConsciousnessError as e:
    print(f"Framework error: {e}")
```

## Performance and Optimization

### JIT Compilation

All major processing functions support JAX JIT compilation for performance:

```python
import jax

# Functions are automatically JIT compiled
processor = create_temporal_processor(config, state_dim=64, key=key)
# First call compiles, subsequent calls use compiled version
```

### Memory Management

The framework includes intelligent memory management:

```python
# Memory usage is automatically optimized
# Use performance monitoring for tracking
from enactive_consciousness.types import PerformanceMetrics

metrics = PerformanceMetrics(
    temporal_coherence=0.85,
    embodiment_stability=0.92,
    processing_time_ms=2.5,
    memory_usage_mb=15.2,
)
```

## Examples

### Basic Usage

```python
import jax
import jax.numpy as jnp
from enactive_consciousness import (
    create_framework_config,
    create_temporal_processor,
    create_body_schema_processor,
    TemporalConsciousnessConfig,
    BodySchemaConfig,
)

# Setup
key = jax.random.PRNGKey(42)
config = create_framework_config()

# Temporal processing
temporal_config = TemporalConsciousnessConfig()
temporal_proc = create_temporal_processor(temporal_config, 64, key)

# Body schema processing  
body_config = BodySchemaConfig()
body_proc = create_body_schema_processor(body_config, key)

# Process inputs
sensory_input = jax.random.normal(key, (64,))
temporal_moment = temporal_proc.temporal_synthesis(sensory_input)

proprioceptive_input = jax.random.normal(key, (48,))
motor_prediction = jax.random.normal(key, (16,))
tactile_feedback = jax.random.normal(key, (24,))

body_state = body_proc.integrate_body_schema(
    proprioceptive_input, motor_prediction, tactile_feedback
)

print(f"Temporal moment synthesized at t={temporal_moment.timestamp}")
print(f"Body schema confidence: {body_state.schema_confidence:.3f}")
```

### Advanced Integration

See `examples/basic_demo.py` for comprehensive usage examples including:
- Sequential temporal processing
- Body schema adaptation
- Integrated temporal-embodied processing
- Performance monitoring
- Visualization

## Migration Guide

### From v0.0.x to v0.1.0

- Replace `TemporalProcessor` direct instantiation with `create_temporal_processor()`
- Update configuration using `create_framework_config()`
- Use new type-safe data structures (`TemporalMoment`, `BodyState`)
- Update exception handling to use specific exception types