# Development Guide

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Make (optional, for convenience commands)

### Development Environment Setup

```bash
# Clone repository
git clone https://github.com/research/enactive-consciousness.git
cd enactive-consciousness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
make test
```

## Development Workflow

### Test-Driven Development (TDD)

The framework follows strict TDD practices:

#### 1. Red Phase - Write Failing Test

```python
# tests/unit/test_new_feature.py
def test_new_consciousness_feature():
    """Test new consciousness processing feature."""
    # Arrange
    processor = create_consciousness_processor()
    test_input = generate_test_data()
    
    # Act & Assert
    with pytest.raises(NotImplementedError):
        result = processor.process_consciousness(test_input)
```

#### 2. Green Phase - Make Test Pass

```python
# src/enactive_consciousness/consciousness.py
class ConsciousnessProcessor:
    def process_consciousness(self, input_data):
        # Minimal implementation to make test pass
        return process_basic_consciousness(input_data)
```

#### 3. Refactor Phase - Improve Code Quality

```python
# Refactored implementation
class ConsciousnessProcessor:
    def process_consciousness(self, input_data):
        validated_input = self._validate_input(input_data)
        processed_data = self._apply_consciousness_transform(validated_input)
        return self._format_output(processed_data)
    
    def _validate_input(self, input_data): ...
    def _apply_consciousness_transform(self, data): ...
    def _format_output(self, data): ...
```

### Development Commands

```bash
# Run tests in watch mode (automatically re-run on changes)
make test-watch

# Run specific test file
pytest tests/unit/test_temporal.py -v

# Run with coverage
make test-coverage

# Format code
make format

# Run all quality checks
make quality
```

## Code Organization

### Module Structure

```
src/enactive_consciousness/
├── __init__.py          # Public API exports
├── types.py            # Type definitions and protocols  
├── temporal.py         # Temporal consciousness implementation
├── embodiment.py       # Body schema and embodied cognition
├── coupling.py         # Structural coupling (future)
├── affordance.py       # Affordance perception (future)  
├── sense_making.py     # Sense-making processes (future)
├── core.py            # Base classes and utilities
├── config.py          # Configuration management
└── framework.py       # Main framework integration
```

### Naming Conventions

#### Classes
- **CamelCase** for class names: `TemporalProcessor`, `BodySchemaIntegration`
- **Descriptive names** reflecting theoretical concepts: `PhenomenologicalTemporalSynthesis`

#### Functions and Methods
- **snake_case** for functions: `create_temporal_processor`, `analyze_coherence`
- **Verbs** for actions: `integrate_body_schema`, `synthesize_temporal_moment`
- **Predicates** for boolean functions: `is_valid_state`, `has_consciousness_threshold`

#### Constants
- **UPPER_SNAKE_CASE**: `DEFAULT_RETENTION_DEPTH`, `MAX_PROTENTION_HORIZON`

#### Private Members
- **Leading underscore**: `_validate_input`, `_internal_state`

### Type Annotations

Comprehensive type annotations are required:

```python
from typing import Optional, List, Dict, Tuple, Protocol
import jax.numpy as jnp
from .types import Array, TemporalMoment, BodyState

def process_temporal_sequence(
    processor: TemporalProcessor,
    input_sequence: List[Array],
    initial_state: Optional[TemporalMoment] = None,
) -> Tuple[List[TemporalMoment], Dict[str, float]]:
    """Process sequence of temporal inputs.
    
    Args:
        processor: Temporal consciousness processor
        input_sequence: Sequence of sensory inputs
        initial_state: Optional initial temporal state
        
    Returns:
        Tuple of processed moments and performance metrics
        
    Raises:
        TemporalSynthesisError: If temporal processing fails
    """
    # Implementation...
```

## Testing Guidelines

### Test Organization

```
tests/
├── unit/               # Unit tests for individual components
│   ├── test_temporal.py
│   ├── test_embodiment.py
│   └── test_types.py
├── integration/        # Integration tests for component interaction
│   ├── test_temporal_embodied_integration.py
│   └── test_framework_integration.py
└── performance/        # Performance and benchmark tests
    ├── test_temporal_performance.py
    └── test_memory_usage.py
```

### Test Patterns

#### Given-When-Then Pattern

```python
def test_temporal_synthesis_with_valid_input():
    """Test temporal synthesis produces valid output."""
    # Given
    config = TemporalConsciousnessConfig(retention_depth=5)
    processor = create_temporal_processor(config, 32, PRNGKey(0))
    sensory_input = jnp.ones((32,))
    
    # When
    moment = processor.temporal_synthesis(sensory_input)
    
    # Then
    assert isinstance(moment, TemporalMoment)
    assert moment.present_moment.shape == (32,)
    assert moment.synthesis_weights.shape == (3,)
    assert jnp.allclose(jnp.sum(moment.synthesis_weights), 1.0)
```

#### Property-Based Testing

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp

@given(
    state_dim=st.integers(min_value=8, max_value=128),
    input_data=hnp.arrays(
        dtype=jnp.float32,
        shape=hnp.array_shapes(min_dims=1, max_dims=1),
        elements=st.floats(min_value=-10.0, max_value=10.0),
    )
)
def test_temporal_synthesis_properties(state_dim, input_data):
    """Test temporal synthesis satisfies properties for any valid input."""
    # Arrange
    processor = create_temporal_processor(
        TemporalConsciousnessConfig(),
        state_dim,
        PRNGKey(42)
    )
    
    # Act
    moment = processor.temporal_synthesis(input_data[:state_dim])
    
    # Assert properties
    assert jnp.all(jnp.isfinite(moment.present_moment))
    assert moment.synthesis_weights.sum() > 0.99  # Approximately normalized
```

#### Fixtures for Complex Setup

```python
import pytest

@pytest.fixture
def temporal_processor():
    """Create temporal processor for testing."""
    config = TemporalConsciousnessConfig(
        retention_depth=10,
        protention_horizon=5,
    )
    return create_temporal_processor(config, 64, PRNGKey(123))

@pytest.fixture
def sample_temporal_sequence():
    """Generate sample temporal sequence for testing."""
    return [
        jnp.sin(t * 0.1) * jnp.ones((64,)) + 
        jax.random.normal(PRNGKey(t), (64,)) * 0.1
        for t in range(20)
    ]

def test_temporal_coherence_analysis(temporal_processor, sample_temporal_sequence):
    """Test temporal coherence analysis with realistic data."""
    # Process sequence
    moments = []
    for input_data in sample_temporal_sequence:
        moment = temporal_processor.temporal_synthesis(input_data)
        moments.append(moment)
    
    # Analyze coherence
    metrics = analyze_temporal_coherence(moments)
    
    # Assert realistic coherence values
    assert 0.0 <= metrics['coherence'] <= 1.0
    assert metrics['stability'] > 0.5  # Should be reasonably stable
```

## Performance Optimization

### JAX Best Practices

#### JIT Compilation

```python
import jax

# Functions are automatically JIT compiled when possible
@jax.jit
def compute_temporal_synthesis(retention, present, protention):
    """JIT-compiled temporal synthesis computation."""
    return jnp.concatenate([retention, present, protention])

# For class methods, use functools.partial
import functools

@functools.partial(jax.jit, static_argnames=['self'])
def temporal_synthesis_method(self, input_data):
    """JIT-compiled method with static self."""
    return self._internal_computation(input_data)
```

#### Memory Management

```python
# Use JAX memory-efficient patterns
def process_large_sequence(sequence):
    """Process large sequence without memory accumulation."""
    
    def scan_fn(carry, x):
        state, processor = carry
        new_moment = processor.temporal_synthesis(x)
        new_state = update_state(state, new_moment)
        return (new_state, processor), new_moment
    
    initial_carry = (initial_state, processor)
    final_carry, moments = jax.lax.scan(scan_fn, initial_carry, sequence)
    return moments
```

### Profiling and Benchmarking

```python
# Use pytest-benchmark for performance tests
def test_temporal_synthesis_performance(benchmark, temporal_processor):
    """Benchmark temporal synthesis performance."""
    input_data = jnp.ones((64,))
    
    # Warm up JIT compilation
    _ = temporal_processor.temporal_synthesis(input_data)
    
    # Benchmark
    result = benchmark(temporal_processor.temporal_synthesis, input_data)
    assert isinstance(result, TemporalMoment)

# Profile memory usage
import tracemalloc

def profile_memory_usage():
    """Profile memory usage of framework components."""
    tracemalloc.start()
    
    # Run framework operations
    processor = create_temporal_processor(config, 128, PRNGKey(0))
    for _ in range(100):
        moment = processor.temporal_synthesis(jnp.ones((128,)))
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

## Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def integrate_consciousness_components(
    temporal_moment: TemporalMoment,
    body_state: BodyState,
    environmental_context: Array,
) -> ConsciousnessState:
    """Integrate temporal and embodied components into unified consciousness.
    
    This function implements the core integration process of the enactive
    consciousness framework, combining Husserlian temporal synthesis with
    Merleau-Ponty embodied cognition.
    
    Args:
        temporal_moment: Current temporal consciousness moment with
            retention-present-protention structure.
        body_state: Integrated body schema including proprioception,
            motor intention, and boundary information.
        environmental_context: Environmental context for coupling dynamics.
            
    Returns:
        Integrated consciousness state with unified temporal-embodied
        representation and confidence metrics.
        
    Raises:
        ConsciousnessIntegrationError: If integration fails due to
            incompatible component states or invalid configurations.
            
    Example:
        >>> config = create_framework_config()
        >>> temporal_proc = create_temporal_processor(config, 64, key)
        >>> body_proc = create_body_schema_processor(config, key)
        >>> 
        >>> moment = temporal_proc.temporal_synthesis(sensory_input)
        >>> body_state = body_proc.integrate_body_schema(prop_input, motor_pred, tactile)
        >>> 
        >>> consciousness = integrate_consciousness_components(
        ...     moment, body_state, env_context
        ... )
        >>> print(f"Consciousness level: {consciousness.level}")
    """
```

### Code Comments

#### When to Comment

- **Complex algorithms**: Explain theoretical background
- **Non-obvious optimizations**: Explain why code is written a certain way
- **Domain-specific concepts**: Link to theoretical sources

```python
def compute_retention_weights(self, depth: int) -> Array:
    """Compute exponentially decaying weights for retention memory.
    
    Based on Husserl's analysis of retention as 'running-off' of now-moments
    into retained consciousness. Recent moments have higher weight than
    distant ones, following phenomenological temporal structure.
    """
    # Exponential decay following φ(t) = e^(-λt) where λ controls decay rate
    # See: Husserl, "Phenomenology of Internal Time Consciousness" §11-15
    decay_rate = 0.1  # Empirically determined for optimal temporal coherence
    time_indices = jnp.arange(depth)
    weights = jnp.exp(-decay_rate * time_indices)
    
    # Normalize weights to sum to 1.0 for proper synthesis
    return weights / jnp.sum(weights)
```

#### What Not to Comment

```python
# Bad: Obvious operations
x = x + 1  # Increment x by 1

# Good: Non-obvious implications
x = x + 1  # Advance temporal index for next retention cycle
```

## Contributing Guidelines

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-consciousness-component
   ```

2. **Implement with TDD**
   - Write failing tests first
   - Implement minimal solution
   - Refactor for quality

3. **Ensure Quality**
   ```bash
   make quality  # Run all quality checks
   make test     # Run full test suite
   ```

4. **Create Pull Request**
   - Clear description of changes
   - Link to related issues
   - Include performance impact assessment

### Code Review Checklist

#### Functionality
- [ ] Tests pass and cover new functionality
- [ ] Code follows TDD principles
- [ ] Error handling is appropriate
- [ ] Edge cases are considered

#### Code Quality  
- [ ] Follows naming conventions
- [ ] Type annotations are complete
- [ ] Docstrings follow Google format
- [ ] No code duplication (DRY principle)

#### Performance
- [ ] JAX best practices followed
- [ ] Memory usage is reasonable
- [ ] JIT compilation considerations

#### Architecture
- [ ] Follows clean architecture principles
- [ ] Maintains separation of concerns
- [ ] Integrates well with existing components

### Issue Reporting

Use issue templates for consistent reporting:

#### Bug Report Template
```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Configure framework with...
2. Run temporal synthesis with...
3. Observe error...

**Expected Behavior**
Description of expected behavior

**Environment**
- Python version:
- JAX version:
- Framework version:

**Additional Context**
Any additional information
```

#### Feature Request Template
```markdown
**Feature Description**
Clear description of requested feature

**Theoretical Background**
Link to relevant theoretical foundations

**Use Cases**
Specific research or application scenarios

**Implementation Suggestions**
Initial ideas for implementation approach
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Test and Quality

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        
    - name: Run quality checks
      run: make quality
      
    - name: Run tests with coverage
      run: make test-coverage
      
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Release Process

### Semantic Versioning

- **Major (X.0.0)**: Breaking API changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible

### Release Checklist

1. [ ] Update version in `pyproject.toml`
2. [ ] Update `CHANGELOG.md`
3. [ ] Run full test suite
4. [ ] Update documentation
5. [ ] Create release PR
6. [ ] Tag release after merge
7. [ ] Build and publish to PyPI

---

This development guide ensures consistent, high-quality contributions to the Enactive Consciousness Framework. For questions or clarifications, please open an issue or discussion.