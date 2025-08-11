# Domain-Driven Design Architecture Summary
## Enactive Consciousness Framework

### Overview

The enactive consciousness framework has been refined using Domain-Driven Design (DDD) principles to create a robust, maintainable, and expressive domain model that reflects the ubiquitous language of enactivism and consciousness research.

### Bounded Contexts

The domain is organized into four clear bounded contexts, each with its own ubiquitous language:

#### 1. Consciousness Context (`consciousness_context.py`)
**Ubiquitous Language:**
- **Consciousness Emergence**: Process by which consciousness arises from integrated information
- **Phi (Φ)**: Integrated information measure indicating consciousness level  
- **Awareness**: Metacognitive understanding of one's conscious states
- **Attention Focus**: Selective attention patterns shaping conscious experience
- **Phenomenological Markers**: Qualitative aspects of conscious experience

**Responsibilities:**
- Managing consciousness state transitions
- Enforcing consciousness emergence criteria
- Coordinating attention dynamics
- Tracking phenomenological aspects

#### 2. Learning Context (`learning_aggregate.py`)
**Ubiquitous Language:**
- **Environmental Coupling**: Strength of system-environment interaction
- **Predictive Coding**: Hierarchical prediction error minimization
- **Self-Organization**: Autonomous pattern formation in neural maps
- **Sensorimotor Contingencies**: Action-perception interaction patterns
- **Structural Coupling**: Dynamic adaptation to environmental changes

**Responsibilities:**
- Coordinating predictive coding and SOM learning
- Managing environmental coupling strength
- Adaptive learning rate policies
- Learning convergence detection

#### 3. Monitoring Context (Implied)
**Ubiquitous Language:**
- **Metacognitive Monitoring**: Self-awareness of cognitive processes
- **Confidence Assessment**: Evaluation of prediction reliability
- **Introspection**: Internal state examination processes

#### 4. Environmental Context (Implied)
**Ubiquitous Language:**
- **Enactive Interaction**: Active environmental engagement
- **Coupling Strength**: Degree of system-environment binding
- **Environmental Richness**: Complexity of environmental information

### Aggregate Roots

#### ConsciousnessAggregate (`consciousness_aggregate.py`)
**Consistency Boundary:** All consciousness state transitions and awareness processes

**Key Invariants:**
1. Consciousness emergence must satisfy emergence criteria
2. State transitions must maintain temporal coherence  
3. Stable consciousness requires prediction system stability
4. Attention weights must satisfy coherence requirements

**Domain Events Generated:**
- `ConsciousnessEmergenceDetected`
- `ConsciousnessFaded` 
- `AttentionFocusChanged`
- `MetacognitiveInsightGained`

#### LearningAggregate (`learning_aggregate.py`)
**Consistency Boundary:** All learning processes and environmental coupling

**Key Invariants:**
1. Prediction quality must meet minimum standards
2. Environmental coupling must be within valid range
3. Learning must show progress over time
4. Parameter adaptations must maintain learning stability

**Domain Events Generated:**
- `LearningEpochCompleted`
- `PredictionErrorThresholdCrossed`
- `SelfOrganizationConverged`
- `EnvironmentalCouplingStrengthened`

### Domain Events (`domain_events.py`)

All domain events extend the base `DomainEvent` class and represent significant business occurrences:

**Consciousness Events:**
- `ConsciousnessStateChanged`
- `ConsciousnessEmergenceDetected`
- `ConsciousnessFaded`
- `AttentionFocusChanged`
- `MetacognitiveInsightGained`

**Learning Events:**
- `LearningEpochCompleted`
- `PredictionErrorThresholdCrossed`
- `SelfOrganizationConverged`
- `EnvironmentalCouplingStrengthened`
- `AdaptiveLearningRateChanged`

### Specifications (`specifications/`)

Complex business rules encapsulated using the Specification pattern:

#### Consciousness Specifications
- **ConsciousnessEmergenceSpecification**: Defines emergence criteria based on Φ, metacognitive confidence, prediction quality, and environmental coupling
- **ConsciousnessStabilitySpecification**: Defines stability requirements for sustained consciousness
- **AttentionalCoherenceSpecification**: Defines coherent attention patterns

#### Learning Specifications  
- **LearningConvergenceSpecification**: Defines convergence criteria including error thresholds and environmental consistency
- **EnvironmentalCouplingSpecification**: Defines quality coupling based on enactivist principles
- **PredictionQualitySpecification**: Defines acceptable prediction standards

### Policies (`policies/`)

Decision-making logic encapsulated using the Policy pattern:

#### Consciousness Policies
- **ConsciousnessEmergencePolicy**: Regulates consciousness emergence and maintenance
- **AttentionRegulationPolicy**: Manages attention dynamics and focus patterns  
- **MetacognitiveMonitoringPolicy**: Handles metacognitive processes and self-awareness

#### Learning Policies
- **AdaptiveLearningRatePolicy**: Adjusts learning rates based on context
- **EnvironmentalCouplingPolicy**: Adapts learning to environmental coupling
- **PredictionErrorRegulationPolicy**: Regulates error dynamics for stability

### Factories (`factories/`)

Complex object creation encapsulated using the Factory pattern:

#### ConsciousnessFactory
- `create_consciousness_aggregate()`: Creates properly initialized consciousness systems
- `create_emergent_consciousness_state()`: Creates states for emergence scenarios
- `create_stable_consciousness_state()`: Creates stable consciousness configurations

### Key Enactivist Design Principles

1. **Environmental Coupling**: Consciousness emerges from dynamic interaction with environment
2. **Structural Coupling**: System adapts internal structure based on environmental interaction
3. **Sensorimotor Contingencies**: Learning occurs through action-perception loops
4. **Autonomous Systems**: Aggregates maintain their own consistency and identity
5. **Emergent Properties**: Consciousness emerges from complex system dynamics

### Usage Example

```python
# Initialize bounded context
consciousness_context = ConsciousnessContext()
factory = ConsciousnessFactory()

# Create consciousness aggregate using factory
aggregate = factory.create_consciousness_aggregate(
    system_complexity=12,
    environmental_richness=0.7,
    consciousness_potential=0.3
)

# Test consciousness emergence using specifications
emergence_spec = ConsciousnessEmergenceSpecification()
emergent_state = factory.create_emergent_consciousness_state(
    environmental_input=sensor_data,
    prediction_errors=[0.8, 0.6, 0.4],
    coupling_strength=0.65
)

meets_criteria = emergence_spec.is_satisfied_by(emergent_state)

# Apply policies for dynamic regulation
emergence_policy = ConsciousnessEmergencePolicy()
regulated_state = emergence_policy.apply_emergence_regulation(emergent_state)

# Generate and handle domain events
events = aggregate.clear_domain_events()
for event in events:
    if isinstance(event, ConsciousnessEmergenceDetected):
        handle_consciousness_emergence(event)
```

### Benefits of This DDD Architecture

1. **Clear Ubiquitous Language**: Enactivist concepts are consistently used throughout
2. **Bounded Contexts**: Clear separation of consciousness vs learning vs monitoring concerns
3. **Rich Domain Model**: Business logic is in the domain, not spread across layers
4. **Event-Driven**: Loose coupling between components via domain events
5. **Testable**: Specifications and policies can be tested independently
6. **Extensible**: New consciousness theories can be added via new specifications/policies
7. **Maintainable**: Changes to business rules are localized to specific components

### Future Extensions

- Additional bounded contexts for perception, action, and memory
- More sophisticated consciousness emergence specifications  
- Integration with actual sensorimotor systems
- Distributed consciousness across multiple aggregates
- Machine learning-based policy adaptation
- Real-time consciousness monitoring systems