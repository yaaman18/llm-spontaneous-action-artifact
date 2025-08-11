"""Domain-Driven Design model for Enactive Consciousness.

This package implements Eric Evans' Domain-Driven Design methodology
for enactive consciousness, following phenomenological and enactivist theory
with precise ubiquitous language and proper bounded contexts.

Bounded Contexts:
- TemporalConsciousness: Husserlian phenomenological time consciousness
- EmbodiedConsciousness: Merleau-Pontian embodied cognition
- EnactiveCoupling: Varela-Maturana circular causality and autopoiesis
- ExperientialMemory: Experience sedimentation and associative recall

Aggregates:
- TemporalConsciousnessAggregate: Retention-Present-Protention synthesis
- EmbodiedExperienceAggregate: Body schema and motor intentionality
- CircularCausalityAggregate: Agent-environment structural coupling
- ExperientialMemoryAggregate: Traces, sedimentation, and recall

Strategic Design Principles:
1. Ubiquitous language matching phenomenological theory exactly
2. Rich domain objects with embedded business rules
3. Immutable value objects preserving theoretical integrity
4. Domain events capturing phenomenological significance
5. Repository abstractions following consciousness theory
6. Domain services for complex cross-aggregate operations
"""

from .value_objects import (
    # Temporal Consciousness Value Objects
    RetentionMoment,
    PrimalImpression, 
    ProtentionalHorizon,
    TemporalSynthesisWeights,
    
    # Embodied Consciousness Value Objects
    BodyBoundary,
    MotorIntention,
    ProprioceptiveField,
    TactileFeedback,
    
    # Enactive Coupling Value Objects
    CouplingStrength,
    AutopoeticProcess,
    StructuralCoupling,
    MeaningEmergence,
    
    # Experiential Memory Value Objects
    ExperientialTrace,
    SedimentLayer,
    RecallContext,
    AssociativeLink,
)

from .entities import (
    # Core Domain Entities
    TemporalMomentEntity,
    EmbodiedExperienceEntity,
    CircularCausalityEntity,
    ExperientialMemoryEntity,
)

from .aggregates import (
    # Domain Aggregates
    TemporalConsciousnessAggregate,
    EmbodiedExperienceAggregate,
    CircularCausalityAggregate,
    ExperientialMemoryAggregate,
)

from .domain_events import (
    # Temporal Consciousness Events
    TemporalSynthesisOccurred,
    RetentionUpdated,
    ProtentionProjected,
    PrimalImpressionFormed,
    
    # Embodied Experience Events
    BodySchemaReconfigured,
    MotorIntentionFormed,
    ProprioceptiveIntegrationCompleted,
    TactileFeedbackProcessed,
    
    # Circular Causality Events
    CircularCausalityCompleted,
    CouplingStrengthChanged,
    MeaningEmerged,
    AutopoeticCycleCompleted,
    
    # Experiential Memory Events
    ExperiencesSedimented,
    MemoryRecalled,
    AssociativeLinkFormed,
    SedimentLayerDeepened,
)

from .repositories import (
    # Domain Repositories
    RetentionMemoryRepository,
    ProtentionProjectionRepository,
    BodySchemaRepository,
    MotorSchemaRepository,
    CouplingHistoryRepository,
    AutopoeticStateRepository,
    ExperientialTraceRepository,
    SedimentLayerRepository,
)

from .domain_services import (
    # Domain Services
    TemporalSynthesisService,
    EmbodiedIntegrationService,
    CircularCausalityOrchestrator,
    ExperientialMemoryService,
)

from .specifications import (
    # Domain Specifications
    TemporalConsistencySpecification,
    EmbodiedCoherenceSpecification,
    CouplingStabilitySpecification,
    MemoryRetentionSpecification,
)

__all__ = [
    # Value Objects
    "RetentionMoment", "PrimalImpression", "ProtentionalHorizon", "TemporalSynthesisWeights",
    "BodyBoundary", "MotorIntention", "ProprioceptiveField", "TactileFeedback",
    "CouplingStrength", "AutopoeticProcess", "StructuralCoupling", "MeaningEmergence",
    "ExperientialTrace", "SedimentLayer", "RecallContext", "AssociativeLink",
    
    # Entities
    "TemporalMomentEntity", "EmbodiedExperienceEntity", "CircularCausalityEntity", "ExperientialMemoryEntity",
    
    # Aggregates
    "TemporalConsciousnessAggregate", "EmbodiedExperienceAggregate", 
    "CircularCausalityAggregate", "ExperientialMemoryAggregate",
    
    # Domain Events
    "TemporalSynthesisOccurred", "RetentionUpdated", "ProtentionProjected", "PrimalImpressionFormed",
    "BodySchemaReconfigured", "MotorIntentionFormed", "ProprioceptiveIntegrationCompleted", "TactileFeedbackProcessed",
    "CircularCausalityCompleted", "CouplingStrengthChanged", "MeaningEmerged", "AutopoeticCycleCompleted",
    "ExperiencesSedimented", "MemoryRecalled", "AssociativeLinkFormed", "SedimentLayerDeepened",
    
    # Repositories
    "RetentionMemoryRepository", "ProtentionProjectionRepository", "BodySchemaRepository", "MotorSchemaRepository",
    "CouplingHistoryRepository", "AutopoeticStateRepository", "ExperientialTraceRepository", "SedimentLayerRepository",
    
    # Domain Services
    "TemporalSynthesisService", "EmbodiedIntegrationService", 
    "CircularCausalityOrchestrator", "ExperientialMemoryService",
    
    # Specifications
    "TemporalConsistencySpecification", "EmbodiedCoherenceSpecification",
    "CouplingStabilitySpecification", "MemoryRetentionSpecification",
]