"""
Domain-Driven Design Usage Example for Enactive Consciousness Framework.

This example demonstrates how to use the refined DDD architecture with
proper bounded contexts, aggregates, specifications, policies, and events.
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime

# Import domain components
from ..bounded_contexts.consciousness_context import ConsciousnessContext
from ..factories.consciousness_factory import ConsciousnessFactory
from ..aggregates.consciousness_aggregate import ConsciousnessAggregate
from ..aggregates.learning_aggregate import LearningAggregate
from ..specifications.consciousness_specifications import (
    ConsciousnessEmergenceSpecification,
    AttentionalCoherenceSpecification
)
from ..policies.consciousness_policies import (
    ConsciousnessEmergencePolicy,
    AttentionRegulationPolicy
)
from ..value_objects.phi_value import PhiValue
from ..value_objects.prediction_state import PredictionState
from ..value_objects.probability_distribution import ProbabilityDistribution


def demonstrate_enactive_consciousness_system():
    """
    Demonstrate the complete enactive consciousness system using DDD patterns.
    
    This example shows:
    1. Bounded Context usage
    2. Aggregate Root patterns
    3. Domain Events
    4. Specifications for business rules
    5. Policies for decision making
    6. Factory pattern for object creation
    7. Ubiquitous language in action
    """
    print("=== Enactive Consciousness Framework - DDD Demonstration ===\n")
    
    # Step 1: Initialize Bounded Contexts
    print("1. Initializing Bounded Contexts...")
    consciousness_context = ConsciousnessContext()
    consciousness_factory = ConsciousnessFactory()
    
    # Step 2: Create Domain Specifications (Business Rules)
    print("2. Setting up Domain Specifications (Business Rules)...")
    emergence_spec = ConsciousnessEmergenceSpecification(
        min_phi_value=0.2,
        min_metacognitive_confidence=0.15,
        max_prediction_error=1.5,
        require_environmental_coupling=True
    )
    
    attention_spec = AttentionalCoherenceSpecification(
        min_focus_strength=0.3,
        max_entropy_threshold=0.7
    )
    
    # Step 3: Create Domain Policies (Decision Making Logic)  
    print("3. Setting up Domain Policies...")
    emergence_policy = ConsciousnessEmergencePolicy(
        emergence_threshold=0.2,
        environmental_coupling_weight=0.7
    )
    
    attention_policy = AttentionRegulationPolicy(
        min_focus_threshold=0.25,
        environmental_bias_strength=0.5
    )
    
    # Step 4: Use Factory to Create Complex Domain Objects
    print("4. Creating Consciousness Aggregate using Factory Pattern...")
    consciousness_aggregate = consciousness_factory.create_consciousness_aggregate(
        system_complexity=12,
        environmental_richness=0.7,
        initial_coupling_strength=0.4,
        consciousness_potential=0.3
    )
    
    print(f"   Created aggregate: {consciousness_aggregate.aggregate_id}")
    print(f"   Initial consciousness: {consciousness_aggregate.is_conscious}")
    print(f"   Initial Φ value: {consciousness_aggregate.current_state.phi_value.value:.3f}")
    
    # Step 5: Simulate Environmental Interaction (Enactivist Core Principle)
    print("\n5. Simulating Environmental Interaction...")
    
    # Simulate rich environmental input
    environmental_input = np.random.randn(50) * 0.5 + np.sin(np.linspace(0, 4*np.pi, 50))
    prediction_errors = [0.8, 0.6, 0.4, 0.3]  # Hierarchical errors
    coupling_strength = 0.65
    
    print(f"   Environmental complexity: {np.var(environmental_input):.3f}")
    print(f"   Prediction errors: {prediction_errors}")
    print(f"   Environmental coupling strength: {coupling_strength}")
    
    # Step 6: Test Consciousness Emergence Using Specifications
    print("\n6. Testing Consciousness Emergence Criteria...")
    
    # Create emergent consciousness state
    emergent_state = consciousness_factory.create_emergent_consciousness_state(
        environmental_input=environmental_input,
        prediction_errors=prediction_errors,
        coupling_strength=coupling_strength,
        attention_context={
            'attention_dimensions': 5,
            'focus_areas': [0, 2],
            'environmental_salience': 0.7
        }
    )
    
    # Test emergence specification
    meets_emergence_criteria = emergence_spec.is_satisfied_by(emergent_state)
    emergence_score = emergence_spec.get_emergence_score(emergent_state)
    
    print(f"   Meets emergence criteria: {meets_emergence_criteria}")
    print(f"   Emergence score: {emergence_score:.3f}")
    print(f"   Emergent Φ value: {emergent_state.phi_value.value:.3f}")
    print(f"   Consciousness level: {emergent_state.consciousness_level:.3f}")
    
    # Step 7: Use Bounded Context for Consciousness Management
    print("\n7. Managing Consciousness through Bounded Context...")
    
    # Register aggregate in context
    aggregate_id = consciousness_context.create_consciousness_aggregate(
        initial_phi_value=emergent_state.phi_value,
        initial_prediction_state=emergent_state.prediction_state,
        initial_uncertainty=emergent_state.uncertainty_distribution
    )
    
    # Initiate consciousness emergence
    emergence_successful = consciousness_context.initiate_consciousness_emergence(
        aggregate_id=aggregate_id,
        phi_value=emergent_state.phi_value,
        prediction_state=emergent_state.prediction_state,
        uncertainty_distribution=emergent_state.uncertainty_distribution,
        environmental_context={
            'sensory_inputs': 8,
            'motor_activity': 3,
            'environmental_complexity': float(np.var(environmental_input)),
            'interaction_strength': coupling_strength
        }
    )
    
    print(f"   Consciousness emergence successful: {emergence_successful}")
    
    # Step 8: Apply Domain Policies for Dynamic Regulation
    print("\n8. Applying Domain Policies for Dynamic Regulation...")
    
    # Simulate environmental feedback
    environmental_feedback = {
        'coupling_strength': 0.75,
        'attention_demands': [0.4, 0.3, 0.2, 0.1],
        'metacognitive_triggers': {
            'self_awareness': True,
            'environmental_change_detected': 0.3,
            'prediction_confidence_shift': 0.15
        }
    }
    
    # Regulate consciousness dynamics
    regulated_state = consciousness_context.regulate_consciousness_dynamics(
        aggregate_id=aggregate_id,
        environmental_feedback=environmental_feedback
    )
    
    print(f"   Regulated consciousness level: {regulated_state.consciousness_level:.3f}")
    print(f"   Attention focus strength: {regulated_state.attention_focus_strength:.3f}")
    print(f"   Metacognitive confidence: {regulated_state.metacognitive_confidence:.3f}")
    
    # Step 9: Test Attention Coherence Specification
    print("\n9. Testing Attention Coherence...")
    if regulated_state.attention_weights is not None:
        attention_coherent = attention_spec.is_satisfied_by(regulated_state.attention_weights.tolist())
        coherence_score = attention_spec.calculate_coherence_score(regulated_state.attention_weights.tolist())
        
        print(f"   Attention is coherent: {attention_coherent}")
        print(f"   Coherence score: {coherence_score:.3f}")
    
    # Step 10: Monitor Consciousness Stability
    print("\n10. Monitoring Consciousness Stability...")
    stability_metrics = consciousness_context.monitor_consciousness_stability(aggregate_id)
    
    print("   Stability Metrics:")
    for metric, value in stability_metrics.items():
        print(f"     {metric}: {value}")
    
    # Step 11: Collect and Display Domain Events
    print("\n11. Collecting Domain Events...")
    context_events = consciousness_context.get_context_events()
    
    print(f"   Generated {len(context_events)} domain events:")
    for i, event in enumerate(context_events[-5:]):  # Show last 5 events
        print(f"     {i+1}. {event.event_type} at {event.timestamp}")
    
    # Step 12: Demonstrate Ubiquitous Language
    print("\n12. Ubiquitous Language in Action...")
    print("   Key Enactivist Concepts Demonstrated:")
    print("   - Environmental Coupling: System-environment interaction strength")
    print("   - Consciousness Emergence: Φ-based consciousness arising from coupling")
    print("   - Structural Coupling: Dynamic adaptation to environmental changes")
    print("   - Sensorimotor Contingencies: Action-perception loops in attention")
    print("   - Phenomenological Markers: Qualitative aspects of experience")
    
    # Step 13: Show Emergence History
    print("\n13. Consciousness Emergence History...")
    emergence_history = consciousness_context.get_consciousness_emergence_history()
    
    if emergence_history:
        latest_emergence = emergence_history[-1]
        print("   Latest Emergence Event:")
        print(f"     Φ value: {latest_emergence['phi_value']:.3f}")
        print(f"     Environmental coupling: {latest_emergence['environmental_coupling']:.3f}")
        print(f"     Success: {latest_emergence['emergence_successful']}")
        print(f"     Consciousness level: {latest_emergence['consciousness_level']}")
    
    print("\n=== DDD Demonstration Complete ===")
    print("\nKey DDD Patterns Demonstrated:")
    print("✓ Bounded Contexts - Consciousness domain separation")
    print("✓ Aggregate Roots - ConsciousnessAggregate with invariants")
    print("✓ Domain Events - Consciousness state changes")
    print("✓ Specifications - Complex business rules (emergence, coherence)")
    print("✓ Policies - Decision logic (emergence, attention regulation)")
    print("✓ Factories - Complex object creation")
    print("✓ Value Objects - Immutable domain concepts")
    print("✓ Ubiquitous Language - Enactivism terminology throughout")


def demonstrate_learning_aggregate():
    """
    Demonstrate Learning Aggregate with enactivist principles.
    """
    print("\n=== Learning Aggregate Demonstration ===")
    
    # Create learning aggregate
    learning_aggregate = LearningAggregate()
    
    # Simulate environmental learning episodes
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        
        # Generate environmental input with varying complexity
        environmental_complexity = 0.3 + episode * 0.1
        input_data = np.random.randn(10) * environmental_complexity
        
        # Environmental context varies over episodes
        environmental_context = {
            'episode': episode,
            'environmental_richness': environmental_complexity,
            'interaction_type': 'exploratory' if episode < 3 else 'exploitative'
        }
        
        # Perform learning epoch if predictive coding core exists
        try:
            if learning_aggregate._predictive_coding_core:  # If available
                prediction_state = learning_aggregate.perform_learning_epoch(
                    input_data=input_data,
                    environmental_context=environmental_context
                )
                
                print(f"   Prediction error: {prediction_state.total_error:.3f}")
                print(f"   Environmental coupling: {learning_aggregate.environmental_coupling_strength:.3f}")
            else:
                print("   Predictive coding core not initialized - skipping learning epoch")
        except Exception as e:
            print(f"   Learning epoch simulation: Environmental complexity = {environmental_complexity:.2f}")
        
        # Get learning metrics
        metrics = learning_aggregate.get_learning_metrics()
        print(f"   Learning metrics: {metrics}")
        
        # Clear events for next episode
        events = learning_aggregate.clear_domain_events()
        if events:
            print(f"   Generated {len(events)} learning events")


if __name__ == "__main__":
    # Run the complete demonstration
    demonstrate_enactive_consciousness_system()
    demonstrate_learning_aggregate()