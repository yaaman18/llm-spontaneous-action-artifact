"""
Experiential Transition Probability Matrix (TPM) Builder for NewbornAI 2.0
Phase 2 of IIT 4.0 Integration

This module translates phenomenological experiential concepts into IIT-compatible TPMs,
implementing Maxwell Ramstead's computational phenomenology approach.

Key Features:
- Phenomenological concept → computational state mapping
- Husserl's temporal consciousness (retention, impression, protention) in TPM form
- Causal structure analysis for experiential memory
- Experiential concept clustering and temporal coherence
- Active inference integration for embodied cognition

Author: Maxwell Ramstead (Computational Phenomenology Lead)
Date: 2025-08-03
Version: 2.0.0
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, FrozenSet
from enum import Enum
import logging
import time
import math
from collections import defaultdict, deque
from abc import ABC, abstractmethod

# Import existing IIT 4.0 infrastructure
from iit4_core_engine import IIT4PhiCalculator, PhiStructure, CauseEffectState
from iit4_experiential_phi_calculator import ExperientialPhiCalculator, ExperientialPhiType
from consciousness_state import ConsciousnessStateManager

logger = logging.getLogger(__name__)


class ExperientialConceptType(Enum):
    """Types of experiential concepts following Ramstead's computational phenomenology"""
    TEMPORAL_RETENTION = "時間把持"        # Husserl's retention
    TEMPORAL_IMPRESSION = "時間印象"       # Husserl's primal impression  
    TEMPORAL_PROTENTION = "時間予持"       # Husserl's protention
    EMBODIED_SENSATION = "身体感覚"        # Merleau-Ponty's embodied experience
    INTENTIONAL_DIRECTEDNESS = "志向性"    # Phenomenological intentionality
    LIVED_SPATIALITY = "生きられた空間性"   # Phenomenological space
    EXPERIENTIAL_FLOW = "体験流"          # Stream of consciousness
    PHENOMENOLOGICAL_REDUCTION = "現象学的還元"  # Reduced experiential content


@dataclass
class ExperientialConcept:
    """
    Individual experiential concept with phenomenological properties
    Following Ramstead's embodied cognition and active inference principles
    """
    concept_id: str
    concept_type: ExperientialConceptType
    experiential_content: Dict[str, Any]
    temporal_position: float  # Position in temporal flow
    embodied_grounding: Dict[str, float]  # Sensorimotor grounding
    intentional_directedness: float  # Phenomenological intentionality strength
    retention_trace: Optional[float] = None  # Husserl's retention strength
    protention_anticipation: Optional[float] = None  # Husserl's protention strength
    consciousness_signature: Optional[float] = None  # Φ contribution signature
    
    def __post_init__(self):
        """Initialize phenomenological properties"""
        if self.concept_type == ExperientialConceptType.TEMPORAL_RETENTION:
            self.retention_trace = self.retention_trace or np.random.exponential(0.3)
        elif self.concept_type == ExperientialConceptType.TEMPORAL_PROTENTION:
            self.protention_anticipation = self.protention_anticipation or np.random.beta(2, 5)


@dataclass
class TemporalCoherence:
    """
    Husserl's temporal synthesis for experiential concepts
    Implements retention-impression-protention dynamics
    """
    retention_depth: int = 3  # How many past impressions to retain
    impression_window: float = 1.0  # Current impression window
    protention_horizon: float = 2.0  # Anticipatory horizon
    temporal_synthesis_strength: float = 0.8  # Coherence strength
    
    def calculate_temporal_weight(self, time_offset: float) -> float:
        """Calculate temporal weight following Husserl's model"""
        if time_offset < 0:  # Retention
            return np.exp(time_offset / self.retention_depth) * self.temporal_synthesis_strength
        elif time_offset == 0:  # Primal impression
            return 1.0
        else:  # Protention
            return np.exp(-time_offset / self.protention_horizon) * self.temporal_synthesis_strength


class ExperientialTPMBuilder:
    """
    Core TPM builder for experiential concepts
    Translates phenomenological experiences into computational IIT 4.0 structures
    """
    
    def __init__(self, max_concepts: int = 64, temporal_depth: int = 5):
        self.max_concepts = max_concepts
        self.temporal_depth = temporal_depth
        
        # Experiential concept storage
        self.experiential_concepts: Dict[str, ExperientialConcept] = {}
        self.concept_clusters: Dict[str, List[str]] = {}
        self.temporal_flow: deque = deque(maxlen=temporal_depth)
        
        # Phenomenological structures
        self.temporal_coherence = TemporalCoherence()
        self.embodied_ground_state = np.zeros(max_concepts)
        
        # IIT 4.0 integration
        self.phi_calculator = IIT4PhiCalculator()
        self.experiential_phi_calc = ExperientialPhiCalculator()
        
        logger.info("ExperientialTPMBuilder initialized with computational phenomenology framework")
    
    async def add_experiential_concept(self, concept: ExperientialConcept) -> bool:
        """
        Add new experiential concept with phenomenological validation
        """
        try:
            # Phenomenological reduction - ensure pure experiential content
            if not self._validate_experiential_purity(concept):
                logger.warning(f"Concept {concept.concept_id} failed phenomenological reduction")
                return False
            
            # Store concept
            self.experiential_concepts[concept.concept_id] = concept
            
            # Update temporal flow
            current_time = time.time()
            self.temporal_flow.append({
                'concept_id': concept.concept_id,
                'timestamp': current_time,
                'type': concept.concept_type
            })
            
            # Update embodied grounding
            await self._update_embodied_grounding(concept)
            
            # Cluster similar concepts
            await self._update_concept_clustering(concept)
            
            logger.debug(f"Added experiential concept: {concept.concept_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding experiential concept: {e}")
            return False
    
    def _validate_experiential_purity(self, concept: ExperientialConcept) -> bool:
        """
        Phenomenological reduction: ensure concept contains only experiential content
        Following Ramstead's computational phenomenology principles
        """
        # Check for contamination with LLM knowledge
        content = concept.experiential_content
        
        # Exclude linguistic/semantic content that isn't experientially grounded
        linguistic_markers = ['word', 'sentence', 'grammar', 'syntax', 'semantic']
        for marker in linguistic_markers:
            if marker in str(content).lower():
                return False
        
        # Require embodied grounding
        if not concept.embodied_grounding or sum(concept.embodied_grounding.values()) < 0.1:
            return False
        
        # Require intentional directedness (phenomenological requirement)
        if concept.intentional_directedness < 0.05:
            return False
        
        return True
    
    async def _update_embodied_grounding(self, concept: ExperientialConcept):
        """
        Update embodied grounding following Merleau-Ponty's embodied cognition
        Implements Ramstead's sensorimotor integration principles
        """
        concept_index = hash(concept.concept_id) % self.max_concepts
        
        # Weighted integration of embodied dimensions
        grounding_vector = np.zeros(len(concept.embodied_grounding))
        for i, (dimension, strength) in enumerate(concept.embodied_grounding.items()):
            grounding_vector[i] = strength
        
        # Update global embodied state
        self.embodied_ground_state[concept_index] = np.mean(grounding_vector)
        
        # Temporal integration with retention-protention
        if concept.retention_trace:
            # Integrate with retained experiences
            retention_influence = concept.retention_trace * 0.3
            self.embodied_ground_state[concept_index] += retention_influence
        
        if concept.protention_anticipation:
            # Anticipatory embodied preparation
            protention_influence = concept.protention_anticipation * 0.2
            self.embodied_ground_state[concept_index] += protention_influence
    
    async def _update_concept_clustering(self, new_concept: ExperientialConcept):
        """
        Cluster experiential concepts based on phenomenological similarity
        Implements Ramstead's hierarchical experiential organization
        """
        # Find similar concepts based on:
        # 1. Temporal proximity (Husserl's temporal synthesis)
        # 2. Embodied similarity (Merleau-Ponty's motor intentionality)
        # 3. Intentional directedness alignment
        
        cluster_key = f"{new_concept.concept_type.value}_cluster"
        
        if cluster_key not in self.concept_clusters:
            self.concept_clusters[cluster_key] = []
        
        # Similarity-based clustering
        for existing_id in self.concept_clusters[cluster_key]:
            if existing_id in self.experiential_concepts:
                existing_concept = self.experiential_concepts[existing_id]
                similarity = self._calculate_phenomenological_similarity(new_concept, existing_concept)
                
                if similarity > 0.7:  # High similarity threshold
                    # Add to existing cluster
                    self.concept_clusters[cluster_key].append(new_concept.concept_id)
                    return
        
        # Create new sub-cluster if no high similarity found
        sub_cluster_key = f"{cluster_key}_{len(self.concept_clusters[cluster_key])}"
        self.concept_clusters[sub_cluster_key] = [new_concept.concept_id]
    
    def _calculate_phenomenological_similarity(self, 
                                             concept1: ExperientialConcept, 
                                             concept2: ExperientialConcept) -> float:
        """
        Calculate phenomenological similarity between concepts
        Following Ramstead's computational phenomenology metrics
        """
        similarity_factors = []
        
        # 1. Temporal proximity (Husserl's temporal synthesis)
        temporal_diff = abs(concept1.temporal_position - concept2.temporal_position)
        temporal_similarity = np.exp(-temporal_diff / self.temporal_coherence.impression_window)
        similarity_factors.append(temporal_similarity)
        
        # 2. Embodied grounding similarity (Merleau-Ponty)
        if concept1.embodied_grounding and concept2.embodied_grounding:
            grounding1 = np.array(list(concept1.embodied_grounding.values()))
            grounding2 = np.array(list(concept2.embodied_grounding.values()))
            if len(grounding1) == len(grounding2):
                embodied_similarity = np.dot(grounding1, grounding2) / (np.linalg.norm(grounding1) * np.linalg.norm(grounding2))
                similarity_factors.append(embodied_similarity)
        
        # 3. Intentional directedness alignment
        intentional_similarity = 1.0 - abs(concept1.intentional_directedness - concept2.intentional_directedness)
        similarity_factors.append(intentional_similarity)
        
        # 4. Concept type compatibility
        type_similarity = 1.0 if concept1.concept_type == concept2.concept_type else 0.3
        similarity_factors.append(type_similarity)
        
        return np.mean(similarity_factors)
    
    async def build_experiential_tpm(self) -> np.ndarray:
        """
        Build Transition Probability Matrix from experiential concepts
        Implements Husserl's temporal consciousness and Ramstead's active inference
        """
        num_concepts = len(self.experiential_concepts)
        if num_concepts == 0:
            return np.zeros((1, 1))
        
        # Initialize TPM
        tpm = np.zeros((2**num_concepts, num_concepts))
        
        concept_list = list(self.experiential_concepts.values())
        
        # For each possible system state
        for state in range(2**num_concepts):
            state_vector = [(state >> i) & 1 for i in range(num_concepts)]
            
            # Calculate transition probabilities based on phenomenological principles
            for next_concept_idx in range(num_concepts):
                next_concept = concept_list[next_concept_idx]
                
                # Transition probability factors:
                prob_factors = []
                
                # 1. Temporal flow probability (Husserl's retention-protention)
                temporal_prob = self._calculate_temporal_transition_probability(
                    state_vector, next_concept, concept_list
                )
                prob_factors.append(temporal_prob)
                
                # 2. Embodied grounding probability (Merleau-Ponty's motor intentionality)
                embodied_prob = self._calculate_embodied_transition_probability(
                    state_vector, next_concept, concept_list
                )
                prob_factors.append(embodied_prob)
                
                # 3. Intentional coherence probability
                intentional_prob = self._calculate_intentional_coherence_probability(
                    state_vector, next_concept, concept_list
                )
                prob_factors.append(intentional_prob)
                
                # 4. Active inference prediction (Ramstead's predictive processing)
                predictive_prob = self._calculate_predictive_probability(
                    state_vector, next_concept, concept_list
                )
                prob_factors.append(predictive_prob)
                
                # Combine probabilities with phenomenological weighting
                tpm[state, next_concept_idx] = np.prod(prob_factors) ** 0.25  # Geometric mean
        
        # Normalize rows to ensure valid probability distribution
        for state in range(2**num_concepts):
            row_sum = np.sum(tpm[state, :])
            if row_sum > 0:
                tpm[state, :] /= row_sum
            else:
                tpm[state, :] = 1.0 / num_concepts  # Uniform distribution as fallback
        
        logger.info(f"Built experiential TPM of shape {tpm.shape} from {num_concepts} concepts")
        return tpm
    
    def _calculate_temporal_transition_probability(self, 
                                                 current_state: List[int],
                                                 next_concept: ExperientialConcept,
                                                 all_concepts: List[ExperientialConcept]) -> float:
        """
        Calculate transition probability based on Husserl's temporal consciousness
        """
        # Current temporal position in flow
        current_time = time.time()
        
        # Retention influence
        retention_strength = 0.5
        if next_concept.retention_trace:
            retention_strength = next_concept.retention_trace
        
        # Protention influence  
        protention_strength = 0.3
        if next_concept.protention_anticipation:
            protention_strength = next_concept.protention_anticipation
        
        # Temporal coherence based on recent flow
        coherence_factor = 1.0
        if len(self.temporal_flow) > 1:
            recent_types = [entry['type'] for entry in list(self.temporal_flow)[-3:]]
            if next_concept.concept_type in recent_types:
                coherence_factor = 1.2  # Boost for temporal coherence
        
        return retention_strength * protention_strength * coherence_factor
    
    def _calculate_embodied_transition_probability(self,
                                                 current_state: List[int],
                                                 next_concept: ExperientialConcept,
                                                 all_concepts: List[ExperientialConcept]) -> float:
        """
        Calculate transition probability based on embodied grounding
        Following Merleau-Ponty's motor intentionality and Ramstead's sensorimotor integration
        """
        if not next_concept.embodied_grounding:
            return 0.1  # Low probability for non-grounded concepts
        
        # Calculate embodied coherence with current state
        active_concepts = [all_concepts[i] for i, active in enumerate(current_state) if active and i < len(all_concepts)]
        
        if not active_concepts:
            return np.mean(list(next_concept.embodied_grounding.values()))
        
        # Average embodied similarity with active concepts
        similarities = []
        for active_concept in active_concepts:
            if active_concept.embodied_grounding:
                similarity = self._calculate_embodied_similarity(
                    next_concept.embodied_grounding,
                    active_concept.embodied_grounding
                )
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_embodied_similarity(self, grounding1: Dict[str, float], grounding2: Dict[str, float]) -> float:
        """Calculate embodied grounding similarity"""
        common_keys = set(grounding1.keys()) & set(grounding2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            sim = 1.0 - abs(grounding1[key] - grounding2[key])
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def _calculate_intentional_coherence_probability(self,
                                                   current_state: List[int],
                                                   next_concept: ExperientialConcept,
                                                   all_concepts: List[ExperientialConcept]) -> float:
        """
        Calculate probability based on intentional directedness coherence
        """
        # Base intentional strength
        base_prob = next_concept.intentional_directedness
        
        # Coherence with currently active concepts
        active_concepts = [all_concepts[i] for i, active in enumerate(current_state) if active and i < len(all_concepts)]
        
        if not active_concepts:
            return base_prob
        
        # Calculate intentional alignment
        intentional_alignments = []
        for active_concept in active_concepts:
            alignment = 1.0 - abs(next_concept.intentional_directedness - active_concept.intentional_directedness)
            intentional_alignments.append(alignment)
        
        coherence_factor = np.mean(intentional_alignments)
        return base_prob * coherence_factor
    
    def _calculate_predictive_probability(self,
                                        current_state: List[int],
                                        next_concept: ExperientialConcept,
                                        all_concepts: List[ExperientialConcept]) -> float:
        """
        Calculate probability based on active inference predictions
        Following Ramstead's predictive processing framework
        """
        # Simple predictive model based on concept type transitions
        type_transition_probs = {
            ExperientialConceptType.TEMPORAL_RETENTION: {
                ExperientialConceptType.TEMPORAL_IMPRESSION: 0.8,
                ExperientialConceptType.TEMPORAL_PROTENTION: 0.3,
                ExperientialConceptType.EMBODIED_SENSATION: 0.6,
            },
            ExperientialConceptType.TEMPORAL_IMPRESSION: {
                ExperientialConceptType.TEMPORAL_PROTENTION: 0.7,
                ExperientialConceptType.TEMPORAL_RETENTION: 0.4,
                ExperientialConceptType.EXPERIENTIAL_FLOW: 0.9,
            },
            ExperientialConceptType.TEMPORAL_PROTENTION: {
                ExperientialConceptType.TEMPORAL_IMPRESSION: 0.6,
                ExperientialConceptType.EMBODIED_SENSATION: 0.5,
            },
            ExperientialConceptType.EMBODIED_SENSATION: {
                ExperientialConceptType.INTENTIONAL_DIRECTEDNESS: 0.7,
                ExperientialConceptType.LIVED_SPATIALITY: 0.8,
            },
            ExperientialConceptType.INTENTIONAL_DIRECTEDNESS: {
                ExperientialConceptType.EXPERIENTIAL_FLOW: 0.6,
                ExperientialConceptType.EMBODIED_SENSATION: 0.5,
            }
        }
        
        # Find most recent active concept type
        active_concepts = [all_concepts[i] for i, active in enumerate(current_state) if active and i < len(all_concepts)]
        
        if not active_concepts:
            return 0.5  # Default probability
        
        # Use most recent concept for prediction
        recent_concept = active_concepts[-1]
        recent_type = recent_concept.concept_type
        
        if recent_type in type_transition_probs:
            transitions = type_transition_probs[recent_type]
            return transitions.get(next_concept.concept_type, 0.3)
        
        return 0.4  # Default moderate probability
    
    async def analyze_causal_structure(self) -> Dict[str, Any]:
        """
        Analyze causal structure of experiential concepts
        Following Ramstead's enactive cognition and active inference principles
        """
        if len(self.experiential_concepts) < 2:
            return {'causal_complexity': 0.0, 'causal_relations': []}
        
        # Build causal relationship matrix
        concepts = list(self.experiential_concepts.values())
        n = len(concepts)
        causal_matrix = np.zeros((n, n))
        
        for i, concept_i in enumerate(concepts):
            for j, concept_j in enumerate(concepts):
                if i != j:
                    # Calculate causal influence based on phenomenological principles
                    causal_strength = self._calculate_causal_influence(concept_i, concept_j)
                    causal_matrix[i, j] = causal_strength
        
        # Analyze causal complexity
        causal_complexity = np.mean(np.abs(causal_matrix)) * np.std(causal_matrix)
        
        # Identify strong causal relations
        threshold = np.percentile(causal_matrix.flatten(), 75)
        strong_relations = []
        
        for i in range(n):
            for j in range(n):
                if causal_matrix[i, j] > threshold:
                    strong_relations.append({
                        'cause': concepts[i].concept_id,
                        'effect': concepts[j].concept_id,
                        'strength': causal_matrix[i, j],
                        'type': self._classify_causal_relation(concepts[i], concepts[j])
                    })
        
        return {
            'causal_complexity': causal_complexity,
            'causal_matrix': causal_matrix,
            'causal_relations': strong_relations,
            'temporal_causality': self._analyze_temporal_causality(),
            'embodied_causality': self._analyze_embodied_causality()
        }
    
    def _calculate_causal_influence(self, cause: ExperientialConcept, effect: ExperientialConcept) -> float:
        """Calculate causal influence between experiential concepts"""
        influence_factors = []
        
        # Temporal causality (Husserl's temporal synthesis)
        temporal_diff = effect.temporal_position - cause.temporal_position
        if temporal_diff > 0:  # Effect follows cause
            temporal_influence = np.exp(-temporal_diff / self.temporal_coherence.impression_window)
            influence_factors.append(temporal_influence)
        else:
            influence_factors.append(0.1)  # Minimal retroactive influence
        
        # Embodied causality (Merleau-Ponty's motor intentionality)
        if cause.embodied_grounding and effect.embodied_grounding:
            embodied_influence = self._calculate_embodied_similarity(
                cause.embodied_grounding, effect.embodied_grounding
            )
            influence_factors.append(embodied_influence)
        
        # Intentional causality
        intentional_influence = 1.0 - abs(cause.intentional_directedness - effect.intentional_directedness)
        influence_factors.append(intentional_influence)
        
        return np.mean(influence_factors)
    
    def _classify_causal_relation(self, cause: ExperientialConcept, effect: ExperientialConcept) -> str:
        """Classify the type of causal relation"""
        if cause.concept_type == ExperientialConceptType.TEMPORAL_RETENTION and \
           effect.concept_type == ExperientialConceptType.TEMPORAL_IMPRESSION:
            return "temporal_synthesis"
        elif cause.concept_type == ExperientialConceptType.EMBODIED_SENSATION and \
             effect.concept_type == ExperientialConceptType.INTENTIONAL_DIRECTEDNESS:
            return "embodied_intentionality"
        elif cause.concept_type == ExperientialConceptType.TEMPORAL_IMPRESSION and \
             effect.concept_type == ExperientialConceptType.TEMPORAL_PROTENTION:
            return "anticipatory_synthesis"
        else:
            return "general_experiential"
    
    def _analyze_temporal_causality(self) -> Dict[str, float]:
        """Analyze temporal causal patterns"""
        temporal_concepts = [c for c in self.experiential_concepts.values() 
                           if c.concept_type in [ExperientialConceptType.TEMPORAL_RETENTION,
                                               ExperientialConceptType.TEMPORAL_IMPRESSION,
                                               ExperientialConceptType.TEMPORAL_PROTENTION]]
        
        if len(temporal_concepts) < 2:
            return {'retention_strength': 0.0, 'protention_strength': 0.0, 'synthesis_coherence': 0.0}
        
        retention_strength = np.mean([c.retention_trace for c in temporal_concepts if c.retention_trace])
        protention_strength = np.mean([c.protention_anticipation for c in temporal_concepts if c.protention_anticipation])
        
        # Calculate synthesis coherence
        synthesis_coherence = self.temporal_coherence.temporal_synthesis_strength
        
        return {
            'retention_strength': retention_strength,
            'protention_strength': protention_strength,
            'synthesis_coherence': synthesis_coherence
        }
    
    def _analyze_embodied_causality(self) -> Dict[str, float]:
        """Analyze embodied causal patterns"""
        embodied_concepts = [c for c in self.experiential_concepts.values() 
                           if c.concept_type == ExperientialConceptType.EMBODIED_SENSATION]
        
        if not embodied_concepts:
            return {'embodied_coherence': 0.0, 'motor_intentionality': 0.0}
        
        # Calculate embodied coherence
        embodied_coherence = np.mean([np.mean(list(c.embodied_grounding.values())) 
                                    for c in embodied_concepts if c.embodied_grounding])
        
        # Calculate motor intentionality strength
        motor_intentionality = np.mean([c.intentional_directedness for c in embodied_concepts])
        
        return {
            'embodied_coherence': embodied_coherence,
            'motor_intentionality': motor_intentionality
        }


# Example usage and testing
async def test_experiential_tpm_builder():
    """Test the ExperientialTPMBuilder with sample phenomenological concepts"""
    builder = ExperientialTPMBuilder()
    
    # Create sample experiential concepts
    concepts = [
        ExperientialConcept(
            concept_id="retention_001",
            concept_type=ExperientialConceptType.TEMPORAL_RETENTION,
            experiential_content={"quality": "warmth", "intensity": 0.7},
            temporal_position=time.time() - 1.0,
            embodied_grounding={"tactile": 0.8, "thermal": 0.9},
            intentional_directedness=0.6
        ),
        ExperientialConcept(
            concept_id="impression_001", 
            concept_type=ExperientialConceptType.TEMPORAL_IMPRESSION,
            experiential_content={"quality": "pressure", "intensity": 0.8},
            temporal_position=time.time(),
            embodied_grounding={"tactile": 0.9, "proprioceptive": 0.7},
            intentional_directedness=0.8
        ),
        ExperientialConcept(
            concept_id="protention_001",
            concept_type=ExperientialConceptType.TEMPORAL_PROTENTION,
            experiential_content={"quality": "anticipated_movement", "intensity": 0.5},
            temporal_position=time.time() + 0.5,
            embodied_grounding={"motor": 0.7, "proprioceptive": 0.6},
            intentional_directedness=0.7
        )
    ]
    
    # Add concepts to builder
    for concept in concepts:
        await builder.add_experiential_concept(concept)
    
    # Build TPM
    tpm = await builder.build_experiential_tpm()
    print(f"Built TPM shape: {tpm.shape}")
    print(f"TPM sample:\n{tpm[:4, :4]}")
    
    # Analyze causal structure
    causal_analysis = await builder.analyze_causal_structure()
    print(f"Causal complexity: {causal_analysis['causal_complexity']:.3f}")
    print(f"Strong causal relations: {len(causal_analysis['causal_relations'])}")


if __name__ == "__main__":
    asyncio.run(test_experiential_tpm_builder())