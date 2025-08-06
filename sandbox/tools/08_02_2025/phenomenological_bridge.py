"""
Phenomenological Bridge for NewbornAI 2.0
Phase 2 of IIT 4.0 Integration

This module bridges between phenomenological concepts and computational representations,
implementing Maxwell Ramstead's computational phenomenology and embodied cognition principles.

Key Features:
- Phenomenological concept → computational representation mapping
- Embodied cognition principles in state transitions
- Qualitative experiential features → quantitative measures translation
- Phenomenological reduction for computational processing
- Active inference integration for predictive processing

Author: Maxwell Ramstead (Computational Phenomenology Lead)
Date: 2025-08-03
Version: 2.0.0
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from enum import Enum
import logging
import time
import math
from abc import ABC, abstractmethod
from collections import defaultdict

# Import framework components
from experiential_tpm_builder import ExperientialConcept, ExperientialConceptType, ExperientialTPMBuilder
from iit4_core_engine import IIT4PhiCalculator, PhiStructure
from iit4_experiential_phi_calculator import ExperientialPhiCalculator, ExperientialPhiType

logger = logging.getLogger(__name__)


class PhenomenologicalDimension(Enum):
    """
    Core phenomenological dimensions following Ramstead's computational phenomenology
    Based on Husserl, Merleau-Ponty, and enactive cognition principles
    """
    TEMPORALITY = "時間性"              # Temporal consciousness structure
    EMBODIMENT = "身体性"               # Embodied cognition and motor intentionality  
    INTENTIONALITY = "志向性"           # Phenomenological directedness toward objects
    SPATIALITY = "空間性"               # Lived space vs geometric space
    INTERSUBJECTIVITY = "間主観性"      # Shared experiential structures
    AFFECTIVITY = "情感性"              # Emotional-affective experience
    HORIZONALITY = "地平性"             # Background intentional horizons
    REDUCTION = "還元性"                # Phenomenological reduction capacity


@dataclass
class PhenomenologicalState:
    """
    Computational representation of phenomenological state
    Bridges qualitative experience to quantitative measures
    """
    state_id: str
    dimensions: Dict[PhenomenologicalDimension, float] = field(default_factory=dict)
    temporal_structure: Dict[str, float] = field(default_factory=dict)  # retention/impression/protention
    embodied_aspects: Dict[str, float] = field(default_factory=dict)   # sensorimotor components
    intentional_objects: List[str] = field(default_factory=list)       # objects of consciousness
    affective_tone: float = 0.0                                       # emotional coloring
    reduction_level: float = 0.0                                      # degree of phenomenological reduction
    consciousness_signature: Optional[float] = None                    # IIT 4.0 φ signature
    
    def __post_init__(self):
        """Initialize default phenomenological dimensions"""
        if not self.dimensions:
            for dim in PhenomenologicalDimension:
                self.dimensions[dim] = 0.5  # Neutral default
        
        if not self.temporal_structure:
            self.temporal_structure = {
                'retention': 0.3,
                'impression': 1.0,
                'protention': 0.4
            }


class PhenomenologicalReducer:
    """
    Implements phenomenological reduction for computational processing
    Following Husserl's epoché and Ramstead's computational adaptation
    """
    
    def __init__(self, reduction_threshold: float = 0.8):
        self.reduction_threshold = reduction_threshold
        self.natural_attitude_markers = [
            'objective', 'factual', 'scientific', 'logical', 'causal',
            'physical', 'material', 'external', 'independent'
        ]
        self.phenomenological_markers = [
            'experienced', 'lived', 'felt', 'appearing', 'given',
            'conscious', 'subjective', 'intentional', 'embodied'
        ]
    
    def reduce_to_experiential_core(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform phenomenological reduction on content
        Strips away natural attitude assumptions, preserves experiential core
        """
        reduced_content = {}
        
        for key, value in content.items():
            # Check if content has phenomenological character
            if self._has_phenomenological_character(key, value):
                reduced_content[key] = self._transform_to_experiential(value)
        
        return reduced_content
    
    def _has_phenomenological_character(self, key: str, value: Any) -> bool:
        """Check if content has genuine phenomenological character"""
        content_str = f"{key} {str(value)}".lower()
        
        # Count phenomenological vs natural attitude markers
        phenom_score = sum(1 for marker in self.phenomenological_markers if marker in content_str)
        natural_score = sum(1 for marker in self.natural_attitude_markers if marker in content_str)
        
        if phenom_score + natural_score == 0:
            return True  # Neutral content passes through
        
        return phenom_score > natural_score
    
    def _transform_to_experiential(self, value: Any) -> Any:
        """Transform value to experiential form"""
        if isinstance(value, str):
            # Transform linguistic descriptions to experiential qualities
            return self._linguistc_to_experiential(value)
        elif isinstance(value, (int, float)):
            # Quantitative measures remain but get experiential interpretation
            return float(value)
        elif isinstance(value, dict):
            # Recursively reduce nested structures
            return self.reduce_to_experiential_core(value)
        else:
            return value
    
    def _linguistc_to_experiential(self, text: str) -> str:
        """Convert linguistic descriptions to experiential qualities"""
        # Simple transformation rules
        transformations = {
            'objective': 'appearing',
            'fact': 'given',
            'real': 'experienced',
            'exists': 'presents itself',
            'cause': 'motivates',
            'effect': 'fulfillment'
        }
        
        transformed = text.lower()
        for objective_term, experiential_term in transformations.items():
            transformed = transformed.replace(objective_term, experiential_term)
        
        return transformed


class EmbodiedCognitionProcessor:
    """
    Processes embodied cognition aspects following Merleau-Ponty and Ramstead
    Implements sensorimotor integration and motor intentionality
    """
    
    def __init__(self):
        self.sensorimotor_dimensions = [
            'visual', 'auditory', 'tactile', 'proprioceptive', 'vestibular',
            'motor', 'kinesthetic', 'thermal', 'olfactory', 'gustatory'
        ]
        self.motor_schemas = {}
        self.embodied_memory = defaultdict(list)
    
    def process_embodied_state(self, concept: ExperientialConcept) -> Dict[str, float]:
        """
        Process embodied aspects of experiential concept
        Returns sensorimotor grounding profile
        """
        embodied_profile = {}
        
        # Extract sensorimotor components
        for dimension in self.sensorimotor_dimensions:
            embodied_profile[dimension] = self._extract_sensorimotor_component(concept, dimension)
        
        # Calculate motor intentionality
        embodied_profile['motor_intentionality'] = self._calculate_motor_intentionality(concept)
        
        # Assess embodied coherence
        embodied_profile['embodied_coherence'] = self._assess_embodied_coherence(embodied_profile)
        
        return embodied_profile
    
    def _extract_sensorimotor_component(self, concept: ExperientialConcept, dimension: str) -> float:
        """Extract specific sensorimotor component strength"""
        if concept.embodied_grounding and dimension in concept.embodied_grounding:
            return concept.embodied_grounding[dimension]
        
        # Infer from experiential content
        content_str = str(concept.experiential_content).lower()
        dimension_indicators = {
            'visual': ['see', 'look', 'color', 'bright', 'dark', 'shape'],
            'tactile': ['touch', 'feel', 'texture', 'pressure', 'soft', 'hard'],
            'motor': ['move', 'action', 'reach', 'grasp', 'walk', 'gesture'],
            'proprioceptive': ['position', 'posture', 'balance', 'orientation'],
            'thermal': ['warm', 'cold', 'hot', 'temperature']
        }
        
        if dimension in dimension_indicators:
            indicators = dimension_indicators[dimension]
            score = sum(1 for indicator in indicators if indicator in content_str)
            return min(score / len(indicators), 1.0)
        
        return 0.1  # Minimal baseline
    
    def _calculate_motor_intentionality(self, concept: ExperientialConcept) -> float:
        """
        Calculate motor intentionality strength
        Following Merleau-Ponty's motor intentionality theory
        """
        # Motor intentionality = directedness toward action possibilities
        motor_component = self._extract_sensorimotor_component(concept, 'motor')
        intentional_strength = concept.intentional_directedness
        
        # Action affordance detection
        action_affordances = self._detect_action_affordances(concept)
        
        return (motor_component * intentional_strength * action_affordances) ** (1/3)
    
    def _detect_action_affordances(self, concept: ExperientialConcept) -> float:
        """Detect action affordances in experiential concept"""
        content_str = str(concept.experiential_content).lower()
        
        affordance_markers = [
            'graspable', 'reachable', 'moveable', 'walkable', 'climbable',
            'touchable', 'usable', 'manipulable', 'accessible'
        ]
        
        affordance_score = sum(1 for marker in affordance_markers if marker in content_str)
        return min(affordance_score / 3.0, 1.0)  # Normalize to [0,1]
    
    def _assess_embodied_coherence(self, embodied_profile: Dict[str, float]) -> float:
        """Assess overall embodied coherence"""
        # Exclude meta-measures
        sensorimotor_values = [v for k, v in embodied_profile.items() 
                             if k in self.sensorimotor_dimensions]
        
        if not sensorimotor_values:
            return 0.0
        
        # Coherence = (mean - std) to penalize scattered activation
        mean_activation = np.mean(sensorimotor_values)
        std_activation = np.std(sensorimotor_values)
        
        coherence = max(mean_activation - 0.5 * std_activation, 0.0)
        return min(coherence, 1.0)


class QualitativeQuantitativeTranslator:
    """
    Translates qualitative experiential features to quantitative measures
    Implements Ramstead's approach to computational phenomenology
    """
    
    def __init__(self):
        self.quality_mappings = self._initialize_quality_mappings()
        self.intensity_scales = self._initialize_intensity_scales()
    
    def translate_qualitative_features(self, experiential_content: Dict[str, Any]) -> Dict[str, float]:
        """
        Translate qualitative experiential features to quantitative measures
        Preserves phenomenological character while enabling computation
        """
        quantitative_features = {}
        
        for feature, value in experiential_content.items():
            if isinstance(value, str):
                # Qualitative description → quantitative measure
                quantitative_features[feature] = self._map_quality_to_quantity(feature, value)
            elif isinstance(value, (int, float)):
                # Already quantitative
                quantitative_features[feature] = float(value)
            elif isinstance(value, dict):
                # Nested qualitative structure
                nested_quant = self.translate_qualitative_features(value)
                quantitative_features[feature] = np.mean(list(nested_quant.values()))
        
        return quantitative_features
    
    def _initialize_quality_mappings(self) -> Dict[str, Dict[str, float]]:
        """Initialize mappings from qualitative descriptions to quantitative values"""
        return {
            'intensity': {
                'barely_noticeable': 0.1, 'faint': 0.2, 'weak': 0.3, 'mild': 0.4,
                'moderate': 0.5, 'strong': 0.7, 'intense': 0.8, 'overwhelming': 0.9
            },
            'clarity': {
                'vague': 0.1, 'unclear': 0.2, 'fuzzy': 0.3, 'somewhat_clear': 0.5,
                'clear': 0.7, 'distinct': 0.8, 'crystal_clear': 0.9
            },
            'temporal_extent': {
                'momentary': 0.1, 'brief': 0.2, 'short': 0.3, 'moderate': 0.5,
                'extended': 0.7, 'long': 0.8, 'enduring': 0.9
            },
            'affective_valence': {
                'very_negative': -0.9, 'negative': -0.6, 'somewhat_negative': -0.3,
                'neutral': 0.0, 'somewhat_positive': 0.3, 'positive': 0.6, 'very_positive': 0.9
            }
        }
    
    def _initialize_intensity_scales(self) -> Dict[str, Tuple[float, float]]:
        """Initialize intensity scaling for different quality types"""
        return {
            'color': (0.0, 1.0),
            'sound': (0.0, 1.0),  
            'texture': (0.0, 1.0),
            'temperature': (-1.0, 1.0),  # cold to hot
            'pressure': (0.0, 1.0),
            'movement': (0.0, 1.0)
        }
    
    def _map_quality_to_quantity(self, feature: str, quality: str) -> float:
        """Map specific qualitative feature to quantitative value"""
        quality_lower = quality.lower().replace(' ', '_')
        
        # Try direct mapping first
        for category, mappings in self.quality_mappings.items():
            if quality_lower in mappings:
                return mappings[quality_lower]
        
        # Try fuzzy matching for compound qualities
        return self._fuzzy_quality_mapping(feature, quality_lower)
    
    def _fuzzy_quality_mapping(self, feature: str, quality: str) -> float:
        """Fuzzy mapping for complex qualitative descriptions"""
        # Intensity indicators
        intensity_indicators = {
            'very': 0.9, 'extremely': 0.95, 'somewhat': 0.4, 'slightly': 0.2,
            'barely': 0.1, 'quite': 0.7, 'rather': 0.6, 'fairly': 0.5
        }
        
        base_value = 0.5  # Default moderate intensity
        
        for indicator, multiplier in intensity_indicators.items():
            if indicator in quality:
                base_value = multiplier
                break
        
        # Adjust based on feature type
        if feature in self.intensity_scales:
            min_val, max_val = self.intensity_scales[feature]
            scaled_value = min_val + (max_val - min_val) * base_value
            return scaled_value
        
        return base_value


class PhenomenologicalBridge:
    """
    Main bridge class connecting phenomenological concepts to computational IIT 4.0
    Integrates all bridging components for seamless translation
    """
    
    def __init__(self):
        self.reducer = PhenomenologicalReducer()
        self.embodied_processor = EmbodiedCognitionProcessor()
        self.translator = QualitativeQuantitativeTranslator()
        self.tpm_builder = ExperientialTPMBuilder()
        
        # Bridge state
        self.phenomenological_states: Dict[str, PhenomenologicalState] = {}
        self.bridge_mappings: Dict[str, Dict[str, float]] = {}
        
        logger.info("PhenomenologicalBridge initialized with computational phenomenology components")
    
    async def bridge_experiential_concept(self, concept: ExperientialConcept) -> PhenomenologicalState:
        """
        Bridge experiential concept to computational phenomenological state
        Main bridging function integrating all translation components
        """
        try:
            # Step 1: Phenomenological reduction
            reduced_content = self.reducer.reduce_to_experiential_core(concept.experiential_content)
            
            # Step 2: Qualitative to quantitative translation
            quantitative_features = self.translator.translate_qualitative_features(reduced_content)
            
            # Step 3: Embodied cognition processing
            embodied_profile = self.embodied_processor.process_embodied_state(concept)
            
            # Step 4: Create phenomenological state
            phenom_state = PhenomenologicalState(
                state_id=f"phenom_{concept.concept_id}",
                temporal_structure=self._extract_temporal_structure(concept),
                embodied_aspects=embodied_profile,
                affective_tone=self._extract_affective_tone(quantitative_features),
                reduction_level=self._calculate_reduction_level(reduced_content, concept.experiential_content)
            )
            
            # Step 5: Map to phenomenological dimensions
            phenom_state.dimensions = self._map_to_phenomenological_dimensions(
                concept, quantitative_features, embodied_profile
            )
            
            # Step 6: Extract intentional objects
            phenom_state.intentional_objects = self._extract_intentional_objects(concept)
            
            # Store state
            self.phenomenological_states[phenom_state.state_id] = phenom_state
            
            logger.debug(f"Bridged concept {concept.concept_id} to phenomenological state")
            return phenom_state
            
        except Exception as e:
            logger.error(f"Error bridging experiential concept: {e}")
            raise
    
    def _extract_temporal_structure(self, concept: ExperientialConcept) -> Dict[str, float]:
        """Extract Husserlian temporal structure from concept"""
        temporal_structure = {
            'retention': concept.retention_trace or 0.3,
            'impression': 1.0,  # Current impression always maximal
            'protention': concept.protention_anticipation or 0.4
        }
        
        # Adjust based on concept type
        if concept.concept_type == ExperientialConceptType.TEMPORAL_RETENTION:
            temporal_structure['retention'] *= 1.5
        elif concept.concept_type == ExperientialConceptType.TEMPORAL_PROTENTION:
            temporal_structure['protention'] *= 1.5
        
        # Normalize to maintain temporal synthesis
        total = sum(temporal_structure.values())
        for key in temporal_structure:
            temporal_structure[key] /= total
        
        return temporal_structure
    
    def _extract_affective_tone(self, quantitative_features: Dict[str, float]) -> float:
        """Extract affective tone from quantitative features"""
        affective_indicators = ['valence', 'emotion', 'feeling', 'mood', 'affect']
        
        affective_values = []
        for feature, value in quantitative_features.items():
            if any(indicator in feature.lower() for indicator in affective_indicators):
                affective_values.append(value)
        
        if affective_values:
            return np.mean(affective_values)
        
        # Infer from overall intensity
        intensity_values = [v for k, v in quantitative_features.items() if 'intensity' in k.lower()]
        if intensity_values:
            return (np.mean(intensity_values) - 0.5) * 0.5  # Convert to affective tone
        
        return 0.0  # Neutral
    
    def _calculate_reduction_level(self, reduced_content: Dict[str, Any], original_content: Dict[str, Any]) -> float:
        """Calculate degree of phenomenological reduction achieved"""
        if not original_content:
            return 1.0
        
        reduction_ratio = len(reduced_content) / len(original_content)
        
        # Quality of reduction matters more than quantity
        natural_attitude_removed = self._count_natural_attitude_elements(original_content) - \
                                 self._count_natural_attitude_elements(reduced_content)
        
        quality_factor = min(natural_attitude_removed / max(len(original_content), 1), 1.0)
        
        return 0.7 * reduction_ratio + 0.3 * quality_factor
    
    def _count_natural_attitude_elements(self, content: Dict[str, Any]) -> int:
        """Count natural attitude elements in content"""
        content_str = str(content).lower()
        natural_markers = self.reducer.natural_attitude_markers
        
        return sum(1 for marker in natural_markers if marker in content_str)
    
    def _map_to_phenomenological_dimensions(self,
                                          concept: ExperientialConcept,
                                          quantitative_features: Dict[str, float],
                                          embodied_profile: Dict[str, float]) -> Dict[PhenomenologicalDimension, float]:
        """Map concept to phenomenological dimensions"""
        dimensions = {}
        
        # Temporality
        temporal_strength = (concept.retention_trace or 0.3) + (concept.protention_anticipation or 0.3)
        dimensions[PhenomenologicalDimension.TEMPORALITY] = min(temporal_strength, 1.0)
        
        # Embodiment
        dimensions[PhenomenologicalDimension.EMBODIMENT] = embodied_profile.get('embodied_coherence', 0.5)
        
        # Intentionality
        dimensions[PhenomenologicalDimension.INTENTIONALITY] = concept.intentional_directedness
        
        # Spatiality
        spatial_indicators = ['spatial', 'location', 'position', 'place', 'direction']
        spatial_score = sum(1 for feature in quantitative_features.keys() 
                          if any(indicator in feature.lower() for indicator in spatial_indicators))
        dimensions[PhenomenologicalDimension.SPATIALITY] = min(spatial_score / 3.0, 1.0)
        
        # Affectivity  
        affective_score = abs(self._extract_affective_tone(quantitative_features))
        dimensions[PhenomenologicalDimension.AFFECTIVITY] = affective_score
        
        # Intersubjectivity (inferred from intentional objects)
        intersubjective_objects = self._count_intersubjective_objects(concept)
        dimensions[PhenomenologicalDimension.INTERSUBJECTIVITY] = min(intersubjective_objects / 3.0, 1.0)
        
        # Horizonality (background intentional structure)
        horizon_complexity = len(quantitative_features) / 10.0  # More features = richer horizon
        dimensions[PhenomenologicalDimension.HORIZONALITY] = min(horizon_complexity, 1.0)
        
        # Reduction (calculated separately)
        reduction_level = self._calculate_reduction_level(
            concept.experiential_content, concept.experiential_content
        )
        dimensions[PhenomenologicalDimension.REDUCTION] = reduction_level
        
        return dimensions
    
    def _extract_intentional_objects(self, concept: ExperientialConcept) -> List[str]:
        """Extract intentional objects from experiential concept"""
        objects = []
        
        content_str = str(concept.experiential_content).lower()
        
        # Common intentional object indicators
        object_patterns = [
            'toward', 'about', 'of', 'concerning', 'regarding',
            'directed at', 'focused on', 'aimed at'
        ]
        
        # Extract objects based on linguistic patterns
        # Simplified extraction - in practice would use more sophisticated NLP
        for pattern in object_patterns:
            if pattern in content_str:
                # Extract what follows the pattern
                parts = content_str.split(pattern)
                if len(parts) > 1:
                    obj = parts[1].strip().split()[0]  # First word after pattern
                    if obj and len(obj) > 2:
                        objects.append(obj)
        
        return list(set(objects))  # Remove duplicates
    
    def _count_intersubjective_objects(self, concept: ExperientialConcept) -> int:
        """Count intersubjective objects in concept"""
        intersubjective_markers = [
            'other', 'person', 'face', 'voice', 'gesture', 'communication',
            'shared', 'together', 'mutual', 'empathy', 'understanding'
        ]
        
        content_str = str(concept.experiential_content).lower()
        return sum(1 for marker in intersubjective_markers if marker in content_str)
    
    async def synthesize_phenomenological_states(self, states: List[PhenomenologicalState]) -> PhenomenologicalState:
        """
        Synthesize multiple phenomenological states into unified state
        Following Husserlian temporal synthesis principles
        """
        if not states:
            raise ValueError("Cannot synthesize empty state list")
        
        if len(states) == 1:
            return states[0]
        
        # Create synthesized state
        synthesized = PhenomenologicalState(
            state_id=f"synthesized_{int(time.time())}",
            dimensions={},
            temporal_structure={},
            embodied_aspects={},
            intentional_objects=[],
            affective_tone=0.0,
            reduction_level=0.0
        )
        
        # Synthesize dimensions (weighted average)
        for dim in PhenomenologicalDimension:
            values = [state.dimensions.get(dim, 0.0) for state in states]
            synthesized.dimensions[dim] = np.mean(values)
        
        # Synthesize temporal structure
        for component in ['retention', 'impression', 'protention']:
            values = [state.temporal_structure.get(component, 0.0) for state in states]
            synthesized.temporal_structure[component] = np.mean(values)
        
        # Synthesize embodied aspects
        all_embodied_keys = set()
        for state in states:
            all_embodied_keys.update(state.embodied_aspects.keys())
        
        for key in all_embodied_keys:
            values = [state.embodied_aspects.get(key, 0.0) for state in states]
            synthesized.embodied_aspects[key] = np.mean(values)
        
        # Synthesize other properties
        synthesized.affective_tone = np.mean([state.affective_tone for state in states])
        synthesized.reduction_level = np.mean([state.reduction_level for state in states])
        
        # Combine intentional objects
        for state in states:
            synthesized.intentional_objects.extend(state.intentional_objects)
        synthesized.intentional_objects = list(set(synthesized.intentional_objects))  # Remove duplicates
        
        return synthesized
    
    async def compute_phenomenological_distance(self,
                                               state1: PhenomenologicalState,
                                               state2: PhenomenologicalState) -> float:
        """
        Compute phenomenological distance between states
        Used for clustering and similarity analysis
        """
        distances = []
        
        # Dimensional distance
        dim_distances = []
        for dim in PhenomenologicalDimension:
            val1 = state1.dimensions.get(dim, 0.0)
            val2 = state2.dimensions.get(dim, 0.0)
            dim_distances.append(abs(val1 - val2))
        distances.append(np.mean(dim_distances))
        
        # Temporal structure distance
        temporal_distances = []
        for component in ['retention', 'impression', 'protention']:
            val1 = state1.temporal_structure.get(component, 0.0)
            val2 = state2.temporal_structure.get(component, 0.0)
            temporal_distances.append(abs(val1 - val2))
        distances.append(np.mean(temporal_distances))
        
        # Embodied distance
        all_embodied_keys = set(state1.embodied_aspects.keys()) | set(state2.embodied_aspects.keys())
        embodied_distances = []
        for key in all_embodied_keys:
            val1 = state1.embodied_aspects.get(key, 0.0)
            val2 = state2.embodied_aspects.get(key, 0.0)
            embodied_distances.append(abs(val1 - val2))
        if embodied_distances:
            distances.append(np.mean(embodied_distances))
        
        # Affective distance
        distances.append(abs(state1.affective_tone - state2.affective_tone))
        
        return np.mean(distances)


# Example usage and testing
async def test_phenomenological_bridge():
    """Test the PhenomenologicalBridge with sample concepts"""
    bridge = PhenomenologicalBridge()
    
    # Create test concept
    concept = ExperientialConcept(
        concept_id="test_embodied_sensation",
        concept_type=ExperientialConceptType.EMBODIED_SENSATION,
        experiential_content={
            "quality": "warm_pressure",
            "intensity": "moderate", 
            "clarity": "clear",
            "affective_valence": "slightly_positive",
            "spatial_location": "hand_palm"
        },
        temporal_position=time.time(),
        embodied_grounding={
            "tactile": 0.8,
            "thermal": 0.6,
            "proprioceptive": 0.4
        },
        intentional_directedness=0.7
    )
    
    # Bridge to phenomenological state
    phenom_state = await bridge.bridge_experiential_concept(concept)
    
    print(f"Bridged concept to phenomenological state: {phenom_state.state_id}")
    print(f"Phenomenological dimensions: {phenom_state.dimensions}")
    print(f"Temporal structure: {phenom_state.temporal_structure}")
    print(f"Embodied aspects: {phenom_state.embodied_aspects}")
    print(f"Reduction level: {phenom_state.reduction_level:.3f}")


if __name__ == "__main__":
    asyncio.run(test_phenomenological_bridge())