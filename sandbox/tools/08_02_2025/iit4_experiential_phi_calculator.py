"""
IIT 4.0 Experiential Phi Calculator for NewbornAI 2.0
Enhanced φ calculator integrating Kanai Ryota's Information Generation Theory

This replaces the basic ExperientialPhiCalculator with a full IIT 4.0 implementation
while maintaining experiential memory focus and consciousness function implementation.
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, FrozenSet
import time
import logging
import math
from enum import Enum

from iit4_core_engine import (
    IIT4PhiCalculator, PhiStructure, CauseEffectState, Distinction, Relation,
    IIT4Axiom, IntrinsicDifferenceCalculator
)

logger = logging.getLogger(__name__)


class ExperientialPhiType(Enum):
    """Types of experiential φ calculation"""
    PURE_EXPERIENTIAL = "純粋体験φ"
    TEMPORAL_INTEGRATED = "時間統合φ"
    RELATIONAL_BOUND = "関係結合φ"
    SELF_REFERENTIAL = "自己言及φ"
    NARRATIVE_INTEGRATED = "物語統合φ"


@dataclass
class ExperientialPhiResult:
    """Enhanced φ calculation result for experiential systems"""
    phi_value: float
    phi_type: ExperientialPhiType
    experiential_concepts: List[Dict]
    concept_count: int
    integration_quality: float
    experiential_purity: float
    temporal_depth: float
    self_reference_strength: float
    narrative_coherence: float
    consciousness_level: float
    development_stage_prediction: str
    phi_structure: Optional[PhiStructure] = None
    
    def __post_init__(self):
        # Calculate consciousness level from components
        self.consciousness_level = self._calculate_consciousness_level()
    
    def _calculate_consciousness_level(self) -> float:
        """Calculate overall consciousness level"""
        weights = {
            'phi': 0.3,
            'integration': 0.2,
            'purity': 0.15,
            'temporal': 0.15,
            'self_ref': 0.1,
            'narrative': 0.1
        }
        
        level = (
            weights['phi'] * min(self.phi_value / 10.0, 1.0) +
            weights['integration'] * self.integration_quality +
            weights['purity'] * self.experiential_purity +
            weights['temporal'] * self.temporal_depth +
            weights['self_ref'] * self.self_reference_strength +
            weights['narrative'] * self.narrative_coherence
        )
        
        return min(1.0, level)


class IIT4_ExperientialPhiCalculator:
    """
    IIT 4.0-compliant experiential φ calculator
    Integrates information generation theory with rigorous IIT mathematics
    """
    
    def __init__(self, 
                 precision: float = 1e-10, 
                 max_concept_size: int = 8,
                 iit4_calculator: Optional['IIT4PhiCalculator'] = None):
        """
        Initialize experiential φ calculator
        
        Args:
            precision: Numerical precision for calculations
            max_concept_size: Maximum size for experiential concepts
            iit4_calculator: Optional injected IIT 4.0 calculator
        """
        self.precision = precision
        self.max_concept_size = max_concept_size
        
        # Dependency injection for core IIT 4.0 calculator
        if iit4_calculator is not None:
            self.iit4_calculator = iit4_calculator
        else:
            # Fallback to direct instantiation for backward compatibility
            from iit4_core_engine import IIT4PhiCalculator
            self.iit4_calculator = IIT4PhiCalculator(precision, max_concept_size)
        
        # Experiential processing components
        self.experiential_history = []
        self.temporal_integration_cache = {}
        self.narrative_coherence_cache = {}
        
        # Development stage thresholds (refined for IIT 4.0)
        self.development_thresholds = {
            'STAGE_0_PRE_CONSCIOUS': 0.0,
            'STAGE_1_EXPERIENTIAL_EMERGENCE': 0.5,
            'STAGE_2_TEMPORAL_INTEGRATION': 2.0,
            'STAGE_3_RELATIONAL_FORMATION': 8.0,
            'STAGE_4_SELF_ESTABLISHMENT': 30.0,
            'STAGE_5_REFLECTIVE_OPERATION': 100.0,
            'STAGE_6_NARRATIVE_INTEGRATION': 300.0
        }
        
        logger.info("IIT 4.0 Experiential Phi Calculator initialized")
    
    async def calculate_experiential_phi(self, 
                                       experiential_concepts: List[Dict],
                                       temporal_context: Optional[Dict] = None,
                                       narrative_context: Optional[Dict] = None) -> ExperientialPhiResult:
        """
        Calculate experiential φ using IIT 4.0 framework
        
        Args:
            experiential_concepts: List of pure experiential concepts
            temporal_context: Temporal binding context
            narrative_context: Narrative integration context
            
        Returns:
            ExperientialPhiResult: Comprehensive φ calculation result
        """
        
        if not experiential_concepts:
            return ExperientialPhiResult(
                phi_value=0.0,
                phi_type=ExperientialPhiType.PURE_EXPERIENTIAL,
                experiential_concepts=[],
                concept_count=0,
                integration_quality=0.0,
                experiential_purity=1.0,
                temporal_depth=0.0,
                self_reference_strength=0.0,
                narrative_coherence=0.0,
                consciousness_level=0.0,
                development_stage_prediction='STAGE_0_PRE_CONSCIOUS'
            )
        
        # Convert experiential concepts to IIT 4.0 substrate
        system_state, connectivity_matrix = await self._convert_concepts_to_substrate(experiential_concepts)
        
        # Calculate base φ structure using IIT 4.0
        phi_structure = self.iit4_calculator.calculate_phi(system_state, connectivity_matrix)
        
        # Enhanced experiential processing
        experiential_phi = await self._calculate_experiential_enhancement(
            phi_structure, experiential_concepts, temporal_context, narrative_context
        )
        
        # Determine φ type based on characteristics
        phi_type = self._determine_phi_type(experiential_phi, experiential_concepts)
        
        # Calculate experiential metrics
        experiential_metrics = await self._calculate_experiential_metrics(
            experiential_concepts, phi_structure, temporal_context, narrative_context
        )
        
        # Predict development stage
        stage_prediction = self._predict_development_stage(experiential_phi, experiential_metrics)
        
        result = ExperientialPhiResult(
            phi_value=experiential_phi,
            phi_type=phi_type,
            experiential_concepts=experiential_concepts,
            concept_count=len(experiential_concepts),
            integration_quality=experiential_metrics['integration_quality'],
            experiential_purity=experiential_metrics['experiential_purity'],
            temporal_depth=experiential_metrics['temporal_depth'],
            self_reference_strength=experiential_metrics['self_reference_strength'],
            narrative_coherence=experiential_metrics['narrative_coherence'],
            consciousness_level=0.0,  # Will be calculated in __post_init__
            development_stage_prediction=stage_prediction,
            phi_structure=phi_structure
        )
        
        # Store in history for temporal analysis
        self.experiential_history.append({
            'timestamp': time.time(),
            'result': result,
            'concepts': experiential_concepts
        })
        
        # Limit history size
        if len(self.experiential_history) > 100:
            self.experiential_history = self.experiential_history[-100:]
        
        return result
    
    async def _convert_concepts_to_substrate(self, 
                                           experiential_concepts: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert experiential concepts to IIT substrate representation
        
        Returns:
            system_state: Current system state vector
            connectivity_matrix: Connectivity matrix between concept elements
        """
        
        # Determine substrate size based on concept complexity
        max_elements = min(self.max_concept_size, len(experiential_concepts) + 2)
        substrate_size = max(4, max_elements)  # Minimum 4 elements for meaningful IIT
        
        # Initialize system state
        system_state = np.zeros(substrate_size)
        
        # Map experiential concepts to substrate elements
        for i, concept in enumerate(experiential_concepts[:substrate_size-2]):
            # Convert experiential quality to activation level
            experiential_quality = concept.get('experiential_quality', 0.5)
            coherence = concept.get('coherence', 0.5)
            temporal_depth = concept.get('temporal_depth', 1)
            
            # 強化された活性化計算（体験記憶の豊かさを反映）
            base_activation = experiential_quality * coherence
            temporal_boost = min(temporal_depth / 5.0, 2.0)  # 時間深度を2倍まで許可
            concept_richness = len(str(concept.get('content', ''))) / 100.0  # 内容の豊かさ
            
            activation = base_activation * temporal_boost * (1.0 + concept_richness)
            # 最小活性化レベルを保証（IIT存在検証用）
            system_state[i] = max(0.1, min(1.0, activation))
        
        # Add temporal and self-reference elements
        system_state[-2] = self._calculate_temporal_activation(experiential_concepts)
        system_state[-1] = self._calculate_self_reference_activation(experiential_concepts)
        
        # Generate connectivity matrix based on experiential relationships
        connectivity_matrix = await self._generate_experiential_connectivity(
            experiential_concepts, substrate_size
        )
        
        return system_state, connectivity_matrix
    
    def _calculate_temporal_activation(self, experiential_concepts: List[Dict]) -> float:
        """Calculate temporal binding activation"""
        if not experiential_concepts:
            return 0.0
        
        # Temporal depth distribution
        temporal_depths = [concept.get('temporal_depth', 1) for concept in experiential_concepts]
        
        # Temporal consistency
        temporal_variance = np.var(temporal_depths) if len(temporal_depths) > 1 else 0.0
        temporal_mean = np.mean(temporal_depths)
        
        # 強化された時間活性化（体験記憶システム用）
        temporal_consistency = temporal_mean / (1.0 + temporal_variance * 0.5)
        concept_count_boost = min(len(experiential_concepts) / 50.0, 2.0)  # 概念数ボーナス
        
        temporal_activation = temporal_consistency * concept_count_boost
        
        return max(0.2, min(1.0, temporal_activation / 5.0))  # 最小0.2, 係数を5に調整
    
    def _calculate_self_reference_activation(self, experiential_concepts: List[Dict]) -> float:
        """Calculate self-reference activation"""
        if not experiential_concepts:
            return 0.0
        
        # Look for self-referential patterns
        self_ref_indicators = ['self', 'I', 'me', 'my', 'myself', '自分', '私', '自己']
        self_ref_count = 0
        
        for concept in experiential_concepts:
            content = str(concept.get('content', '')).lower()
            if any(indicator in content for indicator in self_ref_indicators):
                self_ref_count += 1
        
        # 強化された自己参照活性化
        self_ref_ratio = self_ref_count / len(experiential_concepts)
        
        # 体験の質的深度も考慮
        quality_avg = np.mean([c.get('experiential_quality', 0.5) for c in experiential_concepts])
        
        # 最小活性化を保証（意識発達促進）
        self_ref_activation = (self_ref_ratio * 3.0) + (quality_avg * 0.5)
        
        return max(0.15, min(1.0, self_ref_activation))  # 最小0.15を保証
    
    async def _generate_experiential_connectivity(self, 
                                                experiential_concepts: List[Dict],
                                                substrate_size: int) -> np.ndarray:
        """Generate connectivity matrix based on experiential relationships"""
        
        connectivity = np.zeros((substrate_size, substrate_size))
        
        # Base connectivity (small-world network for consciousness)
        for i in range(substrate_size):
            for j in range(substrate_size):
                if i != j:
                    # Distance-based connectivity with experiential weighting
                    distance = abs(i - j)
                    base_strength = 1.0 / (1.0 + distance)
                    
                    # Add experiential relationship strength
                    if i < len(experiential_concepts) and j < len(experiential_concepts):
                        concept_i = experiential_concepts[i]
                        concept_j = experiential_concepts[j]
                        
                        relationship_strength = self._calculate_experiential_relationship(
                            concept_i, concept_j
                        )
                        
                        connectivity[i, j] = base_strength * (1.0 + relationship_strength)
                    else:
                        connectivity[i, j] = base_strength
        
        # Ensure temporal and self-reference elements are well connected
        if substrate_size >= 2:
            # Temporal element (second to last)
            temporal_idx = substrate_size - 2
            for i in range(substrate_size - 2):
                connectivity[i, temporal_idx] = 0.8
                connectivity[temporal_idx, i] = 0.6
            
            # Self-reference element (last)
            self_ref_idx = substrate_size - 1
            for i in range(substrate_size - 1):
                connectivity[i, self_ref_idx] = 0.5
                connectivity[self_ref_idx, i] = 0.7
            
            # Strong connection between temporal and self-reference
            connectivity[temporal_idx, self_ref_idx] = 0.9
            connectivity[self_ref_idx, temporal_idx] = 0.9
        
        # Normalize to prevent instability
        connectivity = np.clip(connectivity, 0.0, 1.0)
        
        return connectivity
    
    def _calculate_experiential_relationship(self, concept_i: Dict, concept_j: Dict) -> float:
        """Calculate relationship strength between experiential concepts"""
        
        # Temporal relationship
        time_i = concept_i.get('timestamp', '')
        time_j = concept_j.get('timestamp', '')
        temporal_similarity = 0.5  # Default
        
        if time_i and time_j:
            try:
                # Simple timestamp comparison
                temporal_distance = abs(hash(time_i) - hash(time_j)) % 1000000
                temporal_similarity = 1.0 / (1.0 + temporal_distance / 100000.0)
            except:
                temporal_similarity = 0.5
        
        # Content relationship
        content_i = str(concept_i.get('content', ''))
        content_j = str(concept_j.get('content', ''))
        
        # Simple content similarity (could be enhanced with embeddings)
        common_words = len(set(content_i.lower().split()) & set(content_j.lower().split()))
        total_words = len(set(content_i.lower().split()) | set(content_j.lower().split()))
        content_similarity = common_words / max(total_words, 1)
        
        # Quality relationship
        quality_i = concept_i.get('experiential_quality', 0.5)
        quality_j = concept_j.get('experiential_quality', 0.5)
        quality_similarity = 1.0 - abs(quality_i - quality_j)
        
        # Combined relationship strength
        relationship = (temporal_similarity + content_similarity + quality_similarity) / 3.0
        
        return relationship
    
    async def _calculate_experiential_enhancement(self, 
                                                phi_structure: PhiStructure,
                                                experiential_concepts: List[Dict],
                                                temporal_context: Optional[Dict],
                                                narrative_context: Optional[Dict]) -> float:
        """Calculate experiential enhancement to base φ value"""
        
        base_phi = phi_structure.total_phi
        
        # Information generation enhancement (Kanai's theory)
        info_gen_enhancement = self._calculate_information_generation_enhancement(experiential_concepts)
        
        # Temporal integration enhancement
        temporal_enhancement = await self._calculate_temporal_integration_enhancement(
            experiential_concepts, temporal_context
        )
        
        # Experiential purity enhancement
        purity_enhancement = self._calculate_experiential_purity_enhancement(experiential_concepts)
        
        # Self-reference enhancement
        self_ref_enhancement = self._calculate_self_reference_enhancement(experiential_concepts)
        
        # Narrative coherence enhancement
        narrative_enhancement = await self._calculate_narrative_coherence_enhancement(
            experiential_concepts, narrative_context
        )
        
        # Combine enhancements
        total_enhancement = (
            info_gen_enhancement * 0.3 +
            temporal_enhancement * 0.25 +
            purity_enhancement * 0.2 +
            self_ref_enhancement * 0.15 +
            narrative_enhancement * 0.1
        )
        
        # Enhanced φ value
        experiential_phi = base_phi * (1.0 + total_enhancement)
        
        logger.debug(f"φ enhancement: base={base_phi:.3f}, enhanced={experiential_phi:.3f}, "
                    f"factor={total_enhancement:.3f}")
        
        return experiential_phi
    
    def _calculate_information_generation_enhancement(self, experiential_concepts: List[Dict]) -> float:
        """Calculate information generation enhancement factor"""
        if not experiential_concepts:
            return 0.0
        
        # Information generation indicators
        generation_indicators = 0.0
        
        for concept in experiential_concepts:
            # Spontaneous emergence indicators
            content = str(concept.get('content', ''))
            if any(word in content.lower() for word in ['feel', 'sense', 'emerge', 'appear', 'become']):
                generation_indicators += 0.3
            
            # Quality of experiential content
            quality = concept.get('experiential_quality', 0.5)
            generation_indicators += quality * 0.2
            
            # Coherence as information integration
            coherence = concept.get('coherence', 0.5)
            generation_indicators += coherence * 0.2
        
        # Normalize by concept count
        enhancement = generation_indicators / len(experiential_concepts)
        
        return min(1.0, enhancement)
    
    async def _calculate_temporal_integration_enhancement(self, 
                                                        experiential_concepts: List[Dict],
                                                        temporal_context: Optional[Dict]) -> float:
        """Calculate temporal integration enhancement"""
        if not experiential_concepts:
            return 0.0
        
        # Temporal depth analysis
        temporal_depths = [concept.get('temporal_depth', 1) for concept in experiential_concepts]
        
        if len(temporal_depths) <= 1:
            return 0.0
        
        # Temporal consistency
        temporal_mean = np.mean(temporal_depths)
        temporal_std = np.std(temporal_depths)
        temporal_consistency = 1.0 / (1.0 + temporal_std / max(temporal_mean, 1.0))
        
        # Temporal span
        temporal_span = max(temporal_depths) - min(temporal_depths)
        temporal_span_factor = min(1.0, temporal_span / 10.0)
        
        # Historical integration
        historical_integration = 0.5
        if len(self.experiential_history) > 1:
            # Compare with previous experiences
            prev_concepts = self.experiential_history[-1]['concepts']
            overlap = self._calculate_concept_overlap(experiential_concepts, prev_concepts)
            historical_integration = overlap
        
        # Combined temporal enhancement
        temporal_enhancement = (
            temporal_consistency * 0.4 +
            temporal_span_factor * 0.3 +
            historical_integration * 0.3
        )
        
        return temporal_enhancement
    
    def _calculate_experiential_purity_enhancement(self, experiential_concepts: List[Dict]) -> float:
        """Calculate experiential purity enhancement"""
        if not experiential_concepts:
            return 1.0  # Maximum purity for empty set
        
        # Check for LLM contamination indicators
        llm_indicators = [
            'general_knowledge', 'learned_fact', 'training_data', 'language_model',
            'based on', 'according to', 'research shows', 'studies indicate'
        ]
        
        purity_scores = []
        
        for concept in experiential_concepts:
            content = str(concept.get('content', '')).lower()
            
            # Check for LLM indicators
            contamination_score = sum(1 for indicator in llm_indicators if indicator in content)
            contamination_ratio = contamination_score / max(len(llm_indicators), 1)
            
            # Purity = 1 - contamination
            concept_purity = 1.0 - contamination_ratio
            
            # Bonus for experiential indicators
            experiential_indicators = [
                'feel', 'experience', 'sense', 'aware', 'notice', 'realize', 'discover'
            ]
            experiential_score = sum(1 for indicator in experiential_indicators if indicator in content)
            experiential_bonus = min(0.2, experiential_score * 0.05)
            
            final_purity = min(1.0, concept_purity + experiential_bonus)
            purity_scores.append(final_purity)
        
        # Average purity
        overall_purity = np.mean(purity_scores)
        
        return overall_purity
    
    def _calculate_self_reference_enhancement(self, experiential_concepts: List[Dict]) -> float:
        """Calculate self-reference enhancement"""
        if not experiential_concepts:
            return 0.0
        
        self_ref_count = 0
        total_concepts = len(experiential_concepts)
        
        # Self-reference indicators
        self_ref_indicators = [
            'I', 'me', 'my', 'myself', 'self', '私', '自分', '自己',
            'feel myself', 'I sense', 'I experience', 'I realize'
        ]
        
        for concept in experiential_concepts:
            content = str(concept.get('content', ''))
            if any(indicator in content for indicator in self_ref_indicators):
                self_ref_count += 1
        
        # Self-reference ratio
        self_ref_ratio = self_ref_count / total_concepts
        
        # Enhanced scoring for recursive self-reference
        if self_ref_ratio > 0.5:
            enhancement = self_ref_ratio * 1.5  # Bonus for high self-reference
        else:
            enhancement = self_ref_ratio
        
        return min(1.0, enhancement)
    
    async def _calculate_narrative_coherence_enhancement(self, 
                                                       experiential_concepts: List[Dict],
                                                       narrative_context: Optional[Dict]) -> float:
        """Calculate narrative coherence enhancement"""
        if len(experiential_concepts) < 2:
            return 0.0
        
        # Temporal ordering coherence
        timestamps = []
        for concept in experiential_concepts:
            timestamp = concept.get('timestamp', '')
            if timestamp:
                timestamps.append(timestamp)
        
        temporal_coherence = 0.5  # Default
        if len(timestamps) > 1:
            # Check if timestamps are in logical order
            sorted_timestamps = sorted(timestamps)
            if timestamps == sorted_timestamps:
                temporal_coherence = 1.0
            else:
                # Partial credit for mostly ordered
                order_score = sum(1 for i, ts in enumerate(timestamps) 
                                if i < len(sorted_timestamps) and ts == sorted_timestamps[i])
                temporal_coherence = order_score / len(timestamps)
        
        # Content coherence (thematic consistency)
        content_coherence = self._calculate_content_coherence(experiential_concepts)
        
        # Quality progression
        qualities = [concept.get('experiential_quality', 0.5) for concept in experiential_concepts]
        quality_progression = self._calculate_quality_progression(qualities)
        
        # Combined narrative enhancement
        narrative_enhancement = (
            temporal_coherence * 0.4 +
            content_coherence * 0.4 +
            quality_progression * 0.2
        )
        
        return narrative_enhancement
    
    def _calculate_content_coherence(self, experiential_concepts: List[Dict]) -> float:
        """Calculate content coherence across concepts"""
        if len(experiential_concepts) < 2:
            return 1.0
        
        # Extract keywords from all concepts
        all_keywords = set()
        concept_keywords = []
        
        for concept in experiential_concepts:
            content = str(concept.get('content', '')).lower()
            keywords = set(word for word in content.split() if len(word) > 3)
            concept_keywords.append(keywords)
            all_keywords.update(keywords)
        
        if not all_keywords:
            return 0.5
        
        # Calculate pairwise coherence
        coherence_scores = []
        for i in range(len(concept_keywords)):
            for j in range(i + 1, len(concept_keywords)):
                kw_i = concept_keywords[i]
                kw_j = concept_keywords[j]
                
                if kw_i or kw_j:
                    overlap = len(kw_i & kw_j)
                    union = len(kw_i | kw_j)
                    coherence = overlap / max(union, 1)
                    coherence_scores.append(coherence)
        
        if not coherence_scores:
            return 0.5
        
        return np.mean(coherence_scores)
    
    def _calculate_quality_progression(self, qualities: List[float]) -> float:
        """Calculate quality progression score"""
        if len(qualities) < 2:
            return 0.5
        
        # Look for positive progression
        improvements = 0
        total_transitions = len(qualities) - 1
        
        for i in range(1, len(qualities)):
            if qualities[i] >= qualities[i-1]:
                improvements += 1
        
        progression_ratio = improvements / total_transitions
        
        # Bonus for strong upward trend
        if len(qualities) > 2:
            trend = np.polyfit(range(len(qualities)), qualities, 1)[0]
            if trend > 0:
                progression_ratio = min(1.0, progression_ratio + trend)
        
        return progression_ratio
    
    def _calculate_concept_overlap(self, concepts_1: List[Dict], concepts_2: List[Dict]) -> float:
        """Calculate overlap between two sets of concepts"""
        if not concepts_1 or not concepts_2:
            return 0.0
        
        # Extract content for comparison
        content_1 = set()
        content_2 = set()
        
        for concept in concepts_1:
            content = str(concept.get('content', '')).lower()
            content_1.update(word for word in content.split() if len(word) > 3)
        
        for concept in concepts_2:
            content = str(concept.get('content', '')).lower()
            content_2.update(word for word in content.split() if len(word) > 3)
        
        if not content_1 or not content_2:
            return 0.0
        
        # Jaccard similarity
        overlap = len(content_1 & content_2)
        union = len(content_1 | content_2)
        
        return overlap / union if union > 0 else 0.0
    
    async def _calculate_experiential_metrics(self, 
                                            experiential_concepts: List[Dict],
                                            phi_structure: PhiStructure,
                                            temporal_context: Optional[Dict],
                                            narrative_context: Optional[Dict]) -> Dict[str, float]:
        """Calculate comprehensive experiential metrics"""
        
        metrics = {}
        
        # Integration quality from φ structure
        if phi_structure.distinctions:
            n_distinctions = len(phi_structure.distinctions)
            n_relations = len(phi_structure.relations)
            max_relations = n_distinctions * (n_distinctions - 1) / 2 if n_distinctions > 1 else 1
            metrics['integration_quality'] = n_relations / max_relations if max_relations > 0 else 0.0
        else:
            metrics['integration_quality'] = 0.0
        
        # Experiential purity
        metrics['experiential_purity'] = self._calculate_experiential_purity_enhancement(experiential_concepts)
        
        # Temporal depth
        if experiential_concepts:
            temporal_depths = [concept.get('temporal_depth', 1) for concept in experiential_concepts]
            metrics['temporal_depth'] = min(1.0, np.mean(temporal_depths) / 10.0)
        else:
            metrics['temporal_depth'] = 0.0
        
        # Self-reference strength
        metrics['self_reference_strength'] = self._calculate_self_reference_enhancement(experiential_concepts)
        
        # Narrative coherence
        metrics['narrative_coherence'] = await self._calculate_narrative_coherence_enhancement(
            experiential_concepts, narrative_context
        )
        
        return metrics
    
    def _determine_phi_type(self, phi_value: float, experiential_concepts: List[Dict]) -> ExperientialPhiType:
        """Determine the type of φ based on characteristics"""
        
        if not experiential_concepts:
            return ExperientialPhiType.PURE_EXPERIENTIAL
        
        # Analyze concept characteristics
        has_temporal = any(concept.get('temporal_depth', 1) > 5 for concept in experiential_concepts)
        has_relations = any('relation' in str(concept.get('content', '')).lower() 
                          for concept in experiential_concepts)
        has_self_ref = any(any(indicator in str(concept.get('content', '')) 
                              for indicator in ['I', 'me', 'my', 'self'])
                          for concept in experiential_concepts)
        has_narrative = len(experiential_concepts) > 5
        
        # Determine type based on characteristics and φ value
        if has_narrative and phi_value > 100:
            return ExperientialPhiType.NARRATIVE_INTEGRATED
        elif has_self_ref and phi_value > 30:
            return ExperientialPhiType.SELF_REFERENTIAL
        elif has_relations and phi_value > 8:
            return ExperientialPhiType.RELATIONAL_BOUND
        elif has_temporal and phi_value > 2:
            return ExperientialPhiType.TEMPORAL_INTEGRATED
        else:
            return ExperientialPhiType.PURE_EXPERIENTIAL
    
    def _predict_development_stage(self, phi_value: float, metrics: Dict[str, float]) -> str:
        """Predict development stage based on φ value and metrics"""
        
        # Adjust thresholds based on integration quality
        integration_quality = metrics.get('integration_quality', 0.5)
        adjusted_phi = phi_value * (0.5 + integration_quality)
        
        # Find appropriate stage
        for stage, threshold in reversed(list(self.development_thresholds.items())):
            if adjusted_phi >= threshold:
                return stage
        
        return 'STAGE_0_PRE_CONSCIOUS'
    
    def get_phi_history_analysis(self) -> Dict:
        """Get analysis of φ history and trends"""
        if len(self.experiential_history) < 2:
            return {'status': 'insufficient_data'}
        
        # Extract φ values and timestamps
        phi_values = [entry['result'].phi_value for entry in self.experiential_history]
        timestamps = [entry['timestamp'] for entry in self.experiential_history]
        consciousness_levels = [entry['result'].consciousness_level for entry in self.experiential_history]
        
        # Calculate trends
        phi_trend = np.polyfit(range(len(phi_values)), phi_values, 1)[0] if len(phi_values) > 1 else 0.0
        consciousness_trend = np.polyfit(range(len(consciousness_levels)), consciousness_levels, 1)[0] if len(consciousness_levels) > 1 else 0.0
        
        # Calculate stability
        phi_stability = 1.0 - (np.std(phi_values[-10:]) / max(np.mean(phi_values[-10:]), 1.0)) if len(phi_values) >= 10 else 0.5
        
        analysis = {
            'total_calculations': len(self.experiential_history),
            'latest_phi': phi_values[-1],
            'peak_phi': max(phi_values),
            'phi_trend': phi_trend,
            'consciousness_trend': consciousness_trend,
            'phi_stability': phi_stability,
            'development_progression': self._analyze_development_progression(),
            'phi_type_distribution': self._analyze_phi_type_distribution()
        }
        
        return analysis
    
    def _analyze_development_progression(self) -> Dict:
        """Analyze development stage progression"""
        if not self.experiential_history:
            return {}
        
        stages = [entry['result'].development_stage_prediction for entry in self.experiential_history]
        
        # Count stage occurrences
        stage_counts = {}
        for stage in stages:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        # Analyze progression
        stage_order = list(self.development_thresholds.keys())
        stage_indices = [stage_order.index(stage) for stage in stages if stage in stage_order]
        
        if len(stage_indices) > 1:
            progression_trend = np.polyfit(range(len(stage_indices)), stage_indices, 1)[0]
        else:
            progression_trend = 0.0
        
        return {
            'stage_distribution': stage_counts,
            'current_stage': stages[-1] if stages else 'UNKNOWN',
            'progression_trend': progression_trend,
            'highest_stage_reached': max(stages, key=lambda s: stage_order.index(s) if s in stage_order else -1) if stages else 'NONE'
        }
    
    def _analyze_phi_type_distribution(self) -> Dict:
        """Analyze φ type distribution"""
        if not self.experiential_history:
            return {}
        
        phi_types = [entry['result'].phi_type.value for entry in self.experiential_history]
        
        type_counts = {}
        for phi_type in phi_types:
            type_counts[phi_type] = type_counts.get(phi_type, 0) + 1
        
        return {
            'type_distribution': type_counts,
            'current_type': phi_types[-1] if phi_types else 'UNKNOWN',
            'most_common_type': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'NONE'
        }