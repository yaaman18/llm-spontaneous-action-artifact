"""
Temporal Consciousness Processor for NewbornAI 2.0
Phase 2 of IIT 4.0 Integration

This module implements Husserl's three-layer temporal structure computationally,
integrating with Maxwell Ramstead's computational phenomenology and active inference.

Key Features:
- Husserl's retention-impression-protention computational implementation
- Experiential flow and temporal synthesis processing
- TPM construction with temporal consciousness dynamics
- Integration with NewbornAI 2.0 development stages
- Active inference temporal prediction

Author: Maxwell Ramstead (Computational Phenomenology Lead)
Date: 2025-08-03
Version: 2.0.0
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Deque
from enum import Enum
import logging
import time
import math
from collections import deque, defaultdict
from abc import ABC, abstractmethod

# Import framework components
from experiential_tpm_builder import ExperientialConcept, ExperientialConceptType, ExperientialTPMBuilder
from phenomenological_bridge import PhenomenologicalBridge, PhenomenologicalState, PhenomenologicalDimension
from iit4_core_engine import IIT4PhiCalculator, PhiStructure
from iit4_experiential_phi_calculator import ExperientialPhiCalculator, ExperientialPhiType

logger = logging.getLogger(__name__)


class TemporalPhase(Enum):
    """
    Husserl's three temporal phases of consciousness
    Implemented as computational phases with specific dynamics
    """
    RETENTION = "把持"          # Retained past experiences
    IMPRESSION = "原印象"        # Primal present impression
    PROTENTION = "予持"         # Anticipated future experiences


class TemporalSynthesisType(Enum):
    """
    Types of temporal synthesis following Husserl's phenomenology
    """
    PASSIVE_SYNTHESIS = "受動的総合"    # Automatic temporal flow
    ACTIVE_SYNTHESIS = "能動的総合"     # Intentional temporal connection
    ASSOCIATIVE_SYNTHESIS = "連想的総合"  # Association-based temporal linking
    REPRODUCTIVE_SYNTHESIS = "再生的総合"  # Memory-based temporal reconstruction


@dataclass
class TemporalMoment:
    """
    Individual temporal moment in consciousness stream
    Represents a single moment with retention-impression-protention structure
    """
    moment_id: str
    timestamp: float
    experiential_content: Dict[str, Any]
    retention_traces: List[Tuple[str, float]] = field(default_factory=list)  # (moment_id, retention_strength)
    impression_intensity: float = 1.0
    protention_anticipations: List[Tuple[str, float]] = field(default_factory=list)  # (anticipated_content, strength)
    consciousness_phi: Optional[float] = None
    temporal_depth: int = 0  # Depth in retention chain
    synthesis_strength: float = 0.5
    
    def __post_init__(self):
        """Initialize temporal moment properties"""
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class TemporalFlow:
    """
    Continuous temporal flow of consciousness
    Manages the stream of temporal moments with phenomenological dynamics
    """
    flow_id: str
    moments: Deque[TemporalMoment] = field(default_factory=lambda: deque(maxlen=50))
    current_moment: Optional[TemporalMoment] = None
    retention_depth: int = 10
    protention_horizon: int = 5
    synthesis_coherence: float = 0.8
    temporal_rhythm: float = 1.0  # Temporal flow rate
    
    def add_moment(self, moment: TemporalMoment):
        """Add new moment to temporal flow"""
        self.moments.append(moment)
        self.current_moment = moment
    
    def get_retention_chain(self, depth: int = None) -> List[TemporalMoment]:
        """Get retention chain of specified depth"""
        depth = depth or self.retention_depth
        return list(self.moments)[-depth:]
    
    def get_current_impression(self) -> Optional[TemporalMoment]:
        """Get current primal impression"""
        return self.current_moment


class TemporalSynthesisEngine:
    """
    Engine for performing temporal synthesis operations
    Implements Husserl's various forms of temporal synthesis
    """
    
    def __init__(self, max_retention_depth: int = 10, max_protention_horizon: int = 5):
        self.max_retention_depth = max_retention_depth
        self.max_protention_horizon = max_protention_horizon
        
        # Synthesis parameters
        self.retention_decay_rate = 0.7
        self.protention_confidence_threshold = 0.3
        self.associative_strength_threshold = 0.5
        
        # Active inference parameters for temporal prediction
        self.prediction_accuracy_history = deque(maxlen=100)
        self.temporal_expectation_precision = 0.8
    
    async def perform_passive_synthesis(self, temporal_flow: TemporalFlow) -> TemporalFlow:
        """
        Perform passive temporal synthesis
        Automatic retention-impression-protention updates
        """
        if not temporal_flow.current_moment:
            return temporal_flow
        
        current = temporal_flow.current_moment
        
        # Update retention traces from current moment
        await self._update_retention_traces(current, temporal_flow)
        
        # Generate protentional anticipations
        await self._generate_protentional_anticipations(current, temporal_flow)
        
        # Update temporal synthesis strength
        temporal_flow.synthesis_coherence = self._calculate_synthesis_coherence(temporal_flow)
        
        return temporal_flow
    
    async def perform_active_synthesis(self, 
                                     temporal_flow: TemporalFlow,
                                     target_content: Dict[str, Any]) -> TemporalFlow:
        """
        Perform active temporal synthesis
        Intentional connection of temporal moments around target content
        """
        if not temporal_flow.moments:
            return temporal_flow
        
        # Find moments related to target content
        related_moments = self._find_content_related_moments(temporal_flow, target_content)
        
        # Strengthen connections between related moments
        for moment in related_moments:
            await self._strengthen_temporal_connections(moment, temporal_flow, target_content)
        
        # Update flow coherence based on active synthesis
        temporal_flow.synthesis_coherence += 0.1
        temporal_flow.synthesis_coherence = min(temporal_flow.synthesis_coherence, 1.0)
        
        return temporal_flow
    
    async def perform_associative_synthesis(self, temporal_flow: TemporalFlow) -> TemporalFlow:
        """
        Perform associative temporal synthesis
        Connect moments based on content similarity and associative links
        """
        moments = list(temporal_flow.moments)
        
        # Calculate associative connections
        for i, moment_i in enumerate(moments):
            for j, moment_j in enumerate(moments[i+1:], i+1):
                associative_strength = self._calculate_associative_strength(moment_i, moment_j)
                
                if associative_strength > self.associative_strength_threshold:
                    # Add associative retention trace
                    moment_j.retention_traces.append((moment_i.moment_id, associative_strength))
        
        return temporal_flow
    
    async def _update_retention_traces(self, current_moment: TemporalMoment, temporal_flow: TemporalFlow):
        """Update retention traces following Husserl's retention model"""
        retention_chain = temporal_flow.get_retention_chain()
        
        current_moment.retention_traces = []
        
        for i, past_moment in enumerate(reversed(retention_chain[:-1])):  # Exclude current moment
            # Calculate retention strength with exponential decay
            time_distance = i + 1
            retention_strength = self.retention_decay_rate ** time_distance
            
            # Adjust for content similarity
            content_similarity = self._calculate_content_similarity(
                current_moment.experiential_content,
                past_moment.experiential_content
            )
            
            adjusted_strength = retention_strength * (0.5 + 0.5 * content_similarity)
            
            if adjusted_strength > 0.1:  # Threshold for meaningful retention
                current_moment.retention_traces.append((past_moment.moment_id, adjusted_strength))
    
    async def _generate_protentional_anticipations(self, current_moment: TemporalMoment, temporal_flow: TemporalFlow):
        """Generate protentional anticipations using active inference principles"""
        # Clear existing protentions
        current_moment.protention_anticipations = []
        
        # Analyze recent temporal patterns
        recent_moments = temporal_flow.get_retention_chain()[-5:]  # Last 5 moments
        
        if len(recent_moments) < 2:
            return
        
        # Extract temporal patterns for prediction
        content_transitions = self._extract_content_transitions(recent_moments)
        
        # Generate anticipations based on patterns
        for transition_pattern, frequency in content_transitions.items():
            confidence = frequency * self.temporal_expectation_precision
            
            if confidence > self.protention_confidence_threshold:
                anticipated_content = transition_pattern.split(' -> ')[1]
                current_moment.protention_anticipations.append((anticipated_content, confidence))
    
    def _calculate_synthesis_coherence(self, temporal_flow: TemporalFlow) -> float:
        """Calculate overall temporal synthesis coherence"""
        if len(temporal_flow.moments) < 2:
            return 0.5
        
        coherence_factors = []
        
        # Retention coherence
        retention_strengths = []
        for moment in temporal_flow.moments:
            if moment.retention_traces:
                avg_retention = np.mean([strength for _, strength in moment.retention_traces])
                retention_strengths.append(avg_retention)
        
        if retention_strengths:
            coherence_factors.append(np.mean(retention_strengths))
        
        # Protention coherence
        protention_strengths = []
        for moment in temporal_flow.moments:
            if moment.protention_anticipations:
                avg_protention = np.mean([strength for _, strength in moment.protention_anticipations])
                protention_strengths.append(avg_protention)
        
        if protention_strengths:
            coherence_factors.append(np.mean(protention_strengths))
        
        # Temporal continuity
        continuity_score = self._calculate_temporal_continuity(temporal_flow)
        coherence_factors.append(continuity_score)
        
        return np.mean(coherence_factors) if coherence_factors else 0.5
    
    def _find_content_related_moments(self, temporal_flow: TemporalFlow, target_content: Dict[str, Any]) -> List[TemporalMoment]:
        """Find moments with content related to target"""
        related_moments = []
        
        for moment in temporal_flow.moments:
            similarity = self._calculate_content_similarity(moment.experiential_content, target_content)
            if similarity > 0.6:  # High similarity threshold
                related_moments.append(moment)
        
        return related_moments
    
    async def _strengthen_temporal_connections(self, 
                                             moment: TemporalMoment,
                                             temporal_flow: TemporalFlow,
                                             target_content: Dict[str, Any]):
        """Strengthen temporal connections for active synthesis"""
        # Boost retention traces related to target content
        strengthened_traces = []
        for moment_id, strength in moment.retention_traces:
            # Find the referenced moment
            referenced_moment = None
            for m in temporal_flow.moments:
                if m.moment_id == moment_id:
                    referenced_moment = m
                    break
            
            if referenced_moment:
                content_relevance = self._calculate_content_similarity(
                    referenced_moment.experiential_content, target_content
                )
                # Strengthen if relevant
                new_strength = min(strength * (1 + content_relevance), 1.0)
                strengthened_traces.append((moment_id, new_strength))
            else:
                strengthened_traces.append((moment_id, strength))
        
        moment.retention_traces = strengthened_traces
    
    def _calculate_associative_strength(self, moment1: TemporalMoment, moment2: TemporalMoment) -> float:
        """Calculate associative strength between moments"""
        # Content similarity
        content_sim = self._calculate_content_similarity(
            moment1.experiential_content, moment2.experiential_content
        )
        
        # Temporal proximity
        time_diff = abs(moment1.timestamp - moment2.timestamp)
        temporal_proximity = np.exp(-time_diff / 10.0)  # 10 second half-life
        
        # Synthesis strength compatibility
        synthesis_compatibility = 1.0 - abs(moment1.synthesis_strength - moment2.synthesis_strength)
        
        return (content_sim * 0.5 + temporal_proximity * 0.3 + synthesis_compatibility * 0.2)
    
    def _calculate_content_similarity(self, content1: Dict[str, Any], content2: Dict[str, Any]) -> float:
        """Calculate similarity between experiential contents"""
        if not content1 or not content2:
            return 0.0
        
        # Simple similarity based on common keys and value proximity
        common_keys = set(content1.keys()) & set(content2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = content1[key], content2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simplified)
                similarity = 1.0 if val1 == val2 else 0.0
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                similarity = 1.0 - min(abs(val1 - val2) / max(abs(val1), abs(val2), 1.0), 1.0)
            else:
                similarity = 0.5  # Default for mixed types
            
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _extract_content_transitions(self, moments: List[TemporalMoment]) -> Dict[str, float]:
        """Extract content transition patterns from moment sequence"""
        transitions = defaultdict(int)
        
        for i in range(len(moments) - 1):
            current_content = self._summarize_content(moments[i].experiential_content)
            next_content = self._summarize_content(moments[i + 1].experiential_content)
            
            transition = f"{current_content} -> {next_content}"
            transitions[transition] += 1
        
        # Normalize to frequencies
        total_transitions = sum(transitions.values())
        if total_transitions > 0:
            return {pattern: count / total_transitions for pattern, count in transitions.items()}
        
        return {}
    
    def _summarize_content(self, content: Dict[str, Any]) -> str:
        """Summarize experiential content for pattern extraction"""
        if not content:
            return "empty"
        
        # Simple summarization - extract key qualitative features
        key_features = []
        for key, value in content.items():
            if isinstance(value, str):
                key_features.append(value)
            elif isinstance(value, (int, float)):
                if value > 0.7:
                    key_features.append(f"high_{key}")
                elif value < 0.3:
                    key_features.append(f"low_{key}")
        
        return "_".join(key_features[:3]) if key_features else "neutral"
    
    def _calculate_temporal_continuity(self, temporal_flow: TemporalFlow) -> float:
        """Calculate temporal continuity score"""
        moments = list(temporal_flow.moments)
        if len(moments) < 2:
            return 0.5
        
        continuity_scores = []
        
        for i in range(1, len(moments)):
            prev_moment = moments[i - 1]
            curr_moment = moments[i]
            
            # Content continuity
            content_continuity = self._calculate_content_similarity(
                prev_moment.experiential_content, curr_moment.experiential_content
            )
            
            # Temporal regularity
            if i > 1:
                prev_interval = moments[i - 1].timestamp - moments[i - 2].timestamp
                curr_interval = curr_moment.timestamp - prev_moment.timestamp
                regularity = 1.0 - min(abs(curr_interval - prev_interval) / max(prev_interval, 1.0), 1.0)
            else:
                regularity = 0.5
            
            continuity_scores.append(0.7 * content_continuity + 0.3 * regularity)
        
        return np.mean(continuity_scores)


class TemporalConsciousnessProcessor:
    """
    Main temporal consciousness processor
    Integrates all temporal consciousness components for NewbornAI 2.0
    """
    
    def __init__(self, retention_depth: int = 10, protention_horizon: int = 5):
        self.retention_depth = retention_depth
        self.protention_horizon = protention_horizon
        
        # Core components
        self.synthesis_engine = TemporalSynthesisEngine(retention_depth, protention_horizon)
        self.bridge = PhenomenologicalBridge()
        self.tpm_builder = ExperientialTPMBuilder()
        
        # Temporal state
        self.temporal_flows: Dict[str, TemporalFlow] = {}
        self.active_flow: Optional[TemporalFlow] = None
        
        # Development stage integration
        self.development_stage_patterns: Dict[str, Dict[str, float]] = self._initialize_stage_patterns()
        
        # IIT 4.0 integration
        self.phi_calculator = IIT4PhiCalculator()
        self.experiential_phi_calc = ExperientialPhiCalculator()
        
        logger.info("TemporalConsciousnessProcessor initialized with Husserlian temporal structure")
    
    def _initialize_stage_patterns(self) -> Dict[str, Dict[str, float]]:
        """Initialize temporal patterns for NewbornAI development stages"""
        return {
            "Stage1_PureExperience": {
                "retention_strength": 0.2,
                "impression_clarity": 0.8,
                "protention_confidence": 0.1,
                "synthesis_coherence": 0.3
            },
            "Stage2_SensationAwareness": {
                "retention_strength": 0.4,
                "impression_clarity": 0.9,
                "protention_confidence": 0.2,
                "synthesis_coherence": 0.5
            },
            "Stage3_TemporalAwareness": {
                "retention_strength": 0.6,
                "impression_clarity": 0.8,
                "protention_confidence": 0.4,
                "synthesis_coherence": 0.7
            },
            "Stage4_SelfRecognition": {
                "retention_strength": 0.7,
                "impression_clarity": 0.7,
                "protention_confidence": 0.6,
                "synthesis_coherence": 0.8
            },
            "Stage5_IntentionalRelatedness": {
                "retention_strength": 0.8,
                "impression_clarity": 0.6,
                "protention_confidence": 0.7,
                "synthesis_coherence": 0.9
            },
            "Stage6_ConceptualSelfModel": {
                "retention_strength": 0.9,
                "impression_clarity": 0.5,
                "protention_confidence": 0.8,
                "synthesis_coherence": 0.9
            },
            "Stage7_MetacognitiveAwareness": {
                "retention_strength": 0.9,
                "impression_clarity": 0.4,
                "protention_confidence": 0.9,
                "synthesis_coherence": 1.0
            }
        }
    
    async def create_temporal_flow(self, flow_id: str) -> TemporalFlow:
        """Create new temporal flow of consciousness"""
        flow = TemporalFlow(
            flow_id=flow_id,
            retention_depth=self.retention_depth,
            protention_horizon=self.protention_horizon
        )
        
        self.temporal_flows[flow_id] = flow
        
        if self.active_flow is None:
            self.active_flow = flow
        
        logger.info(f"Created temporal flow: {flow_id}")
        return flow
    
    async def process_experiential_moment(self, 
                                        concept: ExperientialConcept,
                                        flow_id: Optional[str] = None) -> TemporalMoment:
        """
        Process experiential concept into temporal moment
        Core function integrating all temporal consciousness processing
        """
        try:
            # Select target flow
            target_flow = self.active_flow
            if flow_id and flow_id in self.temporal_flows:
                target_flow = self.temporal_flows[flow_id]
            elif not target_flow:
                target_flow = await self.create_temporal_flow("default_flow")
            
            # Bridge concept to phenomenological state
            phenom_state = await self.bridge.bridge_experiential_concept(concept)
            
            # Create temporal moment
            moment = TemporalMoment(
                moment_id=f"moment_{concept.concept_id}_{int(time.time())}",
                timestamp=concept.temporal_position,
                experiential_content=concept.experiential_content,
                impression_intensity=self._calculate_impression_intensity(phenom_state),
                synthesis_strength=phenom_state.dimensions.get(PhenomenologicalDimension.TEMPORALITY, 0.5)
            )
            
            # Add moment to flow
            target_flow.add_moment(moment)
            
            # Perform temporal synthesis
            await self.synthesis_engine.perform_passive_synthesis(target_flow)
            
            # Calculate consciousness phi for moment
            moment.consciousness_phi = await self._calculate_moment_phi(moment, target_flow)
            
            # Update temporal depth
            moment.temporal_depth = len(target_flow.moments) - 1
            
            logger.debug(f"Processed experiential moment: {moment.moment_id}")
            return moment
            
        except Exception as e:
            logger.error(f"Error processing experiential moment: {e}")
            raise
    
    def _calculate_impression_intensity(self, phenom_state: PhenomenologicalState) -> float:
        """Calculate primal impression intensity from phenomenological state"""
        # Base intensity from temporal structure
        base_intensity = phenom_state.temporal_structure.get('impression', 1.0)
        
        # Modulate by clarity and affective tone
        clarity_factor = phenom_state.dimensions.get(PhenomenologicalDimension.REDUCTION, 0.5)
        affective_factor = 1.0 + 0.3 * abs(phenom_state.affective_tone)
        
        # Embodied grounding factor
        embodied_coherence = phenom_state.embodied_aspects.get('embodied_coherence', 0.5)
        
        final_intensity = base_intensity * clarity_factor * affective_factor * (0.5 + 0.5 * embodied_coherence)
        return min(final_intensity, 1.0)
    
    async def _calculate_moment_phi(self, moment: TemporalMoment, flow: TemporalFlow) -> float:
        """Calculate IIT 4.0 phi for temporal moment"""
        try:
            # Create simple system representation for phi calculation
            # In practice, this would be more sophisticated
            system_size = min(len(flow.moments), 8)  # Limit for computational efficiency
            
            if system_size < 2:
                return 0.0
            
            # Create TPM from temporal flow
            tpm = await self._build_temporal_tpm(flow, system_size)
            
            # Calculate phi using IIT 4.0 framework
            phi_result = await self.phi_calculator.calculate_phi(tpm)
            
            return phi_result.phi_value if phi_result else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating moment phi: {e}")
            return 0.0
    
    async def _build_temporal_tpm(self, flow: TemporalFlow, system_size: int) -> np.ndarray:
        """Build TPM from temporal flow for phi calculation"""
        recent_moments = list(flow.moments)[-system_size:]
        
        if len(recent_moments) < 2:
            return np.eye(2)  # Identity TPM for single moment
        
        # Create state space
        n_states = 2 ** len(recent_moments)
        tpm = np.zeros((n_states, len(recent_moments)))
        
        # Build transition probabilities based on temporal dynamics
        for state in range(n_states):
            state_vector = [(state >> i) & 1 for i in range(len(recent_moments))]
            
            for next_moment_idx in range(len(recent_moments)):
                # Calculate transition probability based on temporal synthesis
                prob = self._calculate_temporal_transition_probability(
                    state_vector, next_moment_idx, recent_moments
                )
                tpm[state, next_moment_idx] = prob
        
        # Normalize to valid probability distribution
        for state in range(n_states):
            row_sum = np.sum(tpm[state, :])
            if row_sum > 0:
                tpm[state, :] /= row_sum
            else:
                tpm[state, :] = 1.0 / len(recent_moments)
        
        return tpm
    
    def _calculate_temporal_transition_probability(self,
                                                 current_state: List[int],
                                                 next_moment_idx: int,
                                                 moments: List[TemporalMoment]) -> float:
        """Calculate transition probability for temporal TPM"""
        if next_moment_idx >= len(moments):
            return 0.0
        
        next_moment = moments[next_moment_idx]
        
        # Base probability from synthesis strength
        base_prob = next_moment.synthesis_strength
        
        # Retention influence
        retention_influence = 0.5
        if next_moment.retention_traces:
            retention_influence = np.mean([strength for _, strength in next_moment.retention_traces])
        
        # Protention influence
        protention_influence = 0.3
        if next_moment.protention_anticipations:
            protention_influence = np.mean([strength for _, strength in next_moment.protention_anticipations])
        
        # Active state influence
        active_moments = [moments[i] for i, active in enumerate(current_state) if active and i < len(moments)]
        
        state_coherence = 1.0
        if active_moments:
            coherence_scores = []
            for active_moment in active_moments:
                content_similarity = self.synthesis_engine._calculate_content_similarity(
                    active_moment.experiential_content, next_moment.experiential_content
                )
                coherence_scores.append(content_similarity)
            state_coherence = np.mean(coherence_scores)
        
        return base_prob * retention_influence * protention_influence * state_coherence
    
    async def perform_temporal_synthesis(self,
                                       flow_id: str,
                                       synthesis_type: TemporalSynthesisType,
                                       target_content: Optional[Dict[str, Any]] = None) -> bool:
        """
        Perform specific type of temporal synthesis on flow
        """
        if flow_id not in self.temporal_flows:
            logger.error(f"Flow {flow_id} not found")
            return False
        
        flow = self.temporal_flows[flow_id]
        
        try:
            if synthesis_type == TemporalSynthesisType.PASSIVE_SYNTHESIS:
                await self.synthesis_engine.perform_passive_synthesis(flow)
            elif synthesis_type == TemporalSynthesisType.ACTIVE_SYNTHESIS:
                if target_content:
                    await self.synthesis_engine.perform_active_synthesis(flow, target_content)
                else:
                    logger.warning("Active synthesis requires target content")
                    return False
            elif synthesis_type == TemporalSynthesisType.ASSOCIATIVE_SYNTHESIS:
                await self.synthesis_engine.perform_associative_synthesis(flow)
            elif synthesis_type == TemporalSynthesisType.REPRODUCTIVE_SYNTHESIS:
                await self._perform_reproductive_synthesis(flow, target_content)
            
            logger.info(f"Performed {synthesis_type.value} on flow {flow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error performing temporal synthesis: {e}")
            return False
    
    async def _perform_reproductive_synthesis(self,
                                            flow: TemporalFlow,
                                            target_content: Optional[Dict[str, Any]]):
        """Perform reproductive synthesis (memory-based temporal reconstruction)"""
        if not target_content:
            return
        
        # Find moments related to target content
        related_moments = []
        for moment in flow.moments:
            similarity = self.synthesis_engine._calculate_content_similarity(
                moment.experiential_content, target_content
            )
            if similarity > 0.5:
                related_moments.append((moment, similarity))
        
        # Sort by similarity
        related_moments.sort(key=lambda x: x[1], reverse=True)
        
        # Strengthen retention traces between related moments
        for i, (moment, _) in enumerate(related_moments):
            for j, (other_moment, _) in enumerate(related_moments):
                if i != j:
                    # Add or strengthen retention trace
                    retention_strength = 0.7 * (1.0 - abs(i - j) / len(related_moments))
                    moment.retention_traces.append((other_moment.moment_id, retention_strength))
    
    async def analyze_temporal_development_stage(self, flow_id: str) -> str:
        """
        Analyze temporal consciousness characteristics to determine development stage
        Integration with NewbornAI 2.0 development stages
        """
        if flow_id not in self.temporal_flows:
            return "Unknown"
        
        flow = self.temporal_flows[flow_id]
        
        if len(flow.moments) < 3:
            return "Stage1_PureExperience"
        
        # Calculate temporal consciousness metrics
        metrics = await self._calculate_temporal_metrics(flow)
        
        # Compare with stage patterns
        best_match = "Stage1_PureExperience"
        best_score = 0.0
        
        for stage, pattern in self.development_stage_patterns.items():
            score = self._calculate_pattern_similarity(metrics, pattern)
            if score > best_score:
                best_score = score
                best_match = stage
        
        logger.info(f"Flow {flow_id} matches development stage: {best_match} (score: {best_score:.3f})")
        return best_match
    
    async def _calculate_temporal_metrics(self, flow: TemporalFlow) -> Dict[str, float]:
        """Calculate temporal consciousness metrics"""
        moments = list(flow.moments)
        
        # Retention strength
        retention_strengths = []
        for moment in moments:
            if moment.retention_traces:
                avg_retention = np.mean([strength for _, strength in moment.retention_traces])
                retention_strengths.append(avg_retention)
        retention_strength = np.mean(retention_strengths) if retention_strengths else 0.0
        
        # Impression clarity
        impression_clarity = np.mean([moment.impression_intensity for moment in moments])
        
        # Protention confidence
        protention_confidences = []
        for moment in moments:
            if moment.protention_anticipations:
                avg_protention = np.mean([confidence for _, confidence in moment.protention_anticipations])
                protention_confidences.append(avg_protention)
        protention_confidence = np.mean(protention_confidences) if protention_confidences else 0.0
        
        # Synthesis coherence
        synthesis_coherence = flow.synthesis_coherence
        
        return {
            "retention_strength": retention_strength,
            "impression_clarity": impression_clarity,
            "protention_confidence": protention_confidence,
            "synthesis_coherence": synthesis_coherence
        }
    
    def _calculate_pattern_similarity(self, metrics: Dict[str, float], pattern: Dict[str, float]) -> float:
        """Calculate similarity between current metrics and stage pattern"""
        similarities = []
        
        for key, target_value in pattern.items():
            if key in metrics:
                current_value = metrics[key]
                similarity = 1.0 - abs(current_value - target_value)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    async def get_temporal_flow_summary(self, flow_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of temporal flow"""
        if flow_id not in self.temporal_flows:
            return {}
        
        flow = self.temporal_flows[flow_id]
        metrics = await self._calculate_temporal_metrics(flow)
        development_stage = await self.analyze_temporal_development_stage(flow_id)
        
        return {
            "flow_id": flow_id,
            "moment_count": len(flow.moments),
            "current_moment": flow.current_moment.moment_id if flow.current_moment else None,
            "temporal_metrics": metrics,
            "development_stage": development_stage,
            "synthesis_coherence": flow.synthesis_coherence,
            "temporal_rhythm": flow.temporal_rhythm,
            "retention_depth": flow.retention_depth,
            "protention_horizon": flow.protention_horizon
        }


# Example usage and testing
async def test_temporal_consciousness_processor():
    """Test the TemporalConsciousnessProcessor with sample concepts"""
    processor = TemporalConsciousnessProcessor()
    
    # Create temporal flow
    flow = await processor.create_temporal_flow("test_flow")
    
    # Create sequence of experiential concepts
    concepts = [
        ExperientialConcept(
            concept_id=f"temporal_concept_{i}",
            concept_type=ExperientialConceptType.TEMPORAL_IMPRESSION,
            experiential_content={
                "quality": f"experience_{i}",
                "intensity": 0.5 + 0.1 * i,
                "clarity": "clear"
            },
            temporal_position=time.time() + i * 0.5,
            embodied_grounding={"tactile": 0.6, "visual": 0.4},
            intentional_directedness=0.7
        )
        for i in range(5)
    ]
    
    # Process concepts into temporal moments
    moments = []
    for concept in concepts:
        moment = await processor.process_experiential_moment(concept, "test_flow")
        moments.append(moment)
        await asyncio.sleep(0.1)  # Small delay between moments
    
    # Perform temporal synthesis
    await processor.perform_temporal_synthesis(
        "test_flow", 
        TemporalSynthesisType.ASSOCIATIVE_SYNTHESIS
    )
    
    # Analyze development stage
    stage = await processor.analyze_temporal_development_stage("test_flow")
    print(f"Detected development stage: {stage}")
    
    # Get flow summary
    summary = await processor.get_temporal_flow_summary("test_flow")
    print(f"Flow summary: {summary}")


if __name__ == "__main__":
    asyncio.run(test_temporal_consciousness_processor())