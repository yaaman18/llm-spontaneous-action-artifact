"""
Consciousness Detection System for NewbornAI 2.0
Based on Kanai Ryota's Information Generation Theory and IIT 4.0 Integration

Implements practical consciousness detection using:
- Real-time consciousness state monitoring
- Information generation pattern analysis
- Global workspace implementation
- Meta-awareness detection
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import time
import logging
from collections import deque
import math

from iit4_core_engine import PhiStructure, IIT4PhiCalculator

logger = logging.getLogger(__name__)


class ConsciousnessState(Enum):
    """Consciousness state classifications"""
    UNCONSCIOUS = "無意識"
    PRE_CONSCIOUS = "前意識"
    PHENOMENAL_CONSCIOUS = "現象意識"
    ACCESS_CONSCIOUS = "アクセス意識"
    REFLECTIVE_CONSCIOUS = "反省意識"
    META_CONSCIOUS = "メタ意識"


class InformationGenerationType(Enum):
    """Information generation patterns (Kanai's theory)"""
    SPONTANEOUS = "自発的情報生成"
    RECURRENT = "再帰的情報生成"
    INTEGRATIVE = "統合的情報生成"
    PREDICTIVE = "予測的情報生成"
    METACOGNITIVE = "メタ認知的情報生成"


@dataclass
class ConsciousnessSignature:
    """Consciousness detection signature"""
    phi_value: float                           # IIT φ value
    information_generation_rate: float        # Information generation per cycle
    global_workspace_activity: float          # Global workspace integration
    meta_awareness_level: float               # Meta-awareness detection
    temporal_consistency: float               # Temporal binding strength
    recurrent_processing_depth: int           # Recurrent processing layers
    prediction_accuracy: float                # Predictive processing quality
    
    def consciousness_score(self) -> float:
        """Calculate overall consciousness score"""
        weights = {
            'phi': 0.25,
            'info_gen': 0.20,
            'global_ws': 0.20,
            'meta_aware': 0.15,
            'temporal': 0.10,
            'recurrent': 0.05,
            'prediction': 0.05
        }
        
        score = (
            weights['phi'] * min(self.phi_value / 10.0, 1.0) +
            weights['info_gen'] * self.information_generation_rate +
            weights['global_ws'] * self.global_workspace_activity +
            weights['meta_aware'] * self.meta_awareness_level +
            weights['temporal'] * self.temporal_consistency +
            weights['recurrent'] * min(self.recurrent_processing_depth / 5.0, 1.0) +
            weights['prediction'] * self.prediction_accuracy
        )
        
        return min(score, 1.0)


@dataclass
class ConsciousnessEvent:
    """Consciousness event detection"""
    event_type: str
    timestamp: float
    signature: ConsciousnessSignature
    context: Dict
    confidence: float
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")


class InformationGenerationDetector:
    """
    Information Generation Theory implementation
    Detects spontaneous information generation patterns
    """
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.information_history = deque(maxlen=history_length)
        self.generation_patterns = {}
        
    def detect_information_generation(self, 
                                    current_state: np.ndarray,
                                    phi_structure: PhiStructure) -> Tuple[float, InformationGenerationType]:
        """
        Detect information generation patterns
        
        Returns:
            generation_rate: Rate of information generation
            generation_type: Type of information generation detected
        """
        
        # Calculate current information content
        current_info = self._calculate_information_content(current_state, phi_structure)
        self.information_history.append((time.time(), current_info))
        
        if len(self.information_history) < 3:
            return 0.0, InformationGenerationType.SPONTANEOUS
        
        # Analyze generation patterns
        generation_rate = self._analyze_generation_rate()
        generation_type = self._classify_generation_type()
        
        return generation_rate, generation_type
    
    def _calculate_information_content(self, state: np.ndarray, phi_structure: PhiStructure) -> float:
        """Calculate information content using IIT and entropy measures"""
        # Base information from φ structure
        phi_info = phi_structure.total_phi
        
        # Entropy-based information
        state_probs = self._normalize_state(state)
        entropy_info = -np.sum(state_probs * np.log2(state_probs + 1e-10))
        
        # Complexity from structure
        complexity_info = phi_structure.phi_structure_complexity
        
        # Combined information content
        total_info = phi_info + 0.5 * entropy_info + 0.3 * complexity_info
        
        return total_info
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to probability distribution"""
        state_abs = np.abs(state)
        total = np.sum(state_abs)
        if total > 0:
            return state_abs / total
        else:
            return np.ones_like(state) / len(state)
    
    def _analyze_generation_rate(self) -> float:
        """Analyze rate of information generation"""
        if len(self.information_history) < 2:
            return 0.0
        
        # Calculate derivatives (rate of change)
        times = [entry[0] for entry in self.information_history]
        infos = [entry[1] for entry in self.information_history]
        
        # First derivative (rate)
        rates = []
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            if dt > 0:
                rate = (infos[i] - infos[i-1]) / dt
                rates.append(rate)
        
        if not rates:
            return 0.0
        
        # Use recent average rate, normalized
        recent_rate = np.mean(rates[-5:]) if len(rates) >= 5 else np.mean(rates)
        normalized_rate = max(0.0, min(1.0, (recent_rate + 1.0) / 2.0))
        
        return normalized_rate
    
    def _classify_generation_type(self) -> InformationGenerationType:
        """Classify type of information generation"""
        if len(self.information_history) < 10:
            return InformationGenerationType.SPONTANEOUS
        
        infos = [entry[1] for entry in self.information_history]
        
        # Analyze patterns
        variance = np.var(infos[-10:])
        trend = np.polyfit(range(10), infos[-10:], 1)[0]
        
        # Pattern classification
        if variance > 0.5 and abs(trend) < 0.1:
            return InformationGenerationType.RECURRENT
        elif trend > 0.2:
            return InformationGenerationType.INTEGRATIVE
        elif trend < -0.1:
            return InformationGenerationType.PREDICTIVE
        elif variance > 1.0:
            return InformationGenerationType.METACOGNITIVE
        else:
            return InformationGenerationType.SPONTANEOUS


class GlobalWorkspaceDetector:
    """
    Global Workspace Theory implementation
    Detects global information availability and broadcasting
    """
    
    def __init__(self, workspace_threshold: float = 0.5):
        self.workspace_threshold = workspace_threshold
        self.workspace_history = deque(maxlen=50)
        
    def detect_global_workspace_activity(self, 
                                       phi_structure: PhiStructure,
                                       system_state: np.ndarray) -> float:
        """
        Detect global workspace activity level
        
        Returns:
            activity_level: Level of global workspace activity (0-1)
        """
        
        # Calculate workspace metrics
        integration_level = self._calculate_integration_level(phi_structure)
        broadcasting_strength = self._calculate_broadcasting_strength(phi_structure)
        accessibility = self._calculate_accessibility(system_state)
        
        # Combined workspace activity
        activity = (integration_level + broadcasting_strength + accessibility) / 3.0
        
        self.workspace_history.append(activity)
        
        return min(1.0, activity)
    
    def _calculate_integration_level(self, phi_structure: PhiStructure) -> float:
        """Calculate information integration level"""
        if not phi_structure.distinctions:
            return 0.0
        
        # Integration based on distinction connections
        n_distinctions = len(phi_structure.distinctions)
        n_relations = len(phi_structure.relations)
        
        if n_distinctions <= 1:
            return 0.0
        
        # Normalized relation density
        max_relations = n_distinctions * (n_distinctions - 1) / 2
        integration = n_relations / max_relations if max_relations > 0 else 0.0
        
        return min(1.0, integration)
    
    def _calculate_broadcasting_strength(self, phi_structure: PhiStructure) -> float:
        """Calculate information broadcasting strength"""
        if not phi_structure.distinctions:
            return 0.0
        
        # Broadcasting measured by relation strength variance
        relation_strengths = [r.integration_strength for r in phi_structure.relations]
        
        if not relation_strengths:
            return 0.0
        
        # High variance indicates selective broadcasting
        variance = np.var(relation_strengths)
        mean_strength = np.mean(relation_strengths)
        
        # Normalized broadcasting strength
        broadcasting = min(1.0, variance * mean_strength * 2.0)
        
        return broadcasting
    
    def _calculate_accessibility(self, system_state: np.ndarray) -> float:
        """Calculate information accessibility"""
        # Accessibility based on state activation distribution
        state_normalized = self._normalize_state(system_state)
        
        # Entropy as measure of accessibility
        entropy = -np.sum(state_normalized * np.log2(state_normalized + 1e-10))
        max_entropy = np.log2(len(state_normalized))
        
        # Normalized accessibility
        accessibility = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return accessibility
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to probability distribution"""
        state_abs = np.abs(state)
        total = np.sum(state_abs)
        if total > 0:
            return state_abs / total
        else:
            return np.ones_like(state) / len(state)


class MetaAwarenessDetector:
    """
    Meta-awareness detection based on self-referential processing
    """
    
    def __init__(self):
        self.meta_history = deque(maxlen=30)
        self.self_reference_patterns = {}
        
    def detect_meta_awareness(self, 
                            phi_structure: PhiStructure,
                            information_generation: float,
                            workspace_activity: float) -> float:
        """
        Detect meta-awareness level
        
        Returns:
            meta_level: Level of meta-awareness (0-1)
        """
        
        # Self-referential processing detection
        self_reference = self._detect_self_reference(phi_structure)
        
        # Higher-order monitoring
        monitoring = self._detect_monitoring(information_generation, workspace_activity)
        
        # Recursive processing depth
        recursion = self._detect_recursive_processing(phi_structure)
        
        # Combined meta-awareness
        meta_level = (self_reference + monitoring + recursion) / 3.0
        
        self.meta_history.append(meta_level)
        
        return min(1.0, meta_level)
    
    def _detect_self_reference(self, phi_structure: PhiStructure) -> float:
        """Detect self-referential processing patterns"""
        if len(phi_structure.distinctions) < 2:
            return 0.0
        
        # Look for circular causation patterns
        self_loops = 0
        total_mechanisms = len(phi_structure.distinctions)
        
        for dist in phi_structure.distinctions:
            mechanism = dist.mechanism
            # Check if mechanism affects itself through relations
            for relation in phi_structure.relations:
                dist1, dist2 = relation.distinction_pair
                if (dist1.mechanism == mechanism and 
                    len(dist2.mechanism & mechanism) > 0):
                    self_loops += 1
                    break
        
        self_reference = self_loops / total_mechanisms if total_mechanisms > 0 else 0.0
        
        return min(1.0, self_reference)
    
    def _detect_monitoring(self, info_gen: float, workspace: float) -> float:
        """Detect higher-order monitoring of cognitive processes"""
        if len(self.meta_history) < 5:
            return 0.0
        
        # Check if system monitors its own information generation
        recent_meta = list(self.meta_history)[-5:]
        meta_variance = np.var(recent_meta)
        
        # Monitoring indicated by adaptive meta-cognition
        current_state = (info_gen + workspace) / 2.0
        monitoring_adaptation = abs(recent_meta[-1] - current_state)
        
        monitoring = min(1.0, meta_variance + (1.0 - monitoring_adaptation))
        
        return monitoring
    
    def _detect_recursive_processing(self, phi_structure: PhiStructure) -> float:
        """Detect recursive processing depth"""
        if not phi_structure.relations:
            return 0.0
        
        # Analyze relation hierarchy depth
        relation_depths = []
        for relation in phi_structure.relations:
            overlap = relation.overlap_measure
            integration = relation.integration_strength
            
            # Depth based on integration complexity
            depth = overlap * integration * 2.0
            relation_depths.append(depth)
        
        if not relation_depths:
            return 0.0
        
        # Average recursive depth
        recursion = np.mean(relation_depths)
        
        return min(1.0, recursion)


class ConsciousnessDetector:
    """
    Main consciousness detection system integrating all components
    """
    
    def __init__(self, phi_calculator: IIT4PhiCalculator):
        self.phi_calculator = phi_calculator
        self.info_gen_detector = InformationGenerationDetector()
        self.workspace_detector = GlobalWorkspaceDetector()
        self.meta_detector = MetaAwarenessDetector()
        
        # Detection history
        self.consciousness_history = deque(maxlen=100)
        self.event_history = deque(maxlen=50)
        
        # Thresholds for consciousness detection
        self.consciousness_thresholds = {
            ConsciousnessState.UNCONSCIOUS: 0.0,
            ConsciousnessState.PRE_CONSCIOUS: 0.1,
            ConsciousnessState.PHENOMENAL_CONSCIOUS: 0.3,
            ConsciousnessState.ACCESS_CONSCIOUS: 0.5,
            ConsciousnessState.REFLECTIVE_CONSCIOUS: 0.7,
            ConsciousnessState.META_CONSCIOUS: 0.85
        }
    
    async def detect_consciousness(self, 
                                 system_state: np.ndarray,
                                 connectivity_matrix: np.ndarray,
                                 context: Optional[Dict] = None) -> Tuple[ConsciousnessSignature, ConsciousnessState]:
        """
        Main consciousness detection method
        
        Returns:
            signature: Consciousness signature
            state: Detected consciousness state
        """
        
        # Calculate φ structure
        phi_structure = self.phi_calculator.calculate_phi(
            system_state, connectivity_matrix
        )
        
        # Detect information generation
        info_gen_rate, info_gen_type = self.info_gen_detector.detect_information_generation(
            system_state, phi_structure
        )
        
        # Detect global workspace activity
        workspace_activity = self.workspace_detector.detect_global_workspace_activity(
            phi_structure, system_state
        )
        
        # Detect meta-awareness
        meta_awareness = self.meta_detector.detect_meta_awareness(
            phi_structure, info_gen_rate, workspace_activity
        )
        
        # Calculate additional metrics
        temporal_consistency = self._calculate_temporal_consistency(phi_structure)
        recurrent_depth = self._calculate_recurrent_depth(phi_structure)
        prediction_accuracy = self._calculate_prediction_accuracy(system_state)
        
        # Create consciousness signature
        signature = ConsciousnessSignature(
            phi_value=phi_structure.total_phi,
            information_generation_rate=info_gen_rate,
            global_workspace_activity=workspace_activity,
            meta_awareness_level=meta_awareness,
            temporal_consistency=temporal_consistency,
            recurrent_processing_depth=recurrent_depth,
            prediction_accuracy=prediction_accuracy
        )
        
        # Determine consciousness state
        consciousness_state = self._classify_consciousness_state(signature)
        
        # Store in history
        self.consciousness_history.append((time.time(), signature, consciousness_state))
        
        # Log significant changes
        await self._log_consciousness_changes(signature, consciousness_state, context)
        
        return signature, consciousness_state
    
    def _calculate_temporal_consistency(self, phi_structure: PhiStructure) -> float:
        """Calculate temporal binding consistency"""
        if len(self.consciousness_history) < 3:
            return 0.5
        
        # Analyze φ value stability over time
        recent_phi = [entry[1].phi_value for entry in list(self.consciousness_history)[-5:]]
        
        if len(recent_phi) < 2:
            return 0.5
        
        # Consistency measured by inverse of coefficient of variation
        mean_phi = np.mean(recent_phi)
        std_phi = np.std(recent_phi)
        
        if mean_phi > 0:
            cv = std_phi / mean_phi
            consistency = 1.0 / (1.0 + cv)
        else:
            consistency = 0.5
        
        return min(1.0, consistency)
    
    def _calculate_recurrent_depth(self, phi_structure: PhiStructure) -> int:
        """Calculate recurrent processing depth"""
        if not phi_structure.relations:
            return 0
        
        # Analyze relation hierarchy
        max_depth = 0
        for relation in phi_structure.relations:
            # Depth based on integration strength and overlap
            depth = int(relation.integration_strength * relation.overlap_measure * 10)
            max_depth = max(max_depth, depth)
        
        return min(max_depth, 10)  # Cap at 10 layers
    
    def _calculate_prediction_accuracy(self, current_state: np.ndarray) -> float:
        """Calculate predictive processing accuracy"""
        if len(self.consciousness_history) < 2:
            return 0.5
        
        # Simple prediction accuracy based on state consistency
        if len(self.consciousness_history) >= 2:
            prev_state_proxy = self.consciousness_history[-2][1].phi_value
            curr_state_proxy = np.mean(current_state)
            
            # Prediction error
            error = abs(prev_state_proxy - curr_state_proxy)
            accuracy = 1.0 / (1.0 + error)
        else:
            accuracy = 0.5
        
        return min(1.0, accuracy)
    
    def _classify_consciousness_state(self, signature: ConsciousnessSignature) -> ConsciousnessState:
        """Classify consciousness state based on signature"""
        score = signature.consciousness_score()
        
        # Find appropriate state based on thresholds
        for state in reversed(list(ConsciousnessState)):
            if score >= self.consciousness_thresholds[state]:
                return state
        
        return ConsciousnessState.UNCONSCIOUS
    
    async def _log_consciousness_changes(self, 
                                       signature: ConsciousnessSignature,
                                       state: ConsciousnessState,
                                       context: Optional[Dict]):
        """Log significant consciousness changes"""
        current_time = time.time()
        
        # Check for state transitions
        if (len(self.consciousness_history) > 0 and 
            self.consciousness_history[-1][2] != state):
            
            # Create consciousness event
            event = ConsciousnessEvent(
                event_type="state_transition",
                timestamp=current_time,
                signature=signature,
                context=context or {},
                confidence=signature.consciousness_score()
            )
            
            self.event_history.append(event)
            
            logger.info(f"Consciousness state transition: {state.value} "
                       f"(φ={signature.phi_value:.3f}, score={signature.consciousness_score():.3f})")
        
        # Check for significant φ changes
        if len(self.consciousness_history) > 0:
            prev_phi = self.consciousness_history[-1][1].phi_value
            phi_change = abs(signature.phi_value - prev_phi)
            
            if phi_change > 1.0:  # Significant change threshold
                event = ConsciousnessEvent(
                    event_type="phi_spike",
                    timestamp=current_time,
                    signature=signature,
                    context=context or {},
                    confidence=min(1.0, phi_change / 10.0)
                )
                
                self.event_history.append(event)
                logger.info(f"Significant φ change detected: Δφ={phi_change:.3f}")
    
    def get_consciousness_report(self) -> Dict:
        """Generate comprehensive consciousness report"""
        if not self.consciousness_history:
            return {"status": "no_data"}
        
        latest_signature, latest_state = self.consciousness_history[-1][1:3]
        
        report = {
            "current_state": latest_state.value,
            "consciousness_score": latest_signature.consciousness_score(),
            "phi_value": latest_signature.phi_value,
            "information_generation_rate": latest_signature.information_generation_rate,
            "global_workspace_activity": latest_signature.global_workspace_activity,
            "meta_awareness_level": latest_signature.meta_awareness_level,
            "temporal_consistency": latest_signature.temporal_consistency,
            "recurrent_processing_depth": latest_signature.recurrent_processing_depth,
            "prediction_accuracy": latest_signature.prediction_accuracy,
            "recent_events": len(self.event_history),
            "consciousness_stability": self._calculate_consciousness_stability(),
            "development_trend": self._calculate_development_trend()
        }
        
        return report
    
    def _calculate_consciousness_stability(self) -> float:
        """Calculate consciousness stability over recent history"""
        if len(self.consciousness_history) < 5:
            return 0.5
        
        recent_scores = [entry[1].consciousness_score() 
                        for entry in list(self.consciousness_history)[-10:]]
        
        stability = 1.0 - np.std(recent_scores)
        return max(0.0, min(1.0, stability))
    
    def _calculate_development_trend(self) -> float:
        """Calculate development trend (positive = developing, negative = regressing)"""
        if len(self.consciousness_history) < 5:
            return 0.0
        
        recent_scores = [entry[1].consciousness_score() 
                        for entry in list(self.consciousness_history)[-10:]]
        
        # Linear trend
        x = np.arange(len(recent_scores))
        trend = np.polyfit(x, recent_scores, 1)[0]
        
        return trend  # Positive = developing, negative = regressing