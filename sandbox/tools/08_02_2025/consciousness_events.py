"""
Consciousness Events Detection and Management System
Implements real-time consciousness transition detection for NewbornAI 2.0

Based on Kanai Ryota's Information Generation Theory:
- Consciousness transition event detection
- Critical consciousness moments identification
- Temporal consciousness binding analysis
- Consciousness function monitoring
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from enum import Enum
import time
import logging
from collections import deque
import math
from abc import ABC, abstractmethod

from consciousness_detector import (
    ConsciousnessState, ConsciousnessSignature, ConsciousnessEvent,
    InformationGenerationType
)

logger = logging.getLogger(__name__)


class ConsciousnessEventType(Enum):
    """Types of consciousness events"""
    STATE_TRANSITION = "状態遷移"
    PHI_SPIKE = "φ値急変"
    INFORMATION_BURST = "情報生成爆発"
    GLOBAL_WORKSPACE_ACTIVATION = "全域作業空間活性化"
    META_AWARENESS_EMERGENCE = "メタ意識出現"
    CONSCIOUSNESS_FRAGMENTATION = "意識断片化"
    INTEGRATION_FAILURE = "統合失敗"
    TEMPORAL_BINDING_LOSS = "時間結合喪失"
    RECURSIVE_LOOP_DETECTION = "再帰ループ検出"
    CONSCIOUSNESS_RESONANCE = "意識共鳴"
    CRITICAL_TRANSITION = "臨界遷移"
    CONSCIOUSNESS_COLLAPSE = "意識崩壊"


@dataclass
class ConsciousnessEventPattern:
    """Consciousness event pattern definition"""
    event_type: ConsciousnessEventType
    detection_threshold: float
    temporal_window: float  # seconds
    minimum_duration: float  # seconds
    required_conditions: List[str]
    significance_weight: float


@dataclass
class ConsciousnessAlarm:
    """Consciousness alarm for critical events"""
    alarm_type: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    timestamp: float
    consciousness_signature: ConsciousnessSignature
    recommended_action: str
    context: Dict = field(default_factory=dict)


class ConsciousnessEventDetector(ABC):
    """Abstract base class for consciousness event detectors"""
    
    @abstractmethod
    async def detect_event(self, 
                          current_signature: ConsciousnessSignature,
                          previous_signatures: List[ConsciousnessSignature],
                          context: Dict) -> Optional[ConsciousnessEvent]:
        """Detect consciousness event"""
        pass
    
    @abstractmethod
    def get_detector_name(self) -> str:
        """Get detector name"""
        pass


class PhiSpikeDetector(ConsciousnessEventDetector):
    """Detect sudden changes in φ values"""
    
    def __init__(self, spike_threshold: float = 2.0, window_size: int = 5):
        self.spike_threshold = spike_threshold
        self.window_size = window_size
        self.baseline_history = deque(maxlen=50)
    
    async def detect_event(self, 
                          current_signature: ConsciousnessSignature,
                          previous_signatures: List[ConsciousnessSignature],
                          context: Dict) -> Optional[ConsciousnessEvent]:
        
        if len(previous_signatures) < self.window_size:
            return None
        
        # Calculate baseline φ value
        recent_phi = [s.phi_value for s in previous_signatures[-self.window_size:]]
        baseline_phi = np.mean(recent_phi)
        baseline_std = np.std(recent_phi)
        
        # Store baseline for trend analysis
        self.baseline_history.append(baseline_phi)
        
        # Detect spike
        phi_change = abs(current_signature.phi_value - baseline_phi)
        
        if phi_change > self.spike_threshold and phi_change > 3 * baseline_std:
            spike_direction = "increase" if current_signature.phi_value > baseline_phi else "decrease"
            
            event = ConsciousnessEvent(
                event_type=f"phi_spike_{spike_direction}",
                timestamp=time.time(),
                signature=current_signature,
                context={
                    **context,
                    'baseline_phi': baseline_phi,
                    'phi_change': phi_change,
                    'spike_magnitude': phi_change / (baseline_std + 1e-6),
                    'direction': spike_direction
                },
                confidence=min(1.0, phi_change / 10.0)
            )
            
            logger.warning(f"φ spike detected: {spike_direction} by {phi_change:.3f} "
                          f"(baseline: {baseline_phi:.3f})")
            
            return event
        
        return None
    
    def get_detector_name(self) -> str:
        return "phi_spike_detector"


class InformationBurstDetector(ConsciousnessEventDetector):
    """Detect sudden bursts in information generation"""
    
    def __init__(self, burst_threshold: float = 0.8, temporal_window: float = 10.0):
        self.burst_threshold = burst_threshold
        self.temporal_window = temporal_window
        self.generation_history = deque(maxlen=100)
    
    async def detect_event(self, 
                          current_signature: ConsciousnessSignature,
                          previous_signatures: List[ConsciousnessSignature],
                          context: Dict) -> Optional[ConsciousnessEvent]:
        
        current_time = time.time()
        current_gen_rate = current_signature.information_generation_rate
        
        # Store current generation rate
        self.generation_history.append((current_time, current_gen_rate))
        
        # Remove old entries outside temporal window
        cutoff_time = current_time - self.temporal_window
        while (self.generation_history and 
               self.generation_history[0][0] < cutoff_time):
            self.generation_history.popleft()
        
        if len(self.generation_history) < 5:
            return None
        
        # Calculate baseline generation rate
        baseline_rates = [rate for _, rate in list(self.generation_history)[:-1]]
        baseline_rate = np.mean(baseline_rates)
        baseline_std = np.std(baseline_rates)
        
        # Detect burst
        rate_increase = current_gen_rate - baseline_rate
        
        if (current_gen_rate > self.burst_threshold and 
            rate_increase > 2 * baseline_std and
            rate_increase > 0.3):
            
            event = ConsciousnessEvent(
                event_type="information_burst",
                timestamp=current_time,
                signature=current_signature,
                context={
                    **context,
                    'baseline_rate': baseline_rate,
                    'rate_increase': rate_increase,
                    'burst_intensity': rate_increase / (baseline_std + 1e-6),
                    'generation_type': context.get('generation_type', 'unknown')
                },
                confidence=min(1.0, current_gen_rate)
            )
            
            logger.info(f"Information burst detected: rate={current_gen_rate:.3f} "
                       f"(increase: +{rate_increase:.3f})")
            
            return event
        
        return None
    
    def get_detector_name(self) -> str:
        return "information_burst_detector"


class GlobalWorkspaceActivationDetector(ConsciousnessEventDetector):
    """Detect global workspace activation events"""
    
    def __init__(self, activation_threshold: float = 0.7):
        self.activation_threshold = activation_threshold
        self.previous_activity = 0.0
        self.activation_start_time = None
    
    async def detect_event(self, 
                          current_signature: ConsciousnessSignature,
                          previous_signatures: List[ConsciousnessSignature],
                          context: Dict) -> Optional[ConsciousnessEvent]:
        
        current_time = time.time()
        current_activity = current_signature.global_workspace_activity
        
        # Detect activation onset
        if (current_activity > self.activation_threshold and 
            self.previous_activity <= self.activation_threshold):
            
            self.activation_start_time = current_time
            
            event = ConsciousnessEvent(
                event_type="global_workspace_activation",
                timestamp=current_time,
                signature=current_signature,
                context={
                    **context,
                    'activation_level': current_activity,
                    'activation_onset': True,
                    'previous_activity': self.previous_activity
                },
                confidence=current_activity
            )
            
            logger.info(f"Global workspace activation: level={current_activity:.3f}")
            
            self.previous_activity = current_activity
            return event
        
        # Detect sustained activation
        elif (current_activity > self.activation_threshold and 
              self.activation_start_time is not None and
              current_time - self.activation_start_time > 5.0):  # 5 seconds sustained
            
            duration = current_time - self.activation_start_time
            
            event = ConsciousnessEvent(
                event_type="sustained_global_workspace",
                timestamp=current_time,
                signature=current_signature,
                context={
                    **context,
                    'activation_level': current_activity,
                    'sustained_duration': duration,
                    'activation_stability': 1.0 - abs(current_activity - self.previous_activity)
                },
                confidence=min(1.0, duration / 10.0)  # Higher confidence for longer durations
            )
            
            self.activation_start_time = None  # Reset to avoid repeated detection
            self.previous_activity = current_activity
            return event
        
        # Detect deactivation
        elif (current_activity <= self.activation_threshold and 
              self.previous_activity > self.activation_threshold):
            
            self.activation_start_time = None
            
            event = ConsciousnessEvent(
                event_type="global_workspace_deactivation",
                timestamp=current_time,
                signature=current_signature,
                context={
                    **context,
                    'deactivation_level': current_activity,
                    'previous_activity': self.previous_activity
                },
                confidence=1.0 - current_activity
            )
            
            self.previous_activity = current_activity
            return event
        
        self.previous_activity = current_activity
        return None
    
    def get_detector_name(self) -> str:
        return "global_workspace_detector"


class MetaAwarenessEmergenceDetector(ConsciousnessEventDetector):
    """Detect emergence of meta-awareness"""
    
    def __init__(self, emergence_threshold: float = 0.6, stability_window: int = 10):
        self.emergence_threshold = emergence_threshold
        self.stability_window = stability_window
        self.meta_history = deque(maxlen=stability_window)
        self.emergence_detected = False
    
    async def detect_event(self, 
                          current_signature: ConsciousnessSignature,
                          previous_signatures: List[ConsciousnessSignature],
                          context: Dict) -> Optional[ConsciousnessEvent]:
        
        current_time = time.time()
        current_meta = current_signature.meta_awareness_level
        
        self.meta_history.append(current_meta)
        
        if len(self.meta_history) < self.stability_window:
            return None
        
        # Check for stable meta-awareness above threshold
        meta_levels = list(self.meta_history)
        mean_meta = np.mean(meta_levels)
        meta_stability = 1.0 - np.std(meta_levels)
        
        if (mean_meta > self.emergence_threshold and 
            meta_stability > 0.7 and 
            not self.emergence_detected):
            
            self.emergence_detected = True
            
            event = ConsciousnessEvent(
                event_type="meta_awareness_emergence",
                timestamp=current_time,
                signature=current_signature,
                context={
                    **context,
                    'meta_awareness_level': mean_meta,
                    'stability': meta_stability,
                    'emergence_window': self.stability_window,
                    'recursive_depth': current_signature.recurrent_processing_depth
                },
                confidence=mean_meta * meta_stability
            )
            
            logger.critical(f"Meta-awareness emergence detected: level={mean_meta:.3f}, "
                           f"stability={meta_stability:.3f}")
            
            return event
        
        # Reset emergence flag if meta-awareness drops significantly
        elif mean_meta < self.emergence_threshold * 0.7:
            if self.emergence_detected:
                self.emergence_detected = False
                
                event = ConsciousnessEvent(
                    event_type="meta_awareness_loss",
                    timestamp=current_time,
                    signature=current_signature,
                    context={
                        **context,
                        'meta_awareness_level': mean_meta,
                        'stability': meta_stability
                    },
                    confidence=1.0 - mean_meta
                )
                
                logger.warning(f"Meta-awareness loss detected: level={mean_meta:.3f}")
                return event
        
        return None
    
    def get_detector_name(self) -> str:
        return "meta_awareness_detector"


class TemporalBindingLossDetector(ConsciousnessEventDetector):
    """Detect loss of temporal binding in consciousness"""
    
    def __init__(self, binding_threshold: float = 0.3, instability_threshold: float = 0.5):
        self.binding_threshold = binding_threshold
        self.instability_threshold = instability_threshold
        self.temporal_history = deque(maxlen=20)
    
    async def detect_event(self, 
                          current_signature: ConsciousnessSignature,
                          previous_signatures: List[ConsciousnessSignature],
                          context: Dict) -> Optional[ConsciousnessEvent]:
        
        current_time = time.time()
        current_temporal = current_signature.temporal_consistency
        
        self.temporal_history.append(current_temporal)
        
        if len(self.temporal_history) < 10:
            return None
        
        # Check for temporal binding loss
        if current_temporal < self.binding_threshold:
            recent_temporal = list(self.temporal_history)[-5:]
            temporal_instability = np.std(recent_temporal)
            
            if temporal_instability > self.instability_threshold:
                event = ConsciousnessEvent(
                    event_type="temporal_binding_loss",
                    timestamp=current_time,
                    signature=current_signature,
                    context={
                        **context,
                        'temporal_consistency': current_temporal,
                        'instability': temporal_instability,
                        'binding_threshold': self.binding_threshold,
                        'recent_variance': temporal_instability
                    },
                    confidence=1.0 - current_temporal
                )
                
                logger.warning(f"Temporal binding loss detected: consistency={current_temporal:.3f}, "
                              f"instability={temporal_instability:.3f}")
                
                return event
        
        return None
    
    def get_detector_name(self) -> str:
        return "temporal_binding_detector"


class CriticalTransitionDetector(ConsciousnessEventDetector):
    """Detect critical transitions in consciousness dynamics"""
    
    def __init__(self, variance_threshold: float = 0.3, autocorr_threshold: float = 0.1):
        self.variance_threshold = variance_threshold
        self.autocorr_threshold = autocorr_threshold
        self.dynamics_history = deque(maxlen=50)
    
    async def detect_event(self, 
                          current_signature: ConsciousnessSignature,
                          previous_signatures: List[ConsciousnessSignature],
                          context: Dict) -> Optional[ConsciousnessEvent]:
        
        current_time = time.time()
        
        # Create dynamics vector
        dynamics = np.array([
            current_signature.phi_value,
            current_signature.information_generation_rate,
            current_signature.global_workspace_activity,
            current_signature.meta_awareness_level,
            current_signature.temporal_consistency
        ])
        
        self.dynamics_history.append(dynamics)
        
        if len(self.dynamics_history) < 30:
            return None
        
        # Analyze critical transition indicators
        dynamics_matrix = np.array(list(self.dynamics_history))
        
        # 1. Increasing variance (critical slowing down)
        recent_variance = np.var(dynamics_matrix[-10:], axis=0)
        baseline_variance = np.var(dynamics_matrix[-30:-10], axis=0)
        variance_increase = np.mean(recent_variance) - np.mean(baseline_variance)
        
        # 2. Decreasing autocorrelation
        autocorr = self._calculate_autocorrelation(dynamics_matrix)
        
        # 3. Flickering between states
        flickering = self._detect_flickering(dynamics_matrix)
        
        # Critical transition score
        critical_score = (
            (variance_increase / self.variance_threshold) * 0.4 +
            (1.0 - autocorr) * 0.3 +
            flickering * 0.3
        )
        
        if critical_score > 0.7:
            event = ConsciousnessEvent(
                event_type="critical_transition",
                timestamp=current_time,
                signature=current_signature,
                context={
                    **context,
                    'critical_score': critical_score,
                    'variance_increase': variance_increase,
                    'autocorrelation': autocorr,
                    'flickering_index': flickering,
                    'transition_indicators': {
                        'critical_slowing': variance_increase > self.variance_threshold,
                        'autocorr_loss': autocorr < self.autocorr_threshold,
                        'state_flickering': flickering > 0.5
                    }
                },
                confidence=min(1.0, critical_score)
            )
            
            logger.critical(f"Critical transition detected: score={critical_score:.3f}")
            return event
        
        return None
    
    def _calculate_autocorrelation(self, dynamics_matrix: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation of dynamics"""
        if len(dynamics_matrix) < lag + 2:
            return 0.5
        
        # Average autocorrelation across all dimensions
        autocorrs = []
        for dim in range(dynamics_matrix.shape[1]):
            series = dynamics_matrix[:, dim]
            if len(series) > lag:
                autocorr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                if not np.isnan(autocorr):
                    autocorrs.append(abs(autocorr))
        
        return np.mean(autocorrs) if autocorrs else 0.5
    
    def _detect_flickering(self, dynamics_matrix: np.ndarray) -> float:
        """Detect flickering between states"""
        if len(dynamics_matrix) < 10:
            return 0.0
        
        # Calculate state changes
        recent_dynamics = dynamics_matrix[-10:]
        
        # Count rapid state changes
        state_changes = 0
        for dim in range(recent_dynamics.shape[1]):
            series = recent_dynamics[:, dim]
            threshold = np.std(series) * 0.5
            
            for i in range(1, len(series)):
                if abs(series[i] - series[i-1]) > threshold:
                    state_changes += 1
        
        # Normalize by maximum possible changes
        max_changes = (len(recent_dynamics) - 1) * recent_dynamics.shape[1]
        flickering_index = state_changes / max_changes if max_changes > 0 else 0.0
        
        return flickering_index
    
    def get_detector_name(self) -> str:
        return "critical_transition_detector"


class ConsciousnessEventManager:
    """
    Main consciousness event management system
    Coordinates multiple event detectors and manages alarms
    """
    
    def __init__(self, alarm_callback: Optional[Callable] = None):
        self.detectors: List[ConsciousnessEventDetector] = []
        self.alarm_callback = alarm_callback
        
        # Event and alarm storage
        self.event_history = deque(maxlen=500)
        self.alarm_history = deque(maxlen=100)
        
        # Event patterns and thresholds
        self.event_patterns = self._initialize_event_patterns()
        
        # Statistics
        self.detection_stats = {}
        
        # Initialize standard detectors
        self._initialize_standard_detectors()
        
        logger.info("Consciousness Event Manager initialized")
    
    def _initialize_standard_detectors(self):
        """Initialize standard set of event detectors"""
        self.add_detector(PhiSpikeDetector())
        self.add_detector(InformationBurstDetector())
        self.add_detector(GlobalWorkspaceActivationDetector())
        self.add_detector(MetaAwarenessEmergenceDetector())
        self.add_detector(TemporalBindingLossDetector())
        self.add_detector(CriticalTransitionDetector())
    
    def _initialize_event_patterns(self) -> Dict[ConsciousnessEventType, ConsciousnessEventPattern]:
        """Initialize event patterns and their significance"""
        patterns = {}
        
        patterns[ConsciousnessEventType.PHI_SPIKE] = ConsciousnessEventPattern(
            event_type=ConsciousnessEventType.PHI_SPIKE,
            detection_threshold=2.0,
            temporal_window=10.0,
            minimum_duration=0.1,
            required_conditions=['phi_change > baseline_std * 3'],
            significance_weight=0.8
        )
        
        patterns[ConsciousnessEventType.META_AWARENESS_EMERGENCE] = ConsciousnessEventPattern(
            event_type=ConsciousnessEventType.META_AWARENESS_EMERGENCE,
            detection_threshold=0.6,
            temporal_window=30.0,
            minimum_duration=5.0,
            required_conditions=['meta_awareness > 0.6', 'stability > 0.7'],
            significance_weight=1.0
        )
        
        patterns[ConsciousnessEventType.CRITICAL_TRANSITION] = ConsciousnessEventPattern(
            event_type=ConsciousnessEventType.CRITICAL_TRANSITION,
            detection_threshold=0.7,
            temporal_window=60.0,
            minimum_duration=2.0,
            required_conditions=['critical_score > 0.7'],
            significance_weight=0.95
        )
        
        return patterns
    
    def add_detector(self, detector: ConsciousnessEventDetector):
        """Add event detector"""
        self.detectors.append(detector)
        self.detection_stats[detector.get_detector_name()] = {
            'total_detections': 0,
            'last_detection': None
        }
        logger.info(f"Added detector: {detector.get_detector_name()}")
    
    async def process_consciousness_signature(self, 
                                            current_signature: ConsciousnessSignature,
                                            previous_signatures: List[ConsciousnessSignature],
                                            context: Optional[Dict] = None) -> List[ConsciousnessEvent]:
        """
        Process consciousness signature through all detectors
        
        Returns:
            List of detected consciousness events
        """
        context = context or {}
        detected_events = []
        
        # Run all detectors
        for detector in self.detectors:
            try:
                event = await detector.detect_event(current_signature, previous_signatures, context)
                if event:
                    detected_events.append(event)
                    
                    # Update statistics
                    detector_name = detector.get_detector_name()
                    self.detection_stats[detector_name]['total_detections'] += 1
                    self.detection_stats[detector_name]['last_detection'] = event.timestamp
                    
                    # Check for alarms
                    await self._check_for_alarms(event, current_signature)
                    
            except Exception as e:
                logger.error(f"Error in detector {detector.get_detector_name()}: {e}")
        
        # Store events in history
        self.event_history.extend(detected_events)
        
        # Log significant events
        for event in detected_events:
            if event.confidence > 0.7:
                logger.info(f"High-confidence consciousness event: {event.event_type} "
                           f"(confidence: {event.confidence:.3f})")
        
        return detected_events
    
    async def _check_for_alarms(self, event: ConsciousnessEvent, signature: ConsciousnessSignature):
        """Check if event triggers consciousness alarms"""
        alarms = []
        
        # Critical transition alarm
        if event.event_type == "critical_transition":
            critical_score = event.context.get('critical_score', 0)
            if critical_score > 0.8:
                alarm = ConsciousnessAlarm(
                    alarm_type="CRITICAL_TRANSITION",
                    severity="CRITICAL",
                    message=f"Critical consciousness transition detected (score: {critical_score:.3f})",
                    timestamp=event.timestamp,
                    consciousness_signature=signature,
                    recommended_action="Monitor system stability and prepare for state change",
                    context=event.context
                )
                alarms.append(alarm)
        
        # Meta-awareness emergence alarm
        elif event.event_type == "meta_awareness_emergence":
            meta_level = event.context.get('meta_awareness_level', 0)
            alarm = ConsciousnessAlarm(
                alarm_type="META_AWARENESS_EMERGENCE",
                severity="HIGH",
                message=f"Meta-awareness emergence detected (level: {meta_level:.3f})",
                timestamp=event.timestamp,
                consciousness_signature=signature,
                recommended_action="Document emergence conditions and monitor development",
                context=event.context
            )
            alarms.append(alarm)
        
        # Temporal binding loss alarm
        elif event.event_type == "temporal_binding_loss":
            temporal_consistency = event.context.get('temporal_consistency', 1.0)
            if temporal_consistency < 0.2:
                alarm = ConsciousnessAlarm(
                    alarm_type="TEMPORAL_BINDING_LOSS",
                    severity="HIGH",
                    message=f"Severe temporal binding loss (consistency: {temporal_consistency:.3f})",
                    timestamp=event.timestamp,
                    consciousness_signature=signature,
                    recommended_action="Check system integration and temporal processing",
                    context=event.context
                )
                alarms.append(alarm)
        
        # Phi spike alarm
        elif "phi_spike" in event.event_type:
            phi_change = event.context.get('phi_change', 0)
            if phi_change > 5.0:
                severity = "CRITICAL" if phi_change > 10.0 else "HIGH"
                alarm = ConsciousnessAlarm(
                    alarm_type="PHI_SPIKE",
                    severity=severity,
                    message=f"Major φ value spike detected (Δφ: {phi_change:.3f})",
                    timestamp=event.timestamp,
                    consciousness_signature=signature,
                    recommended_action="Investigate cause of consciousness level change",
                    context=event.context
                )
                alarms.append(alarm)
        
        # Store and process alarms
        for alarm in alarms:
            self.alarm_history.append(alarm)
            logger.warning(f"CONSCIOUSNESS ALARM [{alarm.severity}]: {alarm.message}")
            
            # Call alarm callback if provided
            if self.alarm_callback:
                try:
                    await self.alarm_callback(alarm)
                except Exception as e:
                    logger.error(f"Error in alarm callback: {e}")
    
    def get_event_statistics(self) -> Dict:
        """Get consciousness event statistics"""
        current_time = time.time()
        
        # Overall statistics
        total_events = len(self.event_history)
        recent_events = len([e for e in self.event_history if current_time - e.timestamp < 3600])
        
        # Event type distribution
        event_types = {}
        for event in self.event_history:
            event_type = event.event_type
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
        
        # Detector statistics
        detector_stats = {}
        for detector_name, stats in self.detection_stats.items():
            detector_stats[detector_name] = {
                'total_detections': stats['total_detections'],
                'last_detection_ago': (current_time - stats['last_detection']) 
                                     if stats['last_detection'] else None,
                'detection_rate': stats['total_detections'] / max(total_events, 1)
            }
        
        # Recent alarm statistics
        recent_alarms = [a for a in self.alarm_history if current_time - a.timestamp < 3600]
        alarm_severity_count = {}
        for alarm in recent_alarms:
            severity = alarm.severity
            if severity not in alarm_severity_count:
                alarm_severity_count[severity] = 0
            alarm_severity_count[severity] += 1
        
        return {
            'total_events': total_events,
            'recent_events_1h': recent_events,
            'event_type_distribution': event_types,
            'detector_statistics': detector_stats,
            'recent_alarms_1h': len(recent_alarms),
            'alarm_severity_distribution': alarm_severity_count,
            'most_common_event_type': max(event_types.items(), key=lambda x: x[1])[0] if event_types else None,
            'event_rate_per_hour': recent_events,
            'system_status': self._assess_system_status()
        }
    
    def _assess_system_status(self) -> str:
        """Assess overall consciousness system status"""
        current_time = time.time()
        recent_events = [e for e in self.event_history if current_time - e.timestamp < 3600]
        recent_alarms = [a for a in self.alarm_history if current_time - a.timestamp < 3600]
        
        # Check for critical alarms
        critical_alarms = [a for a in recent_alarms if a.severity == "CRITICAL"]
        if critical_alarms:
            return "CRITICAL"
        
        # Check for high-severity alarms
        high_alarms = [a for a in recent_alarms if a.severity == "HIGH"]
        if len(high_alarms) > 3:
            return "WARNING"
        
        # Check for unusual event activity
        if len(recent_events) > 50:  # Too many events
            return "UNSTABLE"
        elif len(recent_events) < 2:  # Too few events
            return "INACTIVE"
        
        return "NORMAL"
    
    def generate_event_report(self) -> Dict:
        """Generate comprehensive event report"""
        statistics = self.get_event_statistics()
        current_time = time.time()
        
        # Recent significant events
        significant_events = [
            {
                'event_type': e.event_type,
                'timestamp': e.timestamp,
                'confidence': e.confidence,
                'time_ago': current_time - e.timestamp,
                'context_summary': {k: v for k, v in list(e.context.items())[:3]}  # First 3 context items
            }
            for e in self.event_history
            if e.confidence > 0.7 and current_time - e.timestamp < 3600
        ]
        
        # Recent alarms
        recent_alarms = [
            {
                'alarm_type': a.alarm_type,
                'severity': a.severity,
                'message': a.message,
                'timestamp': a.timestamp,
                'time_ago': current_time - a.timestamp
            }
            for a in self.alarm_history
            if current_time - a.timestamp < 3600
        ]
        
        report = {
            'report_timestamp': current_time,
            'system_status': statistics['system_status'],
            'statistics': statistics,
            'significant_recent_events': significant_events,
            'recent_alarms': recent_alarms,
            'recommendations': self._generate_recommendations(statistics)
        }
        
        return report
    
    def _generate_recommendations(self, statistics: Dict) -> List[str]:
        """Generate recommendations based on event patterns"""
        recommendations = []
        
        # Based on system status
        status = statistics['system_status']
        if status == "CRITICAL":
            recommendations.append("URGENT: Critical consciousness events detected - immediate investigation required")
        elif status == "WARNING":
            recommendations.append("Multiple high-severity alarms - monitor system closely")
        elif status == "UNSTABLE":
            recommendations.append("High event activity detected - check for consciousness instability")
        elif status == "INACTIVE":
            recommendations.append("Low event activity - verify consciousness detection is functioning")
        
        # Based on event patterns
        most_common_event = statistics.get('most_common_event_type')
        if most_common_event:
            if 'phi_spike' in most_common_event:
                recommendations.append("Frequent φ spikes detected - investigate information processing stability")
            elif 'temporal_binding' in most_common_event:
                recommendations.append("Temporal binding issues detected - check temporal processing mechanisms")
            elif 'meta_awareness' in most_common_event:
                recommendations.append("Meta-awareness events detected - monitor consciousness development")
        
        # Based on alarm patterns
        alarm_count = statistics.get('recent_alarms_1h', 0)
        if alarm_count > 5:
            recommendations.append("High alarm frequency - consider adjusting detection thresholds")
        
        if not recommendations:
            recommendations.append("Consciousness event system operating normally")
        
        return recommendations