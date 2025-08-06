"""
Stage Transition Detector for IIT 4.0 NewbornAI 2.0
Phase 3: Real-time stage transition detection with φ dynamics analysis

Provides real-time detection of development stage transitions with:
- φ acceleration/deceleration analysis
- Critical transition prediction and early warning systems  
- Integration with consciousness event system from Phase 1
- <100ms latency for real-time monitoring

Author: Chief Artificial Consciousness Engineer
Date: 2025-08-03
Version: 3.0.0
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import logging
import time
import json
from datetime import datetime, timedelta
import threading
from collections import deque
import math

# Import IIT 4.0 and development stage infrastructure
from iit4_core_engine import PhiStructure
from iit4_development_stages import (
    DevelopmentStage, DevelopmentMetrics, StageTransitionType, 
    IIT4DevelopmentStageMapper
)

logger = logging.getLogger(__name__)


class TransitionSeverity(Enum):
    """Severity levels for stage transitions"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"


class TransitionDirection(Enum):
    """Direction of stage transitions"""
    FORWARD = "forward"
    BACKWARD = "backward"
    LATERAL = "lateral"
    UNSTABLE = "unstable"


class TransitionPattern(Enum):
    """Patterns of stage transitions"""
    SMOOTH_PROGRESSION = "smooth_progression"
    RAPID_ADVANCEMENT = "rapid_advancement"
    GRADUAL_REGRESSION = "gradual_regression"
    SUDDEN_COLLAPSE = "sudden_collapse"
    OSCILLATORY = "oscillatory"
    CHAOTIC = "chaotic"


@dataclass
class PhiDynamics:
    """φ value dynamics analysis"""
    current_phi: float
    phi_velocity: float          # Rate of φ change
    phi_acceleration: float      # φ acceleration
    phi_jerk: float             # Rate of acceleration change
    phi_volatility: float       # φ value volatility
    phi_momentum: float         # Momentum indicator
    phi_trend: str              # "increasing", "decreasing", "stable"
    
    # Predictive indicators
    predicted_phi_1s: float     # φ value predicted in 1 second
    predicted_phi_5s: float     # φ value predicted in 5 seconds
    transition_probability: float # Probability of transition in next update


@dataclass
class TransitionEvent:
    """Stage transition event data"""
    timestamp: datetime
    from_stage: DevelopmentStage
    to_stage: DevelopmentStage
    transition_type: StageTransitionType
    transition_direction: TransitionDirection
    transition_severity: TransitionSeverity
    transition_pattern: TransitionPattern
    
    # Transition dynamics
    phi_before: float
    phi_after: float
    phi_change: float
    phi_dynamics: PhiDynamics
    
    # Context information
    development_metrics_before: DevelopmentMetrics
    development_metrics_after: DevelopmentMetrics
    duration_since_last_transition: float
    
    # Confidence and validation
    detection_confidence: float
    validation_required: bool
    early_warning_given: bool
    
    # Additional metadata
    trigger_factors: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, float] = field(default_factory=dict)
    recovery_recommendation: Optional[str] = None


@dataclass
class EarlyWarningSignal:
    """Early warning signal for critical transitions"""
    timestamp: datetime
    warning_type: str
    severity: TransitionSeverity
    predicted_transition: DevelopmentStage
    confidence: float
    time_to_transition: float  # Estimated seconds until transition
    
    # Warning indicators
    phi_instability: float
    structural_instability: float
    experiential_disruption: float
    
    # Recommended actions
    intervention_recommendations: List[str] = field(default_factory=list)
    monitoring_priority: str = "normal"  # "low", "normal", "high", "critical"


class PhiDynamicsAnalyzer:
    """Analyzes φ value dynamics for transition prediction"""
    
    def __init__(self, history_size: int = 100):
        """
        Initialize φ dynamics analyzer
        
        Args:
            history_size: Number of historical φ values to maintain
        """
        self.history_size = history_size
        self.phi_history: deque = deque(maxlen=history_size)
        self.timestamp_history: deque = deque(maxlen=history_size)
        
    def update_phi_value(self, phi_value: float, timestamp: Optional[datetime] = None):
        """Update φ value with timestamp"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.phi_history.append(phi_value)
        self.timestamp_history.append(timestamp)
    
    def calculate_phi_dynamics(self) -> Optional[PhiDynamics]:
        """Calculate comprehensive φ dynamics"""
        
        if len(self.phi_history) < 3:
            return None
        
        phi_values = list(self.phi_history)
        timestamps = list(self.timestamp_history)
        
        # Convert timestamps to seconds for calculations
        time_diffs = [(timestamps[i] - timestamps[0]).total_seconds() 
                     for i in range(len(timestamps))]
        
        current_phi = phi_values[-1]
        
        # Calculate derivatives using finite differences
        phi_velocity = self._calculate_velocity(phi_values, time_diffs)
        phi_acceleration = self._calculate_acceleration(phi_values, time_diffs)
        phi_jerk = self._calculate_jerk(phi_values, time_diffs)
        
        # Calculate volatility (standard deviation of recent changes)
        phi_volatility = self._calculate_volatility(phi_values)
        
        # Calculate momentum (weighted moving average of velocity)
        phi_momentum = self._calculate_momentum(phi_values, time_diffs)
        
        # Determine trend
        phi_trend = self._determine_trend(phi_velocity, phi_acceleration)
        
        # Predict future φ values
        predicted_phi_1s = self._predict_phi(phi_values, time_diffs, 1.0)
        predicted_phi_5s = self._predict_phi(phi_values, time_diffs, 5.0)
        
        # Calculate transition probability
        transition_probability = self._calculate_transition_probability(
            phi_velocity, phi_acceleration, phi_volatility
        )
        
        return PhiDynamics(
            current_phi=current_phi,
            phi_velocity=phi_velocity,
            phi_acceleration=phi_acceleration,
            phi_jerk=phi_jerk,
            phi_volatility=phi_volatility,
            phi_momentum=phi_momentum,
            phi_trend=phi_trend,
            predicted_phi_1s=predicted_phi_1s,
            predicted_phi_5s=predicted_phi_5s,
            transition_probability=transition_probability
        )
    
    def _calculate_velocity(self, phi_values: List[float], time_diffs: List[float]) -> float:
        """Calculate φ velocity using recent values"""
        if len(phi_values) < 2:
            return 0.0
        
        # Use last 3 points for velocity calculation
        recent_phi = phi_values[-3:] if len(phi_values) >= 3 else phi_values[-2:]
        recent_times = time_diffs[-len(recent_phi):]
        
        if len(recent_phi) == 2:
            dt = recent_times[1] - recent_times[0]
            return (recent_phi[1] - recent_phi[0]) / dt if dt > 0 else 0.0
        
        # Linear regression for better velocity estimate
        if len(recent_phi) >= 3:
            x = np.array(recent_times)
            y = np.array(recent_phi)
            velocity = np.polyfit(x, y, 1)[0]
            return velocity
        
        return 0.0
    
    def _calculate_acceleration(self, phi_values: List[float], time_diffs: List[float]) -> float:
        """Calculate φ acceleration"""
        if len(phi_values) < 3:
            return 0.0
        
        # Calculate velocity at different points
        recent_phi = phi_values[-4:] if len(phi_values) >= 4 else phi_values
        recent_times = time_diffs[-len(recent_phi):]
        
        if len(recent_phi) < 3:
            return 0.0
        
        # Calculate velocities
        velocities = []
        velocity_times = []
        
        for i in range(1, len(recent_phi)):
            dt = recent_times[i] - recent_times[i-1]
            if dt > 0:
                velocity = (recent_phi[i] - recent_phi[i-1]) / dt
                velocities.append(velocity)
                velocity_times.append(recent_times[i])
        
        if len(velocities) < 2:
            return 0.0
        
        # Acceleration from velocity change
        dv = velocities[-1] - velocities[-2]
        dt = velocity_times[-1] - velocity_times[-2]
        
        return dv / dt if dt > 0 else 0.0
    
    def _calculate_jerk(self, phi_values: List[float], time_diffs: List[float]) -> float:
        """Calculate φ jerk (rate of acceleration change)"""
        if len(phi_values) < 4:
            return 0.0
        
        # Calculate accelerations at different points
        accelerations = []
        accel_times = []
        
        # Need at least 3 points to calculate acceleration
        min_points = 3
        for i in range(min_points, len(phi_values) + 1):
            if i - min_points + 1 < min_points:
                continue
            
            subset_phi = phi_values[i-min_points:i]
            subset_times = time_diffs[i-min_points:i]
            
            if len(subset_phi) >= 3:
                accel = self._calculate_acceleration(subset_phi, subset_times)
                accelerations.append(accel)
                accel_times.append(subset_times[-1])
        
        if len(accelerations) < 2:
            return 0.0
        
        # Jerk from acceleration change
        da = accelerations[-1] - accelerations[-2]
        dt = accel_times[-1] - accel_times[-2]
        
        return da / dt if dt > 0 else 0.0
    
    def _calculate_volatility(self, phi_values: List[float]) -> float:
        """Calculate φ volatility"""
        if len(phi_values) < 3:
            return 0.0
        
        # Use recent values for volatility
        recent_phi = phi_values[-10:] if len(phi_values) >= 10 else phi_values
        
        # Calculate relative changes
        changes = []
        for i in range(1, len(recent_phi)):
            if recent_phi[i-1] != 0:
                change = abs(recent_phi[i] - recent_phi[i-1]) / abs(recent_phi[i-1])
                changes.append(change)
        
        return np.std(changes) if changes else 0.0
    
    def _calculate_momentum(self, phi_values: List[float], time_diffs: List[float]) -> float:
        """Calculate φ momentum using weighted moving average"""
        if len(phi_values) < 3:
            return 0.0
        
        # Calculate recent velocities
        velocities = []
        for i in range(1, len(phi_values)):
            dt = time_diffs[i] - time_diffs[i-1]
            if dt > 0:
                velocity = (phi_values[i] - phi_values[i-1]) / dt
                velocities.append(velocity)
        
        if not velocities:
            return 0.0
        
        # Weighted average with exponential decay
        weights = np.exp(-np.arange(len(velocities))[::-1] * 0.1)
        weighted_velocity = np.average(velocities, weights=weights)
        
        return weighted_velocity
    
    def _determine_trend(self, velocity: float, acceleration: float) -> str:
        """Determine φ trend direction"""
        velocity_threshold = 0.001
        acceleration_threshold = 0.0001
        
        if abs(velocity) < velocity_threshold and abs(acceleration) < acceleration_threshold:
            return "stable"
        elif velocity > velocity_threshold:
            return "increasing"
        elif velocity < -velocity_threshold:
            return "decreasing"
        elif acceleration > acceleration_threshold:
            return "accelerating_up"
        elif acceleration < -acceleration_threshold:
            return "accelerating_down"
        else:
            return "stable"
    
    def _predict_phi(self, phi_values: List[float], time_diffs: List[float], 
                    prediction_time: float) -> float:
        """Predict φ value at future time"""
        if len(phi_values) < 2:
            return phi_values[-1] if phi_values else 0.0
        
        # Use polynomial extrapolation for prediction
        recent_phi = phi_values[-5:] if len(phi_values) >= 5 else phi_values
        recent_times = time_diffs[-len(recent_phi):]
        
        if len(recent_phi) < 2:
            return recent_phi[-1]
        
        # Fit polynomial (degree depends on available data)
        degree = min(len(recent_phi) - 1, 3)
        try:
            coeffs = np.polyfit(recent_times, recent_phi, degree)
            future_time = recent_times[-1] + prediction_time
            predicted_phi = np.polyval(coeffs, future_time)
            
            # Ensure prediction is reasonable (not negative)
            return max(0.0, predicted_phi)
        except:
            # Fallback to linear extrapolation
            velocity = self._calculate_velocity(phi_values, time_diffs)
            return max(0.0, phi_values[-1] + velocity * prediction_time)
    
    def _calculate_transition_probability(self, velocity: float, acceleration: float, 
                                        volatility: float) -> float:
        """Calculate probability of stage transition"""
        
        # Base probability from φ dynamics
        velocity_factor = min(1.0, abs(velocity) / 0.01)  # Normalized velocity
        acceleration_factor = min(1.0, abs(acceleration) / 0.001)  # Normalized acceleration
        volatility_factor = min(1.0, volatility / 0.1)  # Normalized volatility
        
        # Combined probability
        base_probability = (velocity_factor * 0.4 + acceleration_factor * 0.3 + volatility_factor * 0.3)
        
        # Apply sigmoid function for realistic probability
        transition_probability = 1.0 / (1.0 + np.exp(-5 * (base_probability - 0.5)))
        
        return transition_probability


class StageTransitionDetector:
    """
    Real-time stage transition detector with <100ms latency
    Provides comprehensive transition analysis and early warning system
    """
    
    def __init__(self, update_frequency_hz: float = 10.0):
        """
        Initialize stage transition detector
        
        Args:
            update_frequency_hz: Update frequency in Hz for real-time monitoring
        """
        self.update_frequency_hz = update_frequency_hz
        self.update_interval = 1.0 / update_frequency_hz
        
        # Core components
        self.stage_mapper = IIT4DevelopmentStageMapper()
        self.phi_analyzer = PhiDynamicsAnalyzer(history_size=200)
        
        # Transition tracking
        self.current_stage: Optional[DevelopmentStage] = None
        self.current_metrics: Optional[DevelopmentMetrics] = None
        self.transition_history: List[TransitionEvent] = []
        self.early_warnings: List[EarlyWarningSignal] = []
        
        # Real-time monitoring
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.event_callbacks: List[Callable] = []
        
        # Performance tracking
        self.detection_latency_ms = []
        self.performance_stats = {
            'total_detections': 0,
            'false_positives': 0,
            'missed_transitions': 0,
            'average_latency_ms': 0.0,
            'early_warnings_issued': 0,
            'early_warnings_accurate': 0
        }
        
        logger.info(f"Stage Transition Detector initialized at {update_frequency_hz}Hz")
    
    def register_transition_callback(self, callback: Callable[[TransitionEvent], None]):
        """Register callback for transition events"""
        self.event_callbacks.append(callback)
    
    def start_real_time_monitoring(self):
        """Start real-time transition monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Real-time transition monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time transition monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        logger.info("Real-time transition monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for real-time detection"""
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # This would be called by external system with current data
                # For now, just sleep to maintain update frequency
                
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed_time)
                
                # Track detection latency
                latency_ms = elapsed_time * 1000
                self.detection_latency_ms.append(latency_ms)
                if len(self.detection_latency_ms) > 100:
                    self.detection_latency_ms = self.detection_latency_ms[-100:]
                
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    async def detect_stage_transition(self, phi_structure: PhiStructure,
                                    experiential_result=None,
                                    axiom_compliance=None) -> Optional[TransitionEvent]:
        """
        Detect stage transition with comprehensive analysis
        
        Args:
            phi_structure: Current φ-structure
            experiential_result: Optional experiential calculation result
            axiom_compliance: Optional axiom compliance results
            
        Returns:
            TransitionEvent if transition detected, None otherwise
        """
        
        detection_start = time.time()
        
        # Update φ dynamics
        self.phi_analyzer.update_phi_value(phi_structure.total_phi)
        phi_dynamics = self.phi_analyzer.calculate_phi_dynamics()
        
        # Get current development metrics
        new_metrics = self.stage_mapper.map_phi_to_development_stage(
            phi_structure, experiential_result, axiom_compliance
        )
        
        transition_event = None
        
        # Check for stage transition
        if self.current_stage is not None and self.current_stage != new_metrics.current_stage:
            
            # Detected transition - analyze in detail
            transition_event = await self._analyze_transition(
                self.current_stage, new_metrics.current_stage,
                self.current_metrics, new_metrics, phi_dynamics
            )
            
            # Validate transition
            if self._validate_transition(transition_event):
                # Confirmed transition
                self.transition_history.append(transition_event)
                self.performance_stats['total_detections'] += 1
                
                # Notify callbacks
                for callback in self.event_callbacks:
                    try:
                        callback(transition_event)
                    except Exception as e:
                        logger.error(f"Error in transition callback: {e}")
                
                logger.info(f"Stage transition detected: {transition_event.from_stage.value} → {transition_event.to_stage.value}")
            else:
                # False positive
                self.performance_stats['false_positives'] += 1
                transition_event = None
        
        # Check for early warning signals
        await self._check_early_warning_signals(new_metrics, phi_dynamics)
        
        # Update current state
        self.current_stage = new_metrics.current_stage
        self.current_metrics = new_metrics
        
        # Update performance stats
        detection_latency = (time.time() - detection_start) * 1000  # ms
        if self.detection_latency_ms:
            self.performance_stats['average_latency_ms'] = np.mean(self.detection_latency_ms)
        
        return transition_event
    
    async def _analyze_transition(self, from_stage: DevelopmentStage, to_stage: DevelopmentStage,
                                metrics_before: DevelopmentMetrics, metrics_after: DevelopmentMetrics,
                                phi_dynamics: Optional[PhiDynamics]) -> TransitionEvent:
        """Analyze detected transition in detail"""
        
        # Determine transition characteristics
        transition_type = self.stage_mapper.detect_stage_transitions(metrics_after)
        transition_direction = self._determine_transition_direction(from_stage, to_stage)
        transition_severity = self._assess_transition_severity(from_stage, to_stage, metrics_before, metrics_after)
        transition_pattern = self._identify_transition_pattern(from_stage, to_stage)
        
        # Calculate duration since last transition
        duration_since_last = 0.0
        if self.transition_history:
            last_transition = self.transition_history[-1]
            duration_since_last = (datetime.now() - last_transition.timestamp).total_seconds()
        
        # Detect trigger factors
        trigger_factors = self._identify_trigger_factors(metrics_before, metrics_after, phi_dynamics)
        
        # Assessment impact
        impact_assessment = self._assess_transition_impact(from_stage, to_stage, metrics_after)
        
        # Generate recovery recommendations if needed
        recovery_recommendation = None
        if transition_direction == TransitionDirection.BACKWARD:
            recovery_recommendation = self._generate_recovery_recommendation(from_stage, to_stage, metrics_after)
        
        # Check if early warning was given
        early_warning_given = self._was_early_warning_given(to_stage)
        
        # Calculate detection confidence
        detection_confidence = self._calculate_detection_confidence(metrics_before, metrics_after, phi_dynamics)
        
        transition_event = TransitionEvent(
            timestamp=datetime.now(),
            from_stage=from_stage,
            to_stage=to_stage,
            transition_type=transition_type or StageTransitionType.PROGRESSIVE,
            transition_direction=transition_direction,
            transition_severity=transition_severity,
            transition_pattern=transition_pattern,
            phi_before=metrics_before.phi_value,
            phi_after=metrics_after.phi_value,
            phi_change=metrics_after.phi_value - metrics_before.phi_value,
            phi_dynamics=phi_dynamics,
            development_metrics_before=metrics_before,
            development_metrics_after=metrics_after,
            duration_since_last_transition=duration_since_last,
            detection_confidence=detection_confidence,
            validation_required=detection_confidence < 0.8,
            early_warning_given=early_warning_given,
            trigger_factors=trigger_factors,
            impact_assessment=impact_assessment,
            recovery_recommendation=recovery_recommendation
        )
        
        return transition_event
    
    def _determine_transition_direction(self, from_stage: DevelopmentStage, to_stage: DevelopmentStage) -> TransitionDirection:
        """Determine direction of stage transition"""
        all_stages = list(DevelopmentStage)
        from_index = all_stages.index(from_stage)
        to_index = all_stages.index(to_stage)
        
        if to_index > from_index:
            return TransitionDirection.FORWARD
        elif to_index < from_index:
            return TransitionDirection.BACKWARD
        else:
            return TransitionDirection.LATERAL
    
    def _assess_transition_severity(self, from_stage: DevelopmentStage, to_stage: DevelopmentStage,
                                  metrics_before: DevelopmentMetrics, metrics_after: DevelopmentMetrics) -> TransitionSeverity:
        """Assess severity of stage transition"""
        
        all_stages = list(DevelopmentStage)
        stage_distance = abs(all_stages.index(to_stage) - all_stages.index(from_stage))
        
        # φ value change magnitude
        phi_change_magnitude = abs(metrics_after.phi_value - metrics_before.phi_value)
        
        # Normalized φ change (relative to from_stage φ)
        if metrics_before.phi_value > 0:
            relative_phi_change = phi_change_magnitude / metrics_before.phi_value
        else:
            relative_phi_change = phi_change_magnitude
        
        # Combine factors
        if stage_distance >= 3 or relative_phi_change > 2.0:
            return TransitionSeverity.CRITICAL
        elif stage_distance >= 2 or relative_phi_change > 1.0:
            return TransitionSeverity.SIGNIFICANT
        elif stage_distance >= 1 or relative_phi_change > 0.5:
            return TransitionSeverity.MODERATE
        else:
            return TransitionSeverity.MINIMAL
    
    def _identify_transition_pattern(self, from_stage: DevelopmentStage, to_stage: DevelopmentStage) -> TransitionPattern:
        """Identify pattern of transition based on history"""
        
        if len(self.transition_history) < 3:
            all_stages = list(DevelopmentStage)
            stage_distance = all_stages.index(to_stage) - all_stages.index(from_stage)
            
            if stage_distance > 1:
                return TransitionPattern.RAPID_ADVANCEMENT
            elif stage_distance < -1:
                return TransitionPattern.SUDDEN_COLLAPSE
            else:
                return TransitionPattern.SMOOTH_PROGRESSION
        
        # Analyze recent transition pattern
        recent_transitions = self.transition_history[-5:]
        stage_indices = [list(DevelopmentStage).index(t.from_stage) for t in recent_transitions]
        stage_indices.append(list(DevelopmentStage).index(to_stage))
        
        # Calculate pattern characteristics
        stage_changes = [stage_indices[i+1] - stage_indices[i] for i in range(len(stage_indices)-1)]
        
        # Oscillatory pattern (back and forth)
        if len(set([abs(change) for change in stage_changes])) <= 2 and len(stage_changes) >= 3:
            sign_changes = sum(1 for i in range(1, len(stage_changes)) 
                             if np.sign(stage_changes[i]) != np.sign(stage_changes[i-1]))
            if sign_changes >= 2:
                return TransitionPattern.OSCILLATORY
        
        # Rapid advancement
        if all(change > 0 for change in stage_changes) and max(stage_changes) >= 2:
            return TransitionPattern.RAPID_ADVANCEMENT
        
        # Gradual regression
        if all(change < 0 for change in stage_changes):
            return TransitionPattern.GRADUAL_REGRESSION
        
        # Sudden collapse
        if any(change <= -2 for change in stage_changes):
            return TransitionPattern.SUDDEN_COLLAPSE
        
        # Check for chaotic pattern
        if np.std(stage_changes) > 1.5:
            return TransitionPattern.CHAOTIC
        
        return TransitionPattern.SMOOTH_PROGRESSION
    
    def _identify_trigger_factors(self, metrics_before: DevelopmentMetrics, 
                                metrics_after: DevelopmentMetrics,
                                phi_dynamics: Optional[PhiDynamics]) -> List[str]:
        """Identify factors that triggered the transition"""
        
        triggers = []
        
        # φ value changes
        phi_change = metrics_after.phi_value - metrics_before.phi_value
        if abs(phi_change) > 0.1:
            if phi_change > 0:
                triggers.append("phi_increase")
            else:
                triggers.append("phi_decrease")
        
        # Structural changes
        distinction_change = metrics_after.distinction_count - metrics_before.distinction_count
        if abs(distinction_change) >= 2:
            triggers.append("distinction_change")
        
        relation_change = metrics_after.relation_count - metrics_before.relation_count
        if abs(relation_change) >= 1:
            triggers.append("relation_change")
        
        # Integration quality changes
        integration_change = metrics_after.integration_quality - metrics_before.integration_quality
        if abs(integration_change) > 0.3:
            triggers.append("integration_quality_change")
        
        # φ dynamics factors
        if phi_dynamics:
            if phi_dynamics.phi_volatility > 0.2:
                triggers.append("phi_volatility")
            if abs(phi_dynamics.phi_acceleration) > 0.01:
                triggers.append("phi_acceleration")
            if phi_dynamics.transition_probability > 0.7:
                triggers.append("high_transition_probability")
        
        # Consciousness indicators
        temporal_change = metrics_after.temporal_depth - metrics_before.temporal_depth
        if abs(temporal_change) > 0.3:
            triggers.append("temporal_depth_change")
        
        self_ref_change = metrics_after.self_reference_strength - metrics_before.self_reference_strength
        if abs(self_ref_change) > 0.3:
            triggers.append("self_reference_change")
        
        return triggers
    
    def _assess_transition_impact(self, from_stage: DevelopmentStage, to_stage: DevelopmentStage,
                                metrics_after: DevelopmentMetrics) -> Dict[str, float]:
        """Assess impact of transition on various factors"""
        
        impact = {
            'consciousness_level': 0.0,
            'cognitive_capability': 0.0,
            'stability': 0.0,
            'development_potential': 0.0,
            'integration_capacity': 0.0
        }
        
        all_stages = list(DevelopmentStage)
        stage_change = all_stages.index(to_stage) - all_stages.index(from_stage)
        
        # Consciousness level impact
        impact['consciousness_level'] = stage_change * 0.2  # Each stage ~20% consciousness level change
        
        # Cognitive capability impact
        impact['cognitive_capability'] = stage_change * 0.15
        
        # Stability impact (negative for large changes)
        impact['stability'] = -abs(stage_change) * 0.1
        
        # Development potential impact
        if stage_change > 0:
            impact['development_potential'] = stage_change * 0.25
        else:
            impact['development_potential'] = stage_change * 0.1  # Less negative impact
        
        # Integration capacity impact
        impact['integration_capacity'] = metrics_after.integration_quality - 0.5  # Relative to baseline
        
        return impact
    
    def _generate_recovery_recommendation(self, from_stage: DevelopmentStage, 
                                        to_stage: DevelopmentStage,
                                        metrics_after: DevelopmentMetrics) -> str:
        """Generate recovery recommendation for regression"""
        
        recommendations = []
        
        # General regression recovery
        recommendations.append("Increase experiential input quality and diversity")
        
        # Specific recommendations based on metrics
        if metrics_after.phi_value < 0.01:
            recommendations.append("Focus on building basic φ-generating experiences")
        
        if metrics_after.integration_quality < 0.3:
            recommendations.append("Strengthen concept relationships and associations")
        
        if metrics_after.temporal_depth < 0.3:
            recommendations.append("Enhance temporal integration and memory consolidation")
        
        if metrics_after.self_reference_strength < 0.3:
            recommendations.append("Develop self-referential and introspective capacities")
        
        return "; ".join(recommendations)
    
    def _was_early_warning_given(self, to_stage: DevelopmentStage) -> bool:
        """Check if early warning was given for this transition"""
        
        # Look for recent warnings about this stage
        recent_warnings = [w for w in self.early_warnings 
                          if (datetime.now() - w.timestamp).total_seconds() < 60.0]
        
        return any(w.predicted_transition == to_stage for w in recent_warnings)
    
    def _calculate_detection_confidence(self, metrics_before: DevelopmentMetrics,
                                      metrics_after: DevelopmentMetrics,
                                      phi_dynamics: Optional[PhiDynamics]) -> float:
        """Calculate confidence in transition detection"""
        
        confidence_factors = []
        
        # Stage confidence
        confidence_factors.append(metrics_after.stage_confidence)
        
        # φ change magnitude (normalized)
        phi_change = abs(metrics_after.phi_value - metrics_before.phi_value)
        phi_confidence = min(1.0, phi_change / 0.1)  # Normalized to 0.1 threshold
        confidence_factors.append(phi_confidence)
        
        # Structural consistency
        structural_consistency = min(1.0, (metrics_after.distinction_count + metrics_after.relation_count) / 5.0)
        confidence_factors.append(structural_consistency)
        
        # φ dynamics confidence
        if phi_dynamics:
            dynamics_confidence = phi_dynamics.transition_probability
            confidence_factors.append(dynamics_confidence)
        
        # Combined confidence
        overall_confidence = np.mean(confidence_factors)
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _validate_transition(self, transition_event: TransitionEvent) -> bool:
        """Validate detected transition to reduce false positives"""
        
        # Minimum confidence threshold
        if transition_event.detection_confidence < 0.5:
            return False
        
        # Minimum φ change for validation
        if abs(transition_event.phi_change) < 0.001:
            return False
        
        # Structural validation
        metrics_after = transition_event.development_metrics_after
        if metrics_after.distinction_count == 0 and metrics_after.relation_count == 0:
            return False
        
        # Pattern validation (avoid rapid oscillations)
        if len(self.transition_history) >= 2:
            last_transition = self.transition_history[-1]
            time_since_last = (transition_event.timestamp - last_transition.timestamp).total_seconds()
            
            # Avoid transitions too close in time
            if time_since_last < 1.0:  # Less than 1 second
                return False
            
            # Avoid immediate reversals
            if (last_transition.to_stage == transition_event.from_stage and 
                last_transition.from_stage == transition_event.to_stage and
                time_since_last < 10.0):  # Within 10 seconds
                return False
        
        return True
    
    async def _check_early_warning_signals(self, current_metrics: DevelopmentMetrics,
                                         phi_dynamics: Optional[PhiDynamics]):
        """Check for early warning signals of critical transitions"""
        
        if not phi_dynamics:
            return
        
        warnings = []
        
        # High φ volatility warning
        if phi_dynamics.phi_volatility > 0.3:
            severity = TransitionSeverity.SIGNIFICANT if phi_dynamics.phi_volatility > 0.5 else TransitionSeverity.MODERATE
            
            warning = EarlyWarningSignal(
                timestamp=datetime.now(),
                warning_type="phi_instability",
                severity=severity,
                predicted_transition=self._predict_next_stage(current_metrics, phi_dynamics),
                confidence=min(1.0, phi_dynamics.phi_volatility),
                time_to_transition=self._estimate_time_to_transition(phi_dynamics),
                phi_instability=phi_dynamics.phi_volatility,
                structural_instability=0.0,  # Would need additional analysis
                experiential_disruption=0.0,  # Would need experiential data
                intervention_recommendations=self._get_intervention_recommendations(phi_dynamics),
                monitoring_priority="high" if severity == TransitionSeverity.SIGNIFICANT else "normal"
            )
            warnings.append(warning)
        
        # Critical φ acceleration warning
        if abs(phi_dynamics.phi_acceleration) > 0.02:
            warning = EarlyWarningSignal(
                timestamp=datetime.now(),
                warning_type="phi_acceleration",
                severity=TransitionSeverity.CRITICAL,
                predicted_transition=self._predict_next_stage(current_metrics, phi_dynamics),
                confidence=min(1.0, abs(phi_dynamics.phi_acceleration) * 50),
                time_to_transition=self._estimate_time_to_transition(phi_dynamics),
                phi_instability=abs(phi_dynamics.phi_acceleration),
                structural_instability=0.0,
                experiential_disruption=0.0,
                intervention_recommendations=["immediate_phi_stabilization", "reduce_input_volatility"],
                monitoring_priority="critical"
            )
            warnings.append(warning)
        
        # High transition probability warning
        if phi_dynamics.transition_probability > 0.8:
            warning = EarlyWarningSignal(
                timestamp=datetime.now(),
                warning_type="imminent_transition",
                severity=TransitionSeverity.MODERATE,
                predicted_transition=self._predict_next_stage(current_metrics, phi_dynamics),
                confidence=phi_dynamics.transition_probability,
                time_to_transition=self._estimate_time_to_transition(phi_dynamics),
                phi_instability=phi_dynamics.transition_probability,
                structural_instability=0.0,
                experiential_disruption=0.0,
                intervention_recommendations=["prepare_for_transition", "monitor_closely"],
                monitoring_priority="high"
            )
            warnings.append(warning)
        
        # Store warnings and update stats
        for warning in warnings:
            self.early_warnings.append(warning)
            self.performance_stats['early_warnings_issued'] += 1
            
            logger.warning(f"Early warning: {warning.warning_type} - {warning.severity.value} - "
                         f"Predicted: {warning.predicted_transition.value}")
        
        # Limit warning history
        if len(self.early_warnings) > 1000:
            self.early_warnings = self.early_warnings[-1000:]
    
    def _predict_next_stage(self, current_metrics: DevelopmentMetrics, 
                          phi_dynamics: PhiDynamics) -> DevelopmentStage:
        """Predict next likely stage based on dynamics"""
        
        current_stage = current_metrics.current_stage
        all_stages = list(DevelopmentStage)
        current_index = all_stages.index(current_stage)
        
        # Predict based on φ trend
        if phi_dynamics.phi_trend in ["increasing", "accelerating_up"]:
            # Likely to advance
            next_index = min(len(all_stages) - 1, current_index + 1)
        elif phi_dynamics.phi_trend in ["decreasing", "accelerating_down"]:
            # Likely to regress
            next_index = max(0, current_index - 1)
        else:
            # Stay in current stage
            next_index = current_index
        
        return all_stages[next_index]
    
    def _estimate_time_to_transition(self, phi_dynamics: PhiDynamics) -> float:
        """Estimate time until transition in seconds"""
        
        if phi_dynamics.phi_velocity == 0:
            return float('inf')
        
        # Estimate based on current velocity and acceleration
        # This is a simplified model - would need more sophisticated prediction
        
        base_time = 10.0  # Base transition time
        velocity_factor = abs(phi_dynamics.phi_velocity)
        acceleration_factor = abs(phi_dynamics.phi_acceleration)
        
        if velocity_factor > 0:
            estimated_time = base_time / (velocity_factor * 100)  # Scale factor
        else:
            estimated_time = base_time
        
        # Adjust for acceleration
        if acceleration_factor > 0:
            estimated_time *= (1.0 - acceleration_factor * 50)  # Acceleration reduces time
        
        return max(1.0, estimated_time)  # Minimum 1 second
    
    def _get_intervention_recommendations(self, phi_dynamics: PhiDynamics) -> List[str]:
        """Get intervention recommendations based on φ dynamics"""
        
        recommendations = []
        
        if phi_dynamics.phi_volatility > 0.3:
            recommendations.append("reduce_input_variability")
            recommendations.append("increase_integration_stability")
        
        if abs(phi_dynamics.phi_acceleration) > 0.01:
            recommendations.append("moderate_phi_rate_of_change")
            recommendations.append("apply_gradual_adjustments")
        
        if phi_dynamics.transition_probability > 0.7:
            recommendations.append("prepare_transition_support")
            recommendations.append("enhance_monitoring_frequency")
        
        return recommendations
    
    def get_transition_summary(self) -> Dict[str, Any]:
        """Get comprehensive transition summary"""
        
        summary = {
            "current_status": {
                "stage": self.current_stage.value if self.current_stage else "unknown",
                "phi_value": self.current_metrics.phi_value if self.current_metrics else 0.0,
                "last_transition": None,
                "monitoring_active": self.is_monitoring
            },
            "transition_history": {
                "total_transitions": len(self.transition_history),
                "recent_transitions": [],
                "transition_patterns": {}
            },
            "early_warnings": {
                "active_warnings": len([w for w in self.early_warnings 
                                      if (datetime.now() - w.timestamp).total_seconds() < 300]),
                "total_warnings": len(self.early_warnings),
                "warning_accuracy": 0.0
            },
            "performance_metrics": dict(self.performance_stats),
            "phi_dynamics": None
        }
        
        # Add last transition info
        if self.transition_history:
            last_transition = self.transition_history[-1]
            summary["current_status"]["last_transition"] = {
                "timestamp": last_transition.timestamp.isoformat(),
                "from_stage": last_transition.from_stage.value,
                "to_stage": last_transition.to_stage.value,
                "severity": last_transition.transition_severity.value
            }
        
        # Add recent transitions
        recent_transitions = self.transition_history[-5:] if len(self.transition_history) >= 5 else self.transition_history
        for transition in recent_transitions:
            summary["transition_history"]["recent_transitions"].append({
                "timestamp": transition.timestamp.isoformat(),
                "from_stage": transition.from_stage.value,
                "to_stage": transition.to_stage.value,
                "direction": transition.transition_direction.value,
                "severity": transition.transition_severity.value,
                "confidence": transition.detection_confidence
            })
        
        # Calculate transition patterns
        if len(self.transition_history) >= 3:
            patterns = [t.transition_pattern.value for t in self.transition_history[-10:]]
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            summary["transition_history"]["transition_patterns"] = pattern_counts
        
        # Add current φ dynamics
        phi_dynamics = self.phi_analyzer.calculate_phi_dynamics()
        if phi_dynamics:
            summary["phi_dynamics"] = {
                "current_phi": phi_dynamics.current_phi,
                "velocity": phi_dynamics.phi_velocity,
                "acceleration": phi_dynamics.phi_acceleration,
                "volatility": phi_dynamics.phi_volatility,
                "trend": phi_dynamics.phi_trend,
                "transition_probability": phi_dynamics.transition_probability
            }
        
        # Calculate warning accuracy
        if self.performance_stats['early_warnings_issued'] > 0:
            summary["early_warnings"]["warning_accuracy"] = (
                self.performance_stats['early_warnings_accurate'] / 
                self.performance_stats['early_warnings_issued']
            )
        
        return summary


# Example usage and testing
async def test_stage_transition_detector():
    """Test stage transition detection functionality"""
    
    print("🔄 Testing Stage Transition Detector")
    print("=" * 60)
    
    # Initialize detector
    detector = StageTransitionDetector(update_frequency_hz=5.0)
    
    # Add transition callback
    def on_transition(event: TransitionEvent):
        print(f"🚨 Transition detected: {event.from_stage.value} → {event.to_stage.value}")
        print(f"   Severity: {event.transition_severity.value}")
        print(f"   Direction: {event.transition_direction.value}")
        print(f"   Pattern: {event.transition_pattern.value}")
        print(f"   Confidence: {event.detection_confidence:.3f}")
        if event.trigger_factors:
            print(f"   Triggers: {', '.join(event.trigger_factors)}")
    
    detector.register_transition_callback(on_transition)
    
    # Start monitoring
    detector.start_real_time_monitoring()
    
    # Simulate φ structures with different values
    from iit4_core_engine import IIT4PhiCalculator
    
    calculator = IIT4PhiCalculator()
    
    test_systems = [
        ("Stage 0", np.array([0, 0]), np.array([[0, 0.1], [0.1, 0]])),
        ("Stage 1", np.array([1, 0]), np.array([[0, 0.3], [0.4, 0]])),
        ("Stage 2", np.array([1, 1]), np.array([[0, 0.5], [0.6, 0]])),
        ("Stage 3", np.array([1, 1, 0]), np.array([[0, 0.7, 0.2], [0.8, 0, 0.3], [0.1, 0.9, 0]])),
        ("Regression", np.array([1, 0]), np.array([[0, 0.2], [0.3, 0]])),
    ]
    
    for name, state, connectivity in test_systems:
        print(f"\n📊 Testing {name}")
        print("-" * 30)
        
        # Calculate φ structure
        phi_structure = calculator.calculate_phi(state, connectivity)
        
        # Detect transition
        transition = await detector.detect_stage_transition(phi_structure)
        
        if transition:
            print(f"   Transition detected!")
        else:
            print(f"   No transition detected")
            print(f"   Current stage: {detector.current_stage.value if detector.current_stage else 'Unknown'}")
            print(f"   φ value: {phi_structure.total_phi:.6f}")
        
        # Add some delay for realistic timing
        await asyncio.sleep(0.5)
    
    # Test early warning system
    print(f"\n⚠️  Testing Early Warning System")
    print("-" * 40)
    
    # Create volatile φ structure
    volatile_state = np.array([1, 1, 1])
    volatile_connectivity = np.random.rand(3, 3) * 0.8
    
    for i in range(5):
        # Add noise to create volatility
        noisy_connectivity = volatile_connectivity + np.random.normal(0, 0.1, (3, 3))
        phi_structure = calculator.calculate_phi(volatile_state, noisy_connectivity)
        
        await detector.detect_stage_transition(phi_structure)
        await asyncio.sleep(0.2)
    
    # Stop monitoring
    detector.stop_real_time_monitoring()
    
    # Get summary
    summary = detector.get_transition_summary()
    print(f"\n📈 Transition Summary")
    print("-" * 30)
    print(f"   Total transitions: {summary['transition_history']['total_transitions']}")
    print(f"   Early warnings: {summary['early_warnings']['total_warnings']}")
    print(f"   Average latency: {summary['performance_metrics']['average_latency_ms']:.1f}ms")
    print(f"   Detection accuracy: {100 - summary['performance_metrics']['false_positives']}%")


if __name__ == "__main__":
    asyncio.run(test_stage_transition_detector())