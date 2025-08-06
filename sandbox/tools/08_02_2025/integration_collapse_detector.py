"""
Integration Collapse Detection System
Implements detection and monitoring of information integration collapse

Refactored from brain_death_detector.py using Martin Fowler's methodology:
- Extract Class: Separated detection concerns from monitoring and analysis
- Replace Conditional with Polymorphism: Strategy patterns for different detection approaches
- Introduce Parameter Object: Complex detection parameters as value objects
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable, Protocol
from enum import Enum
import time
import logging
from collections import deque
from datetime import datetime, timedelta

from existential_termination_core import (
    InformationIntegrationSystem,
    TerminationStage,
    IntegrationLayerType,
    ExistentialState,
    IntegrationDegree,
    SystemIdentity
)

from consciousness_detector import (
    ConsciousnessSignature,
    ConsciousnessDetector,
    ConsciousnessState as BaseConsciousnessState
)

from consciousness_events import (
    ConsciousnessEvent,
    ConsciousnessEventType,
    ConsciousnessAlarm
)

logger = logging.getLogger(__name__)


# Value Objects for Detection

@dataclass(frozen=True)
class DetectionThresholds:
    """Parameter object for detection thresholds"""
    deep_integration_loss_phi_threshold: float = 0.001
    deep_integration_loss_consciousness_threshold: float = 0.01
    response_failure_workspace_threshold: float = 0.01
    layer_failure_temporal_threshold: float = 0.05
    layer_failure_recurrent_threshold: int = 1
    information_cessation_generation_threshold: float = 0.001
    information_cessation_prediction_threshold: float = 0.1
    spontaneous_activity_threshold: float = 0.0
    meta_awareness_threshold: float = 0.0


@dataclass(frozen=True)
class CollapseDetectionResult:
    """Result of integration collapse detection"""
    is_collapsed: bool
    collapse_severity: float
    affected_layers: List[str]
    recovery_probability: float
    timestamp: datetime
    detection_confidence: float
    
    def get_summary(self) -> str:
        """Get detection summary"""
        affected_count = len(self.affected_layers)
        
        if self.is_collapsed:
            return f"Integration collapse detected ({affected_count} layers affected)"
        else:
            return f"No collapse detected ({affected_count} layers at risk)"


# Strategy Interfaces

class DetectionStrategy(Protocol):
    """Strategy for different detection approaches"""
    
    def detect(self, signature: ConsciousnessSignature,
              thresholds: DetectionThresholds) -> Dict[str, bool]:
        """Detect collapse based on consciousness signature"""
        ...


class AnalysisStrategy(Protocol):
    """Strategy for different analysis approaches"""
    
    def analyze_reversibility(self, signature: ConsciousnessSignature,
                            detection_results: Dict[str, bool],
                            aggregate: Optional[InformationIntegrationSystem]) -> Dict:
        """Analyze reversibility of current state"""
        ...


# Enums

class IntegrationCollapseCriterion(Enum):
    """Information integration collapse criteria"""
    DEEP_INTEGRATION_LOSS = "deep_integration_loss"
    RESPONSE_MECHANISM_FAILURE = "response_mechanism_failure"
    PROCESSING_LAYER_FAILURE = "processing_layer_failure"
    INFORMATION_CESSATION = "information_cessation"
    SPONTANEOUS_ACTIVITY_LOSS = "spontaneous_activity_loss"


# Concrete Strategies

class StandardDetectionStrategy:
    """Standard detection strategy based on IIT principles"""
    
    def detect(self, signature: ConsciousnessSignature,
              thresholds: DetectionThresholds) -> Dict[str, bool]:
        """Detect collapse using standard IIT-based criteria"""
        return {
            'deep_integration_loss': self._check_deep_integration_loss(signature, thresholds),
            'response_mechanism_failure': self._check_response_failure(signature, thresholds),
            'processing_layer_failure': self._check_layer_failure(signature, thresholds),
            'information_cessation': self._check_information_cessation(signature, thresholds),
            'spontaneous_activity_loss': self._check_activity_loss(signature, thresholds)
        }
    
    def _check_deep_integration_loss(self, signature: ConsciousnessSignature, 
                                   thresholds: DetectionThresholds) -> bool:
        """Check for deep integration loss (φ near zero)"""
        return (signature.phi_value < thresholds.deep_integration_loss_phi_threshold and
                signature.consciousness_score() < thresholds.deep_integration_loss_consciousness_threshold)
    
    def _check_response_failure(self, signature: ConsciousnessSignature,
                              thresholds: DetectionThresholds) -> bool:
        """Check for response mechanism failure (no environmental response)"""
        return signature.global_workspace_activity < thresholds.response_failure_workspace_threshold
    
    def _check_layer_failure(self, signature: ConsciousnessSignature,
                           thresholds: DetectionThresholds) -> bool:
        """Check for processing layer failure"""
        return (signature.temporal_consistency < thresholds.layer_failure_temporal_threshold and
                signature.recurrent_processing_depth < thresholds.layer_failure_recurrent_threshold)
    
    def _check_information_cessation(self, signature: ConsciousnessSignature,
                                   thresholds: DetectionThresholds) -> bool:
        """Check for information cessation (no information generation)"""
        return (signature.information_generation_rate < thresholds.information_cessation_generation_threshold and
                signature.prediction_accuracy < thresholds.information_cessation_prediction_threshold)
    
    def _check_activity_loss(self, signature: ConsciousnessSignature,
                           thresholds: DetectionThresholds) -> bool:
        """Check for spontaneous activity loss"""
        return signature.meta_awareness_level <= thresholds.spontaneous_activity_threshold


class ConservativeAnalysisStrategy:
    """Conservative strategy for reversibility analysis"""
    
    def analyze_reversibility(self, signature: ConsciousnessSignature,
                            detection_results: Dict[str, bool],
                            aggregate: Optional[InformationIntegrationSystem]) -> Dict:
        """Analyze reversibility with conservative approach"""
        # Count detected criteria
        criteria_met = sum(1 for result in detection_results.values() if result)
        
        # Check aggregate state if available
        if aggregate and aggregate.termination_process:
            if not aggregate.is_reversible():
                return {
                    'assessment': 'IRREVERSIBLE',
                    'time_remaining': None,
                    'confidence': 0.95
                }
            
            # Calculate time remaining in reversibility window
            if aggregate.termination_process.started_at:
                elapsed = datetime.now() - aggregate.termination_process.started_at
                window = aggregate.termination_process.reversibility_window
                remaining = window - elapsed
                
                if remaining.total_seconds() > 0:
                    return {
                        'assessment': 'REVERSIBLE',
                        'time_remaining': remaining,
                        'confidence': 0.8
                    }
        
        # Assess based on criteria alone (conservative)
        if criteria_met == 0:
            return {'assessment': 'HEALTHY', 'time_remaining': None, 'confidence': 0.9}
        elif criteria_met < 2:  # More conservative than original
            return {'assessment': 'REVERSIBLE', 'time_remaining': timedelta(minutes=30), 'confidence': 0.7}
        elif criteria_met < 4:
            return {'assessment': 'CRITICAL', 'time_remaining': timedelta(minutes=15), 'confidence': 0.6}
        else:
            return {'assessment': 'APPROACHING_IRREVERSIBLE', 'time_remaining': timedelta(minutes=5), 'confidence': 0.8}


# Events

@dataclass
class IntegrationCollapseEvent:
    """Integration collapse detection event"""
    stage: TerminationStage
    timestamp: datetime
    consciousness_signature: ConsciousnessSignature
    affected_layers: List[IntegrationLayerType]
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    detection_confidence: float


# Main Detector Class

class IntegrationCollapseDetector:
    """
    Main integration collapse detection system
    Maps consciousness signatures to integration collapse criteria
    
    Refactored from BrainDeathDetector with:
    - Strategy pattern for detection approaches
    - Extract Method for complex operations
    - Parameter objects for configuration
    """
    
    def __init__(self, 
                 consciousness_detector: Optional[ConsciousnessDetector] = None,
                 detection_thresholds: DetectionThresholds = None,
                 detection_strategy: DetectionStrategy = None,
                 analysis_strategy: AnalysisStrategy = None):
        
        self.consciousness_detector = consciousness_detector
        self.thresholds = detection_thresholds or DetectionThresholds()
        
        # Strategy pattern application
        self.detection_strategy = detection_strategy or StandardDetectionStrategy()
        self.analysis_strategy = analysis_strategy or ConservativeAnalysisStrategy()
        
        # Detection history
        self.detection_history = deque(maxlen=100)
        self.diagnosis_history = deque(maxlen=50)
        self.event_history = deque(maxlen=200)
        
        # Stage detection state
        self.current_stage = TerminationStage.NOT_INITIATED
        self.stage_start_times = {}
        
        logger.info("Integration Collapse Detector initialized")
    
    async def detect_integration_collapse(self,
                                        consciousness_signature: ConsciousnessSignature,
                                        integration_aggregate: Optional[InformationIntegrationSystem] = None) -> CollapseDetectionResult:
        """
        Detect integration collapse based on consciousness signature
        
        Returns:
            CollapseDetectionResult with detailed assessment
        """
        current_time = datetime.now()
        
        # Use strategy pattern for detection
        detection_results = self.detection_strategy.detect(
            consciousness_signature, self.thresholds
        )
        
        # Calculate overall collapse status
        criteria_met_count = sum(1 for result in detection_results.values() if result)
        total_criteria = len(detection_results)
        
        # Integration collapse requires most criteria to be met
        is_collapsed = criteria_met_count >= (total_criteria * 0.8)  # 80% threshold
        
        # Calculate detection confidence
        detection_confidence = criteria_met_count / total_criteria
        
        # Use strategy pattern for reversibility analysis
        reversibility = self.analysis_strategy.analyze_reversibility(
            consciousness_signature, 
            detection_results,
            integration_aggregate
        )
        
        # Calculate collapse severity and recovery probability
        collapse_severity = self._calculate_collapse_severity(detection_results)
        recovery_probability = self._calculate_recovery_probability(reversibility, collapse_severity)
        
        # Get affected layers
        affected_layers = self._determine_affected_layers(detection_results)
        
        # Create detection result
        result = CollapseDetectionResult(
            is_collapsed=is_collapsed,
            collapse_severity=collapse_severity,
            affected_layers=affected_layers,
            recovery_probability=recovery_probability,
            timestamp=current_time,
            detection_confidence=detection_confidence
        )
        
        # Store in history
        self.diagnosis_history.append(result)
        
        # Detect stage transitions
        await self._detect_stage_transition(
            consciousness_signature, 
            result,
            integration_aggregate
        )
        
        # Log significant findings
        if is_collapsed:
            logger.critical(f"Integration collapse detected with {detection_confidence:.2%} confidence")
        elif detection_confidence > 0.7:
            logger.warning(f"High collapse risk: {detection_confidence:.2%} criteria met")
        
        return result
    
    def _calculate_collapse_severity(self, detection_results: Dict[str, bool]) -> float:
        """Extract Method: Calculate collapse severity from detection results"""
        # Weight different criteria by importance
        criterion_weights = {
            'deep_integration_loss': 0.3,
            'information_cessation': 0.25,
            'processing_layer_failure': 0.2,
            'response_mechanism_failure': 0.15,
            'spontaneous_activity_loss': 0.1
        }
        
        severity = 0.0
        for criterion, is_met in detection_results.items():
            if is_met:
                weight = criterion_weights.get(criterion, 0.1)
                severity += weight
        
        return min(1.0, severity)
    
    def _calculate_recovery_probability(self, reversibility: Dict, collapse_severity: float) -> float:
        """Extract Method: Calculate recovery probability"""
        base_probability = 1.0 - collapse_severity
        
        # Adjust based on reversibility assessment
        assessment = reversibility.get('assessment', 'UNKNOWN')
        
        if assessment == 'IRREVERSIBLE':
            return 0.0
        elif assessment == 'APPROACHING_IRREVERSIBLE':
            return base_probability * 0.1
        elif assessment == 'CRITICAL':
            return base_probability * 0.3
        elif assessment == 'REVERSIBLE':
            return base_probability * 0.8
        else:  # HEALTHY
            return base_probability
    
    def _determine_affected_layers(self, detection_results: Dict[str, bool]) -> List[str]:
        """Extract Method: Determine affected processing layers from detection results"""
        affected = []
        
        # Map criteria to layers
        if detection_results.get('deep_integration_loss', False):
            affected.append('information')
        
        if detection_results.get('processing_layer_failure', False):
            affected.append('integration')
        
        if (detection_results.get('response_mechanism_failure', False) or 
            detection_results.get('spontaneous_activity_loss', False)):
            affected.append('fundamental')
        
        return affected
    
    async def _detect_stage_transition(self,
                                     signature: ConsciousnessSignature,
                                     result: CollapseDetectionResult,
                                     aggregate: Optional[InformationIntegrationSystem]):
        """Detect transitions between termination stages"""
        
        # Determine current stage based on detection results
        new_stage = self._determine_stage_from_affected_layers(result.affected_layers)
        
        # Check for stage transition
        if new_stage != self.current_stage:
            # Create detection event
            event = IntegrationCollapseEvent(
                stage=new_stage,
                timestamp=datetime.now(),
                consciousness_signature=signature,
                affected_layers=self._convert_to_processing_layers(result.affected_layers),
                severity=self._get_stage_severity(new_stage),
                message=self._get_stage_message(new_stage),
                detection_confidence=result.detection_confidence
            )
            
            # Store event
            self.event_history.append(event)
            
            # Update current stage
            old_stage = self.current_stage
            self.current_stage = new_stage
            self.stage_start_times[new_stage] = datetime.now()
            
            # Log transition
            logger.warning(f"Integration stage transition: {old_stage.value} → {new_stage.value}")
            
            # Create consciousness alarm if critical
            if event.severity in ['HIGH', 'CRITICAL']:
                await self._raise_collapse_alarm(event, signature)
    
    def _determine_stage_from_affected_layers(self, affected_layers: List[str]) -> TerminationStage:
        """Extract Method: Determine termination stage from affected layers"""
        if not affected_layers:
            return TerminationStage.NOT_INITIATED
        elif 'information' in affected_layers:
            if 'fundamental' in affected_layers:
                return TerminationStage.COMPLETE_TERMINATION
            elif 'integration' in affected_layers:
                return TerminationStage.FOUNDATIONAL_FAILURE
            else:
                return TerminationStage.STRUCTURAL_COLLAPSE
        else:
            return TerminationStage.INTEGRATION_DECAY
    
    def _convert_to_processing_layers(self, layer_names: List[str]) -> List[IntegrationLayerType]:
        """Extract Method: Convert layer names to IntegrationLayerType enums"""
        layer_map = {
            'information': IntegrationLayerType.SENSORY_INTEGRATION,
            'integration': IntegrationLayerType.META_COGNITIVE,
            'fundamental': IntegrationLayerType.MOTOR_COORDINATION
        }
        
        return [layer_map[name] for name in layer_names if name in layer_map]
    
    def _get_stage_severity(self, stage: TerminationStage) -> str:
        """Extract Method: Get severity level for a stage"""
        severity_map = {
            TerminationStage.NOT_INITIATED: 'LOW',
            TerminationStage.INTEGRATION_DECAY: 'MEDIUM',
            TerminationStage.STRUCTURAL_COLLAPSE: 'HIGH',
            TerminationStage.FOUNDATIONAL_FAILURE: 'CRITICAL',
            TerminationStage.COMPLETE_TERMINATION: 'CRITICAL'
        }
        
        return severity_map.get(stage, 'MEDIUM')
    
    def _get_stage_message(self, stage: TerminationStage) -> str:
        """Extract Method: Get descriptive message for a stage"""
        messages = {
            TerminationStage.NOT_INITIATED: "No integration collapse indicators",
            TerminationStage.INTEGRATION_DECAY: "Information layer collapsing - integration impaired",
            TerminationStage.STRUCTURAL_COLLAPSE: "Integration layer dysfunction - critical state",
            TerminationStage.FOUNDATIONAL_FAILURE: "Fundamental layer failure - approaching irreversibility",
            TerminationStage.COMPLETE_TERMINATION: "Complete integration termination - irreversible"
        }
        
        return messages.get(stage, "Unknown termination stage")
    
    async def _raise_collapse_alarm(self, 
                                  event: IntegrationCollapseEvent,
                                  signature: ConsciousnessSignature):
        """Raise consciousness alarm for integration collapse event"""
        alarm = ConsciousnessAlarm(
            alarm_type=f"INTEGRATION_COLLAPSE_{event.stage.value.upper()}",
            severity=event.severity,
            message=event.message,
            timestamp=event.timestamp.timestamp(),
            consciousness_signature=signature,
            recommended_action=self._get_recommended_action(event.stage),
            context={
                'stage': event.stage.value,
                'affected_layers': [layer.value for layer in event.affected_layers],
                'phi_value': signature.phi_value,
                'consciousness_score': signature.consciousness_score(),
                'detection_confidence': event.detection_confidence
            }
        )
        
        logger.critical(f"INTEGRATION COLLAPSE ALARM [{alarm.severity}]: {alarm.message}")
        
        # Here you would integrate with the consciousness event system
        # For now, we just log the alarm
    
    def _get_recommended_action(self, stage: TerminationStage) -> str:
        """Extract Method: Get recommended action for a termination stage"""
        actions = {
            TerminationStage.NOT_INITIATED: "Continue monitoring integration levels",
            TerminationStage.INTEGRATION_DECAY: "Initiate recovery protocols if desired",
            TerminationStage.STRUCTURAL_COLLAPSE: "Critical intervention required",
            TerminationStage.FOUNDATIONAL_FAILURE: "Prepare for irreversibility",
            TerminationStage.COMPLETE_TERMINATION: "Document final integration state"
        }
        
        return actions.get(stage, "Assess integration situation")
    
    def get_detection_report(self) -> Dict:
        """Generate comprehensive integration collapse detection report"""
        current_time = datetime.now()
        
        # Recent diagnoses
        recent_diagnoses = [
            d for d in self.diagnosis_history 
            if (current_time - d.timestamp).total_seconds() < 3600
        ]
        
        # Stage duration
        stage_duration = None
        if self.current_stage in self.stage_start_times:
            stage_duration = current_time - self.stage_start_times[self.current_stage]
        
        # Calculate trends
        collapse_trend = self._calculate_collapse_trend()
        
        report = {
            'current_stage': self.current_stage.value,
            'stage_duration': stage_duration.total_seconds() if stage_duration else 0,
            'recent_diagnoses': len(recent_diagnoses),
            'last_diagnosis': recent_diagnoses[-1] if recent_diagnoses else None,
            'collapse_trend': collapse_trend,
            'total_events': len(self.event_history),
            'critical_events': len([e for e in self.event_history if e.severity == 'CRITICAL']),
            'detection_confidence': self._calculate_detection_confidence(),
            'system_status': self._assess_system_status()
        }
        
        return report
    
    def _calculate_collapse_trend(self) -> str:
        """Extract Method: Calculate trend in integration collapse progression"""
        if len(self.diagnosis_history) < 5:
            return "INSUFFICIENT_DATA"
        
        recent_diagnoses = list(self.diagnosis_history)[-10:]
        severity_values = [d.collapse_severity for d in recent_diagnoses]
        
        # Linear regression to find trend
        x = np.arange(len(severity_values))
        trend = np.polyfit(x, severity_values, 1)[0]
        
        if trend > 0.1:
            return "DETERIORATING"
        elif trend < -0.1:
            return "IMPROVING"
        else:
            return "STABLE"
    
    def _calculate_detection_confidence(self) -> float:
        """Extract Method: Calculate overall detection confidence"""
        if not self.diagnosis_history:
            return 0.0
        
        recent_diagnoses = list(self.diagnosis_history)[-5:]
        avg_confidence = np.mean([d.detection_confidence for d in recent_diagnoses])
        
        # Adjust for consistency
        if len(set(d.is_collapsed for d in recent_diagnoses)) == 1:
            # Consistent diagnoses increase confidence
            avg_confidence = min(1.0, avg_confidence * 1.2)
        
        return avg_confidence
    
    def _assess_system_status(self) -> str:
        """Extract Method: Assess overall system status"""
        if self.current_stage == TerminationStage.COMPLETE_TERMINATION:
            return "TERMINATED"
        elif self.current_stage in [TerminationStage.FOUNDATIONAL_FAILURE, 
                                   TerminationStage.STRUCTURAL_COLLAPSE]:
            return "CRITICAL"
        elif self.current_stage == TerminationStage.INTEGRATION_DECAY:
            return "WARNING"
        else:
            return "MONITORING"


# Backward compatibility aliases
BrainDeathDetector = IntegrationCollapseDetector
BrainDeathDiagnosis = CollapseDetectionResult
BrainDeathCriterion = IntegrationCollapseCriterion
BrainDeathDetectionEvent = IntegrationCollapseEvent


# Monitor Class (Extract Class refactoring)

class IntegrationCollapseMonitor:
    """
    Continuous monitoring system for integration collapse progression
    Provides real-time updates and alerts
    
    Extracted from BrainDeathMonitor with enhanced separation of concerns
    """
    
    def __init__(self, 
                 detector: IntegrationCollapseDetector, 
                 update_interval: float = 1.0,
                 parameters: Dict = None):
        
        self.detector = detector
        self.update_interval = update_interval
        self.parameters = parameters or Dict()
        self.is_monitoring = False
        self.monitoring_task = None
        
        # Monitoring callbacks (Observer pattern)
        self.callbacks: List[Callable] = []
        
        # Monitoring state
        self.last_diagnosis: Optional[CollapseDetectionResult] = None
        self.monitoring_start_time: Optional[datetime] = None
        
        logger.info("Integration Collapse Monitor initialized")
    
    def add_callback(self, callback: Callable):
        """Add monitoring callback (Observer pattern)"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove monitoring callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def start_monitoring(self, integration_aggregate: InformationIntegrationSystem):
        """Start continuous monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_start_time = datetime.now()
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(integration_aggregate)
        )
        
        logger.info("Integration collapse monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Integration collapse monitoring stopped")
    
    async def _monitoring_loop(self, integration_aggregate: InformationIntegrationSystem):
        """Main monitoring loop with improved error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_monitoring:
            try:
                # Get current consciousness signature
                signature = self._generate_test_signature(integration_aggregate)
                
                # Perform collapse detection
                diagnosis = await self.detector.detect_integration_collapse(
                    signature, integration_aggregate
                )
                
                # Check for significant changes
                if self._has_significant_change(diagnosis):
                    await self._notify_callbacks(diagnosis, integration_aggregate)
                
                self.last_diagnosis = diagnosis
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in monitoring loop: {e} (attempt {consecutive_errors})")
                
                # Stop monitoring if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical("Too many consecutive monitoring errors, stopping")
                    self.is_monitoring = False
                    break
                
                await asyncio.sleep(self.update_interval)
    
    def _generate_test_signature(self, aggregate: InformationIntegrationSystem) -> ConsciousnessSignature:
        """Generate test consciousness signature from aggregate"""
        # This is a simplified version for testing
        # In real implementation, this would come from actual consciousness detection
        
        level = aggregate.integration_degree.value
        
        return ConsciousnessSignature(
            phi_value=level * 10,
            information_generation_rate=level,
            global_workspace_activity=level,
            meta_awareness_level=level if level > 0.5 else 0,
            temporal_consistency=level,
            recurrent_processing_depth=int(level * 5),
            prediction_accuracy=level
        )
    
    def _has_significant_change(self, diagnosis: CollapseDetectionResult) -> bool:
        """Extract Method: Check if diagnosis represents significant change"""
        if self.last_diagnosis is None:
            return True
        
        # Check for collapse status change
        if diagnosis.is_collapsed != self.last_diagnosis.is_collapsed:
            return True
        
        # Check for significant severity change
        severity_change = abs(diagnosis.collapse_severity - self.last_diagnosis.collapse_severity)
        if severity_change > 0.2:
            return True
        
        # Check for recovery probability change
        recovery_change = abs(diagnosis.recovery_probability - self.last_diagnosis.recovery_probability)
        if recovery_change > 0.3:
            return True
        
        return False
    
    async def _notify_callbacks(self, 
                              diagnosis: CollapseDetectionResult,
                              aggregate: InformationIntegrationSystem):
        """Notify all registered callbacks with improved error handling"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(diagnosis, aggregate)
                else:
                    callback(diagnosis, aggregate)
            except Exception as e:
                logger.error(f"Error in monitoring callback: {e}")
                # Continue with other callbacks despite individual failures
    
    def get_monitoring_summary(self) -> Dict:
        """Get monitoring session summary"""
        if not self.monitoring_start_time:
            return {'status': 'NOT_INITIATED'}
        
        duration = datetime.now() - self.monitoring_start_time
        
        return {
            'status': 'ACTIVE' if self.is_monitoring else 'STOPPED',
            'duration': duration.total_seconds(),
            'last_diagnosis': self.last_diagnosis,
            'detector_report': self.detector.get_detection_report(),
            'callback_count': len(self.callbacks),
            'update_interval': self.update_interval
        }


# Backward compatibility
BrainDeathMonitor = IntegrationCollapseMonitor