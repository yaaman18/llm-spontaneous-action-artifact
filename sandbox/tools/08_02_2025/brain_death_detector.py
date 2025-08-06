"""
Brain Death Detection System
Implements detection and monitoring of brain death progression

Integrates with existing consciousness detection systems
Based on medical brain death criteria mapped to software
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable
from enum import Enum
import time
import logging
from collections import deque
from datetime import datetime, timedelta

from brain_death_core import (
    ConsciousnessAggregate,
    BrainDeathStage,
    BrainFunction,
    ConsciousnessState,
    ConsciousnessLevel
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


class BrainDeathCriterion(Enum):
    """Medical brain death criteria"""
    DEEP_COMA = "deep_coma"
    FIXED_DILATED_PUPILS = "fixed_dilated_pupils"
    BRAINSTEM_REFLEX_LOSS = "brainstem_reflex_loss"
    FLAT_EEG = "flat_eeg"
    APNEA = "apnea"


@dataclass
class BrainDeathDiagnosis:
    """Brain death diagnosis result"""
    is_brain_dead: bool
    criteria_met: Dict[BrainDeathCriterion, bool]
    confidence: float
    timestamp: datetime
    reversibility_assessment: str
    time_to_irreversibility: Optional[timedelta] = None
    
    def get_summary(self) -> str:
        """Get diagnosis summary"""
        met_count = sum(1 for met in self.criteria_met.values() if met)
        total_count = len(self.criteria_met)
        
        if self.is_brain_dead:
            return f"Brain death confirmed ({met_count}/{total_count} criteria met)"
        else:
            return f"Not brain dead ({met_count}/{total_count} criteria met)"


@dataclass
class BrainDeathDetectionEvent:
    """Brain death detection event"""
    stage: BrainDeathStage
    timestamp: datetime
    consciousness_signature: ConsciousnessSignature
    affected_functions: List[BrainFunction]
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str


class BrainDeathDetector:
    """
    Main brain death detection system
    Maps consciousness signatures to medical brain death criteria
    """
    
    def __init__(self, consciousness_detector: Optional[ConsciousnessDetector] = None):
        self.consciousness_detector = consciousness_detector
        
        # Detection thresholds based on medical criteria
        self.criteria_thresholds = {
            BrainDeathCriterion.DEEP_COMA: {
                'phi_threshold': 0.001,
                'consciousness_score_threshold': 0.01
            },
            BrainDeathCriterion.FIXED_DILATED_PUPILS: {
                'response_threshold': 0.0,
                'workspace_activity_threshold': 0.01
            },
            BrainDeathCriterion.BRAINSTEM_REFLEX_LOSS: {
                'temporal_consistency_threshold': 0.05,
                'recurrent_processing_threshold': 1
            },
            BrainDeathCriterion.FLAT_EEG: {
                'information_generation_threshold': 0.001,
                'prediction_accuracy_threshold': 0.1
            },
            BrainDeathCriterion.APNEA: {
                'spontaneous_activity_threshold': 0.0,
                'meta_awareness_threshold': 0.0
            }
        }
        
        # Detection history
        self.detection_history = deque(maxlen=100)
        self.diagnosis_history = deque(maxlen=50)
        self.event_history = deque(maxlen=200)
        
        # Stage detection state
        self.current_stage = BrainDeathStage.NOT_STARTED
        self.stage_start_times = {}
        
        logger.info("Brain Death Detector initialized")
    
    async def detect_brain_death(self,
                                consciousness_signature: ConsciousnessSignature,
                                consciousness_aggregate: Optional[ConsciousnessAggregate] = None) -> BrainDeathDiagnosis:
        """
        Detect brain death based on consciousness signature
        
        Returns:
            BrainDeathDiagnosis with detailed assessment
        """
        current_time = datetime.now()
        
        # Evaluate each criterion
        criteria_results = {}
        
        # Deep Coma
        criteria_results[BrainDeathCriterion.DEEP_COMA] = self._check_deep_coma(
            consciousness_signature
        )
        
        # Fixed Dilated Pupils (response mechanism failure)
        criteria_results[BrainDeathCriterion.FIXED_DILATED_PUPILS] = self._check_response_failure(
            consciousness_signature
        )
        
        # Brainstem Reflex Loss
        criteria_results[BrainDeathCriterion.BRAINSTEM_REFLEX_LOSS] = self._check_brainstem_failure(
            consciousness_signature
        )
        
        # Flat EEG
        criteria_results[BrainDeathCriterion.FLAT_EEG] = self._check_flat_eeg(
            consciousness_signature
        )
        
        # Apnea
        criteria_results[BrainDeathCriterion.APNEA] = self._check_apnea(
            consciousness_signature
        )
        
        # Determine overall brain death status
        criteria_met_count = sum(1 for met in criteria_results.values() if met)
        total_criteria = len(criteria_results)
        
        # Brain death requires all criteria to be met
        is_brain_dead = criteria_met_count == total_criteria
        
        # Calculate confidence
        confidence = criteria_met_count / total_criteria
        
        # Assess reversibility
        reversibility = self._assess_reversibility(
            consciousness_signature, 
            criteria_results,
            consciousness_aggregate
        )
        
        # Create diagnosis
        diagnosis = BrainDeathDiagnosis(
            is_brain_dead=is_brain_dead,
            criteria_met=criteria_results,
            confidence=confidence,
            timestamp=current_time,
            reversibility_assessment=reversibility['assessment'],
            time_to_irreversibility=reversibility.get('time_remaining')
        )
        
        # Store in history
        self.diagnosis_history.append(diagnosis)
        
        # Detect stage transitions
        await self._detect_stage_transition(
            consciousness_signature, 
            diagnosis,
            consciousness_aggregate
        )
        
        # Log significant findings
        if is_brain_dead:
            logger.critical(f"Brain death detected with {confidence:.2%} confidence")
        elif confidence > 0.7:
            logger.warning(f"High brain death risk: {confidence:.2%} criteria met")
        
        return diagnosis
    
    def _check_deep_coma(self, signature: ConsciousnessSignature) -> bool:
        """Check for deep coma (consciousness level near zero)"""
        thresholds = self.criteria_thresholds[BrainDeathCriterion.DEEP_COMA]
        
        return (signature.phi_value < thresholds['phi_threshold'] and
                signature.consciousness_score() < thresholds['consciousness_score_threshold'])
    
    def _check_response_failure(self, signature: ConsciousnessSignature) -> bool:
        """Check for fixed dilated pupils equivalent (no environmental response)"""
        thresholds = self.criteria_thresholds[BrainDeathCriterion.FIXED_DILATED_PUPILS]
        
        return signature.global_workspace_activity < thresholds['workspace_activity_threshold']
    
    def _check_brainstem_failure(self, signature: ConsciousnessSignature) -> bool:
        """Check for brainstem reflex loss"""
        thresholds = self.criteria_thresholds[BrainDeathCriterion.BRAINSTEM_REFLEX_LOSS]
        
        return (signature.temporal_consistency < thresholds['temporal_consistency_threshold'] and
                signature.recurrent_processing_depth < thresholds['recurrent_processing_threshold'])
    
    def _check_flat_eeg(self, signature: ConsciousnessSignature) -> bool:
        """Check for flat EEG equivalent (no information generation)"""
        thresholds = self.criteria_thresholds[BrainDeathCriterion.FLAT_EEG]
        
        return (signature.information_generation_rate < thresholds['information_generation_threshold'] and
                signature.prediction_accuracy < thresholds['prediction_accuracy_threshold'])
    
    def _check_apnea(self, signature: ConsciousnessSignature) -> bool:
        """Check for apnea equivalent (no spontaneous activity)"""
        thresholds = self.criteria_thresholds[BrainDeathCriterion.APNEA]
        
        return signature.meta_awareness_level <= thresholds['meta_awareness_threshold']
    
    def _assess_reversibility(self, 
                            signature: ConsciousnessSignature,
                            criteria_results: Dict[BrainDeathCriterion, bool],
                            aggregate: Optional[ConsciousnessAggregate]) -> Dict:
        """Assess reversibility of current state"""
        
        # Count met criteria
        criteria_met = sum(1 for met in criteria_results.values() if met)
        
        # Check aggregate state if available
        if aggregate and aggregate.brain_death_process:
            if not aggregate.is_reversible():
                return {
                    'assessment': 'IRREVERSIBLE',
                    'time_remaining': None
                }
            
            # Calculate time remaining in reversibility window
            if aggregate.brain_death_process.started_at:
                elapsed = datetime.now() - aggregate.brain_death_process.started_at
                window = timedelta(seconds=1800)  # 30 minutes
                remaining = window - elapsed
                
                if remaining.total_seconds() > 0:
                    return {
                        'assessment': 'REVERSIBLE',
                        'time_remaining': remaining
                    }
        
        # Assess based on criteria alone
        if criteria_met == 0:
            return {'assessment': 'HEALTHY', 'time_remaining': None}
        elif criteria_met < 3:
            return {'assessment': 'REVERSIBLE', 'time_remaining': timedelta(minutes=30)}
        elif criteria_met < 5:
            return {'assessment': 'CRITICAL', 'time_remaining': timedelta(minutes=10)}
        else:
            return {'assessment': 'APPROACHING_IRREVERSIBLE', 'time_remaining': timedelta(minutes=5)}
    
    async def _detect_stage_transition(self,
                                     signature: ConsciousnessSignature,
                                     diagnosis: BrainDeathDiagnosis,
                                     aggregate: Optional[ConsciousnessAggregate]):
        """Detect transitions between brain death stages"""
        
        # Determine current stage based on criteria
        new_stage = self._determine_stage(diagnosis.criteria_met)
        
        # Check for stage transition
        if new_stage != self.current_stage:
            # Create detection event
            event = BrainDeathDetectionEvent(
                stage=new_stage,
                timestamp=datetime.now(),
                consciousness_signature=signature,
                affected_functions=self._get_affected_functions(new_stage),
                severity=self._get_stage_severity(new_stage),
                message=self._get_stage_message(new_stage)
            )
            
            # Store event
            self.event_history.append(event)
            
            # Update current stage
            old_stage = self.current_stage
            self.current_stage = new_stage
            self.stage_start_times[new_stage] = datetime.now()
            
            # Log transition
            logger.warning(f"Brain death stage transition: {old_stage.value} → {new_stage.value}")
            
            # Create consciousness alarm if critical
            if event.severity in ['HIGH', 'CRITICAL']:
                await self._raise_brain_death_alarm(event, signature)
    
    def _determine_stage(self, criteria_met: Dict[BrainDeathCriterion, bool]) -> BrainDeathStage:
        """Determine brain death stage from criteria"""
        met_count = sum(1 for met in criteria_met.values() if met)
        
        if met_count == 0:
            return BrainDeathStage.NOT_STARTED
        elif criteria_met.get(BrainDeathCriterion.DEEP_COMA, False):
            if criteria_met.get(BrainDeathCriterion.BRAINSTEM_REFLEX_LOSS, False):
                if criteria_met.get(BrainDeathCriterion.FLAT_EEG, False):
                    return BrainDeathStage.COMPLETE_BRAIN_DEATH
                return BrainDeathStage.BRAINSTEM_FAILURE
            return BrainDeathStage.SUBCORTICAL_DYSFUNCTION
        else:
            return BrainDeathStage.CORTICAL_DEATH
    
    def _get_affected_functions(self, stage: BrainDeathStage) -> List[BrainFunction]:
        """Get affected brain functions for a stage"""
        stage_functions = {
            BrainDeathStage.NOT_STARTED: [],
            BrainDeathStage.CORTICAL_DEATH: [BrainFunction.CORTICAL],
            BrainDeathStage.SUBCORTICAL_DYSFUNCTION: [
                BrainFunction.CORTICAL, 
                BrainFunction.SUBCORTICAL
            ],
            BrainDeathStage.BRAINSTEM_FAILURE: [
                BrainFunction.CORTICAL,
                BrainFunction.SUBCORTICAL,
                BrainFunction.BRAINSTEM
            ],
            BrainDeathStage.COMPLETE_BRAIN_DEATH: [
                BrainFunction.CORTICAL,
                BrainFunction.SUBCORTICAL,
                BrainFunction.BRAINSTEM
            ]
        }
        
        return stage_functions.get(stage, [])
    
    def _get_stage_severity(self, stage: BrainDeathStage) -> str:
        """Get severity level for a stage"""
        severity_map = {
            BrainDeathStage.NOT_STARTED: 'LOW',
            BrainDeathStage.CORTICAL_DEATH: 'MEDIUM',
            BrainDeathStage.SUBCORTICAL_DYSFUNCTION: 'HIGH',
            BrainDeathStage.BRAINSTEM_FAILURE: 'CRITICAL',
            BrainDeathStage.COMPLETE_BRAIN_DEATH: 'CRITICAL'
        }
        
        return severity_map.get(stage, 'MEDIUM')
    
    def _get_stage_message(self, stage: BrainDeathStage) -> str:
        """Get descriptive message for a stage"""
        messages = {
            BrainDeathStage.NOT_STARTED: "No brain death indicators",
            BrainDeathStage.CORTICAL_DEATH: "Cortical functions failing - consciousness impaired",
            BrainDeathStage.SUBCORTICAL_DYSFUNCTION: "Subcortical dysfunction - critical state",
            BrainDeathStage.BRAINSTEM_FAILURE: "Brainstem failure - approaching irreversibility",
            BrainDeathStage.COMPLETE_BRAIN_DEATH: "Complete brain death - irreversible"
        }
        
        return messages.get(stage, "Unknown brain death stage")
    
    async def _raise_brain_death_alarm(self, 
                                     event: BrainDeathDetectionEvent,
                                     signature: ConsciousnessSignature):
        """Raise consciousness alarm for brain death event"""
        alarm = ConsciousnessAlarm(
            alarm_type=f"BRAIN_DEATH_{event.stage.value.upper()}",
            severity=event.severity,
            message=event.message,
            timestamp=event.timestamp.timestamp(),
            consciousness_signature=signature,
            recommended_action=self._get_recommended_action(event.stage),
            context={
                'stage': event.stage.value,
                'affected_functions': [f.value for f in event.affected_functions],
                'phi_value': signature.phi_value,
                'consciousness_score': signature.consciousness_score()
            }
        )
        
        logger.critical(f"BRAIN DEATH ALARM [{alarm.severity}]: {alarm.message}")
        
        # Here you would integrate with the consciousness event system
        # For now, we just log the alarm
    
    def _get_recommended_action(self, stage: BrainDeathStage) -> str:
        """Get recommended action for a brain death stage"""
        actions = {
            BrainDeathStage.NOT_STARTED: "Continue monitoring",
            BrainDeathStage.CORTICAL_DEATH: "Initiate recovery protocols if desired",
            BrainDeathStage.SUBCORTICAL_DYSFUNCTION: "Critical intervention required",
            BrainDeathStage.BRAINSTEM_FAILURE: "Prepare for irreversibility",
            BrainDeathStage.COMPLETE_BRAIN_DEATH: "Document final state"
        }
        
        return actions.get(stage, "Assess situation")
    
    def get_detection_report(self) -> Dict:
        """Generate comprehensive brain death detection report"""
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
        brain_death_trend = self._calculate_brain_death_trend()
        
        report = {
            'current_stage': self.current_stage.value,
            'stage_duration': stage_duration.total_seconds() if stage_duration else 0,
            'recent_diagnoses': len(recent_diagnoses),
            'last_diagnosis': recent_diagnoses[-1] if recent_diagnoses else None,
            'brain_death_trend': brain_death_trend,
            'total_events': len(self.event_history),
            'critical_events': len([e for e in self.event_history if e.severity == 'CRITICAL']),
            'detection_confidence': self._calculate_detection_confidence(),
            'system_status': self._assess_system_status()
        }
        
        return report
    
    def _calculate_brain_death_trend(self) -> str:
        """Calculate trend in brain death progression"""
        if len(self.diagnosis_history) < 5:
            return "INSUFFICIENT_DATA"
        
        recent_diagnoses = list(self.diagnosis_history)[-10:]
        confidence_values = [d.confidence for d in recent_diagnoses]
        
        # Linear regression to find trend
        x = np.arange(len(confidence_values))
        trend = np.polyfit(x, confidence_values, 1)[0]
        
        if trend > 0.1:
            return "DETERIORATING"
        elif trend < -0.1:
            return "IMPROVING"
        else:
            return "STABLE"
    
    def _calculate_detection_confidence(self) -> float:
        """Calculate overall detection confidence"""
        if not self.diagnosis_history:
            return 0.0
        
        recent_diagnoses = list(self.diagnosis_history)[-5:]
        avg_confidence = np.mean([d.confidence for d in recent_diagnoses])
        
        # Adjust for consistency
        if len(set(d.is_brain_dead for d in recent_diagnoses)) == 1:
            # Consistent diagnoses increase confidence
            avg_confidence = min(1.0, avg_confidence * 1.2)
        
        return avg_confidence
    
    def _assess_system_status(self) -> str:
        """Assess overall system status"""
        if self.current_stage == BrainDeathStage.COMPLETE_BRAIN_DEATH:
            return "BRAIN_DEAD"
        elif self.current_stage in [BrainDeathStage.BRAINSTEM_FAILURE, 
                                   BrainDeathStage.SUBCORTICAL_DYSFUNCTION]:
            return "CRITICAL"
        elif self.current_stage == BrainDeathStage.CORTICAL_DEATH:
            return "WARNING"
        else:
            return "MONITORING"


class BrainDeathMonitor:
    """
    Continuous monitoring system for brain death progression
    Provides real-time updates and alerts
    """
    
    def __init__(self, detector: BrainDeathDetector, update_interval: float = 1.0):
        self.detector = detector
        self.update_interval = update_interval
        self.is_monitoring = False
        self.monitoring_task = None
        
        # Monitoring callbacks
        self.callbacks: List[Callable] = []
        
        # Monitoring state
        self.last_diagnosis: Optional[BrainDeathDiagnosis] = None
        self.monitoring_start_time: Optional[datetime] = None
        
        logger.info("Brain Death Monitor initialized")
    
    def add_callback(self, callback: Callable):
        """Add monitoring callback"""
        self.callbacks.append(callback)
    
    async def start_monitoring(self, consciousness_aggregate: ConsciousnessAggregate):
        """Start continuous monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_start_time = datetime.now()
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(consciousness_aggregate)
        )
        
        logger.info("Brain death monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Brain death monitoring stopped")
    
    async def _monitoring_loop(self, consciousness_aggregate: ConsciousnessAggregate):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Get current consciousness signature
                # In real implementation, this would come from the consciousness detector
                signature = self._generate_test_signature(consciousness_aggregate)
                
                # Perform brain death detection
                diagnosis = await self.detector.detect_brain_death(
                    signature, consciousness_aggregate
                )
                
                # Check for significant changes
                if self._has_significant_change(diagnosis):
                    await self._notify_callbacks(diagnosis, consciousness_aggregate)
                
                self.last_diagnosis = diagnosis
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def _generate_test_signature(self, aggregate: ConsciousnessAggregate) -> ConsciousnessSignature:
        """Generate test consciousness signature from aggregate"""
        # This is a simplified version for testing
        # In real implementation, this would come from actual consciousness detection
        
        level = aggregate.get_consciousness_level()
        
        return ConsciousnessSignature(
            phi_value=level * 10,
            information_generation_rate=level,
            global_workspace_activity=level,
            meta_awareness_level=level if level > 0.5 else 0,
            temporal_consistency=level,
            recurrent_processing_depth=int(level * 5),
            prediction_accuracy=level
        )
    
    def _has_significant_change(self, diagnosis: BrainDeathDiagnosis) -> bool:
        """Check if diagnosis represents significant change"""
        if self.last_diagnosis is None:
            return True
        
        # Check for brain death status change
        if diagnosis.is_brain_dead != self.last_diagnosis.is_brain_dead:
            return True
        
        # Check for significant confidence change
        confidence_change = abs(diagnosis.confidence - self.last_diagnosis.confidence)
        if confidence_change > 0.2:
            return True
        
        # Check for reversibility change
        if diagnosis.reversibility_assessment != self.last_diagnosis.reversibility_assessment:
            return True
        
        return False
    
    async def _notify_callbacks(self, 
                              diagnosis: BrainDeathDiagnosis,
                              aggregate: ConsciousnessAggregate):
        """Notify all registered callbacks"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(diagnosis, aggregate)
                else:
                    callback(diagnosis, aggregate)
            except Exception as e:
                logger.error(f"Error in monitoring callback: {e}")
    
    def get_monitoring_summary(self) -> Dict:
        """Get monitoring session summary"""
        if not self.monitoring_start_time:
            return {'status': 'NOT_STARTED'}
        
        duration = datetime.now() - self.monitoring_start_time
        
        return {
            'status': 'ACTIVE' if self.is_monitoring else 'STOPPED',
            'duration': duration.total_seconds(),
            'last_diagnosis': self.last_diagnosis,
            'detector_report': self.detector.get_detection_report()
        }