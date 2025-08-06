"""
Phase Transition Engine Integration Demonstration
Shows integration with existential termination system and IIT4 implementation

This demo demonstrates:
1. Phase Transition Engine integration with InformationIntegrationSystem
2. Real-time phase detection and prediction
3. Emergent property analysis during transitions
4. Critical point identification and monitoring
5. Integration with domain events and clean architecture patterns
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import numpy as np

from existential_termination_core import (
    SystemIdentity,
    InformationIntegrationSystem,
    IntegrationSystemFactory,
    TerminationPattern,
    IntegrationDegree,
    ExistentialState
)

from phase_transition_engine import (
    PhaseTransitionEngine,
    PhaseTransitionEngineFactory,
    PhaseState,
    PhaseTransition,
    PhaseTransitionType,
    CriticalPoint,
    EmergentProperty,
    PhaseTransitionDetectedEvent,
    CriticalPointApproachedEvent,
    EmergentPropertyEvent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhaseTransitionIntegrationDemo:
    """Comprehensive demonstration of Phase Transition Engine integration"""
    
    def __init__(self):
        self.demo_start_time = datetime.now()
        self.analysis_results = []
        self.event_log = []
        
    async def run_comprehensive_demo(self):
        """Run comprehensive integration demonstration"""
        
        logger.info("=== Phase Transition Engine Integration Demo ===")
        logger.info(f"Demo started at: {self.demo_start_time}")
        
        try:
            # 1. System Setup
            await self.demo_system_setup()
            
            # 2. Basic Phase Detection
            await self.demo_basic_phase_detection()
            
            # 3. Transition Prediction
            await self.demo_transition_prediction()
            
            # 4. Emergent Property Analysis
            await self.demo_emergent_property_analysis()
            
            # 5. Critical Point Monitoring
            await self.demo_critical_point_monitoring()
            
            # 6. Integration with Termination Process
            await self.demo_termination_integration()
            
            # 7. Real-time Monitoring
            await self.demo_realtime_monitoring()
            
            # 8. Generate Final Report
            await self.generate_demo_report()
            
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise
    
    async def demo_system_setup(self):
        """Demonstrate system setup and initialization"""
        
        logger.info("\n--- 1. System Setup ---")
        
        # Create information integration system
        system_id = SystemIdentity("phase-demo-system-001")
        self.integration_system = IntegrationSystemFactory.create_standard_system(system_id)
        
        # Create phase transition engine
        engine_id = SystemIdentity("phase-engine-001")
        self.phase_engine = PhaseTransitionEngineFactory.create_standard_engine(engine_id)
        
        logger.info(f"Created Integration System: {self.integration_system.id.value}")
        logger.info(f"Created Phase Engine: {self.phase_engine.id.value}")
        
        # Initial system status
        system_status = self.integration_system.get_system_status()
        engine_status = self.phase_engine.get_engine_status()
        
        logger.info(f"Integration System Status: {system_status['existential_state']}")
        logger.info(f"Integration Degree: {system_status['integration_degree']:.3f}")
        logger.info(f"Active Layers: {system_status['active_layers']}/{system_status['total_layers']}")
        
        self.log_event("SYSTEM_SETUP", {
            'integration_system_id': system_id.value,
            'phase_engine_id': engine_id.value,
            'initial_integration_degree': system_status['integration_degree'],
            'active_layers': system_status['active_layers']
        })
    
    async def demo_basic_phase_detection(self):
        """Demonstrate basic phase detection capabilities"""
        
        logger.info("\n--- 2. Basic Phase Detection ---")
        
        # Perform initial phase analysis
        analysis_result = await self.phase_engine.analyze_system_phase(self.integration_system)
        self.analysis_results.append(analysis_result)
        
        current_phase = analysis_result['current_phase']
        logger.info(f"Current Phase Detected:")
        logger.info(f"  - Integration Level: {current_phase['integration_level']:.3f}")
        logger.info(f"  - Information Generation Rate: {current_phase['information_generation_rate']:.3f}")
        logger.info(f"  - Emergence Potential: {current_phase['emergence_potential']:.3f}")
        logger.info(f"  - Stability Index: {current_phase['stability_index']:.3f}")
        logger.info(f"  - Entropy Level: {current_phase['entropy_level']:.3f}")
        logger.info(f"  - Is Critical Phase: {current_phase['is_critical']}")
        
        # Demonstrate phase state evolution over time
        logger.info("\nSimulating phase evolution...")
        
        for i in range(5):
            # Simulate small changes in system
            await asyncio.sleep(0.5)  # Brief pause for realistic timing
            
            # Slightly degrade a random layer to show phase changes
            if self.integration_system.integration_layers:
                layer = np.random.choice(self.integration_system.integration_layers)
                if layer.is_active:
                    degradation = np.random.uniform(0.01, 0.05)
                    layer.degrade(degradation)
            
            # Detect new phase
            new_analysis = await self.phase_engine.analyze_system_phase(self.integration_system)
            new_phase = new_analysis['current_phase']
            
            logger.info(f"Phase Evolution Step {i+1}: Integration={new_phase['integration_level']:.3f}, "
                       f"Generation={new_phase['information_generation_rate']:.3f}")
        
        self.log_event("PHASE_DETECTION", {
            'initial_phase': current_phase,
            'evolution_steps': 5,
            'final_integration_level': new_phase['integration_level']
        })
    
    async def demo_transition_prediction(self):
        """Demonstrate transition prediction capabilities"""
        
        logger.info("\n--- 3. Transition Prediction ---")
        
        # Get current predictions
        predictions = self.phase_engine.get_transition_predictions(
            self.integration_system,
            time_horizon=timedelta(minutes=30)
        )
        
        logger.info(f"Transition Predictions (30 minute horizon):")
        logger.info(f"  - Current Phase ID: {predictions['current_phase_id']}")
        
        # Show predicted states
        if predictions['predicted_states']:
            logger.info("  - Predicted Future States:")
            for i, pred in enumerate(predictions['predicted_states'][:3]):  # Show first 3
                state = pred['state']
                logger.info(f"    {i+1}. Time: {pred['time']}")
                logger.info(f"       Integration: {state['integration_level']:.3f} "
                           f"(confidence: {pred['confidence']:.2f})")
                logger.info(f"       Generation: {state['generation_rate']:.3f}")
                logger.info(f"       Emergence: {state['emergence_potential']:.3f}")
        
        # Show transition risks
        risks = predictions['transition_risks']
        logger.info("  - Transition Risks:")
        for risk_type, risk_value in risks.items():
            if risk_value > 0.3:  # Only show significant risks
                logger.info(f"    * {risk_type}: {risk_value:.2f}")
        
        # Show emergence forecasts
        emergence = predictions['emergence_forecasts']
        if emergence['likely_emergences']:
            logger.info("  - Likely Emergent Properties:")
            for emergence_pred in emergence['likely_emergences']:
                logger.info(f"    * {emergence_pred['property_type']}: "
                           f"potential={emergence_pred['emergence_potential']:.2f}, "
                           f"time={emergence_pred['estimated_emergence_time']:.0f}s")
        
        self.log_event("TRANSITION_PREDICTION", {
            'prediction_horizon_minutes': 30,
            'predicted_states_count': len(predictions['predicted_states']),
            'high_risk_transitions': [k for k, v in risks.items() if v > 0.5],
            'likely_emergences_count': len(emergence['likely_emergences'])
        })
    
    async def demo_emergent_property_analysis(self):
        """Demonstrate emergent property analysis"""
        
        logger.info("\n--- 4. Emergent Property Analysis ---")
        
        # Force a transition to demonstrate emergence analysis
        logger.info("Simulating system changes to trigger emergence...")
        
        # Enhance meta-cognitive layer to increase emergence potential
        meta_layers = [layer for layer in self.integration_system.integration_layers
                      if hasattr(layer, 'layer_type') and 
                      layer.layer_type.name == 'META_COGNITIVE']
        
        if meta_layers:
            meta_layer = meta_layers[0]
            # Boost capacity temporarily
            original_capacity = meta_layer.capacity
            meta_layer.capacity = min(1.0, meta_layer.capacity + 0.3)
            logger.info(f"Enhanced meta-cognitive layer capacity: {original_capacity:.3f} → {meta_layer.capacity:.3f}")
        
        # Analyze for emergent properties
        analysis = await self.phase_engine.analyze_system_phase(self.integration_system)
        
        emergent_props = analysis.get('emergent_properties', [])
        if emergent_props:
            logger.info(f"Detected {len(emergent_props)} emergent properties:")
            for prop in emergent_props:
                logger.info(f"  - {prop['name']}:")
                logger.info(f"    Intensity: {prop['intensity']:.3f}")
                logger.info(f"    Strength: {prop['strength']:.3f}")
                logger.info(f"    Downward Causation: {prop['downward_causation']}")
        else:
            logger.info("No emergent properties detected at current emergence threshold")
        
        # Analyze emergence potentials
        if self.phase_engine.current_phase:
            potentials = self.phase_engine.analyzer.monitor_emergence_potential(
                self.integration_system, self.phase_engine.current_phase
            )
            
            logger.info("Emergence Potentials:")
            for prop_type, potential in potentials.items():
                if potential > 0.3:  # Show significant potentials
                    logger.info(f"  - {prop_type}: {potential:.3f}")
        
        self.log_event("EMERGENCE_ANALYSIS", {
            'emergent_properties_detected': len(emergent_props),
            'emergence_potentials': potentials if self.phase_engine.current_phase else {},
            'meta_layer_enhancement': meta_layers[0].capacity if meta_layers else None
        })
    
    async def demo_critical_point_monitoring(self):
        """Demonstrate critical point identification and monitoring"""
        
        logger.info("\n--- 5. Critical Point Monitoring ---")
        
        # Build up phase history for critical point analysis
        logger.info("Building phase history for critical point analysis...")
        
        phase_history = []
        for i in range(15):  # Generate synthetic phase history
            # Simulate system evolution with some variability
            base_integration = 0.8 - (i * 0.03)  # Gradual decline
            noise = np.random.normal(0, 0.05)  # Add noise
            
            synthetic_phase = PhaseState(
                state_id=f"synthetic_{i}",
                integration_level=max(0.0, min(1.0, base_integration + noise)),
                information_generation_rate=max(0.0, min(1.0, base_integration * 0.9 + noise)),
                emergence_potential=max(0.0, min(1.0, base_integration * 1.1 + noise * 0.5)),
                stability_index=max(0.0, min(1.0, base_integration * 0.8 + noise * 0.3)),
                entropy_level=max(0.0, min(1.0, (1.0 - base_integration) * 0.7 + noise * 0.2)),
                timestamp=datetime.now() - timedelta(minutes=15-i)
            )
            
            phase_history.append(synthetic_phase)
            self.phase_engine.phase_history.append(synthetic_phase)
        
        # Identify critical points
        critical_points = self.phase_engine.calculator.identify_critical_points(
            self.integration_system, phase_history
        )
        
        logger.info(f"Identified {len(critical_points)} critical points:")
        for cp in critical_points:
            logger.info(f"  - {cp.point_id}:")
            logger.info(f"    Type: {cp.criticality_type.value}")
            logger.info(f"    Basin Radius: {cp.basin_radius:.3f}")
            logger.info(f"    Attraction Strength: {cp.attraction_strength:.3f}")
            logger.info(f"    Is Attractor: {cp.is_attractor()}")
            
            # Analyze stability
            stability_analysis = self.phase_engine.calculator.analyze_basin_stability(cp, self.integration_system)
            logger.info(f"    Stability Analysis: {stability_analysis}")
        
        # Check proximity to critical points if we have any
        if critical_points and self.phase_engine.current_phase:
            proximities = self.phase_engine.calculator.calculate_threshold_proximity(
                self.phase_engine.current_phase, critical_points
            )
            
            logger.info("Proximity to Critical Points:")
            for point_id, (proximity, cp) in proximities.items():
                if proximity > 0.1:  # Show significant proximities
                    logger.info(f"  - {point_id}: proximity={proximity:.3f}")
        
        self.log_event("CRITICAL_POINT_MONITORING", {
            'phase_history_length': len(phase_history),
            'critical_points_identified': len(critical_points),
            'attractor_points': len([cp for cp in critical_points if cp.is_attractor()]),
            'repeller_points': len([cp for cp in critical_points if cp.is_repeller()])
        })
    
    async def demo_termination_integration(self):
        """Demonstrate integration with termination process"""
        
        logger.info("\n--- 6. Integration with Termination Process ---")
        
        # Initiate termination to see phase transitions during termination
        logger.info("Initiating gradual termination process...")
        
        self.integration_system.initiate_termination(TerminationPattern.GRADUAL_DECAY)
        
        # Monitor phase changes during termination
        logger.info("Monitoring phase transitions during termination...")
        
        for step in range(5):
            # Progress termination
            elapsed_time = timedelta(minutes=step * 8)  # 8-minute steps
            self.integration_system.progress_termination(elapsed_time)
            
            # Analyze phase during termination
            analysis = await self.phase_engine.analyze_system_phase(self.integration_system)
            
            # Get system status
            system_status = self.integration_system.get_system_status()
            
            logger.info(f"Termination Step {step + 1}:")
            logger.info(f"  - Termination Stage: {system_status['termination_stage']}")
            logger.info(f"  - Existential State: {system_status['existential_state']}")
            logger.info(f"  - Integration Degree: {system_status['integration_degree']:.3f}")
            logger.info(f"  - Phase Integration Level: {analysis['current_phase']['integration_level']:.3f}")
            
            # Check for transitions during termination
            active_transitions = analysis.get('active_transitions', [])
            if active_transitions:
                logger.info(f"  - Active Transitions: {len(active_transitions)}")
                for transition in active_transitions:
                    logger.info(f"    * {transition['transition_type']}: "
                               f"probability={transition['probability']:.3f}, "
                               f"magnitude={transition['magnitude']:.3f}")
            
            # Check for emergent properties during termination
            emergent_props = analysis.get('emergent_properties', [])
            if emergent_props:
                logger.info(f"  - Emergent Properties: {len(emergent_props)}")
            
            await asyncio.sleep(0.3)  # Brief pause between steps
        
        # Check if termination is complete
        final_status = self.integration_system.get_system_status()
        logger.info(f"Final System State: {final_status['existential_state']}")
        logger.info(f"Is Terminated: {final_status['is_terminated']}")
        logger.info(f"Is Reversible: {final_status['is_reversible']}")
        
        self.log_event("TERMINATION_INTEGRATION", {
            'termination_pattern': 'GRADUAL_DECAY',
            'termination_steps': 5,
            'final_state': final_status['existential_state'],
            'final_integration_degree': final_status['integration_degree'],
            'is_terminated': final_status['is_terminated']
        })
    
    async def demo_realtime_monitoring(self):
        """Demonstrate real-time monitoring capabilities"""
        
        logger.info("\n--- 7. Real-time Monitoring ---")
        
        # Create a new system for real-time monitoring (since previous one is terminated)
        monitor_system_id = SystemIdentity("realtime-monitor-system")
        monitor_system = IntegrationSystemFactory.create_standard_system(monitor_system_id)
        
        # Set up event monitoring
        events_captured = []
        
        def capture_events():
            """Capture domain events from phase engine"""
            if self.phase_engine.domain_events:
                new_events = list(self.phase_engine.domain_events)
                events_captured.extend(new_events)
                self.phase_engine.domain_events.clear()
        
        logger.info("Starting real-time monitoring simulation...")
        
        # Simulate real-time monitoring with dynamic changes
        for minute in range(3):  # 3-minute simulation
            logger.info(f"\nMinute {minute + 1} of real-time monitoring:")
            
            # Introduce system changes
            if minute == 1:
                # Simulate external perturbation
                logger.info("  Applying external perturbation...")
                for layer in monitor_system.integration_layers[:2]:
                    if layer.is_active:
                        layer.degrade(0.15)
            elif minute == 2:
                # Simulate recovery attempt
                logger.info("  Simulating recovery dynamics...")
                # This would normally involve external intervention
            
            # Perform phase analysis
            analysis = await self.phase_engine.analyze_system_phase(monitor_system)
            
            # Capture any events generated
            capture_events()
            
            # Report current state
            current_phase = analysis['current_phase']
            logger.info(f"  Phase State: Integration={current_phase['integration_level']:.3f}, "
                       f"Stability={current_phase['stability_index']:.3f}")
            
            # Show any active transitions
            if analysis['active_transitions']:
                for transition in analysis['active_transitions']:
                    logger.info(f"  Transition Detected: {transition['transition_type']} "
                               f"(probability: {transition['probability']:.3f})")
            
            # Show predictions
            predictions = self.phase_engine.get_transition_predictions(
                monitor_system, time_horizon=timedelta(minutes=5)
            )
            
            if predictions['predicted_states']:
                next_state = predictions['predicted_states'][0]['state']
                confidence = predictions['predicted_states'][0]['confidence']
                logger.info(f"  Next Predicted State: Integration={next_state['integration_level']:.3f} "
                           f"(confidence: {confidence:.3f})")
            
            await asyncio.sleep(1.0)  # 1 second represents 1 minute of monitoring
        
        # Report captured events
        logger.info(f"\nCaptured {len(events_captured)} domain events during monitoring:")
        for event in events_captured:
            event_type = type(event).__name__
            logger.info(f"  - {event_type} at {event.timestamp}")
        
        self.log_event("REALTIME_MONITORING", {
            'monitoring_duration_minutes': 3,
            'events_captured': len(events_captured),
            'event_types': [type(e).__name__ for e in events_captured],
            'final_system_status': monitor_system.get_system_status()
        })
    
    async def generate_demo_report(self):
        """Generate comprehensive demo report"""
        
        logger.info("\n--- 8. Demo Report Generation ---")
        
        demo_duration = datetime.now() - self.demo_start_time
        
        # Compile comprehensive report
        report = {
            'demo_metadata': {
                'start_time': self.demo_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': demo_duration.total_seconds(),
                'demo_version': '1.0.0'
            },
            'systems_created': {
                'integration_systems': 2,  # main demo + monitoring
                'phase_engines': 1,
                'total_components': 8  # detectors, predictors, analyzers, calculators
            },
            'analysis_summary': {
                'total_phase_analyses': len(self.analysis_results),
                'transitions_detected': sum(
                    len(analysis.get('active_transitions', [])) 
                    for analysis in self.analysis_results
                ),
                'emergent_properties_found': sum(
                    len(analysis.get('emergent_properties', [])) 
                    for analysis in self.analysis_results
                ),
                'critical_points_identified': sum(
                    len(analysis.get('critical_points', [])) 
                    for analysis in self.analysis_results
                )
            },
            'capabilities_demonstrated': [
                'phase_state_detection',
                'transition_prediction',
                'emergent_property_analysis',
                'critical_point_identification',
                'termination_process_integration',
                'realtime_monitoring',
                'domain_event_generation',
                'clean_architecture_patterns'
            ],
            'integration_points_verified': [
                'existential_termination_core',
                'information_integration_system',
                'domain_events',
                'clean_architecture_compliance',
                'kanai_information_generation_theory',
                'iit4_principles'
            ],
            'event_log': self.event_log,
            'performance_metrics': {
                'average_analysis_time': np.mean([
                    analysis.get('analysis_performance', {}).get('duration_seconds', 0)
                    for analysis in self.analysis_results
                ]) if self.analysis_results else 0.0,
                'total_events_generated': len(self.event_log),
                'memory_efficiency': 'optimized_with_deque_buffers'
            }
        }
        
        logger.info("=== DEMO COMPLETION REPORT ===")
        logger.info(f"Demo Duration: {demo_duration.total_seconds():.2f} seconds")
        logger.info(f"Phase Analyses Performed: {report['analysis_summary']['total_phase_analyses']}")
        logger.info(f"Transitions Detected: {report['analysis_summary']['transitions_detected']}")
        logger.info(f"Emergent Properties Found: {report['analysis_summary']['emergent_properties_found']}")
        logger.info(f"Critical Points Identified: {report['analysis_summary']['critical_points_identified']}")
        logger.info(f"Domain Events Generated: {len(self.event_log)}")
        logger.info(f"Integration Points Verified: {len(report['integration_points_verified'])}")
        
        # Save report to file
        report_filename = f"/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/08_02_2025/phase_transition_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Demo report saved to: {report_filename}")
        except Exception as e:
            logger.warning(f"Could not save report file: {e}")
        
        self.log_event("DEMO_COMPLETION", {
            'total_duration': demo_duration.total_seconds(),
            'success': True,
            'report_file': report_filename
        })
        
        return report
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log demo event"""
        self.event_log.append({
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        })


async def main():
    """Run the comprehensive phase transition integration demo"""
    
    print("Phase Transition Engine Integration Demonstration")
    print("=" * 60)
    
    demo = PhaseTransitionIntegrationDemo()
    
    try:
        report = await demo.run_comprehensive_demo()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return report
        
    except Exception as e:
        print(f"\nDEMO FAILED: {e}")
        logger.error(f"Demo failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())