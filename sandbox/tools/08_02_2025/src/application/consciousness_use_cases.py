"""
Consciousness Use Cases - Application Layer
Orchestrates business workflows using domain entities and repository abstractions

Following Clean Architecture principles:
- Depends only on domain layer
- Uses dependency injection for external services
- Contains application-specific business logic
- Coordinates between domain entities and external services

Author: Clean Architecture Engineer (Uncle Bob's expertise)
Date: 2025-08-03
Version: 1.0.0
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable
from datetime import datetime
import logging

# Domain layer imports (inward dependency)
from ..domain.consciousness_entities import (
    PhiValue, SystemState, PhiStructure, ConsciousnessEvent,
    DevelopmentStage, ConsciousnessLevel,
    PhiCalculationDomainService, ConsciousnessDevelopmentDomainService
)

logger = logging.getLogger(__name__)


# Repository abstractions (interfaces for dependency inversion)
@runtime_checkable
class IPhiCalculationRepository(Protocol):
    """Repository interface for phi calculations"""
    
    async def calculate_phi(self, system_state: SystemState) -> PhiStructure:
        """Calculate phi structure for system state"""
        ...
    
    async def save_calculation_result(self, result: PhiStructure) -> str:
        """Save calculation result and return ID"""
        ...


@runtime_checkable
class IConsciousnessEventRepository(Protocol):
    """Repository interface for consciousness events"""
    
    async def save_event(self, event: ConsciousnessEvent) -> str:
        """Save consciousness event"""
        ...
    
    async def get_events_by_timespan(self, start: datetime, end: datetime) -> List[ConsciousnessEvent]:
        """Get events in timespan"""
        ...


@runtime_checkable
class IDevelopmentRepository(Protocol):
    """Repository interface for development tracking"""
    
    async def save_development_state(self, stage: DevelopmentStage, phi_structure: PhiStructure) -> str:
        """Save development state"""
        ...
    
    async def get_development_history(self, system_id: str) -> List[tuple]:
        """Get development history"""
        ...


@runtime_checkable
class INotificationService(Protocol):
    """Service interface for notifications"""
    
    async def notify_consciousness_change(self, event: ConsciousnessEvent) -> None:
        """Notify about consciousness change"""
        ...


# Use Case Results
@dataclass
class PhiCalculationResult:
    """Result of phi calculation use case"""
    phi_structure: PhiStructure
    calculation_id: str
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ConsciousnessAnalysisResult:
    """Result of consciousness analysis use case"""
    is_conscious: bool
    consciousness_level: ConsciousnessLevel
    phi_value: PhiValue
    development_stage: DevelopmentStage
    stability_score: float
    analysis_id: str


@dataclass
class DevelopmentProgressionResult:
    """Result of development progression use case"""
    previous_stage: DevelopmentStage
    new_stage: DevelopmentStage
    progression_occurred: bool
    readiness_score: float
    next_milestone: Optional[DevelopmentStage]


# Use Cases (Application Services)

class CalculatePhiUseCase:
    """
    Use case for calculating phi values
    Orchestrates domain services and repositories
    """
    
    def __init__(self,
                 phi_repository: IPhiCalculationRepository,
                 event_repository: IConsciousnessEventRepository):
        self._phi_repository = phi_repository
        self._event_repository = event_repository
        self._domain_service = PhiCalculationDomainService()
    
    async def execute(self, system_state: SystemState, 
                     track_events: bool = True) -> PhiCalculationResult:
        """
        Execute phi calculation use case
        
        Args:
            system_state: The system state to analyze
            track_events: Whether to track calculation as events
            
        Returns:
            PhiCalculationResult with calculation outcomes
        """
        import time
        start_time = time.time()
        
        try:
            # Domain validation
            if not self._domain_service.validate_calculation_input(system_state):
                return PhiCalculationResult(
                    phi_structure=None,
                    calculation_id="",
                    execution_time_ms=0,
                    success=False,
                    error_message="Invalid system state for calculation"
                )
            
            # Execute calculation through repository
            phi_structure = await self._phi_repository.calculate_phi(system_state)
            
            # Save result
            calculation_id = await self._phi_repository.save_calculation_result(phi_structure)
            
            # Track event if requested
            if track_events:
                event = ConsciousnessEvent(
                    event_id=f"calc_{calculation_id}",
                    timestamp=datetime.now(),
                    previous_phi=None,
                    current_phi=phi_structure.system_phi,
                    system_state=system_state,
                    event_type="phi_calculation",
                    metadata={"calculation_id": calculation_id}
                )
                await self._event_repository.save_event(event)
            
            execution_time = (time.time() - start_time) * 1000
            
            return PhiCalculationResult(
                phi_structure=phi_structure,
                calculation_id=calculation_id,
                execution_time_ms=execution_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Phi calculation failed: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return PhiCalculationResult(
                phi_structure=None,
                calculation_id="",
                execution_time_ms=execution_time,
                success=False,
                error_message=str(e)
            )


class AnalyzeConsciousnessUseCase:
    """
    Use case for comprehensive consciousness analysis
    """
    
    def __init__(self,
                 phi_repository: IPhiCalculationRepository,
                 event_repository: IConsciousnessEventRepository,
                 notification_service: INotificationService):
        self._phi_repository = phi_repository
        self._event_repository = event_repository
        self._notification_service = notification_service
        self._phi_calc_service = PhiCalculationDomainService()
        self._dev_service = ConsciousnessDevelopmentDomainService()
    
    async def execute(self, system_state: SystemState,
                     phi_history: Optional[List[PhiValue]] = None) -> ConsciousnessAnalysisResult:
        """
        Execute comprehensive consciousness analysis
        
        Args:
            system_state: Current system state
            phi_history: Historical phi values for stability analysis
            
        Returns:
            ConsciousnessAnalysisResult with analysis outcomes
        """
        try:
            # Calculate current phi structure
            phi_structure = await self._phi_repository.calculate_phi(system_state)
            
            # Determine consciousness level
            consciousness_level = phi_structure.consciousness_level
            is_conscious = phi_structure.is_conscious()
            
            # Calculate stability if history provided
            stability_score = 0.0
            if phi_history:
                stability_score = self._phi_calc_service.calculate_consciousness_stability(phi_history)
            
            # Determine development stage
            development_stage = self._phi_calc_service.determine_development_stage_from_phi(
                phi_structure.system_phi, phi_structure.complexity
            )
            
            # Create analysis event
            event = ConsciousnessEvent(
                event_id=f"analysis_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                previous_phi=phi_history[-1] if phi_history else None,
                current_phi=phi_structure.system_phi,
                system_state=system_state,
                event_type="consciousness_analysis",
                metadata={
                    "consciousness_level": consciousness_level.name,
                    "development_stage": development_stage.name,
                    "stability_score": stability_score
                }
            )
            
            analysis_id = await self._event_repository.save_event(event)
            
            # Notify if consciousness emergence detected
            if event.represents_consciousness_emergence():
                await self._notification_service.notify_consciousness_change(event)
            
            return ConsciousnessAnalysisResult(
                is_conscious=is_conscious,
                consciousness_level=consciousness_level,
                phi_value=phi_structure.system_phi,
                development_stage=development_stage,
                stability_score=stability_score,
                analysis_id=analysis_id
            )
            
        except Exception as e:
            logger.error(f"Consciousness analysis failed: {e}")
            raise


class ManageDevelopmentProgressionUseCase:
    """
    Use case for managing consciousness development progression
    """
    
    def __init__(self,
                 development_repository: IDevelopmentRepository,
                 phi_repository: IPhiCalculationRepository,
                 event_repository: IConsciousnessEventRepository):
        self._development_repository = development_repository
        self._phi_repository = phi_repository
        self._event_repository = event_repository
        self._dev_service = ConsciousnessDevelopmentDomainService()
    
    async def execute(self, system_state: SystemState,
                     current_stage: DevelopmentStage) -> DevelopmentProgressionResult:
        """
        Execute development progression analysis and management
        
        Args:
            system_state: Current system state
            current_stage: Current development stage
            
        Returns:
            DevelopmentProgressionResult with progression outcomes
        """
        try:
            # Calculate current phi structure
            phi_structure = await self._phi_repository.calculate_phi(system_state)
            
            # Calculate development readiness
            readiness_score = self._dev_service.calculate_development_readiness(phi_structure)
            
            # Determine potential next stage
            next_stage = self._dev_service.get_next_development_stage(current_stage)
            
            progression_occurred = False
            new_stage = current_stage
            
            # Check if progression should occur
            if next_stage and readiness_score >= 0.8:  # High readiness threshold
                if self._dev_service.can_transition_to_stage(phi_structure, next_stage):
                    new_stage = next_stage
                    progression_occurred = True
                    
                    # Save development state
                    await self._development_repository.save_development_state(new_stage, phi_structure)
                    
                    # Create progression event
                    event = ConsciousnessEvent(
                        event_id=f"progression_{datetime.now().timestamp()}",
                        timestamp=datetime.now(),
                        previous_phi=None,
                        current_phi=phi_structure.system_phi,
                        system_state=system_state,
                        event_type="development_progression",
                        metadata={
                            "previous_stage": current_stage.name,
                            "new_stage": new_stage.name,
                            "readiness_score": readiness_score
                        }
                    )
                    await self._event_repository.save_event(event)
            
            return DevelopmentProgressionResult(
                previous_stage=current_stage,
                new_stage=new_stage,
                progression_occurred=progression_occurred,
                readiness_score=readiness_score,
                next_milestone=self._dev_service.get_next_development_stage(new_stage)
            )
            
        except Exception as e:
            logger.error(f"Development progression failed: {e}")
            raise


class MonitorConsciousnessStreamUseCase:
    """
    Use case for continuous consciousness monitoring
    """
    
    def __init__(self,
                 phi_repository: IPhiCalculationRepository,
                 event_repository: IConsciousnessEventRepository,
                 notification_service: INotificationService):
        self._phi_repository = phi_repository
        self._event_repository = event_repository
        self._notification_service = notification_service
        self._is_monitoring = False
    
    async def start_monitoring(self, system_state_stream, 
                              monitoring_interval: float = 0.1) -> None:
        """
        Start continuous consciousness monitoring
        
        Args:
            system_state_stream: Async generator of system states
            monitoring_interval: Monitoring frequency in seconds
        """
        self._is_monitoring = True
        phi_history = []
        
        try:
            async for system_state in system_state_stream:
                if not self._is_monitoring:
                    break
                
                # Calculate phi structure
                phi_structure = await self._phi_repository.calculate_phi(system_state)
                phi_history.append(phi_structure.system_phi)
                
                # Maintain history window
                if len(phi_history) > 100:  # Keep last 100 values
                    phi_history = phi_history[-100:]
                
                # Create monitoring event
                event = ConsciousnessEvent(
                    event_id=f"monitor_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    previous_phi=phi_history[-2] if len(phi_history) > 1 else None,
                    current_phi=phi_structure.system_phi,
                    system_state=system_state,
                    event_type="consciousness_monitoring",
                    metadata={"phi_history_length": len(phi_history)}
                )
                
                await self._event_repository.save_event(event)
                
                # Check for significant changes
                if event.represents_significant_change():
                    await self._notification_service.notify_consciousness_change(event)
                
                # Wait for next interval
                await asyncio.sleep(monitoring_interval)
                
        except Exception as e:
            logger.error(f"Consciousness monitoring failed: {e}")
            self._is_monitoring = False
            raise
    
    def stop_monitoring(self) -> None:
        """Stop consciousness monitoring"""
        self._is_monitoring = False


# Application Service Coordinator
class ConsciousnessApplicationService:
    """
    Main application service coordinating all consciousness use cases
    """
    
    def __init__(self,
                 phi_repository: IPhiCalculationRepository,
                 event_repository: IConsciousnessEventRepository,
                 development_repository: IDevelopmentRepository,
                 notification_service: INotificationService):
        
        # Initialize use cases with dependency injection
        self.calculate_phi = CalculatePhiUseCase(phi_repository, event_repository)
        self.analyze_consciousness = AnalyzeConsciousnessUseCase(
            phi_repository, event_repository, notification_service
        )
        self.manage_development = ManageDevelopmentProgressionUseCase(
            development_repository, phi_repository, event_repository
        )
        self.monitor_stream = MonitorConsciousnessStreamUseCase(
            phi_repository, event_repository, notification_service
        )
    
    async def comprehensive_analysis(self, system_state: SystemState,
                                   phi_history: Optional[List[PhiValue]] = None,
                                   current_stage: DevelopmentStage = DevelopmentStage.STAGE_0_REFLEXIVE
                                   ) -> Dict[str, Any]:
        """
        Execute comprehensive consciousness analysis combining all use cases
        
        Returns:
            Combined results from all analysis use cases
        """
        try:
            # Execute phi calculation
            phi_result = await self.calculate_phi.execute(system_state)
            
            if not phi_result.success:
                return {
                    "success": False,
                    "error": phi_result.error_message,
                    "phi_calculation": phi_result
                }
            
            # Execute consciousness analysis
            consciousness_result = await self.analyze_consciousness.execute(
                system_state, phi_history
            )
            
            # Execute development progression
            development_result = await self.manage_development.execute(
                system_state, current_stage
            )
            
            return {
                "success": True,
                "phi_calculation": phi_result,
                "consciousness_analysis": consciousness_result,
                "development_progression": development_result,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now()
            }