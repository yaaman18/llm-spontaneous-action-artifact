"""
Consciousness Repository Implementations - Interface Adapters Layer
Implement repository interfaces defined in application layer

Following Clean Architecture principles:
- Implements repository interfaces from application layer
- Handles data persistence and retrieval
- Isolates external data sources from business logic
- Uses dependency injection for external services

Author: Clean Architecture Engineer (Uncle Bob's expertise)
Date: 2025-08-03
Version: 1.0.0
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime
from dataclasses import asdict
import logging

# Application layer imports (implementing protocols)
from ..application.consciousness_use_cases import (
    IPhiCalculationRepository, IConsciousnessEventRepository, 
    IDevelopmentRepository, INotificationService
)

# Domain layer imports
from ..domain.consciousness_entities import (
    SystemState, PhiStructure, PhiValue, ConsciousnessEvent, 
    DevelopmentStage, Distinction, CauseEffectState
)

logger = logging.getLogger(__name__)


# External service abstractions (dependency inversion)
class IPhiCalculationEngine(Protocol):
    """Interface for external phi calculation engine"""
    
    async def compute_phi_structure(self, system_state: SystemState) -> PhiStructure:
        """Compute phi structure using external engine"""
        ...


class IDataStore(Protocol):
    """Interface for external data storage"""
    
    async def save(self, collection: str, data: Dict[str, Any]) -> str:
        """Save data to collection"""
        ...
    
    async def find(self, collection: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find data in collection"""
        ...
    
    async def find_by_timespan(self, collection: str, 
                              start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Find data in timespan"""
        ...


class IMessageBroker(Protocol):
    """Interface for external message broker"""
    
    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publish message to topic"""
        ...


# Repository Implementations

class PhiCalculationRepository(IPhiCalculationRepository):
    """
    Repository for phi calculations
    Coordinates with external calculation engines and data storage
    """
    
    def __init__(self, 
                 calculation_engine: IPhiCalculationEngine,
                 data_store: IDataStore):
        self._calculation_engine = calculation_engine
        self._data_store = data_store
        self._calculation_cache = {}
    
    async def calculate_phi(self, system_state: SystemState) -> PhiStructure:
        """
        Calculate phi structure using external engine
        
        Args:
            system_state: System state to analyze
            
        Returns:
            PhiStructure: Calculated phi structure
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(system_state)
            if cache_key in self._calculation_cache:
                logger.debug(f"Returning cached phi calculation for key: {cache_key}")
                return self._calculation_cache[cache_key]
            
            # Calculate using external engine
            phi_structure = await self._calculation_engine.compute_phi_structure(system_state)
            
            # Cache result
            self._calculation_cache[cache_key] = phi_structure
            
            # Maintain cache size
            if len(self._calculation_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self._calculation_cache.keys())[:100]
                for key in oldest_keys:
                    del self._calculation_cache[key]
            
            return phi_structure
            
        except Exception as e:
            logger.error(f"Phi calculation failed: {e}")
            raise
    
    async def save_calculation_result(self, result: PhiStructure) -> str:
        """
        Save calculation result to persistent storage
        
        Args:
            result: PhiStructure to save
            
        Returns:
            str: Unique ID of saved calculation
        """
        try:
            # Convert to serializable format
            calculation_data = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "system_phi_value": result.system_phi.value,
                "system_phi_precision": result.system_phi.precision,
                "consciousness_level": result.consciousness_level.name,
                "development_stage": result.development_stage.name,
                "distinction_count": len(result.distinctions),
                "relation_count": len(result.relations),
                "complexity": result.complexity,
                "is_conscious": result.is_conscious(),
                "system_state": {
                    "nodes": list(result.system_state.nodes),
                    "state_vector": list(result.system_state.state_vector),
                    "connectivity_matrix": [list(row) for row in result.system_state.connectivity_matrix],
                    "dimension": result.system_state.dimension
                }
            }
            
            # Save to data store
            calculation_id = await self._data_store.save("phi_calculations", calculation_data)
            
            logger.info(f"Saved phi calculation result with ID: {calculation_id}")
            return calculation_id
            
        except Exception as e:
            logger.error(f"Failed to save calculation result: {e}")
            raise
    
    def _generate_cache_key(self, system_state: SystemState) -> str:
        """Generate cache key for system state"""
        # Create deterministic hash from system state
        state_str = f"{sorted(system_state.nodes)}_{system_state.state_vector}_{system_state.connectivity_matrix}"
        return str(hash(state_str))


class ConsciousnessEventRepository(IConsciousnessEventRepository):
    """
    Repository for consciousness events
    Handles event persistence and retrieval
    """
    
    def __init__(self, data_store: IDataStore):
        self._data_store = data_store
    
    async def save_event(self, event: ConsciousnessEvent) -> str:
        """
        Save consciousness event to persistent storage
        
        Args:
            event: ConsciousnessEvent to save
            
        Returns:
            str: Event ID
        """
        try:
            # Convert to serializable format
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "current_phi_value": event.current_phi.value,
                "current_phi_precision": event.current_phi.precision,
                "previous_phi_value": event.previous_phi.value if event.previous_phi else None,
                "previous_phi_precision": event.previous_phi.precision if event.previous_phi else None,
                "phi_change": event.phi_change,
                "consciousness_emergence": event.represents_consciousness_emergence(),
                "significant_change": event.represents_significant_change(),
                "system_state": {
                    "nodes": list(event.system_state.nodes),
                    "state_vector": list(event.system_state.state_vector),
                    "connectivity_matrix": [list(row) for row in event.system_state.connectivity_matrix],
                    "dimension": event.system_state.dimension
                },
                "metadata": event.metadata
            }
            
            # Save to data store
            saved_id = await self._data_store.save("consciousness_events", event_data)
            
            logger.info(f"Saved consciousness event: {event.event_type} with ID: {saved_id}")
            return saved_id
            
        except Exception as e:
            logger.error(f"Failed to save consciousness event: {e}")
            raise
    
    async def get_events_by_timespan(self, start: datetime, end: datetime) -> List[ConsciousnessEvent]:
        """
        Retrieve events within timespan
        
        Args:
            start: Start datetime
            end: End datetime
            
        Returns:
            List of ConsciousnessEvent objects
        """
        try:
            # Query data store
            event_data_list = await self._data_store.find_by_timespan(
                "consciousness_events", start, end
            )
            
            # Convert back to domain objects
            events = []
            for event_data in event_data_list:
                event = self._convert_to_consciousness_event(event_data)
                events.append(event)
            
            logger.info(f"Retrieved {len(events)} events between {start} and {end}")
            return events
            
        except Exception as e:
            logger.error(f"Failed to retrieve events by timespan: {e}")
            raise
    
    def _convert_to_consciousness_event(self, event_data: Dict[str, Any]) -> ConsciousnessEvent:
        """Convert stored data back to ConsciousnessEvent domain object"""
        # Reconstruct system state
        system_state = SystemState(
            nodes=frozenset(event_data["system_state"]["nodes"]),
            state_vector=tuple(event_data["system_state"]["state_vector"]),
            connectivity_matrix=tuple(
                tuple(row) for row in event_data["system_state"]["connectivity_matrix"]
            ),
            timestamp=datetime.fromisoformat(event_data["timestamp"])
        )
        
        # Reconstruct phi values
        current_phi = PhiValue(
            value=event_data["current_phi_value"],
            precision=event_data["current_phi_precision"],
            timestamp=datetime.fromisoformat(event_data["timestamp"])
        )
        
        previous_phi = None
        if event_data["previous_phi_value"] is not None:
            previous_phi = PhiValue(
                value=event_data["previous_phi_value"],
                precision=event_data["previous_phi_precision"],
                timestamp=datetime.fromisoformat(event_data["timestamp"])
            )
        
        return ConsciousnessEvent(
            event_id=event_data["event_id"],
            timestamp=datetime.fromisoformat(event_data["timestamp"]),
            previous_phi=previous_phi,
            current_phi=current_phi,
            system_state=system_state,
            event_type=event_data["event_type"],
            metadata=event_data["metadata"]
        )


class DevelopmentRepository(IDevelopmentRepository):
    """
    Repository for consciousness development tracking
    """
    
    def __init__(self, data_store: IDataStore):
        self._data_store = data_store
    
    async def save_development_state(self, stage: DevelopmentStage, phi_structure: PhiStructure) -> str:
        """
        Save development state
        
        Args:
            stage: Current development stage
            phi_structure: Associated phi structure
            
        Returns:
            str: Development record ID
        """
        try:
            development_data = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "development_stage": stage.name,
                "stage_value": stage.value,
                "phi_value": phi_structure.system_phi.value,
                "consciousness_level": phi_structure.consciousness_level.name,
                "complexity": phi_structure.complexity,
                "distinction_count": len(phi_structure.distinctions),
                "is_conscious": phi_structure.is_conscious(),
                "system_state": {
                    "nodes": list(phi_structure.system_state.nodes),
                    "dimension": phi_structure.system_state.dimension
                }
            }
            
            development_id = await self._data_store.save("development_states", development_data)
            
            logger.info(f"Saved development state: {stage.name} with ID: {development_id}")
            return development_id
            
        except Exception as e:
            logger.error(f"Failed to save development state: {e}")
            raise
    
    async def get_development_history(self, system_id: str) -> List[tuple]:
        """
        Get development history for system
        
        Args:
            system_id: System identifier
            
        Returns:
            List of (timestamp, stage, phi_value) tuples
        """
        try:
            # Query development states
            query = {"system_id": system_id} if system_id else {}
            development_data = await self._data_store.find("development_states", query)
            
            # Convert to tuples
            history = []
            for data in development_data:
                timestamp = datetime.fromisoformat(data["timestamp"])
                stage = DevelopmentStage(data["stage_value"])
                phi_value = data["phi_value"]
                history.append((timestamp, stage, phi_value))
            
            # Sort by timestamp
            history.sort(key=lambda x: x[0])
            
            logger.info(f"Retrieved development history with {len(history)} entries")
            return history
            
        except Exception as e:
            logger.error(f"Failed to retrieve development history: {e}")
            raise


class NotificationService(INotificationService):
    """
    Service for consciousness change notifications
    """
    
    def __init__(self, message_broker: IMessageBroker):
        self._message_broker = message_broker
    
    async def notify_consciousness_change(self, event: ConsciousnessEvent) -> None:
        """
        Send notification about consciousness change
        
        Args:
            event: ConsciousnessEvent to notify about
        """
        try:
            # Create notification message
            notification = {
                "notification_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "event_id": event.event_id,
                "event_type": event.event_type,
                "phi_value": event.current_phi.value,
                "phi_change": event.phi_change,
                "consciousness_emergence": event.represents_consciousness_emergence(),
                "significant_change": event.represents_significant_change(),
                "system_dimension": event.system_state.dimension,
                "metadata": event.metadata
            }
            
            # Determine notification topic based on event type
            if event.represents_consciousness_emergence():
                topic = "consciousness.emergence"
            elif event.represents_significant_change():
                topic = "consciousness.significant_change"
            else:
                topic = "consciousness.general"
            
            # Publish notification
            await self._message_broker.publish(topic, notification)
            
            logger.info(f"Published consciousness notification to topic: {topic}")
            
        except Exception as e:
            logger.error(f"Failed to send consciousness notification: {e}")
            raise


# Adapter Factory for Repository Creation
class ConsciousnessRepositoryFactory:
    """
    Factory for creating repository instances with dependency injection
    """
    
    def __init__(self,
                 calculation_engine: IPhiCalculationEngine,
                 data_store: IDataStore,
                 message_broker: IMessageBroker):
        self._calculation_engine = calculation_engine
        self._data_store = data_store
        self._message_broker = message_broker
    
    def create_phi_calculation_repository(self) -> IPhiCalculationRepository:
        """Create phi calculation repository"""
        return PhiCalculationRepository(self._calculation_engine, self._data_store)
    
    def create_consciousness_event_repository(self) -> IConsciousnessEventRepository:
        """Create consciousness event repository"""
        return ConsciousnessEventRepository(self._data_store)
    
    def create_development_repository(self) -> IDevelopmentRepository:
        """Create development repository"""
        return DevelopmentRepository(self._data_store)
    
    def create_notification_service(self) -> INotificationService:
        """Create notification service"""
        return NotificationService(self._message_broker)