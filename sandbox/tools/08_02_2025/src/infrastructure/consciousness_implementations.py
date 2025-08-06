"""
Consciousness Infrastructure Implementations - Infrastructure Layer
Concrete implementations of external services and frameworks

Following Clean Architecture principles:
- Implements interfaces defined in adapter layer
- Contains framework-specific code
- Handles external system integration
- Isolated from business logic

Author: Clean Architecture Engineer (Uncle Bob's expertise)
Date: 2025-08-03
Version: 1.0.0
"""

import asyncio
import json
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging
import pickle
import gzip
from contextlib import asynccontextmanager

# Optional numpy import (will create simple fallback if not available)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Simple numpy-like operations for basic functionality
    class SimpleArray:
        def __init__(self, data):
            if isinstance(data, (list, tuple)):
                self.data = list(data)
            else:
                self.data = [data]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def sum(self):
            return sum(self.data)
        
        def mean(self):
            return sum(self.data) / len(self.data) if self.data else 0
        
        def clip(self, min_val, max_val):
            return SimpleArray([max(min_val, min(max_val, x)) for x in self.data])
    
    class MockNumpy:
        def array(self, data):
            return SimpleArray(data)
        
        def sum(self, arr, axis=None):
            if hasattr(arr, 'sum'):
                return arr.sum()
            return sum(arr)
        
        def any(self, arr):
            return any(arr)
        
        def log(self, arr):
            import math
            if hasattr(arr, 'data'):
                return SimpleArray([math.log(max(1e-10, x)) for x in arr.data])
            return math.log(max(1e-10, arr))
    
    np = MockNumpy()

# Adapter layer imports (implementing interfaces)
from ..adapters.consciousness_repositories import (
    IPhiCalculationEngine, IDataStore, IMessageBroker
)

# Domain layer imports  
from ..domain.consciousness_entities import (
    SystemState, PhiStructure, PhiValue, Distinction, 
    CauseEffectState, DevelopmentStage, ConsciousnessLevel
)

logger = logging.getLogger(__name__)


class IIT4PhiCalculationEngine(IPhiCalculationEngine):
    """
    Concrete implementation of phi calculation using IIT 4.0 algorithms
    Wraps existing implementations while isolating infrastructure concerns
    """
    
    def __init__(self, precision: float = 1e-10, max_mechanism_size: int = 8):
        self._precision = precision
        self._max_mechanism_size = max_mechanism_size
        self._calculation_stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "average_time_ms": 0.0
        }
    
    async def compute_phi_structure(self, system_state: SystemState) -> PhiStructure:
        """
        Compute phi structure using IIT 4.0 algorithms
        
        Args:
            system_state: System state to analyze
            
        Returns:
            PhiStructure: Complete phi structure with distinctions and relations
        """
        import time
        start_time = time.time()
        
        try:
            self._calculation_stats["total_calculations"] += 1
            
            # Convert domain entities to computation format
            state_vector = np.array(system_state.state_vector)
            connectivity_matrix = np.array(system_state.connectivity_matrix)
            
            # Validate inputs
            if not self._validate_inputs(state_vector, connectivity_matrix):
                raise ValueError("Invalid inputs for phi calculation")
            
            # Calculate phi structure components
            distinctions = await self._calculate_distinctions(
                system_state, state_vector, connectivity_matrix
            )
            
            relations = await self._calculate_relations(distinctions)
            
            system_phi = await self._calculate_system_phi(
                distinctions, relations, state_vector, connectivity_matrix
            )
            
            # Determine development stage
            development_stage = self._determine_development_stage(system_phi, len(distinctions))
            
            # Create phi structure
            phi_structure = PhiStructure(
                distinctions=tuple(distinctions),
                relations=tuple(relations),
                system_phi=system_phi,
                system_state=system_state,
                development_stage=development_stage
            )
            
            # Update statistics
            self._calculation_stats["successful_calculations"] += 1
            calc_time = (time.time() - start_time) * 1000
            self._update_average_time(calc_time)
            
            logger.debug(f"Phi calculation completed in {calc_time:.2f}ms")
            return phi_structure
            
        except Exception as e:
            self._calculation_stats["failed_calculations"] += 1
            logger.error(f"Phi calculation failed: {e}")
            raise
    
    async def _calculate_distinctions(self, system_state: SystemState,
                                    state_vector: np.ndarray,
                                    connectivity_matrix: np.ndarray) -> List[Distinction]:
        """Calculate distinctions for all valid mechanisms"""
        distinctions = []
        nodes = list(system_state.nodes)
        
        # Calculate distinctions for all possible mechanisms
        for size in range(1, min(len(nodes) + 1, self._max_mechanism_size + 1)):
            for mechanism_nodes in self._get_combinations(nodes, size):
                mechanism = frozenset(mechanism_nodes)
                
                # Calculate cause-effect state
                ces = await self._calculate_cause_effect_state(
                    mechanism, state_vector, connectivity_matrix
                )
                
                # Only create distinction if phi > 0
                if ces.phi_value.value > 0:
                    distinction = Distinction(
                        mechanism=mechanism,
                        cause_effect_state=ces,
                        phi_value=ces.phi_value
                    )
                    distinctions.append(distinction)
        
        # Sort by phi value (highest first)
        distinctions.sort(key=lambda d: d.phi_value.value, reverse=True)
        
        return distinctions
    
    async def _calculate_cause_effect_state(self, mechanism: frozenset,
                                          state_vector: np.ndarray,
                                          connectivity_matrix: np.ndarray) -> CauseEffectState:
        """Calculate cause-effect state for a mechanism"""
        # Simplified IIT 4.0 calculation (in real implementation, this would be much more complex)
        mechanism_indices = sorted(list(mechanism))
        
        # Calculate cause state distribution
        cause_state = self._calculate_cause_distribution(
            mechanism_indices, state_vector, connectivity_matrix
        )
        
        # Calculate effect state distribution  
        effect_state = self._calculate_effect_distribution(
            mechanism_indices, state_vector, connectivity_matrix
        )
        
        # Calculate intrinsic difference
        intrinsic_difference = self._calculate_intrinsic_difference(
            cause_state, effect_state
        )
        
        # Calculate phi value for this mechanism
        phi_value = PhiValue(
            value=max(0.0, intrinsic_difference * 0.5),  # Simplified calculation
            precision=self._precision
        )
        
        return CauseEffectState(
            mechanism=mechanism,
            cause_state=tuple(cause_state),
            effect_state=tuple(effect_state),
            intrinsic_difference=intrinsic_difference,
            phi_value=phi_value
        )
    
    def _calculate_cause_distribution(self, mechanism_indices: List[int],
                                    state_vector, connectivity_matrix):
        """Calculate cause state distribution"""
        # Simplified calculation - in reality this involves complex probability calculations
        if HAS_NUMPY:
            cause_weights = np.sum(connectivity_matrix[mechanism_indices, :], axis=1)
            cause_state = state_vector[mechanism_indices] * cause_weights
            
            # Normalize to probability distribution
            if np.sum(cause_state) > 0:
                cause_state = cause_state / np.sum(cause_state)
        else:
            # Simple fallback calculation
            cause_state = [state_vector[i] * 0.5 for i in mechanism_indices]
            total = sum(cause_state)
            if total > 0:
                cause_state = [x / total for x in cause_state]
        
        return cause_state
    
    def _calculate_effect_distribution(self, mechanism_indices: List[int],
                                     state_vector: np.ndarray,
                                     connectivity_matrix: np.ndarray) -> np.ndarray:
        """Calculate effect state distribution"""
        # Simplified calculation
        effect_weights = np.sum(connectivity_matrix[:, mechanism_indices], axis=0)
        effect_state = state_vector[mechanism_indices] * effect_weights
        
        # Normalize to probability distribution
        if np.sum(effect_state) > 0:
            effect_state = effect_state / np.sum(effect_state)
        
        return effect_state
    
    def _calculate_intrinsic_difference(self, cause_state: np.ndarray,
                                      effect_state: np.ndarray) -> float:
        """Calculate intrinsic difference between cause and effect states"""
        # Jensen-Shannon divergence as measure of intrinsic difference
        def kl_divergence(p, q):
            epsilon = 1e-10
            p = np.clip(p, epsilon, 1.0)
            q = np.clip(q, epsilon, 1.0)
            return np.sum(p * np.log(p / q))
        
        # Average of cause and effect distributions
        m = (cause_state + effect_state) / 2
        
        # Jensen-Shannon divergence
        js_divergence = 0.5 * kl_divergence(cause_state, m) + 0.5 * kl_divergence(effect_state, m)
        
        return float(js_divergence)
    
    async def _calculate_relations(self, distinctions: List[Distinction]) -> List[Tuple[int, int, float]]:
        """Calculate relations between distinctions"""
        relations = []
        
        for i, dist1 in enumerate(distinctions):
            for j, dist2 in enumerate(distinctions[i+1:], i+1):
                # Calculate overlap between mechanisms
                overlap = len(dist1.mechanism.intersection(dist2.mechanism))
                total_nodes = len(dist1.mechanism.union(dist2.mechanism))
                
                if total_nodes > 0:
                    overlap_ratio = overlap / total_nodes
                    
                    # Calculate integration strength based on phi values and overlap
                    integration_strength = (
                        (dist1.phi_value.value * dist2.phi_value.value * overlap_ratio) ** 0.5
                    )
                    
                    if integration_strength > 0.01:  # Threshold for significant relations
                        relations.append((i, j, integration_strength))
        
        return relations
    
    async def _calculate_system_phi(self, distinctions: List[Distinction],
                                  relations: List[Tuple[int, int, float]],
                                  state_vector: np.ndarray,
                                  connectivity_matrix: np.ndarray) -> PhiValue:
        """Calculate overall system phi value"""
        if not distinctions:
            return PhiValue(value=0.0, precision=self._precision)
        
        # Aggregate phi from distinctions
        distinction_phi = sum(d.phi_value.value for d in distinctions)
        
        # Add integration bonus from relations
        relation_bonus = sum(strength for _, _, strength in relations) * 0.1
        
        # Apply normalization
        system_size = len(state_vector)
        normalized_phi = (distinction_phi + relation_bonus) / system_size
        
        return PhiValue(value=normalized_phi, precision=self._precision)
    
    def _determine_development_stage(self, system_phi: PhiValue, distinction_count: int) -> DevelopmentStage:
        """Determine development stage based on phi value and complexity"""
        phi_val = system_phi.value
        
        if phi_val < 0.1:
            return DevelopmentStage.STAGE_0_REFLEXIVE
        elif phi_val < 0.2:
            return DevelopmentStage.STAGE_1_REACTIVE
        elif phi_val < 0.3:
            return DevelopmentStage.STAGE_2_ADAPTIVE
        elif phi_val < 0.5 and distinction_count < 10:
            return DevelopmentStage.STAGE_3_PREDICTIVE
        elif phi_val < 0.7 and distinction_count < 20:
            return DevelopmentStage.STAGE_4_REFLECTIVE
        elif phi_val < 0.8:
            return DevelopmentStage.STAGE_5_INTROSPECTIVE
        else:
            return DevelopmentStage.STAGE_6_METACOGNITIVE
    
    def _validate_inputs(self, state_vector: np.ndarray, connectivity_matrix: np.ndarray) -> bool:
        """Validate calculation inputs"""
        if state_vector.size == 0:
            return False
        
        if connectivity_matrix.shape[0] != connectivity_matrix.shape[1]:
            return False
        
        if len(state_vector) != connectivity_matrix.shape[0]:
            return False
        
        # Check state vector bounds
        if np.any(state_vector < 0) or np.any(state_vector > 1):
            return False
        
        return True
    
    def _get_combinations(self, items: List[Any], size: int) -> List[List[Any]]:
        """Get all combinations of specified size"""
        import itertools
        return list(itertools.combinations(items, size))
    
    def _update_average_time(self, calc_time_ms: float) -> None:
        """Update average calculation time"""
        current_avg = self._calculation_stats["average_time_ms"]
        total_calcs = self._calculation_stats["successful_calculations"]
        
        if total_calcs == 1:
            self._calculation_stats["average_time_ms"] = calc_time_ms
        else:
            # Moving average
            self._calculation_stats["average_time_ms"] = (
                (current_avg * (total_calcs - 1) + calc_time_ms) / total_calcs
            )
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """Get calculation performance statistics"""
        return self._calculation_stats.copy()


class SqliteDataStore(IDataStore):
    """
    SQLite implementation of data store interface
    Provides persistent storage with async operations
    """
    
    def __init__(self, database_path: str = "consciousness.db"):
        self._database_path = Path(database_path)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database tables"""
        with sqlite3.connect(self._database_path) as conn:
            cursor = conn.cursor()
            
            # Phi calculations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS phi_calculations (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            # Consciousness events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS consciousness_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            # Development states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS development_states (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    development_stage TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_phi_timestamp ON phi_calculations(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_timestamp ON consciousness_events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dev_timestamp ON development_states(timestamp)")
            
            conn.commit()
    
    async def save(self, collection: str, data: Dict[str, Any]) -> str:
        """Save data to collection"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._save_sync, collection, data
        )
    
    def _save_sync(self, collection: str, data: Dict[str, Any]) -> str:
        """Synchronous save operation"""
        with sqlite3.connect(self._database_path) as conn:
            cursor = conn.cursor()
            
            record_id = data.get("id", data.get("event_id", str(hash(str(data)))))
            timestamp = data.get("timestamp", datetime.now().isoformat())
            data_json = json.dumps(data)
            
            if collection == "phi_calculations":
                cursor.execute(
                    "INSERT OR REPLACE INTO phi_calculations (id, timestamp, data) VALUES (?, ?, ?)",
                    (record_id, timestamp, data_json)
                )
            elif collection == "consciousness_events":
                event_type = data.get("event_type", "unknown")
                cursor.execute(
                    "INSERT OR REPLACE INTO consciousness_events (id, timestamp, event_type, data) VALUES (?, ?, ?, ?)",
                    (record_id, timestamp, event_type, data_json)
                )
            elif collection == "development_states":
                dev_stage = data.get("development_stage", "unknown")
                cursor.execute(
                    "INSERT OR REPLACE INTO development_states (id, timestamp, development_stage, data) VALUES (?, ?, ?, ?)",
                    (record_id, timestamp, dev_stage, data_json)
                )
            else:
                raise ValueError(f"Unknown collection: {collection}")
            
            conn.commit()
            return record_id
    
    async def find(self, collection: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find data in collection"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._find_sync, collection, query
        )
    
    def _find_sync(self, collection: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synchronous find operation"""
        with sqlite3.connect(self._database_path) as conn:
            cursor = conn.cursor()
            
            if collection == "phi_calculations":
                cursor.execute("SELECT data FROM phi_calculations ORDER BY timestamp DESC LIMIT 1000")
            elif collection == "consciousness_events":
                cursor.execute("SELECT data FROM consciousness_events ORDER BY timestamp DESC LIMIT 1000")
            elif collection == "development_states":
                cursor.execute("SELECT data FROM development_states ORDER BY timestamp DESC LIMIT 1000")
            else:
                raise ValueError(f"Unknown collection: {collection}")
            
            results = []
            for (data_json,) in cursor.fetchall():
                data = json.loads(data_json)
                
                # Apply simple query filtering
                match = True
                for key, value in query.items():
                    if key in data and data[key] != value:
                        match = False
                        break
                
                if match:
                    results.append(data)
            
            return results
    
    async def find_by_timespan(self, collection: str, 
                              start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Find data within timespan"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._find_by_timespan_sync, collection, start, end
        )
    
    def _find_by_timespan_sync(self, collection: str, 
                              start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Synchronous timespan find operation"""
        with sqlite3.connect(self._database_path) as conn:
            cursor = conn.cursor()
            
            start_str = start.isoformat()
            end_str = end.isoformat()
            
            if collection == "phi_calculations":
                cursor.execute(
                    "SELECT data FROM phi_calculations WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                    (start_str, end_str)
                )
            elif collection == "consciousness_events":
                cursor.execute(
                    "SELECT data FROM consciousness_events WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                    (start_str, end_str)
                )
            elif collection == "development_states":
                cursor.execute(
                    "SELECT data FROM development_states WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                    (start_str, end_str)
                )
            else:
                raise ValueError(f"Unknown collection: {collection}")
            
            results = []
            for (data_json,) in cursor.fetchall():
                data = json.loads(data_json)
                results.append(data)
            
            return results


class InMemoryMessageBroker(IMessageBroker):
    """
    In-memory message broker implementation
    For development and testing purposes
    """
    
    def __init__(self):
        self._subscribers = {}
        self._message_history = []
    
    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publish message to topic"""
        timestamped_message = {
            **message,
            "published_at": datetime.now().isoformat(),
            "topic": topic
        }
        
        # Store in history
        self._message_history.append(timestamped_message)
        
        # Keep only last 1000 messages
        if len(self._message_history) > 1000:
            self._message_history = self._message_history[-1000:]
        
        # Notify subscribers (if any)
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    await callback(timestamped_message)
                except Exception as e:
                    logger.error(f"Subscriber callback failed for topic {topic}: {e}")
        
        logger.info(f"Published message to topic: {topic}")
    
    def subscribe(self, topic: str, callback):
        """Subscribe to topic (for testing/monitoring)"""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
    
    def get_message_history(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get message history for debugging"""
        if topic:
            return [msg for msg in self._message_history if msg["topic"] == topic]
        return self._message_history.copy()


# Infrastructure Factory
class ConsciousnessInfrastructureFactory:
    """
    Factory for creating infrastructure components
    """
    
    def __init__(self, database_path: str = "consciousness.db"):
        self._database_path = database_path
        self._phi_engine = None
        self._data_store = None
        self._message_broker = None
    
    def create_phi_calculation_engine(self, precision: float = 1e-10, 
                                    max_mechanism_size: int = 8) -> IPhiCalculationEngine:
        """Create phi calculation engine"""
        if self._phi_engine is None:
            self._phi_engine = IIT4PhiCalculationEngine(precision, max_mechanism_size)
        return self._phi_engine
    
    def create_data_store(self) -> IDataStore:
        """Create data store"""
        if self._data_store is None:
            self._data_store = SqliteDataStore(self._database_path)
        return self._data_store
    
    def create_message_broker(self) -> IMessageBroker:
        """Create message broker"""
        if self._message_broker is None:
            self._message_broker = InMemoryMessageBroker()
        return self._message_broker