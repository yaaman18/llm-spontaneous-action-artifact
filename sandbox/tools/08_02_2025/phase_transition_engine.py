"""
Phase Transition Engine - 相転移予測エンジン
Implements phase transition detection and prediction for information integration systems
Based on Kanai Ryota's information generation theory and IIT4 principles

Architecture:
- Clean Architecture patterns (Robert C. Martin)
- Domain-Driven Design (Eric Evans)
- Strategy Pattern for different transition types
- Observer Pattern for event propagation
- Value Objects for immutable state representation

Integration:
- Works with InformationIntegrationSystem from existential_termination_core.py
- Uses IntegrationCollapseDetector for real-time analysis
- Follows same abstraction patterns (no biological metaphors)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Protocol, TypeVar, Generic, Callable, Tuple
from enum import Enum, auto
import numpy as np
import logging
import secrets
import hashlib
import asyncio
from collections import deque
import time

from existential_termination_core import (
    SystemIdentity,
    IntegrationDegree,
    ExistentialState,
    ExistentialTransition,
    InformationIntegrationSystem,
    IntegrationLayer,
    IntegrationLayerType,
    TerminationPattern,
    TerminationStage,
    DomainEvent
)

logger = logging.getLogger(__name__)

# Type Variables
T = TypeVar('T')
StateType = TypeVar('StateType', bound='PhaseState')
TransitionType = TypeVar('TransitionType', bound='PhaseTransition')


# ============================================================================
# VALUE OBJECTS (Immutable phase transition representations)
# ============================================================================

@dataclass(frozen=True)
class PhaseState:
    """Represents a distinct phase state in information integration"""
    state_id: str
    integration_level: float  # 0.0 to 1.0
    information_generation_rate: float
    emergence_potential: float
    stability_index: float
    entropy_level: float
    timestamp: datetime
    
    def __post_init__(self):
        if not 0.0 <= self.integration_level <= 1.0:
            raise ValueError("Integration level must be between 0.0 and 1.0")
        if not 0.0 <= self.emergence_potential <= 1.0:
            raise ValueError("Emergence potential must be between 0.0 and 1.0")
        if not 0.0 <= self.stability_index <= 1.0:
            raise ValueError("Stability index must be between 0.0 and 1.0")
        if not 0.0 <= self.entropy_level <= 1.0:
            raise ValueError("Entropy level must be between 0.0 and 1.0")
    
    def is_critical_phase(self) -> bool:
        """Check if this is a critical phase state"""
        return (self.stability_index < 0.3 and 
                self.emergence_potential > 0.7)
    
    def distance_to(self, other: 'PhaseState') -> float:
        """Calculate phase distance to another state"""
        return np.sqrt(
            (self.integration_level - other.integration_level) ** 2 +
            (self.information_generation_rate - other.information_generation_rate) ** 2 +
            (self.emergence_potential - other.emergence_potential) ** 2 +
            (self.stability_index - other.stability_index) ** 2
        )


@dataclass(frozen=True)
class PhaseTransition:
    """Represents a transition between phase states"""
    from_state: PhaseState
    to_state: PhaseState
    transition_type: 'PhaseTransitionType'
    transition_probability: float
    estimated_duration: timedelta
    energy_barrier: float
    reversibility: bool
    catalyst_required: bool
    timestamp: datetime
    
    def __post_init__(self):
        if not 0.0 <= self.transition_probability <= 1.0:
            raise ValueError("Transition probability must be between 0.0 and 1.0")
        if not 0.0 <= self.energy_barrier <= 1.0:
            raise ValueError("Energy barrier must be between 0.0 and 1.0")
    
    def magnitude(self) -> float:
        """Calculate transition magnitude"""
        return self.from_state.distance_to(self.to_state)
    
    def is_spontaneous(self) -> bool:
        """Check if transition is spontaneous (low energy barrier)"""
        return self.energy_barrier < 0.2
    
    def criticality_level(self) -> float:
        """Calculate criticality level of transition"""
        magnitude = self.magnitude()
        return magnitude * (1.0 - self.transition_probability) * self.energy_barrier


@dataclass(frozen=True)
class CriticalPoint:
    """Represents a critical point in phase space"""
    point_id: str
    phase_coordinates: Tuple[float, ...]  # Multi-dimensional phase coordinates
    criticality_type: 'CriticalPointType'
    basin_radius: float
    attraction_strength: float
    stability_eigenvalues: List[float]
    timestamp: datetime
    
    def is_attractor(self) -> bool:
        """Check if this is an attracting critical point"""
        return all(eigenval < 0 for eigenval in self.stability_eigenvalues)
    
    def is_repeller(self) -> bool:
        """Check if this is a repelling critical point"""
        return all(eigenval > 0 for eigenval in self.stability_eigenvalues)
    
    def is_saddle_point(self) -> bool:
        """Check if this is a saddle point"""
        positive = sum(1 for eigenval in self.stability_eigenvalues if eigenval > 0)
        negative = sum(1 for eigenval in self.stability_eigenvalues if eigenval < 0)
        return positive > 0 and negative > 0


@dataclass(frozen=True)
class EmergentProperty:
    """Represents an emergent property during phase transitions"""
    property_id: str
    property_name: str
    emergence_threshold: float
    current_intensity: float
    temporal_persistence: float
    causal_efficacy: float
    downward_causation: bool
    timestamp: datetime
    
    def is_emergent(self) -> bool:
        """Check if property has emerged"""
        return self.current_intensity >= self.emergence_threshold
    
    def emergence_strength(self) -> float:
        """Calculate strength of emergence"""
        if not self.is_emergent():
            return 0.0
        return (self.current_intensity - self.emergence_threshold) * self.causal_efficacy


# ============================================================================
# ENUMS (Phase transition classifications)
# ============================================================================

class PhaseTransitionType(Enum):
    """Types of phase transitions in information integration"""
    CONTINUOUS = "continuous"           # Second-order phase transition
    DISCONTINUOUS = "discontinuous"     # First-order phase transition
    CRITICAL = "critical"               # At critical point
    HYSTERETIC = "hysteretic"          # Path-dependent transition
    AVALANCHE = "avalanche"            # Cascade transition
    QUANTUM = "quantum"                # Quantum-like transition


class CriticalPointType(Enum):
    """Types of critical points in phase space"""
    ATTRACTOR = "attractor"            # Stable equilibrium
    REPELLER = "repeller"              # Unstable equilibrium
    SADDLE = "saddle"                  # Mixed stability
    CENTER = "center"                  # Neutral stability
    SPIRAL_ATTRACTOR = "spiral_attractor"  # Spiral convergence
    SPIRAL_REPELLER = "spiral_repeller"    # Spiral divergence


class EmergenceType(Enum):
    """Types of emergent properties"""
    WEAK_EMERGENCE = "weak_emergence"           # Predictable from components
    STRONG_EMERGENCE = "strong_emergence"       # Novel causal powers
    DIACHRONIC_EMERGENCE = "diachronic_emergence"  # Temporal emergence
    SYNCHRONIC_EMERGENCE = "synchronic_emergence"   # Spatial emergence


# ============================================================================
# DOMAIN EVENTS (Phase transition events)
# ============================================================================

@dataclass
class PhaseTransitionDetectedEvent(DomainEvent):
    """Phase transition detection event"""
    system_id: SystemIdentity
    phase_transition: PhaseTransition
    detection_confidence: float
    affected_layers: List[IntegrationLayerType]
    
    def __post_init__(self):
        super().__init__()


@dataclass
class CriticalPointApproachedEvent(DomainEvent):
    """Critical point approach event"""
    system_id: SystemIdentity
    critical_point: CriticalPoint
    approach_velocity: float
    estimated_arrival: timedelta
    
    def __post_init__(self):
        super().__init__()


@dataclass
class EmergentPropertyEvent(DomainEvent):
    """Emergent property manifestation event"""
    system_id: SystemIdentity
    emergent_property: EmergentProperty
    emergence_context: Dict[str, Any]
    
    def __post_init__(self):
        super().__init__()


@dataclass
class PhaseTransitionCompletedEvent(DomainEvent):
    """Phase transition completion event"""
    system_id: SystemIdentity
    completed_transition: PhaseTransition
    actual_duration: timedelta
    final_state: PhaseState
    
    def __post_init__(self):
        super().__init__()


# ============================================================================
# EXCEPTIONS (Phase transition domain errors)
# ============================================================================

class PhaseTransitionError(Exception):
    """Base exception for phase transition domain"""
    pass


class InvalidPhaseStateError(PhaseTransitionError):
    """Invalid phase state configuration"""
    pass


class TransitionBarrierError(PhaseTransitionError):
    """Cannot overcome transition energy barrier"""
    pass


class CriticalPointStabilityError(PhaseTransitionError):
    """Critical point stability analysis failed"""
    pass


class EmergenceDetectionError(PhaseTransitionError):
    """Emergent property detection failed"""
    pass


# ============================================================================
# PROTOCOLS (Interface abstractions)
# ============================================================================

class PhaseDetectable(Protocol):
    """Protocol for systems that can detect phases"""
    
    def detect_current_phase(self) -> PhaseState:
        """Detect current phase state"""
        ...
    
    def calculate_phase_stability(self) -> float:
        """Calculate current phase stability"""
        ...


class TransitionPredictable(Protocol):
    """Protocol for systems that can predict transitions"""
    
    def predict_next_transition(self, time_horizon: timedelta) -> Optional[PhaseTransition]:
        """Predict next phase transition"""
        ...
    
    def calculate_transition_probability(self, target_state: PhaseState) -> float:
        """Calculate probability of transition to target state"""
        ...


class EmergenceObservable(Protocol):
    """Protocol for systems that can observe emergence"""
    
    def detect_emergent_properties(self) -> List[EmergentProperty]:
        """Detect currently emergent properties"""
        ...
    
    def monitor_emergence_potential(self) -> float:
        """Monitor potential for new emergence"""
        ...


# ============================================================================
# STRATEGY INTERFACES
# ============================================================================

class PhaseTransitionStrategy(ABC):
    """Strategy for different phase transition detection methods"""
    
    @abstractmethod
    def detect_transition(self, 
                         current_state: PhaseState,
                         system: InformationIntegrationSystem) -> Optional[PhaseTransition]:
        """Detect phase transition using specific strategy"""
        pass
    
    @abstractmethod
    def calculate_transition_probability(self,
                                       from_state: PhaseState,
                                       to_state: PhaseState) -> float:
        """Calculate transition probability between states"""
        pass


class CriticalPointStrategy(ABC):
    """Strategy for critical point identification"""
    
    @abstractmethod
    def identify_critical_points(self,
                               system: InformationIntegrationSystem,
                               phase_history: List[PhaseState]) -> List[CriticalPoint]:
        """Identify critical points in phase space"""
        pass
    
    @abstractmethod
    def analyze_stability(self, point: CriticalPoint) -> Dict[str, float]:
        """Analyze stability characteristics of critical point"""
        pass


class EmergenceStrategy(ABC):
    """Strategy for emergent property detection"""
    
    @abstractmethod
    def detect_emergence(self,
                        system: InformationIntegrationSystem,
                        phase_transition: PhaseTransition) -> List[EmergentProperty]:
        """Detect emergent properties during transition"""
        pass
    
    @abstractmethod
    def assess_emergence_strength(self, property: EmergentProperty) -> float:
        """Assess strength of emergent property"""
        pass


# ============================================================================
# CONCRETE STRATEGIES
# ============================================================================

class KanaiInformationGenerationStrategy(PhaseTransitionStrategy):
    """Phase transition strategy based on Kanai Ryota's information generation theory"""
    
    def __init__(self, sensitivity_threshold: float = 0.1):
        self.sensitivity_threshold = sensitivity_threshold
        self.information_generation_cache = deque(maxlen=100)
    
    def detect_transition(self,
                         current_state: PhaseState,
                         system: InformationIntegrationSystem) -> Optional[PhaseTransition]:
        """Detect transition using information generation rate changes"""
        
        # Calculate current information generation
        current_generation = self._calculate_information_generation(system)
        self.information_generation_cache.append((datetime.now(), current_generation))
        
        # Need history for trend detection
        if len(self.information_generation_cache) < 5:
            return None
        
        # Detect significant changes in generation rate
        recent_rates = [rate for _, rate in list(self.information_generation_cache)[-5:]]
        rate_variance = np.var(recent_rates)
        
        if rate_variance > self.sensitivity_threshold:
            # Potential transition detected
            target_state = self._predict_target_state(current_state, current_generation)
            
            return PhaseTransition(
                from_state=current_state,
                to_state=target_state,
                transition_type=PhaseTransitionType.CONTINUOUS,
                transition_probability=min(1.0, rate_variance * 2),
                estimated_duration=timedelta(seconds=30),
                energy_barrier=min(1.0, max(0.0, rate_variance)),  # Ensure valid range
                reversibility=True,
                catalyst_required=False,
                timestamp=datetime.now()
            )
        
        return None
    
    def calculate_transition_probability(self,
                                       from_state: PhaseState,
                                       to_state: PhaseState) -> float:
        """Calculate transition probability based on information generation theory"""
        
        # Distance in information generation space
        generation_distance = abs(to_state.information_generation_rate - 
                                from_state.information_generation_rate)
        
        # Stability difference
        stability_difference = abs(to_state.stability_index - from_state.stability_index)
        
        # Integration level change
        integration_change = abs(to_state.integration_level - from_state.integration_level)
        
        # Combine factors using Kanai's information generation principles
        probability = 1.0 / (1.0 + generation_distance * 2 + stability_difference + integration_change)
        
        return min(1.0, probability)
    
    def _calculate_information_generation(self, system: InformationIntegrationSystem) -> float:
        """Calculate current information generation rate"""
        active_layers = [layer for layer in system.integration_layers if layer.is_active]
        
        if not active_layers:
            return 0.0
        
        # Calculate based on layer interactions and integration degree
        layer_contributions = []
        for layer in active_layers:
            # Layer contribution based on capacity and dependencies
            contribution = layer.capacity * (1.0 + len(layer.dependencies) * 0.1)
            layer_contributions.append(contribution)
        
        # Total generation rate
        total_contribution = sum(layer_contributions)
        integration_factor = system.integration_degree.value
        
        return total_contribution * integration_factor
    
    def _predict_target_state(self, current_state: PhaseState, new_generation: float) -> PhaseState:
        """Predict target state based on new information generation rate"""
        
        # Adjust other parameters based on generation change
        generation_change = new_generation - current_state.information_generation_rate
        
        # Higher generation usually increases emergence potential
        new_emergence = min(1.0, current_state.emergence_potential + generation_change * 0.5)
        
        # Stability might decrease with rapid changes
        new_stability = max(0.0, current_state.stability_index - abs(generation_change) * 0.3)
        
        # Integration level correlates with generation
        new_integration = min(1.0, max(0.0, current_state.integration_level + generation_change * 0.2))
        
        # Entropy changes inversely with organization
        new_entropy = max(0.0, min(1.0, current_state.entropy_level - generation_change * 0.1))
        
        return PhaseState(
            state_id=f"predicted_{secrets.token_hex(8)}",
            integration_level=new_integration,
            information_generation_rate=new_generation,
            emergence_potential=new_emergence,
            stability_index=new_stability,
            entropy_level=new_entropy,
            timestamp=datetime.now()
        )


class AttractorBasedCriticalPointStrategy(CriticalPointStrategy):
    """Critical point identification using attractor dynamics"""
    
    def identify_critical_points(self,
                               system: InformationIntegrationSystem,
                               phase_history: List[PhaseState]) -> List[CriticalPoint]:
        """Identify critical points using attractor basin analysis"""
        
        if len(phase_history) < 10:
            return []
        
        critical_points = []
        
        # Identify potential attractors from clustering in phase space
        coordinates = self._extract_phase_coordinates(phase_history)
        clusters = self._identify_clusters(coordinates)
        
        for i, cluster_center in enumerate(clusters):
            # Analyze stability around cluster center
            eigenvalues = self._calculate_stability_eigenvalues(cluster_center, coordinates)
            
            # Determine critical point type
            if all(eigenval < 0 for eigenval in eigenvalues):
                point_type = CriticalPointType.ATTRACTOR
            elif all(eigenval > 0 for eigenval in eigenvalues):
                point_type = CriticalPointType.REPELLER
            else:
                point_type = CriticalPointType.SADDLE
            
            critical_point = CriticalPoint(
                point_id=f"critical_point_{i}_{secrets.token_hex(4)}",
                phase_coordinates=tuple(cluster_center),
                criticality_type=point_type,
                basin_radius=self._calculate_basin_radius(cluster_center, coordinates),
                attraction_strength=abs(np.mean(eigenvalues)),
                stability_eigenvalues=eigenvalues,
                timestamp=datetime.now()
            )
            
            critical_points.append(critical_point)
        
        return critical_points
    
    def analyze_stability(self, point: CriticalPoint) -> Dict[str, float]:
        """Analyze stability characteristics"""
        eigenvalues = np.array(point.stability_eigenvalues)
        
        return {
            'max_eigenvalue': float(np.max(eigenvalues)),
            'min_eigenvalue': float(np.min(eigenvalues)),
            'stability_measure': float(-np.max(eigenvalues)) if point.is_attractor() else float(np.min(eigenvalues)),
            'oscillation_frequency': float(np.mean(np.abs(eigenvalues))),
            'convergence_rate': float(1.0 / (1.0 + np.max(np.abs(eigenvalues))))
        }
    
    def _extract_phase_coordinates(self, phase_history: List[PhaseState]) -> np.ndarray:
        """Extract coordinates for phase space analysis"""
        coordinates = []
        for state in phase_history:
            coord = [
                state.integration_level,
                state.information_generation_rate,
                state.emergence_potential,
                state.stability_index,
                state.entropy_level
            ]
            coordinates.append(coord)
        
        return np.array(coordinates)
    
    def _identify_clusters(self, coordinates: np.ndarray, n_clusters: int = 3) -> List[np.ndarray]:
        """Simple k-means clustering for critical point identification"""
        if len(coordinates) < n_clusters:
            return [np.mean(coordinates, axis=0)]
        
        # Initialize cluster centers randomly
        centers = coordinates[np.random.choice(len(coordinates), n_clusters, replace=False)]
        
        # Simple k-means iterations
        for _ in range(10):
            # Assign points to nearest cluster
            distances = np.array([[np.linalg.norm(point - center) for center in centers] 
                                 for point in coordinates])
            assignments = np.argmin(distances, axis=1)
            
            # Update cluster centers
            new_centers = []
            for i in range(n_clusters):
                cluster_points = coordinates[assignments == i]
                if len(cluster_points) > 0:
                    new_centers.append(np.mean(cluster_points, axis=0))
                else:
                    new_centers.append(centers[i])
            
            centers = np.array(new_centers)
        
        return centers.tolist()
    
    def _calculate_stability_eigenvalues(self, center: np.ndarray, coordinates: np.ndarray) -> List[float]:
        """Calculate stability eigenvalues around critical point"""
        # Find points near the center
        distances = [np.linalg.norm(coord - center) for coord in coordinates]
        threshold = np.percentile(distances, 30)  # Use closest 30% of points
        
        near_points = coordinates[np.array(distances) < threshold]
        
        if len(near_points) < 3:
            return [-0.1, -0.1]  # Default stable eigenvalues
        
        # Calculate covariance matrix
        centered_points = near_points - center
        cov_matrix = np.cov(centered_points.T)
        
        # Eigenvalues of covariance matrix (simplified stability analysis)
        try:
            eigenvals = np.linalg.eigvals(cov_matrix)
            # Convert to stability eigenvalues (negative for attraction)
            stability_eigenvals = [-abs(val) for val in eigenvals[:2]]  # Use first two
            return stability_eigenvals
        except:
            return [-0.1, -0.1]  # Default on numerical issues
    
    def _calculate_basin_radius(self, center: np.ndarray, coordinates: np.ndarray) -> float:
        """Calculate attraction basin radius"""
        distances = [np.linalg.norm(coord - center) for coord in coordinates]
        return float(np.percentile(distances, 75))  # 75th percentile as basin radius


class DownwardCausationEmergenceStrategy(EmergenceStrategy):
    """Emergence detection strategy focusing on downward causation"""
    
    def detect_emergence(self,
                        system: InformationIntegrationSystem,
                        phase_transition: PhaseTransition) -> List[EmergentProperty]:
        """Detect emergent properties with downward causation"""
        
        emergent_properties = []
        
        # Analyze integration patterns for emergence
        if phase_transition.to_state.emergence_potential > 0.7:
            
            # Meta-cognitive emergence
            if self._has_meta_cognitive_emergence(system):
                meta_property = EmergentProperty(
                    property_id=f"meta_cognitive_{secrets.token_hex(6)}",
                    property_name="Meta-Cognitive Integration",
                    emergence_threshold=0.6,
                    current_intensity=phase_transition.to_state.emergence_potential,
                    temporal_persistence=0.8,
                    causal_efficacy=0.7,
                    downward_causation=True,
                    timestamp=datetime.now()
                )
                emergent_properties.append(meta_property)
            
            # Temporal synthesis emergence
            if self._has_temporal_synthesis_emergence(system):
                temporal_property = EmergentProperty(
                    property_id=f"temporal_synthesis_{secrets.token_hex(6)}",
                    property_name="Temporal Synthesis Coherence",
                    emergence_threshold=0.5,
                    current_intensity=phase_transition.to_state.integration_level,
                    temporal_persistence=0.9,
                    causal_efficacy=0.6,
                    downward_causation=True,
                    timestamp=datetime.now()
                )
                emergent_properties.append(temporal_property)
            
            # Predictive modeling emergence
            if phase_transition.to_state.information_generation_rate > 0.8:
                predictive_property = EmergentProperty(
                    property_id=f"predictive_modeling_{secrets.token_hex(6)}",
                    property_name="Predictive Integration Capability",
                    emergence_threshold=0.7,
                    current_intensity=phase_transition.to_state.information_generation_rate,
                    temporal_persistence=0.7,
                    causal_efficacy=0.8,
                    downward_causation=True,
                    timestamp=datetime.now()
                )
                emergent_properties.append(predictive_property)
        
        return emergent_properties
    
    def assess_emergence_strength(self, property: EmergentProperty) -> float:
        """Assess strength of emergent property"""
        if not property.is_emergent():
            return 0.0
        
        # Combine intensity, persistence, and causal efficacy
        strength = (property.current_intensity * 0.4 +
                   property.temporal_persistence * 0.3 +
                   property.causal_efficacy * 0.3)
        
        # Bonus for downward causation
        if property.downward_causation:
            strength *= 1.2
        
        return min(1.0, strength)
    
    def _has_meta_cognitive_emergence(self, system: InformationIntegrationSystem) -> bool:
        """Check for meta-cognitive emergence"""
        meta_layers = [layer for layer in system.integration_layers 
                      if layer.layer_type == IntegrationLayerType.META_COGNITIVE and layer.is_active]
        
        return len(meta_layers) > 0 and system.integration_degree.value > 0.6
    
    def _has_temporal_synthesis_emergence(self, system: InformationIntegrationSystem) -> bool:
        """Check for temporal synthesis emergence"""
        temporal_layers = [layer for layer in system.integration_layers 
                          if layer.layer_type == IntegrationLayerType.TEMPORAL_SYNTHESIS and layer.is_active]
        
        return len(temporal_layers) > 0 and system.integration_degree.value > 0.5
    
    def _has_predictive_modeling_emergence(self, system: InformationIntegrationSystem) -> bool:
        """Check for predictive modeling emergence"""
        predictive_layers = [layer for layer in system.integration_layers 
                           if layer.layer_type == IntegrationLayerType.PREDICTIVE_MODELING and layer.is_active]
        
        return len(predictive_layers) > 0 and system.integration_degree.value > 0.7


# ============================================================================
# CORE COMPONENTS
# ============================================================================

class PhaseTransitionDetector:
    """Detects approaching and occurring phase transitions"""
    
    def __init__(self,
                 transition_strategy: PhaseTransitionStrategy = None,
                 detection_sensitivity: float = 0.1):
        self.transition_strategy = transition_strategy or KanaiInformationGenerationStrategy()
        self.detection_sensitivity = detection_sensitivity
        self.phase_history = deque(maxlen=200)
        self.transition_history = deque(maxlen=50)
        
        logger.info("Phase Transition Detector initialized")
    
    def detect_current_phase(self, system: InformationIntegrationSystem) -> PhaseState:
        """Detect current phase state of the system"""
        
        # Calculate phase parameters
        integration_level = system.integration_degree.value
        information_generation = self._calculate_information_generation_rate(system)
        emergence_potential = self._calculate_emergence_potential(system)
        stability_index = system.assess_integration_stability()
        entropy_level = self._calculate_entropy_level(system)
        
        current_phase = PhaseState(
            state_id=f"phase_{secrets.token_hex(8)}",
            integration_level=integration_level,
            information_generation_rate=information_generation,
            emergence_potential=emergence_potential,
            stability_index=stability_index,
            entropy_level=entropy_level,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.phase_history.append(current_phase)
        
        return current_phase
    
    def detect_transition(self, system: InformationIntegrationSystem) -> Optional[PhaseTransition]:
        """Detect phase transition using configured strategy"""
        
        current_phase = self.detect_current_phase(system)
        
        # Use strategy to detect transition
        transition = self.transition_strategy.detect_transition(current_phase, system)
        
        if transition:
            self.transition_history.append(transition)
            logger.info(f"Phase transition detected: {transition.transition_type.value}")
        
        return transition
    
    def _calculate_information_generation_rate(self, system: InformationIntegrationSystem) -> float:
        """Calculate information generation rate"""
        if not system.integration_layers:
            return 0.0
        
        # Sum of active layer contributions weighted by dependencies
        total_generation = 0.0
        for layer in system.integration_layers:
            if layer.is_active:
                layer_generation = layer.capacity * (1.0 + len(layer.dependencies) * 0.15)
                total_generation += layer_generation
        
        # Normalize by total layers and apply integration factor
        if system.integration_layers:
            normalized_generation = total_generation / len(system.integration_layers)
            return normalized_generation * system.integration_degree.value
        
        return 0.0
    
    def _calculate_emergence_potential(self, system: InformationIntegrationSystem) -> float:
        """Calculate emergence potential"""
        # Based on layer interactions and non-linear integration effects
        active_layers = [layer for layer in system.integration_layers if layer.is_active]
        
        if len(active_layers) < 2:
            return 0.0
        
        # Calculate interaction potential between layers
        interaction_sum = 0.0
        interaction_count = 0
        
        for i, layer1 in enumerate(active_layers):
            for layer2 in active_layers[i+1:]:
                # Interaction strength based on capacity product and dependency relations
                interaction_strength = layer1.capacity * layer2.capacity
                
                # Bonus for dependency relationships
                if layer2 in layer1.dependencies or layer1 in layer2.dependencies:
                    interaction_strength *= 1.5
                
                interaction_sum += interaction_strength
                interaction_count += 1
        
        if interaction_count == 0:
            return 0.0
        
        # Average interaction strength as emergence potential
        emergence_potential = interaction_sum / interaction_count
        
        # Amplify with integration degree (non-linear effect)
        emergence_potential *= (system.integration_degree.value ** 1.5)
        
        return min(1.0, emergence_potential)
    
    def _calculate_entropy_level(self, system: InformationIntegrationSystem) -> float:
        """Calculate entropy level (disorder/randomness)"""
        if not system.integration_layers:
            return 1.0  # Maximum entropy
        
        # Calculate based on layer capacity variance and integration stability
        layer_capacities = [layer.capacity for layer in system.integration_layers if layer.is_active]
        
        if not layer_capacities:
            return 1.0
        
        # Higher variance in capacities indicates higher entropy
        capacity_variance = np.var(layer_capacities)
        
        # Integration stability reduces entropy
        stability = system.assess_integration_stability()
        
        # Combine factors
        entropy = capacity_variance + (1.0 - stability) * 0.5
        
        return min(1.0, entropy)


class TransitionPredictor:
    """Predicts future system states and transition paths"""
    
    def __init__(self,
                 transition_strategy: PhaseTransitionStrategy = None,
                 prediction_horizon: timedelta = timedelta(minutes=60)):
        self.transition_strategy = transition_strategy or KanaiInformationGenerationStrategy()
        self.prediction_horizon = prediction_horizon
        self.prediction_cache = {}
        
        logger.info("Transition Predictor initialized")
    
    def predict_future_states(self,
                            system: InformationIntegrationSystem,
                            current_phase: PhaseState,
                            time_steps: int = 10) -> List[Tuple[datetime, PhaseState, float]]:
        """Predict future phase states"""
        
        predictions = []
        current_state = current_phase
        step_duration = self.prediction_horizon / time_steps
        
        for step in range(time_steps):
            future_time = datetime.now() + step_duration * (step + 1)
            
            # Predict next state
            next_state, confidence = self._predict_next_state(system, current_state, step_duration)
            
            predictions.append((future_time, next_state, confidence))
            current_state = next_state
        
        return predictions
    
    def calculate_transition_probabilities(self,
                                        system: InformationIntegrationSystem,
                                        current_phase: PhaseState,
                                        candidate_states: List[PhaseState]) -> Dict[str, float]:
        """Calculate transition probabilities to candidate states"""
        
        probabilities = {}
        
        for candidate_state in candidate_states:
            probability = self.transition_strategy.calculate_transition_probability(
                current_phase, candidate_state
            )
            probabilities[candidate_state.state_id] = probability
        
        return probabilities
    
    def predict_transition_timing(self,
                                system: InformationIntegrationSystem,
                                target_state: PhaseState) -> Optional[Tuple[datetime, float]]:
        """Predict when a transition to target state will occur"""
        
        current_phase = self._get_current_phase(system)
        
        # Calculate energy barrier and transition rate
        transition_probability = self.transition_strategy.calculate_transition_probability(
            current_phase, target_state
        )
        
        if transition_probability < 0.1:
            return None  # Very unlikely transition
        
        # Estimate timing based on probability and system dynamics
        base_time_constant = 300  # 5 minutes base time
        adjusted_time = base_time_constant / transition_probability
        
        # Add some randomness for realistic prediction
        variation = adjusted_time * 0.3 * np.random.random()
        estimated_time = adjusted_time + variation
        
        predicted_time = datetime.now() + timedelta(seconds=estimated_time)
        confidence = transition_probability * 0.8  # Slightly lower confidence for timing
        
        return predicted_time, confidence
    
    def _predict_next_state(self,
                          system: InformationIntegrationSystem,
                          current_state: PhaseState,
                          time_step: timedelta) -> Tuple[PhaseState, float]:
        """Predict next state after time step"""
        
        # Simple evolution model based on system dynamics
        dt = time_step.total_seconds() / 3600.0  # Convert to hours
        
        # Evolution rates (simplified model)
        integration_rate = (0.5 - current_state.integration_level) * 0.1  # Tendency toward 0.5
        generation_rate = -current_state.information_generation_rate * 0.05  # Decay
        emergence_rate = (current_state.integration_level - current_state.emergence_potential) * 0.2
        stability_rate = -abs(integration_rate) * 0.3  # Stability decreases with change
        entropy_rate = abs(integration_rate) * 0.1  # Entropy increases with change
        
        # Apply evolution
        new_integration = max(0.0, min(1.0, 
            current_state.integration_level + integration_rate * dt))
        new_generation = max(0.0, min(1.0,
            current_state.information_generation_rate + generation_rate * dt))
        new_emergence = max(0.0, min(1.0,
            current_state.emergence_potential + emergence_rate * dt))
        new_stability = max(0.0, min(1.0,
            current_state.stability_index + stability_rate * dt))
        new_entropy = max(0.0, min(1.0,
            current_state.entropy_level + entropy_rate * dt))
        
        # Create predicted state
        predicted_state = PhaseState(
            state_id=f"predicted_{secrets.token_hex(8)}",
            integration_level=new_integration,
            information_generation_rate=new_generation,
            emergence_potential=new_emergence,
            stability_index=new_stability,
            entropy_level=new_entropy,
            timestamp=datetime.now() + time_step
        )
        
        # Calculate prediction confidence
        state_change = current_state.distance_to(predicted_state)
        confidence = max(0.3, 1.0 - state_change)  # Higher confidence for smaller changes
        
        return predicted_state, confidence
    
    def _get_current_phase(self, system: InformationIntegrationSystem) -> PhaseState:
        """Get current phase state (simplified)"""
        # This would typically use a PhaseTransitionDetector
        return PhaseState(
            state_id=f"current_{secrets.token_hex(8)}",
            integration_level=system.integration_degree.value,
            information_generation_rate=0.5,  # Placeholder
            emergence_potential=0.3,  # Placeholder
            stability_index=system.assess_integration_stability(),
            entropy_level=0.4,  # Placeholder
            timestamp=datetime.now()
        )


class EmergentPropertyAnalyzer:
    """Analyzes emergent behaviors during transitions"""
    
    def __init__(self,
                 emergence_strategy: EmergenceStrategy = None,
                 emergence_threshold: float = 0.5):
        self.emergence_strategy = emergence_strategy or DownwardCausationEmergenceStrategy()
        self.emergence_threshold = emergence_threshold
        self.property_history = deque(maxlen=100)
        
        logger.info("Emergent Property Analyzer initialized")
    
    def analyze_emergence_during_transition(self,
                                          system: InformationIntegrationSystem,
                                          transition: PhaseTransition) -> List[EmergentProperty]:
        """Analyze emergent properties during phase transition"""
        
        emergent_properties = self.emergence_strategy.detect_emergence(system, transition)
        
        # Filter by emergence threshold
        significant_properties = [
            prop for prop in emergent_properties
            if prop.is_emergent() and self.emergence_strategy.assess_emergence_strength(prop) >= self.emergence_threshold
        ]
        
        # Store in history
        for prop in significant_properties:
            self.property_history.append(prop)
        
        if significant_properties:
            logger.info(f"Detected {len(significant_properties)} emergent properties during transition")
        
        return significant_properties
    
    def monitor_emergence_potential(self,
                                  system: InformationIntegrationSystem,
                                  current_phase: PhaseState) -> Dict[str, float]:
        """Monitor potential for different types of emergence"""
        
        potentials = {
            'meta_cognitive': self._assess_meta_cognitive_potential(system, current_phase),
            'temporal_synthesis': self._assess_temporal_synthesis_potential(system, current_phase),
            'predictive_modeling': self._assess_predictive_modeling_potential(system, current_phase),
            'global_workspace': self._assess_global_workspace_potential(system, current_phase),
            'self_awareness': self._assess_self_awareness_potential(system, current_phase)
        }
        
        return potentials
    
    def assess_downward_causation(self,
                                property: EmergentProperty,
                                system: InformationIntegrationSystem) -> float:
        """Assess strength of downward causation"""
        
        if not property.downward_causation:
            return 0.0
        
        # Measure influence of emergent property on lower levels
        causal_strength = property.causal_efficacy
        
        # Enhance based on system integration
        integration_enhancement = system.integration_degree.value * 0.3
        
        # Temporal persistence adds to causation strength
        persistence_factor = property.temporal_persistence * 0.2
        
        total_causation = causal_strength + integration_enhancement + persistence_factor
        
        return min(1.0, total_causation)
    
    def _assess_meta_cognitive_potential(self,
                                       system: InformationIntegrationSystem,
                                       phase: PhaseState) -> float:
        """Assess potential for meta-cognitive emergence"""
        
        meta_layers = [layer for layer in system.integration_layers
                      if layer.layer_type == IntegrationLayerType.META_COGNITIVE]
        
        if not meta_layers:
            return 0.0
        
        # Based on integration level and emergence potential
        potential = (phase.integration_level * 0.4 + 
                    phase.emergence_potential * 0.4 +
                    phase.information_generation_rate * 0.2)
        
        # Bonus for active meta-cognitive layers
        if any(layer.is_active for layer in meta_layers):
            potential *= 1.3
        
        return min(1.0, potential)
    
    def _assess_temporal_synthesis_potential(self,
                                          system: InformationIntegrationSystem,
                                          phase: PhaseState) -> float:
        """Assess potential for temporal synthesis emergence"""
        
        temporal_layers = [layer for layer in system.integration_layers
                          if layer.layer_type == IntegrationLayerType.TEMPORAL_SYNTHESIS]
        
        if not temporal_layers:
            return 0.0
        
        # Based on stability and integration
        potential = (phase.stability_index * 0.5 +
                    phase.integration_level * 0.3 +
                    (1.0 - phase.entropy_level) * 0.2)
        
        return min(1.0, potential)
    
    def _assess_predictive_modeling_potential(self,
                                           system: InformationIntegrationSystem,
                                           phase: PhaseState) -> float:
        """Assess potential for predictive modeling emergence"""
        
        predictive_layers = [layer for layer in system.integration_layers
                           if layer.layer_type == IntegrationLayerType.PREDICTIVE_MODELING]
        
        if not predictive_layers:
            return 0.0
        
        # Based on information generation and emergence potential
        potential = (phase.information_generation_rate * 0.6 +
                    phase.emergence_potential * 0.4)
        
        return min(1.0, potential)
    
    def _assess_global_workspace_potential(self,
                                        system: InformationIntegrationSystem,
                                        phase: PhaseState) -> float:
        """Assess potential for global workspace emergence"""
        
        # Requires multiple active layers with good integration
        active_layers = [layer for layer in system.integration_layers if layer.is_active]
        
        if len(active_layers) < 3:
            return 0.0
        
        # Based on integration level and inter-layer connectivity
        connectivity_factor = sum(len(layer.dependencies) for layer in active_layers) / len(active_layers)
        
        potential = (phase.integration_level * 0.5 +
                    min(1.0, connectivity_factor * 0.2) * 0.3 +
                    phase.emergence_potential * 0.2)
        
        return min(1.0, potential)
    
    def _assess_self_awareness_potential(self,
                                       system: InformationIntegrationSystem,
                                       phase: PhaseState) -> float:
        """Assess potential for self-awareness emergence"""
        
        # Requires meta-cognitive layer and high integration
        meta_layers = [layer for layer in system.integration_layers
                      if layer.layer_type == IntegrationLayerType.META_COGNITIVE and layer.is_active]
        
        if not meta_layers:
            return 0.0
        
        # High requirements for self-awareness
        potential = (phase.integration_level ** 2 * 0.4 +  # Non-linear requirement
                    phase.emergence_potential * 0.3 +
                    phase.information_generation_rate * 0.3)
        
        # Threshold effect - needs minimum integration
        if phase.integration_level < 0.6:
            potential *= 0.3
        
        return min(1.0, potential)


class CriticalPointCalculator:
    """Identifies critical thresholds and phase boundaries"""
    
    def __init__(self,
                 critical_point_strategy: CriticalPointStrategy = None,
                 analysis_window: int = 50):
        self.critical_point_strategy = critical_point_strategy or AttractorBasedCriticalPointStrategy()
        self.analysis_window = analysis_window
        self.critical_points_cache = {}
        
        logger.info("Critical Point Calculator initialized")
    
    def identify_critical_points(self,
                               system: InformationIntegrationSystem,
                               phase_history: List[PhaseState]) -> List[CriticalPoint]:
        """Identify critical points in phase space"""
        
        if len(phase_history) < 10:
            logger.warning("Insufficient phase history for critical point analysis")
            return []
        
        # Use recent history window
        recent_history = phase_history[-self.analysis_window:] if len(phase_history) > self.analysis_window else phase_history
        
        # Use strategy to identify critical points
        critical_points = self.critical_point_strategy.identify_critical_points(
            system, recent_history
        )
        
        # Cache results
        cache_key = f"system_{system.id.value}_{datetime.now().strftime('%Y%m%d%H')}"
        self.critical_points_cache[cache_key] = critical_points
        
        logger.info(f"Identified {len(critical_points)} critical points")
        
        return critical_points
    
    def calculate_threshold_proximity(self,
                                    current_phase: PhaseState,
                                    critical_points: List[CriticalPoint]) -> Dict[str, Tuple[float, CriticalPoint]]:
        """Calculate proximity to critical thresholds"""
        
        proximities = {}
        
        for critical_point in critical_points:
            # Calculate distance in phase space
            phase_coords = np.array([
                current_phase.integration_level,
                current_phase.information_generation_rate,
                current_phase.emergence_potential,
                current_phase.stability_index,
                current_phase.entropy_level
            ])
            
            critical_coords = np.array(critical_point.phase_coordinates[:len(phase_coords)])
            distance = np.linalg.norm(phase_coords - critical_coords)
            
            # Convert distance to proximity (closer = higher proximity)
            proximity = max(0.0, 1.0 - distance / critical_point.basin_radius)
            
            proximities[critical_point.point_id] = (proximity, critical_point)
        
        return proximities
    
    def predict_critical_point_approach(self,
                                      system: InformationIntegrationSystem,
                                      current_phase: PhaseState,
                                      phase_velocity: np.ndarray,
                                      critical_points: List[CriticalPoint]) -> List[Tuple[CriticalPoint, timedelta, float]]:
        """Predict approach to critical points"""
        
        approaches = []
        
        for critical_point in critical_points:
            # Calculate trajectory toward critical point
            current_coords = np.array([
                current_phase.integration_level,
                current_phase.information_generation_rate,
                current_phase.emergence_potential,
                current_phase.stability_index,
                current_phase.entropy_level
            ])
            
            critical_coords = np.array(critical_point.phase_coordinates[:len(current_coords)])
            direction_vector = critical_coords - current_coords
            distance = np.linalg.norm(direction_vector)
            
            if distance < 0.01:  # Already at critical point
                continue
            
            # Project velocity onto direction to critical point
            if np.linalg.norm(phase_velocity) > 0:
                direction_unit = direction_vector / distance
                velocity_component = np.dot(phase_velocity, direction_unit)
                
                if velocity_component > 0:  # Moving toward critical point
                    # Estimate arrival time
                    time_to_arrival = distance / velocity_component
                    arrival_time = timedelta(seconds=time_to_arrival * 3600)  # Convert from hours
                    
                    # Calculate confidence based on trajectory alignment
                    velocity_alignment = abs(velocity_component) / np.linalg.norm(phase_velocity)
                    confidence = velocity_alignment * min(1.0, critical_point.attraction_strength)
                    
                    approaches.append((critical_point, arrival_time, confidence))
        
        # Sort by arrival time
        approaches.sort(key=lambda x: x[1])
        
        return approaches
    
    def analyze_basin_stability(self,
                              critical_point: CriticalPoint,
                              system: InformationIntegrationSystem) -> Dict[str, Any]:
        """Analyze stability of critical point basin"""
        
        stability_analysis = self.critical_point_strategy.analyze_stability(critical_point)
        
        # Add system-specific analysis
        if critical_point.is_attractor():
            # For attractors, assess how system state affects stability
            integration_factor = system.integration_degree.value
            stability_analysis['integration_stability'] = integration_factor
            stability_analysis['overall_stability'] = (
                stability_analysis['stability_measure'] * integration_factor
            )
        
        return stability_analysis


# ============================================================================
# MAIN PHASE TRANSITION ENGINE (Aggregate Root)
# ============================================================================

class PhaseTransitionEngine:
    """
    Main aggregate root for phase transition detection and prediction
    Integrates all phase transition components following Clean Architecture
    """
    
    def __init__(self,
                 system_id: SystemIdentity,
                 detector: PhaseTransitionDetector = None,
                 predictor: TransitionPredictor = None,
                 analyzer: EmergentPropertyAnalyzer = None,
                 calculator: CriticalPointCalculator = None):
        
        # Core identity
        self.id = system_id
        self.created_at = datetime.now()
        
        # Core components
        self.detector = detector or PhaseTransitionDetector()
        self.predictor = predictor or TransitionPredictor()
        self.analyzer = analyzer or EmergentPropertyAnalyzer()
        self.calculator = calculator or CriticalPointCalculator()
        
        # State tracking
        self.current_phase: Optional[PhaseState] = None
        self.active_transitions: List[PhaseTransition] = []
        self.critical_points: List[CriticalPoint] = []
        self.emergent_properties: List[EmergentProperty] = []
        
        # Event tracking
        self.domain_events: List[DomainEvent] = []
        
        # Analysis history
        self.phase_history = deque(maxlen=500)
        self.transition_history = deque(maxlen=100)
        self.emergence_history = deque(maxlen=200)
        
        # Performance tracking
        self.last_analysis_time: Optional[datetime] = None
        self.analysis_duration_history = deque(maxlen=20)
        
        logger.info(f"Phase Transition Engine {system_id.value} initialized")
    
    async def analyze_system_phase(self,
                                 system: InformationIntegrationSystem) -> Dict[str, Any]:
        """Comprehensive phase analysis of the system"""
        
        analysis_start = time.time()
        
        try:
            # Detect current phase
            self.current_phase = self.detector.detect_current_phase(system)
            self.phase_history.append(self.current_phase)
            
            # Detect transitions
            transition = self.detector.detect_transition(system)
            if transition:
                self.active_transitions.append(transition)
                self.transition_history.append(transition)
                
                # Generate transition event
                self.domain_events.append(
                    PhaseTransitionDetectedEvent(
                        system_id=self.id,
                        phase_transition=transition,
                        detection_confidence=0.8,  # Default confidence
                        affected_layers=self._determine_affected_layers(system, transition)
                    )
                )
                
                # Analyze emergent properties during transition
                emergent_properties = self.analyzer.analyze_emergence_during_transition(
                    system, transition
                )
                
                self.emergent_properties.extend(emergent_properties)
                self.emergence_history.extend(emergent_properties)
                
                # Generate emergence events
                for prop in emergent_properties:
                    self.domain_events.append(
                        EmergentPropertyEvent(
                            system_id=self.id,
                            emergent_property=prop,
                            emergence_context={
                                'transition_type': transition.transition_type.value,
                                'phase_change_magnitude': transition.magnitude(),
                                'system_integration_level': system.integration_degree.value
                            }
                        )
                    )
            
            # Update critical points
            if len(self.phase_history) >= 10:
                self.critical_points = self.calculator.identify_critical_points(
                    system, list(self.phase_history)
                )
                
                # Check proximity to critical points
                if self.critical_points:
                    proximities = self.calculator.calculate_threshold_proximity(
                        self.current_phase, self.critical_points
                    )
                    
                    # Generate events for approaching critical points
                    for point_id, (proximity, critical_point) in proximities.items():
                        if proximity > 0.8:  # Very close to critical point
                            self.domain_events.append(
                                CriticalPointApproachedEvent(
                                    system_id=self.id,
                                    critical_point=critical_point,
                                    approach_velocity=0.0,  # Simplified
                                    estimated_arrival=timedelta(minutes=10)  # Estimated
                                )
                            )
            
            # Predict future states
            future_predictions = self.predictor.predict_future_states(
                system, self.current_phase, time_steps=5
            )
            
            # Monitor emergence potential
            emergence_potentials = self.analyzer.monitor_emergence_potential(
                system, self.current_phase
            )
            
            # Performance tracking
            analysis_duration = time.time() - analysis_start
            self.analysis_duration_history.append(analysis_duration)
            self.last_analysis_time = datetime.now()
            
            # Compile comprehensive analysis
            analysis_result = {
                'system_id': self.id.value,
                'timestamp': self.last_analysis_time.isoformat(),
                'current_phase': {
                    'state_id': self.current_phase.state_id,
                    'integration_level': self.current_phase.integration_level,
                    'information_generation_rate': self.current_phase.information_generation_rate,
                    'emergence_potential': self.current_phase.emergence_potential,
                    'stability_index': self.current_phase.stability_index,
                    'entropy_level': self.current_phase.entropy_level,
                    'is_critical': self.current_phase.is_critical_phase()
                },
                'active_transitions': [
                    {
                        'transition_type': t.transition_type.value,
                        'probability': t.transition_probability,
                        'magnitude': t.magnitude(),
                        'reversible': t.reversibility,
                        'criticality': t.criticality_level()
                    }
                    for t in self.active_transitions
                ],
                'critical_points': [
                    {
                        'point_id': cp.point_id,
                        'type': cp.criticality_type.value,
                        'basin_radius': cp.basin_radius,
                        'attraction_strength': cp.attraction_strength,
                        'is_attractor': cp.is_attractor()
                    }
                    for cp in self.critical_points
                ],
                'emergent_properties': [
                    {
                        'property_id': ep.property_id,
                        'name': ep.property_name,
                        'intensity': ep.current_intensity,
                        'strength': self.analyzer.emergence_strategy.assess_emergence_strength(ep),
                        'downward_causation': ep.downward_causation
                    }
                    for ep in self.emergent_properties
                ],
                'future_predictions': [
                    {
                        'timestamp': pred_time.isoformat(),
                        'predicted_state': {
                            'integration_level': pred_state.integration_level,
                            'generation_rate': pred_state.information_generation_rate,
                            'emergence_potential': pred_state.emergence_potential,
                            'stability_index': pred_state.stability_index
                        },
                        'confidence': confidence
                    }
                    for pred_time, pred_state, confidence in future_predictions
                ],
                'emergence_potentials': emergence_potentials,
                'analysis_performance': {
                    'duration_seconds': analysis_duration,
                    'average_duration': np.mean(self.analysis_duration_history) if self.analysis_duration_history else 0.0,
                    'total_analyses': len(self.analysis_duration_history)
                }
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error during phase analysis: {e}")
            raise PhaseTransitionError(f"Phase analysis failed: {str(e)}")
    
    def get_transition_predictions(self,
                                 system: InformationIntegrationSystem,
                                 time_horizon: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get comprehensive transition predictions"""
        
        if not self.current_phase:
            self.current_phase = self.detector.detect_current_phase(system)
        
        # Predict future states
        predictions = self.predictor.predict_future_states(
            system, self.current_phase, time_steps=10
        )
        
        # Analyze critical point approaches
        if self.critical_points and len(self.phase_history) >= 2:
            # Estimate phase velocity from recent history
            recent_states = list(self.phase_history)[-2:]
            if len(recent_states) == 2:
                dt = (recent_states[1].timestamp - recent_states[0].timestamp).total_seconds() / 3600.0
                
                velocity = np.array([
                    (recent_states[1].integration_level - recent_states[0].integration_level) / dt,
                    (recent_states[1].information_generation_rate - recent_states[0].information_generation_rate) / dt,
                    (recent_states[1].emergence_potential - recent_states[0].emergence_potential) / dt,
                    (recent_states[1].stability_index - recent_states[0].stability_index) / dt,
                    (recent_states[1].entropy_level - recent_states[0].entropy_level) / dt
                ])
                
                critical_approaches = self.calculator.predict_critical_point_approach(
                    system, self.current_phase, velocity, self.critical_points
                )
            else:
                critical_approaches = []
        else:
            critical_approaches = []
        
        return {
            'time_horizon': time_horizon.total_seconds(),
            'current_phase_id': self.current_phase.state_id,
            'predicted_states': [
                {
                    'time': pred_time.isoformat(),
                    'state': {
                        'integration_level': pred_state.integration_level,
                        'generation_rate': pred_state.information_generation_rate,
                        'emergence_potential': pred_state.emergence_potential,
                        'stability_index': pred_state.stability_index,
                        'entropy_level': pred_state.entropy_level
                    },
                    'confidence': confidence
                }
                for pred_time, pred_state, confidence in predictions
            ],
            'critical_point_approaches': [
                {
                    'critical_point_id': cp.point_id,
                    'critical_point_type': cp.criticality_type.value,
                    'estimated_arrival': arrival_time.total_seconds(),
                    'approach_confidence': confidence
                }
                for cp, arrival_time, confidence in critical_approaches
            ],
            'transition_risks': self._assess_transition_risks(system),
            'emergence_forecasts': self._forecast_emergence(system, time_horizon)
        }
    
    def _determine_affected_layers(self,
                                 system: InformationIntegrationSystem,
                                 transition: PhaseTransition) -> List[IntegrationLayerType]:
        """Determine which integration layers are affected by transition"""
        
        affected_layers = []
        
        # High magnitude transitions affect more layers
        if transition.magnitude() > 0.5:
            # Major transitions affect all layer types
            affected_layers = list(IntegrationLayerType)
        elif transition.magnitude() > 0.3:
            # Medium transitions affect cognitive and temporal layers
            affected_layers = [
                IntegrationLayerType.META_COGNITIVE,
                IntegrationLayerType.TEMPORAL_SYNTHESIS,
                IntegrationLayerType.PREDICTIVE_MODELING
            ]
        else:
            # Minor transitions might only affect specific layers
            if transition.transition_type == PhaseTransitionType.CONTINUOUS:
                affected_layers = [IntegrationLayerType.SENSORY_INTEGRATION]
            else:
                affected_layers = [IntegrationLayerType.META_COGNITIVE]
        
        # Filter by actually existing layers
        existing_layer_types = {layer.layer_type for layer in system.integration_layers}
        affected_layers = [layer_type for layer_type in affected_layers 
                          if layer_type in existing_layer_types]
        
        return affected_layers
    
    def _assess_transition_risks(self, system: InformationIntegrationSystem) -> Dict[str, float]:
        """Assess risks associated with potential transitions"""
        
        risks = {
            'integration_collapse_risk': 0.0,
            'emergence_failure_risk': 0.0,
            'stability_loss_risk': 0.0,
            'irreversible_transition_risk': 0.0
        }
        
        if self.current_phase:
            # Integration collapse risk
            risks['integration_collapse_risk'] = max(0.0, 1.0 - self.current_phase.integration_level)
            
            # Emergence failure risk (high emergence potential but low stability)
            if self.current_phase.emergence_potential > 0.7 and self.current_phase.stability_index < 0.3:
                risks['emergence_failure_risk'] = 0.8
            
            # Stability loss risk
            risks['stability_loss_risk'] = 1.0 - self.current_phase.stability_index
            
            # Irreversible transition risk (high entropy, low stability)
            if self.current_phase.entropy_level > 0.7 and self.current_phase.stability_index < 0.2:
                risks['irreversible_transition_risk'] = 0.9
        
        return risks
    
    def _forecast_emergence(self,
                          system: InformationIntegrationSystem,
                          time_horizon: timedelta) -> Dict[str, Any]:
        """Forecast emergent property manifestations"""
        
        emergence_potentials = self.analyzer.monitor_emergence_potential(
            system, self.current_phase
        )
        
        # Predict which properties might emerge
        likely_emergences = []
        for property_type, potential in emergence_potentials.items():
            if potential > 0.6:
                # Estimate emergence time based on potential
                emergence_time = time_horizon.total_seconds() * (1.0 - potential)
                likely_emergences.append({
                    'property_type': property_type,
                    'emergence_potential': potential,
                    'estimated_emergence_time': emergence_time,
                    'confidence': potential * 0.8
                })
        
        return {
            'emergence_potentials': emergence_potentials,
            'likely_emergences': likely_emergences,
            'total_emergence_potential': sum(emergence_potentials.values()) / len(emergence_potentials),
            'emergence_timeline': time_horizon.total_seconds()
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        
        return {
            'engine_id': self.id.value,
            'uptime': (datetime.now() - self.created_at).total_seconds(),
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'current_phase': {
                'state_id': self.current_phase.state_id,
                'is_critical': self.current_phase.is_critical_phase()
            } if self.current_phase else None,
            'active_transitions_count': len(self.active_transitions),
            'critical_points_count': len(self.critical_points),
            'emergent_properties_count': len(self.emergent_properties),
            'phase_history_length': len(self.phase_history),
            'transition_history_length': len(self.transition_history),
            'emergence_history_length': len(self.emergence_history),
            'pending_events_count': len(self.domain_events),
            'average_analysis_duration': (
                np.mean(self.analysis_duration_history) 
                if self.analysis_duration_history else 0.0
            ),
            'component_status': {
                'detector': 'ACTIVE',
                'predictor': 'ACTIVE', 
                'analyzer': 'ACTIVE',
                'calculator': 'ACTIVE'
            }
        }


# ============================================================================
# FACTORY AND UTILITIES
# ============================================================================

class PhaseTransitionEngineFactory:
    """Factory for creating phase transition engines"""
    
    @staticmethod
    def create_standard_engine(system_id: SystemIdentity) -> PhaseTransitionEngine:
        """Create engine with standard configuration"""
        
        # Create components with standard strategies
        detector = PhaseTransitionDetector(
            transition_strategy=KanaiInformationGenerationStrategy(),
            detection_sensitivity=0.1
        )
        
        predictor = TransitionPredictor(
            transition_strategy=KanaiInformationGenerationStrategy(),
            prediction_horizon=timedelta(hours=1)
        )
        
        analyzer = EmergentPropertyAnalyzer(
            emergence_strategy=DownwardCausationEmergenceStrategy(),
            emergence_threshold=0.5
        )
        
        calculator = CriticalPointCalculator(
            critical_point_strategy=AttractorBasedCriticalPointStrategy(),
            analysis_window=50
        )
        
        return PhaseTransitionEngine(
            system_id=system_id,
            detector=detector,
            predictor=predictor,
            analyzer=analyzer,
            calculator=calculator
        )
    
    @staticmethod
    def create_research_engine(system_id: SystemIdentity) -> PhaseTransitionEngine:
        """Create engine optimized for research"""
        
        # High sensitivity detector for research
        detector = PhaseTransitionDetector(
            transition_strategy=KanaiInformationGenerationStrategy(sensitivity_threshold=0.05),
            detection_sensitivity=0.05
        )
        
        # Extended prediction horizon
        predictor = TransitionPredictor(
            prediction_horizon=timedelta(hours=6)
        )
        
        # Lower emergence threshold for research
        analyzer = EmergentPropertyAnalyzer(
            emergence_threshold=0.3
        )
        
        # Larger analysis window
        calculator = CriticalPointCalculator(
            analysis_window=200
        )
        
        return PhaseTransitionEngine(
            system_id=system_id,
            detector=detector,
            predictor=predictor,
            analyzer=analyzer,
            calculator=calculator
        )


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# Maintain compatibility with existing naming conventions
PhaseTransitionDetector = PhaseTransitionDetector
TransitionPredictor = TransitionPredictor
EmergentPropertyAnalyzer = EmergentPropertyAnalyzer
CriticalPointCalculator = CriticalPointCalculator
PhaseTransitionEngine = PhaseTransitionEngine

# Legacy aliases for integration with existing systems
相転移予測エンジン = PhaseTransitionEngine
相転移検出器 = PhaseTransitionDetector
創発特性解析器 = EmergentPropertyAnalyzer
臨界点計算器 = CriticalPointCalculator


if __name__ == "__main__":
    # Demonstration
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a phase transition engine
    engine_id = SystemIdentity("phase-engine-demo-001")
    engine = PhaseTransitionEngineFactory.create_standard_engine(engine_id)
    
    print(f"Created Phase Transition Engine: {engine.get_engine_status()}")
    
    # This would typically be integrated with an InformationIntegrationSystem
    # For demonstration purposes only
    logger.info("Phase Transition Engine demonstration completed")