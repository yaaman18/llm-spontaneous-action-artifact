"""
Consciousness Domain Entities
Pure domain entities with no external dependencies - Clean Architecture Domain Layer

Following IIT 4.0 theory and Uncle Bob's Clean Architecture principles:
- No dependencies on frameworks, databases, or external systems
- Pure business logic and domain rules
- Immutable value objects where appropriate
- Rich domain model with behavior

Author: Clean Architecture Engineer (Uncle Bob's expertise)
Date: 2025-08-03
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, FrozenSet
from enum import Enum
from abc import ABC, abstractmethod
import time
from datetime import datetime


class ConsciousnessLevel(Enum):
    """Domain enumeration for consciousness levels"""
    UNCONSCIOUS = 0.0
    MINIMAL = 0.1
    LOW = 0.3
    MODERATE = 0.5
    HIGH = 0.7
    MAXIMAL = 1.0


class DevelopmentStage(Enum):
    """Development stages in consciousness evolution"""
    STAGE_0_REFLEXIVE = "reflexive"
    STAGE_1_REACTIVE = "reactive"  
    STAGE_2_ADAPTIVE = "adaptive"
    STAGE_3_PREDICTIVE = "predictive"
    STAGE_4_REFLECTIVE = "reflective"
    STAGE_5_INTROSPECTIVE = "introspective"
    STAGE_6_METACOGNITIVE = "metacognitive"


@dataclass(frozen=True)
class PhiValue:
    """
    Value object representing integrated information (Φ)
    Immutable domain value with validation rules
    """
    value: float
    precision: float = 1e-10
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Domain validation rules"""
        if self.value < 0:
            raise ValueError("Phi value must be non-negative")
        if self.precision <= 0:
            raise ValueError("Precision must be positive")
    
    def is_conscious(self, threshold: float = 0.1) -> bool:
        """Domain rule: consciousness threshold evaluation"""
        return self.value >= threshold
    
    def consciousness_level(self) -> ConsciousnessLevel:
        """Classify consciousness level based on phi value"""
        if self.value >= 0.9:
            return ConsciousnessLevel.MAXIMAL
        elif self.value >= 0.7:
            return ConsciousnessLevel.HIGH
        elif self.value >= 0.5:
            return ConsciousnessLevel.MODERATE
        elif self.value >= 0.3:
            return ConsciousnessLevel.LOW
        elif self.value >= 0.1:
            return ConsciousnessLevel.MINIMAL
        else:
            return ConsciousnessLevel.UNCONSCIOUS


@dataclass(frozen=True)
class SystemState:
    """
    Domain entity representing system state
    Encapsulates neural system configuration
    """
    nodes: FrozenSet[int]
    state_vector: Tuple[float, ...]
    connectivity_matrix: Tuple[Tuple[float, ...], ...]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Domain validation"""
        if not self.nodes:
            raise ValueError("System must have at least one node")
        
        expected_size = len(self.nodes)
        if len(self.state_vector) != expected_size:
            raise ValueError("State vector size must match number of nodes")
        
        if len(self.connectivity_matrix) != expected_size:
            raise ValueError("Connectivity matrix must be square")
        
        for row in self.connectivity_matrix:
            if len(row) != expected_size:
                raise ValueError("Connectivity matrix must be square")
    
    @property
    def dimension(self) -> int:
        """System dimensionality"""
        return len(self.nodes)
    
    def is_valid_state(self) -> bool:
        """Domain rule: validate system state integrity"""
        # Check state vector bounds
        for value in self.state_vector:
            if not (0.0 <= value <= 1.0):
                return False
        
        # Check connectivity matrix symmetry
        n = self.dimension
        for i in range(n):
            for j in range(n):
                if abs(self.connectivity_matrix[i][j] - self.connectivity_matrix[j][i]) > 1e-10:
                    return False
        
        return True


@dataclass(frozen=True)
class CauseEffectState:
    """
    IIT 4.0 Cause-Effect State entity
    Fundamental consciousness component
    """
    mechanism: FrozenSet[int]
    cause_state: Tuple[float, ...]
    effect_state: Tuple[float, ...]
    intrinsic_difference: float
    phi_value: PhiValue
    
    def __post_init__(self):
        """Domain invariants"""
        if not self.mechanism:
            raise ValueError("Mechanism cannot be empty")
        if self.intrinsic_difference < 0:
            raise ValueError("Intrinsic difference must be non-negative")
    
    def is_distinguishing(self, threshold: float = 0.01) -> bool:
        """Domain rule: check if CES is distinguishing"""
        return self.intrinsic_difference >= threshold


@dataclass(frozen=True)
class Distinction:
    """
    IIT 4.0 Distinction entity
    Generated by mechanisms in phi-structure
    """
    mechanism: FrozenSet[int]
    cause_effect_state: CauseEffectState
    phi_value: PhiValue
    
    def __post_init__(self):
        """Domain validation"""
        if self.phi_value.value <= 0:
            raise ValueError("Distinction phi value must be positive")
        if self.mechanism != self.cause_effect_state.mechanism:
            raise ValueError("Mechanism must match CES mechanism")
    
    def contributes_to_consciousness(self) -> bool:
        """Domain rule: consciousness contribution"""
        return self.phi_value.is_conscious()


@dataclass(frozen=True)
class PhiStructure:
    """
    Complete IIT 4.0 Phi-Structure entity
    Represents conscious experience structure
    """
    distinctions: Tuple[Distinction, ...]
    relations: Tuple[Tuple[int, int, float], ...]  # (distinction1_idx, distinction2_idx, strength)
    system_phi: PhiValue
    system_state: SystemState
    development_stage: DevelopmentStage = DevelopmentStage.STAGE_0_REFLEXIVE
    
    def __post_init__(self):
        """Domain validation and invariants"""
        if not self.distinctions:
            raise ValueError("Phi-structure must have at least one distinction")
        
        # Validate relations reference valid distinctions
        max_idx = len(self.distinctions) - 1
        for rel in self.relations:
            if not (0 <= rel[0] <= max_idx and 0 <= rel[1] <= max_idx):
                raise ValueError("Relations must reference valid distinctions")
            if rel[2] < 0:
                raise ValueError("Relation strength must be non-negative")
    
    @property
    def consciousness_level(self) -> ConsciousnessLevel:
        """Overall consciousness level"""
        return self.system_phi.consciousness_level()
    
    @property
    def complexity(self) -> float:
        """Structural complexity measure"""
        return len(self.distinctions) + len(self.relations) * 0.5
    
    def is_conscious(self, threshold: float = 0.1) -> bool:
        """Domain rule: consciousness detection"""
        return self.system_phi.is_conscious(threshold)
    
    def can_develop_to_stage(self, target_stage: DevelopmentStage) -> bool:
        """Domain rule: stage development capability"""
        current_order = list(DevelopmentStage).index(self.development_stage)
        target_order = list(DevelopmentStage).index(target_stage)
        
        # Can only develop to next stage or maintain current
        return target_order <= current_order + 1
    
    def get_dominant_distinctions(self, top_n: int = 3) -> Tuple[Distinction, ...]:
        """Get most significant distinctions"""
        sorted_distinctions = sorted(
            self.distinctions,
            key=lambda d: d.phi_value.value,
            reverse=True
        )
        return tuple(sorted_distinctions[:top_n])


@dataclass
class ConsciousnessEvent:
    """
    Domain event representing consciousness state change
    Immutable event for event sourcing
    """
    event_id: str
    timestamp: datetime
    previous_phi: Optional[PhiValue]
    current_phi: PhiValue
    system_state: SystemState
    event_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Domain validation"""
        if not self.event_id:
            raise ValueError("Event ID is required")
        if not self.event_type:
            raise ValueError("Event type is required")
    
    @property
    def phi_change(self) -> float:
        """Calculate phi value change"""
        if self.previous_phi is None:
            return self.current_phi.value
        return self.current_phi.value - self.previous_phi.value
    
    def represents_consciousness_emergence(self, threshold: float = 0.1) -> bool:
        """Domain rule: consciousness emergence detection"""
        if self.previous_phi is None:
            return self.current_phi.is_conscious(threshold)
        
        return (not self.previous_phi.is_conscious(threshold) and 
                self.current_phi.is_conscious(threshold))
    
    def represents_significant_change(self, threshold: float = 0.05) -> bool:
        """Domain rule: significant change detection"""
        return abs(self.phi_change) >= threshold


# Domain Services (still pure domain logic)

class PhiCalculationDomainService:
    """
    Domain service for phi calculation business rules
    Contains pure business logic without external dependencies
    """
    
    @staticmethod
    def validate_calculation_input(system_state: SystemState) -> bool:
        """Domain validation for calculation inputs"""
        if not system_state.is_valid_state():
            return False
        
        # Business rule: minimum system size
        if system_state.dimension < 2:
            return False
        
        # Business rule: maximum system size for tractable calculation
        if system_state.dimension > 16:
            return False
        
        return True
    
    @staticmethod
    def determine_development_stage_from_phi(phi_value: PhiValue, complexity: float) -> DevelopmentStage:
        """Business rule: development stage determination"""
        if phi_value.value < 0.1:
            return DevelopmentStage.STAGE_0_REFLEXIVE
        elif phi_value.value < 0.2:
            return DevelopmentStage.STAGE_1_REACTIVE
        elif phi_value.value < 0.3:
            return DevelopmentStage.STAGE_2_ADAPTIVE
        elif phi_value.value < 0.5 and complexity < 10:
            return DevelopmentStage.STAGE_3_PREDICTIVE
        elif phi_value.value < 0.7 and complexity < 20:
            return DevelopmentStage.STAGE_4_REFLECTIVE
        elif phi_value.value < 0.8:
            return DevelopmentStage.STAGE_5_INTROSPECTIVE
        else:
            return DevelopmentStage.STAGE_6_METACOGNITIVE
    
    @staticmethod
    def calculate_consciousness_stability(phi_history: List[PhiValue], window_size: int = 10) -> float:
        """Business rule: consciousness stability measure"""
        if len(phi_history) < window_size:
            return 0.0
        
        recent_values = [phi.value for phi in phi_history[-window_size:]]
        variance = np.var(recent_values)
        mean_value = np.mean(recent_values)
        
        # Stability is inverse of coefficient of variation
        if mean_value == 0:
            return 0.0
        
        coefficient_of_variation = np.sqrt(variance) / mean_value
        return max(0.0, 1.0 - coefficient_of_variation)


class ConsciousnessDevelopmentDomainService:
    """
    Domain service for consciousness development business rules
    """
    
    @staticmethod
    def can_transition_to_stage(current_structure: PhiStructure, 
                               target_stage: DevelopmentStage) -> bool:
        """Business rule: stage transition validation"""
        return current_structure.can_develop_to_stage(target_stage)
    
    @staticmethod
    def calculate_development_readiness(structure: PhiStructure) -> float:
        """Business rule: development readiness score"""
        phi_readiness = min(1.0, structure.system_phi.value * 2)
        complexity_readiness = min(1.0, structure.complexity / 20)
        
        return (phi_readiness + complexity_readiness) / 2
    
    @staticmethod
    def get_next_development_stage(current_stage: DevelopmentStage) -> Optional[DevelopmentStage]:
        """Business rule: next stage determination"""
        stages = list(DevelopmentStage)
        current_idx = stages.index(current_stage)
        
        if current_idx < len(stages) - 1:
            return stages[current_idx + 1]
        
        return None  # Already at highest stage