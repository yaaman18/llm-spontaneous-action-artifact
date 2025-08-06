"""
SRP-Compliant IIT 4.0 Core Engine
Refactored to follow Single Responsibility Principle

This module replaces the monolithic IIT4PhiCalculator with several 
single-responsibility classes that work together through composition.

Each class has exactly one reason to change and one clear responsibility.

Author: Clean Architecture Engineer (Uncle Bob's expertise)
Date: 2025-08-03
Version: 4.0.0 - SRP Compliant
"""

import numpy as np
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, FrozenSet
from abc import ABC, abstractmethod
import itertools
import time
import logging

logger = logging.getLogger(__name__)


# ===== SINGLE RESPONSIBILITY INTERFACES =====

class IPhiCalculationStrategy(ABC):
    """Strategy for phi calculation algorithms"""
    
    @abstractmethod
    def calculate_phi_value(self, distinctions: List, relations: List) -> float:
        """Calculate phi value from distinctions and relations"""
        pass


class ISubstrateDiscovery(ABC):
    """Strategy for substrate discovery"""
    
    @abstractmethod
    def find_maximal_substrate(self, system_state: np.ndarray, tpm: np.ndarray) -> FrozenSet[int]:
        """Find maximal substrate for given system"""
        pass


class IExistenceValidator(ABC):
    """Validator for system existence"""
    
    @abstractmethod
    def validate_existence(self, system_state: np.ndarray) -> bool:
        """Validate that system meets existence criteria"""
        pass


class IStructureAnalyzer(ABC):
    """Analyzer for phi structure properties"""
    
    @abstractmethod
    def analyze_complexity(self, distinctions: List, relations: List) -> float:
        """Analyze structure complexity"""
        pass
    
    @abstractmethod
    def analyze_definiteness(self, distinctions: List) -> float:
        """Analyze exclusion definiteness"""
        pass
    
    @abstractmethod
    def analyze_richness(self, distinctions: List, substrate: FrozenSet[int]) -> float:
        """Analyze composition richness"""
        pass


class ITpmBuilder(ABC):
    """Builder for transition probability matrices"""
    
    @abstractmethod
    def build_from_connectivity(self, connectivity_matrix: np.ndarray) -> np.ndarray:
        """Build TPM from connectivity matrix"""
        pass


# ===== SINGLE RESPONSIBILITY IMPLEMENTATIONS =====

class StandardPhiCalculation(IPhiCalculationStrategy):
    """
    Single Responsibility: Calculate phi values using standard IIT algorithm
    
    Reason to change: Changes in phi calculation methodology
    """
    
    def __init__(self, precision: float = 1e-10):
        self.precision = precision
    
    def calculate_phi_value(self, distinctions: List, relations: List) -> float:
        """Calculate total phi from distinctions"""
        if not distinctions:
            return 0.0
        
        # Sum phi values from all distinctions
        total_phi = sum(getattr(d, 'phi_value', 0.0) for d in distinctions)
        
        # Apply integration penalty based on relations
        if relations:
            integration_factor = len(relations) / max(len(distinctions) * (len(distinctions) - 1) / 2, 1)
            total_phi *= (1.0 + integration_factor * 0.1)
        
        return max(0.0, total_phi)


class ActivityBasedSubstrateDiscovery(ISubstrateDiscovery):
    """
    Single Responsibility: Discover optimal substrate based on activity
    
    Reason to change: Changes in substrate discovery algorithms
    """
    
    def __init__(self, max_substrate_size: int = 8, precision: float = 1e-10):
        self.max_substrate_size = max_substrate_size
        self.precision = precision
    
    def find_maximal_substrate(self, system_state: np.ndarray, tmp: np.ndarray) -> FrozenSet[int]:
        """Find maximal substrate using activity-based heuristics"""
        n_nodes = len(system_state)
        
        if n_nodes <= self.max_substrate_size:
            return frozenset(range(n_nodes))
        
        # Select nodes with highest activity levels
        activity_scores = self._calculate_activity_scores(system_state)
        top_indices = np.argsort(activity_scores)[-self.max_substrate_size:]
        
        return frozenset(top_indices)
    
    def _calculate_activity_scores(self, system_state: np.ndarray) -> np.ndarray:
        """Calculate activity scores for each node"""
        # Simple activity scoring - could be enhanced
        return system_state.copy()


class StandardExistenceValidator(IExistenceValidator):
    """
    Single Responsibility: Validate system existence according to IIT axioms
    
    Reason to change: Changes in existence validation criteria
    """
    
    def __init__(self, min_activity_ratio: float = 0.1, precision: float = 1e-10):
        self.min_activity_ratio = min_activity_ratio
        self.precision = precision
    
    def validate_existence(self, system_state: np.ndarray) -> bool:
        """Validate system meets existence criteria (Axiom 0)"""
        if len(system_state) == 0:
            return False
        
        # Check minimum activity level
        active_nodes = np.sum(system_state > self.precision)
        min_activity_threshold = max(1, len(system_state) * self.min_activity_ratio)
        
        return active_nodes >= min_activity_threshold


class ComprehensiveStructureAnalyzer(IStructureAnalyzer):
    """
    Single Responsibility: Analyze phi structure properties
    
    Reason to change: Changes in structure analysis methods
    """
    
    def analyze_complexity(self, distinctions: List, relations: List) -> float:
        """Calculate phi structure complexity"""
        if not distinctions:
            return 0.0
        
        n_distinctions = len(distinctions)
        n_relations = len(relations)
        
        # Complexity based on distinction count and relation density
        max_possible_relations = n_distinctions * (n_distinctions - 1) / 2 if n_distinctions > 1 else 1
        relation_density = n_relations / max_possible_relations if max_possible_relations > 0 else 0.0
        
        complexity = n_distinctions * relation_density
        return complexity
    
    def analyze_definiteness(self, distinctions: List) -> float:
        """Calculate exclusion definiteness"""
        if not distinctions:
            return 0.0
        
        # Extract phi values
        phi_values = [getattr(d, 'phi_value', 0.0) for d in distinctions]
        
        if not phi_values or len(phi_values) < 2:
            return 0.0
        
        # High variance indicates clear exclusion
        phi_variance = np.var(phi_values)
        return min(phi_variance, 1.0)
    
    def analyze_richness(self, distinctions: List, substrate: FrozenSet[int]) -> float:
        """Calculate composition richness"""
        if not distinctions or not substrate:
            return 0.0
        
        # Analyze mechanism size diversity
        mechanism_sizes = [len(getattr(d, 'mechanism', set())) for d in distinctions]
        unique_sizes = len(set(mechanism_sizes))
        max_possible_sizes = len(substrate)
        
        richness = unique_sizes / max(max_possible_sizes, 1)
        return richness


class SigmoidTpmBuilder(ITpmBuilder):
    """
    Single Responsibility: Build transition probability matrices
    
    Reason to change: Changes in TPM construction algorithms
    """
    
    def build_from_connectivity(self, connectivity_matrix: np.ndarray) -> np.ndarray:
        """Build TPM using sigmoid activation functions"""
        n_nodes = connectivity_matrix.shape[0]
        n_states = 2 ** n_nodes
        tpm = np.zeros((n_states, n_nodes))
        
        for state_idx in range(n_states):
            current_state = self._index_to_state(state_idx, n_nodes)
            
            for node in range(n_nodes):
                input_sum = np.dot(connectivity_matrix[node], current_state)
                activation_prob = self._sigmoid(input_sum)
                tpm[state_idx, node] = activation_prob
        
        return tpm
    
    def _index_to_state(self, index: int, n_nodes: int) -> np.ndarray:
        """Convert state index to binary state vector"""
        binary_str = format(index, f'0{n_nodes}b')
        return np.array([int(bit) for bit in binary_str])
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ===== COMPOSITION ROOT - SRP COMPLIANT PHI CALCULATOR =====

@dataclass
class PhiStructureResult:
    """Result of phi structure calculation"""
    total_phi: float
    maximal_substrate: FrozenSet[int]
    distinctions: List
    relations: List
    complexity: float
    definiteness: float
    richness: float
    calculation_time_ms: float


class SRPCompliantPhiCalculator:
    """
    SRP-Compliant Phi Calculator using Composition
    
    Single Responsibility: Orchestrate phi calculation process
    Reason to change: Changes in overall calculation workflow
    
    All domain-specific logic is delegated to single-responsibility components
    """
    
    def __init__(self,
                 phi_strategy: IPhiCalculationStrategy,
                 substrate_discovery: ISubstrateDiscovery,
                 existence_validator: IExistenceValidator,
                 structure_analyzer: IStructureAnalyzer,
                 tpm_builder: ITpmBuilder):
        
        # Dependency injection - all components have single responsibilities
        self.phi_strategy = phi_strategy
        self.substrate_discovery = substrate_discovery
        self.existence_validator = existence_validator
        self.structure_analyzer = structure_analyzer
        self.tpm_builder = tpm_builder
    
    def calculate_phi_structure(self, 
                              system_state: np.ndarray, 
                              connectivity_matrix: np.ndarray,
                              tpm: Optional[np.ndarray] = None) -> PhiStructureResult:
        """
        Calculate complete phi structure using single-responsibility components
        
        This method orchestrates the calculation but delegates all domain logic
        to specialized single-responsibility classes.
        """
        start_time = time.time()
        
        # Step 1: Validate existence using dedicated validator
        if not self.existence_validator.validate_existence(system_state):
            return self._empty_result(start_time)
        
        # Step 2: Build TPM using dedicated builder
        if tpm is None:
            tpm = self.tpm_builder.build_from_connectivity(connectivity_matrix)
        
        # Step 3: Discover maximal substrate using dedicated discoverer
        maximal_substrate = self.substrate_discovery.find_maximal_substrate(system_state, tpm)
        
        # Step 4: Calculate distinctions and relations (simplified for demo)
        distinctions, relations = self._calculate_distinctions_and_relations(
            system_state, tpm, maximal_substrate
        )
        
        # Step 5: Calculate phi using dedicated strategy
        total_phi = self.phi_strategy.calculate_phi_value(distinctions, relations)
        
        # Step 6: Analyze structure using dedicated analyzer
        complexity = self.structure_analyzer.analyze_complexity(distinctions, relations)
        definiteness = self.structure_analyzer.analyze_definiteness(distinctions)
        richness = self.structure_analyzer.analyze_richness(distinctions, maximal_substrate)
        
        calculation_time = (time.time() - start_time) * 1000
        
        return PhiStructureResult(
            total_phi=total_phi,
            maximal_substrate=maximal_substrate,
            distinctions=distinctions,
            relations=relations,
            complexity=complexity,
            definiteness=definiteness,
            richness=richness,
            calculation_time_ms=calculation_time
        )
    
    def _calculate_distinctions_and_relations(self, 
                                            system_state: np.ndarray,
                                            tpm: np.ndarray,
                                            substrate: FrozenSet[int]) -> Tuple[List, List]:
        """Calculate distinctions and relations (simplified for demo)"""
        
        distinctions = []
        
        # Create mock distinctions for demonstration
        for i, node in enumerate(substrate):
            if i < 3:  # Limit for demo
                distinction = type('Distinction', (), {
                    'mechanism': frozenset([node]),
                    'phi_value': system_state[node] * 0.1 if node < len(system_state) else 0.1
                })()
                distinctions.append(distinction)
        
        # Create mock relations
        relations = []
        for i in range(len(distinctions)):
            for j in range(i + 1, len(distinctions)):
                relation = type('Relation', (), {
                    'distinction_pair': (distinctions[i], distinctions[j]),
                    'strength': 0.5
                })()
                relations.append(relation)
        
        return distinctions, relations
    
    def _empty_result(self, start_time: float) -> PhiStructureResult:
        """Return empty result for non-existent systems"""
        return PhiStructureResult(
            total_phi=0.0,
            maximal_substrate=frozenset(),
            distinctions=[],
            relations=[],
            complexity=0.0,
            definiteness=0.0,
            richness=0.0,
            calculation_time_ms=(time.time() - start_time) * 1000
        )


# ===== FACTORY FOR CREATING SRP-COMPLIANT SYSTEM =====

class SRPCompliantPhiCalculatorFactory:
    """Factory for creating SRP-compliant phi calculator with default components"""
    
    @staticmethod
    def create_standard_calculator(max_substrate_size: int = 8, 
                                 precision: float = 1e-10) -> SRPCompliantPhiCalculator:
        """Create phi calculator with standard single-responsibility components"""
        
        # Create all single-responsibility components
        phi_strategy = StandardPhiCalculation(precision)
        substrate_discovery = ActivityBasedSubstrateDiscovery(max_substrate_size, precision)
        existence_validator = StandardExistenceValidator(precision=precision)
        structure_analyzer = ComprehensiveStructureAnalyzer()
        tpm_builder = SigmoidTpmBuilder()
        
        # Compose calculator using dependency injection
        return SRPCompliantPhiCalculator(
            phi_strategy=phi_strategy,
            substrate_discovery=substrate_discovery,
            existence_validator=existence_validator,
            structure_analyzer=structure_analyzer,
            tpm_builder=tpm_builder
        )
    
    @staticmethod
    def create_custom_calculator(phi_strategy: IPhiCalculationStrategy,
                               substrate_discovery: ISubstrateDiscovery,
                               existence_validator: IExistenceValidator,
                               structure_analyzer: IStructureAnalyzer,
                               tpm_builder: ITpmBuilder) -> SRPCompliantPhiCalculator:
        """Create phi calculator with custom components"""
        
        return SRPCompliantPhiCalculator(
            phi_strategy=phi_strategy,
            substrate_discovery=substrate_discovery, 
            existence_validator=existence_validator,
            structure_analyzer=structure_analyzer,
            tpm_builder=tpm_builder
        )


# ===== DEMONSTRATION =====

async def demonstrate_srp_compliant_phi_calculator():
    """Demonstrate SRP-compliant phi calculator"""
    
    print("🧮 SRP-Compliant IIT 4.0 Phi Calculator")
    print("=" * 50)
    
    # Create SRP-compliant calculator
    calculator = SRPCompliantPhiCalculatorFactory.create_standard_calculator()
    
    print("✅ Created calculator with single-responsibility components:")
    print("   • StandardPhiCalculation - phi calculation only")
    print("   • ActivityBasedSubstrateDiscovery - substrate discovery only")
    print("   • StandardExistenceValidator - existence validation only")
    print("   • ComprehensiveStructureAnalyzer - structure analysis only")
    print("   • SigmoidTmpBuilder - TPM building only")
    print()
    
    # Test calculation
    print("🧪 Testing Calculation")
    print("-" * 30)
    
    system_state = np.array([0.8, 0.6, 0.7, 0.5])
    connectivity_matrix = np.array([
        [0.0, 0.5, 0.3, 0.2],
        [0.4, 0.0, 0.6, 0.1],
        [0.2, 0.7, 0.0, 0.5],
        [0.3, 0.1, 0.4, 0.0]
    ])
    
    result = calculator.calculate_phi_structure(system_state, connectivity_matrix)
    
    print(f"Total Phi: {result.total_phi:.6f}")
    print(f"Maximal Substrate: {result.maximal_substrate}")
    print(f"Distinctions: {len(result.distinctions)}")
    print(f"Relations: {len(result.relations)}")
    print(f"Complexity: {result.complexity:.3f}")
    print(f"Definiteness: {result.definiteness:.3f}")
    print(f"Richness: {result.richness:.3f}")
    print(f"Calculation Time: {result.calculation_time_ms:.2f}ms")
    print()
    
    print("✅ SRP Benefits Demonstrated:")
    print("   • Each component has single, clear responsibility")
    print("   • Easy to test individual components in isolation")
    print("   • Easy to replace/extend individual components")
    print("   • Clear separation of concerns")
    print("   • High cohesion within each component")
    print("   • Loose coupling between components")


if __name__ == "__main__":
    asyncio.run(demonstrate_srp_compliant_phi_calculator())