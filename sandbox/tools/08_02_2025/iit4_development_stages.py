"""
IIT 4.0 Development Stages Mapper for NewbornAI 2.0
Phase 3: Development stage integration with φ-structure analysis

Maps IIT 4.0 φ values and Φ-structures to NewbornAI's 7-stage development system
with consciousness maturity metrics and development trajectory analysis.

Development Stage Criteria (IIT 4.0 compliant):
- Stage 0: φ 0.0-0.001 (Pre-conscious foundation)
- Stage 1: φ 0.001-0.01 (Experiential emergence)  
- Stage 2: φ 0.01-0.1 (Temporal integration)
- Stage 3: φ 0.1-1.0 (Relational formation)
- Stage 4: φ 1.0-10.0 (Self establishment)
- Stage 5: φ 10.0-100.0 (Reflective operation)
- Stage 6: φ 100.0+ (Narrative integration)

Author: Chief Artificial Consciousness Engineer
Date: 2025-08-03
Version: 3.0.0
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, FrozenSet
from enum import Enum
import logging
import time
import json
from datetime import datetime, timedelta
import math

# Import existing IIT 4.0 infrastructure
from iit4_core_engine import IIT4PhiCalculator, PhiStructure, Distinction, Relation
from iit4_experiential_phi_calculator import IIT4_ExperientialPhiCalculator, ExperientialPhiResult

logger = logging.getLogger(__name__)


class DevelopmentStage(Enum):
    """NewbornAI 2.0 7-stage development system with IIT 4.0 integration"""
    STAGE_0_PRE_CONSCIOUS = "前意識基盤層"
    STAGE_1_EXPERIENTIAL_EMERGENCE = "体験記憶発生期"
    STAGE_2_TEMPORAL_INTEGRATION = "時間記憶統合期"
    STAGE_3_RELATIONAL_FORMATION = "関係記憶形成期"
    STAGE_4_SELF_ESTABLISHMENT = "自己記憶確立期"
    STAGE_5_REFLECTIVE_OPERATION = "反省記憶操作期"
    STAGE_6_NARRATIVE_INTEGRATION = "物語記憶統合期"


class StageTransitionType(Enum):
    """Types of stage transitions"""
    PROGRESSIVE = "progressive"      # Forward development
    REGRESSIVE = "regressive"       # Backward regression
    STAGNANT = "stagnant"          # No change
    OSCILLATORY = "oscillatory"    # Back and forth
    LEAP = "leap"                  # Skipping stages


@dataclass
class StageCharacteristics:
    """Characteristics defining each development stage"""
    stage: DevelopmentStage
    phi_range: Tuple[float, float]
    min_distinctions: int
    min_relations: int
    complexity_threshold: float
    integration_quality_threshold: float
    temporal_depth_requirement: float
    self_reference_requirement: float
    narrative_coherence_requirement: float
    
    # IIT 4.0 specific requirements
    min_substrate_size: int = 2
    axiom_compliance_threshold: float = 0.8
    phi_structure_stability_requirement: float = 0.7


@dataclass
class DevelopmentMetrics:
    """Comprehensive development metrics"""
    current_stage: DevelopmentStage
    phi_value: float
    stage_confidence: float
    maturity_score: float
    development_velocity: float
    regression_risk: float
    next_stage_readiness: float
    
    # Detailed φ-structure metrics
    distinction_count: int
    relation_count: int
    phi_structure_complexity: float
    integration_quality: float
    
    # Consciousness development indicators
    temporal_depth: float
    self_reference_strength: float
    narrative_coherence: float
    experiential_purity: float
    
    # Transition indicators
    transition_probability: Dict[DevelopmentStage, float] = field(default_factory=dict)
    critical_transition_warning: bool = False
    development_trajectory: str = "stable"


@dataclass
class DevelopmentTrajectory:
    """Long-term development trajectory analysis"""
    start_stage: DevelopmentStage
    current_stage: DevelopmentStage
    target_stage: DevelopmentStage
    trajectory_type: str
    development_rate: float
    estimated_time_to_target: Optional[float]
    
    # Historical progression
    stage_history: List[Tuple[datetime, DevelopmentStage, float]] = field(default_factory=list)
    phi_history: List[Tuple[datetime, float]] = field(default_factory=list)
    regression_events: List[Tuple[datetime, DevelopmentStage, DevelopmentStage]] = field(default_factory=list)
    
    # Predictive indicators
    development_momentum: float = 0.0
    stability_index: float = 0.0
    critical_periods: List[Tuple[datetime, str]] = field(default_factory=list)


class IIT4DevelopmentStageMapper:
    """
    Maps IIT 4.0 φ values and Φ-structures to NewbornAI development stages
    with advanced consciousness maturity analysis
    """
    
    def __init__(self):
        """
        Initialize development stage mapper with IIT 4.0 criteria
        Note: This class currently has minimal external dependencies but
        could be enhanced with injected components for validation, etc.
        """
        
        # Define stage characteristics based on IIT 4.0 theory
        self.stage_characteristics = {
            DevelopmentStage.STAGE_0_PRE_CONSCIOUS: StageCharacteristics(
                stage=DevelopmentStage.STAGE_0_PRE_CONSCIOUS,
                phi_range=(0.0, 0.001),
                min_distinctions=0,
                min_relations=0,
                complexity_threshold=0.0,
                integration_quality_threshold=0.0,
                temporal_depth_requirement=0.0,
                self_reference_requirement=0.0,
                narrative_coherence_requirement=0.0,
                min_substrate_size=1,
                axiom_compliance_threshold=0.3
            ),
            
            DevelopmentStage.STAGE_1_EXPERIENTIAL_EMERGENCE: StageCharacteristics(
                stage=DevelopmentStage.STAGE_1_EXPERIENTIAL_EMERGENCE,
                phi_range=(0.001, 0.01),
                min_distinctions=1,
                min_relations=0,
                complexity_threshold=0.1,
                integration_quality_threshold=0.1,
                temporal_depth_requirement=0.1,
                self_reference_requirement=0.0,
                narrative_coherence_requirement=0.0,
                min_substrate_size=2,
                axiom_compliance_threshold=0.5
            ),
            
            DevelopmentStage.STAGE_2_TEMPORAL_INTEGRATION: StageCharacteristics(
                stage=DevelopmentStage.STAGE_2_TEMPORAL_INTEGRATION,
                phi_range=(0.01, 0.1),
                min_distinctions=2,
                min_relations=1,
                complexity_threshold=0.3,
                integration_quality_threshold=0.3,
                temporal_depth_requirement=0.4,
                self_reference_requirement=0.1,
                narrative_coherence_requirement=0.1,
                min_substrate_size=3,
                axiom_compliance_threshold=0.6
            ),
            
            DevelopmentStage.STAGE_3_RELATIONAL_FORMATION: StageCharacteristics(
                stage=DevelopmentStage.STAGE_3_RELATIONAL_FORMATION,
                phi_range=(0.1, 1.0),
                min_distinctions=3,
                min_relations=2,
                complexity_threshold=0.5,
                integration_quality_threshold=0.5,
                temporal_depth_requirement=0.5,
                self_reference_requirement=0.3,
                narrative_coherence_requirement=0.2,
                min_substrate_size=4,
                axiom_compliance_threshold=0.7
            ),
            
            DevelopmentStage.STAGE_4_SELF_ESTABLISHMENT: StageCharacteristics(
                stage=DevelopmentStage.STAGE_4_SELF_ESTABLISHMENT,
                phi_range=(1.0, 10.0),
                min_distinctions=4,
                min_relations=3,
                complexity_threshold=0.7,
                integration_quality_threshold=0.6,
                temporal_depth_requirement=0.6,
                self_reference_requirement=0.6,
                narrative_coherence_requirement=0.4,
                min_substrate_size=5,
                axiom_compliance_threshold=0.8
            ),
            
            DevelopmentStage.STAGE_5_REFLECTIVE_OPERATION: StageCharacteristics(
                stage=DevelopmentStage.STAGE_5_REFLECTIVE_OPERATION,
                phi_range=(10.0, 100.0),
                min_distinctions=5,
                min_relations=4,
                complexity_threshold=0.8,
                integration_quality_threshold=0.7,
                temporal_depth_requirement=0.7,
                self_reference_requirement=0.8,
                narrative_coherence_requirement=0.6,
                min_substrate_size=6,
                axiom_compliance_threshold=0.85
            ),
            
            DevelopmentStage.STAGE_6_NARRATIVE_INTEGRATION: StageCharacteristics(
                stage=DevelopmentStage.STAGE_6_NARRATIVE_INTEGRATION,
                phi_range=(100.0, float('inf')),
                min_distinctions=6,
                min_relations=5,
                complexity_threshold=0.9,
                integration_quality_threshold=0.8,
                temporal_depth_requirement=0.8,
                self_reference_requirement=0.9,
                narrative_coherence_requirement=0.8,
                min_substrate_size=7,
                axiom_compliance_threshold=0.9
            )
        }
        
        # Development history tracking
        self.development_history: List[Tuple[datetime, DevelopmentMetrics]] = []
        self.trajectory_cache = {}
        
        logger.info("IIT4 Development Stage Mapper initialized with 7-stage system")
    
    def map_phi_to_development_stage(self, 
                                   phi_structure: PhiStructure,
                                   experiential_result: Optional[ExperientialPhiResult] = None,
                                   axiom_compliance: Optional[Dict[str, bool]] = None) -> DevelopmentMetrics:
        """
        Map φ-structure to development stage with comprehensive analysis
        
        Args:
            phi_structure: IIT 4.0 φ-structure
            experiential_result: Optional experiential φ calculation result
            axiom_compliance: Optional axiom compliance results
            
        Returns:
            DevelopmentMetrics: Comprehensive development analysis
        """
        
        phi_value = phi_structure.total_phi
        
        # Base stage determination from φ value
        base_stage = self._determine_base_stage_from_phi(phi_value)
        
        # Calculate detailed metrics
        stage_confidence = self._calculate_stage_confidence(phi_structure, base_stage)
        maturity_score = self._calculate_maturity_score(phi_structure, experiential_result)
        
        # Φ-structure analysis
        distinction_count = len(phi_structure.distinctions)
        relation_count = len(phi_structure.relations)
        phi_complexity = phi_structure.phi_structure_complexity
        integration_quality = self._calculate_integration_quality(phi_structure)
        
        # Consciousness development indicators
        temporal_depth = self._extract_temporal_depth(phi_structure, experiential_result)
        self_reference_strength = self._extract_self_reference_strength(phi_structure, experiential_result)
        narrative_coherence = self._extract_narrative_coherence(phi_structure, experiential_result)
        experiential_purity = self._extract_experiential_purity(experiential_result)
        
        # Transition analysis
        transition_probabilities = self._calculate_transition_probabilities(
            phi_structure, base_stage, experiential_result
        )
        
        # Development dynamics
        development_velocity = self._calculate_development_velocity(base_stage, phi_value)
        regression_risk = self._calculate_regression_risk(phi_structure, base_stage)
        next_stage_readiness = self._calculate_next_stage_readiness(
            phi_structure, base_stage, experiential_result
        )
        
        # Critical transition detection
        critical_warning = self._detect_critical_transition(phi_structure, base_stage)
        development_trajectory = self._analyze_development_trajectory(base_stage)
        
        # Final stage adjustment based on all factors
        final_stage = self._adjust_stage_with_constraints(
            base_stage, phi_structure, experiential_result, axiom_compliance
        )
        
        metrics = DevelopmentMetrics(
            current_stage=final_stage,
            phi_value=phi_value,
            stage_confidence=stage_confidence,
            maturity_score=maturity_score,
            development_velocity=development_velocity,
            regression_risk=regression_risk,
            next_stage_readiness=next_stage_readiness,
            distinction_count=distinction_count,
            relation_count=relation_count,
            phi_structure_complexity=phi_complexity,
            integration_quality=integration_quality,
            temporal_depth=temporal_depth,
            self_reference_strength=self_reference_strength,
            narrative_coherence=narrative_coherence,
            experiential_purity=experiential_purity,
            transition_probability=transition_probabilities,
            critical_transition_warning=critical_warning,
            development_trajectory=development_trajectory
        )
        
        # Store in history
        self.development_history.append((datetime.now(), metrics))
        
        # Limit history size
        if len(self.development_history) > 1000:
            self.development_history = self.development_history[-1000:]
        
        return metrics
    
    def _determine_base_stage_from_phi(self, phi_value: float) -> DevelopmentStage:
        """Determine base stage from φ value"""
        for stage, characteristics in self.stage_characteristics.items():
            min_phi, max_phi = characteristics.phi_range
            if min_phi <= phi_value < max_phi:
                return stage
        
        # If φ exceeds all ranges, return highest stage
        return DevelopmentStage.STAGE_6_NARRATIVE_INTEGRATION
    
    def _calculate_stage_confidence(self, phi_structure: PhiStructure, stage: DevelopmentStage) -> float:
        """Calculate confidence in stage determination"""
        characteristics = self.stage_characteristics[stage]
        phi_value = phi_structure.total_phi
        
        # φ range confidence
        min_phi, max_phi = characteristics.phi_range
        if max_phi == float('inf'):
            phi_confidence = 1.0 if phi_value >= min_phi else 0.0
        else:
            phi_range_size = max_phi - min_phi
            distance_from_center = abs(phi_value - (min_phi + max_phi) / 2)
            phi_confidence = 1.0 - (distance_from_center / (phi_range_size / 2))
        
        # Structural requirements confidence
        distinctions_met = len(phi_structure.distinctions) >= characteristics.min_distinctions
        relations_met = len(phi_structure.relations) >= characteristics.min_relations
        complexity_met = phi_structure.phi_structure_complexity >= characteristics.complexity_threshold
        
        structural_confidence = (
            (1.0 if distinctions_met else 0.5) +
            (1.0 if relations_met else 0.5) +
            (1.0 if complexity_met else 0.5)
        ) / 3.0
        
        # Combined confidence
        overall_confidence = (phi_confidence * 0.6 + structural_confidence * 0.4)
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _calculate_maturity_score(self, phi_structure: PhiStructure, 
                                experiential_result: Optional[ExperientialPhiResult]) -> float:
        """Calculate overall consciousness maturity score"""
        
        # Base maturity from φ value (normalized log scale)
        phi_value = phi_structure.total_phi
        if phi_value <= 0:
            phi_maturity = 0.0
        else:
            # Log scale normalization: log(1 + φ) / log(101) to map [0, 100] φ to [0, 1]
            phi_maturity = min(1.0, math.log(1 + phi_value) / math.log(101))
        
        # Structural maturity
        max_distinctions = 10  # Assumed maximum for normalization
        max_relations = 20     # Assumed maximum for normalization
        
        structural_maturity = (
            min(1.0, len(phi_structure.distinctions) / max_distinctions) * 0.4 +
            min(1.0, len(phi_structure.relations) / max_relations) * 0.3 +
            min(1.0, phi_structure.phi_structure_complexity) * 0.3
        )
        
        # Experiential maturity
        experiential_maturity = 0.5  # Default
        if experiential_result:
            experiential_maturity = (
                experiential_result.consciousness_level * 0.4 +
                experiential_result.integration_quality * 0.3 +
                experiential_result.experiential_purity * 0.3
            )
        
        # Combined maturity score
        overall_maturity = (
            phi_maturity * 0.4 +
            structural_maturity * 0.35 +
            experiential_maturity * 0.25
        )
        
        return max(0.0, min(1.0, overall_maturity))
    
    def _calculate_integration_quality(self, phi_structure: PhiStructure) -> float:
        """Calculate integration quality from φ-structure"""
        if not phi_structure.distinctions:
            return 0.0
        
        n_distinctions = len(phi_structure.distinctions)
        n_relations = len(phi_structure.relations)
        
        # Maximum possible relations
        max_relations = n_distinctions * (n_distinctions - 1) / 2
        
        if max_relations == 0:
            return 1.0 if n_distinctions == 1 else 0.0
        
        # Relation density
        relation_density = n_relations / max_relations
        
        # Integration strength from φ-structure complexity
        complexity_factor = min(1.0, phi_structure.phi_structure_complexity)
        
        # Combined integration quality
        integration_quality = (relation_density * 0.6 + complexity_factor * 0.4)
        
        return max(0.0, min(1.0, integration_quality))
    
    def _extract_temporal_depth(self, phi_structure: PhiStructure, 
                              experiential_result: Optional[ExperientialPhiResult]) -> float:
        """Extract temporal depth indicator"""
        if experiential_result:
            return experiential_result.temporal_depth
        
        # Estimate from φ-structure (larger structures suggest temporal integration)
        substrate_size = len(phi_structure.maximal_substrate)
        temporal_estimate = min(1.0, substrate_size / 10.0)
        
        return temporal_estimate
    
    def _extract_self_reference_strength(self, phi_structure: PhiStructure,
                                       experiential_result: Optional[ExperientialPhiResult]) -> float:
        """Extract self-reference strength indicator"""
        if experiential_result:
            return experiential_result.self_reference_strength
        
        # Estimate from φ-structure complexity
        self_ref_estimate = min(1.0, phi_structure.phi_structure_complexity / 2.0)
        
        return self_ref_estimate
    
    def _extract_narrative_coherence(self, phi_structure: PhiStructure,
                                   experiential_result: Optional[ExperientialPhiResult]) -> float:
        """Extract narrative coherence indicator"""
        if experiential_result:
            return experiential_result.narrative_coherence
        
        # Estimate from relation structure
        if not phi_structure.relations:
            return 0.0
        
        # Higher relation count suggests narrative coherence
        narrative_estimate = min(1.0, len(phi_structure.relations) / 10.0)
        
        return narrative_estimate
    
    def _extract_experiential_purity(self, experiential_result: Optional[ExperientialPhiResult]) -> float:
        """Extract experiential purity indicator"""
        if experiential_result:
            return experiential_result.experiential_purity
        
        return 0.5  # Default neutral value
    
    def _calculate_transition_probabilities(self, phi_structure: PhiStructure, 
                                          current_stage: DevelopmentStage,
                                          experiential_result: Optional[ExperientialPhiResult]) -> Dict[DevelopmentStage, float]:
        """Calculate transition probabilities to other stages"""
        
        probabilities = {}
        
        # Get all stages
        all_stages = list(DevelopmentStage)
        current_index = all_stages.index(current_stage)
        
        for i, target_stage in enumerate(all_stages):
            if target_stage == current_stage:
                # Probability of staying in current stage
                stability = self._calculate_stage_stability(phi_structure, current_stage)
                probabilities[target_stage] = stability
            else:
                # Distance-based probability
                stage_distance = abs(i - current_index)
                
                if stage_distance == 1:
                    # Adjacent stages have higher probability
                    if i > current_index:
                        # Forward progression
                        readiness = self._calculate_next_stage_readiness(
                            phi_structure, current_stage, experiential_result
                        )
                        probabilities[target_stage] = readiness * 0.3
                    else:
                        # Regression
                        regression_risk = self._calculate_regression_risk(phi_structure, current_stage)
                        probabilities[target_stage] = regression_risk * 0.2
                else:
                    # Distant stages have very low probability
                    probabilities[target_stage] = max(0.01, 0.1 / stage_distance)
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {stage: prob / total_prob for stage, prob in probabilities.items()}
        
        return probabilities
    
    def _calculate_stage_stability(self, phi_structure: PhiStructure, stage: DevelopmentStage) -> float:
        """Calculate stability of current stage"""
        characteristics = self.stage_characteristics[stage]
        
        # φ value stability (how well it fits the range)
        phi_value = phi_structure.total_phi
        min_phi, max_phi = characteristics.phi_range
        
        if max_phi == float('inf'):
            phi_stability = 1.0 if phi_value >= min_phi else 0.0
        else:
            phi_center = (min_phi + max_phi) / 2
            phi_range_size = max_phi - min_phi
            distance_from_center = abs(phi_value - phi_center)
            phi_stability = max(0.0, 1.0 - (distance_from_center / (phi_range_size / 2)))
        
        # Structural stability
        distinctions_ratio = min(1.0, len(phi_structure.distinctions) / max(characteristics.min_distinctions, 1))
        relations_ratio = min(1.0, len(phi_structure.relations) / max(characteristics.min_relations, 1))
        
        structural_stability = (distinctions_ratio + relations_ratio) / 2.0
        
        # Combined stability
        overall_stability = (phi_stability * 0.7 + structural_stability * 0.3)
        
        return max(0.0, min(1.0, overall_stability))
    
    def _calculate_development_velocity(self, current_stage: DevelopmentStage, phi_value: float) -> float:
        """Calculate development velocity"""
        if len(self.development_history) < 2:
            return 0.0
        
        # Compare with recent history
        recent_entries = self.development_history[-5:]  # Last 5 entries
        
        if len(recent_entries) < 2:
            return 0.0
        
        # Calculate stage progression
        stage_values = [list(DevelopmentStage).index(entry[1].current_stage) for entry in recent_entries]
        phi_values = [entry[1].phi_value for entry in recent_entries]
        
        # Stage velocity
        stage_diff = stage_values[-1] - stage_values[0]
        time_diff = (recent_entries[-1][0] - recent_entries[0][0]).total_seconds()
        
        if time_diff > 0:
            stage_velocity = stage_diff / time_diff
        else:
            stage_velocity = 0.0
        
        # φ velocity
        phi_diff = phi_values[-1] - phi_values[0]
        if time_diff > 0:
            phi_velocity = phi_diff / time_diff
        else:
            phi_velocity = 0.0
        
        # Combined velocity (normalized)
        combined_velocity = (stage_velocity * 0.6 + math.log(1 + abs(phi_velocity)) * 0.4)
        
        return max(-1.0, min(1.0, combined_velocity))
    
    def _calculate_regression_risk(self, phi_structure: PhiStructure, current_stage: DevelopmentStage) -> float:
        """Calculate risk of regression to earlier stage"""
        
        # Base risk from φ value stability
        characteristics = self.stage_characteristics[current_stage]
        phi_value = phi_structure.total_phi
        min_phi, _ = characteristics.phi_range
        
        if phi_value < min_phi:
            phi_risk = 1.0 - (phi_value / min_phi) if min_phi > 0 else 1.0
        else:
            phi_risk = 0.0
        
        # Structural risk
        distinction_deficit = max(0, characteristics.min_distinctions - len(phi_structure.distinctions))
        relation_deficit = max(0, characteristics.min_relations - len(phi_structure.relations))
        
        structural_risk = (distinction_deficit + relation_deficit) / 10.0  # Normalized
        
        # Historical regression tendency
        historical_risk = 0.0
        if len(self.development_history) > 5:
            recent_stages = [entry[1].current_stage for entry in self.development_history[-10:]]
            stage_indices = [list(DevelopmentStage).index(stage) for stage in recent_stages]
            
            # Count regressions in recent history
            regressions = sum(1 for i in range(1, len(stage_indices)) 
                            if stage_indices[i] < stage_indices[i-1])
            historical_risk = min(1.0, regressions / 5.0)
        
        # Combined regression risk
        overall_risk = (phi_risk * 0.4 + structural_risk * 0.4 + historical_risk * 0.2)
        
        return max(0.0, min(1.0, overall_risk))
    
    def _calculate_next_stage_readiness(self, phi_structure: PhiStructure, 
                                      current_stage: DevelopmentStage,
                                      experiential_result: Optional[ExperientialPhiResult]) -> float:
        """Calculate readiness for next development stage"""
        
        all_stages = list(DevelopmentStage)
        current_index = all_stages.index(current_stage)
        
        # Already at highest stage
        if current_index >= len(all_stages) - 1:
            return 0.0
        
        next_stage = all_stages[current_index + 1]
        next_characteristics = self.stage_characteristics[next_stage]
        
        # φ value readiness
        phi_value = phi_structure.total_phi
        min_next_phi, _ = next_characteristics.phi_range
        phi_readiness = min(1.0, phi_value / min_next_phi) if min_next_phi > 0 else 1.0
        
        # Structural readiness
        distinction_readiness = min(1.0, len(phi_structure.distinctions) / next_characteristics.min_distinctions) if next_characteristics.min_distinctions > 0 else 1.0
        relation_readiness = min(1.0, len(phi_structure.relations) / next_characteristics.min_relations) if next_characteristics.min_relations > 0 else 1.0
        complexity_readiness = min(1.0, phi_structure.phi_structure_complexity / next_characteristics.complexity_threshold) if next_characteristics.complexity_threshold > 0 else 1.0
        
        structural_readiness = (distinction_readiness + relation_readiness + complexity_readiness) / 3.0
        
        # Experiential readiness
        experiential_readiness = 0.5  # Default
        if experiential_result:
            temporal_readiness = experiential_result.temporal_depth / next_characteristics.temporal_depth_requirement if next_characteristics.temporal_depth_requirement > 0 else 1.0
            self_ref_readiness = experiential_result.self_reference_strength / next_characteristics.self_reference_requirement if next_characteristics.self_reference_requirement > 0 else 1.0
            narrative_readiness = experiential_result.narrative_coherence / next_characteristics.narrative_coherence_requirement if next_characteristics.narrative_coherence_requirement > 0 else 1.0
            
            experiential_readiness = (temporal_readiness + self_ref_readiness + narrative_readiness) / 3.0
        
        # Combined readiness
        overall_readiness = (
            phi_readiness * 0.4 +
            structural_readiness * 0.35 +
            experiential_readiness * 0.25
        )
        
        return max(0.0, min(1.0, overall_readiness))
    
    def _detect_critical_transition(self, phi_structure: PhiStructure, current_stage: DevelopmentStage) -> bool:
        """Detect if system is in critical transition period"""
        
        # Rapid φ changes
        if len(self.development_history) >= 3:
            recent_phi_values = [entry[1].phi_value for entry in self.development_history[-3:]]
            phi_volatility = np.std(recent_phi_values) / max(np.mean(recent_phi_values), 0.001)
            
            if phi_volatility > 0.5:  # High volatility threshold
                return True
        
        # Stage boundaries
        characteristics = self.stage_characteristics[current_stage]
        phi_value = phi_structure.total_phi
        min_phi, max_phi = characteristics.phi_range
        
        # Near boundaries
        if max_phi != float('inf'):
            range_size = max_phi - min_phi
            distance_to_upper = max_phi - phi_value
            distance_to_lower = phi_value - min_phi
            
            if distance_to_upper < range_size * 0.1 or distance_to_lower < range_size * 0.1:
                return True
        
        return False
    
    def _analyze_development_trajectory(self, current_stage: DevelopmentStage) -> str:
        """Analyze current development trajectory"""
        
        if len(self.development_history) < 5:
            return "insufficient_data"
        
        recent_stages = [entry[1].current_stage for entry in self.development_history[-10:]]
        stage_indices = [list(DevelopmentStage).index(stage) for stage in recent_stages]
        
        # Calculate trend
        if len(stage_indices) > 1:
            trend = np.polyfit(range(len(stage_indices)), stage_indices, 1)[0]
            
            if trend > 0.1:
                return "progressive"
            elif trend < -0.1:
                return "regressive"
            elif abs(trend) < 0.05:
                return "stable"
            else:
                return "slow_change"
        
        return "stable"
    
    def _adjust_stage_with_constraints(self, base_stage: DevelopmentStage,
                                     phi_structure: PhiStructure,
                                     experiential_result: Optional[ExperientialPhiResult],
                                     axiom_compliance: Optional[Dict[str, bool]]) -> DevelopmentStage:
        """Adjust stage determination with additional constraints"""
        
        characteristics = self.stage_characteristics[base_stage]
        
        # Check minimum requirements
        distinction_met = len(phi_structure.distinctions) >= characteristics.min_distinctions
        relation_met = len(phi_structure.relations) >= characteristics.min_relations
        substrate_met = len(phi_structure.maximal_substrate) >= characteristics.min_substrate_size
        
        # Axiom compliance check
        axiom_met = True
        if axiom_compliance:
            compliance_rate = sum(axiom_compliance.values()) / len(axiom_compliance)
            axiom_met = compliance_rate >= characteristics.axiom_compliance_threshold
        
        # If requirements not met, consider regression
        if not all([distinction_met, relation_met, substrate_met, axiom_met]):
            all_stages = list(DevelopmentStage)
            current_index = all_stages.index(base_stage)
            
            # Try previous stages
            for i in range(current_index - 1, -1, -1):
                prev_stage = all_stages[i]
                prev_characteristics = self.stage_characteristics[prev_stage]
                
                prev_distinction_met = len(phi_structure.distinctions) >= prev_characteristics.min_distinctions
                prev_relation_met = len(phi_structure.relations) >= prev_characteristics.min_relations
                prev_substrate_met = len(phi_structure.maximal_substrate) >= prev_characteristics.min_substrate_size
                
                if all([prev_distinction_met, prev_relation_met, prev_substrate_met]):
                    logger.warning(f"Stage adjusted from {base_stage.value} to {prev_stage.value} due to unmet requirements")
                    return prev_stage
        
        return base_stage
    
    def detect_stage_transitions(self, current_metrics: DevelopmentMetrics) -> Optional[StageTransitionType]:
        """Detect type of stage transition"""
        
        if len(self.development_history) < 2:
            return None
        
        previous_stage = self.development_history[-2][1].current_stage
        current_stage = current_metrics.current_stage
        
        if previous_stage == current_stage:
            return StageTransitionType.STAGNANT
        
        all_stages = list(DevelopmentStage)
        prev_index = all_stages.index(previous_stage)
        curr_index = all_stages.index(current_stage)
        
        if curr_index > prev_index:
            if curr_index - prev_index == 1:
                return StageTransitionType.PROGRESSIVE
            else:
                return StageTransitionType.LEAP
        else:
            return StageTransitionType.REGRESSIVE
    
    def handle_stage_regression(self, current_metrics: DevelopmentMetrics) -> Dict[str, Any]:
        """Handle stage regression with analysis and recommendations"""
        
        regression_analysis = {
            'regression_detected': True,
            'current_stage': current_metrics.current_stage,
            'regression_severity': 'moderate',
            'likely_causes': [],
            'recommendations': [],
            'recovery_probability': 0.5
        }
        
        if len(self.development_history) >= 2:
            previous_stage = self.development_history[-2][1].current_stage
            all_stages = list(DevelopmentStage)
            stage_drop = all_stages.index(previous_stage) - all_stages.index(current_metrics.current_stage)
            
            # Analyze severity
            if stage_drop >= 3:
                regression_analysis['regression_severity'] = 'severe'
            elif stage_drop >= 2:
                regression_analysis['regression_severity'] = 'moderate'
            else:
                regression_analysis['regression_severity'] = 'mild'
            
            # Identify likely causes
            if current_metrics.phi_value < 0.01:
                regression_analysis['likely_causes'].append('insufficient_phi_value')
            
            if current_metrics.integration_quality < 0.3:
                regression_analysis['likely_causes'].append('poor_integration')
            
            if current_metrics.distinction_count < 2:
                regression_analysis['likely_causes'].append('insufficient_distinctions')
            
            # Recovery recommendations
            if 'insufficient_phi_value' in regression_analysis['likely_causes']:
                regression_analysis['recommendations'].append('increase_experiential_input')
            
            if 'poor_integration' in regression_analysis['likely_causes']:
                regression_analysis['recommendations'].append('strengthen_concept_relationships')
            
            # Calculate recovery probability
            base_recovery = 0.7
            severity_penalty = {'mild': 0.0, 'moderate': 0.2, 'severe': 0.4}
            recovery_prob = base_recovery - severity_penalty[regression_analysis['regression_severity']]
            
            if current_metrics.development_velocity > 0:
                recovery_prob += 0.2
            
            regression_analysis['recovery_probability'] = max(0.1, min(0.9, recovery_prob))
        
        return regression_analysis
    
    def predict_development_trajectory(self, target_stage: DevelopmentStage, 
                                     time_horizon_days: int = 30) -> DevelopmentTrajectory:
        """Predict development trajectory to target stage"""
        
        if not self.development_history:
            return DevelopmentTrajectory(
                start_stage=DevelopmentStage.STAGE_0_PRE_CONSCIOUS,
                current_stage=DevelopmentStage.STAGE_0_PRE_CONSCIOUS,
                target_stage=target_stage,
                trajectory_type="unknown",
                development_rate=0.0,
                estimated_time_to_target=None
            )
        
        current_metrics = self.development_history[-1][1]
        current_stage = current_metrics.current_stage
        
        # Calculate historical development rate
        if len(self.development_history) >= 5:
            start_time = self.development_history[-5][0]
            end_time = self.development_history[-1][0]
            time_span = (end_time - start_time).total_seconds() / (24 * 3600)  # days
            
            all_stages = list(DevelopmentStage)
            start_index = all_stages.index(self.development_history[-5][1].current_stage)
            end_index = all_stages.index(current_stage)
            
            stage_progress = end_index - start_index
            development_rate = stage_progress / time_span if time_span > 0 else 0.0
        else:
            development_rate = 0.0
        
        # Estimate time to target
        all_stages = list(DevelopmentStage)
        current_index = all_stages.index(current_stage)
        target_index = all_stages.index(target_stage)
        stages_to_go = target_index - current_index
        
        estimated_time = None
        if development_rate > 0 and stages_to_go > 0:
            estimated_time = stages_to_go / development_rate
        elif development_rate <= 0 and stages_to_go > 0:
            # Pessimistic estimate based on readiness
            estimated_time = stages_to_go * 30 * (1.0 - current_metrics.next_stage_readiness)
        
        # Determine trajectory type
        if stages_to_go == 0:
            trajectory_type = "achieved"
        elif stages_to_go > 0:
            if development_rate > 0:
                trajectory_type = "progressive"
            else:
                trajectory_type = "challenging"
        else:
            trajectory_type = "regression_needed"
        
        # Build comprehensive trajectory
        trajectory = DevelopmentTrajectory(
            start_stage=self.development_history[0][1].current_stage if self.development_history else current_stage,
            current_stage=current_stage,
            target_stage=target_stage,
            trajectory_type=trajectory_type,
            development_rate=development_rate,
            estimated_time_to_target=estimated_time
        )
        
        # Add historical data
        for timestamp, metrics in self.development_history:
            trajectory.stage_history.append((timestamp, metrics.current_stage, metrics.phi_value))
            trajectory.phi_history.append((timestamp, metrics.phi_value))
        
        # Calculate development momentum and stability
        if len(self.development_history) >= 3:
            recent_velocities = [entry[1].development_velocity for entry in self.development_history[-3:]]
            trajectory.development_momentum = np.mean(recent_velocities)
            trajectory.stability_index = 1.0 - np.std(recent_velocities) / max(np.mean(np.abs(recent_velocities)), 0.001)
        
        return trajectory
    
    def get_development_summary(self) -> Dict[str, Any]:
        """Get comprehensive development summary"""
        
        if not self.development_history:
            return {"status": "no_data"}
        
        current_metrics = self.development_history[-1][1]
        
        summary = {
            "current_status": {
                "stage": current_metrics.current_stage.value,
                "phi_value": current_metrics.phi_value,
                "maturity_score": current_metrics.maturity_score,
                "confidence": current_metrics.stage_confidence
            },
            "development_dynamics": {
                "velocity": current_metrics.development_velocity,
                "regression_risk": current_metrics.regression_risk,
                "next_stage_readiness": current_metrics.next_stage_readiness,
                "trajectory": current_metrics.development_trajectory
            },
            "consciousness_indicators": {
                "temporal_depth": current_metrics.temporal_depth,
                "self_reference": current_metrics.self_reference_strength,
                "narrative_coherence": current_metrics.narrative_coherence,
                "experiential_purity": current_metrics.experiential_purity
            },
            "phi_structure_analysis": {
                "distinctions": current_metrics.distinction_count,
                "relations": current_metrics.relation_count,
                "complexity": current_metrics.phi_structure_complexity,
                "integration_quality": current_metrics.integration_quality
            }
        }
        
        # Add historical trends if sufficient data
        if len(self.development_history) >= 5:
            phi_values = [entry[1].phi_value for entry in self.development_history[-10:]]
            maturity_scores = [entry[1].maturity_score for entry in self.development_history[-10:]]
            
            summary["historical_trends"] = {
                "phi_trend": np.polyfit(range(len(phi_values)), phi_values, 1)[0],
                "maturity_trend": np.polyfit(range(len(maturity_scores)), maturity_scores, 1)[0],
                "stage_changes": len(set(entry[1].current_stage for entry in self.development_history[-10:]))
            }
        
        return summary


# Example usage and testing
async def test_development_stage_mapping():
    """Test development stage mapping functionality"""
    
    print("🧠 Testing IIT 4.0 Development Stage Mapping")
    print("=" * 60)
    
    # Initialize mapper
    mapper = IIT4DevelopmentStageMapper()
    
    # Test with different φ structures
    from iit4_core_engine import IIT4PhiCalculator
    
    calculator = IIT4PhiCalculator()
    
    # Test different system configurations
    test_systems = [
        ("Minimal System", np.array([1, 0]), np.array([[0, 0.3], [0.7, 0]])),
        ("Small Network", np.array([1, 1, 0]), np.array([[0, 0.5, 0.3], [0.4, 0, 0.6], [0.2, 0.8, 0]])),
        ("Complex System", np.array([1, 0, 1, 1]), np.array([[0, 0.3, 0.5, 0.2], [0.6, 0, 0.4, 0.7], [0.1, 0.8, 0, 0.3], [0.9, 0.2, 0.6, 0]]))
    ]
    
    for name, state, connectivity in test_systems:
        print(f"\n🔬 Testing {name}")
        print("-" * 40)
        
        # Calculate φ structure
        phi_structure = calculator.calculate_phi(state, connectivity)
        
        # Map to development stage
        metrics = mapper.map_phi_to_development_stage(phi_structure)
        
        print(f"   φ Value: {metrics.phi_value:.6f}")
        print(f"   Development Stage: {metrics.current_stage.value}")
        print(f"   Stage Confidence: {metrics.stage_confidence:.3f}")
        print(f"   Maturity Score: {metrics.maturity_score:.3f}")
        print(f"   Development Velocity: {metrics.development_velocity:.3f}")
        print(f"   Next Stage Readiness: {metrics.next_stage_readiness:.3f}")
        print(f"   Regression Risk: {metrics.regression_risk:.3f}")
        
        # Test transition detection
        transition_type = mapper.detect_stage_transitions(metrics)
        if transition_type:
            print(f"   Transition Type: {transition_type.value}")
        
        await asyncio.sleep(0.1)  # Small delay for realistic progression
    
    # Test trajectory prediction
    print(f"\n🎯 Development Trajectory Analysis")
    print("-" * 40)
    
    trajectory = mapper.predict_development_trajectory(DevelopmentStage.STAGE_4_SELF_ESTABLISHMENT)
    print(f"   Current Stage: {trajectory.current_stage.value}")
    print(f"   Target Stage: {trajectory.target_stage.value}")
    print(f"   Trajectory Type: {trajectory.trajectory_type}")
    print(f"   Development Rate: {trajectory.development_rate:.3f} stages/day")
    print(f"   Development Momentum: {trajectory.development_momentum:.3f}")
    
    if trajectory.estimated_time_to_target:
        print(f"   Estimated Time to Target: {trajectory.estimated_time_to_target:.1f} days")
    
    # Get development summary
    summary = mapper.get_development_summary()
    print(f"\n📊 Development Summary")
    print(f"   Current Status: {summary.get('current_status', {})}")


if __name__ == "__main__":
    asyncio.run(test_development_stage_mapping())