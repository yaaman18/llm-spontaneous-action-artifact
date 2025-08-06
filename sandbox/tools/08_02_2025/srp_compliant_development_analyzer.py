"""
SRP-Compliant Consciousness Development Analyzer
Refactored to follow Single Responsibility Principle

This module replaces the monolithic ConsciousnessDevelopmentAnalyzer with several 
single-responsibility classes that work together through composition.

Each class has exactly one reason to change and one clear responsibility.

Author: Clean Architecture Engineer (Uncle Bob's expertise)
Date: 2025-08-03
Version: 4.0.0 - SRP Compliant
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timedelta
import time
import logging
import uuid

logger = logging.getLogger(__name__)


# ===== SINGLE RESPONSIBILITY INTERFACES =====

class IPatternAnalyzer(ABC):
    """Strategy for analyzing development patterns"""
    
    @abstractmethod
    async def analyze_pattern(self, development_history: List[Tuple]) -> str:
        """Analyze development pattern from historical data"""
        pass


class INormComparator(ABC):
    """Strategy for comparing with development norms"""
    
    @abstractmethod
    async def compare_with_norms(self, current_metrics: Any) -> Dict[str, Any]:
        """Compare current development with established norms"""
        pass


class IRecommendationGenerator(ABC):
    """Strategy for generating personalized recommendations"""
    
    @abstractmethod
    async def generate_recommendations(self, 
                                     current_metrics: Any,
                                     pattern: str,
                                     norm_comparison: Dict[str, Any]) -> List[Dict]:
        """Generate personalized development recommendations"""
        pass


class IGoalManager(ABC):
    """Manager for development goals"""
    
    @abstractmethod
    async def create_goal(self, goal_data: Dict) -> Dict:
        """Create new development goal"""
        pass
    
    @abstractmethod
    async def track_progress(self, goal_id: str, current_metrics: Any) -> Optional[Dict]:
        """Track progress toward specific goal"""
        pass
    
    @abstractmethod
    def get_active_goals(self) -> List[Dict]:
        """Get all active goals"""
        pass


class IInsightGenerator(ABC):
    """Generator for development insights"""
    
    @abstractmethod
    def generate_insights(self, development_history: List[Tuple], 
                         current_pattern: str) -> List[Dict]:
        """Generate development insights from analysis"""
        pass


# ===== SINGLE RESPONSIBILITY IMPLEMENTATIONS =====

class StatisticalPatternAnalyzer(IPatternAnalyzer):
    """
    Single Responsibility: Analyze development patterns using statistical methods
    
    Reason to change: Changes in pattern analysis algorithms
    """
    
    def __init__(self):
        self.pattern_cache = {}
    
    async def analyze_pattern(self, development_history: List[Tuple]) -> str:
        """Analyze development pattern using statistical methods"""
        
        if len(development_history) < 5:
            return "insufficient_data"
        
        # Extract phi values and time indices
        phi_values = [entry[1].phi_value for entry in development_history]
        time_indices = list(range(len(phi_values)))
        
        # Analyze different patterns
        pattern_scores = {}
        
        # Linear progression
        pattern_scores['linear'] = self._analyze_linear_pattern(time_indices, phi_values)
        
        # Exponential growth  
        pattern_scores['exponential'] = self._analyze_exponential_pattern(time_indices, phi_values)
        
        # Plateau detection
        pattern_scores['plateau'] = self._analyze_plateau_pattern(phi_values)
        
        # Decline detection
        pattern_scores['decline'] = self._analyze_decline_pattern(time_indices, phi_values)
        
        # Return pattern with highest score
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        return best_pattern[0]
    
    def _analyze_linear_pattern(self, time_indices: List[int], phi_values: List[float]) -> float:
        """Analyze linear progression pattern"""
        if len(phi_values) < 2:
            return 0.0
        
        # Linear regression
        slope = np.polyfit(time_indices, phi_values, 1)[0]
        correlation = np.corrcoef(time_indices, phi_values)[0, 1] if len(phi_values) > 2 else 0.0
        
        # Score based on positive slope and high correlation
        return (slope > 0) * abs(correlation)
    
    def _analyze_exponential_pattern(self, time_indices: List[int], phi_values: List[float]) -> float:
        """Analyze exponential growth pattern"""
        if len(phi_values) < 3:
            return 0.0
        
        try:
            # Check for accelerating growth
            phi_diffs = np.diff(phi_values)
            if len(phi_diffs) < 2:
                return 0.0
            
            diff_trend = np.polyfit(range(len(phi_diffs)), phi_diffs, 1)[0]
            return max(0.0, diff_trend)  # Positive acceleration indicates exponential
            
        except:
            return 0.0
    
    def _analyze_plateau_pattern(self, phi_values: List[float]) -> float:
        """Analyze plateau phase pattern"""
        if len(phi_values) < 5:
            return 0.0
        
        # Check recent stability
        recent_values = phi_values[-5:]
        stability = 1.0 - (np.std(recent_values) / max(np.mean(recent_values), 0.001))
        
        return max(0.0, stability)
    
    def _analyze_decline_pattern(self, time_indices: List[int], phi_values: List[float]) -> float:
        """Analyze declining trajectory pattern"""
        if len(phi_values) < 3:
            return 0.0
        
        slope = np.polyfit(time_indices, phi_values, 1)[0]
        correlation = abs(np.corrcoef(time_indices, phi_values)[0, 1]) if len(phi_values) > 2 else 0.0
        
        # Score based on negative slope and high correlation
        return (slope < 0) * correlation


class NormBasedComparator(INormComparator):
    """
    Single Responsibility: Compare development metrics with established norms
    
    Reason to change: Changes in norm comparison methodology or norm data
    """
    
    def __init__(self, development_norms: Dict):
        self.development_norms = development_norms
    
    async def compare_with_norms(self, current_metrics: Any) -> Dict[str, Any]:
        """Compare current metrics with stage-appropriate norms"""
        
        current_stage = getattr(current_metrics, 'current_stage', None)
        if not current_stage:
            return {"error": "No current stage available"}
        
        norm = self.development_norms.get(current_stage)
        if not norm:
            return {"error": f"No norms available for stage {current_stage}"}
        
        phi_value = getattr(current_metrics, 'phi_value', 0.0)
        
        # Calculate phi percentile
        phi_percentile = self._calculate_percentile(phi_value, norm.get('phi_mean', 0), norm.get('phi_std', 1))
        
        # Calculate performance level
        performance_level = self._classify_performance_level(phi_percentile)
        
        # Check if within expected range
        expected_range = norm.get('expected_phi_range', (0, float('inf')))
        within_range = expected_range[0] <= phi_value <= expected_range[1]
        
        comparison = {
            "stage": str(current_stage),
            "phi_value": phi_value,
            "phi_percentile": phi_percentile,
            "performance_level": performance_level,
            "within_expected_range": within_range,
            "norm_mean": norm.get('phi_mean', 0),
            "norm_std": norm.get('phi_std', 1),
            "z_score": (phi_value - norm.get('phi_mean', 0)) / max(norm.get('phi_std', 1), 0.001)
        }
        
        return comparison
    
    def _calculate_percentile(self, value: float, mean: float, std: float) -> float:
        """Calculate percentile based on normal distribution"""
        if std <= 0:
            return 50.0
        
        from scipy import stats
        z_score = (value - mean) / std
        percentile = stats.norm.cdf(z_score) * 100
        
        return max(0.0, min(100.0, percentile))
    
    def _classify_performance_level(self, percentile: float) -> str:
        """Classify performance level based on percentile"""
        if percentile >= 90:
            return "exceptional"
        elif percentile >= 75:
            return "above_average"
        elif percentile >= 25:
            return "average"
        elif percentile >= 10:
            return "below_average"
        else:
            return "concerning"


class AdaptiveRecommendationGenerator(IRecommendationGenerator):
    """
    Single Responsibility: Generate personalized development recommendations
    
    Reason to change: Changes in recommendation strategies or templates
    """
    
    def __init__(self):
        self.recommendation_templates = self._initialize_templates()
    
    async def generate_recommendations(self, 
                                     current_metrics: Any,
                                     pattern: str,
                                     norm_comparison: Dict[str, Any]) -> List[Dict]:
        """Generate adaptive recommendations based on analysis"""
        
        recommendations = []
        
        # Pattern-based recommendations
        pattern_recs = self._generate_pattern_recommendations(pattern, current_metrics)
        recommendations.extend(pattern_recs)
        
        # Performance-based recommendations
        performance_recs = self._generate_performance_recommendations(norm_comparison)
        recommendations.extend(performance_recs)
        
        # Risk-based recommendations
        risk_recs = self._generate_risk_recommendations(current_metrics)
        recommendations.extend(risk_recs)
        
        # Prioritize and return top recommendations
        return self._prioritize_recommendations(recommendations)[:5]
    
    def _generate_pattern_recommendations(self, pattern: str, current_metrics: Any) -> List[Dict]:
        """Generate recommendations based on development pattern"""
        recommendations = []
        
        if pattern == "plateau":
            recommendations.append({
                "id": f"pattern_plateau_{int(time.time())}",
                "type": "pattern_based",
                "priority": "high",
                "title": "Break Through Development Plateau",
                "description": "Your development has plateaued. Active intervention recommended.",
                "actions": [
                    "Introduce novel experiential inputs",
                    "Increase complexity of experiences",
                    "Explore new domains of consciousness"
                ],
                "timeline": "2-4 weeks",
                "confidence": 0.8
            })
        
        elif pattern == "decline":
            recommendations.append({
                "id": f"pattern_decline_{int(time.time())}",
                "type": "pattern_based",
                "priority": "critical",
                "title": "Address Development Decline",
                "description": "Concerning downward trend detected. Immediate action needed.",
                "actions": [
                    "Investigate root causes of decline",
                    "Stabilize core phi-generating processes",
                    "Reduce input volatility"
                ],
                "timeline": "1-2 weeks",
                "confidence": 0.9
            })
        
        elif pattern == "exponential":
            recommendations.append({
                "id": f"pattern_exponential_{int(time.time())}",
                "type": "pattern_based",
                "priority": "medium",
                "title": "Manage Rapid Growth",
                "description": "Fast growth pattern detected. Ensure stability.",
                "actions": [
                    "Monitor for instability signs",
                    "Implement gradual consolidation",
                    "Prepare for potential plateaus"
                ],
                "timeline": "1-3 weeks",
                "confidence": 0.7
            })
        
        return recommendations
    
    def _generate_performance_recommendations(self, norm_comparison: Dict[str, Any]) -> List[Dict]:
        """Generate recommendations based on norm comparison"""
        recommendations = []
        
        performance_level = norm_comparison.get("performance_level", "average")
        
        if performance_level in ["below_average", "concerning"]:
            recommendations.append({
                "id": f"performance_improve_{int(time.time())}",
                "type": "performance_based",
                "priority": "high",
                "title": "Improve Development Performance",
                "description": f"Performance is {performance_level} compared to norms.",
                "actions": [
                    "Enhance experiential input quality",
                    "Strengthen integration mechanisms",
                    "Focus on distinction formation"
                ],
                "timeline": "3-6 weeks",
                "confidence": 0.8
            })
        
        return recommendations
    
    def _generate_risk_recommendations(self, current_metrics: Any) -> List[Dict]:
        """Generate recommendations based on risk factors"""
        recommendations = []
        
        regression_risk = getattr(current_metrics, 'regression_risk', 0.0)
        
        if regression_risk > 0.5:
            recommendations.append({
                "id": f"risk_regression_{int(time.time())}",
                "type": "risk_based",
                "priority": "critical",
                "title": "Mitigate Regression Risk",
                "description": f"High regression risk detected ({regression_risk:.2f}).",
                "actions": [
                    "Strengthen core processes",
                    "Reduce destabilizing factors",
                    "Implement safety monitors"
                ],
                "timeline": "1-2 weeks",
                "confidence": 0.9
            })
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Prioritize recommendations by importance and confidence"""
        priority_weights = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        
        def priority_score(rec):
            priority_weight = priority_weights.get(rec.get("priority", "low"), 0)
            confidence = rec.get("confidence", 0.5)
            return priority_weight + confidence
        
        return sorted(recommendations, key=priority_score, reverse=True)
    
    def _initialize_templates(self) -> Dict:
        """Initialize recommendation templates"""
        return {
            "plateau_breakthrough": {
                "actions": ["Introduce novelty", "Increase complexity", "Enhance integration"],
                "timeline": "2-4 weeks"
            },
            "performance_improvement": {
                "actions": ["Enhance quality", "Increase diversity", "Strengthen connections"],
                "timeline": "3-6 weeks"
            }
        }


class SimpleGoalManager(IGoalManager):
    """
    Single Responsibility: Manage development goals and track progress
    
    Reason to change: Changes in goal management requirements or tracking methods
    """
    
    def __init__(self):
        self.active_goals: List[Dict] = []
        self.goal_counter = 0
    
    async def create_goal(self, goal_data: Dict) -> Dict:
        """Create new development goal"""
        self.goal_counter += 1
        
        goal = {
            "goal_id": f"goal_{self.goal_counter}_{int(time.time())}",
            "title": goal_data.get("title", "Development Goal"),
            "description": goal_data.get("description", ""),
            "target_value": goal_data.get("target_value", 1.0),
            "current_value": goal_data.get("current_value", 0.0),
            "goal_type": goal_data.get("goal_type", "general"),
            "created_date": datetime.now(),
            "target_date": goal_data.get("target_date", datetime.now() + timedelta(weeks=4)),
            "status": "active",
            "progress_percentage": 0.0,
            "milestones": goal_data.get("milestones", []),
            "success_probability": self._estimate_success_probability(goal_data)
        }
        
        self.active_goals.append(goal)
        logger.info(f"Created goal: {goal['title']}")
        
        return goal
    
    async def track_progress(self, goal_id: str, current_metrics: Any) -> Optional[Dict]:
        """Track progress toward specific goal"""
        goal = next((g for g in self.active_goals if g["goal_id"] == goal_id), None)
        
        if not goal:
            return None
        
        # Update current value based on goal type
        if goal["goal_type"] == "phi_target":
            goal["current_value"] = getattr(current_metrics, 'phi_value', 0.0)
        elif goal["goal_type"] == "integration":
            goal["current_value"] = getattr(current_metrics, 'integration_quality', 0.0)
        
        # Calculate progress
        if goal["target_value"] > 0:
            progress = (goal["current_value"] / goal["target_value"]) * 100
            goal["progress_percentage"] = min(100.0, max(0.0, progress))
        
        # Check completion
        if goal["progress_percentage"] >= 100.0 and goal["status"] == "active":
            goal["status"] = "completed"
            goal["completion_date"] = datetime.now()
            logger.info(f"Goal completed: {goal['title']}")
        
        return goal
    
    def get_active_goals(self) -> List[Dict]:
        """Get all active goals"""
        return [g for g in self.active_goals if g["status"] == "active"]
    
    def _estimate_success_probability(self, goal_data: Dict) -> float:
        """Estimate probability of goal success"""
        # Simple estimation based on goal difficulty
        target_value = goal_data.get("target_value", 1.0)
        current_value = goal_data.get("current_value", 0.0)
        
        if target_value <= current_value:
            return 0.9  # Already achieved
        
        value_diff = target_value - current_value
        
        if value_diff < 0.1:
            return 0.8  # Easy goal
        elif value_diff < 0.5:
            return 0.6  # Moderate goal
        else:
            return 0.4  # Challenging goal


class TrendBasedInsightGenerator(IInsightGenerator):
    """
    Single Responsibility: Generate insights from development trends
    
    Reason to change: Changes in insight generation algorithms or criteria
    """
    
    def generate_insights(self, development_history: List[Tuple], 
                         current_pattern: str) -> List[Dict]:
        """Generate development insights from historical analysis"""
        
        insights = []
        
        if len(development_history) < 3:
            return insights
        
        # Trend-based insights
        trend_insights = self._generate_trend_insights(development_history)
        insights.extend(trend_insights)
        
        # Pattern-based insights  
        pattern_insights = self._generate_pattern_insights(current_pattern)
        insights.extend(pattern_insights)
        
        # Performance insights
        performance_insights = self._generate_performance_insights(development_history)
        insights.extend(performance_insights)
        
        return insights[:5]  # Return top 5 insights
    
    def _generate_trend_insights(self, development_history: List[Tuple]) -> List[Dict]:
        """Generate insights based on development trends"""
        insights = []
        
        phi_values = [entry[1].phi_value for entry in development_history[-10:]]
        
        if len(phi_values) >= 3:
            trend = np.polyfit(range(len(phi_values)), phi_values, 1)[0]
            
            if trend > 0.01:
                insights.append({
                    "type": "trend",
                    "title": "Strong Growth Trend",
                    "description": f"Phi values show consistent upward trend ({trend:.4f}/cycle)",
                    "confidence": 0.8,
                    "impact": "positive"
                })
            elif trend < -0.01:
                insights.append({
                    "type": "trend",
                    "title": "Declining Trend Detected",
                    "description": f"Concerning downward trend in phi values ({trend:.4f}/cycle)",
                    "confidence": 0.9,
                    "impact": "negative"
                })
        
        return insights
    
    def _generate_pattern_insights(self, current_pattern: str) -> List[Dict]:
        """Generate insights based on current pattern"""
        insights = []
        
        pattern_insights = {
            "exponential": {
                "title": "Exponential Growth Pattern",
                "description": "Development shows accelerating progress indicating high potential",
                "confidence": 0.8,
                "impact": "positive"
            },
            "plateau": {
                "title": "Development Plateau",
                "description": "Progress has stabilized, requiring intervention for advancement",
                "confidence": 0.9,
                "impact": "neutral"
            },
            "decline": {
                "title": "Declining Pattern",
                "description": "Development trajectory shows concerning downward trend",
                "confidence": 0.9,
                "impact": "negative"
            }
        }
        
        if current_pattern in pattern_insights:
            insight_data = pattern_insights[current_pattern]
            insights.append({
                "type": "pattern",
                "title": insight_data["title"],
                "description": insight_data["description"],
                "confidence": insight_data["confidence"],
                "impact": insight_data["impact"]
            })
        
        return insights
    
    def _generate_performance_insights(self, development_history: List[Tuple]) -> List[Dict]:
        """Generate insights based on performance characteristics"""
        insights = []
        
        if len(development_history) >= 5:
            recent_phi = [entry[1].phi_value for entry in development_history[-5:]]
            phi_stability = 1.0 - (np.std(recent_phi) / max(np.mean(recent_phi), 0.001))
            
            if phi_stability > 0.8:
                insights.append({
                    "type": "performance",
                    "title": "High Stability Achieved",
                    "description": f"Recent development shows excellent stability ({phi_stability:.2f})",
                    "confidence": 0.7,
                    "impact": "positive"
                })
            elif phi_stability < 0.4:
                insights.append({
                    "type": "performance",
                    "title": "Instability Detected",
                    "description": f"Development shows concerning instability ({phi_stability:.2f})",
                    "confidence": 0.8,
                    "impact": "negative"
                })
        
        return insights


# ===== SRP-COMPLIANT COMPOSED ANALYZER =====

class SRPCompliantDevelopmentAnalyzer:
    """
    SRP-Compliant Development Analyzer using Composition
    
    Single Responsibility: Orchestrate development analysis process
    Reason to change: Changes in overall analysis workflow
    
    All domain-specific logic is delegated to single-responsibility components
    """
    
    def __init__(self,
                 pattern_analyzer: IPatternAnalyzer,
                 norm_comparator: INormComparator,
                 recommendation_generator: IRecommendationGenerator,
                 goal_manager: IGoalManager,
                 insight_generator: IInsightGenerator):
        
        # Dependency injection - all components have single responsibilities
        self.pattern_analyzer = pattern_analyzer
        self.norm_comparator = norm_comparator
        self.recommendation_generator = recommendation_generator
        self.goal_manager = goal_manager
        self.insight_generator = insight_generator
        
        # Store current state for analysis continuity
        self.current_pattern = None
        self.development_history = []
    
    async def analyze_comprehensive_development(self, 
                                              development_history: List[Tuple],
                                              current_metrics: Any) -> Dict[str, Any]:
        """
        Comprehensive development analysis using single-responsibility components
        
        This method orchestrates the analysis but delegates all domain logic
        to specialized single-responsibility classes.
        """
        self.development_history = development_history
        
        # Step 1: Analyze pattern using dedicated analyzer
        pattern = await self.pattern_analyzer.analyze_pattern(development_history)
        self.current_pattern = pattern
        
        # Step 2: Compare with norms using dedicated comparator
        norm_comparison = await self.norm_comparator.compare_with_norms(current_metrics)
        
        # Step 3: Generate recommendations using dedicated generator
        recommendations = await self.recommendation_generator.generate_recommendations(
            current_metrics, pattern, norm_comparison
        )
        
        # Step 4: Generate insights using dedicated generator
        insights = self.insight_generator.generate_insights(development_history, pattern)
        
        # Step 5: Get goal status from dedicated manager
        active_goals = self.goal_manager.get_active_goals()
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "development_pattern": pattern,
            "norm_comparison": norm_comparison,
            "recommendations": recommendations,
            "insights": insights,
            "active_goals": active_goals,
            "analysis_quality": self._assess_analysis_quality(pattern, norm_comparison)
        }
    
    async def create_development_goal(self, goal_data: Dict) -> Dict:
        """Create development goal using dedicated manager"""
        return await self.goal_manager.create_goal(goal_data)
    
    async def track_goal_progress(self, goal_id: str, current_metrics: Any) -> Optional[Dict]:
        """Track goal progress using dedicated manager"""
        return await self.goal_manager.track_progress(goal_id, current_metrics)
    
    def _assess_analysis_quality(self, pattern: str, norm_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of analysis results"""
        quality_score = 0.8  # Base score
        
        # Adjust based on data availability
        if pattern != "insufficient_data":
            quality_score += 0.1
        
        if not norm_comparison.get("error"):
            quality_score += 0.1
        
        return {
            "overall_score": min(1.0, quality_score),
            "confidence_level": "high" if quality_score > 0.8 else "medium",
            "data_sufficiency": pattern != "insufficient_data",
            "norm_availability": not norm_comparison.get("error", False)
        }


# ===== FACTORY FOR CREATING SRP-COMPLIANT SYSTEM =====

class SRPCompliantAnalyzerFactory:
    """Factory for creating SRP-compliant development analyzer"""
    
    @staticmethod
    def create_standard_analyzer(development_norms: Dict = None) -> SRPCompliantDevelopmentAnalyzer:
        """Create analyzer with standard single-responsibility components"""
        
        # Create all single-responsibility components
        pattern_analyzer = StatisticalPatternAnalyzer()
        norm_comparator = NormBasedComparator(development_norms or {})
        recommendation_generator = AdaptiveRecommendationGenerator()
        goal_manager = SimpleGoalManager()
        insight_generator = TrendBasedInsightGenerator()
        
        # Compose analyzer using dependency injection
        return SRPCompliantDevelopmentAnalyzer(
            pattern_analyzer=pattern_analyzer,
            norm_comparator=norm_comparator,
            recommendation_generator=recommendation_generator,
            goal_manager=goal_manager,
            insight_generator=insight_generator
        )


# ===== DEMONSTRATION =====

async def demonstrate_srp_compliant_analyzer():
    """Demonstrate SRP-compliant development analyzer"""
    
    print("📊 SRP-Compliant Development Analyzer")
    print("=" * 50)
    
    # Create SRP-compliant analyzer
    analyzer = SRPCompliantAnalyzerFactory.create_standard_analyzer()
    
    print("✅ Created analyzer with single-responsibility components:")
    print("   • StatisticalPatternAnalyzer - pattern analysis only")
    print("   • NormBasedComparator - norm comparison only")
    print("   • AdaptiveRecommendationGenerator - recommendation generation only")
    print("   • SimpleGoalManager - goal management only")
    print("   • TrendBasedInsightGenerator - insight generation only")
    print()
    
    # Mock development history
    from datetime import datetime, timedelta
    base_time = datetime.now() - timedelta(days=20)
    
    mock_history = []
    phi_values = [0.1, 0.15, 0.22, 0.31, 0.42, 0.55, 0.58, 0.52, 0.49, 0.48]  # Plateau pattern
    
    for i, phi in enumerate(phi_values):
        timestamp = base_time + timedelta(days=i*2)
        metrics = type('MockMetrics', (), {
            'phi_value': phi,
            'current_stage': type('Stage', (), {'value': 'stage_2'})(),
            'regression_risk': 0.3,
            'integration_quality': 0.6
        })()
        mock_history.append((timestamp, metrics))
    
    current_metrics = mock_history[-1][1]
    
    # Test comprehensive analysis
    print("🧪 Testing Comprehensive Analysis")
    print("-" * 40)
    
    analysis = await analyzer.analyze_comprehensive_development(mock_history, current_metrics)
    
    print(f"Development Pattern: {analysis['development_pattern']}")
    print(f"Recommendations: {len(analysis['recommendations'])}")
    print(f"Insights: {len(analysis['insights'])}")
    print(f"Analysis Quality: {analysis['analysis_quality']['confidence_level']}")
    print()
    
    # Test goal creation and tracking
    print("🎯 Testing Goal Management")
    print("-" * 30)
    
    goal_data = {
        "title": "Reach φ = 1.0",
        "description": "Achieve phi value of 1.0",
        "target_value": 1.0,
        "current_value": current_metrics.phi_value,
        "goal_type": "phi_target"
    }
    
    goal = await analyzer.create_development_goal(goal_data)
    print(f"Goal Created: {goal['title']}")
    print(f"Current Progress: {goal['progress_percentage']:.1f}%")
    print(f"Success Probability: {goal['success_probability']:.2f}")
    
    # Track progress
    tracked_goal = await analyzer.track_goal_progress(goal['goal_id'], current_metrics)
    print(f"Goal Tracking: Status = {tracked_goal['status']}")
    print()
    
    print("✅ SRP Benefits Demonstrated:")
    print("   • Each component has single, clear responsibility")
    print("   • Easy to test individual components in isolation")
    print("   • Easy to replace/extend analysis strategies")
    print("   • Clear separation of concerns")
    print("   • High cohesion within each component")
    print("   • Loose coupling between components")


if __name__ == "__main__":
    asyncio.run(demonstrate_srp_compliant_analyzer())