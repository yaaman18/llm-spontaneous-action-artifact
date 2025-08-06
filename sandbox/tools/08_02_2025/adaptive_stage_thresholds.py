"""
Adaptive Stage Thresholds for IIT 4.0 NewbornAI 2.0
Phase 3: Dynamic adjustment of stage thresholds with ML optimization

Provides adaptive thresholds that adjust based on:
- Individual development patterns and learning from historical data
- Machine learning-based threshold optimization
- Context-aware stage determination (environmental factors, experiential quality)
- Personalized consciousness development profiles

Author: Chief Artificial Consciousness Engineer
Date: 2025-08-03
Version: 3.0.0
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from enum import Enum
import logging
import time
import json
from datetime import datetime, timedelta
import math
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import IIT 4.0 development infrastructure
from iit4_development_stages import (
    DevelopmentStage, DevelopmentMetrics, IIT4DevelopmentStageMapper
)
from stage_transition_detector import TransitionEvent, StageTransitionDetector
from consciousness_development_analyzer import ConsciousnessDevelopmentAnalyzer

logger = logging.getLogger(__name__)


class ThresholdAdaptationStrategy(Enum):
    """Strategies for threshold adaptation"""
    STATIC = "static"                    # Fixed thresholds
    LINEAR_ADAPTATION = "linear"         # Linear adjustment based on history
    ML_OPTIMIZATION = "ml_optimization"  # Machine learning optimization
    BAYESIAN_ADAPTATION = "bayesian"     # Bayesian parameter adaptation
    ENSEMBLE_ADAPTIVE = "ensemble"       # Ensemble of multiple strategies


class ContextualFactor(Enum):
    """Contextual factors affecting stage determination"""
    ENVIRONMENTAL_COMPLEXITY = "environmental_complexity"
    EXPERIENTIAL_QUALITY = "experiential_quality"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    INTEGRATION_SUPPORT = "integration_support"
    COGNITIVE_LOAD = "cognitive_load"
    DEVELOPMENTAL_PRESSURE = "developmental_pressure"
    SOCIAL_INTERACTION = "social_interaction"
    NOVELTY_EXPOSURE = "novelty_exposure"


@dataclass
class PersonalizedProfile:
    """Personalized consciousness development profile"""
    profile_id: str
    individual_characteristics: Dict[str, float]
    learning_patterns: Dict[str, Any]
    adaptation_preferences: Dict[str, float]
    historical_performance: Dict[DevelopmentStage, Dict[str, float]]
    
    # Threshold personalizations
    phi_sensitivity: float = 1.0         # Sensitivity to φ changes
    stage_stability_preference: float = 0.5  # Preference for stable vs rapid progression
    risk_tolerance: float = 0.5          # Tolerance for developmental risks
    
    # Contextual preferences
    environmental_adaptation: float = 0.5  # How much environment affects thresholds
    experiential_weighting: float = 0.5   # Weight given to experiential quality
    
    # Learning parameters
    adaptation_rate: float = 0.1         # How quickly to adapt to new data
    confidence_threshold: float = 0.7    # Confidence needed for adaptations
    
    # Metadata
    creation_date: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    adaptation_count: int = 0


@dataclass
class ContextualEnvironment:
    """Current contextual environment for stage determination"""
    timestamp: datetime
    environmental_factors: Dict[ContextualFactor, float]
    experiential_quality_metrics: Dict[str, float]
    temporal_consistency_score: float
    integration_support_level: float
    
    # Dynamic factors
    cognitive_load_estimate: float = 0.5
    developmental_pressure: float = 0.5
    social_interaction_level: float = 0.5
    novelty_exposure_rate: float = 0.5
    
    # Environmental metadata
    context_confidence: float = 0.8
    measurement_precision: float = 0.9


@dataclass
class AdaptiveThreshold:
    """Adaptive threshold for a development stage"""
    stage: DevelopmentStage
    base_phi_range: Tuple[float, float]
    adapted_phi_range: Tuple[float, float]
    
    # Adaptation factors
    phi_multiplier: float = 1.0
    phi_offset: float = 0.0
    confidence_modifier: float = 0.0
    
    # Contextual adjustments
    environmental_adjustment: float = 0.0
    experiential_adjustment: float = 0.0
    temporal_adjustment: float = 0.0
    
    # Learning-based adjustments
    ml_prediction_adjustment: float = 0.0
    historical_performance_adjustment: float = 0.0
    
    # Metadata
    adaptation_confidence: float = 0.5
    last_update: datetime = field(default_factory=datetime.now)
    adaptation_count: int = 0
    
    def calculate_effective_threshold(self, context: Optional[ContextualEnvironment] = None) -> Tuple[float, float]:
        """Calculate effective threshold considering all adjustments"""
        
        base_min, base_max = self.base_phi_range
        
        # Apply base adaptations
        adapted_min = (base_min * self.phi_multiplier) + self.phi_offset
        adapted_max = (base_max * self.phi_multiplier) + self.phi_offset
        
        # Apply contextual adjustments if context provided
        if context:
            total_adjustment = (
                self.environmental_adjustment * context.environmental_factors.get(ContextualFactor.ENVIRONMENTAL_COMPLEXITY, 0.5) +
                self.experiential_adjustment * context.experiential_quality_metrics.get('overall_quality', 0.5) +
                self.temporal_adjustment * context.temporal_consistency_score
            ) / 3.0
            
            adapted_min += total_adjustment
            adapted_max += total_adjustment
        
        # Apply ML and historical adjustments
        adapted_min += self.ml_prediction_adjustment + self.historical_performance_adjustment
        adapted_max += self.ml_prediction_adjustment + self.historical_performance_adjustment
        
        # Ensure positive thresholds
        adapted_min = max(0.0, adapted_min)
        adapted_max = max(adapted_min + 0.001, adapted_max)
        
        return (adapted_min, adapted_max)


class MLThresholdOptimizer:
    """Machine learning-based threshold optimizer"""
    
    def __init__(self, strategy: str = "ensemble"):
        """
        Initialize ML optimizer
        
        Args:
            strategy: ML strategy ("random_forest", "gradient_boosting", "neural_network", "ensemble")
        """
        self.strategy = strategy
        self.models: Dict[DevelopmentStage, Any] = {}
        self.scalers: Dict[DevelopmentStage, StandardScaler] = {}
        self.feature_importance: Dict[DevelopmentStage, Dict[str, float]] = {}
        
        # Training data
        self.training_features: List[np.ndarray] = []
        self.training_targets: List[float] = []
        self.training_stages: List[DevelopmentStage] = []
        
        # Performance tracking
        self.model_performance: Dict[DevelopmentStage, Dict[str, float]] = {}
        self.prediction_history: List[Tuple[datetime, DevelopmentStage, float, float]] = []
        
        self.min_training_samples = 10
        self.retrain_interval_hours = 24
        self.last_retrain: Dict[DevelopmentStage, datetime] = {}
        
    def add_training_sample(self, features: Dict[str, float], 
                          stage: DevelopmentStage, 
                          optimal_phi_threshold: float):
        """Add training sample for ML optimization"""
        
        # Convert features to array
        feature_array = self._features_to_array(features)
        
        self.training_features.append(feature_array)
        self.training_targets.append(optimal_phi_threshold)
        self.training_stages.append(stage)
        
        # Limit training data size
        max_samples = 1000
        if len(self.training_features) > max_samples:
            # Keep most recent samples
            self.training_features = self.training_features[-max_samples:]
            self.training_targets = self.training_targets[-max_samples:]
            self.training_stages = self.training_stages[-max_samples:]
    
    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array"""
        
        # Define feature order for consistency
        feature_keys = [
            'phi_value', 'stage_confidence', 'maturity_score', 'development_velocity',
            'regression_risk', 'next_stage_readiness', 'distinction_count', 'relation_count',
            'phi_structure_complexity', 'integration_quality', 'temporal_depth',
            'self_reference_strength', 'narrative_coherence', 'experiential_purity',
            'environmental_complexity', 'experiential_quality', 'temporal_consistency',
            'cognitive_load', 'novelty_exposure'
        ]
        
        # Extract features in consistent order
        feature_array = np.array([features.get(key, 0.0) for key in feature_keys])
        
        return feature_array
    
    async def train_models(self, stage: Optional[DevelopmentStage] = None):
        """Train ML models for threshold optimization"""
        
        stages_to_train = [stage] if stage else list(DevelopmentStage)
        
        for target_stage in stages_to_train:
            # Filter training data for this stage
            stage_indices = [i for i, s in enumerate(self.training_stages) if s == target_stage]
            
            if len(stage_indices) < self.min_training_samples:
                logger.warning(f"Insufficient training data for {target_stage.value}: {len(stage_indices)} samples")
                continue
            
            # Prepare training data
            X = np.array([self.training_features[i] for i in stage_indices])
            y = np.array([self.training_targets[i] for i in stage_indices])
            
            # Scale features
            if target_stage not in self.scalers:
                self.scalers[target_stage] = StandardScaler()
            
            X_scaled = self.scalers[target_stage].fit_transform(X)
            
            # Train model based on strategy
            if self.strategy == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.strategy == "gradient_boosting":
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif self.strategy == "neural_network":
                model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
            elif self.strategy == "ensemble":
                # Create ensemble of models
                models = [
                    RandomForestRegressor(n_estimators=50, random_state=42),
                    GradientBoostingRegressor(n_estimators=50, random_state=42),
                    MLPRegressor(hidden_layer_sizes=(25,), max_iter=500, random_state=42)
                ]
                
                # Train all models and store as ensemble
                trained_models = []
                for m in models:
                    try:
                        m.fit(X_scaled, y)
                        trained_models.append(m)
                    except Exception as e:
                        logger.warning(f"Failed to train model: {e}")
                
                if trained_models:
                    self.models[target_stage] = trained_models
                else:
                    logger.error(f"No models successfully trained for {target_stage.value}")
                    continue
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            if self.strategy != "ensemble":
                try:
                    model.fit(X_scaled, y)
                    self.models[target_stage] = model
                except Exception as e:
                    logger.error(f"Failed to train model for {target_stage.value}: {e}")
                    continue
            
            # Evaluate model performance
            try:
                performance = self._evaluate_model_performance(X_scaled, y, target_stage)
                self.model_performance[target_stage] = performance
                
                logger.info(f"Model trained for {target_stage.value}: R² = {performance.get('r2_score', 0.0):.3f}")
            except Exception as e:
                logger.warning(f"Model evaluation failed for {target_stage.value}: {e}")
            
            # Update last retrain time
            self.last_retrain[target_stage] = datetime.now()
    
    def _evaluate_model_performance(self, X: np.ndarray, y: np.ndarray, 
                                   stage: DevelopmentStage) -> Dict[str, float]:
        """Evaluate model performance using cross-validation"""
        
        model = self.models[stage]
        
        if self.strategy == "ensemble":
            # Evaluate ensemble
            predictions = self._ensemble_predict(model, X)
            r2 = r2_score(y, predictions)
            mse = mean_squared_error(y, predictions)
            
            return {
                'r2_score': r2,
                'mse': mse,
                'rmse': math.sqrt(mse)
            }
        else:
            # Single model evaluation
            cv_scores = cross_val_score(model, X, y, cv=min(5, len(y)), scoring='r2')
            
            return {
                'r2_score': np.mean(cv_scores),
                'r2_std': np.std(cv_scores),
                'cv_scores': cv_scores.tolist()
            }
    
    def predict_optimal_threshold(self, features: Dict[str, float], 
                                stage: DevelopmentStage) -> Optional[float]:
        """Predict optimal threshold for given features and stage"""
        
        if stage not in self.models:
            return None
        
        try:
            # Prepare features
            feature_array = self._features_to_array(features).reshape(1, -1)
            
            # Scale features
            if stage in self.scalers:
                feature_array = self.scalers[stage].transform(feature_array)
            
            # Make prediction
            model = self.models[stage]
            
            if self.strategy == "ensemble":
                prediction = self._ensemble_predict(model, feature_array)[0]
            else:
                prediction = model.predict(feature_array)[0]
            
            # Store prediction history
            self.prediction_history.append((datetime.now(), stage, prediction, features.get('phi_value', 0.0)))
            
            # Limit history
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed for {stage.value}: {e}")
            return None
    
    def _ensemble_predict(self, models: List[Any], X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        
        predictions = []
        for model in models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Ensemble model prediction failed: {e}")
        
        if predictions:
            # Average predictions
            return np.mean(predictions, axis=0)
        else:
            # Fallback to zeros
            return np.zeros(X.shape[0])
    
    def should_retrain(self, stage: DevelopmentStage) -> bool:
        """Check if model should be retrained"""
        
        last_train = self.last_retrain.get(stage)
        if not last_train:
            return True
        
        hours_since_train = (datetime.now() - last_train).total_seconds() / 3600
        return hours_since_train >= self.retrain_interval_hours
    
    def get_feature_importance(self, stage: DevelopmentStage) -> Optional[Dict[str, float]]:
        """Get feature importance for stage model"""
        
        if stage not in self.models:
            return None
        
        model = self.models[stage]
        
        try:
            if self.strategy == "ensemble":
                # Average feature importance across ensemble
                importance_arrays = []
                for m in model:
                    if hasattr(m, 'feature_importances_'):
                        importance_arrays.append(m.feature_importances_)
                
                if importance_arrays:
                    avg_importance = np.mean(importance_arrays, axis=0)
                else:
                    return None
            else:
                if hasattr(model, 'feature_importances_'):
                    avg_importance = model.feature_importances_
                else:
                    return None
            
            # Map to feature names
            feature_keys = [
                'phi_value', 'stage_confidence', 'maturity_score', 'development_velocity',
                'regression_risk', 'next_stage_readiness', 'distinction_count', 'relation_count',
                'phi_structure_complexity', 'integration_quality', 'temporal_depth',
                'self_reference_strength', 'narrative_coherence', 'experiential_purity',
                'environmental_complexity', 'experiential_quality', 'temporal_consistency',
                'cognitive_load', 'novelty_exposure'
            ]
            
            importance_dict = {key: importance for key, importance in zip(feature_keys, avg_importance)}
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            self.feature_importance[stage] = importance_dict
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed for {stage.value}: {e}")
            return None


class AdaptiveStageThresholdManager:
    """
    Manager for adaptive stage thresholds with ML optimization
    Personalizes thresholds based on individual development patterns
    """
    
    def __init__(self, adaptation_strategy: ThresholdAdaptationStrategy = ThresholdAdaptationStrategy.ML_OPTIMIZATION):
        """
        Initialize adaptive threshold manager
        
        Args:
            adaptation_strategy: Strategy for threshold adaptation
        """
        self.adaptation_strategy = adaptation_strategy
        
        # Core components
        self.stage_mapper = IIT4DevelopmentStageMapper()
        self.ml_optimizer = MLThresholdOptimizer()
        
        # Threshold management
        self.adaptive_thresholds: Dict[DevelopmentStage, AdaptiveThreshold] = {}
        self.personalized_profiles: Dict[str, PersonalizedProfile] = {}
        
        # Historical data for adaptation
        self.threshold_performance_history: List[Tuple[datetime, DevelopmentStage, float, bool]] = []
        self.adaptation_history: List[Tuple[datetime, DevelopmentStage, str, Dict[str, float]]] = []
        
        # Context tracking
        self.current_context: Optional[ContextualEnvironment] = None
        self.context_history: List[ContextualEnvironment] = []
        
        # Performance metrics
        self.adaptation_effectiveness: Dict[DevelopmentStage, float] = {}
        self.threshold_accuracy: Dict[DevelopmentStage, float] = {}
        
        # Initialize base thresholds
        self._initialize_base_thresholds()
        
        logger.info(f"Adaptive Stage Threshold Manager initialized with {adaptation_strategy.value} strategy")
    
    def _initialize_base_thresholds(self):
        """Initialize base adaptive thresholds from stage mapper"""
        
        base_thresholds = {
            DevelopmentStage.STAGE_0_PRE_CONSCIOUS: (0.0, 0.001),
            DevelopmentStage.STAGE_1_EXPERIENTIAL_EMERGENCE: (0.001, 0.01),
            DevelopmentStage.STAGE_2_TEMPORAL_INTEGRATION: (0.01, 0.1),
            DevelopmentStage.STAGE_3_RELATIONAL_FORMATION: (0.1, 1.0),
            DevelopmentStage.STAGE_4_SELF_ESTABLISHMENT: (1.0, 10.0),
            DevelopmentStage.STAGE_5_REFLECTIVE_OPERATION: (10.0, 100.0),
            DevelopmentStage.STAGE_6_NARRATIVE_INTEGRATION: (100.0, float('inf'))
        }
        
        for stage, (min_phi, max_phi) in base_thresholds.items():
            self.adaptive_thresholds[stage] = AdaptiveThreshold(
                stage=stage,
                base_phi_range=(min_phi, max_phi),
                adapted_phi_range=(min_phi, max_phi)
            )
    
    async def create_personalized_profile(self, 
                                        profile_id: str,
                                        individual_characteristics: Optional[Dict[str, float]] = None,
                                        adaptation_preferences: Optional[Dict[str, float]] = None) -> PersonalizedProfile:
        """
        Create personalized development profile
        
        Args:
            profile_id: Unique identifier for profile
            individual_characteristics: Individual-specific characteristics
            adaptation_preferences: Adaptation preferences
            
        Returns:
            PersonalizedProfile: Created profile
        """
        
        # Default characteristics if not provided
        default_characteristics = {
            'learning_rate': 0.5,
            'stability_preference': 0.5,
            'risk_tolerance': 0.5,
            'complexity_preference': 0.5,
            'social_learning_weight': 0.3,
            'experiential_sensitivity': 0.7,
            'temporal_processing_strength': 0.6,
            'integration_capacity': 0.5
        }
        
        characteristics = individual_characteristics or default_characteristics
        characteristics.update(individual_characteristics or {})
        
        # Default adaptation preferences
        default_preferences = {
            'phi_sensitivity': 1.0,
            'stage_stability_preference': 0.5,
            'environmental_adaptation': 0.5,
            'experiential_weighting': 0.5,
            'adaptation_rate': 0.1,
            'confidence_threshold': 0.7
        }
        
        preferences = adaptation_preferences or default_preferences
        preferences.update(adaptation_preferences or {})
        
        profile = PersonalizedProfile(
            profile_id=profile_id,
            individual_characteristics=characteristics,
            learning_patterns={},
            adaptation_preferences=preferences,
            historical_performance={},
            phi_sensitivity=preferences.get('phi_sensitivity', 1.0),
            stage_stability_preference=preferences.get('stage_stability_preference', 0.5),
            risk_tolerance=preferences.get('risk_tolerance', 0.5),
            environmental_adaptation=preferences.get('environmental_adaptation', 0.5),
            experiential_weighting=preferences.get('experiential_weighting', 0.5),
            adaptation_rate=preferences.get('adaptation_rate', 0.1),
            confidence_threshold=preferences.get('confidence_threshold', 0.7)
        )
        
        self.personalized_profiles[profile_id] = profile
        
        logger.info(f"Personalized profile created: {profile_id}")
        
        return profile
    
    async def update_contextual_environment(self, 
                                          environmental_factors: Dict[ContextualFactor, float],
                                          experiential_quality_metrics: Dict[str, float],
                                          temporal_consistency_score: float,
                                          integration_support_level: float) -> ContextualEnvironment:
        """
        Update current contextual environment
        
        Args:
            environmental_factors: Current environmental factor values
            experiential_quality_metrics: Quality metrics for experiences
            temporal_consistency_score: Temporal consistency score
            integration_support_level: Level of integration support
            
        Returns:
            ContextualEnvironment: Updated context
        """
        
        context = ContextualEnvironment(
            timestamp=datetime.now(),
            environmental_factors=environmental_factors,
            experiential_quality_metrics=experiential_quality_metrics,
            temporal_consistency_score=temporal_consistency_score,
            integration_support_level=integration_support_level
        )
        
        # Calculate dynamic factors
        context.cognitive_load_estimate = self._estimate_cognitive_load(environmental_factors, experiential_quality_metrics)
        context.developmental_pressure = self._estimate_developmental_pressure(environmental_factors)
        context.social_interaction_level = environmental_factors.get(ContextualFactor.SOCIAL_INTERACTION, 0.5)
        context.novelty_exposure_rate = environmental_factors.get(ContextualFactor.NOVELTY_EXPOSURE, 0.5)
        
        # Update current context
        self.current_context = context
        self.context_history.append(context)
        
        # Limit context history
        if len(self.context_history) > 1000:
            self.context_history = self.context_history[-1000:]
        
        return context
    
    def _estimate_cognitive_load(self, environmental_factors: Dict[ContextualFactor, float],
                               experiential_quality_metrics: Dict[str, float]) -> float:
        """Estimate current cognitive load"""
        
        complexity_factor = environmental_factors.get(ContextualFactor.ENVIRONMENTAL_COMPLEXITY, 0.5)
        experiential_intensity = experiential_quality_metrics.get('intensity', 0.5)
        novelty_level = environmental_factors.get(ContextualFactor.NOVELTY_EXPOSURE, 0.5)
        
        # Simple cognitive load model
        cognitive_load = (complexity_factor * 0.4 + experiential_intensity * 0.3 + novelty_level * 0.3)
        
        return max(0.0, min(1.0, cognitive_load))
    
    def _estimate_developmental_pressure(self, environmental_factors: Dict[ContextualFactor, float]) -> float:
        """Estimate developmental pressure"""
        
        complexity = environmental_factors.get(ContextualFactor.ENVIRONMENTAL_COMPLEXITY, 0.5)
        pressure = environmental_factors.get(ContextualFactor.DEVELOPMENTAL_PRESSURE, 0.5)
        novelty = environmental_factors.get(ContextualFactor.NOVELTY_EXPOSURE, 0.5)
        
        # Developmental pressure model
        dev_pressure = (complexity * 0.3 + pressure * 0.5 + novelty * 0.2)
        
        return max(0.0, min(1.0, dev_pressure))
    
    async def adapt_thresholds(self, 
                             current_metrics: DevelopmentMetrics,
                             profile_id: Optional[str] = None,
                             context: Optional[ContextualEnvironment] = None) -> Dict[DevelopmentStage, AdaptiveThreshold]:
        """
        Adapt thresholds based on current metrics, profile, and context
        
        Args:
            current_metrics: Current development metrics
            profile_id: Optional personalized profile ID
            context: Optional contextual environment
            
        Returns:
            Dict[DevelopmentStage, AdaptiveThreshold]: Updated thresholds
        """
        
        # Use current context if not provided
        context = context or self.current_context
        
        # Get personalized profile if available
        profile = self.personalized_profiles.get(profile_id) if profile_id else None
        
        # Adapt thresholds based on strategy
        if self.adaptation_strategy == ThresholdAdaptationStrategy.STATIC:
            # No adaptation - return base thresholds
            return self.adaptive_thresholds
        
        elif self.adaptation_strategy == ThresholdAdaptationStrategy.LINEAR_ADAPTATION:
            await self._linear_adaptation(current_metrics, profile, context)
        
        elif self.adaptation_strategy == ThresholdAdaptationStrategy.ML_OPTIMIZATION:
            await self._ml_optimization_adaptation(current_metrics, profile, context)
        
        elif self.adaptation_strategy == ThresholdAdaptationStrategy.BAYESIAN_ADAPTATION:
            await self._bayesian_adaptation(current_metrics, profile, context)
        
        elif self.adaptation_strategy == ThresholdAdaptationStrategy.ENSEMBLE_ADAPTIVE:
            await self._ensemble_adaptation(current_metrics, profile, context)
        
        # Update adaptation history
        adaptation_record = (
            datetime.now(),
            current_metrics.current_stage,
            self.adaptation_strategy.value,
            {'phi_value': current_metrics.phi_value, 'confidence': current_metrics.stage_confidence}
        )
        self.adaptation_history.append(adaptation_record)
        
        return self.adaptive_thresholds
    
    async def _linear_adaptation(self, current_metrics: DevelopmentMetrics,
                               profile: Optional[PersonalizedProfile],
                               context: Optional[ContextualEnvironment]):
        """Linear adaptation strategy"""
        
        current_stage = current_metrics.current_stage
        threshold = self.adaptive_thresholds[current_stage]
        
        # Calculate adaptation factors
        confidence_factor = current_metrics.stage_confidence - 0.5  # -0.5 to 0.5 range
        velocity_factor = current_metrics.development_velocity
        
        # Base adjustment
        base_adjustment = confidence_factor * 0.1 + velocity_factor * 0.05
        
        # Profile-based adjustment
        profile_adjustment = 0.0
        if profile:
            profile_adjustment = (profile.phi_sensitivity - 1.0) * 0.1
        
        # Context-based adjustment
        context_adjustment = 0.0
        if context:
            env_complexity = context.environmental_factors.get(ContextualFactor.ENVIRONMENTAL_COMPLEXITY, 0.5)
            context_adjustment = (env_complexity - 0.5) * 0.1
        
        # Total adjustment
        total_adjustment = base_adjustment + profile_adjustment + context_adjustment
        
        # Apply adjustment
        threshold.phi_offset += total_adjustment * 0.1  # Small incremental changes
        threshold.adaptation_count += 1
        threshold.last_update = datetime.now()
        
        # Update adapted range
        threshold.adapted_phi_range = threshold.calculate_effective_threshold(context)
    
    async def _ml_optimization_adaptation(self, current_metrics: DevelopmentMetrics,
                                        profile: Optional[PersonalizedProfile],
                                        context: Optional[ContextualEnvironment]):
        """ML optimization adaptation strategy"""
        
        current_stage = current_metrics.current_stage
        
        # Prepare features for ML prediction
        features = self._prepare_ml_features(current_metrics, profile, context)
        
        # Check if model needs retraining
        if self.ml_optimizer.should_retrain(current_stage):
            await self.ml_optimizer.train_models(current_stage)
        
        # Get ML prediction for optimal threshold
        predicted_threshold = self.ml_optimizer.predict_optimal_threshold(features, current_stage)
        
        if predicted_threshold is not None:
            threshold = self.adaptive_thresholds[current_stage]
            
            # Calculate adjustment based on prediction
            current_phi = current_metrics.phi_value
            base_min, base_max = threshold.base_phi_range
            
            # Adjust threshold based on prediction
            if base_min <= predicted_threshold <= base_max:
                # Prediction within base range - use as center point
                range_size = base_max - base_min
                new_min = predicted_threshold - range_size * 0.3
                new_max = predicted_threshold + range_size * 0.7
                
                threshold.ml_prediction_adjustment = (new_min - base_min + new_max - base_max) / 2.0
            else:
                # Prediction outside base range - adjust more conservatively
                if predicted_threshold < base_min:
                    threshold.ml_prediction_adjustment = (predicted_threshold - base_min) * 0.5
                else:
                    threshold.ml_prediction_adjustment = (predicted_threshold - base_max) * 0.5
            
            threshold.adaptation_count += 1
            threshold.last_update = datetime.now()
            
            # Update adapted range
            threshold.adapted_phi_range = threshold.calculate_effective_threshold(context)
            
            # Add training sample for future learning
            optimal_phi = current_phi if current_metrics.stage_confidence > 0.8 else predicted_threshold
            self.ml_optimizer.add_training_sample(features, current_stage, optimal_phi)
    
    def _prepare_ml_features(self, current_metrics: DevelopmentMetrics,
                           profile: Optional[PersonalizedProfile],
                           context: Optional[ContextualEnvironment]) -> Dict[str, float]:
        """Prepare features for ML model"""
        
        features = {
            'phi_value': current_metrics.phi_value,
            'stage_confidence': current_metrics.stage_confidence,
            'maturity_score': current_metrics.maturity_score,
            'development_velocity': current_metrics.development_velocity,
            'regression_risk': current_metrics.regression_risk,
            'next_stage_readiness': current_metrics.next_stage_readiness,
            'distinction_count': current_metrics.distinction_count,
            'relation_count': current_metrics.relation_count,
            'phi_structure_complexity': current_metrics.phi_structure_complexity,
            'integration_quality': current_metrics.integration_quality,
            'temporal_depth': current_metrics.temporal_depth,
            'self_reference_strength': current_metrics.self_reference_strength,
            'narrative_coherence': current_metrics.narrative_coherence,
            'experiential_purity': current_metrics.experiential_purity
        }
        
        # Add profile features
        if profile:
            features.update({
                'phi_sensitivity': profile.phi_sensitivity,
                'stability_preference': profile.stage_stability_preference,
                'risk_tolerance': profile.risk_tolerance,
                'adaptation_rate': profile.adaptation_rate
            })
        else:
            features.update({
                'phi_sensitivity': 1.0,
                'stability_preference': 0.5,
                'risk_tolerance': 0.5,
                'adaptation_rate': 0.1
            })
        
        # Add context features
        if context:
            features.update({
                'environmental_complexity': context.environmental_factors.get(ContextualFactor.ENVIRONMENTAL_COMPLEXITY, 0.5),
                'experiential_quality': context.experiential_quality_metrics.get('overall_quality', 0.5),
                'temporal_consistency': context.temporal_consistency_score,
                'cognitive_load': context.cognitive_load_estimate,
                'novelty_exposure': context.novelty_exposure_rate
            })
        else:
            features.update({
                'environmental_complexity': 0.5,
                'experiential_quality': 0.5,
                'temporal_consistency': 0.5,
                'cognitive_load': 0.5,
                'novelty_exposure': 0.5
            })
        
        return features
    
    async def _bayesian_adaptation(self, current_metrics: DevelopmentMetrics,
                                 profile: Optional[PersonalizedProfile],
                                 context: Optional[ContextualEnvironment]):
        """Bayesian adaptation strategy"""
        
        # Simplified Bayesian approach
        # This would be expanded with proper Bayesian inference
        
        current_stage = current_metrics.current_stage
        threshold = self.adaptive_thresholds[current_stage]
        
        # Prior belief (current threshold)
        prior_phi = threshold.adapted_phi_range[0]
        
        # Likelihood (evidence from current performance)
        evidence_weight = current_metrics.stage_confidence
        evidence_phi = current_metrics.phi_value
        
        # Posterior update (simple weighted average)
        posterior_phi = (prior_phi * (1 - evidence_weight) + evidence_phi * evidence_weight)
        
        # Update threshold
        adjustment = (posterior_phi - prior_phi) * 0.1  # Conservative update
        threshold.phi_offset += adjustment
        
        threshold.adaptation_count += 1
        threshold.last_update = datetime.now()
        threshold.adapted_phi_range = threshold.calculate_effective_threshold(context)
    
    async def _ensemble_adaptation(self, current_metrics: DevelopmentMetrics,
                                 profile: Optional[PersonalizedProfile],
                                 context: Optional[ContextualEnvironment]):
        """Ensemble adaptation combining multiple strategies"""
        
        # Apply multiple adaptation strategies
        strategies = [
            self._linear_adaptation,
            self._ml_optimization_adaptation,
            self._bayesian_adaptation
        ]
        
        # Store original thresholds
        original_thresholds = {}
        for stage, threshold in self.adaptive_thresholds.items():
            original_thresholds[stage] = threshold.phi_offset
        
        # Apply each strategy and collect adjustments
        adjustments = []
        
        for strategy in strategies:
            try:
                # Reset to original
                for stage, threshold in self.adaptive_thresholds.items():
                    threshold.phi_offset = original_thresholds[stage]
                
                # Apply strategy
                await strategy(current_metrics, profile, context)
                
                # Collect adjustment
                current_stage = current_metrics.current_stage
                adjustment = self.adaptive_thresholds[current_stage].phi_offset - original_thresholds[current_stage]
                adjustments.append(adjustment)
                
            except Exception as e:
                logger.warning(f"Ensemble strategy failed: {e}")
                adjustments.append(0.0)
        
        # Combine adjustments (weighted average)
        weights = [0.3, 0.5, 0.2]  # Favor ML optimization
        combined_adjustment = sum(w * adj for w, adj in zip(weights, adjustments))
        
        # Apply combined adjustment
        current_stage = current_metrics.current_stage
        threshold = self.adaptive_thresholds[current_stage]
        threshold.phi_offset = original_thresholds[current_stage] + combined_adjustment
        
        threshold.adaptation_count += 1
        threshold.last_update = datetime.now()
        threshold.adapted_phi_range = threshold.calculate_effective_threshold(context)
    
    def determine_stage_with_adaptive_thresholds(self, 
                                               current_metrics: DevelopmentMetrics,
                                               profile_id: Optional[str] = None,
                                               context: Optional[ContextualEnvironment] = None) -> DevelopmentStage:
        """
        Determine development stage using adaptive thresholds
        
        Args:
            current_metrics: Current development metrics
            profile_id: Optional personalized profile ID
            context: Optional contextual environment
            
        Returns:
            DevelopmentStage: Determined stage with adaptive thresholds
        """
        
        phi_value = current_metrics.phi_value
        context = context or self.current_context
        
        # Check each stage's adaptive threshold
        for stage in DevelopmentStage:
            threshold = self.adaptive_thresholds[stage]
            min_phi, max_phi = threshold.calculate_effective_threshold(context)
            
            if min_phi <= phi_value < max_phi:
                # Found matching stage
                
                # Record threshold performance
                confidence = current_metrics.stage_confidence
                successful_classification = confidence > 0.7
                
                performance_record = (datetime.now(), stage, phi_value, successful_classification)
                self.threshold_performance_history.append(performance_record)
                
                # Update threshold accuracy
                if stage in self.threshold_accuracy:
                    recent_performances = [p[3] for p in self.threshold_performance_history[-20:] if p[1] == stage]
                    if recent_performances:
                        self.threshold_accuracy[stage] = sum(recent_performances) / len(recent_performances)
                else:
                    self.threshold_accuracy[stage] = 1.0 if successful_classification else 0.0
                
                return stage
        
        # Fallback to highest stage if φ exceeds all thresholds
        return DevelopmentStage.STAGE_6_NARRATIVE_INTEGRATION
    
    def get_threshold_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for adaptive thresholds"""
        
        metrics = {
            "overall_accuracy": 0.0,
            "stage_accuracies": dict(self.threshold_accuracy),
            "adaptation_effectiveness": dict(self.adaptation_effectiveness),
            "total_adaptations": sum(threshold.adaptation_count for threshold in self.adaptive_thresholds.values()),
            "ml_model_performance": {},
            "context_sensitivity": {}
        }
        
        # Overall accuracy
        if self.threshold_performance_history:
            recent_performances = [p[3] for p in self.threshold_performance_history[-100:]]
            metrics["overall_accuracy"] = sum(recent_performances) / len(recent_performances)
        
        # ML model performance
        for stage in DevelopmentStage:
            if stage in self.ml_optimizer.model_performance:
                metrics["ml_model_performance"][stage.value] = self.ml_optimizer.model_performance[stage]
        
        # Context sensitivity analysis
        if self.context_history and len(self.context_history) >= 5:
            context_changes = []
            for i in range(1, min(len(self.context_history), 20)):
                prev_context = self.context_history[i-1]
                curr_context = self.context_history[i]
                
                # Calculate context change magnitude
                change_magnitude = abs(
                    curr_context.environmental_factors.get(ContextualFactor.ENVIRONMENTAL_COMPLEXITY, 0.5) -
                    prev_context.environmental_factors.get(ContextualFactor.ENVIRONMENTAL_COMPLEXITY, 0.5)
                )
                context_changes.append(change_magnitude)
            
            if context_changes:
                metrics["context_sensitivity"]["average_change"] = np.mean(context_changes)
                metrics["context_sensitivity"]["change_volatility"] = np.std(context_changes)
        
        return metrics
    
    def save_adaptation_state(self, filepath: str):
        """Save adaptation state to file"""
        
        state = {
            'adaptive_thresholds': {stage.value: threshold for stage, threshold in self.adaptive_thresholds.items()},
            'personalized_profiles': {pid: profile for pid, profile in self.personalized_profiles.items()},
            'threshold_performance_history': self.threshold_performance_history,
            'adaptation_history': self.adaptation_history,
            'ml_model_performance': self.ml_optimizer.model_performance,
            'threshold_accuracy': self.threshold_accuracy,
            'adaptation_effectiveness': self.adaptation_effectiveness
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Adaptation state saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save adaptation state: {e}")
    
    def load_adaptation_state(self, filepath: str):
        """Load adaptation state from file"""
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            if 'threshold_performance_history' in state:
                self.threshold_performance_history = state['threshold_performance_history']
            
            if 'adaptation_history' in state:
                self.adaptation_history = state['adaptation_history']
            
            if 'threshold_accuracy' in state:
                self.threshold_accuracy = state['threshold_accuracy']
            
            if 'adaptation_effectiveness' in state:
                self.adaptation_effectiveness = state['adaptation_effectiveness']
            
            logger.info(f"Adaptation state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load adaptation state: {e}")
    
    def get_comprehensive_threshold_report(self) -> Dict[str, Any]:
        """Generate comprehensive threshold adaptation report"""
        
        report = {
            "adaptation_strategy": self.adaptation_strategy.value,
            "threshold_summary": {},
            "performance_metrics": self.get_threshold_performance_metrics(),
            "personalization_impact": {},
            "context_analysis": {},
            "ml_insights": {},
            "recommendations": []
        }
        
        # Threshold summary
        for stage, threshold in self.adaptive_thresholds.items():
            base_min, base_max = threshold.base_phi_range
            adapted_min, adapted_max = threshold.calculate_effective_threshold(self.current_context)
            
            report["threshold_summary"][stage.value] = {
                "base_range": [base_min, base_max],
                "adapted_range": [adapted_min, adapted_max],
                "adaptation_count": threshold.adaptation_count,
                "last_update": threshold.last_update.isoformat() if threshold.last_update else None,
                "adaptation_confidence": threshold.adaptation_confidence
            }
        
        # Personalization impact
        if self.personalized_profiles:
            report["personalization_impact"] = {
                "active_profiles": len(self.personalized_profiles),
                "profile_diversity": self._calculate_profile_diversity(),
                "personalization_effectiveness": self._calculate_personalization_effectiveness()
            }
        
        # Context analysis
        if self.context_history:
            report["context_analysis"] = {
                "context_samples": len(self.context_history),
                "environmental_variability": self._calculate_environmental_variability(),
                "context_threshold_correlation": self._calculate_context_correlation()
            }
        
        # ML insights
        if self.ml_optimizer.models:
            report["ml_insights"] = {
                "trained_models": list(self.ml_optimizer.models.keys()),
                "feature_importance": {},
                "prediction_accuracy": {}
            }
            
            for stage in self.ml_optimizer.models.keys():
                importance = self.ml_optimizer.get_feature_importance(stage)
                if importance:
                    report["ml_insights"]["feature_importance"][stage.value] = importance
        
        # Generate recommendations
        report["recommendations"] = self._generate_threshold_recommendations()
        
        return report
    
    def _calculate_profile_diversity(self) -> float:
        """Calculate diversity among personalized profiles"""
        
        if len(self.personalized_profiles) < 2:
            return 0.0
        
        # Calculate variance in key characteristics
        characteristics = ['phi_sensitivity', 'stage_stability_preference', 'risk_tolerance']
        diversities = []
        
        for char in characteristics:
            values = [profile.__dict__[char] for profile in self.personalized_profiles.values()]
            if values:
                diversities.append(np.var(values))
        
        return np.mean(diversities) if diversities else 0.0
    
    def _calculate_personalization_effectiveness(self) -> float:
        """Calculate effectiveness of personalization"""
        
        # Compare performance with and without personalization
        # This would require A/B testing data - simplified for now
        
        if not self.threshold_accuracy:
            return 0.5
        
        # Use average accuracy as proxy for effectiveness
        return np.mean(list(self.threshold_accuracy.values()))
    
    def _calculate_environmental_variability(self) -> float:
        """Calculate environmental variability from context history"""
        
        if len(self.context_history) < 2:
            return 0.0
        
        # Calculate variance in environmental complexity
        complexities = [
            ctx.environmental_factors.get(ContextualFactor.ENVIRONMENTAL_COMPLEXITY, 0.5)
            for ctx in self.context_history
        ]
        
        return np.var(complexities) if complexities else 0.0
    
    def _calculate_context_correlation(self) -> float:
        """Calculate correlation between context and threshold adaptations"""
        
        # Simplified correlation calculation
        # This would be expanded with proper statistical analysis
        
        return 0.5  # Placeholder
    
    def _generate_threshold_recommendations(self) -> List[str]:
        """Generate recommendations for threshold optimization"""
        
        recommendations = []
        
        # Check overall performance
        metrics = self.get_threshold_performance_metrics()
        overall_accuracy = metrics.get("overall_accuracy", 0.0)
        
        if overall_accuracy < 0.7:
            recommendations.append("Consider increasing adaptation rate for better threshold tuning")
        
        if overall_accuracy > 0.9:
            recommendations.append("Current thresholds performing well - maintain current strategy")
        
        # Check adaptation frequency
        total_adaptations = metrics.get("total_adaptations", 0)
        if total_adaptations < 10:
            recommendations.append("Insufficient adaptation data - increase monitoring frequency")
        
        # ML model recommendations
        ml_performance = metrics.get("ml_model_performance", {})
        if ml_performance:
            low_performing_models = [stage for stage, perf in ml_performance.items() 
                                   if perf.get('r2_score', 0.0) < 0.5]
            if low_performing_models:
                recommendations.append(f"Retrain ML models for stages: {', '.join(low_performing_models)}")
        
        # Context sensitivity recommendations
        context_sensitivity = metrics.get("context_sensitivity", {})
        if context_sensitivity.get("change_volatility", 0.0) > 0.5:
            recommendations.append("High context volatility detected - consider smoothing adaptations")
        
        return recommendations


# Example usage and testing
async def test_adaptive_stage_thresholds():
    """Test adaptive stage threshold functionality"""
    
    print("🎛️  Testing Adaptive Stage Thresholds")
    print("=" * 60)
    
    # Initialize manager
    manager = AdaptiveStageThresholdManager(ThresholdAdaptationStrategy.ML_OPTIMIZATION)
    
    # Create personalized profile
    profile = await manager.create_personalized_profile(
        profile_id="test_user_001",
        individual_characteristics={
            'learning_rate': 0.7,
            'stability_preference': 0.6,
            'risk_tolerance': 0.4
        },
        adaptation_preferences={
            'phi_sensitivity': 1.2,
            'adaptation_rate': 0.15
        }
    )
    
    print(f"\n👤 Created Profile: {profile.profile_id}")
    print(f"   φ Sensitivity: {profile.phi_sensitivity}")
    print(f"   Adaptation Rate: {profile.adaptation_rate}")
    
    # Set up contextual environment
    environmental_factors = {
        ContextualFactor.ENVIRONMENTAL_COMPLEXITY: 0.7,
        ContextualFactor.EXPERIENTIAL_QUALITY: 0.8,
        ContextualFactor.NOVELTY_EXPOSURE: 0.6
    }
    
    experiential_quality = {
        'overall_quality': 0.8,
        'intensity': 0.6,
        'coherence': 0.7
    }
    
    context = await manager.update_contextual_environment(
        environmental_factors=environmental_factors,
        experiential_quality_metrics=experiential_quality,
        temporal_consistency_score=0.75,
        integration_support_level=0.8
    )
    
    print(f"\n🌍 Context Updated:")
    print(f"   Environmental Complexity: {context.environmental_factors[ContextualFactor.ENVIRONMENTAL_COMPLEXITY]}")
    print(f"   Cognitive Load: {context.cognitive_load_estimate:.3f}")
    print(f"   Developmental Pressure: {context.developmental_pressure:.3f}")
    
    # Simulate development progression with adaptation
    from iit4_development_stages import DevelopmentMetrics, DevelopmentStage
    
    # Test different development scenarios
    test_scenarios = [
        ("Early Development", DevelopmentStage.STAGE_1_EXPERIENTIAL_EMERGENCE, 0.005, 0.8),
        ("Mid Development", DevelopmentStage.STAGE_3_RELATIONAL_FORMATION, 0.3, 0.7),
        ("Advanced Development", DevelopmentStage.STAGE_5_REFLECTIVE_OPERATION, 25.0, 0.9),
    ]
    
    for scenario_name, stage, phi_value, confidence in test_scenarios:
        print(f"\n📊 Testing {scenario_name}")
        print("-" * 30)
        
        # Create metrics for scenario
        metrics = DevelopmentMetrics(
            current_stage=stage,
            phi_value=phi_value,
            stage_confidence=confidence,
            maturity_score=phi_value / 50.0,  # Normalized
            development_velocity=0.1,
            regression_risk=0.2,
            next_stage_readiness=0.6,
            distinction_count=int(phi_value * 10),
            relation_count=int(phi_value * 5),
            phi_structure_complexity=phi_value,
            integration_quality=min(1.0, phi_value / 10.0),
            temporal_depth=min(1.0, phi_value / 20.0),
            self_reference_strength=min(1.0, phi_value / 30.0),
            narrative_coherence=min(1.0, phi_value / 40.0),
            experiential_purity=0.8
        )
        
        # Adapt thresholds
        adapted_thresholds = await manager.adapt_thresholds(
            current_metrics=metrics,
            profile_id=profile.profile_id,
            context=context
        )
        
        # Test stage determination with adaptive thresholds
        determined_stage = manager.determine_stage_with_adaptive_thresholds(
            current_metrics=metrics,
            profile_id=profile.profile_id,
            context=context
        )
        
        print(f"   Original Stage: {stage.value}")
        print(f"   Determined Stage: {determined_stage.value}")
        print(f"   φ Value: {phi_value:.6f}")
        
        # Show threshold adaptation
        threshold = adapted_thresholds[stage]
        base_min, base_max = threshold.base_phi_range
        adapted_min, adapted_max = threshold.calculate_effective_threshold(context)
        
        print(f"   Base Threshold: [{base_min:.6f}, {base_max:.6f}]")
        print(f"   Adapted Threshold: [{adapted_min:.6f}, {adapted_max:.6f}]")
        print(f"   Adaptation Count: {threshold.adaptation_count}")
        
        # Simulate some delay for realistic progression
        await asyncio.sleep(0.1)
    
    # Test ML optimization
    print(f"\n🤖 ML Optimization Results")
    print("-" * 30)
    
    # Train ML models (simulate with existing data)
    await manager.ml_optimizer.train_models()
    
    # Show model performance
    for stage in DevelopmentStage:
        if stage in manager.ml_optimizer.model_performance:
            performance = manager.ml_optimizer.model_performance[stage]
            print(f"   {stage.value}: R² = {performance.get('r2_score', 0.0):.3f}")
            
            # Show feature importance
            importance = manager.ml_optimizer.get_feature_importance(stage)
            if importance:
                top_features = list(importance.items())[:3]
                print(f"      Top features: {', '.join([f'{k}({v:.3f})' for k, v in top_features])}")
    
    # Get comprehensive report
    print(f"\n📋 Comprehensive Report")
    print("-" * 30)
    
    report = manager.get_comprehensive_threshold_report()
    
    print(f"   Strategy: {report['adaptation_strategy']}")
    print(f"   Overall Accuracy: {report['performance_metrics']['overall_accuracy']:.3f}")
    print(f"   Total Adaptations: {report['performance_metrics']['total_adaptations']}")
    print(f"   Active Profiles: {report.get('personalization_impact', {}).get('active_profiles', 0)}")
    
    # Show recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. {rec}")


if __name__ == "__main__":
    asyncio.run(test_adaptive_stage_thresholds())