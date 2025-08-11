"""
Predictive Coding Inference Service.

Application service for making predictions and analyzing the hierarchical
predictive coding system. Provides high-level interfaces for inference,
analysis, and introspection following the Application Service pattern.
"""

from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import numpy.typing as npt

from domain.entities.predictive_coding_core import PredictiveCodingCore
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.precision_weights import PrecisionWeights
from domain.value_objects.probability_distribution import ProbabilityDistribution
from domain.events.domain_events import DomainEvent


@dataclass
class InferenceResult:
    """Result of predictive coding inference."""
    predictions: List[npt.NDArray]
    prediction_errors: List[float]
    precision_weights: PrecisionWeights
    prediction_state: PredictionState
    confidence: float
    free_energy_estimate: Optional[float] = None
    hierarchical_representations: Optional[List[npt.NDArray]] = None
    temporal_consistency: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AnalysisResult:
    """Result of predictive coding system analysis."""
    system_state: Dict[str, Any]
    hierarchical_metrics: Dict[int, Dict[str, float]]
    precision_analysis: Dict[str, float]
    convergence_analysis: Dict[str, Any]
    attention_analysis: Dict[str, float]
    temporal_dynamics: Optional[Dict[str, Any]] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class PredictiveCodingInferenceService:
    """
    Application service for predictive coding inference and analysis.
    
    Provides high-level interfaces for:
    - Making predictions on new data
    - Analyzing system state and performance  
    - Computing confidence and uncertainty measures
    - Extracting hierarchical representations
    - Monitoring attention and precision dynamics
    """
    
    def __init__(self, predictive_coding_core: PredictiveCodingCore):
        """
        Initialize inference service.
        
        Args:
            predictive_coding_core: Trained predictive coding system
        """
        self._core = predictive_coding_core
        self._inference_history: List[InferenceResult] = []
        
    def predict(
        self,
        input_data: npt.NDArray,
        precision_weights: Optional[PrecisionWeights] = None,
        return_uncertainty: bool = True,
        return_representations: bool = False
    ) -> InferenceResult:
        """
        Make predictions on input data.
        
        Args:
            input_data: Input data for prediction
            precision_weights: Precision weights (defaults to uniform)
            return_uncertainty: Whether to compute uncertainty estimates
            return_representations: Whether to return hierarchical representations
            
        Returns:
            InferenceResult with predictions and analysis
        """
        # Use uniform precision weights if none provided
        if precision_weights is None:
            precision_weights = PrecisionWeights.create_uniform(
                self._core.hierarchy_levels
            )
        
        # Generate predictions
        predictions = self._core.generate_predictions(input_data, precision_weights)
        
        # Create targets and compute errors
        targets = self._core._create_targets_from_input(input_data, predictions)
        errors = self._core.compute_prediction_errors(predictions, targets)
        
        # Propagate errors and get state
        propagated_errors, prediction_state = self._core.propagate_errors(
            errors, precision_weights
        )
        
        # Compute confidence
        confidence = self._compute_prediction_confidence(
            prediction_state, precision_weights
        )
        
        # Optional: free energy estimate
        free_energy_estimate = None
        if hasattr(self._core, 'get_free_energy_estimate'):
            free_energy_estimate = self._core.get_free_energy_estimate()
        
        # Optional: hierarchical representations
        hierarchical_representations = None
        if return_representations and hasattr(self._core, 'get_hierarchical_representations'):
            hierarchical_representations = self._core.get_hierarchical_representations()
        
        # Optional: temporal consistency
        temporal_consistency = None
        if len(self._inference_history) > 0:
            temporal_consistency = self._compute_temporal_consistency(prediction_state)
        
        # Create result
        result = InferenceResult(
            predictions=predictions,
            prediction_errors=[float(np.mean(np.abs(err))) for err in errors],
            precision_weights=precision_weights,
            prediction_state=prediction_state,
            confidence=confidence,
            free_energy_estimate=free_energy_estimate,
            hierarchical_representations=hierarchical_representations,
            temporal_consistency=temporal_consistency
        )
        
        # Store in history
        self._inference_history.append(result)
        
        return result
    
    def predict_batch(
        self,
        input_batch: List[npt.NDArray],
        precision_weights_batch: Optional[List[PrecisionWeights]] = None,
        aggregate_results: bool = True
    ) -> Union[List[InferenceResult], InferenceResult]:
        """
        Make predictions on batch of input data.
        
        Args:
            input_batch: List of input data arrays
            precision_weights_batch: List of precision weights (optional)
            aggregate_results: Whether to aggregate results into single result
            
        Returns:
            List of InferenceResults or aggregated InferenceResult
        """
        results = []
        
        for i, input_data in enumerate(input_batch):
            precision_weights = (
                precision_weights_batch[i] if precision_weights_batch 
                else None
            )
            
            result = self.predict(
                input_data, 
                precision_weights, 
                return_uncertainty=True,
                return_representations=False  # Don't return for batch to save memory
            )
            results.append(result)
        
        if aggregate_results:
            return self._aggregate_inference_results(results)
        else:
            return results
    
    def analyze_system_state(self) -> AnalysisResult:
        """
        Analyze current state of the predictive coding system.
        
        Returns:
            AnalysisResult with comprehensive system analysis
        """
        # System state
        system_state = {
            "hierarchy_levels": self._core.hierarchy_levels,
            "input_dimensions": self._core.input_dimensions,
            "current_state_available": self._core.current_state is not None,
            "total_inferences": len(self._inference_history)
        }
        
        if self._core.current_state:
            system_state.update({
                "current_total_error": self._core.current_state.total_error,
                "current_convergence_status": self._core.current_state.convergence_status,
                "current_learning_iteration": self._core.current_state.learning_iteration,
                "prediction_quality": self._core.current_state.prediction_quality
            })
        
        # Hierarchical metrics
        hierarchical_metrics = self._analyze_hierarchical_performance()
        
        # Precision analysis
        precision_analysis = self._analyze_precision_dynamics()
        
        # Convergence analysis
        convergence_analysis = self._analyze_convergence_patterns()
        
        # Attention analysis
        attention_analysis = self._analyze_attention_patterns()
        
        # Temporal dynamics (if available)
        temporal_dynamics = None
        if len(self._inference_history) > 5:
            temporal_dynamics = self._analyze_temporal_dynamics()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            system_state, hierarchical_metrics, precision_analysis, convergence_analysis
        )
        
        return AnalysisResult(
            system_state=system_state,
            hierarchical_metrics=hierarchical_metrics,
            precision_analysis=precision_analysis,
            convergence_analysis=convergence_analysis,
            attention_analysis=attention_analysis,
            temporal_dynamics=temporal_dynamics,
            recommendations=recommendations
        )
    
    def _compute_prediction_confidence(
        self, 
        prediction_state: PredictionState,
        precision_weights: PrecisionWeights
    ) -> float:
        """
        Compute confidence in predictions.
        
        Confidence is based on:
        1. Prediction error magnitude (lower = higher confidence)
        2. Precision weights entropy (focused attention = higher confidence)
        3. Convergence status
        4. Historical performance
        """
        # Error-based confidence (inverse of normalized error)
        max_reasonable_error = 10.0
        error_confidence = 1.0 - min(
            prediction_state.total_error / max_reasonable_error, 1.0
        )
        
        # Attention-based confidence (focused attention = higher confidence)
        attention_confidence = precision_weights.attention_focus
        
        # Convergence-based confidence
        convergence_confidence = 1.0 if prediction_state.is_converged else 0.5
        
        # Historical confidence (if available)
        historical_confidence = 1.0
        if len(self._inference_history) > 5:
            recent_errors = [
                result.prediction_state.total_error 
                for result in self._inference_history[-5:]
            ]
            error_stability = 1.0 / (1.0 + np.var(recent_errors))
            historical_confidence = min(error_stability, 1.0)
        
        # Weighted combination
        confidence = (
            0.4 * error_confidence +
            0.3 * attention_confidence +
            0.2 * convergence_confidence +
            0.1 * historical_confidence
        )
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _compute_temporal_consistency(
        self, 
        current_state: PredictionState
    ) -> float:
        """Compute temporal consistency of predictions."""
        if len(self._inference_history) < 2:
            return 1.0
        
        # Compare with previous prediction state
        previous_state = self._inference_history[-1].prediction_state
        
        # Compute error trajectory similarity
        current_errors = np.array(current_state.hierarchical_errors)
        previous_errors = np.array(previous_state.hierarchical_errors)
        
        # Cosine similarity between error patterns
        dot_product = np.dot(current_errors, previous_errors)
        norms = np.linalg.norm(current_errors) * np.linalg.norm(previous_errors)
        
        if norms > 0:
            similarity = dot_product / norms
            consistency = (similarity + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        else:
            consistency = 1.0
        
        return float(consistency)
    
    def _aggregate_inference_results(
        self, 
        results: List[InferenceResult]
    ) -> InferenceResult:
        """Aggregate multiple inference results."""
        if not results:
            raise ValueError("Cannot aggregate empty results list")
        
        # Average predictions
        all_predictions = [result.predictions for result in results]
        avg_predictions = []
        for level in range(len(all_predictions[0])):
            level_predictions = [preds[level] for preds in all_predictions]
            avg_pred = np.mean(level_predictions, axis=0)
            avg_predictions.append(avg_pred)
        
        # Average errors
        avg_errors = []
        for level in range(len(results[0].prediction_errors)):
            level_errors = [result.prediction_errors[level] for result in results]
            avg_errors.append(np.mean(level_errors))
        
        # Use first result's precision weights (could be improved)
        precision_weights = results[0].precision_weights
        
        # Create aggregated prediction state
        hierarchical_errors = [
            np.mean([result.prediction_state.hierarchical_errors[i] for result in results])
            for i in range(len(results[0].prediction_state.hierarchical_errors))
        ]
        
        aggregated_state = PredictionState(
            hierarchical_errors=hierarchical_errors,
            convergence_status="aggregated",
            learning_iteration=0,
            metadata={"batch_size": len(results)}
        )
        
        # Average confidence
        avg_confidence = np.mean([result.confidence for result in results])
        
        return InferenceResult(
            predictions=avg_predictions,
            prediction_errors=avg_errors,
            precision_weights=precision_weights,
            prediction_state=aggregated_state,
            confidence=avg_confidence
        )
    
    def _analyze_hierarchical_performance(self) -> Dict[int, Dict[str, float]]:
        """Analyze performance at each hierarchical level."""
        metrics = {}
        
        if not self._core.current_state:
            return metrics
        
        for level in range(self._core.hierarchy_levels):
            level_error = self._core.current_state.get_error_at_level(level)
            
            # Recent error history for this level
            recent_level_errors = []
            for result in self._inference_history[-10:]:
                if level < len(result.prediction_errors):
                    recent_level_errors.append(result.prediction_errors[level])
            
            level_metrics = {
                "current_error": level_error,
                "mean_recent_error": np.mean(recent_level_errors) if recent_level_errors else level_error,
                "error_stability": 1.0 / (1.0 + np.var(recent_level_errors)) if len(recent_level_errors) > 1 else 1.0,
                "error_trend": self._compute_error_trend(recent_level_errors),
                "relative_importance": level_error / (self._core.current_state.total_error + 1e-10)
            }
            
            metrics[level] = level_metrics
        
        return metrics
    
    def _analyze_precision_dynamics(self) -> Dict[str, float]:
        """Analyze precision weight dynamics."""
        precision_analysis = {}
        
        if hasattr(self._core, 'get_precision_estimates'):
            precision_estimates = self._core.get_precision_estimates()
            
            if precision_estimates:
                precisions = list(precision_estimates.values())
                precision_analysis.update({
                    "mean_precision": float(np.mean(precisions)),
                    "precision_variance": float(np.var(precisions)),
                    "max_precision": float(np.max(precisions)),
                    "min_precision": float(np.min(precisions)),
                    "precision_range": float(np.max(precisions) - np.min(precisions))
                })
        
        # Analyze precision weight evolution over inference history
        if len(self._inference_history) > 5:
            recent_attention_focus = [
                result.precision_weights.attention_focus 
                for result in self._inference_history[-10:]
            ]
            
            precision_analysis.update({
                "attention_focus_mean": float(np.mean(recent_attention_focus)),
                "attention_focus_stability": 1.0 / (1.0 + np.var(recent_attention_focus)),
                "attention_focus_trend": self._compute_error_trend(recent_attention_focus)
            })
        
        return precision_analysis
    
    def _analyze_convergence_patterns(self) -> Dict[str, Any]:
        """Analyze convergence patterns and stability."""
        convergence_analysis = {}
        
        if not self._inference_history:
            return convergence_analysis
        
        # Error trajectory analysis
        error_trajectory = [
            result.prediction_state.total_error 
            for result in self._inference_history
        ]
        
        convergence_analysis.update({
            "error_trajectory_length": len(error_trajectory),
            "initial_error": error_trajectory[0] if error_trajectory else 0.0,
            "current_error": error_trajectory[-1] if error_trajectory else 0.0,
            "error_reduction": (
                error_trajectory[0] - error_trajectory[-1] 
                if len(error_trajectory) > 1 else 0.0
            ),
            "convergence_rate": self._compute_error_trend(error_trajectory),
            "error_oscillations": self._count_error_oscillations(error_trajectory),
            "is_converging": error_trajectory[-1] < error_trajectory[0] if len(error_trajectory) > 1 else False
        })
        
        # Convergence quality
        if len(error_trajectory) > 10:
            recent_errors = error_trajectory[-10:]
            convergence_analysis.update({
                "recent_error_mean": float(np.mean(recent_errors)),
                "recent_error_variance": float(np.var(recent_errors)),
                "convergence_quality": 1.0 / (1.0 + np.var(recent_errors))
            })
        
        return convergence_analysis
    
    def _analyze_attention_patterns(self) -> Dict[str, float]:
        """Analyze attention and precision weight patterns."""
        if not self._inference_history:
            return {}
        
        # Collect attention patterns
        attention_focus_history = [
            result.precision_weights.attention_focus 
            for result in self._inference_history
        ]
        
        dominant_levels = [
            result.precision_weights.dominant_level 
            for result in self._inference_history
        ]
        
        return {
            "mean_attention_focus": float(np.mean(attention_focus_history)),
            "attention_focus_variance": float(np.var(attention_focus_history)),
            "attention_stability": 1.0 / (1.0 + np.var(attention_focus_history)),
            "most_frequent_dominant_level": float(max(set(dominant_levels), key=dominant_levels.count)),
            "attention_switching_frequency": self._compute_switching_frequency(dominant_levels)
        }
    
    def _analyze_temporal_dynamics(self) -> Dict[str, Any]:
        """Analyze temporal dynamics of the system."""
        if len(self._inference_history) < 5:
            return {}
        
        # Temporal consistency analysis
        temporal_consistencies = [
            result.temporal_consistency 
            for result in self._inference_history 
            if result.temporal_consistency is not None
        ]
        
        # Error autocorrelation
        error_trajectory = [
            result.prediction_state.total_error 
            for result in self._inference_history
        ]
        
        autocorrelation = self._compute_autocorrelation(error_trajectory)
        
        return {
            "mean_temporal_consistency": float(np.mean(temporal_consistencies)) if temporal_consistencies else 0.0,
            "temporal_stability": 1.0 / (1.0 + np.var(temporal_consistencies)) if temporal_consistencies else 0.0,
            "error_autocorrelation": autocorrelation,
            "temporal_memory": self._estimate_temporal_memory(error_trajectory)
        }
    
    def _generate_recommendations(
        self,
        system_state: Dict[str, Any],
        hierarchical_metrics: Dict[int, Dict[str, float]],
        precision_analysis: Dict[str, float],
        convergence_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate system recommendations based on analysis."""
        recommendations = []
        
        # High error recommendations
        if system_state.get("current_total_error", 0) > 1.0:
            recommendations.append(
                "システムエラーが高い - 学習率の調整を検討してください"
            )
        
        # Convergence recommendations
        if convergence_analysis.get("convergence_rate", 0) < 0.01:
            recommendations.append(
                "収束が遅い - 精度重みの調整または学習戦略の見直しを推奨"
            )
        
        # Attention recommendations
        if precision_analysis.get("attention_focus_mean", 0) < 0.3:
            recommendations.append(
                "注意の集中度が低い - より集中した精度重みの使用を推奨"
            )
        
        # Hierarchical performance recommendations
        for level, metrics in hierarchical_metrics.items():
            if metrics.get("error_stability", 0) < 0.5:
                recommendations.append(
                    f"レベル{level}のエラーが不安定 - この階層の調整が必要"
                )
        
        # Temporal recommendations
        if convergence_analysis.get("error_oscillations", 0) > 5:
            recommendations.append(
                "エラーの振動が多い - 学習率を下げることを推奨"
            )
        
        return recommendations
    
    # Utility methods
    
    def _compute_error_trend(self, values: List[float]) -> float:
        """Compute trend in a series of values (positive = increasing)."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)
    
    def _count_error_oscillations(self, values: List[float]) -> int:
        """Count number of oscillations in error trajectory."""
        if len(values) < 3:
            return 0
        
        oscillations = 0
        for i in range(1, len(values) - 1):
            if (values[i] > values[i-1] and values[i] > values[i+1]) or \
               (values[i] < values[i-1] and values[i] < values[i+1]):
                oscillations += 1
        
        return oscillations
    
    def _compute_switching_frequency(self, sequence: List[int]) -> float:
        """Compute frequency of switches in a sequence."""
        if len(sequence) < 2:
            return 0.0
        
        switches = sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i-1])
        return switches / (len(sequence) - 1)
    
    def _compute_autocorrelation(self, values: List[float], lag: int = 1) -> float:
        """Compute autocorrelation of a time series."""
        if len(values) < lag + 1:
            return 0.0
        
        values_array = np.array(values)
        mean_val = np.mean(values_array)
        
        numerator = np.sum(
            (values_array[:-lag] - mean_val) * (values_array[lag:] - mean_val)
        )
        denominator = np.sum((values_array - mean_val) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _estimate_temporal_memory(self, values: List[float]) -> float:
        """Estimate how much the system remembers past states."""
        # Simple estimate based on how long autocorrelations remain significant
        memory_estimate = 0.0
        
        for lag in range(1, min(10, len(values) // 2)):
            autocorr = self._compute_autocorrelation(values, lag)
            if abs(autocorr) > 0.1:  # Threshold for significance
                memory_estimate = lag
            else:
                break
        
        return memory_estimate
    
    # Public interface methods
    
    def get_inference_history(self) -> List[InferenceResult]:
        """Get complete inference history."""
        return self._inference_history.copy()
    
    def clear_inference_history(self) -> None:
        """Clear inference history."""
        self._inference_history.clear()
    
    def get_latest_inference(self) -> Optional[InferenceResult]:
        """Get most recent inference result."""
        return self._inference_history[-1] if self._inference_history else None
    
    def compute_prediction_uncertainty(
        self,
        input_data: npt.NDArray,
        num_samples: int = 10,
        precision_noise_scale: float = 0.1
    ) -> Dict[str, float]:
        """
        Compute prediction uncertainty using Monte Carlo sampling.
        
        Args:
            input_data: Input data for uncertainty estimation
            num_samples: Number of Monte Carlo samples
            precision_noise_scale: Noise scale for precision weights
            
        Returns:
            Dictionary with uncertainty estimates
        """
        predictions_samples = []
        
        # Base precision weights
        base_precision = PrecisionWeights.create_uniform(self._core.hierarchy_levels)
        
        for _ in range(num_samples):
            # Add noise to precision weights
            noisy_weights = base_precision.weights + np.random.normal(
                0, precision_noise_scale, size=base_precision.weights.shape
            )
            noisy_weights = np.clip(noisy_weights, 0.01, None)  # Ensure positive
            
            noisy_precision = PrecisionWeights(
                weights=noisy_weights,
                normalization_method=base_precision.normalization_method
            )
            
            # Make prediction with noisy precision
            result = self.predict(input_data, noisy_precision, return_uncertainty=False)
            predictions_samples.append(result.predictions)
        
        # Compute uncertainty statistics
        prediction_means = []
        prediction_stds = []
        
        for level in range(len(predictions_samples[0])):
            level_predictions = np.array([sample[level] for sample in predictions_samples])
            prediction_means.append(np.mean(level_predictions, axis=0))
            prediction_stds.append(np.std(level_predictions, axis=0))
        
        # Overall uncertainty measures
        total_epistemic_uncertainty = np.mean([np.mean(std) for std in prediction_stds])
        max_uncertainty_level = np.argmax([np.mean(std) for std in prediction_stds])
        
        return {
            "total_epistemic_uncertainty": float(total_epistemic_uncertainty),
            "max_uncertainty_level": int(max_uncertainty_level),
            "level_uncertainties": [float(np.mean(std)) for std in prediction_stds],
            "uncertainty_distribution": [std.tolist() for std in prediction_stds]
        }