"""
Bayesian Inference Domain Service.

Abstract service interface for Bayesian inference and uncertainty
quantification in the enactive consciousness system. Handles belief
updates, posterior computations, and uncertainty propagation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy.typing as npt
from ..value_objects.probability_distribution import ProbabilityDistribution
from ..value_objects.prediction_state import PredictionState
from ..value_objects.precision_weights import PrecisionWeights


class BayesianInferenceService(ABC):
    """
    Abstract domain service for Bayesian inference operations.
    
    This service encapsulates complex Bayesian reasoning operations
    that span multiple domain objects. It implements the Service pattern
    to handle domain logic that doesn't naturally fit in entities
    or value objects.
    
    Key responsibilities:
    - Prior and posterior distribution management
    - Belief updating with new evidence
    - Uncertainty quantification and propagation
    - Model evidence computation
    - Hyperparameter inference
    """

    @abstractmethod
    def update_beliefs(
        self,
        prior_distribution: ProbabilityDistribution,
        evidence: npt.NDArray,
        likelihood_params: Dict[str, Any]
    ) -> ProbabilityDistribution:
        """
        Update beliefs using Bayes' rule given new evidence.
        
        Implements P(θ|D) ∝ P(D|θ) * P(θ) where:
        - θ: parameters/hypotheses
        - D: observed data/evidence
        - P(θ): prior beliefs
        - P(D|θ): likelihood function
        - P(θ|D): posterior beliefs
        
        Args:
            prior_distribution: Prior belief distribution P(θ)
            evidence: Observed evidence/data D
            likelihood_params: Parameters defining the likelihood function
            
        Returns:
            Updated posterior distribution P(θ|D)
            
        Raises:
            ValueError: If evidence or parameters are invalid
            ComputationError: If Bayesian update fails
        """
        pass

    @abstractmethod
    def compute_model_evidence(
        self,
        data: npt.NDArray,
        model_params: Dict[str, Any],
        prior_distribution: ProbabilityDistribution
    ) -> float:
        """
        Compute marginal likelihood (model evidence) P(D|M).
        
        The model evidence represents how well a model explains
        the observed data, integrated over all possible parameter values.
        
        Args:
            data: Observed data
            model_params: Model configuration parameters
            prior_distribution: Prior distribution over model parameters
            
        Returns:
            Log marginal likelihood P(D|M)
            
        Raises:
            ComputationError: If evidence computation fails
        """
        pass

    @abstractmethod
    def compute_posterior_predictive(
        self,
        posterior_distribution: ProbabilityDistribution,
        prediction_contexts: List[npt.NDArray]
    ) -> List[ProbabilityDistribution]:
        """
        Compute posterior predictive distributions.
        
        For each prediction context, computes P(y*|D) by integrating
        over the posterior distribution of parameters.
        
        Args:
            posterior_distribution: Posterior over parameters P(θ|D)
            prediction_contexts: Input contexts for predictions
            
        Returns:
            List of posterior predictive distributions P(y*|x*, D)
            
        Raises:
            ValueError: If contexts are invalid
            ComputationError: If prediction computation fails
        """
        pass

    @abstractmethod
    def estimate_uncertainty(
        self,
        prediction_state: PredictionState,
        model_confidence: float
    ) -> Dict[str, float]:
        """
        Estimate different types of uncertainty in predictions.
        
        Distinguishes between:
        - Aleatoric (data) uncertainty: irreducible noise in data
        - Epistemic (model) uncertainty: reducible uncertainty due to limited data
        - Total uncertainty: combination of both types
        
        Args:
            prediction_state: Current prediction state
            model_confidence: Confidence in the model structure
            
        Returns:
            Dictionary with uncertainty estimates:
            - aleatoric_uncertainty: Data-driven uncertainty
            - epistemic_uncertainty: Model parameter uncertainty  
            - total_uncertainty: Combined uncertainty
            - confidence_interval: Credible interval bounds
            
        Raises:
            ValueError: If prediction state is invalid
        """
        pass

    @abstractmethod
    def optimize_precision_weights(
        self,
        prediction_errors: List[float],
        current_weights: PrecisionWeights,
        adaptation_rate: float
    ) -> PrecisionWeights:
        """
        Optimize precision weights using Bayesian optimization.
        
        Updates precision weights (inverse variances) based on
        prediction error patterns to implement attention mechanisms
        and adaptive error scaling.
        
        Args:
            prediction_errors: Recent prediction errors across levels
            current_weights: Current precision weight configuration
            adaptation_rate: Rate of weight adaptation
            
        Returns:
            Optimized precision weights
            
        Raises:
            ValueError: If inputs are inconsistent
            OptimizationError: If optimization fails to converge
        """
        pass

    @abstractmethod
    def perform_model_selection(
        self,
        candidate_models: List[Dict[str, Any]],
        data: npt.NDArray,
        selection_criterion: str = "bayes_factor"
    ) -> Tuple[int, Dict[str, float]]:
        """
        Perform Bayesian model selection among candidate models.
        
        Compares models using Bayesian criteria such as Bayes factors,
        information criteria, or cross-validation estimates.
        
        Args:
            candidate_models: List of model configurations
            data: Data for model comparison
            selection_criterion: Selection method ("bayes_factor", "aic", "bic", "waic")
            
        Returns:
            Tuple of (best_model_index, selection_scores)
            
        Raises:
            ValueError: If models or criterion are invalid
            ComputationError: If model comparison fails
        """
        pass

    @abstractmethod
    def estimate_hyperparameters(
        self,
        data: npt.NDArray,
        model_structure: Dict[str, Any],
        prior_hyperparams: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Estimate hyperparameters using empirical Bayes or full Bayes.
        
        Learns hyperparameters of prior distributions from data,
        implementing hierarchical Bayesian modeling.
        
        Args:
            data: Training data for hyperparameter estimation
            model_structure: Structure of the hierarchical model
            prior_hyperparams: Prior beliefs about hyperparameters
            
        Returns:
            Dictionary of estimated hyperparameter values
            
        Raises:
            ValueError: If model structure is invalid
            EstimationError: If hyperparameter estimation fails
        """
        pass

    @abstractmethod
    def compute_information_gain(
        self,
        prior_distribution: ProbabilityDistribution,
        posterior_distribution: ProbabilityDistribution
    ) -> float:
        """
        Compute information gain from prior to posterior.
        
        Measures how much information was gained by observing evidence,
        computed as KL divergence between posterior and prior.
        
        Args:
            prior_distribution: Prior belief distribution
            posterior_distribution: Posterior belief distribution
            
        Returns:
            Information gain in bits (KL divergence)
            
        Raises:
            ValueError: If distributions are incompatible
        """
        pass

    @abstractmethod
    def propagate_uncertainty(
        self,
        input_uncertainty: ProbabilityDistribution,
        transformation_function: str,
        transformation_params: Dict[str, Any]
    ) -> ProbabilityDistribution:
        """
        Propagate uncertainty through transformations.
        
        Computes how uncertainty in inputs translates to uncertainty
        in outputs through various transformation functions.
        
        Args:
            input_uncertainty: Uncertainty distribution in inputs
            transformation_function: Type of transformation ("linear", "nonlinear", "neural_network")
            transformation_params: Parameters of the transformation
            
        Returns:
            Output uncertainty distribution
            
        Raises:
            ValueError: If transformation is not supported
            ComputationError: If uncertainty propagation fails
        """
        pass

    @abstractmethod
    def compute_surprise(
        self,
        observed_data: npt.NDArray,
        predicted_distribution: ProbabilityDistribution
    ) -> float:
        """
        Compute surprise (negative log-likelihood) of observations.
        
        Measures how surprising the observed data is under the
        current predictive distribution, implementing the free
        energy principle's surprise minimization.
        
        Args:
            observed_data: Actually observed data
            predicted_distribution: Predicted probability distribution
            
        Returns:
            Surprise value (negative log-likelihood)
            
        Raises:
            ValueError: If data doesn't match distribution support
        """
        pass

    @abstractmethod
    def sample_from_posterior(
        self,
        posterior_distribution: ProbabilityDistribution,
        n_samples: int,
        sampling_method: str = "monte_carlo"
    ) -> npt.NDArray:
        """
        Generate samples from posterior distribution.
        
        Provides samples for Monte Carlo estimation and uncertainty
        quantification in downstream computations.
        
        Args:
            posterior_distribution: Distribution to sample from
            n_samples: Number of samples to generate
            sampling_method: Sampling algorithm ("monte_carlo", "gibbs", "metropolis")
            
        Returns:
            Array of samples from the posterior
            
        Raises:
            ValueError: If sampling parameters are invalid
            SamplingError: If sampling fails
        """
        pass


class BayesianInferenceError(Exception):
    """Base exception for Bayesian inference operations."""
    pass


class ComputationError(BayesianInferenceError):
    """Raised when Bayesian computations fail."""
    pass


class OptimizationError(BayesianInferenceError):
    """Raised when optimization procedures fail to converge."""
    pass


class EstimationError(BayesianInferenceError):
    """Raised when parameter estimation fails."""
    pass


class SamplingError(BayesianInferenceError):
    """Raised when sampling procedures fail."""
    pass