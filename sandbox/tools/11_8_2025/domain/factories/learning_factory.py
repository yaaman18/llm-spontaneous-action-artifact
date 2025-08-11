"""
Factory for creating learning-related domain objects.

This factory creates properly initialized learning aggregates, parameters,
and related components following enactivist principles.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from ..aggregates.learning_aggregate import LearningAggregate
from ..value_objects.learning_parameters import LearningParameters
from ..value_objects.precision_weights import PrecisionWeights
from ..value_objects.som_topology import SOMTopology


class LearningFactory:
    """
    Factory for creating learning components with proper initialization.
    
    This factory ensures that learning components are created with
    enactivist principles and proper environmental coupling setup.
    """
    
    @staticmethod
    def create_learning_parameters(
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        adaptation_rate: float = 0.001,
        min_learning_rate: float = 1e-6,
        max_learning_rate: float = 0.1,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> LearningParameters:
        """
        Create learning parameters with enactivist defaults.
        
        Args:
            learning_rate: Base learning rate
            momentum: Momentum coefficient for optimization
            adaptation_rate: Rate of learning rate adaptation
            min_learning_rate: Minimum allowed learning rate
            max_learning_rate: Maximum allowed learning rate
            custom_params: Additional custom parameters
            
        Returns:
            Initialized learning parameters
        """
        return LearningParameters(
            learning_rate=learning_rate,
            momentum=momentum,
            adaptation_rate=adaptation_rate,
            min_learning_rate=min_learning_rate,
            max_learning_rate=max_learning_rate,
            custom_parameters=custom_params or {}
        )
    
    @staticmethod
    def create_precision_weights(
        layer_count: int,
        initial_precision: float = 1.0,
        attention_modulation: float = 0.1
    ) -> PrecisionWeights:
        """
        Create precision weights for hierarchical layers.
        
        Args:
            layer_count: Number of hierarchical layers
            initial_precision: Initial precision value for all layers
            attention_modulation: Strength of attentional modulation
            
        Returns:
            Initialized precision weights
        """
        weights = np.full(layer_count, initial_precision)
        return PrecisionWeights(
            layer_precisions=weights.tolist(),
            attention_modulation=attention_modulation
        )
    
    @staticmethod
    def create_som_topology(
        map_width: int = 10,
        map_height: int = 10,
        input_dimension: int = 2,
        neighborhood_function: str = "gaussian",
        initial_radius: Optional[float] = None
    ) -> SOMTopology:
        """
        Create SOM topology configuration.
        
        Args:
            map_width: Width of the SOM grid
            map_height: Height of the SOM grid  
            input_dimension: Dimensionality of input vectors
            neighborhood_function: Type of neighborhood function
            initial_radius: Initial neighborhood radius
            
        Returns:
            Initialized SOM topology
        """
        if initial_radius is None:
            initial_radius = max(map_width, map_height) / 2.0
            
        return SOMTopology(
            width=map_width,
            height=map_height,
            input_dimension=input_dimension,
            neighborhood_function=neighborhood_function,
            initial_radius=initial_radius
        )
    
    @staticmethod
    def create_learning_aggregate(
        learning_params: Optional[LearningParameters] = None,
        precision_weights: Optional[PrecisionWeights] = None,
        som_topology: Optional[SOMTopology] = None,
        environmental_coupling_strength: float = 0.5
    ) -> LearningAggregate:
        """
        Create a complete learning aggregate with all components.
        
        Args:
            learning_params: Learning parameters (default if None)
            precision_weights: Precision weights (default if None) 
            som_topology: SOM topology (default if None)
            environmental_coupling_strength: Initial coupling strength
            
        Returns:
            Fully initialized learning aggregate
        """
        if learning_params is None:
            learning_params = LearningFactory.create_learning_parameters()
            
        if precision_weights is None:
            precision_weights = LearningFactory.create_precision_weights(layer_count=3)
            
        if som_topology is None:
            som_topology = LearningFactory.create_som_topology()
            
        return LearningAggregate(
            learning_params=learning_params,
            precision_weights=precision_weights,
            som_topology=som_topology,
            environmental_coupling_strength=environmental_coupling_strength
        )