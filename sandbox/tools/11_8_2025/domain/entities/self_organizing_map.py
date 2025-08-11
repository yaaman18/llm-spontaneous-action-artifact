"""
Self-Organizing Map (SOM) Entity.

Domain entity implementing Kohonen's Self-Organizing Map for spatial
representation and concept space organization. Follows SRP by focusing
on topological learning and neighborhood preservation.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import numpy.typing as npt
from ..value_objects.som_topology import SOMTopology
from ..value_objects.learning_parameters import LearningParameters


class SelfOrganizingMap(ABC):
    """
    Abstract base class for Self-Organizing Map implementation.
    
    This entity encapsulates the core business logic for topological
    learning and spatial organization of high-dimensional input data.
    Implements the Interface Segregation Principle by providing focused
    interfaces for different aspects of SOM functionality.
    
    Responsibilities:
    - Topological organization of input space
    - Best matching unit (BMU) computation
    - Neighborhood function calculation
    - Competitive learning dynamics
    """

    def __init__(
        self,
        map_dimensions: Tuple[int, int],
        input_dimensions: int,
        topology: SOMTopology
    ):
        """
        Initialize the self-organizing map.
        
        Args:
            map_dimensions: (width, height) of the SOM grid
            input_dimensions: Dimensionality of input vectors
            topology: Topology configuration for the map
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if any(dim < 1 for dim in map_dimensions):
            raise ValueError("Map dimensions must be positive")
        if input_dimensions < 1:
            raise ValueError("Input dimensions must be positive")
            
        self._map_dimensions = map_dimensions
        self._input_dimensions = input_dimensions
        self._topology = topology
        self._weight_matrix: Optional[npt.NDArray] = None
        self._training_iterations = 0

    @property
    def map_dimensions(self) -> Tuple[int, int]:
        """Dimensions of the SOM grid (width, height)."""
        return self._map_dimensions

    @property
    def input_dimensions(self) -> int:
        """Dimensionality of input vectors."""
        return self._input_dimensions

    @property
    def topology(self) -> SOMTopology:
        """Topology configuration of the map."""
        return self._topology

    @property
    def weight_matrix(self) -> Optional[npt.NDArray]:
        """Current weight matrix of the SOM."""
        return self._weight_matrix

    @property
    def training_iterations(self) -> int:
        """Number of training iterations completed."""
        return self._training_iterations

    @property
    def is_trained(self) -> bool:
        """Check if the SOM has been initialized with weights."""
        return self._weight_matrix is not None

    @abstractmethod
    def initialize_weights(self, initialization_method: str = "random") -> None:
        """
        Initialize the weight matrix of the SOM.
        
        Args:
            initialization_method: Method for weight initialization
                ("random", "pca", "uniform")
                
        Raises:
            ValueError: If initialization method is not supported
        """
        pass

    @abstractmethod
    def find_best_matching_unit(self, input_vector: npt.NDArray) -> Tuple[int, int]:
        """
        Find the best matching unit (BMU) for the input vector.
        
        Args:
            input_vector: Input data vector
            
        Returns:
            Coordinates (x, y) of the BMU on the map
            
        Raises:
            ValueError: If input vector shape doesn't match expected dimensions
            RuntimeError: If weights are not initialized
        """
        pass

    @abstractmethod
    def compute_neighborhood_function(
        self,
        bmu_position: Tuple[int, int],
        current_iteration: int,
        learning_params: LearningParameters
    ) -> npt.NDArray:
        """
        Compute the neighborhood function centered on the BMU.
        
        Args:
            bmu_position: Position of the best matching unit
            current_iteration: Current training iteration
            learning_params: Learning parameters including neighborhood radius
            
        Returns:
            Neighborhood function values for all units in the map
            
        Raises:
            ValueError: If BMU position is out of bounds
        """
        pass

    @abstractmethod
    def update_weights(
        self,
        input_vector: npt.NDArray,
        bmu_position: Tuple[int, int],
        neighborhood_function: npt.NDArray,
        learning_rate: float
    ) -> None:
        """
        Update weights based on input vector and neighborhood function.
        
        Args:
            input_vector: Current input vector
            bmu_position: Position of the best matching unit
            neighborhood_function: Neighborhood function values
            learning_rate: Current learning rate
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If weights are not initialized
        """
        pass

    def train_single_iteration(
        self,
        input_vector: npt.NDArray,
        learning_params: LearningParameters
    ) -> Tuple[int, int]:
        """
        Perform a single training iteration.
        
        Template method implementing the SOM learning algorithm.
        Follows the Template Method pattern for the training cycle.
        
        Args:
            input_vector: Input data vector
            learning_params: Learning parameters for this iteration
            
        Returns:
            Position of the best matching unit
            
        Raises:
            RuntimeError: If SOM is not properly initialized
        """
        if not self.is_trained:
            raise RuntimeError("SOM weights must be initialized before training")

        # Find best matching unit
        bmu_position = self.find_best_matching_unit(input_vector)
        
        # Compute neighborhood function
        neighborhood_function = self.compute_neighborhood_function(
            bmu_position, self._training_iterations, learning_params
        )
        
        # Update weights
        self.update_weights(
            input_vector,
            bmu_position,
            neighborhood_function,
            learning_params.current_learning_rate(self._training_iterations)
        )
        
        # Increment training iteration counter
        self._training_iterations += 1
        
        return bmu_position

    def train_batch(
        self,
        input_data: npt.NDArray,
        learning_params: LearningParameters,
        max_iterations: int
    ) -> List[Tuple[int, int]]:
        """
        Train the SOM on a batch of input data.
        
        Args:
            input_data: Batch of input vectors (n_samples, input_dimensions)
            learning_params: Learning parameters for training
            max_iterations: Maximum number of training iterations
            
        Returns:
            List of BMU positions for each training sample
            
        Raises:
            ValueError: If input data shape is invalid
            RuntimeError: If SOM is not properly initialized
        """
        if input_data.ndim != 2:
            raise ValueError("Input data must be 2D array")
        if input_data.shape[1] != self._input_dimensions:
            raise ValueError(
                f"Input data must have {self._input_dimensions} dimensions"
            )

        bmu_positions = []
        
        for iteration in range(max_iterations):
            # Cycle through input data
            input_index = iteration % len(input_data)
            input_vector = input_data[input_index]
            
            bmu_position = self.train_single_iteration(input_vector, learning_params)
            bmu_positions.append(bmu_position)
            
        return bmu_positions

    @abstractmethod
    def compute_quantization_error(self, input_data: npt.NDArray) -> float:
        """
        Compute the quantization error of the SOM.
        
        Args:
            input_data: Test data for error computation
            
        Returns:
            Average quantization error
            
        Raises:
            RuntimeError: If SOM is not trained
        """
        pass

    @abstractmethod
    def compute_topographic_error(self, input_data: npt.NDArray) -> float:
        """
        Compute the topographic error of the SOM.
        
        Args:
            input_data: Test data for error computation
            
        Returns:
            Topographic error ratio
            
        Raises:
            RuntimeError: If SOM is not trained
        """
        pass

    def get_map_state(self) -> dict:
        """
        Get current state of the SOM for persistence or analysis.
        
        Returns:
            Dictionary containing SOM state information
        """
        return {
            "map_dimensions": self._map_dimensions,
            "input_dimensions": self._input_dimensions,
            "training_iterations": self._training_iterations,
            "is_trained": self.is_trained,
            "topology": self._topology.to_dict() if hasattr(self._topology, 'to_dict') else str(self._topology)
        }

    def reset_training(self) -> None:
        """Reset the training state of the SOM."""
        self._training_iterations = 0
        self._weight_matrix = None