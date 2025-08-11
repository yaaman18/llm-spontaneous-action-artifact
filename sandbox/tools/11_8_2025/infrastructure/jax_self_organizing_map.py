"""
JAX-based Self-Organizing Map Implementation.

High-performance SOM implementation using JAX for JIT compilation and 
vectorized operations. Maintains Clean Architecture principles by 
implementing the abstract SelfOrganizingMap interface.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple, Optional, List
import numpy as np
import numpy.typing as npt

from domain.entities.self_organizing_map import SelfOrganizingMap
from domain.value_objects.som_topology import SOMTopology
from domain.value_objects.learning_parameters import LearningParameters


class JaxSelfOrganizingMap(SelfOrganizingMap):
    """
    JAX-based Self-Organizing Map implementation.
    
    Provides high-performance SOM operations using JAX transformations
    while maintaining compatibility with domain layer abstractions.
    Follows Dependency Inversion Principle by depending on abstractions.
    """
    
    def __init__(
        self,
        map_dimensions: Tuple[int, int],
        input_dimensions: int,
        topology: SOMTopology,
        random_seed: int = 42
    ):
        """
        Initialize JAX-based SOM.
        
        Args:
            map_dimensions: (width, height) of SOM grid
            input_dimensions: Dimensionality of input vectors
            topology: SOM topology configuration
            random_seed: Random seed for reproducibility
        """
        super().__init__(map_dimensions, input_dimensions, topology)
        
        # JAX-specific initialization
        self._key = jax.random.PRNGKey(random_seed)
        self._weight_matrix: Optional[jnp.ndarray] = None
        
        # Compile JIT functions for performance
        self._jit_functions = self._compile_jax_functions()
    
    def _compile_jax_functions(self) -> dict:
        """Compile JAX functions with JIT for performance."""
        
        @jit
        def _jit_find_bmu(weights: jnp.ndarray, input_vec: jnp.ndarray) -> Tuple[int, int]:
            """JIT-compiled BMU finding."""
            # Compute distances to all units
            distances = jnp.linalg.norm(weights - input_vec, axis=2)
            # Find minimum distance position
            flat_idx = jnp.argmin(distances)
            return jnp.unravel_index(flat_idx, distances.shape)
        
        @jit  
        def _jit_update_weights(
            weights: jnp.ndarray,
            input_vec: jnp.ndarray,
            neighborhood: jnp.ndarray,
            learning_rate: float
        ) -> jnp.ndarray:
            """JIT-compiled weight update."""
            # Vectorized weight update with neighborhood modulation
            delta = learning_rate * neighborhood[..., None] * (input_vec - weights)
            return weights + delta
        
        @jit
        def _jit_compute_distances(
            center: Tuple[int, int],
            map_dims: Tuple[int, int],
            topology_type: str
        ) -> jnp.ndarray:
            """JIT-compiled distance computation."""
            width, height = map_dims
            cx, cy = center
            
            # Create coordinate grids
            x_coords, y_coords = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
            
            # Compute distances based on topology
            dx = jnp.abs(x_coords - cx)
            dy = jnp.abs(y_coords - cy)
            
            # Apply wrap-around for toroidal topology
            if topology_type == "toroidal":
                dx = jnp.minimum(dx, width - dx)
                dy = jnp.minimum(dy, height - dy)
            
            return jnp.sqrt(dx**2 + dy**2)
        
        return {
            'find_bmu': _jit_find_bmu,
            'update_weights': _jit_update_weights,
            'compute_distances': _jit_compute_distances
        }
    
    def initialize_weights(self, initialization_method: str = "random") -> None:
        """
        Initialize weight matrix using JAX operations.
        
        Args:
            initialization_method: Weight initialization method
        """
        if initialization_method not in ["random", "pca", "uniform"]:
            raise ValueError(f"Unknown initialization method: {initialization_method}")
        
        width, height = self.map_dimensions
        
        if initialization_method == "random":
            # Random initialization with normal distribution
            self._key, subkey = jax.random.split(self._key)
            self._weight_matrix = jax.random.normal(
                subkey, (height, width, self.input_dimensions)
            ) * 0.1
        elif initialization_method == "uniform":
            # Uniform initialization in [-0.1, 0.1]
            self._key, subkey = jax.random.split(self._key)
            self._weight_matrix = jax.random.uniform(
                subkey, (height, width, self.input_dimensions), 
                minval=-0.1, maxval=0.1
            )
        else:  # pca
            # PCA initialization would require data, simplified to linear
            coords = jnp.linspace(-1, 1, width * height)
            coords = coords.reshape(height, width)
            # Create linear gradients for each dimension
            weights = jnp.stack([
                coords * (i + 1) * 0.1 for i in range(self.input_dimensions)
            ], axis=-1)
            self._weight_matrix = weights
    
    def find_best_matching_unit(self, input_vector: npt.NDArray) -> Tuple[int, int]:
        """
        Find BMU using JIT-compiled JAX function.
        
        Args:
            input_vector: Input vector for BMU search
            
        Returns:
            BMU coordinates (x, y)
        """
        if not self.is_trained:
            raise RuntimeError("SOM must be initialized before BMU search")
        
        if input_vector.shape[-1] != self.input_dimensions:
            raise ValueError(f"Input vector dimensions mismatch")
        
        # Convert to JAX array and find BMU
        jax_input = jnp.array(input_vector)
        bmu_y, bmu_x = self._jit_functions['find_bmu'](self._weight_matrix, jax_input)
        
        return int(bmu_x), int(bmu_y)
    
    def compute_neighborhood_function(
        self,
        bmu_position: Tuple[int, int],
        current_iteration: int,
        learning_params: LearningParameters
    ) -> npt.NDArray:
        """
        Compute neighborhood function using topology-aware distance.
        
        Args:
            bmu_position: BMU coordinates
            current_iteration: Current training iteration
            learning_params: Learning parameters with neighborhood config
            
        Returns:
            Neighborhood function values
        """
        # Get current neighborhood radius
        radius = learning_params.current_neighborhood_radius(current_iteration)
        
        # Compute distances using JIT function
        distances = self._jit_functions['compute_distances'](
            bmu_position, self.map_dimensions, self.topology.topology_type.value
        )
        
        # Apply neighborhood function
        if self.topology.neighborhood_function.value == "gaussian":
            sigma = radius / 3.0
            neighborhood = jnp.exp(-(distances**2) / (2 * sigma**2))
        elif self.topology.neighborhood_function.value == "bubble":
            neighborhood = (distances <= radius).astype(jnp.float32)
        else:  # triangular
            neighborhood = jnp.maximum(0.0, 1.0 - distances / radius)
        
        return np.array(neighborhood)
    
    def update_weights(
        self,
        input_vector: npt.NDArray,
        bmu_position: Tuple[int, int],
        neighborhood_function: npt.NDArray,
        learning_rate: float
    ) -> None:
        """
        Update weights using JIT-compiled vectorized operations.
        
        Args:
            input_vector: Current input vector
            bmu_position: BMU coordinates (unused in vectorized version)
            neighborhood_function: Neighborhood values
            learning_rate: Current learning rate
        """
        if not self.is_trained:
            raise RuntimeError("SOM must be initialized before weight updates")
        
        # Convert inputs to JAX arrays
        jax_input = jnp.array(input_vector)
        jax_neighborhood = jnp.array(neighborhood_function)
        
        # Update weights using JIT function
        self._weight_matrix = self._jit_functions['update_weights'](
            self._weight_matrix, jax_input, jax_neighborhood, learning_rate
        )
    
    def compute_quantization_error(self, input_data: npt.NDArray) -> float:
        """
        Compute quantization error efficiently using JAX.
        
        Args:
            input_data: Test data for error computation
            
        Returns:
            Average quantization error
        """
        if not self.is_trained:
            raise RuntimeError("SOM must be trained before error computation")
        
        if input_data.ndim != 2 or input_data.shape[1] != self.input_dimensions:
            raise ValueError("Input data shape mismatch")
        
        total_error = 0.0
        
        # Vectorized error computation
        for input_vec in input_data:
            bmu_pos = self.find_best_matching_unit(input_vec)
            bmu_weight = self._weight_matrix[bmu_pos[1], bmu_pos[0]]  # y, x indexing
            error = float(jnp.linalg.norm(jnp.array(input_vec) - bmu_weight))
            total_error += error
        
        return total_error / len(input_data)
    
    def compute_topographic_error(self, input_data: npt.NDArray) -> float:
        """
        Compute topographic error using neighborhood consistency.
        
        Args:
            input_data: Test data for error computation
            
        Returns:
            Topographic error ratio [0, 1]
        """
        if not self.is_trained:
            raise RuntimeError("SOM must be trained before error computation")
        
        topographic_errors = 0
        
        for input_vec in input_data:
            # Find BMU and second best matching unit
            distances = jnp.linalg.norm(self._weight_matrix - jnp.array(input_vec), axis=2)
            flat_distances = distances.flatten()
            
            # Get indices of two smallest distances
            sorted_indices = jnp.argsort(flat_distances)
            bmu_flat_idx = sorted_indices[0]
            second_bmu_flat_idx = sorted_indices[1]
            
            # Convert to 2D coordinates
            height, width = self.map_dimensions
            bmu_pos = jnp.unravel_index(bmu_flat_idx, (height, width))
            second_bmu_pos = jnp.unravel_index(second_bmu_flat_idx, (height, width))
            
            # Check if BMU and second BMU are neighbors
            distance = self.topology.calculate_grid_distance(
                (int(bmu_pos[1]), int(bmu_pos[0])),  # Convert to (x, y)
                (int(second_bmu_pos[1]), int(second_bmu_pos[0])),
                (width, height)
            )
            
            # Topographic error if not immediate neighbors
            if distance > 1.5:  # Allow some tolerance for diagonal neighbors
                topographic_errors += 1
        
        return topographic_errors / len(input_data)
    
    @property
    def weight_matrix(self) -> Optional[npt.NDArray]:
        """Get weight matrix as numpy array."""
        if self._weight_matrix is None:
            return None
        return np.array(self._weight_matrix)
    
    def get_jax_weight_matrix(self) -> Optional[jnp.ndarray]:
        """Get JAX weight matrix for internal operations."""
        return self._weight_matrix
    
    def set_weights_from_numpy(self, weights: npt.NDArray) -> None:
        """Set weights from numpy array (for integration)."""
        expected_shape = (self.map_dimensions[1], self.map_dimensions[0], self.input_dimensions)
        if weights.shape != expected_shape:
            raise ValueError(f"Weight shape {weights.shape} != expected {expected_shape}")
        
        self._weight_matrix = jnp.array(weights)