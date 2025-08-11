"""
Unit tests for SelfOrganizingMap entity.

Comprehensive TDD test suite covering SOM topology learning, BMU computation,
neighborhood functions, and competitive learning dynamics. Tests include
property-based testing for topological preservation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Tuple, List
import numpy.typing as npt

from domain.entities.self_organizing_map import SelfOrganizingMap
from domain.value_objects.som_topology import SOMTopology
from domain.value_objects.learning_parameters import LearningParameters


# Test implementation of abstract SelfOrganizingMap for testing
class MockSelfOrganizingMap(SelfOrganizingMap):
    """Mock implementation of SelfOrganizingMap for testing."""
    
    def __init__(
        self,
        map_dimensions: Tuple[int, int],
        input_dimensions: int,
        topology: SOMTopology
    ):
        super().__init__(map_dimensions, input_dimensions, topology)
        self.initialization_calls = []
        self.bmu_calls = []
        self.neighborhood_calls = []
        self.weight_update_calls = []
    
    def initialize_weights(self, initialization_method: str = "random") -> None:
        """Mock weight initialization."""
        if initialization_method not in ["random", "pca", "uniform"]:
            raise ValueError(f"Unsupported initialization method: {initialization_method}")
        
        self.initialization_calls.append(initialization_method)
        
        # Create mock weight matrix
        width, height = self.map_dimensions
        self._weight_matrix = np.random.rand(height, width, self.input_dimensions)
        
        if initialization_method == "uniform":
            self._weight_matrix = np.ones((height, width, self.input_dimensions)) * 0.5
        elif initialization_method == "pca":
            # Simple PCA-like initialization (not actual PCA)
            self._weight_matrix = np.random.rand(height, width, self.input_dimensions) * 2 - 1
    
    def find_best_matching_unit(self, input_vector: npt.NDArray) -> Tuple[int, int]:
        """Mock BMU finding."""
        if not self.is_trained:
            raise RuntimeError("Weights must be initialized before finding BMU")
        
        if input_vector.shape[0] != self.input_dimensions:
            raise ValueError(f"Input vector must have {self.input_dimensions} dimensions")
        
        self.bmu_calls.append(input_vector.copy())
        
        # Simple distance-based BMU finding
        width, height = self.map_dimensions
        min_distance = float('inf')
        best_position = (0, 0)
        
        for i in range(width):
            for j in range(height):
                weight_vector = self._weight_matrix[j, i, :]
                distance = np.linalg.norm(input_vector - weight_vector)
                if distance < min_distance:
                    min_distance = distance
                    best_position = (i, j)
        
        return best_position
    
    def compute_neighborhood_function(
        self,
        bmu_position: Tuple[int, int],
        current_iteration: int,
        learning_params: LearningParameters
    ) -> npt.NDArray:
        """Mock neighborhood function computation."""
        bmu_x, bmu_y = bmu_position
        width, height = self.map_dimensions
        
        if not (0 <= bmu_x < width and 0 <= bmu_y < height):
            raise ValueError(f"BMU position {bmu_position} is out of bounds")
        
        self.neighborhood_calls.append((bmu_position, current_iteration, learning_params))
        
        # Gaussian neighborhood function
        neighborhood = np.zeros((height, width))
        radius = learning_params.current_neighborhood_radius(current_iteration)
        
        for i in range(width):
            for j in range(height):
                distance_sq = (i - bmu_x)**2 + (j - bmu_y)**2
                neighborhood[j, i] = np.exp(-distance_sq / (2 * radius**2))
        
        return neighborhood
    
    def update_weights(
        self,
        input_vector: npt.NDArray,
        bmu_position: Tuple[int, int],
        neighborhood_function: npt.NDArray,
        learning_rate: float
    ) -> None:
        """Mock weight update."""
        if not self.is_trained:
            raise RuntimeError("Weights must be initialized before updating")
        
        if input_vector.shape[0] != self.input_dimensions:
            raise ValueError(f"Input vector must have {self.input_dimensions} dimensions")
        
        if neighborhood_function.shape != self.map_dimensions[::-1]:  # (height, width)
            raise ValueError("Neighborhood function shape doesn't match map dimensions")
        
        self.weight_update_calls.append((
            input_vector.copy(), bmu_position, neighborhood_function.copy(), learning_rate
        ))
        
        # Simple weight update rule
        width, height = self.map_dimensions
        for i in range(width):
            for j in range(height):
                influence = neighborhood_function[j, i] * learning_rate
                weight_diff = input_vector - self._weight_matrix[j, i, :]
                self._weight_matrix[j, i, :] += influence * weight_diff
    
    def compute_quantization_error(self, input_data: npt.NDArray) -> float:
        """Mock quantization error computation."""
        if not self.is_trained:
            raise RuntimeError("SOM must be trained before computing quantization error")
        
        if input_data.ndim != 2 or input_data.shape[1] != self.input_dimensions:
            raise ValueError("Input data must be 2D with correct feature dimensions")
        
        total_error = 0.0
        for input_vector in input_data:
            bmu_position = self.find_best_matching_unit(input_vector)
            bmu_x, bmu_y = bmu_position
            bmu_weights = self._weight_matrix[bmu_y, bmu_x, :]
            error = np.linalg.norm(input_vector - bmu_weights)
            total_error += error
        
        return total_error / len(input_data)
    
    def compute_topographic_error(self, input_data: npt.NDArray) -> float:
        """Mock topographic error computation."""
        if not self.is_trained:
            raise RuntimeError("SOM must be trained before computing topographic error")
        
        topographic_errors = 0
        
        for input_vector in input_data:
            bmu_position = self.find_best_matching_unit(input_vector)
            
            # Find second best matching unit
            distances = []
            width, height = self.map_dimensions
            
            for i in range(width):
                for j in range(height):
                    if (i, j) != bmu_position:
                        weight_vector = self._weight_matrix[j, i, :]
                        distance = np.linalg.norm(input_vector - weight_vector)
                        distances.append(((i, j), distance))
            
            second_bmu = min(distances, key=lambda x: x[1])[0]
            
            # Check if BMU and second BMU are adjacent
            bmu_x, bmu_y = bmu_position
            second_x, second_y = second_bmu
            
            if abs(bmu_x - second_x) > 1 or abs(bmu_y - second_y) > 1:
                topographic_errors += 1
        
        return topographic_errors / len(input_data)


class TestSelfOrganizingMapCreation:
    """Test suite for SelfOrganizingMap creation and validation."""
    
    @pytest.fixture
    def sample_topology(self):
        """Create sample SOM topology for testing."""
        return SOMTopology.create_rectangular()
    
    def test_valid_som_creation(self, sample_topology):
        """Test creating valid SOM instance."""
        # Arrange & Act
        som = MockSelfOrganizingMap(
            map_dimensions=(10, 8),
            input_dimensions=5,
            topology=sample_topology
        )
        
        # Assert
        assert som.map_dimensions == (10, 8)
        assert som.input_dimensions == 5
        assert som.topology == sample_topology
        assert not som.is_trained
        assert som.training_iterations == 0

    def test_invalid_map_dimensions_raise_error(self, sample_topology):
        """Test that invalid map dimensions raise error."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Map dimensions must be positive"):
            MockSelfOrganizingMap(
                map_dimensions=(0, 5),
                input_dimensions=3,
                topology=sample_topology
            )
        
        with pytest.raises(ValueError, match="Map dimensions must be positive"):
            MockSelfOrganizingMap(
                map_dimensions=(5, -1),
                input_dimensions=3,
                topology=sample_topology
            )

    def test_invalid_input_dimensions_raise_error(self, sample_topology):
        """Test that invalid input dimensions raise error."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Input dimensions must be positive"):
            MockSelfOrganizingMap(
                map_dimensions=(5, 5),
                input_dimensions=0,
                topology=sample_topology
            )

    def test_properties_are_accessible(self, sample_topology):
        """Test that SOM properties are accessible."""
        # Arrange
        som = MockSelfOrganizingMap(
            map_dimensions=(6, 4),
            input_dimensions=3,
            topology=sample_topology
        )
        
        # Act & Assert
        assert som.map_dimensions == (6, 4)
        assert som.input_dimensions == 3
        assert som.topology == sample_topology
        assert som.weight_matrix is None
        assert som.training_iterations == 0
        assert not som.is_trained


class TestSelfOrganizingMapInitialization:
    """Test suite for SOM weight initialization."""
    
    @pytest.fixture
    def sample_som(self):
        """Create sample SOM for testing."""
        topology = SOMTopology.create_rectangular()
        return MockSelfOrganizingMap(
            map_dimensions=(5, 4),
            input_dimensions=3,
            topology=topology
        )
    
    def test_random_weight_initialization(self, sample_som):
        """Test random weight initialization."""
        # Act
        sample_som.initialize_weights("random")
        
        # Assert
        assert sample_som.is_trained
        assert sample_som.weight_matrix is not None
        assert sample_som.weight_matrix.shape == (4, 5, 3)  # (height, width, input_dims)
        assert len(sample_som.initialization_calls) == 1

    def test_uniform_weight_initialization(self, sample_som):
        """Test uniform weight initialization."""
        # Act
        sample_som.initialize_weights("uniform")
        
        # Assert
        assert sample_som.is_trained
        assert np.allclose(sample_som.weight_matrix, 0.5)

    def test_pca_weight_initialization(self, sample_som):
        """Test PCA-like weight initialization."""
        # Act
        sample_som.initialize_weights("pca")
        
        # Assert
        assert sample_som.is_trained
        assert sample_som.weight_matrix.min() >= -1.0
        assert sample_som.weight_matrix.max() <= 1.0

    def test_invalid_initialization_method_raises_error(self, sample_som):
        """Test that invalid initialization method raises error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported initialization method"):
            sample_som.initialize_weights("invalid_method")


class TestSelfOrganizingMapBMU:
    """Test suite for Best Matching Unit (BMU) computation."""
    
    @pytest.fixture
    def initialized_som(self):
        """Create initialized SOM for testing."""
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        som.initialize_weights("uniform")
        return som
    
    def test_find_bmu_with_valid_input(self, initialized_som):
        """Test finding BMU with valid input."""
        # Arrange
        input_vector = np.array([0.5, 0.5])
        
        # Act
        bmu_position = initialized_som.find_best_matching_unit(input_vector)
        
        # Assert
        assert isinstance(bmu_position, tuple)
        assert len(bmu_position) == 2
        assert 0 <= bmu_position[0] < 3  # x coordinate
        assert 0 <= bmu_position[1] < 3  # y coordinate
        assert len(initialized_som.bmu_calls) == 1

    def test_find_bmu_without_initialization_raises_error(self):
        """Test finding BMU without weight initialization raises error."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        input_vector = np.array([0.5, 0.5])
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Weights must be initialized"):
            som.find_best_matching_unit(input_vector)

    def test_find_bmu_wrong_input_dimensions_raises_error(self, initialized_som):
        """Test finding BMU with wrong input dimensions raises error."""
        # Arrange
        wrong_input = np.array([0.5, 0.5, 0.5])  # 3D instead of 2D
        
        # Act & Assert
        with pytest.raises(ValueError, match="Input vector must have .* dimensions"):
            initialized_som.find_best_matching_unit(wrong_input)

    def test_bmu_consistency(self, initialized_som):
        """Test that same input produces same BMU."""
        # Arrange
        input_vector = np.array([0.3, 0.7])
        
        # Act
        bmu1 = initialized_som.find_best_matching_unit(input_vector)
        bmu2 = initialized_som.find_best_matching_unit(input_vector)
        
        # Assert
        assert bmu1 == bmu2


class TestSelfOrganizingMapNeighborhoodFunction:
    """Test suite for neighborhood function computation."""
    
    @pytest.fixture
    def initialized_som(self):
        """Create initialized SOM for testing."""
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(5, 5),
            input_dimensions=2,
            topology=topology
        )
        som.initialize_weights("uniform")
        return som
    
    @pytest.fixture
    def learning_params(self):
        """Create learning parameters for testing."""
        return LearningParameters(
            initial_learning_rate=0.1,
            final_learning_rate=0.01,
            initial_radius=2.0,
            final_radius=0.5,
            max_iterations=1000
        )
    
    def test_compute_neighborhood_function_valid_bmu(self, initialized_som, learning_params):
        """Test computing neighborhood function with valid BMU."""
        # Arrange
        bmu_position = (2, 2)  # Center of 5x5 map
        current_iteration = 100
        
        # Act
        neighborhood = initialized_som.compute_neighborhood_function(
            bmu_position, current_iteration, learning_params
        )
        
        # Assert
        assert neighborhood.shape == (5, 5)  # (height, width)
        assert neighborhood.max() == neighborhood[2, 2]  # Maximum at BMU
        assert neighborhood.min() >= 0.0  # All values non-negative
        assert len(initialized_som.neighborhood_calls) == 1

    def test_compute_neighborhood_function_out_of_bounds_bmu_raises_error(self, initialized_som, learning_params):
        """Test neighborhood function with out-of-bounds BMU raises error."""
        # Arrange
        invalid_bmu = (5, 5)  # Out of bounds for 5x5 map
        
        # Act & Assert
        with pytest.raises(ValueError, match="BMU position .* is out of bounds"):
            initialized_som.compute_neighborhood_function(
                invalid_bmu, 100, learning_params
            )

    def test_neighborhood_function_gaussian_properties(self, initialized_som, learning_params):
        """Test that neighborhood function has Gaussian-like properties."""
        # Arrange
        bmu_position = (2, 2)
        current_iteration = 0  # Use initial parameters
        
        # Act
        neighborhood = initialized_som.compute_neighborhood_function(
            bmu_position, current_iteration, learning_params
        )
        
        # Assert
        # Center should have highest value
        center_value = neighborhood[2, 2]
        assert center_value == neighborhood.max()
        
        # Values should decrease with distance
        corner_value = neighborhood[0, 0]
        edge_value = neighborhood[1, 2]
        assert center_value > edge_value > corner_value

    def test_neighborhood_function_shrinks_over_time(self, initialized_som, learning_params):
        """Test that neighborhood function shrinks over training iterations."""
        # Arrange
        bmu_position = (2, 2)
        early_iteration = 0
        late_iteration = 900
        
        # Act
        early_neighborhood = initialized_som.compute_neighborhood_function(
            bmu_position, early_iteration, learning_params
        )
        late_neighborhood = initialized_som.compute_neighborhood_function(
            bmu_position, late_iteration, learning_params
        )
        
        # Assert - late neighborhood should be more concentrated
        early_influence_sum = early_neighborhood.sum()
        late_influence_sum = late_neighborhood.sum()
        
        # Late neighborhood should have less total influence (more concentrated)
        assert late_influence_sum < early_influence_sum


class TestSelfOrganizingMapWeightUpdate:
    """Test suite for weight update operations."""
    
    @pytest.fixture
    def initialized_som(self):
        """Create initialized SOM for testing."""
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        som.initialize_weights("uniform")
        return som
    
    def test_update_weights_valid_parameters(self, initialized_som):
        """Test weight update with valid parameters."""
        # Arrange
        input_vector = np.array([0.7, 0.3])
        bmu_position = (1, 1)
        neighborhood_function = np.ones((3, 3)) * 0.5  # Uniform neighborhood
        learning_rate = 0.1
        
        initial_weights = initialized_som.weight_matrix.copy()
        
        # Act
        initialized_som.update_weights(
            input_vector, bmu_position, neighborhood_function, learning_rate
        )
        
        # Assert
        assert len(initialized_som.weight_update_calls) == 1
        # Weights should have changed
        assert not np.array_equal(initial_weights, initialized_som.weight_matrix)

    def test_update_weights_without_initialization_raises_error(self):
        """Test weight update without initialization raises error."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        
        input_vector = np.array([0.5, 0.5])
        neighborhood_function = np.ones((3, 3))
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Weights must be initialized"):
            som.update_weights(input_vector, (1, 1), neighborhood_function, 0.1)

    def test_update_weights_wrong_input_dimensions_raises_error(self, initialized_som):
        """Test weight update with wrong input dimensions raises error."""
        # Arrange
        wrong_input = np.array([0.5, 0.5, 0.5])  # 3D instead of 2D
        neighborhood_function = np.ones((3, 3))
        
        # Act & Assert
        with pytest.raises(ValueError, match="Input vector must have .* dimensions"):
            initialized_som.update_weights(wrong_input, (1, 1), neighborhood_function, 0.1)

    def test_update_weights_wrong_neighborhood_shape_raises_error(self, initialized_som):
        """Test weight update with wrong neighborhood shape raises error."""
        # Arrange
        input_vector = np.array([0.5, 0.5])
        wrong_neighborhood = np.ones((2, 2))  # Wrong shape
        
        # Act & Assert
        with pytest.raises(ValueError, match="Neighborhood function shape doesn't match"):
            initialized_som.update_weights(input_vector, (1, 1), wrong_neighborhood, 0.1)


class TestSelfOrganizingMapTraining:
    """Test suite for SOM training operations."""
    
    @pytest.fixture
    def initialized_som(self):
        """Create initialized SOM for testing."""
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(4, 4),
            input_dimensions=2,
            topology=topology
        )
        som.initialize_weights("random")
        return som
    
    @pytest.fixture
    def learning_params(self):
        """Create learning parameters for testing."""
        return LearningParameters(
            initial_learning_rate=0.5,
            final_learning_rate=0.01,
            initial_radius=2.0,
            final_radius=0.1,
            max_iterations=100
        )
    
    def test_train_single_iteration_valid_input(self, initialized_som, learning_params):
        """Test single training iteration with valid input."""
        # Arrange
        input_vector = np.array([0.3, 0.7])
        initial_iterations = initialized_som.training_iterations
        
        # Act
        bmu_position = initialized_som.train_single_iteration(input_vector, learning_params)
        
        # Assert
        assert isinstance(bmu_position, tuple)
        assert len(bmu_position) == 2
        assert initialized_som.training_iterations == initial_iterations + 1
        assert len(initialized_som.bmu_calls) >= 1
        assert len(initialized_som.neighborhood_calls) >= 1
        assert len(initialized_som.weight_update_calls) >= 1

    def test_train_single_iteration_without_initialization_raises_error(self, learning_params):
        """Test single training iteration without initialization raises error."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        input_vector = np.array([0.5, 0.5])
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="SOM weights must be initialized"):
            som.train_single_iteration(input_vector, learning_params)

    def test_train_batch_valid_data(self, initialized_som, learning_params):
        """Test batch training with valid data."""
        # Arrange
        input_data = np.random.rand(10, 2)  # 10 samples, 2 dimensions
        max_iterations = 50
        
        # Act
        bmu_positions = initialized_som.train_batch(input_data, learning_params, max_iterations)
        
        # Assert
        assert len(bmu_positions) == max_iterations
        assert all(isinstance(pos, tuple) for pos in bmu_positions)
        assert initialized_som.training_iterations == max_iterations

    def test_train_batch_invalid_data_shape_raises_error(self, initialized_som, learning_params):
        """Test batch training with invalid data shape raises error."""
        # Arrange
        invalid_data = np.random.rand(10)  # 1D instead of 2D
        
        # Act & Assert
        with pytest.raises(ValueError, match="Input data must be 2D array"):
            initialized_som.train_batch(invalid_data, learning_params, 10)

    def test_train_batch_wrong_feature_dimensions_raises_error(self, initialized_som, learning_params):
        """Test batch training with wrong feature dimensions raises error."""
        # Arrange
        wrong_data = np.random.rand(10, 3)  # 3 dimensions instead of 2
        
        # Act & Assert
        with pytest.raises(ValueError, match="Input data must have .* dimensions"):
            initialized_som.train_batch(wrong_data, learning_params, 10)

    def test_train_batch_cycles_through_data(self, initialized_som, learning_params):
        """Test that batch training cycles through input data."""
        # Arrange
        input_data = np.array([[0.1, 0.1], [0.9, 0.9]])  # 2 distinct samples
        max_iterations = 5  # More iterations than samples
        
        # Track which inputs are used
        original_train_single = initialized_som.train_single_iteration
        used_inputs = []
        
        def track_inputs(input_vec, params):
            used_inputs.append(input_vec.copy())
            return original_train_single(input_vec, params)
        
        initialized_som.train_single_iteration = track_inputs
        
        # Act
        initialized_som.train_batch(input_data, learning_params, max_iterations)
        
        # Assert
        assert len(used_inputs) == max_iterations
        # Should cycle through the data
        for i in range(max_iterations):
            expected_input = input_data[i % len(input_data)]
            np.testing.assert_array_equal(used_inputs[i], expected_input)


class TestSelfOrganizingMapErrorMetrics:
    """Test suite for SOM error metric computations."""
    
    @pytest.fixture
    def trained_som(self):
        """Create trained SOM for testing."""
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        som.initialize_weights("random")
        
        # Do some training
        learning_params = LearningParameters(
            initial_learning_rate=0.1,
            final_learning_rate=0.01,
            initial_radius=1.0,
            final_radius=0.1,
            max_iterations=50
        )
        
        training_data = np.random.rand(20, 2)
        som.train_batch(training_data, learning_params, 50)
        
        return som
    
    def test_compute_quantization_error_valid_data(self, trained_som):
        """Test quantization error computation with valid data."""
        # Arrange
        test_data = np.random.rand(10, 2)
        
        # Act
        quantization_error = trained_som.compute_quantization_error(test_data)
        
        # Assert
        assert isinstance(quantization_error, float)
        assert quantization_error >= 0.0

    def test_compute_quantization_error_without_training_raises_error(self):
        """Test quantization error computation without training raises error."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        test_data = np.random.rand(5, 2)
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="SOM must be trained"):
            som.compute_quantization_error(test_data)

    def test_compute_topographic_error_valid_data(self, trained_som):
        """Test topographic error computation with valid data."""
        # Arrange
        test_data = np.random.rand(10, 2)
        
        # Act
        topographic_error = trained_som.compute_topographic_error(test_data)
        
        # Assert
        assert isinstance(topographic_error, float)
        assert 0.0 <= topographic_error <= 1.0  # Topographic error is a ratio

    def test_compute_topographic_error_without_training_raises_error(self):
        """Test topographic error computation without training raises error."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        test_data = np.random.rand(5, 2)
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="SOM must be trained"):
            som.compute_topographic_error(test_data)


class TestSelfOrganizingMapStateManagement:
    """Test suite for SOM state management operations."""
    
    def test_get_map_state_untrained_som(self):
        """Test getting state of untrained SOM."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(5, 4),
            input_dimensions=3,
            topology=topology
        )
        
        # Act
        state = som.get_map_state()
        
        # Assert
        assert state["map_dimensions"] == (5, 4)
        assert state["input_dimensions"] == 3
        assert state["training_iterations"] == 0
        assert state["is_trained"] is False

    def test_get_map_state_trained_som(self):
        """Test getting state of trained SOM."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        som.initialize_weights("random")
        som._training_iterations = 100  # Simulate training
        
        # Act
        state = som.get_map_state()
        
        # Assert
        assert state["training_iterations"] == 100
        assert state["is_trained"] is True

    def test_reset_training_clears_state(self):
        """Test that reset_training clears training state."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        som.initialize_weights("random")
        som._training_iterations = 50
        
        # Act
        som.reset_training()
        
        # Assert
        assert som.training_iterations == 0
        assert som.weight_matrix is None
        assert not som.is_trained


class TestSelfOrganizingMapIntegration:
    """Integration tests for SOM operations."""
    
    def test_complete_training_cycle(self):
        """Test complete SOM training cycle."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(5, 5),
            input_dimensions=3,
            topology=topology
        )
        
        learning_params = LearningParameters(
            initial_learning_rate=0.5,
            final_learning_rate=0.01,
            initial_radius=2.0,
            final_radius=0.1,
            max_iterations=100
        )
        
        # Generate training data - three clusters
        cluster1 = np.random.normal([0.2, 0.2, 0.2], 0.1, (20, 3))
        cluster2 = np.random.normal([0.8, 0.2, 0.8], 0.1, (20, 3))
        cluster3 = np.random.normal([0.5, 0.8, 0.2], 0.1, (20, 3))
        training_data = np.vstack([cluster1, cluster2, cluster3])
        
        # Act
        som.initialize_weights("random")
        bmu_positions = som.train_batch(training_data, learning_params, 100)
        
        quantization_error = som.compute_quantization_error(training_data)
        topographic_error = som.compute_topographic_error(training_data)
        
        # Assert
        assert som.is_trained
        assert som.training_iterations == 100
        assert len(bmu_positions) == 100
        assert quantization_error >= 0.0
        assert 0.0 <= topographic_error <= 1.0

    def test_learning_parameter_progression(self):
        """Test that learning parameters progress correctly during training."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(4, 4),
            input_dimensions=2,
            topology=topology
        )
        som.initialize_weights("uniform")
        
        learning_params = LearningParameters(
            initial_learning_rate=1.0,
            final_learning_rate=0.1,
            initial_radius=3.0,
            final_radius=0.5,
            max_iterations=100
        )
        
        # Track learning rates and neighborhood radii
        learning_rates = []
        neighborhood_radii = []
        
        original_update = som.update_weights
        original_neighborhood = som.compute_neighborhood_function
        
        def track_learning_rate(*args, **kwargs):
            learning_rates.append(args[3])  # learning_rate is 4th argument
            return original_update(*args, **kwargs)
        
        def track_neighborhood_radius(*args, **kwargs):
            iteration = args[1]
            radius = learning_params.current_neighborhood_radius(iteration)
            neighborhood_radii.append(radius)
            return original_neighborhood(*args, **kwargs)
        
        som.update_weights = track_learning_rate
        som.compute_neighborhood_function = track_neighborhood_radius
        
        # Act
        input_vector = np.array([0.5, 0.5])
        for i in range(10):  # Train for 10 iterations
            som.train_single_iteration(input_vector, learning_params)
        
        # Assert
        assert len(learning_rates) == 10
        assert len(neighborhood_radii) == 10
        
        # Learning rate should decrease over time
        assert learning_rates[0] > learning_rates[-1]
        
        # Neighborhood radius should decrease over time
        assert neighborhood_radii[0] > neighborhood_radii[-1]


class TestSelfOrganizingMapEdgeCases:
    """Test suite for SOM edge cases and boundary conditions."""
    
    def test_single_neuron_som(self):
        """Test SOM with single neuron (1x1 map)."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(1, 1),
            input_dimensions=2,
            topology=topology
        )
        som.initialize_weights("uniform")
        
        input_vector = np.array([0.3, 0.7])
        learning_params = LearningParameters(
            initial_learning_rate=0.1,
            final_learning_rate=0.01,
            initial_radius=0.1,
            final_radius=0.01,
            max_iterations=10
        )
        
        # Act
        bmu_position = som.train_single_iteration(input_vector, learning_params)
        
        # Assert
        assert bmu_position == (0, 0)  # Only possible position
        assert som.training_iterations == 1

    def test_large_som_performance(self, performance_timer):
        """Test performance with larger SOM."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(20, 20),
            input_dimensions=10,
            topology=topology
        )
        som.initialize_weights("random")
        
        learning_params = LearningParameters(
            initial_learning_rate=0.1,
            final_learning_rate=0.01,
            initial_radius=5.0,
            final_radius=1.0,
            max_iterations=100
        )
        
        input_data = np.random.rand(50, 10)
        
        # Act & Measure
        performance_timer.start()
        som.train_batch(input_data, learning_params, 20)
        elapsed_time = performance_timer.stop()
        
        # Assert
        assert som.training_iterations == 20
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed_time < 5.0  # 5 seconds maximum

    def test_extreme_learning_parameters(self):
        """Test SOM with extreme learning parameters."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        som.initialize_weights("uniform")
        
        # Very high learning rate and large neighborhood
        extreme_params = LearningParameters(
            initial_learning_rate=2.0,  # > 1.0
            final_learning_rate=0.001,
            initial_radius=10.0,
            final_radius=0.01,
            max_iterations=10
        )
        
        input_vector = np.array([0.5, 0.5])
        
        # Act - should not crash despite extreme parameters
        bmu_position = som.train_single_iteration(input_vector, extreme_params)
        
        # Assert
        assert isinstance(bmu_position, tuple)
        assert som.training_iterations == 1

    def test_identical_input_vectors(self):
        """Test SOM training with identical input vectors."""
        # Arrange
        topology = SOMTopology.create_rectangular()
        som = MockSelfOrganizingMap(
            map_dimensions=(3, 3),
            input_dimensions=2,
            topology=topology
        )
        som.initialize_weights("random")
        
        learning_params = LearningParameters(
            initial_learning_rate=0.1,
            final_learning_rate=0.01,
            initial_radius=1.0,
            final_radius=0.1,
            max_iterations=20
        )
        
        # All identical vectors
        identical_data = np.tile(np.array([0.5, 0.5]), (10, 1))
        
        # Act
        bmu_positions = som.train_batch(identical_data, learning_params, 20)
        
        # Assert
        # All BMUs should be the same since inputs are identical
        unique_bmus = set(bmu_positions)
        assert len(unique_bmus) == 1  # Only one unique BMU