"""
SOM Topology Value Object.

Immutable representation of Self-Organizing Map topology configuration
including grid structure, neighborhood functions, and distance metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Callable
from enum import Enum


class TopologyType(Enum):
    """Enumeration of supported SOM topology types."""
    RECTANGULAR = "rectangular"
    HEXAGONAL = "hexagonal"
    CYLINDRICAL = "cylindrical"
    TOROIDAL = "toroidal"


class DistanceMetric(Enum):
    """Enumeration of distance metrics for SOM."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"


class NeighborhoodFunction(Enum):
    """Enumeration of neighborhood functions."""
    GAUSSIAN = "gaussian"
    MEXICAN_HAT = "mexican_hat"
    BUBBLE = "bubble"
    TRIANGULAR = "triangular"


@dataclass(frozen=True)
class SOMTopology:
    """
    Immutable representation of SOM topology configuration.
    
    Encapsulates all topological aspects of the Self-Organizing Map
    including grid structure, neighborhood relationships, and
    distance calculations.
    """
    
    topology_type: TopologyType
    distance_metric: DistanceMetric
    neighborhood_function: NeighborhoodFunction
    wrap_around: bool = field(default=False)
    periodic_boundary: bool = field(default=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate topology configuration."""
        self._validate_topology_consistency()

    def _validate_topology_consistency(self) -> None:
        """Validate that topology settings are consistent."""
        # Toroidal topology requires wrap around
        if self.topology_type == TopologyType.TOROIDAL and not self.wrap_around:
            raise ValueError("Toroidal topology requires wrap_around=True")
        
        # Cylindrical topology should have periodic boundary
        if self.topology_type == TopologyType.CYLINDRICAL and not self.periodic_boundary:
            raise ValueError("Cylindrical topology should have periodic_boundary=True")

    @property
    def is_periodic(self) -> bool:
        """Check if topology has periodic boundaries."""
        return self.periodic_boundary or self.topology_type in {
            TopologyType.CYLINDRICAL, 
            TopologyType.TOROIDAL
        }

    @property
    def supports_wrap_around(self) -> bool:
        """Check if topology supports wrap-around connections."""
        return self.wrap_around or self.topology_type == TopologyType.TOROIDAL

    def calculate_grid_distance(
        self, 
        pos1: Tuple[int, int], 
        pos2: Tuple[int, int],
        map_dimensions: Tuple[int, int]
    ) -> float:
        """
        Calculate distance between two grid positions.
        
        Args:
            pos1: First grid position (x, y)
            pos2: Second grid position (x, y)
            map_dimensions: Map dimensions (width, height)
            
        Returns:
            Distance according to the configured metric and topology
        """
        x1, y1 = pos1
        x2, y2 = pos2
        width, height = map_dimensions
        
        # Calculate raw differences
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        # Apply wrap-around if supported
        if self.supports_wrap_around:
            dx = min(dx, width - dx)
            dy = min(dy, height - dy)
        
        # Apply distance metric
        if self.distance_metric == DistanceMetric.EUCLIDEAN:
            return (dx**2 + dy**2)**0.5
        elif self.distance_metric == DistanceMetric.MANHATTAN:
            return dx + dy
        elif self.distance_metric == DistanceMetric.CHEBYSHEV:
            return max(dx, dy)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def get_neighborhood_coordinates(
        self, 
        center: Tuple[int, int], 
        radius: float,
        map_dimensions: Tuple[int, int]
    ) -> list[Tuple[int, int]]:
        """
        Get all coordinates within neighborhood radius of center.
        
        Args:
            center: Center position (x, y)
            radius: Neighborhood radius
            map_dimensions: Map dimensions (width, height)
            
        Returns:
            List of coordinates within the neighborhood
        """
        cx, cy = center
        width, height = map_dimensions
        neighbors = []
        
        # Search within bounding box
        search_radius = int(radius) + 1
        
        for x in range(max(0, cx - search_radius), min(width, cx + search_radius + 1)):
            for y in range(max(0, cy - search_radius), min(height, cy + search_radius + 1)):
                distance = self.calculate_grid_distance((cx, cy), (x, y), map_dimensions)
                if distance <= radius:
                    neighbors.append((x, y))
        
        return neighbors

    def compute_neighborhood_strength(
        self, 
        distance: float, 
        radius: float,
        learning_iteration: int = 0
    ) -> float:
        """
        Compute neighborhood function strength based on distance.
        
        Args:
            distance: Distance from center
            radius: Current neighborhood radius
            learning_iteration: Current learning iteration (for adaptive functions)
            
        Returns:
            Neighborhood strength [0, 1]
        """
        if distance > radius:
            return 0.0
        
        if self.neighborhood_function == NeighborhoodFunction.GAUSSIAN:
            return self._gaussian_neighborhood(distance, radius)
        elif self.neighborhood_function == NeighborhoodFunction.MEXICAN_HAT:
            return self._mexican_hat_neighborhood(distance, radius)
        elif self.neighborhood_function == NeighborhoodFunction.BUBBLE:
            return 1.0  # Constant within radius
        elif self.neighborhood_function == NeighborhoodFunction.TRIANGULAR:
            return self._triangular_neighborhood(distance, radius)
        else:
            raise ValueError(f"Unknown neighborhood function: {self.neighborhood_function}")

    def _gaussian_neighborhood(self, distance: float, radius: float) -> float:
        """Gaussian neighborhood function."""
        sigma = radius / 3.0  # Standard deviation
        return float(np.exp(-(distance**2) / (2 * sigma**2)))

    def _mexican_hat_neighborhood(self, distance: float, radius: float) -> float:
        """Mexican hat (difference of Gaussians) neighborhood function."""
        sigma1 = radius / 3.0
        sigma2 = radius / 1.5
        
        gauss1 = np.exp(-(distance**2) / (2 * sigma1**2))
        gauss2 = 0.5 * np.exp(-(distance**2) / (2 * sigma2**2))
        
        return float(max(0.0, gauss1 - gauss2))

    def _triangular_neighborhood(self, distance: float, radius: float) -> float:
        """Triangular (linear decay) neighborhood function."""
        return float(max(0.0, 1.0 - distance / radius))

    def get_grid_coordinates_for_topology(
        self, 
        map_dimensions: Tuple[int, int]
    ) -> list[Tuple[int, int]]:
        """
        Get all valid grid coordinates for the topology.
        
        Args:
            map_dimensions: Map dimensions (width, height)
            
        Returns:
            List of all valid grid coordinates
        """
        width, height = map_dimensions
        coordinates = []
        
        if self.topology_type == TopologyType.HEXAGONAL:
            # Hexagonal grid has offset rows
            for y in range(height):
                for x in range(width):
                    if y % 2 == 0:  # Even rows
                        coordinates.append((x, y))
                    else:  # Odd rows - offset by half unit
                        if x < width - 1:  # Avoid edge issues
                            coordinates.append((x, y))
        else:
            # Standard rectangular grid for other topologies
            for y in range(height):
                for x in range(width):
                    coordinates.append((x, y))
        
        return coordinates

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert topology to dictionary representation.
        
        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "topology_type": self.topology_type.value,
            "distance_metric": self.distance_metric.value,
            "neighborhood_function": self.neighborhood_function.value,
            "wrap_around": self.wrap_around,
            "periodic_boundary": self.periodic_boundary,
            "metadata": self.metadata,
            "is_periodic": self.is_periodic,
            "supports_wrap_around": self.supports_wrap_around
        }

    @classmethod
    def create_rectangular(cls, distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN) -> 'SOMTopology':
        """
        Create rectangular topology configuration.
        
        Args:
            distance_metric: Distance metric to use
            
        Returns:
            SOMTopology with rectangular configuration
        """
        return cls(
            topology_type=TopologyType.RECTANGULAR,
            distance_metric=distance_metric,
            neighborhood_function=NeighborhoodFunction.GAUSSIAN,
            wrap_around=False,
            periodic_boundary=False,
            metadata={"standard": "rectangular_grid"}
        )

    @classmethod
    def create_hexagonal(cls) -> 'SOMTopology':
        """
        Create hexagonal topology configuration.
        
        Returns:
            SOMTopology with hexagonal configuration
        """
        return cls(
            topology_type=TopologyType.HEXAGONAL,
            distance_metric=DistanceMetric.EUCLIDEAN,
            neighborhood_function=NeighborhoodFunction.GAUSSIAN,
            wrap_around=False,
            periodic_boundary=False,
            metadata={"standard": "hexagonal_grid"}
        )

    @classmethod
    def create_toroidal(cls) -> 'SOMTopology':
        """
        Create toroidal topology configuration.
        
        Returns:
            SOMTopology with toroidal configuration
        """
        return cls(
            topology_type=TopologyType.TOROIDAL,
            distance_metric=DistanceMetric.EUCLIDEAN,
            neighborhood_function=NeighborhoodFunction.GAUSSIAN,
            wrap_around=True,
            periodic_boundary=True,
            metadata={"standard": "toroidal_grid"}
        )


# Import numpy here to avoid circular imports
import numpy as np