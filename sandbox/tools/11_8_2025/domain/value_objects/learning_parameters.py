"""
Learning Parameters Value Object.

Immutable representation of learning parameters for adaptive algorithms
including learning rates, decay schedules, and adaptation strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional
from enum import Enum
import math


class LearningSchedule(Enum):
    """Enumeration of learning rate schedules."""
    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    INVERSE_TIME_DECAY = "inverse_time_decay"
    COSINE_ANNEALING = "cosine_annealing"
    STEP_DECAY = "step_decay"


@dataclass(frozen=True)
class LearningParameters:
    """
    Immutable representation of learning parameters.
    
    Encapsulates all parameters controlling the learning process
    including rates, schedules, and adaptive mechanisms.
    """
    
    initial_learning_rate: float
    final_learning_rate: float
    initial_radius: float
    final_radius: float
    learning_schedule: LearningSchedule = field(default=LearningSchedule.EXPONENTIAL_DECAY)
    radius_schedule: LearningSchedule = field(default=LearningSchedule.EXPONENTIAL_DECAY)
    max_iterations: int = field(default=1000)
    decay_rate: float = field(default=0.01)
    step_size: int = field(default=100)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate learning parameters."""
        self._validate_learning_rates()
        self._validate_radii()
        self._validate_iterations()
        self._validate_decay_parameters()

    def _validate_learning_rates(self) -> None:
        """Validate learning rate parameters."""
        if self.initial_learning_rate <= 0:
            raise ValueError("Initial learning rate must be positive")
        if self.final_learning_rate < 0:
            raise ValueError("Final learning rate must be non-negative")
        if self.final_learning_rate > self.initial_learning_rate:
            raise ValueError("Final learning rate should not exceed initial rate")

    def _validate_radii(self) -> None:
        """Validate radius parameters."""
        if self.initial_radius <= 0:
            raise ValueError("Initial radius must be positive")
        if self.final_radius < 0:
            raise ValueError("Final radius must be non-negative")
        if self.final_radius > self.initial_radius:
            raise ValueError("Final radius should not exceed initial radius")

    def _validate_iterations(self) -> None:
        """Validate iteration parameters."""
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")

    def _validate_decay_parameters(self) -> None:
        """Validate decay parameters."""
        if self.decay_rate <= 0:
            raise ValueError("Decay rate must be positive")
        if self.step_size <= 0:
            raise ValueError("Step size must be positive")

    def current_learning_rate(self, iteration: int) -> float:
        """
        Calculate current learning rate based on iteration.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Current learning rate
        """
        if iteration >= self.max_iterations:
            return self.final_learning_rate
        
        progress = iteration / self.max_iterations
        
        if self.learning_schedule == LearningSchedule.CONSTANT:
            return self.initial_learning_rate
        elif self.learning_schedule == LearningSchedule.LINEAR_DECAY:
            return self._linear_decay(
                self.initial_learning_rate, 
                self.final_learning_rate, 
                progress
            )
        elif self.learning_schedule == LearningSchedule.EXPONENTIAL_DECAY:
            return self._exponential_decay(
                self.initial_learning_rate,
                self.final_learning_rate,
                progress,
                self.decay_rate
            )
        elif self.learning_schedule == LearningSchedule.INVERSE_TIME_DECAY:
            return self._inverse_time_decay(
                self.initial_learning_rate,
                iteration,
                self.decay_rate
            )
        elif self.learning_schedule == LearningSchedule.COSINE_ANNEALING:
            return self._cosine_annealing(
                self.initial_learning_rate,
                self.final_learning_rate,
                progress
            )
        elif self.learning_schedule == LearningSchedule.STEP_DECAY:
            return self._step_decay(
                self.initial_learning_rate,
                iteration,
                self.step_size,
                self.decay_rate
            )
        else:
            raise ValueError(f"Unknown learning schedule: {self.learning_schedule}")

    def current_radius(self, iteration: int) -> float:
        """
        Calculate current neighborhood radius based on iteration.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Current neighborhood radius
        """
        if iteration >= self.max_iterations:
            return self.final_radius
        
        progress = iteration / self.max_iterations
        
        if self.radius_schedule == LearningSchedule.CONSTANT:
            return self.initial_radius
        elif self.radius_schedule == LearningSchedule.LINEAR_DECAY:
            return self._linear_decay(
                self.initial_radius,
                self.final_radius,
                progress
            )
        elif self.radius_schedule == LearningSchedule.EXPONENTIAL_DECAY:
            return self._exponential_decay(
                self.initial_radius,
                self.final_radius,
                progress,
                self.decay_rate
            )
        elif self.radius_schedule == LearningSchedule.INVERSE_TIME_DECAY:
            return self._inverse_time_decay(
                self.initial_radius,
                iteration,
                self.decay_rate
            )
        elif self.radius_schedule == LearningSchedule.COSINE_ANNEALING:
            return self._cosine_annealing(
                self.initial_radius,
                self.final_radius,
                progress
            )
        else:
            raise ValueError(f"Unknown radius schedule: {self.radius_schedule}")

    def _linear_decay(self, initial: float, final: float, progress: float) -> float:
        """Linear decay schedule."""
        return initial - (initial - final) * progress

    def _exponential_decay(
        self, 
        initial: float, 
        final: float, 
        progress: float, 
        decay_rate: float
    ) -> float:
        """Exponential decay schedule."""
        decay_factor = math.exp(-decay_rate * progress * self.max_iterations)
        return final + (initial - final) * decay_factor

    def _inverse_time_decay(
        self, 
        initial: float, 
        iteration: int, 
        decay_rate: float
    ) -> float:
        """Inverse time decay schedule."""
        return initial / (1.0 + decay_rate * iteration)

    def _cosine_annealing(
        self, 
        initial: float, 
        final: float, 
        progress: float
    ) -> float:
        """Cosine annealing schedule."""
        return final + (initial - final) * (1 + math.cos(math.pi * progress)) / 2

    def _step_decay(
        self, 
        initial: float, 
        iteration: int, 
        step_size: int, 
        decay_rate: float
    ) -> float:
        """Step decay schedule."""
        steps = iteration // step_size
        return initial * (decay_rate ** steps)

    def get_parameter_trajectory(self) -> Dict[str, list]:
        """
        Get the complete trajectory of learning parameters.
        
        Returns:
            Dictionary with learning rate and radius trajectories
        """
        iterations = range(self.max_iterations + 1)
        
        return {
            "iterations": list(iterations),
            "learning_rates": [self.current_learning_rate(i) for i in iterations],
            "radii": [self.current_radius(i) for i in iterations]
        }

    def with_updated_schedule(
        self, 
        learning_schedule: Optional[LearningSchedule] = None,
        radius_schedule: Optional[LearningSchedule] = None
    ) -> 'LearningParameters':
        """
        Create new parameters with updated schedules.
        
        Args:
            learning_schedule: New learning rate schedule
            radius_schedule: New radius schedule
            
        Returns:
            New LearningParameters with updated schedules
        """
        return LearningParameters(
            initial_learning_rate=self.initial_learning_rate,
            final_learning_rate=self.final_learning_rate,
            initial_radius=self.initial_radius,
            final_radius=self.final_radius,
            learning_schedule=learning_schedule or self.learning_schedule,
            radius_schedule=radius_schedule or self.radius_schedule,
            max_iterations=self.max_iterations,
            decay_rate=self.decay_rate,
            step_size=self.step_size,
            metadata=self.metadata.copy()
        )

    def with_updated_iterations(self, new_max_iterations: int) -> 'LearningParameters':
        """
        Create new parameters with updated max iterations.
        
        Args:
            new_max_iterations: New maximum iteration count
            
        Returns:
            New LearningParameters with updated iterations
        """
        return LearningParameters(
            initial_learning_rate=self.initial_learning_rate,
            final_learning_rate=self.final_learning_rate,
            initial_radius=self.initial_radius,
            final_radius=self.final_radius,
            learning_schedule=self.learning_schedule,
            radius_schedule=self.radius_schedule,
            max_iterations=new_max_iterations,
            decay_rate=self.decay_rate,
            step_size=self.step_size,
            metadata=self.metadata.copy()
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert learning parameters to dictionary.
        
        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "initial_radius": self.initial_radius,
            "final_radius": self.final_radius,
            "learning_schedule": self.learning_schedule.value,
            "radius_schedule": self.radius_schedule.value,
            "max_iterations": self.max_iterations,
            "decay_rate": self.decay_rate,
            "step_size": self.step_size,
            "metadata": self.metadata
        }

    @classmethod
    def create_som_defaults(cls) -> 'LearningParameters':
        """
        Create default parameters for SOM training.
        
        Returns:
            LearningParameters with SOM-appropriate defaults
        """
        return cls(
            initial_learning_rate=0.5,
            final_learning_rate=0.01,
            initial_radius=5.0,
            final_radius=0.5,
            learning_schedule=LearningSchedule.EXPONENTIAL_DECAY,
            radius_schedule=LearningSchedule.EXPONENTIAL_DECAY,
            max_iterations=1000,
            decay_rate=0.01,
            step_size=100,
            metadata={"algorithm": "som", "defaults": True}
        )

    @classmethod
    def create_predictive_coding_defaults(cls) -> 'LearningParameters':
        """
        Create default parameters for predictive coding.
        
        Returns:
            LearningParameters with predictive coding defaults
        """
        return cls(
            initial_learning_rate=0.01,
            final_learning_rate=0.001,
            initial_radius=2.0,
            final_radius=0.1,
            learning_schedule=LearningSchedule.EXPONENTIAL_DECAY,
            radius_schedule=LearningSchedule.LINEAR_DECAY,
            max_iterations=5000,
            decay_rate=0.001,
            step_size=500,
            metadata={"algorithm": "predictive_coding", "defaults": True}
        )