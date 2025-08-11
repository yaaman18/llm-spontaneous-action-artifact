"""
Consciousness Repository Interface.

Abstract repository interface for persistence and retrieval of consciousness
states and related data. Follows Repository pattern and Dependency Inversion
Principle to decouple domain logic from infrastructure concerns.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from ..value_objects.consciousness_state import ConsciousnessState
from ..value_objects.phi_value import PhiValue


class ConsciousnessRepository(ABC):
    """
    Abstract repository interface for consciousness state persistence.
    
    This interface defines the contract for storing and retrieving
    consciousness states, following the Repository pattern to abstract
    persistence mechanisms from domain logic.
    
    Implementations must ensure:
    - Data integrity and consistency
    - Efficient querying capabilities
    - Thread-safe operations
    - Proper error handling
    """

    @abstractmethod
    async def save_consciousness_state(self, state: ConsciousnessState) -> str:
        """
        Save a consciousness state to the repository.
        
        Args:
            state: ConsciousnessState to persist
            
        Returns:
            Unique identifier for the saved state
            
        Raises:
            RepositoryError: If save operation fails
        """
        pass

    @abstractmethod
    async def get_consciousness_state(self, state_id: str) -> Optional[ConsciousnessState]:
        """
        Retrieve a consciousness state by its identifier.
        
        Args:
            state_id: Unique identifier of the state
            
        Returns:
            ConsciousnessState if found, None otherwise
            
        Raises:
            RepositoryError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def get_latest_consciousness_state(self) -> Optional[ConsciousnessState]:
        """
        Get the most recent consciousness state.
        
        Returns:
            Latest ConsciousnessState if available, None otherwise
            
        Raises:
            RepositoryError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def get_consciousness_states_in_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> List[ConsciousnessState]:
        """
        Retrieve consciousness states within a time range.
        
        Args:
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            limit: Maximum number of states to return
            
        Returns:
            List of ConsciousnessState objects in chronological order
            
        Raises:
            RepositoryError: If retrieval operation fails
            ValueError: If time range is invalid
        """
        pass

    @abstractmethod
    async def get_conscious_states_only(
        self,
        limit: Optional[int] = None
    ) -> List[ConsciousnessState]:
        """
        Retrieve only states that indicate consciousness (is_conscious=True).
        
        Args:
            limit: Maximum number of states to return
            
        Returns:
            List of conscious ConsciousnessState objects
            
        Raises:
            RepositoryError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def get_consciousness_statistics(
        self,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get statistical summary of consciousness states.
        
        Args:
            time_window: Time window for statistics (None for all time)
            
        Returns:
            Dictionary containing statistical metrics:
            - total_states: Total number of states
            - conscious_ratio: Ratio of conscious states
            - avg_phi_value: Average Φ value
            - avg_consciousness_level: Average consciousness level
            - consciousness_duration: Time spent conscious
            
        Raises:
            RepositoryError: If calculation fails
        """
        pass

    @abstractmethod
    async def search_by_phi_range(
        self,
        min_phi: float,
        max_phi: float,
        limit: Optional[int] = None
    ) -> List[ConsciousnessState]:
        """
        Search consciousness states by Φ value range.
        
        Args:
            min_phi: Minimum Φ value (inclusive)
            max_phi: Maximum Φ value (inclusive)  
            limit: Maximum number of states to return
            
        Returns:
            List of ConsciousnessState objects with Φ in range
            
        Raises:
            RepositoryError: If search operation fails
            ValueError: If Φ range is invalid
        """
        pass

    @abstractmethod
    async def search_by_consciousness_level(
        self,
        min_level: float,
        max_level: float,
        limit: Optional[int] = None
    ) -> List[ConsciousnessState]:
        """
        Search consciousness states by consciousness level range.
        
        Args:
            min_level: Minimum consciousness level [0, 1]
            max_level: Maximum consciousness level [0, 1]
            limit: Maximum number of states to return
            
        Returns:
            List of ConsciousnessState objects with level in range
            
        Raises:
            RepositoryError: If search operation fails
            ValueError: If level range is invalid
        """
        pass

    @abstractmethod
    async def search_by_phenomenological_markers(
        self,
        markers: Dict[str, Any],
        match_all: bool = True,
        limit: Optional[int] = None
    ) -> List[ConsciousnessState]:
        """
        Search consciousness states by phenomenological markers.
        
        Args:
            markers: Dictionary of marker key-value pairs to search for
            match_all: If True, all markers must match; if False, any match
            limit: Maximum number of states to return
            
        Returns:
            List of ConsciousnessState objects matching markers
            
        Raises:
            RepositoryError: If search operation fails
        """
        pass

    @abstractmethod
    async def delete_consciousness_state(self, state_id: str) -> bool:
        """
        Delete a consciousness state from the repository.
        
        Args:
            state_id: Unique identifier of the state to delete
            
        Returns:
            True if deletion successful, False if state not found
            
        Raises:
            RepositoryError: If deletion operation fails
        """
        pass

    @abstractmethod
    async def delete_states_older_than(self, cutoff_time: datetime) -> int:
        """
        Delete consciousness states older than specified time.
        
        Args:
            cutoff_time: Delete states older than this time
            
        Returns:
            Number of states deleted
            
        Raises:
            RepositoryError: If deletion operation fails
        """
        pass

    @abstractmethod
    async def get_state_count(self) -> int:
        """
        Get total count of consciousness states in repository.
        
        Returns:
            Total number of stored consciousness states
            
        Raises:
            RepositoryError: If count operation fails
        """
        pass

    @abstractmethod
    async def export_states_to_format(
        self,
        format_type: str,
        states: Optional[List[ConsciousnessState]] = None,
        filepath: Optional[str] = None
    ) -> str:
        """
        Export consciousness states to specified format.
        
        Args:
            format_type: Export format ("json", "csv", "parquet", etc.)
            states: Specific states to export (None for all)
            filepath: Output file path (None for string return)
            
        Returns:
            Exported data as string (if filepath is None)
            
        Raises:
            RepositoryError: If export operation fails
            ValueError: If format is not supported
        """
        pass

    @abstractmethod
    async def import_states_from_format(
        self,
        format_type: str,
        data_source: str
    ) -> int:
        """
        Import consciousness states from specified format.
        
        Args:
            format_type: Import format ("json", "csv", "parquet", etc.)
            data_source: Data source (file path or string data)
            
        Returns:
            Number of states imported
            
        Raises:
            RepositoryError: If import operation fails
            ValueError: If format is not supported or data is invalid
        """
        pass

    @abstractmethod
    async def create_index(self, field_name: str) -> bool:
        """
        Create index on specified field for faster queries.
        
        Args:
            field_name: Name of field to index
            
        Returns:
            True if index created successfully
            
        Raises:
            RepositoryError: If index creation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the repository.
        
        Returns:
            Dictionary with health status information:
            - status: "healthy", "degraded", or "unhealthy"
            - response_time: Query response time in milliseconds
            - storage_usage: Storage space usage information
            - error_count: Number of recent errors
            
        Raises:
            RepositoryError: If health check fails
        """
        pass


class RepositoryError(Exception):
    """
    Base exception for repository operations.
    
    Raised when repository operations fail due to infrastructure
    issues, data corruption, or other persistence-related problems.
    """
    
    def __init__(self, message: str, operation: Optional[str] = None, cause: Optional[Exception] = None):
        """
        Initialize repository error.
        
        Args:
            message: Error description
            operation: Repository operation that failed
            cause: Underlying exception that caused this error
        """
        super().__init__(message)
        self.operation = operation
        self.cause = cause

    def __str__(self) -> str:
        """String representation of the error."""
        base_msg = super().__str__()
        if self.operation:
            return f"Repository operation '{self.operation}' failed: {base_msg}"
        return f"Repository error: {base_msg}"