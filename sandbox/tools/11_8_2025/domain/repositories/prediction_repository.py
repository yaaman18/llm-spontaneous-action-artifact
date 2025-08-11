"""
Repository interface for prediction state persistence.

This repository handles the persistence and retrieval of prediction states,
prediction histories, and learning trajectories.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..value_objects.prediction_state import PredictionState


class PredictionRepository(ABC):
    """
    Abstract repository for prediction state persistence.
    
    This repository provides an abstraction for storing and retrieving
    prediction states, enabling analysis of learning trajectories and
    prediction performance over time.
    """
    
    @abstractmethod
    async def save_prediction_state(
        self, 
        state: PredictionState,
        session_id: str
    ) -> None:
        """
        Save a prediction state to the repository.
        
        Args:
            state: The prediction state to save
            session_id: Unique identifier for the learning session
        """
        pass
    
    @abstractmethod
    async def get_prediction_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[PredictionState]:
        """
        Get prediction history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of states to return
            
        Returns:
            List of prediction states in chronological order
        """
        pass
    
    @abstractmethod
    async def get_latest_prediction_state(
        self,
        session_id: str
    ) -> Optional[PredictionState]:
        """
        Get the most recent prediction state for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Latest prediction state or None if no states exist
        """
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """
        Delete all prediction states for a session.
        
        Args:
            session_id: Session identifier to delete
        """
        pass
    
    @abstractmethod
    async def get_session_metrics(
        self, 
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics for a learning session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of session metrics (convergence, performance, etc.)
        """
        pass