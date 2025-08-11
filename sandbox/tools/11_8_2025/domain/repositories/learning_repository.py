"""
Repository interface for learning data persistence.

This repository handles the persistence and retrieval of learning parameters,
training histories, and model checkpoints.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..value_objects.learning_parameters import LearningParameters


class LearningRepository(ABC):
    """
    Abstract repository for learning data persistence.
    
    This repository provides an abstraction for storing and retrieving
    learning parameters, training histories, and model states.
    """
    
    @abstractmethod
    async def save_learning_parameters(
        self,
        params: LearningParameters,
        session_id: str,
        epoch: int
    ) -> None:
        """
        Save learning parameters for a specific epoch.
        
        Args:
            params: Learning parameters to save
            session_id: Unique session identifier
            epoch: Training epoch number
        """
        pass
    
    @abstractmethod
    async def get_learning_parameters(
        self,
        session_id: str,
        epoch: Optional[int] = None
    ) -> Optional[LearningParameters]:
        """
        Get learning parameters for a session and epoch.
        
        Args:
            session_id: Session identifier
            epoch: Specific epoch (None for latest)
            
        Returns:
            Learning parameters or None if not found
        """
        pass
    
    @abstractmethod
    async def save_model_checkpoint(
        self,
        model_state: Dict[str, Any],
        session_id: str,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Save a model checkpoint with associated metrics.
        
        Args:
            model_state: Serialized model state
            session_id: Session identifier
            epoch: Training epoch
            metrics: Performance metrics
        """
        pass
    
    @abstractmethod
    async def load_model_checkpoint(
        self,
        session_id: str,
        epoch: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load a model checkpoint.
        
        Args:
            session_id: Session identifier
            epoch: Specific epoch (None for latest)
            
        Returns:
            Model state dictionary or None if not found
        """
        pass
    
    @abstractmethod
    async def get_training_history(
        self,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get complete training history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of training records with metrics
        """
        pass
    
    @abstractmethod
    async def delete_session_data(self, session_id: str) -> None:
        """
        Delete all learning data for a session.
        
        Args:
            session_id: Session identifier to delete
        """
        pass