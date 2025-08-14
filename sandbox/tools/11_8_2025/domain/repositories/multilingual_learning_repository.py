"""
Multilingual Learning Repository Interface.

Repository contract for persisting and retrieving multilingual learning data,
language clusters, and tokenization models. Follows the Repository pattern
for clean separation between domain logic and persistence concerns.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from ..value_objects.language_cluster import LanguageCluster
from ..entities.multilingual_tokenizer import MultilingualTokenizer


class MultilingualLearningRepository(ABC):
    """
    Abstract repository for multilingual learning data persistence.
    
    This repository provides an abstraction for storing and retrieving
    language clusters, tokenizer states, learning progress, and
    multilingual corpus data. Follows the Interface Segregation Principle
    by providing specific methods for multilingual learning concerns.
    
    Domain Responsibilities:
    - Language cluster persistence and retrieval
    - Tokenizer state management
    - Learning session tracking
    - Corpus data storage and indexing
    """
    
    @abstractmethod
    async def save_language_cluster(
        self,
        cluster: LanguageCluster,
        session_id: str
    ) -> None:
        """
        Save a language cluster to persistent storage.
        
        Args:
            cluster: LanguageCluster to persist
            session_id: Unique session identifier
            
        Raises:
            RepositoryError: If persistence fails
        """
        pass
    
    @abstractmethod
    async def get_language_cluster(
        self,
        cluster_id: str,
        session_id: str
    ) -> Optional[LanguageCluster]:
        """
        Retrieve a specific language cluster.
        
        Args:
            cluster_id: Unique cluster identifier
            session_id: Session identifier
            
        Returns:
            LanguageCluster instance or None if not found
        """
        pass
    
    @abstractmethod
    async def get_all_language_clusters(
        self,
        session_id: str
    ) -> List[LanguageCluster]:
        """
        Retrieve all language clusters for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of all LanguageCluster instances
        """
        pass
    
    @abstractmethod
    async def delete_language_cluster(
        self,
        cluster_id: str,
        session_id: str
    ) -> bool:
        """
        Delete a specific language cluster.
        
        Args:
            cluster_id: Cluster identifier to delete
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def save_tokenizer_state(
        self,
        tokenizer: MultilingualTokenizer,
        session_id: str,
        checkpoint_name: str
    ) -> None:
        """
        Save complete tokenizer state including all clusters.
        
        Args:
            tokenizer: MultilingualTokenizer to persist
            session_id: Session identifier
            checkpoint_name: Name for this checkpoint
        """
        pass
    
    @abstractmethod
    async def load_tokenizer_state(
        self,
        session_id: str,
        checkpoint_name: Optional[str] = None
    ) -> Optional[MultilingualTokenizer]:
        """
        Load tokenizer state from storage.
        
        Args:
            session_id: Session identifier
            checkpoint_name: Specific checkpoint (None for latest)
            
        Returns:
            Restored MultilingualTokenizer or None if not found
        """
        pass
    
    @abstractmethod
    async def save_learning_progress(
        self,
        session_id: str,
        text_sample: str,
        tokens: List[str],
        cluster_id: str,
        learning_metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record learning progress for analysis and replay.
        
        Args:
            session_id: Session identifier
            text_sample: Original text that was tokenized
            tokens: Resulting tokens
            cluster_id: Language cluster used
            learning_metrics: Performance metrics
            timestamp: When learning occurred (defaults to now)
        """
        pass
    
    @abstractmethod
    async def get_learning_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        cluster_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve learning history records.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of records (None for all)
            cluster_id: Filter by specific cluster (None for all)
            
        Returns:
            List of learning records with metadata
        """
        pass
    
    @abstractmethod
    async def save_corpus_sample(
        self,
        text: str,
        language_hint: Optional[str],
        source: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Store a text sample in the multilingual corpus.
        
        Args:
            text: Text content to store
            language_hint: Suggested language (if known)
            source: Source identifier (file, URL, etc.)
            metadata: Additional metadata about the sample
            
        Returns:
            Unique corpus sample identifier
        """
        pass
    
    @abstractmethod
    async def get_corpus_samples(
        self,
        language_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve corpus samples with optional filtering.
        
        Args:
            language_filter: Filter by language
            source_filter: Filter by source
            limit: Maximum number of samples
            
        Returns:
            List of corpus samples with metadata
        """
        pass
    
    @abstractmethod
    async def find_similar_clusters(
        self,
        reference_cluster: LanguageCluster,
        similarity_threshold: float = 0.8,
        session_id: Optional[str] = None
    ) -> List[Tuple[LanguageCluster, float]]:
        """
        Find clusters similar to the reference cluster.
        
        Args:
            reference_cluster: Cluster to find similarities for
            similarity_threshold: Minimum similarity score
            session_id: Session to search in (None for all sessions)
            
        Returns:
            List of (cluster, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    async def get_cluster_statistics(
        self,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics about stored clusters.
        
        Args:
            session_id: Session to analyze (None for all sessions)
            
        Returns:
            Statistics dictionary with cluster metrics
        """
        pass
    
    @abstractmethod
    async def cleanup_old_sessions(
        self,
        cutoff_date: datetime,
        preserve_checkpoints: bool = True
    ) -> int:
        """
        Clean up old session data to manage storage.
        
        Args:
            cutoff_date: Delete sessions older than this date
            preserve_checkpoints: Keep tokenizer checkpoints
            
        Returns:
            Number of sessions cleaned up
        """
        pass
    
    @abstractmethod
    async def export_session_data(
        self,
        session_id: str,
        export_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export complete session data for backup or analysis.
        
        Args:
            session_id: Session to export
            export_format: Export format ("json", "pickle", etc.)
            
        Returns:
            Exported data structure
        """
        pass
    
    @abstractmethod
    async def import_session_data(
        self,
        session_data: Dict[str, Any],
        new_session_id: Optional[str] = None
    ) -> str:
        """
        Import session data from backup.
        
        Args:
            session_data: Data structure to import
            new_session_id: New session ID (None to generate)
            
        Returns:
            Imported session identifier
        """
        pass


class RepositoryError(Exception):
    """Base exception for repository operations."""
    pass


class ClusterNotFoundError(RepositoryError):
    """Raised when a requested cluster is not found."""
    pass


class SessionNotFoundError(RepositoryError):
    """Raised when a requested session is not found."""
    pass


class PersistenceError(RepositoryError):
    """Raised when persistence operations fail."""
    pass