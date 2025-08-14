"""
Multilingual Learning Application Service.

Application service that coordinates use cases and provides a high-level
interface for multilingual learning operations. Follows the Application
Service pattern to orchestrate complex workflows.
"""

import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..use_cases.learn_from_text_use_case import LearnFromTextUseCase
from ..use_cases.tokenize_text_use_case import TokenizeTextUseCase
from ...domain.entities.multilingual_tokenizer import MultilingualTokenizer
from ...domain.repositories.multilingual_learning_repository import (
    MultilingualLearningRepository,
    RepositoryError
)
from ...domain.value_objects.consciousness_state import ConsciousnessState
from ...domain.value_objects.language_cluster import LanguageCluster
from ...infrastructure.config.system_config import SystemConfig


class MultilingualLearningService:
    """
    Application service for multilingual learning operations.
    
    This service provides a high-level interface for all multilingual
    learning operations, coordinating between use cases and maintaining
    system state. It implements the Facade pattern for complex domain
    operations while ensuring proper separation of concerns.
    
    Responsibilities:
    - Coordinate learning and tokenization workflows
    - Manage tokenizer lifecycle and state
    - Provide unified interface for client applications
    - Handle cross-cutting concerns like error handling and logging
    - Integrate with consciousness framework
    """
    
    def __init__(
        self,
        repository: MultilingualLearningRepository,
        config: SystemConfig
    ):
        """
        Initialize the service with dependencies.
        
        Args:
            repository: Repository for persistence operations
            config: System configuration
        """
        self.repository = repository
        self.config = config
        
        # Initialize use cases
        self.learn_from_text_use_case = LearnFromTextUseCase(repository, config)
        self.tokenize_text_use_case = TokenizeTextUseCase(repository, config)
        
        # Service state
        self._active_tokenizers: Dict[str, MultilingualTokenizer] = {}
        self._session_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def create_learning_session(
        self,
        session_name: Optional[str] = None,
        tokenizer_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new learning session with a fresh tokenizer.
        
        Args:
            session_name: Optional name for the session
            tokenizer_config: Configuration for the tokenizer
            
        Returns:
            Unique session identifier
        """
        session_id = str(uuid.uuid4())
        
        # Create tokenizer with specified or default configuration
        config = tokenizer_config or {}
        tokenizer = MultilingualTokenizer(
            max_clusters=config.get('max_clusters', 20),
            similarity_threshold=config.get('similarity_threshold', 0.8),
            boundary_confidence_threshold=config.get('boundary_confidence_threshold', 0.6),
            learning_rate=config.get('learning_rate', 0.01)
        )
        
        # Store tokenizer and metadata
        self._active_tokenizers[session_id] = tokenizer
        self._session_metadata[session_id] = {
            'session_name': session_name or f"Session_{session_id[:8]}",
            'created_at': datetime.now().isoformat(),
            'tokenizer_config': config,
            'learning_samples_count': 0,
            'tokenization_count': 0
        }
        
        return session_id
    
    async def load_learning_session(
        self,
        session_id: str,
        checkpoint_name: Optional[str] = None
    ) -> bool:
        """
        Load an existing learning session from storage.
        
        Args:
            session_id: Session identifier to load
            checkpoint_name: Specific checkpoint to load (None for latest)
            
        Returns:
            True if session was loaded successfully, False otherwise
        """
        try:
            # Load tokenizer state from repository
            tokenizer = await self.repository.load_tokenizer_state(
                session_id, checkpoint_name
            )
            
            if tokenizer is None:
                return False
            
            # Store loaded tokenizer
            self._active_tokenizers[session_id] = tokenizer
            
            # Create or update session metadata
            if session_id not in self._session_metadata:
                self._session_metadata[session_id] = {}
            
            self._session_metadata[session_id].update({
                'loaded_at': datetime.now().isoformat(),
                'checkpoint_name': checkpoint_name,
                'loaded_from_storage': True
            })
            
            return True
            
        except RepositoryError:
            return False
    
    async def learn_from_text(
        self,
        session_id: str,
        text: str,
        consciousness_state: Optional[ConsciousnessState] = None,
        save_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Learn from text input in the specified session.
        
        Args:
            session_id: Session identifier
            text: Text to learn from
            consciousness_state: Current consciousness state
            save_progress: Whether to save learning progress
            
        Returns:
            Learning result dictionary
            
        Raises:
            ValueError: If session not found or invalid input
        """
        tokenizer = self._get_tokenizer(session_id)
        
        # Execute learning through use case
        result = await self.learn_from_text_use_case.execute(
            text, tokenizer, consciousness_state, save_progress
        )
        
        # Update session metadata
        self._update_session_metadata(session_id, 'learning_samples_count', 1)
        
        return result
    
    async def learn_from_batch(
        self,
        session_id: str,
        text_samples: List[str],
        consciousness_state: Optional[ConsciousnessState] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Learn from multiple text samples in batches.
        
        Args:
            session_id: Session identifier
            text_samples: List of text samples to learn from
            consciousness_state: Current consciousness state
            batch_size: Number of samples per batch
            
        Returns:
            List of learning results
        """
        tokenizer = self._get_tokenizer(session_id)
        
        # Execute batch learning through use case
        results = await self.learn_from_text_use_case.execute_batch(
            text_samples, tokenizer, consciousness_state, batch_size
        )
        
        # Update session metadata
        successful_samples = sum(1 for r in results if r.get('success', False))
        self._update_session_metadata(session_id, 'learning_samples_count', successful_samples)
        
        return results
    
    async def tokenize_text(
        self,
        session_id: str,
        text: str,
        consciousness_state: Optional[ConsciousnessState] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Tokenize text using the session's tokenizer.
        
        Args:
            session_id: Session identifier
            text: Text to tokenize
            consciousness_state: Current consciousness state
            include_metadata: Whether to include detailed metadata
            
        Returns:
            Tokenization result dictionary
        """
        tokenizer = self._get_tokenizer(session_id)
        
        # Execute tokenization through use case
        result = await self.tokenize_text_use_case.execute(
            text, tokenizer, consciousness_state,
            enable_caching=True, include_metadata=include_metadata
        )
        
        # Update session metadata
        self._update_session_metadata(session_id, 'tokenization_count', 1)
        
        return result
    
    async def tokenize_batch(
        self,
        session_id: str,
        text_samples: List[str],
        consciousness_state: Optional[ConsciousnessState] = None,
        parallel_processing: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Tokenize multiple text samples.
        
        Args:
            session_id: Session identifier
            text_samples: List of text samples to tokenize
            consciousness_state: Current consciousness state
            parallel_processing: Whether to process in parallel
            
        Returns:
            List of tokenization results
        """
        tokenizer = self._get_tokenizer(session_id)
        
        # Execute batch tokenization through use case
        results = await self.tokenize_text_use_case.execute_batch(
            text_samples, tokenizer, consciousness_state, parallel_processing
        )
        
        # Update session metadata
        successful_tokenizations = sum(1 for r in results if r.get('success', False))
        self._update_session_metadata(session_id, 'tokenization_count', successful_tokenizations)
        
        return results
    
    async def get_tokenization_analysis(
        self,
        session_id: str,
        text: str,
        consciousness_state: Optional[ConsciousnessState] = None
    ) -> Dict[str, Any]:
        """
        Get detailed analysis of tokenization process.
        
        Args:
            session_id: Session identifier
            text: Text to analyze
            consciousness_state: Current consciousness state
            
        Returns:
            Detailed tokenization analysis
        """
        tokenizer = self._get_tokenizer(session_id)
        
        return await self.tokenize_text_use_case.get_tokenization_analysis(
            text, tokenizer, consciousness_state
        )
    
    async def save_session_checkpoint(
        self,
        session_id: str,
        checkpoint_name: str
    ) -> bool:
        """
        Save a checkpoint of the current session state.
        
        Args:
            session_id: Session identifier
            checkpoint_name: Name for the checkpoint
            
        Returns:
            True if checkpoint was saved successfully
        """
        try:
            tokenizer = self._get_tokenizer(session_id)
            
            await self.repository.save_tokenizer_state(
                tokenizer, session_id, checkpoint_name
            )
            
            # Update session metadata
            self._session_metadata[session_id]['last_checkpoint'] = {
                'name': checkpoint_name,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except (ValueError, RepositoryError):
            return False
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of session state and progress.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary dictionary
        """
        # Validate session exists
        tokenizer = self._get_tokenizer(session_id)
        
        # Get base session metadata
        session_metadata = self._session_metadata.get(session_id, {})
        
        # Get learning summary from use case
        learning_summary = await self.learn_from_text_use_case.get_learning_summary()
        
        # Get tokenization performance statistics
        tokenization_stats = self.tokenize_text_use_case.get_performance_statistics()
        
        # Get tokenizer statistics
        tokenizer_stats = tokenizer.get_cluster_statistics()
        
        # Get language clusters
        clusters = tokenizer.language_clusters
        cluster_summary = {
            cluster_id: {
                'sample_count': cluster.sample_count,
                'confidence_threshold': cluster.confidence_threshold,
                'creation_timestamp': cluster.creation_timestamp
            }
            for cluster_id, cluster in clusters.items()
        }
        
        return {
            'session_id': session_id,
            'session_metadata': session_metadata,
            'learning_summary': learning_summary,
            'tokenization_statistics': tokenization_stats,
            'tokenizer_statistics': tokenizer_stats,
            'language_clusters': cluster_summary,
            'summary_timestamp': datetime.now().isoformat()
        }
    
    async def get_language_clusters(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all language clusters for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of language cluster dictionaries
        """
        tokenizer = self._get_tokenizer(session_id)
        clusters = tokenizer.language_clusters
        
        return [cluster.to_dict() for cluster in clusters.values()]
    
    async def export_session(
        self,
        session_id: str,
        export_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export complete session data.
        
        Args:
            session_id: Session identifier
            export_format: Export format
            
        Returns:
            Exported session data
        """
        try:
            return await self.repository.export_session_data(session_id, export_format)
        except RepositoryError as e:
            # Fall back to in-memory export
            return await self._export_session_fallback(session_id)
    
    async def import_session(
        self,
        session_data: Dict[str, Any],
        new_session_id: Optional[str] = None
    ) -> str:
        """
        Import session data.
        
        Args:
            session_data: Session data to import
            new_session_id: New session ID (None to generate)
            
        Returns:
            Imported session identifier
        """
        try:
            return await self.repository.import_session_data(session_data, new_session_id)
        except RepositoryError:
            # Fall back to in-memory import
            return await self._import_session_fallback(session_data, new_session_id)
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session identifiers.
        
        Returns:
            List of active session IDs
        """
        return list(self._active_tokenizers.keys())
    
    def close_session(self, session_id: str) -> bool:
        """
        Close and clean up a session.
        
        Args:
            session_id: Session identifier to close
            
        Returns:
            True if session was closed successfully
        """
        if session_id in self._active_tokenizers:
            del self._active_tokenizers[session_id]
        
        if session_id in self._session_metadata:
            del self._session_metadata[session_id]
        
        return True
    
    def clear_cache(self) -> Dict[str, int]:
        """
        Clear all caches in the service.
        
        Returns:
            Dictionary with counts of cleared items
        """
        tokenization_cache_cleared = self.tokenize_text_use_case.clear_cache()
        
        return {
            'tokenization_cache_cleared': tokenization_cache_cleared,
            'active_sessions': len(self._active_tokenizers)
        }
    
    def _get_tokenizer(self, session_id: str) -> MultilingualTokenizer:
        """
        Get tokenizer for session, raising error if not found.
        
        Args:
            session_id: Session identifier
            
        Returns:
            MultilingualTokenizer instance
            
        Raises:
            ValueError: If session not found
        """
        if session_id not in self._active_tokenizers:
            raise ValueError(f"Session {session_id} not found or not active")
        
        return self._active_tokenizers[session_id]
    
    def _update_session_metadata(
        self,
        session_id: str,
        key: str,
        increment: int = 1
    ) -> None:
        """Update session metadata with incremental values."""
        if session_id not in self._session_metadata:
            self._session_metadata[session_id] = {}
        
        current_value = self._session_metadata[session_id].get(key, 0)
        self._session_metadata[session_id][key] = current_value + increment
        self._session_metadata[session_id]['last_updated'] = datetime.now().isoformat()
    
    async def _export_session_fallback(self, session_id: str) -> Dict[str, Any]:
        """Fallback export implementation using in-memory data."""
        tokenizer = self._get_tokenizer(session_id)
        session_metadata = self._session_metadata.get(session_id, {})
        
        return {
            'session_id': session_id,
            'session_metadata': session_metadata,
            'tokenizer_state': {
                'max_clusters': tokenizer.max_clusters,
                'similarity_threshold': tokenizer.similarity_threshold,
                'boundary_confidence_threshold': tokenizer.boundary_confidence_threshold,
                'learning_rate': tokenizer.learning_rate,
                'language_clusters': {
                    cluster_id: cluster.to_dict()
                    for cluster_id, cluster in tokenizer.language_clusters.items()
                }
            },
            'export_timestamp': datetime.now().isoformat(),
            'export_method': 'fallback'
        }
    
    async def _import_session_fallback(
        self,
        session_data: Dict[str, Any],
        new_session_id: Optional[str]
    ) -> str:
        """Fallback import implementation using in-memory restoration."""
        session_id = new_session_id or str(uuid.uuid4())
        
        # Extract tokenizer configuration
        tokenizer_state = session_data.get('tokenizer_state', {})
        
        # Create tokenizer with imported configuration
        tokenizer = MultilingualTokenizer(
            max_clusters=tokenizer_state.get('max_clusters', 20),
            similarity_threshold=tokenizer_state.get('similarity_threshold', 0.8),
            boundary_confidence_threshold=tokenizer_state.get('boundary_confidence_threshold', 0.6),
            learning_rate=tokenizer_state.get('learning_rate', 0.01)
        )
        
        # Restore language clusters
        cluster_data = tokenizer_state.get('language_clusters', {})
        for cluster_id, cluster_dict in cluster_data.items():
            cluster = LanguageCluster.from_dict(cluster_dict)
            tokenizer.language_detection_service._clusters[cluster_id] = cluster
        
        # Store restored session
        self._active_tokenizers[session_id] = tokenizer
        self._session_metadata[session_id] = session_data.get('session_metadata', {})
        self._session_metadata[session_id]['imported_at'] = datetime.now().isoformat()
        
        return session_id