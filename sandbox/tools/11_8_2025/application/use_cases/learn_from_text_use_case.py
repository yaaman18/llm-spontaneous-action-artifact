"""
Learn From Text Use Case.

Application use case that orchestrates the learning process from text input.
Coordinates between domain entities, services, and repository to implement
the complete learning workflow following Clean Architecture principles.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from ...domain.entities.multilingual_tokenizer import MultilingualTokenizer
from ...domain.repositories.multilingual_learning_repository import (
    MultilingualLearningRepository,
    RepositoryError
)
from ...domain.value_objects.consciousness_state import ConsciousnessState
from ...infrastructure.config.system_config import SystemConfig


class LearnFromTextUseCase:
    """
    Use case for learning from text input.
    
    This use case orchestrates the complete learning process:
    1. Input validation and preprocessing
    2. Tokenization and language detection
    3. Learning and adaptation
    4. Progress tracking and persistence
    5. Consciousness state integration
    
    Follows the Single Responsibility Principle by focusing solely
    on the learning orchestration workflow.
    """
    
    def __init__(
        self,
        repository: MultilingualLearningRepository,
        config: SystemConfig
    ):
        """
        Initialize the use case with dependencies.
        
        Args:
            repository: Repository for persistence operations
            config: System configuration
        """
        self.repository = repository
        self.config = config
        self._session_id = str(uuid.uuid4())
        self._learning_history: List[Dict[str, Any]] = []
    
    async def execute(
        self,
        text: str,
        tokenizer: MultilingualTokenizer,
        consciousness_state: Optional[ConsciousnessState] = None,
        save_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the learning from text workflow.
        
        Args:
            text: Input text to learn from
            tokenizer: Multilingual tokenizer to use
            consciousness_state: Current consciousness state for integration
            save_progress: Whether to save learning progress
            
        Returns:
            Dictionary containing learning results and metrics
            
        Raises:
            ValueError: If input validation fails
            RepositoryError: If persistence operations fail
        """
        # Input validation
        self._validate_input(text, tokenizer)
        
        start_time = time.time()
        
        try:
            # Step 1: Tokenize the text
            tokens = tokenizer.tokenize(text)
            
            # Step 2: Extract learning metrics
            learning_metrics = self._extract_learning_metrics(
                text, tokens, tokenizer, consciousness_state
            )
            
            # Step 3: Update consciousness integration if provided
            if consciousness_state:
                learning_metrics.update(
                    self._integrate_consciousness_feedback(
                        consciousness_state, learning_metrics
                    )
                )
            
            # Step 4: Create learning result
            result = self._create_learning_result(
                text, tokens, learning_metrics, time.time() - start_time
            )
            
            # Step 5: Save progress if requested
            if save_progress:
                await self._save_learning_progress(
                    text, tokens, tokenizer, learning_metrics
                )
            
            # Step 6: Update learning history
            self._learning_history.append(result)
            
            return result
            
        except Exception as e:
            # Log error and re-raise with context
            error_context = {
                'text_length': len(text),
                'session_id': self._session_id,
                'timestamp': datetime.now().isoformat()
            }
            
            if isinstance(e, RepositoryError):
                raise e
            else:
                raise ValueError(f"Learning failed: {str(e)}") from e
    
    async def execute_batch(
        self,
        text_samples: List[str],
        tokenizer: MultilingualTokenizer,
        consciousness_state: Optional[ConsciousnessState] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute learning from multiple text samples in batches.
        
        Args:
            text_samples: List of text samples to learn from
            tokenizer: Multilingual tokenizer to use
            consciousness_state: Current consciousness state
            batch_size: Number of samples to process in each batch
            
        Returns:
            List of learning results for each sample
        """
        results = []
        
        for i in range(0, len(text_samples), batch_size):
            batch = text_samples[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                try:
                    result = await self.execute(
                        text, tokenizer, consciousness_state, save_progress=True
                    )
                    batch_results.append(result)
                except Exception as e:
                    # Log error but continue with next sample
                    error_result = {
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    batch_results.append(error_result)
            
            results.extend(batch_results)
            
            # Optional: Save batch checkpoint
            if self.config.enable_data_persistence:
                await self._save_batch_checkpoint(i // batch_size, batch_results)
        
        return results
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of learning progress.
        
        Returns:
            Dictionary with learning statistics and progress
        """
        if not self._learning_history:
            return {
                'total_samples': 0,
                'session_id': self._session_id,
                'summary': 'No learning samples processed yet'
            }
        
        # Calculate statistics from learning history
        total_samples = len(self._learning_history)
        successful_samples = sum(1 for r in self._learning_history if r.get('success', False))
        
        total_chars = sum(r.get('input_length', 0) for r in self._learning_history)
        total_tokens = sum(len(r.get('tokens', [])) for r in self._learning_history)
        
        avg_processing_time = sum(
            r.get('processing_time', 0) for r in self._learning_history
        ) / total_samples if total_samples > 0 else 0
        
        # Language cluster statistics
        cluster_usage = {}
        for result in self._learning_history:
            cluster_id = result.get('cluster_id', 'unknown')
            cluster_usage[cluster_id] = cluster_usage.get(cluster_id, 0) + 1
        
        return {
            'session_id': self._session_id,
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'success_rate': successful_samples / total_samples if total_samples > 0 else 0,
            'total_characters_processed': total_chars,
            'total_tokens_generated': total_tokens,
            'average_processing_time': avg_processing_time,
            'cluster_usage_distribution': cluster_usage,
            'learning_efficiency': total_tokens / total_chars if total_chars > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _validate_input(self, text: str, tokenizer: MultilingualTokenizer) -> None:
        """Validate input parameters."""
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty or whitespace-only")
        
        if not isinstance(tokenizer, MultilingualTokenizer):
            raise ValueError("Tokenizer must be a MultilingualTokenizer instance")
        
        if len(text) > 1_000_000:  # 1MB text limit
            raise ValueError(f"Text too large: {len(text)} characters exceeds 1MB limit")
    
    def _extract_learning_metrics(
        self,
        text: str,
        tokens: List[str],
        tokenizer: MultilingualTokenizer,
        consciousness_state: Optional[ConsciousnessState]
    ) -> Dict[str, Any]:
        """Extract learning metrics from tokenization process."""
        # Get tokenizer statistics
        tokenizer_stats = tokenizer.get_cluster_statistics()
        
        # Calculate basic metrics
        metrics = {
            'input_length': len(text),
            'token_count': len(tokens),
            'avg_token_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
            'compression_ratio': len(tokens) / len(text) if len(text) > 0 else 0,
            'cluster_count': tokenizer_stats.get('total_clusters', 0),
            'active_cluster': self._get_active_cluster_id(tokenizer, text)
        }
        
        # Add consciousness-aware metrics if available
        if consciousness_state:
            metrics.update({
                'consciousness_level': consciousness_state.consciousness_level,
                'phi_value': consciousness_state.phi_value.value,
                'metacognitive_confidence': consciousness_state.metacognitive_confidence,
                'attention_focus': consciousness_state.attention_focus_strength
            })
        
        return metrics
    
    def _integrate_consciousness_feedback(
        self,
        consciousness_state: ConsciousnessState,
        learning_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate consciousness state feedback into learning metrics."""
        consciousness_feedback = {}
        
        # Adjust learning based on consciousness level
        consciousness_level = consciousness_state.consciousness_level
        
        if consciousness_level > 0.7:
            consciousness_feedback['learning_confidence_boost'] = 0.2
            consciousness_feedback['attention_enhanced'] = True
        elif consciousness_level < 0.3:
            consciousness_feedback['learning_confidence_penalty'] = -0.1
            consciousness_feedback['attention_degraded'] = True
        
        # Factor in prediction quality
        prediction_error = consciousness_state.prediction_state.total_error
        if prediction_error < 0.5:
            consciousness_feedback['prediction_quality'] = 'high'
        elif prediction_error > 1.5:
            consciousness_feedback['prediction_quality'] = 'low'
        else:
            consciousness_feedback['prediction_quality'] = 'medium'
        
        consciousness_feedback['consciousness_integration_timestamp'] = datetime.now().isoformat()
        
        return consciousness_feedback
    
    def _create_learning_result(
        self,
        text: str,
        tokens: List[str],
        learning_metrics: Dict[str, Any],
        processing_time: float
    ) -> Dict[str, Any]:
        """Create comprehensive learning result."""
        return {
            'session_id': self._session_id,
            'text_preview': text[:200] + '...' if len(text) > 200 else text,
            'tokens': tokens,
            'success': True,
            'processing_time': processing_time,
            'learning_metrics': learning_metrics,
            'timestamp': datetime.now().isoformat(),
            'input_length': len(text),
            'cluster_id': learning_metrics.get('active_cluster')
        }
    
    async def _save_learning_progress(
        self,
        text: str,
        tokens: List[str],
        tokenizer: MultilingualTokenizer,
        learning_metrics: Dict[str, Any]
    ) -> None:
        """Save learning progress to repository."""
        try:
            cluster_id = learning_metrics.get('active_cluster', 'unknown')
            
            await self.repository.save_learning_progress(
                session_id=self._session_id,
                text_sample=text,
                tokens=tokens,
                cluster_id=cluster_id,
                learning_metrics=learning_metrics
            )
            
            # Save updated tokenizer state periodically
            if len(self._learning_history) % 10 == 0:  # Every 10 samples
                checkpoint_name = f"auto_checkpoint_{len(self._learning_history)}"
                await self.repository.save_tokenizer_state(
                    tokenizer=tokenizer,
                    session_id=self._session_id,
                    checkpoint_name=checkpoint_name
                )
                
        except RepositoryError as e:
            # Log but don't fail the learning process
            if self.config.debug_mode:
                print(f"Warning: Failed to save learning progress: {e}")
    
    async def _save_batch_checkpoint(
        self,
        batch_number: int,
        batch_results: List[Dict[str, Any]]
    ) -> None:
        """Save checkpoint after processing a batch."""
        try:
            checkpoint_data = {
                'batch_number': batch_number,
                'batch_size': len(batch_results),
                'successful_samples': sum(1 for r in batch_results if r.get('success', False)),
                'timestamp': datetime.now().isoformat(),
                'results_summary': [
                    {
                        'success': r.get('success', False),
                        'token_count': len(r.get('tokens', [])),
                        'processing_time': r.get('processing_time', 0)
                    }
                    for r in batch_results
                ]
            }
            
            # Save as custom metadata in repository
            if hasattr(self.repository, 'save_metadata'):
                await self.repository.save_metadata(
                    session_id=self._session_id,
                    key=f"batch_checkpoint_{batch_number}",
                    data=checkpoint_data
                )
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"Warning: Failed to save batch checkpoint: {e}")
    
    def _get_active_cluster_id(
        self,
        tokenizer: MultilingualTokenizer,
        text: str
    ) -> Optional[str]:
        """Get the cluster ID that would be used for the given text."""
        try:
            # Find the best matching cluster without modifying the tokenizer
            best_cluster = tokenizer._find_best_cluster(text)
            return best_cluster.cluster_id if best_cluster else None
        except Exception:
            return None