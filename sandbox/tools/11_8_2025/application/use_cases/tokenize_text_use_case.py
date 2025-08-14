"""
Tokenize Text Use Case.

Application use case that orchestrates text tokenization with proper
error handling, validation, and result formatting following Clean
Architecture principles.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ...domain.entities.multilingual_tokenizer import MultilingualTokenizer
from ...domain.repositories.multilingual_learning_repository import (
    MultilingualLearningRepository,
    RepositoryError
)
from ...domain.value_objects.consciousness_state import ConsciousnessState
from ...infrastructure.config.system_config import SystemConfig


class TokenizeTextUseCase:
    """
    Use case for text tokenization operations.
    
    This use case orchestrates the tokenization process:
    1. Input validation and preprocessing
    2. Tokenization execution
    3. Result post-processing and enrichment
    4. Performance monitoring
    5. Optional result caching
    
    Follows the Single Responsibility Principle by focusing solely
    on tokenization orchestration without learning concerns.
    """
    
    def __init__(
        self,
        repository: Optional[MultilingualLearningRepository],
        config: SystemConfig
    ):
        """
        Initialize the use case with dependencies.
        
        Args:
            repository: Optional repository for caching/retrieval
            config: System configuration
        """
        self.repository = repository
        self.config = config
        self._tokenization_cache: Dict[str, Dict[str, Any]] = {}
        self._performance_metrics: List[Dict[str, Any]] = []
    
    async def execute(
        self,
        text: str,
        tokenizer: MultilingualTokenizer,
        consciousness_state: Optional[ConsciousnessState] = None,
        enable_caching: bool = True,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Execute text tokenization with optional consciousness integration.
        
        Args:
            text: Input text to tokenize
            tokenizer: Multilingual tokenizer to use
            consciousness_state: Current consciousness state for integration
            enable_caching: Whether to use caching for repeated text
            include_metadata: Whether to include detailed metadata
            
        Returns:
            Dictionary containing tokenization results and metadata
            
        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        self._validate_input(text, tokenizer)
        
        # Check cache if enabled
        if enable_caching:
            cached_result = self._get_cached_result(text, tokenizer)
            if cached_result:
                return self._enrich_cached_result(cached_result, consciousness_state)
        
        start_time = time.time()
        
        try:
            # Execute tokenization
            tokens = tokenizer.tokenize(text)
            processing_time = time.time() - start_time
            
            # Create result dictionary
            result = self._create_tokenization_result(
                text, tokens, tokenizer, processing_time, include_metadata
            )
            
            # Integrate consciousness feedback if provided
            if consciousness_state:
                result.update(
                    self._integrate_consciousness_analysis(
                        consciousness_state, result
                    )
                )
            
            # Cache result if enabled
            if enable_caching:
                self._cache_result(text, tokenizer, result)
            
            # Record performance metrics
            self._record_performance_metrics(result)
            
            return result
            
        except Exception as e:
            # Create error result
            error_result = self._create_error_result(
                text, str(e), time.time() - start_time
            )
            
            # Still record performance for error analysis
            self._record_performance_metrics(error_result)
            
            raise ValueError(f"Tokenization failed: {str(e)}") from e
    
    async def execute_batch(
        self,
        text_samples: List[str],
        tokenizer: MultilingualTokenizer,
        consciousness_state: Optional[ConsciousnessState] = None,
        parallel_processing: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute tokenization for multiple text samples.
        
        Args:
            text_samples: List of text samples to tokenize
            tokenizer: Multilingual tokenizer to use
            consciousness_state: Current consciousness state
            parallel_processing: Whether to process samples in parallel
            
        Returns:
            List of tokenization results for each sample
        """
        results = []
        
        if parallel_processing and len(text_samples) > 1:
            # Implement parallel processing if supported by the framework
            results = await self._execute_batch_parallel(
                text_samples, tokenizer, consciousness_state
            )
        else:
            # Sequential processing
            for text in text_samples:
                try:
                    result = await self.execute(
                        text, tokenizer, consciousness_state,
                        enable_caching=True, include_metadata=True
                    )
                    results.append(result)
                except Exception as e:
                    error_result = {
                        'text_preview': text[:100] + '...' if len(text) > 100 else text,
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(error_result)
        
        return results
    
    async def get_tokenization_analysis(
        self,
        text: str,
        tokenizer: MultilingualTokenizer,
        consciousness_state: Optional[ConsciousnessState] = None
    ) -> Dict[str, Any]:
        """
        Get detailed analysis of tokenization process.
        
        Args:
            text: Text to analyze
            tokenizer: Tokenizer to use
            consciousness_state: Current consciousness state
            
        Returns:
            Detailed analysis including boundary detection and clustering
        """
        result = await self.execute(
            text, tokenizer, consciousness_state,
            enable_caching=False, include_metadata=True
        )
        
        # Add detailed analysis
        analysis = {
            'tokenization_result': result,
            'boundary_analysis': self._analyze_boundaries(text, result['tokens']),
            'cluster_analysis': self._analyze_cluster_usage(tokenizer, text),
            'linguistic_features': self._extract_linguistic_features(text, result['tokens'])
        }
        
        if consciousness_state:
            analysis['consciousness_analysis'] = self._analyze_consciousness_impact(
                consciousness_state, result
            )
        
        return analysis
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics for all tokenization operations.
        
        Returns:
            Dictionary with performance metrics and statistics
        """
        if not self._performance_metrics:
            return {
                'total_operations': 0,
                'average_processing_time': 0,
                'cache_hit_rate': 0,
                'summary': 'No tokenization operations recorded'
            }
        
        total_ops = len(self._performance_metrics)
        successful_ops = sum(1 for m in self._performance_metrics if m.get('success', False))
        
        avg_time = sum(
            m.get('processing_time', 0) for m in self._performance_metrics
        ) / total_ops
        
        cache_hits = sum(1 for m in self._performance_metrics if m.get('cache_hit', False))
        cache_hit_rate = cache_hits / total_ops if total_ops > 0 else 0
        
        total_chars = sum(m.get('input_length', 0) for m in self._performance_metrics)
        total_tokens = sum(m.get('token_count', 0) for m in self._performance_metrics)
        
        return {
            'total_operations': total_ops,
            'successful_operations': successful_ops,
            'success_rate': successful_ops / total_ops if total_ops > 0 else 0,
            'average_processing_time': avg_time,
            'cache_hit_rate': cache_hit_rate,
            'total_characters_processed': total_chars,
            'total_tokens_generated': total_tokens,
            'average_tokens_per_character': total_tokens / total_chars if total_chars > 0 else 0,
            'cache_size': len(self._tokenization_cache),
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_cache(self) -> int:
        """
        Clear the tokenization cache.
        
        Returns:
            Number of cached entries removed
        """
        cache_size = len(self._tokenization_cache)
        self._tokenization_cache.clear()
        return cache_size
    
    def _validate_input(self, text: str, tokenizer: MultilingualTokenizer) -> None:
        """Validate input parameters."""
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty or whitespace-only")
        
        if not isinstance(tokenizer, MultilingualTokenizer):
            raise ValueError("Tokenizer must be a MultilingualTokenizer instance")
        
        if len(text) > 10_000_000:  # 10MB text limit for tokenization
            raise ValueError(f"Text too large: {len(text)} characters exceeds 10MB limit")
    
    def _get_cached_result(
        self,
        text: str,
        tokenizer: MultilingualTokenizer
    ) -> Optional[Dict[str, Any]]:
        """Get cached tokenization result if available."""
        if not self.config.enable_memory_mapping:
            return None
        
        # Create cache key based on text hash and tokenizer configuration
        cache_key = self._create_cache_key(text, tokenizer)
        return self._tokenization_cache.get(cache_key)
    
    def _cache_result(
        self,
        text: str,
        tokenizer: MultilingualTokenizer,
        result: Dict[str, Any]
    ) -> None:
        """Cache tokenization result."""
        if not self.config.enable_memory_mapping:
            return
        
        cache_key = self._create_cache_key(text, tokenizer)
        
        # Limit cache size to prevent memory issues
        max_cache_size = 1000
        if len(self._tokenization_cache) >= max_cache_size:
            # Remove oldest entries (simple LRU approximation)
            oldest_key = next(iter(self._tokenization_cache))
            del self._tokenization_cache[oldest_key]
        
        self._tokenization_cache[cache_key] = result.copy()
    
    def _create_cache_key(
        self,
        text: str,
        tokenizer: MultilingualTokenizer
    ) -> str:
        """Create cache key for text and tokenizer configuration."""
        import hashlib
        
        # Hash text content
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        
        # Include relevant tokenizer parameters
        tokenizer_signature = f"{tokenizer.max_clusters}_{tokenizer.similarity_threshold}"
        
        return f"{text_hash}_{tokenizer_signature}"
    
    def _create_tokenization_result(
        self,
        text: str,
        tokens: List[str],
        tokenizer: MultilingualTokenizer,
        processing_time: float,
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Create comprehensive tokenization result."""
        result = {
            'text_preview': text[:200] + '...' if len(text) > 200 else text,
            'tokens': tokens,
            'success': True,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'input_length': len(text),
            'token_count': len(tokens),
            'cache_hit': False
        }
        
        if include_metadata:
            # Add detailed metadata
            tokenizer_stats = tokenizer.get_cluster_statistics()
            
            result['metadata'] = {
                'compression_ratio': len(tokens) / len(text) if len(text) > 0 else 0,
                'avg_token_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
                'cluster_count': tokenizer_stats.get('total_clusters', 0),
                'active_cluster': self._get_active_cluster_id(tokenizer, text),
                'tokenizer_performance': tokenizer_stats.get('performance_metrics', {}),
                'character_types': self._analyze_character_types(text),
                'processing_rate': len(text) / processing_time if processing_time > 0 else 0
            }
        
        return result
    
    def _create_error_result(
        self,
        text: str,
        error_message: str,
        processing_time: float
    ) -> Dict[str, Any]:
        """Create error result for failed tokenization."""
        return {
            'text_preview': text[:200] + '...' if len(text) > 200 else text,
            'tokens': [],
            'success': False,
            'error': error_message,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'input_length': len(text),
            'token_count': 0,
            'cache_hit': False
        }
    
    def _integrate_consciousness_analysis(
        self,
        consciousness_state: ConsciousnessState,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate consciousness state analysis into tokenization result."""
        consciousness_analysis = {
            'consciousness_level': consciousness_state.consciousness_level,
            'attention_influence': self._calculate_attention_influence(
                consciousness_state, result
            ),
            'metacognitive_assessment': self._assess_metacognitive_quality(
                consciousness_state, result
            ),
            'integrated_information': consciousness_state.phi_value.value,
            'prediction_coherence': 1.0 - min(consciousness_state.prediction_state.total_error, 1.0)
        }
        
        return {'consciousness_analysis': consciousness_analysis}
    
    def _calculate_attention_influence(
        self,
        consciousness_state: ConsciousnessState,
        result: Dict[str, Any]
    ) -> float:
        """Calculate how attention state influenced tokenization."""
        if consciousness_state.attention_weights is None:
            return 0.0
        
        # Simple heuristic: higher attention focus should correlate with better tokenization
        attention_focus = consciousness_state.attention_focus_strength
        compression_quality = result.get('metadata', {}).get('compression_ratio', 0)
        
        # Normalize and combine factors
        return min(attention_focus * compression_quality * 2, 1.0)
    
    def _assess_metacognitive_quality(
        self,
        consciousness_state: ConsciousnessState,
        result: Dict[str, Any]
    ) -> str:
        """Assess metacognitive quality of tokenization."""
        confidence = consciousness_state.metacognitive_confidence
        processing_time = result.get('processing_time', float('inf'))
        
        if confidence > 0.8 and processing_time < 0.1:
            return 'excellent'
        elif confidence > 0.6 and processing_time < 0.5:
            return 'good'
        elif confidence > 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _enrich_cached_result(
        self,
        cached_result: Dict[str, Any],
        consciousness_state: Optional[ConsciousnessState]
    ) -> Dict[str, Any]:
        """Enrich cached result with current consciousness state."""
        result = cached_result.copy()
        result['cache_hit'] = True
        result['timestamp'] = datetime.now().isoformat()
        
        if consciousness_state:
            result.update(
                self._integrate_consciousness_analysis(consciousness_state, result)
            )
        
        return result
    
    def _record_performance_metrics(self, result: Dict[str, Any]) -> None:
        """Record performance metrics for monitoring."""
        metrics = {
            'success': result.get('success', False),
            'processing_time': result.get('processing_time', 0),
            'input_length': result.get('input_length', 0),
            'token_count': result.get('token_count', 0),
            'cache_hit': result.get('cache_hit', False),
            'timestamp': result.get('timestamp')
        }
        
        self._performance_metrics.append(metrics)
        
        # Keep only recent metrics to manage memory
        max_metrics = 10000
        if len(self._performance_metrics) > max_metrics:
            self._performance_metrics = self._performance_metrics[-max_metrics:]
    
    def _get_active_cluster_id(
        self,
        tokenizer: MultilingualTokenizer,
        text: str
    ) -> Optional[str]:
        """Get the cluster ID that was used for tokenization."""
        try:
            best_cluster = tokenizer._find_best_cluster(text)
            return best_cluster.cluster_id if best_cluster else None
        except Exception:
            return None
    
    def _analyze_character_types(self, text: str) -> Dict[str, float]:
        """Analyze character type distribution in text."""
        total_chars = len(text)
        if total_chars == 0:
            return {}
        
        return {
            'alphabetic': sum(1 for c in text if c.isalpha()) / total_chars,
            'numeric': sum(1 for c in text if c.isdigit()) / total_chars,
            'whitespace': sum(1 for c in text if c.isspace()) / total_chars,
            'punctuation': sum(1 for c in text if not c.isalnum() and not c.isspace()) / total_chars
        }
    
    def _analyze_boundaries(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """Analyze boundary detection quality."""
        if not tokens:
            return {'total_boundaries': 0, 'quality_score': 0.0}
        
        # Calculate boundary positions
        boundaries = []
        pos = 0
        for token in tokens:
            pos += len(token)
            if pos < len(text):
                boundaries.append(pos)
        
        return {
            'total_boundaries': len(boundaries),
            'avg_segment_length': len(text) / len(tokens) if tokens else 0,
            'boundary_positions': boundaries[:10],  # First 10 for analysis
            'quality_score': self._calculate_boundary_quality(text, boundaries)
        }
    
    def _calculate_boundary_quality(self, text: str, boundaries: List[int]) -> float:
        """Calculate quality score for detected boundaries."""
        if not boundaries:
            return 0.0
        
        # Simple heuristic: prefer boundaries at word breaks
        quality_score = 0.0
        for boundary in boundaries:
            if boundary > 0 and boundary < len(text):
                if text[boundary - 1].isalnum() and text[boundary].isspace():
                    quality_score += 1.0
                elif text[boundary - 1].isspace() and text[boundary].isalnum():
                    quality_score += 1.0
                else:
                    quality_score += 0.5
        
        return quality_score / len(boundaries) if boundaries else 0.0
    
    def _analyze_cluster_usage(
        self,
        tokenizer: MultilingualTokenizer,
        text: str
    ) -> Dict[str, Any]:
        """Analyze language cluster usage for the text."""
        cluster_stats = tokenizer.get_cluster_statistics()
        active_cluster_id = self._get_active_cluster_id(tokenizer, text)
        
        return {
            'active_cluster_id': active_cluster_id,
            'total_clusters': cluster_stats.get('total_clusters', 0),
            'cluster_distribution': cluster_stats.get('cluster_sizes', {}),
            'new_cluster_created': active_cluster_id and active_cluster_id.startswith('cluster_')
        }
    
    def _extract_linguistic_features(
        self,
        text: str,
        tokens: List[str]
    ) -> Dict[str, Any]:
        """Extract linguistic features from text and tokens."""
        return {
            'text_length': len(text),
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'type_token_ratio': len(set(tokens)) / len(tokens) if tokens else 0,
            'avg_token_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
            'longest_token': max(tokens, key=len) if tokens else '',
            'shortest_token': min(tokens, key=len) if tokens else ''
        }
    
    def _analyze_consciousness_impact(
        self,
        consciousness_state: ConsciousnessState,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze how consciousness state impacted tokenization."""
        return {
            'consciousness_level_impact': consciousness_state.consciousness_level,
            'attention_correlation': self._calculate_attention_influence(
                consciousness_state, result
            ),
            'prediction_coherence': 1.0 - min(consciousness_state.prediction_state.total_error, 1.0),
            'metacognitive_quality': self._assess_metacognitive_quality(
                consciousness_state, result
            ),
            'integrated_information_influence': consciousness_state.phi_value.value
        }
    
    async def _execute_batch_parallel(
        self,
        text_samples: List[str],
        tokenizer: MultilingualTokenizer,
        consciousness_state: Optional[ConsciousnessState]
    ) -> List[Dict[str, Any]]:
        """Execute batch tokenization in parallel (placeholder for future implementation)."""
        # For now, fall back to sequential processing
        # In a full implementation, this would use asyncio.gather or similar
        results = []
        for text in text_samples:
            try:
                result = await self.execute(
                    text, tokenizer, consciousness_state,
                    enable_caching=True, include_metadata=True
                )
                results.append(result)
            except Exception as e:
                error_result = {
                    'text_preview': text[:100] + '...' if len(text) > 100 else text,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
        
        return results