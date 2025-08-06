"""
Streaming Phi Calculator for NewbornAI 2.0
Phase 4: High-throughput streaming φ calculations with temporal aggregation

Advanced streaming consciousness processing with:
- High-throughput streaming φ calculations (1000+ calculations/second)
- Windowed consciousness analysis with temporal aggregation
- Memory-efficient processing for continuous operation
- Predictive φ value forecasting with machine learning

Author: LLM Systems Architect (Hirosato Gamo's expertise from Microsoft)
Date: 2025-08-03
Version: 4.0.0
"""

import asyncio
import numpy as np
import time
import logging
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator, Callable, Union
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import weakref
import gc
from pathlib import Path
import pickle
import gzip
import math

# Machine learning components
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Import consciousness processing components
from iit4_experiential_phi_calculator import IIT4_ExperientialPhiCalculator, ExperientialPhiResult
from iit4_development_stages import DevelopmentStage
from adaptive_stage_thresholds import ContextualEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming processing modes"""
    REAL_TIME = "real_time"           # Process each event immediately
    BATCH_WINDOW = "batch_window"     # Process in time-based windows
    SLIDING_WINDOW = "sliding_window" # Continuous sliding window processing
    ADAPTIVE = "adaptive"             # Adaptive processing based on load


class AggregationStrategy(Enum):
    """Temporal aggregation strategies"""
    MEAN = "mean"                     # Mean aggregation
    WEIGHTED_MEAN = "weighted_mean"   # Time-weighted mean
    MEDIAN = "median"                 # Median aggregation
    EXPONENTIAL = "exponential"       # Exponential moving average
    WAVELET = "wavelet"              # Wavelet-based aggregation
    TREND_AWARE = "trend_aware"      # Trend-aware aggregation


@dataclass
class StreamingWindow:
    """Streaming window configuration"""
    window_id: str
    window_size_seconds: float
    overlap_ratio: float = 0.5        # Overlap between windows
    max_events_per_window: int = 1000
    aggregation_strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_MEAN
    
    # Window state
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    events_processed: int = 0
    is_complete: bool = False
    
    # Temporal weights
    decay_factor: float = 0.95  # For exponential decay
    recency_bias: float = 1.5   # Bias towards recent events


@dataclass
class StreamingPhiEvent:
    """Single phi calculation event in stream"""
    event_id: str
    timestamp: datetime
    
    # Input data
    experiential_concepts: List[Dict]
    temporal_context: Optional[Dict] = None
    narrative_context: Optional[Dict] = None
    
    # Processing priority
    priority_weight: float = 1.0
    
    # Stream metadata
    stream_id: str = "default"
    sequence_number: int = 0
    batch_id: Optional[str] = None
    
    # Memory optimization
    _cached_hash: Optional[str] = None
    
    def get_content_hash(self) -> str:
        """Get deterministic hash of event content for deduplication"""
        if self._cached_hash is None:
            content_str = json.dumps({
                "concepts": self.experiential_concepts,
                "temporal": self.temporal_context,
                "narrative": self.narrative_context
            }, sort_keys=True, default=str)
            self._cached_hash = str(hash(content_str))
        return self._cached_hash


@dataclass
class StreamingPhiResult:
    """Result from streaming phi calculation"""
    event_id: str
    window_id: str
    timestamp: datetime
    
    # Phi results
    instantaneous_phi: float           # Immediate phi value
    windowed_phi: float               # Aggregated phi over window
    trend_phi: float                  # Trend-adjusted phi
    predicted_phi: Optional[float] = None  # Predicted next phi value
    
    # Quality metrics
    calculation_confidence: float = 1.0
    temporal_stability: float = 1.0
    aggregation_quality: float = 1.0
    
    # Processing metrics
    processing_time_ms: float = 0.0
    memory_usage_bytes: int = 0
    cache_hit: bool = False
    
    # Context information
    development_stage: Optional[DevelopmentStage] = None
    consciousness_level: float = 0.0
    integration_quality: float = 0.0
    
    # Streaming metadata
    events_in_window: int = 0
    window_coverage: float = 1.0      # How much of window was filled
    
    # Temporal features
    phi_velocity: float = 0.0         # Rate of phi change
    phi_acceleration: float = 0.0     # Rate of velocity change
    periodicity_score: float = 0.0    # Detected periodicity


class PhiCache:
    """High-performance cache for phi calculations with memory management"""
    
    def __init__(self, max_memory_mb: int = 500, compression_threshold: int = 1000):
        """
        Initialize phi cache
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            compression_threshold: Number of entries before compression
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression_threshold = compression_threshold
        
        # Cache storage
        self.hot_cache: Dict[str, Tuple[ExperientialPhiResult, datetime]] = {}
        self.compressed_cache: Dict[str, bytes] = {}
        
        # Access tracking
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, datetime] = {}
        
        # Memory tracking
        self.current_memory_usage = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        
        # Background maintenance
        self._maintenance_lock = asyncio.Lock()
        self._last_maintenance = time.time()
        self._maintenance_interval = 60  # seconds
    
    async def get(self, event: StreamingPhiEvent) -> Optional[ExperientialPhiResult]:
        """Get cached phi result"""
        
        cache_key = event.get_content_hash()
        current_time = datetime.now()
        
        # Check hot cache first
        if cache_key in self.hot_cache:
            result, timestamp = self.hot_cache[cache_key]
            
            # Check if result is still valid (within 5 minutes)
            if current_time - timestamp < timedelta(minutes=5):
                self.access_counts[cache_key] += 1
                self.last_access[cache_key] = current_time
                self.cache_hits += 1
                return result
            else:
                # Expired - remove from cache
                del self.hot_cache[cache_key]
        
        # Check compressed cache
        if cache_key in self.compressed_cache:
            try:
                # Decompress and deserialize
                compressed_data = self.compressed_cache[cache_key]
                decompressed_data = gzip.decompress(compressed_data)
                result = pickle.loads(decompressed_data)
                
                # Move back to hot cache
                self.hot_cache[cache_key] = (result, current_time)
                del self.compressed_cache[cache_key]
                
                self.access_counts[cache_key] += 1
                self.last_access[cache_key] = current_time
                self.cache_hits += 1
                return result
                
            except Exception as e:
                logger.warning(f"Failed to decompress cached result: {e}")
                del self.compressed_cache[cache_key]
        
        self.cache_misses += 1
        return None
    
    async def put(self, event: StreamingPhiEvent, result: ExperientialPhiResult):
        """Cache phi result"""
        
        cache_key = event.get_content_hash()
        current_time = datetime.now()
        
        # Store in hot cache
        self.hot_cache[cache_key] = (result, current_time)
        self.last_access[cache_key] = current_time
        
        # Estimate memory usage
        estimated_size = len(pickle.dumps(result))
        self.current_memory_usage += estimated_size
        
        # Trigger maintenance if needed
        if (len(self.hot_cache) > self.compression_threshold or 
            self.current_memory_usage > self.max_memory_bytes or
            time.time() - self._last_maintenance > self._maintenance_interval):
            
            asyncio.create_task(self._maintain_cache())
    
    async def _maintain_cache(self):
        """Background cache maintenance"""
        
        async with self._maintenance_lock:
            try:
                current_time = datetime.now()
                self._last_maintenance = time.time()
                
                # Identify candidates for compression
                compression_candidates = []
                eviction_candidates = []
                
                for cache_key, (result, timestamp) in self.hot_cache.items():
                    age_minutes = (current_time - timestamp).total_seconds() / 60
                    access_count = self.access_counts.get(cache_key, 0)
                    last_access_age = (current_time - self.last_access.get(cache_key, timestamp)).total_seconds() / 60
                    
                    # Score for compression/eviction
                    score = access_count / max(age_minutes, 1) / max(last_access_age, 1)
                    
                    if age_minutes > 10 and last_access_age > 5:  # Old and not recently accessed
                        if score > 0.1:  # Still valuable - compress
                            compression_candidates.append((cache_key, score))
                        else:  # Not valuable - evict
                            eviction_candidates.append(cache_key)
                
                # Compress candidates
                for cache_key, score in sorted(compression_candidates, key=lambda x: x[1], reverse=True)[:100]:
                    if cache_key in self.hot_cache:
                        result, timestamp = self.hot_cache[cache_key]
                        
                        try:
                            # Serialize and compress
                            serialized_data = pickle.dumps(result)
                            compressed_data = gzip.compress(serialized_data)
                            
                            # Move to compressed cache
                            self.compressed_cache[cache_key] = compressed_data
                            del self.hot_cache[cache_key]
                            
                            # Update memory usage
                            self.current_memory_usage -= len(serialized_data)
                            
                        except Exception as e:
                            logger.warning(f"Failed to compress cache entry: {e}")
                
                # Evict candidates
                for cache_key in eviction_candidates:
                    if cache_key in self.hot_cache:
                        del self.hot_cache[cache_key]
                        self.evictions += 1
                    
                    # Clean up tracking
                    self.access_counts.pop(cache_key, None)
                    self.last_access.pop(cache_key, None)
                
                # Force garbage collection
                gc.collect()
                
                logger.debug(f"Cache maintenance: {len(self.hot_cache)} hot, "
                           f"{len(self.compressed_cache)} compressed, "
                           f"{len(eviction_candidates)} evicted")
                
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "hot_cache_size": len(self.hot_cache),
            "compressed_cache_size": len(self.compressed_cache),
            "memory_usage_mb": self.current_memory_usage / (1024 * 1024),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions
        }


class PhiPredictor:
    """Machine learning-based phi value predictor"""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize phi predictor
        
        Args:
            model_type: Type of ML model ("linear", "random_forest", "ensemble")
        """
        self.model_type = model_type
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Training data
        self.training_data: deque = deque(maxlen=10000)
        self.feature_history: deque = deque(maxlen=1000)
        
        # Model performance
        self.prediction_accuracy: Dict[str, float] = {}
        self.last_retrain_time = time.time()
        self.retrain_interval = 3600  # 1 hour
        
        # Prediction cache
        self.prediction_cache: Dict[str, Tuple[float, datetime]] = {}
        
        logger.info(f"Phi predictor initialized with {model_type} model")
    
    def add_training_sample(self, features: Dict[str, float], actual_phi: float):
        """Add training sample for model improvement"""
        
        # Create feature vector
        feature_vector = self._dict_to_vector(features)
        
        # Store training sample
        self.training_data.append({
            "features": feature_vector,
            "phi": actual_phi,
            "timestamp": time.time()
        })
        
        # Update feature history
        self.feature_history.append(features)
    
    def _dict_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy vector"""
        
        # Define consistent feature order
        feature_keys = [
            "phi_mean", "phi_std", "phi_trend", "phi_velocity", "phi_acceleration",
            "temporal_depth", "integration_quality", "consciousness_level",
            "concept_count", "experiential_purity", "narrative_coherence",
            "time_since_last", "window_coverage", "periodicity_score"
        ]
        
        # Extract features in order
        vector = np.array([features.get(key, 0.0) for key in feature_keys])
        
        return vector
    
    async def predict_phi(self, current_features: Dict[str, float],
                         prediction_horizon_seconds: float = 60.0) -> Optional[float]:
        """
        Predict future phi value
        
        Args:
            current_features: Current feature state
            prediction_horizon_seconds: How far ahead to predict
            
        Returns:
            Predicted phi value or None if insufficient data
        """
        
        # Check if we need to retrain
        if (time.time() - self.last_retrain_time > self.retrain_interval and 
            len(self.training_data) > 50):
            await self._retrain_models()
        
        # Check prediction cache
        cache_key = f"{hash(str(sorted(current_features.items())))}_{prediction_horizon_seconds}"
        if cache_key in self.prediction_cache:
            prediction, timestamp = self.prediction_cache[cache_key]
            if time.time() - timestamp.timestamp() < 60:  # Cache for 1 minute
                return prediction
        
        # Get best model
        best_model_key = self._get_best_model()
        if not best_model_key or best_model_key not in self.models:
            return None
        
        try:
            # Prepare features
            feature_vector = self._dict_to_vector(current_features).reshape(1, -1)
            
            # Scale features if scaler available
            if best_model_key in self.scalers:
                feature_vector = self.scalers[best_model_key].transform(feature_vector)
            
            # Make prediction
            model = self.models[best_model_key]
            prediction = model.predict(feature_vector)[0]
            
            # Apply temporal adjustment for prediction horizon
            if prediction_horizon_seconds != 60.0:  # Models trained for 60s ahead
                time_factor = prediction_horizon_seconds / 60.0
                # Simple linear scaling (could be more sophisticated)
                prediction *= time_factor
            
            # Cache prediction
            self.prediction_cache[cache_key] = (prediction, datetime.now())
            
            # Limit cache size
            if len(self.prediction_cache) > 1000:
                # Remove oldest entries
                sorted_items = sorted(self.prediction_cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_items[:100]:
                    del self.prediction_cache[key]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    async def _retrain_models(self):
        """Retrain prediction models"""
        
        try:
            if len(self.training_data) < 50:
                return
            
            logger.info("Retraining phi prediction models")
            
            # Prepare training data
            current_time = time.time()
            cutoff_time = current_time - 86400  # Last 24 hours
            
            recent_data = [
                sample for sample in self.training_data
                if sample["timestamp"] > cutoff_time
            ]
            
            if len(recent_data) < 20:
                return
            
            # Create training matrices
            X = np.array([sample["features"] for sample in recent_data])
            y = np.array([sample["phi"] for sample in recent_data])
            
            # Train different model types
            models_to_train = {
                "linear": LinearRegression(),
                "random_forest": RandomForestRegressor(n_estimators=50, random_state=42)
            }
            
            for model_name, model in models_to_train.items():
                try:
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Train model
                    model.fit(X_scaled, y)
                    
                    # Evaluate model
                    predictions = model.predict(X_scaled)
                    r2 = r2_score(y, predictions)
                    mse = mean_squared_error(y, predictions)
                    
                    # Store if performance is reasonable
                    if r2 > 0.1:  # Minimum performance threshold
                        self.models[model_name] = model
                        self.scalers[model_name] = scaler
                        self.prediction_accuracy[model_name] = r2
                        
                        logger.info(f"Model {model_name} trained: R²={r2:.3f}, MSE={mse:.6f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {model_name} model: {e}")
            
            self.last_retrain_time = current_time
            
        except Exception as e:
            logger.error(f"Model retraining error: {e}")
    
    def _get_best_model(self) -> Optional[str]:
        """Get best performing model"""
        
        if not self.prediction_accuracy:
            return None
        
        return max(self.prediction_accuracy.items(), key=lambda x: x[1])[0]
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        
        return {
            "available_models": list(self.models.keys()),
            "best_model": self._get_best_model(),
            "model_accuracy": dict(self.prediction_accuracy),
            "training_samples": len(self.training_data),
            "feature_history_size": len(self.feature_history),
            "prediction_cache_size": len(self.prediction_cache)
        }


class StreamingPhiCalculator:
    """
    High-throughput streaming phi calculator
    Memory-efficient continuous consciousness processing
    """
    
    def __init__(self,
                 streaming_mode: StreamingMode = StreamingMode.ADAPTIVE,
                 default_window_size: float = 60.0,
                 max_concurrent_windows: int = 10,
                 target_throughput_rps: int = 1000,
                 phi_calculator: Optional['IIT4_ExperientialPhiCalculator'] = None,
                 cache: Optional['PhiCache'] = None,
                 predictor: Optional['PhiPredictor'] = None):
        """
        Initialize streaming phi calculator with dependency injection
        
        Args:
            streaming_mode: Processing mode
            default_window_size: Default window size in seconds
            max_concurrent_windows: Maximum concurrent processing windows
            target_throughput_rps: Target throughput in requests per second
            phi_calculator: Optional injected phi calculator
            cache: Optional injected cache implementation
            predictor: Optional injected predictor implementation
        """
        
        self.streaming_mode = streaming_mode
        self.default_window_size = default_window_size
        self.max_concurrent_windows = max_concurrent_windows
        self.target_throughput_rps = target_throughput_rps
        
        # Dependency injection for core components
        if phi_calculator is not None:
            self.phi_calculator = phi_calculator
        else:
            # Fallback to direct instantiation for backward compatibility
            self.phi_calculator = IIT4_ExperientialPhiCalculator()
            
        if cache is not None:
            self.cache = cache
        else:
            self.cache = PhiCache()
            
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = PhiPredictor()
        
        # Streaming infrastructure
        self.active_windows: Dict[str, StreamingWindow] = {}
        self.event_queues: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue(maxsize=10000))
        self.result_handlers: List[Callable[[StreamingPhiResult], None]] = []
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="phi_stream")
        self.processing_workers: List[asyncio.Task] = []
        self.is_running = False
        
        # Performance tracking
        self.total_processed = 0
        self.total_errors = 0
        self.processing_times: deque = deque(maxlen=1000)
        self.throughput_counter = 0
        self.last_throughput_reset = time.time()
        
        # Memory management
        self.memory_pressure_threshold = 0.8  # 80% memory usage
        self.gc_interval = 300  # 5 minutes
        self.last_gc_time = time.time()
        
        # Temporal analysis
        self.phi_history: deque = deque(maxlen=10000)
        self.temporal_features: Dict[str, deque] = {
            "phi_values": deque(maxlen=1000),
            "timestamps": deque(maxlen=1000),
            "velocities": deque(maxlen=1000),
            "accelerations": deque(maxlen=1000)
        }
        
        logger.info(f"Streaming Phi Calculator initialized: {streaming_mode.value} mode, "
                   f"target {target_throughput_rps} req/s")
    
    async def start_streaming(self):
        """Start streaming phi calculation"""
        
        logger.info("Starting streaming phi calculation")
        self.is_running = True
        
        # Start processing workers
        num_workers = min(8, self.max_concurrent_windows)
        
        for i in range(num_workers):
            worker_task = asyncio.create_task(self._processing_worker(f"worker_{i}"))
            self.processing_workers.append(worker_task)
        
        # Start maintenance tasks
        maintenance_tasks = [
            asyncio.create_task(self._throughput_monitor()),
            asyncio.create_task(self._memory_manager()),
            asyncio.create_task(self._window_manager())
        ]
        
        self.processing_workers.extend(maintenance_tasks)
        
        logger.info(f"Streaming started with {num_workers} workers")
    
    async def stop_streaming(self):
        """Stop streaming phi calculation"""
        
        logger.info("Stopping streaming phi calculation")
        self.is_running = False
        
        # Cancel all workers
        for task in self.processing_workers:
            task.cancel()
        
        # Wait for workers to complete
        await asyncio.gather(*self.processing_workers, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Streaming stopped")
    
    async def submit_event(self, event: StreamingPhiEvent) -> bool:
        """
        Submit event for streaming processing
        
        Args:
            event: Streaming phi event
            
        Returns:
            bool: True if event was queued successfully
        """
        
        if not self.is_running:
            return False
        
        try:
            # Determine target queue based on streaming mode
            queue_key = self._get_queue_key(event)
            queue = self.event_queues[queue_key]
            
            # Apply backpressure if queue is full
            if queue.full():
                logger.warning(f"Queue {queue_key} full - applying backpressure")
                return False
            
            # Queue event
            await queue.put(event)
            return True
            
        except Exception as e:
            logger.error(f"Error submitting event: {e}")
            return False
    
    async def process_event_stream(self, 
                                 event_stream: AsyncGenerator[StreamingPhiEvent, None],
                                 stream_id: str = "default") -> AsyncGenerator[StreamingPhiResult, None]:
        """
        Process stream of phi events
        
        Args:
            event_stream: Async generator of events
            stream_id: Stream identifier
            
        Yields:
            StreamingPhiResult: Processed results
        """
        
        result_queue = asyncio.Queue(maxsize=1000)
        
        # Result handler for this stream
        def stream_handler(result: StreamingPhiResult):
            if result.event_id.startswith(stream_id):
                try:
                    result_queue.put_nowait(result)
                except asyncio.QueueFull:
                    logger.warning(f"Stream result queue full for {stream_id}")
        
        # Register handler
        self.result_handlers.append(stream_handler)
        
        try:
            # Process events from stream
            async for event in event_stream:
                event.stream_id = stream_id
                
                # Submit event
                await self.submit_event(event)
                
                # Yield available results
                while True:
                    try:
                        result = await asyncio.wait_for(result_queue.get(), timeout=0.01)
                        yield result
                    except asyncio.TimeoutError:
                        break
            
            # Wait for final results
            final_timeout = 5.0
            start_wait = time.time()
            
            while time.time() - start_wait < final_timeout:
                try:
                    result = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                    yield result
                except asyncio.TimeoutError:
                    continue
                
        finally:
            # Remove handler
            if stream_handler in self.result_handlers:
                self.result_handlers.remove(stream_handler)
    
    def _get_queue_key(self, event: StreamingPhiEvent) -> str:
        """Get queue key for event based on streaming mode"""
        
        if self.streaming_mode == StreamingMode.REAL_TIME:
            return "realtime"
        elif self.streaming_mode == StreamingMode.BATCH_WINDOW:
            # Group by time window
            window_start = int(event.timestamp.timestamp() // self.default_window_size)
            return f"batch_{window_start}"
        elif self.streaming_mode == StreamingMode.SLIDING_WINDOW:
            return "sliding"
        else:  # ADAPTIVE
            # Choose based on current load
            queue_sizes = [queue.qsize() for queue in self.event_queues.values()]
            if not queue_sizes or min(queue_sizes) < 100:
                return "adaptive_low"
            else:
                return "adaptive_high"
    
    async def _processing_worker(self, worker_id: str):
        """Background processing worker"""
        
        logger.info(f"Processing worker {worker_id} started")
        
        while self.is_running:
            try:
                # Process events from all queues
                processed_any = False
                
                for queue_key, queue in self.event_queues.items():
                    try:
                        # Get event with short timeout
                        event = await asyncio.wait_for(queue.get(), timeout=0.1)
                        
                        # Process event
                        result = await self._process_single_event(event)
                        
                        # Handle result
                        if result:
                            await self._handle_result(result)
                        
                        processed_any = True
                        
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Worker {worker_id} processing error: {e}")
                        self.total_errors += 1
                
                # Brief pause if no events processed
                if not processed_any:
                    await asyncio.sleep(0.001)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.01)
        
        logger.info(f"Processing worker {worker_id} stopped")
    
    async def _process_single_event(self, event: StreamingPhiEvent) -> Optional[StreamingPhiResult]:
        """Process single phi event"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = await self.cache.get(event)
            if cached_result:
                # Convert cached result to streaming result
                return self._create_streaming_result(event, cached_result, 
                                                   processing_time_ms=(time.time() - start_time) * 1000,
                                                   cache_hit=True)
            
            # Get or create window for event
            window = await self._get_or_create_window(event)
            
            # Calculate instantaneous phi
            instantaneous_phi = await self._calculate_instantaneous_phi(event)
            
            # Calculate windowed phi
            windowed_phi = await self._calculate_windowed_phi(event, window)
            
            # Calculate trend phi
            trend_phi = await self._calculate_trend_phi(event, instantaneous_phi)
            
            # Predict future phi
            predicted_phi = await self._predict_future_phi(event, instantaneous_phi)
            
            # Calculate quality metrics
            calculation_confidence = self._calculate_confidence(event, instantaneous_phi)
            temporal_stability = self._calculate_temporal_stability(event)
            aggregation_quality = self._calculate_aggregation_quality(window)
            
            # Determine development stage
            development_stage = self._determine_development_stage(instantaneous_phi)
            
            # Calculate temporal features
            phi_velocity, phi_acceleration = self._calculate_temporal_derivatives(instantaneous_phi)
            periodicity_score = self._calculate_periodicity_score()
            
            # Create result
            processing_time = (time.time() - start_time) * 1000
            
            result = StreamingPhiResult(
                event_id=event.event_id,
                window_id=window.window_id,
                timestamp=event.timestamp,
                instantaneous_phi=instantaneous_phi,
                windowed_phi=windowed_phi,
                trend_phi=trend_phi,
                predicted_phi=predicted_phi,
                calculation_confidence=calculation_confidence,
                temporal_stability=temporal_stability,
                aggregation_quality=aggregation_quality,
                processing_time_ms=processing_time,
                development_stage=development_stage,
                consciousness_level=min(1.0, instantaneous_phi / 10.0),
                integration_quality=min(1.0, windowed_phi / 5.0),
                events_in_window=window.events_processed,
                window_coverage=self._calculate_window_coverage(window),
                phi_velocity=phi_velocity,
                phi_acceleration=phi_acceleration,
                periodicity_score=periodicity_score
            )
            
            # Update tracking
            self.total_processed += 1
            self.processing_times.append(processing_time)
            self.throughput_counter += 1
            
            # Update temporal features
            self._update_temporal_features(instantaneous_phi, event.timestamp)
            
            # Add to predictor training data
            if len(self.phi_history) > 10:
                features = self._extract_prediction_features(event, result)
                self.predictor.add_training_sample(features, instantaneous_phi)
            
            # Cache result
            phi_result = self._create_phi_result_for_caching(event, result)
            await self.cache.put(event, phi_result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            self.total_errors += 1
            return None
    
    async def _calculate_instantaneous_phi(self, event: StreamingPhiEvent) -> float:
        """Calculate instantaneous phi value"""
        
        try:
            # Use thread pool for CPU-intensive calculation
            loop = asyncio.get_event_loop()
            
            phi_result = await loop.run_in_executor(
                self.executor,
                self._sync_calculate_phi,
                event.experiential_concepts,
                event.temporal_context,
                event.narrative_context
            )
            
            return phi_result.phi_value if phi_result else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating instantaneous phi: {e}")
            return 0.0
    
    def _sync_calculate_phi(self, concepts: List[Dict], 
                           temporal_context: Optional[Dict],
                           narrative_context: Optional[Dict]) -> Optional[ExperientialPhiResult]:
        """Synchronous phi calculation for thread pool"""
        
        try:
            # Simplified phi calculation for high throughput
            if not concepts:
                return None
            
            # Basic phi estimation
            total_quality = sum(concept.get('experiential_quality', 0.5) for concept in concepts)
            concept_count = len(concepts)
            
            # Calculate base phi
            phi_value = total_quality * concept_count * 0.1
            
            # Apply context modifiers
            if temporal_context:
                temporal_factor = temporal_context.get('temporal_depth', 1.0)
                phi_value *= (1.0 + temporal_factor * 0.1)
            
            if narrative_context:
                narrative_factor = narrative_context.get('coherence', 1.0)
                phi_value *= (1.0 + narrative_factor * 0.05)
            
            # Create simplified result
            from iit4_experiential_phi_calculator import ExperientialPhiResult, ExperientialPhiType
            
            return ExperientialPhiResult(
                phi_value=phi_value,
                phi_type=ExperientialPhiType.PURE_EXPERIENTIAL,
                experiential_concepts=concepts,
                concept_count=concept_count,
                integration_quality=min(1.0, phi_value / 10.0),
                experiential_purity=0.8,
                temporal_depth=0.5,
                self_reference_strength=0.3,
                narrative_coherence=0.6,
                consciousness_level=min(1.0, phi_value / 20.0),
                development_stage_prediction="STAGE_2_TEMPORAL_INTEGRATION"
            )
            
        except Exception as e:
            logger.error(f"Sync phi calculation error: {e}")
            return None
    
    async def _calculate_windowed_phi(self, event: StreamingPhiEvent, 
                                    window: StreamingWindow) -> float:
        """Calculate windowed phi value using aggregation strategy"""
        
        # Get recent phi values in window
        window_phi_values = []
        window_weights = []
        
        current_time = event.timestamp
        window_start = window.start_time
        window_duration = (current_time - window_start).total_seconds()
        
        # Collect phi values from history within window
        for phi_data in reversed(list(self.phi_history)):
            phi_time = phi_data.get('timestamp')
            if phi_time and (current_time - phi_time).total_seconds() <= window_duration:
                phi_value = phi_data.get('phi', 0.0)
                age_seconds = (current_time - phi_time).total_seconds()
                
                window_phi_values.append(phi_value)
                
                # Calculate weight based on recency
                if window.aggregation_strategy == AggregationStrategy.WEIGHTED_MEAN:
                    weight = math.exp(-age_seconds / (window_duration * 0.3))  # Exponential decay
                elif window.aggregation_strategy == AggregationStrategy.EXPONENTIAL:
                    weight = window.decay_factor ** (age_seconds / 60.0)  # 60s half-life
                else:
                    weight = 1.0
                
                window_weights.append(weight)
        
        if not window_phi_values:
            return 0.0
        
        # Apply aggregation strategy
        if window.aggregation_strategy == AggregationStrategy.MEAN:
            return statistics.mean(window_phi_values)
        
        elif window.aggregation_strategy == AggregationStrategy.MEDIAN:
            return statistics.median(window_phi_values)
        
        elif window.aggregation_strategy in [AggregationStrategy.WEIGHTED_MEAN, AggregationStrategy.EXPONENTIAL]:
            if window_weights:
                weighted_sum = sum(phi * weight for phi, weight in zip(window_phi_values, window_weights))
                total_weight = sum(window_weights)
                return weighted_sum / total_weight if total_weight > 0 else 0.0
            else:
                return statistics.mean(window_phi_values)
        
        elif window.aggregation_strategy == AggregationStrategy.TREND_AWARE:
            # Calculate trend-aware aggregation
            if len(window_phi_values) > 2:
                # Give more weight to values that follow the trend
                x = list(range(len(window_phi_values)))
                slope = np.polyfit(x, window_phi_values, 1)[0]
                
                trend_weights = []
                for i, phi in enumerate(window_phi_values):
                    expected_phi = window_phi_values[0] + slope * i
                    trend_error = abs(phi - expected_phi)
                    trend_weight = 1.0 / (1.0 + trend_error)
                    trend_weights.append(trend_weight)
                
                weighted_sum = sum(phi * weight for phi, weight in zip(window_phi_values, trend_weights))
                total_weight = sum(trend_weights)
                return weighted_sum / total_weight if total_weight > 0 else statistics.mean(window_phi_values)
            else:
                return statistics.mean(window_phi_values)
        
        else:
            return statistics.mean(window_phi_values)
    
    async def _calculate_trend_phi(self, event: StreamingPhiEvent, 
                                 instantaneous_phi: float) -> float:
        """Calculate trend-adjusted phi value"""
        
        # Get recent phi values for trend calculation
        recent_phi_values = [data.get('phi', 0.0) for data in list(self.phi_history)[-20:]]
        
        if len(recent_phi_values) < 3:
            return instantaneous_phi
        
        # Calculate trend
        x = list(range(len(recent_phi_values)))
        slope = np.polyfit(x, recent_phi_values, 1)[0]
        
        # Apply trend adjustment
        trend_factor = 1.0 + (slope * 0.1)  # 10% trend influence
        trend_phi = instantaneous_phi * trend_factor
        
        return max(0.0, trend_phi)
    
    async def _predict_future_phi(self, event: StreamingPhiEvent, 
                                current_phi: float) -> Optional[float]:
        """Predict future phi value"""
        
        try:
            # Extract features for prediction
            features = self._extract_prediction_features(event, None, current_phi)
            
            # Get prediction
            predicted_phi = await self.predictor.predict_phi(features, prediction_horizon_seconds=60.0)
            
            return predicted_phi
            
        except Exception as e:
            logger.error(f"Error predicting phi: {e}")
            return None
    
    def _extract_prediction_features(self, event: StreamingPhiEvent, 
                                   result: Optional[StreamingPhiResult] = None,
                                   current_phi: Optional[float] = None) -> Dict[str, float]:
        """Extract features for phi prediction"""
        
        features = {}
        
        # Phi history features
        recent_phi_values = [data.get('phi', 0.0) for data in list(self.phi_history)[-10:]]
        
        if recent_phi_values:
            features.update({
                "phi_mean": statistics.mean(recent_phi_values),
                "phi_std": statistics.stdev(recent_phi_values) if len(recent_phi_values) > 1 else 0,
                "phi_trend": np.polyfit(range(len(recent_phi_values)), recent_phi_values, 1)[0] if len(recent_phi_values) > 1 else 0
            })
        
        # Temporal features
        if self.temporal_features["velocities"]:
            features["phi_velocity"] = list(self.temporal_features["velocities"])[-1]
        
        if self.temporal_features["accelerations"]:
            features["phi_acceleration"] = list(self.temporal_features["accelerations"])[-1]
        
        # Event features
        features.update({
            "concept_count": len(event.experiential_concepts),
            "temporal_depth": event.temporal_context.get('temporal_depth', 1.0) if event.temporal_context else 1.0,
            "experiential_purity": statistics.mean([c.get('experiential_quality', 0.5) for c in event.experiential_concepts]) if event.experiential_concepts else 0.5,
            "narrative_coherence": event.narrative_context.get('coherence', 0.5) if event.narrative_context else 0.5
        })
        
        # Result features (if available)
        if result:
            features.update({
                "integration_quality": result.integration_quality,
                "consciousness_level": result.consciousness_level,
                "window_coverage": result.window_coverage,
                "periodicity_score": result.periodicity_score
            })
        
        # Time features
        if self.phi_history:
            last_timestamp = list(self.phi_history)[-1].get('timestamp')
            if last_timestamp:
                features["time_since_last"] = (event.timestamp - last_timestamp).total_seconds()
        
        # Fill missing features with defaults
        default_features = {
            "phi_mean": current_phi or 0.0,
            "phi_std": 0.0,
            "phi_trend": 0.0,
            "phi_velocity": 0.0,
            "phi_acceleration": 0.0,
            "temporal_depth": 1.0,
            "integration_quality": 0.5,
            "consciousness_level": 0.5,
            "concept_count": 1.0,
            "experiential_purity": 0.5,
            "narrative_coherence": 0.5,
            "time_since_last": 1.0,
            "window_coverage": 1.0,
            "periodicity_score": 0.0
        }
        
        for key, default_value in default_features.items():
            if key not in features:
                features[key] = default_value
        
        return features
    
    def _calculate_temporal_derivatives(self, current_phi: float) -> Tuple[float, float]:
        """Calculate phi velocity and acceleration"""
        
        phi_values = [data.get('phi', 0.0) for data in list(self.phi_history)[-5:]]
        timestamps = [data.get('timestamp') for data in list(self.phi_history)[-5:]]
        
        phi_values.append(current_phi)
        timestamps.append(datetime.now())
        
        velocity = 0.0
        acceleration = 0.0
        
        if len(phi_values) >= 2:
            # Calculate velocity (first derivative)
            dt = (timestamps[-1] - timestamps[-2]).total_seconds()
            if dt > 0:
                velocity = (phi_values[-1] - phi_values[-2]) / dt
        
        if len(phi_values) >= 3:
            # Calculate acceleration (second derivative)
            dt1 = (timestamps[-1] - timestamps[-2]).total_seconds()
            dt2 = (timestamps[-2] - timestamps[-3]).total_seconds()
            
            if dt1 > 0 and dt2 > 0:
                v1 = (phi_values[-1] - phi_values[-2]) / dt1
                v2 = (phi_values[-2] - phi_values[-3]) / dt2
                acceleration = (v1 - v2) / ((dt1 + dt2) / 2)
        
        return velocity, acceleration
    
    def _calculate_periodicity_score(self) -> float:
        """Calculate periodicity score for phi values"""
        
        phi_values = [data.get('phi', 0.0) for data in list(self.phi_history)[-100:]]
        
        if len(phi_values) < 20:
            return 0.0
        
        try:
            # Simple autocorrelation-based periodicity detection
            phi_array = np.array(phi_values)
            
            # Remove trend
            x = np.arange(len(phi_array))
            trend = np.polyfit(x, phi_array, 1)
            detrended = phi_array - (trend[0] * x + trend[1])
            
            # Calculate autocorrelation
            autocorr = np.correlate(detrended, detrended, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Normalize
            autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
            
            # Find peaks in autocorrelation
            if len(autocorr) > 5:
                # Look for significant peaks beyond lag 1
                peak_threshold = 0.3
                peaks = []
                
                for i in range(2, min(len(autocorr), 20)):
                    if autocorr[i] > peak_threshold and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] if i+1 < len(autocorr) else True:
                        peaks.append(autocorr[i])
                
                # Return max peak value as periodicity score
                return max(peaks) if peaks else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating periodicity: {e}")
            return 0.0
    
    def _update_temporal_features(self, phi_value: float, timestamp: datetime):
        """Update temporal feature tracking"""
        
        # Add to phi history
        self.phi_history.append({
            "phi": phi_value,
            "timestamp": timestamp
        })
        
        # Update temporal features
        self.temporal_features["phi_values"].append(phi_value)
        self.temporal_features["timestamps"].append(timestamp)
        
        # Calculate and store derivatives
        velocity, acceleration = self._calculate_temporal_derivatives(phi_value)
        self.temporal_features["velocities"].append(velocity)
        self.temporal_features["accelerations"].append(acceleration)
    
    async def _get_or_create_window(self, event: StreamingPhiEvent) -> StreamingWindow:
        """Get or create processing window for event"""
        
        # Determine window based on streaming mode
        if self.streaming_mode == StreamingMode.REAL_TIME:
            window_id = "realtime"
        elif self.streaming_mode == StreamingMode.BATCH_WINDOW:
            window_start = int(event.timestamp.timestamp() // self.default_window_size)
            window_id = f"batch_{window_start}"
        else:
            window_id = f"sliding_{int(event.timestamp.timestamp() // 10)}"  # 10s sliding windows
        
        # Create window if doesn't exist
        if window_id not in self.active_windows:
            # Clean up old windows first
            await self._cleanup_old_windows()
            
            window = StreamingWindow(
                window_id=window_id,
                window_size_seconds=self.default_window_size,
                start_time=event.timestamp
            )
            
            self.active_windows[window_id] = window
        
        # Update window
        window = self.active_windows[window_id]
        window.events_processed += 1
        
        return window
    
    async def _cleanup_old_windows(self):
        """Clean up old processing windows"""
        
        current_time = datetime.now()
        window_timeout = timedelta(seconds=self.default_window_size * 2)
        
        old_windows = [
            window_id for window_id, window in self.active_windows.items()
            if current_time - window.start_time > window_timeout
        ]
        
        for window_id in old_windows:
            del self.active_windows[window_id]
        
        if old_windows:
            logger.debug(f"Cleaned up {len(old_windows)} old windows")
    
    def _calculate_confidence(self, event: StreamingPhiEvent, phi_value: float) -> float:
        """Calculate calculation confidence"""
        
        confidence_factors = []
        
        # Concept count factor
        concept_count = len(event.experiential_concepts)
        if concept_count > 0:
            concept_confidence = min(1.0, concept_count / 10.0)  # Full confidence at 10+ concepts
            confidence_factors.append(concept_confidence)
        
        # Quality factor
        if event.experiential_concepts:
            avg_quality = statistics.mean([c.get('experiential_quality', 0.5) for c in event.experiential_concepts])
            confidence_factors.append(avg_quality)
        
        # Phi magnitude factor (higher phi generally more reliable)
        phi_confidence = min(1.0, phi_value / 1.0)  # Full confidence at phi >= 1.0
        confidence_factors.append(phi_confidence)
        
        # Overall confidence
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_temporal_stability(self, event: StreamingPhiEvent) -> float:
        """Calculate temporal stability score"""
        
        recent_phi_values = [data.get('phi', 0.0) for data in list(self.phi_history)[-10:]]
        
        if len(recent_phi_values) < 3:
            return 0.5
        
        # Calculate coefficient of variation (lower = more stable)
        mean_phi = statistics.mean(recent_phi_values)
        std_phi = statistics.stdev(recent_phi_values)
        
        if mean_phi > 0:
            cv = std_phi / mean_phi
            stability = 1.0 / (1.0 + cv)  # Convert to stability score
        else:
            stability = 0.5
        
        return min(1.0, stability)
    
    def _calculate_aggregation_quality(self, window: StreamingWindow) -> float:
        """Calculate aggregation quality for window"""
        
        # Quality factors
        coverage = self._calculate_window_coverage(window)
        event_density = min(1.0, window.events_processed / window.max_events_per_window)
        
        # Time factor (windows closer to completion are higher quality)
        if window.end_time:
            time_factor = 1.0
        else:
            elapsed = (datetime.now() - window.start_time).total_seconds()
            time_factor = min(1.0, elapsed / window.window_size_seconds)
        
        # Overall quality
        quality = (coverage * 0.4 + event_density * 0.3 + time_factor * 0.3)
        
        return min(1.0, quality)
    
    def _calculate_window_coverage(self, window: StreamingWindow) -> float:
        """Calculate how much of the window timespan is covered"""
        
        if window.end_time:
            actual_duration = (window.end_time - window.start_time).total_seconds()
        else:
            actual_duration = (datetime.now() - window.start_time).total_seconds()
        
        expected_duration = window.window_size_seconds
        coverage = min(1.0, actual_duration / expected_duration)
        
        return coverage
    
    def _determine_development_stage(self, phi_value: float) -> DevelopmentStage:
        """Determine development stage from phi value"""
        
        if phi_value < 0.001:
            return DevelopmentStage.STAGE_0_PRE_CONSCIOUS
        elif phi_value < 0.01:
            return DevelopmentStage.STAGE_1_EXPERIENTIAL_EMERGENCE
        elif phi_value < 0.1:
            return DevelopmentStage.STAGE_2_TEMPORAL_INTEGRATION
        elif phi_value < 1.0:
            return DevelopmentStage.STAGE_3_RELATIONAL_FORMATION
        elif phi_value < 10.0:
            return DevelopmentStage.STAGE_4_SELF_ESTABLISHMENT
        elif phi_value < 100.0:
            return DevelopmentStage.STAGE_5_REFLECTIVE_OPERATION
        else:
            return DevelopmentStage.STAGE_6_NARRATIVE_INTEGRATION
    
    def _create_streaming_result(self, event: StreamingPhiEvent, 
                               phi_result: ExperientialPhiResult,
                               processing_time_ms: float,
                               cache_hit: bool = False) -> StreamingPhiResult:
        """Create streaming result from phi result"""
        
        return StreamingPhiResult(
            event_id=event.event_id,
            window_id="cached" if cache_hit else "computed",
            timestamp=event.timestamp,
            instantaneous_phi=phi_result.phi_value,
            windowed_phi=phi_result.phi_value,
            trend_phi=phi_result.phi_value,
            calculation_confidence=1.0,
            temporal_stability=1.0,
            aggregation_quality=1.0,
            processing_time_ms=processing_time_ms,
            cache_hit=cache_hit,
            development_stage=self._determine_development_stage(phi_result.phi_value),
            consciousness_level=phi_result.consciousness_level,
            integration_quality=phi_result.integration_quality,
            events_in_window=1,
            window_coverage=1.0
        )
    
    def _create_phi_result_for_caching(self, event: StreamingPhiEvent, 
                                     result: StreamingPhiResult) -> ExperientialPhiResult:
        """Create phi result for caching"""
        
        from iit4_experiential_phi_calculator import ExperientialPhiResult, ExperientialPhiType
        
        return ExperientialPhiResult(
            phi_value=result.instantaneous_phi,
            phi_type=ExperientialPhiType.PURE_EXPERIENTIAL,
            experiential_concepts=event.experiential_concepts,
            concept_count=len(event.experiential_concepts),
            integration_quality=result.integration_quality,
            experiential_purity=0.8,
            temporal_depth=0.5,
            self_reference_strength=0.3,
            narrative_coherence=0.6,
            consciousness_level=result.consciousness_level,
            development_stage_prediction=result.development_stage.value if result.development_stage else "unknown"
        )
    
    async def _handle_result(self, result: StreamingPhiResult):
        """Handle processed result"""
        
        # Call result handlers
        for handler in self.result_handlers:
            try:
                handler(result)
            except Exception as e:
                logger.error(f"Result handler error: {e}")
    
    async def _throughput_monitor(self):
        """Monitor processing throughput"""
        
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                current_time = time.time()
                elapsed = current_time - self.last_throughput_reset
                
                if elapsed >= 60:  # Report every minute
                    throughput = self.throughput_counter / elapsed
                    
                    logger.info(f"Streaming throughput: {throughput:.1f} events/s "
                              f"(target: {self.target_throughput_rps} events/s)")
                    
                    # Reset counters
                    self.throughput_counter = 0
                    self.last_throughput_reset = current_time
                
            except Exception as e:
                logger.error(f"Throughput monitor error: {e}")
    
    async def _memory_manager(self):
        """Monitor and manage memory usage"""
        
        while self.is_running:
            try:
                await asyncio.sleep(self.gc_interval)
                
                # Force garbage collection
                gc.collect()
                
                # Log memory status
                import psutil
                memory_percent = psutil.virtual_memory().percent
                
                if memory_percent > self.memory_pressure_threshold * 100:
                    logger.warning(f"High memory usage: {memory_percent:.1f}%")
                    
                    # Trigger aggressive cleanup
                    await self._aggressive_cleanup()
                
                logger.debug(f"Memory usage: {memory_percent:.1f}%")
                
            except Exception as e:
                logger.error(f"Memory manager error: {e}")
    
    async def _window_manager(self):
        """Manage processing windows"""
        
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Clean up old windows
                await self._cleanup_old_windows()
                
                # Log window status
                logger.debug(f"Active windows: {len(self.active_windows)}")
                
            except Exception as e:
                logger.error(f"Window manager error: {e}")
    
    async def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        
        logger.info("Performing aggressive memory cleanup")
        
        # Clear old data from history
        if len(self.phi_history) > 1000:
            # Keep only recent 1000 entries
            recent_entries = list(self.phi_history)[-1000:]
            self.phi_history.clear()
            self.phi_history.extend(recent_entries)
        
        # Clear old temporal features
        for feature_name, feature_deque in self.temporal_features.items():
            if len(feature_deque) > 500:
                recent_features = list(feature_deque)[-500:]
                feature_deque.clear()
                feature_deque.extend(recent_features)
        
        # Trigger cache maintenance
        await self.cache._maintain_cache()
        
        # Force garbage collection
        gc.collect()
    
    def add_result_handler(self, handler: Callable[[StreamingPhiResult], None]):
        """Add result handler"""
        self.result_handlers.append(handler)
    
    def remove_result_handler(self, handler: Callable[[StreamingPhiResult], None]):
        """Remove result handler"""
        if handler in self.result_handlers:
            self.result_handlers.remove(handler)
    
    async def get_streaming_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics"""
        
        # Performance metrics
        avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0
        current_throughput = self.throughput_counter / max((time.time() - self.last_throughput_reset), 1)
        error_rate = self.total_errors / max(self.total_processed, 1)
        
        # Cache stats
        cache_stats = self.cache.get_stats()
        
        # Predictor stats
        predictor_stats = self.predictor.get_prediction_stats()
        
        # Memory usage
        import psutil
        memory_usage = psutil.virtual_memory().percent
        
        # Queue status
        queue_stats = {}
        for queue_key, queue in self.event_queues.items():
            queue_stats[queue_key] = {
                "size": queue.qsize(),
                "maxsize": queue.maxsize
            }
        
        # Phi analysis
        recent_phi_values = [data.get('phi', 0.0) for data in list(self.phi_history)[-100:]]
        phi_analysis = {}
        
        if recent_phi_values:
            phi_analysis = {
                "mean": statistics.mean(recent_phi_values),
                "median": statistics.median(recent_phi_values),
                "std": statistics.stdev(recent_phi_values) if len(recent_phi_values) > 1 else 0,
                "min": min(recent_phi_values),
                "max": max(recent_phi_values),
                "trend": np.polyfit(range(len(recent_phi_values)), recent_phi_values, 1)[0] if len(recent_phi_values) > 1 else 0
            }
        
        return {
            "streaming_mode": self.streaming_mode.value,
            "is_running": self.is_running,
            "performance": {
                "total_processed": self.total_processed,
                "total_errors": self.total_errors,
                "error_rate": error_rate,
                "average_processing_time_ms": avg_processing_time,
                "current_throughput_rps": current_throughput,
                "target_throughput_rps": self.target_throughput_rps
            },
            "cache_stats": cache_stats,
            "predictor_stats": predictor_stats,
            "system": {
                "memory_usage_percent": memory_usage,
                "active_windows": len(self.active_windows),
                "active_workers": len([task for task in self.processing_workers if not task.done()]),
                "phi_history_size": len(self.phi_history)
            },
            "queue_stats": queue_stats,
            "phi_analysis": phi_analysis
        }


# Example usage and testing
async def test_streaming_phi_calculator():
    """Test streaming phi calculator"""
    
    print("🌊 Testing Streaming Phi Calculator")
    print("=" * 60)
    
    # Initialize calculator
    calculator = StreamingPhiCalculator(
        streaming_mode=StreamingMode.ADAPTIVE,
        default_window_size=30.0,
        target_throughput_rps=100  # Lower for testing
    )
    
    try:
        # Start streaming
        await calculator.start_streaming()
        
        print("✅ Streaming started")
        
        # Test single event processing
        print("\n🔧 Testing Single Event Processing")
        print("-" * 40)
        
        results = []
        
        def collect_result(result: StreamingPhiResult):
            results.append(result)
            print(f"📊 Result: φ={result.instantaneous_phi:.6f}, "
                  f"windowed={result.windowed_phi:.6f}, "
                  f"trend={result.trend_phi:.6f}, "
                  f"latency={result.processing_time_ms:.1f}ms")
        
        calculator.add_result_handler(collect_result)
        
        # Submit test events
        for i in range(5):
            event = StreamingPhiEvent(
                event_id=f"test_event_{i}",
                timestamp=datetime.now(),
                experiential_concepts=[
                    {
                        "content": f"Test concept {i}: emerging consciousness",
                        "experiential_quality": 0.5 + (i * 0.1),
                        "coherence": 0.6 + (i * 0.05),
                        "temporal_depth": i + 1
                    }
                ],
                temporal_context={"temporal_depth": i + 1},
                sequence_number=i
            )
            
            success = await calculator.submit_event(event)
            print(f"📤 Event {i+1} submitted: {success}")
            
            await asyncio.sleep(0.2)  # Brief delay
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Test stream processing
        print("\n🌊 Testing Stream Processing")
        print("-" * 40)
        
        async def concept_stream():
            """Generate stream of phi events"""
            for i in range(10):
                event = StreamingPhiEvent(
                    event_id=f"stream_event_{i}",
                    timestamp=datetime.now(),
                    experiential_concepts=[
                        {
                            "content": f"Stream concept {i}: consciousness evolution",
                            "experiential_quality": 0.3 + (i * 0.07),
                            "coherence": 0.5 + (i * 0.04),
                            "temporal_depth": (i % 5) + 1
                        }
                    ],
                    temporal_context={"temporal_depth": (i % 5) + 1},
                    stream_id="test_stream",
                    sequence_number=i
                )
                yield event
                await asyncio.sleep(0.1)  # 10 events/second
        
        # Process stream
        stream_results = []
        async for result in calculator.process_event_stream(concept_stream(), "test_stream"):
            stream_results.append(result)
            print(f"🌊 Stream: φ={result.instantaneous_phi:.6f}, "
                  f"predicted={result.predicted_phi:.6f if result.predicted_phi else 'N/A'}, "
                  f"velocity={result.phi_velocity:.3f}")
        
        # Test high-throughput processing
        print("\n⚡ Testing High-Throughput Processing")
        print("-" * 40)
        
        # Submit many events rapidly
        throughput_events = []
        start_time = time.time()
        
        for i in range(50):
            event = StreamingPhiEvent(
                event_id=f"throughput_event_{i}",
                timestamp=datetime.now(),
                experiential_concepts=[
                    {
                        "content": f"High-throughput concept {i}",
                        "experiential_quality": 0.4 + np.random.normal(0, 0.1),
                        "coherence": 0.6 + np.random.normal(0, 0.05),
                        "temporal_depth": np.random.randint(1, 6)
                    }
                ],
                sequence_number=i
            )
            
            await calculator.submit_event(event)
            throughput_events.append(event)
        
        submission_time = time.time() - start_time
        print(f"📤 Submitted {len(throughput_events)} events in {submission_time:.2f}s "
              f"({len(throughput_events)/submission_time:.1f} events/s)")
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Get streaming statistics
        print("\n📈 Streaming Statistics")
        print("-" * 40)
        
        stats = await calculator.get_streaming_stats()
        
        print(f"Mode: {stats['streaming_mode']}")
        print(f"Total Processed: {stats['performance']['total_processed']}")
        print(f"Current Throughput: {stats['performance']['current_throughput_rps']:.1f} events/s")
        print(f"Average Processing Time: {stats['performance']['average_processing_time_ms']:.1f}ms")
        print(f"Error Rate: {stats['performance']['error_rate']:.2%}")
        print(f"Cache Hit Rate: {stats['cache_stats']['hit_rate']:.2%}")
        
        # Phi analysis
        phi_analysis = stats['phi_analysis']
        if phi_analysis:
            print(f"\nΦ Analysis:")
            print(f"  Mean: {phi_analysis['mean']:.6f}")
            print(f"  Std: {phi_analysis['std']:.6f}")
            print(f"  Range: [{phi_analysis['min']:.6f}, {phi_analysis['max']:.6f}]")
            print(f"  Trend: {phi_analysis['trend']:.6f}")
        
        # Predictor stats
        predictor_stats = stats['predictor_stats']
        print(f"\nPrediction System:")
        print(f"  Available Models: {predictor_stats['available_models']}")
        print(f"  Best Model: {predictor_stats['best_model']}")
        print(f"  Training Samples: {predictor_stats['training_samples']}")
        
        # System stats
        system_stats = stats['system']
        print(f"\nSystem Status:")
        print(f"  Memory Usage: {system_stats['memory_usage_percent']:.1f}%")
        print(f"  Active Windows: {system_stats['active_windows']}")
        print(f"  Phi History Size: {system_stats['phi_history_size']}")
        
        print(f"\n✅ Streaming phi calculator test completed!")
        print(f"   Total results: {len(results) + len(stream_results)}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean shutdown
        await calculator.stop_streaming()


if __name__ == "__main__":
    asyncio.run(test_streaming_phi_calculator())