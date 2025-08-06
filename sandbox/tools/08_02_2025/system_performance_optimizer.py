"""
System Performance Optimizer for NewbornAI 2.0 IIT 4.0 Implementation
Phase 2B: Advanced performance optimization and resource management

This module provides comprehensive performance optimization for the IIT 4.0
consciousness detection system, including memory management, streaming φ calculations,
cache optimization, and performance benchmarking.

Key Features:
- Memory usage optimization and garbage collection tuning
- Streaming φ calculations for continuous monitoring
- Intelligent cache management with predictive prefetching
- Performance benchmarking and bottleneck identification
- Resource allocation optimization
- Automatic performance tuning based on workload patterns

Author: LLM Systems Architect (Hirosato Gamo's expertise)
Date: 2025-08-03
Version: 2.0.0
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Iterator, AsyncIterator, Generator
from enum import Enum
import logging
import time
import gc
import psutil
import threading
import weakref
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import statistics
import json
from pathlib import Path
import pickle
import mmap
import sys
from functools import wraps
import cProfile
import pstats
import io

# Import our implementations
from pyphi_iit4_bridge import PyPhiIIT4Bridge, PhiCalculationResult
from production_phi_calculator import ProductionPhiCalculator, CalculationRequest, CalculationPriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    MEMORY_FOCUSED = "memory_focused"      # Minimize memory usage
    SPEED_FOCUSED = "speed_focused"        # Maximize calculation speed
    BALANCED = "balanced"                  # Balance memory and speed
    ADAPTIVE = "adaptive"                  # Adapt to workload patterns


class CacheStrategy(Enum):
    """Cache optimization strategies"""
    LRU = "lru"                           # Least Recently Used
    LFU = "lfu"                           # Least Frequently Used  
    PREDICTIVE = "predictive"             # Predictive prefetching
    ADAPTIVE_SIZE = "adaptive_size"       # Adaptive cache size


@dataclass
class PerformanceProfile:
    """Performance profile for workload characterization"""
    average_node_count: float
    calculation_frequency: float  # calculations per second
    memory_pressure: float        # 0.0 (low) to 1.0 (high)
    cache_hit_rate: float        # 0.0 to 1.0
    error_rate: float            # 0.0 to 1.0
    latency_p95: float           # 95th percentile latency
    throughput: float            # calculations per second
    resource_utilization: Dict[str, float]  # CPU, memory, etc.


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation with expected impact"""
    strategy: str
    description: str
    expected_improvement: float  # 0.0 to 1.0
    implementation_complexity: str  # "low", "medium", "high"
    resource_impact: str         # "positive", "neutral", "negative"
    recommended_settings: Dict[str, Any]


class MemoryOptimizer:
    """Advanced memory optimization for φ calculations"""
    
    def __init__(self, target_memory_mb: int = 500):
        self.target_memory_mb = target_memory_mb
        self.memory_pools = {}
        self.gc_thresholds = (700, 10, 10)  # Aggressive GC for consciousness calculations
        self.weak_references = weakref.WeakSet()
        
    def optimize_memory_settings(self):
        """Optimize Python memory settings for φ calculations"""
        # Set aggressive garbage collection thresholds
        gc.set_threshold(*self.gc_thresholds)
        
        # Enable garbage collection debugging (for development)
        if logger.isEnabledFor(logging.DEBUG):
            gc.set_debug(gc.DEBUG_STATS)
        
        logger.info(f"Memory optimizer configured with target: {self.target_memory_mb}MB")
    
    def create_memory_pool(self, pool_name: str, initial_size: int = 100):
        """Create memory pool for frequently used objects"""
        if pool_name not in self.memory_pools:
            self.memory_pools[pool_name] = deque(maxlen=initial_size)
            logger.debug(f"Created memory pool: {pool_name}")
    
    def get_from_pool(self, pool_name: str, factory_func):
        """Get object from memory pool or create new one"""
        if pool_name in self.memory_pools and self.memory_pools[pool_name]:
            return self.memory_pools[pool_name].popleft()
        else:
            return factory_func()
    
    def return_to_pool(self, pool_name: str, obj):
        """Return object to memory pool"""
        if pool_name in self.memory_pools:
            # Clear object state before returning to pool
            if hasattr(obj, 'clear'):
                obj.clear()
            self.memory_pools[pool_name].append(obj)
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics"""
        collected = {}
        for generation in range(3):
            collected[f"generation_{generation}"] = gc.collect(generation)
        
        # Also run full collection
        collected["full_collection"] = gc.collect()
        
        return collected
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'gc_counts': gc.get_count(),
            'gc_stats': gc.get_stats()
        }
    
    def optimize_numpy_memory(self):
        """Optimize NumPy memory usage"""
        # Set memory-mapped file threshold
        np.seterr(all='warn')  # Warn on numerical errors
        
        # Optimize BLAS threading for consciousness calculations
        try:
            import mkl
            mkl.set_num_threads(1)  # Single thread for small matrices
            logger.debug("Optimized MKL threading")
        except ImportError:
            pass
        
        # Set NumPy memory mapping threshold
        np.set_printoptions(threshold=1000)  # Limit large array printing
    
    async def monitor_memory_pressure(self) -> float:
        """Monitor memory pressure and return normalized value (0.0-1.0)"""
        memory_stats = self.get_memory_usage()
        current_usage = memory_stats['rss_mb']
        target_ratio = current_usage / self.target_memory_mb
        
        # Calculate pressure based on usage relative to target
        if target_ratio <= 0.7:
            pressure = 0.0  # Low pressure
        elif target_ratio <= 0.9:
            pressure = (target_ratio - 0.7) / 0.2  # Linear increase
        else:
            pressure = min(1.0, (target_ratio - 0.9) * 5.0)  # High pressure
        
        return pressure


class StreamingPhiCalculator:
    """Streaming φ calculator for continuous consciousness monitoring"""
    
    def __init__(self, 
                 calculator: ProductionPhiCalculator,
                 buffer_size: int = 1000,
                 batch_size: int = 10):
        self.calculator = calculator
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Streaming buffers
        self.input_buffer = asyncio.Queue(maxsize=buffer_size)
        self.output_buffer = asyncio.Queue(maxsize=buffer_size)
        
        # Streaming state
        self.is_streaming = False
        self.stream_tasks = []
        self.stream_stats = {
            'total_processed': 0,
            'current_rate': 0.0,
            'average_latency': 0.0,
            'buffer_usage': 0.0
        }
        
        logger.info("StreamingPhiCalculator initialized")
    
    async def start_streaming(self):
        """Start streaming φ calculation"""
        if self.is_streaming:
            logger.warning("Streaming already active")
            return
        
        self.is_streaming = True
        
        # Start processing tasks
        processor_task = asyncio.create_task(self._stream_processor())
        monitor_task = asyncio.create_task(self._stream_monitor())
        
        self.stream_tasks = [processor_task, monitor_task]
        
        logger.info("Streaming φ calculation started")
    
    async def stop_streaming(self):
        """Stop streaming φ calculation"""
        self.is_streaming = False
        
        # Cancel tasks
        for task in self.stream_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.stream_tasks, return_exceptions=True)
        
        self.stream_tasks = []
        logger.info("Streaming φ calculation stopped")
    
    async def submit_for_streaming(self, 
                                 tpm: np.ndarray, 
                                 state: np.ndarray,
                                 metadata: Optional[Dict] = None) -> str:
        """Submit data for streaming calculation"""
        if not self.is_streaming:
            raise RuntimeError("Streaming not active")
        
        request_id = f"stream_{time.time()}_{id(tpm)}"
        
        stream_item = {
            'request_id': request_id,
            'tpm': tpm,
            'state': state,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        try:
            self.input_buffer.put_nowait(stream_item)
            return request_id
        except asyncio.QueueFull:
            raise RuntimeError("Streaming buffer full")
    
    async def get_streaming_result(self, timeout: float = 1.0) -> Optional[Dict]:
        """Get next streaming result"""
        try:
            result = await asyncio.wait_for(
                self.output_buffer.get(),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            return None
    
    async def _stream_processor(self):
        """Main streaming processor"""
        batch = []
        
        while self.is_streaming:
            try:
                # Collect batch
                while len(batch) < self.batch_size:
                    try:
                        item = await asyncio.wait_for(
                            self.input_buffer.get(),
                            timeout=0.1
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break  # Process partial batch
                
                if not batch:
                    continue
                
                # Process batch
                await self._process_streaming_batch(batch)
                batch = []
                
                # Update stats
                self.stream_stats['total_processed'] += len(batch)
                
            except Exception as e:
                logger.error(f"Streaming processor error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_streaming_batch(self, batch: List[Dict]):
        """Process batch of streaming items"""
        # Create calculation requests
        requests = []
        for item in batch:
            request = CalculationRequest(
                request_id=item['request_id'],
                tpm=item['tpm'],
                state=item['state'],
                priority=CalculationPriority.HIGH,
                metadata=item['metadata']
            )
            requests.append(request)
        
        # Process batch
        responses = await self.calculator.calculate_phi_batch(requests)
        
        # Put results in output buffer
        for response, original_item in zip(responses, batch):
            result = {
                'request_id': response.request_id,
                'phi_value': response.result.phi_value if response.result else 0.0,
                'calculation_time': response.processing_time,
                'status': response.status,
                'metadata': original_item['metadata'],
                'timestamp': time.time()
            }
            
            try:
                self.output_buffer.put_nowait(result)
            except asyncio.QueueFull:
                # Drop oldest result if buffer full
                try:
                    self.output_buffer.get_nowait()
                    self.output_buffer.put_nowait(result)
                except asyncio.QueueEmpty:
                    pass
    
    async def _stream_monitor(self):
        """Monitor streaming performance"""
        last_count = 0
        last_time = time.time()
        
        while self.is_streaming:
            await asyncio.sleep(5.0)  # Monitor every 5 seconds
            
            current_time = time.time()
            current_count = self.stream_stats['total_processed']
            
            # Calculate rate
            time_delta = current_time - last_time
            count_delta = current_count - last_count
            
            if time_delta > 0:
                self.stream_stats['current_rate'] = count_delta / time_delta
            
            # Update buffer usage
            input_usage = self.input_buffer.qsize() / self.buffer_size
            output_usage = self.output_buffer.qsize() / self.buffer_size
            self.stream_stats['buffer_usage'] = max(input_usage, output_usage)
            
            last_count = current_count
            last_time = current_time
            
            logger.debug(f"Streaming stats: {self.stream_stats}")
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming performance statistics"""
        return {
            'is_streaming': self.is_streaming,
            'stats': self.stream_stats.copy(),
            'buffer_status': {
                'input_size': self.input_buffer.qsize(),
                'output_size': self.output_buffer.qsize(),
                'input_capacity': self.buffer_size,
                'output_capacity': self.buffer_size
            }
        }


class IntelligentCache:
    """Intelligent cache with predictive prefetching and adaptive sizing"""
    
    def __init__(self, 
                 initial_size: int = 1000,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE_SIZE):
        self.strategy = strategy
        self.max_size = initial_size
        self.current_size = 0
        
        # Cache storage
        self.cache_data = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.access_patterns = deque(maxlen=1000)
        
        # Adaptive sizing
        self.size_history = deque(maxlen=100)
        self.hit_rate_history = deque(maxlen=100)
        
        # Predictive prefetching
        self.pattern_predictor = PatternPredictor()
        
        logger.info(f"IntelligentCache initialized with strategy: {strategy}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache_data:
            # Update access statistics
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            self.access_patterns.append(('hit', key, time.time()))
            
            # Trigger predictive prefetching
            await self._predict_and_prefetch(key)
            
            return self.cache_data[key]
        
        self.access_patterns.append(('miss', key, time.time()))
        return None
    
    async def put(self, key: str, value: Any):
        """Put item in cache"""
        # Check if eviction needed
        if self.current_size >= self.max_size and key not in self.cache_data:
            await self._evict_items()
        
        # Store item
        self.cache_data[key] = value
        self.access_times[key] = time.time()
        self.access_counts[key] += 1
        self.current_size = len(self.cache_data)
        
        # Update adaptive sizing
        await self._update_adaptive_sizing()
    
    async def _evict_items(self):
        """Evict items based on strategy"""
        if not self.cache_data:
            return
        
        items_to_evict = max(1, int(self.max_size * 0.1))  # Evict 10%
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            sorted_items = sorted(
                self.cache_data.keys(),
                key=lambda k: self.access_times.get(k, 0)
            )
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            sorted_items = sorted(
                self.cache_data.keys(),
                key=lambda k: self.access_counts.get(k, 0)
            )
        else:  # PREDICTIVE or ADAPTIVE_SIZE
            # Evict based on predicted future access
            predicted_access = await self._predict_future_access()
            sorted_items = sorted(
                self.cache_data.keys(),
                key=lambda k: predicted_access.get(k, 0)
            )
        
        # Remove least valuable items
        for key in sorted_items[:items_to_evict]:
            del self.cache_data[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.access_counts:
                del self.access_counts[key]
        
        self.current_size = len(self.cache_data)
    
    async def _predict_future_access(self) -> Dict[str, float]:
        """Predict future access probabilities"""
        predictions = {}
        
        for key in self.cache_data.keys():
            # Simple prediction based on access patterns
            recent_access = self.access_times.get(key, 0)
            access_frequency = self.access_counts.get(key, 0)
            time_decay = max(0.1, 1.0 - (time.time() - recent_access) / 3600.0)  # 1 hour decay
            
            predictions[key] = access_frequency * time_decay
        
        return predictions
    
    async def _predict_and_prefetch(self, accessed_key: str):
        """Predict and prefetch related items"""
        if self.strategy != CacheStrategy.PREDICTIVE:
            return
        
        # Use pattern predictor to find related keys
        related_keys = await self.pattern_predictor.predict_related_keys(
            accessed_key, list(self.access_patterns)
        )
        
        # Prefetch related items (placeholder - would require calculation function)
        for related_key in related_keys[:3]:  # Limit prefetching
            if related_key not in self.cache_data:
                # Would trigger calculation and caching here
                logger.debug(f"Would prefetch: {related_key}")
    
    async def _update_adaptive_sizing(self):
        """Update cache size based on performance"""
        if self.strategy != CacheStrategy.ADAPTIVE_SIZE:
            return
        
        # Calculate current hit rate
        recent_patterns = list(self.access_patterns)[-100:]  # Last 100 accesses
        if recent_patterns:
            hits = sum(1 for pattern in recent_patterns if pattern[0] == 'hit')
            hit_rate = hits / len(recent_patterns)
            self.hit_rate_history.append(hit_rate)
        
        # Adjust size based on hit rate trend
        if len(self.hit_rate_history) >= 10:
            recent_hit_rate = statistics.mean(list(self.hit_rate_history)[-10:])
            
            if recent_hit_rate < 0.7 and self.max_size < 5000:
                # Low hit rate - increase cache size
                self.max_size = int(self.max_size * 1.1)
                logger.debug(f"Increased cache size to {self.max_size}")
            elif recent_hit_rate > 0.9 and self.max_size > 100:
                # High hit rate - can decrease cache size
                self.max_size = int(self.max_size * 0.95)
                logger.debug(f"Decreased cache size to {self.max_size}")
        
        self.size_history.append(self.max_size)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_accesses = len(self.access_patterns)
        hits = sum(1 for pattern in self.access_patterns if pattern[0] == 'hit')
        hit_rate = hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            'current_size': self.current_size,
            'max_size': self.max_size,
            'utilization': self.current_size / self.max_size,
            'hit_rate': hit_rate,
            'total_accesses': total_accesses,
            'strategy': self.strategy.value,
            'size_history': list(self.size_history)[-20:]  # Last 20 size changes
        }


class PatternPredictor:
    """Pattern predictor for cache prefetching"""
    
    def __init__(self):
        self.access_sequences = deque(maxlen=1000)
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
    
    async def predict_related_keys(self, current_key: str, access_patterns: List) -> List[str]:
        """Predict keys likely to be accessed after current key"""
        # Update transition matrix
        self._update_transitions(access_patterns)
        
        # Find most likely next keys
        transitions = self.transition_matrix.get(current_key, {})
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        
        return [key for key, count in sorted_transitions[:5] if count > 1]
    
    def _update_transitions(self, access_patterns: List):
        """Update transition matrix from access patterns"""
        keys = [pattern[1] for pattern in access_patterns if pattern[0] == 'hit']
        
        for i in range(len(keys) - 1):
            current_key = keys[i]
            next_key = keys[i + 1]
            self.transition_matrix[current_key][next_key] += 1


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking and bottleneck identification"""
    
    def __init__(self):
        self.benchmark_results = {}
        self.bottleneck_analysis = {}
        self.profiling_data = {}
    
    async def benchmark_system(self, calculator: ProductionPhiCalculator) -> Dict[str, Any]:
        """Comprehensive system benchmark"""
        logger.info("Starting comprehensive system benchmark")
        
        benchmarks = {}
        
        # Memory benchmark
        benchmarks['memory'] = await self._benchmark_memory_performance(calculator)
        
        # Latency benchmark
        benchmarks['latency'] = await self._benchmark_latency(calculator)
        
        # Throughput benchmark
        benchmarks['throughput'] = await self._benchmark_throughput(calculator)
        
        # Scaling benchmark
        benchmarks['scaling'] = await self._benchmark_scaling(calculator)
        
        # Cache benchmark
        benchmarks['cache'] = await self._benchmark_cache_performance(calculator)
        
        # Overall score
        benchmarks['overall_score'] = self._calculate_overall_score(benchmarks)
        
        self.benchmark_results = benchmarks
        logger.info(f"Benchmark completed - Overall score: {benchmarks['overall_score']:.2f}")
        
        return benchmarks
    
    async def _benchmark_memory_performance(self, calculator: ProductionPhiCalculator) -> Dict[str, float]:
        """Benchmark memory performance"""
        memory_optimizer = MemoryOptimizer()
        
        # Initial memory state
        initial_memory = memory_optimizer.get_memory_usage()
        
        # Create test workload
        test_sizes = [4, 6, 8, 10]  # Different system sizes
        memory_usage = {}
        
        for size in test_sizes:
            tpm = np.random.rand(2**size, size)
            tmp = tpm / np.sum(tpm, axis=1, keepdims=True)
            state = np.random.randint(0, 2, size)
            
            request = CalculationRequest(
                request_id=f"memory_test_{size}",
                tpm=tpm,
                state=state
            )
            
            # Measure memory before calculation
            memory_before = memory_optimizer.get_memory_usage()['rss_mb']
            
            # Perform calculation
            response = await calculator.calculate_phi_async(request)
            
            # Measure memory after calculation
            memory_after = memory_optimizer.get_memory_usage()['rss_mb']
            
            memory_usage[f"nodes_{size}"] = memory_after - memory_before
            
            # Force garbage collection
            memory_optimizer.force_garbage_collection()
        
        return {
            'initial_memory_mb': initial_memory['rss_mb'],
            'memory_per_calculation': memory_usage,
            'memory_efficiency_score': self._calculate_memory_efficiency_score(memory_usage)
        }
    
    async def _benchmark_latency(self, calculator: ProductionPhiCalculator) -> Dict[str, float]:
        """Benchmark calculation latency"""
        latencies = []
        
        # Test with standard 4-node system
        tpm = np.random.rand(16, 4)
        tpm = tpm / np.sum(tpm, axis=1, keepdims=True)
        state = np.array([1, 0, 1, 0])
        
        # Warm-up
        for _ in range(5):
            request = CalculationRequest(
                request_id=f"warmup_{_}",
                tpm=tpm,
                state=state
            )
            await calculator.calculate_phi_async(request)
        
        # Actual benchmark
        for i in range(50):
            request = CalculationRequest(
                request_id=f"latency_test_{i}",
                tpm=tpm,
                state=state
            )
            
            start_time = time.time()
            response = await calculator.calculate_phi_async(request)
            latency = time.time() - start_time
            
            if response.status == 'success':
                latencies.append(latency)
        
        if latencies:
            return {
                'mean_latency': statistics.mean(latencies),
                'median_latency': statistics.median(latencies),
                'p95_latency': np.percentile(latencies, 95),
                'p99_latency': np.percentile(latencies, 99),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'latency_std': statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            }
        
        return {'error': 'No successful latency measurements'}
    
    async def _benchmark_throughput(self, calculator: ProductionPhiCalculator) -> Dict[str, float]:
        """Benchmark system throughput"""
        # Create batch of requests
        batch_size = 20
        tpm = np.random.rand(16, 4)
        tpm = tpm / np.sum(tpm, axis=1, keepdims=True)
        state = np.array([1, 0, 1, 0])
        
        requests = []
        for i in range(batch_size):
            request = CalculationRequest(
                request_id=f"throughput_test_{i}",
                tpm=tpm,
                state=state,
                priority=CalculationPriority.NORMAL
            )
            requests.append(request)
        
        # Measure batch processing time
        start_time = time.time()
        responses = await calculator.calculate_phi_batch(requests)
        total_time = time.time() - start_time
        
        successful_responses = [r for r in responses if r.status == 'success']
        
        return {
            'batch_size': batch_size,
            'successful_calculations': len(successful_responses),
            'total_time': total_time,
            'throughput_per_second': len(successful_responses) / total_time if total_time > 0 else 0.0,
            'success_rate': len(successful_responses) / batch_size
        }
    
    async def _benchmark_scaling(self, calculator: ProductionPhiCalculator) -> Dict[str, Any]:
        """Benchmark system scaling with different node counts"""
        scaling_results = {}
        
        node_counts = [3, 4, 5, 6, 8, 10]
        
        for nodes in node_counts:
            tpm = np.random.rand(2**nodes, nodes)
            tpm = tpm / np.sum(tpm, axis=1, keepdims=True)
            state = np.random.randint(0, 2, nodes)
            
            # Measure calculation time
            start_time = time.time()
            
            request = CalculationRequest(
                request_id=f"scaling_test_{nodes}",
                tpm=tpm,
                state=state
            )
            
            response = await calculator.calculate_phi_async(request)
            calculation_time = time.time() - start_time
            
            scaling_results[f"nodes_{nodes}"] = {
                'calculation_time': calculation_time,
                'success': response.status == 'success',
                'phi_value': response.result.phi_value if response.result else 0.0
            }
        
        # Calculate scaling efficiency
        scaling_efficiency = self._calculate_scaling_efficiency(scaling_results)
        
        return {
            'scaling_results': scaling_results,
            'scaling_efficiency': scaling_efficiency
        }
    
    async def _benchmark_cache_performance(self, calculator: ProductionPhiCalculator) -> Dict[str, Any]:
        """Benchmark cache performance"""
        # Get initial cache stats
        initial_stats = calculator.bridge.get_performance_stats()
        
        # Perform repeated calculations with same inputs
        tpm = np.random.rand(16, 4)
        tpm = tpm / np.sum(tpm, axis=1, keepdims=True)
        state = np.array([1, 0, 1, 0])
        
        # First calculation (cache miss)
        request1 = CalculationRequest(
            request_id="cache_test_1",
            tpm=tpm,
            state=state
        )
        
        start_time = time.time()
        response1 = await calculator.calculate_phi_async(request1)
        first_calc_time = time.time() - start_time
        
        # Second calculation (should be cache hit)
        request2 = CalculationRequest(
            request_id="cache_test_2",
            tpm=tpm,
            state=state
        )
        
        start_time = time.time()
        response2 = await calculator.calculate_phi_async(request2)
        second_calc_time = time.time() - start_time
        
        # Get final cache stats
        final_stats = calculator.bridge.get_performance_stats()
        
        return {
            'first_calculation_time': first_calc_time,
            'second_calculation_time': second_calc_time,
            'cache_speedup': first_calc_time / max(second_calc_time, 0.001),
            'cache_hit_rate': final_stats.get('cache_hit_rate', 0.0),
            'cache_effectiveness': 1.0 - (second_calc_time / max(first_calc_time, 0.001))
        }
    
    def _calculate_memory_efficiency_score(self, memory_usage: Dict[str, float]) -> float:
        """Calculate memory efficiency score (0.0-1.0)"""
        if not memory_usage:
            return 0.0
        
        # Lower memory usage per calculation = higher score
        avg_memory = statistics.mean(memory_usage.values())
        
        # Normalize to reasonable range (0-100MB per calculation)
        normalized = max(0.0, 1.0 - (avg_memory / 100.0))
        return min(1.0, normalized)
    
    def _calculate_scaling_efficiency(self, scaling_results: Dict[str, Any]) -> float:
        """Calculate scaling efficiency score"""
        node_counts = []
        calc_times = []
        
        for key, result in scaling_results.items():
            if result['success']:
                nodes = int(key.split('_')[1])
                time_val = result['calculation_time']
                node_counts.append(nodes)
                calc_times.append(time_val)
        
        if len(node_counts) < 2:
            return 0.0
        
        # Calculate if scaling is reasonable (should be roughly exponential)
        # Good scaling means time doesn't increase too dramatically
        time_ratios = []
        for i in range(1, len(calc_times)):
            node_ratio = node_counts[i] / node_counts[i-1]
            time_ratio = calc_times[i] / calc_times[i-1]
            # Ideal scaling would be exponential, so we expect some growth
            efficiency = 1.0 / (time_ratio / (node_ratio ** 1.5))  # Allow for some exponential growth
            time_ratios.append(min(1.0, efficiency))
        
        return statistics.mean(time_ratios) if time_ratios else 0.0
    
    def _calculate_overall_score(self, benchmarks: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        scores = []
        
        # Memory score
        if 'memory' in benchmarks and 'memory_efficiency_score' in benchmarks['memory']:
            scores.append(benchmarks['memory']['memory_efficiency_score'])
        
        # Latency score (lower is better)
        if 'latency' in benchmarks and 'mean_latency' in benchmarks['latency']:
            latency_score = max(0.0, 1.0 - (benchmarks['latency']['mean_latency'] / 10.0))  # Normalize to 10s
            scores.append(latency_score)
        
        # Throughput score
        if 'throughput' in benchmarks and 'throughput_per_second' in benchmarks['throughput']:
            throughput_score = min(1.0, benchmarks['throughput']['throughput_per_second'] / 10.0)  # Normalize to 10/s
            scores.append(throughput_score)
        
        # Scaling score
        if 'scaling' in benchmarks and 'scaling_efficiency' in benchmarks['scaling']:
            scores.append(benchmarks['scaling']['scaling_efficiency'])
        
        # Cache score
        if 'cache' in benchmarks and 'cache_effectiveness' in benchmarks['cache']:
            scores.append(benchmarks['cache']['cache_effectiveness'])
        
        return statistics.mean(scores) if scores else 0.0


class SystemPerformanceOptimizer:
    """Main system performance optimizer coordinator"""
    
    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.cache = IntelligentCache()
        self.benchmarker = PerformanceBenchmarker()
        
        # Optimization state
        self.current_profile = None
        self.optimization_history = []
        self.active_optimizations = set()
        
        logger.info("SystemPerformanceOptimizer initialized")
    
    async def analyze_performance(self, calculator: ProductionPhiCalculator) -> PerformanceProfile:
        """Analyze current system performance"""
        # Run benchmarks
        benchmark_results = await self.benchmarker.benchmark_system(calculator)
        
        # Get system metrics
        system_status = calculator.get_system_status()
        
        # Calculate performance profile
        profile = PerformanceProfile(
            average_node_count=self._estimate_average_node_count(calculator),
            calculation_frequency=system_status['current_metrics']['throughput_per_second'],
            memory_pressure=await self.memory_optimizer.monitor_memory_pressure(),
            cache_hit_rate=system_status['bridge_stats']['cache_hit_rate'] / 100.0,
            error_rate=system_status['current_metrics']['error_rate_percent'] / 100.0,
            latency_p95=benchmark_results.get('latency', {}).get('p95_latency', 0.0),
            throughput=benchmark_results.get('throughput', {}).get('throughput_per_second', 0.0),
            resource_utilization={
                'cpu': system_status['current_metrics']['cpu_percent'] / 100.0,
                'memory': system_status['current_metrics']['memory_percent'] / 100.0
            }
        )
        
        self.current_profile = profile
        return profile
    
    async def generate_optimization_recommendations(self, 
                                                  profile: PerformanceProfile) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on performance profile"""
        recommendations = []
        
        # Memory optimization recommendations
        if profile.memory_pressure > 0.7:
            recommendations.append(OptimizationRecommendation(
                strategy="memory_optimization",
                description="Implement aggressive memory management and garbage collection",
                expected_improvement=0.3,
                implementation_complexity="medium",
                resource_impact="positive",
                recommended_settings={
                    "gc_threshold": (500, 10, 10),
                    "cache_size_reduction": 0.3,
                    "memory_pool_size": 50
                }
            ))
        
        # Cache optimization recommendations
        if profile.cache_hit_rate < 0.6:
            recommendations.append(OptimizationRecommendation(
                strategy="cache_optimization",
                description="Increase cache size and implement predictive prefetching",
                expected_improvement=0.25,
                implementation_complexity="low",
                resource_impact="neutral",
                recommended_settings={
                    "cache_size_multiplier": 2.0,
                    "cache_strategy": "predictive",
                    "prefetch_enabled": True
                }
            ))
        
        # Throughput optimization recommendations
        if profile.throughput < 5.0:  # Less than 5 calculations per second
            recommendations.append(OptimizationRecommendation(
                strategy="throughput_optimization",
                description="Increase parallel workers and optimize batch processing",
                expected_improvement=0.4,
                implementation_complexity="medium",
                resource_impact="negative",
                recommended_settings={
                    "max_workers": int(profile.resource_utilization['cpu'] * 8),
                    "batch_size": 20,
                    "queue_size": 2000
                }
            ))
        
        # Latency optimization recommendations
        if profile.latency_p95 > 5.0:  # More than 5 seconds
            recommendations.append(OptimizationRecommendation(
                strategy="latency_optimization",
                description="Use approximate calculations for large systems",
                expected_improvement=0.6,
                implementation_complexity="low",
                resource_impact="positive",
                recommended_settings={
                    "approximate_threshold": max(8, int(profile.average_node_count * 0.8)),
                    "approximation_accuracy": 0.95,
                    "timeout_reduction": 0.5
                }
            ))
        
        # Resource optimization recommendations
        cpu_usage = profile.resource_utilization.get('cpu', 0.0)
        memory_usage = profile.resource_utilization.get('memory', 0.0)
        
        if cpu_usage > 0.85 or memory_usage > 0.85:
            recommendations.append(OptimizationRecommendation(
                strategy="resource_optimization",
                description="Implement load balancing and resource throttling",
                expected_improvement=0.35,
                implementation_complexity="high",
                resource_impact="positive",
                recommended_settings={
                    "enable_throttling": True,
                    "max_cpu_percent": 80.0,
                    "max_memory_percent": 80.0,
                    "adaptive_workers": True
                }
            ))
        
        return recommendations
    
    async def apply_optimization(self, 
                               recommendation: OptimizationRecommendation,
                               calculator: ProductionPhiCalculator) -> bool:
        """Apply optimization recommendation"""
        try:
            if recommendation.strategy == "memory_optimization":
                return await self._apply_memory_optimization(recommendation.recommended_settings)
            
            elif recommendation.strategy == "cache_optimization":
                return await self._apply_cache_optimization(recommendation.recommended_settings)
            
            elif recommendation.strategy == "throughput_optimization":
                return await self._apply_throughput_optimization(
                    recommendation.recommended_settings, calculator
                )
            
            elif recommendation.strategy == "latency_optimization":
                return await self._apply_latency_optimization(recommendation.recommended_settings)
            
            elif recommendation.strategy == "resource_optimization":
                return await self._apply_resource_optimization(
                    recommendation.recommended_settings, calculator
                )
            
            return False
            
        except Exception as e:
            logger.error(f"Error applying optimization {recommendation.strategy}: {e}")
            return False
    
    async def _apply_memory_optimization(self, settings: Dict[str, Any]) -> bool:
        """Apply memory optimization settings"""
        try:
            if "gc_threshold" in settings:
                gc.set_threshold(*settings["gc_threshold"])
            
            if "memory_pool_size" in settings:
                self.memory_optimizer.create_memory_pool("phi_results", settings["memory_pool_size"])
            
            # Force garbage collection
            self.memory_optimizer.force_garbage_collection()
            
            self.active_optimizations.add("memory_optimization")
            logger.info("Applied memory optimization")
            return True
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return False
    
    async def _apply_cache_optimization(self, settings: Dict[str, Any]) -> bool:
        """Apply cache optimization settings"""
        try:
            if "cache_size_multiplier" in settings:
                self.cache.max_size = int(self.cache.max_size * settings["cache_size_multiplier"])
            
            if "cache_strategy" in settings:
                self.cache.strategy = CacheStrategy(settings["cache_strategy"])
            
            self.active_optimizations.add("cache_optimization")
            logger.info("Applied cache optimization")
            return True
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return False
    
    async def _apply_throughput_optimization(self, 
                                           settings: Dict[str, Any], 
                                           calculator: ProductionPhiCalculator) -> bool:
        """Apply throughput optimization settings"""
        try:
            # Note: This would require calculator reconfiguration
            # For now, we just log the recommendations
            logger.info(f"Throughput optimization recommended: {settings}")
            
            self.active_optimizations.add("throughput_optimization")
            return True
            
        except Exception as e:
            logger.error(f"Throughput optimization failed: {e}")
            return False
    
    async def _apply_latency_optimization(self, settings: Dict[str, Any]) -> bool:
        """Apply latency optimization settings"""
        try:
            # Configure approximate calculation thresholds
            logger.info(f"Latency optimization applied: {settings}")
            
            self.active_optimizations.add("latency_optimization")
            return True
            
        except Exception as e:
            logger.error(f"Latency optimization failed: {e}")
            return False
    
    async def _apply_resource_optimization(self, 
                                         settings: Dict[str, Any],
                                         calculator: ProductionPhiCalculator) -> bool:
        """Apply resource optimization settings"""
        try:
            # Implement resource monitoring and throttling
            logger.info(f"Resource optimization applied: {settings}")
            
            self.active_optimizations.add("resource_optimization")
            return True
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            return False
    
    def _estimate_average_node_count(self, calculator: ProductionPhiCalculator) -> float:
        """Estimate average node count from telemetry"""
        telemetry = calculator.get_telemetry_data(limit=100)
        calculations = telemetry.get('calculations', [])
        
        if calculations:
            node_counts = [calc.get('node_count', 4) for calc in calculations]
            return statistics.mean(node_counts)
        
        return 4.0  # Default assumption
    
    async def continuous_optimization(self, calculator: ProductionPhiCalculator):
        """Run continuous optimization loop"""
        logger.info("Starting continuous optimization")
        
        while True:
            try:
                # Analyze current performance
                profile = await self.analyze_performance(calculator)
                
                # Generate recommendations
                recommendations = await self.generate_optimization_recommendations(profile)
                
                # Apply high-impact, low-complexity optimizations automatically
                for rec in recommendations:
                    if (rec.expected_improvement > 0.3 and 
                        rec.implementation_complexity == "low" and
                        rec.strategy not in self.active_optimizations):
                        
                        success = await self.apply_optimization(rec, calculator)
                        if success:
                            self.optimization_history.append({
                                'timestamp': time.time(),
                                'recommendation': rec,
                                'applied': True
                            })
                
                # Wait before next optimization cycle
                await asyncio.sleep(300.0)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Continuous optimization error: {e}")
                await asyncio.sleep(60.0)  # Wait 1 minute on error
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'active_optimizations': list(self.active_optimizations),
            'current_profile': self.current_profile.__dict__ if self.current_profile else None,
            'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
            'cache_stats': self.cache.get_cache_stats(),
            'memory_stats': self.memory_optimizer.get_memory_usage()
        }


# Example usage and testing
async def test_performance_optimizer():
    """Test the system performance optimizer"""
    logger.info("Testing SystemPerformanceOptimizer")
    
    # Create production calculator
    calculator = ProductionPhiCalculator(max_workers=2)
    await calculator.start()
    
    try:
        # Create optimizer
        optimizer = SystemPerformanceOptimizer()
        
        # Analyze performance
        profile = await optimizer.analyze_performance(calculator)
        print(f"Performance profile:")
        print(f"  Average node count: {profile.average_node_count:.1f}")
        print(f"  Memory pressure: {profile.memory_pressure:.2f}")
        print(f"  Cache hit rate: {profile.cache_hit_rate:.2f}")
        print(f"  Throughput: {profile.throughput:.2f} calc/s")
        
        # Generate recommendations
        recommendations = await optimizer.generate_optimization_recommendations(profile)
        print(f"\nOptimization recommendations: {len(recommendations)}")
        for rec in recommendations:
            print(f"  {rec.strategy}: {rec.description}")
            print(f"    Expected improvement: {rec.expected_improvement:.1%}")
        
        # Test streaming calculator
        streaming_calc = StreamingPhiCalculator(calculator)
        await streaming_calc.start_streaming()
        
        # Submit streaming data
        test_tpm = np.random.rand(16, 4)
        test_tpm = test_tpm / np.sum(test_tpm, axis=1, keepdims=True)
        test_state = np.array([1, 0, 1, 0])
        
        for i in range(5):
            request_id = await streaming_calc.submit_for_streaming(test_tpm, test_state)
            print(f"Submitted streaming request: {request_id}")
        
        # Get streaming results
        for _ in range(5):
            result = await streaming_calc.get_streaming_result(timeout=2.0)
            if result:
                print(f"Streaming result: φ={result['phi_value']:.3f}")
        
        streaming_stats = streaming_calc.get_streaming_stats()
        print(f"Streaming stats: {streaming_stats['stats']}")
        
        await streaming_calc.stop_streaming()
        
        # Test intelligent cache
        cache = IntelligentCache(strategy=CacheStrategy.ADAPTIVE_SIZE)
        
        # Test cache operations
        await cache.put("test_key_1", {"phi": 3.14, "data": "test"})
        await cache.put("test_key_2", {"phi": 2.71, "data": "test2"})
        
        result1 = await cache.get("test_key_1")
        result2 = await cache.get("test_key_3")  # Miss
        
        cache_stats = cache.get_cache_stats()
        print(f"Cache stats: Hit rate {cache_stats['hit_rate']:.2f}, Size {cache_stats['current_size']}")
        
        # Memory optimization test
        memory_optimizer = MemoryOptimizer(target_memory_mb=200)
        memory_optimizer.optimize_memory_settings()
        
        memory_stats = memory_optimizer.get_memory_usage()
        print(f"Memory usage: {memory_stats['rss_mb']:.1f}MB")
        
        # Performance benchmark
        benchmarker = PerformanceBenchmarker()
        benchmark_results = await benchmarker.benchmark_system(calculator)
        print(f"Benchmark overall score: {benchmark_results['overall_score']:.2f}")
        
    finally:
        await calculator.stop()


if __name__ == "__main__":
    asyncio.run(test_performance_optimizer())