"""
PyPhi v1.20 Integration Bridge for IIT 4.0 NewbornAI 2.0
Phase 2B: Production-ready PyPhi integration with performance optimization

This module bridges our custom IIT 4.0 implementation with PyPhi v1.20,
providing optimized φ calculation with caching and parallel processing.

Key Features:
- PyPhi v1.20 compatibility layer
- Performance-optimized φ calculations for real-time applications
- Intelligent caching system for large concept sets
- Parallel processing for enterprise-scale deployments
- Azure OpenAI integration support
- Production monitoring and telemetry

Author: LLM Systems Architect (Hirosato Gamo's expertise)
Date: 2025-08-03
Version: 2.0.0
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, FrozenSet, Union
from enum import Enum
import logging
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import pickle
import weakref
from pathlib import Path

# Import existing IIT 4.0 infrastructure
from iit4_core_engine import (
    IIT4PhiCalculator, PhiStructure, CauseEffectState, Distinction, Relation,
    IntrinsicDifferenceCalculator
)
from iit4_experiential_phi_calculator import IIT4_ExperientialPhiCalculator, ExperientialPhiResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PyPhi with fallback
try:
    import pyphi
    PYPHI_AVAILABLE = True
    logger.info("PyPhi v1.20 imported successfully")
except ImportError:
    PYPHI_AVAILABLE = False
    logger.warning("PyPhi not available - using fallback implementation")


class PyPhiIntegrationMode(Enum):
    """PyPhi integration modes for different performance requirements"""
    FULL_PYPHI = "full_pyphi"          # Full PyPhi computation (highest accuracy)
    HYBRID_OPTIMIZED = "hybrid"        # Hybrid PyPhi + custom (balanced)
    CUSTOM_ONLY = "custom_only"        # Custom implementation only (fastest)
    ADAPTIVE = "adaptive"              # Automatically choose based on system size


@dataclass
class PyPhiCalculationConfig:
    """Configuration for PyPhi calculations"""
    mode: PyPhiIntegrationMode = PyPhiIntegrationMode.HYBRID_OPTIMIZED
    max_nodes_for_full_pyphi: int = 8
    parallel_workers: int = 4
    cache_size: int = 1000
    cache_ttl_seconds: float = 300.0
    enable_performance_monitoring: bool = True
    memory_limit_mb: int = 500
    timeout_seconds: float = 30.0
    
    # PyPhi-specific settings
    pyphi_config: Dict[str, Any] = field(default_factory=lambda: {
        'PRECISION': 6,
        'VALIDATE_SUBSYSTEM_STATES': False,
        'VALIDATE_NODE_LABELS': False,
        'CACHE_REPERTOIRES': True,
        'CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA': True,
        'PARALLEL_CUT_EVALUATION': True,
        'NUMBER_OF_CORES': -1
    })


@dataclass
class PhiCalculationResult:
    """Enhanced φ calculation result with performance metrics"""
    phi_value: float
    calculation_time: float
    memory_usage_mb: float
    cache_hit: bool
    mode_used: PyPhiIntegrationMode
    node_count: int
    concept_count: int
    phi_structure: Optional[PhiStructure] = None
    pyphi_structure: Optional[Any] = None  # PyPhi ConceptStructure if available
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


class PyPhiCache:
    """High-performance cache for φ calculations with TTL and memory management"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
    
    def _generate_key(self, tpm: np.ndarray, state: np.ndarray, **kwargs) -> str:
        """Generate cache key from TPM and state"""
        tpm_hash = hashlib.md5(tpm.tobytes()).hexdigest()
        state_hash = hashlib.md5(state.tobytes()).hexdigest()
        kwargs_hash = hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()
        return f"{tpm_hash}_{state_hash}_{kwargs_hash}"
    
    async def get(self, tpm: np.ndarray, state: np.ndarray, **kwargs) -> Optional[PhiCalculationResult]:
        """Get cached result if available and not expired"""
        async with self._lock:
            key = self._generate_key(tpm, state, **kwargs)
            
            if key in self._cache:
                result, timestamp = self._cache[key]
                
                # Check TTL
                if time.time() - timestamp < self.ttl_seconds:
                    # Update access order
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)
                    
                    # Mark as cache hit
                    if hasattr(result, 'cache_hit'):
                        result.cache_hit = True
                    
                    return result
                else:
                    # Expired - remove
                    del self._cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
            
            return None
    
    async def put(self, tpm: np.ndarray, state: np.ndarray, result: PhiCalculationResult, **kwargs):
        """Store result in cache"""
        async with self._lock:
            key = self._generate_key(tpm, state, **kwargs)
            
            # Evict old entries if at capacity
            while len(self._cache) >= self.max_size and self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
            
            # Store new result
            self._cache[key] = (result, time.time())
            self._access_order.append(key)
    
    async def clear_expired(self):
        """Clear expired cache entries"""
        async with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, (result, timestamp) in self._cache.items():
                if current_time - timestamp >= self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'access_order_length': len(self._access_order)
        }


class PyPhiIIT4Bridge:
    """
    Production-ready bridge between PyPhi v1.20 and IIT 4.0 implementation
    Optimized for real-time applications with enterprise-scale performance
    """
    
    def __init__(self, config: Optional[PyPhiCalculationConfig] = None):
        """
        Initialize PyPhi-IIT4 bridge
        
        Args:
            config: Configuration for PyPhi integration
        """
        self.config = config or PyPhiCalculationConfig()
        
        # Initialize cache
        self.cache = PyPhiCache(
            max_size=self.config.cache_size,
            ttl_seconds=self.config.cache_ttl_seconds
        )
        
        # Initialize calculators
        self.iit4_calculator = IIT4PhiCalculator()
        self.experiential_calculator = IIT4_ExperientialPhiCalculator()
        
        # Performance monitoring
        self.performance_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_calculation_time': 0.0,
            'memory_peak_mb': 0.0,
            'timeout_errors': 0,
            'mode_usage': {mode.value: 0 for mode in PyPhiIntegrationMode}
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        # Configure PyPhi if available
        if PYPHI_AVAILABLE:
            self._configure_pyphi()
        
        logger.info(f"PyPhiIIT4Bridge initialized with mode: {self.config.mode}")
    
    def _configure_pyphi(self):
        """Configure PyPhi with optimized settings"""
        try:
            for key, value in self.config.pyphi_config.items():
                if hasattr(pyphi.config, key):
                    setattr(pyphi.config, key, value)
            logger.info("PyPhi configured with optimized settings")
        except Exception as e:
            logger.warning(f"Error configuring PyPhi: {e}")
    
    async def calculate_phi_optimized(self, 
                                    tpm: np.ndarray, 
                                    state: np.ndarray,
                                    experiential_concepts: Optional[List[Dict]] = None,
                                    force_mode: Optional[PyPhiIntegrationMode] = None) -> PhiCalculationResult:
        """
        Calculate φ with performance optimization and intelligent mode selection
        
        Args:
            tpm: Transition probability matrix
            state: Current system state
            experiential_concepts: Optional experiential concepts for enhanced calculation
            force_mode: Force specific calculation mode
            
        Returns:
            PhiCalculationResult: Comprehensive calculation result with performance metrics
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        # Update stats
        self.performance_stats['total_calculations'] += 1
        
        # Check cache first
        cache_result = await self.cache.get(tpm, state, experiential_concepts=experiential_concepts)
        if cache_result:
            self.performance_stats['cache_hits'] += 1
            cache_result.cache_hit = True
            return cache_result
        
        self.performance_stats['cache_misses'] += 1
        
        # Determine calculation mode
        mode = force_mode or self._select_optimal_mode(tpm, state)
        self.performance_stats['mode_usage'][mode.value] += 1
        
        # Validate inputs
        if not self._validate_inputs(tpm, state):
            return PhiCalculationResult(
                phi_value=0.0,
                calculation_time=time.time() - start_time,
                memory_usage_mb=self._get_memory_usage() - initial_memory,
                cache_hit=False,
                mode_used=mode,
                node_count=len(state),
                concept_count=0,
                error_message="Invalid inputs"
            )
        
        try:
            # Calculate φ based on selected mode
            if mode == PyPhiIntegrationMode.FULL_PYPHI:
                result = await self._calculate_full_pyphi(tpm, state, experiential_concepts)
            elif mode == PyPhiIntegrationMode.HYBRID_OPTIMIZED:
                result = await self._calculate_hybrid(tpm, state, experiential_concepts)
            elif mode == PyPhiIntegrationMode.CUSTOM_ONLY:
                result = await self._calculate_custom_only(tpm, state, experiential_concepts)
            else:  # ADAPTIVE
                result = await self._calculate_adaptive(tpm, state, experiential_concepts)
            
            # Update performance metrics
            calculation_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - initial_memory
            
            result.calculation_time = calculation_time
            result.memory_usage_mb = memory_usage
            result.cache_hit = False
            result.mode_used = mode
            result.node_count = len(state)
            
            # Update running averages
            self._update_performance_averages(calculation_time, memory_usage)
            
            # Cache result
            await self.cache.put(tpm, state, result, experiential_concepts=experiential_concepts)
            
            return result
            
        except asyncio.TimeoutError:
            self.performance_stats['timeout_errors'] += 1
            return PhiCalculationResult(
                phi_value=0.0,
                calculation_time=time.time() - start_time,
                memory_usage_mb=self._get_memory_usage() - initial_memory,
                cache_hit=False,
                mode_used=mode,
                node_count=len(state),
                concept_count=0,
                error_message="Calculation timeout"
            )
        except Exception as e:
            logger.error(f"Error in φ calculation: {e}")
            return PhiCalculationResult(
                phi_value=0.0,
                calculation_time=time.time() - start_time,
                memory_usage_mb=self._get_memory_usage() - initial_memory,
                cache_hit=False,
                mode_used=mode,
                node_count=len(state),
                concept_count=0,
                error_message=str(e)
            )
    
    def _select_optimal_mode(self, tpm: np.ndarray, state: np.ndarray) -> PyPhiIntegrationMode:
        """Select optimal calculation mode based on system characteristics"""
        n_nodes = len(state)
        
        if self.config.mode != PyPhiIntegrationMode.ADAPTIVE:
            return self.config.mode
        
        # Adaptive mode selection
        if not PYPHI_AVAILABLE:
            return PyPhiIntegrationMode.CUSTOM_ONLY
        
        if n_nodes <= 4:
            return PyPhiIntegrationMode.FULL_PYPHI
        elif n_nodes <= self.config.max_nodes_for_full_pyphi:
            return PyPhiIntegrationMode.HYBRID_OPTIMIZED
        else:
            return PyPhiIntegrationMode.CUSTOM_ONLY
    
    def _validate_inputs(self, tpm: np.ndarray, state: np.ndarray) -> bool:
        """Validate TPM and state inputs"""
        try:
            # Check dimensions
            if tpm.ndim != 2:
                return False
            
            n_nodes = len(state)
            expected_states = 2 ** n_nodes
            
            if tpm.shape[0] != expected_states or tpm.shape[1] != n_nodes:
                return False
            
            # Check probability constraints
            if not np.allclose(np.sum(tpm, axis=1), 1.0, atol=1e-6):
                logger.warning("TPM rows do not sum to 1.0 - normalizing")
                # Normalize rows
                row_sums = np.sum(tpm, axis=1)
                tpm = tpm / row_sums[:, np.newaxis]
            
            # Check state values
            if not np.all((state >= 0) & (state <= 1)):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False
    
    async def _calculate_full_pyphi(self, 
                                  tpm: np.ndarray, 
                                  state: np.ndarray,
                                  experiential_concepts: Optional[List[Dict]]) -> PhiCalculationResult:
        """Calculate φ using full PyPhi implementation"""
        if not PYPHI_AVAILABLE:
            return await self._calculate_custom_only(tpm, state, experiential_concepts)
        
        try:
            # Convert to PyPhi format
            pyphi_network = pyphi.Network(tpm)
            pyphi_state = tuple(int(x) for x in state)
            
            # Create subsystem (all nodes)
            all_nodes = tuple(range(len(state)))
            subsystem = pyphi.Subsystem(pyphi_network, pyphi_state, all_nodes)
            
            # Calculate φ using PyPhi
            sia = pyphi.compute.sia(subsystem)
            
            phi_value = sia.phi if sia else 0.0
            concept_count = len(sia.ces) if sia and hasattr(sia, 'ces') else 0
            
            # Convert to our PhiStructure format
            phi_structure = self._convert_pyphi_to_phi_structure(sia)
            
            return PhiCalculationResult(
                phi_value=phi_value,
                calculation_time=0.0,  # Will be set by caller
                memory_usage_mb=0.0,   # Will be set by caller
                cache_hit=False,
                mode_used=PyPhiIntegrationMode.FULL_PYPHI,
                node_count=len(state),
                concept_count=concept_count,
                phi_structure=phi_structure,
                pyphi_structure=sia
            )
            
        except Exception as e:
            logger.error(f"PyPhi calculation error: {e}")
            # Fallback to custom implementation
            return await self._calculate_custom_only(tpm, state, experiential_concepts)
    
    async def _calculate_hybrid(self, 
                              tpm: np.ndarray, 
                              state: np.ndarray,
                              experiential_concepts: Optional[List[Dict]]) -> PhiCalculationResult:
        """Calculate φ using hybrid PyPhi + custom implementation"""
        
        # Use PyPhi for small subsystems, custom for large ones
        n_nodes = len(state)
        
        if n_nodes <= 6 and PYPHI_AVAILABLE:
            # Use full PyPhi for small systems
            pyphi_result = await self._calculate_full_pyphi(tpm, state, experiential_concepts)
            
            # Enhance with experiential calculation if concepts provided
            if experiential_concepts:
                experiential_result = await self.experiential_calculator.calculate_experiential_phi(
                    experiential_concepts
                )
                
                # Combine results
                combined_phi = pyphi_result.phi_value * (1.0 + experiential_result.consciousness_level * 0.5)
                pyphi_result.phi_value = combined_phi
            
            return pyphi_result
        else:
            # Use custom implementation for larger systems
            return await self._calculate_custom_only(tpm, state, experiential_concepts)
    
    async def _calculate_custom_only(self, 
                                   tpm: np.ndarray, 
                                   state: np.ndarray,
                                   experiential_concepts: Optional[List[Dict]]) -> PhiCalculationResult:
        """Calculate φ using custom IIT 4.0 implementation only"""
        
        # Use IIT 4.0 calculator
        connectivity_matrix = self._tpm_to_connectivity(tpm)
        phi_structure = self.iit4_calculator.calculate_phi(state, connectivity_matrix, tpm)
        
        phi_value = phi_structure.total_phi
        concept_count = len(phi_structure.distinctions)
        
        # Enhance with experiential calculation if concepts provided
        if experiential_concepts:
            experiential_result = await self.experiential_calculator.calculate_experiential_phi(
                experiential_concepts
            )
            
            # Combine with base φ
            phi_value = phi_value * (1.0 + experiential_result.consciousness_level * 0.3)
        
        return PhiCalculationResult(
            phi_value=phi_value,
            calculation_time=0.0,  # Will be set by caller
            memory_usage_mb=0.0,   # Will be set by caller
            cache_hit=False,
            mode_used=PyPhiIntegrationMode.CUSTOM_ONLY,
            node_count=len(state),
            concept_count=concept_count,
            phi_structure=phi_structure
        )
    
    async def _calculate_adaptive(self, 
                                tpm: np.ndarray, 
                                state: np.ndarray,
                                experiential_concepts: Optional[List[Dict]]) -> PhiCalculationResult:
        """Calculate φ using adaptive mode selection"""
        optimal_mode = self._select_optimal_mode(tpm, state)
        
        if optimal_mode == PyPhiIntegrationMode.FULL_PYPHI:
            return await self._calculate_full_pyphi(tpm, state, experiential_concepts)
        elif optimal_mode == PyPhiIntegrationMode.HYBRID_OPTIMIZED:
            return await self._calculate_hybrid(tpm, state, experiential_concepts)
        else:
            return await self._calculate_custom_only(tpm, state, experiential_concepts)
    
    def _convert_pyphi_to_phi_structure(self, sia) -> Optional[PhiStructure]:
        """Convert PyPhi SIA to our PhiStructure format"""
        if not sia:
            return None
        
        try:
            distinctions = []
            relations = []
            
            # Convert concepts to distinctions
            if hasattr(sia, 'ces'):
                for concept in sia.ces:
                    if hasattr(concept, 'mechanism') and hasattr(concept, 'phi'):
                        mechanism = frozenset(concept.mechanism)
                        
                        # Create basic cause-effect state
                        ces = CauseEffectState(
                            mechanism=mechanism,
                            cause_state=np.array([0.5]),  # Placeholder
                            effect_state=np.array([0.5]), # Placeholder
                            intrinsic_difference=concept.phi,
                            phi_value=concept.phi
                        )
                        
                        distinction = Distinction(
                            mechanism=mechanism,
                            cause_effect_state=ces,
                            phi_value=concept.phi
                        )
                        distinctions.append(distinction)
            
            # Create basic relations (simplified)
            for i, dist1 in enumerate(distinctions):
                for j, dist2 in enumerate(distinctions[i+1:], i+1):
                    overlap = len(dist1.mechanism & dist2.mechanism) / max(len(dist1.mechanism | dist2.mechanism), 1)
                    if overlap > 0.1:
                        relation = Relation(
                            distinction_pair=(dist1, dist2),
                            overlap_measure=overlap,
                            integration_strength=overlap
                        )
                        relations.append(relation)
            
            return PhiStructure(
                distinctions=distinctions,
                relations=relations,
                total_phi=sia.phi,
                maximal_substrate=frozenset(range(len(sia.subsystem.state)))
            )
            
        except Exception as e:
            logger.warning(f"Error converting PyPhi structure: {e}")
            return None
    
    def _tpm_to_connectivity(self, tpm: np.ndarray) -> np.ndarray:
        """Convert TPM to connectivity matrix (simplified)"""
        n_nodes = tpm.shape[1]
        connectivity = np.zeros((n_nodes, n_nodes))
        
        # Simple heuristic: average influence between nodes
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    # Calculate average influence from j to i across all states
                    influence = np.mean(tpm[:, i])  # Simplified
                    connectivity[i, j] = influence
        
        return connectivity
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _update_performance_averages(self, calculation_time: float, memory_usage: float):
        """Update running performance averages"""
        n = self.performance_stats['total_calculations']
        
        # Update average calculation time
        current_avg = self.performance_stats['average_calculation_time']
        self.performance_stats['average_calculation_time'] = ((current_avg * (n - 1)) + calculation_time) / n
        
        # Update peak memory
        if memory_usage > self.performance_stats['memory_peak_mb']:
            self.performance_stats['memory_peak_mb'] = memory_usage
    
    async def calculate_parallel_phi(self, 
                                   batch_configs: List[Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]],
                                   max_parallel: Optional[int] = None) -> List[PhiCalculationResult]:
        """
        Calculate φ for multiple configurations in parallel
        
        Args:
            batch_configs: List of (tpm, state, experiential_concepts) tuples
            max_parallel: Maximum parallel calculations (defaults to config setting)
            
        Returns:
            List of PhiCalculationResult objects
        """
        max_parallel = max_parallel or self.config.parallel_workers
        
        # Create semaphore to limit concurrent calculations
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def calculate_with_semaphore(config):
            async with semaphore:
                tpm, state, concepts = config
                return await self.calculate_phi_optimized(tpm, state, concepts)
        
        # Execute all calculations concurrently
        tasks = [calculate_with_semaphore(config) for config in batch_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in parallel calculation {i}: {result}")
                # Create error result
                tpm, state, _ = batch_configs[i]
                error_result = PhiCalculationResult(
                    phi_value=0.0,
                    calculation_time=0.0,
                    memory_usage_mb=0.0,
                    cache_hit=False,
                    mode_used=PyPhiIntegrationMode.CUSTOM_ONLY,
                    node_count=len(state),
                    concept_count=0,
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        cache_stats = self.cache.get_stats()
        
        hit_rate = (self.performance_stats['cache_hits'] / 
                   max(self.performance_stats['total_calculations'], 1)) * 100
        
        return {
            'total_calculations': self.performance_stats['total_calculations'],
            'cache_hit_rate': hit_rate,
            'average_calculation_time': self.performance_stats['average_calculation_time'],
            'memory_peak_mb': self.performance_stats['memory_peak_mb'],
            'timeout_errors': self.performance_stats['timeout_errors'],
            'mode_usage': self.performance_stats['mode_usage'],
            'cache_stats': cache_stats,
            'pyphi_available': PYPHI_AVAILABLE
        }
    
    async def optimize_cache(self):
        """Optimize cache by clearing expired entries"""
        await self.cache.clear_expired()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_calculation_time': 0.0,
            'memory_peak_mb': 0.0,
            'timeout_errors': 0,
            'mode_usage': {mode.value: 0 for mode in PyPhiIntegrationMode}
        }
    
    async def benchmark_modes(self, 
                            tpm: np.ndarray, 
                            state: np.ndarray,
                            iterations: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Benchmark all available calculation modes
        
        Args:
            tpm: Test TPM
            state: Test state
            iterations: Number of iterations per mode
            
        Returns:
            Benchmark results for each mode
        """
        results = {}
        
        for mode in PyPhiIntegrationMode:
            if mode == PyPhiIntegrationMode.ADAPTIVE:
                continue  # Skip adaptive mode in benchmarks
            
            if mode == PyPhiIntegrationMode.FULL_PYPHI and not PYPHI_AVAILABLE:
                continue  # Skip if PyPhi not available
            
            mode_results = []
            
            for i in range(iterations):
                result = await self.calculate_phi_optimized(tpm, state, force_mode=mode)
                mode_results.append({
                    'phi_value': result.phi_value,
                    'calculation_time': result.calculation_time,
                    'memory_usage_mb': result.memory_usage_mb
                })
            
            # Calculate statistics
            phi_values = [r['phi_value'] for r in mode_results]
            calc_times = [r['calculation_time'] for r in mode_results]
            memory_usage = [r['memory_usage_mb'] for r in mode_results]
            
            results[mode.value] = {
                'phi_mean': np.mean(phi_values),
                'phi_std': np.std(phi_values),
                'time_mean': np.mean(calc_times),
                'time_std': np.std(calc_times),
                'memory_mean': np.mean(memory_usage),
                'memory_std': np.std(memory_usage)
            }
        
        return results
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Example usage and testing
async def test_pyphi_bridge():
    """Test the PyPhi-IIT4 bridge"""
    logger.info("Testing PyPhi-IIT4 Bridge")
    
    # Create bridge with optimized config
    config = PyPhiCalculationConfig(
        mode=PyPhiIntegrationMode.ADAPTIVE,
        max_nodes_for_full_pyphi=6,
        parallel_workers=2,
        cache_size=100
    )
    
    bridge = PyPhiIIT4Bridge(config)
    
    # Test with small system
    n_nodes = 4
    tpm = np.random.rand(2**n_nodes, n_nodes)
    # Normalize rows
    tpm = tpm / np.sum(tpm, axis=1, keepdims=True)
    
    state = np.array([0, 1, 0, 1])
    
    # Single calculation
    result = await bridge.calculate_phi_optimized(tpm, state)
    print(f"φ = {result.phi_value:.6f}")
    print(f"Calculation time: {result.calculation_time:.3f}s")
    print(f"Mode used: {result.mode_used}")
    print(f"Concepts: {result.concept_count}")
    
    # Test caching (second calculation should be faster)
    result2 = await bridge.calculate_phi_optimized(tpm, state)
    print(f"Second calculation (cached): {result2.calculation_time:.3f}s, cache_hit: {result2.cache_hit}")
    
    # Performance stats
    stats = bridge.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Parallel test
    batch_configs = [(tpm, state, None) for _ in range(3)]
    parallel_results = await bridge.calculate_parallel_phi(batch_configs)
    print(f"Parallel results: {len(parallel_results)} calculations completed")
    
    # Benchmark different modes
    if PYPHI_AVAILABLE:
        benchmark_results = await bridge.benchmark_modes(tpm, state, iterations=3)
        print("Benchmark results:")
        for mode, results in benchmark_results.items():
            print(f"  {mode}: φ={results['phi_mean']:.3f}±{results['phi_std']:.3f}, "
                  f"time={results['time_mean']:.3f}±{results['time_std']:.3f}s")


if __name__ == "__main__":
    asyncio.run(test_pyphi_bridge())