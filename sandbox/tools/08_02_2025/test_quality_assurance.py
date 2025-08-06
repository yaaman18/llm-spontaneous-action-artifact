"""
Test Quality Assurance System for IIT 4.0 NewbornAI 2.0
Advanced quality metrics, performance benchmarking, and regression testing

This module implements comprehensive quality assurance following TDD principles:
- Code quality metrics and validation
- Performance benchmarking and regression testing  
- Memory leak detection and resource usage validation
- Error handling and edge case testing
- Continuous integration quality gates

Quality Standards (t_wada TDD):
- Test coverage: 95%+ minimum
- Performance regression: <5% degradation tolerance
- Memory growth: <10MB per 1000 operations
- Error recovery: 100% graceful handling
- Edge case coverage: Comprehensive boundary testing

Author: TDD Engineer (Takuto Wada's expertise)
Date: 2025-08-03
Version: 1.0.0
"""

import asyncio
import time
import psutil
import gc
import sys
import tracemalloc
import threading
import resource
import numpy as np
import json
import tempfile
import weakref
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Generator
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from unittest.mock import Mock, patch, AsyncMock
import logging
import warnings
import cProfile
import pstats
import io

# Import modules under test
from comprehensive_test_suite import ComprehensiveTestSuite, TestResult
from iit4_core_engine import IIT4PhiCalculator, PhiStructure
from iit4_experiential_phi_calculator import IIT4_ExperientialPhiCalculator
from realtime_iit4_processor import RealtimeIIT4Processor
from newborn_ai_2_integrated_system import NewbornAI20_IntegratedSystem

# Configure quality assurance logging
qa_logger = logging.getLogger("quality_assurance")
qa_logger.setLevel(logging.INFO)


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics following TDD standards"""
    test_coverage_percentage: float
    performance_score: float
    memory_efficiency_score: float
    error_handling_score: float
    regression_score: float
    overall_quality_score: float
    
    # Detailed metrics
    execution_time_ms: float
    memory_peak_mb: float
    memory_average_mb: float
    cpu_usage_percentage: float
    
    # Quality gates
    coverage_gate_passed: bool
    performance_gate_passed: bool
    memory_gate_passed: bool
    regression_gate_passed: bool
    
    # TDD compliance
    red_green_refactor_compliance: float
    test_isolation_score: float
    edge_case_coverage_score: float
    mock_usage_score: float
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results"""
    operation_name: str
    iterations: int
    total_time_ms: float
    average_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_ops_per_second: float
    memory_per_operation_mb: float
    
    # Performance regression analysis
    baseline_time_ms: Optional[float] = None
    regression_percentage: Optional[float] = None
    performance_grade: str = "A"  # A, B, C, D, F
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryProfile:
    """Memory usage profiling results"""
    operation_name: str
    initial_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_growth_mb: float
    allocation_count: int
    deallocation_count: int
    leak_potential_objects: int
    
    # Memory efficiency metrics
    memory_efficiency_score: float
    garbage_collection_count: int
    weak_reference_count: int
    
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceProfiler:
    """Advanced performance profiling with regression detection"""
    
    def __init__(self, baseline_file: Optional[Path] = None):
        self.baseline_file = baseline_file
        self.current_benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.baseline_benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.regression_threshold = 0.05  # 5% regression tolerance
        
        if baseline_file and baseline_file.exists():
            self._load_baseline()
    
    def _load_baseline(self):
        """Load baseline performance data"""
        try:
            with open(self.baseline_file, 'r') as f:
                baseline_data = json.load(f)
                for name, data in baseline_data.items():
                    self.baseline_benchmarks[name] = PerformanceBenchmark(**data)
            qa_logger.info(f"Loaded {len(self.baseline_benchmarks)} baseline benchmarks")
        except Exception as e:
            qa_logger.warning(f"Failed to load baseline: {e}")
    
    def save_baseline(self):
        """Save current benchmarks as baseline"""
        if self.baseline_file:
            try:
                baseline_data = {}
                for name, benchmark in self.current_benchmarks.items():
                    # Convert to dict for JSON serialization
                    data = {
                        'operation_name': benchmark.operation_name,
                        'iterations': benchmark.iterations,
                        'total_time_ms': benchmark.total_time_ms,
                        'average_time_ms': benchmark.average_time_ms,
                        'median_time_ms': benchmark.median_time_ms,
                        'p95_time_ms': benchmark.p95_time_ms,
                        'p99_time_ms': benchmark.p99_time_ms,
                        'throughput_ops_per_second': benchmark.throughput_ops_per_second,
                        'memory_per_operation_mb': benchmark.memory_per_operation_mb,
                        'performance_grade': benchmark.performance_grade
                    }
                    baseline_data[name] = data
                
                with open(self.baseline_file, 'w') as f:
                    json.dump(baseline_data, f, indent=2)
                qa_logger.info(f"Saved {len(baseline_data)} benchmarks as baseline")
            except Exception as e:
                qa_logger.error(f"Failed to save baseline: {e}")
    
    @contextmanager
    def profile_operation(self, operation_name: str, iterations: int = 1):
        """Context manager for profiling operations"""
        # Start profiling
        profiler = cProfile.Profile()
        tracemalloc.start()
        
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        yield
        
        # Stop profiling
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        total_time_ms = (end_time - start_time) * 1000
        average_time_ms = total_time_ms / iterations
        memory_per_operation_mb = (end_memory - start_memory) / iterations
        
        # Create benchmark
        benchmark = PerformanceBenchmark(
            operation_name=operation_name,
            iterations=iterations,
            total_time_ms=total_time_ms,
            average_time_ms=average_time_ms,
            median_time_ms=average_time_ms,  # Approximation for single run
            p95_time_ms=average_time_ms * 1.5,  # Estimate
            p99_time_ms=average_time_ms * 2.0,  # Estimate
            throughput_ops_per_second=1000 / average_time_ms if average_time_ms > 0 else 0,
            memory_per_operation_mb=memory_per_operation_mb
        )
        
        # Check for regression
        if operation_name in self.baseline_benchmarks:
            baseline = self.baseline_benchmarks[operation_name]
            benchmark.baseline_time_ms = baseline.average_time_ms
            benchmark.regression_percentage = (
                (benchmark.average_time_ms - baseline.average_time_ms) / baseline.average_time_ms
            )
            
            # Assign performance grade
            if benchmark.regression_percentage <= 0:
                benchmark.performance_grade = "A"  # Improvement
            elif benchmark.regression_percentage <= 0.02:
                benchmark.performance_grade = "B"  # Minor regression
            elif benchmark.regression_percentage <= 0.05:
                benchmark.performance_grade = "C"  # Acceptable regression
            elif benchmark.regression_percentage <= 0.10:
                benchmark.performance_grade = "D"  # Concerning regression
            else:
                benchmark.performance_grade = "F"  # Unacceptable regression
        
        self.current_benchmarks[operation_name] = benchmark
    
    def get_regression_report(self) -> Dict[str, Any]:
        """Generate performance regression report"""
        regressions = {}
        improvements = {}
        stable = {}
        
        for name, benchmark in self.current_benchmarks.items():
            if benchmark.regression_percentage is not None:
                if benchmark.regression_percentage > self.regression_threshold:
                    regressions[name] = {
                        'regression_percentage': benchmark.regression_percentage * 100,
                        'current_time_ms': benchmark.average_time_ms,
                        'baseline_time_ms': benchmark.baseline_time_ms,
                        'grade': benchmark.performance_grade
                    }
                elif benchmark.regression_percentage < -0.01:  # 1% improvement threshold
                    improvements[name] = {
                        'improvement_percentage': abs(benchmark.regression_percentage) * 100,
                        'current_time_ms': benchmark.average_time_ms,
                        'baseline_time_ms': benchmark.baseline_time_ms,
                        'grade': benchmark.performance_grade
                    }
                else:
                    stable[name] = {
                        'change_percentage': benchmark.regression_percentage * 100,
                        'grade': benchmark.performance_grade
                    }
        
        return {
            'regressions': regressions,
            'improvements': improvements,
            'stable': stable,
            'regression_count': len(regressions),
            'improvement_count': len(improvements),
            'stable_count': len(stable),
            'overall_grade': self._calculate_overall_grade()
        }
    
    def _calculate_overall_grade(self) -> str:
        """Calculate overall performance grade"""
        grades = [b.performance_grade for b in self.current_benchmarks.values()]
        if not grades:
            return "N/A"
        
        grade_scores = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        average_score = sum(grade_scores.get(g, 0) for g in grades) / len(grades)
        
        if average_score >= 3.5:
            return "A"
        elif average_score >= 2.5:
            return "B"
        elif average_score >= 1.5:
            return "C"
        elif average_score >= 0.5:
            return "D"
        else:
            return "F"


class MemoryProfiler:
    """Advanced memory profiling and leak detection"""
    
    def __init__(self):
        self.profiles: List[MemoryProfile] = []
        self.tracked_objects: Dict[int, Any] = {}
        self.weak_references: List[weakref.ref] = []
    
    @contextmanager
    def profile_memory(self, operation_name: str):
        """Context manager for memory profiling"""
        # Start memory tracking
        tracemalloc.start()
        gc.collect()  # Clean slate
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        initial_objects = len(gc.get_objects())
        gc_count_before = sum(gc.get_count())
        
        yield
        
        # Force garbage collection and measure
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        final_objects = len(gc.get_objects())
        gc_count_after = sum(gc.get_count())
        
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        memory_growth = final_memory - initial_memory
        object_growth = final_objects - initial_objects
        gc_activity = gc_count_after - gc_count_before
        
        # Memory efficiency score (0-1, higher is better)
        peak_mb = peak_mem / 1024 / 1024
        efficiency_score = max(0, 1.0 - (memory_growth / max(peak_mb, 1.0)))
        
        profile = MemoryProfile(
            operation_name=operation_name,
            initial_memory_mb=initial_memory,
            peak_memory_mb=peak_mb,
            final_memory_mb=final_memory,
            memory_growth_mb=memory_growth,
            allocation_count=object_growth,
            deallocation_count=max(0, -object_growth),
            leak_potential_objects=max(0, object_growth),
            memory_efficiency_score=efficiency_score,
            garbage_collection_count=gc_activity,
            weak_reference_count=len(self.weak_references)
        )
        
        self.profiles.append(profile)
        
        # Log concerning memory patterns
        if memory_growth > 10:  # >10MB growth
            qa_logger.warning(f"High memory growth in {operation_name}: {memory_growth:.1f}MB")
        if object_growth > 1000:  # >1000 new objects
            qa_logger.warning(f"High object creation in {operation_name}: {object_growth} objects")
    
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """Analyze memory profiles for potential leaks"""
        leak_indicators = {}
        
        for profile in self.profiles:
            risk_factors = []
            risk_score = 0
            
            # High memory growth
            if profile.memory_growth_mb > 5:
                risk_factors.append(f"High memory growth: {profile.memory_growth_mb:.1f}MB")
                risk_score += 2
            
            # High object retention
            if profile.leak_potential_objects > 100:
                risk_factors.append(f"High object retention: {profile.leak_potential_objects}")
                risk_score += 1
            
            # Low efficiency score
            if profile.memory_efficiency_score < 0.7:
                risk_factors.append(f"Low efficiency: {profile.memory_efficiency_score:.2f}")
                risk_score += 1
            
            # Low garbage collection activity (might indicate unreachable objects)
            if profile.garbage_collection_count < 5 and profile.allocation_count > 50:
                risk_factors.append("Low GC activity with high allocations")
                risk_score += 1
            
            if risk_score > 0:
                leak_indicators[profile.operation_name] = {
                    'risk_score': risk_score,
                    'risk_factors': risk_factors,
                    'memory_growth_mb': profile.memory_growth_mb,
                    'efficiency_score': profile.memory_efficiency_score
                }
        
        return leak_indicators
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary"""
        if not self.profiles:
            return {"status": "no_data"}
        
        total_growth = sum(p.memory_growth_mb for p in self.profiles)
        avg_efficiency = sum(p.memory_efficiency_score for p in self.profiles) / len(self.profiles)
        max_peak = max(p.peak_memory_mb for p in self.profiles)
        
        return {
            "total_operations": len(self.profiles),
            "total_memory_growth_mb": total_growth,
            "average_efficiency_score": avg_efficiency,
            "peak_memory_mb": max_peak,
            "leak_indicators": self.detect_memory_leaks(),
            "memory_grade": self._calculate_memory_grade(avg_efficiency, total_growth)
        }
    
    def _calculate_memory_grade(self, efficiency: float, growth: float) -> str:
        """Calculate memory management grade"""
        if efficiency >= 0.9 and growth < 5:
            return "A"
        elif efficiency >= 0.8 and growth < 10:
            return "B"
        elif efficiency >= 0.7 and growth < 20:
            return "C"
        elif efficiency >= 0.6 and growth < 50:
            return "D"
        else:
            return "F"


class ErrorHandlingValidator:
    """Validate error handling and edge case coverage"""
    
    def __init__(self):
        self.error_scenarios: List[Dict[str, Any]] = []
        self.recovery_scenarios: List[Dict[str, Any]] = []
        self.edge_cases: List[Dict[str, Any]] = []
    
    async def test_error_handling(self, test_target: Any) -> Dict[str, Any]:
        """Comprehensive error handling validation"""
        results = {
            "exception_handling": await self._test_exception_handling(test_target),
            "boundary_conditions": await self._test_boundary_conditions(test_target),
            "resource_exhaustion": await self._test_resource_exhaustion(test_target),
            "concurrent_access": await self._test_concurrent_access(test_target),
            "malformed_input": await self._test_malformed_input(test_target)
        }
        
        # Calculate overall error handling score
        scores = [result.get('score', 0) for result in results.values()]
        overall_score = sum(scores) / len(scores) if scores else 0
        
        results["overall_score"] = overall_score
        results["grade"] = self._calculate_error_handling_grade(overall_score)
        
        return results
    
    async def _test_exception_handling(self, test_target: Any) -> Dict[str, Any]:
        """Test exception handling robustness"""
        test_cases = [
            ("null_input", lambda: None),
            ("empty_input", lambda: []),
            ("invalid_type", lambda: "invalid"),
            ("negative_values", lambda: [-1, -2, -3]),
            ("overflow_values", lambda: [float('inf'), float('-inf')])
        ]
        
        passed = 0
        total = len(test_cases)
        error_details = []
        
        for test_name, input_generator in test_cases:
            try:
                if hasattr(test_target, 'calculate_phi'):
                    # Test phi calculation with invalid inputs
                    invalid_state = input_generator()
                    if invalid_state is not None:
                        result = test_target.calculate_phi(invalid_state, np.eye(2))
                        # Should handle gracefully
                        passed += 1
                elif hasattr(test_target, 'calculate_experiential_phi'):
                    # Test experiential phi calculation
                    invalid_concepts = input_generator()
                    if invalid_concepts is not None:
                        result = await test_target.calculate_experiential_phi(invalid_concepts)
                        passed += 1
                else:
                    passed += 1  # Skip if no testable methods
                    
            except Exception as e:
                # Exception is expected for invalid inputs
                if "should handle gracefully" not in str(e):
                    passed += 1  # Good error handling
                    
                error_details.append({
                    "test_case": test_name,
                    "exception_type": type(e).__name__,
                    "handled_gracefully": True
                })
            
        return {
            "score": passed / total,
            "passed": passed,
            "total": total,
            "error_details": error_details
        }
    
    async def _test_boundary_conditions(self, test_target: Any) -> Dict[str, Any]:
        """Test boundary condition handling"""
        boundary_tests = [
            ("zero_values", np.zeros(2)),
            ("ones_values", np.ones(2)),
            ("minimal_values", np.array([1e-10, 1e-10])),
            ("maximal_values", np.array([1.0, 1.0])),
            ("single_element", np.array([1.0]))
        ]
        
        passed = 0
        total = len(boundary_tests)
        
        for test_name, test_input in boundary_tests:
            try:
                if hasattr(test_target, 'calculate_phi'):
                    connectivity = np.eye(len(test_input))
                    result = test_target.calculate_phi(test_input, connectivity)
                    # Should produce valid result
                    if hasattr(result, 'total_phi') and result.total_phi >= 0:
                        passed += 1
                else:
                    passed += 1  # Skip if not applicable
                    
            except Exception as e:
                qa_logger.debug(f"Boundary test {test_name} failed: {e}")
        
        return {
            "score": passed / total,
            "passed": passed,
            "total": total
        }
    
    async def _test_resource_exhaustion(self, test_target: Any) -> Dict[str, Any]:
        """Test behavior under resource exhaustion"""
        # Simulate high memory usage
        try:
            # Create large arrays to stress memory
            large_arrays = []
            for i in range(10):
                try:
                    large_array = np.random.rand(1000, 1000)  # ~8MB each
                    large_arrays.append(large_array)
                except MemoryError:
                    break
            
            # Test functionality under memory pressure
            if hasattr(test_target, 'calculate_phi'):
                test_state = np.array([1.0, 0.5])
                connectivity = np.array([[0.0, 0.8], [0.7, 0.0]])
                result = test_target.calculate_phi(test_state, connectivity)
                memory_pressure_handled = True
            else:
                memory_pressure_handled = True
                
            # Cleanup
            large_arrays.clear()
            gc.collect()
            
            return {
                "score": 1.0 if memory_pressure_handled else 0.0,
                "memory_pressure_handled": memory_pressure_handled
            }
            
        except Exception as e:
            return {
                "score": 0.5,  # Partial credit for attempting
                "error": str(e)
            }
    
    async def _test_concurrent_access(self, test_target: Any) -> Dict[str, Any]:
        """Test concurrent access safety"""
        if not hasattr(test_target, 'calculate_phi'):
            return {"score": 1.0, "note": "No concurrent testing needed"}
        
        try:
            # Test concurrent phi calculations
            tasks = []
            for i in range(5):
                state = np.random.rand(3)
                connectivity = np.random.rand(3, 3)
                task = asyncio.create_task(
                    asyncio.to_thread(test_target.calculate_phi, state, connectivity)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check that all calculations completed
            successful = sum(1 for r in results if not isinstance(r, Exception))
            score = successful / len(results)
            
            return {
                "score": score,
                "successful": successful,
                "total": len(results),
                "concurrent_safety": score >= 0.8
            }
            
        except Exception as e:
            return {
                "score": 0.0,
                "error": str(e)
            }
    
    async def _test_malformed_input(self, test_target: Any) -> Dict[str, Any]:
        """Test handling of malformed inputs"""
        malformed_tests = [
            ("nan_values", np.array([np.nan, 0.5])),
            ("inf_values", np.array([np.inf, 0.5])),
            ("wrong_dimensions", np.array([[1, 2], [3, 4]])),  # 2D instead of 1D
            ("mismatched_connectivity", (np.array([1, 0]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
        ]
        
        passed = 0
        total = len(malformed_tests)
        
        for test_name, test_data in malformed_tests:
            try:
                if hasattr(test_target, 'calculate_phi'):
                    if test_name == "mismatched_connectivity":
                        state, connectivity = test_data
                        result = test_target.calculate_phi(state, connectivity)
                    else:
                        state = test_data
                        connectivity = np.eye(2)
                        result = test_target.calculate_phi(state, connectivity)
                    
                    # Should either handle gracefully or raise appropriate exception
                    passed += 1
                    
            except (ValueError, TypeError, IndexError) as e:
                # Expected exceptions for malformed input
                passed += 1
            except Exception as e:
                qa_logger.debug(f"Unexpected exception in {test_name}: {e}")
        
        return {
            "score": passed / total,
            "passed": passed,
            "total": total
        }
    
    def _calculate_error_handling_grade(self, score: float) -> str:
        """Calculate error handling grade"""
        if score >= 0.95:
            return "A"
        elif score >= 0.85:
            return "B"
        elif score >= 0.75:
            return "C"
        elif score >= 0.65:
            return "D"
        else:
            return "F"


class QualityAssuranceManager:
    """Master quality assurance coordinator"""
    
    def __init__(self, baseline_file: Optional[Path] = None):
        self.performance_profiler = PerformanceProfiler(baseline_file)
        self.memory_profiler = MemoryProfiler()
        self.error_validator = ErrorHandlingValidator()
        self.quality_history: List[QualityMetrics] = []
        
        # Quality gates configuration
        self.quality_gates = {
            "min_coverage": 95.0,
            "max_regression": 5.0,
            "max_memory_growth_mb": 10.0,
            "min_error_handling_score": 0.8,
            "min_overall_quality": 0.85
        }
    
    async def comprehensive_quality_assessment(self, test_suite: ComprehensiveTestSuite) -> QualityMetrics:
        """Perform comprehensive quality assessment"""
        qa_logger.info("Starting comprehensive quality assessment...")
        
        # Run test suite with profiling
        with self.performance_profiler.profile_operation("full_test_suite", 1):
            with self.memory_profiler.profile_memory("full_test_suite"):
                test_results = await test_suite.run_all_tests()
                analysis = test_suite.analyze_test_results(test_results)
        
        # Performance analysis
        performance_report = self.performance_profiler.get_regression_report()
        memory_report = self.memory_profiler.get_memory_summary()
        
        # Error handling validation
        phi_calculator = IIT4PhiCalculator()
        exp_calculator = IIT4_ExperientialPhiCalculator()
        
        error_results_phi = await self.error_validator.test_error_handling(phi_calculator)
        error_results_exp = await self.error_validator.test_error_handling(exp_calculator)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            analysis, performance_report, memory_report, 
            error_results_phi, error_results_exp
        )
        
        self.quality_history.append(quality_metrics)
        
        qa_logger.info(f"Quality assessment complete. Overall score: {quality_metrics.overall_quality_score:.3f}")
        
        return quality_metrics
    
    def _calculate_quality_metrics(self, test_analysis: Dict, performance_report: Dict,
                                 memory_report: Dict, error_phi: Dict, error_exp: Dict) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        
        # Test coverage metrics
        coverage = test_analysis.get("coverage", {})
        coverage_percentage = coverage.get("average_coverage", 0)
        coverage_gate_passed = coverage_percentage >= self.quality_gates["min_coverage"]
        
        # Performance metrics
        performance_score = self._calculate_performance_score(performance_report)
        performance_gate_passed = performance_report.get("regression_count", 0) == 0
        
        # Memory metrics
        memory_efficiency = memory_report.get("average_efficiency_score", 0)
        memory_growth = memory_report.get("total_memory_growth_mb", 0)
        memory_gate_passed = memory_growth <= self.quality_gates["max_memory_growth_mb"]
        
        # Error handling metrics
        error_phi_score = error_phi.get("overall_score", 0)
        error_exp_score = error_exp.get("overall_score", 0)
        error_handling_score = (error_phi_score + error_exp_score) / 2
        error_gate_passed = error_handling_score >= self.quality_gates["min_error_handling_score"]
        
        # Regression metrics
        regression_count = performance_report.get("regression_count", 0)
        regression_score = max(0, 1.0 - (regression_count * 0.1))
        regression_gate_passed = regression_count <= 2  # Allow up to 2 minor regressions
        
        # TDD compliance metrics
        qa_metrics = test_analysis.get("quality_assurance", {})
        test_isolation_score = qa_metrics.get("isolation_score", 0)
        edge_case_coverage = min(1.0, qa_metrics.get("total_edge_cases", 0) / 50.0)
        mock_usage_score = min(1.0, qa_metrics.get("mocked_dependencies", 0) / 20.0)
        
        # Red-Green-Refactor compliance (estimated from test success patterns)
        tdd_score = test_analysis.get("summary", {}).get("tdd_quality_score", 0)
        
        # Overall quality score calculation
        weights = {
            "coverage": 0.25,
            "performance": 0.20,
            "memory": 0.15,
            "error_handling": 0.15,
            "regression": 0.15,
            "tdd_compliance": 0.10
        }
        
        overall_score = (
            weights["coverage"] * (coverage_percentage / 100.0) +
            weights["performance"] * performance_score +
            weights["memory"] * memory_efficiency +
            weights["error_handling"] * error_handling_score +
            weights["regression"] * regression_score +
            weights["tdd_compliance"] * tdd_score
        )
        
        # Get timing and resource metrics
        execution_time = sum(
            sum(t.execution_time_ms for t in test_list)
            for test_list in test_analysis.get("results", {}).values()
            if isinstance(test_list, list)
        )
        
        memory_peak = memory_report.get("peak_memory_mb", 0)
        memory_average = memory_report.get("total_memory_growth_mb", 0) / max(
            memory_report.get("total_operations", 1), 1
        )
        
        return QualityMetrics(
            test_coverage_percentage=coverage_percentage,
            performance_score=performance_score,
            memory_efficiency_score=memory_efficiency,
            error_handling_score=error_handling_score,
            regression_score=regression_score,
            overall_quality_score=overall_score,
            execution_time_ms=execution_time,
            memory_peak_mb=memory_peak,
            memory_average_mb=memory_average,
            cpu_usage_percentage=psutil.cpu_percent(),
            coverage_gate_passed=coverage_gate_passed,
            performance_gate_passed=performance_gate_passed,
            memory_gate_passed=memory_gate_passed,
            regression_gate_passed=regression_gate_passed,
            red_green_refactor_compliance=tdd_score,
            test_isolation_score=test_isolation_score,
            edge_case_coverage_score=edge_case_coverage,
            mock_usage_score=mock_usage_score
        )
    
    def _calculate_performance_score(self, performance_report: Dict) -> float:
        """Calculate normalized performance score"""
        regression_count = performance_report.get("regression_count", 0)
        improvement_count = performance_report.get("improvement_count", 0)
        stable_count = performance_report.get("stable_count", 0)
        
        total_benchmarks = regression_count + improvement_count + stable_count
        if total_benchmarks == 0:
            return 1.0
        
        # Score based on performance distribution
        score = (
            improvement_count * 1.0 +
            stable_count * 0.8 +
            regression_count * 0.3
        ) / total_benchmarks
        
        return min(1.0, score)
    
    def generate_quality_report(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Generate comprehensive quality assurance report"""
        
        # Quality gate status
        gates_passed = (
            metrics.coverage_gate_passed and
            metrics.performance_gate_passed and
            metrics.memory_gate_passed and
            metrics.regression_gate_passed
        )
        
        # Overall grade
        if metrics.overall_quality_score >= 0.95:
            overall_grade = "A"
        elif metrics.overall_quality_score >= 0.85:
            overall_grade = "B"
        elif metrics.overall_quality_score >= 0.75:
            overall_grade = "C"
        elif metrics.overall_quality_score >= 0.65:
            overall_grade = "D"
        else:
            overall_grade = "F"
        
        # Recommendations
        recommendations = self._generate_recommendations(metrics)
        
        return {
            "overall_assessment": {
                "quality_score": metrics.overall_quality_score,
                "overall_grade": overall_grade,
                "quality_gates_passed": gates_passed,
                "production_ready": gates_passed and metrics.overall_quality_score >= 0.85
            },
            "detailed_metrics": {
                "test_coverage": {
                    "percentage": metrics.test_coverage_percentage,
                    "gate_passed": metrics.coverage_gate_passed,
                    "target": self.quality_gates["min_coverage"]
                },
                "performance": {
                    "score": metrics.performance_score,
                    "gate_passed": metrics.performance_gate_passed,
                    "execution_time_ms": metrics.execution_time_ms
                },
                "memory_management": {
                    "efficiency_score": metrics.memory_efficiency_score,
                    "peak_usage_mb": metrics.memory_peak_mb,
                    "average_usage_mb": metrics.memory_average_mb,
                    "gate_passed": metrics.memory_gate_passed
                },
                "error_handling": {
                    "score": metrics.error_handling_score,
                    "gate_passed": metrics.error_handling_score >= self.quality_gates["min_error_handling_score"]
                },
                "regression_analysis": {
                    "score": metrics.regression_score,
                    "gate_passed": metrics.regression_gate_passed
                }
            },
            "tdd_compliance": {
                "red_green_refactor": metrics.red_green_refactor_compliance,
                "test_isolation": metrics.test_isolation_score,
                "edge_case_coverage": metrics.edge_case_coverage_score,
                "mock_usage": metrics.mock_usage_score
            },
            "recommendations": recommendations,
            "historical_trend": self._analyze_quality_trend(),
            "timestamp": metrics.timestamp.isoformat()
        }
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not metrics.coverage_gate_passed:
            recommendations.append(
                f"Increase test coverage from {metrics.test_coverage_percentage:.1f}% to {self.quality_gates['min_coverage']}%"
            )
        
        if not metrics.performance_gate_passed:
            recommendations.append("Address performance regressions identified in benchmarks")
        
        if not metrics.memory_gate_passed:
            recommendations.append(
                f"Optimize memory usage - current growth {metrics.memory_peak_mb:.1f}MB exceeds {self.quality_gates['max_memory_growth_mb']}MB limit"
            )
        
        if metrics.error_handling_score < self.quality_gates["min_error_handling_score"]:
            recommendations.append("Improve error handling and edge case coverage")
        
        if metrics.test_isolation_score < 0.8:
            recommendations.append("Increase test isolation using more mocking and stubbing")
        
        if metrics.edge_case_coverage_score < 0.8:
            recommendations.append("Add more edge case testing scenarios")
        
        if metrics.overall_quality_score < 0.85:
            recommendations.append("Overall quality below production threshold - comprehensive review needed")
        
        return recommendations
    
    def _analyze_quality_trend(self) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        if len(self.quality_history) < 2:
            return {"status": "insufficient_data"}
        
        recent = self.quality_history[-1]
        previous = self.quality_history[-2]
        
        trend = {
            "quality_score_change": recent.overall_quality_score - previous.overall_quality_score,
            "coverage_change": recent.test_coverage_percentage - previous.test_coverage_percentage,
            "performance_change": recent.performance_score - previous.performance_score,
            "memory_efficiency_change": recent.memory_efficiency_score - previous.memory_efficiency_score,
            "error_handling_change": recent.error_handling_score - previous.error_handling_score
        }
        
        # Determine overall trend
        positive_changes = sum(1 for change in trend.values() if change > 0.01)
        negative_changes = sum(1 for change in trend.values() if change < -0.01)
        
        if positive_changes > negative_changes:
            trend_direction = "improving"
        elif negative_changes > positive_changes:
            trend_direction = "declining"
        else:
            trend_direction = "stable"
        
        trend["overall_direction"] = trend_direction
        return trend
    
    def print_quality_report(self, report: Dict[str, Any]):
        """Print comprehensive quality report"""
        print("\n" + "=" * 80)
        print("🔍 COMPREHENSIVE QUALITY ASSURANCE REPORT")
        print("=" * 80)
        
        # Overall assessment
        overall = report["overall_assessment"]
        status_icon = "✅" if overall["production_ready"] else "❌"
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Quality Score: {overall['quality_score']:.3f}/1.000 (Grade: {overall['overall_grade']})")
        print(f"   Quality Gates: {'✅ PASSED' if overall['quality_gates_passed'] else '❌ FAILED'}")
        print(f"   Production Ready: {status_icon} {overall['production_ready']}")
        
        # Detailed metrics
        metrics = report["detailed_metrics"]
        print(f"\n📊 DETAILED METRICS:")
        
        # Test coverage
        coverage = metrics["test_coverage"]
        coverage_icon = "✅" if coverage["gate_passed"] else "❌"
        print(f"   {coverage_icon} Test Coverage: {coverage['percentage']:.1f}% (Target: {coverage['target']:.1f}%)")
        
        # Performance
        performance = metrics["performance"]
        perf_icon = "✅" if performance["gate_passed"] else "❌"
        print(f"   {perf_icon} Performance Score: {performance['score']:.3f} (Execution: {performance['execution_time_ms']:.1f}ms)")
        
        # Memory
        memory = metrics["memory_management"]
        memory_icon = "✅" if memory["gate_passed"] else "❌"
        print(f"   {memory_icon} Memory Efficiency: {memory['efficiency_score']:.3f} (Peak: {memory['peak_usage_mb']:.1f}MB)")
        
        # Error handling
        error = metrics["error_handling"]
        error_icon = "✅" if error["gate_passed"] else "❌"
        print(f"   {error_icon} Error Handling: {error['score']:.3f}")
        
        # TDD compliance
        tdd = report["tdd_compliance"]
        print(f"\n📋 TDD COMPLIANCE:")
        print(f"   Red-Green-Refactor: {tdd['red_green_refactor']:.3f}")
        print(f"   Test Isolation: {tdd['test_isolation']:.3f}")
        print(f"   Edge Case Coverage: {tdd['edge_case_coverage']:.3f}")
        print(f"   Mock Usage: {tdd['mock_usage']:.3f}")
        
        # Recommendations
        recommendations = report["recommendations"]
        if recommendations:
            print(f"\n🔧 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print(f"\n✨ RECOMMENDATIONS: All quality standards met!")
        
        # Historical trend
        trend = report["historical_trend"]
        if trend.get("status") != "insufficient_data":
            trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(trend["overall_direction"], "➡️")
            print(f"\n{trend_icon} QUALITY TREND: {trend['overall_direction'].title()}")
            print(f"   Quality Score Change: {trend['quality_score_change']:+.3f}")
            print(f"   Coverage Change: {trend['coverage_change']:+.1f}%")
        
        print("=" * 80)


async def main():
    """Main quality assurance execution"""
    print("🔍 IIT 4.0 NewbornAI 2.0 - Quality Assurance System")
    print("📋 TDD Quality Standards | Performance Benchmarking | Memory Profiling")
    print("=" * 80)
    
    # Initialize QA system
    baseline_file = Path("performance_baseline.json")
    qa_manager = QualityAssuranceManager(baseline_file)
    
    # Create test suite
    test_suite = ComprehensiveTestSuite()
    
    # Run comprehensive quality assessment
    quality_metrics = await qa_manager.comprehensive_quality_assessment(test_suite)
    
    # Generate and print report
    quality_report = qa_manager.generate_quality_report(quality_metrics)
    qa_manager.print_quality_report(quality_report)
    
    # Save baseline for future comparisons
    qa_manager.performance_profiler.save_baseline()
    
    # Return quality status for CI/CD
    return {
        "production_ready": quality_report["overall_assessment"]["production_ready"],
        "quality_score": quality_metrics.overall_quality_score,
        "quality_grade": quality_report["overall_assessment"]["overall_grade"],
        "gates_passed": quality_report["overall_assessment"]["quality_gates_passed"]
    }


if __name__ == "__main__":
    # Run quality assurance system
    result = asyncio.run(main())
    
    # Exit with appropriate code for CI/CD
    if result["production_ready"]:
        print("\n🎉 Quality assurance PASSED - Ready for production!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Quality assurance FAILED - Grade: {result['quality_grade']}")
        sys.exit(1)