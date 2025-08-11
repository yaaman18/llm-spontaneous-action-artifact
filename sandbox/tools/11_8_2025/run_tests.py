#!/usr/bin/env python3
"""
Test runner script for enactive consciousness system.

Provides structured execution of different test suites following TDD principles.
Supports running tests by phase, category, or comprehensive test execution
with detailed reporting and coverage analysis.
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional


class ConsciousnessTestRunner:
    """Main test runner for consciousness system tests."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_categories = {
            'unit': {
                'description': 'Unit tests for individual components',
                'markers': 'unit',
                'path': 'tests/unit',
                'timeout': 120
            },
            'integration': {
                'description': 'Integration tests for component interactions',
                'markers': 'integration',
                'path': 'tests/integration', 
                'timeout': 300
            },
            'acceptance': {
                'description': 'Acceptance tests for user requirements',
                'markers': 'acceptance',
                'path': 'tests/acceptance',
                'timeout': 600
            },
            'property': {
                'description': 'Property-based tests with Hypothesis',
                'markers': 'property',
                'path': 'tests/test_properties.py',
                'timeout': 300
            }
        }
        
        self.phase_tests = {
            'phase1': {
                'description': 'Phase 1: Basic consciousness and Φ computation',
                'markers': 'phase1',
                'components': ['phi_value', 'consciousness_state'],
                'timeout': 180
            },
            'phase2': {
                'description': 'Phase 2: Predictive coding and hierarchical processing',
                'markers': 'phase2',
                'components': ['predictive_coding_core', 'prediction_state', 'bayesian'],
                'timeout': 300
            },
            'phase3': {
                'description': 'Phase 3: Advanced features and metacognition',
                'markers': 'phase3',
                'components': ['metacognitive', 'som', 'gui'],
                'timeout': 400
            }
        }
        
        self.component_tests = {
            'phi_value': 'tests/unit/test_phi_value.py',
            'consciousness_state': 'tests/unit/test_consciousness_state.py',
            'prediction_state': 'tests/unit/test_prediction_state.py',
            'predictive_coding_core': 'tests/unit/test_predictive_coding_core.py',
            'self_organizing_map': 'tests/unit/test_self_organizing_map.py',
            'bayesian_inference': 'tests/unit/test_bayesian_inference_service.py'
        }
        
    def run_category_tests(self, category: str, options: Optional[Dict] = None) -> bool:
        """Run tests for a specific category."""
        if category not in self.test_categories:
            print(f"❌ Unknown test category: {category}")
            return False
        
        category_info = self.test_categories[category]
        print(f"\n🧪 Running {category_info['description']}")
        print(f"   Path: {category_info['path']}")
        
        cmd = self._build_pytest_command(
            path=category_info['path'],
            markers=category_info.get('markers'),
            timeout=category_info.get('timeout'),
            options=options or {}
        )
        
        start_time = time.time()
        success = self._execute_command(cmd)
        duration = time.time() - start_time
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {category} tests completed in {duration:.2f}s")
        
        return success
    
    def run_phase_tests(self, phase: str, options: Optional[Dict] = None) -> bool:
        """Run tests for a specific implementation phase."""
        if phase not in self.phase_tests:
            print(f"❌ Unknown phase: {phase}")
            return False
        
        phase_info = self.phase_tests[phase]
        print(f"\n🎯 Running {phase_info['description']}")
        print(f"   Components: {', '.join(phase_info['components'])}")
        
        cmd = self._build_pytest_command(
            markers=phase_info.get('markers'),
            timeout=phase_info.get('timeout'),
            options=options or {}
        )
        
        start_time = time.time()
        success = self._execute_command(cmd)
        duration = time.time() - start_time
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {phase} tests completed in {duration:.2f}s")
        
        return success
    
    def run_component_tests(self, component: str, options: Optional[Dict] = None) -> bool:
        """Run tests for a specific component."""
        if component not in self.component_tests:
            print(f"❌ Unknown component: {component}")
            print(f"Available components: {', '.join(self.component_tests.keys())}")
            return False
        
        test_file = self.component_tests[component]
        print(f"\n🔧 Running tests for {component}")
        print(f"   File: {test_file}")
        
        cmd = self._build_pytest_command(
            path=test_file,
            options=options or {}
        )
        
        start_time = time.time()
        success = self._execute_command(cmd)
        duration = time.time() - start_time
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {component} tests completed in {duration:.2f}s")
        
        return success
    
    def run_all_tests(self, options: Optional[Dict] = None) -> bool:
        """Run comprehensive test suite."""
        print("\n🚀 Running comprehensive consciousness system test suite")
        print("=" * 60)
        
        results = {}
        overall_start = time.time()
        
        # Run all categories in order
        test_order = ['unit', 'integration', 'property', 'acceptance']
        
        for category in test_order:
            print(f"\n{'='*20} {category.upper()} TESTS {'='*20}")
            success = self.run_category_tests(category, options)
            results[category] = success
            
            if not success and options and options.get('fail_fast'):
                print(f"\n❌ Test suite failed at {category} tests (fail-fast enabled)")
                break
        
        # Summary
        overall_duration = time.time() - overall_start
        print(f"\n{'='*60}")
        print("🎯 TEST SUITE SUMMARY")
        print(f"{'='*60}")
        
        passed = sum(results.values())
        total = len(results)
        
        for category, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {status} {category}")
        
        print(f"\nOverall: {passed}/{total} test categories passed")
        print(f"Duration: {overall_duration:.2f}s")
        
        overall_success = all(results.values())
        final_status = "🎉 ALL TESTS PASSED" if overall_success else "💥 SOME TESTS FAILED"
        print(f"\n{final_status}")
        
        return overall_success
    
    def run_tdd_cycle(self, component: str, options: Optional[Dict] = None) -> bool:
        """Run TDD Red-Green-Refactor cycle for a component."""
        print(f"\n🔄 Running TDD cycle for {component}")
        print("   Red -> Green -> Refactor")
        
        # Red: Run tests first (should fail for new features)
        print("\n🔴 RED: Running tests (expecting failures for new features)")
        red_success = self.run_component_tests(component, {**options, 'continue_on_failure': True})
        
        if red_success:
            print("⚠️  All tests passed in RED phase - no new failing tests for TDD cycle")
        else:
            print("✅ RED phase complete - tests failing as expected for new features")
        
        # Green: Implementation would happen here (not automated)
        print("\n🟢 GREEN: Implement minimal code to make tests pass")
        print("   (Manual implementation step - run tests again after implementation)")
        
        # Refactor: Re-run tests after refactoring
        print("\n🔵 REFACTOR: Run tests after code improvements")
        print("   (Run this command again after refactoring)")
        
        return True
    
    def run_performance_benchmark(self, options: Optional[Dict] = None) -> bool:
        """Run performance benchmarks."""
        print("\n⚡ Running performance benchmarks")
        
        cmd = self._build_pytest_command(
            markers='performance',
            options={
                **options,
                'benchmark': True,
                'benchmark_sort': 'mean',
                'benchmark_columns': 'min,max,mean,stddev,rounds,iterations'
            }
        )
        
        return self._execute_command(cmd)
    
    def _build_pytest_command(
        self, 
        path: str = None, 
        markers: str = None, 
        timeout: int = None,
        options: Dict = None
    ) -> List[str]:
        """Build pytest command with appropriate options."""
        cmd = ['python', '-m', 'pytest']
        
        if path:
            cmd.append(str(self.project_root / path))
        
        if markers:
            cmd.extend(['-m', markers])
        
        if timeout:
            cmd.extend(['--timeout', str(timeout)])
        
        # Add common options
        cmd.extend([
            '--verbose',
            '--tb=short',
            '--durations=10'
        ])
        
        # Handle specific options
        if options:
            if options.get('coverage', True):
                cmd.extend(['--cov=domain', '--cov-report=term-missing'])
            
            if options.get('fail_fast'):
                cmd.append('--maxfail=1')
            
            if options.get('parallel'):
                cmd.extend(['-n', 'auto'])
            
            if options.get('benchmark'):
                cmd.append('--benchmark-only')
                if options.get('benchmark_sort'):
                    cmd.extend(['--benchmark-sort', options['benchmark_sort']])
            
            if options.get('property_examples'):
                cmd.extend(['--hypothesis-max-examples', str(options['property_examples'])])
            
            if options.get('verbose_hypothesis'):
                cmd.append('--hypothesis-show-statistics')
            
            if options.get('continue_on_failure'):
                cmd.extend(['--maxfail=999'])  # Continue despite failures
        
        return cmd
    
    def _execute_command(self, cmd: List[str]) -> bool:
        """Execute command and return success status."""
        print(f"🔧 Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=False,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Command execution failed: {e}")
            return False
    
    def list_available_tests(self):
        """List all available test categories and components."""
        print("\n📋 Available Test Categories:")
        print("=" * 40)
        for category, info in self.test_categories.items():
            print(f"  {category:12} - {info['description']}")
        
        print("\n📋 Available Implementation Phases:")
        print("=" * 40)
        for phase, info in self.phase_tests.items():
            print(f"  {phase:12} - {info['description']}")
        
        print("\n📋 Available Components:")
        print("=" * 40)
        for component, test_file in self.component_tests.items():
            print(f"  {component:20} - {test_file}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for enactive consciousness system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --category unit          # Run unit tests only  
  python run_tests.py --phase phase1           # Run Phase 1 tests
  python run_tests.py --component phi_value    # Test specific component
  python run_tests.py --tdd phi_value          # Run TDD cycle
  python run_tests.py --list                   # List available tests
  python run_tests.py --benchmark              # Run performance benchmarks
        """
    )
    
    # Test selection options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Run all tests')
    group.add_argument('--category', choices=['unit', 'integration', 'acceptance', 'property'], 
                      help='Run specific test category')
    group.add_argument('--phase', choices=['phase1', 'phase2', 'phase3'],
                      help='Run tests for specific implementation phase')
    group.add_argument('--component', help='Run tests for specific component')
    group.add_argument('--tdd', help='Run TDD cycle for component')
    group.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    group.add_argument('--list', action='store_true', help='List available test options')
    
    # Test execution options
    parser.add_argument('--no-coverage', action='store_true', help='Disable coverage reporting')
    parser.add_argument('--fail-fast', action='store_true', help='Stop on first failure')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--property-examples', type=int, default=100,
                       help='Number of examples for property-based tests')
    parser.add_argument('--verbose-hypothesis', action='store_true',
                       help='Show detailed Hypothesis statistics')
    
    args = parser.parse_args()
    
    runner = ConsciousnessTestRunner()
    
    # Handle list command
    if args.list:
        runner.list_available_tests()
        return 0
    
    # Build options dictionary
    options = {
        'coverage': not args.no_coverage,
        'fail_fast': args.fail_fast,
        'parallel': args.parallel,
        'property_examples': args.property_examples,
        'verbose_hypothesis': args.verbose_hypothesis
    }
    
    # Execute appropriate test command
    success = False
    
    if args.all:
        success = runner.run_all_tests(options)
    elif args.category:
        success = runner.run_category_tests(args.category, options)
    elif args.phase:
        success = runner.run_phase_tests(args.phase, options)
    elif args.component:
        success = runner.run_component_tests(args.component, options)
    elif args.tdd:
        success = runner.run_tdd_cycle(args.tdd, options)
    elif args.benchmark:
        success = runner.run_performance_benchmark(options)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())