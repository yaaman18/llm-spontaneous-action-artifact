#!/usr/bin/env python
"""
Comprehensive Test Runner for Enactive Consciousness Modules

This script runs the complete test suite following TDD principles and
generates detailed coverage reports for all target modules.

Usage:
    python run_comprehensive_tests.py [--module MODULE] [--coverage] [--html] [--parallel]

Modules tested:
- Information Theory (target: 85% coverage)
- Dynamic Networks (target: 85% coverage)
- Sparse Representations (target: 85% coverage)
- Predictive Coding (target: 85% coverage)
- Continuous Dynamics (target: 85% coverage)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class TestRunner:
    """Comprehensive test runner for enactive consciousness modules."""
    
    def __init__(self, project_root: str = None):
        """Initialize test runner.
        
        Args:
            project_root: Root directory of the project
        """
        if project_root is None:
            self.project_root = Path(__file__).parent
        else:
            self.project_root = Path(project_root)
        
        self.test_modules = {
            'information_theory': {
                'test_file': 'tests/test_information_theory_comprehensive.py',
                'source_module': 'src/enactive_consciousness/information_theory.py',
                'target_coverage': 85,
                'current_coverage': 17,
                'description': 'Information theory measures for consciousness analysis'
            },
            'dynamic_networks': {
                'test_file': 'tests/test_dynamic_networks_comprehensive.py',
                'source_module': 'src/enactive_consciousness/dynamic_networks.py',
                'target_coverage': 85,
                'current_coverage': 16,
                'description': 'Dynamic network processing and adaptation'
            },
            'sparse_representations': {
                'test_file': 'tests/test_sparse_representations_comprehensive.py',
                'source_module': 'src/enactive_consciousness/sparse_representations.py',
                'target_coverage': 85,
                'current_coverage': 0,
                'description': 'Sparse coding and representation learning'
            },
            'predictive_coding': {
                'test_file': 'tests/test_predictive_coding_comprehensive.py',
                'source_module': 'src/enactive_consciousness/predictive_coding.py',
                'target_coverage': 85,
                'current_coverage': 29,
                'description': 'Hierarchical predictive coding systems'
            },
            'continuous_dynamics': {
                'test_file': 'tests/test_continuous_dynamics_comprehensive.py',
                'source_module': 'src/enactive_consciousness/continuous_dynamics.py',
                'target_coverage': 85,
                'current_coverage': 0,
                'description': 'Continuous-time dynamics and differential equations'
            }
        }
    
    def run_single_module(self, module_name: str, verbose: bool = True, 
                         coverage: bool = True, html_report: bool = False) -> Dict:
        """Run tests for a single module.
        
        Args:
            module_name: Name of the module to test
            verbose: Enable verbose output
            coverage: Enable coverage reporting
            html_report: Generate HTML coverage report
            
        Returns:
            Dictionary with test results
        """
        if module_name not in self.test_modules:
            raise ValueError(f"Unknown module: {module_name}. Available: {list(self.test_modules.keys())}")
        
        module_info = self.test_modules[module_name]
        test_file = self.project_root / module_info['test_file']
        source_module = module_info['source_module']
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        print(f"\n{'='*80}")
        print(f"Running tests for {module_name.replace('_', ' ').title()}")
        print(f"Description: {module_info['description']}")
        print(f"Target Coverage: {module_info['target_coverage']}%")
        print(f"Current Coverage: {module_info['current_coverage']}%")
        print(f"{'='*80}")
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest', str(test_file)]
        
        if verbose:
            cmd.append('-v')
        
        if coverage:
            cmd.extend([
                f'--cov={source_module.replace("src/", "").replace("/", ".").replace(".py", "")}',
                '--cov-report=term-missing',
                '--cov-fail-under=0',  # Don't fail on coverage for individual module runs
            ])
            
            if html_report:
                cmd.extend([
                    f'--cov-report=html:htmlcov_{module_name}',
                    '--cov-report=xml:coverage.xml'
                ])
        
        # Add test markers and options
        cmd.extend([
            '--tb=short',
            '--strict-markers',
            '--disable-warnings'
        ])
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {self.project_root}")
        
        # Run tests
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            # Parse results
            output = result.stdout
            error = result.stderr
            
            # Extract test statistics
            test_stats = self._parse_test_output(output)
            coverage_stats = self._parse_coverage_output(output)
            
            results = {
                'module': module_name,
                'success': result.returncode == 0,
                'output': output,
                'error': error,
                'return_code': result.returncode,
                'test_stats': test_stats,
                'coverage_stats': coverage_stats,
                'target_coverage': module_info['target_coverage'],
                'coverage_achieved': coverage_stats.get('coverage_percent', 0) >= module_info['target_coverage']
            }
            
            # Print summary
            self._print_module_summary(results)
            
            return results
            
        except subprocess.TimeoutExpired:
            print(f"❌ Tests timed out after 10 minutes")
            return {
                'module': module_name,
                'success': False,
                'error': 'Test execution timed out',
                'test_stats': {},
                'coverage_stats': {},
                'coverage_achieved': False
            }
        
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return {
                'module': module_name,
                'success': False,
                'error': str(e),
                'test_stats': {},
                'coverage_stats': {},
                'coverage_achieved': False
            }
    
    def run_all_modules(self, verbose: bool = True, coverage: bool = True, 
                       html_report: bool = False, parallel: bool = False) -> Dict:
        """Run tests for all modules.
        
        Args:
            verbose: Enable verbose output
            coverage: Enable coverage reporting
            html_report: Generate HTML coverage reports
            parallel: Run tests in parallel (not implemented yet)
            
        Returns:
            Dictionary with all test results
        """
        print(f"\n🚀 Starting Comprehensive Test Suite")
        print(f"Target: Improve coverage from current 7% to 85%+")
        print(f"Modules to test: {len(self.test_modules)}")
        
        all_results = {}
        successful_modules = 0
        coverage_achieved_modules = 0
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for module_name in self.test_modules.keys():
            try:
                result = self.run_single_module(
                    module_name, verbose=verbose, coverage=coverage, html_report=html_report
                )
                all_results[module_name] = result
                
                if result['success']:
                    successful_modules += 1
                
                if result.get('coverage_achieved', False):
                    coverage_achieved_modules += 1
                
                # Accumulate test statistics
                test_stats = result.get('test_stats', {})
                total_tests += test_stats.get('total', 0)
                total_passed += test_stats.get('passed', 0)
                total_failed += test_stats.get('failed', 0)
                
            except Exception as e:
                print(f"❌ Failed to run tests for {module_name}: {e}")
                all_results[module_name] = {
                    'success': False,
                    'error': str(e),
                    'coverage_achieved': False
                }
        
        # Print overall summary
        self._print_overall_summary(
            all_results, successful_modules, coverage_achieved_modules,
            total_tests, total_passed, total_failed
        )
        
        return all_results
    
    def _parse_test_output(self, output: str) -> Dict:
        """Parse pytest output for test statistics."""
        stats = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0, 'warnings': 0}
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for summary lines like "5 passed, 2 failed, 1 skipped in 3.45s"
            if 'passed' in line and ('failed' in line or 'in ' in line):
                # Extract numbers
                import re
                passed = re.search(r'(\d+) passed', line)
                failed = re.search(r'(\d+) failed', line)
                skipped = re.search(r'(\d+) skipped', line)
                warnings = re.search(r'(\d+) warning', line)
                
                if passed:
                    stats['passed'] = int(passed.group(1))
                if failed:
                    stats['failed'] = int(failed.group(1))
                if skipped:
                    stats['skipped'] = int(skipped.group(1))
                if warnings:
                    stats['warnings'] = int(warnings.group(1))
                
                stats['total'] = stats['passed'] + stats['failed'] + stats['skipped']
                break
        
        return stats
    
    def _parse_coverage_output(self, output: str) -> Dict:
        """Parse coverage output for coverage statistics."""
        coverage_stats = {}
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for coverage percentage lines
            if 'Total coverage:' in line:
                import re
                coverage_match = re.search(r'(\d+(?:\.\d+)?)%', line)
                if coverage_match:
                    coverage_stats['coverage_percent'] = float(coverage_match.group(1))
            
            # Look for TOTAL coverage line in coverage report
            if line.startswith('TOTAL') and '%' in line:
                parts = line.split()
                for part in parts:
                    if '%' in part and part.replace('%', '').replace('.', '').isdigit():
                        coverage_stats['coverage_percent'] = float(part.replace('%', ''))
                        break
        
        return coverage_stats
    
    def _print_module_summary(self, results: Dict):
        """Print summary for a single module."""
        module = results['module']
        success = results['success']
        test_stats = results.get('test_stats', {})
        coverage_stats = results.get('coverage_stats', {})
        
        status = "✅ PASSED" if success else "❌ FAILED"
        
        print(f"\n{'-'*60}")
        print(f"📊 {module.replace('_', ' ').title()} Results: {status}")
        
        if test_stats:
            print(f"Tests: {test_stats.get('total', 0)} total, "
                  f"{test_stats.get('passed', 0)} passed, "
                  f"{test_stats.get('failed', 0)} failed")
        
        coverage_percent = coverage_stats.get('coverage_percent', 0)
        target_coverage = results.get('target_coverage', 85)
        coverage_status = "✅" if coverage_percent >= target_coverage else "⚠️"
        
        print(f"Coverage: {coverage_percent:.1f}% {coverage_status} (target: {target_coverage}%)")
        
        if not success and 'error' in results:
            print(f"Error: {results['error']}")
        
        print(f"{'-'*60}")
    
    def _print_overall_summary(self, all_results: Dict, successful_modules: int,
                              coverage_achieved_modules: int, total_tests: int,
                              total_passed: int, total_failed: int):
        """Print overall summary for all modules."""
        total_modules = len(self.test_modules)
        
        print(f"\n{'='*80}")
        print(f"📈 COMPREHENSIVE TEST SUITE RESULTS")
        print(f"{'='*80}")
        
        print(f"Module Success Rate: {successful_modules}/{total_modules} "
              f"({successful_modules/total_modules*100:.1f}%)")
        
        print(f"Coverage Target Achievement: {coverage_achieved_modules}/{total_modules} "
              f"({coverage_achieved_modules/total_modules*100:.1f}%)")
        
        print(f"Total Tests: {total_tests}")
        print(f"Tests Passed: {total_passed} ({total_passed/max(total_tests,1)*100:.1f}%)")
        print(f"Tests Failed: {total_failed} ({total_failed/max(total_tests,1)*100:.1f}%)")
        
        # Detailed module breakdown
        print(f"\n📋 Module Breakdown:")
        for module_name, results in all_results.items():
            status = "✅" if results['success'] else "❌"
            coverage_percent = results.get('coverage_stats', {}).get('coverage_percent', 0)
            coverage_target = results.get('target_coverage', 85)
            coverage_status = "🎯" if coverage_percent >= coverage_target else "⚠️"
            
            print(f"  {status} {module_name.replace('_', ' ').title():25} "
                  f"Coverage: {coverage_percent:5.1f}% {coverage_status}")
        
        # Success criteria
        print(f"\n🎯 Success Criteria:")
        if successful_modules == total_modules:
            print(f"✅ All {total_modules} modules have passing tests")
        else:
            print(f"⚠️  {total_modules - successful_modules} modules have failing tests")
        
        if coverage_achieved_modules == total_modules:
            print(f"✅ All {total_modules} modules achieved 85%+ coverage target")
        else:
            print(f"⚠️  {total_modules - coverage_achieved_modules} modules below 85% coverage")
        
        overall_success = (successful_modules == total_modules and 
                          coverage_achieved_modules == total_modules)
        
        if overall_success:
            print(f"\n🎉 SUCCESS: All modules meet quality criteria!")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: Some modules need attention")
        
        print(f"{'='*80}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for enactive consciousness modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tests with coverage
    python run_comprehensive_tests.py --coverage --html
    
    # Run specific module
    python run_comprehensive_tests.py --module information_theory
    
    # Quick test run without coverage
    python run_comprehensive_tests.py --no-coverage
        """
    )
    
    parser.add_argument(
        '--module', '-m',
        choices=['information_theory', 'dynamic_networks', 'sparse_representations', 
                'predictive_coding', 'continuous_dynamics'],
        help='Run tests for specific module only'
    )
    
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        default=True,
        help='Enable coverage reporting (default: True)'
    )
    
    parser.add_argument(
        '--no-coverage',
        action='store_true',
        help='Disable coverage reporting'
    )
    
    parser.add_argument(
        '--html',
        action='store_true',
        help='Generate HTML coverage reports'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Run tests in parallel (not implemented yet)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Handle coverage flags
    coverage = args.coverage and not args.no_coverage
    
    # Initialize test runner
    runner = TestRunner()
    
    try:
        if args.module:
            # Run single module
            result = runner.run_single_module(
                args.module,
                verbose=args.verbose,
                coverage=coverage,
                html_report=args.html
            )
            
            sys.exit(0 if result['success'] else 1)
        
        else:
            # Run all modules
            results = runner.run_all_modules(
                verbose=args.verbose,
                coverage=coverage,
                html_report=args.html,
                parallel=args.parallel
            )
            
            # Exit with success only if all modules pass
            all_success = all(result['success'] for result in results.values())
            sys.exit(0 if all_success else 1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Test run interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()