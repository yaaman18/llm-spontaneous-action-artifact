#!/usr/bin/env python3
"""
Comprehensive TDD Test Runner for Equinox Stateful Operations

This script demonstrates the complete Test-Driven Development approach
for Equinox state management in enactive consciousness, showing how
tests drive proper implementation patterns.

Usage:
    python run_equinox_tdd_tests.py [--test-category=<category>] [--show-failures]
    
Categories:
    - state_management: Core eqx.nn.State and eqx.tree_at patterns
    - rnn_integration: eqx.nn.GRUCell and temporal processing
    - circular_causality: Memory sedimentation and coupling dynamics
    - all: Run complete test suite (default)
"""

import subprocess
import sys
import argparse
from typing import List, Dict, Tuple
import time


def run_test_category(category: str, show_failures: bool = False) -> Tuple[bool, str]:
    """Run a specific test category and return success status with output."""
    
    test_files = {
        'state_management': 'tests/test_equinox_state_management_corrected.py',
        'rnn_integration': 'tests/test_rnn_integration.py', 
        'circular_causality': 'tests/test_circular_causality_state.py'
    }
    
    if category not in test_files:
        return False, f"Unknown test category: {category}"
    
    # Build pytest command
    cmd = [
        'python', '-m', 'pytest',
        test_files[category],
        '-v',
        '--no-cov',
        '--tb=short' if not show_failures else '--tb=long'
    ]
    
    print(f"\n{'='*60}")
    print(f"Running {category.replace('_', ' ').title()} Tests")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        status_msg = f"✅ PASSED" if success else f"❌ FAILED"
        print(f"\n{status_msg} - {category} tests completed in {duration:.2f}s")
        
        return success, result.stdout
        
    except subprocess.TimeoutExpired:
        return False, f"Tests timed out after 5 minutes"
    except Exception as e:
        return False, f"Error running tests: {str(e)}"


def demonstrate_tdd_methodology():
    """Demonstrate TDD Red-Green-Refactor methodology with examples."""
    
    print("\n" + "="*80)
    print("TDD METHODOLOGY DEMONSTRATION")
    print("="*80)
    
    print("""
🔴 RED Phase: Write failing tests that specify desired behavior
   - Tests drive implementation requirements
   - Clear assertions define success criteria
   - Initial failures guide development direction
   
🟢 GREEN Phase: Implement minimal code to make tests pass  
   - Focus on correctness over optimization
   - Satisfy test requirements exactly
   - Build confidence through passing tests
   
🔵 REFACTOR Phase: Improve code structure while maintaining tests
   - Enhance performance and readability
   - Add sophisticated features on solid foundation
   - Tests prevent regressions during improvements

Key TDD Principles Applied:
✓ Test-First Development: Specifications before implementation
✓ A-A-A Pattern: Arrange-Act-Assert structure
✓ Given-When-Then: Clear behavioral specifications  
✓ Edge Case Coverage: Comprehensive scenario testing
✓ Living Documentation: Tests as executable specifications
    """)


def show_implementation_examples():
    """Show key implementation patterns driven by tests."""
    
    print("\n" + "="*80) 
    print("IMPLEMENTATION PATTERNS DRIVEN BY TESTS")
    print("="*80)
    
    examples = {
        "Equinox Module Structure": '''
class EnactiveTemporalProcessor(eqx.Module):
    gru_cell: eqx.nn.GRUCell
    memory_projector: eqx.nn.Linear  
    coupling_network: eqx.nn.Linear
    hidden_dim: int
    buffer_depth: int
        ''',
        
        "Immutable State Updates": '''
# Using eqx.tree_at for immutable PyTree updates
updated_state = eqx.tree_at(
    lambda s: (s.retention_buffer, s.iteration_count),
    current_state,
    (new_buffer, current_state.iteration_count + 1)
)
        ''',
        
        "Stateful RNN Processing": '''
# Proper GRU cell usage with vmap
def step_fn(hidden_state, x):
    new_hidden = jax.vmap(gru_cell)(x, hidden_state)
    output = jax.vmap(projector)(new_hidden)
    return new_hidden, output

final_hidden, outputs = jax.lax.scan(step_fn, initial_hidden, sequence)
        ''',
        
        "Circular Buffer Management": '''
# Immutable circular buffer with proper shifting
shifted_buffer = jnp.roll(buffer, -1, axis=0)
updated_buffer = shifted_buffer.at[-1].set(new_experience)
        '''
    }
    
    for title, code in examples.items():
        print(f"\n{title}:")
        print("-" * len(title))
        print(code)


def run_comprehensive_validation():
    """Run comprehensive validation demonstrating test coverage."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST VALIDATION")  
    print("="*80)
    
    # Define validation tests
    validation_tests = [
        ("State Management", "tests/test_equinox_state_management_corrected.py::TestEquinoxStatefulLayers::test_gru_cell_stateful_operation_red"),
        ("Tree Operations", "tests/test_equinox_state_management_corrected.py::TestEquinoxTreeAtOperations::test_tree_at_memory_buffer_update_red"),
        ("RNN Integration", "tests/test_rnn_integration.py::TestGRUCellIntegration::test_gru_cell_initialization_follows_equinox_pattern"),
        ("Circular Causality", "tests/test_circular_causality_state.py::TestHistoryBufferImmutableUpdates::test_history_buffer_circular_update_immutably")
    ]
    
    results = []
    
    for test_name, test_path in validation_tests:
        print(f"\n🧪 Validating {test_name}...")
        
        cmd = ['python', '-m', 'pytest', test_path, '-v', '--no-cov', '--tb=short']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            success = result.returncode == 0
            status = "✅ PASS" if success else "❌ FAIL"
            results.append((test_name, success))
            print(f"   {status}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nValidation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core TDD patterns validated successfully!")
        print("   The test suite properly drives Equinox implementation.")
    else:
        print("⚠️  Some tests failed - implementation needs attention.")
        
    return passed == total


def main():
    """Main entry point for TDD test runner."""
    
    parser = argparse.ArgumentParser(
        description="TDD Test Runner for Equinox Stateful Operations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--test-category',
        choices=['state_management', 'rnn_integration', 'circular_causality', 'all'],
        default='all',
        help='Test category to run (default: all)'
    )
    
    parser.add_argument(
        '--show-failures',
        action='store_true',
        help='Show detailed failure information'
    )
    
    parser.add_argument(
        '--demo-only',
        action='store_true', 
        help='Show TDD methodology demonstration without running tests'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run comprehensive validation of key TDD patterns'
    )
    
    args = parser.parse_args()
    
    print("🧪 TDD Test Suite for Equinox Stateful Operations")
    print("   Following Takuto Wada's Test-Driven Development Methodology")
    
    # Show methodology demonstration
    demonstrate_tdd_methodology()
    
    if args.demo_only:
        show_implementation_examples()
        return
        
    if args.validate:
        success = run_comprehensive_validation()
        sys.exit(0 if success else 1)
    
    # Run requested test categories
    categories = ['state_management', 'rnn_integration', 'circular_causality'] if args.test_category == 'all' else [args.test_category]
    
    results = {}
    overall_success = True
    
    for category in categories:
        success, output = run_test_category(category, args.show_failures)
        results[category] = success
        overall_success &= success
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    for category, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {category.replace('_', ' ').title()}")
    
    if overall_success:
        print("\n🎉 All tests passed! Equinox state management patterns validated.")
        print("   The TDD approach successfully drives proper implementation.")
    else:
        print("\n⚠️  Some tests failed. Review implementation against test requirements.")
        
    show_implementation_examples()
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()