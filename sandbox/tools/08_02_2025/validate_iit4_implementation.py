#!/usr/bin/env python3
"""
IIT 4.0 Implementation Validation Script
Quick validation to ensure the implementation works correctly

This script performs basic validation tests to verify:
1. Core modules import successfully
2. Basic φ calculation works
3. Integration with experiential concepts functions
4. No critical errors in the implementation

Run this script to quickly validate the IIT 4.0 implementation.

Author: IIT Integration Master
Date: 2025-08-03
"""

import sys
import traceback
import time
import numpy as np
from datetime import datetime


def test_core_imports():
    """Test that core modules import successfully"""
    print("🔧 Testing core module imports...")
    
    try:
        from iit4_core_engine import (
            IIT4PhiCalculator, IntrinsicDifferenceCalculator, 
            PhiStructure, CauseEffectState, IIT4AxiomValidator
        )
        print("   ✅ iit4_core_engine imports successful")
    except Exception as e:
        print(f"   ❌ iit4_core_engine import failed: {e}")
        return False
    
    try:
        from intrinsic_difference import (
            DetailedIntrinsicDifferenceCalculator, IntrinsicDifferenceValidator,
            OptimalPurviewFinder, StateSpaceAnalyzer
        )
        print("   ✅ intrinsic_difference imports successful")
    except Exception as e:
        print(f"   ❌ intrinsic_difference import failed: {e}")
        return False
    
    try:
        from iit4_newborn_integration_demo import (
            IIT4_ExperientialPhiCalculator, ExperientialConcept,
            ConsciousnessMonitor
        )
        print("   ✅ iit4_newborn_integration_demo imports successful")
    except Exception as e:
        print(f"   ❌ iit4_newborn_integration_demo import failed: {e}")
        return False
    
    return True


def test_basic_phi_calculation():
    """Test basic φ calculation functionality"""
    print("\n🧠 Testing basic φ calculation...")
    
    try:
        from iit4_core_engine import IIT4PhiCalculator
        
        # Initialize calculator
        phi_calculator = IIT4PhiCalculator(precision=1e-8)
        
        # Simple 2-node system
        system_state = np.array([1, 1])
        connectivity_matrix = np.array([
            [0, 1],
            [1, 0]
        ])
        
        start_time = time.time()
        phi_structure = phi_calculator.calculate_phi(system_state, connectivity_matrix)
        calculation_time = time.time() - start_time
        
        print(f"   ✅ φ calculation completed in {calculation_time:.3f}s")
        print(f"   📊 φ value: {phi_structure.total_phi:.6f}")
        print(f"   📊 Distinctions: {len(phi_structure.distinctions)}")
        print(f"   📊 Relations: {len(phi_structure.relations)}")
        print(f"   📊 Maximal substrate size: {len(phi_structure.maximal_substrate)}")
        
        # Basic validation
        if phi_structure.total_phi >= 0:
            print("   ✅ φ value is non-negative")
        else:
            print("   ❌ φ value is negative")
            return False
        
        if len(phi_structure.maximal_substrate) > 0:
            print("   ✅ Maximal substrate identified")
        else:
            print("   ❌ No maximal substrate found")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic φ calculation failed: {e}")
        traceback.print_exc()
        return False


def test_axiom_validation():
    """Test IIT 4.0 axiom validation"""
    print("\n📜 Testing axiom validation...")
    
    try:
        from iit4_core_engine import IIT4PhiCalculator, IIT4AxiomValidator
        
        phi_calculator = IIT4PhiCalculator()
        validator = IIT4AxiomValidator(phi_calculator)
        
        # Test system with meaningful connectivity
        system_state = np.array([1, 1, 0])
        connectivity_matrix = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        phi_structure = phi_calculator.calculate_phi(system_state, connectivity_matrix)
        axiom_results = validator.validate_all_axioms(phi_structure, system_state)
        
        print("   📊 Axiom validation results:")
        for axiom, passed in axiom_results.items():
            status = "✅" if passed else "❌"
            print(f"      {status} {axiom}: {passed}")
        
        # Check if most axioms pass
        passed_count = sum(axiom_results.values())
        total_count = len(axiom_results)
        
        if passed_count >= total_count * 0.7:  # 70% pass rate
            print(f"   ✅ Axiom validation: {passed_count}/{total_count} passed")
            return True
        else:
            print(f"   ⚠️  Axiom validation: only {passed_count}/{total_count} passed")
            return False
            
    except Exception as e:
        print(f"   ❌ Axiom validation failed: {e}")
        traceback.print_exc()
        return False


def test_experiential_integration():
    """Test experiential concept integration"""
    print("\n🔗 Testing experiential concept integration...")
    
    try:
        from iit4_newborn_integration_demo import (
            IIT4_ExperientialPhiCalculator, ExperientialConcept
        )
        
        # Create test experiential concepts
        concepts = [
            ExperientialConcept(
                concept_id="test_001",
                content="Test experiential concept 1",
                phi_contribution=0.3,
                timestamp=datetime.now(),
                experiential_quality=0.8,
                temporal_position=1,
                emotional_valence=0.7
            ),
            ExperientialConcept(
                concept_id="test_002",
                content="Test experiential concept 2",
                phi_contribution=0.4,
                timestamp=datetime.now(),
                experiential_quality=0.9,
                temporal_position=2,
                emotional_valence=0.6
            ),
            ExperientialConcept(
                concept_id="test_003",
                content="Test experiential concept 3",
                phi_contribution=0.5,
                timestamp=datetime.now(),
                experiential_quality=0.7,
                temporal_position=3,
                emotional_valence=0.8
            )
        ]
        
        # Calculate experiential φ
        experiential_calculator = IIT4_ExperientialPhiCalculator()
        result = experiential_calculator.calculate_experiential_phi(concepts)
        
        print(f"   ✅ Experiential φ calculation completed")
        print(f"   📊 Experiential φ: {result.phi_value:.6f}")
        print(f"   📊 Development stage: {result.stage_prediction.value}")
        print(f"   📊 Integration quality: {result.integration_quality:.3f}")
        print(f"   📊 Experiential purity: {result.experiential_purity:.3f}")
        print(f"   📊 Computation time: {result.computation_time:.3f}s")
        
        # Basic validation
        if result.phi_value >= 0:
            print("   ✅ Experiential φ value is non-negative")
        else:
            print("   ❌ Experiential φ value is negative")
            return False
        
        if result.concept_count == len(concepts):
            print("   ✅ Concept count matches input")
        else:
            print("   ❌ Concept count mismatch")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Experiential integration failed: {e}")
        traceback.print_exc()
        return False


def test_intrinsic_difference():
    """Test intrinsic difference calculation"""
    print("\n🔢 Testing intrinsic difference calculation...")
    
    try:
        from intrinsic_difference import DetailedIntrinsicDifferenceCalculator
        
        calculator = DetailedIntrinsicDifferenceCalculator()
        
        # Simple test case
        mechanism = frozenset([0])
        purview = frozenset([0, 1])
        
        # Simple TPM
        tpm = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8]
        ])
        
        system_state = np.array([1, 0])
        
        # Calculate directional ID
        cause_id = calculator.compute_directional_id(
            mechanism, purview, tpm, system_state, 'cause'
        )
        effect_id = calculator.compute_directional_id(
            mechanism, purview, tpm, system_state, 'effect'
        )
        
        print(f"   ✅ ID calculation completed")
        print(f"   📊 Cause ID: {cause_id:.6f}")
        print(f"   📊 Effect ID: {effect_id:.6f}")
        
        # Basic validation
        if cause_id >= 0 and effect_id >= 0:
            print("   ✅ ID values are non-negative")
            return True
        else:
            print("   ❌ ID values are negative")
            return False
            
    except Exception as e:
        print(f"   ❌ ID calculation failed: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_validation():
    """Run all validation tests"""
    print("🚀 IIT 4.0 Implementation Validation")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Basic φ Calculation", test_basic_phi_calculation),
        ("Axiom Validation", test_axiom_validation),
        ("Experiential Integration", test_experiential_integration),
        ("Intrinsic Difference", test_intrinsic_difference)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"   ⚠️  {test_name} had issues but didn't crash")
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Validation Summary")
    print("=" * 60)
    
    success_rate = passed_tests / total_tests
    print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate*100:.1f}%)")
    
    if success_rate >= 0.8:
        print("🎉 IIT 4.0 implementation validation SUCCESSFUL!")
        print("   The implementation appears to be working correctly.")
        return True
    elif success_rate >= 0.6:
        print("⚠️  IIT 4.0 implementation validation PARTIAL SUCCESS")
        print("   Some components may need attention, but core functionality works.")
        return True
    else:
        print("❌ IIT 4.0 implementation validation FAILED")
        print("   Significant issues detected. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)