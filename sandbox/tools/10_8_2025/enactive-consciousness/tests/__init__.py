#!/usr/bin/env python3
"""Comprehensive TDD test suite for Equinox stateful operations in enactive consciousness.

This test package implements rigorous Test-Driven Development methodology
for ensuring proper Equinox state management patterns in the enactive 
consciousness system.

Test Modules:
- test_equinox_state_management: Core eqx.nn.State and eqx.tree_at patterns
- test_rnn_integration: eqx.nn.GRUCell and temporal sequence processing
- test_circular_causality_state: Circular causality and memory sedimentation

Usage:
    pytest tests/ -v                    # Run all tests
    pytest tests/test_equinox_state_management.py -v  # Run specific module
    python -m pytest tests/ --tb=short # Run with concise tracebacks
"""

__version__ = "1.0.0"
__author__ = "TDD Specialist (following Takuto Wada's methodology)"

# Test configuration constants
DEFAULT_TEST_KEY = 42
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQUENCE_LENGTH = 15
DEFAULT_HIDDEN_DIM = 64
DEFAULT_COUPLING_DIM = 32

# Import key test utilities for easy access
from .test_equinox_state_management import (
    TestEquinoxStateModule,
    StatefulMemoryTrace,
)

from .test_rnn_integration import (
    TestConfig,
    TestResultsValidator,
    SimpleGRUProcessor,
)

from .test_circular_causality_state import (
    CircularCausalityProcessor,
    CircularCausalityState,
    CausalityConfig,
)