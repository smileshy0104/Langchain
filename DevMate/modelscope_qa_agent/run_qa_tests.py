#!/usr/bin/env python
"""
Test runner for QA Agent tests with dependency mocking

This script mocks missing dependencies (langchain_milvus) before running tests.
"""

import sys
from unittest.mock import MagicMock

# Mock missing dependencies before any imports
sys.modules['langchain_milvus'] = MagicMock()

import pytest

if __name__ == "__main__":
    # Run TestSingleTurnQA tests
    exit_code = pytest.main([
        "tests/test_qa_agent.py::TestSingleTurnQA",
        "-v",
        "--tb=short"
    ])

    sys.exit(exit_code)
