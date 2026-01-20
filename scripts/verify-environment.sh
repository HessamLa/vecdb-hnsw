#!/bin/bash
# VecDB Environment Verification Script
# Run this script to verify your development environment is correctly set up

set -e

echo "========================================"
echo "VecDB Environment Verification"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

check_pass() {
    echo -e "${GREEN}[✓]${NC} $1"
}

check_fail() {
    echo -e "${RED}[✗]${NC} $1"
    ERRORS=$((ERRORS + 1))
}

check_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# ============================================
# Python Checks
# ============================================
echo "Python Environment"
echo "-------------------"

# Python version
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    PYTHON_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
    PYTHON_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
        check_pass "Python: $PYTHON_VERSION"
    else
        check_fail "Python: $PYTHON_VERSION (requires >= 3.9)"
    fi
else
    check_fail "Python: not found"
fi

# NumPy
if python -c "import numpy" 2>/dev/null; then
    NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)")
    check_pass "NumPy: $NUMPY_VERSION"
else
    check_fail "NumPy: not installed"
fi

# pybind11
if python -c "import pybind11" 2>/dev/null; then
    PYBIND_VERSION=$(python -c "import pybind11; print(pybind11.__version__)")
    check_pass "pybind11: $PYBIND_VERSION"
else
    check_fail "pybind11: not installed"
fi

# pytest
if python -c "import pytest" 2>/dev/null; then
    PYTEST_VERSION=$(python -c "import pytest; print(pytest.__version__)")
    check_pass "pytest: $PYTEST_VERSION"
else
    check_fail "pytest: not installed"
fi

# ============================================
# C++ Toolchain Checks
# ============================================
echo ""
echo "C++ Toolchain"
echo "-------------------"

# g++
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n 1)
    check_pass "g++: $GCC_VERSION"
else
    check_fail "g++: not found"
fi

# CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n 1)
    check_pass "CMake: $CMAKE_VERSION"
else
    check_fail "CMake: not found"
fi

# pybind11 CMake directory
PYBIND11_CMAKE=$(python -m pybind11 --cmakedir 2>/dev/null || echo "")
if [ -n "$PYBIND11_CMAKE" ] && [ -d "$PYBIND11_CMAKE" ]; then
    check_pass "pybind11 CMake dir: $PYBIND11_CMAKE"
else
    check_warn "pybind11 CMake dir: not found (may need: pip install pybind11[global])"
fi

# ============================================
# Project Structure Checks
# ============================================
echo ""
echo "Project Structure"
echo "-------------------"

# Check we're in the right directory
if [ -f "Dockerfile" ] && [ -d "src" ]; then
    check_pass "Project root: $(pwd)"
else
    check_warn "Project root: may not be in vecdb directory"
fi

# Check key directories
for dir in "src/python/vecdb" "src/cpp" "tests/python" "tests/cpp" "docs"; do
    if [ -d "$dir" ]; then
        check_pass "Directory exists: $dir"
    else
        check_warn "Directory missing: $dir"
    fi
done

# ============================================
# Environment Variables
# ============================================
echo ""
echo "Environment Variables"
echo "-------------------"

if [ -n "$PYTHONPATH" ]; then
    check_pass "PYTHONPATH: $PYTHONPATH"
else
    check_warn "PYTHONPATH: not set (should include src/python)"
fi

# ============================================
# Optional: Check if HNSW module is built
# ============================================
echo ""
echo "HNSW Module (optional)"
echo "-------------------"

if python -c "from vecdb._hnsw import HNSWIndex" 2>/dev/null; then
    check_pass "HNSW C++ module: available"
elif python -c "from vecdb._hnsw_stub import HNSWIndex" 2>/dev/null; then
    check_warn "HNSW C++ module: using stub (C++ not yet built)"
else
    check_warn "HNSW module: not available (expected during initial setup)"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "========================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All critical checks passed!${NC}"
    echo "========================================"
    exit 0
else
    echo -e "${RED}$ERRORS critical check(s) failed${NC}"
    echo "========================================"
    echo ""
    echo "Please fix the issues above before proceeding."
    exit 1
fi
