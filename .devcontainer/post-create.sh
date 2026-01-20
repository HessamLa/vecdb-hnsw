#!/bin/bash
# Post-create script for VecDB development container
# This script runs once after the container is created

set -e

echo "========================================"
echo "VecDB Development Environment Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Navigate to workspace
cd /workspaces/vecdb

# Install Python package in development mode
echo ""
echo "Installing Python package in development mode..."
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    pip install -e ".[dev]" --quiet 2>/dev/null || pip install -e . --quiet 2>/dev/null || print_warning "Python package install skipped (setup.py/pyproject.toml may be incomplete)"
    print_status "Python package installed (or skipped if not ready)"
else
    print_warning "No setup.py or pyproject.toml found - skipping Python install"
fi

# Install additional dev requirements if present
if [ -f "requirements-dev.txt" ]; then
    echo ""
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt --quiet
    print_status "Development dependencies installed"
fi

# Create build directory for CMake
echo ""
echo "Creating build directory..."
mkdir -p build
print_status "Build directory created at /workspaces/vecdb/build"

# Verify environment
echo ""
echo "Verifying environment..."
echo ""

# Check Python
python_version=$(python --version 2>&1)
print_status "Python: $python_version"

# Check pip packages
if python -c "import numpy" 2>/dev/null; then
    numpy_version=$(python -c "import numpy; print(numpy.__version__)")
    print_status "NumPy: $numpy_version"
else
    print_error "NumPy not installed"
fi

if python -c "import pybind11" 2>/dev/null; then
    pybind11_version=$(python -c "import pybind11; print(pybind11.__version__)")
    print_status "pybind11: $pybind11_version"
else
    print_error "pybind11 not installed"
fi

if python -c "import pytest" 2>/dev/null; then
    pytest_version=$(python -c "import pytest; print(pytest.__version__)")
    print_status "pytest: $pytest_version"
else
    print_error "pytest not installed"
fi

# Check C++ toolchain
echo ""
gcc_version=$(g++ --version | head -n 1)
print_status "g++: $gcc_version"

cmake_version=$(cmake --version | head -n 1)
print_status "CMake: $cmake_version"

# Check pybind11 CMake discovery
echo ""
pybind11_cmake=$(python -m pybind11 --cmakedir 2>/dev/null || echo "not found")
if [ "$pybind11_cmake" != "not found" ]; then
    print_status "pybind11 CMake dir: $pybind11_cmake"
else
    print_warning "pybind11 CMake directory not found"
fi

# Summary
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Quick start commands:"
echo "  - Run tests:        pytest tests/"
echo "  - Build C++ module: cd build && cmake .. && make"
echo "  - Verify setup:     bash scripts/verify-environment.sh"
echo ""
