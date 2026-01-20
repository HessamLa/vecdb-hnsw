"""
Basic tests to verify the testing infrastructure works.
These tests should pass immediately after environment setup.
"""

import pytest


class TestSetup:
    """Tests to verify the development environment is working."""

    def test_python_version(self):
        """Verify Python version is 3.9+."""
        import sys
        assert sys.version_info >= (3, 9), "Python 3.9+ required"

    def test_numpy_import(self):
        """Verify NumPy is installed and importable."""
        import numpy as np
        assert np.__version__ >= "1.20", "NumPy 1.20+ required"

    def test_pybind11_import(self):
        """Verify pybind11 is installed."""
        import pybind11
        assert pybind11.__version__ is not None

    def test_vecdb_package_import(self):
        """Verify vecdb package is importable."""
        import vecdb
        assert vecdb.__version__ == "0.1.0"

    def test_fixtures_work(self, sample_vectors, sample_query, temp_db_path):
        """Verify pytest fixtures are working."""
        assert len(sample_vectors) == 100
        assert len(sample_vectors[0]) == 128
        assert len(sample_query) == 128
        assert temp_db_path is not None
