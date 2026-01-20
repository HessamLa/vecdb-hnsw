"""
Pytest configuration and shared fixtures for VecDB tests.
"""

import pytest
import sys
from pathlib import Path

# Ensure src/python is in path
src_path = Path(__file__).parent.parent / "src" / "python"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    import numpy as np
    
    np.random.seed(42)
    return np.random.rand(100, 128).astype(np.float32).tolist()


@pytest.fixture
def sample_query():
    """Generate a sample query vector."""
    import numpy as np
    
    np.random.seed(123)
    return np.random.rand(128).astype(np.float32).tolist()


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a temporary directory for database files."""
    db_path = tmp_path / "test_vecdb"
    db_path.mkdir()
    return str(db_path)
