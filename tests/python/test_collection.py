"""
Unit tests for Collection Manager.

Tests the Collection class that manages user ID mappings and HNSW interactions.
"""

import pytest
import numpy as np

from vecdb.collection import Collection
from vecdb.exceptions import DimensionError, DuplicateIDError


class TestCollectionInit:
    """Tests for Collection initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        col = Collection('test', dimension=128, metric='l2')
        assert col.name == 'test'
        assert col.dimension == 128
        assert col.metric == 'l2'
        assert col.count() == 0
        assert len(col) == 0

    def test_init_with_hnsw_params(self):
        """Test initialization with HNSW parameters."""
        col = Collection(
            'test',
            dimension=64,
            metric='cosine',
            hnsw_params={'M': 32, 'ef_construction': 400}
        )
        assert col.dimension == 64
        assert col.metric == 'cosine'

    def test_init_invalid_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Collection('', dimension=128, metric='l2')

    def test_init_invalid_dimension(self):
        """Test that dimension < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Dimension must be >= 1"):
            Collection('test', dimension=0, metric='l2')

    def test_init_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="Invalid metric"):
            Collection('test', dimension=128, metric='invalid')


class TestInsertAndGet:
    """Tests for insert and get operations."""

    def test_insert_and_get(self):
        """Test inserting and retrieving a vector."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1001, [1.0, 2.0, 3.0])

        vec = col.get(1001)
        assert vec == [1.0, 2.0, 3.0]

    def test_insert_multiple(self):
        """Test inserting multiple vectors."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 0.0, 0.0])
        col.insert(2, [0.0, 1.0, 0.0])
        col.insert(3, [0.0, 0.0, 1.0])

        assert col.count() == 3
        assert col.get(1) == [1.0, 0.0, 0.0]
        assert col.get(2) == [0.0, 1.0, 0.0]
        assert col.get(3) == [0.0, 0.0, 1.0]

    def test_insert_numpy_array(self):
        """Test inserting numpy array."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, np.array([1.0, 2.0, 3.0]))

        vec = col.get(1)
        assert vec == [1.0, 2.0, 3.0]

    def test_get_nonexistent(self):
        """Test that get returns None for non-existent ID."""
        col = Collection('test', dimension=3, metric='l2')
        assert col.get(999) is None

    def test_get_returns_copy(self):
        """Test that get returns a copy, not the original."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 2.0, 3.0])

        vec = col.get(1)
        vec[0] = 999.0  # Modify the returned vector

        # Original should be unchanged
        assert col.get(1) == [1.0, 2.0, 3.0]


class TestSearch:
    """Tests for search operation."""

    def test_search_returns_user_ids(self):
        """Test that search returns user IDs, not internal IDs."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1001, [1.0, 0.0, 0.0])
        col.insert(1002, [0.0, 1.0, 0.0])
        col.insert(1003, [0.0, 0.0, 1.0])

        results = col.search([1.0, 0.0, 0.0], k=3)

        # Verify results contain user IDs
        user_ids = [r[0] for r in results]
        assert 1001 in user_ids
        assert 1002 in user_ids
        assert 1003 in user_ids

        # Verify ordering (closest first)
        assert results[0][0] == 1001

    def test_search_with_numpy(self):
        """Test search with numpy array query."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 0.0, 0.0])
        col.insert(2, [0.0, 1.0, 0.0])

        results = col.search(np.array([1.0, 0.0, 0.0]), k=2)
        assert results[0][0] == 1

    def test_search_empty_collection(self):
        """Test searching empty collection returns empty list."""
        col = Collection('test', dimension=3, metric='l2')
        results = col.search([1.0, 2.0, 3.0], k=5)
        assert results == []

    def test_search_k_larger_than_size(self):
        """Test that k > collection size returns all vectors."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 0.0, 0.0])
        col.insert(2, [0.0, 1.0, 0.0])

        results = col.search([0.0, 0.0, 0.0], k=100)
        assert len(results) == 2


class TestDelete:
    """Tests for delete operation."""

    def test_delete_existing(self):
        """Test deleting an existing vector."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 2.0, 3.0])
        assert col.count() == 1

        result = col.delete(1)
        assert result is True
        assert col.count() == 0

    def test_delete_nonexistent(self):
        """Test deleting non-existent vector returns False."""
        col = Collection('test', dimension=3, metric='l2')
        result = col.delete(999)
        assert result is False

    def test_delete_removes_from_get(self):
        """Test that deleted vector is not returned by get."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 2.0, 3.0])
        col.delete(1)
        assert col.get(1) is None

    def test_delete_removes_from_search(self):
        """Test that deleted vector is excluded from search."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 0.0, 0.0])
        col.insert(2, [0.0, 1.0, 0.0])

        col.delete(1)

        results = col.search([1.0, 0.0, 0.0], k=10)
        user_ids = [r[0] for r in results]
        assert 1 not in user_ids
        assert 2 in user_ids

    def test_delete_removes_from_contains(self):
        """Test that deleted vector is not in collection."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 2.0, 3.0])
        assert col.contains(1) is True
        assert 1 in col

        col.delete(1)
        assert col.contains(1) is False
        assert 1 not in col


class TestContains:
    """Tests for contains operation."""

    def test_contains_existing(self):
        """Test contains returns True for existing ID."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 2.0, 3.0])
        assert col.contains(1) is True
        assert 1 in col

    def test_contains_nonexistent(self):
        """Test contains returns False for non-existent ID."""
        col = Collection('test', dimension=3, metric='l2')
        assert col.contains(999) is False
        assert 999 not in col


class TestErrorHandling:
    """Tests for error handling."""

    def test_duplicate_user_id(self):
        """Test that duplicate user ID raises DuplicateIDError."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 2.0, 3.0])

        with pytest.raises(DuplicateIDError, match="User ID 1 already exists"):
            col.insert(1, [4.0, 5.0, 6.0])

    def test_dimension_validation_insert(self):
        """Test that wrong dimension on insert raises DimensionError."""
        col = Collection('test', dimension=3, metric='l2')

        with pytest.raises(DimensionError, match="Expected dimension 3"):
            col.insert(1, [1.0, 2.0])  # Wrong dimension

    def test_dimension_validation_search(self):
        """Test that wrong dimension on search raises DimensionError."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 2.0, 3.0])

        with pytest.raises(DimensionError, match="Expected dimension 3"):
            col.search([1.0, 2.0], k=1)  # Wrong dimension


class TestDistanceMetrics:
    """Tests for different distance metrics."""

    def test_l2_metric(self):
        """Test L2 distance metric."""
        col = Collection('test', dimension=2, metric='l2')
        col.insert(1, [0.0, 0.0])
        col.insert(2, [3.0, 4.0])

        results = col.search([0.0, 0.0], k=2)
        assert results[0][0] == 1
        assert abs(results[0][1] - 0.0) < 1e-6
        assert abs(results[1][1] - 5.0) < 1e-6  # 3-4-5 triangle

    def test_cosine_metric(self):
        """Test cosine distance metric."""
        col = Collection('test', dimension=2, metric='cosine')
        col.insert(1, [1.0, 0.0])
        col.insert(2, [0.0, 1.0])

        results = col.search([1.0, 0.0], k=2)
        assert results[0][0] == 1
        assert abs(results[0][1] - 0.0) < 1e-6  # Same direction
        assert abs(results[1][1] - 1.0) < 1e-6  # Perpendicular

    def test_dot_metric(self):
        """Test dot product distance metric."""
        col = Collection('test', dimension=2, metric='dot')
        col.insert(1, [1.0, 1.0])
        col.insert(2, [2.0, 2.0])

        results = col.search([1.0, 1.0], k=2)
        # Higher dot product comes first
        assert results[0][0] == 2


class TestInternalState:
    """Tests for internal state management."""

    def test_get_state(self):
        """Test _get_state returns correct state."""
        col = Collection('test', dimension=3, metric='l2')
        col.insert(1, [1.0, 2.0, 3.0])
        col.insert(2, [4.0, 5.0, 6.0])

        state = col._get_state()

        assert state['name'] == 'test'
        assert state['dimension'] == 3
        assert state['metric'] == 'l2'
        assert len(state['vectors']) == 2
        assert 1 in state['user_to_internal']
        assert 2 in state['user_to_internal']

    def test_from_state(self):
        """Test _from_state reconstructs collection correctly."""
        # Create original
        col1 = Collection('test', dimension=3, metric='l2')
        col1.insert(1, [1.0, 2.0, 3.0])
        col1.insert(2, [4.0, 5.0, 6.0])

        # Get state and index
        state = col1._get_state()
        hnsw_index = col1._get_hnsw_index()

        # Reconstruct
        col2 = Collection._from_state(state, hnsw_index)

        assert col2.name == col1.name
        assert col2.dimension == col1.dimension
        assert col2.metric == col1.metric
        assert col2.count() == col1.count()
        assert col2.get(1) == col1.get(1)
        assert col2.get(2) == col1.get(2)
