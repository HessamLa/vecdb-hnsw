"""
Unit tests for Mock HNSW Index.

Tests the brute-force mock implementation that matches the C++ interface.
"""

import math
import pytest
import numpy as np

from vecdb._hnsw_mock import HNSWIndex
from vecdb.exceptions import DimensionError, DuplicateIDError, DeserializationError


class TestHNSWIndexInit:
    """Tests for HNSWIndex initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        index = HNSWIndex(dimension=128, metric='l2')
        assert index.dimension == 128
        assert index.metric == 'l2'
        assert index.M == 16  # Default
        assert index.ef_construction == 200  # Default
        assert len(index) == 0

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        index = HNSWIndex(dimension=64, metric='cosine', M=32, ef_construction=400)
        assert index.dimension == 64
        assert index.metric == 'cosine'
        assert index.M == 32
        assert index.ef_construction == 400

    def test_init_invalid_dimension(self):
        """Test that dimension < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Dimension must be >= 1"):
            HNSWIndex(dimension=0, metric='l2')

        with pytest.raises(ValueError, match="Dimension must be >= 1"):
            HNSWIndex(dimension=-1, metric='l2')

    def test_init_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="Invalid metric"):
            HNSWIndex(dimension=128, metric='invalid')


class TestDistanceL2:
    """Tests for L2 (Euclidean) distance calculation."""

    def test_distance_l2_identical(self):
        """Test L2 distance between identical vectors is 0."""
        index = HNSWIndex(dimension=3, metric='l2')
        index.add(0, [1.0, 2.0, 3.0])
        results = index.search([1.0, 2.0, 3.0], k=1)
        assert abs(results[0][1]) < 1e-6

    def test_distance_l2_known(self):
        """Test L2 distance with known values (3-4-5 triangle)."""
        index = HNSWIndex(dimension=2, metric='l2')
        index.add(0, [0.0, 0.0])
        index.add(1, [3.0, 4.0])
        results = index.search([0.0, 0.0], k=2)
        assert abs(results[0][1] - 0.0) < 1e-6  # Distance to self
        assert abs(results[1][1] - 5.0) < 1e-6  # 3-4-5 triangle

    def test_distance_l2_ordering(self):
        """Test that L2 distance orders correctly."""
        index = HNSWIndex(dimension=2, metric='l2')
        index.add(0, [0.0, 0.0])
        index.add(1, [1.0, 0.0])
        index.add(2, [2.0, 0.0])
        results = index.search([0.0, 0.0], k=3)
        assert results[0][0] == 0  # Closest
        assert results[1][0] == 1  # Second closest
        assert results[2][0] == 2  # Farthest


class TestDistanceCosine:
    """Tests for cosine distance calculation."""

    def test_distance_cosine_identical_direction(self):
        """Test cosine distance for vectors in same direction is 0."""
        index = HNSWIndex(dimension=3, metric='cosine')
        index.add(0, [1.0, 0.0, 0.0])
        index.add(1, [2.0, 0.0, 0.0])  # Same direction, different magnitude
        results = index.search([1.0, 0.0, 0.0], k=2)
        assert abs(results[0][1]) < 1e-6
        assert abs(results[1][1]) < 1e-6

    def test_distance_cosine_perpendicular(self):
        """Test cosine distance for perpendicular vectors is 1."""
        index = HNSWIndex(dimension=2, metric='cosine')
        index.add(0, [1.0, 0.0])
        index.add(1, [0.0, 1.0])
        results = index.search([1.0, 0.0], k=2)
        # First result is self (distance 0)
        assert abs(results[0][1]) < 1e-6
        # Second is perpendicular (cosine sim 0, distance 1)
        assert abs(results[1][1] - 1.0) < 1e-6

    def test_distance_cosine_opposite(self):
        """Test cosine distance for opposite vectors is 2."""
        index = HNSWIndex(dimension=2, metric='cosine')
        index.add(0, [1.0, 0.0])
        index.add(1, [-1.0, 0.0])
        results = index.search([1.0, 0.0], k=2)
        assert abs(results[0][1]) < 1e-6  # Self
        assert abs(results[1][1] - 2.0) < 1e-6  # Opposite


class TestDistanceDot:
    """Tests for dot product distance calculation."""

    def test_distance_dot_ordering(self):
        """Test that dot distance orders by maximum inner product."""
        index = HNSWIndex(dimension=2, metric='dot')
        index.add(0, [1.0, 1.0])  # dot = 2
        index.add(1, [2.0, 2.0])  # dot = 4
        index.add(2, [0.5, 0.5])  # dot = 1
        results = index.search([1.0, 1.0], k=3)
        # Higher dot product should come first (lower negative distance)
        assert results[0][0] == 1  # dot=4, dist=-4
        assert results[1][0] == 0  # dot=2, dist=-2
        assert results[2][0] == 2  # dot=1, dist=-1

    def test_distance_dot_values(self):
        """Test dot distance returns negative dot product."""
        index = HNSWIndex(dimension=2, metric='dot')
        index.add(0, [3.0, 4.0])
        results = index.search([1.0, 2.0], k=1)
        # dot([1,2], [3,4]) = 3 + 8 = 11
        assert abs(results[0][1] - (-11.0)) < 1e-6


class TestAddAndSearch:
    """Tests for add and search operations."""

    def test_add_single(self):
        """Test adding a single vector."""
        index = HNSWIndex(dimension=3, metric='l2')
        index.add(0, [1.0, 2.0, 3.0])
        assert len(index) == 1

    def test_add_multiple(self):
        """Test adding multiple vectors."""
        index = HNSWIndex(dimension=3, metric='l2')
        for i in range(100):
            index.add(i, [float(i), float(i), float(i)])
        assert len(index) == 100

    def test_add_numpy_array(self):
        """Test adding numpy arrays."""
        index = HNSWIndex(dimension=3, metric='l2')
        index.add(0, np.array([1.0, 2.0, 3.0]))
        assert len(index) == 1
        results = index.search(np.array([1.0, 2.0, 3.0]), k=1)
        assert abs(results[0][1]) < 1e-6

    def test_search_exact(self):
        """Test that exact match is returned first."""
        index = HNSWIndex(dimension=3, metric='l2')
        index.add(0, [1.0, 0.0, 0.0])
        index.add(1, [0.0, 1.0, 0.0])
        index.add(2, [0.0, 0.0, 1.0])
        results = index.search([1.0, 0.0, 0.0], k=3)
        assert results[0][0] == 0
        assert abs(results[0][1]) < 1e-6

    def test_search_empty_index(self):
        """Test searching empty index returns empty list."""
        index = HNSWIndex(dimension=3, metric='l2')
        results = index.search([1.0, 2.0, 3.0], k=5)
        assert results == []

    def test_search_k_larger_than_size(self):
        """Test that k > index size returns all vectors."""
        index = HNSWIndex(dimension=3, metric='l2')
        index.add(0, [1.0, 0.0, 0.0])
        index.add(1, [0.0, 1.0, 0.0])
        results = index.search([0.0, 0.0, 0.0], k=100)
        assert len(results) == 2

    def test_search_recall(self):
        """Test search recall against brute force."""
        np.random.seed(42)
        dim = 32
        n_vectors = 100

        index = HNSWIndex(dimension=dim, metric='l2')
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)

        for i, vec in enumerate(vectors):
            index.add(i, vec.tolist())

        # Query
        query = np.random.randn(dim).astype(np.float32)
        k = 10

        # Get results from index
        results = index.search(query.tolist(), k=k)
        result_ids = set(r[0] for r in results)

        # Brute force ground truth
        distances = [np.linalg.norm(query - vec) for vec in vectors]
        sorted_indices = np.argsort(distances)[:k]
        ground_truth = set(sorted_indices)

        # Since mock is brute-force, recall should be 100%
        recall = len(result_ids & ground_truth) / k
        assert recall == 1.0


class TestRemove:
    """Tests for remove operation."""

    def test_remove_existing(self):
        """Test removing an existing vector."""
        index = HNSWIndex(dimension=3, metric='l2')
        index.add(0, [1.0, 2.0, 3.0])
        assert len(index) == 1

        result = index.remove(0)
        assert result is True
        assert len(index) == 0

    def test_remove_nonexistent(self):
        """Test removing a non-existent vector returns False."""
        index = HNSWIndex(dimension=3, metric='l2')
        result = index.remove(999)
        assert result is False

    def test_remove_excluded_from_search(self):
        """Test that removed vectors are excluded from search."""
        index = HNSWIndex(dimension=3, metric='l2')
        index.add(0, [1.0, 0.0, 0.0])
        index.add(1, [0.0, 1.0, 0.0])

        index.remove(0)

        results = index.search([1.0, 0.0, 0.0], k=10)
        result_ids = [r[0] for r in results]
        assert 0 not in result_ids
        assert 1 in result_ids

    def test_remove_and_readd(self):
        """Test that removed ID can be re-added."""
        index = HNSWIndex(dimension=3, metric='l2')
        index.add(0, [1.0, 2.0, 3.0])
        index.remove(0)
        index.add(0, [4.0, 5.0, 6.0])  # Should not raise
        assert len(index) == 1


class TestSerializeDeserialize:
    """Tests for serialization and deserialization."""

    def test_serialize_deserialize_roundtrip(self):
        """Test that serialize/deserialize produces identical results."""
        index = HNSWIndex(dimension=3, metric='cosine', M=24, ef_construction=150)
        index.add(0, [1.0, 0.0, 0.0])
        index.add(1, [0.0, 1.0, 0.0])
        index.add(2, [0.0, 0.0, 1.0])

        # Serialize
        data = index.serialize()
        assert isinstance(data, bytes)

        # Deserialize
        index2 = HNSWIndex.deserialize(data)

        # Verify properties
        assert index2.dimension == index.dimension
        assert index2.metric == index.metric
        assert index2.M == index.M
        assert index2.ef_construction == index.ef_construction
        assert len(index2) == len(index)

        # Verify search results are identical
        query = [0.5, 0.5, 0.0]
        results1 = index.search(query, k=3)
        results2 = index2.search(query, k=3)
        assert results1 == results2

    def test_deserialize_corrupt_data(self):
        """Test that corrupt data raises DeserializationError."""
        with pytest.raises(DeserializationError):
            HNSWIndex.deserialize(b'corrupt data')

    def test_serialize_empty_index(self):
        """Test serialization of empty index."""
        index = HNSWIndex(dimension=3, metric='l2')
        data = index.serialize()
        index2 = HNSWIndex.deserialize(data)
        assert len(index2) == 0
        assert index2.dimension == 3
        assert index2.metric == 'l2'


class TestErrorHandling:
    """Tests for error handling."""

    def test_dimension_mismatch_add(self):
        """Test that dimension mismatch on add raises DimensionError."""
        index = HNSWIndex(dimension=3, metric='l2')
        with pytest.raises(DimensionError):
            index.add(0, [1.0, 2.0])  # Wrong dimension

    def test_dimension_mismatch_search(self):
        """Test that dimension mismatch on search raises DimensionError."""
        index = HNSWIndex(dimension=3, metric='l2')
        index.add(0, [1.0, 2.0, 3.0])
        with pytest.raises(DimensionError):
            index.search([1.0, 2.0], k=1)  # Wrong dimension

    def test_duplicate_id(self):
        """Test that duplicate ID raises DuplicateIDError."""
        index = HNSWIndex(dimension=3, metric='l2')
        index.add(0, [1.0, 2.0, 3.0])
        with pytest.raises(DuplicateIDError):
            index.add(0, [4.0, 5.0, 6.0])

    def test_invalid_k(self):
        """Test that k < 1 raises ValueError."""
        index = HNSWIndex(dimension=3, metric='l2')
        index.add(0, [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="k must be >= 1"):
            index.search([1.0, 2.0, 3.0], k=0)
