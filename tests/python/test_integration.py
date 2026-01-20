"""
Phase 1 Integration Tests

Tests the full workflow of VecDB using the mock HNSW implementation.
Validates the complete integration between all Python modules.
"""

import os
import tempfile
import numpy as np
import pytest

from vecdb import VecDB, Collection, DimensionError


class TestFullWorkflow:
    """Test complete create → insert → search → save → reload → search workflow."""

    def test_full_workflow_l2(self):
        """Test full workflow with L2 distance metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Phase 1: Create, insert, search
            with VecDB(tmpdir) as db:
                col = db.create_collection('vectors', dimension=4, metric='l2')

                # Insert test vectors
                col.insert(100, [1.0, 0.0, 0.0, 0.0])
                col.insert(200, [0.0, 1.0, 0.0, 0.0])
                col.insert(300, [0.0, 0.0, 1.0, 0.0])
                col.insert(400, [0.0, 0.0, 0.0, 1.0])
                col.insert(500, [0.5, 0.5, 0.0, 0.0])

                # Search before save
                results = col.search([1.0, 0.0, 0.0, 0.0], k=3)
                assert len(results) == 3
                assert results[0][0] == 100  # Exact match
                assert abs(results[0][1]) < 1e-6  # Distance ~0

                # Get vector
                vec = col.get(100)
                assert vec == [1.0, 0.0, 0.0, 0.0]

                # Context manager auto-saves on exit

            # Phase 2: Reload and verify
            with VecDB(tmpdir) as db:
                assert 'vectors' in db
                col = db.get_collection('vectors')

                assert col.count() == 5
                assert col.dimension == 4
                assert col.metric == 'l2'

                # Search after reload
                results = col.search([1.0, 0.0, 0.0, 0.0], k=3)
                assert results[0][0] == 100
                assert abs(results[0][1]) < 1e-6

                # Verify all vectors persisted correctly
                assert col.get(100) == [1.0, 0.0, 0.0, 0.0]
                assert col.get(200) == [0.0, 1.0, 0.0, 0.0]
                assert col.get(500) == [0.5, 0.5, 0.0, 0.0]

    def test_full_workflow_cosine(self):
        """Test full workflow with cosine distance metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col = db.create_collection('embeddings', dimension=3, metric='cosine')

                # Insert vectors with different directions
                col.insert(1, [1.0, 0.0, 0.0])  # Unit X
                col.insert(2, [0.0, 1.0, 0.0])  # Unit Y
                col.insert(3, [1.0, 1.0, 0.0])  # 45 degrees
                col.insert(4, [2.0, 0.0, 0.0])  # Same direction as 1, different magnitude

                results = col.search([1.0, 0.0, 0.0], k=4)

                # IDs 1 and 4 should be closest (same direction)
                closest_ids = {results[0][0], results[1][0]}
                assert closest_ids == {1, 4}
                assert abs(results[0][1]) < 1e-6  # Distance ~0 for same direction

            # Reload and verify
            db = VecDB(tmpdir)
            col = db.get_collection('embeddings')

            results = col.search([1.0, 0.0, 0.0], k=2)
            closest_ids = {results[0][0], results[1][0]}
            assert closest_ids == {1, 4}

    def test_full_workflow_dot(self):
        """Test full workflow with dot product distance metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col = db.create_collection('mips', dimension=2, metric='dot')

                col.insert(1, [1.0, 1.0])  # dot = 2
                col.insert(2, [2.0, 2.0])  # dot = 4
                col.insert(3, [3.0, 3.0])  # dot = 6
                col.insert(4, [0.5, 0.5])  # dot = 1

                # Search - higher dot product should come first
                results = col.search([1.0, 1.0], k=4)
                assert results[0][0] == 3  # Highest dot product
                assert results[1][0] == 2
                assert results[2][0] == 1
                assert results[3][0] == 4

            # Reload and verify
            db = VecDB(tmpdir)
            col = db.get_collection('mips')
            results = col.search([1.0, 1.0], k=2)
            assert results[0][0] == 3


class TestMultipleCollections:
    """Test working with multiple collections of different dimensions."""

    def test_multiple_collections_different_dimensions(self):
        """Test two collections with different dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                # Collection for small embeddings
                col_small = db.create_collection('small', dimension=32, metric='l2')
                col_small.insert(1, [0.1] * 32)
                col_small.insert(2, [0.2] * 32)

                # Collection for large embeddings
                col_large = db.create_collection('large', dimension=128, metric='cosine')
                col_large.insert(1, [0.3] * 128)
                col_large.insert(2, [0.4] * 128)

                assert db.list_collections() == ['large', 'small']

            # Reload and verify
            db = VecDB(tmpdir)
            assert len(db) == 2

            col_small = db.get_collection('small')
            col_large = db.get_collection('large')

            assert col_small.dimension == 32
            assert col_large.dimension == 128
            assert col_small.metric == 'l2'
            assert col_large.metric == 'cosine'

            assert col_small.count() == 2
            assert col_large.count() == 2

            # Verify dimension enforcement still works
            with pytest.raises(DimensionError):
                col_small.insert(3, [0.1] * 128)  # Wrong dimension

            with pytest.raises(DimensionError):
                col_large.insert(3, [0.1] * 32)  # Wrong dimension

    def test_multiple_collections_same_dimension(self):
        """Test multiple collections with same dimension but different metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col_l2 = db.create_collection('l2_index', dimension=64, metric='l2')
                col_cos = db.create_collection('cos_index', dimension=64, metric='cosine')
                col_dot = db.create_collection('dot_index', dimension=64, metric='dot')

                # Same vectors in all collections
                vec = [0.1] * 64
                col_l2.insert(1, vec)
                col_cos.insert(1, vec)
                col_dot.insert(1, vec)

            # Reload
            db = VecDB(tmpdir)
            assert len(db) == 3

            # Verify each preserved its metric
            assert db.get_collection('l2_index').metric == 'l2'
            assert db.get_collection('cos_index').metric == 'cosine'
            assert db.get_collection('dot_index').metric == 'dot'


class TestContextManager:
    """Test context manager auto-save behavior."""

    def test_context_manager_auto_saves(self):
        """Test that context manager auto-saves on normal exit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col = db.create_collection('test', dimension=3)
                col.insert(1, [1.0, 2.0, 3.0])
                # No explicit save

            # Verify data persisted
            db = VecDB(tmpdir)
            assert 'test' in db
            assert db.get_collection('test').get(1) == [1.0, 2.0, 3.0]

    def test_context_manager_auto_saves_on_exception(self):
        """Test that context manager saves even when exception occurs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with VecDB(tmpdir) as db:
                    col = db.create_collection('test', dimension=3)
                    col.insert(1, [1.0, 2.0, 3.0])
                    raise ValueError("Simulated error")
            except ValueError:
                pass

            # Verify data persisted
            db = VecDB(tmpdir)
            assert 'test' in db
            assert db.get_collection('test').get(1) == [1.0, 2.0, 3.0]

    def test_nested_operations(self):
        """Test nested insert/search/delete operations with context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col = db.create_collection('test', dimension=3)

                # Insert several vectors
                for i in range(10):
                    col.insert(i, [float(i), float(i), float(i)])

                # Delete some
                col.delete(3)
                col.delete(7)

                # Search
                results = col.search([5.0, 5.0, 5.0], k=3)
                assert 3 not in [r[0] for r in results]
                assert 7 not in [r[0] for r in results]

            # Reload and verify deletions persisted
            db = VecDB(tmpdir)
            col = db.get_collection('test')
            assert col.count() == 8
            assert col.get(3) is None
            assert col.get(7) is None
            assert col.get(5) == [5.0, 5.0, 5.0]


class TestCppModuleIntegration:
    """Test C++ module integration."""

    def test_cpp_module_imports(self):
        """Test that C++ module can be imported."""
        try:
            from vecdb._hnsw_cpp import is_stub, HNSWIndex
            # In Phase 2, is_stub returns False (real implementation)
            assert is_stub() is False
        except ImportError:
            pytest.skip("C++ module not built")

    def test_cpp_module_has_version(self):
        """Test that C++ module has version info."""
        try:
            import vecdb._hnsw_cpp as m
            assert hasattr(m, '__version__')
        except ImportError:
            pytest.skip("C++ module not built")


class TestLargeDataset:
    """Test with larger datasets to validate scalability."""

    def test_1000_vectors(self):
        """Test with 1000 vectors."""
        np.random.seed(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col = db.create_collection('large', dimension=64, metric='l2')

                # Insert 1000 vectors
                for i in range(1000):
                    vec = np.random.randn(64).tolist()
                    col.insert(i, vec)

                assert col.count() == 1000

                # Search
                query = np.random.randn(64).tolist()
                results = col.search(query, k=10)
                assert len(results) == 10

            # Reload and verify
            db = VecDB(tmpdir)
            col = db.get_collection('large')
            assert col.count() == 1000

            # Search should still work
            results = col.search(query, k=10)
            assert len(results) == 10

    def test_high_dimension(self):
        """Test with high dimensional vectors (512d)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col = db.create_collection('highd', dimension=512, metric='cosine')

                # Insert 100 vectors
                for i in range(100):
                    vec = [float(i % 10) / 10.0] * 512
                    col.insert(i, vec)

                # Search
                query = [0.5] * 512
                results = col.search(query, k=5)
                assert len(results) == 5

            # Reload
            db = VecDB(tmpdir)
            col = db.get_collection('highd')
            assert col.count() == 100
            assert col.dimension == 512


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_collection_persistence(self):
        """Test saving and loading empty collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                db.create_collection('empty', dimension=3)

            db = VecDB(tmpdir)
            col = db.get_collection('empty')
            assert col.count() == 0
            assert col.search([1.0, 2.0, 3.0], k=5) == []

    def test_single_vector_collection(self):
        """Test collection with single vector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col = db.create_collection('single', dimension=3)
                col.insert(42, [1.0, 2.0, 3.0])

            db = VecDB(tmpdir)
            col = db.get_collection('single')
            assert col.count() == 1
            results = col.search([1.0, 2.0, 3.0], k=10)
            assert len(results) == 1
            assert results[0][0] == 42

    def test_delete_and_recreate_collection(self):
        """Test deleting and recreating a collection with same name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first version
            with VecDB(tmpdir) as db:
                col = db.create_collection('reuse', dimension=3)
                col.insert(1, [1.0, 0.0, 0.0])
                col.insert(2, [0.0, 1.0, 0.0])

            # Delete and recreate
            with VecDB(tmpdir) as db:
                db.delete_collection('reuse')
                col = db.create_collection('reuse', dimension=4)  # Different dimension!
                col.insert(100, [1.0, 2.0, 3.0, 4.0])

            # Verify new collection
            db = VecDB(tmpdir)
            col = db.get_collection('reuse')
            assert col.dimension == 4
            assert col.count() == 1
            assert col.get(100) == [1.0, 2.0, 3.0, 4.0]
            assert col.get(1) is None  # Old ID doesn't exist

    def test_special_characters_in_collection_name(self):
        """Test collection names with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                # Underscores and numbers should work
                col = db.create_collection('test_collection_123', dimension=3)
                col.insert(1, [1.0, 2.0, 3.0])

            db = VecDB(tmpdir)
            col = db.get_collection('test_collection_123')
            assert col.count() == 1


class TestNumpyIntegration:
    """Test numpy array support throughout the system."""

    def test_numpy_insert_and_search(self):
        """Test inserting and searching with numpy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col = db.create_collection('numpy', dimension=4, metric='l2')

                # Insert numpy arrays
                col.insert(1, np.array([1.0, 0.0, 0.0, 0.0]))
                col.insert(2, np.array([0.0, 1.0, 0.0, 0.0]))

                # Search with numpy array
                results = col.search(np.array([1.0, 0.0, 0.0, 0.0]), k=2)
                assert results[0][0] == 1

            # Reload and verify
            db = VecDB(tmpdir)
            col = db.get_collection('numpy')
            results = col.search(np.array([1.0, 0.0, 0.0, 0.0]), k=2)
            assert results[0][0] == 1

    def test_mixed_list_and_numpy(self):
        """Test mixing list and numpy array inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col = db.create_collection('mixed', dimension=3, metric='l2')

                col.insert(1, [1.0, 2.0, 3.0])  # List
                col.insert(2, np.array([4.0, 5.0, 6.0]))  # Numpy

                # Get returns list
                vec1 = col.get(1)
                vec2 = col.get(2)
                assert isinstance(vec1, list)
                assert isinstance(vec2, list)
