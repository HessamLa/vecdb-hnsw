"""
Unit tests for VecDB API Layer.

Tests the main VecDB class that provides the public interface.
"""

import os
import tempfile
import pytest

from vecdb import (
    VecDB,
    Collection,
    CollectionExistsError,
    CollectionNotFoundError,
    DimensionError,
)


class TestVecDBInit:
    """Tests for VecDB initialization."""

    def test_init_creates_directory(self):
        """Test that initialization creates the database directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'new_db')
            db = VecDB(db_path)

            assert os.path.exists(db_path)
            assert len(db) == 0

    def test_init_loads_existing_collections(self):
        """Test that initialization loads existing collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a database
            with VecDB(tmpdir) as db:
                col = db.create_collection('test', dimension=3, metric='l2')
                col.insert(1, [1.0, 2.0, 3.0])

            # Reopen
            db2 = VecDB(tmpdir)
            assert 'test' in db2
            assert db2.get_collection('test').count() == 1


class TestCreateCollection:
    """Tests for create_collection method."""

    def test_create_basic(self):
        """Test creating a basic collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            col = db.create_collection('test', dimension=128, metric='l2')

            assert isinstance(col, Collection)
            assert col.name == 'test'
            assert col.dimension == 128
            assert col.metric == 'l2'
            assert 'test' in db

    def test_create_with_all_params(self):
        """Test creating collection with all parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            col = db.create_collection(
                name='test',
                dimension=64,
                metric='cosine',
                hnsw_params={'M': 32, 'ef_construction': 400}
            )

            assert col.dimension == 64
            assert col.metric == 'cosine'

    def test_create_duplicate_raises_error(self):
        """Test that creating duplicate collection raises CollectionExistsError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            db.create_collection('test', dimension=128)

            with pytest.raises(CollectionExistsError, match="'test' already exists"):
                db.create_collection('test', dimension=128)

    def test_create_different_metrics(self):
        """Test creating collections with different metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)

            for metric in ['l2', 'cosine', 'dot']:
                col = db.create_collection(f'test_{metric}', dimension=32, metric=metric)
                assert col.metric == metric


class TestGetCollection:
    """Tests for get_collection method."""

    def test_get_existing(self):
        """Test getting an existing collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            created = db.create_collection('test', dimension=128)
            retrieved = db.get_collection('test')

            assert retrieved is created

    def test_get_nonexistent_raises_error(self):
        """Test that getting non-existent collection raises CollectionNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)

            with pytest.raises(CollectionNotFoundError, match="'nonexistent' not found"):
                db.get_collection('nonexistent')


class TestDeleteCollection:
    """Tests for delete_collection method."""

    def test_delete_existing(self):
        """Test deleting an existing collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            db.create_collection('test', dimension=128)

            result = db.delete_collection('test')
            assert result is True
            assert 'test' not in db

    def test_delete_nonexistent(self):
        """Test that deleting non-existent collection returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            result = db.delete_collection('nonexistent')
            assert result is False

    def test_delete_persisted_files(self):
        """Test that deleting removes persisted files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col = db.create_collection('test', dimension=3)
                col.insert(1, [1.0, 2.0, 3.0])

            # Files should exist
            assert os.path.exists(os.path.join(tmpdir, 'collections', 'test.meta'))

            db2 = VecDB(tmpdir)
            db2.delete_collection('test')

            # Files should be deleted
            assert not os.path.exists(os.path.join(tmpdir, 'collections', 'test.meta'))


class TestListCollections:
    """Tests for list_collections method."""

    def test_list_empty(self):
        """Test listing collections when none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            assert db.list_collections() == []

    def test_list_multiple(self):
        """Test listing multiple collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            db.create_collection('beta', dimension=32)
            db.create_collection('alpha', dimension=64)
            db.create_collection('gamma', dimension=128)

            collections = db.list_collections()
            assert collections == ['alpha', 'beta', 'gamma']  # Sorted


class TestSave:
    """Tests for save method."""

    def test_save_persists_data(self):
        """Test that save persists collection data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            col = db.create_collection('test', dimension=3)
            col.insert(1, [1.0, 2.0, 3.0])
            col.insert(2, [4.0, 5.0, 6.0])

            db.save()

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, 'collections', 'test.meta'))
            assert os.path.exists(os.path.join(tmpdir, 'collections', 'test.hnsw'))
            assert os.path.exists(os.path.join(tmpdir, 'collections', 'test.vectors'))

            # Reopen and verify data
            db2 = VecDB(tmpdir)
            col2 = db2.get_collection('test')
            assert col2.count() == 2
            assert col2.get(1) == [1.0, 2.0, 3.0]

    def test_save_multiple_collections(self):
        """Test saving multiple collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)

            for name in ['a', 'b', 'c']:
                col = db.create_collection(name, dimension=3)
                col.insert(1, [1.0, 2.0, 3.0])

            db.save()

            db2 = VecDB(tmpdir)
            assert db2.list_collections() == ['a', 'b', 'c']


class TestContextManager:
    """Tests for context manager support."""

    def test_context_manager_saves_on_exit(self):
        """Test that context manager saves on exit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                col = db.create_collection('test', dimension=3)
                col.insert(1, [1.0, 2.0, 3.0])
                # No explicit save

            # Data should be persisted
            db2 = VecDB(tmpdir)
            assert 'test' in db2
            assert db2.get_collection('test').count() == 1

    def test_context_manager_returns_self(self):
        """Test that context manager returns the VecDB instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                assert isinstance(db, VecDB)


class TestContains:
    """Tests for __contains__ support."""

    def test_in_operator(self):
        """Test 'in' operator for checking collection existence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            db.create_collection('test', dimension=3)

            assert 'test' in db
            assert 'nonexistent' not in db


class TestLen:
    """Tests for __len__ support."""

    def test_len_empty(self):
        """Test len() on empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            assert len(db) == 0

    def test_len_multiple(self):
        """Test len() with multiple collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            db.create_collection('a', dimension=3)
            db.create_collection('b', dimension=3)
            db.create_collection('c', dimension=3)

            assert len(db) == 3


class TestRepr:
    """Tests for __repr__ support."""

    def test_repr(self):
        """Test string representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = VecDB(tmpdir)
            db.create_collection('test', dimension=3)

            repr_str = repr(db)
            assert 'VecDB' in repr_str
            assert tmpdir in repr_str
            assert 'collections=1' in repr_str


class TestFullWorkflow:
    """Integration-style tests for full workflows."""

    def test_full_crud_workflow(self):
        """Test a complete CRUD workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                # Create
                col = db.create_collection('docs', dimension=4, metric='cosine')

                # Insert
                col.insert(1, [1.0, 0.0, 0.0, 0.0])
                col.insert(2, [0.0, 1.0, 0.0, 0.0])
                col.insert(3, [0.5, 0.5, 0.0, 0.0])

                # Search
                results = col.search([1.0, 0.0, 0.0, 0.0], k=2)
                assert results[0][0] == 1  # Exact match first

                # Get
                vec = col.get(1)
                assert vec == [1.0, 0.0, 0.0, 0.0]

                # Delete vector
                col.delete(2)
                assert col.count() == 2
                assert col.get(2) is None

            # Reopen and verify persistence
            with VecDB(tmpdir) as db:
                col = db.get_collection('docs')
                assert col.count() == 2
                assert col.get(1) == [1.0, 0.0, 0.0, 0.0]
                assert col.get(2) is None

                # Delete collection
                db.delete_collection('docs')
                assert 'docs' not in db

    def test_multiple_collections_workflow(self):
        """Test workflow with multiple collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with VecDB(tmpdir) as db:
                # Create collections with different dimensions
                col_128 = db.create_collection('embeddings_small', dimension=128, metric='cosine')
                col_512 = db.create_collection('embeddings_large', dimension=512, metric='l2')

                # Insert into each
                col_128.insert(1, [0.1] * 128)
                col_512.insert(1, [0.2] * 512)

            # Reopen
            db = VecDB(tmpdir)
            assert len(db) == 2

            col_128 = db.get_collection('embeddings_small')
            col_512 = db.get_collection('embeddings_large')

            assert col_128.dimension == 128
            assert col_512.dimension == 512

            # Verify dimension enforcement
            with pytest.raises(DimensionError):
                col_128.insert(2, [0.1] * 512)  # Wrong dimension
