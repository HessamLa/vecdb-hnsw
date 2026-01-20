"""
Unit tests for Persistence Manager.

Tests the PersistenceManager class for saving and loading collections.
"""

import json
import os
import tempfile
import pytest

from vecdb.collection import Collection
from vecdb.persistence import PersistenceManager
from vecdb.exceptions import DeserializationError


class TestPersistenceManagerInit:
    """Tests for PersistenceManager initialization."""

    def test_init_creates_directories(self):
        """Test that initialization creates required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'new_db')
            pm = PersistenceManager(db_path)

            assert os.path.exists(db_path)
            assert os.path.exists(os.path.join(db_path, 'collections'))

    def test_init_existing_directory(self):
        """Test initialization with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, 'collections'))


class TestSaveLoadCollection:
    """Tests for save and load collection operations."""

    def test_save_load_collection(self):
        """Test saving and loading a collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            # Create and populate collection
            col = Collection('test', dimension=3, metric='l2')
            col.insert(1001, [1.0, 2.0, 3.0])
            col.insert(1002, [4.0, 5.0, 6.0])

            # Save
            pm.save_collection(col)

            # Verify files created
            assert os.path.exists(os.path.join(tmpdir, 'collections', 'test.meta'))
            assert os.path.exists(os.path.join(tmpdir, 'collections', 'test.hnsw'))
            assert os.path.exists(os.path.join(tmpdir, 'collections', 'test.vectors'))

            # Load
            loaded = pm.load_collection('test')

            assert loaded is not None
            assert loaded.name == 'test'
            assert loaded.dimension == 3
            assert loaded.metric == 'l2'
            assert loaded.count() == 2
            assert loaded.get(1001) == [1.0, 2.0, 3.0]
            assert loaded.get(1002) == [4.0, 5.0, 6.0]

    def test_save_load_different_metrics(self):
        """Test saving and loading collections with different metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            for metric in ['l2', 'cosine', 'dot']:
                col = Collection(f'test_{metric}', dimension=3, metric=metric)
                col.insert(1, [1.0, 2.0, 3.0])

                pm.save_collection(col)
                loaded = pm.load_collection(f'test_{metric}')

                assert loaded.metric == metric

    def test_save_load_preserves_search_results(self):
        """Test that search results are identical after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            col = Collection('test', dimension=3, metric='l2')
            col.insert(1, [1.0, 0.0, 0.0])
            col.insert(2, [0.0, 1.0, 0.0])
            col.insert(3, [0.0, 0.0, 1.0])

            query = [0.5, 0.5, 0.0]
            results_before = col.search(query, k=3)

            pm.save_collection(col)
            loaded = pm.load_collection('test')

            results_after = loaded.search(query, k=3)
            assert results_before == results_after

    def test_save_overwrites_existing(self):
        """Test that saving overwrites existing collection files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            # Save version 1
            col1 = Collection('test', dimension=3, metric='l2')
            col1.insert(1, [1.0, 2.0, 3.0])
            pm.save_collection(col1)

            # Save version 2 (same name, different data)
            col2 = Collection('test', dimension=3, metric='l2')
            col2.insert(2, [4.0, 5.0, 6.0])
            col2.insert(3, [7.0, 8.0, 9.0])
            pm.save_collection(col2)

            # Load should get version 2
            loaded = pm.load_collection('test')
            assert loaded.count() == 2
            assert loaded.get(1) is None
            assert loaded.get(2) == [4.0, 5.0, 6.0]


class TestLoadNonExistent:
    """Tests for loading non-existent collections."""

    def test_load_nonexistent_returns_none(self):
        """Test that loading non-existent collection returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)
            result = pm.load_collection('nonexistent')
            assert result is None

    def test_load_partial_files_returns_none(self):
        """Test that partial files (missing some) returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            # Create only meta file
            meta_path = os.path.join(tmpdir, 'collections', 'partial.meta')
            with open(meta_path, 'w') as f:
                json.dump({'name': 'partial', 'dimension': 3, 'metric': 'l2'}, f)

            result = pm.load_collection('partial')
            assert result is None


class TestDeleteCollection:
    """Tests for delete collection operation."""

    def test_delete_existing_collection(self):
        """Test deleting an existing collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            col = Collection('test', dimension=3, metric='l2')
            col.insert(1, [1.0, 2.0, 3.0])
            pm.save_collection(col)

            result = pm.delete_collection('test')
            assert result is True

            # Files should be deleted
            assert not os.path.exists(os.path.join(tmpdir, 'collections', 'test.meta'))
            assert not os.path.exists(os.path.join(tmpdir, 'collections', 'test.hnsw'))
            assert not os.path.exists(os.path.join(tmpdir, 'collections', 'test.vectors'))

            # Load should return None
            assert pm.load_collection('test') is None

    def test_delete_nonexistent_returns_false(self):
        """Test that deleting non-existent collection returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)
            result = pm.delete_collection('nonexistent')
            assert result is False


class TestListCollections:
    """Tests for list collections operation."""

    def test_list_empty(self):
        """Test listing collections when none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)
            collections = pm.list_collections()
            assert collections == []

    def test_list_multiple(self):
        """Test listing multiple collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            for name in ['alpha', 'beta', 'gamma']:
                col = Collection(name, dimension=3, metric='l2')
                col.insert(1, [1.0, 2.0, 3.0])
                pm.save_collection(col)

            collections = pm.list_collections()
            assert collections == ['alpha', 'beta', 'gamma']  # Sorted

    def test_list_after_delete(self):
        """Test listing after deleting a collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            for name in ['a', 'b', 'c']:
                col = Collection(name, dimension=3, metric='l2')
                pm.save_collection(col)

            pm.delete_collection('b')

            collections = pm.list_collections()
            assert collections == ['a', 'c']


class TestMetadata:
    """Tests for database-level metadata."""

    def test_save_load_metadata(self):
        """Test saving and loading metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            metadata = {
                'created': '2025-01-01',
                'description': 'Test database',
                'custom_field': 123
            }
            pm.save_metadata(metadata)

            loaded = pm.load_metadata()
            assert loaded['created'] == '2025-01-01'
            assert loaded['description'] == 'Test database'
            assert loaded['custom_field'] == 123
            assert 'version' in loaded

    def test_load_metadata_empty(self):
        """Test loading metadata when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)
            metadata = pm.load_metadata()
            assert metadata == {}


class TestCorruptFiles:
    """Tests for handling corrupt files."""

    def test_corrupt_meta_file(self):
        """Test that corrupt meta file raises DeserializationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            # Create all required files
            col = Collection('test', dimension=3, metric='l2')
            col.insert(1, [1.0, 2.0, 3.0])
            pm.save_collection(col)

            # Corrupt the meta file
            meta_path = os.path.join(tmpdir, 'collections', 'test.meta')
            with open(meta_path, 'w') as f:
                f.write('not valid json')

            with pytest.raises(DeserializationError, match="Corrupt metadata"):
                pm.load_collection('test')

    def test_corrupt_hnsw_file(self):
        """Test that corrupt HNSW file raises DeserializationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            col = Collection('test', dimension=3, metric='l2')
            col.insert(1, [1.0, 2.0, 3.0])
            pm.save_collection(col)

            # Corrupt the HNSW file
            hnsw_path = os.path.join(tmpdir, 'collections', 'test.hnsw')
            with open(hnsw_path, 'wb') as f:
                f.write(b'corrupt data')

            with pytest.raises(DeserializationError):
                pm.load_collection('test')

    def test_corrupt_vectors_file(self):
        """Test that corrupt vectors file raises DeserializationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            col = Collection('test', dimension=3, metric='l2')
            col.insert(1, [1.0, 2.0, 3.0])
            pm.save_collection(col)

            # Corrupt the vectors file (truncate it)
            vectors_path = os.path.join(tmpdir, 'collections', 'test.vectors')
            with open(vectors_path, 'wb') as f:
                f.write(b'short')

            with pytest.raises(DeserializationError, match="too small"):
                pm.load_collection('test')


class TestAtomicWrites:
    """Tests for atomic write behavior."""

    def test_no_temp_files_left(self):
        """Test that no .tmp files are left after save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            col = Collection('test', dimension=3, metric='l2')
            col.insert(1, [1.0, 2.0, 3.0])
            pm.save_collection(col)

            collections_dir = os.path.join(tmpdir, 'collections')
            files = os.listdir(collections_dir)
            tmp_files = [f for f in files if f.endswith('.tmp')]
            assert tmp_files == []


class TestLargeCollections:
    """Tests for larger collections."""

    def test_save_load_100_vectors(self):
        """Test saving and loading 100 vectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistenceManager(tmpdir)

            col = Collection('test', dimension=32, metric='l2')
            for i in range(100):
                col.insert(i, [float(j) for j in range(32)])

            pm.save_collection(col)
            loaded = pm.load_collection('test')

            assert loaded.count() == 100
            for i in range(100):
                assert loaded.contains(i)
