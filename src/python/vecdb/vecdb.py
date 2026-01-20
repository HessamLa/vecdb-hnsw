"""
VecDB - Main API Layer

Provides the public interface for VecDB users.
Orchestrates Collection Manager and Persistence modules.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from vecdb.collection import Collection
from vecdb.exceptions import CollectionExistsError, CollectionNotFoundError
from vecdb.persistence import PersistenceManager


class VecDB:
    """
    VecDB - A minimal vector database with HNSW indexing.

    Provides a clean API for managing collections of vectors with
    efficient similarity search capabilities.

    Args:
        path: Directory path for database storage (default: './vecdb_data').
              Creates the directory if it doesn't exist.
              Loads existing collections from disk if present.

    Example:
        >>> from vecdb import VecDB
        >>>
        >>> # Create database
        >>> with VecDB('./my_database') as db:
        ...     # Create collection for 384-dim embeddings
        ...     collection = db.create_collection(
        ...         name='documents',
        ...         dimension=384,
        ...         metric='cosine'
        ...     )
        ...
        ...     # Insert vectors
        ...     collection.insert(user_id=1001, vector=[0.1, 0.2, ...])
        ...     collection.insert(user_id=1002, vector=[0.3, 0.4, ...])
        ...
        ...     # Search
        ...     results = collection.search(query=[0.15, 0.25, ...], k=5)
        ...     for user_id, distance in results:
        ...         print(f'ID: {user_id}, Distance: {distance}')
        ...
        ...     # Explicit save (also happens on context exit)
        ...     db.save()
        >>>
        >>> # Later: reopen and data persists
        >>> db = VecDB('./my_database')
        >>> collection = db.get_collection('documents')
        >>> print(collection.count())  # Still has the vectors
    """

    def __init__(self, path: str = './vecdb_data') -> None:
        self._path = path
        self._persistence = PersistenceManager(path)
        self._collections: Dict[str, Collection] = {}

        # Load existing collections from disk
        self._load_existing_collections()

    def _load_existing_collections(self) -> None:
        """Load all existing collections from disk."""
        collection_names = self._persistence.list_collections()

        for name in collection_names:
            collection = self._persistence.load_collection(name)
            if collection is not None:
                self._collections[name] = collection

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = 'l2',
        hnsw_params: Optional[Dict] = None
    ) -> Collection:
        """
        Create a new collection.

        Args:
            name: Unique collection identifier
            dimension: Fixed vector dimension for this collection
            metric: Distance metric - 'l2', 'cosine', or 'dot' (default: 'l2')
            hnsw_params: Optional dict with 'M' and 'ef_construction' parameters

        Returns:
            The newly created Collection object

        Raises:
            CollectionExistsError: If a collection with this name already exists
            ValueError: If name is empty, dimension < 1, or metric is invalid
        """
        if name in self._collections:
            raise CollectionExistsError(f"Collection '{name}' already exists")

        collection = Collection(
            name=name,
            dimension=dimension,
            metric=metric,
            hnsw_params=hnsw_params
        )

        self._collections[name] = collection
        return collection

    def get_collection(self, name: str) -> Collection:
        """
        Get an existing collection by name.

        Args:
            name: The collection name to retrieve

        Returns:
            The Collection object

        Raises:
            CollectionNotFoundError: If the collection doesn't exist
        """
        if name not in self._collections:
            raise CollectionNotFoundError(f"Collection '{name}' not found")

        return self._collections[name]

    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection and its persisted data.

        Args:
            name: The collection name to delete

        Returns:
            True if the collection was deleted, False if it didn't exist
        """
        if name not in self._collections:
            # Still try to delete from disk in case of orphaned files
            return self._persistence.delete_collection(name)

        del self._collections[name]
        self._persistence.delete_collection(name)
        return True

    def list_collections(self) -> List[str]:
        """
        List all collection names.

        Returns:
            Sorted list of collection names
        """
        return sorted(self._collections.keys())

    def save(self) -> None:
        """
        Persist all collections to disk.

        Saves all collections regardless of whether they've been modified.
        """
        # Save metadata
        self._persistence.save_metadata({
            'collections': self.list_collections(),
        })

        # Save all collections
        for collection in self._collections.values():
            self._persistence.save_collection(collection)

    def close(self) -> None:
        """
        Save and release resources.

        Persists all collections to disk.
        """
        self.save()

    def __enter__(self) -> 'VecDB':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - automatically saves on exit."""
        self.close()
        return None  # Don't suppress exceptions

    def __repr__(self) -> str:
        return f"VecDB(path='{self._path}', collections={len(self._collections)})"

    def __len__(self) -> int:
        """Return the number of collections."""
        return len(self._collections)

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator for checking collection existence."""
        return name in self._collections
