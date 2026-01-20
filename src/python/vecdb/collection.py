"""
VecDB Collection Manager

Manages the mapping between user-provided IDs and internal HNSW indices,
handles collection metadata, and validates inputs before passing to HNSW.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from vecdb.exceptions import DimensionError, DuplicateIDError

# Type alias for vectors
VectorType = Union[List[float], np.ndarray]

# Import HNSW - try C++ first, fall back to mock
try:
    from vecdb._hnsw_cpp import HNSWIndex
except ImportError:
    from vecdb._hnsw_mock import HNSWIndex


class Collection:
    """
    A named collection of vectors with a fixed dimension.

    Manages the mapping between user-provided IDs and internal HNSW indices,
    validates inputs, and provides a clean interface for vector operations.

    Args:
        name: Unique collection identifier
        dimension: Fixed vector dimension for this collection
        metric: Distance metric - 'l2', 'cosine', or 'dot'
        hnsw_params: Optional dict with 'M' and 'ef_construction' parameters
    """

    VALID_METRICS = {'l2', 'cosine', 'dot'}

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: str = 'l2',
        hnsw_params: Optional[Dict] = None
    ) -> None:
        if not name:
            raise ValueError("Collection name cannot be empty")
        if dimension < 1:
            raise ValueError(f"Dimension must be >= 1, got {dimension}")
        if metric not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric '{metric}'. Must be one of: {self.VALID_METRICS}")

        self._name = name
        self._dimension = dimension
        self._metric = metric

        # Parse HNSW parameters
        hnsw_params = hnsw_params or {}
        M = hnsw_params.get('M', 16)
        ef_construction = hnsw_params.get('ef_construction', 200)

        # Initialize HNSW index
        self._hnsw_index = HNSWIndex(
            dimension=dimension,
            metric=metric,
            M=M,
            ef_construction=ef_construction
        )

        # ID mappings
        self._user_to_internal: Dict[int, int] = {}
        self._internal_to_user: Dict[int, int] = {}

        # Store original vectors for get() retrieval
        self._vectors: Dict[int, List[float]] = {}

        # Counter for assigning internal IDs
        self._next_internal_id: int = 0

    @property
    def name(self) -> str:
        """Get the collection name."""
        return self._name

    @property
    def dimension(self) -> int:
        """Get the vector dimension for this collection."""
        return self._dimension

    @property
    def metric(self) -> str:
        """Get the distance metric for this collection."""
        return self._metric

    def insert(self, user_id: int, vector: VectorType) -> None:
        """
        Insert a vector with a user-provided ID.

        Args:
            user_id: Unique user identifier for this vector
            vector: The vector to insert (list or numpy array)

        Raises:
            DuplicateIDError: If user_id already exists in the collection
            DimensionError: If vector dimension doesn't match collection dimension
        """
        # Convert numpy array to list if needed
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        # Validate dimension
        if len(vector) != self._dimension:
            raise DimensionError(
                f"Expected dimension {self._dimension}, got {len(vector)}"
            )

        # Check for duplicate user ID
        if user_id in self._user_to_internal:
            raise DuplicateIDError(f"User ID {user_id} already exists in collection '{self._name}'")

        # Assign internal ID
        internal_id = self._next_internal_id
        self._next_internal_id += 1

        # Store mappings
        self._user_to_internal[user_id] = internal_id
        self._internal_to_user[internal_id] = user_id

        # Store original vector
        self._vectors[user_id] = list(vector)

        # Add to HNSW index
        self._hnsw_index.add(internal_id, vector)

    def search(
        self,
        query: VectorType,
        k: int = 10,
        ef_search: int = 50
    ) -> List[Tuple[int, float]]:
        """
        Search for the k nearest neighbors to a query vector.

        Args:
            query: Query vector
            k: Number of neighbors to return (default: 10)
            ef_search: Search width parameter (default: 50)

        Returns:
            List of (user_id, distance) tuples sorted by distance ascending.
            Returns fewer than k results if collection has fewer vectors.

        Raises:
            DimensionError: If query dimension doesn't match collection dimension
        """
        # Convert numpy array to list if needed
        if isinstance(query, np.ndarray):
            query = query.tolist()

        # Validate dimension
        if len(query) != self._dimension:
            raise DimensionError(
                f"Expected dimension {self._dimension}, got {len(query)}"
            )

        # Search HNSW index
        internal_results = self._hnsw_index.search(query, k, ef_search)

        # Translate internal IDs to user IDs
        user_results = [
            (self._internal_to_user[internal_id], distance)
            for internal_id, distance in internal_results
        ]

        return user_results

    def delete(self, user_id: int) -> bool:
        """
        Delete a vector by user ID.

        Args:
            user_id: The user ID to delete

        Returns:
            True if the vector was found and deleted, False otherwise
        """
        if user_id not in self._user_to_internal:
            return False

        internal_id = self._user_to_internal[user_id]

        # Remove from HNSW index
        self._hnsw_index.remove(internal_id)

        # Remove from mappings
        del self._user_to_internal[user_id]
        del self._internal_to_user[internal_id]

        # Remove stored vector
        del self._vectors[user_id]

        return True

    def get(self, user_id: int) -> Optional[List[float]]:
        """
        Retrieve a vector by user ID.

        Args:
            user_id: The user ID to retrieve

        Returns:
            The vector as a list of floats, or None if not found
        """
        vector = self._vectors.get(user_id)
        if vector is not None:
            return list(vector)  # Return a copy
        return None

    def contains(self, user_id: int) -> bool:
        """
        Check if a user ID exists in the collection.

        Args:
            user_id: The user ID to check

        Returns:
            True if the ID exists, False otherwise
        """
        return user_id in self._user_to_internal

    def count(self) -> int:
        """
        Return the number of vectors in the collection.

        Returns:
            Number of non-deleted vectors
        """
        return len(self._user_to_internal)

    def __len__(self) -> int:
        """Return the number of vectors in the collection."""
        return self.count()

    def __contains__(self, user_id: int) -> bool:
        """Support 'in' operator for checking user ID existence."""
        return self.contains(user_id)

    # Internal methods for persistence support

    def _get_hnsw_index(self) -> HNSWIndex:
        """Get the underlying HNSW index (for persistence)."""
        return self._hnsw_index

    def _get_state(self) -> dict:
        """Get collection state for persistence."""
        return {
            'name': self._name,
            'dimension': self._dimension,
            'metric': self._metric,
            'user_to_internal': self._user_to_internal,
            'internal_to_user': self._internal_to_user,
            'vectors': self._vectors,
            'next_internal_id': self._next_internal_id,
        }

    @classmethod
    def _from_state(
        cls,
        state: dict,
        hnsw_index: HNSWIndex,
        hnsw_params: Optional[Dict] = None
    ) -> 'Collection':
        """Reconstruct collection from persisted state."""
        collection = cls.__new__(cls)
        collection._name = state['name']
        collection._dimension = state['dimension']
        collection._metric = state['metric']
        collection._user_to_internal = state['user_to_internal']
        collection._internal_to_user = state['internal_to_user']
        collection._vectors = state['vectors']
        collection._next_internal_id = state['next_internal_id']
        collection._hnsw_index = hnsw_index
        return collection
