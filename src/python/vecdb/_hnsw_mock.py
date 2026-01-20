"""
Mock HNSW Index using brute-force search.

This module provides a functional mock of the HNSW index that uses O(N) brute-force
search instead of the actual HNSW algorithm. It implements the exact same interface
as the C++ HNSW module for testing and development purposes.

The mock stores but ignores HNSW-specific parameters (M, ef_construction, ef_search)
to maintain interface compatibility.
"""

from __future__ import annotations

import math
import pickle
from typing import List, Tuple, Union

import numpy as np

from vecdb.exceptions import DimensionError, DuplicateIDError, DeserializationError


# Type alias for vectors - accept both list and numpy array
VectorType = Union[List[float], np.ndarray]


class HNSWIndex:
    """
    Mock HNSW index using brute-force search.

    Matches the real C++ HNSWIndex interface exactly for seamless swapping.

    Args:
        dimension: Vector dimensionality (fixed for this index)
        metric: Distance metric - 'l2', 'cosine', or 'dot'
        M: Max connections per node per layer (stored but unused in mock)
        ef_construction: Search width during construction (stored but unused in mock)
    """

    VALID_METRICS = {'l2', 'cosine', 'dot'}

    def __init__(
        self,
        dimension: int,
        metric: str,
        M: int = 16,
        ef_construction: int = 200
    ) -> None:
        if dimension < 1:
            raise ValueError(f"Dimension must be >= 1, got {dimension}")
        if metric not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric '{metric}'. Must be one of: {self.VALID_METRICS}")

        self.dimension = dimension
        self.metric = metric
        self.M = M  # Stored but unused in mock
        self.ef_construction = ef_construction  # Stored but unused in mock

        # Internal storage
        self._vectors: dict[int, List[float]] = {}
        self._deleted: set[int] = set()

    def add(self, internal_id: int, vector: VectorType) -> None:
        """
        Add vector with given internal ID.

        Args:
            internal_id: Unique identifier for this vector
            vector: The vector to add (list or numpy array)

        Raises:
            DimensionError: If vector dimension doesn't match index dimension
            DuplicateIDError: If internal_id already exists and is not deleted
        """
        # Convert numpy array to list if needed
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        if len(vector) != self.dimension:
            raise DimensionError(f"Expected dimension {self.dimension}, got {len(vector)}")

        if internal_id in self._vectors and internal_id not in self._deleted:
            raise DuplicateIDError(f"ID {internal_id} already exists")

        self._vectors[internal_id] = list(vector)  # Store a copy
        self._deleted.discard(internal_id)

    def search(
        self,
        query: VectorType,
        k: int,
        ef_search: int = 50
    ) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors.

        Args:
            query: Query vector
            k: Number of neighbors to return
            ef_search: Search width (stored but unused in mock)

        Returns:
            List of (internal_id, distance) sorted by distance ascending.
            Returns fewer than k results if index has fewer vectors.

        Raises:
            DimensionError: If query dimension doesn't match index dimension
            ValueError: If k < 1
        """
        # Convert numpy array to list if needed
        if isinstance(query, np.ndarray):
            query = query.tolist()

        if len(query) != self.dimension:
            raise DimensionError(f"Expected dimension {self.dimension}, got {len(query)}")

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        # Return empty list for empty index
        if len(self) == 0:
            return []

        # Brute-force: compute all distances
        results = []
        for internal_id, vec in self._vectors.items():
            if internal_id not in self._deleted:
                dist = self._compute_distance(query, vec)
                results.append((internal_id, dist))

        # Sort by distance ascending
        results.sort(key=lambda x: x[1])

        # Return up to k results
        return results[:k]

    def remove(self, internal_id: int) -> bool:
        """
        Mark vector as deleted (lazy deletion).

        Args:
            internal_id: The ID to remove

        Returns:
            True if the ID was found and removed, False otherwise
        """
        if internal_id in self._vectors and internal_id not in self._deleted:
            self._deleted.add(internal_id)
            return True
        return False

    def serialize(self) -> bytes:
        """
        Serialize entire index to bytes for persistence.

        Returns:
            Binary representation of the index
        """
        state = {
            'version': 1,
            'dimension': self.dimension,
            'metric': self.metric,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'vectors': self._vectors,
            'deleted': self._deleted,
        }
        return pickle.dumps(state)

    @staticmethod
    def deserialize(data: bytes) -> 'HNSWIndex':
        """
        Reconstruct index from serialized bytes.

        Args:
            data: Binary data from serialize()

        Returns:
            Reconstructed HNSWIndex

        Raises:
            DeserializationError: If data is corrupt or incompatible
        """
        try:
            state = pickle.loads(data)

            # Validate required fields
            required_fields = {'dimension', 'metric', 'vectors', 'deleted'}
            if not required_fields.issubset(state.keys()):
                raise DeserializationError("Missing required fields in serialized data")

            index = HNSWIndex(
                dimension=state['dimension'],
                metric=state['metric'],
                M=state.get('M', 16),
                ef_construction=state.get('ef_construction', 200)
            )
            index._vectors = state['vectors']
            index._deleted = state['deleted']

            return index

        except pickle.UnpicklingError as e:
            raise DeserializationError(f"Failed to deserialize HNSW index: {e}")
        except (KeyError, TypeError) as e:
            raise DeserializationError(f"Invalid serialized data format: {e}")

    def __len__(self) -> int:
        """Return count of non-deleted vectors."""
        return len(self._vectors) - len(self._deleted)

    def _compute_distance(self, a: List[float], b: List[float]) -> float:
        """
        Compute distance between two vectors based on configured metric.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Distance value (lower is more similar for all metrics)
        """
        if self.metric == 'l2':
            return self._l2_distance(a, b)
        elif self.metric == 'cosine':
            return self._cosine_distance(a, b)
        elif self.metric == 'dot':
            return self._dot_distance(a, b)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    @staticmethod
    def _l2_distance(a: List[float], b: List[float]) -> float:
        """Compute Euclidean (L2) distance."""
        sum_sq = 0.0
        for i in range(len(a)):
            diff = a[i] - b[i]
            sum_sq += diff * diff
        return math.sqrt(sum_sq)

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        """
        Compute cosine distance (1 - cosine_similarity).

        Returns value in [0, 2] where 0 means identical direction.
        """
        dot_product = 0.0
        norm_a = 0.0
        norm_b = 0.0

        for i in range(len(a)):
            dot_product += a[i] * b[i]
            norm_a += a[i] * a[i]
            norm_b += b[i] * b[i]

        norm_a = math.sqrt(norm_a)
        norm_b = math.sqrt(norm_b)

        # Handle zero vectors
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0  # Undefined, return neutral distance

        cosine_sim = dot_product / (norm_a * norm_b)

        # Clamp to [-1, 1] to handle floating point errors
        cosine_sim = max(-1.0, min(1.0, cosine_sim))

        return 1.0 - cosine_sim

    @staticmethod
    def _dot_distance(a: List[float], b: List[float]) -> float:
        """
        Compute negative dot product distance.

        For Maximum Inner Product Search (MIPS), we want maximum dot product.
        Since HNSW minimizes distance, we return -dot_product.
        """
        dot_product = 0.0
        for i in range(len(a)):
            dot_product += a[i] * b[i]
        return -dot_product
