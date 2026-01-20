"""
VecDB Persistence Manager

Handles serialization and deserialization of all database state to disk,
enabling restart recovery.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from vecdb.exceptions import DeserializationError

if TYPE_CHECKING:
    from vecdb.collection import Collection

# Import HNSW - try C++ first, fall back to mock
try:
    from vecdb._hnsw_cpp import HNSWIndex
except ImportError:
    from vecdb._hnsw_mock import HNSWIndex


# File format version for compatibility checking
FILE_FORMAT_VERSION = 1


class PersistenceManager:
    """
    Manages persistence of VecDB collections to disk.

    File structure:
        db_path/
        ├── metadata.json          # Database-level metadata
        └── collections/
            ├── {name}.hnsw       # Serialized HNSW index (binary)
            ├── {name}.meta       # Collection metadata (JSON)
            └── {name}.vectors    # Original vectors + ID mappings (binary)

    Args:
        db_path: Directory path for database storage.
                 Creates the directory if it doesn't exist.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.collections_path = self.db_path / "collections"

        # Create directories if they don't exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collections_path.mkdir(parents=True, exist_ok=True)

    def save_collection(self, collection: 'Collection') -> None:
        """
        Save a single collection to disk.

        Uses atomic writes (write to .tmp, then rename) for safety.

        Args:
            collection: The Collection object to save
        """
        name = collection.name
        state = collection._get_state()
        hnsw_index = collection._get_hnsw_index()

        # Save HNSW index (binary)
        hnsw_path = self.collections_path / f"{name}.hnsw"
        hnsw_data = hnsw_index.serialize()
        self._atomic_write(hnsw_path, hnsw_data, binary=True)

        # Save collection metadata (JSON)
        meta_path = self.collections_path / f"{name}.meta"
        meta = {
            'version': FILE_FORMAT_VERSION,
            'name': state['name'],
            'dimension': state['dimension'],
            'metric': state['metric'],
            'count': len(state['vectors']),
            'next_internal_id': state['next_internal_id'],
        }
        self._atomic_write(meta_path, json.dumps(meta, indent=2).encode('utf-8'), binary=True)

        # Save vectors and ID mappings (binary)
        vectors_path = self.collections_path / f"{name}.vectors"
        vectors_data = self._serialize_vectors(state)
        self._atomic_write(vectors_path, vectors_data, binary=True)

    def load_collection(self, name: str) -> Optional['Collection']:
        """
        Load a collection by name.

        Args:
            name: The collection name to load

        Returns:
            The loaded Collection object, or None if not found

        Raises:
            DeserializationError: If files are corrupt or incompatible
        """
        from vecdb.collection import Collection

        meta_path = self.collections_path / f"{name}.meta"
        hnsw_path = self.collections_path / f"{name}.hnsw"
        vectors_path = self.collections_path / f"{name}.vectors"

        # Check if all files exist
        if not all(p.exists() for p in [meta_path, hnsw_path, vectors_path]):
            return None

        try:
            # Load metadata
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            # Check version
            if meta.get('version', 0) > FILE_FORMAT_VERSION:
                raise DeserializationError(
                    f"File format version {meta.get('version')} is newer than supported {FILE_FORMAT_VERSION}"
                )

            # Load HNSW index
            with open(hnsw_path, 'rb') as f:
                hnsw_data = f.read()
            hnsw_index = HNSWIndex.deserialize(hnsw_data)

            # Load vectors and mappings
            with open(vectors_path, 'rb') as f:
                vectors_data = f.read()
            vectors_state = self._deserialize_vectors(vectors_data, meta['dimension'])

            # Reconstruct collection state
            state = {
                'name': meta['name'],
                'dimension': meta['dimension'],
                'metric': meta['metric'],
                'user_to_internal': vectors_state['user_to_internal'],
                'internal_to_user': vectors_state['internal_to_user'],
                'vectors': vectors_state['vectors'],
                'next_internal_id': meta['next_internal_id'],
            }

            return Collection._from_state(state, hnsw_index)

        except json.JSONDecodeError as e:
            raise DeserializationError(f"Corrupt metadata file for collection '{name}': {e}")
        except (KeyError, TypeError) as e:
            raise DeserializationError(f"Invalid data format for collection '{name}': {e}")
        except struct.error as e:
            raise DeserializationError(f"Corrupt vectors file for collection '{name}': {e}")
        except Exception as e:
            # Catch C++ DeserializationError and convert to Python exception
            if 'DeserializationError' in type(e).__name__ or 'Unsupported' in str(e):
                raise DeserializationError(f"Corrupt HNSW file for collection '{name}': {e}")
            raise

    def delete_collection(self, name: str) -> bool:
        """
        Delete collection files from disk.

        Args:
            name: The collection name to delete

        Returns:
            True if files were deleted, False if collection didn't exist
        """
        meta_path = self.collections_path / f"{name}.meta"
        hnsw_path = self.collections_path / f"{name}.hnsw"
        vectors_path = self.collections_path / f"{name}.vectors"

        deleted_any = False
        for path in [meta_path, hnsw_path, vectors_path]:
            if path.exists():
                path.unlink()
                deleted_any = True

        return deleted_any

    def list_collections(self) -> List[str]:
        """
        List all persisted collection names.

        Returns:
            List of collection names found on disk
        """
        collections = set()

        if self.collections_path.exists():
            for path in self.collections_path.iterdir():
                if path.suffix == '.meta':
                    collections.add(path.stem)

        return sorted(collections)

    def save_metadata(self, metadata: dict) -> None:
        """
        Save database-level metadata.

        Args:
            metadata: Dictionary of metadata to save
        """
        meta_path = self.db_path / "metadata.json"
        data = {
            'version': FILE_FORMAT_VERSION,
            **metadata
        }
        self._atomic_write(meta_path, json.dumps(data, indent=2).encode('utf-8'), binary=True)

    def load_metadata(self) -> dict:
        """
        Load database-level metadata.

        Returns:
            Dictionary of metadata, empty dict if file doesn't exist
        """
        meta_path = self.db_path / "metadata.json"

        if not meta_path.exists():
            return {}

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _atomic_write(self, path: Path, data: bytes, binary: bool = True) -> None:
        """
        Write data to file atomically using temp file + rename.

        Args:
            path: Target file path
            data: Data to write
            binary: Whether to write in binary mode
        """
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        mode = 'wb' if binary else 'w'

        with open(tmp_path, mode) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        tmp_path.rename(path)

    def _serialize_vectors(self, state: dict) -> bytes:
        """
        Serialize vectors and ID mappings to binary format.

        Format:
            Header: [version: uint32][count: uint64][dimension: uint32]
            Per vector: [user_id: uint64][internal_id: uint64][floats: float32 * dim]

        Args:
            state: Collection state dict

        Returns:
            Binary serialized data
        """
        vectors = state['vectors']
        user_to_internal = state['user_to_internal']
        dimension = state['dimension']

        # Header
        data = struct.pack('<IQI', FILE_FORMAT_VERSION, len(vectors), dimension)

        # Per-vector data
        for user_id, vector in vectors.items():
            internal_id = user_to_internal[user_id]
            # Pack user_id, internal_id, then vector floats
            data += struct.pack('<QQ', user_id, internal_id)
            data += struct.pack(f'<{dimension}f', *vector)

        return data

    def _deserialize_vectors(self, data: bytes, expected_dimension: int) -> dict:
        """
        Deserialize vectors and ID mappings from binary format.

        Args:
            data: Binary data from _serialize_vectors
            expected_dimension: Expected vector dimension for validation

        Returns:
            Dict with 'vectors', 'user_to_internal', 'internal_to_user'

        Raises:
            DeserializationError: If data is corrupt or dimension mismatch
        """
        if len(data) < 16:  # Minimum header size
            raise DeserializationError("Vectors file too small")

        # Read header
        version, count, dimension = struct.unpack('<IQI', data[:16])

        if version > FILE_FORMAT_VERSION:
            raise DeserializationError(
                f"Vectors file version {version} is newer than supported {FILE_FORMAT_VERSION}"
            )

        if dimension != expected_dimension:
            raise DeserializationError(
                f"Dimension mismatch: file has {dimension}, expected {expected_dimension}"
            )

        # Calculate expected size
        vector_size = 16 + (dimension * 4)  # 2 uint64 + dimension floats
        expected_size = 16 + (count * vector_size)

        if len(data) < expected_size:
            raise DeserializationError(
                f"Vectors file truncated: expected {expected_size} bytes, got {len(data)}"
            )

        # Read vectors
        vectors: Dict[int, List[float]] = {}
        user_to_internal: Dict[int, int] = {}
        internal_to_user: Dict[int, int] = {}

        offset = 16
        for _ in range(count):
            user_id, internal_id = struct.unpack('<QQ', data[offset:offset + 16])
            offset += 16

            vector = list(struct.unpack(f'<{dimension}f', data[offset:offset + dimension * 4]))
            offset += dimension * 4

            vectors[user_id] = vector
            user_to_internal[user_id] = internal_id
            internal_to_user[internal_id] = user_id

        return {
            'vectors': vectors,
            'user_to_internal': user_to_internal,
            'internal_to_user': internal_to_user,
        }
