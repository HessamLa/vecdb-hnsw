"""
VecDB Exception Classes

All custom exceptions inherit from VecDBError for easy catching.
"""


class VecDBError(Exception):
    """Base exception for all VecDB errors."""

    def __init__(self, message: str = "A VecDB error occurred"):
        self.message = message
        super().__init__(self.message)


class DimensionError(VecDBError):
    """Raised when vector dimension doesn't match expected dimension."""

    def __init__(self, message: str = "Vector dimension mismatch"):
        super().__init__(message)


class DuplicateIDError(VecDBError):
    """Raised when attempting to insert a vector with an existing ID."""

    def __init__(self, message: str = "ID already exists"):
        super().__init__(message)


class CollectionExistsError(VecDBError):
    """Raised when attempting to create a collection that already exists."""

    def __init__(self, message: str = "Collection already exists"):
        super().__init__(message)


class CollectionNotFoundError(VecDBError):
    """Raised when attempting to access a collection that doesn't exist."""

    def __init__(self, message: str = "Collection not found"):
        super().__init__(message)


class DeserializationError(VecDBError):
    """Raised when deserialization of stored data fails."""

    def __init__(self, message: str = "Failed to deserialize data"):
        super().__init__(message)
