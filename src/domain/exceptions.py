"""Domain-specific exceptions."""


class DomainError(Exception):
    """Base class for domain errors."""


class DocumentNotFoundError(DomainError):
    """Raised when a document is not found."""


class PermissionDeniedError(DomainError):
    """Raised when user lacks permissions."""


class IndexingError(DomainError):
    """Raised when document indexing fails."""
