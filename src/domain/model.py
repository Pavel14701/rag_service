"""Domain model: entities, value objects and domain exceptions."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class DocStatus(str, Enum):
    """Possible statuses of a document."""

    PENDING = 'pending'
    INDEXED = 'indexed'
    FAILED = 'failed'
    FAILED_INVALID = 'failed_invalid'


@dataclass
class Document:
    """Domain entity representing a document."""

    id: uuid.UUID
    file_name: str
    file_path: str
    file_hash: str
    owner_id: str
    access_group: str | None
    uploaded_at: datetime
    status: DocStatus
    deleted: bool = False


@dataclass
class Conversation:
    """User conversation record."""

    id: uuid.UUID
    user_id: str
    query: str
    response: str
    sources: list[dict[str, Any]]
    created_at: datetime


class DomainError(Exception):
    """Base class for domain errors."""


class TransientError(DomainError):
    """Retryable failure (infrastructure hiccup, concurrent work)."""


class PermanentError(DomainError):
    """Non-retryable failure (invalid input, poison message)."""


class DocumentNotFoundError(DomainError):
    """Raised when a document is not found."""


class PermissionDeniedError(DomainError):
    """Raised when user lacks permissions."""


class IndexingError(TransientError):
    """Raised when document indexing fails (retryable)."""


class IndexLockedError(TransientError):
    """Raised when another worker holds the indexing lock for a document.

    The message must be retried later, not failed permanently.
    """


class PermanentIndexingError(PermanentError, IndexingError):
    """Raised when indexing fails due to invalid input (no retry).

    Example: a binary file renamed to ``.pdf`` will never parse.
    """


class ParseTimeoutError(DomainError):
    """A synchronous parse exceeded its hard time budget.

    Raised by isolated parse runners after the child process tree
    was killed; services map it to a permanent indexing failure.
    """
