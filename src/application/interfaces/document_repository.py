"""Interface for document metadata persistence."""

import uuid
from typing import Protocol, Any, runtime_checkable

from domain.entities.document import Document, DocStatus


@runtime_checkable
class DocumentRepository(Protocol):
    """Abstract interface for document metadata storage."""

    async def get_document(self, doc_id: uuid.UUID) -> Document | None:
        """Retrieve a document by its ID."""
        ...

    async def get_all_active(self) -> list[Document]:
        """Get all non-deleted documents."""
        ...

    async def save(self, document: Document) -> None:
        """Persist a document record."""
        ...

    async def update_status(
        self, doc_id: uuid.UUID, status: DocStatus
    ) -> None:
        """Update the indexing status of a document."""
        ...

    async def mark_deleted(self, doc_id: uuid.UUID) -> None:
        """Soft-delete a document."""
        ...

    async def get_user_groups(self, user_id: str) -> list[str]:
        """Retrieve access groups for a given user from the DB."""
        ...

    async def set_user_groups(self, user_id: str, groups: list[str]) -> None:
        """Replace the set of access groups for a given user."""
        ...

    async def save_conversation(
        self,
        user_id: str,
        query: str,
        response: str,
        sources: list[dict[str, Any]],
    ) -> None:
        """Store a conversation record."""
        ...
