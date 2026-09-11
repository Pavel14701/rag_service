"""Document management service for deletion and reindexing."""

import logging
import uuid
from typing import Optional

import structlog

from domain.exceptions import DocumentNotFoundError, PermissionDeniedError
from application.interfaces import FileStorage, VectorStore, DocumentRepository
from application.services.indexer import IndexerService
from application.interfaces import EmbeddingModel

logger = structlog.get_logger(__name__)


class DocumentManager:
    """Service for managing document lifecycle."""

    def __init__(
        self,
        file_storage: FileStorage,
        vector_store: VectorStore,
        repo: DocumentRepository,
        indexer: IndexerService,
        embedding_model: EmbeddingModel,
        embedding_model_name: str | None = None,
    ) -> None:
        self._file_storage = file_storage
        self._vector_store = vector_store
        self._repo = repo
        self._indexer = indexer
        self._embedding_model = embedding_model
        # Current embedding model version (config marker).
        self._embedding_model_name = embedding_model_name

    async def delete_document(
        self,
        doc_id: uuid.UUID,
        user_id: str,
        remove_file: bool = False,
    ) -> None:
        """
        Delete a document (soft delete) and remove its vectors.

        Args:
            doc_id: Document ID.
            user_id: ID of the user requesting deletion.
            remove_file: If True, also delete the physical file from storage.

        Raises:
            DocumentNotFoundError: If document not found.
            PermissionDeniedError: If user is not the owner.
        """
        doc = await self._repo.get_document(doc_id)
        if doc is None or doc.deleted:
            raise DocumentNotFoundError(f"Document {doc_id} not found")
        if doc.owner_id != user_id:
            raise PermissionDeniedError("User is not the owner of this document")

        await self._vector_store.delete_by_filter(
            {"key": "doc_id", "match": {"value": str(doc_id)}}
        )
        if remove_file:
            await self._file_storage.delete_file(doc.file_path)
        await self._repo.mark_deleted(doc_id)

    async def reindex_all(
        self,
        admin_user_id: str,
        vector_dimension: Optional[int] = None,
    ) -> None:
        """
        Re-index all active documents.

        This is an admin operation. It clears the vector collection,
        recreates it, and re-indexes every active document.

        Args:
            admin_user_id: ID of the admin user.
            vector_dimension: Dimension of embeddings. If not provided, obtained from model.

        Raises:
            PermissionDeniedError: If user is not an admin.
        """
        is_admin = await self._is_admin(admin_user_id)
        if not is_admin:
            raise PermissionDeniedError("User is not an administrator")

        stored_version, expected_version = await self.check_embedding_version()
        if stored_version and stored_version != expected_version:
            logger.warning(
                "embedding_model_migration_reindex",
                stored=stored_version,
                expected=expected_version,
            )
        elif stored_version:
            logger.info("reindex_same_model_refresh", model=stored_version)

        if vector_dimension is None:
            vector_dimension = self._embedding_model.dimension()

        await self._vector_store.drop_collection()
        await self._vector_store.create_collection(vector_dimension)

        active_docs = await self._repo.get_all_active()
        for doc in active_docs:
            try:
                await self._indexer.index_document(doc.id)
            except Exception as e:
                logger.error(f"Failed to re-index document {doc.id}: {e}")

    async def check_embedding_version(self) -> tuple[str | None, str | None]:
        """Compare the indexed model version marker with the configured one.

        Returns:
            (stored_version, expected_version): ``stored_version`` is the
            marker read from an indexed point (None when the collection
            is empty or unreadable); ``expected_version`` is the model
            currently configured. When both are non-None and differ, a
            migration reindex is required.
        """
        payload = await self._vector_store.scroll_first_payload()
        stored = payload.get("embedding_model") if payload else None
        return stored, self._embedding_model_name

    async def needs_reindex(self) -> bool:
        """True when indexed vectors were built by a different model."""
        stored, expected = await self.check_embedding_version()
        return stored is not None and expected is not None and stored != expected

    async def _is_admin(self, user_id: str) -> bool:
        """Check if user is admin (placeholder)."""
        groups = await self._repo.get_user_groups(user_id)
        return "admin" in groups or user_id.startswith("admin_")