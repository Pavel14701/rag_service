"""Consumer for document indexing tasks."""

import uuid
from typing import Any

from dishka import AsyncContainer

from entrypoints.base_consumer import BaseConsumer
from application.services.indexer import IndexerService
from application.interfaces import TokenValidator, DocumentRepository
from domain.exceptions import PermissionDeniedError


class IngestConsumer(BaseConsumer):
    """Consumer for ingest_queue: indexes a document."""

    def __init__(self, container: AsyncContainer) -> None:
        super().__init__("ingest_queue", container)

    async def handle(self, data: dict[str, Any]) -> None:
        token = data.get("token")
        if not token:
            raise ValueError("Missing token")

        validator = await self._container.get(TokenValidator)
        payload = validator.validate(token)
        user_id = payload["sub"]

        doc_id = uuid.UUID(data["doc_id"])
        repo = await self._container.get(DocumentRepository)
        doc = await repo.get_document(doc_id)
        if doc is None or doc.owner_id != user_id:
            raise PermissionDeniedError("User is not owner of this document")

        indexer = await self._container.get(IndexerService)
        await indexer.index_document(doc_id)
