"""Consumer for document deletion tasks."""

import uuid
from typing import Any

from dishka import AsyncContainer

from entrypoints.base_consumer import BaseConsumer
from application.services.document_manager import DocumentManager
from application.interfaces import TokenValidator, DocumentRepository
from domain.exceptions import PermissionDeniedError
from config import Settings


class DeleteConsumer(BaseConsumer):
    """Consumer for delete_queue: deletes a document."""

    def __init__(self, container: AsyncContainer) -> None:
        settings = container.get(Settings)
        super().__init__("delete_queue", settings)
        self._container = container

    async def handle(self, data: dict[str, Any]) -> None:
        token = data.get("token")
        if not token:
            raise ValueError("Missing token")

        validator = await self._container.get(TokenValidator)
        payload = validator.validate(token)
        user_id = payload["sub"]

        doc_id = uuid.UUID(data["doc_id"])
        remove_file = data.get("remove_file", False)

        manager = await self._container.get(DocumentManager)
        await manager.delete_document(doc_id, user_id, remove_file)
