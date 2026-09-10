"""Consumer for reindexing tasks."""

from typing import Any

from dishka import AsyncContainer

from entrypoints.base_consumer import BaseConsumer
from application.services.document_manager import DocumentManager
from application.interfaces import TokenValidator
from domain.exceptions import PermissionDeniedError


class ReindexConsumer(BaseConsumer):
    """Consumer for reindex_queue: triggers full reindexing."""

    def __init__(self, container: AsyncContainer) -> None:
        super().__init__("reindex_queue", container)

    async def handle(self, data: dict[str, Any]) -> None:
        token = data.get("token")
        if not token:
            raise ValueError("Missing token")

        validator = await self._container.get(TokenValidator)
        payload = validator.validate(token)
        user_id = payload["sub"]

        dimension = data.get("vector_dimension")

        manager = await self._container.get(DocumentManager)
        await manager.reindex_all(user_id, dimension)