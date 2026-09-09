"""Consumer for query (RAG) tasks."""

import uuid
from typing import Any

from dishka import AsyncContainer

from entrypoints.base_consumer import BaseConsumer
from application.services.retriever import RetrieverService
from application.interfaces import TokenValidator
from domain.exceptions import PermissionDeniedError
from config import Settings


class QueryConsumer(BaseConsumer):
    """Consumer for query_queue: processes user questions."""

    def __init__(self, container: AsyncContainer) -> None:
        settings = container.get(Settings)
        super().__init__("query_queue", settings)
        self._container = container

    async def handle(self, data: dict[str, Any]) -> None:
        token = data.get("token")
        if not token:
            raise ValueError("Missing token")

        validator = await self._container.get(TokenValidator)
        payload = validator.validate(token)
        user_id = payload["sub"]

        query = data["query"]
        query_id = data.get("query_id", str(uuid.uuid4()))

        retriever = await self._container.get(RetrieverService)
        result = await retriever.answer_query(
            user_id=user_id,
            query=query,
            conversation_id=data.get("conversation_id"),
        )

        # In a real implementation, send result back via reply_queue
        # For now, just log or store.
        print(f"Result for {query_id}: {result}")