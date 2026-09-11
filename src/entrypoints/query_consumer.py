"""Consumer for query (RAG) tasks."""

import uuid
from typing import Any

import structlog
from dishka import AsyncContainer

from entrypoints.base_consumer import BaseConsumer
from application.services.retriever import RetrieverService
from application.interfaces import TokenValidator
from domain.exceptions import PermissionDeniedError

logger = structlog.get_logger(__name__)


class QueryConsumer(BaseConsumer):
    """Consumer for query_queue: processes user questions."""

    def __init__(self, container: AsyncContainer) -> None:
        super().__init__("query_queue", container)

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
        logger.info(f"Result for {query_id}: {result}")