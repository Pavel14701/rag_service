"""Consumer for query (RAG) tasks."""

import json
import uuid
from typing import Any

import aio_pika
import structlog
from aio_pika import DeliveryMode
from dishka import AsyncContainer

from entrypoints.base_consumer import BaseConsumer
from application.services.retriever import RetrieverService
from application.interfaces import TokenValidator
from domain.exceptions import PermissionDeniedError

logger = structlog.get_logger(__name__)


class QueryConsumer(BaseConsumer):
    """Consumer for query_queue: processes user questions.

    The RAG result is sent back to the producer: either to the
    ``reply_to`` queue carried by the incoming message (RPC pattern,
    recommended) or to the shared ``query_reply_queue`` when the
    producer did not request a dedicated reply queue. The
    ``correlation_id`` property of the incoming message is propagated
    to the reply so the producer can match request/response pairs.
    """

    def __init__(
        self, container: AsyncContainer, reply_queue: str = "query_reply_queue"
    ) -> None:
        super().__init__("query_queue", container)
        self._reply_queue = reply_queue

    async def _declare_extra_topology(
        self, channel: aio_pika.abc.AbstractChannel
    ) -> None:
        # Ensure the fallback reply queue exists for producers that do
        # not use per-request reply_to queues.
        await channel.declare_queue(self._reply_queue, durable=True)

    async def handle(self, data: dict[str, Any]) -> None:
        token = data.get("token")
        if not token:
            raise ValueError("Missing token")

        validator = await self._container.get(TokenValidator)
        payload = validator.validate(token)
        user_id = payload["sub"]
        # Access groups may be embedded in the token (e.g. "groups" claim);
        # when absent, the retriever falls back to the DB membership table.
        user_groups = payload.get("groups")

        query = data["query"]
        query_id = data.get("query_id", str(uuid.uuid4()))

        retriever = await self._container.get(RetrieverService)
        result = await retriever.answer_query(
            user_id=user_id,
            query=query,
            conversation_id=data.get("conversation_id"),
            user_groups=user_groups,
        )

        await self._send_reply(query_id, result)
        logger.info(
            "query_processed",
            query_id=query_id,
            user_id=user_id,
            conversation_id=result.get("conversation_id"),
        )

    async def _send_reply(self, query_id: str, result: dict[str, Any]) -> None:
        """Publish the result to reply_to / the shared reply queue."""
        message = self._current_message
        reply_to = getattr(message, "reply_to", None) if message else None
        correlation_id = getattr(message, "correlation_id", None) if message else None
        routing_key = reply_to or self._reply_queue

        reply_body = json.dumps(
            {
                "query_id": query_id,
                "conversation_id": result.get("conversation_id"),
                "answer": result["answer"],
                "sources": result["sources"],
            },
            ensure_ascii=False,
        ).encode()

        try:
            if message is None:
                # handle() called directly (tests / programmatic use):
                # publish through the queue channel is unavailable, log only.
                logger.warning(
                    "reply_skipped_no_message_context",
                    query_id=query_id,
                    routing_key=routing_key,
                )
                return
            await message.channel.default_exchange.publish(
                aio_pika.Message(
                    body=reply_body,
                    correlation_id=correlation_id,
                    delivery_mode=DeliveryMode.PERSISTENT,
                ),
                routing_key=routing_key,
            )
            logger.info("reply_published", query_id=query_id, routing_key=routing_key)
        except Exception:
            # The answer is already persisted in the conversation history;
            # a failed reply publication must not fail the whole message
            # (it would be retried and answered twice).
            logger.exception("reply_publish_failed", query_id=query_id)