"""Base consumer class with RabbitMQ connection.

Implements reliable delivery:
- Each queue gets a Dead Letter Exchange (``<queue>.dlx``) with a DLQ
  (``<queue>.dlq``); messages rejected as poison or failed after all
  retries land there instead of being lost.
- Failed messages are retried with exponential backoff via per-delay
  TTL queues (``<queue>.retry.<ms>``) that dead-letter back to the
  main queue.
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

import aio_pika
import structlog
from aio_pika import DeliveryMode, IncomingMessage
from dishka import AsyncContainer
from opentelemetry.trace import Status, StatusCode

from config import Settings
from shared.metrics import MESSAGE_PROCESSING_SECONDS, MESSAGES_TOTAL
from shared.tracing import get_tracer

logger = structlog.get_logger(__name__)

# Exponential backoff delays for message retries (in milliseconds).
RETRY_DELAYS_MS: tuple[int, ...] = (5_000, 30_000, 120_000)

RETRY_COUNT_HEADER = "x-retry-count"
REQUEST_ID_HEADER = "x-request-id"


class BaseConsumer(ABC):
    """Abstract base for a RabbitMQ consumer.

    Settings are resolved from the DI container lazily in ``start()``,
    because ``AsyncContainer.get`` is a coroutine and cannot be awaited
    in a synchronous ``__init__``.
    """

    def __init__(self, queue_name: str, container: AsyncContainer) -> None:
        self._queue_name = queue_name
        self._container = container

    async def start(self) -> None:
        """Start consuming messages from the queue."""
        settings = await self._container.get(Settings)
        connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        async with connection:
            channel = await connection.channel()
            await self._declare_topology(channel)
            queue = await channel.declare_queue(
                self._queue_name,
                durable=True,
                arguments={
                    # Rejected (nack without requeue) messages go to the DLQ.
                    "x-dead-letter-exchange": f"{self._queue_name}.dlx",
                    "x-dead-letter-routing-key": self._queue_name,
                },
            )
            await queue.consume(self._on_message)
            await asyncio.Future()

    async def _declare_topology(self, channel: aio_pika.abc.AbstractChannel) -> None:
        """Declare DLX/DLQ and retry TTL queues for this consumer's queue."""
        dlx = await channel.declare_exchange(
            f"{self._queue_name}.dlx",
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        dlq = await channel.declare_queue(f"{self._queue_name}.dlq", durable=True)
        await dlq.bind(dlx, routing_key=self._queue_name)

        retry_exchange = await channel.declare_exchange(
            f"{self._queue_name}.retry",
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        for delay in RETRY_DELAYS_MS:
            retry_queue = await channel.declare_queue(
                f"{self._queue_name}.retry.{delay}",
                durable=True,
                arguments={
                    # After the TTL expires, the message returns to the main queue.
                    "x-dead-letter-exchange": "",
                    "x-dead-letter-routing-key": self._queue_name,
                    "x-message-ttl": delay,
                },
            )
            await retry_queue.bind(retry_exchange, routing_key=str(delay))

    async def _on_message(self, message: IncomingMessage) -> None:
        headers = dict(message.headers or {})
        request_id = str(headers.get(REQUEST_ID_HEADER) or uuid.uuid4())

        structlog.contextvars.bind_contextvars(
            request_id=request_id, queue=self._queue_name
        )
        start = time.perf_counter()
        try:
            with get_tracer().start_as_current_span("message.process") as span:
                span.set_attribute("mq.queue", self._queue_name)
                span.set_attribute("request.id", request_id)
                async with message.process(ignore_processed=True):
                    try:
                        body = json.loads(message.body.decode())
                        await self.handle(body)
                        MESSAGES_TOTAL.labels(
                            queue=self._queue_name, status="success"
                        ).inc()
                    except Exception as e:
                        logger.exception(
                            "message_processing_failed",
                            error=str(e),
                            queue=self._queue_name,
                        )
                        MESSAGES_TOTAL.labels(
                            queue=self._queue_name, status="error"
                        ).inc()
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        await self._retry_or_dead_letter(message, request_id)
        finally:
            MESSAGE_PROCESSING_SECONDS.labels(queue=self._queue_name).observe(
                time.perf_counter() - start
            )
            structlog.contextvars.clear_contextvars()

    async def _retry_or_dead_letter(
        self, message: IncomingMessage, request_id: str
    ) -> None:
        """Schedule a retry with exponential backoff or move to the DLQ.

        The original message is acknowledged; its body is republished
        either to a retry queue (which dead-letters back to the main
        queue after the TTL) or, when retries are exhausted, to the DLQ.
        The ``request_id`` is propagated so logs stay traceable across
        retries.
        """
        headers = dict(message.headers or {})
        headers[REQUEST_ID_HEADER] = request_id
        attempt = int(headers.get(RETRY_COUNT_HEADER, 0))
        try:
            if attempt < len(RETRY_DELAYS_MS):
                delay = RETRY_DELAYS_MS[attempt]
                headers[RETRY_COUNT_HEADER] = attempt + 1
                await message.channel.default_exchange.publish(
                    aio_pika.Message(
                        body=message.body,
                        headers=headers,
                        delivery_mode=DeliveryMode.PERSISTENT,
                    ),
                    routing_key=f"{self._queue_name}.retry.{delay}",
                )
                MESSAGES_TOTAL.labels(queue=self._queue_name, status="retry").inc()
                logger.warning(
                    "message_retry_scheduled",
                    attempt=attempt + 1,
                    max_retries=len(RETRY_DELAYS_MS),
                    delay_ms=delay,
                    queue=self._queue_name,
                )
            else:
                await message.channel.default_exchange.publish(
                    aio_pika.Message(
                        body=message.body,
                        headers=headers,
                        delivery_mode=DeliveryMode.PERSISTENT,
                    ),
                    routing_key=f"{self._queue_name}.dlq",
                )
                MESSAGES_TOTAL.labels(queue=self._queue_name, status="dlq").inc()
                logger.error(
                    "message_moved_to_dlq",
                    queue=self._queue_name,
                    retries=attempt,
                )
        except Exception:
            logger.exception("retry_publish_failed", queue=self._queue_name)

    @abstractmethod
    async def handle(self, data: dict[str, Any]) -> None:
        """Process a message payload."""
        ...