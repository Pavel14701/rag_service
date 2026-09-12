"""RabbitMQ consumers: base plumbing and the four worker handlers.

Reliable delivery in the base consumer:
- Each queue gets a Dead Letter Exchange (``<queue>.dlx``) with a DLQ
  (``<queue>.dlq``); messages rejected as poison or failed after all
  retries land there instead of being lost.
- Failed messages are retried with exponential backoff via per-delay
  TTL queues (``<queue>.retry.<ms>``) that dead-letter back to the
  main queue.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, cast

import aio_pika
import orjson
import structlog
from aio_pika import DeliveryMode
from dishka import AsyncContainer
from opentelemetry.trace import Status, StatusCode

from application.interfaces import DocumentRepository, TokenValidator
from application.services import (
    DocumentManager,
    IndexerService,
    RetrieverService,
)
from config import Settings
from domain.model import PermanentError, PermissionDeniedError
from infrastructure.caching import RedisCache, SharedCaches, TTLCache
from infrastructure.observability import (
    MESSAGES_TOTAL,
    MESSAGE_PROCESSING_SECONDS,
    get_tracer,
)

logger = structlog.get_logger(__name__)

# Exponential backoff delays for message retries (in milliseconds).
RETRY_DELAYS_MS: tuple[int, ...] = (5_000, 30_000, 120_000)

RETRY_COUNT_HEADER = 'x-retry-count'
REQUEST_ID_HEADER = 'x-request-id'


class BaseConsumer(ABC):
    """Abstract base for a RabbitMQ consumer.

    Settings are resolved from the DI container lazily in ``start()``,
    because ``AsyncContainer.get`` is a coroutine and cannot be awaited
    in a synchronous ``__init__``.
    """

    def __init__(self, queue_name: str, container: AsyncContainer) -> None:
        self._queue_name = queue_name
        self._container = container
        # Currently processed message (reply_to / correlation_id context
        # for handlers that respond to the producer).
        self._current_message: aio_pika.abc.AbstractIncomingMessage | None = (
            None
        )

    async def start(self) -> None:
        """Start consuming messages from the queue."""
        settings = await self._container.get(Settings)
        connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        async with connection:
            channel = await connection.channel()
            await self._declare_topology(channel, settings)
            await self._declare_extra_topology(channel)
            queue = await channel.declare_queue(
                self._queue_name,
                durable=True,
                arguments=self._main_queue_arguments(settings.queue_max_length)
                | {
                    # Rejected (nack without requeue) messages go to the DLQ.
                    'x-dead-letter-exchange': f'{self._queue_name}.dlx',
                    'x-dead-letter-routing-key': self._queue_name,
                },
            )
            await queue.consume(self._on_message)
            await asyncio.Future()

    @staticmethod
    def _main_queue_arguments(max_length: int) -> dict[str, Any]:
        """Main-queue arguments: DLX wiring + optional length cap.

        ``max_length > 0`` enables ``reject-publish`` overflow: RabbitMQ
        rejects new messages when the queue is full instead of growing
        without bound while consumers are down.
        """
        arguments: dict[str, Any] = {
            'x-dead-letter-exchange': '',
        }
        if max_length > 0:
            arguments['x-max-length'] = max_length
            arguments['x-overflow'] = 'reject-publish'
        return arguments

    async def _declare_topology(
        self,
        channel: aio_pika.abc.AbstractChannel,
        settings: Settings,
    ) -> None:
        """Declare DLX/DLQ and retry TTL queues for this consumer's queue."""
        dlx = await channel.declare_exchange(
            f'{self._queue_name}.dlx',
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        dlq = await channel.declare_queue(
            f'{self._queue_name}.dlq', durable=True
        )
        await dlq.bind(dlx, routing_key=self._queue_name)

        retry_exchange = await channel.declare_exchange(
            f'{self._queue_name}.retry',
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        for delay in RETRY_DELAYS_MS:
            retry_queue = await channel.declare_queue(
                f'{self._queue_name}.retry.{delay}',
                durable=True,
                arguments={
                    # After the TTL expires, the message
                    # returns to the main queue.
                    'x-dead-letter-exchange': '',
                    'x-dead-letter-routing-key': self._queue_name,
                    'x-message-ttl': delay,
                },
            )
            await retry_queue.bind(retry_exchange, routing_key=str(delay))

    async def _declare_extra_topology(
        self, channel: aio_pika.abc.AbstractChannel
    ) -> None:
        """Declare consumer-specific queues/exchanges (no-op by default)."""
        return None

    async def _on_message(
        self, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        headers = dict(message.headers or {})
        request_id = str(headers.get(REQUEST_ID_HEADER) or uuid.uuid4())

        structlog.contextvars.bind_contextvars(
            request_id=request_id, queue=self._queue_name
        )
        start = time.perf_counter()
        self._current_message = message
        try:
            with get_tracer().start_as_current_span('message.process') as span:
                span.set_attribute('mq.queue', self._queue_name)
                span.set_attribute('request.id', request_id)
                async with message.process(ignore_processed=True):
                    try:
                        if await self._skip_duplicate(message):
                            MESSAGES_TOTAL.labels(
                                queue=self._queue_name, status='duplicate'
                            ).inc()
                        else:
                            # orjson: ~5-10x faster parsing on the hot path
                            body = orjson.loads(message.body)
                            await self.handle(body)
                            await self._mark_processed(message)
                            MESSAGES_TOTAL.labels(
                                queue=self._queue_name, status='success'
                            ).inc()
                    except Exception as e:
                        logger.exception(
                            'message_processing_failed',
                            error=str(e),
                            queue=self._queue_name,
                        )
                        MESSAGES_TOTAL.labels(
                            queue=self._queue_name, status='error'
                        ).inc()
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        await self._retry_or_dead_letter(
                            message, request_id, error=e
                        )
        finally:
            MESSAGE_PROCESSING_SECONDS.labels(queue=self._queue_name).observe(
                time.perf_counter() - start
            )
            structlog.contextvars.clear_contextvars()
            self._current_message = None

    async def _idempotency_cache(self) -> RedisCache | None:
        """Optional shared Redis cache used for idempotency markers."""
        try:
            caches = await self._container.get(SharedCaches)
        except Exception:  # noqa: BLE001 - fakes/tests may not provide it
            return None
        return caches.redis

    def _idempotency_key(
        self, message: aio_pika.abc.AbstractIncomingMessage
    ) -> str | None:
        """Stable per-message key (AMQP message_id survives redelivery)."""
        if message_id := getattr(message, 'message_id', None):
            return TTLCache.make_key('idem', self._queue_name, message_id)
        else:
            return None

    async def _skip_duplicate(
        self, message: aio_pika.abc.AbstractIncomingMessage
    ) -> bool:
        """Drop a redelivered message that was already processed.

        Broker redeliveries (after a worker crash mid-processing) can
        duplicate side effects; with Redis configured, processed
        message ids are remembered and duplicates are skipped.
        """
        if not getattr(message, 'redelivered', False):
            return False
        idem = await self._idempotency_cache()
        key = self._idempotency_key(message)
        if idem is None or key is None:
            return False
        if idem.get(key) is not None:
            logger.warning(
                'duplicate_redelivery_skipped',
                queue=self._queue_name,
                message_id=getattr(message, 'message_id', None),
            )
            return True
        return False

    async def _mark_processed(
        self, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        """Remember a processed message id (no-op without Redis)."""
        idem = await self._idempotency_cache()
        key = self._idempotency_key(message)
        if idem is not None and key is not None:
            idem.set(key, '1')

    async def _retry_or_dead_letter(
        self,
        message: aio_pika.abc.AbstractIncomingMessage,
        request_id: str,
        error: Exception | None = None,
    ) -> None:
        """Schedule a retry with exponential backoff or move to the DLQ.

        Permanent errors (invalid input, poison messages) go straight
        to the DLQ - retrying can never succeed. The original message
        is acknowledged; its body is republished either to a retry
        queue (which dead-letters back to the main queue after the
        TTL) or, when retries are exhausted, to the DLQ. The
        ``request_id`` is propagated so logs stay traceable.
        """
        headers = dict(message.headers or {})
        headers[REQUEST_ID_HEADER] = request_id
        raw_count: Any = headers.get(RETRY_COUNT_HEADER, 0)
        attempt = int(raw_count)
        channel = cast(aio_pika.RobustChannel, message.channel)
        try:
            if isinstance(error, PermanentError):
                # Poison message: never retry, move to the DLQ at once.
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=message.body,
                        headers=headers,
                        delivery_mode=DeliveryMode.PERSISTENT,
                    ),
                    routing_key=f'{self._queue_name}.dlq',
                )
                MESSAGES_TOTAL.labels(
                    queue=self._queue_name, status='dlq'
                ).inc()
                logger.error(
                    'message_moved_to_dlq',
                    queue=self._queue_name,
                    reason='permanent_error',
                    error=str(error),
                )
            elif attempt < len(RETRY_DELAYS_MS):
                delay = RETRY_DELAYS_MS[attempt]
                headers[RETRY_COUNT_HEADER] = attempt + 1
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=message.body,
                        headers=headers,
                        delivery_mode=DeliveryMode.PERSISTENT,
                    ),
                    routing_key=f'{self._queue_name}.retry.{delay}',
                )
                MESSAGES_TOTAL.labels(
                    queue=self._queue_name, status='retry'
                ).inc()
                logger.warning(
                    'message_retry_scheduled',
                    attempt=attempt + 1,
                    max_retries=len(RETRY_DELAYS_MS),
                    delay_ms=delay,
                    queue=self._queue_name,
                )
            else:
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=message.body,
                        headers=headers,
                        delivery_mode=DeliveryMode.PERSISTENT,
                    ),
                    routing_key=f'{self._queue_name}.dlq',
                )
                MESSAGES_TOTAL.labels(
                    queue=self._queue_name, status='dlq'
                ).inc()
                logger.error(
                    'message_moved_to_dlq',
                    queue=self._queue_name,
                    retries=attempt,
                )
        except Exception:
            logger.exception('retry_publish_failed', queue=self._queue_name)

    @abstractmethod
    async def handle(self, data: dict[str, Any]) -> None:
        """Process a message payload."""
        ...


class IngestConsumer(BaseConsumer):
    """Consumer for ingest_queue: indexes a document."""

    def __init__(self, container: AsyncContainer) -> None:
        super().__init__('ingest_queue', container)

    async def handle(self, data: dict[str, Any]) -> None:
        """Process one ingest message from the queue."""
        token = data.get('token')
        if not token:
            raise ValueError('Missing token')

        validator = await self._container.get(TokenValidator)
        payload = validator.validate(token)
        user_id = payload['sub']

        doc_id = uuid.UUID(data['doc_id'])
        repo = await self._container.get(DocumentRepository)
        doc = await repo.get_document(doc_id)
        if doc is None or doc.owner_id != user_id:
            raise PermissionDeniedError('User is not owner of this document')

        indexer = await self._container.get(IndexerService)
        await indexer.index_document(doc_id)


class DeleteConsumer(BaseConsumer):
    """Consumer for delete_queue: deletes a document."""

    def __init__(self, container: AsyncContainer) -> None:
        super().__init__('delete_queue', container)

    async def handle(self, data: dict[str, Any]) -> None:
        """Process one delete message from the queue."""
        token = data.get('token')
        if not token:
            raise ValueError('Missing token')

        validator = await self._container.get(TokenValidator)
        payload = validator.validate(token)
        user_id = payload['sub']

        doc_id = uuid.UUID(data['doc_id'])
        remove_file = data.get('remove_file', False)

        manager = await self._container.get(DocumentManager)
        await manager.delete_document(doc_id, user_id, remove_file)


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
        self, container: AsyncContainer, reply_queue: str = 'query_reply_queue'
    ) -> None:
        super().__init__('query_queue', container)
        self._reply_queue = reply_queue

    async def _declare_extra_topology(
        self, channel: aio_pika.abc.AbstractChannel
    ) -> None:
        # Ensure the fallback reply queue exists for producers that do
        # not use per-request reply_to queues.
        await channel.declare_queue(self._reply_queue, durable=True)

    async def handle(self, data: dict[str, Any]) -> None:
        """Process one query message and publish the reply."""
        token = data.get('token')
        if not token:
            raise ValueError('Missing token')

        validator = await self._container.get(TokenValidator)
        payload = validator.validate(token)
        user_id = payload['sub']
        # Access groups may be embedded in the token (e.g. "groups" claim);
        # when absent, the retriever falls back to the DB membership table.
        user_groups = payload.get('groups')

        query = data['query']
        query_id = data.get('query_id', str(uuid.uuid4()))

        # Validate per-request routing choices before any work: a bogus
        # model string must not reach the router client cache (unbounded
        # growth) or trigger a paid embedding/search round-trip.
        try:
            settings = await self._container.get(Settings)
        except Exception:  # noqa: BLE001 - fakes may omit Settings
            settings = None
        if settings is not None:
            requested_provider = data.get('llm_provider')
            requested_model = data.get('llm_model')
            if (
                requested_provider
                and requested_provider
                not in settings.llm_enabled_providers_list
            ):
                raise ValueError(
                    f'LLM provider {requested_provider!r} is not enabled'
                )
            if requested_model:
                allowed = settings.llm_allowed_models_list
                if allowed and requested_model not in allowed:
                    raise ValueError(
                        f'LLM model {requested_model!r} is not allowed'
                    )

        retriever = await self._container.get(RetrieverService)
        result = await retriever.answer_query(
            user_id=user_id,
            query=query,
            conversation_id=data.get('conversation_id'),
            user_groups=user_groups,
            # Optional per-request LLM routing (multi-tenant support):
            # validated against the enabled providers/models allowlists.
            llm_provider=data.get('llm_provider'),
            llm_model=data.get('llm_model'),
        )

        await self._send_reply(query_id, result)
        logger.info(
            'query_processed',
            query_id=query_id,
            user_id=user_id,
            conversation_id=result.get('conversation_id'),
        )

    async def _send_reply(self, query_id: str, result: dict[str, Any]) -> None:
        """Publish the result to reply_to / the shared reply queue."""
        message = self._current_message
        reply_to = getattr(message, 'reply_to', None) if message else None
        correlation_id = (
            getattr(message, 'correlation_id', None) if message else None
        )
        routing_key = reply_to or self._reply_queue

        # orjson emits compact UTF-8 bytes directly (no .encode()).
        reply_body = orjson.dumps(
            {
                'query_id': query_id,
                'conversation_id': result.get('conversation_id'),
                'answer': result['answer'],
                'sources': result['sources'],
            }
        )

        try:
            if message is None:
                # handle() called directly (tests / programmatic use):
                # publish through the queue channel is unavailable, log only.
                logger.warning(
                    'reply_skipped_no_message_context',
                    query_id=query_id,
                    routing_key=routing_key,
                )
                return
            channel = cast(aio_pika.RobustChannel, message.channel)
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=reply_body,
                    correlation_id=correlation_id,
                    delivery_mode=DeliveryMode.PERSISTENT,
                ),
                routing_key=routing_key,
            )
            logger.info(
                'reply_published', query_id=query_id, routing_key=routing_key
            )
        except Exception:
            # The answer is already persisted in the conversation history;
            # a failed reply publication must not fail the whole message
            # (it would be retried and answered twice).
            logger.exception('reply_publish_failed', query_id=query_id)


class ReindexConsumer(BaseConsumer):
    """Consumer for reindex_queue: triggers full reindexing."""

    def __init__(self, container: AsyncContainer) -> None:
        super().__init__('reindex_queue', container)

    async def handle(self, data: dict[str, Any]) -> None:
        """Process one reindex message from the queue."""
        token = data.get('token')
        if not token:
            raise ValueError('Missing token')

        validator = await self._container.get(TokenValidator)
        payload = validator.validate(token)
        user_id = payload['sub']

        dimension = data.get('vector_dimension')

        manager = await self._container.get(DocumentManager)
        await manager.reindex_all(
            user_id, dimension, user_groups=payload.get('groups')
        )
