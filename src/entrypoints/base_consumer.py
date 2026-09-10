"""Base consumer class with RabbitMQ connection."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import aio_pika
from aio_pika import IncomingMessage
from dishka import AsyncContainer

from config import Settings

logger = logging.getLogger(__name__)


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
            queue = await channel.declare_queue(self._queue_name, durable=True)
            await queue.consume(self._on_message)
            await asyncio.Future()

    async def _on_message(self, message: IncomingMessage) -> None:
        async with message.process():
            try:
                body = json.loads(message.body.decode())
                await self.handle(body)
            except Exception as e:
                logger.exception("Error processing message: %s", e)

    @abstractmethod
    async def handle(self, data: dict[str, Any]) -> None:
        """Process a message payload."""
        ...