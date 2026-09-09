"""Base consumer class with RabbitMQ connection."""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

import aio_pika
from aio_pika import IncomingMessage

from config import Settings


class BaseConsumer(ABC):
    """Abstract base for a RabbitMQ consumer."""

    def __init__(self, queue_name: str, settings: Settings) -> None:
        self._queue_name = queue_name
        self._settings = settings

    async def start(self) -> None:
        """Start consuming messages from the queue."""
        connection = await aio_pika.connect_robust(self._settings.rabbitmq_url)
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
                # In production, log with structlog and send to DLX
                print(f"Error processing message: {e}")

    @abstractmethod
    async def handle(self, data: dict[str, Any]) -> None:
        """Process a message payload."""
        ...