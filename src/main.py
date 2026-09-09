"""Application entry point: start all consumers."""

import asyncio
import structlog

from container import create_container
from entrypoints.ingest_consumer import IngestConsumer
from entrypoints.query_consumer import QueryConsumer
from entrypoints.delete_consumer import DeleteConsumer
from entrypoints.reindex_consumer import ReindexConsumer

logger = structlog.get_logger()


async def main() -> None:
    """Start all RabbitMQ consumers."""
    container = await create_container()
    consumers = [
        IngestConsumer(container),
        QueryConsumer(container),
        DeleteConsumer(container),
        ReindexConsumer(container),
    ]
    tasks = [c.start() for c in consumers]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())