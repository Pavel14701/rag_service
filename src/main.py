"""Application entry point.

Starts the health/metrics server and the RabbitMQ consumers selected by
``WORKER_QUEUES``:
- ``query``      — only the real-time query consumer;
- ``background`` — ingest/delete/reindex consumers;
- ``all``        — everything in one process (default, for dev).

Run separate worker replicas (``WORKER_QUEUES=query`` and
``WORKER_QUEUES=background``) so long-running indexing jobs never block
real-time question answering.
"""

import asyncio

import structlog
from dishka import AsyncContainer

from config import Settings
from container import create_container
from application.services.document_manager import DocumentManager
from infrastructure.parsing.factory import ParserFactory
from entrypoints.base_consumer import BaseConsumer
from entrypoints.health import HealthServer, build_dependency_checks
from entrypoints.ingest_consumer import IngestConsumer
from entrypoints.query_consumer import QueryConsumer
from entrypoints.delete_consumer import DeleteConsumer
from entrypoints.reindex_consumer import ReindexConsumer
from shared.logging import configure_logging
from shared.tracing import setup_tracing

logger = structlog.get_logger()


def select_consumers(
    worker_queues: str, container: AsyncContainer
) -> list[BaseConsumer]:
    """Build the consumer set for this worker role."""
    factories = {
        'query': QueryConsumer,
        'ingest': IngestConsumer,
        'delete': DeleteConsumer,
        'reindex': ReindexConsumer,
    }
    mode = worker_queues.strip().lower() or 'all'
    if mode == 'all':
        roles = list(factories)
    elif mode in ('query', 'background'):
        roles = (
            ['query'] if mode == 'query' else ['ingest', 'delete', 'reindex']
        )
    else:
        raise ValueError(
            f'Invalid WORKER_QUEUES={worker_queues!r}; '
            "expected 'query', 'background' or 'all'"
        )
    return [factories[role](container) for role in roles]


async def warn_on_embedding_model_mismatch(container: AsyncContainer) -> None:
    """Log a warning when stored vectors were built by another model.

    Vectors incompatible with the current embedding model silently
    degrade retrieval quality; the remediation is a full reindex
    (admin: send a reindex message).
    """
    manager = await container.get(DocumentManager)
    stored, expected = await manager.check_embedding_version()
    if stored is not None and expected is not None and stored != expected:
        logger.warning(
            'embedding_model_mismatch',
            stored=stored,
            expected=expected,
            hint='full reindex required (send a reindex message)',
        )


def configure_parsing(settings: Settings) -> None:
    """Apply OCR settings to the parsing pipeline."""
    ParserFactory.configure(
        settings.pdf_ocr_strategy, settings.pdf_ocr_languages
    )


async def main() -> None:
    """Start the health/metrics server and the selected RabbitMQ consumers."""
    container = create_container()
    settings = await container.get(Settings)

    configure_logging(settings.log_level, json_output=settings.log_json)
    if settings.otel_endpoint:
        setup_tracing(settings.otel_endpoint)

    health = HealthServer(
        checks=build_dependency_checks(container, settings),
        port=settings.metrics_port,
        check_timeout=settings.health_check_timeout,
    )
    await health.start()
    logger.info('health_server_started', port=settings.metrics_port)

    configure_parsing(settings)
    await warn_on_embedding_model_mismatch(container)

    consumers = select_consumers(settings.worker_queues, container)
    logger.info(
        'workers_starting',
        roles=settings.worker_queues,
        consumers=len(consumers),
    )
    try:
        await asyncio.gather(*(c.start() for c in consumers))
    finally:
        await health.stop()


if __name__ == '__main__':
    asyncio.run(main())
