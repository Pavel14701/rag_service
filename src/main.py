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
from application.interfaces import EmbeddingModel, VectorStore
from application.services import DocumentManager
from entrypoints.consumers import BaseConsumer
from entrypoints.health import HealthServer, build_dependency_checks
from entrypoints.consumers import IngestConsumer
from entrypoints.consumers import QueryConsumer
from entrypoints.consumers import DeleteConsumer
from entrypoints.consumers import ReindexConsumer
from infrastructure.observability import configure_logging
from infrastructure.observability import setup_tracing

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


async def verify_embedding_dimension(container: AsyncContainer) -> None:
    """Fail fast when the embedding dimension mismatches the collection.

    Qdrant rejects searches whose vector size differs from the
    collection config, so an EMBEDDING_MODEL change without a reindex
    would paralyze the query path entirely. Exiting at startup makes
    the mismatch loud (orchestrator alerts) instead of failing every
    query at runtime.
    """
    embedding = await container.get(EmbeddingModel)
    store = await container.get(VectorStore)
    configured = embedding.dimension()
    stored = await store.get_collection_dimension()
    if stored is not None and stored != configured:
        logger.critical(
            'embedding_dimension_mismatch_fail_fast',
            stored_dimension=stored,
            configured_dimension=configured,
            hint=(
                'restore the previous EMBEDDING_MODEL or reindex into '
                'a new collection (send a reindex message)'
            ),
        )
        raise SystemExit(1)


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

    await warn_on_embedding_model_mismatch(container)
    await verify_embedding_dimension(container)

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
