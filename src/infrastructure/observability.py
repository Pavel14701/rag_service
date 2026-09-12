"""Observability: structured logging, metrics and tracing.

- Logging: structured records via structlog; JSON lines when
  ``json_output`` is enabled, colored console otherwise.
- Metrics: Prometheus counters/histograms (names keep the ``rag_``
  prefix so dashboards survive refactors).
- Tracing: optional OpenTelemetry; a no-op unless ``setup_tracing``
  is called with an OTLP/HTTP endpoint (e.g. Jaeger).
"""

import logging
import sys

import structlog
from prometheus_client import Counter, Histogram
from opentelemetry import trace


def configure_logging(level: str = 'INFO', json_output: bool = True) -> None:
    """Configure structlog and stdlib logging."""
    logging.basicConfig(
        level=level.upper(), stream=sys.stdout, format='%(message)s'
    )

    renderer: structlog.typing.Processor
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


MESSAGE_PROCESSING_SECONDS = Histogram(
    'rag_message_processing_seconds',
    'Time spent processing a message, by queue.',
    ['queue'],
    buckets=(0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 60.0, 300.0),
)
MESSAGES_TOTAL = Counter(
    'rag_messages_total',
    'Processed messages by queue and outcome (success/error/retry/dlq).',
    ['queue', 'status'],
)
VECTOR_SEARCH_SECONDS = Histogram(
    'rag_vector_search_seconds',
    'Latency of vector store search calls.',
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)
LLM_GENERATION_SECONDS = Histogram(
    'rag_llm_generation_seconds',
    'Latency of LLM generation calls.',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)
LLM_TOKENS_TOTAL = Counter(
    'rag_llm_tokens_total',
    'LLM tokens consumed, by kind (prompt/completion).',
    ['kind'],
)
LLM_CACHE_TOTAL = Counter(
    'rag_llm_cache_total',
    'LLM answer cache lookups, by result (hit/miss).',
    ['result'],
)
LLM_TRUNCATIONS_TOTAL = Counter(
    'rag_llm_truncations_total',
    'LLM responses cut by the max_tokens limit, by provider.',
    ['provider'],
)


_TRACER_NAME = 'rag_service'


def setup_tracing(endpoint: str, service_name: str = 'rag_service') -> None:
    """Install an SDK tracer provider exporting spans via OTLP/HTTP."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({'service.name': service_name})
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )
    trace.set_tracer_provider(provider)


def get_tracer() -> trace.Tracer:
    """Return the application tracer (no-op unless tracing is configured)."""
    return trace.get_tracer(_TRACER_NAME)
