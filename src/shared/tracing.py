"""Optional OpenTelemetry tracing support.

Tracing is a no-op unless ``setup_tracing`` is called with an OTLP/HTTP
endpoint (e.g. Jaeger). All instrumentation goes through ``get_tracer()``,
which falls back to the global (no-op) tracer when the SDK is not installed
by the application.
"""

from opentelemetry import trace

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
