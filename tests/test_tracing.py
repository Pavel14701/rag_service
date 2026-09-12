from typing import Any

"""Tests for OpenTelemetry tracing instrumentation.

Installs an SDK tracer provider with an in-memory exporter once for this
module and asserts that spans are created for message processing, vector
search and LLM generation.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from conftest import FakeContainer, FakeMessage

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
otel_trace.set_tracer_provider(_provider)

from entrypoints.consumers import BaseConsumer  # noqa: E402
from infrastructure.llm import DeepSeekClient  # noqa: E402
from infrastructure.vector_store import QdrantStore  # noqa: E402

pytestmark = pytest.mark.observability


def _span_names() -> list[str]:
    return [span.name for span in _exporter.get_finished_spans()]


class QuietConsumer(BaseConsumer):
    def __init__(self) -> None:
        super().__init__("trace_queue", FakeContainer({}))

    async def handle(self, data: dict[str, Any]):
        return None


async def test_message_processing_span_created() -> None:
    consumer = QuietConsumer()
    await consumer._on_message(FakeMessage(b"{}"))
    assert "message.process" in _span_names()


async def test_vector_search_span_created() -> None:
    client = MagicMock()
    response = MagicMock()
    response.points = []
    client.query_points.return_value = response
    store = QdrantStore(client, "test-collection")

    await store.search(vector=[0.0], top_k=1)

    assert "vector.search" in _span_names()


async def test_llm_generate_span_created_with_attributes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
        )

    client = DeepSeekClient(
        "key", "http://test", http_transport=httpx.MockTransport(handler)
    )
    await client.generate("s", "u")

    names = _span_names()
    assert "llm.generate" in names
    span = [s for s in _exporter.get_finished_spans() if s.name == "llm.generate"][-1]
    assert span.attributes["llm.model"] == "deepseek-chat"
    assert span.attributes["llm.completion_tokens"] == 2
