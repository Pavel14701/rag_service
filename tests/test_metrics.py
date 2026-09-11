"""Tests for Prometheus metrics instrumentation."""

from prometheus_client import REGISTRY

from entrypoints.base_consumer import BaseConsumer

from conftest import FakeContainer, FakeMessage


class OutcomeConsumer(BaseConsumer):
    def __init__(self, container, fail: bool = False):
        super().__init__("metric_queue", container)
        self.fail = fail

    async def handle(self, data):
        if self.fail:
            raise RuntimeError("boom")


def metric(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


async def test_success_increments_messages_total():
    consumer = OutcomeConsumer(FakeContainer({}), fail=False)
    before = metric("rag_messages_total", queue="metric_queue", status="success")
    await consumer._on_message(FakeMessage(b"{}"))
    after = metric("rag_messages_total", queue="metric_queue", status="success")
    assert after == before + 1


async def test_error_increments_error_and_retry_counters():
    consumer = OutcomeConsumer(FakeContainer({}), fail=True)
    before_err = metric("rag_messages_total", queue="metric_queue", status="error")
    before_retry = metric("rag_messages_total", queue="metric_queue", status="retry")
    await consumer._on_message(FakeMessage(b"{}"))
    assert (
        metric("rag_messages_total", queue="metric_queue", status="error")
        == before_err + 1
    )
    assert (
        metric("rag_messages_total", queue="metric_queue", status="retry")
        == before_retry + 1
    )


async def test_message_duration_histogram_observed():
    consumer = OutcomeConsumer(FakeContainer({}), fail=False)
    before = metric("rag_message_processing_seconds_count", queue="metric_queue")
    await consumer._on_message(FakeMessage(b"{}"))
    after = metric("rag_message_processing_seconds_count", queue="metric_queue")
    assert after == before + 1


async def test_dlq_counter():
    consumer = OutcomeConsumer(FakeContainer({}), fail=True)
    before = metric("rag_messages_total", queue="metric_queue", status="dlq")
    await consumer._on_message(FakeMessage(b"{}", headers={"x-retry-count": 3}))
    assert (
        metric("rag_messages_total", queue="metric_queue", status="dlq") == before + 1
    )