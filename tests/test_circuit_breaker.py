"""Tests for the circuit breaker and its integration with DeepSeekClient."""

import httpx
import pytest

from infrastructure.llm import DeepSeekClient
from infrastructure.resilience import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
    CircuitOpenError,
)

pytestmark = pytest.mark.infra


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _breaker(threshold: int = 3, timeout: float = 60.0):
    clock = FakeClock()
    return CircuitBreaker(
        name="test",
        failure_threshold=threshold,
        reset_timeout=timeout,
        clock=clock,
    ), clock


async def test_starts_closed_and_allows_calls() -> None:
    breaker, _ = _breaker()
    assert breaker.state == CLOSED
    assert breaker.allow() is True


async def test_opens_after_failure_threshold() -> None:
    breaker, _ = _breaker(threshold=3)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == OPEN
    assert breaker.allow() is False


async def test_rejections_do_not_count_as_failures() -> None:
    breaker, _ = _breaker(threshold=2)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.allow() is False
    # while open, allow() must not flip state or record failures
    assert breaker.state == OPEN


async def test_half_open_after_reset_timeout_then_success_closes() -> None:
    breaker, clock = _breaker(threshold=1, timeout=60.0)
    breaker.record_failure()
    assert breaker.allow() is False
    clock.now = 61.0
    assert breaker.state == OPEN  # lazily transitioned on allow()
    assert breaker.allow() is True
    assert breaker.state == HALF_OPEN
    breaker.record_success()
    assert breaker.state == CLOSED


async def test_half_open_failure_reopens() -> None:
    breaker, clock = _breaker(threshold=1, timeout=60.0)
    breaker.record_failure()
    clock.now = 61.0
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state == OPEN
    assert breaker.allow() is False


async def test_success_resets_failure_count() -> None:
    breaker, _ = _breaker(threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CLOSED  # 2 < 3 after reset


async def test_call_wrapper_records_success_and_failure():
    breaker, _ = _breaker(threshold=2)

    async def ok():
        return "value"

    async def boom() -> None:
        raise RuntimeError("down")

    assert await breaker.call(ok) == "value"
    with pytest.raises(RuntimeError):
        await breaker.call(boom)
    with pytest.raises(RuntimeError):
        await breaker.call(boom)
    with pytest.raises(CircuitOpenError):
        await breaker.call(ok)


def _mock_transport(calls: list, status_code: int = 500):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status_code, json={"error": "boom"})

    return httpx.MockTransport(handler)


async def test_deepseek_client_opens_circuit_after_failures() -> None:
    calls: list = []
    breaker = CircuitBreaker(name="llm-test", failure_threshold=2, reset_timeout=60.0)
    client = DeepSeekClient(
        api_key="k",
        base_url="https://api.test/v1",
        http_transport=_mock_transport(calls),
        circuit_breaker=breaker,
    )
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await client.generate("system", "user")
    assert breaker.state == OPEN
    # Third call fails fast without touching the network.
    with pytest.raises(CircuitOpenError):
        await client.generate("system", "user")
    assert len(calls) == 2


async def test_deepseek_client_success_closes_circuit():
    breaker = CircuitBreaker(name="llm-test", failure_threshold=1, reset_timeout=60.0)
    clock = FakeClock()
    breaker._clock = clock
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}], "usage": {}},
        )

    client = DeepSeekClient(
        api_key="k",
        base_url="https://api.test/v1",
        http_transport=httpx.MockTransport(handler),
        circuit_breaker=breaker,
    )
    with pytest.raises(RuntimeError):
        await client.generate("system", "user")
    assert breaker.state == OPEN
    # While open, the call is rejected without touching the network.
    with pytest.raises(CircuitOpenError):
        await client.generate("system", "user")
    assert len(calls) == 1
    clock.now = 61.0
    assert await client.generate("system", "user") == "hello"
    assert breaker.state == CLOSED
    assert len(calls) == 2


async def test_cache_hit_records_success() -> None:
    from infrastructure.caching import TTLCache

    breaker = CircuitBreaker(name="llm-test", failure_threshold=1, reset_timeout=60.0)
    breaker.record_failure()
    assert breaker.state == OPEN
    cache = TTLCache(maxsize=8, ttl=60.0)
    key = TTLCache.make_key("s", "u", 0.1)
    cache.set(key, "cached answer")
    client = DeepSeekClient(
        api_key="k",
        base_url="https://api.test/v1",
        http_transport=_mock_transport([]),
        cache=cache,
        circuit_breaker=breaker,
    )
    # Circuit is open, but a cache hit short-circuits and heals the breaker.
    assert await client.generate("s", "u") == "cached answer"
    assert breaker.state == CLOSED
