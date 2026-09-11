"""Tests for DeepSeekClient instrumentation."""

import httpx
import pytest
from prometheus_client import REGISTRY

from infrastructure.llm.deepseek_client import DeepSeekClient


def _client(handler) -> DeepSeekClient:
    return DeepSeekClient("key", "http://test", http_transport=httpx.MockTransport(handler))


def metric(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "the answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


async def test_generate_returns_content_and_records_tokens():
    client = _client(lambda request: _ok_response())
    before_prompt = metric("rag_llm_tokens_total", kind="prompt")
    before_completion = metric("rag_llm_tokens_total", kind="completion")

    result = await client.generate("system", "user")

    assert result == "the answer"
    assert metric("rag_llm_tokens_total", kind="prompt") == before_prompt + 10
    assert (
        metric("rag_llm_tokens_total", kind="completion") == before_completion + 5
    )


async def test_generate_records_latency_histogram():
    client = _client(lambda request: _ok_response())
    before = metric("rag_llm_generation_seconds_count")
    await client.generate("s", "u")
    assert metric("rag_llm_generation_seconds_count") == before + 1


async def test_generate_error_raises_runtime_error():
    client = _client(
        lambda request: httpx.Response(500, text="server error")
    )
    with pytest.raises(RuntimeError, match="DeepSeek API error"):
        await client.generate("s", "u")


def _counting_client(calls: list[int], cache=None) -> DeepSeekClient:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _ok_response()

    return DeepSeekClient(
        "key", "http://test", http_transport=httpx.MockTransport(handler), cache=cache
    )


async def test_cache_hit_skips_http_call():
    from shared.caching import TTLCache
    from prometheus_client import REGISTRY

    calls: list[int] = []
    client = _counting_client(calls, cache=TTLCache(maxsize=8, ttl=60))

    first = await client.generate("sys", "user")
    second = await client.generate("sys", "user")

    assert first == second == "the answer"
    assert len(calls) == 1  # second answer served from cache
    hits = REGISTRY.get_sample_value("rag_llm_cache_total", {"result": "hit"})
    misses = REGISTRY.get_sample_value("rag_llm_cache_total", {"result": "miss"})
    # Global counters: other tests may have incremented them; check deltas
    # via the client's own behaviour (1 call) plus positive hit/miss counts.
    assert hits >= 1 and misses >= 1


async def test_cache_respects_temperature_in_key():
    from shared.caching import TTLCache

    calls: list[int] = []
    client = _counting_client(calls, cache=TTLCache(maxsize=8, ttl=60))

    await client.generate("s", "u", temperature=0.1)
    await client.generate("s", "u", temperature=0.9)

    assert len(calls) == 2  # different temperature -> cache miss


async def test_generate_sends_chat_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return _ok_response()

    import json

    client = _client(handler)
    await client.generate("sys prompt", "user prompt", temperature=0.3)

    assert captured["url"] == "http://test/chat/completions"
    assert captured["payload"]["messages"][0] == {
        "role": "system",
        "content": "sys prompt",
    }
    assert captured["payload"]["temperature"] == 0.3