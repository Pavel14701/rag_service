"""Tests for LLM truncation handling (finish_reason / stop_reason)."""

import json

import httpx
from prometheus_client import REGISTRY
from infrastructure.caching import TTLCache

from infrastructure.llm import AnthropicClient, OpenAIChatClient

import pytest

pytestmark = pytest.mark.llm



def metric(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _openai_payload(content: str, finish_reason: str | None) -> dict:
    return {
        "choices": [
            {"message": {"content": content}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _anthropic_payload(text: str, stop_reason: str | None) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "stop_reason": stop_reason,
    }


async def test_truncated_openai_answer_is_not_cached() -> None:

    cache = TTLCache(maxsize=8, ttl=60)
    client = OpenAIChatClient(
        "key",
        "http://test",
        model="m",
        http_transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json=_openai_payload("cut text", "length")
            )
        ),
        cache=cache,
    )

    result = await client.generate("s", "u")

    assert result == "cut text"
    assert len(cache) == 0  # truncated answer must not poison the cache
    assert (
        metric("rag_llm_truncations_total", provider="openai-compatible") == 1.0
    )


async def test_truncated_openai_answer_continuation_merges():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(200, json=_openai_payload("part1", "length"))
        return httpx.Response(200, json=_openai_payload("part2", "stop"))

    client = OpenAIChatClient(
        "key",
        "http://test",
        model="m",
        http_transport=httpx.MockTransport(handler),
        continue_on_truncation=True,
    )

    result = await client.generate("s", "u")

    assert result == "part1part2"
    assert len(calls) == 2
    roles = [m["role"] for m in calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "Continue" in calls[1]["messages"][-1]["content"]


async def test_normal_openai_finish_is_cached() -> None:
    from infrastructure.caching import TTLCache

    cache = TTLCache(maxsize=8, ttl=60)
    client = OpenAIChatClient(
        "key",
        "http://test",
        model="m",
        http_transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json=_openai_payload("full answer", "stop")
            )
        ),
        cache=cache,
    )

    await client.generate("s", "u")

    assert len(cache) == 1


async def test_truncated_anthropic_answer_is_not_cached_and_continues():
    calls: list[dict] = []
    cache = TTLCache(maxsize=8, ttl=60)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200, json=_anthropic_payload("half ", "max_tokens")
            )
        return httpx.Response(
            200, json=_anthropic_payload("answer", "end_turn")
        )

    client = AnthropicClient(
        "key",
        "http://test",
        model="m",
        http_transport=httpx.MockTransport(handler),
        cache=cache,
        continue_on_truncation=True,
    )

    result = await client.generate("s", "u")

    assert result == "half answer"
    # only the merged FULL answer is cached, never the truncated part
    assert len(cache) == 1
    assert cache.get(TTLCache.make_key("s", "u", 0.1)) == "half answer"
    assert len(calls) == 2
    roles = [m["role"] for m in calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert (
        metric("rag_llm_truncations_total", provider="anthropic") == 1.0
    )


async def test_shared_http_client_is_reused() -> None:
    """When a pooled client is injected, no per-call client is built."""
    from unittest.mock import AsyncMock, MagicMock

    shared = MagicMock()
    shared.post = AsyncMock(
        return_value=httpx.Response(
            200, json=_openai_payload("pooled answer", "stop")
        )
    )
    client = OpenAIChatClient(
        "key", "http://test", model="m", http_client=shared
    )

    result = await client.generate("s", "u")

    assert result == "pooled answer"
    shared.post.assert_awaited_once()
    # the injected client must not be closed by the adapter
    assert not isinstance(shared, httpx.AsyncClient) or not shared.is_closed
