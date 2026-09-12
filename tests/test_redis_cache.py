"""Tests for the Redis-backed cache (fake client, fail-open behavior)."""

from infrastructure.caching import RedisCache, TTLCache

import pytest

pytestmark = pytest.mark.llm



class FakeRedis:
    def __init__(self, store: dict[str, str] | None = None, fail: bool = False) -> None:
        self.store = store if store is not None else {}
        self.fail = fail
        self.setex_calls: list[tuple] = []

    def get(self, key: str):
        if self.fail:
            raise ConnectionError("down")
        return self.store.get(key)

    def setex(self, key, ttl, value) -> None:
        if self.fail:
            raise ConnectionError("down")
        self.setex_calls.append((key, ttl, value))
        self.store[key] = value

    def scan_iter(self, match: str | None = None):
        for key in list(self.store):
            if match and key.startswith(match[:-1]):
                yield key

    def delete(self, *keys) -> None:
        for key in keys:
            self.store.pop(key, None)


def make_cache(**kwargs) -> RedisCache:
    return RedisCache(url="redis://t", ttl=60, prefix="rag", client=FakeRedis(**kwargs))


def test_roundtrip() -> None:
    cache = make_cache()
    cache.set("k", ["v", 1])
    assert cache.get("k") == ["v", 1]


def test_get_missing_is_none() -> None:
    assert make_cache().get("nope") is None


def test_uses_prefix_and_ttl() -> None:
    fake = FakeRedis()
    cache = RedisCache(url="r", ttl=120, prefix="rag", client=fake)
    cache.set("abc", {"x": 1})
    key, ttl, _ = fake.setex_calls[0]
    assert key == "rag:abc"
    assert ttl == 120


def test_fail_open_on_redis_error() -> None:
    cache = RedisCache(url="r", ttl=60, prefix="p", client=FakeRedis(fail=True))
    cache.set("k", "v")  # must not raise
    assert cache.get("k") is None  # behaves like a miss


def test_corrupted_value_behaves_as_miss() -> None:
    fake = FakeRedis(store={"rag:k": "not-json{"})
    cache = RedisCache(url="r", ttl=60, prefix="rag", client=fake)
    assert cache.get("k") is None


def test_clear_removes_prefixed_keys() -> None:
    fake = FakeRedis(store={"rag:a": "1", "other:b": "2"})
    cache = RedisCache(url="r", ttl=60, prefix="rag", client=fake)
    cache.clear()
    assert "rag:a" not in fake.store
    assert fake.store["other:b"] == "2"


def test_llm_answer_stored_in_redis_cache():
    import httpx

    from infrastructure.llm import DeepSeekClient
    from infrastructure.caching import RedisCache, TTLCache

    fake = FakeRedis()
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer"}}],
                "usage": {},
            },
        )

    client = DeepSeekClient(
        "key",
        "http://test",
        http_transport=httpx.MockTransport(handler),
        cache=RedisCache(url="r", ttl=60, prefix="rag", client=fake),
    )

    import asyncio

    async def run():
        first = await client.generate("s", "u")
        second = await client.generate("s", "u")
        return first, second

    first, second = asyncio.run(run())
    assert first == second == "answer"
    assert len(calls) == 1  # second served from Redis
    assert len(fake.store) == 1
# ---------- container wiring ----------


async def test_container_uses_ttl_cache_without_redis_url(monkeypatch) -> None:
    from container import create_container
    from application.interfaces import LLMGenerator

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.delenv("REDIS_URL", raising=False)
    container = create_container()
    client = (await container.get(LLMGenerator)).default_client
    assert type(client._cache) is TTLCache


async def test_container_uses_redis_cache_with_redis_url(monkeypatch) -> None:
    from container import create_container
    from application.interfaces import LLMGenerator

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    container = create_container()
    client = (await container.get(LLMGenerator)).default_client
    assert type(client._cache) is RedisCache  # redis-py connects lazily
