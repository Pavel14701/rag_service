"""Tests for the TTLCache."""

import asyncio

from shared.caching import TTLCache


def test_get_set_roundtrip():
    cache = TTLCache(maxsize=10, ttl=60)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert len(cache) == 1


def test_get_missing_returns_none():
    cache = TTLCache()
    assert cache.get("nope") is None


def test_ttl_expiry():
    cache = TTLCache(maxsize=10, ttl=0.01)
    cache.set("k", "v")
    asyncio.run(asyncio.sleep(0.03))
    assert cache.get("k") is None


def test_maxsize_evicts_oldest():
    cache = TTLCache(maxsize=2, ttl=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # evicts "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_lru_refresh_on_get():
    cache = TTLCache(maxsize=2, ttl=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # touch "a" -> "b" becomes LRU
    cache.set("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_make_key_distinguishes_parts():
    assert TTLCache.make_key("a", "b") == TTLCache.make_key("a", "b")
    assert TTLCache.make_key("a", "b") != TTLCache.make_key("b", "a")
    assert TTLCache.make_key("a b") != TTLCache.make_key("a", "b")


def test_clear():
    cache = TTLCache()
    cache.set("k", "v")
    cache.clear()
    assert len(cache) == 0
    assert cache.get("k") is None