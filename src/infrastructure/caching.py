"""Caching infrastructure: in-process TTL cache and Redis-backed cache.

Used for LLM answers and query embeddings. Not thread-safe by design:
access happens on the event loop (cache checks are performed before
offloading heavy work to executors), which is safe under CPython's GIL
for dict operations used here.
"""

import contextlib
import hashlib
import json
import math
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    """LRU cache with a shared time-to-live for all entries."""

    def __init__(self, maxsize: int = 256, ttl: float = 3600.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    @staticmethod
    def make_key(*parts: Any) -> str:
        """Build a stable cache key from arbitrary parts."""
        digest = hashlib.sha256()
        for part in parts:
            digest.update(str(part).encode())
            digest.update(b'\x00')
        return digest.hexdigest()

    def get(self, key: str) -> Any | None:
        """Return the cached value or None if missing/expired."""
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at <= time.monotonic():
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value, evicting the oldest entry beyond ``maxsize``."""
        self._data[key] = (time.monotonic() + self._ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        """Drop all cached entries."""
        self._data.clear()

    def __len__(self) -> int:
        """Number of currently stored entries."""
        return len(self._data)


class RedisCache:
    """Redis-backed cache with the sync ``get``/``set`` protocol of TTLCache.

    Values are JSON-serialized (cache stores embedding vectors and LLM
    answers, both JSON-safe). The cache is fail-open: any Redis failure
    is swallowed and behaves like a cache miss, so a broken Redis never
    breaks message processing. ``maxsize`` is not enforced (Redis evicts
    by TTL / its own maxmemory policy).
    """

    def __init__(
        self,
        url: str,
        ttl: float = 3600.0,
        prefix: str = 'rag',
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client  # test seam
        else:
            import redis

            self._client = redis.Redis.from_url(
                url,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                decode_responses=True,
            )
        self._ttl = ttl
        self._prefix = prefix

    def _key(self, key: str) -> str:
        return f'{self._prefix}:{key}'

    def get(self, key: str) -> Any | None:
        """Return the cached value or None on miss, expiry or error."""
        try:
            raw = self._client.get(self._key(key))
        except Exception:  # noqa: BLE001 - cache must never break the app
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def set(self, key: str, value: Any) -> None:
        """Store the value under the namespaced key."""
        with contextlib.suppress(Exception):
            self._client.setex(
                self._key(key),
                int(self._ttl),
                json.dumps(value, ensure_ascii=False),
            )

    def clear(self) -> None:
        """Delete all keys under the namespace prefix."""
        with contextlib.suppress(Exception):
            pattern = f'{self._prefix}:*'
            if keys := list(self._client.scan_iter(match=pattern)):
                self._client.delete(*keys)

    def __len__(self) -> int:
        """Number of keys under the namespace prefix."""
        try:
            return sum(
                1 for _ in self._client.scan_iter(match=f'{self._prefix}:*')
            )
        except Exception:  # noqa: BLE001
            return 0


class SharedCaches:
    """Container-level holder for the optional shared Redis cache.

    Exists because DI factories registered with ``Optional`` return types
    don't match ``Optional`` consumer parameters reliably; this wrapper is
    always provided, ``redis`` is ``None`` when ``REDIS_URL`` is unset.
    """

    def __init__(self, redis: RedisCache | None) -> None:
        self.redis = redis

    def pick(self, maxsize: int, ttl: float) -> 'RedisCache | TTLCache':
        """Shared Redis cache when configured, else a fresh TTLCache."""
        return (
            self.redis
            if self.redis is not None
            else TTLCache(maxsize=maxsize, ttl=ttl)
        )


Cache = TTLCache | RedisCache


class InMemorySemanticCache:
    """Cosine-similarity answer cache (single process).

    Stores ``(vector, answer)`` pairs; ``lookup`` returns the best
    answer whose cosine similarity with the query vector is at least
    ``threshold``. Deliberately in-process: a distributed variant
    would need a vector index in Redis. Use ``enabled=False`` (no-op)
    in multi-tenant setups where cached answers leak ACL context.
    """

    def __init__(
        self,
        maxsize: int = 256,
        ttl: float = 3600.0,
        threshold: float = 0.95,
        enabled: bool = True,
        clock: Any = None,
    ) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._threshold = threshold
        self._enabled = enabled
        self._clock = clock if clock is not None else time.monotonic
        # key -> (expires_at, vector, answer); OrderedDict = LRU order
        self._entries: OrderedDict[str, tuple[float, list[float], str]] = (
            OrderedDict()
        )

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def lookup(self, vector: list[float]) -> str | None:
        """Best cached answer within the similarity threshold."""
        if not self._enabled or not self._entries:
            return None
        now = self._clock()
        best_key: str | None = None
        best_score = 0.0
        best_answer: str | None = None
        expired: list[str] = []
        for key, (expires_at, entry_vector, answer) in self._entries.items():
            if expires_at <= now:
                expired.append(key)
                continue
            score = self._cosine(vector, entry_vector)
            if score > best_score:
                best_key, best_score, best_answer = key, score, answer
        for key in expired:
            del self._entries[key]
        if best_key is None or best_score < self._threshold:
            return None
        self._entries.move_to_end(best_key)
        return best_answer

    async def store(self, vector: list[float], answer: str) -> None:
        """Remember the answer, evicting the oldest beyond ``maxsize``."""
        if not self._enabled:
            return
        key = hashlib.sha256(str(vector).encode()).hexdigest()
        self._entries[key] = (self._clock() + self._ttl, vector, answer)
        self._entries.move_to_end(key)
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)
