"""Small in-process TTL cache used for LLM answers and query embeddings.

Not thread-safe by design: access happens on the event loop (cache checks
are performed before offloading heavy work to executors), which is safe
under CPython's GIL for dict operations used here.
"""

import hashlib
import json
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
            digest.update(b"\x00")
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
        self._data.clear()

    def __len__(self) -> int:
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
        prefix: str = "rag",
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
        return f"{self._prefix}:{key}"

    def get(self, key: str) -> Any | None:
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
        try:
            self._client.setex(
                self._key(key),
                int(self._ttl),
                json.dumps(value, ensure_ascii=False),
            )
        except Exception:  # noqa: BLE001
            pass

    def clear(self) -> None:
        try:
            pattern = f"{self._prefix}:*"
            keys = list(self._client.scan_iter(match=pattern))
            if keys:
                self._client.delete(*keys)
        except Exception:  # noqa: BLE001
            pass

    def __len__(self) -> int:
        try:
            return sum(1 for _ in self._client.scan_iter(match=f"{self._prefix}:*"))
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

    def pick(self, maxsize: int, ttl: float) -> "RedisCache | TTLCache":
        """Shared Redis cache when configured, else a fresh TTLCache."""
        return self.redis if self.redis is not None else TTLCache(maxsize=maxsize, ttl=ttl)