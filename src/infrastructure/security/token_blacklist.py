"""Token revocation stores (JWT blacklist by ``jti`` claim)."""

import time
from typing import Any


class InMemoryTokenBlacklist:
    """In-process revocation store with per-entry TTL.

    Fits single-process deployments and tests; use
    :class:`RedisTokenBlacklist` when several worker replicas share the
    revocation state.
    """

    def __init__(self) -> None:
        self._revoked: dict[str, float] = {}  # jti -> revoked-until timestamp

    def revoke(self, jti: str, ttl: float) -> None:
        """Revoke ``jti`` for ``ttl`` seconds (until the token expires)."""
        self._revoked[jti] = time.time() + ttl
        # Opportunistic cleanup of expired entries.
        now = time.time()
        expired = [k for k, until in self._revoked.items() if until <= now]
        for key in expired:
            del self._revoked[key]

    def contains(self, jti: str) -> bool:
        """True when the given token ID is revoked."""
        until = self._revoked.get(jti)
        return until is not None and until > time.time()


class RedisTokenBlacklist:
    """Redis-backed revocation store shared by all worker replicas.

    Each revoked ``jti`` is stored with an expiry equal to the token's
    remaining lifetime, so the blacklist self-cleans.
    """

    def __init__(
        self, redis_client: Any, prefix: str = 'rag:revoked-jti'
    ) -> None:
        self._client = redis_client
        self._prefix = prefix

    def revoke(self, jti: str, ttl: float) -> None:
        """Store the revoked jti in Redis until the token would expire."""
        self._client.setex(f'{self._prefix}:{jti}', max(int(ttl), 1), '1')

    def contains(self, jti: str) -> bool:
        """True when the token ID is present in Redis."""
        return bool(self._client.exists(f'{self._prefix}:{jti}'))
