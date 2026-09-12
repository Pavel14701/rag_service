"""Circuit breaker for external dependencies (e.g. the LLM API).

Prevents cascading failures: after ``failure_threshold`` consecutive
failures the circuit opens and calls are rejected immediately
(``CircuitOpenError``) without hitting the dependency. After
``reset_timeout`` seconds the circuit becomes half-open and lets a
limited number of trial calls through; a successful trial closes it
again.

Designed for single-event-loop use (no locks; fail-fast counters).
"""

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from prometheus_client import Counter

CIRCUIT_BREAKER_TOTAL = Counter(
    'rag_circuit_breaker_total',
    'Circuit breaker events, by breaker name and event kind.',
    ['breaker', 'event'],
)

CLOSED = 'closed'
OPEN = 'open'
HALF_OPEN = 'half_open'


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    """Async-friendly circuit breaker (call ``allow`` before the call)."""

    def __init__(
        self,
        name: str = 'llm',
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._name = name
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._clock = clock
        self._state = CLOSED
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        """Current breaker state (CLOSED, OPEN or HALF_OPEN)."""
        return self._state

    @property
    def failures(self) -> int:
        """Number of consecutive failures since the last success."""
        return self._failures

    def allow(self) -> bool:
        """Whether a call may proceed right now."""
        if self._state == CLOSED:
            return True
        elapsed = self._clock() - (self._opened_at or 0)
        if elapsed >= self._reset_timeout:
            self._state = HALF_OPEN
            CIRCUIT_BREAKER_TOTAL.labels(
                breaker=self._name, event='half_open'
            ).inc()
            return True
        CIRCUIT_BREAKER_TOTAL.labels(
            breaker=self._name, event='rejected'
        ).inc()
        return False

    def record_success(self) -> None:
        """A call succeeded: reset the breaker to closed."""
        self._failures = 0
        self._opened_at = None
        if self._state != CLOSED:
            CIRCUIT_BREAKER_TOTAL.labels(
                breaker=self._name, event='closed'
            ).inc()
        self._state = CLOSED

    def record_failure(self) -> None:
        """A call failed; open the circuit after the threshold."""
        self._failures += 1
        if self._state == HALF_OPEN or self._failures >= self._threshold:
            self._open()

    def _open(self) -> None:
        self._state = OPEN
        self._opened_at = self._clock()
        self._failures = 0
        CIRCUIT_BREAKER_TOTAL.labels(breaker=self._name, event='opened').inc()

    async def call(
        self,
        factory: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Convenience wrapper: guard an async call with the breaker."""
        if not self.allow():
            raise CircuitOpenError(
                f"Circuit '{self._name}' is open; call rejected"
            )
        try:
            result = await factory(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


class InProcessDistributedLock:
    """Single-process fallback used when Redis is not configured.

    Provides the same acquire/release protocol so application code is
    identical with and without Redis; exclusivity only holds within
    one worker process.
    """

    def __init__(self) -> None:
        self._held: set[str] = set()

    async def acquire(self, name: str, ttl: float = 300.0) -> bool:
        """Lock ``name`` in-process (``ttl`` accepted for protocol parity)."""
        if name in self._held:
            return False
        self._held.add(name)
        return True

    async def release(self, name: str) -> None:
        """Unlock ``name`` if held by this instance."""
        self._held.discard(name)


# Release must delete the key only when this owner's token is still
# stored: a plain DEL could drop a lock re-acquired by another worker
# after our TTL expired.
_RELEASE_SCRIPT = (
    'if redis.call("get", KEYS[1]) == ARGV[1] then '
    'return redis.call("del", KEYS[1]) else return 0 end'
)


class RedisDistributedLock:
    """Redis lock: ``SET key token NX EX ttl`` + compare-and-delete release.

    ``client`` is a ``redis.asyncio.Redis`` instance.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._tokens: dict[str, str] = {}

    async def acquire(self, name: str, ttl: float = 300.0) -> bool:
        """Set the lock key with NX; True when this call won the race."""
        token = uuid.uuid4().hex
        acquired = await self._client.set(
            f'lock:{name}', token, nx=True, ex=max(int(ttl), 1)
        )
        if acquired:
            self._tokens[name] = token
        return bool(acquired)

    async def release(self, name: str) -> None:
        """Compare-and-delete the lock key for an owned token."""
        token = self._tokens.pop(name, None)
        if token is not None:
            await self._client.eval(_RELEASE_SCRIPT, 1, f'lock:{name}', token)
