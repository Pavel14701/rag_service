"""Tests for distributed lock implementations."""

from unittest.mock import AsyncMock, MagicMock

from infrastructure.resilience import (
    InProcessDistributedLock,
    RedisDistributedLock,
)

import pytest

pytestmark = pytest.mark.infra



async def test_in_process_lock_is_exclusive() -> None:
    lock = InProcessDistributedLock()
    assert await lock.acquire('doc-1') is True
    assert await lock.acquire('doc-1') is False
    await lock.release('doc-1')
    assert await lock.acquire('doc-1') is True


async def test_in_process_lock_namespaces_are_independent() -> None:
    lock = InProcessDistributedLock()
    assert await lock.acquire('doc-1') is True
    assert await lock.acquire('doc-2') is True


async def test_redis_lock_acquires_with_nx_and_ttl() -> None:
    client = MagicMock()
    client.set = AsyncMock(return_value=True)
    lock = RedisDistributedLock(client)

    assert await lock.acquire('doc-1', ttl=300) is True

    args = client.set.call_args.args
    assert args[0] == 'lock:doc-1'
    token = args[1]
    assert isinstance(token, str) and len(token) == 32
    kwargs = client.set.call_args.kwargs
    assert kwargs['nx'] is True
    assert kwargs['ex'] == 300


async def test_redis_lock_rejects_when_held_by_another_worker() -> None:
    client = MagicMock()
    client.set = AsyncMock(return_value=None)  # NX failed: already held
    lock = RedisDistributedLock(client)

    assert await lock.acquire('doc-1') is False


async def test_redis_lock_release_uses_compare_and_delete() -> None:
    client = MagicMock()
    client.set = AsyncMock(return_value=True)
    client.eval = AsyncMock(return_value=1)
    lock = RedisDistributedLock(client)

    await lock.acquire('doc-1')
    await lock.release('doc-1')

    client.eval.assert_called_once()
    args = client.eval.call_args.args
    script, num_keys, key, token = args
    assert 'redis.call("get", KEYS[1]) == ARGV[1]' in script
    assert num_keys == 1
    assert key == 'lock:doc-1'
    assert isinstance(token, str)


async def test_redis_lock_release_unknown_lock_is_noop() -> None:
    client = MagicMock()
    client.eval = AsyncMock()
    lock = RedisDistributedLock(client)

    await lock.release('never-acquired')

    client.eval.assert_not_called()
