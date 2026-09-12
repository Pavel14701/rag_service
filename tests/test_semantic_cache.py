"""Tests for the near-duplicate (semantic) answer cache."""

import pytest

from application.services import RetrieverService
from infrastructure.caching import InMemorySemanticCache

from conftest import FakeEmbedding, FakeLLM

pytestmark = pytest.mark.retrieval


async def test_store_then_exact_lookup_hits() -> None:
    cache = InMemorySemanticCache(maxsize=8, ttl=60, threshold=0.95)
    vector = [0.1, 0.2, 0.3]

    await cache.store(vector, "cached answer")

    assert await cache.lookup(vector) == "cached answer"


async def test_near_identical_vector_hits_above_threshold() -> None:
    cache = InMemorySemanticCache(maxsize=8, ttl=60, threshold=0.95)
    await cache.store([1.0, 0.0, 0.0], "cached answer")

    # almost the same direction -> cosine ~0.999
    assert await cache.lookup([0.9999, 0.0001, 0.0]) == "cached answer"


def test_cosine_scale_invariant() -> None:
    cache = InMemorySemanticCache()
    # unnormalized embeddings: same direction, different magnitudes
    assert cache._cosine([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)


async def test_different_vector_misses() -> None:
    cache = InMemorySemanticCache(maxsize=8, ttl=60, threshold=0.95)
    await cache.store([1.0, 0.0], "cached answer")

    assert await cache.lookup([0.0, 1.0]) is None


async def test_disabled_cache_is_noop() -> None:
    cache = InMemorySemanticCache(enabled=False)
    await cache.store([1.0, 0.0], "answer")

    assert await cache.lookup([1.0, 0.0]) is None


async def test_maxsize_evicts_oldest() -> None:
    cache = InMemorySemanticCache(maxsize=1, ttl=60, threshold=0.5)
    await cache.store([1.0, 0.0], "first")
    await cache.store([0.0, 1.0], "second")

    # first entry was evicted beyond maxsize
    assert await cache.lookup([1.0, 0.0]) is None
    assert await cache.lookup([0.0, 1.0]) == "second"


async def test_expired_entries_are_dropped() -> None:
    clock_time = [100.0]
    cache = InMemorySemanticCache(
        ttl=10.0, threshold=0.5, clock=lambda: clock_time[0]
    )
    await cache.store([1.0, 0.0], "answer")
    clock_time[0] += 11.0  # past ttl

    assert await cache.lookup([1.0, 0.0]) is None
    assert len(cache._entries) == 0  # expired entry cleaned up


async def test_retriever_semantic_hit_skips_llm(
    repo, vector_store, embedding
):
    class FakeSemanticCache:
        def __init__(self) -> None:
            self.stored = []
            self.canned = "cached answer"

        async def lookup(self, vector: list[float]):
            return self.canned

        async def store(self, vector, answer) -> None:
            self.stored.append((vector, answer))

    llm = FakeLLM()
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "relevant"}
    ]
    cache = FakeSemanticCache()
    service = RetrieverService(
        vector_store, repo, embedding, llm, semantic_cache=cache
    )

    result = await service.answer_query("u1", "question")

    assert result["answer"] == "cached answer"
    assert llm.calls == []  # LLM was not called
    # sources still come from this user's own (ACL-filtered) search
    assert result["sources"][0]["chunk_id"] == "c1"


async def test_retriever_stores_answer_after_llm_generation(
    repo, vector_store, embedding, llm
):
    class FakeSemanticCache:
        def __init__(self) -> None:
            self.stored = []

        async def lookup(self, vector: list[float]):
            return None

        async def store(self, vector, answer) -> None:
            self.stored.append((vector, answer))

    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "relevant"}
    ]
    cache = FakeSemanticCache()
    service = RetrieverService(
        vector_store, repo, embedding, llm, semantic_cache=cache
    )

    result = await service.answer_query("u1", "question")

    assert result["answer"] == "fake answer"
    assert len(cache.stored) == 1
    assert cache.stored[0][1] == "fake answer"
    # the stored vector is the one the retriever computed for the query
    expected_vec = [float(len("question") % 10)] * embedding.dim
    assert cache.stored[0][0] == expected_vec


async def test_retriever_without_cache_unchanged(repo, vector_store, embedding, llm) -> None:
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "relevant"}
    ]
    service = RetrieverService(vector_store, repo, embedding, llm)

    result = await service.answer_query("u1", "question")

    assert result["answer"] == "fake answer"
    assert len(llm.calls) == 1
