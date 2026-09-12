"""Cache-then-validate: ACL-safe semantic cache behavior in RetrieverService."""

import hashlib

import pytest

from application.interfaces import CachedAnswer, CacheMeta
from application.services import RetrieverService

from conftest import FakeLLM

pytestmark = pytest.mark.retrieval


def _meta(chunk_ids=(), embedding_model='e5', llm_provider='', llm_model='', acl_key=''):
    return CacheMeta(
        chunk_ids=tuple(chunk_ids),
        embedding_model=embedding_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=0.1,
        acl_key=acl_key,
    )


class ScriptedCache:
    """SemanticCache double with a canned entry, recording stores."""

    def __init__(self, canned: CachedAnswer | None = None) -> None:
        self.canned = canned
        self.stored: list[tuple[list[float], str, CacheMeta]] = []

    async def lookup(self, vector: list[float]) -> CachedAnswer | None:
        return self.canned

    async def store(self, vector, answer, meta) -> None:
        self.stored.append((vector, answer, meta))


HIT = {"id": "c9", "score": 0.9, "payload": {}, "text": "relevant"}

ACL_FILTER = {
    "should": [
        {"key": "owner_id", "match": {"value": "u1"}},
        {"key": "access_group", "match": {"value": ["hr"]}},
    ]
}


def _service(vector_store, repo, embedding, llm, cache, **kwargs):
    return RetrieverService(
        vector_store,
        repo,
        embedding,
        llm,
        semantic_cache=cache,
        embedding_model=kwargs.pop("embedding_model", "e5"),
        **kwargs,
    )


def _seed(vector_store, payloads):
    """Seed the fake store so retrieve_by_ids sees the given payloads."""
    ids = [f"c{i}" for i in range(len(payloads))]
    vector_store.upserts.append(
        {"ids": ids, "vectors": [], "payloads": payloads}
    )
    for pid, payload in zip(ids, payloads):
        vector_store.payload_by_id[pid] = payload
    return ids


async def test_hit_when_all_chunks_visible(repo, vector_store, embedding, llm):
    vector_store.search_results = [HIT]
    ids = _seed(vector_store, [{"owner_id": "u1"}, {"owner_id": "u1"}])
    cache = ScriptedCache(
        CachedAnswer(answer="cached", meta=_meta(chunk_ids=ids))
    )
    service = _service(vector_store, repo, embedding, llm, cache)

    result = await service.answer_query("u1", "question")

    assert result["answer"] == "cached"
    assert llm.calls == []  # validated hit: no LLM call


async def test_miss_when_chunk_hidden_by_acl(repo, vector_store, embedding):
    llm = FakeLLM()
    # c1 belongs to another owner: hidden from u1 by the ACL filter
    vector_store.search_results = [HIT]
    _seed(vector_store, [{"owner_id": "u1"}, {"owner_id": "someone-else"}])
    cache = ScriptedCache(
        CachedAnswer(answer="cached", meta=_meta(chunk_ids=("c0", "c1")))
    )
    service = _service(vector_store, repo, embedding, llm, cache)

    result = await service.answer_query("u1", "question")

    # cache is not trusted: the full pipeline regenerates the answer
    assert result["answer"] == "fake answer"
    assert len(llm.calls) == 1


async def test_miss_on_embedding_model_mismatch(repo, vector_store, embedding, llm):
    vector_store.search_results = [HIT]
    cache = ScriptedCache(
        CachedAnswer(answer="cached", meta=_meta(chunk_ids=(), embedding_model="other-model"))
    )
    service = _service(vector_store, repo, embedding, llm, cache)

    result = await service.answer_query("u1", "question")

    assert result["answer"] == "fake answer"


async def test_miss_on_llm_routing_mismatch(repo, vector_store, embedding, llm):
    vector_store.search_results = [HIT]
    cache = ScriptedCache(
        CachedAnswer(answer="cached", meta=_meta(chunk_ids=(), llm_provider="openai"))
    )
    service = _service(vector_store, repo, embedding, llm, cache)

    # request without explicit routing: stored provider != '' -> miss
    result = await service.answer_query("u1", "question")
    assert result["answer"] == "fake answer"


async def test_hit_respects_explicit_routing(repo, vector_store, embedding, llm):
    vector_store.search_results = [HIT]
    cache = ScriptedCache(
        CachedAnswer(
            answer="cached",
            meta=_meta(chunk_ids=(), llm_provider="openai", llm_model="gpt-x"),
        )
    )
    service = _service(vector_store, repo, embedding, llm, cache)

    result = await service.answer_query(
        "u1", "question", llm_provider="openai", llm_model="gpt-x"
    )

    assert result["answer"] == "cached"


async def test_strict_acl_requires_group_set_match(repo, vector_store, embedding, llm):
    # groups of the requester: hr -> acl_key fingerprint of ["hr"]
    hr_key = hashlib.sha256(b"hr").hexdigest()
    legal_key = hashlib.sha256(b"legal").hexdigest()

    vector_store.search_results = [HIT]
    cache = ScriptedCache(
        CachedAnswer(answer="cached", meta=_meta(chunk_ids=(), acl_key=legal_key))
    )
    service = _service(
        vector_store, repo, embedding, llm, cache, cache_strict_acl=True
    )
    repo.user_groups["u1"] = ["hr"]

    result = await service.answer_query("u1", "question")
    assert result["answer"] == "fake answer"  # group set mismatch -> miss

    # same group set -> hit (chunks empty, visibility trivially satisfied)
    vector_store.search_results = [HIT]
    cache2 = ScriptedCache(
        CachedAnswer(answer="cached", meta=_meta(chunk_ids=(), acl_key=hr_key))
    )
    service2 = _service(
        vector_store, repo, embedding, llm, cache2, cache_strict_acl=True
    )

    result2 = await service2.answer_query("u1", "question")
    assert result2["answer"] == "cached"


async def test_store_records_chunk_ids_and_acl_key(repo, vector_store, embedding, llm):
    _seed(vector_store, [{"owner_id": "u1"}])
    vector_store.search_results = [
        {"id": "c0", "score": 0.9, "payload": {"owner_id": "u1"}, "text": "t"},
        {"id": "c0", "score": 0.8, "payload": {"owner_id": "u1"}, "text": "t"},
    ]
    cache = ScriptedCache(canned=None)
    service = _service(
        vector_store, repo, embedding, llm, cache, cache_strict_acl=True
    )
    repo.user_groups["u1"] = ["hr", "legal"]

    await service.answer_query("u1", "question")

    assert len(cache.stored) == 1
    _vec, answer, meta = cache.stored[0]
    assert answer == "fake answer"
    assert meta.chunk_ids == ("c0",)  # deduplicated, actual chunk ids
    assert meta.embedding_model == "e5"
    expected_key = hashlib.sha256(b"hr,legal").hexdigest()
    assert meta.acl_key == expected_key


async def test_store_skipped_when_cache_hit_validated(repo, vector_store, embedding, llm):
    cache = ScriptedCache(
        CachedAnswer(answer="cached", meta=_meta(chunk_ids=()))
    )
    service = _service(vector_store, repo, embedding, llm, cache)

    await service.answer_query("u1", "question")

    assert cache.stored == []  # no re-store on a validated hit


async def test_broken_cache_never_breaks_answering(repo, vector_store, embedding, llm):
    class ExplodingCache:
        async def lookup(self, vector):
            raise RuntimeError("cache down")

        async def store(self, vector, answer, meta):
            raise RuntimeError("cache down")

    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "t"}
    ]
    service = _service(vector_store, repo, embedding, llm, ExplodingCache())

    result = await service.answer_query("u1", "question")

    assert result["answer"] == "fake answer"
