"""Tests for hybrid search: BM25 ranking, RRF fusion, QdrantStore fusion."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from qdrant_client.http import models

from infrastructure.vector_store.qdrant_store import QdrantStore
from shared.hybrid import bm25_rank, rrf_fuse, tokenize


def test_tokenize_latin_cyrillic():
    tokens = tokenize("Привет, World 42!")
    assert tokens == ["привет", "world", "42"]


def test_bm25_ranks_relevant_doc_first():
    corpus = [
        "the quick brown fox jumps",
        "machine learning with vectors",
        "foxes in the garden",
    ]
    ranked = bm25_rank("fox", corpus)
    # "fox" appears exactly in doc 0; "foxes" is a different token
    assert ranked[0] == 0
    assert 1 not in ranked


def test_bm25_multi_term_query():
    corpus = ["vector database search", "database of recipes", "unrelated text"]
    ranked = bm25_rank("database vector", corpus)
    assert ranked[0] == 0  # matches both terms
    assert ranked[1] == 1  # matches one term
    assert 2 not in ranked


def test_bm25_empty_corpus():
    assert bm25_rank("query", []) == []


def test_rrf_fuse_prefers_top_of_both_lists():
    fused = rrf_fuse([["a", "b"], ["b", "c"]], k=60)
    # "b" appears in both lists -> first
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c"}


def test_rrf_fuse_respects_top_k():
    fused = rrf_fuse([["a", "b", "c"], ["c", "d"]], k=60, top_k=2)
    assert len(fused) == 2


def _point(pid, text, score=0.0):
    return SimpleNamespace(
        id=pid, score=score, payload={"text": text, "doc_id": "d"}, **{}
    )


def _hybrid_store(client) -> QdrantStore:
    return QdrantStore(
        client,
        "docs",
        fulltext_enabled=True,
        hybrid_candidates=10,
        hybrid_rrf_k=60,
    )


async def test_search_without_keyword_skips_scroll():
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=[_point("v1", "text")])
    store = _hybrid_store(client)
    hits = await store.search(vector=[0.1], top_k=1)
    assert [h["id"] for h in hits] == ["v1"]
    client.scroll.assert_not_called()


async def test_search_with_keyword_fuses_results():
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(id="v1", score=0.9, payload={"text": "quantum physics intro"}),
        ]
    )
    # keyword leg finds the doc about foxes (not in vector results)
    client.scroll.return_value = (
        [
            SimpleNamespace(id="k1", score=None, payload={"text": "quantum physics intro"}),
            SimpleNamespace(id="k2", score=None, payload={"text": "foxes in the garden"}),
        ],
        None,
    )
    store = _hybrid_store(client)

    hits = await store.search(vector=[0.1], top_k=2, keyword_query="quantum physics")

    ids = [h["id"] for h in hits]
    assert ids[0] == "v1"  # in both lists -> highest fused score
    assert ids == ["v1", "k1"]  # k2 has zero BM25 -> excluded
    assert len(hits) == 2
    # scroll filter combines access filter with MatchText condition
    scroll_filter = client.scroll.call_args.kwargs["scroll_filter"]
    text_conditions = [c for c in scroll_filter.must if c.key == "text"]
    assert len(text_conditions) == 1
    assert text_conditions[0].match.text == "quantum physics"


async def test_search_hybrid_degrades_to_vector_when_scroll_fails():
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(id="v1", score=0.9, payload={"text": "t"})]
    )
    client.scroll.side_effect = RuntimeError("full-text index missing")
    store = _hybrid_store(client)

    hits = await store.search(vector=[0.1], top_k=1, keyword_query="q")
    assert [h["id"] for h in hits] == ["v1"]


async def test_create_collection_creates_fulltext_index_when_enabled():
    client = MagicMock()
    store = _hybrid_store(client)
    await store.create_collection(384)
    kwargs = client.create_payload_index.call_args.kwargs
    assert kwargs["field_name"] == "text"
    assert kwargs["field_schema"].type == models.PayloadSchemaType.TEXT


async def test_create_collection_without_fulltext_index():
    client = MagicMock()
    store = QdrantStore(client, "docs")
    await store.create_collection(384)
    client.create_payload_index.assert_not_called()


async def test_scroll_first_payload():
    client = MagicMock()
    client.scroll.return_value = (
        [SimpleNamespace(id="p1", payload={"embedding_model": "e5"})],
        None,
    )
    store = QdrantStore(client, "docs")
    assert (await store.scroll_first_payload()) == {"embedding_model": "e5"}


async def test_scroll_first_payload_missing_collection_returns_none():
    client = MagicMock()
    client.scroll.side_effect = RuntimeError("not found")
    store = QdrantStore(client, "docs")
    assert (await store.scroll_first_payload()) is None


async def test_scroll_first_payload_empty_collection_returns_none():
    client = MagicMock()
    client.scroll.return_value = ([], None)
    store = QdrantStore(client, "docs")
    assert (await store.scroll_first_payload()) is None