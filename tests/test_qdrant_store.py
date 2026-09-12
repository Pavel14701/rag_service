"""Tests for QdrantStore filter conversion and client calls."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from qdrant_client.http import models

from infrastructure.vector_store import QdrantStore


from typing import Any, cast

pytestmark = pytest.mark.retrieval


def _first_condition(
    qfilter: models.Filter, clause: str = "must"
) -> models.FieldCondition:
    """First condition of a filter clause, narrowed for mypy."""
    conditions: Any = getattr(qfilter, clause)
    assert conditions is not None and len(conditions) == 1
    condition = conditions[0]
    assert isinstance(condition, models.FieldCondition)
    return condition


def make_store() -> tuple[QdrantStore, MagicMock]:
    client = MagicMock()
    return QdrantStore(client, "docs"), client


def test_missing_optional_import_regression() -> None:
    """Module imports fine (regression: `Optional` was not imported)."""
    import infrastructure.vector_store as mod
    assert hasattr(mod, "QdrantStore")


async def test_create_collection() -> None:
    store, client = make_store()
    await store.create_collection(384)
    kwargs = client.recreate_collection.call_args.kwargs
    assert kwargs["collection_name"] == "docs"
    assert kwargs["vectors_config"].size == 384


async def test_drop_collection() -> None:
    store, client = make_store()
    await store.drop_collection()
    client.delete_collection.assert_called_once_with(collection_name="docs")


async def test_upsert_builds_point_structs() -> None:
    store, client = make_store()
    await store.upsert(["id1"], [[0.1, 0.2]], [{"doc_id": "d"}])
    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].id == "id1"
    assert points[0].vector == [0.1, 0.2]
    assert points[0].payload == {"doc_id": "d"}


async def test_simple_filter_becomes_must_condition() -> None:
    store, client = make_store()
    client.query_points.return_value = SimpleNamespace(points=[])
    condition = {"key": "doc_id", "match": {"value": "abc"}}

    await store.search(vector=[0.1], top_k=5, filter_condition=condition)

    kwargs = client.query_points.call_args.kwargs
    qfilter = kwargs["query_filter"]
    assert isinstance(qfilter, models.Filter)
    condition = _first_condition(qfilter)
    assert condition.key == "doc_id"
    assert condition.match == models.MatchValue(value="abc")


async def test_list_value_becomes_match_any() -> None:
    store, _ = make_store()
    condition = {"key": "access_group", "match": {"value": ["a", "b"]}}
    qfilter = store._build_filter(condition)
    assert _first_condition(qfilter).match == models.MatchAny(any=["a", "b"])


async def test_should_filter_conversion() -> None:
    """Regression: retriever builds `should` filters that must be understood."""
    store, _ = make_store()
    condition = {
        "should": [
            {"key": "owner_id", "match": {"value": "u1"}},
            {"key": "access_group", "match": {"value": ["team-a"]}},
        ]
    }
    qfilter = store._build_filter(condition)
    assert qfilter.must is None
    should = qfilter.should
    assert should is not None and len(list(should)) == 2
    first = cast(models.FieldCondition, should[0])
    second = cast(models.FieldCondition, should[1])
    assert first.key == "owner_id"
    assert first.match == models.MatchValue(value="u1")
    assert second.match == models.MatchAny(any=["team-a"])


async def test_search_maps_hits_to_dicts() -> None:
    store, client = make_store()
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(id="p1", score=0.98, payload={"text": "hello", "doc_id": "d"}),
            SimpleNamespace(id="p2", score=0.42, payload=None),
        ]
    )

    hits = await store.search(vector=[0.1], top_k=2)

    assert hits[0] == {
        "id": "p1",
        "score": 0.98,
        "payload": {"text": "hello", "doc_id": "d"},
        "text": "hello",
    }
    assert hits[1]["text"] == ""
    assert hits[1]["payload"] == {}


async def test_delete_by_filter() -> None:
    store, client = make_store()
    await store.delete_by_filter({"key": "doc_id", "match": {"value": "abc"}})
    selector = client.delete.call_args.kwargs["points_selector"]
    assert isinstance(selector, models.FilterSelector)
    assert selector.filter.must[0].key == "doc_id"
    assert selector.filter.must[0].match == models.MatchValue(value="abc")


async def test_create_collection_with_hnsw_config() -> None:
    client = MagicMock()
    store = QdrantStore(client, "docs", hnsw_m=32, hnsw_ef_construct=200)
    await store.create_collection(384)
    kwargs = client.recreate_collection.call_args.kwargs
    hnsw = kwargs["hnsw_config"]
    assert isinstance(hnsw, models.HnswConfigDiff)
    assert hnsw.m == 32
    assert hnsw.ef_construct == 200


async def test_create_collection_without_hnsw_config() -> None:
    store, client = make_store()
    await store.create_collection(384)
    assert "hnsw_config" not in client.recreate_collection.call_args.kwargs


async def test_search_with_hnsw_ef_search_param() -> None:
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=[])
    store = QdrantStore(client, "docs", hnsw_ef=256)
    await store.search(vector=[0.1], top_k=5)
    kwargs = client.query_points.call_args.kwargs
    assert isinstance(kwargs["search_params"], models.SearchParams)
    assert kwargs["search_params"].hnsw_ef == 256


async def test_search_without_hnsw_ef_has_no_search_params() -> None:
    store, client = make_store()
    client.query_points.return_value = SimpleNamespace(points=[])
    await store.search(vector=[0.1], top_k=5)
    assert "search_params" not in client.query_points.call_args.kwargs

async def test_delete_by_doc_id_uses_doc_id_filter() -> None:
    import uuid as uuid_mod

    store, client = make_store()
    doc_id = uuid_mod.uuid4()

    await store.delete_by_doc_id(doc_id)

    kwargs = client.delete.call_args.kwargs
    assert kwargs["collection_name"] == "docs"
    selector = kwargs["points_selector"]
    condition = selector.filter.must[0]
    assert condition.key == "doc_id"
    assert condition.match == models.MatchValue(value=str(doc_id))


async def test_get_collection_dimension_reads_vectors_size() -> None:
    store, client = make_store()
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=384))
        )
    )

    assert await store.get_collection_dimension() == 384


async def test_get_collection_dimension_missing_collection_is_none() -> None:
    store, client = make_store()
    client.get_collection.side_effect = RuntimeError("collection not found")

    assert await store.get_collection_dimension() is None


async def test_get_collection_dimension_unexpected_shape_is_none() -> None:
    store, client = make_store()
    client.get_collection.return_value = SimpleNamespace(config=None)

    assert await store.get_collection_dimension() is None


async def test_search_hybrid_bm25_threshold_drops_weak_keyword_leg() -> None:
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(id="v1", score=0.9, payload={"text": "text"})]
    )
    client.scroll.return_value = (
        [SimpleNamespace(id="k1", score=None, payload={"text": "quantum physics"})],
        None,
    )
    store = QdrantStore(
        client, "docs", fulltext_enabled=True, bm25_score_threshold=10_000.0
    )

    hits = await store.search(vector=[0.1], top_k=2, keyword_query="quantum")

    # BM25 scores cannot pass the huge threshold: vector-only fallback
    assert [h["id"] for h in hits] == ["v1"]


async def test_search_hybrid_stopwords_cleaned_from_keyword_query() -> None:
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(id="v1", score=0.9, payload={"text": "text"})]
    )
    client.scroll.return_value = (
        [SimpleNamespace(id="k1", score=None, payload={"text": "garden foxes"})],
        None,
    )
    store = QdrantStore(
        client,
        "docs",
        fulltext_enabled=True,
        bm25_stopwords=frozenset({"ромашка"}),
    )

    hits = await store.search(vector=[0.1], top_k=2, keyword_query="Ромашка")

    # every token removed -> empty lexical query -> vector-only fallback
    assert [h["id"] for h in hits] == ["v1"]


def _aliases(entries: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        aliases=[
            SimpleNamespace(alias_name=k, collection_name=v)
            for k, v in entries.items()
        ]
    )


async def test_promote_creates_alias_when_missing() -> None:
    store, client = make_store()
    client.get_aliases.return_value = _aliases({})
    client.get_collections.return_value = SimpleNamespace(collections=[])

    shadow = await store.create_shadow_collection(384)
    assert shadow.startswith("docs__green_")

    await store.promote_shadow()

    op = client.update_collection_aliases.call_args.kwargs[
        "change_aliases_operations"
    ]
    assert op[0].create_alias.collection_name == shadow
    assert op[0].create_alias.alias_name == "docs"
    # nothing superseded -> nothing dropped
    client.delete_collection.assert_not_called()


async def test_promote_flips_existing_alias_atomically() -> None:
    store, client = make_store()
    client.get_aliases.return_value = _aliases({"docs": "docs_v1"})
    client.get_collections.return_value = SimpleNamespace(collections=[])

    shadow = await store.create_shadow_collection(384)
    await store.promote_shadow()

    op = client.update_collection_aliases.call_args.kwargs[
        "change_aliases_operations"
    ]
    assert isinstance(op[0], models.DeleteAliasOperation)
    assert isinstance(op[1], models.CreateAliasOperation)
    assert op[1].create_alias.collection_name == shadow
    # superseded collection is dropped after the flip
    client.delete_collection.assert_called_with(collection_name="docs_v1")


async def test_promote_physical_collection_conflict_raises() -> None:
    store, client = make_store()
    client.get_aliases.return_value = _aliases({})
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="docs")]
    )
    await store.create_shadow_collection(384)

    with pytest.raises(RuntimeError, match="physical collection"):
        await store.promote_shadow()


async def test_promote_without_shadow_raises() -> None:
    store, _ = make_store()
    with pytest.raises(ValueError, match="No shadow"):
        await store.promote_shadow()


async def test_discard_shadow_drops_without_promoting() -> None:
    store, client = make_store()
    shadow = await store.create_shadow_collection(384)

    await store.discard_shadow()

    client.delete_collection.assert_called_with(collection_name=shadow)
    with pytest.raises(ValueError):
        await store.promote_shadow()


async def test_writes_are_dual_routed_into_shadow() -> None:
    store, client = make_store()
    shadow = await store.create_shadow_collection(384)

    await store.upsert(["id1"], [[0.1]], [{"doc_id": "d"}])

    targets = [
        call.kwargs["collection_name"]
        for call in client.upsert.call_args_list
    ]
    assert targets == ["docs", shadow]
