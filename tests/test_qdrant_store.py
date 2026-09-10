"""Tests for QdrantStore filter conversion and client calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from qdrant_client.http import models

from infrastructure.vector_store.qdrant_store import QdrantStore


def make_store() -> tuple[QdrantStore, MagicMock]:
    client = MagicMock()
    return QdrantStore(client, "docs"), client


def test_missing_optional_import_regression():
    """Module imports fine (regression: `Optional` was not imported)."""
    import infrastructure.vector_store.qdrant_store as mod
    assert hasattr(mod, "QdrantStore")


async def test_create_collection():
    store, client = make_store()
    await store.create_collection(384)
    kwargs = client.recreate_collection.call_args.kwargs
    assert kwargs["collection_name"] == "docs"
    assert kwargs["vectors_config"].size == 384


async def test_drop_collection():
    store, client = make_store()
    await store.drop_collection()
    client.delete_collection.assert_called_once_with(collection_name="docs")


async def test_upsert_builds_point_structs():
    store, client = make_store()
    await store.upsert(["id1"], [[0.1, 0.2]], [{"doc_id": "d"}])
    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].id == "id1"
    assert points[0].vector == [0.1, 0.2]
    assert points[0].payload == {"doc_id": "d"}


async def test_simple_filter_becomes_must_condition():
    store, client = make_store()
    client.query_points.return_value = SimpleNamespace(points=[])
    condition = {"key": "doc_id", "match": {"value": "abc"}}

    await store.search(vector=[0.1], top_k=5, filter_condition=condition)

    kwargs = client.query_points.call_args.kwargs
    qfilter = kwargs["query_filter"]
    assert isinstance(qfilter, models.Filter)
    assert len(qfilter.must) == 1
    assert qfilter.must[0].key == "doc_id"
    assert qfilter.must[0].match == models.MatchValue(value="abc")


async def test_list_value_becomes_match_any():
    store, _ = make_store()
    condition = {"key": "access_group", "match": {"value": ["a", "b"]}}
    qfilter = store._build_filter(condition)
    assert qfilter.must[0].match == models.MatchAny(any=["a", "b"])


async def test_should_filter_conversion():
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
    assert len(qfilter.should) == 2
    assert qfilter.should[0].key == "owner_id"
    assert qfilter.should[0].match == models.MatchValue(value="u1")
    assert qfilter.should[1].match == models.MatchAny(any=["team-a"])


async def test_search_maps_hits_to_dicts():
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


async def test_delete_by_filter():
    store, client = make_store()
    await store.delete_by_filter({"key": "doc_id", "match": {"value": "abc"}})
    selector = client.delete.call_args.kwargs["points_selector"]
    assert isinstance(selector, models.FilterSelector)
    assert selector.filter.must[0].key == "doc_id"
    assert selector.filter.must[0].match == models.MatchValue(value="abc")