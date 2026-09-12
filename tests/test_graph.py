"""Tests for GraphRAG: extraction adapter, Qdrant graph store, expansion."""

import uuid
from unittest.mock import MagicMock

import pytest

from application.interfaces import ExtractedRelation
from application.services import IndexerService, RetrieverService
from infrastructure.graph_extract import LLMEntityExtractor, NoOpEntityExtractor
from infrastructure.graph_store import QdrantGraphStore

from conftest import FakeEmbedding, FakeLLM, make_document

pytestmark = pytest.mark.retrieval


# ---------- fakes ----------


class FakeEntityExtractor:
    """EntityExtractor double returning fixed triples, recording calls."""

    def __init__(self, relations: list[ExtractedRelation]) -> None:
        self.relations = relations
        self.calls: list[str] = []

    async def extract(self, text: str) -> list[ExtractedRelation]:
        self.calls.append(text)
        return list(self.relations)


class FakeGraphStore:
    """GraphStore double recording writes, serving fixed neighbor docs."""

    def __init__(self, doc_ids: list[uuid.UUID] | None = None) -> None:
        self.added: list[tuple[uuid.UUID, list[ExtractedRelation]]] = []
        self.queries: list[list[str]] = []
        self.doc_ids = doc_ids or []

    async def add_relations(
        self, doc_id: uuid.UUID, relations: list[ExtractedRelation]
    ) -> None:
        self.added.append((doc_id, list(relations)))

    async def related_doc_ids(
        self, entity_names: list[str], max_hops: int = 1
    ) -> list[uuid.UUID]:
        self.queries.append(list(entity_names))
        return list(self.doc_ids)


class FakeQdrantPoint:
    def __init__(self, payload: dict) -> None:
        self.payload = payload


def _make_graph_store() -> tuple[QdrantGraphStore, MagicMock]:
    client = MagicMock()
    client.collection_exists.return_value = False
    client.retrieve.return_value = []
    store = QdrantGraphStore(
        client=client,
        entities_collection="docs__entities",
        embedding=FakeEmbedding(dim=4),
    )
    return store, client


# ---------- LLM extraction ----------


async def test_llm_extractor_parses_triples():
    llm = FakeLLM(
        '[{"subject": "Atlas", "predicate": "uses", '
        '"object": "LIDAR"}]'
    )
    triples = await LLMEntityExtractor(llm).extract("some text")
    assert triples == [
        ExtractedRelation(subject="Atlas", predicate="uses", obj="LIDAR")
    ]


async def test_llm_extractor_degrades_gracefully():
    # LLM failure -> empty list, never an exception
    broken = LLMEntityExtractor(FakeLLM("no json"))
    assert await broken.extract("text") == []

    class DownLLM(FakeLLM):
        async def generate(self, system_prompt, user_prompt, temperature=0.1):
            raise RuntimeError("down")

    assert await LLMEntityExtractor(DownLLM()).extract("text") == []


async def test_noop_extractor_returns_empty():
    assert await NoOpEntityExtractor().extract("text") == []


# ---------- Qdrant graph store ----------


async def test_add_relations_creates_entity_points():
    store, client = _make_graph_store()
    doc_id = uuid.uuid4()

    await store.add_relations(
        doc_id,
        [ExtractedRelation(subject="Atlas", predicate="uses", obj="LIDAR")],
    )

    # collection ensured once with the embedding dimension
    client.create_collection.assert_called_once()
    # one upsert per entity point: subject and object
    assert client.upsert.call_count == 2
    points = [
        p
        for call in client.upsert.call_args_list
        for p in call.kwargs["points"]
    ]
    names = {p.payload["name"] for p in points}
    assert names == {"Atlas", "LIDAR"}
    subject = next(p for p in points if p.payload["name"] == "Atlas")
    assert subject.payload["relations"] == [
        {"predicate": "uses", "object": "LIDAR"}
    ]
    assert subject.payload["doc_ids"] == [str(doc_id)]


async def test_add_relations_merges_doc_ids_without_duplicates():
    store, client = _make_graph_store()
    client.collection_exists.return_value = True
    doc_id = uuid.uuid4()
    point_id = uuid.uuid5(
        uuid.NAMESPACE_URL, "graph-entity:atlas"
    )
    client.retrieve.return_value = [
        FakeQdrantPoint(
            {
                "name": "Atlas",
                "doc_ids": [str(doc_id)],
                "relations": [
                    {"predicate": "uses", "object": "LIDAR"}
                ],
            }
        )
    ]

    await store.add_relations(
        doc_id,
        [ExtractedRelation(subject="Atlas", predicate="uses", obj="LIDAR")],
    )

    points = client.upsert.call_args.kwargs["points"]
    payload = points[0].payload
    assert payload["doc_ids"] == [str(doc_id)]  # deduplicated
    assert payload["relations"] == [  # relation not duplicated
        {"predicate": "uses", "object": "LIDAR"}
    ]


async def test_related_doc_ids_traverses_hops():
    store, client = _make_graph_store()
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    hop1 = {
        "name": "Atlas",
        "doc_ids": [str(doc_a)],
        "relations": [{"predicate": "uses", "object": "LIDAR"}],
    }
    hop2 = {"name": "LIDAR", "doc_ids": [str(doc_b)], "relations": []}
    client.scroll.return_value = ([FakeQdrantPoint(hop1)], None)

    # hop 1: only direct doc ids
    client.scroll.side_effect = [([FakeQdrantPoint(hop1)], None)]
    ids = await store.related_doc_ids(["atlas"], max_hops=1)
    assert set(ids) == {doc_a}

    # hop 2: neighbors' docs too (LIDAR reached via the relation)
    client.scroll.side_effect = [
        ([FakeQdrantPoint(hop1)], None),
        ([FakeQdrantPoint(hop2)], None),
    ]
    ids = await store.related_doc_ids(["atlas"], max_hops=2)
    assert set(ids) == {doc_a, doc_b}


async def test_related_doc_ids_skips_malformed_ids():
    store, client = _make_graph_store()
    client.scroll.return_value = (
        [FakeQdrantPoint({"name": "x", "doc_ids": ["not-a-uuid"]})],
        None,
    )
    assert await store.related_doc_ids(["x"]) == []


# ---------- retrieval expansion ----------


async def test_retriever_expands_via_graph(repo, embedding, llm):
    from conftest import FakeVectorStore

    neighbor_doc = uuid.uuid4()

    class MultiResultVectorStore(FakeVectorStore):
        async def search(self, vector, top_k, filter_condition=None, keyword_query=None):
            self.searches.append({"filter": filter_condition})
            if filter_condition and "should" in filter_condition and any(
                c.get("key") == "doc_id" for c in filter_condition["should"]
            ):
                return [
                    {
                        "id": "n1",
                        "score": 0.4,
                        "payload": {"doc_id": str(neighbor_doc)},
                        "text": "neighbor chunk",
                    }
                ]
            return [
                {
                    "id": "c1",
                    "score": 0.9,
                    "payload": {"doc_id": "11111111-1111-1111-1111-111111111111"},
                    "text": "direct hit",
                }
            ]

    vector_store = MultiResultVectorStore()
    extractor = FakeEntityExtractor(
        [ExtractedRelation(subject="Atlas", predicate="uses", obj="LIDAR")]
    )
    graph = FakeGraphStore(doc_ids=[neighbor_doc])
    service = RetrieverService(
        vector_store,
        repo,
        embedding,
        llm,
        entity_extractor=extractor,
        graph_store=graph,
    )

    result = await service.answer_query("u1", "what does Atlas use")

    # the extractor saw the query, the graph was queried for its entities
    assert len(extractor.calls) == 1
    assert graph.queries == [["Atlas", "LIDAR"]]
    # two searches: the main one + the graph-expansion one;
    # the neighbor chunk is merged into the sources
    assert len(vector_store.searches) == 2
    source_docs = {s["doc_id"] for s in result["sources"]}
    assert str(neighbor_doc) in source_docs
async def test_retriever_without_graph_stays_single_search(
    repo, vector_store, embedding, llm
):
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "t"}
    ]
    service = RetrieverService(vector_store, repo, embedding, llm)

    await service.answer_query("u1", "question")

    assert len(vector_store.searches) == 1
async def test_retriever_skips_already_retrieved_neighbors(
    repo, vector_store, embedding, llm
):
    extractor = FakeEntityExtractor(
        [ExtractedRelation(subject="A", predicate="r", obj="B")]
    )
    graph = FakeGraphStore(doc_ids=[uuid.uuid4()])
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "t"}
    ]
    service = RetrieverService(
        vector_store,
        repo,
        embedding,
        llm,
        entity_extractor=extractor,
        graph_store=graph,
    )

    result = await service.answer_query("u1", "question")

    # expansion ran, but the neighbor search returned the same chunk id,
    # which is filtered out: exactly one source, no duplicates
    assert len(extractor.calls) == 1
    assert len(graph.queries) == 1
    assert len(result["sources"]) == 1

# ---------- indexing hook ----------


class FakeParser:
    def __init__(self, elements):
        self.elements = elements

    def parse(self, file_path):
        return self.elements


class FakeSelector:
    def __init__(self, parser):
        self.parser = parser

    def get_parser(self, file_path):
        return self.parser


async def test_indexer_writes_relations_after_upsert(
    repo, vector_store, embedding, storage,
):
    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"
    elements = [{"text": "Atlas uses LIDAR.", "metadata": {"type": "text"}}]
    extractor = FakeEntityExtractor(
        [ExtractedRelation(subject="Atlas", predicate="uses", obj="LIDAR")]
    )
    graph = FakeGraphStore()
    indexer = IndexerService(
        storage, vector_store, repo, embedding, FakeSelector(FakeParser(elements)),
        entity_extractor=extractor,
        graph_store=graph,
    )

    await indexer.index_document(doc.id)

    assert len(graph.added) == 1
    added_doc, relations = graph.added[0]
    assert added_doc == doc.id
    assert relations[0].subject == "Atlas"
    assert vector_store.upserts  # normal indexing happened


async def test_indexer_survives_graph_failures(
    repo, vector_store, embedding, storage,
):
    class ExplodingGraph(FakeGraphStore):
        async def add_relations(self, doc_id, relations):
            raise RuntimeError("graph down")

    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"
    elements = [{"text": "some text", "metadata": {"type": "text"}}]
    extractor = FakeEntityExtractor(
        [ExtractedRelation(subject="A", predicate="r", obj="B")]
    )
    indexer = IndexerService(
        storage, vector_store, repo, embedding, FakeSelector(FakeParser(elements)),
        entity_extractor=extractor,
        graph_store=ExplodingGraph(),
    )

    await indexer.index_document(doc.id)  # must not raise
    assert vector_store.upserts

