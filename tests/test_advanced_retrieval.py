"""Tests for advanced retrieval: query rewriting and parent-child merge."""

import pytest

from application.services import IndexerService, RetrieverService
from infrastructure.llm import LLMQueryRewriter, NoOpQueryRewriter

from conftest import FakeLLM, make_document

pytestmark = pytest.mark.retrieval


# ---------- Query rewriting ----------


class StaticRewriter:
    def __init__(self, rewritten: str = "rewritten question") -> None:
        self.rewritten = rewritten

    async def rewrite(self, query: str, feedback: str | None = None) -> str:
        return self.rewritten


async def test_rewriter_feeds_embedding_and_search(
    repo, vector_store, embedding, llm
):
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "t"}
    ]
    service = RetrieverService(
        vector_store,
        repo,
        embedding,
        llm,
        query_rewriter=StaticRewriter("normalized question"),
    )

    await service.answer_query("u1", "qestion about docs")

    # the rewritten query is what gets embedded
    assert embedding.query_calls[-1] == ["normalized question"]
    # the ORIGINAL question is persisted, not the rewritten one
    assert repo.conversations[-1]["query"] == "qestion about docs"


async def test_noop_rewriter_keeps_query(repo, vector_store, embedding, llm) -> None:
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "t"}
    ]
    service = RetrieverService(
        vector_store,
        repo,
        embedding,
        llm,
        query_rewriter=NoOpQueryRewriter(),
    )

    await service.answer_query("u1", "original question")

    assert embedding.query_calls[-1] == ["original question"]


async def test_llm_rewriter_strips_and_falls_back():
    class EchoLLM:
        def __init__(self, reply) -> None:
            self.reply = reply

        async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
            if self.reply is None:
                raise RuntimeError("llm down")
            return self.reply

    rewriter = LLMQueryRewriter(EchoLLM("  self-contained question  "))
    assert await rewriter.rewrite("q") == "self-contained question"

    fallback = LLMQueryRewriter(EchoLLM(None))
    assert await fallback.rewrite("original") == "original"


async def test_llm_rewriter_empty_reply_falls_back():
    class EmptyLLM:
        async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
            return "   "

    rewriter = LLMQueryRewriter(EmptyLLM())
    assert await rewriter.rewrite("original") == "original"


# ---------- Parent-Child indexing + auto-merge ----------


def _make_indexer(child_chars: int = 0) -> IndexerService:
    return IndexerService(
        file_storage=None,  # type: ignore[arg-type]
        vector_store=None,  # type: ignore[arg-type]
        repo=None,  # type: ignore[arg-type]
        embedding=None,  # type: ignore[arg-type]
        parser_selector=None,  # type: ignore[arg-type]
        child_chars=child_chars,
    )


def test_parent_child_indexing_attaches_parent_payload() -> None:
    indexer = _make_indexer(child_chars=40)
    doc = make_document()
    long_text = " ".join(f"word{i}" for i in range(12))  # ~90 chars
    elements = [{"text": long_text, "metadata": {"type": "text"}}]

    chunks = indexer._prepare_chunks(elements, doc.id, doc, max_tokens=120)

    assert len(chunks) == 2  # one parent split into two children
    parents = {c["metadata"]["parent_id"] for c in chunks}
    assert len(parents) == 1  # one parent element
    parent_text = chunks[0]["metadata"]["parent_text"]
    # each child carries the full parent context in its payload
    assert all(c["metadata"]["parent_text"] == parent_text for c in chunks)
    # the parent text covers the whole element
    assert "word0" in parent_text and "word11" in parent_text


def test_parent_child_disabled_keeps_flat_chunks() -> None:
    indexer = _make_indexer(child_chars=0)
    doc = make_document()
    elements = [{"text": "hello world", "metadata": {"type": "text"}}]

    chunks = indexer._prepare_chunks(elements, doc.id, doc)

    assert "parent_id" not in chunks[0]["metadata"]
    assert "parent_text" not in chunks[0]["metadata"]


async def test_auto_merge_collapses_sibling_children(
    repo, vector_store, embedding, llm
):
    parent_text = "full parent paragraph context"
    vector_store.search_results = [
        {
            "id": "child-1",
            "score": 0.9,
            "payload": {"parent_id": "p1", "parent_text": parent_text},
            "text": "child one snippet",
        },
        {
            "id": "child-2",
            "score": 0.8,
            "payload": {"parent_id": "p1", "parent_text": parent_text},
            "text": "child two snippet",
        },
        {
            "id": "other",
            "score": 0.7,
            "payload": {},
            "text": "unrelated chunk",
        },
    ]
    service = RetrieverService(vector_store, repo, embedding, llm)

    result = await service.answer_query("u1", "question")

    prompt = llm.calls[0]["user_prompt"]
    # parent context replaces child snippets, exactly once
    assert prompt.count(parent_text) == 1
    assert "child one snippet" not in prompt
    assert "unrelated chunk" in prompt
    # source points at the parent id
    chunk_ids = [s["chunk_id"] for s in result["sources"]]
    assert chunk_ids == ["p1", "other"]


async def test_auto_merge_keeps_best_child_order(
    repo, vector_store, embedding, llm
):
    hits = [
        {
            "id": "a",
            "score": 0.9,
            "payload": {"parent_id": "p1", "parent_text": "P1"},
            "text": "a",
        },
        {
            "id": "b",
            "score": 0.85,
            "payload": {},
            "text": "b",
        },
        {
            "id": "c",
            "score": 0.8,
            "payload": {"parent_id": "p2", "parent_text": "P2"},
            "text": "c",
        },
    ]

    merged = RetrieverService._auto_merge_parents(hits)

    assert [h["text"] for h in merged] == ["P1", "b", "P2"]
