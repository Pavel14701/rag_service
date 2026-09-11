"""Tests for RetrieverService."""

import pytest

from application.services.retriever import RetrieverService

from conftest import make_document


@pytest.fixture
def service(repo, vector_store, embedding, llm):
    return RetrieverService(vector_store, repo, embedding, llm)


async def test_empty_query_returns_fallback(service, llm):
    result = await service.answer_query(user_id="u1", query="   ")

    assert result["sources"] == []
    assert "non-empty" in result["answer"]
    assert llm.calls == []


async def test_no_context_returns_no_info_answer(service, repo, vector_store, llm):
    result = await service.answer_query(user_id="u1", query="question")

    assert result["answer"] == "I don't have enough information to answer that."
    assert result["sources"] == []
    assert llm.calls == []


async def test_answer_with_context(service, repo, vector_store, embedding, llm):
    doc = make_document(owner_id="u1")
    await repo.save(doc)
    vector_store.search_results = [
        {"id": "chunk-1", "score": 0.9, "payload": {"doc_id": str(doc.id), "page": 1},
         "text": "relevant text"}
    ]

    result = await service.answer_query(user_id="u1", query="question")

    assert result["answer"] == "fake answer"
    assert result["sources"] == [{"doc_id": str(doc.id), "chunk_id": "chunk-1", "page": 1}]
    # LLM called with context and question
    assert len(llm.calls) == 1
    assert "relevant text" in llm.calls[0]["user_prompt"]
    assert "question" in llm.calls[0]["user_prompt"]
    # Conversation persisted
    assert len(repo.conversations) == 1
    assert repo.conversations[0]["user_id"] == "u1"


async def test_owner_filter_passed_to_vector_store(service, repo, vector_store):
    await service.answer_query(user_id="u1", query="question")

    assert len(vector_store.searches) == 1
    filt = vector_store.searches[0]["filter_condition"]
    assert filt == {"key": "owner_id", "match": {"value": "u1"}}


async def test_access_group_filter_uses_should_clause(service, repo, vector_store):
    repo.user_groups["u1"] = ["team-a", "team-b"]

    await service.answer_query(user_id="u1", query="question")

    filt = vector_store.searches[0]["filter_condition"]
    assert set(filt.keys()) == {"should"}
    conditions = filt["should"]
    owner_match = [c for c in conditions if c["key"] == "owner_id"]
    group_match = [c for c in conditions if c["key"] == "access_group"]
    assert owner_match and owner_match[0]["match"]["value"] == "u1"
    assert group_match and group_match[0]["match"]["value"] == ["team-a", "team-b"]


async def test_temperature_zero_not_overridden(service, repo, vector_store, llm):
    """Regression: temperature=0.0 must not fall back to the default (falsy `or`)."""
    vector_store.search_results = [
        {"id": "c", "score": 0.9, "payload": {"doc_id": "d"}, "text": "t"}
    ]
    await service.answer_query(user_id="u1", query="q", temperature=0.0)
    assert llm.calls[0]["temperature"] == 0.0


async def test_top_k_explicit_override(service, repo, vector_store):
    await service.answer_query(user_id="u1", query="q", top_k=3)
    assert vector_store.searches[0]["top_k"] == 3


async def test_top_k_zero_not_overridden(service, repo, vector_store):
    """Regression: top_k=0 must be passed through, not replaced by default."""
    await service.answer_query(user_id="u1", query="q", top_k=0)
    assert vector_store.searches[0]["top_k"] == 0


async def test_query_embedding_generated(service, vector_store, embedding):
    await service.answer_query(user_id="u1", query="find me")
    assert embedding.calls == [["find me"]]


async def test_explicit_user_groups_override_repo(service, repo, vector_store):
    result = await service.answer_query(
        user_id="u1", query="question", user_groups=["team-a"]
    )

    filt = vector_store.searches[0]["filter_condition"]
    assert filt["should"][1] == {
        "key": "access_group",
        "match": {"value": ["team-a"]},
    }
    # groups from the argument, not from the repo membership table
    assert repo.user_groups.get("u1") is None
    assert result["answer"] == "I don't have enough information to answer that."


async def test_empty_user_groups_means_owner_only_without_repo_lookup(
    service, repo, vector_store
):
    repo.user_groups["u1"] = ["team-a"]
    await service.answer_query(user_id="u1", query="question", user_groups=[])

    filt = vector_store.searches[0]["filter_condition"]
    # explicit empty list wins over the membership table
    assert filt == {"key": "owner_id", "match": {"value": "u1"}}


async def test_owner_document_accessible_even_when_not_in_doc_group(service, repo, vector_store):
    """Regression: access filter must be OR (owner OR group), not AND —
    an owner must not lose access to their own document just because
    they are not a member of the document's access group."""
    await service.answer_query(user_id="u1", query="question", user_groups=["team-a"])

    filt = vector_store.searches[0]["filter_condition"]
    assert set(filt) == {"should"}  # OR-semantics, no "must"
    assert filt["should"][0] == {"key": "owner_id", "match": {"value": "u1"}}


async def test_per_request_llm_routing(service, repo, vector_store, llm):
    doc = make_document(owner_id="u1")
    await repo.save(doc)
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {"doc_id": str(doc.id)}, "text": "t"}
    ]

    await service.answer_query(
        user_id="u1", query="question", llm_provider="openai", llm_model="gpt-4o"
    )

    assert llm.calls[-1]["provider"] == "openai"
    assert llm.calls[-1]["model"] == "gpt-4o"


async def test_no_routing_request_uses_default_generate(service, repo, vector_store, llm):
    doc = make_document(owner_id="u1")
    await repo.save(doc)
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {"doc_id": str(doc.id)}, "text": "t"}
    ]

    await service.answer_query(user_id="u1", query="question")

    assert llm.calls[-1]["provider"] is None  # default generate path



def test_system_prompt_rules(service):
    prompt = service._build_system_prompt()
    assert "ONLY information from the context" in prompt
    assert "I don't know" in prompt


async def test_query_uses_embed_query(service, embedding):
    await service.answer_query(user_id="u1", query="find me")
    assert embedding.query_calls == [["find me"]]
    assert embedding.passage_calls == []


async def test_hybrid_disabled_passes_no_keyword_query(repo, vector_store, embedding, llm):
    service = RetrieverService(vector_store, repo, embedding, llm, hybrid_enabled=False)
    await service.answer_query(user_id="u1", query="question")
    assert vector_store.searches[0]["keyword_query"] is None


async def test_hybrid_enabled_passes_keyword_query(repo, vector_store, embedding, llm):
    service = RetrieverService(
        vector_store, repo, embedding, llm, hybrid_enabled=True, hybrid_rrf_k=30
    )
    await service.answer_query(user_id="u1", query="question text")
    assert vector_store.searches[0]["keyword_query"] == "question text"


async def test_default_temperature_from_constructor(repo, vector_store, embedding, llm):
    service = RetrieverService(
        vector_store, repo, embedding, llm, default_temperature=0.7
    )
    vector_store.search_results = [
        {"id": "c", "score": 0.9, "payload": {"doc_id": "d"}, "text": "t"}
    ]
    await service.answer_query(user_id="u1", query="q")
    assert llm.calls[0]["temperature"] == 0.7