"""Tests for agentic retrieval: grader loop, planner sub-queries, adapters."""

import pytest

from application.interfaces import GradeVerdict
from application.services import RetrieverService
from infrastructure.agentic import (
    LLMDocumentGrader,
    LLMQueryPlanner,
    NoOpDocumentGrader,
    NoOpQueryPlanner,
)

from conftest import FakeLLM, FakeVectorStore

pytestmark = pytest.mark.retrieval


# ---------- fakes ----------


class RecordingRewriter:
    """Rewriter double that records feedback and returns a fixed text."""

    def __init__(self, rewritten: str = "better question") -> None:
        self.rewritten = rewritten
        self.feedbacks: list[str | None] = []

    async def rewrite(self, query: str, feedback: str | None = None) -> str:
        self.feedbacks.append(feedback)
        return self.rewritten


class ScriptedGrader:
    """DocumentGrader double returning queued verdicts in order."""

    def __init__(self, verdicts: list[GradeVerdict]) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[tuple[str, int]] = []  # (query, n_hits)

    async def grade(self, query: str, hits: list[dict]) -> GradeVerdict:
        self.calls.append((query, len(hits)))
        return self.verdicts.pop(0)


class StaticPlanner:
    """QueryPlanner double returning fixed sub-questions."""

    def __init__(self, subqueries: list[str]) -> None:
        self.subqueries = subqueries
        self.calls: list[str] = []

    async def plan(self, query: str) -> list[str]:
        self.calls.append(query)
        return self.subqueries


class MultiResultVectorStore(FakeVectorStore):
    """VectorStore double returning different hits per filter condition."""

    def __init__(self) -> None:
        super().__init__()
        self.results_by_filter: list[tuple[dict | None, list[dict]]] = []

    async def search(self, vector, top_k, filter_condition=None, keyword_query=None):
        self.searches.append(
            {
                "vector": vector,
                "top_k": top_k,
                "filter_condition": filter_condition,
                "keyword_query": keyword_query,
            }
        )
        for cond, results in self.results_by_filter:
            if cond == filter_condition:
                return list(results)
        return list(self.search_results)


def _service(vector_store, repo, embedding, llm, **kwargs) -> RetrieverService:
    return RetrieverService(
        vector_store, repo, embedding, llm, **kwargs
    )


# ---------- grader loop ----------


async def test_relevant_verdict_single_pass(repo, vector_store, embedding, llm):
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "t"}
    ]
    grader = ScriptedGrader([GradeVerdict(relevant=True, reason="ok")])
    service = _service(
        vector_store, repo, embedding, llm, document_grader=grader
    )

    result = await service.answer_query("u1", "question")

    assert len(vector_store.searches) == 1  # exactly one search round
    assert result["answer"] == "fake answer"


async def test_irrelevant_round_triggers_rewrite(repo, vector_store, embedding, llm):
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "t"}
    ]
    grader = ScriptedGrader(
        [
            GradeVerdict(relevant=False, reason="hits about the wrong topic"),
            GradeVerdict(relevant=True, reason="now relevant"),
        ]
    )
    rewriter = RecordingRewriter("better question")
    service = _service(
        vector_store,
        repo,
        embedding,
        llm,
        query_rewriter=rewriter,
        document_grader=grader,
        agentic_max_rounds=3,
    )

    result = await service.answer_query("u1", "original question")

    # exactly two search rounds: bad one, then the corrective one
    assert len(vector_store.searches) == 2
    # the rewriter received the grader reason as feedback on round 2
    assert rewriter.feedbacks == [None, "hits about the wrong topic"]
    # the grader saw hits on both rounds
    assert [n for _, n in grader.calls] == [1, 1]
    assert result["answer"] == "fake answer"


async def test_rounds_exhausted_answers_from_best_hits(
    repo, vector_store, embedding, llm
):
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "t"}
    ]
    grader = ScriptedGrader(
        [
            GradeVerdict(relevant=False, reason="bad"),
            GradeVerdict(relevant=False, reason="still bad"),
        ]
    )
    rewriter = RecordingRewriter("rewrite attempt")
    service = _service(
        vector_store,
        repo,
        embedding,
        llm,
        query_rewriter=rewriter,
        document_grader=grader,
        agentic_max_rounds=2,
    )

    result = await service.answer_query("u1", "question")

    assert len(vector_store.searches) == 2  # exhausted, no third round
    # a real answer is still produced from the best available hits
    assert result["answer"] == "fake answer"


async def test_no_grader_is_single_pass(repo, vector_store, embedding, llm):
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "t"}
    ]
    service = _service(
        vector_store, repo, embedding, llm, agentic_max_rounds=5
    )

    await service.answer_query("u1", "question")

    assert len(vector_store.searches) == 1


# ---------- planner sub-queries ----------


async def test_planner_splits_and_searches_in_parallel(
    repo, vector_store, embedding, llm
):
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "t"}
    ]
    planner = StaticPlanner(["sub question a", "sub question b"])
    service = _service(
        vector_store, repo, embedding, llm, query_planner=planner
    )

    result = await service.answer_query("u1", "complex question")

    # both sub-queries embedded and searched
    assert embedding.query_calls[-1] == ["sub question a", "sub question b"]
    assert len(vector_store.searches) == 2
    assert result["answer"] == "fake answer"


async def test_planner_respects_subquery_limit(
    repo, vector_store, embedding, llm
):
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {}, "text": "t"}
    ]
    planner = StaticPlanner(["a", "b", "c", "d"])
    service = _service(
        vector_store,
        repo,
        embedding,
        llm,
        query_planner=planner,
        agentic_subquery_limit=2,
    )

    await service.answer_query("u1", "question")

    assert embedding.query_calls[-1] == ["a", "b"]  # only the first two


# ---------- LLM adapters ----------


class EchoLLM:
    def __init__(self, reply: str | None) -> None:
        self.reply = reply

    async def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.1
    ) -> str:
        if self.reply is None:
            raise RuntimeError("llm down")
        return self.reply


async def test_llm_grader_parses_json_verdict():
    grader = LLMDocumentGrader(
        EchoLLM('{"relevant": false, "reason": "off topic"}')
    )
    verdict = await grader.grade(
        "q", [{"text": "chunk", "score": 0.9}]
    )
    assert verdict.relevant is False
    assert verdict.reason == "off topic"


async def test_llm_grader_falls_back_to_threshold_on_llm_failure():
    grader = LLMDocumentGrader(EchoLLM(None), fallback_threshold=0.5)
    above = await grader.grade("q", [{"text": "c", "score": 0.9}])
    below = await grader.grade("q", [{"text": "c", "score": 0.1}])
    assert above.relevant is True
    assert "fallback" in above.reason
    assert below.relevant is False


async def test_llm_grader_empty_hits_never_relevant():
    grader = LLMDocumentGrader(EchoLLM('{"relevant": true}'))
    verdict = await grader.grade("q", [])
    assert verdict.relevant is False




async def test_llm_grader_unparsable_reply_falls_back():
    grader = LLMDocumentGrader(EchoLLM("no json here"), fallback_threshold=0.5)
    verdict = await grader.grade("q", [{"text": "c", "score": 0.9}])
    assert verdict.relevant is True  # threshold fallback


async def test_llm_planner_parses_json_list():
    planner = LLMQueryPlanner(EchoLLM('["one", "two"]'))
    assert await planner.plan("complex q") == ["one", "two"]


async def test_llm_planner_falls_back_to_original():
    broken = LLMQueryPlanner(EchoLLM(None))
    assert await broken.plan("original q") == ["original q"]

    unparsable = LLMQueryPlanner(EchoLLM("sure thing"))
    assert await unparsable.plan("original q") == ["original q"]

    empty = LLMQueryPlanner(EchoLLM("[]"))
    assert await empty.plan("original q") == ["original q"]


async def test_noop_adapters_are_pass_through():
    assert (await NoOpDocumentGrader().grade("q", [])).relevant is True
    assert await NoOpQueryPlanner().plan("q") == ["q"]
