"""Tests for offline RAG evaluation: metrics, dataset loading, runner."""

import json

import pytest

from evaluation.dataset import EvalCase, load_cases
from evaluation.metrics import (
    faithfulness,
    hit_rate,
    is_refusal,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    refusal_rate,
)
from evaluation.runner import EvalRunner, compare_reports

pytestmark = pytest.mark.evaluation


# ---------- retrieval metrics ----------


def test_precision_at_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert precision_at_k(retrieved, ["b", "d"], 4) == 0.5
    assert precision_at_k(retrieved, ["b", "d"], 2) == 0.5
    assert precision_at_k(retrieved, [], 3) == 0.0
    assert precision_at_k([], ["a"], 3) == 0.0


def test_recall_at_k() -> None:
    retrieved = ["a", "b"]
    assert recall_at_k(retrieved, ["a", "b", "c"], 2) == pytest.approx(2 / 3)
    # k=None/0 means "all retrieved"
    assert recall_at_k(retrieved, ["a", "b", "c"], 0) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, [], 2) == 0.0


def test_mrr() -> None:
    assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)
    assert mrr(["a", "x"], ["a"]) == 1.0
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_hit_rate() -> None:
    assert hit_rate(["x", "y", "a"], ["a"], 2) == 0.0
    assert hit_rate(["x", "y", "a"], ["a"], 3) == 1.0


def test_ndcg_at_k() -> None:
    # single relevant at rank 2: dcg=1/log2(3), idcg=1
    assert ndcg_at_k(["x", "a"], ["a"], 2) == pytest.approx(1 / 1.5849625, rel=1e-4)
    assert ndcg_at_k(["a", "x"], ["a"], 2) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], ["a"], 2) == 0.0


# ---------- answer metrics ----------


def test_faithfulness_grounded_vs_hallucinated() -> None:
    context = ["the capital of France is Paris and it is large"]
    grounded = "Paris is the capital of France"
    wild = "zebra quantum spaghetti moonlight"
    assert faithfulness(grounded, context, n=2) > 0.5
    assert faithfulness(wild, context, n=2) == 0.0
    assert faithfulness("", context, n=2) == 0.0


def test_refusal_detection() -> None:
    assert is_refusal("I don't have enough information to answer.")
    assert is_refusal("Sorry, I don't know that.")
    assert not is_refusal("The capital of France is Paris.")
    assert refusal_rate(["I don't know.", "Paris.", "not in the context"]) == pytest.approx(2 / 3)
    assert refusal_rate([]) == 0.0


# ---------- dataset ----------


def test_load_cases(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query": "q1",
                    "relevant_doc_ids": ["d1"],
                    "user_id": "u1",
                    "top_k": 3,
                },
                {"query": "q2"},
            ]
        ),
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert cases[0] == EvalCase(query="q1", relevant_doc_ids=["d1"], user_id="u1", top_k=3)
    assert cases[1].query == "q2"
    assert cases[1].user_id == "eval-user"


def test_load_cases_validates(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"no_query": True}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)


# ---------- runner ----------


class FakeRetriever:
    """Answer function returning fixed retrieval + answer per query."""

    def __init__(self, responses: dict[str, dict]) -> None:
        self._responses = responses

    async def __call__(self, user_id: str, query: str, top_k: int | None = None):
        return self._responses[query]


async def test_runner_computes_all_metrics() -> None:
    responses = {
        "good": {
            "answer": "Paris is the capital of France",
            "sources": [{"doc_id": "d1"}, {"doc_id": "d2"}],
        },
        "refusal": {
            "answer": "I don't have enough information",
            "sources": [],
        },
    }
    cases = [
        EvalCase(query="good", relevant_doc_ids=["d1"], source_texts=["Paris is the capital of France"]),
        EvalCase(query="refusal", relevant_doc_ids=["d1"]),
    ]
    report = await EvalRunner(
        FakeRetriever(responses),  # type: ignore[arg-type]
        top_k=2
    ).run(cases)
    d = report.to_dict()
    assert d["num_cases"] == 2
    # case1: 1 of top-2 relevant (0.5); case2: 0 of top-2 (0.0) -> 0.25
    assert d["precision"] == pytest.approx(0.25)
    assert d["recall"] == pytest.approx(0.5)
    assert d["mrr"] == 0.5  # rank 1 and 0
    assert d["hit_rate"] == 0.5
    assert d["faithfulness"] == pytest.approx(0.5)  # 1.0 and 0.0
    assert d["refusal_rate"] == 0.5
    assert d["errors"] == 0


async def test_runner_reports_errors_as_failed_cases() -> None:
    async def failing(user_id, query, top_k=None) -> None:
        raise RuntimeError("llm down")

    report = await EvalRunner(
        failing,  # type: ignore[arg-type]
    ).run([EvalCase(query="q")])
    d = report.to_dict()
    assert d["errors"] == 1
    assert d["cases"][0]["error"] == "llm down"


async def test_compare_reports_deltas() -> None:
    base_responses = {"q": {"answer": "a", "sources": [{"doc_id": "x"}]}}
    variant_responses = {"q": {"answer": "a", "sources": [{"doc_id": "d1"}]}}
    cases = [EvalCase(query="q", relevant_doc_ids=["d1"])]
    baseline = await EvalRunner(FakeRetriever(base_responses)).run(cases)
    variant = await EvalRunner(FakeRetriever(variant_responses)).run(cases)
    deltas = compare_reports(baseline, variant)
    assert deltas["hit_rate"] == 1.0
    assert deltas["recall"] == 1.0
    # precision@5: baseline 0.0 (irrelevant only), variant 1/5
    assert deltas["precision"] == pytest.approx(0.2)
