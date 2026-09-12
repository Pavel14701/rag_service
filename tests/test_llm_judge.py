
from collections.abc import Sequence

"""Tests for the faithfulness judge (LLM-as-a-judge + fallbacks)."""

import pytest

from evaluation.judge import (
    LexicalFaithfulnessJudge,
    LLMFaithfulnessJudge,
)

pytestmark = pytest.mark.evaluation


CONTEXTS = ["RAG combines retrieval with generation. Paris is the capital of France."]


class FakeLLM:
    def __init__(self, reply=None, error: bool = False) -> None:
        self.reply = reply
        self.error = error
        self.calls = []

    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise RuntimeError("llm down")
        return self.reply


async def test_json_faithful_true_scores_one() -> None:
    llm = FakeLLM('{"faithful": true}')
    judge = LLMFaithfulnessJudge(llm)
    assert await judge.judge("q", "a", CONTEXTS) == 1.0


async def test_json_faithful_false_scores_zero() -> None:
    llm = FakeLLM('Verdict: {"faithful": false}')
    judge = LLMFaithfulnessJudge(llm)
    assert await judge.judge("q", "a", CONTEXTS) == 0.0


async def test_json_score_is_clamped() -> None:
    llm = FakeLLM('{"score": 1.5}')
    judge = LLMFaithfulnessJudge(llm)
    assert await judge.judge("q", "a", CONTEXTS) == 1.0

    llm = FakeLLM('{"score": 0.8}')
    judge = LLMFaithfulnessJudge(llm)
    assert await judge.judge("q", "a", CONTEXTS) == 0.8


async def test_yes_no_verdicts() -> None:
    yes = LLMFaithfulnessJudge(FakeLLM("YES - fully grounded"))
    assert await yes.judge("q", "a", CONTEXTS) == 1.0
    no = LLMFaithfulnessJudge(FakeLLM("NO - hallucinated"))
    assert await no.judge("q", "a", CONTEXTS) == 0.0


async def test_unparsable_verdict_falls_back_to_lexical() -> None:
    llm = FakeLLM("bananas")
    judge = LLMFaithfulnessJudge(llm)
    lexical = LexicalFaithfulnessJudge()
    answer = "RAG combines retrieval with generation."
    expected = await lexical.judge("q", answer, CONTEXTS)
    assert await judge.judge("q", answer, CONTEXTS) == expected


async def test_llm_error_falls_back_to_lexical() -> None:
    llm = FakeLLM(error=True)
    judge = LLMFaithfulnessJudge(llm)
    lexical = LexicalFaithfulnessJudge()
    answer = "Paris is the capital of France."
    expected = await lexical.judge("q", answer, CONTEXTS)
    assert await judge.judge("q", answer, CONTEXTS) == expected


async def test_prompt_contains_question_answer_context() -> None:
    llm = FakeLLM('{"faithful": true}')
    judge = LLMFaithfulnessJudge(llm)
    await judge.judge("my question", "my answer", ["ctx one", "ctx two"])
    _, user_prompt = llm.calls[0]
    assert "my question" in user_prompt
    assert "my answer" in user_prompt
    assert "ctx one" in user_prompt and "ctx two" in user_prompt


async def test_lexical_judge_matches_metrics_faithfulness() -> None:
    from evaluation.metrics import faithfulness

    judge = LexicalFaithfulnessJudge(n=3)
    answer = "Paris is the capital of France."
    assert await judge.judge("q", answer, CONTEXTS) == faithfulness(
        answer, CONTEXTS, n=3
    )


async def test_runner_uses_judge_when_configured():
    from evaluation import EvalRunner

    class FakeJudge:
        def __init__(self) -> None:
            self.calls = []

        async def judge(self, question: str, answer: str, contexts: Sequence[str]):
            self.calls.append((question, answer, tuple(contexts)))
            return 0.42

    async def answer_fn(user_id: str, query: str, top_k: int | None = None):
        return {
            "answer": "grounded",
            "sources": [{"doc_id": "d1"}],
            "source_texts": ["source text"],
        }

    judge = FakeJudge()
    runner = EvalRunner(answer_fn, faithfulness_judge=judge)
    report = await runner.run([])  # empty run: wiring smoke only
    assert report.to_dict()["faithfulness"] == 0.0
    assert judge.calls == []
