"""Faithfulness judging: lexical heuristic vs LLM-as-a-judge.

The lexical n-gram heuristic (``metrics.faithfulness``) is cheap and
deterministic but cannot recognize paraphrases: a perfectly grounded
answer phrased differently scores low. An LLM judge scores grounding
semantically; it falls back to the lexical heuristic whenever the LLM
call fails or returns an unparsable verdict, so the offline eval
pipeline never breaks.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Sequence
from typing import Protocol

from application.interfaces import LLMGenerator

from .metrics import faithfulness as lexical_faithfulness


class FaithfulnessJudge(Protocol):
    """Grades whether an answer is grounded in the given contexts."""

    async def judge(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> float:
        """Return a faithfulness score in [0.0, 1.0]."""
        ...


class LexicalFaithfulnessJudge:
    """The original n-gram overlap heuristic (no LLM, deterministic)."""

    def __init__(self, n: int = 3) -> None:
        self._n = n

    async def judge(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> float:
        """Lexical grounding score of the answer against the contexts."""
        return lexical_faithfulness(answer, contexts, n=self._n)


class LLMFaithfulnessJudge:
    """LLM-as-a-judge with graceful fallback to the lexical heuristic."""

    _SYSTEM_PROMPT = (
        'You are a strict faithfulness judge for a retrieval-augmented '
        'system. Given a question, an answer and the retrieved context, '
        'decide whether every claim in the answer is supported by the '
        'context. An answer that refuses to answer (e.g. "I don\'t have '
        'enough information") counts as faithful. Reply with a JSON '
        'object ONLY, either {"faithful": true} / {"faithful": false} '
        'or {"score": <number between 0 and 1>}.'
    )

    def __init__(
        self, llm: LLMGenerator, fallback: FaithfulnessJudge | None = None
    ) -> None:
        self._llm = llm
        self._fallback = fallback or LexicalFaithfulnessJudge()

    async def judge(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> float:
        """Judge grounding; fall back to the lexical heuristic on failure."""
        context_block = (
            '\n'.join(f'- {context}' for context in contexts) or '(empty)'
        )
        user_prompt = (
            f'Question:\n{question}\n\n'
            f'Answer:\n{answer}\n\n'
            f'Context:\n{context_block}\n\n'
            'Verdict (JSON only):'
        )
        try:
            raw = await self._llm.generate(
                self._SYSTEM_PROMPT, user_prompt, 0.0
            )
        except Exception:  # noqa: BLE001 - eval must not break on LLM errors
            return await self._fallback.judge(question, answer, contexts)
        score = self._parse_verdict(raw)
        if score is None:
            return await self._fallback.judge(question, answer, contexts)
        return score

    @staticmethod
    def _parse_verdict(raw: str) -> float | None:
        """Extract a [0, 1] score from an LLM verdict; None if unparsable."""
        text = raw.strip()
        if match := re.search(r'\{.*\}', text, re.DOTALL):
            try:
                data = json.loads(match[0])
            except ValueError:
                data = None
            if isinstance(data, dict):
                if 'score' in data:
                    with contextlib.suppress(TypeError, ValueError):
                        return max(0.0, min(1.0, float(data['score'])))
                if 'faithful' in data:
                    return 1.0 if data['faithful'] else 0.0
        upper = text.upper()
        if 'YES' in upper:
            return 1.0
        return 0.0 if 'NO' in upper else None
