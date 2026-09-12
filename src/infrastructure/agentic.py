"""Agentic retrieval adapters: hit grading and query planning.

All adapters degrade gracefully - the retrieval loop must never break
because of an auxiliary LLM call:

- ``LLMDocumentGrader`` - LLM verdict on hit relevance; falls back to a
  score threshold when the LLM is unavailable or unparsable;
- ``LLMQueryPlanner`` - splits a complex question into sub-questions;
  falls back to the original question;
- ``NoOpDocumentGrader`` / ``NoOpQueryPlanner`` - pass-through doubles
  used when the agentic mode is disabled.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from application.interfaces import GradeVerdict, LLMGenerator


class LLMDocumentGrader:
    """Grades retrieved hits via the LLM, with a threshold fallback."""

    _SYSTEM_PROMPT = (
        'You grade retrieval results for a question-answer system. Given a '
        'question and retrieved text chunks, decide whether the chunks '
        'contain enough information to answer the question. Reply with a '
        'JSON object ONLY: {"relevant": true, "reason": "..."} or '
        '{"relevant": false, "reason": "..."}.'
    )

    def __init__(
        self,
        llm: Any,
        fallback_threshold: float = 0.0,
    ) -> None:
        self._llm = llm
        # used when the LLM is unavailable: hits at/above the threshold
        # are considered relevant
        self._fallback_threshold = fallback_threshold

    async def grade(
        self, query: str, hits: Sequence[dict[str, Any]]
    ) -> GradeVerdict:
        """Return the relevance verdict; never raises."""
        if not hits:
            return GradeVerdict(
                relevant=False, reason='no hits retrieved', score=0.0
            )
        context = '\n\n'.join(
            f'[{i}] {hit.get("text", "")}'
            for i, hit in enumerate(hits, start=1)
        )
        user_prompt = f'Question:\n{query}\n\nChunks:\n{context}\n\nVerdict:'
        try:
            raw = await self._llm.generate(
                self._SYSTEM_PROMPT, user_prompt, 0.0
            )
            verdict = self._parse_verdict(raw)
        except Exception:  # noqa: BLE001 - grader must not break retrieval
            verdict = None
        if verdict is not None:
            return verdict
        # fallback: threshold on the best vector score
        best = max((float(hit.get('score') or 0) for hit in hits), default=0.0)
        gate = self._fallback_threshold
        relevant = gate > 0 and best >= gate
        reason = (
            'fallback: threshold gate'
            if relevant
            else 'fallback: below threshold'
        )
        return GradeVerdict(relevant=relevant, reason=reason, score=best)

    @staticmethod
    def _parse_verdict(raw: str) -> GradeVerdict | None:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return None
        if not isinstance(data, dict) or 'relevant' not in data:
            return None
        reason = str(data.get('reason', ''))
        score = float(data.get('score', 1.0 if data['relevant'] else 0.0))
        return GradeVerdict(
            relevant=bool(data['relevant']), reason=reason, score=score
        )


class NoOpDocumentGrader:
    """Pass-through grader used when grading is disabled."""

    async def grade(
        self, query: str, hits: Sequence[dict[str, Any]]
    ) -> GradeVerdict:
        """Always relevant: the loop exits after a single round."""
        return GradeVerdict(relevant=True, reason='grading disabled')


class LLMQueryPlanner:
    """Splits a question into sub-questions via the LLM."""

    _SYSTEM_PROMPT = (
        'You decompose a user question into independent search sub-questions. '
        'Reply with a JSON array of strings ONLY, e.g. '
        '["first sub-question", "second sub-question"]. One element when the '
        'question is already self-contained.'
    )

    def __init__(self, llm: LLMGenerator, temperature: float = 0.0) -> None:
        self._llm = llm
        self._temperature = temperature

    async def plan(self, query: str) -> list[str]:
        """Return sub-questions; the original on any failure."""
        try:
            raw = await self._llm.generate(
                self._SYSTEM_PROMPT, query, self._temperature
            )
        except Exception:  # noqa: BLE001 - planner must not break search
            return [query]
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            return [query]
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return [query]
        if not isinstance(data, list):
            return [query]
        questions = [str(q).strip() for q in data if str(q).strip()]
        return questions or [query]


class NoOpQueryPlanner:
    """Pass-through planner used when planning is disabled."""

    async def plan(self, query: str) -> list[str]:
        """Single sub-question: the query itself."""
        return [query]
