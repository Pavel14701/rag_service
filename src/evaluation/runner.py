"""Eval runner: executes retriever/generation against golden cases.

``answer_fn`` is any callable with the ``RetrieverService.answer_query``
signature shape::

    async def answer_fn(
        user_id: str, query: str, top_k: int | None = None
    ) -> dict

returning ``{"answer": str, "sources": [{"doc_id": ...}, ...]}``. Using
an injected callable makes A/B comparisons trivial: build two runner
instances over different configurations (e.g. hybrid on/off) and
compare the aggregated reports.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .dataset import EvalCase
from .metrics import (
    faithfulness,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    refusal_rate,
)

AnswerFn = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class CaseResult:
    """Per-case evaluation outcome."""

    query: str
    retrieved_doc_ids: list[str]
    answer: str
    error: str | None = None
    precision: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    hit: float = 0.0
    ndcg: float = 0.0
    faithfulness: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the JSON report."""
        return {
            'query': self.query,
            'retrieved_doc_ids': self.retrieved_doc_ids,
            'answer': self.answer,
            'error': self.error,
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'mrr': round(self.mrr, 4),
            'hit': round(self.hit, 4),
            'ndcg': round(self.ndcg, 4),
            'faithfulness': round(self.faithfulness, 4),
        }


@dataclass
class EvalReport:
    """Aggregated evaluation results over all cases."""

    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report, including per-case results."""
        return {
            'num_cases': len(self.cases),
            'precision': self._avg('precision'),
            'recall': self._avg('recall'),
            'mrr': self._avg('mrr'),
            'hit_rate': self._avg('hit'),
            'ndcg': self._avg('ndcg'),
            'faithfulness': self._avg('faithfulness'),
            'refusal_rate': refusal_rate([c.answer for c in self.cases]),
            'errors': sum(1 for c in self.cases if c.error),
            'cases': [c.to_dict() for c in self.cases],
        }

    def _avg(self, attr: str) -> float:
        if not self.cases:
            return 0.0
        values: list[float] = [float(getattr(c, attr)) for c in self.cases]
        return round(sum(values) / len(values), 4)


class EvalRunner:
    """Runs an answer function over golden cases and aggregates metrics."""

    def __init__(
        self,
        answer_fn: AnswerFn,
        top_k: int = 5,
        faithfulness_n: int = 3,
    ) -> None:
        self._answer_fn = answer_fn
        self._top_k = top_k
        self._faithfulness_n = faithfulness_n

    async def run(self, cases: Sequence[EvalCase]) -> EvalReport:
        """Run all cases sequentially and aggregate metrics."""
        results = [await self._run_case(case) for case in cases]
        return EvalReport(cases=results)

    async def run_concurrently(self, cases: Sequence[EvalCase]) -> EvalReport:
        """Run all cases concurrently and aggregate metrics."""
        results = list(
            await asyncio.gather(*(self._run_case(c) for c in cases))
        )
        return EvalReport(cases=results)

    async def _run_case(self, case: EvalCase) -> CaseResult:
        top_k = case.top_k if case.top_k is not None else self._top_k
        try:
            result = await self._answer_fn(
                user_id=case.user_id, query=case.query, top_k=top_k
            )
        except Exception as e:  # noqa: BLE001 - report errors as failed cases
            return CaseResult(
                query=case.query, retrieved_doc_ids=[], answer='', error=str(e)
            )

        answer = str(result.get('answer', ''))
        sources = result.get('sources') or []
        retrieved_doc_ids = [
            str(source.get('doc_id'))
            for source in sources
            if source.get('doc_id')
        ]
        source_texts = case.source_texts or [
            text for text in result.get('source_texts', []) if text
        ]

        return CaseResult(
            query=case.query,
            retrieved_doc_ids=retrieved_doc_ids,
            answer=answer,
            precision=precision_at_k(
                retrieved_doc_ids, case.relevant_doc_ids, top_k
            ),
            recall=recall_at_k(
                retrieved_doc_ids, case.relevant_doc_ids, top_k
            ),
            mrr=mrr(retrieved_doc_ids, case.relevant_doc_ids),
            hit=hit_rate(retrieved_doc_ids, case.relevant_doc_ids, top_k),
            ndcg=ndcg_at_k(retrieved_doc_ids, case.relevant_doc_ids, top_k),
            faithfulness=faithfulness(
                answer, source_texts, n=self._faithfulness_n
            ),
        )


def compare_reports(
    baseline: EvalReport, variant: EvalReport
) -> dict[str, float]:
    """Deltas of the variant against the baseline (variant - baseline)."""
    base, var = baseline.to_dict(), variant.to_dict()
    return {
        key: round(var[key] - base[key], 4)
        for key in (
            'precision',
            'recall',
            'mrr',
            'hit_rate',
            'ndcg',
            'faithfulness',
            'refusal_rate',
        )
    }
