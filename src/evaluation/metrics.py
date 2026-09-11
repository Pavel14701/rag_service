"""Retrieval and answer quality metrics for offline RAG evaluation.

Relevance is expressed by document IDs: an eval case lists the document
IDs (``relevant_doc_ids``) that should be retrieved for a query; the
runner collects the IDs of the chunks returned by the retriever.

All functions are pure and dependency-free so they can be reused for
A/B comparisons of retrieval/generation configurations.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)

_REFUSAL_MARKERS = (
    "i don't have enough information",
    "i don't know",
    "i cannot answer",
    "not in the context",
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def precision_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k <= 0 or not retrieved_ids:
        return 0.0
    top = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return sum(1 for doc_id in top if doc_id in relevant) / k


def recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    """Fraction of relevant items found within the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    top = retrieved_ids[:k] if k and k > 0 else list(retrieved_ids)
    return sum(1 for doc_id in top if doc_id in relevant) / len(relevant)


def mrr(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Mean reciprocal rank: 1 / rank of the first relevant item."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    """1.0 when at least one relevant item appears in the top-k."""
    relevant = set(relevant_ids)
    top = retrieved_ids[:k] if k and k > 0 else list(retrieved_ids)
    return 1.0 if any(doc_id in relevant for doc_id in top) else 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    """Normalized discounted cumulative gain (binary relevance)."""
    if k <= 0 or not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def faithfulness(answer: str, source_texts: Sequence[str], n: int = 3) -> float:
    """Lexical grounding heuristic for an answer against its sources.

    The fraction of the answer's word n-grams that also appear in the
    retrieved sources. 1.0 means every n-gram of the answer is literally
    present in the context (strictly grounded); low values suggest the
    answer uses phrasing absent from the context, which may indicate
    hallucination. This is a cheap lexical proxy, not a semantic judge.
    """
    answer_grams = _ngrams(_tokenize(answer), n)
    if not answer_grams:
        return 0.0
    source_grams: set[tuple[str, ...]] = set()
    for text in source_texts:
        source_grams.update(_ngrams(_tokenize(text), n))
    if not source_grams:
        return 0.0
    grounded = sum(1 for gram in answer_grams if gram in source_grams)
    return grounded / len(answer_grams)


def is_refusal(answer: str) -> bool:
    """True when the answer is a non-informative refusal."""
    normalized = " ".join(answer.lower().split())
    return any(marker in normalized for marker in _REFUSAL_MARKERS)


def refusal_rate(answers: Sequence[str]) -> float:
    """Fraction of answers that are refusals."""
    if not answers:
        return 0.0
    return sum(1 for answer in answers if is_refusal(answer)) / len(answers)