"""Hybrid search helpers: BM25 ranking and Reciprocal Rank Fusion.

Used by the vector store adapter to combine semantic (vector) hits with
lexical (keyword) hits without external dependencies.
"""

import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9а-яё]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization over latin/cyrillic words and digits."""
    return _TOKEN_RE.findall(text.lower())


def bm25_rank(
    query: str,
    corpus: list[str],
    k: float = 1.5,
    b: float = 0.75,
) -> list[int]:
    """Rank corpus documents against the query with BM25.

    Args:
        query: Keyword query text.
        corpus: Document texts; index in the corpus is the document ID.
        k, b: BM25 parameters (k controls term-frequency saturation,
            b the length normalization).

    Returns:
        Corpus indices sorted by descending BM25 score (documents with
        zero score are excluded).
    """
    doc_tokens = [tokenize(doc) for doc in corpus]
    doc_count = len(doc_tokens)
    if doc_count == 0:
        return []
    avg_len = sum(len(toks) for toks in doc_tokens) / doc_count or 1.0

    df: dict[str, int] = {}
    for toks in doc_tokens:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1

    query_terms = tokenize(query)
    if not query_terms:
        return []

    scores: list[float] = [0.0] * doc_count
    for index, toks in enumerate(doc_tokens):
        tf: dict[str, int] = {}
        for term in toks:
            tf[term] = tf.get(term, 0) + 1
        length_norm = len(toks) / avg_len
        for term in query_terms:
            occurrences = tf.get(term, 0)
            if not occurrences:
                continue
            idf = _idf(df.get(term, 0), doc_count)
            scores[index] += idf * occurrences * (k + 1) / (
                occurrences + k * (1 - b + b * length_norm)
            )

    ranked = sorted(
        (i for i in range(doc_count) if scores[i] > 0),
        key=lambda i: scores[i],
        reverse=True,
    )
    return ranked


def _idf(df: int, doc_count: int) -> float:
    """BM25 inverse document frequency."""
    import math

    return math.log(1 + (doc_count - df + 0.5) / (df + 0.5))


def rrf_fuse(
    ranked_lists: list[list[Any]],
    k: int = 60,
    top_k: int | None = None,
) -> list[Any]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    score(item) = sum over lists of 1 / (k + rank), rank starting at 1.

    Args:
        ranked_lists: Ordered items per list (best first).
        k: RRF damping constant (higher reduces the impact of top ranks).
        top_k: Limit the fused result length; None = all fused items.

    Returns:
        Items ordered by descending fused score (ties keep the order of
        first appearance in the first list).
    """
    scores: dict[Any, float] = {}
    first_seen: dict[Any, int] = {}
    for items in ranked_lists:
        for rank, item in enumerate(items, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    ordered = sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
    return ordered if top_k is None else ordered[:top_k]