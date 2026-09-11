"""Golden eval dataset loading.

Dataset format (JSON)::

    [
        {
            "query": "What is RAG?",
            "user_id": "eval-user",
            "relevant_doc_ids": ["doc-uuid-1"],
            "top_k": 5,
            "source_texts": ["optional expected context text"]
        }
    ]

``source_texts`` is optional; when present, ``faithfulness`` of the
answer is computed against these texts instead of the retrieved chunk
texts (useful for golden answers authored from the source documents).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    """One golden evaluation case."""

    query: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    user_id: str = "eval-user"
    top_k: int | None = None
    source_texts: list[str] = field(default_factory=list)


def load_cases(path: Path | str) -> list[EvalCase]:
    """Load and validate eval cases from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Eval dataset must be a JSON array of cases")
    cases = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not item.get("query"):
            raise ValueError(f"Case #{index} must be an object with a 'query'")
        cases.append(
            EvalCase(
                query=str(item["query"]),
                relevant_doc_ids=[
                    str(doc_id) for doc_id in item.get("relevant_doc_ids", [])
                ],
                user_id=str(item.get("user_id", "eval-user")),
                top_k=item.get("top_k"),
                source_texts=[str(t) for t in item.get("source_texts", [])],
            )
        )
    return cases