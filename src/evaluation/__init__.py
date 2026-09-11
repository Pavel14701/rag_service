"""Offline evaluation of RAG quality (retrieval + answer metrics)."""

from .dataset import EvalCase, load_cases
from .metrics import (
    faithfulness,
    hit_rate,
    mrr,
    precision_at_k,
    recall_at_k,
    refusal_rate,
)
from .runner import CaseResult, EvalReport, EvalRunner

__all__ = (
    'CaseResult',
    'EvalCase',
    'EvalReport',
    'EvalRunner',
    'faithfulness',
    'hit_rate',
    'load_cases',
    'mrr',
    'precision_at_k',
    'recall_at_k',
    'refusal_rate',
)
