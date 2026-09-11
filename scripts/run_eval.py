#!/usr/bin/env python
"""Offline RAG quality evaluation runner.

Loads a golden dataset (JSON), runs every case through a
RetrieverService and reports retrieval metrics (precision@k, recall@k,
MRR, hit rate, NDCG) plus answer metrics (faithfulness heuristic,
refusal rate).

Usage (from the ``src`` layout root, with PYTHONPATH pointing at src)::

    PYTHONPATH=src python scripts/run_eval.py \
        --dataset eval/golden_dataset.json
    PYTHONPATH=src python scripts/run_eval.py \
        --dataset eval/golden.json --json-out report.json

The golden dataset format is documented in ``evaluation/dataset.py``;
a sample lives in ``eval/golden_dataset.json``. For A/B comparisons run
the script twice with different ``SEARCH_HYBRID``/model settings and
diff the aggregate reports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from evaluation import EvalRunner, load_cases  # noqa: E402
from container import create_container  # noqa: E402
from application.services.retriever import RetrieverService  # noqa: E402


async def main() -> int:
    """Run the evaluation over the golden dataset and print a report."""
    parser = argparse.ArgumentParser(description='Run offline RAG evaluation')
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('eval/golden_dataset.json'),
        help='Path to the golden dataset JSON',
    )
    parser.add_argument('--top-k', type=int, default=5, help='Default top_k')
    parser.add_argument(
        '--json-out', type=Path, default=None, help='Write the report as JSON'
    )
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if not cases:
        print('No eval cases found; nothing to do.')
        return 1

    container = create_container()
    retriever = await container.get(RetrieverService)

    async def answer_fn(
        user_id: str, query: str, top_k: int | None = None
    ) -> dict[str, Any]:
        return await retriever.answer_query(
            user_id=user_id, query=query, top_k=top_k
        )

    runner = EvalRunner(answer_fn, top_k=args.top_k)
    report = await runner.run(cases)

    report_dict = report.to_dict()
    print(json.dumps(report_dict, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(
            json.dumps(report_dict, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f'Report written to {args.json_out}')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
